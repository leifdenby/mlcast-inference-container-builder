"""Mode 1, ``run``: read a zarr store, forecast with the bundled model, write a zarr store.

Every option ``--some-option`` can also be set with the environment variable
``MLCAST_SOME_OPTION`` (see :mod:`mlcast_icb.options`), so the container behaves the
same however it is launched (``docker run``, a Kubernetes Job, an Argo step). Every
value is checked before anything is loaded, and every problem is reported at once, as a
single :class:`~mlcast_icb.errors.ConfigError` (exit 2).

The flow is: resolve the device, load the bundle (fail fast on a broken image), open the
input lazily, select only the variables and timesteps the model needs, read them,
forecast, write. The startup log records the version first, then the effective
configuration one option per line, with its variable and where its value came from,
then what the bundle is.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import xarray as xr
from loguru import logger

from mlcast_icb import ENV_PREFIX, __version__
from mlcast_icb.bundle import DEFAULT_MODEL_DIR, load_bundle, manifest_summary, mlcast_version
from mlcast_icb.device import DEVICES, select_device
from mlcast_icb.errors import ConfigError, InputError, MlcastIcbError, format_problems
from mlcast_icb.io.storage import SIDES, StorageSettings, log_storage_options, redact_json, storage_options
from mlcast_icb.io.zarr_io import ZarrFormat, load_zarr, open_zarr, parse_zarr_format, write_zarr
from mlcast_icb.logs import configure_logging, log_lines, log_settings
from mlcast_icb.options import (
    ParsedOptions,
    attempt,
    blank_to_none,
    get_choice,
    get_datetime,
    get_int,
    log_level,
    parse_options,
    require,
)
from mlcast_icb.upstream.forecast import ForecastInputError, forecast
from mlcast_icb.upstream.spec import InferenceSpec

#: Distribution name, as logged at startup.
DISTRIBUTION = "mlcast-inference-container-builder"


@dataclass(frozen=True)
class RunConfig:
    """The parsed, validated configuration of one ``run``.

    Parameters
    ----------
    input_zarr, output_zarr : str
        Store URIs: local paths, ``s3://``, ``gs://``.
    model_dir : str
        The model bundle directory.
    forecast_reference_time : numpy.datetime64 or None
        The last input time to forecast from (naive UTC), or ``None`` for the last time
        in the input.
    forecast_steps, ensemble_size : int or None
        Overrides of the model's defaults, or ``None`` to use them.
    seed : int or None
        Seed for the ensemble noise, or ``None`` for an unseeded run.
    device : {'auto', 'cuda', 'cpu'}
        Compute device.
    zarr_format : {'auto', 2, 3}
        Output store format; ``'auto'`` matches the input.
    log_level : str
        loguru level name.
    input_storage, output_storage : StorageSettings
        The storage options of each side, as given.
    input_storage_options, output_storage_options : dict
        The fsspec options they resolve to, for the input and output stores.
    """

    input_zarr: str
    output_zarr: str
    model_dir: str = DEFAULT_MODEL_DIR
    forecast_reference_time: np.datetime64 | None = None
    forecast_steps: int | None = None
    ensemble_size: int | None = None
    seed: int | None = None
    device: str = "auto"
    zarr_format: ZarrFormat = "auto"
    log_level: str = "INFO"
    input_storage: StorageSettings = field(default_factory=StorageSettings)
    output_storage: StorageSettings = field(default_factory=StorageSettings)
    input_storage_options: dict[str, Any] = field(default_factory=dict)
    output_storage_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_options(cls, parsed: ParsedOptions, env: Mapping[str, str] | None = None) -> RunConfig:
        """Validate the parsed options of ``run``.

        Parameters
        ----------
        parsed : ParsedOptions
            The options, from :func:`~mlcast_icb.options.parse_options`.
        env : mapping of str to str, optional
            The environment, for the standard AWS variables that decide whether S3
            access is anonymous. Defaults to :data:`os.environ`.

        Returns
        -------
        RunConfig

        Raises
        ------
        ConfigError
            Listing every problem found: every option is checked before failing, each
            named as it was given (its variable, or its option). A side's storage
            options are resolved only when its store URI is set, because the URI's
            scheme decides which options apply.
        """
        args, label = parsed.args, parsed.label
        problems: list[str] = []

        input_zarr = attempt(problems, lambda: require(args.input_zarr, label("input_zarr")))
        output_zarr = attempt(problems, lambda: require(args.output_zarr, label("output_zarr")))
        reference_time = attempt(
            problems, lambda: get_datetime(args.forecast_reference_time, label("forecast_reference_time"))
        )
        forecast_steps = attempt(problems, lambda: get_int(args.forecast_steps, label("forecast_steps"), minimum=1))
        ensemble_size = attempt(problems, lambda: get_int(args.ensemble_size, label("ensemble_size"), minimum=1))
        seed = attempt(problems, lambda: get_int(args.seed, label("seed"), minimum=0))
        device = attempt(problems, lambda: get_choice(args.device, label("device"), DEVICES, "auto"))
        zarr_format = attempt(problems, lambda: parse_zarr_format(args.zarr_format, label("zarr_format")))
        settings = {side: _storage_settings(args, side) for side in SIDES}
        resolved = {}
        for side, uri in (("input", input_zarr), ("output", output_zarr)):
            resolved[side] = (
                None
                if uri is None
                else attempt(problems, lambda s=side, u=uri: storage_options(s, u, settings[s], env))
            )
        if input_zarr is not None and input_zarr == output_zarr:
            problems.append(
                f"{label('input_zarr')} and {label('output_zarr')} are the same store ({input_zarr}); "
                "the output replaces it"
            )

        if len(problems) == 1:
            raise ConfigError(problems[0])
        if problems:
            raise ConfigError(format_problems(f"The configuration has {len(problems)} problems:", problems))

        return cls(
            input_zarr=input_zarr,
            output_zarr=output_zarr,
            model_dir=blank_to_none(args.model_dir) or DEFAULT_MODEL_DIR,
            forecast_reference_time=reference_time,
            forecast_steps=forecast_steps,
            ensemble_size=ensemble_size,
            seed=seed,
            device=device,
            zarr_format=zarr_format,
            log_level=log_level(args.log_level),
            input_storage=settings["input"],
            output_storage=settings["output"],
            input_storage_options=resolved["input"],
            output_storage_options=resolved["output"],
        )

    @classmethod
    def from_command_line(cls, argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> RunConfig:
        """Parse and validate the options of ``run`` from a command line and an environment.

        Parameters
        ----------
        argv : sequence of str, optional
            The arguments after ``run``. Defaults to none.
        env : mapping of str to str, optional
            The environment. Defaults to :data:`os.environ`; tests pass a dict.

        Returns
        -------
        RunConfig

        Raises
        ------
        SystemExit
            Exit code 2 for what argparse rejects (see
            :func:`~mlcast_icb.options.parse_options`).
        ConfigError
            For invalid values (see :meth:`from_options`).
        """
        return cls.from_options(parse_options("run", argv, env), env)


def describe_config(config: RunConfig) -> dict[str, str]:
    """Render the effective configuration for the startup log.

    Parameters
    ----------
    config : RunConfig
        The parsed configuration.

    Returns
    -------
    dict of str to str
        One entry per option of ``run``, defaults included, keyed by its ``dest``, in
        the parser's order. An optional setting left unset reads ``unset (...)``, naming
        what happens instead. The storage options JSON is shown with its credentials
        masked.
    """
    reference = config.forecast_reference_time
    described = {
        "input_zarr": config.input_zarr,
        "output_zarr": config.output_zarr,
        "model_dir": config.model_dir,
        "forecast_reference_time": _or_unset(
            None if reference is None else np.datetime_as_string(reference, unit="s"), "last time in the input"
        ),
        "forecast_steps": _or_unset(config.forecast_steps, "model default"),
        "ensemble_size": _or_unset(config.ensemble_size, "model default"),
        "seed": _or_unset(config.seed, "not seeded"),
        "device": config.device,
        "zarr_format": str(config.zarr_format),
        "log_level": config.log_level,
    }
    for side, settings in (("input", config.input_storage), ("output", config.output_storage)):
        described[f"{side}_aws_profile"] = _or_unset(settings.aws_profile, "AWS configuration")
        described[f"{side}_s3_endpoint_url"] = _or_unset(settings.s3_endpoint_url, "AWS configuration")
        described[f"{side}_s3_anon"] = _or_unset(
            None if settings.s3_anon is None else str(settings.s3_anon).lower(), "anonymous unless credentials"
        )
        described[f"{side}_storage_options"] = _or_unset(redact_json(settings.storage_options), "none")
    return described


def _storage_settings(args: Any, side: str) -> StorageSettings:
    """Collect one side's ``--<side>-*`` storage options."""
    return StorageSettings(
        aws_profile=blank_to_none(getattr(args, f"{side}_aws_profile")),
        s3_endpoint_url=blank_to_none(getattr(args, f"{side}_s3_endpoint_url")),
        s3_anon=getattr(args, f"{side}_s3_anon"),
        storage_options=blank_to_none(getattr(args, f"{side}_storage_options")),
    )


def select_inputs(ds: xr.Dataset, spec: InferenceSpec, reference_time: np.datetime64 | None) -> xr.Dataset:
    """Narrow a lazily opened input to what the forecast reads, before anything is loaded.

    This is only an optimisation: :func:`~mlcast_icb.upstream.forecast.forecast` does
    the real validation. Anything this cannot narrow safely is passed through, so the
    forecast reports it with its own, complete message.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened input.
    spec : InferenceSpec
        The model's spec, naming the standard names and the number of input steps.
    reference_time : numpy.datetime64 or None
        The last input time, or ``None`` for the last time in the input.

    Returns
    -------
    xarray.Dataset
        The variables carrying one of ``spec.standard_names`` and the grid-mapping
        variables they reference (every data variable if a standard name is not found),
        and the ``spec.input_steps`` timesteps ending at ``reference_time`` (all
        timesteps if no single datetime dimension is found).

    Raises
    ------
    InputError
        If ``reference_time`` is given but is not one of the input's times.
    """
    names = [name for name, var in ds.data_vars.items() if var.attrs.get("standard_name") in spec.standard_names]
    found = {ds[name].attrs["standard_name"] for name in names}
    if found >= set(spec.standard_names):
        keep = set(names)
        for name in names:
            mapping = str(ds[name].attrs.get("grid_mapping", "")).split(":")[0].strip()
            if mapping in ds.data_vars:
                keep.add(mapping)
        ds = ds[[name for name in ds.data_vars if name in keep]]

    time_dims = {
        dim
        for name in ds.data_vars
        for dim in ds[name].dims
        if dim in ds.coords and np.issubdtype(ds[dim].dtype, np.datetime64)
    }
    if len(time_dims) != 1:
        return ds
    (time_dim,) = time_dims
    times = ds[time_dim].values
    if reference_time is None:
        end = len(times) - 1
    else:
        matches = np.flatnonzero(times == reference_time)
        if len(matches) == 0:
            span = (
                f"the input has {len(times)} times from {times.min()} to {times.max()}" if len(times) else "it is empty"
            )
            raise InputError(
                f"The forecast reference time {np.datetime_as_string(reference_time, unit='s')} is not one of the "
                f"input's times; {span}"
            )
        end = int(matches[-1])
    start = max(0, end - spec.input_steps + 1)
    return ds.isel({time_dim: slice(start, end + 1)})


def run(config: RunConfig) -> xr.Dataset:
    """Execute one forecast, start to finish.

    Parameters
    ----------
    config : RunConfig
        The parsed configuration.

    Returns
    -------
    xarray.Dataset
        The forecast, as written.

    Raises
    ------
    DeviceError
        If the requested device is unusable (exit 4).
    ModelError
        If the bundle is missing or corrupt (exit 6).
    InputError
        If the input is missing or does not match what the model expects (exit 3).
    StorageError
        If a store cannot be reached, read or written (exit 5).
    """
    started = time.perf_counter()
    device = select_device(config.device)

    tic = time.perf_counter()
    model, spec, manifest = load_bundle(config.model_dir, device)
    log_settings(
        f"Model bundle {config.model_dir} (loaded in {time.perf_counter() - tic:.2f} s):", manifest_summary(manifest)
    )
    if manifest.get("mlcast_version") not in (None, mlcast_version()):
        logger.warning(
            f"The bundle was made with mlcast {manifest['mlcast_version']}, but this image runs mlcast "
            f"{mlcast_version()}; rebuild the bundle with this builder release if results look wrong"
        )

    forecast_steps = config.forecast_steps or spec.forecast_steps
    ensemble_size = config.ensemble_size or spec.default_ensemble_size
    logger.info(
        f"Forecasting {forecast_steps} steps ({'as configured' if config.forecast_steps else 'model default'}), "
        f"{ensemble_size} members ({'as configured' if config.ensemble_size else 'model default'}), "
        f"seed {'unset' if config.seed is None else config.seed}"
    )

    ds = open_zarr(config.input_zarr, config.input_storage_options)
    ds = select_inputs(ds, spec, config.forecast_reference_time)
    tic = time.perf_counter()
    ds = load_zarr(ds, config.input_zarr, config.input_storage_options)
    logger.info(
        f"Read {', '.join(ds.data_vars)} {dict(ds.sizes)} ({ds.nbytes / 1024**2:.1f} MiB) "
        f"in {time.perf_counter() - tic:.2f} s"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tic = time.perf_counter()
    try:
        out = forecast(
            model,
            spec,
            ds,
            forecast_steps=forecast_steps,
            ensemble_size=ensemble_size,
            seed=config.seed,
            reference_time=config.forecast_reference_time,
        )
    except ForecastInputError as exc:
        headline = f"The input {config.input_zarr} does not match what the model expects:"
        problems = getattr(exc, "problems", None)
        message = format_problems(headline, [str(p) for p in problems]) if problems else f"{headline} {exc}"
        raise InputError(message) from exc
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    message = f"Forecast took {time.perf_counter() - tic:.2f} s on {device.type}"
    if device.type == "cuda":
        message += f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB"
    logger.info(message)

    out = _with_provenance(out, config, manifest)
    write_zarr(
        out,
        config.output_zarr,
        config.output_storage_options,
        config.zarr_format,
        source_uri=config.input_zarr,
        source_storage_options=config.input_storage_options,
        encoding=_encoding(out),
    )
    logger.info(f"Done in {time.perf_counter() - started:.2f} s")
    return out


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Run Mode 1 from the command line and the environment, and return a process exit code.

    Parameters
    ----------
    argv : sequence of str, optional
        The arguments after ``run``. Defaults to none.
    env : mapping of str to str, optional
        Environment to read. Defaults to :data:`os.environ`.

    Returns
    -------
    int
        0 on success (and after ``--help``), otherwise the failing error's exit code
        (see :mod:`mlcast_icb.errors`; 2 for any configuration error), or 1 for anything
        unexpected, with its traceback.

    Notes
    -----
    The version is logged first, before the configuration is parsed, so even a run that
    exits 2 says which version failed; the effective configuration follows as soon as
    it parses, before any store or model is touched.
    """
    env = os.environ if env is None else env
    configure_logging(log_level(env.get(f"{ENV_PREFIX}_LOG_LEVEL")))
    logger.info(f"Starting {DISTRIBUTION} {__version__} (run)")
    try:
        parsed = parse_options("run", argv, env)
    except SystemExit as exc:  # argparse has printed the usage and the error, or the help
        return int(exc.code or 0)
    try:
        config = RunConfig.from_options(parsed, env)
    except MlcastIcbError as exc:
        logger.error(f"{exc}")
        return exc.exit_code

    configure_logging(config.log_level)
    log_lines("Effective configuration (defaults included):", parsed.config_lines(describe_config(config)))
    for name in parsed.ignored:
        logger.debug(f"Ignoring {name}: it is an option of `serve`, not of `run`")
    log_storage_options("input", config.input_storage_options)
    log_storage_options("output", config.output_storage_options)
    try:
        run(config)
    except MlcastIcbError as exc:
        logger.error(f"{exc}")
        return exc.exit_code
    except Exception:
        logger.opt(exception=True).error("Unexpected failure")
        return 1
    return 0


def _with_provenance(out: xr.Dataset, config: RunConfig, manifest: Mapping[str, Any]) -> xr.Dataset:
    """Record which model, builder and input produced a forecast, and drop inherited encodings.

    Encodings carried over from the input (zarr v3 codecs, say) can be invalid for the
    output's format, so every variable is written with fresh defaults.
    """
    out = out.copy()
    for variable in out.variables.values():
        variable.encoding = {}
    attrs = {
        "source": f"{DISTRIBUTION} {__version__}",
        "model_name": manifest.get("name"),
        "model_version": manifest.get("model_version"),
        "model_sha256": manifest.get("model_sha256"),
        "mlcast_version": manifest.get("mlcast_version"),
        "input_zarr": config.input_zarr,
        "seed": config.seed,
        "history": f"{datetime.now(UTC).isoformat(timespec='seconds')}: forecast by {DISTRIBUTION} {__version__}",
    }
    out.attrs.update({key: value for key, value in attrs.items() if value is not None})
    return out


def _encoding(out: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Chunk each forecast field per member and time step, so a reader can fetch one map at a time."""
    encoding = {}
    for name, var in out.data_vars.items():
        if var.ndim == 4:
            encoding[name] = {"chunks": (1, 1, *var.shape[2:])}
    return encoding


def _or_unset(value: object, meaning: str) -> str:
    """Render an optional setting, spelling out what an unset one means."""
    return f"unset ({meaning})" if value is None else str(value)
