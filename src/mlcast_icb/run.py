"""Mode 1, ``run``: read a zarr store, forecast with the bundled model, write a zarr store.

Configuration comes entirely from environment variables, so the container behaves the
same however it is launched (``docker run``, a Kubernetes Job, an Argo step). Every
variable is parsed before anything is loaded, and every problem is reported at once, as
a single :class:`~mlcast_icb.errors.ConfigError` (exit 2).

The flow is: resolve the device, load the bundle (fail fast on a broken image), open the
input lazily, select only the variables and timesteps the model needs, read them,
forecast, write. The startup log records the version first, then the effective
configuration one setting per line, then what the bundle is.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from mlcast_icb import __version__
from mlcast_icb.bundle import DEFAULT_MODEL_DIR, load_bundle, manifest_summary, mlcast_version
from mlcast_icb.device import DEVICES, select_device
from mlcast_icb.envvars import warn_unknown_env_vars
from mlcast_icb.errors import ConfigError, InputError, MlcastIcbError, format_problems
from mlcast_icb.io.storage import storage_options, storage_variables
from mlcast_icb.io.zarr_io import ZarrFormat, load_zarr, open_zarr, parse_zarr_format, write_zarr
from mlcast_icb.logs import configure_logging, log_settings
from mlcast_icb.upstream.forecast import ForecastInputError, forecast
from mlcast_icb.upstream.spec import InferenceSpec

#: Distribution name, as logged at startup.
DISTRIBUTION = "mlcast-inference-container-builder"

#: The variables ``run`` itself reads, in the order the README documents them.
RUN_VARIABLES = (
    "INPUT_ZARR",
    "OUTPUT_ZARR",
    "MODEL_DIR",
    "FORECAST_REFERENCE_TIME",
    "FORECAST_STEPS",
    "ENSEMBLE_SIZE",
    "SEED",
    "DEVICE",
    "ZARR_FORMAT",
    "LOG_LEVEL",
)

#: Every variable ``run`` reads, storage variables included, for the typo check.
KNOWN_VARIABLES = frozenset(RUN_VARIABLES) | storage_variables()

_T = TypeVar("_T")


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
    src_storage_options, dst_storage_options : dict
        fsspec options for the input and output stores.
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
    src_storage_options: dict[str, Any] = field(default_factory=dict)
    dst_storage_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RunConfig:
        """Build the configuration from environment variables.

        Parameters
        ----------
        env : mapping of str to str, optional
            Environment to read. Defaults to :data:`os.environ`; tests pass a dict.

        Returns
        -------
        RunConfig

        Raises
        ------
        ConfigError
            Listing every problem found: every variable is checked before failing. A
            side's storage options are parsed only when its store URI is set, because
            the URI's scheme decides which options apply.
        """
        env = os.environ if env is None else env
        problems: list[str] = []

        input_zarr = _attempt(problems, lambda: _require(env, "INPUT_ZARR"))
        output_zarr = _attempt(problems, lambda: _require(env, "OUTPUT_ZARR"))
        reference_time = _attempt(problems, lambda: _get_datetime(env, "FORECAST_REFERENCE_TIME"))
        forecast_steps = _attempt(problems, lambda: _get_int(env, "FORECAST_STEPS", minimum=1))
        ensemble_size = _attempt(problems, lambda: _get_int(env, "ENSEMBLE_SIZE", minimum=1))
        seed = _attempt(problems, lambda: _get_int(env, "SEED", minimum=0))
        device = _attempt(problems, lambda: _get_choice(env, "DEVICE", DEVICES, "auto"))
        zarr_format = _attempt(problems, lambda: parse_zarr_format(env.get("ZARR_FORMAT")))
        src_options = (
            None if input_zarr is None else _attempt(problems, lambda: storage_options(env, input_zarr, "SRC"))
        )
        dst_options = (
            None if output_zarr is None else _attempt(problems, lambda: storage_options(env, output_zarr, "DST"))
        )
        if input_zarr is not None and input_zarr == output_zarr:
            problems.append(f"INPUT_ZARR and OUTPUT_ZARR are the same store ({input_zarr}); the output replaces it")

        if len(problems) == 1:
            raise ConfigError(problems[0])
        if problems:
            raise ConfigError(format_problems(f"The environment configuration has {len(problems)} problems:", problems))

        return cls(
            input_zarr=input_zarr,
            output_zarr=output_zarr,
            model_dir=env.get("MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR,
            forecast_reference_time=reference_time,
            forecast_steps=forecast_steps,
            ensemble_size=ensemble_size,
            seed=seed,
            device=device,
            zarr_format=zarr_format,
            log_level=env.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            src_storage_options=src_options,
            dst_storage_options=dst_options,
        )


def describe_config(config: RunConfig) -> dict[str, str]:
    """Render the effective configuration for the startup log.

    Parameters
    ----------
    config : RunConfig
        The parsed configuration.

    Returns
    -------
    dict of str to str
        One entry per setting, defaults included, keyed by the variable that controls
        it, in :data:`RUN_VARIABLES` order. An optional setting left unset reads
        ``unset (...)``, naming what happens instead.

    Notes
    -----
    The storage options are left out: they come from many variables rather than one,
    and :func:`~mlcast_icb.io.storage.storage_options` has already logged them with
    credentials masked. One place rendering them means one place that has to get the
    redaction right.
    """
    reference = config.forecast_reference_time
    return {
        "INPUT_ZARR": config.input_zarr,
        "OUTPUT_ZARR": config.output_zarr,
        "MODEL_DIR": config.model_dir,
        "FORECAST_REFERENCE_TIME": _or_unset(
            None if reference is None else np.datetime_as_string(reference, unit="s"), "last time in the input"
        ),
        "FORECAST_STEPS": _or_unset(config.forecast_steps, "model default"),
        "ENSEMBLE_SIZE": _or_unset(config.ensemble_size, "model default"),
        "SEED": _or_unset(config.seed, "not seeded"),
        "DEVICE": config.device,
        "ZARR_FORMAT": str(config.zarr_format),
        "LOG_LEVEL": config.log_level,
    }


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
                f"FORECAST_REFERENCE_TIME {np.datetime_as_string(reference_time, unit='s')} is not one of the "
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
        f"Forecasting {forecast_steps} steps ({'FORECAST_STEPS' if config.forecast_steps else 'model default'}), "
        f"{ensemble_size} members ({'ENSEMBLE_SIZE' if config.ensemble_size else 'model default'}), "
        f"seed {'unset' if config.seed is None else config.seed}"
    )

    ds = open_zarr(config.input_zarr, config.src_storage_options)
    ds = select_inputs(ds, spec, config.forecast_reference_time)
    tic = time.perf_counter()
    ds = load_zarr(ds, config.input_zarr, config.src_storage_options)
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
        config.dst_storage_options,
        config.zarr_format,
        source_uri=config.input_zarr,
        source_storage_options=config.src_storage_options,
        encoding=_encoding(out),
    )
    logger.info(f"Done in {time.perf_counter() - started:.2f} s")
    return out


def main(env: Mapping[str, str] | None = None) -> int:
    """Run Mode 1 from the environment and return a process exit code.

    Parameters
    ----------
    env : mapping of str to str, optional
        Environment to read. Defaults to :data:`os.environ`.

    Returns
    -------
    int
        0 on success, otherwise the failing error's exit code (see
        :mod:`mlcast_icb.errors`), or 1 for anything unexpected, with its traceback.

    Notes
    -----
    The version is logged first, before the configuration is parsed, so even a run that
    exits 2 says which version failed; the effective configuration follows as soon as
    it parses, before any store or model is touched.
    """
    env = os.environ if env is None else env
    configure_logging(env.get("LOG_LEVEL", "INFO").strip().upper() or "INFO")
    logger.info(f"Starting {DISTRIBUTION} {__version__} (run)")
    warn_unknown_env_vars(KNOWN_VARIABLES, env)
    try:
        config = RunConfig.from_env(env)
    except MlcastIcbError as exc:
        logger.error(f"{exc}")
        return exc.exit_code

    configure_logging(config.log_level)
    log_settings("Effective configuration (defaults included):", describe_config(config))
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


def _attempt(problems: list[str], parse: Callable[[], _T]) -> _T | None:
    """Run one parse step, recording a :class:`ConfigError` instead of raising it."""
    try:
        return parse()
    except ConfigError as exc:
        problems.append(str(exc))
        return None


def _or_unset(value: object, meaning: str) -> str:
    """Render an optional setting, spelling out what an unset one means."""
    return f"unset ({meaning})" if value is None else str(value)


def _require(env: Mapping[str, str], key: str) -> str:
    """Read a required variable, failing when it is unset or blank."""
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(f"{key} is required but not set")
    return value


def _get_int(env: Mapping[str, str], key: str, minimum: int) -> int | None:
    """Read an optional integer of at least ``minimum``; ``None`` when unset."""
    raw = env.get(key, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}, got {value}")
    return value


def _get_choice(env: Mapping[str, str], key: str, choices: tuple[str, ...], default: str) -> str:
    """Read a case-insensitive choice."""
    raw = env.get(key, "").strip()
    if not raw:
        return default
    if raw.lower() not in choices:
        raise ConfigError(f"{key} must be one of {', '.join(choices)}, got {raw!r}")
    return raw.lower()


def _get_datetime(env: Mapping[str, str], key: str) -> np.datetime64 | None:
    """Read an ISO 8601 date-time as naive UTC; a time zone, if given, is converted to UTC."""
    raw = env.get(key, "").strip()
    if not raw:
        return None
    try:
        stamp = pd.Timestamp(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an ISO 8601 date-time such as '2026-05-01T12:00:00Z', got {raw!r}") from exc
    if stamp is pd.NaT:
        raise ConfigError(f"{key} must be an ISO 8601 date-time such as '2026-05-01T12:00:00Z', got {raw!r}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return np.datetime64(stamp.to_datetime64(), "ns")
