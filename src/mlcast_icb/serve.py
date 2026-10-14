"""Mode 2, ``serve``: an HTTP server that takes a netCDF of past fields and returns a forecast netCDF.

The API is a superset of IRENE's (``it4lia/irene``, FBK) FastAPI server, so its clients
keep working unchanged: ``GET /health``, ``GET /model/info`` and ``POST /predict`` with a
multipart ``file`` upload and the query parameters ``variable``, ``forecast_steps`` and
``ensemble_size``, answered with ``predictions.nc`` holding
``precipitation_forecast(ensemble_member, forecast_step, y, x)`` in mm/h. That layout is
``format=irene``, the default. ``format=cf`` returns the CF-complete forecast of
:func:`~mlcast_icb.upstream.forecast.forecast` instead, optionally reduced to ensemble
products (mean, quantiles, exceedance probabilities).

Configuration comes from options mirrored by ``MLCAST_*`` environment variables, like
``run`` (see :mod:`mlcast_icb.options`): every value is checked before anything is
loaded and every problem is reported at once (exit 2). The device is
resolved (exit 4) and the bundle loaded (exit 6) before the port is bound, so a broken
image never accepts a connection. One warm-up forward pass then runs in the background;
``/readyz`` answers 200 only after it.

Concurrency
-----------
Inference runs in the threadpool, never on the event loop, so ``/health``, ``/healthz``,
``/readyz`` and ``/metrics`` answer while a forecast runs. One lock serialises the
model: the GPU holds one forecast at a time, and the seeded forecast's global RNG state
(see :mod:`mlcast_icb.upstream.forecast`) is only safe that way. At most ``1 + --max-queue``
predictions are in progress at once; a request beyond that is answered 429 at once,
with a ``Retry-After`` estimate, rather than queueing without bound.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import IO, Any, Literal

import numpy as np
import pandas as pd
import torch
import uvicorn
import xarray as xr
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from mlcast_icb import ENV_PREFIX, __version__
from mlcast_icb.bundle import DEFAULT_MODEL_DIR, load_bundle, manifest_summary, mlcast_version
from mlcast_icb.device import DEVICES, select_device
from mlcast_icb.errors import ConfigError, InputError, MlcastIcbError, PayloadTooLargeError, format_problems
from mlcast_icb.io.netcdf_io import SPOOL_MAX_MEMORY, read_netcdf_bytes, write_netcdf
from mlcast_icb.logs import log_lines, log_settings
from mlcast_icb.options import (
    ParsedOptions,
    attempt,
    blank_to_none,
    get_choice,
    get_int,
    get_positive_float,
    log_level,
    parse_options,
)
from mlcast_icb.upstream.forecast import (
    COMPATIBLE_STANDARD_NAMES,
    ForecastInputError,
    forecast,
    unit_conversion_factor,
)
from mlcast_icb.upstream.spec import InferenceSpec

#: Distribution name, as logged at startup.
DISTRIBUTION = "mlcast-inference-container-builder"

#: IRENE's name for the input variable, the default of ``variable=``.
IRENE_VARIABLE = "RR"

#: IRENE's output variable and file name.
IRENE_OUTPUT_VARIABLE = "precipitation_forecast"
OUTPUT_FILENAME = "predictions.nc"
NETCDF_MEDIA_TYPE = "application/x-netcdf"

#: Dimension names IRENE accepts as a leading time dimension.
IRENE_TIME_DIMS = frozenset({"time", "t", "step", "forecast_time", "lead_time"})

#: Content types accepted for a raw-body (non-multipart) upload.
RAW_CONTENT_TYPES = frozenset(
    {
        "application/x-netcdf",
        "application/netcdf",
        "application/x-netcdf4",
        "application/x-hdf5",
        "application/octet-stream",
    }
)

#: Ensemble products ``products=`` may name.
PRODUCTS = ("members", "mean", "quantiles", "exceedance")
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_THRESHOLDS = (0.5, 1.0, 5.0)

#: Timestep assumed for synthesised input times when the model's spec does not record one.
_FALLBACK_TIMESTEP = "PT5M"

#: An incoming X-Request-Id is echoed only if it looks like an identifier.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:/+=-]{1,128}")

#: The loguru format of ``serve``: ``logs.LOG_FORMAT`` plus the request id.
SERVE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> | req={extra[request_id]} - <level>{message}</level>"
)


# --- configuration -------------------------------------------------------------------


@dataclass(frozen=True)
class ServeConfig:
    """The parsed, validated configuration of the server.

    Parameters
    ----------
    model_dir : str
        The model bundle directory.
    device : {'auto', 'cuda', 'cpu'}
        Compute device.
    host : str
        Address to bind.
    port : int
        Port to bind.
    max_upload_mb : int
        Largest accepted upload, in MiB.
    max_queue : int
        Predictions allowed to wait while one runs; more are answered 429.
    request_timeout : float or None
        Seconds a prediction may wait for the model before it is answered 503, or
        ``None`` to wait as long as it takes.
    max_forecast_steps, max_ensemble_size : int
        Upper limits of the ``forecast_steps`` and ``ensemble_size`` query parameters.
    log_level : str
        loguru level name.
    """

    model_dir: str = DEFAULT_MODEL_DIR
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 100
    max_queue: int = 2
    request_timeout: float | None = None
    max_forecast_steps: int = 48
    max_ensemble_size: int = 10
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        """The upload limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_options(cls, parsed: ParsedOptions) -> ServeConfig:
        """Validate the parsed options of ``serve``.

        Parameters
        ----------
        parsed : ParsedOptions
            The options, from :func:`~mlcast_icb.options.parse_options`.

        Returns
        -------
        ServeConfig

        Raises
        ------
        ConfigError
            Listing every problem found, each named as it was given (its variable, or
            its option).
        """
        args, label = parsed.args, parsed.label
        problems: list[str] = []
        defaults = cls()

        device = attempt(problems, lambda: get_choice(args.device, label("device"), DEVICES, defaults.device))
        port = attempt(problems, lambda: get_int(args.port, label("port"), minimum=1, maximum=65535))
        max_upload_mb = attempt(problems, lambda: get_int(args.max_upload_mb, label("max_upload_mb"), minimum=1))
        max_queue = attempt(problems, lambda: get_int(args.max_queue, label("max_queue"), minimum=0))
        request_timeout = attempt(problems, lambda: get_positive_float(args.request_timeout, label("request_timeout")))
        max_forecast_steps = attempt(
            problems, lambda: get_int(args.max_forecast_steps, label("max_forecast_steps"), minimum=1)
        )
        max_ensemble_size = attempt(
            problems, lambda: get_int(args.max_ensemble_size, label("max_ensemble_size"), minimum=1)
        )

        if len(problems) == 1:
            raise ConfigError(problems[0])
        if problems:
            raise ConfigError(format_problems(f"The configuration has {len(problems)} problems:", problems))

        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return cls(
            model_dir=blank_to_none(args.model_dir) or DEFAULT_MODEL_DIR,
            device=device,
            host=blank_to_none(args.host) or defaults.host,
            port=pick(port, defaults.port),
            max_upload_mb=pick(max_upload_mb, defaults.max_upload_mb),
            max_queue=pick(max_queue, defaults.max_queue),
            request_timeout=request_timeout,
            max_forecast_steps=pick(max_forecast_steps, defaults.max_forecast_steps),
            max_ensemble_size=pick(max_ensemble_size, defaults.max_ensemble_size),
            log_level=log_level(args.log_level),
        )

    @classmethod
    def from_command_line(cls, argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> ServeConfig:
        """Parse and validate the options of ``serve`` from a command line and an environment.

        Parameters
        ----------
        argv : sequence of str, optional
            The arguments after ``serve``. Defaults to none.
        env : mapping of str to str, optional
            The environment. Defaults to :data:`os.environ`; tests pass a dict.

        Returns
        -------
        ServeConfig

        Raises
        ------
        SystemExit
            Exit code 2 for what argparse rejects.
        ConfigError
            For invalid values (see :meth:`from_options`).
        """
        return cls.from_options(parse_options("serve", argv, env))


def describe_config(config: ServeConfig) -> dict[str, str]:
    """Render the effective configuration for the startup log.

    Parameters
    ----------
    config : ServeConfig
        The parsed configuration.

    Returns
    -------
    dict of str to str
        One entry per option of ``serve``, keyed by its ``dest``, in the parser's order.
    """
    return {
        "model_dir": config.model_dir,
        "device": config.device,
        "host": config.host,
        "port": str(config.port),
        "max_upload_mb": str(config.max_upload_mb),
        "max_queue": str(config.max_queue),
        "request_timeout": "unset (wait for the model as long as it takes)"
        if config.request_timeout is None
        else f"{config.request_timeout:g} s",
        "max_forecast_steps": str(config.max_forecast_steps),
        "max_ensemble_size": str(config.max_ensemble_size),
        "log_level": config.log_level,
    }


# --- the model and its gate ------------------------------------------------------------


@dataclass
class LoadedModel:
    """A loaded bundle: the model, its spec and manifest, and the device it runs on."""

    model: torch.nn.Module
    spec: InferenceSpec
    manifest: dict[str, Any]
    device: torch.device


def load_model(config: ServeConfig) -> LoadedModel:
    """Resolve the device and load the bundle, logging what it is.

    Parameters
    ----------
    config : ServeConfig
        The parsed configuration.

    Returns
    -------
    LoadedModel

    Raises
    ------
    DeviceError
        If the requested device is unusable (exit 4).
    ModelError
        If the bundle is missing or corrupt (exit 6).
    """
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
    return LoadedModel(model=model, spec=spec, manifest=manifest, device=device)


class QueueFullError(Exception):
    """Too many predictions are already in progress (HTTP 429)."""


class QueueTimeoutError(Exception):
    """A prediction waited longer than ``--request-timeout`` for the model (HTTP 503)."""


class Metrics:
    """The server's Prometheus metrics, in a registry of their own (one per app)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "mlcast_icb_requests",
            "HTTP requests, by method, route and status code",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "mlcast_icb_request_duration_seconds",
            "HTTP request latency, by method and route",
            ["method", "endpoint"],
            buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "mlcast_icb_inference_duration_seconds",
            "Time the model takes for one forecast (the forward pass, not reading or writing)",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )
        self.queue_depth = Gauge("mlcast_icb_queue_depth", "Predictions waiting for the model", registry=self.registry)
        self.in_progress = Gauge(
            "mlcast_icb_predictions_in_progress",
            "Predictions admitted and not yet answered (running, waiting, reading or writing)",
            registry=self.registry,
        )
        self.ready = Gauge("mlcast_icb_ready", "1 once the model is loaded and warmed up", registry=self.registry)
        self.model_info = Gauge(
            "mlcast_icb_model_info",
            "The served model (always 1)",
            ["name", "model_version", "model_sha256", "mlcast_version", "builder_version"],
            registry=self.registry,
        )


class InferenceGate:
    """Admission control and the model lock.

    Parameters
    ----------
    max_queue : int
        Predictions allowed to wait while one runs.
    timeout : float or None
        Seconds :meth:`model` waits for the lock, or ``None`` for no limit.
    metrics : Metrics
        Where queue depth and in-progress counts are reported.
    """

    def __init__(self, max_queue: int, timeout: float | None, metrics: Metrics) -> None:
        self.max_queue = max_queue
        self.timeout = timeout
        self.metrics = metrics
        self.admitted = 0
        self.waiting = 0
        self._model_lock = threading.Lock()
        self._mutex = threading.Lock()
        self._inference_seconds = 1.0  # running estimate, for Retry-After

    @contextmanager
    def admit(self) -> Iterator[None]:
        """Admit one prediction for its whole lifetime, or refuse it when ``1 + max_queue`` are in progress.

        Raises
        ------
        QueueFullError
            If the server is at capacity.
        """
        with self._mutex:
            if self.admitted >= 1 + self.max_queue:
                raise QueueFullError(f"{self.admitted} predictions in progress, at most {1 + self.max_queue}")
            self.admitted += 1
            self.metrics.in_progress.set(self.admitted)
        try:
            yield
        finally:
            with self._mutex:
                self.admitted -= 1
                self.metrics.in_progress.set(self.admitted)

    @contextmanager
    def model(self) -> Iterator[None]:
        """Hold the model lock for the duration of the block.

        Raises
        ------
        QueueTimeoutError
            If the lock was not acquired within ``timeout`` seconds.
        """
        with self._mutex:
            self.waiting += 1
            self.metrics.queue_depth.set(self.waiting)
        try:
            acquired = self._model_lock.acquire(timeout=-1 if self.timeout is None else self.timeout)
        finally:
            with self._mutex:
                self.waiting -= 1
                self.metrics.queue_depth.set(self.waiting)
        if not acquired:
            raise QueueTimeoutError(f"waited more than {self.timeout:g} s for the model")
        try:
            yield
        finally:
            self._model_lock.release()

    def record_inference(self, seconds: float) -> None:
        """Feed one inference duration into the metrics and the Retry-After estimate."""
        self.metrics.inference_duration.observe(seconds)
        self._inference_seconds = 0.7 * self._inference_seconds + 0.3 * seconds

    def retry_after(self) -> int:
        """Seconds a refused client should wait before retrying: the estimated time to drain the queue."""
        return max(1, math.ceil(self._inference_seconds * max(1, self.admitted)))


@dataclass
class ServerState:
    """Everything the routes share."""

    config: ServeConfig
    loaded: LoadedModel | None
    metrics: Metrics = field(default_factory=Metrics)
    ready: threading.Event = field(default_factory=threading.Event)
    warm_up_error: str | None = None
    gate: InferenceGate = field(init=False)

    def __post_init__(self) -> None:
        self.gate = InferenceGate(self.config.max_queue, self.config.request_timeout, self.metrics)


# --- input preparation -----------------------------------------------------------------


@dataclass
class PreparedInput:
    """An upload, reshaped into what :func:`forecast` reads.

    Attributes
    ----------
    ds : xarray.Dataset
        The input variable (and, in CF mode, its grid mapping and auxiliary coordinates)
        with CF time, y and x coordinates and ``standard_name``/``units`` attributes.
    name : str or None
        The input variable's name; ``None`` for a multi-variable model's input, which is
        passed through unchanged.
    valid : xarray.DataArray or None
        ``(y, x)``: True where any of the last ``input_steps`` fields has a finite value.
    synthetic_time : bool
        True if the times were made up because the upload had no datetime coordinate.
    """

    ds: xr.Dataset
    name: str | None
    valid: xr.DataArray | None
    synthetic_time: bool = False


def prepare_input(ds: xr.Dataset, spec: InferenceSpec, variable: str | None, irene: bool) -> PreparedInput:
    """Find the input variable in an upload and annotate it so that :func:`forecast` can read it.

    IRENE's clients send bare ``(time, y, x)`` arrays without CF metadata, so this is
    lenient. In IRENE mode (``irene=True``) the upload is read exactly as IRENE reads
    it: the variable by name, its dims positionally as (time, y, x), the values as mm/h
    whatever its ``units`` say, and its time values ignored. In CF mode the variable's own
    ``standard_name``, ``units``, datetime coordinate, CF axes, grid mapping and
    auxiliary coordinates are used where present, with the IRENE assumptions filling
    in only what is missing.

    Parameters
    ----------
    ds : xarray.Dataset
        The upload.
    spec : InferenceSpec
        The model's spec.
    variable : str or None
        The variable the client named, or ``None`` to look for one (``RR`` first in
        IRENE mode, a variable with the model's standard name first in CF mode).
    irene : bool
        Whether to read the upload the way IRENE does.

    Returns
    -------
    PreparedInput

    Raises
    ------
    InputError
        If no usable variable is found, it is not 3-D, its first dimension is not time,
        it has fewer than ``spec.input_steps`` times, or it holds infinities.
    """
    if len(spec.standard_names) != 1:
        if irene:
            raise InputError(
                f"format=irene needs a single-variable model; this model reads {list(spec.standard_names)}. "
                "Use format=cf with one CF variable per standard name."
            )
        return PreparedInput(ds=ds, name=None, valid=None)

    standard_name, spec_units = spec.standard_names[0], spec.units[0]
    accepted = (standard_name, *COMPATIBLE_STANDARD_NAMES.get(standard_name, ()))
    name = _choose_variable(ds, accepted, variable, irene)
    da = ds[name]
    if da.ndim != 3:
        raise InputError(f"Expected 3D variable (time, y, x), got {da.ndim}D with dims {da.dims}.")

    t_dim, y_dim, x_dim, has_datetime = _axes(da, irene)
    da = da.transpose(t_dim, y_dim, x_dim)
    n_times = da.sizes[t_dim]
    if n_times < spec.input_steps:
        raise InputError(
            f"The model needs at least {spec.input_steps} input time steps, got {n_times} "
            f"(variable {name!r}, dims {da.dims}, shape {da.shape})."
        )
    values = np.asarray(da.values, dtype=np.float32)
    if np.isinf(values).any():
        raise InputError("Input data contains Inf values.")

    attrs = {} if irene else dict(da.attrs)
    attrs.pop("_FillValue", None)
    if irene:
        if "units" in da.attrs and unit_conversion_factor(str(da.attrs["units"]), "mm h-1") != 1.0:
            logger.warning(
                f"Variable {name!r} has units {da.attrs['units']!r}; format=irene reads it as mm/h, as IRENE does"
            )
        attrs.update(standard_name=standard_name, units="mm h-1")
    else:
        own = da.attrs.get("standard_name")
        if own is None:
            logger.info(f"Variable {name!r} has no standard_name; assuming {standard_name!r}")
            attrs["standard_name"] = standard_name
        elif own not in accepted:
            raise InputError(
                f"Variable {name!r} has standard_name {own!r}, but the model reads {standard_name!r}"
                + (f" (or {', '.join(repr(a) for a in accepted[1:])})" if len(accepted) > 1 else "")
            )
        if "units" not in attrs:
            logger.info(f"Variable {name!r} has no units; assuming {spec_units!r}")
            attrs["units"] = spec_units

    synthetic_time = not has_datetime
    if synthetic_time:
        step = pd.Timedelta(spec.timestep or _FALLBACK_TIMESTEP)
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor(step)
        times = pd.date_range(end=end, periods=n_times, freq=step).values
        if not irene:
            logger.warning(
                f"Variable {name!r} has no datetime time coordinate; assuming {spec.timestep or _FALLBACK_TIMESTEP} "
                f"steps ending at {end.isoformat()}"
            )
    else:
        times = da[t_dim].values

    coords: dict[str, Any] = {t_dim: xr.Variable(t_dim, times, {"standard_name": "time", "axis": "T"})}
    for dim, axis in ((y_dim, "Y"), (x_dim, "X")):
        if not irene and dim in ds.coords and ds[dim].ndim == 1 and np.issubdtype(ds[dim].dtype, np.number):
            coord = ds[dim].variable.copy(deep=False)
            coord.encoding = {}
            coord.attrs = {**coord.attrs, "axis": axis}
        else:
            index = "row" if axis == "Y" else "column"
            coord = xr.Variable(dim, np.arange(da.sizes[dim]), {"axis": axis, "long_name": f"grid {index} index"})
        coords[dim] = coord

    data_vars: dict[str, Any] = {name: xr.Variable((t_dim, y_dim, x_dim), values, attrs)}
    if not irene:
        grid_mapping = attrs.get("grid_mapping", da.encoding.get("grid_mapping"))
        if grid_mapping is not None and grid_mapping in ds.variables:
            data_vars[name].attrs["grid_mapping"] = grid_mapping
            coords[grid_mapping] = ds[grid_mapping].variable
        for coord_name, coord in da.coords.items():
            if coord_name not in (t_dim, y_dim, x_dim) and set(coord.dims) <= {y_dim, x_dim}:
                coords.setdefault(str(coord_name), coord.variable)

    valid = np.isfinite(values[-spec.input_steps :]).any(axis=0)
    return PreparedInput(
        ds=xr.Dataset(data_vars, coords=coords),
        name=name,
        valid=xr.DataArray(valid, dims=(y_dim, x_dim)),
        synthetic_time=synthetic_time,
    )


def _choose_variable(ds: xr.Dataset, accepted: Sequence[str], variable: str | None, irene: bool) -> str:
    """Pick the input variable: the one named, else ``RR`` / one with an accepted standard name."""
    available = [str(name) for name in ds.data_vars]
    if variable is not None:
        if variable not in ds.data_vars:
            raise InputError(f"Variable '{variable}' not found. Available: {available}")
        return variable
    by_standard_name = [str(n) for n, v in ds.data_vars.items() if v.attrs.get("standard_name") in accepted]
    if irene and IRENE_VARIABLE in ds.data_vars:
        return IRENE_VARIABLE
    if len(by_standard_name) == 1:
        return by_standard_name[0]
    if len(by_standard_name) > 1:
        raise InputError(
            f"Several variables have standard_name {accepted[0]!r}: {by_standard_name}; choose one with variable="
        )
    if IRENE_VARIABLE in ds.data_vars:
        return IRENE_VARIABLE
    raise InputError(
        f"Variable '{IRENE_VARIABLE}' not found, and no variable has standard_name {accepted[0]!r}. "
        f"Available: {available}"
    )


def _axes(da: xr.DataArray, irene: bool) -> tuple[str, str, str, bool]:
    """Return the time, y and x dims of a 3-D variable, and whether the time dim has datetime values.

    In CF mode the time dim is the one with a datetime coordinate and y/x come from CF
    axis attributes where those identify them; whatever is not identified is taken
    positionally from (time, y, x). In IRENE mode everything is positional.
    """
    dims = [str(d) for d in da.dims]
    datetime_dims = [d for d in dims if d in da.coords and np.issubdtype(da[d].dtype, np.datetime64)]
    if not irene and len(datetime_dims) == 1:
        t_dim = datetime_dims[0]
    else:
        t_dim = dims[0]
        if t_dim.lower() not in IRENE_TIME_DIMS and da.shape[0] >= da.shape[1]:
            raise InputError(
                f"First dimension should be time, got dims {da.dims} with shape {da.shape}. "
                "Expected shape (T, H, W) where T < H and T < W."
            )
    rest = [d for d in dims if d != t_dim]
    y_dim, x_dim = rest
    if not irene:
        cf_axes = da.cf.axes
        y_cf = [d for d in cf_axes.get("Y", []) if d in rest]
        x_cf = [d for d in cf_axes.get("X", []) if d in rest]
        if len(y_cf) == 1 and len(x_cf) == 1 and y_cf != x_cf:
            y_dim, x_dim = y_cf[0], x_cf[0]
    has_datetime = not irene and t_dim in datetime_dims
    return t_dim, y_dim, x_dim, has_datetime


# --- output ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictOptions:
    """The parsed query parameters of one prediction."""

    variable: str | None
    forecast_steps: int
    ensemble_size: int
    format: Literal["irene", "cf"]
    products: tuple[str, ...]
    quantiles: tuple[float, ...]
    thresholds: tuple[float, ...]
    dtype: Literal["float32", "float16"]
    complevel: int
    seed: int | None
    mask: bool


def ensemble_products(
    out: xr.Dataset,
    names: Sequence[str],
    products: Sequence[str],
    quantiles: Sequence[float],
    thresholds: Sequence[float],
) -> xr.Dataset:
    """Reduce a member forecast to the requested ensemble products.

    Parameters
    ----------
    out : xarray.Dataset
        The forecast, with each of ``names`` dimensioned ``(ensemble_member, time, y, x)``.
    names : sequence of str
        The forecast variables.
    products : sequence of str
        Any of :data:`PRODUCTS`.
    quantiles : sequence of float
        Quantile levels in [0, 1], for ``'quantiles'``.
    thresholds : sequence of float
        Thresholds in the variable's units, for ``'exceedance'``.

    Returns
    -------
    xarray.Dataset
        For each variable ``v``: ``v`` itself (``members``), ``v_mean``,
        ``v_quantile(quantile, time, y, x)`` and
        ``v_exceedance_probability(threshold, time, y, x)`` (the fraction of members
        strictly above the threshold), as requested. Where every member is NaN (a
        masked point) every product is NaN. ``ensemble_member`` is dropped when no
        variable uses it any more.
    """
    result = out.copy()
    for name in names:
        da = out[name]
        values = da.values
        missing = np.isnan(values).all(axis=0)
        base = {k: v for k, v in da.attrs.items() if k in ("standard_name", "units", "long_name", "grid_mapping")}
        dims = da.dims[1:]
        if "mean" in products:
            with np.errstate(invalid="ignore"):
                mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
            result[f"{name}_mean"] = xr.Variable(dims, mean, {**base, "cell_methods": "ensemble_member: mean"})
        if "quantiles" in products:
            levels = np.asarray(quantiles, dtype=np.float64)
            q = np.quantile(values, levels, axis=0).astype(np.float32)
            result = result.assign_coords(
                quantile=xr.Variable("quantile", levels, {"long_name": "quantile level", "units": "1"})
            )
            result[f"{name}_quantile"] = xr.Variable(
                ("quantile", *dims), q, {**base, "long_name": f"{base.get('long_name', name)}, ensemble quantile"}
            )
        if "exceedance" in products:
            levels = np.asarray(thresholds, dtype=np.float64)
            probability = np.stack([(values > level).mean(axis=0) for level in levels]).astype(np.float32)
            probability[:, missing] = np.nan
            threshold_attrs = {"long_name": "threshold"}
            if "units" in base:
                threshold_attrs["units"] = base["units"]
            result = result.assign_coords(threshold=xr.Variable("threshold", levels, threshold_attrs))
            prob_attrs = {
                "long_name": f"probability of {base.get('standard_name', name)} above threshold",
                "units": "1",
            }
            if "grid_mapping" in base:
                prob_attrs["grid_mapping"] = base["grid_mapping"]
            result[f"{name}_exceedance_probability"] = xr.Variable(("threshold", *dims), probability, prob_attrs)
        if "members" not in products:
            result = result.drop_vars(name)
    if not any("ensemble_member" in result[v].dims for v in result.data_vars):
        result = result.drop_vars("ensemble_member", errors="ignore")
    return result


def irene_layout(out: xr.Dataset, name: str, spec_units: str, attrs: Mapping[str, Any]) -> xr.Dataset:
    """Re-shape a forecast into IRENE's bare layout.

    Parameters
    ----------
    out : xarray.Dataset
        The forecast of :func:`forecast`.
    name : str
        Its forecast variable.
    spec_units : str
        The variable's units, converted to mm/h.
    attrs : mapping
        Global attributes.

    Returns
    -------
    xarray.Dataset
        ``precipitation_forecast(ensemble_member, forecast_step, y, x)`` in mm/h, with
        no coordinates, as IRENE writes it.
    """
    factor = unit_conversion_factor(spec_units, "mm h-1")
    if factor is None:
        raise InputError(f"format=irene returns mm/h, but this model forecasts {spec_units!r}; use format=cf")
    values = out[name].values
    if factor != 1.0:
        values = values * np.float32(factor)
    return xr.Dataset(
        {
            IRENE_OUTPUT_VARIABLE: xr.DataArray(
                values.astype(np.float32, copy=False),
                dims=["ensemble_member", "forecast_step", "y", "x"],
                attrs={"units": "mm/h", "long_name": "Ensemble precipitation forecast"},
            )
        },
        attrs=dict(attrs),
    )


# --- the app ---------------------------------------------------------------------------


def create_app(config: ServeConfig, loaded: LoadedModel | None = None) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    config : ServeConfig
        The parsed configuration.
    loaded : LoadedModel, optional
        The model, already loaded (as :func:`main` does, before binding the port). If
        ``None``, it is loaded when the app starts.

    Returns
    -------
    fastapi.FastAPI
    """
    state = ServerState(config=config, loaded=loaded)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        if state.loaded is None:
            state.loaded = load_model(config)
        manifest = state.loaded.manifest
        state.metrics.model_info.labels(
            name=str(manifest.get("name")),
            model_version=str(manifest.get("model_version")),
            model_sha256=str(manifest.get("model_sha256")),
            mlcast_version=mlcast_version(),
            builder_version=__version__,
        ).set(1)
        threading.Thread(target=_warm_up, args=(state,), name="warm-up", daemon=True).start()
        yield

    app = FastAPI(
        title="mlcast nowcasting API",
        version=__version__,
        description=(
            "Ensemble nowcasting from a netCDF of past fields, served by mlcast-inference-container-builder. "
            "Compatible with the IRENE (ConvGRU-Ensemble) API: `/health`, `/model/info` and `/predict` with a "
            "multipart `file` upload behave as IRENE's."
        ),
        lifespan=lifespan,
    )
    app.state.server = state
    app.add_middleware(_RequestMiddleware, state=state)

    @app.get("/health", tags=["health"], summary="IRENE-compatible health check")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": state.loaded is not None}

    @app.get("/healthz", tags=["health"], summary="Liveness: the process answers")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get(
        "/readyz",
        tags=["health"],
        summary="Readiness: the model is loaded and warmed up",
        responses={503: {"description": "Not ready yet (loading, warming up) or the warm-up failed"}},
    )
    async def readyz() -> JSONResponse:
        if state.ready.is_set():
            return JSONResponse({"status": "ready"})
        if state.warm_up_error is not None:
            return JSONResponse({"status": "warm-up failed", "detail": state.warm_up_error}, status_code=503)
        return JSONResponse({"status": "warming up"}, status_code=503)

    @app.get("/metrics", tags=["health"], summary="Prometheus metrics")
    async def metrics() -> Response:
        return Response(generate_latest(state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model/info", tags=["model"], summary="Model metadata (a superset of IRENE's)")
    async def model_info() -> dict[str, Any]:
        if state.loaded is None:
            return {"error": "Model not loaded"}
        return _model_info(state)

    raw_body = {"schema": {"type": "string", "format": "binary"}}

    @app.api_route(
        "/predict",
        methods=["POST", "PUT"],
        tags=["model"],
        summary="Forecast from a netCDF of past fields",
        response_class=FileResponse,
        responses={
            200: {"content": {NETCDF_MEDIA_TYPE: {}}, "description": "The forecast, as `predictions.nc`"},
            413: {"description": "Upload larger than MLCAST_MAX_UPLOAD_MB"},
            415: {"description": "Raw body with an unsupported Content-Type"},
            422: {"description": "Invalid parameters, or an input the model cannot use"},
            429: {"description": "Too many predictions in progress; retry after Retry-After seconds"},
            503: {"description": "Waited longer than MLCAST_REQUEST_TIMEOUT for the model"},
        },
        openapi_extra={
            "requestBody": {
                "content": {"application/x-netcdf": raw_body, "application/octet-stream": raw_body},
            }
        },
    )
    async def predict(
        request: Request,
        file: UploadFile | None = File(  # noqa: B008
            None, description="netCDF file with past fields (T, H, W); or send the file as the raw request body"
        ),
        variable: str | None = Query(  # noqa: B008
            None,
            description="Name of the input variable. Default: `RR` (format=irene: `RR` first, then the model's "
            "standard_name; format=cf: the model's standard_name first, then `RR`)",
        ),
        forecast_steps: int | None = Query(  # noqa: B008
            None, ge=1, le=config.max_forecast_steps, description="Number of steps to forecast (default: the model's)"
        ),
        ensemble_size: int | None = Query(  # noqa: B008
            None, ge=1, le=config.max_ensemble_size, description="Number of ensemble members (default: the model's)"
        ),
        format: Literal["irene", "cf"] = Query(  # noqa: A002, B008
            "irene", description="`irene`: IRENE's bare layout; `cf`: CF-complete with time, lead_time and grid"
        ),
        products: str = Query(  # noqa: B008
            "members", description=f"format=cf: comma-separated subset of {', '.join(PRODUCTS)}"
        ),
        quantiles: str = Query(  # noqa: B008
            ",".join(f"{q:g}" for q in DEFAULT_QUANTILES), description="Quantile levels for products=quantiles"
        ),
        thresholds: str = Query(  # noqa: B008
            ",".join(f"{t:g}" for t in DEFAULT_THRESHOLDS),
            description="Thresholds (in the variable's units, mm/h for rain rate) for products=exceedance",
        ),
        dtype: Literal["float32", "float16"] = Query(  # noqa: B008
            "float32", description="Storage type; float16 is CF-packed int16 (scale_factor/add_offset)"
        ),
        complevel: int = Query(4, ge=0, le=9, description="zlib compression level"),  # noqa: B008
        seed: int | None = Query(None, ge=0, description="Seed for the ensemble noise (reproducible)"),  # noqa: B008
        mask: bool = Query(  # noqa: B008
            False, description="Set the forecast to NaN where the input had no valid value in any input step"
        ),
    ) -> Response:
        loaded = state.loaded
        assert loaded is not None  # the lifespan has loaded it
        started = time.perf_counter()
        options = _parse_options(
            loaded.spec,
            variable=variable,
            forecast_steps=forecast_steps,
            ensemble_size=ensemble_size,
            format=format,
            products=products,
            quantiles=quantiles,
            thresholds=thresholds,
            dtype=dtype,
            complevel=complevel,
            seed=seed,
            mask=mask,
        )
        request_id = request.state.request_id
        try:
            with state.gate.admit():
                payload: IO[bytes]
                if file is not None:
                    if file.size is not None and file.size > config.max_upload_bytes:
                        raise PayloadTooLargeError("too large")
                    payload = file.file
                else:
                    payload = await _read_raw_body(request, config.max_upload_bytes)
                try:
                    path = await run_in_threadpool(_predict, state, payload, options, request_id)
                finally:
                    payload.close()
        except QueueFullError as exc:
            logger.warning(f"Refusing a prediction: {exc}")
            raise HTTPException(
                status_code=429,
                detail=f"Too many predictions in progress ({exc}); retry later.",
                headers={"Retry-After": str(state.gate.retry_after())},
            ) from exc
        except QueueTimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"The model is busy: {exc} (MLCAST_REQUEST_TIMEOUT); retry later.",
                headers={"Retry-After": str(state.gate.retry_after())},
            ) from exc
        except PayloadTooLargeError as exc:
            raise HTTPException(
                status_code=413, detail=f"File too large. Maximum is {config.max_upload_mb} MB."
            ) from exc
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ForecastInputError as exc:
            detail = format_problems("The input does not match what the model expects:", exc.problems)
            raise HTTPException(status_code=422, detail=detail) from exc

        elapsed = time.perf_counter() - started
        headers = {
            "Content-Disposition": f"attachment; filename={OUTPUT_FILENAME}",
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
            **_model_headers(loaded.manifest),
        }
        return FileResponse(
            path, media_type=NETCDF_MEDIA_TYPE, headers=headers, background=BackgroundTask(_unlink, path)
        )

    return app


def _parse_options(spec: InferenceSpec, **params: Any) -> PredictOptions:
    """Validate the query parameters FastAPI cannot check by type alone; 422 on a problem."""
    problems: list[str] = []

    def floats(raw: str, name: str, low: float | None = None, high: float | None = None) -> tuple[float, ...]:
        try:
            values = tuple(float(item) for item in raw.split(",") if item.strip())
        except ValueError:
            problems.append(f"{name} must be comma-separated numbers, got {raw!r}")
            return ()
        if not values:
            problems.append(f"{name} must list at least one number")
        bad = [
            v
            for v in values
            if not math.isfinite(v) or (low is not None and v < low) or (high is not None and v > high)
        ]
        if bad:
            bounds = f" between {low:g} and {high:g}" if low is not None and high is not None else ""
            problems.append(f"{name} must be finite numbers{bounds}, got {bad}")
        return values

    products = tuple(dict.fromkeys(p.strip() for p in params["products"].split(",") if p.strip()))
    unknown = [p for p in products if p not in PRODUCTS]
    if unknown or not products:
        problems.append(
            f"products must be a comma-separated subset of {', '.join(PRODUCTS)}, got {params['products']!r}"
        )
    if params["format"] == "irene" and products != ("members",) and not unknown:
        problems.append("format=irene returns ensemble members only; use format=cf for other products")
    quantiles = floats(params["quantiles"], "quantiles", 0.0, 1.0) if "quantiles" in products else ()
    thresholds = floats(params["thresholds"], "thresholds") if "exceedance" in products else ()
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))
    return PredictOptions(
        variable=params["variable"],
        forecast_steps=params["forecast_steps"] or spec.forecast_steps,
        ensemble_size=params["ensemble_size"] or spec.default_ensemble_size,
        format=params["format"],
        products=products,
        quantiles=quantiles,
        thresholds=thresholds,
        dtype=params["dtype"],
        complevel=params["complevel"],
        seed=params["seed"],
        mask=params["mask"],
    )


async def _read_raw_body(request: Request, max_bytes: int) -> IO[bytes]:
    """Copy a raw request body into a spooled temporary file, enforcing the size limit as it streams.

    Raises
    ------
    HTTPException
        415 for a Content-Type that is not netCDF or octet-stream, 422 for a multipart
        upload without a ``file`` field.
    PayloadTooLargeError
        If the body (or its declared Content-Length) is over ``max_bytes``.
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type.startswith("multipart/") or not content_type:  # IRENE: a missing `file` is 422
        raise HTTPException(
            status_code=422,
            detail="No file uploaded: send the netCDF as the multipart 'file' field, or as the raw body with "
            "Content-Type application/x-netcdf.",
        )
    if content_type not in RAW_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Send the netCDF as a multipart 'file' field, or as the raw body with Content-Type "
            f"application/x-netcdf or application/octet-stream; got {content_type or 'no Content-Type'!r}.",
        )
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise PayloadTooLargeError("too large")
    spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_MAX_MEMORY)  # noqa: SIM115 - closed by the caller
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_bytes:
                raise PayloadTooLargeError("too large")
            spool.write(chunk)
    except BaseException:
        spool.close()
        raise
    spool.seek(0)
    return spool


def _predict(state: ServerState, payload: IO[bytes], options: PredictOptions, request_id: str) -> str:
    """Read, forecast and write one prediction (in the threadpool); return the output file's path."""
    loaded = state.loaded
    assert loaded is not None
    spec = loaded.spec
    irene = options.format == "irene"

    tic = time.perf_counter()
    ds = read_netcdf_bytes(payload, state.config.max_upload_bytes)
    prepared = prepare_input(ds, spec, options.variable, irene)
    read_seconds = time.perf_counter() - tic

    with state.gate.model():
        tic = time.perf_counter()
        out = forecast(
            loaded.model,
            spec,
            prepared.ds,
            forecast_steps=options.forecast_steps,
            ensemble_size=options.ensemble_size,
            seed=options.seed,
        )
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)
        inference_seconds = time.perf_counter() - tic
    state.gate.record_inference(inference_seconds)

    names = [prepared.name] if prepared.name is not None else [n for n in out.data_vars if out[n].ndim == 4]
    if options.mask and prepared.valid is not None:
        for name in names:
            out[name] = out[name].where(prepared.valid)

    provenance = _provenance(loaded.manifest, options, request_id)
    if irene:
        assert prepared.name is not None
        attrs = {
            "model": loaded.manifest.get("name") or type(getattr(loaded.model, "network", loaded.model)).__name__,
            "forecast_steps": options.forecast_steps,
            "ensemble_size": options.ensemble_size,
            "elapsed_seconds": f"{read_seconds + inference_seconds:.3f}",
            **provenance,
        }
        out = irene_layout(out, prepared.name, spec.units[0], attrs)
    else:
        out = ensemble_products(out, names, options.products, options.quantiles, options.thresholds)
        history = f"{out.attrs.pop('history')}\n" if "history" in out.attrs else ""
        out.attrs.update(provenance)
        out.attrs["history"] = history + provenance["history"]
        if prepared.synthetic_time:
            out.attrs["comment"] = (
                "The input had no datetime time coordinate: its times were assumed to be consecutive "
                f"{spec.timestep or _FALLBACK_TIMESTEP} steps ending at the request time"
            )
    if options.mask:
        out.attrs["mask"] = "the forecast is NaN where the input had no valid value in any input step"

    fd, path = tempfile.mkstemp(prefix="mlcast-icb-", suffix=".nc")
    os.close(fd)
    try:
        write_netcdf(out, path, complevel=options.complevel, dtype=options.dtype)
    except BaseException:
        _unlink(path)
        raise
    logger.info(
        f"Forecast {options.ensemble_size} members x {options.forecast_steps} steps of "
        f"{dict(prepared.ds.sizes)} (format={options.format}, products={','.join(options.products)}, "
        f"seed={options.seed}): read {read_seconds:.2f} s, inference {inference_seconds:.2f} s, "
        f"{os.path.getsize(path) / 1024**2:.2f} MiB out"
    )
    return path


def _provenance(manifest: Mapping[str, Any], options: PredictOptions, request_id: str) -> dict[str, Any]:
    """Global attributes recording which model, builder and request produced a forecast."""
    created = datetime.now(UTC).isoformat(timespec="seconds")
    attrs = {
        "source": f"{DISTRIBUTION} {__version__}",
        "model_name": manifest.get("name"),
        "model_version": manifest.get("model_version"),
        "model_sha256": manifest.get("model_sha256"),
        "mlcast_version": mlcast_version(),
        "builder_version": __version__,
        "request_id": request_id,
        "seed": options.seed,
        "history": f"{created}: forecast by {DISTRIBUTION} {__version__} serve (request {request_id})",
    }
    return {key: value for key, value in attrs.items() if value is not None}


def _model_headers(manifest: Mapping[str, Any]) -> dict[str, str]:
    """The model-identity response headers, leaving out what the manifest does not record."""
    headers = {
        "X-Model-Sha256": manifest.get("model_sha256"),
        "X-Model-Name": manifest.get("name"),
        "X-Model-Version": manifest.get("model_version"),
        "X-Mlcast-Version": mlcast_version(),
        "X-Builder-Version": __version__,
    }
    return {key: str(value) for key, value in headers.items() if value is not None}


def _model_info(state: ServerState) -> dict[str, Any]:
    """``/model/info``: IRENE's fields where the model has them, then the bundle's identity and spec."""
    loaded = state.loaded
    assert loaded is not None
    spec, manifest = loaded.spec, loaded.manifest
    network = getattr(loaded.model, "network", loaded.model)
    hparams = getattr(loaded.model, "hparams", {}) or {}
    return {
        # IRENE's keys
        "architecture": f"{type(network).__module__}.{type(network).__name__}",
        "input_channels": getattr(network, "input_channels", len(spec.standard_names)),
        "num_blocks": getattr(network, "num_blocks", None),
        "forecast_steps": spec.forecast_steps,
        "ensemble_size": spec.default_ensemble_size,
        "noisy_decoder": getattr(network, "noisy_decoder", None),
        "loss_class": str(hparams.get("loss_class")) if "loss_class" in hparams else None,
        "device": str(loaded.device),
        # Ours
        "name": manifest.get("name"),
        "model_version": manifest.get("model_version"),
        "model_sha256": manifest.get("model_sha256"),
        "mlcast_version": mlcast_version(),
        "bundle_mlcast_version": manifest.get("mlcast_version"),
        "builder_version": __version__,
        "standard_names": list(spec.standard_names),
        "units": list(spec.units),
        "input_steps": spec.input_steps,
        "timestep": spec.timestep,
        "spatial_divisor": spec.spatial_divisor,
        "max_forecast_steps": state.config.max_forecast_steps,
        "max_ensemble_size": state.config.max_ensemble_size,
        "ready": state.ready.is_set(),
    }


def _warm_up(state: ServerState) -> None:
    """Run one small forecast through the full input path, then mark the server ready."""
    loaded = state.loaded
    assert loaded is not None
    spec = loaded.spec
    size = 2 * spec.spatial_divisor
    tic = time.perf_counter()
    try:
        if len(spec.standard_names) == 1:
            dummy = xr.Dataset(
                {IRENE_VARIABLE: (("time", "y", "x"), np.zeros((spec.input_steps, size, size), np.float32))}
            )
            ds = prepare_input(dummy, spec, IRENE_VARIABLE, irene=True).ds
        else:
            ds = _dummy_cf_input(spec, size)
        with state.gate.model():
            forecast(loaded.model, spec, ds, forecast_steps=1, ensemble_size=1, seed=0)
    except Exception as exc:  # noqa: BLE001 - reported through /readyz and the log
        state.warm_up_error = f"{type(exc).__name__}: {exc}"
        logger.opt(exception=True).error(f"Warm-up forecast failed; /readyz stays 503: {state.warm_up_error}")
        return
    state.ready.set()
    state.metrics.ready.set(1)
    logger.info(f"Warm-up forecast took {time.perf_counter() - tic:.2f} s; ready")


def _dummy_cf_input(spec: InferenceSpec, size: int) -> xr.Dataset:
    """An all-zero CF input with one variable per standard name, for a multi-variable model's warm-up."""
    step = pd.Timedelta(spec.timestep or _FALLBACK_TIMESTEP)
    times = pd.date_range("2000-01-01", periods=spec.input_steps, freq=step)
    data_vars = {
        f"v{i}": (
            ("time", "y", "x"),
            np.zeros((spec.input_steps, size, size), np.float32),
            {"standard_name": sn, "units": u},
        )
        for i, (sn, u) in enumerate(zip(spec.standard_names, spec.units, strict=True))
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": ("time", times, {"standard_name": "time", "axis": "T"}),
            "y": ("y", np.arange(size, dtype=float), {"axis": "Y"}),
            "x": ("x", np.arange(size, dtype=float), {"axis": "X"}),
        },
    )


def _unlink(path: str) -> None:
    """Delete a temporary output file, ignoring one that is already gone."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class _RequestMiddleware:
    """Pure ASGI middleware: request ids, the request log line and the request metrics.

    The request id is the incoming ``X-Request-Id`` when it looks like one, else a new
    one; it is echoed in the response, bound to every log record of the request (the
    threadpool inherits the context), and recorded in the forecast's attributes.
    """

    def __init__(self, app: Callable, state: ServerState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = ""
        for key, value in scope.get("headers", []):
            if key == b"x-request-id":
                incoming = value.decode("latin-1")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        started = time.perf_counter()

        async def send_with_id(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", request_id.encode("latin-1"))]
            await send(message)

        with logger.contextualize(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            finally:
                duration = time.perf_counter() - started
                route = scope.get("route")
                endpoint = getattr(route, "path", None) or "unmatched"
                method = scope.get("method", "")
                self.state.metrics.requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
                self.state.metrics.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
                if endpoint not in ("/healthz", "/readyz", "/health", "/metrics"):
                    logger.info(f"{method} {scope.get('path')} {status} in {duration:.3f} s")
                else:
                    logger.debug(f"{method} {scope.get('path')} {status} in {duration:.3f} s")


# --- entrypoint ------------------------------------------------------------------------


def configure_serve_logging(level: str) -> None:
    """Point loguru at stderr with the request id in every record, and route uvicorn's logging through it.

    Parameters
    ----------
    level : str
        A loguru level name; an unknown one warns and falls back to ``INFO``.
    """
    logger.remove()
    logger.configure(extra={"request_id": "-"})
    colorize = sys.stderr.isatty()
    try:
        logger.add(sys.stderr, level=level, format=SERVE_LOG_FORMAT, colorize=colorize)
    except ValueError:
        logger.add(sys.stderr, level="INFO", format=SERVE_LOG_FORMAT, colorize=colorize)
        logger.warning(f"Log level {level!r} is not a known level; using INFO")
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        std = logging.getLogger(name)
        std.handlers = [_InterceptHandler()]
        std.setLevel(logging.DEBUG)  # loguru's sink filters by the log level
        std.propagate = False


class _InterceptHandler(logging.Handler):
    """Forward standard-library log records (uvicorn's) to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Run Mode 2 from the command line and the environment, and return a process exit code.

    Parameters
    ----------
    argv : sequence of str, optional
        The arguments after ``serve``. Defaults to none.
    env : mapping of str to str, optional
        Environment to read. Defaults to :data:`os.environ`.

    Returns
    -------
    int
        0 after a clean shutdown (and after ``--help``); 2 for a configuration error, 4
        for an unusable device and 6 for a broken bundle, all before the port is bound;
        1 if the server cannot start (the port is taken, say) or for anything unexpected.
    """
    env = os.environ if env is None else env
    configure_serve_logging(log_level(env.get(f"{ENV_PREFIX}_LOG_LEVEL")))
    logger.info(f"Starting {DISTRIBUTION} {__version__} (serve)")
    try:
        parsed = parse_options("serve", argv, env)
    except SystemExit as exc:  # argparse has printed the usage and the error, or the help
        return int(exc.code or 0)
    try:
        config = ServeConfig.from_options(parsed)
    except MlcastIcbError as exc:
        logger.error(f"{exc}")
        return exc.exit_code

    configure_serve_logging(config.log_level)
    log_lines("Effective configuration (defaults included):", parsed.config_lines(describe_config(config)))
    for name in parsed.ignored:
        logger.debug(f"Ignoring {name}: it is an option of `run`, not of `serve`")
    try:
        loaded = load_model(config)
    except MlcastIcbError as exc:
        logger.error(f"{exc}")
        return exc.exit_code
    except Exception:
        logger.opt(exception=True).error("Unexpected failure")
        return 1

    app = create_app(config, loaded)
    server = uvicorn.Server(
        uvicorn.Config(app, host=config.host, port=config.port, log_config=None, access_log=False, workers=1)
    )
    logger.info(f"Listening on http://{config.host}:{config.port} (OpenAPI docs at /docs)")
    try:
        server.run()
    except SystemExit as exc:  # uvicorn exits when it cannot bind
        logger.error(f"The server could not start (exit {exc.code}); see the log above")
        return 1
    if not server.started:
        logger.error("The server could not start; see the log above")
        return 1
    return 0
