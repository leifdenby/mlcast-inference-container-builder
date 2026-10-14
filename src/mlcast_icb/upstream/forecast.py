"""Run a model on a CF-compliant xarray Dataset and return a CF-compliant ensemble forecast.

:func:`forecast` is the xarray-level inference entry point: it finds the input variables
by CF standard name and the time/y/x axes by CF attributes, checks units and time
spacing, normalises with :func:`~.processing.preprocess`, runs the network, and returns
the denormalised (and clamped) forecast as an ``xarray.Dataset`` with valid times,
forecast reference time, lead times, the input's grid and grid mapping.

Proposed for mlcast as ``mlcast/inference/forecast.py`` (see ``README.md``, PR D).

Padding
-------
The input is padded at the bottom and right to a multiple of ``spec.spatial_divisor``
with ``scaling.norm_min`` (no rain) *after* normalisation, and the output is cropped back.
``ConvGruModel.forward`` would otherwise pad with ``0.0`` in normalised space, i.e. 30 dBZ
(PR H); since the input already has a valid size, the network's own padding never runs.

Seeding
-------
The noisy decoder draws its noise from the global torch RNG (``torch.randn`` in
``ConvGruModel.forward``). A ``seed`` is therefore applied inside
``torch.random.fork_rng``, which restores the global RNG state afterwards. The global
RNG is still shared while the forecast runs, so concurrent seeded forecasts in several
threads must be serialised by the caller until ``ConvGruModel.forward`` takes a
``generator`` (PR D).
"""

import contextlib
import datetime
from collections.abc import Iterator, Sequence
from typing import Any

import cf_xarray  # noqa: F401  (registers the .cf accessor)
import numpy as np
import pandas as pd
import torch
import xarray as xr
from beartype import beartype
from jaxtyping import Float32, jaxtyped
from loguru import logger

import mlcast
from mlcast.nowcasting_module import NowcastLightningModule

from .processing import postprocess, preprocess
from .spec import UNITS, InferenceSpec

#: Accepted spellings of each canonical unit (the canonical spellings are those in ``spec.UNITS``).
UNIT_ALIASES: dict[str, frozenset[str]] = {
    "mm h-1": frozenset({"mm h-1", "mm/h", "mm hr-1", "mm/hr", "mm h**-1", "mm h^-1", "mm.h-1", "mm hour-1"}),
    "kg m-2 s-1": frozenset({"kg m-2 s-1", "kg/m2/s", "kg m**-2 s**-1", "kg m^-2 s^-1", "kg.m-2.s-1"}),
    "kg m-2": frozenset({"kg m-2", "kg/m2", "kg m**-2", "kg m^-2", "mm"}),
    "dBZ": frozenset({"dBZ", "dbz", "dBz"}),
}

#: Multiplicative factors converting data in the first unit to the second.
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("kg m-2 s-1", "mm h-1"): 3600.0,  # 1 kg m-2 of water is 1 mm
    ("mm h-1", "kg m-2 s-1"): 1.0 / 3600.0,
}

#: Standard names whose data can stand in for another's after a unit conversion.
COMPATIBLE_STANDARD_NAMES: dict[str, tuple[str, ...]] = {
    "rainfall_rate": ("rainfall_flux",),
    "rainfall_flux": ("rainfall_rate",),
}

# Default standard names for the horizontal coordinates, if the input's lack one.
_DEFAULT_COORD_STANDARD_NAMES = {
    ("Y", True): "latitude",
    ("X", True): "longitude",
    ("Y", False): "projection_y_coordinate",
    ("X", False): "projection_x_coordinate",
}


class ForecastInputError(ValueError):
    """The input Dataset cannot be used for a forecast.

    Parameters
    ----------
    problems : Sequence[str]
        Every problem found, each a self-contained sentence. All are reported in the message.
    """

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Invalid forecast input:\n" + "\n".join(f"- {p}" for p in self.problems))


def canonical_units(units: str) -> str | None:
    """Return the canonical spelling of ``units``, or ``None`` if it is not a known spelling.

    Parameters
    ----------
    units : str
        A units string, e.g. ``"mm/h"``.

    Returns
    -------
    canonical : str or None
        A key of ``UNIT_ALIASES``, e.g. ``"mm h-1"``.
    """
    normalised = " ".join(units.split())
    for canonical, aliases in UNIT_ALIASES.items():
        if normalised in aliases:
            return canonical
    return None


def unit_conversion_factor(from_units: str, to_units: str) -> float | None:
    """Return the factor that converts data in ``from_units`` to ``to_units``.

    Parameters
    ----------
    from_units, to_units : str
        Units strings, in any spelling listed in ``UNIT_ALIASES``.

    Returns
    -------
    factor : float or None
        ``1.0`` for equivalent spellings, the factor from ``UNIT_CONVERSIONS`` for convertible
        units, ``None`` if the units are unknown or incompatible.
    """
    source = canonical_units(from_units) or from_units
    target = canonical_units(to_units) or to_units
    if source == target:
        return 1.0
    return UNIT_CONVERSIONS.get((source, target))


def _find_variable(ds: xr.Dataset, standard_name: str, problems: list[str]) -> str | None:
    """Return the name of the one variable with ``standard_name`` (or a compatible one)."""
    by_standard_name = ds.cf.standard_names
    for candidate in (standard_name, *COMPATIBLE_STANDARD_NAMES.get(standard_name, ())):
        names = [name for name in by_standard_name.get(candidate, []) if name in ds.data_vars]
        if len(names) > 1:
            problems.append(f"Several variables have standard_name {candidate!r}: {sorted(names)}")
            return None
        if names:
            if candidate != standard_name:
                logger.info(f"Using variable {names[0]!r} ({candidate}) as {standard_name}")
            return names[0]
    alternatives = COMPATIBLE_STANDARD_NAMES.get(standard_name, ())
    alternatives_text = f" (or {', '.join(repr(a) for a in alternatives)})" if alternatives else ""
    problems.append(
        f"No variable with standard_name {standard_name!r}{alternatives_text}; "
        f"data variables are {sorted(str(name) for name in ds.data_vars)}"
    )
    return None


def _detect_axes(da: xr.DataArray) -> tuple[str, str, str]:
    """Return the time, y and x dimension names of ``da`` from its CF attributes.

    Like ``mlcast.data.source_data_datasets._detect_axes``, but a horizontal axis is also
    found by a ``latitude``/``longitude`` coordinate, and no axis is guessed from its name.

    Parameters
    ----------
    da : xr.DataArray
        A variable with CF time and horizontal coordinates.

    Returns
    -------
    t_dim, y_dim, x_dim : str
        Dimension names.

    Raises
    ------
    KeyError
        Naming each axis that could not be identified.
    """
    missing = []
    try:
        t_dim = da.cf["time"].dims[0]
    except KeyError:
        t_dim = None
        missing.append("time (T)")

    found = {}
    for axis, geographic in (("Y", "latitude"), ("X", "longitude")):
        candidates = list(da.cf.axes.get(axis, []))
        if not candidates:
            candidates = [name for name in da.cf.coordinates.get(geographic, []) if name in da.dims]
        if len(candidates) == 1 and candidates[0] in da.dims:
            found[axis] = candidates[0]
        else:
            missing.append(f"{axis} (axis={axis!r} or a standard_name such as projection_{axis.lower()}_coordinate)")
    if missing:
        raise KeyError(", ".join(missing))
    return str(t_dim), str(found["Y"]), str(found["X"])


def _dataset_time_dim(ds: xr.Dataset) -> str | None:
    """The dimension of ``ds``'s CF time coordinate, used to check times when no input variable was found."""
    try:
        time = ds.cf["time"]
    except KeyError:
        return None
    return str(time.dims[0]) if time.ndim == 1 else None


def _clean_variable(variable: xr.Variable) -> xr.Variable:
    """A copy of ``variable`` without its (input-specific) encoding."""
    return xr.Variable(variable.dims, variable.values, dict(variable.attrs))


@contextlib.contextmanager
def _seeded(seed: int | None, device: torch.device) -> Iterator[None]:
    """Seed the global torch RNG of ``device`` (and the CPU) for the duration of the block.

    The RNG state is restored afterwards. With ``seed=None``, nothing is changed.
    """
    if seed is None:
        yield
        return
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index]):
            torch.default_generator.manual_seed(seed)
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
            yield
    elif device.type == "cpu":
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            yield
    else:
        with torch.random.fork_rng(devices=[0], device_type=device.type):
            torch.manual_seed(seed)
            yield


@jaxtyped(typechecker=beartype)
def _pad(
    data: Float32[np.ndarray, "time channels height width"], divisor: int, fill_value: float
) -> Float32[np.ndarray, "time channels padded_height padded_width"]:
    """Pad H and W at the bottom/right to a multiple of ``divisor`` with ``fill_value``."""
    height, width = data.shape[-2:]
    pad_h = -height % divisor
    pad_w = -width % divisor
    if pad_h == 0 and pad_w == 0:
        return data
    return np.pad(data, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=fill_value)


def forecast(
    model: NowcastLightningModule,
    spec: InferenceSpec,
    ds: xr.Dataset,
    *,
    forecast_steps: int | None = None,
    ensemble_size: int | None = None,
    seed: int | None = None,
    reference_time: np.datetime64 | None = None,
) -> xr.Dataset:
    """Forecast from the latest ``spec.input_steps`` fields in a CF-compliant Dataset.

    Each of ``spec.standard_names`` is looked up by its CF ``standard_name`` attribute
    (``ds.cf[standard_name]``); ``rainfall_rate`` and ``rainfall_flux`` may stand in for one
    another. Callers that want to select a variable by name instead should rename it or
    set its ``standard_name`` before calling. Time, y and x are found by CF attributes
    (``axis``, ``standard_name``, or datetime values for time).

    Parameters
    ----------
    model : NowcastLightningModule
        The model, e.g. from ``load_model``. It is put in eval mode and run on ``model.device``.
    spec : InferenceSpec
        The model's inference spec.
    ds : xr.Dataset
        Input with one variable per standard name in ``spec.standard_names``, each with
        dims time, y and x (in any order) and a ``units`` attribute. Units in another
        spelling (``"mm/h"``) or convertible units (``kg m-2 s-1`` for ``mm h-1``) are
        accepted; missing units are assumed to be the spec's, with a warning. NaN is
        treated as no signal.
    forecast_steps : int or None, optional
        Number of steps to forecast. Default is ``spec.forecast_steps``.
    ensemble_size : int or None, optional
        Number of ensemble members. Default is ``spec.default_ensemble_size``.
    seed : int or None, optional
        Seed for the ensemble noise. The same seed gives the same forecast on the same
        device; the global torch RNG state is left unchanged. Default is ``None`` (draw
        from the global RNG).
    reference_time : np.datetime64 or None, optional
        Time of the last input field, which must be one of ``ds``'s times; the inputs are
        the ``spec.input_steps`` times ending here. Default is the last time in ``ds``.

    Returns
    -------
    forecast : xr.Dataset
        One float32 variable per input variable, with the input's name and dims
        ``("ensemble_member", "time", y, x)``, and attributes ``standard_name``, ``units``
        (the spec's), ``long_name`` and ``grid_mapping`` (if the input had one). Coordinates
        are ``ensemble_member``, the valid ``time``, ``lead_time`` along time, the scalar
        ``forecast_reference_time``, the input's horizontal coordinates and its grid
        mapping variable.

    Raises
    ------
    ForecastInputError
        With every problem found in the input: missing or ambiguous variables, axes that
        are not CF-identifiable, incompatible units, a missing reference time, too few
        input times or a time spacing other than ``spec.timestep``.
    """
    forecast_steps = spec.forecast_steps if forecast_steps is None else forecast_steps
    ensemble_size = spec.default_ensemble_size if ensemble_size is None else ensemble_size
    problems: list[str] = []
    if forecast_steps < 1:
        problems.append(f"forecast_steps must be at least 1, got {forecast_steps}")
    if ensemble_size < 1:
        problems.append(f"ensemble_size must be at least 1, got {ensemble_size}")

    # Variables, axes and units
    var_names: list[str] = []
    factors: list[float] = []
    axes: tuple[str, str, str] | None = None
    for standard_name, spec_units in zip(spec.standard_names, spec.units, strict=True):
        name = _find_variable(ds, standard_name, problems)
        if name is None:
            continue
        da = ds[name]
        try:
            var_axes = _detect_axes(da)
        except KeyError as err:
            problems.append(f"Variable {name!r}: cannot identify the CF axes {err.args[0]}")
            continue
        extra_dims = set(da.dims) - set(var_axes)
        if extra_dims:
            problems.append(f"Variable {name!r} has dims {da.dims}; only time, y and x {var_axes} are supported")
        if axes is None:
            axes = var_axes
        elif var_axes != axes:
            problems.append(f"Variable {name!r} has axes {var_axes}, unlike {axes} of {var_names[0]!r}")

        units = da.attrs.get("units")
        if units is None:
            # The units its standard name implies (which is a compatible one, not necessarily the spec's)
            units = UNITS.get(da.attrs.get("standard_name", standard_name), spec_units)
            logger.warning(f"Variable {name!r} has no units attribute; assuming {units!r}")
        factor = unit_conversion_factor(str(units), spec_units)
        if factor is None:
            problems.append(f"Variable {name!r} has units {units!r}, which cannot be converted to {spec_units!r}")
        var_names.append(name)
        factors.append(1.0 if factor is None else factor)

    # Time selection
    input_steps = spec.input_steps
    timestep = pd.Timedelta(spec.timestep) if spec.timestep is not None else None
    input_slice = None
    t_dim = axes[0] if axes is not None else _dataset_time_dim(ds)
    if t_dim is None:
        problems.append("The input has no CF time coordinate")
    else:
        times = pd.DatetimeIndex(ds[t_dim].values)
        if len(times) == 0:
            ref_index = None
            problems.append("The input has no times")
        elif reference_time is None:
            ref_index = len(times) - 1
        else:
            matches = np.flatnonzero(times == pd.Timestamp(reference_time))
            ref_index = int(matches[0]) if len(matches) == 1 else None
            if ref_index is None:
                problems.append(f"reference_time {reference_time} is not one of the input times ({len(times)} times)")
        if ref_index is not None:
            start = ref_index - input_steps + 1
            if start < 0:
                problems.append(
                    f"The model needs {input_steps} input times ending at {times[ref_index]}, "
                    f"but the input has only {ref_index + 1}"
                )
            input_times = times[max(start, 0) : ref_index + 1]
            spacings = np.unique(np.diff(input_times.values))
            if timestep is not None:
                wrong = [pd.Timedelta(s) for s in spacings if pd.Timedelta(s) != timestep]
                if wrong:
                    problems.append(
                        f"Input times must be {timestep} apart (spec.timestep {spec.timestep}), "
                        f"found spacing(s) {[str(w) for w in wrong]}"
                    )
            elif len(spacings) > 1:
                problems.append(f"Input times are not evenly spaced: {[str(pd.Timedelta(s)) for s in spacings]}")
            elif len(spacings) == 0:
                problems.append("Cannot infer the timestep from a single input time, and spec.timestep is not set")
            else:
                timestep = pd.Timedelta(spacings[0])
            if timestep is not None and timestep <= pd.Timedelta(0):
                problems.append(f"Input times must increase, found spacing {timestep}")
            input_slice = slice(max(start, 0), ref_index + 1)

    if problems:
        raise ForecastInputError(problems)
    assert axes is not None and input_slice is not None and timestep is not None
    t_dim, y_dim, x_dim = axes
    reference = pd.Timestamp(ds[t_dim].values[input_slice.stop - 1])

    # Preprocess, pad with "no rain", run, crop, postprocess
    arrays = [
        ds[name].isel({t_dim: input_slice}).transpose(t_dim, y_dim, x_dim).values * factor
        for name, factor in zip(var_names, factors, strict=True)
    ]
    data = preprocess(arrays, spec.standard_names, spec.scaling)
    height, width = data.shape[-2:]
    data = _pad(data, spec.spatial_divisor, spec.scaling.norm_min)

    model.eval()
    device = model.device
    x = torch.from_numpy(data).unsqueeze(0).to(device)
    with torch.no_grad(), _seeded(seed, device):
        preds = model.network(x, steps=forecast_steps, ensemble_size=ensemble_size)
    preds = preds[..., :height, :width]
    out = postprocess(preds, spec.standard_names, spec.scaling)[0].astype(np.float32, copy=False)  # (E, T, C, H, W)

    return _to_dataset(out, ds, spec, var_names, axes, reference, timestep, seed)


def _to_dataset(
    out: np.ndarray,
    ds: xr.Dataset,
    spec: InferenceSpec,
    var_names: Sequence[str],
    axes: tuple[str, str, str],
    reference: pd.Timestamp,
    timestep: pd.Timedelta,
    seed: int | None,
) -> xr.Dataset:
    """Wrap the ``(E, T, C, H, W)`` forecast in a CF-compliant Dataset (see :func:`forecast`)."""
    t_dim, y_dim, x_dim = axes
    n_members, n_steps = out.shape[:2]
    lead_times = np.arange(1, n_steps + 1) * timestep.to_timedelta64().astype("timedelta64[ns]")
    reference_ns = np.datetime64(reference.to_datetime64(), "ns")

    coords: dict[str, Any] = {
        "ensemble_member": xr.Variable(
            "ensemble_member",
            np.arange(n_members),
            {"standard_name": "realization", "long_name": "ensemble member"},
        ),
        "time": xr.Variable("time", reference_ns + lead_times, {"standard_name": "time", "axis": "T"}),
        "lead_time": xr.Variable(
            "time", lead_times, {"standard_name": "forecast_period", "long_name": "time since forecast reference"}
        ),
        "forecast_reference_time": xr.Variable(
            (), reference_ns, {"standard_name": "forecast_reference_time", "long_name": "time of the last input"}
        ),
    }
    for dim, axis in ((y_dim, "Y"), (x_dim, "X")):
        variable = _clean_variable(ds[dim].variable)
        variable.attrs["axis"] = axis
        geographic = str(variable.attrs.get("units", "")).startswith("degree")
        variable.attrs.setdefault("standard_name", _DEFAULT_COORD_STANDARD_NAMES[(axis, geographic)])
        coords[dim] = variable

    data_vars = {}
    for c, (name, standard_name, units) in enumerate(zip(var_names, spec.standard_names, spec.units, strict=True)):
        da = ds[name]
        # Auxiliary horizontal coordinates (e.g. 2D lat/lon) and scalar ones (e.g. a crs coordinate)
        for coord_name, coord in da.coords.items():
            if coord_name not in (t_dim, y_dim, x_dim) and set(coord.dims) <= {y_dim, x_dim}:
                coords.setdefault(str(coord_name), _clean_variable(coord.variable))
        attrs = {
            "standard_name": standard_name,
            "units": units,
            "long_name": da.attrs.get("long_name", standard_name.replace("_", " ")),
        }
        grid_mapping = da.attrs.get("grid_mapping", da.encoding.get("grid_mapping"))
        if grid_mapping is not None:
            if grid_mapping in ds.variables:
                attrs["grid_mapping"] = grid_mapping
                coords.setdefault(grid_mapping, _clean_variable(ds[grid_mapping].variable))
            else:
                logger.warning(f"Variable {name!r} refers to grid_mapping {grid_mapping!r}, which is not in the input")
        data_vars[name] = xr.Variable(("ensemble_member", "time", y_dim, x_dim), out[:, :, c], attrs)

    version = mlcast.__version__
    created = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    attrs: dict[str, Any] = {
        "Conventions": "CF-1.8",
        "source": f"mlcast {version} via mlcast_icb",
        "mlcast_version": version,
    }
    if seed is not None:
        attrs["seed"] = seed
    attrs["history"] = (
        f"{created}: {n_members}-member, {n_steps}-step forecast from {reference.isoformat()} with mlcast {version}"
    )
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)
