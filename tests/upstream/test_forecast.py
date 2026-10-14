"""Tests for the xarray forecast entry point (``forecast.py``).

Written to move into mlcast's ``tests/inference/test_forecast.py`` unchanged, apart from
the import of the modules under test.
"""

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from loguru import logger

from mlcast.data.normalization import DEFAULT_SCALING
from mlcast.models.convgru import ConvGruModel
from mlcast.nowcasting_module import NowcastLightningModule
from mlcast_icb.upstream.checkpoint import load_model
from mlcast_icb.upstream.forecast import ForecastInputError, forecast, unit_conversion_factor
from mlcast_icb.upstream.processing import postprocess, preprocess
from mlcast_icb.upstream.spec import InferenceSpec

START = np.datetime64("2025-06-01T12:00:00", "ns")
STEP = np.timedelta64(5, "m")


@pytest.fixture(scope="module")
def model_and_spec(tiny_trained_model) -> tuple[NowcastLightningModule, InferenceSpec]:
    """The tiny trained model, with a 5-minute timestep."""
    return load_model(tiny_trained_model.checkpoint_path, tiny_trained_model.config_path, timestep="PT5M")


def _rain(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Synthetic rain in mm/h: about half the cells dry, the rest up to a few tens of mm/h."""
    rng = np.random.default_rng(seed)
    return (rng.gamma(0.7, 3.0, shape) * (rng.random(shape) > 0.5)).astype(np.float32)


def _dataset(
    n_times: int = 4,
    height: int = 8,
    width: int = 8,
    *,
    lat_lon: bool = False,
    name: str = "precip",
    standard_name: str = "rainfall_rate",
    units: str | None = "mm h-1",
    values: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> xr.Dataset:
    """A CF-compliant radar-like Dataset: projected x/y with a ``crs`` grid mapping, or lat/lon."""
    times = START + STEP * np.arange(n_times) if times is None else times
    values = _rain((len(times), height, width)) if values is None else values
    attrs = {"standard_name": standard_name, "long_name": "precipitation"}
    if units is not None:
        attrs["units"] = units
    if lat_lon:
        y_dim, x_dim = "lat", "lon"
        coords = {
            "lat": ("lat", 55.0 + 0.01 * np.arange(height), {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": ("lon", 10.0 + 0.01 * np.arange(width), {"standard_name": "longitude", "units": "degrees_east"}),
        }
        data_vars = {}
    else:
        y_dim, x_dim = "y", "x"
        coords = {
            "y": ("y", 1000.0 * np.arange(height), {"standard_name": "projection_y_coordinate", "units": "m"}),
            "x": ("x", 1000.0 * np.arange(width), {"standard_name": "projection_x_coordinate", "units": "m"}),
        }
        attrs["grid_mapping"] = "crs"
        data_vars = {"crs": ((), np.int32(0), {"grid_mapping_name": "lambert_azimuthal_equal_area"})}
    coords["time"] = ("time", times, {"standard_name": "time"})
    data_vars[name] = (("time", y_dim, x_dim), values, attrs)
    return xr.Dataset(data_vars, coords=coords)


def test_output_schema(model_and_spec: tuple[NowcastLightningModule, InferenceSpec], tmp_path: Path) -> None:
    """The output has the documented dims, coordinates and attributes."""
    # Given a projected input with a grid mapping
    model, spec = model_and_spec
    ds = _dataset()

    # When forecasting
    out = forecast(model, spec, ds, ensemble_size=3, seed=1)

    # Then the forecast variable has the input's name, the expected dims, shape and attributes
    assert list(out.data_vars) == ["precip"]
    da = out["precip"]
    assert da.dims == ("ensemble_member", "time", "y", "x")
    assert da.shape == (3, spec.forecast_steps, 8, 8)
    assert da.dtype == np.float32
    assert da.attrs == {
        "standard_name": "rainfall_rate",
        "units": "mm h-1",
        "long_name": "precipitation",
        "grid_mapping": "crs",
    }
    assert np.isfinite(da.values).all() and (da.values >= 0).all()

    # And the coordinates describe the ensemble, valid times, lead times and grid
    reference = START + 3 * STEP
    lead_times = STEP * np.arange(1, spec.forecast_steps + 1)
    np.testing.assert_array_equal(out["ensemble_member"], np.arange(3))
    np.testing.assert_array_equal(out["time"], reference + lead_times)
    np.testing.assert_array_equal(out["lead_time"], lead_times)
    assert out["lead_time"].dims == ("time",)
    assert out["lead_time"].attrs["standard_name"] == "forecast_period"
    assert out["forecast_reference_time"].values == reference
    assert out["forecast_reference_time"].attrs["standard_name"] == "forecast_reference_time"
    assert out["time"].attrs["axis"] == "T" and out["time"].attrs["standard_name"] == "time"
    for dim, axis in (("y", "Y"), ("x", "X")):
        np.testing.assert_array_equal(out[dim], ds[dim])
        assert out[dim].attrs == {**ds[dim].attrs, "axis": axis}
    assert "crs" in out.coords and out["crs"].attrs == ds["crs"].attrs

    # And the global attributes record provenance
    assert out.attrs["source"].startswith(f"mlcast {out.attrs['mlcast_version']}")
    assert out.attrs["seed"] == 1
    assert "history" in out.attrs

    # And it round-trips through netCDF
    out.to_netcdf(tmp_path / "forecast.nc", engine="h5netcdf")
    with xr.open_dataset(tmp_path / "forecast.nc", engine="h5netcdf") as roundtrip:
        xr.testing.assert_identical(roundtrip.load(), out)


def test_lat_lon_input(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """Latitude/longitude axes are found by standard name and carried through."""
    # Given a lat/lon input without axis attributes
    model, spec = model_and_spec
    ds = _dataset(lat_lon=True, height=8, width=12)

    # When forecasting with the spec's defaults
    out = forecast(model, spec, ds, seed=0)

    # Then the output uses the input's dim names and coordinates
    da = out["precip"]
    assert da.dims == ("ensemble_member", "time", "lat", "lon")
    assert da.shape == (spec.default_ensemble_size, spec.forecast_steps, 8, 12)
    assert "grid_mapping" not in da.attrs
    assert out["lat"].attrs == {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    assert out["lon"].attrs == {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}


def test_odd_domain_pads_with_no_rain(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """An odd-sized domain is padded with no rain and cropped back (the PR H regression test)."""
    # Given an input whose size is not a multiple of the spatial divisor, and the same input
    # padded by hand with 0 mm/h to the next multiple
    model, spec = model_and_spec
    height, width = 37, 53
    divisor = spec.spatial_divisor
    padded_height, padded_width = -(-height // divisor) * divisor, -(-width // divisor) * divisor
    assert (padded_height, padded_width) != (height, width)
    values = _rain((4, height, width), seed=3)
    padded_values = np.zeros((4, padded_height, padded_width), dtype=np.float32)
    padded_values[:, :height, :width] = values

    # When forecasting both with the same seed
    out = forecast(model, spec, _dataset(values=values, height=height, width=width), seed=5)
    out_padded = forecast(model, spec, _dataset(values=padded_values, height=padded_height, width=padded_width), seed=5)

    # Then the output is cropped to the input's shape and equals the hand-padded forecast
    assert out["precip"].shape[-2:] == (height, width)
    np.testing.assert_allclose(out["precip"].values, out_padded["precip"].values[..., :height, :width], atol=1e-6)

    # And the network's own padding (0.0 in normalised space, i.e. 30 dBZ) would give a different forecast
    x = torch.from_numpy(preprocess([values[1:]], spec.standard_names, spec.scaling)).unsqueeze(0)
    torch.manual_seed(5)
    with torch.no_grad():
        raw = model.network(x, steps=spec.forecast_steps, ensemble_size=spec.default_ensemble_size)
    raw_out = postprocess(raw, spec.standard_names, spec.scaling)[0, :, :, 0]
    assert not np.allclose(raw_out, out["precip"].values, atol=1e-3)


def test_seed_reproducibility(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """The same seed gives the same ensemble, different seeds different ones; the global RNG is untouched."""
    # Given an input and the global RNG state
    model, spec = model_and_spec
    ds = _dataset()
    torch.manual_seed(123)
    state_before = torch.get_rng_state()

    # When forecasting twice with one seed and once with another
    first = forecast(model, spec, ds, ensemble_size=2, seed=7)
    second = forecast(model, spec, ds, ensemble_size=2, seed=7)
    other = forecast(model, spec, ds, ensemble_size=2, seed=8)

    # Then the same seed reproduces the forecast and a different seed changes it
    np.testing.assert_array_equal(first["precip"].values, second["precip"].values)
    assert not np.array_equal(first["precip"].values, other["precip"].values)

    # And the members of one forecast differ from each other
    assert not np.array_equal(first["precip"].values[0], first["precip"].values[1])

    # And the global RNG state is as before
    assert torch.equal(torch.get_rng_state(), state_before)


def test_reference_time_selects_inputs(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """The inputs are the ``input_steps`` times ending at ``reference_time``."""
    # Given a longer input series
    model, spec = model_and_spec
    ds = _dataset(n_times=7)
    reference = ds["time"].values[4]

    # When forecasting from an earlier reference time
    out = forecast(model, spec, ds, seed=0, reference_time=reference)

    # Then the forecast starts after the reference time and equals one from just those inputs
    assert out["forecast_reference_time"].values == reference
    assert out["time"].values[0] == reference + STEP
    subset = ds.isel(time=slice(5 - spec.input_steps, 5))
    np.testing.assert_array_equal(out["precip"].values, forecast(model, spec, subset, seed=0)["precip"].values)


def test_problems_are_reported_together(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """A missing variable, too few input times and the wrong spacing are all reported in one error."""
    # Given an input with the wrong variable and two times 10 minutes apart
    model, spec = model_and_spec
    times = START + np.timedelta64(10, "m") * np.arange(2)
    ds = _dataset(times=times, standard_name="air_temperature", units="K")

    # When forecasting
    with pytest.raises(ForecastInputError) as excinfo:
        forecast(model, spec, ds)

    # Then the error lists all three problems
    problems = excinfo.value.problems
    assert len(problems) == 3, problems
    message = str(excinfo.value)
    assert "No variable with standard_name 'rainfall_rate'" in message
    assert f"needs {spec.input_steps} input times" in message
    assert "0 days 00:05:00 apart" in message
    assert isinstance(excinfo.value, ValueError)


def test_unknown_reference_time_and_units(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """A reference time not in the input and unconvertible units are reported."""
    # Given an input in dBZ (not convertible to mm/h)
    model, spec = model_and_spec
    ds = _dataset(units="dBZ")

    # When forecasting from a time that is not in the input
    with pytest.raises(ForecastInputError) as excinfo:
        forecast(model, spec, ds, reference_time=START + STEP // 2)

    # Then both problems are reported
    assert len(excinfo.value.problems) == 2
    assert "cannot be converted to 'mm h-1'" in str(excinfo.value)
    assert "is not one of the input times" in str(excinfo.value)


def test_units(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """Other spellings, rainfall flux and missing units are accepted."""
    # Given the same rain as mm/h, as "mm/h", as a flux in kg m-2 s-1 and without units
    model, spec = model_and_spec
    values = _rain((4, 8, 8), seed=4)
    reference = forecast(model, spec, _dataset(values=values), seed=0)["precip"].values
    flux = _dataset(values=values / 3600, standard_name="rainfall_flux", units="kg m-2 s-1")
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        # When forecasting from each
        spelled = forecast(model, spec, _dataset(values=values, units="mm/h"), seed=0)
        converted = forecast(model, spec, flux, seed=0)
        unitless = forecast(model, spec, _dataset(values=values, units=None), seed=0)
    finally:
        logger.remove(sink)

    # Then all give the same forecast, in the spec's units
    np.testing.assert_array_equal(spelled["precip"].values, reference)
    np.testing.assert_allclose(converted["precip"].values, reference, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(unitless["precip"].values, reference)
    assert converted["precip"].attrs["units"] == "mm h-1"
    assert converted["precip"].attrs["standard_name"] == "rainfall_rate"

    # And the missing units were warned about
    assert any("no units attribute" in w for w in warnings)


def test_unit_conversion_factor() -> None:
    """Equivalent spellings convert with 1, flux and rate with 3600, incompatible units not at all."""
    assert unit_conversion_factor("mm hr-1", "mm h-1") == 1.0
    assert unit_conversion_factor("kg m-2 s-1", "mm/h") == 3600.0
    assert unit_conversion_factor("mm h-1", "kg m-2 s-1") == pytest.approx(1 / 3600)
    assert unit_conversion_factor("K", "mm h-1") is None


def test_output_is_clamped(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """No forecast exceeds the rain rate of the 60 dBZ ceiling, even for extreme input."""
    # Given extreme rain everywhere
    model, spec = model_and_spec
    ds = _dataset(values=np.full((4, 8, 8), 1000.0, dtype=np.float32))

    # When forecasting
    out = forecast(model, spec, ds, ensemble_size=4, seed=0)

    # Then the output stays at or below the 60 dBZ rain rate
    ceiling = DEFAULT_SCALING.reflectivity_to_rainfall_rate(np.array([DEFAULT_SCALING.dbz_ceiling], dtype=np.float32))
    assert out["precip"].values.max() <= ceiling[0] * (1 + 1e-6)


def test_multichannel_model(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """A model with several input channels gets one output variable per channel, in spec order."""
    # Given a two-channel network and a spec for rain rate and reflectivity
    _, spec = model_and_spec
    torch.manual_seed(0)
    model = NowcastLightningModule(ConvGruModel(input_channels=2, num_blocks=2), ensemble_size=2)
    spec2 = dataclasses.replace(
        spec, standard_names=("rainfall_rate", "equivalent_reflectivity_factor"), units=("mm h-1", "dBZ")
    )
    ds = _dataset(height=9, width=10)
    dbz = np.random.default_rng(1).uniform(0, 50, ds["precip"].shape).astype(np.float32)
    ds["refl"] = (ds["precip"].dims, dbz, {"standard_name": "equivalent_reflectivity_factor", "units": "dBZ"})

    # When forecasting
    out = forecast(model, spec2, ds, seed=0)

    # Then each channel is its own variable, with its own units, on the input grid
    assert list(out.data_vars) == ["precip", "refl"]
    assert out["refl"].attrs["units"] == "dBZ"
    assert out["refl"].shape == (2, spec.forecast_steps, 9, 10)
    assert 0.0 <= out["refl"].values.min() and out["refl"].values.max() <= 60.0


def test_pads_input_to_spec_divisor_with_norm_min(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """The network receives an input padded with ``norm_min`` to a multiple of the divisor."""
    # Given a network that records its input
    model, spec = model_and_spec
    seen: list[torch.Tensor] = []
    handle = model.network.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(args[0]), with_kwargs=True
    )

    # When forecasting on a 9x10 domain
    try:
        forecast(model, spec, _dataset(height=9, width=10), ensemble_size=1)
    finally:
        handle.remove()

    # Then its input was padded at the bottom and right with no rain
    (x,) = seen
    assert x.shape[-2:] == (12, 12)
    assert torch.all(x[..., 9:, :] == spec.scaling.norm_min)
    assert torch.all(x[..., :, 10:] == spec.scaling.norm_min)


def test_custom_input_names_by_time_index(model_and_spec: tuple[NowcastLightningModule, InferenceSpec]) -> None:
    """The time dimension is found by CF attributes, whatever its name."""
    # Given an input whose time dimension is called "t"
    model, spec = model_and_spec
    ds = _dataset().rename({"time": "t"})

    # When forecasting
    out = forecast(model, spec, ds, seed=0)

    # Then the output still uses "time" for valid times
    assert out["precip"].dims == ("ensemble_member", "time", "y", "x")
    assert pd.Timestamp(out["forecast_reference_time"].values) == pd.Timestamp(START + 3 * STEP)


def test_import_avoids_training_dependencies() -> None:
    """Importing the module does not import the training orchestrator or its loggers."""
    # Given a fresh interpreter
    code = (
        "import sys, mlcast_icb.upstream.forecast; "
        "print([m for m in ('mlflow', 'tensorboard', 'wandb', 'mlcast.config.orchestrator') if m in sys.modules])"
    )

    # When importing the module there
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    # Then none of the training-only modules were imported
    assert result.stdout.strip() == "[]"
