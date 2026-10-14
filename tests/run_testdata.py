"""Synthetic inputs and a fake forecast shared by the bundle, run and CLI tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

#: Times of the synthetic input: 5-minute steps.
START = "2026-05-01T12:00"
STEP = np.timedelta64(5, "m")


def build_input(nt: int = 5, ny: int = 8, nx: int = 8, seed: int = 0, extra_variable: bool = True) -> xr.Dataset:
    """A CF radar-like input: ``rr`` (rainfall_rate, mm h-1) on a projected grid, plus clutter.

    ``extra`` is a second variable the model does not read, so tests can check that it
    is never loaded.
    """
    rng = np.random.default_rng(seed)
    data_vars: dict[str, Any] = {
        "rr": (
            ("time", "y", "x"),
            rng.gamma(0.5, 2.0, size=(nt, ny, nx)).astype("float32"),
            {"standard_name": "rainfall_rate", "units": "mm h-1", "grid_mapping": "crs"},
        ),
        "crs": ((), np.int32(0), {"grid_mapping_name": "lambert_azimuthal_equal_area"}),
    }
    if extra_variable:
        data_vars["quality"] = (("time", "y", "x"), np.ones((nt, ny, nx), "float32"), {"long_name": "quality"})
    return xr.Dataset(
        data_vars,
        coords={
            "time": ("time", pd.date_range(START, periods=nt, freq="5min"), {"standard_name": "time", "axis": "T"}),
            "y": (
                "y",
                np.arange(ny, dtype="float64") * 1000.0,
                {"axis": "Y", "standard_name": "projection_y_coordinate", "units": "m"},
            ),
            "x": (
                "x",
                np.arange(nx, dtype="float64") * 1000.0,
                {"axis": "X", "standard_name": "projection_x_coordinate", "units": "m"},
            ),
        },
    )


def write_input(path: Path, zarr_format: int = 3, **kwargs: Any) -> Path:
    """Write :func:`build_input` to a zarr store and return its path."""
    build_input(**kwargs).to_zarr(path, mode="w", consolidated=True, zarr_format=zarr_format)
    return path


def fake_forecast_output(
    ds: xr.Dataset, forecast_steps: int, ensemble_size: int, reference_time: np.datetime64 | None
) -> xr.Dataset:
    """A dataset with the schema of ``mlcast_icb.upstream.forecast.forecast``'s output."""
    reference = ds.time.values[-1] if reference_time is None else reference_time
    lead = np.arange(1, forecast_steps + 1) * STEP.astype("timedelta64[ns]")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    values = np.full((ensemble_size, forecast_steps, ny, nx), 0.5, dtype="float32")
    return xr.Dataset(
        {
            "rr": (
                ("ensemble_member", "time", "y", "x"),
                values,
                {"standard_name": "rainfall_rate", "units": "mm h-1", "grid_mapping": "crs"},
            ),
            "crs": ((), np.int32(0), dict(ds.crs.attrs)),
        },
        coords={
            "ensemble_member": ("ensemble_member", np.arange(ensemble_size), {"standard_name": "realization"}),
            "time": ("time", reference + lead, {"standard_name": "time", "axis": "T"}),
            "lead_time": ("time", lead, {"standard_name": "forecast_period"}),
            "forecast_reference_time": ((), reference, {"standard_name": "forecast_reference_time"}),
            "y": ds.y.variable.copy(),
            "x": ds.x.variable.copy(),
        },
    )
