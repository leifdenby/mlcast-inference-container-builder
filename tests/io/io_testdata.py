"""Synthetic data shared by the I/O tests. Fixtures belong in conftest.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def build_dataset(nt: int = 4, ny: int = 12, nx: int = 16, seed: int = 0) -> xr.Dataset:
    """Build a small CF-flavoured radar-like dataset on a projected grid.

    Parameters
    ----------
    nt, ny, nx : int, optional
        Sizes of the time, y and x dimensions.
    seed : int, optional
        Seed for the field values.

    Returns
    -------
    xarray.Dataset
        A float32 ``rr`` field (with a NaN, as radar composites have) referencing a
        ``crs`` grid-mapping variable, with time, y and x coordinates.
    """
    rng = np.random.default_rng(seed)
    rr = rng.gamma(0.5, 2.0, size=(nt, ny, nx)).astype("float32")
    rr[0, 0, 0] = np.nan
    return xr.Dataset(
        {
            "rr": (
                ("time", "y", "x"),
                rr,
                {"standard_name": "rainfall_rate", "units": "mm h-1", "grid_mapping": "crs"},
            ),
            "crs": ((), np.int32(0), {"grid_mapping_name": "lambert_azimuthal_equal_area"}),
        },
        coords={
            "time": ("time", pd.date_range("2026-01-01", periods=nt, freq="5min"), {"standard_name": "time"}),
            "y": ("y", np.arange(ny, dtype="float64") * 1000.0, {"axis": "Y", "units": "m"}),
            "x": ("x", np.arange(nx, dtype="float64") * 1000.0, {"axis": "X", "units": "m"}),
        },
        attrs={"title": "synthetic"},
    )
