"""Reading and writing the data an inference container consumes and produces.

* :mod:`.storage` — per-side (``SRC_`` / ``DST_``) fsspec storage options from the
  environment, and the classification of failures into storage vs input errors.
* :mod:`.zarr_io` — zarr stores, local or remote (mode 1, ``run``).
* :mod:`.netcdf_io` — netCDF payloads as bytes in and a file out (mode 2, ``serve``).

Nothing is imported here, so importing :mod:`.storage` (as :mod:`mlcast_icb.envvars`
does) does not pull in xarray.
"""
