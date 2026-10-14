"""netCDF payloads for the HTTP server: an uploaded body in, a compressed file out.

Reading goes through a spooled temporary file — in memory up to a threshold, on disk
beyond it — so a large upload never sits in memory twice, and the size limit is enforced
while copying rather than after. The format is sniffed from the first bytes instead of
trusting a filename or content type a client may have got wrong:

=============================== ======================== ==========
Leading bytes                   Format                   Engine
=============================== ======================== ==========
``\\x89HDF\\r\\n\\x1a\\n``      netCDF4 (HDF5)           h5netcdf
``CDF\\x01``, ``CDF\\x02``      netCDF3 classic / 64-bit scipy
``CDF\\x05``                    netCDF3 CDF-5            unsupported
=============================== ======================== ==========

Writing produces netCDF4 with zlib compression, through h5netcdf, or the netCDF4
library if h5netcdf is not installed.

Requesting ``dtype='float16'`` does **not** write IEEE half floats: the netCDF data
model has no 16-bit float type, and xarray has no special handling for one — h5netcdf
would store an HDF5 half float that netCDF-C based readers (netCDF4-python, ncdump,
most GIS tools) cannot read, and the netCDF4 library rejects it outright. Instead each
floating-point variable is packed CF-style into ``int16`` with ``scale_factor`` and
``add_offset`` chosen from its own range, and missing values as ``_FillValue``: the same
two bytes per value, readable everywhere, and unpacked back to float32 by any CF-aware
reader, xarray included. The quantisation step is ``(max - min) / 65533``.
"""

from __future__ import annotations

import importlib.util
import tempfile
from os import PathLike
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import xarray as xr
from loguru import logger

from mlcast_icb.errors import InputError, PayloadTooLargeError

#: Signature at the start of an HDF5 file, and so of a netCDF4 file.
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

#: Signatures of the netCDF3 classic and 64-bit-offset formats.
NETCDF3_MAGICS = (b"CDF\x01", b"CDF\x02")

#: Signature of the netCDF3 CDF-5 (64-bit data) format, which scipy cannot read.
CDF5_MAGIC = b"CDF\x05"

#: Uploads up to this size stay in memory; larger ones spill to a temporary file.
SPOOL_MAX_MEMORY = 64 * 1024 * 1024

#: Read size when copying an upload into the spooled file.
_COPY_CHUNK = 1024 * 1024

NetcdfFormat = Literal["NETCDF4", "NETCDF3"]

#: Packed range of ``int16`` with one value kept back for ``_FillValue``.
_INT16_FILL = np.int16(-32768)
_INT16_MIN, _INT16_MAX = -32767, 32767


def sniff_netcdf_format(head: bytes) -> NetcdfFormat:
    """Identify a netCDF payload from its leading bytes.

    Parameters
    ----------
    head : bytes
        At least the first 8 bytes of the payload; more is ignored.

    Returns
    -------
    {'NETCDF4', 'NETCDF3'}
        ``'NETCDF4'`` for an HDF5 file, ``'NETCDF3'`` for the classic and
        64-bit-offset formats.

    Raises
    ------
    InputError
        If the payload is empty, is CDF-5, or is not netCDF at all.

    Notes
    -----
    Only an HDF5 signature at offset 0 is recognised. HDF5 allows a user block that
    moves it to 512, 1024, ... bytes in, but netCDF libraries never write one.
    """
    if not head:
        raise InputError("The uploaded payload is empty; expected a netCDF file")
    if head.startswith(HDF5_MAGIC):
        return "NETCDF4"
    if head.startswith(NETCDF3_MAGICS):
        return "NETCDF3"
    if head.startswith(CDF5_MAGIC):
        raise InputError(
            "The uploaded file is netCDF3 CDF-5 (64-bit data), which is not supported; send netCDF4 or classic netCDF3"
        )
    raise InputError(
        f"The uploaded payload is not a netCDF file: it starts with {head[:8]!r}, not an HDF5 or netCDF3 signature"
    )


def read_netcdf_bytes(
    data: bytes | bytearray | memoryview | IO[bytes],
    max_bytes: int | None = None,
) -> xr.Dataset:
    """Load an uploaded netCDF payload into memory as a dataset.

    Parameters
    ----------
    data : bytes or binary file-like
        The payload, whole or as a stream positioned at its start (an upload's file
        object, a request body stream).
    max_bytes : int, optional
        Largest payload accepted. ``None`` for no limit.

    Returns
    -------
    xarray.Dataset
        The fully loaded dataset, detached from the temporary file, which is gone by
        the time this returns.

    Raises
    ------
    PayloadTooLargeError
        If the payload exceeds ``max_bytes``. Detected while copying, so an oversize
        stream is not read further than one chunk past the limit.
    InputError
        If the payload is empty, not netCDF, or netCDF that cannot be decoded.
    ImportError
        If the payload is netCDF4 and h5netcdf is not installed: a packaging fault of
        the image, not the client's.
    """
    with tempfile.SpooledTemporaryFile(max_size=SPOOL_MAX_MEMORY) as spool:
        size = _copy_limited(data, spool, max_bytes)
        spool.seek(0)
        file_format = sniff_netcdf_format(spool.read(len(HDF5_MAGIC)))
        spool.seek(0)
        engine = _require_h5netcdf() if file_format == "NETCDF4" else "scipy"
        logger.debug(f"Reading {size} byte {file_format} payload with {engine}")
        try:
            with xr.open_dataset(spool, engine=engine, decode_timedelta=True) as ds:
                return ds.load()
        except Exception as exc:
            # The signature matched but the decoder choked: truncated, corrupt or
            # otherwise malformed bytes. Whatever a decoder raises on bytes a client
            # sent is that client's input problem, so the net is deliberately wide;
            # it covers only the decode, never our own code.
            raise InputError(f"The uploaded {file_format} file could not be read: {type(exc).__name__}: {exc}") from exc


def _copy_limited(data: bytes | bytearray | memoryview | IO[bytes], target: IO[bytes], max_bytes: int | None) -> int:
    """Copy a payload into ``target``, enforcing the size limit as it goes.

    Parameters
    ----------
    data : bytes or binary file-like
        The payload.
    target : binary file-like
        Where to copy it.
    max_bytes : int or None
        Largest payload accepted.

    Returns
    -------
    int
        Number of bytes copied.

    Raises
    ------
    PayloadTooLargeError
        As soon as more than ``max_bytes`` have been seen.
    """
    if isinstance(data, bytes | bytearray | memoryview):
        size = len(data)
        _check_size(size, max_bytes)
        target.write(data)
        return size
    size = 0
    while chunk := data.read(_COPY_CHUNK):
        size += len(chunk)
        _check_size(size, max_bytes)
        target.write(chunk)
    return size


def _check_size(size: int, max_bytes: int | None) -> None:
    """Raise when ``size`` is over the limit.

    Parameters
    ----------
    size : int
        Bytes seen so far.
    max_bytes : int or None
        Largest payload accepted.

    Raises
    ------
    PayloadTooLargeError
        If ``size`` exceeds ``max_bytes``.
    """
    if max_bytes is not None and size > max_bytes:
        raise PayloadTooLargeError(f"The uploaded payload is larger than the limit of {max_bytes} bytes")


def write_netcdf(
    ds: xr.Dataset,
    path: str | PathLike[str],
    complevel: int = 4,
    dtype: Literal["float16", "float32"] | None = None,
) -> Path:
    """Write a dataset as zlib-compressed netCDF4.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to write.
    path : str or path-like
        Destination file, typically a temporary file the server streams from.
    complevel : int, optional
        zlib level 0-9 for the data variables; 0 writes them uncompressed.
    dtype : {'float16', 'float32'}, optional
        Storage type of the floating-point data variables. ``None`` keeps their own
        dtype; ``'float32'`` casts on write; ``'float16'`` packs into int16 as the
        module docstring describes. Coordinates are never changed.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    ValueError
        If ``complevel`` or ``dtype`` is out of range.
    ImportError
        If neither h5netcdf nor netCDF4 is installed.
    """
    path = Path(path)
    engine = _netcdf4_engine()
    ds.to_netcdf(path, engine=engine, format="NETCDF4", encoding=netcdf_encoding(ds, complevel, dtype))
    logger.debug(f"Wrote {path} with {engine} (complevel={complevel}, dtype={dtype})")
    return path


def netcdf_encoding(
    ds: xr.Dataset,
    complevel: int = 4,
    dtype: Literal["float16", "float32"] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the per-variable netCDF4 encoding :func:`write_netcdf` uses.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset about to be written.
    complevel : int, optional
        zlib level 0-9.
    dtype : {'float16', 'float32'}, optional
        Storage type of the floating-point data variables.

    Returns
    -------
    dict of str to dict
        A ``to_netcdf`` encoding for every data variable: ``zlib``/``complevel`` when
        ``complevel`` > 0, plus the dtype or int16 packing requested.

    Raises
    ------
    ValueError
        If ``complevel`` is not in 0-9 or ``dtype`` is not a supported value.
    """
    if not 0 <= complevel <= 9:
        raise ValueError(f"complevel must be between 0 and 9, got {complevel}")
    if dtype not in (None, "float16", "float32"):
        raise ValueError(f"dtype must be 'float16' or 'float32', got {dtype!r}")

    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        var: dict[str, Any] = {"zlib": True, "complevel": complevel} if complevel else {}
        if dtype is not None and np.issubdtype(da.dtype, np.floating):
            var.update(_pack_int16(da.values) if dtype == "float16" else {"dtype": "float32"})
        encoding[str(name)] = var
    return encoding


def _pack_int16(values: np.ndarray) -> dict[str, Any]:
    """Choose CF int16 packing parameters that span the finite range of ``values``.

    Parameters
    ----------
    values : numpy.ndarray
        The floating-point data to pack.

    Returns
    -------
    dict
        ``dtype``, ``scale_factor``, ``add_offset`` and ``_FillValue`` encoding
        entries. The minimum maps to -32767 and the maximum to 32766 or so, keeping
        one step spare so float32 rounding of the parameters cannot overflow int16;
        -32768 is the fill value, which is where NaN (and infinities) end up.
        ``scale_factor``/``add_offset`` are float32, so CF readers unpack to float32.
    """
    finite = values[np.isfinite(values)].astype("float64")
    if finite.size == 0:
        low = high = 0.0
    else:
        low, high = float(finite.min()), float(finite.max())
    scale = (high - low) / (_INT16_MAX - _INT16_MIN - 1) if high > low else 1.0
    offset = low - _INT16_MIN * scale
    return {
        "dtype": "int16",
        "scale_factor": np.float32(scale),
        "add_offset": np.float32(offset),
        "_FillValue": _INT16_FILL,
    }


def _netcdf4_engine() -> str:
    """Pick the xarray engine for writing netCDF4.

    Returns
    -------
    str
        ``'h5netcdf'`` when installed, otherwise ``'netcdf4'``.

    Raises
    ------
    ImportError
        If neither is installed.
    """
    if _has_h5netcdf():
        return "h5netcdf"
    if importlib.util.find_spec("netCDF4") is not None:
        return "netcdf4"
    raise ImportError("Writing netCDF4 needs h5netcdf with h5py (preferred) or netCDF4; neither is installed")


def _require_h5netcdf() -> str:
    """Check that h5netcdf can read netCDF4 here.

    Returns
    -------
    str
        ``'h5netcdf'``, the xarray engine name.

    Raises
    ------
    ImportError
        If h5netcdf or h5py is not installed.
    """
    if not _has_h5netcdf():
        raise ImportError("Reading netCDF4 uploads needs h5netcdf and h5py, which are not both installed")
    return "h5netcdf"


def _has_h5netcdf() -> bool:
    """Report whether h5netcdf is usable.

    Returns
    -------
    bool
        True when both h5netcdf and h5py are installed. h5netcdf >= 1.7 no longer
        depends on h5py itself (it can also use pyfive, read-only), but without one it
        imports fine and fails on the first file it opens.
    """
    return all(importlib.util.find_spec(module) is not None for module in ("h5netcdf", "h5py"))
