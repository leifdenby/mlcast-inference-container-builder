"""Opening, loading and writing zarr stores, local or remote, with failures classified.

A failure is reported as one of two things, so the exit code says what to fix:

* :class:`~mlcast_icb.errors.InputError` (exit 3) — the input store is not there, or is
  not zarr: a wrong input URI.
* :class:`~mlcast_icb.errors.StorageError` (exit 5) — a store could not be reached, read
  or written: credentials, access, endpoint, a read-only destination.

Anything else — a bug — propagates unchanged, with its traceback.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import fsspec
import xarray as xr
from loguru import logger

from mlcast_icb.errors import ConfigError, InputError
from mlcast_icb.io.storage import storage_error, storage_errors, storage_exceptions

#: Accepted values of a zarr format setting such as ``--zarr-format``.
ZARR_FORMATS = ("auto", "2", "3")

#: Written when ``'auto'`` has nothing to follow, or detection fails.
DEFAULT_ZARR_FORMAT = 3

ZarrFormat = Literal["auto", 2, 3]


def open_zarr(uri: str, storage_options: Mapping[str, Any] | None = None) -> xr.Dataset:
    """Open a zarr store, local or remote, as the input.

    Parameters
    ----------
    uri : str
        Store location. Local paths, ``s3://`` and ``gs://`` all work, the latter two
        through fsspec.
    storage_options : mapping, optional
        fsspec options for this store, as built by
        :func:`~mlcast_icb.io.storage.storage_options` for the ``input`` side. ``None``
        or an empty mapping leaves fsspec to pick up ambient environment credentials.

    Returns
    -------
    xarray.Dataset
        The opened store, lazily loaded: only metadata and coordinates have been read.
        Load the data through :func:`load_zarr`, so a failure to read it is
        classified too.

    Raises
    ------
    InputError
        If the store is missing or cannot be read as zarr.
    StorageError
        If the store cannot be reached: credentials, access, endpoint.

    Notes
    -----
    No ``chunks=`` is requested, so no dask is needed: the data is read into numpy
    when it is loaded.
    """
    logger.info(f"Opening input store {uri}")
    try:
        return xr.open_dataset(
            uri,
            engine="zarr",
            decode_timedelta=True,
            storage_options=dict(storage_options) if storage_options else None,
        )
    except FileNotFoundError as exc:
        # Nothing there, or nothing zarr there (zarr's GroupNotFoundError is one of
        # these): a wrong input URI, not a failure to reach the store. s3fs reports a
        # missing bucket or key this way too.
        raise InputError(f"Could not open input store {uri!r}: {exc}") from exc
    except storage_exceptions() as exc:
        # Before ValueError: a few botocore errors (an unknown endpoint or region) are
        # ValueErrors too, and they are storage failures, not a malformed store.
        raise storage_error("input", uri, storage_options, "open", exc) from exc
    except (KeyError, ValueError) as exc:
        raise InputError(f"Could not open input store {uri!r}: {exc}") from exc


def load_zarr(ds: xr.Dataset, uri: str, storage_options: Mapping[str, Any] | None = None) -> xr.Dataset:
    """Read the data of a dataset opened with :func:`open_zarr` into memory.

    Parameters
    ----------
    ds : xarray.Dataset
        The lazily opened input, or a selection of it — select first, so only what is
        needed is fetched.
    uri : str
        The store it came from, for the error message.
    storage_options : mapping, optional
        The options it was opened with, for the error hint.

    Returns
    -------
    xarray.Dataset
        ``ds`` with every variable loaded.

    Raises
    ------
    StorageError
        If reading the data fails. Opening reads only metadata, so this is where the
        chunks are fetched, and the first place a read permission or a flaky endpoint
        can bite.
    """
    with storage_errors("input", uri, storage_options, "read"):
        return ds.load()


def detect_zarr_format(uri: str, storage_options: Mapping[str, Any] | None = None) -> int | None:
    """Detect whether a store is zarr format 2 or 3, so the output can match it.

    Parameters
    ----------
    uri : str
        Store location.
    storage_options : mapping, optional
        fsspec options for this store.

    Returns
    -------
    int or None
        ``3`` if the store has a ``zarr.json``, ``2`` if it has a ``.zgroup``, and
        ``None`` when neither is found or the store cannot be inspected. Detection is
        best-effort by design: a failure here should not fail the run.
    """
    try:
        fs, path = fsspec.core.url_to_fs(uri, **(storage_options or {}))
        if fs.exists(f"{path.rstrip('/')}/zarr.json"):
            return 3
        if fs.exists(f"{path.rstrip('/')}/.zgroup"):
            return 2
    except Exception as exc:  # noqa: BLE001 - detection is best-effort by design
        logger.warning(f"Could not detect the zarr format of {uri} ({exc}); defaulting")
    return None


def parse_zarr_format(value: str | None, key: str = "--zarr-format") -> ZarrFormat:
    """Parse a zarr format setting.

    Parameters
    ----------
    value : str or None
        The raw value; ``None`` or blank means ``'auto'``. Case-insensitive.
    key : str, optional
        Option or variable name, used in the error message.

    Returns
    -------
    {'auto', 2, 3}
        The parsed format.

    Raises
    ------
    ConfigError
        If the value is not one of :data:`ZARR_FORMATS`.
    """
    raw = (value or "").strip().lower() or "auto"
    if raw not in ZARR_FORMATS:
        raise ConfigError(f"{key} must be one of {', '.join(ZARR_FORMATS)}, got {value!r}")
    return "auto" if raw == "auto" else int(raw)  # type: ignore[return-value]


def resolve_zarr_format(
    zarr_format: ZarrFormat,
    source_uri: str | None = None,
    source_storage_options: Mapping[str, Any] | None = None,
) -> int:
    """Decide which zarr format to write.

    Parameters
    ----------
    zarr_format : {'auto', 2, 3}
        The requested format.
    source_uri : str, optional
        The input store, whose format ``'auto'`` follows.
    source_storage_options : mapping, optional
        fsspec options for the input store.

    Returns
    -------
    {2, 3}
        An explicit format as given, otherwise the input store's own format. Falls
        back to :data:`DEFAULT_ZARR_FORMAT`, with a warning, when there is no input to
        follow or detection fails.
    """
    if zarr_format != "auto":
        return int(zarr_format)
    if source_uri is None:
        return DEFAULT_ZARR_FORMAT
    detected = detect_zarr_format(source_uri, source_storage_options)
    if detected is None:
        logger.warning(f"Could not detect the input's zarr format; writing format {DEFAULT_ZARR_FORMAT}")
        return DEFAULT_ZARR_FORMAT
    logger.info(f"Input store is zarr format {detected}; writing the output in the same format")
    return detected


def write_zarr(
    ds: xr.Dataset,
    uri: str,
    storage_options: Mapping[str, Any] | None = None,
    zarr_format: ZarrFormat = "auto",
    *,
    source_uri: str | None = None,
    source_storage_options: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Write a dataset to a zarr store, replacing whatever is there.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to write.
    uri : str
        Destination store.
    storage_options : mapping, optional
        fsspec options for the destination, as built by
        :func:`~mlcast_icb.io.storage.storage_options` for the ``output`` side.
    zarr_format : {'auto', 2, 3}, optional
        Store format. ``'auto'`` matches ``source_uri`` (see :func:`resolve_zarr_format`).
    source_uri : str, optional
        The input store, for ``'auto'``.
    source_storage_options : mapping, optional
        fsspec options for the input store.
    encoding : mapping, optional
        Per-variable ``to_zarr`` encoding, e.g. chunks.

    Returns
    -------
    int
        The zarr format written.

    Raises
    ------
    StorageError
        If the destination cannot be reached or written. Every ``OSError`` counts here,
        a missing bucket or parent directory included: unlike on the input side,
        nothing about the output store is supposed to exist beforehand.
    """
    resolved = resolve_zarr_format(zarr_format, source_uri, source_storage_options)
    logger.info(f"Writing {uri} (zarr format {resolved})")
    with storage_errors("output", uri, storage_options, "write"):
        ds.to_zarr(
            uri,
            mode="w",
            consolidated=True,
            zarr_format=resolved,
            encoding=dict(encoding) if encoding else None,
            storage_options=dict(storage_options) if storage_options else None,
        )
    logger.info(f"Wrote {len(ds.data_vars)} variable(s) to {uri}")
    return resolved
