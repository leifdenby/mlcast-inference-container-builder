"""netCDF payloads: sniffing, reading uploaded bytes, rejecting bad uploads, writing.

netCDF3 goes through scipy, which is always installed. netCDF4 needs h5netcdf (or
netCDF4 for writing); those tests skip without it.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from mlcast_icb.errors import InputError, PayloadTooLargeError
from mlcast_icb.io.netcdf_io import (
    HDF5_MAGIC,
    netcdf_encoding,
    read_netcdf_bytes,
    sniff_netcdf_format,
    write_netcdf,
)


def netcdf3_bytes(ds: xr.Dataset) -> bytes:
    """Serialise a dataset as netCDF3 with scipy, as a client might upload it."""
    return ds.to_netcdf(engine="scipy")


def drop_compression(encoding: dict[str, dict]) -> dict[str, dict]:
    """Strip the zlib entries, which netCDF3 cannot store, keeping dtype and packing."""
    return {name: {k: v for k, v in enc.items() if k not in ("zlib", "complevel")} for name, enc in encoding.items()}


# --- sniffing ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "expected"),
    [(HDF5_MAGIC + b"\x00" * 8, "NETCDF4"), (b"CDF\x01\x00\x00", "NETCDF3"), (b"CDF\x02\x00\x00", "NETCDF3")],
)
def test_formats_are_recognised(head: bytes, expected: str) -> None:
    assert sniff_netcdf_format(head) == expected


@pytest.mark.parametrize(
    ("head", "match"),
    [
        (b"", "empty"),
        (b"GRIB\x00\x00\x00\x01", "not a netCDF file"),
        (b"\x89PNG\r\n\x1a\n", "not a netCDF file"),
        (b"CDF", "not a netCDF file"),
        (b"CDF\x05\x00\x00", "CDF-5"),
    ],
)
def test_other_payloads_are_rejected(head: bytes, match: str) -> None:
    with pytest.raises(InputError, match=match):
        sniff_netcdf_format(head)


# --- reading -------------------------------------------------------------------------


@pytest.mark.parametrize("as_stream", [False, True], ids=["bytes", "stream"])
def test_netcdf3_round_trip(make_dataset: Callable[..., xr.Dataset], as_stream: bool) -> None:
    # Given a netCDF3 upload, as bytes or as a file object
    source = make_dataset()
    payload = netcdf3_bytes(source)
    data = io.BytesIO(payload) if as_stream else payload

    # When it is read
    ds = read_netcdf_bytes(data, max_bytes=len(payload))

    # Then the dataset is complete and in memory
    xr.testing.assert_identical(ds, source)
    assert all(var._in_memory for var in ds.variables.values())


def test_a_spilled_upload_is_still_read(
    monkeypatch: pytest.MonkeyPatch, make_dataset: Callable[..., xr.Dataset]
) -> None:
    """Above the spool threshold the upload goes to disk; reading must not notice."""
    monkeypatch.setattr("mlcast_icb.io.netcdf_io.SPOOL_MAX_MEMORY", 16)
    source = make_dataset()
    xr.testing.assert_identical(read_netcdf_bytes(io.BytesIO(netcdf3_bytes(source))), source)


def test_netcdf4_round_trip(tmp_path: Path, make_dataset: Callable[..., xr.Dataset]) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    source = make_dataset()
    path = write_netcdf(source, tmp_path / "in.nc")
    assert path.read_bytes().startswith(HDF5_MAGIC)
    ds = read_netcdf_bytes(path.read_bytes())
    xr.testing.assert_identical(ds, source)


def test_a_netcdf4_upload_without_h5netcdf_is_a_packaging_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not the client's fault, so not an InputError: the server should answer 500."""
    monkeypatch.setattr("mlcast_icb.io.netcdf_io._has_h5netcdf", lambda: False)
    with pytest.raises(ImportError, match="h5netcdf"):
        read_netcdf_bytes(HDF5_MAGIC + b"\x00" * 64)


@pytest.mark.parametrize("data", [b"", io.BytesIO(b"")], ids=["bytes", "stream"])
def test_an_empty_upload_is_rejected(data: bytes | io.BytesIO) -> None:
    with pytest.raises(InputError, match="empty"):
        read_netcdf_bytes(data)


def test_garbage_is_rejected() -> None:
    with pytest.raises(InputError, match="not a netCDF file") as info:
        read_netcdf_bytes(b"this is not netCDF at all" * 10)
    assert info.value.exit_code == 3


def test_a_truncated_netcdf3_file_is_rejected(make_dataset: Callable[..., xr.Dataset]) -> None:
    payload = netcdf3_bytes(make_dataset())
    with pytest.raises(InputError, match="could not be read"):
        read_netcdf_bytes(payload[: len(payload) // 2])


def test_a_corrupt_hdf5_file_is_rejected() -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    with pytest.raises(InputError, match="could not be read"):
        read_netcdf_bytes(HDF5_MAGIC + b"\x00garbage" * 64)


@pytest.mark.parametrize("as_stream", [False, True], ids=["bytes", "stream"])
def test_an_oversize_upload_is_rejected(make_dataset: Callable[..., xr.Dataset], as_stream: bool) -> None:
    payload = netcdf3_bytes(make_dataset())
    data = io.BytesIO(payload) if as_stream else payload
    with pytest.raises(PayloadTooLargeError, match=f"limit of {len(payload) - 1} bytes") as info:
        read_netcdf_bytes(data, max_bytes=len(payload) - 1)
    # Still an input error, so a CLI exit is 3; only the server tells it apart (413).
    assert isinstance(info.value, InputError)


def test_an_oversize_stream_is_not_read_to_the_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlcast_icb.io.netcdf_io._COPY_CHUNK", 10)
    stream = io.BytesIO(b"CDF\x01" + b"\x00" * 1000)
    with pytest.raises(PayloadTooLargeError):
        read_netcdf_bytes(stream, max_bytes=50)
    assert stream.tell() <= 60


# --- writing -------------------------------------------------------------------------


def test_compression_applies_to_data_variables_only(make_dataset: Callable[..., xr.Dataset]) -> None:
    encoding = netcdf_encoding(make_dataset(), complevel=6)
    assert set(encoding) == {"rr", "crs"}
    assert encoding["rr"] == {"zlib": True, "complevel": 6}


def test_complevel_0_writes_uncompressed(make_dataset: Callable[..., xr.Dataset]) -> None:
    assert netcdf_encoding(make_dataset(), complevel=0)["rr"] == {}


@pytest.mark.parametrize(("complevel", "dtype"), [(-1, None), (10, None), (4, "float64"), (4, "int8")])
def test_bad_options_are_rejected(make_dataset: Callable[..., xr.Dataset], complevel: int, dtype: str | None) -> None:
    with pytest.raises(ValueError):
        netcdf_encoding(make_dataset(), complevel=complevel, dtype=dtype)  # type: ignore[arg-type]


def test_float32_casts_float_variables_only(make_dataset: Callable[..., xr.Dataset]) -> None:
    ds = make_dataset()
    ds["rr"] = ds.rr.astype("float64")
    encoding = netcdf_encoding(ds, dtype="float32")
    assert encoding["rr"]["dtype"] == "float32"
    assert "dtype" not in encoding["crs"]  # int32 grid mapping is left alone


def test_float16_packs_into_int16_and_unpacks_within_one_step(make_dataset: Callable[..., xr.Dataset]) -> None:
    # Given a float field with a NaN, packed as int16
    source = make_dataset()
    encoding = netcdf_encoding(source, dtype="float16")
    rr = encoding["rr"]
    assert rr["dtype"] == "int16"

    # When it is written (netCDF3 here, so this runs without h5netcdf) and read back
    ds = xr.open_dataset(source.to_netcdf(engine="scipy", encoding=drop_compression(encoding)), engine="scipy")
    raw = xr.open_dataset(
        source.to_netcdf(engine="scipy", encoding=drop_compression(encoding)), engine="scipy", mask_and_scale=False
    )

    # Then it is stored as int16, decodes to float32, keeps the NaN, and is within one
    # quantisation step everywhere
    assert raw.rr.dtype == np.int16
    assert ds.rr.dtype == np.float32
    assert np.isnan(ds.rr.values[0, 0, 0])
    np.testing.assert_allclose(ds.rr.values, source.rr.values, atol=float(rr["scale_factor"]), equal_nan=True)
    assert int(raw.rr.min()) == -32768  # the NaN, as fill
    assert int(raw.rr.where(raw.rr != -32768).min()) == -32767
    assert int(raw.rr.max()) <= 32767


@pytest.mark.parametrize("values", [np.full((2, 3), 7.5), np.full((2, 3), np.nan)], ids=["constant", "all-nan"])
def test_float16_packing_handles_a_degenerate_range(values: np.ndarray) -> None:
    source = xr.Dataset({"v": (("y", "x"), values.astype("float32"))})
    encoding = drop_compression(netcdf_encoding(source, dtype="float16"))
    ds = xr.open_dataset(source.to_netcdf(engine="scipy", encoding=encoding), engine="scipy")
    np.testing.assert_allclose(ds.v.values, values, equal_nan=True)


@pytest.mark.parametrize("dtype", [None, "float32", "float16"])
def test_write_netcdf_produces_compressed_netcdf4(
    tmp_path: Path, make_dataset: Callable[..., xr.Dataset], dtype: str | None
) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    source = make_dataset()
    path = write_netcdf(source, tmp_path / "out.nc", complevel=4, dtype=dtype)  # type: ignore[arg-type]
    assert path == tmp_path / "out.nc"
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert ds.rr.encoding.get("zlib") or ds.rr.encoding.get("compression") == "gzip"
        assert ds.rr.dtype == np.float32
        packed = dtype == "float16"
        assert ds.rr.encoding["dtype"] == (np.int16 if packed else np.float32)
        atol = float(netcdf_encoding(source, dtype="float16")["rr"]["scale_factor"]) if packed else 0
        np.testing.assert_allclose(ds.rr.values, source.rr.values, atol=atol, equal_nan=True)
