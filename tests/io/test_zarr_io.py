"""Zarr round trips, in either format, and choosing the format to write."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import xarray as xr
import zarr

from mlcast_icb.errors import ConfigError
from mlcast_icb.io import zarr_io
from mlcast_icb.io.zarr_io import (
    detect_zarr_format,
    load_zarr,
    open_zarr,
    parse_zarr_format,
    resolve_zarr_format,
    write_zarr,
)

# The root metadata file each format writes, and the one detect_zarr_format looks for.
MARKER = {2: ".zgroup", 3: "zarr.json"}


def written_format(path: Path) -> int:
    """Return the format a store was written in, after checking all of it agrees."""
    group = zarr.open_group(path, mode="r")
    formats = {group.metadata.zarr_format}
    formats |= {array.metadata.zarr_format for _, array in group.arrays()}
    assert len(formats) == 1, f"{path} mixes zarr formats {sorted(formats)}"
    (found,) = formats
    assert {m for m in MARKER.values() if (path / m).exists()} == {MARKER[found]}
    return found


# --- round trip ----------------------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_round_trip_preserves_values_coordinates_and_attributes(
    tmp_path: Path, make_dataset: Callable[..., xr.Dataset], zarr_format: int
) -> None:
    # Given a dataset with a NaN, CF attributes and a grid-mapping variable
    source = make_dataset()
    target = tmp_path / "out.zarr"

    # When it is written and read back
    assert write_zarr(source, str(target), zarr_format=zarr_format) == zarr_format
    opened = open_zarr(str(target))
    loaded = load_zarr(opened, str(target))

    # Then nothing a reader sees has changed, and the store is in the requested format
    xr.testing.assert_identical(loaded, source)
    assert written_format(target) == zarr_format


def test_write_replaces_an_existing_store(tmp_path: Path, make_dataset: Callable[..., xr.Dataset]) -> None:
    target = str(tmp_path / "out.zarr")
    write_zarr(make_dataset(nt=4), target, zarr_format=3)
    write_zarr(make_dataset(nt=2, seed=1), target, zarr_format=3)
    assert open_zarr(target).sizes["time"] == 2


def test_encoding_reaches_the_store(tmp_path: Path, make_dataset: Callable[..., xr.Dataset]) -> None:
    ds = make_dataset()
    target = str(tmp_path / "out.zarr")
    write_zarr(ds, target, zarr_format=3, encoding={"rr": {"chunks": (1, 12, 16)}})
    assert open_zarr(target).rr.encoding["chunks"] == (1, 12, 16)


def test_open_is_lazy(make_store: Callable[..., Path]) -> None:
    ds = open_zarr(str(make_store()))
    assert not ds.rr.variable._in_memory


# --- detection and resolution --------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_detects_the_format_of_a_store(make_store: Callable[..., Path], zarr_format: int) -> None:
    assert detect_zarr_format(str(make_store(zarr_format=zarr_format))) == zarr_format


def test_a_directory_that_is_not_a_store_is_undetected(tmp_path: Path) -> None:
    assert detect_zarr_format(str(tmp_path)) is None


def test_detection_failure_is_not_fatal() -> None:
    assert detect_zarr_format("nosuchprotocol://x/in.zarr") is None


@pytest.mark.parametrize(
    ("value", "expected"), [(None, "auto"), ("", "auto"), ("auto", "auto"), ("AUTO", "auto"), ("2", 2), (" 3 ", 3)]
)
def test_accepted_values(value: str | None, expected: object) -> None:
    assert parse_zarr_format(value) == expected


@pytest.mark.parametrize("value", ["1", "4", "v3", "zarr3", "3.0"])
def test_other_values_are_rejected(value: str) -> None:
    with pytest.raises(ConfigError, match="--zarr-format must be one of"):
        parse_zarr_format(value)
    with pytest.raises(ConfigError, match="MLCAST_ZARR_FORMAT must be one of"):
        parse_zarr_format(value, "MLCAST_ZARR_FORMAT")


@pytest.mark.parametrize(("requested", "detected"), [(2, 3), (3, 2)])
def test_an_explicit_format_overrides_the_input(monkeypatch: pytest.MonkeyPatch, requested: int, detected: int) -> None:
    monkeypatch.setattr(zarr_io, "detect_zarr_format", lambda *_: detected)
    assert resolve_zarr_format(requested, "/in.zarr") == requested


@pytest.mark.parametrize("detected", [2, 3])
def test_auto_follows_the_input(monkeypatch: pytest.MonkeyPatch, detected: int) -> None:
    monkeypatch.setattr(zarr_io, "detect_zarr_format", lambda *_: detected)
    assert resolve_zarr_format("auto", "/in.zarr") == detected


def test_auto_falls_back_to_format_3_when_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zarr_io, "detect_zarr_format", lambda *_: None)
    assert resolve_zarr_format("auto", "/in.zarr") == 3


def test_auto_without_an_input_writes_format_3() -> None:
    assert resolve_zarr_format("auto") == 3


@pytest.mark.parametrize("source_format", [2, 3])
def test_write_auto_matches_the_source_store(
    tmp_path: Path, make_store: Callable[..., Path], source_format: int
) -> None:
    source = make_store(zarr_format=source_format)
    target = tmp_path / "out.zarr"
    written = write_zarr(open_zarr(str(source)).load(), str(target), source_uri=str(source))
    assert written == source_format
    assert written_format(target) == source_format
