"""Fixtures for the I/O tests: synthetic datasets, a failure-injecting filesystem, no AWS."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import xarray as xr
from fsspec import register_implementation
from fsspec.implementations.memory import MemoryFileSystem
from io_testdata import build_dataset


@pytest.fixture
def make_dataset() -> Callable[..., xr.Dataset]:
    """Return :func:`build_dataset`, for tests that never touch disk."""
    return build_dataset


@pytest.fixture
def make_store(tmp_path: Path) -> Callable[..., Path]:
    """Return a function that writes a synthetic zarr store and returns its path."""

    def _make(name: str = "in.zarr", zarr_format: int = 3, **kwargs: int) -> Path:
        path = tmp_path / name
        build_dataset(**kwargs).to_zarr(path, mode="w", consolidated=True, zarr_format=zarr_format)
        return path

    return _make


class FlakyFileSystem(MemoryFileSystem):
    """An in-memory filesystem that raises ``error`` for every path containing ``match``."""

    protocol = "flaky"
    store: dict = {}
    pseudo_dirs = [""]
    error: BaseException | None = None
    match = ""
    #: Spare suffix reads, which is how zarr reads a shard's index.
    chunks_only = False

    def cat_file(self, path, start=None, end=None, **kwargs):
        suffix = self.chunks_only and start is not None and start < 0
        if self.error is not None and self.match in path and not suffix:
            raise self.error
        return super().cat_file(path, start=start, end=end, **kwargs)

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        if self.error is not None and self.match in path:
            raise self.error
        return super().pipe_file(path, value, mode=mode, **kwargs)


register_implementation("flaky", FlakyFileSystem, clobber=True)


@pytest.fixture
def flaky(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Return a function that arms the flaky filesystem, starting from an empty store."""
    FlakyFileSystem.store.clear()
    FlakyFileSystem.pseudo_dirs[:] = [""]

    def _arm(error: BaseException, match: str = "", chunks_only: bool = False) -> None:
        monkeypatch.setattr(FlakyFileSystem, "error", error)
        monkeypatch.setattr(FlakyFileSystem, "match", match)
        monkeypatch.setattr(FlakyFileSystem, "chunks_only", chunks_only)

    yield _arm
    FlakyFileSystem.store.clear()


@pytest.fixture
def flaky_input(flaky: Callable[..., None]) -> str:
    """Write a valid synthetic input store onto the flaky filesystem."""
    build_dataset().to_zarr("flaky://in.zarr", mode="w", consolidated=True, zarr_format=3)
    return "flaky://in.zarr"


@pytest.fixture
def no_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Switch off every AWS credential source, so nothing here can find real ones."""
    for key in list(os.environ):
        if key.startswith(("AWS_", "SRC_", "DST_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
