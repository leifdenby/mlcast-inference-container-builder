"""Failing to reach a store: StorageError, exit code 5, and a message that says what to check.

The failures are produced three ways, none of which touches a real object store:

* ``flaky://`` — an in-memory fsspec filesystem that raises a chosen exception on a
  chosen path, so the real xarray -> zarr -> fsspec stack carries it to our code;
* local permissions — a read-only output directory, an unreadable input store;
* s3fs against ``127.0.0.1`` on a closed port, with every AWS credential source
  switched off, for the botocore errors exactly as s3fs surfaces them.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
import xarray as xr

from mlcast_icb.errors import InputError, StorageError
from mlcast_icb.io.storage import storage_error, storage_errors, storage_exceptions, storage_options
from mlcast_icb.io.zarr_io import load_zarr, open_zarr, write_zarr

try:
    from botocore import exceptions as botocore_exceptions
except ImportError:  # optional: it arrives with s3fs
    botocore_exceptions = None

try:
    import aiohttp
except ImportError:  # optional: it arrives with s3fs
    aiohttp = None

needs_botocore = pytest.mark.skipif(botocore_exceptions is None, reason="needs botocore")

#: One attempt only, so an unreachable endpoint fails in well under a second.
ONE_ATTEMPT = json.dumps({"config_kwargs": {"retries": {"max_attempts": 1}}})

needs_permissions = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="file permissions do not stop root",
)


def closed_port_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def storage_failures() -> list[BaseException]:
    """Errors that mean the store could not be reached, as the backends raise them."""
    failures: list[BaseException] = [
        PermissionError("Access Denied"),
        OSError(errno.EIO, "Internal Error"),
        TimeoutError("timed out"),
        ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"),
    ]
    if aiohttp is not None:
        failures.append(aiohttp.ClientPayloadError("Response payload is not completed"))
    if botocore_exceptions is None:
        return failures
    return failures + [
        botocore_exceptions.NoCredentialsError(),
        botocore_exceptions.ProfileNotFound(profile="nope"),
        botocore_exceptions.EndpointConnectionError(endpoint_url="http://s3.example"),
        botocore_exceptions.ClientError({"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject"),
        # A ValueError as well as a botocore error: must not be mistaken for a bad store.
        botocore_exceptions.UnknownEndpointError(service_name="s3", region_name="nowhere"),
    ]


def ids(errors: list[BaseException]) -> list[str]:
    return [type(e).__name__ for e in errors]


# --- which exceptions count ----------------------------------------------------------


def test_oserror_always_counts_and_exception_never_does() -> None:
    found = storage_exceptions()
    assert OSError in found
    assert Exception not in found
    assert BaseException not in found


@needs_botocore
def test_botocore_errors_count_once_botocore_is_loaded() -> None:
    found = storage_exceptions()
    assert botocore_exceptions.BotoCoreError in found
    assert botocore_exceptions.ClientError in found


def test_backends_are_looked_up_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("gcsfs.retry")

    class HttpError(Exception):
        pass

    fake.HttpError = HttpError
    monkeypatch.delitem(sys.modules, "gcsfs.retry", raising=False)
    assert HttpError not in storage_exceptions()
    assert "gcsfs.retry" not in sys.modules

    monkeypatch.setitem(sys.modules, "gcsfs.retry", fake)
    assert HttpError in storage_exceptions()


# --- the message ---------------------------------------------------------------------


def test_message_names_side_uri_error_and_its_cause() -> None:
    """s3fs raises PermissionError from a ClientError, which names the refused operation."""
    cause = OSError("An error occurred (AccessDenied) when calling the GetObject operation")
    exc = PermissionError("Access Denied")
    exc.__cause__ = cause

    message = str(storage_error("SRC", "s3://bucket/in.zarr", {"anon": True}, "open", exc))
    assert message.startswith("Could not open the input store 's3://bucket/in.zarr': ")
    assert "PermissionError: Access Denied" in message
    assert "when calling the GetObject operation" in message


def test_an_empty_error_message_still_names_the_type() -> None:
    message = str(storage_error("DST", "/out.zarr", {}, "write", OSError()))
    assert "Could not write the output store '/out.zarr': OSError" in message


def test_local_hints_depend_on_the_side() -> None:
    read = str(storage_error("SRC", "/data/in.zarr", {}, "open", PermissionError()))
    write = str(storage_error("DST", "file:///data/out.zarr", {}, "write", PermissionError()))
    assert "can read every file" in read
    assert "read-only" in write
    assert "AWS" not in read + write


def test_anonymous_s3_hint_names_the_side_variables() -> None:
    message = str(storage_error("SRC", "s3://b/in.zarr", {"anon": True}, "open", OSError()))
    assert "SRC is using anonymous access" in message
    assert "SRC_AWS_PROFILE" in message
    assert "SRC_S3_ENDPOINT_URL" in message


def test_signed_s3_hint_names_the_profile_and_endpoint() -> None:
    options = {"anon": False, "profile": "dmi-minio", "client_kwargs": {"endpoint_url": "https://s3.dmi.dk"}}
    message = str(storage_error("DST", "s3://b/out.zarr", options, "write", OSError()))
    assert "DST is signing requests with profile 'dmi-minio'" in message
    assert "allowed to write this bucket" in message
    assert "https://s3.dmi.dk" in message


def test_other_schemes_get_a_generic_hint() -> None:
    message = str(storage_error("SRC", "gs://b/in.zarr", {}, "open", OSError()))
    assert "gs://" in message
    assert "SRC_STORAGE_OPTIONS" in message


def test_the_storage_error_exit_code_is_5() -> None:
    assert storage_error("SRC", "/in.zarr", {}, "open", OSError()).exit_code == 5


# --- the context manager -------------------------------------------------------------


def test_storage_errors_translates_and_chains() -> None:
    error = PermissionError("denied")
    with pytest.raises(StorageError, match="Could not read the input store '/in.zarr'") as info:
        with storage_errors("SRC", "/in.zarr", None, "read"):
            raise error
    assert info.value.__cause__ is error


def test_storage_errors_unwraps_a_group_of_storage_failures() -> None:
    group = ExceptionGroup("shard", [ExceptionGroup("inner", [PermissionError("denied")])])
    with pytest.raises(StorageError, match="PermissionError: denied") as info:
        with storage_errors("DST", "/out.zarr", None, "write"):
            raise group
    assert info.value.__cause__ is group


def test_storage_errors_leaves_a_mixed_group_alone() -> None:
    group = ExceptionGroup("mixed", [PermissionError("denied"), RuntimeError("bug")])
    with pytest.raises(ExceptionGroup):
        with storage_errors("SRC", "/in.zarr", None, "read"):
            raise group


def test_storage_errors_leaves_a_programming_error_alone() -> None:
    with pytest.raises(RuntimeError, match="a bug"):
        with storage_errors("SRC", "/in.zarr", None, "read"):
            raise RuntimeError("a bug")


# --- input side ----------------------------------------------------------------------


@pytest.mark.parametrize("error", storage_failures(), ids=ids(storage_failures()))
def test_failing_to_open_the_input_is_a_storage_error(
    flaky: Callable[..., None], flaky_input: str, error: BaseException
) -> None:
    flaky(error)
    with pytest.raises(StorageError, match="Could not open the input store") as info:
        open_zarr(flaky_input)
    assert type(info.value.__cause__) is type(error)
    assert info.value.exit_code == 5


def test_a_missing_input_store_is_still_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError) as info:
        open_zarr(str(tmp_path / "nope.zarr"))
    assert not isinstance(info.value, StorageError)
    assert info.value.exit_code == 3


def test_a_directory_that_is_not_zarr_is_an_input_error(tmp_path: Path) -> None:
    (tmp_path / "junk.zarr").mkdir()
    (tmp_path / "junk.zarr" / "README").write_text("not a store")
    with pytest.raises(InputError):
        open_zarr(str(tmp_path / "junk.zarr"))


def test_a_missing_object_on_the_input_is_an_input_error(flaky: Callable[..., None], flaky_input: str) -> None:
    """s3fs reports a missing bucket or key as FileNotFoundError: a wrong input URI."""
    flaky(FileNotFoundError("NoSuchBucket"))
    with pytest.raises(InputError):
        open_zarr(flaky_input)


def test_a_programming_error_is_not_mistaken_for_storage(flaky: Callable[..., None], flaky_input: str) -> None:
    flaky(RuntimeError("a bug, not a storage problem"))
    with pytest.raises(RuntimeError, match="a bug"):
        open_zarr(flaky_input)


def test_failing_to_read_the_data_is_a_storage_error(flaky: Callable[..., None], flaky_input: str) -> None:
    """Opening reads only metadata and coordinates; the fields are fetched on load."""
    ds = open_zarr(flaky_input)
    flaky(PermissionError("Access Denied"), match="rr/c/")
    with pytest.raises(StorageError, match="Could not read the input store 'flaky://in.zarr'"):
        load_zarr(ds, flaky_input)


def test_failing_to_read_part_of_a_shard_is_a_storage_error(
    flaky: Callable[..., None], make_dataset: Callable[..., xr.Dataset]
) -> None:
    source = make_dataset()
    nt, ny, nx = source.rr.shape
    encoding = {"rr": {"chunks": (1, ny, nx), "shards": (nt, ny, nx)}}
    source.to_zarr("flaky://in.zarr", mode="w", zarr_format=3, encoding=encoding)
    ds = open_zarr("flaky://in.zarr")
    flaky(PermissionError("Access Denied"), match="rr/c/", chunks_only=True)
    with pytest.raises(StorageError, match="Could not read the input store") as info:
        load_zarr(ds, "flaky://in.zarr")
    assert "PermissionError: Access Denied" in str(info.value)


def test_a_programming_error_in_an_exception_group_is_not_mistaken_for_storage(
    flaky: Callable[..., None], flaky_input: str
) -> None:
    ds = open_zarr(flaky_input)
    group = ExceptionGroup("chunk read failed", [PermissionError("denied"), RuntimeError("bug")])
    flaky(group, match="rr/c/")
    with pytest.raises(ExceptionGroup):
        load_zarr(ds, flaky_input)


@needs_permissions
def test_unreadable_input_store_is_a_storage_error(make_store: Callable[..., Path]) -> None:
    store = make_store()
    os.chmod(store, 0)
    try:
        with pytest.raises(StorageError, match=f"Could not open the input store '{store}'") as info:
            open_zarr(str(store))
    finally:
        os.chmod(store, 0o755)
    assert "PermissionError" in str(info.value)


# --- output side ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [*storage_failures(), FileNotFoundError("NoSuchBucket")],
    ids=[*ids(storage_failures()), "FileNotFoundError"],
)
def test_failing_to_write_the_output_is_a_storage_error(
    flaky: Callable[..., None], make_dataset: Callable[..., xr.Dataset], error: BaseException
) -> None:
    """Unlike the input, a missing destination is a storage failure: nothing should be there."""
    flaky(error)
    with pytest.raises(StorageError, match="Could not write the output store") as info:
        write_zarr(make_dataset(), "flaky://out.zarr", zarr_format=3)
    assert type(info.value.__cause__) is type(error)


def test_a_programming_error_while_writing_is_not_mistaken_for_storage(
    flaky: Callable[..., None], make_dataset: Callable[..., xr.Dataset]
) -> None:
    flaky(RuntimeError("a bug, not a storage problem"))
    with pytest.raises(RuntimeError, match="a bug"):
        write_zarr(make_dataset(), "flaky://out.zarr", zarr_format=3)


@needs_permissions
def test_read_only_output_directory_is_a_storage_error(tmp_path: Path, make_dataset: Callable[..., xr.Dataset]) -> None:
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    target = str(readonly / "out.zarr")
    try:
        with pytest.raises(StorageError, match=f"Could not write the output store '{target}'") as info:
            write_zarr(make_dataset(), target, zarr_format=3)
    finally:
        readonly.chmod(0o755)
    assert "PermissionError" in str(info.value)
    assert "read-only" in str(info.value)


# --- s3fs against a closed port ------------------------------------------------------


@pytest.mark.usefixtures("no_aws")
def test_s3_input_without_credentials_is_a_storage_error() -> None:
    pytest.importorskip("s3fs")
    uri = "s3://bucket/in.zarr"
    env = {"SRC_S3_ANON": "false", "SRC_S3_ENDPOINT_URL": closed_port_endpoint(), "SRC_STORAGE_OPTIONS": ONE_ATTEMPT}
    with pytest.raises(StorageError) as info:
        open_zarr(uri, storage_options(env, uri, "SRC"))
    message = str(info.value)
    assert "Could not open the input store 's3://bucket/in.zarr'" in message
    assert "NoCredentialsError" in message
    assert "SRC_AWS_PROFILE" in message


@pytest.mark.usefixtures("no_aws")
def test_s3_output_endpoint_unreachable_is_a_storage_error(make_dataset: Callable[..., xr.Dataset]) -> None:
    pytest.importorskip("s3fs")
    endpoint = closed_port_endpoint()
    uri = "s3://bucket/out.zarr"
    env = {"DST_S3_ENDPOINT_URL": endpoint, "DST_STORAGE_OPTIONS": ONE_ATTEMPT}
    with pytest.raises(StorageError) as info:
        write_zarr(make_dataset(), uri, storage_options(env, uri, "DST"), zarr_format=3)
    message = str(info.value)
    assert "Could not write the output store 's3://bucket/out.zarr'" in message
    assert "EndpointConnectionError" in message
    assert "DST is using anonymous access" in message
    assert endpoint in message
