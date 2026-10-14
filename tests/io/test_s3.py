"""Real S3 traffic, served by local moto servers.

``test_storage.py`` checks the options we build; these check that s3fs actually reads
and writes with them. Each server is a separate ``moto_server`` process, so two of them
are genuinely two hosts with separate state — which is what the per-side
(``--input-*`` / ``--output-*``) storage options are for.
Nothing here touches the network beyond localhost.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import xarray as xr
from io_testdata import build_dataset

from mlcast_icb.errors import InputError
from mlcast_icb.io.storage import StorageSettings, storage_options
from mlcast_icb.io.zarr_io import load_zarr, open_zarr, write_zarr

boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto.server")
s3fs = pytest.importorskip("s3fs")

KEY = "testing-key"
SECRET = "testing-secret"
REGION = "us-east-1"

#: Every ambient AWS variable that could leak a developer's or CI runner's real
#: configuration into these tests, or sign requests the test means to send unsigned.
_AMBIENT_AWS_VARS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_moto() -> tuple[subprocess.Popen, str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/moto-api/", timeout=1)
            return proc, url
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("moto server did not start")


@pytest.fixture(scope="module")
def s3_hosts() -> Iterator[list[str]]:
    """Two independent S3 endpoints, standing in for two different object stores."""
    procs, urls = [], []
    try:
        for _ in range(2):
            proc, url = _start_moto()
            procs.append(proc)
            urls.append(url)
        yield urls
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait(timeout=10)


@pytest.fixture
def aws_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], None]:
    """Isolate botocore from the real ~/.aws and AWS_* environment.

    Returns a function that writes profiles into a throwaway config and credentials
    file, the way a mounted ``~/.aws`` would carry them into the container.
    """
    for name in _AMBIENT_AWS_VARS:
        monkeypatch.delenv(name, raising=False)
    # s3fs reuses a filesystem for the same options, and with the standard AWS variables
    # the options alone ({"anon": False}) no longer say which configuration it was made
    # with; each test has its own, so start each with no cached filesystem.
    s3fs.S3FileSystem.clear_instance_cache()
    config = tmp_path / "aws_config"
    credentials = tmp_path / "aws_credentials"
    config.write_text("")
    credentials.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def _add_profile(name: str, endpoint: str) -> None:
        with config.open("a") as f:
            f.write(f"[profile {name}]\nregion = {REGION}\nendpoint_url = {endpoint}\n\n")
        with credentials.open("a") as f:
            f.write(f"[{name}]\naws_access_key_id = {KEY}\naws_secret_access_key = {SECRET}\n\n")

    return _add_profile


def _client(endpoint: str):
    # A fresh session: boto3's default one keeps the AWS configuration of the test that made it.
    return boto3.session.Session().client(
        "s3", endpoint_url=endpoint, region_name=REGION, aws_access_key_id=KEY, aws_secret_access_key=SECRET
    )


def _bucket(endpoint: str) -> str:
    """Create a uniquely named bucket, so no test sees another's objects or caches."""
    name = f"b-{uuid.uuid4().hex[:12]}"
    _client(endpoint).create_bucket(Bucket=name)
    return name


def _keys(endpoint: str, bucket: str) -> set[str]:
    pages = _client(endpoint).get_paginator("list_objects_v2").paginate(Bucket=bucket)
    return {obj["Key"] for page in pages for obj in page.get("Contents", [])}


def _signed(endpoint: str) -> dict:
    return {"key": KEY, "secret": SECRET, "client_kwargs": {"endpoint_url": endpoint}}


def _put_input(endpoint: str, zarr_format: int = 3, public: bool = False) -> str:
    """Upload a synthetic input store and return its s3:// URI.

    ``public`` uploads every object as ``public-read``, as a public dataset bucket
    would serve them; moto, like S3, refuses unsigned reads of private objects.
    """
    uri = f"s3://{_bucket(endpoint)}/in.zarr"
    options = _signed(endpoint)
    if public:
        options["s3_additional_kwargs"] = {"ACL": "public-read"}
    build_dataset().to_zarr(uri, mode="w", consolidated=True, zarr_format=zarr_format, storage_options=options)
    return uri


def _copy(
    source: str, target: str, input: StorageSettings | None = None, output: StorageSettings | None = None
) -> None:
    """Read the input and write it to the output, as a run would, with each side's options."""
    input_options = storage_options("input", source, input)
    output_options = storage_options("output", target, output)
    ds = load_zarr(open_zarr(source, input_options), source, input_options)
    write_zarr(ds, target, output_options, source_uri=source, source_storage_options=input_options)


def _open_output(endpoint: str, uri: str) -> xr.Dataset:
    return xr.open_zarr(uri, storage_options=_signed(endpoint))


def test_profile_endpoint_from_aws_config(
    s3_hosts: list[str], aws_home: Callable[[str, str], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The standard AWS configuration: AWS_PROFILE names a profile carrying host and credentials.

    No endpoint is set at all, so if s3fs ignored the profile's ``endpoint_url`` the
    requests would go to real AWS and fail.
    """
    host = s3_hosts[0]
    aws_home("rig", host)
    monkeypatch.setenv("AWS_PROFILE", "rig")
    source = _put_input(host)
    target = f"s3://{_bucket(host)}/out.zarr"

    _copy(source, target)

    xr.testing.assert_identical(_open_output(host, target).load(), build_dataset())


def test_two_hosts_with_separate_profiles(s3_hosts: list[str], aws_home: Callable[[str, str], None]) -> None:
    """--input-aws-profile and --output-aws-profile route each side to its own host."""
    src_host, dst_host = s3_hosts
    aws_home("reader", src_host)
    aws_home("writer", dst_host)
    source = _put_input(src_host)
    target_bucket = _bucket(dst_host)
    target = f"s3://{target_bucket}/out.zarr"

    _copy(source, target, StorageSettings(aws_profile="reader"), StorageSettings(aws_profile="writer"))

    assert _open_output(dst_host, target).sizes["time"] == 4
    # The output bucket exists only on the destination host, so a write that went to
    # the source host instead would have failed rather than landing there silently.
    source_buckets = {b["Name"] for b in _client(src_host).list_buckets()["Buckets"]}
    assert target_bucket not in source_buckets


def test_a_side_profile_wins_over_the_standard_profile(
    s3_hosts: list[str], aws_home: Callable[[str, str], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AWS_PROFILE configures the input; --output-aws-profile takes the output elsewhere."""
    src_host, dst_host = s3_hosts
    aws_home("reader", src_host)
    aws_home("writer", dst_host)
    monkeypatch.setenv("AWS_PROFILE", "reader")
    source = _put_input(src_host)
    target = f"s3://{_bucket(dst_host)}/out.zarr"

    _copy(source, target, output=StorageSettings(aws_profile="writer"))
    assert _open_output(dst_host, target).sizes["time"] == 4


@pytest.mark.usefixtures("aws_home")
def test_endpoint_options_and_standard_keys(s3_hosts: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-config-file case: each side's endpoint as an option, keys in the standard variables."""
    src_host, dst_host = s3_hosts
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    source = _put_input(src_host)
    target = f"s3://{_bucket(dst_host)}/out.zarr"

    _copy(source, target, StorageSettings(s3_endpoint_url=src_host), StorageSettings(s3_endpoint_url=dst_host))
    assert _open_output(dst_host, target).sizes["time"] == 4


@pytest.mark.usefixtures("aws_home")
def test_standard_endpoint_variable_applies_to_both_sides(s3_hosts: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """AWS_ENDPOINT_URL reaches s3fs through botocore, with no option set."""
    host = s3_hosts[0]
    for name, value in (("AWS_ACCESS_KEY_ID", KEY), ("AWS_SECRET_ACCESS_KEY", SECRET), ("AWS_ENDPOINT_URL", host)):
        monkeypatch.setenv(name, value)
    source = _put_input(host)
    target = f"s3://{_bucket(host)}/out.zarr"

    _copy(source, target)
    assert _open_output(host, target).sizes["time"] == 4


def test_anonymous_source_with_signed_destination(s3_hosts: list[str], aws_home: Callable[[str, str], None]) -> None:
    """With no credentials anywhere, the source side must go out unsigned.

    Signing here would raise ``NoCredentialsError``, so a successful read is the proof
    that anonymous access was actually used rather than merely configured.
    """
    src_host, dst_host = s3_hosts
    aws_home("writer", dst_host)
    source = _put_input(src_host, public=True)
    target = f"s3://{_bucket(dst_host)}/out.zarr"

    _copy(source, target, StorageSettings(s3_endpoint_url=src_host), StorageSettings(aws_profile="writer"))
    assert _open_output(dst_host, target).sizes["time"] == 4


def test_output_matches_input_zarr_format_over_s3(
    s3_hosts: list[str], aws_home: Callable[[str, str], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Format detection has to work through s3fs, not just on a local path."""
    host = s3_hosts[0]
    aws_home("rig", host)
    monkeypatch.setenv("AWS_PROFILE", "rig")
    source = _put_input(host, zarr_format=2)
    target_bucket = _bucket(host)
    target = f"s3://{target_bucket}/out.zarr"

    _copy(source, target)

    keys = _keys(host, target_bucket)
    assert "out.zarr/.zgroup" in keys
    assert "out.zarr/zarr.json" not in keys


def test_missing_input_store_on_s3_is_an_input_error(s3_hosts: list[str], aws_home: Callable[[str, str], None]) -> None:
    host = s3_hosts[0]
    aws_home("rig", host)
    uri = f"s3://{_bucket(host)}/does-not-exist.zarr"
    with pytest.raises(InputError):
        open_zarr(uri, storage_options("input", uri, StorageSettings(aws_profile="rig")))
