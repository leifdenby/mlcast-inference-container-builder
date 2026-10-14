"""Per-side storage options (``--input-*`` / ``--output-*``), and their masking in the log."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from typing import Any

import fsspec
import fsspec.implementations.memory
import pytest
import xarray as xr
from loguru import logger

from mlcast_icb.errors import ConfigError
from mlcast_icb.io.storage import (
    AMBIENT_CREDENTIAL_VARS,
    StorageSettings,
    log_storage_options,
    redact,
    redact_json,
    side_variable,
    storage_options,
)
from mlcast_icb.io.zarr_io import write_zarr

S3 = "s3://bucket/in.zarr"
OTHER_S3 = "s3://other-bucket/out.zarr"


def opts(uri: str, side: str = "input", env: dict[str, str] | None = None, **settings: Any) -> dict[str, Any]:
    """The storage options for ``uri`` with the side's ``settings``, in the environment ``env`` (none by default)."""
    return storage_options(side, uri, StorageSettings(**settings), env or {})


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    """Collect every message logged at INFO or above, formatted as a sink sees it."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    yield messages
    logger.remove(sink)


def test_local_paths_get_no_options() -> None:
    assert opts("/data/in.zarr") == {}


def test_profile_applies_to_s3() -> None:
    assert opts(S3, aws_profile="reader")["profile"] == "reader"


def test_input_and_output_are_independent() -> None:
    assert opts(S3, "input", aws_profile="era5-reader")["profile"] == "era5-reader"
    assert opts(OTHER_S3, "output", aws_profile="dmi-minio")["profile"] == "dmi-minio"


def test_the_standard_aws_profile_is_left_to_botocore() -> None:
    """AWS_PROFILE applies to both sides through botocore itself, with its usual precedence."""
    options = opts(S3, env={"AWS_PROFILE": "shared"})
    assert "profile" not in options
    assert options["anon"] is False  # but it counts as credentials


def test_no_endpoint_option_when_the_profile_carries_it() -> None:
    """The AWS config is the normal home for endpoint_url, so we pass nothing."""
    assert "client_kwargs" not in opts(S3, aws_profile="dmi-minio")


def test_endpoint_option_is_available() -> None:
    options = opts(OTHER_S3, "output", s3_endpoint_url="https://s3.dmi.dk")
    assert options["client_kwargs"] == {"endpoint_url": "https://s3.dmi.dk"}


def test_the_standard_endpoint_variables_are_left_to_botocore() -> None:
    options = opts(S3, env={"AWS_ENDPOINT_URL": "https://s3.dmi.dk", "AWS_ENDPOINT_URL_S3": "https://s3.dmi.dk"})
    assert "client_kwargs" not in options


def test_the_old_unprefixed_s3_endpoint_url_is_not_read() -> None:
    assert "client_kwargs" not in opts(S3, env={"S3_ENDPOINT_URL": "https://s3.dmi.dk"})


def test_side_variable_names() -> None:
    assert side_variable("input", "AWS_PROFILE") == "MLCAST_INPUT_AWS_PROFILE"
    assert side_variable("output", "S3_ENDPOINT_URL") == "MLCAST_OUTPUT_S3_ENDPOINT_URL"


# --- anonymous by default ------------------------------------------------------------


def test_anon_by_default_when_nothing_is_configured() -> None:
    assert opts(S3)["anon"] is True


def test_a_profile_turns_signing_on() -> None:
    assert opts(S3, aws_profile="reader")["anon"] is False


def test_ambient_aws_keys_turn_signing_on() -> None:
    assert opts(S3, env={"AWS_ACCESS_KEY_ID": "AKIA...", "AWS_SECRET_ACCESS_KEY": "..."})["anon"] is False


def test_the_environment_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in AMBIENT_CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    assert storage_options("input", S3)["anon"] is True
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    assert storage_options("input", S3)["anon"] is False


@pytest.mark.parametrize(
    "var",
    ["AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_SESSION_TOKEN"],
)
def test_ambient_role_credentials_turn_signing_on(var: str) -> None:
    """IAM roles on ECS/EKS/EC2 are credentials, even with no keys in the environment."""
    assert opts(S3, env={var: "something"})["anon"] is False


def test_credentials_in_the_json_escape_hatch_turn_signing_on() -> None:
    assert opts(S3, storage_options='{"key": "AKIA...", "secret": "shh"}')["anon"] is False


def test_s3fs_credential_aliases_turn_signing_on() -> None:
    assert opts(S3, storage_options='{"username": "AKIA...", "password": "shh"}')["anon"] is False


@pytest.mark.parametrize("name", ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"])
def test_credentials_inside_client_kwargs_turn_signing_on(name: str) -> None:
    assert opts(S3, storage_options=json.dumps({"client_kwargs": {name: "shh"}}))["anon"] is False


def test_client_kwargs_without_credentials_stay_anonymous() -> None:
    assert opts(S3, storage_options='{"client_kwargs": {"region_name": "eu-north-1"}}')["anon"] is True


def test_explicit_anon_wins_over_credentials() -> None:
    assert opts(S3, aws_profile="reader", s3_anon=True)["anon"] is True


def test_explicit_anon_false_wins_over_the_default() -> None:
    assert opts(S3, s3_anon=False)["anon"] is False


def test_json_can_override_anon() -> None:
    assert opts(S3, s3_anon=True, storage_options='{"anon": false}')["anon"] is False


def test_anon_is_not_set_for_non_s3_schemes() -> None:
    assert "anon" not in opts("gs://bucket/in.zarr")
    assert opts("/data/in.zarr") == {}


# --- the JSON escape hatch -----------------------------------------------------------


def test_s3_options_are_ignored_for_other_schemes(log_lines: list[str]) -> None:
    assert opts("gs://bucket/in.zarr", aws_profile="reader") == {}
    assert opts("/data/in.zarr", s3_anon=True) == {}
    assert any("MLCAST_INPUT_AWS_PROFILE" in line and "not an S3 URI" in line for line in log_lines)


def test_json_reaches_any_option() -> None:
    options = opts(S3, storage_options='{"requester_pays": true, "config_kwargs": {"max_pool": 4}}')
    assert options["requester_pays"] is True
    assert options["config_kwargs"] == {"max_pool": 4}


def test_json_applies_to_non_s3_schemes_too() -> None:
    options = opts("gs://bucket/out.zarr", "output", storage_options='{"project": "my-gcp-project"}')
    assert options == {"project": "my-gcp-project"}


def test_json_overrides_the_dedicated_options() -> None:
    assert opts(S3, aws_profile="reader", storage_options='{"profile": "override"}')["profile"] == "override"


@pytest.mark.parametrize(("value", "message"), [("not json", "must be valid JSON"), ('["a"]', "must be a JSON object")])
def test_malformed_json_is_rejected_naming_the_variable(value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=f"MLCAST_OUTPUT_STORAGE_OPTIONS {message}"):
        opts(S3, "output", storage_options=value)


def test_redact_json_masks_secrets_and_malformed_values() -> None:
    assert json.loads(redact_json('{"key": "AKIA", "profile": "p"}')) == {"key": "***", "profile": "p"}
    assert redact_json("not json with a secret") == "*** (not a JSON object)"
    assert redact_json(None) is None


# --- redaction of the log line -------------------------------------------------------


def test_nested_secrets_are_redacted() -> None:
    options = {
        "profile": "dmi-minio",
        "client_kwargs": {
            "endpoint_url": "https://s3.dmi.dk",
            "region_name": "eu-north-1",
            "aws_access_key_id": "AKIAnested",
            "aws_secret_access_key": "nested-shh",
            "aws_session_token": "nested-session",
        },
        "config_kwargs": {"s3": {"addressing_style": "path"}, "max_pool_connections": 4},
    }
    assert redact(options) == {
        "profile": "dmi-minio",
        "client_kwargs": {
            "endpoint_url": "https://s3.dmi.dk",
            "region_name": "eu-north-1",
            "aws_access_key_id": "***",
            "aws_secret_access_key": "***",
            "aws_session_token": "***",
        },
        "config_kwargs": {"s3": {"addressing_style": "path"}, "max_pool_connections": 4},
    }


def test_secret_name_masks_the_whole_value_even_when_it_is_an_object() -> None:
    """A gcsfs service-account dict under ``token`` goes as one, not field by field."""
    token = {"type": "service_account", "client_email": "sa@p.iam", "private_key": "-----"}
    assert redact({"project": "my-gcp-project", "token": token}) == {"project": "my-gcp-project", "token": "***"}


def test_service_account_fields_are_masked_under_a_non_secret_name() -> None:
    info = {
        "type": "service_account",
        "project_id": "my-gcp-project",
        "private_key_id": "abc123",
        "private_key": "-----BEGIN PRIVATE KEY-----",
        "client_email": "runner@my-gcp-project.iam.gserviceaccount.com",
        "client_secret": "oauth-shh",
        "refresh_token": "refresh-shh",
    }
    redacted = redact({"session_kwargs": {"info": info}})["session_kwargs"]["info"]
    assert redacted["private_key_id"] == "***"
    assert redacted["private_key"] == "***"
    assert redacted["client_secret"] == "***"
    assert redacted["refresh_token"] == "***"
    # Identifies the account rather than authenticating as it, so it stays readable.
    assert redacted["client_email"] == info["client_email"]
    assert redacted["project_id"] == "my-gcp-project"


def test_lists_of_dicts_are_redacted() -> None:
    options = {
        "candidates": [
            {"profile": "first", "secret": "list-shh"},
            ("tuple-item", {"password": "tuple-shh"}),
            "plain",
            3,
        ]
    }
    assert redact(options) == {
        "candidates": [
            {"profile": "first", "secret": "***"},
            ("tuple-item", {"password": "***"}),
            "plain",
            3,
        ]
    }


@pytest.mark.parametrize(
    "name",
    [
        "key",
        "secret",
        "token",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "auth",
        "Authorization",
        "SSECustomerKey",
        "sas_token",
        "account_key",
        "connection_string",
    ],
)
def test_secret_names_are_masked(name: str) -> None:
    assert redact({"headers": {name: "shh"}}) == {"headers": {name: "***"}}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("profile", "dmi-minio"),
        ("region_name", "eu-north-1"),
        ("endpoint_url", "https://s3.dmi.dk"),
        ("project", "my-gcp-project"),
        ("client_email", "runner@my-gcp-project.iam.gserviceaccount.com"),
        ("anon", True),
        ("requester_pays", True),
        ("version_aware", False),
        ("default_block_size", 5242880),
        ("signature_version", "s3v4"),
        ("tcp_keepalive", True),
        ("client_kwargs", {}),
        ("s3_additional_kwargs", {"ACL": "private"}),
    ],
)
def test_realistic_non_secret_options_stay_visible(name: str, value: object) -> None:
    assert redact({name: value}) == {name: value}


def test_url_passwords_are_masked_but_user_and_host_kept() -> None:
    options = {
        "client_kwargs": {"endpoint_url": "https://minio-user:url-shh@s3.dmi.dk:9000/"},
        "config_kwargs": {"proxies": {"https": "http://proxy-user:proxy-shh@proxy.dmi.dk:3128"}},
    }
    redacted = redact(options)
    assert redacted["client_kwargs"]["endpoint_url"] == "https://minio-user:***@s3.dmi.dk:9000/"
    assert redacted["config_kwargs"]["proxies"]["https"] == "http://proxy-user:***@proxy.dmi.dk:3128"


@pytest.mark.parametrize(
    "value",
    [
        "https://s3.dmi.dk",
        "https://only-a-user@s3.dmi.dk",
        "eu-north-1",
        "not a url: at all",
        "http://[::1",  # urlsplit raises on this; redaction must not
    ],
)
def test_strings_without_a_url_password_are_unchanged(value: str) -> None:
    assert redact({"endpoint_url": value}) == {"endpoint_url": value}


def test_redaction_does_not_modify_the_options_passed_to_fsspec() -> None:
    options = {
        "secret": "top-shh",
        "client_kwargs": {"aws_secret_access_key": "nested-shh", "endpoint_url": "https://s3"},
        "items": [{"token": "list-shh"}],
    }
    snapshot = copy.deepcopy(options)

    redacted = redact(options)

    assert options == snapshot
    assert redacted["client_kwargs"] is not options["client_kwargs"]
    assert redacted["items"][0] is not options["items"][0]


def test_storage_options_returns_the_real_secrets() -> None:
    """Only the log is masked; s3fs still gets the credentials it was given."""
    options = opts(S3, storage_options='{"client_kwargs": {"aws_secret_access_key": "nested-shh"}}')
    assert options["client_kwargs"]["aws_secret_access_key"] == "nested-shh"


def test_credentials_are_redacted_from_the_log(log_lines: list[str]) -> None:
    log_storage_options("input", opts(S3, storage_options='{"key": "AKIAsecret", "secret": "shhh"}'))

    combined = "".join(log_lines)
    assert "AKIAsecret" not in combined
    assert "shhh" not in combined
    assert "***" in combined


def test_anonymous_access_is_logged_with_what_to_set(log_lines: list[str]) -> None:
    log_storage_options("output", opts(OTHER_S3, "output"))
    assert any("output store is using anonymous access; set MLCAST_OUTPUT_AWS_PROFILE" in line for line in log_lines)


class _OptionsMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
    """fsspec's in-memory filesystem under a protocol of its own.

    zarr >= 3.2 handles ``memory://`` itself and rejects storage options for it, so a
    test that needs options to reach the filesystem cannot use ``memory://``. Under
    any other scheme zarr hands the URL, options included, to fsspec.
    """

    protocol = ("optsmem",)

    @classmethod
    def _strip_protocol(cls, path):
        """Map ``optsmem://`` onto ``memory://`` before the parent strips it."""
        if isinstance(path, str) and path.startswith("optsmem://"):
            path = "memory://" + path[len("optsmem://") :]
        return super()._strip_protocol(path)


def test_nested_destination_secret_stays_out_of_the_log(
    log_lines: list[str], make_dataset: Callable[..., xr.Dataset]
) -> None:
    """Build the options and write with them, then read the log a pipeline would see."""
    # Given an output on a filesystem that accepts (and ignores) arbitrary options
    fsspec.register_implementation("optsmem", _OptionsMemoryFileSystem, clobber=True)
    storage = {
        "client_kwargs": {"region_name": "eu-north-1", "aws_secret_access_key": "nested-shh"},
        "config_kwargs": {"proxies": {"https": "http://proxy-user:proxy-shh@proxy.dmi.dk"}},
        "session_kwargs": [{"private_key": "pem-shh", "client_email": "sa@p.iam"}],
    }
    uri = "optsmem://redacted-log.zarr"

    # When the options are built, logged and used for a real write
    options = opts(uri, "output", storage_options=json.dumps(storage))
    log_storage_options("output", options)
    write_zarr(make_dataset(), uri, options, zarr_format=3)

    # Then the secrets never reach the log, but everything identifying does
    combined = "\n".join(log_lines)
    line = next(line for line in log_lines if "output storage options" in line)
    for secret in ("nested-shh", "proxy-shh", "pem-shh"):
        assert secret not in combined
    assert "eu-north-1" in line
    assert "proxy-user:***@proxy.dmi.dk" in line
    assert "sa@p.iam" in line


# --- the options have to be ones s3fs actually understands --------------------------


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"aws_profile": "some-profile"},
        {"s3_endpoint_url": "https://s3.dmi.dk"},
        {"s3_anon": True},
        {"s3_anon": False},
        {"storage_options": '{"requester_pays": true}'},
    ],
)
def test_options_construct_a_real_s3_filesystem(settings: dict[str, Any]) -> None:
    """Guards against inventing an option name s3fs would silently ignore or reject.

    No network: constructing the filesystem only resolves configuration.
    """
    s3fs = pytest.importorskip("s3fs")
    fs = s3fs.S3FileSystem(**opts(S3, **settings), skip_instance_cache=True)
    assert "s3" in fs.protocol


def test_endpoint_option_reaches_the_filesystem() -> None:
    s3fs = pytest.importorskip("s3fs")
    fs = s3fs.S3FileSystem(**opts(S3, s3_endpoint_url="https://s3.dmi.dk"), skip_instance_cache=True)
    assert fs.client_kwargs["endpoint_url"] == "https://s3.dmi.dk"


def test_anon_default_reaches_the_filesystem() -> None:
    s3fs = pytest.importorskip("s3fs")
    fs = s3fs.S3FileSystem(**opts(S3), skip_instance_cache=True)
    assert fs.anon is True
