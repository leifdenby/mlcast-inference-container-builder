"""Per-side storage options built from the environment, and their masking in the log."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator

import fsspec
import fsspec.implementations.memory
import pytest
import xarray as xr
from loguru import logger

from mlcast_icb.errors import ConfigError
from mlcast_icb.io.storage import redact, storage_options, storage_variables
from mlcast_icb.io.zarr_io import write_zarr

S3 = "s3://bucket/in.zarr"
OTHER_S3 = "s3://other-bucket/out.zarr"


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    """Collect every message logged at INFO or above, formatted as a sink sees it."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    yield messages
    logger.remove(sink)


def test_local_paths_get_no_options() -> None:
    assert storage_options({}, "/data/in.zarr", "SRC") == {}


def test_profile_applies_to_s3() -> None:
    options = storage_options({"SRC_AWS_PROFILE": "reader"}, S3, "SRC")
    assert options["profile"] == "reader"


def test_source_and_destination_are_independent() -> None:
    env = {"SRC_AWS_PROFILE": "era5-reader", "DST_AWS_PROFILE": "dmi-minio"}
    assert storage_options(env, S3, "SRC")["profile"] == "era5-reader"
    assert storage_options(env, OTHER_S3, "DST")["profile"] == "dmi-minio"


def test_unprefixed_profile_applies_to_both_sides() -> None:
    env = {"AWS_PROFILE": "shared"}
    assert storage_options(env, S3, "SRC")["profile"] == "shared"
    assert storage_options(env, S3, "DST")["profile"] == "shared"


def test_side_specific_wins_over_unprefixed() -> None:
    env = {"AWS_PROFILE": "shared", "DST_AWS_PROFILE": "writer"}
    assert storage_options(env, S3, "SRC")["profile"] == "shared"
    assert storage_options(env, S3, "DST")["profile"] == "writer"


def test_no_endpoint_option_when_the_profile_carries_it() -> None:
    """The AWS config is the normal home for endpoint_url, so we pass nothing."""
    options = storage_options({"SRC_AWS_PROFILE": "dmi-minio"}, S3, "SRC")
    assert "client_kwargs" not in options


def test_endpoint_override_is_available() -> None:
    options = storage_options({"DST_S3_ENDPOINT_URL": "https://s3.dmi.dk"}, OTHER_S3, "DST")
    assert options["client_kwargs"] == {"endpoint_url": "https://s3.dmi.dk"}


def test_aws_endpoint_url_is_accepted_as_a_fallback() -> None:
    options = storage_options({"AWS_ENDPOINT_URL": "https://s3.dmi.dk"}, S3, "SRC")
    assert options["client_kwargs"] == {"endpoint_url": "https://s3.dmi.dk"}


def test_storage_variables_cover_both_sides_and_the_shared_names() -> None:
    names = storage_variables()
    assert {"SRC_AWS_PROFILE", "DST_STORAGE_OPTIONS", "AWS_PROFILE", "AWS_ROLE_ARN"} <= names
    assert not any(name.startswith("SRC_") for name in storage_variables(("DST",)))


# --- anonymous by default ------------------------------------------------------------


def test_anon_by_default_when_nothing_is_configured() -> None:
    assert storage_options({}, S3, "SRC")["anon"] is True


def test_a_profile_turns_signing_on() -> None:
    assert storage_options({"SRC_AWS_PROFILE": "reader"}, S3, "SRC")["anon"] is False


def test_ambient_aws_keys_turn_signing_on() -> None:
    env = {"AWS_ACCESS_KEY_ID": "AKIA...", "AWS_SECRET_ACCESS_KEY": "..."}
    assert storage_options(env, S3, "SRC")["anon"] is False


def test_side_specific_keys_turn_signing_on_and_are_passed() -> None:
    env = {"SRC_AWS_ACCESS_KEY_ID": "AKIA...", "SRC_AWS_SECRET_ACCESS_KEY": "shh"}
    options = storage_options(env, S3, "SRC")
    assert options["anon"] is False
    assert options["key"] == "AKIA..."
    assert options["secret"] == "shh"


def test_side_specific_keys_do_not_leak_to_the_other_side() -> None:
    env = {"SRC_AWS_ACCESS_KEY_ID": "AKIA...", "SRC_AWS_SECRET_ACCESS_KEY": "shh"}
    assert "key" not in storage_options(env, OTHER_S3, "DST")


@pytest.mark.parametrize(
    "var",
    ["AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_SESSION_TOKEN"],
)
def test_ambient_role_credentials_turn_signing_on(var: str) -> None:
    """IAM roles on ECS/EKS/EC2 are credentials, even with no keys in the environment."""
    assert storage_options({var: "something"}, S3, "SRC")["anon"] is False


def test_credentials_in_the_json_escape_hatch_turn_signing_on() -> None:
    env = {"SRC_STORAGE_OPTIONS": '{"key": "AKIA...", "secret": "shh"}'}
    assert storage_options(env, S3, "SRC")["anon"] is False


def test_s3fs_credential_aliases_turn_signing_on() -> None:
    env = {"SRC_STORAGE_OPTIONS": '{"username": "AKIA...", "password": "shh"}'}
    assert storage_options(env, S3, "SRC")["anon"] is False


@pytest.mark.parametrize("name", ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"])
def test_credentials_inside_client_kwargs_turn_signing_on(name: str) -> None:
    env = {"SRC_STORAGE_OPTIONS": json.dumps({"client_kwargs": {name: "shh"}})}
    assert storage_options(env, S3, "SRC")["anon"] is False


def test_client_kwargs_without_credentials_stay_anonymous() -> None:
    env = {"SRC_STORAGE_OPTIONS": '{"client_kwargs": {"region_name": "eu-north-1"}}'}
    assert storage_options(env, S3, "SRC")["anon"] is True


def test_explicit_anon_wins_over_credentials() -> None:
    env = {"SRC_AWS_PROFILE": "reader", "SRC_S3_ANON": "true"}
    assert storage_options(env, S3, "SRC")["anon"] is True


def test_explicit_anon_false_wins_over_the_default() -> None:
    assert storage_options({"SRC_S3_ANON": "false"}, S3, "SRC")["anon"] is False


def test_json_can_override_anon() -> None:
    env = {"SRC_STORAGE_OPTIONS": '{"anon": false}'}
    assert storage_options(env, S3, "SRC")["anon"] is False


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("1", True), ("false", False), ("no", False)])
def test_anon_spellings(value: str, expected: bool) -> None:
    assert storage_options({"SRC_S3_ANON": value}, S3, "SRC")["anon"] is expected


def test_bad_anon_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a boolean"):
        storage_options({"SRC_S3_ANON": "maybe"}, S3, "SRC")


def test_anon_is_not_set_for_non_s3_schemes() -> None:
    assert "anon" not in storage_options({}, "gs://bucket/in.zarr", "SRC")
    assert storage_options({}, "/data/in.zarr", "SRC") == {}


# --- the JSON escape hatch -----------------------------------------------------------


def test_s3_options_are_ignored_for_other_schemes() -> None:
    env = {"SRC_AWS_PROFILE": "reader"}
    assert storage_options(env, "gs://bucket/in.zarr", "SRC") == {}
    assert storage_options(env, "/data/in.zarr", "SRC") == {}


def test_json_reaches_any_option() -> None:
    env = {"SRC_STORAGE_OPTIONS": '{"requester_pays": true, "config_kwargs": {"max_pool": 4}}'}
    options = storage_options(env, S3, "SRC")
    assert options["requester_pays"] is True
    assert options["config_kwargs"] == {"max_pool": 4}


def test_json_applies_to_non_s3_schemes_too() -> None:
    env = {"DST_STORAGE_OPTIONS": '{"project": "my-gcp-project"}'}
    assert storage_options(env, "gs://bucket/out.zarr", "DST") == {"project": "my-gcp-project"}


def test_json_overrides_the_dedicated_variables() -> None:
    env = {"SRC_AWS_PROFILE": "reader", "SRC_STORAGE_OPTIONS": '{"profile": "override"}'}
    assert storage_options(env, S3, "SRC")["profile"] == "override"


@pytest.mark.parametrize(("value", "message"), [("not json", "must be valid JSON"), ('["a"]', "must be a JSON object")])
def test_malformed_json_is_rejected(value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        storage_options({"SRC_STORAGE_OPTIONS": value}, S3, "SRC")


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
    env = {"SRC_STORAGE_OPTIONS": '{"client_kwargs": {"aws_secret_access_key": "nested-shh"}}'}
    options = storage_options(env, S3, "SRC")
    assert options["client_kwargs"]["aws_secret_access_key"] == "nested-shh"


def test_credentials_are_redacted_from_the_log(log_lines: list[str]) -> None:
    env = {"SRC_STORAGE_OPTIONS": '{"key": "AKIAsecret", "secret": "shhh"}'}
    storage_options(env, S3, "SRC")

    combined = "".join(log_lines)
    assert "AKIAsecret" not in combined
    assert "shhh" not in combined
    assert "***" in combined


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
    env = {"DST_STORAGE_OPTIONS": json.dumps(storage)}

    # When the options are built and used for a real write
    write_zarr(make_dataset(), uri, storage_options(env, uri, "DST"), zarr_format=3)

    # Then the secrets never reach the log, but everything identifying does
    combined = "\n".join(log_lines)
    line = next(line for line in log_lines if "DST storage options" in line)
    for secret in ("nested-shh", "proxy-shh", "pem-shh"):
        assert secret not in combined
    assert "eu-north-1" in line
    assert "proxy-user:***@proxy.dmi.dk" in line
    assert "sa@p.iam" in line


# --- the options have to be ones s3fs actually understands --------------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"SRC_AWS_PROFILE": "some-profile"},
        {"SRC_S3_ENDPOINT_URL": "https://s3.dmi.dk"},
        {"SRC_AWS_ACCESS_KEY_ID": "AKIA...", "SRC_AWS_SECRET_ACCESS_KEY": "shh"},
        {"SRC_S3_ANON": "true"},
        {"SRC_STORAGE_OPTIONS": '{"requester_pays": true}'},
    ],
)
def test_options_construct_a_real_s3_filesystem(env: dict[str, str]) -> None:
    """Guards against inventing an option name s3fs would silently ignore or reject.

    No network: constructing the filesystem only resolves configuration.
    """
    s3fs = pytest.importorskip("s3fs")
    fs = s3fs.S3FileSystem(**storage_options(env, S3, "SRC"), skip_instance_cache=True)
    assert "s3" in fs.protocol


def test_endpoint_override_reaches_the_filesystem() -> None:
    s3fs = pytest.importorskip("s3fs")
    options = storage_options({"SRC_S3_ENDPOINT_URL": "https://s3.dmi.dk"}, S3, "SRC")
    fs = s3fs.S3FileSystem(**options, skip_instance_cache=True)
    assert fs.client_kwargs["endpoint_url"] == "https://s3.dmi.dk"


def test_anon_default_reaches_the_filesystem() -> None:
    s3fs = pytest.importorskip("s3fs")
    fs = s3fs.S3FileSystem(**storage_options({}, S3, "SRC"), skip_instance_cache=True)
    assert fs.anon is True
