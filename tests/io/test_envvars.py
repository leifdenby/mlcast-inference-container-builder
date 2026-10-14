"""The misspelt-environment-variable warning, against a known-names list a mode supplies."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger

from mlcast_icb.envvars import warn_unknown_env_vars
from mlcast_icb.io.storage import storage_options, storage_variables

#: Stands in for what a ``run`` mode would declare: its own variables plus storage.
KNOWN = frozenset(
    {
        "INPUT_ZARR",
        "OUTPUT_ZARR",
        "INPUT_VARIABLES",
        "N_FORECAST_STEPS",
        "N_INPUT_TIMESTEPS",
        "OUTPUT_MODE",
        "MODEL_LAYERS",
        "DEVICE",
        "ZARR_FORMAT",
        "LOG_LEVEL",
    }
    | storage_variables()
)


def warn(env: dict[str, str]) -> dict[str, str]:
    return warn_unknown_env_vars(KNOWN, env)


@pytest.fixture
def logged() -> Iterator[list[tuple[str, str]]]:
    """Collect (level, message) for everything logged at WARNING or above."""
    records: list[tuple[str, str]] = []
    handler = logger.add(
        lambda message: records.append((message.record["level"].name, message.record["message"])),
        level="WARNING",
    )
    yield records
    logger.remove(handler)


# --- misspellings --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typo", "meant"),
    [
        ("N_FORECAST_STEP", "N_FORECAST_STEPS"),
        ("N_FORCAST_STEPS", "N_FORECAST_STEPS"),
        ("N_INPUT_TIMESTEP", "N_INPUT_TIMESTEPS"),
        ("INPUT_VARIBLES", "INPUT_VARIABLES"),
        ("OUTPUTMODE", "OUTPUT_MODE"),
        ("MODEL_LAYER", "MODEL_LAYERS"),
        ("LOGLEVEL", "LOG_LEVEL"),
        ("DEVCIE", "DEVICE"),
        ("AWS_PROFLE", "AWS_PROFILE"),
        ("SRC_AWS_PROFIL", "SRC_AWS_PROFILE"),
        ("DST_S3_ENDPOINT", "DST_S3_ENDPOINT_URL"),
    ],
)
def test_a_misspelling_warns_with_a_suggestion(typo: str, meant: str) -> None:
    warnings = warn({typo: "8"})
    assert list(warnings) == [typo]
    assert f"{typo} has no effect" in warnings[typo]
    assert f"Did you mean {meant}?" in warnings[typo]


def test_the_warning_is_logged_at_warning_level(logged: list[tuple[str, str]]) -> None:
    warn({"N_FORECAST_STEP": "8"})
    assert len(logged) == 1
    level, message = logged[0]
    assert level == "WARNING"
    assert "Did you mean N_FORECAST_STEPS?" in message


def test_each_offending_name_warns_once(logged: list[tuple[str, str]]) -> None:
    warnings = warn({"N_FORECAST_STEP": "8", "INPUT_VARIBLES": "t2m"})
    assert list(warnings) == ["INPUT_VARIBLES", "N_FORECAST_STEP"]
    assert len(logged) == 2


def test_wrong_case_is_reported() -> None:
    warnings = warn({"n_forecast_steps": "8"})
    assert "Did you mean N_FORECAST_STEPS?" in warnings["n_forecast_steps"]
    assert "case-sensitive" in warnings["n_forecast_steps"]


def test_equally_close_names_are_all_suggested() -> None:
    """There is no unprefixed STORAGE_OPTIONS; either side's is an equally good guess."""
    warnings = warn({"STORAGE_OPTIONS": "{}"})
    assert "Did you mean DST_STORAGE_OPTIONS or SRC_STORAGE_OPTIONS?" in warnings["STORAGE_OPTIONS"]


def test_values_are_never_logged(logged: list[tuple[str, str]]) -> None:
    """A misspelled variable may hold a secret, so only its name may appear."""
    warnings = warn({"SRC_AWS_SECRET_ACESS_KEY": "hunter2"})
    assert "Did you mean SRC_AWS_SECRET_ACCESS_KEY?" in warnings["SRC_AWS_SECRET_ACESS_KEY"]
    assert not any("hunter2" in message for _, message in logged)


def test_reads_the_process_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N_FORECAST_STEP", "8")
    assert "N_FORECAST_STEP" in warn_unknown_env_vars(KNOWN)


# --- the known names are the caller's ------------------------------------------------


def test_each_mode_declares_its_own_names() -> None:
    """A variable one mode reads is unknown, and so reported, in a mode that does not."""
    serve_known = {"PORT", "MAX_UPLOAD_BYTES"} | storage_variables(("DST",))
    assert warn_unknown_env_vars(serve_known, {"MAX_UPLOAD_BYTES": "1"}) == {}
    warnings = warn_unknown_env_vars(serve_known, {"MAX_UPLOAD_BYTE": "1"})
    assert "Did you mean MAX_UPLOAD_BYTES?" in warnings["MAX_UPLOAD_BYTE"]
    assert "SRC_AWS_PROFILE" in warn_unknown_env_vars(serve_known, {"SRC_AWS_PROFILE": "x"})


def test_an_empty_known_list_only_flags_store_prefixes() -> None:
    assert warn_unknown_env_vars((), {"HOSTNAME": "x"}) == {}
    assert "SRC_ prefix is reserved" in warn_unknown_env_vars((), {"SRC_THING": "x"})["SRC_THING"]


# --- the store prefixes --------------------------------------------------------------


@pytest.mark.parametrize(("name", "side", "store"), [("SRC_REGION", "SRC", "input"), ("DST_BUCKET", "DST", "output")])
def test_unknown_store_prefixed_names_warn(name: str, side: str, store: str) -> None:
    warnings = warn({name: "x"})
    assert f"{side}_ prefix is reserved for the {store} store" in warnings[name]
    assert f"{side}_STORAGE_OPTIONS" in warnings[name]


# --- what must stay quiet ------------------------------------------------------------


def test_known_variables_do_not_warn() -> None:
    assert warn({name: "x" for name in KNOWN}) == {}


def test_kubernetes_service_links_do_not_warn() -> None:
    """Services named input, src, dst or input-zarr inject these into every pod."""
    env = {
        "INPUT_SERVICE_HOST": "10.0.0.11",
        "INPUT_SERVICE_PORT": "8080",
        "INPUT_SERVICE_PORT_HTTP": "8080",
        "INPUT_PORT": "tcp://10.0.0.11:8080",
        "INPUT_PORT_8080_TCP": "tcp://10.0.0.11:8080",
        "INPUT_PORT_8080_TCP_ADDR": "10.0.0.11",
        "INPUT_PORT_8080_TCP_PORT": "8080",
        "INPUT_PORT_8080_TCP_PROTO": "tcp",
        "SRC_SERVICE_HOST": "10.0.0.12",
        "SRC_PORT": "tcp://10.0.0.12:9000",
        "SRC_PORT_9000_TCP_ADDR": "10.0.0.12",
        "DST_SERVICE_PORT": "9000",
        "DST_PORT_53_UDP": "udp://10.0.0.13:53",
        "DST_PORT_9000_SCTP_PROTO": "sctp",
        # Close enough to INPUT_ZARR to count as a misspelling, were it not a link.
        "INPUT_ZARR_SERVICE_HOST": "10.0.0.14",
        "INPUT_ZARR_PORT": "tcp://10.0.0.14:80",
        "KUBERNETES_SERVICE_HOST": "10.96.0.1",
        "KUBERNETES_PORT_443_TCP_PROTO": "tcp",
    }
    assert warn(env) == {}


@pytest.mark.parametrize(
    "name",
    [
        "HOSTNAME",
        "PATH",
        "HOME",
        "LANG",
        "TERM",
        "TZ",
        "http_proxy",
        "LD_LIBRARY_PATH",
        "PYTHONUNBUFFERED",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VERSION",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SECURITY_TOKEN",
        # Read by boto, and close to a known name: allowed explicitly.
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL_STS",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "POD_NAME",
        "JOB_COMPLETION_INDEX",
        "MODEL_NAME",
        "OUTPUT_DIR",
    ],
)
def test_unrelated_variables_do_not_warn(name: str) -> None:
    assert warn({name: "x"}) == {}


@pytest.mark.parametrize(
    ("near_miss", "meant"),
    [
        ("INPUT_VARS", "INPUT_VARIABLES"),
        ("DST_PROFILE", "DST_AWS_PROFILE"),
        ("S3_ENDPOINT", "S3_ENDPOINT_URL"),
    ],
)
def test_the_cutoff_also_catches_near_misses(near_miss: str, meant: str) -> None:
    """Pin the cutoff from below; test_unrelated_variables_do_not_warn pins it from above."""
    assert meant in warn({near_miss: "x"})[near_miss]


# --- storage_variables must match what storage_options reads --------------------------


class RecordingEnv(dict):
    """A dict that remembers every key looked up in it."""

    def __init__(self, values: dict[str, str]):
        super().__init__(values)
        self.read: set[str] = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        """Record the key, then look it up."""
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        """Record the key, then test for it."""
        self.read.add(key)
        return super().__contains__(key)


# S3 URIs exercise the S3 options and the credential probe; local paths exercise the
# "S3 variables set for a non-S3 URI" check instead. Between them, every read happens.
URIS = ("s3://bucket/store.zarr", "/data/store.zarr")


def variables_read_by_storage_options() -> set[str]:
    read: set[str] = set()
    for uri in URIS:
        for side in ("SRC", "DST"):
            env = RecordingEnv({})
            storage_options(env, uri, side)
            read |= env.read
    return read


def test_every_variable_storage_options_reads_is_declared() -> None:
    undeclared = variables_read_by_storage_options() - storage_variables()
    assert not undeclared, f"add these to storage_variables(): {sorted(undeclared)}"


def test_storage_variables_has_no_stale_entries() -> None:
    stale = storage_variables() - variables_read_by_storage_options()
    assert not stale, f"declared but never read, remove from storage_variables(): {sorted(stale)}"
