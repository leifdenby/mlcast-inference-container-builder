"""Mode 1, ``run``: environment configuration, the startup log, exit codes, and zarr end to end."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr
from loguru import logger
from run_testdata import START, STEP, fake_forecast_output, write_input

import mlcast_icb.run as run_module
from mlcast_icb import __version__
from mlcast_icb.bundle import create_bundle, load_bundle
from mlcast_icb.envvars import warn_unknown_env_vars
from mlcast_icb.errors import ConfigError, InputError
from mlcast_icb.run import KNOWN_VARIABLES, RUN_VARIABLES, RunConfig, describe_config, main, select_inputs

MINIMAL = {"INPUT_ZARR": "/in.zarr", "OUTPUT_ZARR": "/out.zarr"}


@pytest.fixture(scope="module")
def bundle_dir(tiny_trained_model, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return create_bundle(
        tiny_trained_model.checkpoint_path,
        tmp_path_factory.mktemp("run_bundle") / "model",
        config_path=tiny_trained_model.config_path,
        timestep="PT5M",
        name="tiny",
        model_version="0.1",
    )


@pytest.fixture
def base_env(bundle_dir: Path, tmp_path: Path) -> dict[str, str]:
    return {
        "INPUT_ZARR": str(write_input(tmp_path / "in.zarr")),
        "OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "MODEL_DIR": str(bundle_dir),
        "DEVICE": "cpu",
    }


@pytest.fixture
def fake_forecast(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the forecast with one returning its output schema; record every call."""
    calls: list[dict[str, Any]] = []

    def _forecast(model, spec, ds, *, forecast_steps=None, ensemble_size=None, seed=None, reference_time=None):
        calls.append(
            {
                "model": model,
                "spec": spec,
                "ds": ds,
                "forecast_steps": forecast_steps,
                "ensemble_size": ensemble_size,
                "seed": seed,
                "reference_time": reference_time,
            }
        )
        return fake_forecast_output(ds, forecast_steps, ensemble_size, reference_time)

    monkeypatch.setattr(run_module, "forecast", _forecast)
    return calls


def invoke(env: dict[str, str]) -> int:
    return main(env)


# --- configuration -------------------------------------------------------------------


def test_minimal_config_uses_defaults() -> None:
    config = RunConfig.from_env(MINIMAL)
    assert config.model_dir == "/model"
    assert config.forecast_reference_time is None
    assert config.forecast_steps is None
    assert config.ensemble_size is None
    assert config.seed is None
    assert config.device == "auto"
    assert config.zarr_format == "auto"
    assert config.log_level == "INFO"


def test_values_are_parsed() -> None:
    config = RunConfig.from_env(
        MINIMAL
        | {
            "MODEL_DIR": "/opt/bundle",
            "FORECAST_REFERENCE_TIME": "2026-05-01T14:00:00+02:00",
            "FORECAST_STEPS": "6",
            "ENSEMBLE_SIZE": "3",
            "SEED": "42",
            "DEVICE": "CPU",
            "ZARR_FORMAT": "2",
            "LOG_LEVEL": "debug",
        }
    )
    assert config.model_dir == "/opt/bundle"
    assert config.forecast_reference_time == np.datetime64("2026-05-01T12:00:00", "ns")  # converted to UTC
    assert (config.forecast_steps, config.ensemble_size, config.seed) == (6, 3, 42)
    assert config.device == "cpu"
    assert config.zarr_format == 2
    assert config.log_level == "DEBUG"


def test_naive_reference_time_is_taken_as_utc() -> None:
    config = RunConfig.from_env(MINIMAL | {"FORECAST_REFERENCE_TIME": "2026-05-01T12:00"})
    assert config.forecast_reference_time == np.datetime64("2026-05-01T12:00", "ns")


def test_missing_required_variables_are_both_reported() -> None:
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_env({})
    message = str(excinfo.value)
    assert "2 problems" in message
    assert "INPUT_ZARR is required" in message
    assert "OUTPUT_ZARR is required" in message


def test_every_problem_is_collected() -> None:
    env = MINIMAL | {
        "FORECAST_REFERENCE_TIME": "yesterday-ish",
        "FORECAST_STEPS": "0",
        "ENSEMBLE_SIZE": "many",
        "SEED": "-1",
        "DEVICE": "tpu",
        "ZARR_FORMAT": "4",
        "DST_STORAGE_OPTIONS": "[1]",
    }
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_env(env | {"OUTPUT_ZARR": "s3://bucket/out.zarr"})
    message = str(excinfo.value)
    assert "7 problems" in message
    for name in ("FORECAST_REFERENCE_TIME", "FORECAST_STEPS", "ENSEMBLE_SIZE", "SEED", "DEVICE", "ZARR_FORMAT"):
        assert f"- {name}" in message, name
    assert "DST_STORAGE_OPTIONS" in message


def test_a_lone_problem_is_reported_bare() -> None:
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_env(MINIMAL | {"FORECAST_STEPS": "x"})
    assert str(excinfo.value) == "FORECAST_STEPS must be an integer, got 'x'"


def test_input_and_output_must_differ() -> None:
    with pytest.raises(ConfigError, match="same store"):
        RunConfig.from_env({"INPUT_ZARR": "/a.zarr", "OUTPUT_ZARR": "/a.zarr"})


# --- typo warnings -------------------------------------------------------------------


def test_every_run_variable_is_known() -> None:
    assert set(RUN_VARIABLES) <= KNOWN_VARIABLES
    assert {"SRC_AWS_PROFILE", "DST_STORAGE_OPTIONS", "AWS_PROFILE"} <= KNOWN_VARIABLES


@pytest.mark.parametrize(
    ("typo", "meant"),
    [("FORECAST_STEP", "FORECAST_STEPS"), ("ENSEMBLE_SIZ", "ENSEMBLE_SIZE"), ("model_dir", "MODEL_DIR")],
)
def test_misspelt_variables_are_warned_about(typo: str, meant: str) -> None:
    warnings = warn_unknown_env_vars(KNOWN_VARIABLES, {typo: "1", "HOSTNAME": "pod"})
    assert list(warnings) == [typo]
    assert f"Did you mean {meant}?" in warnings[typo]


def test_typo_warning_appears_in_the_run_log(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoke({"FORECAST_STEP": "3"}) == 2
    assert "Did you mean FORECAST_STEPS?" in capsys.readouterr().err


# --- startup log ---------------------------------------------------------------------


def test_every_setting_is_described() -> None:
    assert list(describe_config(RunConfig.from_env(MINIMAL))) == list(RUN_VARIABLES)


def test_defaults_are_rendered() -> None:
    settings = describe_config(RunConfig.from_env(MINIMAL))
    assert settings["MODEL_DIR"] == "/model"
    assert settings["FORECAST_REFERENCE_TIME"] == "unset (last time in the input)"
    assert settings["FORECAST_STEPS"] == "unset (model default)"
    assert settings["ENSEMBLE_SIZE"] == "unset (model default)"
    assert settings["SEED"] == "unset (not seeded)"
    assert settings["DEVICE"] == "auto"
    assert settings["ZARR_FORMAT"] == "auto"
    assert settings["LOG_LEVEL"] == "INFO"


def test_reference_time_is_rendered_in_iso_8601() -> None:
    settings = describe_config(RunConfig.from_env(MINIMAL | {"FORECAST_REFERENCE_TIME": "2026-05-01T12:05Z"}))
    assert settings["FORECAST_REFERENCE_TIME"] == "2026-05-01T12:05:00"


def test_no_secrets_are_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    env = {
        "INPUT_ZARR": "s3://bucket/in.zarr",
        "OUTPUT_ZARR": "s3://bucket/out.zarr",
        "SRC_AWS_ACCESS_KEY_ID": "AKIAsrcKey",
        "SRC_AWS_SECRET_ACCESS_KEY": "srcSecret",
        "DST_AWS_SESSION_TOKEN": "dstToken",
        "DST_STORAGE_OPTIONS": '{"secret": "jsonSecret"}',
        "MODEL_DIR": "/nowhere",
        "DEVICE": "cpu",
    }
    rendered = " ".join(f"{k}={v}" for k, v in describe_config(RunConfig.from_env(env)).items())
    assert invoke(env) == 6
    logged = capsys.readouterr().err
    for secret in ("AKIAsrcKey", "srcSecret", "dstToken", "jsonSecret"):
        assert secret not in rendered
        assert secret not in logged


def test_config_error_still_logs_the_version_first(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoke({"INPUT_ZARR": "/in.zarr"}) == 2
    lines = capsys.readouterr().err.splitlines()
    assert f"Starting mlcast-inference-container-builder {__version__} (run)" in lines[0]
    assert "ERROR" in lines[-1]
    assert "OUTPUT_ZARR is required" in lines[-1]
    assert not any("Effective configuration" in line for line in lines)


def test_successful_run_logs_version_configuration_and_bundle(
    base_env: dict[str, str], fake_forecast: list, capsys: pytest.CaptureFixture[str]
) -> None:
    assert invoke(base_env | {"SEED": "7"}) == 0
    lines = capsys.readouterr().err.splitlines()
    assert f"Starting mlcast-inference-container-builder {__version__} (run)" in lines[0]
    stderr = "\n".join(lines)
    expected = {
        "INPUT_ZARR": base_env["INPUT_ZARR"],
        "MODEL_DIR": base_env["MODEL_DIR"],
        "FORECAST_STEPS": "unset (model default)",
        "SEED": "7",
        "DEVICE": "cpu",
        "name": "tiny",
        "model_version": "0.1",
        "standard_names": "rainfall_rate",
        "timestep": "PT5M",
    }
    for name, value in expected.items():
        assert re.search(rf"\b{name} += {re.escape(value)}$", stderr, re.MULTILINE), name
    assert re.search(r"model_sha256 += [0-9a-f]{64}$", stderr, re.MULTILINE)
    # The configuration is logged before the bundle is loaded, and both before the input is opened.
    assert stderr.index("Effective configuration") < stderr.index("Model bundle") < stderr.index("Opening input")
    assert "Forecast took" in stderr
    assert "Done in" in stderr


def test_one_log_record_per_setting(
    base_env: dict[str, str], fake_forecast: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Separate records keep their timestamp and level once a collector splits lines."""
    monkeypatch.setattr(run_module, "configure_logging", lambda level: None)  # keep the test's sink
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    try:
        assert invoke(base_env) == 0
    finally:
        logger.remove(sink)
    assert not any("\n" in message for message in messages)


# --- input selection -----------------------------------------------------------------


def test_select_inputs_keeps_only_what_the_model_reads(bundle_dir: Path) -> None:
    from run_testdata import build_input

    _, spec, _ = load_bundle(bundle_dir)
    ds = build_input(nt=6)
    selected = select_inputs(ds, spec, None)
    assert set(selected.data_vars) == {"rr", "crs"}
    assert list(selected.time.values) == list(ds.time.values[-spec.input_steps :])

    reference = ds.time.values[3]
    selected = select_inputs(ds, spec, reference)
    assert selected.time.values[-1] == reference
    assert selected.sizes["time"] == spec.input_steps

    with pytest.raises(InputError, match="not one of the input's times"):
        select_inputs(ds, spec, np.datetime64("2000-01-01T00:00", "ns"))


# --- end to end with the fake forecast -----------------------------------------------


def test_run_writes_the_forecast(base_env: dict[str, str], fake_forecast: list) -> None:
    env = base_env | {"FORECAST_STEPS": "4", "ENSEMBLE_SIZE": "3", "SEED": "1"}
    assert invoke(env) == 0

    (call,) = fake_forecast
    spec = call["spec"]
    assert (call["forecast_steps"], call["ensemble_size"], call["seed"]) == (4, 3, 1)
    assert call["reference_time"] is None
    # Only the model's variable (and its grid mapping) and only its input window were read.
    assert set(call["ds"].data_vars) == {"rr", "crs"}
    assert call["ds"].sizes["time"] == spec.input_steps
    assert call["ds"].rr.variable._in_memory

    out = xr.open_zarr(env["OUTPUT_ZARR"], decode_timedelta=True)
    assert out.rr.dims == ("ensemble_member", "time", "y", "x")
    assert out.sizes["ensemble_member"] == 3
    assert out.sizes["time"] == 4
    assert out.attrs["model_name"] == "tiny"
    assert out.attrs["model_version"] == "0.1"
    assert len(out.attrs["model_sha256"]) == 64
    assert out.attrs["source"] == f"mlcast-inference-container-builder {__version__}"
    assert out.rr.encoding["chunks"] == (1, 1, 8, 8)


def test_model_defaults_apply_when_unset(base_env: dict[str, str], fake_forecast: list) -> None:
    assert invoke(base_env) == 0
    (call,) = fake_forecast
    assert call["forecast_steps"] == call["spec"].forecast_steps
    assert call["ensemble_size"] == call["spec"].default_ensemble_size
    assert call["seed"] is None


def test_reference_time_selects_the_input_window(base_env: dict[str, str], fake_forecast: list) -> None:
    reference = np.datetime64(START) + 3 * STEP
    env = base_env | {"FORECAST_REFERENCE_TIME": str(reference) + "Z"}
    assert invoke(env) == 0
    (call,) = fake_forecast
    assert call["reference_time"] == reference
    assert call["ds"].time.values[-1] == reference


@pytest.mark.parametrize(("zarr_format", "expected"), [("auto", 2), ("3", 3)])
def test_zarr_format(base_env: dict[str, str], fake_forecast: list, tmp_path: Path, zarr_format, expected) -> None:
    env = base_env | {"INPUT_ZARR": str(write_input(tmp_path / "in2.zarr", zarr_format=2)), "ZARR_FORMAT": zarr_format}
    assert invoke(env) == 0
    out = Path(env["OUTPUT_ZARR"])
    assert (out / ("zarr.json" if expected == 3 else ".zgroup")).exists()


# --- exit codes ----------------------------------------------------------------------


def test_missing_bundle_exits_6(base_env: dict[str, str], tmp_path: Path) -> None:
    assert invoke(base_env | {"MODEL_DIR": str(tmp_path / "nothing")}) == 6


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_cuda_without_a_gpu_exits_4(base_env: dict[str, str]) -> None:
    assert invoke(base_env | {"DEVICE": "cuda"}) == 4


def test_missing_input_exits_3(base_env: dict[str, str], tmp_path: Path) -> None:
    assert invoke(base_env | {"INPUT_ZARR": str(tmp_path / "missing.zarr")}) == 3


def test_reference_time_not_in_input_exits_3(base_env: dict[str, str], fake_forecast: list) -> None:
    assert invoke(base_env | {"FORECAST_REFERENCE_TIME": "1999-01-01T00:00Z"}) == 3
    assert fake_forecast == []


def test_forecast_input_error_exits_3(
    base_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mlcast_icb.upstream.forecast import ForecastInputError

    def _forecast(*args, **kwargs):
        raise ForecastInputError(["no variable has standard_name 'rainfall_rate'", "time spacing is not PT5M"])

    monkeypatch.setattr(run_module, "forecast", _forecast)
    assert invoke(base_env) == 3
    stderr = capsys.readouterr().err
    assert "does not match what the model expects" in stderr
    assert "no variable has standard_name 'rainfall_rate'" in stderr
    assert "time spacing is not PT5M" in stderr


def test_unwritable_output_exits_5(base_env: dict[str, str], fake_forecast: list, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert invoke(base_env | {"OUTPUT_ZARR": str(blocker / "out.zarr")}) == 5


def test_unexpected_failure_exits_1(
    base_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _forecast(*args, **kwargs):
        raise ZeroDivisionError("bug")

    monkeypatch.setattr(run_module, "forecast", _forecast)
    assert invoke(base_env) == 1
    stderr = capsys.readouterr().err
    assert "Unexpected failure" in stderr
    assert "ZeroDivisionError" in stderr


# --- end to end with the real forecast -----------------------------------------------


def test_real_forecast_end_to_end(base_env: dict[str, str], tiny_trained_model) -> None:
    """The tiny model forecasts a local zarr store through the real upstream forecast()."""
    try:
        import mlcast_icb.upstream.forecast  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"mlcast_icb.upstream.forecast is not available: {exc}")

    env = base_env | {"ENSEMBLE_SIZE": "2", "SEED": "3", "FORECAST_STEPS": "3"}
    assert invoke(env) == 0

    source = xr.open_zarr(env["INPUT_ZARR"])
    out = xr.open_zarr(env["OUTPUT_ZARR"], decode_timedelta=True)
    assert out.rr.dims == ("ensemble_member", "time", "y", "x")
    assert out.sizes["ensemble_member"] == 2
    assert out.sizes["time"] == 3
    assert out.rr.dtype == np.float32
    assert out.forecast_reference_time.values == source.time.values[-1]
    assert out.time.values[0] == source.time.values[-1] + STEP
    np.testing.assert_array_equal(out.x.values, source.x.values)
    values = out.rr.values
    assert np.isfinite(values).all()
    assert (values >= 0).all()

    # The same seed gives the same ensemble.
    env2 = env | {"OUTPUT_ZARR": env["OUTPUT_ZARR"] + "-again"}
    assert invoke(env2) == 0
    np.testing.assert_array_equal(xr.open_zarr(env2["OUTPUT_ZARR"]).rr.values, values)
