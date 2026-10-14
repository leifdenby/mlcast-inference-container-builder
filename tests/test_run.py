"""Mode 1, ``run``: options and MLCAST_* variables, the startup log, exit codes, and zarr end to end."""

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
from mlcast_icb.errors import ConfigError, InputError
from mlcast_icb.io.storage import StorageSettings
from mlcast_icb.options import run_parser
from mlcast_icb.run import RunConfig, describe_config, main, select_inputs

MINIMAL = {"MLCAST_INPUT_ZARR": "/in.zarr", "MLCAST_OUTPUT_ZARR": "/out.zarr"}


def config(env: dict[str, str], argv: list[str] | None = None) -> RunConfig:
    return RunConfig.from_command_line(argv or [], env)


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
        "MLCAST_INPUT_ZARR": str(write_input(tmp_path / "in.zarr")),
        "MLCAST_OUTPUT_ZARR": str(tmp_path / "out.zarr"),
        "MLCAST_MODEL_DIR": str(bundle_dir),
        "MLCAST_DEVICE": "cpu",
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


def invoke(env: dict[str, str], argv: list[str] | None = None) -> int:
    return main(argv or [], env)


# --- configuration -------------------------------------------------------------------


def test_minimal_config_uses_defaults() -> None:
    parsed = config(MINIMAL)
    assert parsed.model_dir == "/model"
    assert parsed.forecast_reference_time is None
    assert parsed.forecast_steps is None
    assert parsed.ensemble_size is None
    assert parsed.seed is None
    assert parsed.device == "auto"
    assert parsed.zarr_format == "auto"
    assert parsed.log_level == "INFO"
    assert parsed.input_storage == parsed.output_storage == StorageSettings()


VALUES = {
    "model_dir": "/opt/bundle",
    "forecast_reference_time": "2026-05-01T14:00:00+02:00",
    "forecast_steps": "6",
    "ensemble_size": "3",
    "seed": "42",
    "device": "CPU",
    "zarr_format": "2",
    "log_level": "debug",
}


def _check_values(parsed: RunConfig) -> None:
    assert parsed.model_dir == "/opt/bundle"
    assert parsed.forecast_reference_time == np.datetime64("2026-05-01T12:00:00", "ns")  # converted to UTC
    assert (parsed.forecast_steps, parsed.ensemble_size, parsed.seed) == (6, 3, 42)
    assert parsed.device == "cpu"
    assert parsed.zarr_format == 2
    assert parsed.log_level == "DEBUG"


def test_values_are_parsed_from_the_environment() -> None:
    _check_values(config(MINIMAL | {f"MLCAST_{dest.upper()}": value for dest, value in VALUES.items()}))


def test_values_are_parsed_from_the_command_line() -> None:
    argv = ["--input-zarr", "/in.zarr", "--output-zarr=/out.zarr"]
    for dest, value in VALUES.items():
        argv += [f"--{dest.replace('_', '-')}", value]
    _check_values(config({}, argv))


def test_the_command_line_wins_over_the_environment() -> None:
    parsed = config(MINIMAL | {"MLCAST_SEED": "1", "MLCAST_ENSEMBLE_SIZE": "4"}, ["--seed", "2"])
    assert (parsed.seed, parsed.ensemble_size) == (2, 4)


def test_an_empty_variable_counts_as_unset() -> None:
    assert config(MINIMAL | {"MLCAST_SEED": "", "MLCAST_MODEL_DIR": ""}).model_dir == "/model"


def test_storage_options_per_side() -> None:
    env = MINIMAL | {
        "MLCAST_INPUT_AWS_PROFILE": "reader",
        "MLCAST_INPUT_S3_ANON": "false",
        "MLCAST_OUTPUT_S3_ENDPOINT_URL": "https://s3.example.org",
        "MLCAST_OUTPUT_STORAGE_OPTIONS": '{"requester_pays": true}',
    }
    parsed = config(env, ["--no-output-s3-anon", "--output-aws-profile", "writer"])
    assert parsed.input_storage == StorageSettings(aws_profile="reader", s3_anon=False)
    assert parsed.output_storage == StorageSettings(
        aws_profile="writer",
        s3_endpoint_url="https://s3.example.org",
        s3_anon=False,
        storage_options=env["MLCAST_OUTPUT_STORAGE_OPTIONS"],
    )


def test_storage_options_resolve_against_the_store_uri() -> None:
    env = {
        "MLCAST_INPUT_ZARR": "s3://bucket/in.zarr",
        "MLCAST_OUTPUT_ZARR": "s3://bucket/out.zarr",
        "MLCAST_OUTPUT_AWS_PROFILE": "writer",
        "MLCAST_OUTPUT_S3_ENDPOINT_URL": "https://s3.example.org",
    }
    parsed = config(env)
    assert parsed.input_storage_options == {"anon": True}
    assert parsed.output_storage_options == {
        "profile": "writer",
        "client_kwargs": {"endpoint_url": "https://s3.example.org"},
        "anon": False,
    }


def test_a_bad_s3_anon_value_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoke(MINIMAL | {"MLCAST_INPUT_S3_ANON": "maybe"}) == 2
    assert "MLCAST_INPUT_S3_ANON='maybe'" in capsys.readouterr().err


def test_naive_reference_time_is_taken_as_utc() -> None:
    parsed = config(MINIMAL | {"MLCAST_FORECAST_REFERENCE_TIME": "2026-05-01T12:00"})
    assert parsed.forecast_reference_time == np.datetime64("2026-05-01T12:00", "ns")


def test_missing_required_options_are_both_reported() -> None:
    with pytest.raises(ConfigError) as excinfo:
        config({})
    message = str(excinfo.value)
    assert "2 problems" in message
    assert "MLCAST_INPUT_ZARR (--input-zarr) is required" in message
    assert "MLCAST_OUTPUT_ZARR (--output-zarr) is required" in message


def test_every_problem_is_collected() -> None:
    env = MINIMAL | {
        "MLCAST_FORECAST_REFERENCE_TIME": "yesterday-ish",
        "MLCAST_FORECAST_STEPS": "0",
        "MLCAST_ENSEMBLE_SIZE": "many",
        "MLCAST_SEED": "-1",
        "MLCAST_DEVICE": "tpu",
        "MLCAST_OUTPUT_STORAGE_OPTIONS": "[1]",
        "MLCAST_OUTPUT_ZARR": "s3://bucket/out.zarr",
    }
    with pytest.raises(ConfigError) as excinfo:
        config(env, ["--zarr-format", "4"])
    message = str(excinfo.value)
    assert "7 problems" in message
    for name in ("FORECAST_REFERENCE_TIME", "FORECAST_STEPS", "ENSEMBLE_SIZE", "SEED", "DEVICE"):
        assert f"- MLCAST_{name}" in message, name
    assert "- --zarr-format" in message  # named as it was given: on the command line
    assert "MLCAST_OUTPUT_STORAGE_OPTIONS" in message


def test_a_lone_problem_is_reported_bare() -> None:
    with pytest.raises(ConfigError) as excinfo:
        config(MINIMAL | {"MLCAST_FORECAST_STEPS": "x"})
    assert str(excinfo.value) == "MLCAST_FORECAST_STEPS must be an integer, got 'x'"


def test_input_and_output_must_differ() -> None:
    with pytest.raises(ConfigError, match="same store"):
        config({"MLCAST_INPUT_ZARR": "/a.zarr", "MLCAST_OUTPUT_ZARR": "/a.zarr"})


# --- unknown variables ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("typo", "meant"),
    [
        ("MLCAST_FORECAST_STEP", "MLCAST_FORECAST_STEPS"),
        ("MLCAST_ENSEMBLE_SIZ", "MLCAST_ENSEMBLE_SIZE"),
        ("MLCAST_INPUT_AWS_PROFIL", "MLCAST_INPUT_AWS_PROFILE"),
    ],
)
def test_a_misspelt_variable_exits_2_with_a_suggestion(
    typo: str, meant: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert invoke(MINIMAL | {typo: "3", "HOSTNAME": "pod", "FORECAST_STEPS": "x"}) == 2
    err = capsys.readouterr().err
    assert f"{typo} (did you mean {meant}?)" in err
    assert "HOSTNAME" not in err and "FORECAST_STEPS=" not in err  # unprefixed: not ours


def test_old_per_side_variables_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoke(MINIMAL | {"MLCAST_INPUT_AWS_ACCESS_KEY_ID": "AKIA"}) == 2
    assert "MLCAST_INPUT_AWS_ACCESS_KEY_ID" in capsys.readouterr().err


def test_a_serve_variable_is_ignored(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # Given the image's shared environment, with a serve option set and debug logging
    env = MINIMAL | {"MLCAST_PORT": "9000", "MLCAST_MAX_QUEUE": "3", "MLCAST_LOG_LEVEL": "DEBUG"}
    monkeypatch.setattr(run_module, "run", lambda config: None)

    # When run starts, then it runs, and says at DEBUG that it ignored them
    assert invoke(env) == 0
    err = capsys.readouterr().err
    assert "Ignoring MLCAST_MAX_QUEUE: it is an option of `serve`" in err
    assert "Ignoring MLCAST_PORT" in err


# --- startup log ---------------------------------------------------------------------


def test_every_option_is_described() -> None:
    dests = [action.dest for action in run_parser()._actions if action.dest != "help"]
    assert list(describe_config(config(MINIMAL))) == dests


def test_defaults_are_rendered() -> None:
    settings = describe_config(config(MINIMAL))
    assert settings["model_dir"] == "/model"
    assert settings["forecast_reference_time"] == "unset (last time in the input)"
    assert settings["forecast_steps"] == "unset (model default)"
    assert settings["ensemble_size"] == "unset (model default)"
    assert settings["seed"] == "unset (not seeded)"
    assert settings["device"] == "auto"
    assert settings["zarr_format"] == "auto"
    assert settings["log_level"] == "INFO"
    assert settings["input_aws_profile"] == "unset (AWS configuration)"
    assert settings["output_s3_anon"] == "unset (anonymous unless credentials)"


def test_reference_time_is_rendered_in_iso_8601() -> None:
    settings = describe_config(config(MINIMAL | {"MLCAST_FORECAST_REFERENCE_TIME": "2026-05-01T12:05Z"}))
    assert settings["forecast_reference_time"] == "2026-05-01T12:05:00"


def test_no_secrets_are_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    env = {
        "MLCAST_INPUT_ZARR": "s3://bucket/in.zarr",
        "MLCAST_OUTPUT_ZARR": "s3://bucket/out.zarr",
        "AWS_ACCESS_KEY_ID": "AKIAsrcKey",
        "AWS_SECRET_ACCESS_KEY": "srcSecret",
        "MLCAST_INPUT_STORAGE_OPTIONS": '{"key": "AKIAjsonKey", "client_kwargs": {"region_name": "eu-north-1"}}',
        "MLCAST_OUTPUT_STORAGE_OPTIONS": '{"secret": "jsonSecret"}',
        "MLCAST_MODEL_DIR": "/nowhere",
        "MLCAST_DEVICE": "cpu",
    }
    rendered = " ".join(f"{k}={v}" for k, v in describe_config(config(env)).items())
    assert invoke(env) == 6
    logged = capsys.readouterr().err
    for secret in ("AKIAsrcKey", "srcSecret", "AKIAjsonKey", "jsonSecret"):
        assert secret not in rendered
        assert secret not in logged
    # The storage options are logged, masked, with what identifies the store kept.
    assert re.search(r"MLCAST_INPUT_STORAGE_OPTIONS += .*\*\*\*.*eu-north-1.* \(env\)$", logged, re.MULTILINE)


def test_config_error_still_logs_the_version_first(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoke({"MLCAST_INPUT_ZARR": "/in.zarr"}) == 2
    lines = capsys.readouterr().err.splitlines()
    assert f"Starting mlcast-inference-container-builder {__version__} (run)" in lines[0]
    assert "ERROR" in lines[-1]
    assert "MLCAST_OUTPUT_ZARR (--output-zarr) is required" in lines[-1]
    assert not any("Effective configuration" in line for line in lines)


def test_successful_run_logs_version_configuration_and_bundle(
    base_env: dict[str, str], fake_forecast: list, capsys: pytest.CaptureFixture[str]
) -> None:
    assert invoke(base_env | {"MLCAST_SEED": "7"}, ["--ensemble-size", "2"]) == 0
    lines = capsys.readouterr().err.splitlines()
    assert f"Starting mlcast-inference-container-builder {__version__} (run)" in lines[0]
    stderr = "\n".join(lines)
    expected = {
        "MLCAST_INPUT_ZARR": f"{base_env['MLCAST_INPUT_ZARR']} (env)",
        "MLCAST_MODEL_DIR": f"{base_env['MLCAST_MODEL_DIR']} (env)",
        "MLCAST_FORECAST_STEPS": "unset (model default) (default)",
        "MLCAST_ENSEMBLE_SIZE": "2 (cli)",
        "MLCAST_SEED": "7 (env)",
        "MLCAST_DEVICE": "cpu (env)",
        "MLCAST_ZARR_FORMAT": "auto (default)",
        "MLCAST_OUTPUT_STORAGE_OPTIONS": "unset (none) (default)",
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
    env = base_env | {"MLCAST_FORECAST_STEPS": "4", "MLCAST_ENSEMBLE_SIZE": "3", "MLCAST_SEED": "1"}
    assert invoke(env) == 0

    (call,) = fake_forecast
    spec = call["spec"]
    assert (call["forecast_steps"], call["ensemble_size"], call["seed"]) == (4, 3, 1)
    assert call["reference_time"] is None
    # Only the model's variable (and its grid mapping) and only its input window were read.
    assert set(call["ds"].data_vars) == {"rr", "crs"}
    assert call["ds"].sizes["time"] == spec.input_steps
    assert call["ds"].rr.variable._in_memory

    out = xr.open_zarr(env["MLCAST_OUTPUT_ZARR"], decode_timedelta=True)
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
    env = base_env | {"MLCAST_FORECAST_REFERENCE_TIME": str(reference) + "Z"}
    assert invoke(env) == 0
    (call,) = fake_forecast
    assert call["reference_time"] == reference
    assert call["ds"].time.values[-1] == reference


@pytest.mark.parametrize(("zarr_format", "expected"), [("auto", 2), ("3", 3)])
def test_zarr_format(base_env: dict[str, str], fake_forecast: list, tmp_path: Path, zarr_format, expected) -> None:
    env = base_env | {"MLCAST_INPUT_ZARR": str(write_input(tmp_path / "in2.zarr", zarr_format=2))}
    assert invoke(env, ["--zarr-format", zarr_format]) == 0
    out = Path(env["MLCAST_OUTPUT_ZARR"])
    assert (out / ("zarr.json" if expected == 3 else ".zgroup")).exists()


# --- exit codes ----------------------------------------------------------------------


def test_missing_bundle_exits_6(base_env: dict[str, str], tmp_path: Path) -> None:
    assert invoke(base_env, ["--model-dir", str(tmp_path / "nothing")]) == 6


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_cuda_without_a_gpu_exits_4(base_env: dict[str, str]) -> None:
    assert invoke(base_env | {"MLCAST_DEVICE": "cuda"}) == 4


def test_missing_input_exits_3(base_env: dict[str, str], tmp_path: Path) -> None:
    assert invoke(base_env | {"MLCAST_INPUT_ZARR": str(tmp_path / "missing.zarr")}) == 3


def test_reference_time_not_in_input_exits_3(base_env: dict[str, str], fake_forecast: list) -> None:
    assert invoke(base_env | {"MLCAST_FORECAST_REFERENCE_TIME": "1999-01-01T00:00Z"}) == 3
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
    assert invoke(base_env | {"MLCAST_OUTPUT_ZARR": str(blocker / "out.zarr")}) == 5


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

    env = base_env | {"MLCAST_ENSEMBLE_SIZE": "2", "MLCAST_SEED": "3", "MLCAST_FORECAST_STEPS": "3"}
    assert invoke(env) == 0

    source = xr.open_zarr(env["MLCAST_INPUT_ZARR"])
    out = xr.open_zarr(env["MLCAST_OUTPUT_ZARR"], decode_timedelta=True)
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

    # The same seed gives the same ensemble, and the command line works as the variables do.
    again = env["MLCAST_OUTPUT_ZARR"] + "-again"
    argv = ["--output-zarr", again, "--ensemble-size", "2", "--seed", "3", "--forecast-steps", "3"]
    assert invoke({k: v for k, v in base_env.items() if k != "MLCAST_OUTPUT_ZARR"}, argv) == 0
    np.testing.assert_array_equal(xr.open_zarr(again).rr.values, values)
