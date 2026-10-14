"""The command line: ``mlcast-icb {run,serve,bundle,info}`` and their exit codes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlcast_icb import __version__
from mlcast_icb.__main__ import main
from mlcast_icb.bundle import MANIFEST_FILE, MODEL_FILE


@pytest.fixture(scope="module")
def bundle_dir(tiny_trained_model, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "model"
    args = [
        "bundle",
        "--checkpoint",
        str(tiny_trained_model.checkpoint_path),
        "--config",
        str(tiny_trained_model.config_path),
        "--timestep",
        "PT5M",
        "--out",
        str(out),
        "--name",
        "tiny",
        "--model-version",
        "1.2",
    ]
    assert main(args) == 0
    return out


def test_bundle_writes_checkpoint_and_manifest(bundle_dir: Path) -> None:
    assert (bundle_dir / MODEL_FILE).is_file()
    manifest = json.loads((bundle_dir / MANIFEST_FILE).read_text())
    assert (manifest["name"], manifest["model_version"]) == ("tiny", "1.2")
    assert manifest["spec"]["timestep"] == "PT5M"


def test_bundle_refuses_a_non_empty_output(bundle_dir: Path, tiny_trained_model) -> None:
    args = ["bundle", "--checkpoint", str(tiny_trained_model.checkpoint_path), "--out", str(bundle_dir)]
    assert main(args) == 2


def test_bundle_of_a_missing_checkpoint_exits_6(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bundle", "--checkpoint", str(tmp_path / "no.ckpt"), "--out", str(tmp_path / "out")]) == 6
    assert "no.ckpt" in capsys.readouterr().err


def test_bundle_requires_its_options() -> None:
    assert main(["bundle", "--out", "x"]) == 2


def test_info_prints_the_manifest(bundle_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(bundle_dir)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((bundle_dir / MANIFEST_FILE).read_text())


def test_info_reads_model_dir_from_the_environment(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(bundle_dir))
    assert main(["info", "--verify"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["name"] == "tiny"
    assert "verified" in captured.err


def test_info_on_a_missing_bundle_exits_6(tmp_path: Path) -> None:
    assert main(["info", str(tmp_path / "nothing")]) == 6


def test_serve_is_not_implemented_yet(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_no_command_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: mlcast-icb" in capsys.readouterr().err


def test_unknown_command_exits_2() -> None:
    assert main(["frobnicate"]) == 2


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_run_reads_the_environment(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("INPUT_ZARR", "OUTPUT_ZARR"):
        monkeypatch.delenv(name, raising=False)
    assert main(["run"]) == 2
    assert "INPUT_ZARR is required" in capsys.readouterr().err


def test_module_entry_point_returns_the_exit_code() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mlcast_icb", "serve"], capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 2
    assert "not implemented yet" in result.stderr
