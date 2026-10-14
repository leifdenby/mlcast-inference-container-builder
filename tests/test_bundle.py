"""The model bundle: create from a checkpoint, verify, load, and refuse anything broken."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import torch

from mlcast_icb import __version__
from mlcast_icb.bundle import (
    MANIFEST_FILE,
    MODEL_FILE,
    create_bundle,
    load_bundle,
    manifest_summary,
    read_manifest,
    sha256_file,
)
from mlcast_icb.errors import ConfigError, ModelError
from mlcast_icb.upstream.checkpoint import load_model
from mlcast_icb.upstream.convert_convgru_ensemble import convert


@pytest.fixture(scope="module")
def bundle_dir(tiny_trained_model, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return create_bundle(
        tiny_trained_model.checkpoint_path,
        tmp_path_factory.mktemp("bundle") / "model",
        config_path=tiny_trained_model.config_path,
        timestep="PT5M",
        name="tiny",
        model_version="0.1",
    )


@pytest.fixture
def bundle_copy(bundle_dir: Path, tmp_path: Path) -> Path:
    """A private copy of the bundle, for tests that damage it."""
    return Path(shutil.copytree(bundle_dir, tmp_path / "model"))


def test_bundle_holds_exactly_the_checkpoint_and_manifest(bundle_dir: Path) -> None:
    assert sorted(p.name for p in bundle_dir.iterdir()) == [MANIFEST_FILE, MODEL_FILE]


def test_manifest_records_provenance_and_spec(bundle_dir: Path, tiny_trained_model) -> None:
    manifest = read_manifest(bundle_dir)
    assert manifest["format_version"] == 1
    assert manifest["name"] == "tiny"
    assert manifest["model_version"] == "0.1"
    assert manifest["builder_version"] == __version__
    assert manifest["mlcast_version"]
    assert manifest["model_sha256"] == sha256_file(bundle_dir / MODEL_FILE)
    assert manifest["source_checkpoint_sha256"] == sha256_file(tiny_trained_model.checkpoint_path)
    assert manifest["source_config_sha256"] == sha256_file(tiny_trained_model.config_path)
    assert datetime.fromisoformat(manifest["created"]).tzinfo is not None

    spec = manifest["spec"]
    assert spec["standard_names"] == ["rainfall_rate"]
    assert spec["input_steps"] == tiny_trained_model.input_steps
    assert spec["forecast_steps"] == tiny_trained_model.forecast_steps
    assert spec["timestep"] == "PT5M"


def test_bundle_checkpoint_is_stripped_and_self_describing(bundle_dir: Path) -> None:
    checkpoint = torch.load(bundle_dir / MODEL_FILE, weights_only=False)
    assert set(checkpoint) == {"state_dict", "hyper_parameters", "pytorch-lightning_version", "mlcast_inference"}


def test_load_bundle_gives_the_source_model(bundle_dir: Path, tiny_trained_model) -> None:
    model, spec, manifest = load_bundle(bundle_dir, "cpu")
    source, source_spec = load_model(tiny_trained_model.checkpoint_path, tiny_trained_model.config_path)

    assert not model.training
    assert spec.to_dict() == manifest["spec"]
    assert spec.network_config == source_spec.network_config
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key


def test_manifest_summary_is_one_line_per_field(bundle_dir: Path) -> None:
    summary = manifest_summary(read_manifest(bundle_dir))
    assert summary["name"] == "tiny"
    assert summary["standard_names"] == "rainfall_rate"
    assert summary["timestep"] == "PT5M"
    assert not any("\n" in value for value in summary.values())


def test_missing_directory_is_a_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="does not exist"):
        load_bundle(tmp_path / "nowhere")


def test_missing_manifest_is_a_model_error(bundle_copy: Path) -> None:
    (bundle_copy / MANIFEST_FILE).unlink()
    with pytest.raises(ModelError, match="manifest.json is missing"):
        load_bundle(bundle_copy)


def test_missing_checkpoint_is_a_model_error(bundle_copy: Path) -> None:
    (bundle_copy / MODEL_FILE).unlink()
    with pytest.raises(ModelError, match="model.ckpt is missing"):
        load_bundle(bundle_copy)


def test_invalid_manifest_is_a_model_error(bundle_copy: Path) -> None:
    (bundle_copy / MANIFEST_FILE).write_text("{not json")
    with pytest.raises(ModelError, match="cannot read manifest.json"):
        load_bundle(bundle_copy)


def test_newer_manifest_format_is_a_model_error(bundle_copy: Path) -> None:
    manifest = json.loads((bundle_copy / MANIFEST_FILE).read_text())
    (bundle_copy / MANIFEST_FILE).write_text(json.dumps(manifest | {"format_version": 99}))
    with pytest.raises(ModelError, match="manifest format 99"):
        read_manifest(bundle_copy)


def test_sha_mismatch_is_a_model_error(bundle_copy: Path) -> None:
    with open(bundle_copy / MODEL_FILE, "ab") as f:
        f.write(b"\0")
    with pytest.raises(ModelError, match="sha256"):
        load_bundle(bundle_copy)


def test_corrupt_checkpoint_is_a_model_error(bundle_copy: Path) -> None:
    """A checkpoint whose sha matches (the manifest was rewritten) but that is not a checkpoint."""
    (bundle_copy / MODEL_FILE).write_bytes(b"this is not a torch checkpoint")
    manifest = json.loads((bundle_copy / MANIFEST_FILE).read_text())
    manifest["model_sha256"] = sha256_file(bundle_copy / MODEL_FILE)
    (bundle_copy / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelError, match="cannot load model.ckpt"):
        load_bundle(bundle_copy)


def test_spec_disagreeing_with_manifest_is_a_model_error(bundle_copy: Path) -> None:
    manifest = json.loads((bundle_copy / MANIFEST_FILE).read_text())
    manifest["spec"]["input_steps"] += 1
    (bundle_copy / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ModelError, match="differs"):
        load_bundle(bundle_copy)


def test_plain_checkpoint_without_config_cannot_be_bundled(tiny_trained_model, tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="config"):
        create_bundle(tiny_trained_model.checkpoint_path, tmp_path / "out")


def test_missing_inputs_are_all_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelError) as excinfo:
        create_bundle(tmp_path / "no.ckpt", tmp_path / "out", config_path=tmp_path / "no.yaml")
    assert "no.ckpt" in str(excinfo.value)
    assert "no.yaml" in str(excinfo.value)


def test_non_empty_output_is_refused_unless_overwriting(bundle_copy: Path, tiny_trained_model) -> None:
    with pytest.raises(ConfigError, match="not empty"):
        create_bundle(tiny_trained_model.checkpoint_path, bundle_copy, config_path=tiny_trained_model.config_path)
    create_bundle(
        tiny_trained_model.checkpoint_path, bundle_copy, config_path=tiny_trained_model.config_path, overwrite=True
    )
    assert read_manifest(bundle_copy)["name"] is None


def test_failed_verification_leaves_no_bundle(
    tiny_trained_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose forward pass gives NaN is never bundled."""
    import mlcast_icb.bundle as bundle_module

    real_load_model = bundle_module.load_model

    def load_broken_model(*args, **kwargs):
        model, spec = real_load_model(*args, **kwargs)
        model.forward = lambda x, forecast_steps, ensemble_size=None: torch.full(
            (x.shape[0], forecast_steps, x.shape[2], x.shape[3], x.shape[4]), float("nan")
        )
        return model, spec

    monkeypatch.setattr(bundle_module, "load_model", load_broken_model)
    out = tmp_path / "out"
    with pytest.raises(ModelError, match="non-finite"):
        create_bundle(tiny_trained_model.checkpoint_path, out, config_path=tiny_trained_model.config_path)
    assert list(out.iterdir()) == []


def test_converted_convgru_ensemble_checkpoint_bundles_without_config(
    tiny_convgru_ensemble_checkpoint: Path, tmp_path: Path
) -> None:
    """An IRENE-shaped checkpoint, converted to mlcast, embeds its spec: no YAML needed."""
    checkpoint_path, _ = convert(tiny_convgru_ensemble_checkpoint, tmp_path / "converted")
    out = create_bundle(checkpoint_path, tmp_path / "bundle", name="irene-tiny")

    _, spec, manifest = load_bundle(out)
    assert spec.input_steps == 6
    assert spec.forecast_steps == 12
    assert spec.timestep == "PT5M"
    assert manifest["source_config_sha256"] is None
