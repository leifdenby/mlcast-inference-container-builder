"""Tests for loading models for inference (``checkpoint.py``).

Written to move into mlcast's ``tests/inference/test_checkpoint.py`` unchanged, apart from
the import of the module under test.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import fiddle as fdl
import pytest
import torch

from mlcast.nowcasting_module import NowcastLightningModule
from mlcast_icb.upstream.checkpoint import (
    INFERENCE_SPEC_KEY,
    build_network,
    load_inference_metadata,
    load_model,
    save_inference_metadata,
)
from mlcast_icb.upstream.spec import load_config_loose, spec_from_config, to_buildable


def _forecast(model: NowcastLightningModule, input_steps: int, size: int, seed: int = 0) -> torch.Tensor:
    """A seeded 2-member forecast from a fixed random input."""
    x = torch.rand(1, input_steps, 1, size, size, generator=torch.Generator().manual_seed(1)) * 2 - 1
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(x, forecast_steps=3, ensemble_size=2)


def _self_describing_copy(tiny_trained_model, path: Path, timestep: str | None = "PT5M") -> Path:
    """Copy the tiny model's checkpoint with an inference spec added."""
    checkpoint = torch.load(tiny_trained_model.checkpoint_path, weights_only=False)
    spec = spec_from_config(load_config_loose(tiny_trained_model.config_path), timestep=timestep)
    save_inference_metadata(checkpoint, spec)
    torch.save(checkpoint, path)
    return path


def test_load_model_from_config_matches_lightning(tiny_trained_model) -> None:
    """load_model with the training YAML gives the same weights and output as Lightning's loader."""
    # Given a checkpoint and config.yaml written by mlcast training
    # When loading through load_model, and through Lightning with the network built by hand
    model, spec = load_model(tiny_trained_model.checkpoint_path, tiny_trained_model.config_path)
    network = fdl.build(to_buildable(load_config_loose(tiny_trained_model.config_path).pl_module.network))
    reference = NowcastLightningModule.load_from_checkpoint(
        tiny_trained_model.checkpoint_path, network=network, strict=True, weights_only=False
    ).eval()

    # Then the model is in eval mode, with the trained hyperparameters and weights
    assert not model.training
    assert model.hparams["ensemble_size"] == 2
    raw = torch.load(tiny_trained_model.checkpoint_path, weights_only=False)["state_dict"]
    assert model.state_dict().keys() == raw.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, raw[key]), key
    # And the same seed gives an identical forecast
    torch.testing.assert_close(
        _forecast(model, spec.input_steps, tiny_trained_model.size),
        _forecast(reference, spec.input_steps, tiny_trained_model.size),
        rtol=0,
        atol=0,
    )
    assert spec.timestep is None


def test_self_describing_checkpoint_needs_no_config(tiny_trained_model, tmp_path: Path) -> None:
    """A checkpoint holding an inference spec loads without the YAML, to the same model."""
    # Given the tiny model's checkpoint with an inference spec added
    path = _self_describing_copy(tiny_trained_model, tmp_path / "model.ckpt")

    # When loading it without a config, and the original with its config
    model, spec = load_model(path)
    reference, reference_spec = load_model(tiny_trained_model.checkpoint_path, tiny_trained_model.config_path)

    # Then spec and forecast agree (the spec additionally knows the timestep)
    assert spec == dataclasses.replace(reference_spec, timestep="PT5M")
    torch.testing.assert_close(
        _forecast(model, spec.input_steps, tiny_trained_model.size),
        _forecast(reference, spec.input_steps, tiny_trained_model.size),
        rtol=0,
        atol=0,
    )


def test_stored_spec_is_plain_json(tiny_trained_model, tmp_path: Path) -> None:
    """The stored spec is plain data and reads back equal."""
    # Given a self-describing checkpoint
    path = _self_describing_copy(tiny_trained_model, tmp_path / "model.ckpt")

    # When reading the raw checkpoint dictionary
    checkpoint = torch.load(path, weights_only=False)

    # Then the spec under the documented key survives JSON and reads back equal
    assert json.loads(json.dumps(checkpoint[INFERENCE_SPEC_KEY])) == checkpoint[INFERENCE_SPEC_KEY]
    spec = load_inference_metadata(checkpoint)
    assert spec is not None
    assert isinstance(build_network(spec), type(load_model(path)[0].network))
    assert load_inference_metadata({"state_dict": {}}) is None


def test_timestep_override(tiny_trained_model, tmp_path: Path) -> None:
    """The timestep argument overrides the spec's, whether it came from the YAML or the checkpoint."""
    _, spec = load_model(tiny_trained_model.checkpoint_path, tiny_trained_model.config_path, timestep="PT10M")
    assert spec.timestep == "PT10M"
    _, spec = load_model(_self_describing_copy(tiny_trained_model, tmp_path / "model.ckpt"), timestep="PT2M")
    assert spec.timestep == "PT2M"


def test_checkpoint_without_spec_needs_config(tiny_trained_model) -> None:
    """A plain training checkpoint without a config is a clear error."""
    with pytest.raises(ValueError, match="config_path"):
        load_model(tiny_trained_model.checkpoint_path)


def test_mismatched_network_is_rejected(tiny_trained_model, tmp_path: Path) -> None:
    """Weights that do not fit the configured network fail to load, rather than load partially."""
    # Given a config whose network has one block more than the trained one
    text = tiny_trained_model.config_path.read_text()
    assert "num_blocks: 2" in text
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text.replace("num_blocks: 2", "num_blocks: 3"))

    # Then loading fails
    with pytest.raises(RuntimeError, match="state_dict"):
        load_model(tiny_trained_model.checkpoint_path, config_path)


def test_import_avoids_training_dependencies() -> None:
    """Importing the module does not import the training orchestrator or its loggers."""
    # Given a fresh interpreter
    code = (
        "import sys, mlcast_icb.upstream.checkpoint; "
        "print([m for m in ('mlflow', 'tensorboard', 'wandb', 'mlcast.config.orchestrator') if m in sys.modules])"
    )

    # When importing the module there
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    # Then none of the training-only modules were imported
    assert result.stdout.strip() == "[]"
