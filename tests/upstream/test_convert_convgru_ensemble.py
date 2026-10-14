"""Tests for the ``convgru_ensemble`` checkpoint converter (``convert_convgru_ensemble.py``).

Written to move into mlcast's ``tests/inference/test_convert.py`` unchanged, apart from the
import of the modules under test.

The tests marked ``irene`` convert the real IRENE checkpoint (``it4lia/irene`` on Hugging
Face, 770 MB, cached in the Hugging Face cache). They are skipped unless pytest runs with
``--run-irene`` or ``ICB_TEST_IRENE=1``. ``ICB_TEST_IRENE_CKPT`` and
``ICB_TEST_IRENE_SAMPLE`` point them at local copies of ``model.ckpt`` and
``examples/sample_data.nc`` instead.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from mlcast.config.consistency_checks import validate_config
from mlcast.config.loader import load_yaml_config
from mlcast.models.convgru import ConvGruModel
from mlcast.nowcasting_module import NowcastLightningModule
from mlcast_icb.upstream.checkpoint import INFERENCE_SPEC_KEY, load_model
from mlcast_icb.upstream.convert_convgru_ensemble import (
    convert,
    convert_checkpoint,
    fetch_checkpoint,
    main,
    read_convgru_ensemble_checkpoint,
)
from mlcast_icb.upstream.processing import postprocess, preprocess

IRENE_SOURCE = "hf://it4lia/irene"
IRENE_SAMPLE_SOURCE = "hf://it4lia/irene/examples/sample_data.nc"


def _source_network(path: Path, num_blocks: int) -> ConvGruModel:
    """The source checkpoint's network, loaded directly into ConvGruModel (the bare-network route)."""
    state_dict = read_convgru_ensemble_checkpoint(path)["state_dict"]
    network = ConvGruModel(input_channels=1, num_blocks=num_blocks, noisy_decoder=True)
    network.load_state_dict({k.removeprefix("model."): v for k, v in state_dict.items()}, strict=True)
    return network.eval()


def _seeded(network: torch.nn.Module, x: torch.Tensor, steps: int, ensemble_size: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return network(x, steps=steps, ensemble_size=ensemble_size)


def test_converted_checkpoint_matches_source(tiny_convgru_ensemble_checkpoint: Path, tmp_path: Path) -> None:
    """The converted model loads strictly and forecasts exactly like the source network."""
    # Given a convgru_ensemble-shaped checkpoint with optimizer state
    # When converting it
    checkpoint_path, config_path = convert(tiny_convgru_ensemble_checkpoint, tmp_path / "out")

    # Then the checkpoint is stripped, plain data, with network.* weights
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert "optimizer_states" not in checkpoint
    assert all(key.startswith("network.") for key in checkpoint["state_dict"])
    assert checkpoint_path.stat().st_size < tiny_convgru_ensemble_checkpoint.stat().st_size / 2
    # And it loads like a native mlcast checkpoint, through Lightning and through load_model
    network = ConvGruModel(input_channels=1, num_blocks=2, noisy_decoder=True)
    lightning = NowcastLightningModule.load_from_checkpoint(checkpoint_path, network=network, strict=True).eval()
    model, spec = load_model(checkpoint_path)
    assert model.hparams["ensemble_size"] == 10
    # And the spec describes IRENE-style data
    assert spec.standard_names == ("rainfall_rate",)
    assert spec.units == ("mm h-1",)
    assert (spec.input_steps, spec.forecast_steps, spec.timestep) == (6, 12, "PT5M")
    assert (spec.spatial_divisor, spec.default_ensemble_size) == (4, 10)
    # And for the same seed the forecast equals the source network's, bit for bit
    x = torch.rand(1, 6, 1, 8, 12, generator=torch.Generator().manual_seed(1)) * 2 - 1
    expected = _seeded(_source_network(tiny_convgru_ensemble_checkpoint, num_blocks=2), x, 4, 3)
    torch.testing.assert_close(_seeded(model.network, x, 4, 3), expected, rtol=0, atol=0)
    torch.testing.assert_close(_seeded(lightning.network, x, 4, 3), expected, rtol=0, atol=0)


def test_converted_config_is_a_valid_training_config(tiny_convgru_ensemble_checkpoint: Path, tmp_path: Path) -> None:
    """config.yaml loads with mlcast's own loader and passes its training-config checks."""
    # Given a converted checkpoint
    _, config_path = convert(tiny_convgru_ensemble_checkpoint, tmp_path / "out")

    # When loading its config with mlcast's loader
    cfg = load_yaml_config(config_path)

    # Then it is a consistent training config for the source model
    validate_config(cfg)
    assert cfg.pl_module.network.__fn_or_cls__ is ConvGruModel
    assert (cfg.pl_module.network.num_blocks, cfg.pl_module.network.noisy_decoder) == (2, True)
    assert (cfg.pl_module.ensemble_size, cfg.pl_module.loss_class, cfg.pl_module.masked_loss) == (10, "crps", False)
    assert cfg.pl_module.optimizer.__fn_or_cls__ is torch.optim.Adam
    assert cfg.pl_module.optimizer.lr == 1e-4
    factory = cfg.data.dataset_factory
    assert (factory.standard_names, factory.input_steps, factory.forecast_steps) == (["rainfall_rate"], 6, 12)


def test_cli_without_embedded_spec(tiny_convgru_ensemble_checkpoint: Path, tmp_path: Path) -> None:
    """The CLI converts; without an embedded spec, the checkpoint loads through config.yaml."""
    # Given the CLI run with --no-embed-spec and custom input steps
    out_dir = tmp_path / "out"
    main([str(tiny_convgru_ensemble_checkpoint), str(out_dir), "--no-embed-spec", "--input-steps", "4"])

    # Then the checkpoint holds no spec, and needs and accepts the config
    assert INFERENCE_SPEC_KEY not in torch.load(out_dir / "model.ckpt", weights_only=True)
    with pytest.raises(ValueError, match="config_path"):
        load_model(out_dir / "model.ckpt")
    _, spec = load_model(out_dir / "model.ckpt", out_dir / "config.yaml", timestep="PT5M")
    assert spec.input_steps == 4


def test_unexpected_checkpoint_layouts_are_rejected(tiny_convgru_ensemble_checkpoint: Path) -> None:
    """Weights outside ``model.``, missing network hyperparameters or wrong channel counts are errors."""
    checkpoint = read_convgru_ensemble_checkpoint(tiny_convgru_ensemble_checkpoint)
    with pytest.raises(ValueError, match="extra.weight"):
        convert_checkpoint({**checkpoint, "state_dict": {**checkpoint["state_dict"], "extra.weight": torch.zeros(1)}})
    hparams = {k: v for k, v in checkpoint["hyper_parameters"].items() if k != "num_blocks"}
    with pytest.raises(ValueError, match="num_blocks"):
        convert_checkpoint({**checkpoint, "hyper_parameters": hparams})
    with pytest.raises(ValueError, match="channels"):
        convert_checkpoint(checkpoint, standard_names=["rainfall_rate", "rainfall_flux"])


# --------------------------------------------------------------------------------------
# The real IRENE checkpoint (opt-in)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def irene_checkpoint() -> Path:
    return fetch_checkpoint(os.environ.get("ICB_TEST_IRENE_CKPT", IRENE_SOURCE))


@pytest.fixture(scope="module")
def irene_sample() -> xr.DataArray:
    """The last 6 steps of IRENE's sample data, centre-cropped to 256x256.

    The file labels ``RR`` as ``kg m-2 s-1``, but its values are mm/h, which is how IRENE
    reads them and how they are used here.
    """
    path = fetch_checkpoint(os.environ.get("ICB_TEST_IRENE_SAMPLE", IRENE_SAMPLE_SOURCE))
    with xr.open_dataset(path) as ds:
        rr = ds["RR"].load()
    y0, x0 = (rr.sizes["y"] - 256) // 2, (rr.sizes["x"] - 256) // 2
    return rr.isel(time=slice(-6, None), y=slice(y0, y0 + 256), x=slice(x0, x0 + 256))


@pytest.mark.irene
def test_irene_conversion_parity(irene_checkpoint: Path, irene_sample: xr.DataArray, tmp_path: Path) -> None:
    """Converted IRENE loads strictly and matches the raw IRENE weights bit for bit on real data."""
    # Given IRENE converted to mlcast
    checkpoint_path, _ = convert(irene_checkpoint, tmp_path / "irene")
    model, spec = load_model(checkpoint_path)
    assert (spec.input_steps, spec.forecast_steps, spec.spatial_divisor) == (6, 12, 32)
    assert checkpoint_path.stat().st_size < 0.4 * irene_checkpoint.stat().st_size

    # When forecasting 256x256 of the sample data with the converted model and with the
    # raw IRENE weights loaded directly into ConvGruModel, with the same seed
    x = torch.from_numpy(preprocess([irene_sample.values.astype(np.float32)], spec.standard_names))[None]
    out = _seeded(model.network, x, spec.forecast_steps, 2, seed=1234)
    expected = _seeded(_source_network(irene_checkpoint, num_blocks=5), x, spec.forecast_steps, 2, seed=1234)

    # Then the normalised outputs are identical
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # And the forecast is plausible rain: finite, within [0, 60 dBZ], and neither empty nor saturated
    rain = postprocess(out, spec.standard_names)
    assert rain.shape == (1, 2, 12, 1, 256, 256)
    assert np.isfinite(rain).all() and rain.min() >= 0 and rain.max() <= 206
    past_wet = float((irene_sample.values > 0.1).mean())
    future_wet = float((rain[0, :, 0] > 0.1).mean())
    assert past_wet > 0.01
    assert 0.2 * past_wet < future_wet < min(1.0, 5 * past_wet)
