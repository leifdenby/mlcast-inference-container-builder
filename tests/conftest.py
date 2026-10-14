"""Shared test fixtures: tiny trained models and checkpoints, and the opt-in IRENE tests."""

import os
from dataclasses import dataclass
from pathlib import Path

import fiddle as fdl
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from mlcast.config import training_experiment
from mlcast.config.orchestrator import _log_experiment_config_yaml_file
from mlcast.models.convgru import ConvGruModel

#: Set to 1 to run the slow tests on the real IRENE checkpoint (like ``--run-irene``).
IRENE_ENV = "MLCAST_ICB_IRENE"

TINY_INPUT_STEPS = 3
TINY_FORECAST_STEPS = 2
TINY_NUM_BLOCKS = 2
TINY_SIZE = 8  # a multiple of 2**TINY_NUM_BLOCKS


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-irene",
        action="store_true",
        default=False,
        help=f"run the slow tests on the real IRENE checkpoint (or set {IRENE_ENV}=1)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-irene") or os.environ.get(IRENE_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"IRENE test; run with --run-irene or {IRENE_ENV}=1")
    for item in items:
        if "irene" in item.keywords:
            item.add_marker(skip)


@dataclass(frozen=True)
class TrainedModel:
    """A model trained by mlcast: its checkpoint, its training config YAML and its shape."""

    checkpoint_path: Path
    config_path: Path
    input_steps: int
    forecast_steps: int
    size: int


def _synthetic_batches(n_samples: int, input_steps: int, forecast_steps: int, size: int) -> list[dict]:
    """Samples as mlcast's datasets yield them: normalised (T, C, H, W) input/target plus a target mask."""
    generator = torch.Generator().manual_seed(0)
    steps = input_steps + forecast_steps
    samples = []
    for _ in range(n_samples):
        data = torch.rand(steps, 1, size, size, generator=generator) * 2 - 1
        samples.append(
            {
                "input": data[:input_steps],
                "target": data[input_steps:],
                "target_mask": torch.ones(forecast_steps, 1, size, size, dtype=torch.bool),
            }
        )
    return samples


@pytest.fixture(scope="session")
def tiny_trained_model(tmp_path_factory: pytest.TempPathFactory) -> TrainedModel:
    """A tiny ConvGruModel trained for two steps with mlcast, with its config.yaml written as training does."""
    root = tmp_path_factory.mktemp("tiny_trained_model")

    # The default training config, shrunk to a tiny network and synthetic data
    cfg = training_experiment.as_buildable()
    cfg.pl_module.network = fdl.Config(ConvGruModel, input_channels=1, num_blocks=TINY_NUM_BLOCKS)
    cfg.pl_module.optimizer = fdl.Partial(torch.optim.Adam, lr=1e-3)
    cfg.pl_module.lr_scheduler = None  # ReduceLROnPlateau needs a val_loss, which this run has no data for
    cfg.data.dataset_factory.standard_names = ["rainfall_rate"]
    cfg.data.dataset_factory.input_steps = TINY_INPUT_STEPS
    cfg.data.dataset_factory.forecast_steps = TINY_FORECAST_STEPS

    torch.manual_seed(0)
    module = fdl.build(cfg.pl_module)
    checkpoint_callback = ModelCheckpoint(dirpath=root / "checkpoints", filename="model")
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        callbacks=[checkpoint_callback],
        default_root_dir=root,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    _log_experiment_config_yaml_file(cfg, trainer)
    loader = DataLoader(_synthetic_batches(4, TINY_INPUT_STEPS, TINY_FORECAST_STEPS, TINY_SIZE), batch_size=2)
    trainer.fit(module, train_dataloaders=loader)

    return TrainedModel(
        checkpoint_path=Path(checkpoint_callback.best_model_path),
        config_path=root / "config.yaml",
        input_steps=TINY_INPUT_STEPS,
        forecast_steps=TINY_FORECAST_STEPS,
        size=TINY_SIZE,
    )


@pytest.fixture(scope="session")
def tiny_convgru_ensemble_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny checkpoint shaped like FBK convgru_ensemble's (IRENE's), with fake optimizer state."""
    torch.manual_seed(0)
    network = ConvGruModel(input_channels=1, num_blocks=TINY_NUM_BLOCKS, noisy_decoder=True)
    state_dict = {f"model.{k}": v for k, v in network.state_dict().items()}
    checkpoint = {
        "epoch": 3,
        "global_step": 42,
        "pytorch-lightning_version": "2.6.0",
        "state_dict": state_dict,
        "loops": {"fit_loop": {"state_dict": {}}},
        "callbacks": {"EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}": {"best_score": torch.tensor(0.05)}},
        "optimizer_states": [
            {
                "state": {
                    i: {"step": torch.tensor(42.0), "exp_avg": torch.randn_like(v), "exp_avg_sq": torch.rand_like(v)}
                    for i, v in enumerate(state_dict.values())
                },
                "param_groups": [{"lr": 1e-4, "params": list(range(len(state_dict)))}],
            }
        ],
        "lr_schedulers": [{"factor": 0.5, "patience": 10, "mode": "min"}],
        "hparams_name": "kwargs",
        "hyper_parameters": {
            "input_channels": 1,
            "forecast_steps": 12,
            "num_blocks": TINY_NUM_BLOCKS,
            "ensemble_size": 10,
            "noisy_decoder": True,
            "loss_class": "crps",
            "loss_params": {"temporal_lambda": 0.01},
            "masked_loss": False,
            "optimizer_class": torch.optim.Adam,
            "optimizer_params": {"lr": 0.0001, "fused": True},
            "lr_scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "lr_scheduler_params": {"mode": "min", "factor": 0.5, "patience": 10},
        },
    }
    path = tmp_path_factory.mktemp("convgru_ensemble") / "model.ckpt"
    torch.save(checkpoint, path)
    return path
