"""Convert an FBK ``convgru_ensemble`` checkpoint (e.g. IRENE) into an mlcast checkpoint.

FBK's ``convgru_ensemble`` package trains a ``RadarLightningModel`` that holds an
``EncoderDecoder`` network as ``self.model``. That network has the same module tree as
mlcast's ``ConvGruModel``, which ``NowcastLightningModule`` holds as ``self.network``, and
both use the same normalisation (Z = 200 R^1.6, 0-60 dBZ mapped onto [-1, 1]). Converting
is therefore a rename of the ``model.`` state-dict prefix to ``network.``, plus writing the
training config mlcast would have written for such a model.

The converter writes, to an output directory:

- ``model.ckpt``: a stripped ``NowcastLightningModule`` checkpoint. It holds the renamed
  weights and the module's hyperparameters, but no optimizer, scheduler, loop or callback
  state (513 of IRENE's 770 MB are Adam state). By default it also holds an
  ``InferenceSpec`` (``checkpoint["mlcast_inference"]``), so it is self-describing. It
  contains only tensors and plain values, so it loads with ``weights_only=True``, and
  ``NowcastLightningModule.load_from_checkpoint(path, network=net, strict=True)`` works.
- ``config.yaml``: the Fiddle training config, in the format training writes
  (``mlcast.config.orchestrator``), starting from ``training_experiment`` with the
  network, loss, optimizer and ``data.dataset_factory`` (standard names and step counts)
  set from the source checkpoint. Its data paths are mlcast's placeholders.

The conversion is checked before returning: the weights load strictly, the written files
load again with ``load_model`` to bit-identical weights, and the spec read from
``config.yaml`` matches the one in the checkpoint.

Usage::

    python -m mlcast_icb.upstream.convert_convgru_ensemble hf://it4lia/irene out/
    python -m mlcast_icb.upstream.convert_convgru_ensemble path/to/model.ckpt out/

Proposed for mlcast as ``mlcast/inference/convert.py`` with an ``import-convgru-ensemble``
subcommand (see ``README.md``, PR G). Unlike ``spec.py`` and ``checkpoint.py``, this module
imports ``mlcast.config`` (and so the training loggers), since it writes a training config.
"""

import argparse
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fiddle as fdl
import pytorch_lightning as pl
import torch
from fiddle.experimental.yaml_serialization import dump_yaml
from loguru import logger

from mlcast.config import training_experiment, validate_config
from mlcast.models.convgru import ConvGruModel

from .checkpoint import INFERENCE_SPEC_KEY, load_model, save_inference_metadata
from .spec import InferenceSpec, load_config_loose, spec_from_config

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # huggingface_hub is only needed for hf:// sources
    hf_hub_download = None  # type: ignore[assignment]

SRC_PREFIX = "model."
DST_PREFIX = "network."
HF_SCHEME = "hf://"
HF_DEFAULT_FILENAME = "model.ckpt"

#: Source hyperparameters that become ``ConvGruModel`` arguments.
NETWORK_HPARAMS = ("input_channels", "num_blocks", "noisy_decoder")
#: Source hyperparameters that become ``NowcastLightningModule`` arguments (the ones it saves).
MODULE_HPARAMS = ("ensemble_size", "loss_class", "loss_params", "masked_loss")

CHECKPOINT_FILENAME = "model.ckpt"
CONFIG_FILENAME = "config.yaml"


def _torch_optim_classes() -> list[type]:
    """Optimizer and LR-scheduler classes, which ``convgru_ensemble`` pickles into its hparams."""
    return [
        obj
        for module in (torch.optim, torch.optim.lr_scheduler)
        for obj in vars(module).values()
        if inspect.isclass(obj)
    ]


def fetch_checkpoint(source: str | Path) -> Path:
    """Return a local path for a checkpoint, downloading it from Hugging Face if needed.

    Parameters
    ----------
    source : str or Path
        A local path, or ``hf://<org>/<repo>[/<file>]`` (``<file>`` defaults to
        ``model.ckpt``). Downloads go to the Hugging Face cache, so they happen once.

    Returns
    -------
    path : Path
        Local path of the checkpoint.

    Raises
    ------
    ImportError
        For an ``hf://`` source when ``huggingface_hub`` is not installed.
    """
    source = str(source)
    if not source.startswith(HF_SCHEME):
        return Path(source)
    if hf_hub_download is None:
        raise ImportError(f"Downloading {source} needs huggingface_hub; install it or pass a local path")
    parts = source.removeprefix(HF_SCHEME).split("/")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:]) or HF_DEFAULT_FILENAME
    logger.info("Fetching {} from Hugging Face repo {}", filename, repo_id)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def read_convgru_ensemble_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a ``convgru_ensemble`` checkpoint without unpickling arbitrary objects.

    Parameters
    ----------
    path : str or Path
        Path to the checkpoint.

    Returns
    -------
    checkpoint : dict of str to Any
        The checkpoint dictionary, loaded with ``weights_only=True`` and only the
        ``torch.optim`` classes its hyperparameters reference allowed.
    """
    with torch.serialization.safe_globals(_torch_optim_classes()):
        return torch.load(path, map_location="cpu", weights_only=True)


def _qualified_name(value: Any) -> Any:
    """Replace classes by their qualified name, so the value is plain data."""
    if inspect.isclass(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {k: _qualified_name(v) for k, v in value.items()}
    return value


def _training_config(
    hparams: dict[str, Any], standard_names: Sequence[str], input_steps: int, forecast_steps: int
) -> fdl.Config:
    """The training config mlcast would have written for a model with these hyperparameters."""
    cfg = training_experiment.as_buildable()
    cfg.pl_module.network = fdl.Config(ConvGruModel, **{k: hparams[k] for k in NETWORK_HPARAMS})
    for key in MODULE_HPARAMS:
        if key in hparams:
            setattr(cfg.pl_module, key, hparams[key])
    if inspect.isclass(hparams.get("optimizer_class")):
        cfg.pl_module.optimizer = fdl.Partial(hparams["optimizer_class"], **hparams.get("optimizer_params") or {})
    if inspect.isclass(hparams.get("lr_scheduler_class")):
        cfg.pl_module.lr_scheduler = fdl.Partial(
            hparams["lr_scheduler_class"], **hparams.get("lr_scheduler_params") or {}
        )
    cfg.data.dataset_factory.standard_names = list(standard_names)
    cfg.data.dataset_factory.input_steps = input_steps
    cfg.data.dataset_factory.forecast_steps = forecast_steps
    cfg.data.dataset_factory.return_mask = bool(cfg.pl_module.masked_loss)
    validate_config(cfg)
    return cfg


def convert_checkpoint(
    checkpoint: dict[str, Any],
    standard_names: Sequence[str] = ("rainfall_rate",),
    input_steps: int = 6,
    timestep: str | None = "PT5M",
    embed_spec: bool = True,
    source: str | None = None,
) -> tuple[dict[str, Any], fdl.Config]:
    """Convert a loaded ``convgru_ensemble`` checkpoint into an mlcast checkpoint and config.

    Parameters
    ----------
    checkpoint : dict of str to Any
        The source checkpoint, as returned by :func:`read_convgru_ensemble_checkpoint`.
    standard_names : Sequence[str], optional
        CF standard names of the model's channels. ``convgru_ensemble`` models take rain
        rates in mm/h. Default is ``("rainfall_rate",)``.
    input_steps : int, optional
        Number of past steps the model was trained on, which ``convgru_ensemble`` does
        not record. Default is ``6`` (IRENE).
    timestep : str or None, optional
        Time between steps as an ISO 8601 duration. Default is ``"PT5M"`` (IRENE).
    embed_spec : bool, optional
        Store the ``InferenceSpec`` in the checkpoint. Default is ``True``.
    source : str or None, optional
        Where the checkpoint came from, recorded in ``checkpoint["converted_from"]``.

    Returns
    -------
    checkpoint : dict of str to Any
        The mlcast checkpoint: ``state_dict`` with ``network.`` keys, ``hyper_parameters``
        for ``NowcastLightningModule``, and no optimizer state.
    cfg : fdl.Config
        The matching training config.

    Raises
    ------
    ValueError
        If the source state dict has keys outside ``model.``, or its hyperparameters lack
        the network arguments.
    RuntimeError
        If the weights do not load strictly into the configured network.
    """
    state_dict, hparams = checkpoint["state_dict"], dict(checkpoint["hyper_parameters"])
    unexpected = [k for k in state_dict if not k.startswith(SRC_PREFIX)]
    if unexpected:
        raise ValueError(f"Expected only {SRC_PREFIX!r} keys in the state dict, also found {unexpected}")
    missing = [k for k in (*NETWORK_HPARAMS, "forecast_steps") if k not in hparams]
    if missing:
        raise ValueError(f"Checkpoint hyperparameters lack {missing}; got {sorted(hparams)}")
    if hparams["input_channels"] != len(standard_names):
        raise ValueError(f"Model has {hparams['input_channels']} channels but got standard names {standard_names}")

    cfg = _training_config(hparams, standard_names, input_steps, hparams["forecast_steps"])
    new_state_dict = {DST_PREFIX + k.removeprefix(SRC_PREFIX): v.contiguous() for k, v in state_dict.items()}
    module_hparams = {k: hparams[k] for k in MODULE_HPARAMS if k in hparams}

    # Strict-load check: every tensor has a place in the network and vice versa.
    fdl.build(cfg.pl_module).load_state_dict(new_state_dict, strict=True)

    converted = {
        "state_dict": new_state_dict,
        "hyper_parameters": module_hparams,
        "pytorch-lightning_version": pl.__version__,
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "converted_from": {
            "format": "convgru_ensemble",
            "source": source,
            "pytorch-lightning_version": checkpoint.get("pytorch-lightning_version"),
            "hyper_parameters": _qualified_name(hparams),
        },
    }
    if embed_spec:
        save_inference_metadata(converted, spec_from_config(cfg, timestep=timestep))
    return converted, cfg


def convert(
    source: str | Path,
    out_dir: str | Path,
    standard_names: Sequence[str] = ("rainfall_rate",),
    input_steps: int = 6,
    timestep: str | None = "PT5M",
    embed_spec: bool = True,
) -> tuple[Path, Path]:
    """Convert a ``convgru_ensemble`` checkpoint file and write ``model.ckpt`` + ``config.yaml``.

    Parameters
    ----------
    source : str or Path
        Local checkpoint path, or ``hf://<org>/<repo>[/<file>]``.
    out_dir : str or Path
        Output directory, created if needed.
    standard_names, input_steps, timestep, embed_spec
        See :func:`convert_checkpoint`.

    Returns
    -------
    checkpoint_path : Path
        The written ``model.ckpt``.
    config_path : Path
        The written ``config.yaml``.

    Raises
    ------
    RuntimeError
        If the written files do not load back to the same weights and spec.
    """
    src_path = fetch_checkpoint(source)
    checkpoint = read_convgru_ensemble_checkpoint(src_path)
    converted, cfg = convert_checkpoint(
        checkpoint,
        standard_names=standard_names,
        input_steps=input_steps,
        timestep=timestep,
        embed_spec=embed_spec,
        source=str(source),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, config_path = out_dir / CHECKPOINT_FILENAME, out_dir / CONFIG_FILENAME
    torch.save(converted, checkpoint_path)
    config_path.write_text(dump_yaml(cfg))

    # Check the written files: weights reload strictly and unchanged, and the config
    # describes the same model as the spec in the checkpoint.
    model, spec = load_model(checkpoint_path, None if embed_spec else config_path, timestep=timestep)
    for key, value in model.state_dict().items():
        if not torch.equal(value, converted["state_dict"][key]):
            raise RuntimeError(f"Weight {key} changed in the round trip through {checkpoint_path}")
    spec_from_yaml = spec_from_config(load_config_loose(config_path), timestep=timestep)
    if spec_from_yaml != spec:
        raise RuntimeError(f"{config_path} describes a different model than the spec in {checkpoint_path}")
    if embed_spec and InferenceSpec.from_dict(converted[INFERENCE_SPEC_KEY]) != spec:
        raise RuntimeError(f"The inference spec in {checkpoint_path} changed in the round trip")

    logger.info(
        "Converted {} ({:.1f} MB) to {} ({:.1f} MB) and {}",
        src_path,
        src_path.stat().st_size / 1e6,
        checkpoint_path,
        checkpoint_path.stat().st_size / 1e6,
        config_path,
    )
    return checkpoint_path, config_path


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        prog="python -m mlcast_icb.upstream.convert_convgru_ensemble",
        description="Convert an FBK convgru_ensemble checkpoint (e.g. IRENE) into an mlcast checkpoint + config.",
    )
    parser.add_argument("source", help="checkpoint path, or hf://<org>/<repo>[/<file>] (e.g. hf://it4lia/irene)")
    parser.add_argument("out_dir", help=f"output directory for {CHECKPOINT_FILENAME} and {CONFIG_FILENAME}")
    parser.add_argument(
        "--standard-name",
        dest="standard_names",
        action="append",
        help="CF standard name of each channel, repeated per channel (default: rainfall_rate)",
    )
    parser.add_argument("--input-steps", type=int, default=6, help="past steps the model takes (default: 6)")
    parser.add_argument("--timestep", default="PT5M", help="ISO 8601 time between steps (default: PT5M)")
    parser.add_argument(
        "--no-embed-spec",
        dest="embed_spec",
        action="store_false",
        help="do not store the inference spec in the checkpoint (config.yaml is then needed to load it)",
    )
    args = parser.parse_args(argv)
    convert(
        args.source,
        args.out_dir,
        standard_names=args.standard_names or ["rainfall_rate"],
        input_steps=args.input_steps,
        timestep=args.timestep,
        embed_spec=args.embed_spec,
    )


if __name__ == "__main__":
    main()
