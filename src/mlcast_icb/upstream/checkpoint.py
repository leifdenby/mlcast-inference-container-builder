"""Load a trained model for inference from a checkpoint, with or without its training config.

A checkpoint written by ``NowcastLightningModule`` today holds the weights and four
scalar hyperparameters, but not the network config, so loading it needs the training
YAML too. This module loads either kind:

- a *self-describing* checkpoint, holding an :class:`~.spec.InferenceSpec` under
  ``checkpoint["mlcast_inference"]`` (written by :func:`save_inference_metadata`), needs
  nothing else;
- a plain training checkpoint is loaded together with its training YAML, from which only
  ``cfg.pl_module.network`` is built (never the whole ``Experiment``, whose data and
  trainer sections may reference classes that no longer import).

Proposed for mlcast as ``NowcastLightningModule.on_save_checkpoint`` and a config-free
``NowcastLightningModule.from_checkpoint`` (see ``README.md``, PR A).
"""

import dataclasses
from pathlib import Path
from typing import Any

import fiddle as fdl
import torch
from loguru import logger

from mlcast.nowcasting_module import NowcastLightningModule

from .spec import InferenceSpec, load_config_loose, loads_config_loose, spec_from_config, to_buildable

#: Checkpoint key under which the inference spec is stored.
INFERENCE_SPEC_KEY = "mlcast_inference"


def save_inference_metadata(checkpoint: dict[str, Any], spec: InferenceSpec) -> None:
    """Store an inference spec in a checkpoint dictionary, making it self-describing.

    Meant to be called from ``NowcastLightningModule.on_save_checkpoint``, or on a loaded
    checkpoint dictionary before saving it again.

    Parameters
    ----------
    checkpoint : dict of str to Any
        The checkpoint dictionary, modified in place.
    spec : InferenceSpec
        The spec to store, as ``spec.to_dict()`` (plain JSON-able values, so the
        checkpoint still loads with ``torch.load(..., weights_only=True)``).
    """
    checkpoint[INFERENCE_SPEC_KEY] = spec.to_dict()


def load_inference_metadata(checkpoint: dict[str, Any]) -> InferenceSpec | None:
    """Return the inference spec stored in a checkpoint dictionary, if there is one.

    Parameters
    ----------
    checkpoint : dict of str to Any
        A loaded checkpoint dictionary.

    Returns
    -------
    spec : InferenceSpec or None
        The stored spec, or ``None`` for a checkpoint without one.
    """
    spec_dict = checkpoint.get(INFERENCE_SPEC_KEY)
    return None if spec_dict is None else InferenceSpec.from_dict(spec_dict)


def build_network(spec: InferenceSpec) -> torch.nn.Module:
    """Build the (untrained) network described by a spec's ``network_config``.

    Parameters
    ----------
    spec : InferenceSpec
        The inference spec.

    Returns
    -------
    network : torch.nn.Module
        A freshly initialised network.
    """
    return fdl.build(to_buildable(loads_config_loose(spec.network_config)))


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str = "cpu",
    timestep: str | None = None,
) -> tuple[NowcastLightningModule, InferenceSpec]:
    """Load a trained model and its inference spec, ready for inference.

    The spec comes from the checkpoint when it holds one (``checkpoint["mlcast_inference"]``),
    and otherwise from the training config at ``config_path``. Only the network is built
    from the config; the weights are then loaded with ``strict=True``.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a ``NowcastLightningModule`` checkpoint (``.ckpt``).
    config_path : str or Path or None, optional
        Path to the training config YAML. Required unless the checkpoint holds an
        inference spec, and ignored (with a warning) if it does. Default is ``None``.
    device : str, optional
        Device to load the model onto. Default is ``"cpu"``.
    timestep : str or None, optional
        ISO 8601 duration (e.g. ``"PT5M"``) that overrides the spec's ``timestep``, which
        is ``None`` for specs read from a training config. Default is ``None`` (keep).

    Returns
    -------
    model : NowcastLightningModule
        The model, on ``device`` and in eval mode.
    spec : InferenceSpec
        The model's inference spec.

    Raises
    ------
    ValueError
        If the checkpoint holds no inference spec and no ``config_path`` is given.
    RuntimeError
        If the weights do not match the network exactly (missing or unexpected keys,
        or shape mismatches).

    Notes
    -----
    The checkpoint is loaded with ``weights_only=False``, as
    ``NowcastLightningModule.from_checkpoint`` does, because training checkpoints can
    contain arbitrary pickled objects. Only load checkpoints from trusted sources.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)

    spec = load_inference_metadata(checkpoint)
    if spec is not None:
        if config_path is not None:
            logger.warning(
                "Checkpoint {} holds an inference spec; ignoring the config {}", checkpoint_path, config_path
            )
    elif config_path is not None:
        spec = spec_from_config(load_config_loose(config_path))
    else:
        raise ValueError(
            f"Checkpoint {checkpoint_path} holds no inference spec (key {INFERENCE_SPEC_KEY!r}), so the "
            "training config YAML it was trained with is needed to rebuild the network; pass config_path."
        )
    if timestep is not None:
        spec = dataclasses.replace(spec, timestep=timestep)

    model = NowcastLightningModule(network=build_network(spec), **checkpoint.get("hyper_parameters", {}))
    # The same order as Lightning's load_from_checkpoint: the hook first, then the weights.
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, spec
