"""The model bundle: the one directory an inference image adds on top of the runtime base.

A bundle is two files, baked into the image at ``/model`` (``--model-dir`` /
``MLCAST_MODEL_DIR`` overrides it):

``model.ckpt``
    A stripped ``NowcastLightningModule`` checkpoint: the weights, the module's four
    hyperparameters and the :class:`~mlcast_icb.upstream.spec.InferenceSpec` embedded
    under ``mlcast_inference``. No optimizer, scheduler, loop or callback state, so it is
    roughly a third of a training checkpoint, and no training YAML is needed to load it.
``manifest.json``
    What the bundle is, readable without torch: the spec, the sha256 of ``model.ckpt``
    and of the checkpoint it was made from, the builder and mlcast versions that made
    it, when, and an optional user-supplied model name and version.

:func:`create_bundle` refuses to write a bundle that would not work at run time: it
reloads what it wrote strictly and runs one forward pass on CPU before writing the
manifest. A bundle without a manifest is not a bundle, so a failed verification leaves
nothing loadable behind. :func:`load_bundle` checks the checksum and that the manifest
and the checkpoint agree, and reports anything wrong as a
:class:`~mlcast_icb.errors.ModelError` (exit 6): the image itself is broken, and the
fix is to rebuild it.

All model loading goes through :mod:`mlcast_icb.upstream.checkpoint`; nothing here
imports mlcast.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger

from mlcast_icb import __version__
from mlcast_icb.errors import ConfigError, ModelError
from mlcast_icb.upstream.checkpoint import load_model, save_inference_metadata
from mlcast_icb.upstream.spec import InferenceSpec

#: Where images carry the bundle, unless ``--model-dir`` (``MLCAST_MODEL_DIR``) says otherwise.
DEFAULT_MODEL_DIR = "/model"

#: File names inside a bundle.
MODEL_FILE = "model.ckpt"
MANIFEST_FILE = "manifest.json"

#: Version of the manifest layout written by :func:`create_bundle`.
MANIFEST_FORMAT_VERSION = 1

#: Keys every manifest must have to be usable.
_REQUIRED_MANIFEST_KEYS = ("format_version", "spec", "model_sha256")

#: Seed of the verification forward pass's input, so a failure is reproducible.
_VERIFY_SEED = 0


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks.

    Parameters
    ----------
    path : str or Path
        The file.
    chunk_size : int, optional
        Bytes read at a time.

    Returns
    -------
    str
        The 64-character hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def mlcast_version() -> str:
    """Return the version of the installed mlcast, read from package metadata.

    Returns
    -------
    str
        The version, or ``'unknown'`` when mlcast is not installed as a distribution.
    """
    try:
        return version("mlcast")
    except PackageNotFoundError:  # pragma: no cover - mlcast is a core dependency
        return "unknown"


def create_bundle(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    timestep: str | None = None,
    name: str | None = None,
    model_version: str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a verified model bundle from a training (or converted) checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        The ``NowcastLightningModule`` checkpoint.
    out_dir : str or Path
        Directory to write ``model.ckpt`` and ``manifest.json`` into; created if needed.
    config_path : str or Path or None, optional
        The training config YAML. Needed unless the checkpoint already embeds an
        inference spec (as converted checkpoints do).
    timestep : str or None, optional
        ISO 8601 duration between timesteps (e.g. ``'PT5M'``), overriding the spec's.
        Training configs do not record it, so pass it for a checkpoint loaded through
        its YAML.
    name, model_version : str or None, optional
        Free-form model name and version, recorded in the manifest and logged at run time.
    overwrite : bool, optional
        Replace an existing bundle in ``out_dir``. Without it, a non-empty ``out_dir``
        is refused.

    Returns
    -------
    Path
        ``out_dir``.

    Raises
    ------
    ConfigError
        If ``out_dir`` is not empty (and ``overwrite`` is false) or is not a directory.
    ModelError
        If the checkpoint or config cannot be read, the model cannot be rebuilt, or the
        written bundle fails verification (strict reload, identical weights and spec,
        and one finite CPU forward pass at ``(1, input_steps, C, d, d)`` with
        ``d = spec.spatial_divisor``). Nothing loadable is left in ``out_dir`` then.
    """
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    problems = []
    if not checkpoint_path.is_file():
        problems.append(f"checkpoint {checkpoint_path} does not exist or is not a file")
    if config_path is not None and not Path(config_path).is_file():
        problems.append(f"config {config_path} does not exist or is not a file")
    if problems:
        raise ModelError("Cannot create a bundle: " + "; ".join(problems))
    _prepare_out_dir(out_dir, overwrite)

    logger.info(f"Loading {checkpoint_path}" + (f" with config {config_path}" if config_path else ""))
    try:
        model, spec = load_model(checkpoint_path, config_path, device="cpu", timestep=timestep)
    except Exception as exc:  # noqa: BLE001 - any failure here means this checkpoint cannot be bundled
        raise ModelError(f"Could not load the model from {checkpoint_path}: {type(exc).__name__}: {exc}") from exc
    if spec.timestep is None:
        logger.warning(
            "The model's timestep is unknown (training configs do not record it); pass --timestep, e.g. PT5M, "
            "so the time spacing of the input can be checked and forecast times computed"
        )

    model_path = out_dir / MODEL_FILE
    manifest_path = out_dir / MANIFEST_FILE
    manifest_path.unlink(missing_ok=True)  # until verified, this is not a bundle
    partial = out_dir / (MODEL_FILE + ".partial")
    torch.save(_stripped_checkpoint(model, spec), partial)
    try:
        _verify(partial, model, spec)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(model_path)

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "name": name,
        "model_version": model_version,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder_version": __version__,
        "mlcast_version": mlcast_version(),
        "model_file": MODEL_FILE,
        "model_sha256": sha256_file(model_path),
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_config_sha256": None if config_path is None else sha256_file(config_path),
        "spec": spec.to_dict(),
    }
    _write_json(manifest_path, manifest)
    source_mb = checkpoint_path.stat().st_size / 1024**2
    bundle_mb = model_path.stat().st_size / 1024**2
    logger.info(f"Wrote bundle {out_dir} ({bundle_mb:.1f} MB weights, from a {source_mb:.1f} MB checkpoint)")
    return out_dir


def read_manifest(model_dir: str | Path) -> dict[str, Any]:
    """Read and check a bundle's ``manifest.json``, without loading the model.

    Parameters
    ----------
    model_dir : str or Path
        The bundle directory.

    Returns
    -------
    dict of str to Any
        The manifest.

    Raises
    ------
    ModelError
        If the directory or the manifest is missing, the manifest is not a JSON object,
        lacks a required key, or has a newer format than this builder understands.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ModelError(
            f"No model bundle at {model_dir}: the directory does not exist. Images carry the bundle at "
            f"{DEFAULT_MODEL_DIR}; MLCAST_MODEL_DIR (--model-dir) points elsewhere only for a bundle mounted at "
            "another path."
        )
    path = model_dir / MANIFEST_FILE
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ModelError(f"No model bundle at {model_dir}: {MANIFEST_FILE} is missing") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError(f"Corrupt model bundle at {model_dir}: cannot read {MANIFEST_FILE}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelError(f"Corrupt model bundle at {model_dir}: {MANIFEST_FILE} is not a JSON object")
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ModelError(f"Corrupt model bundle at {model_dir}: {MANIFEST_FILE} lacks {', '.join(missing)}")
    if not isinstance(manifest["format_version"], int) or manifest["format_version"] > MANIFEST_FORMAT_VERSION:
        raise ModelError(
            f"Model bundle at {model_dir} has manifest format {manifest['format_version']!r}; this builder "
            f"({__version__}) reads up to {MANIFEST_FORMAT_VERSION}. Rebuild the image with a matching builder."
        )
    return manifest


def load_bundle(
    model_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, InferenceSpec, dict[str, Any]]:
    """Load a bundle's model, spec and manifest, checking that they are intact and agree.

    Parameters
    ----------
    model_dir : str or Path
        The bundle directory.
    device : str or torch.device, optional
        Device to load the model onto.

    Returns
    -------
    model : NowcastLightningModule
        The model, on ``device`` and in eval mode.
    spec : InferenceSpec
        The spec embedded in the checkpoint.
    manifest : dict of str to Any
        The manifest.

    Raises
    ------
    ModelError
        If anything is missing, the checkpoint's sha256 differs from the manifest's, the
        checkpoint cannot be loaded, or its spec differs from the manifest's.
    """
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    model_path = model_dir / MODEL_FILE
    if not model_path.is_file():
        raise ModelError(f"Corrupt model bundle at {model_dir}: {model_path.name} is missing")

    actual = sha256_file(model_path)
    if actual != manifest["model_sha256"]:
        raise ModelError(
            f"Corrupt model bundle at {model_dir}: {model_path.name} has sha256 {actual}, but the manifest "
            f"records {manifest['model_sha256']}. The file was changed or truncated after the bundle was made."
        )

    try:
        model, spec = load_model(model_path, device=str(device))
    except Exception as exc:  # noqa: BLE001 - any failure to load the bundle is a broken image
        raise ModelError(
            f"Corrupt model bundle at {model_dir}: cannot load {model_path.name}: {type(exc).__name__}: {exc}"
        ) from exc
    if _json_round_trip(spec.to_dict()) != manifest["spec"]:
        raise ModelError(
            f"Corrupt model bundle at {model_dir}: the inference spec in {model_path.name} differs from the one "
            f"in {MANIFEST_FILE}"
        )
    return model, spec, manifest


def manifest_summary(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Render a manifest for the log, one entry per line.

    Parameters
    ----------
    manifest : mapping
        As returned by :func:`read_manifest`.

    Returns
    -------
    dict of str to str
        The model's identity and provenance, then the spec fields a deployer cares
        about. ``network_config`` is left out: it is long, and the sha256 already
        identifies the model.
    """
    spec = manifest.get("spec", {})

    def show(value: Any, unset: str = "unset") -> str:
        if value is None:
            return unset
        if isinstance(value, list | tuple):
            return ",".join(str(v) for v in value)
        return str(value)

    return {
        "name": show(manifest.get("name")),
        "model_version": show(manifest.get("model_version")),
        "model_sha256": show(manifest.get("model_sha256")),
        "created": show(manifest.get("created")),
        "builder_version": show(manifest.get("builder_version")),
        "mlcast_version": show(manifest.get("mlcast_version")),
        "source_checkpoint": show(manifest.get("source_checkpoint")),
        "standard_names": show(spec.get("standard_names")),
        "units": show(spec.get("units")),
        "input_steps": show(spec.get("input_steps")),
        "forecast_steps": show(spec.get("forecast_steps")),
        "timestep": show(spec.get("timestep"), "unknown"),
        "default_ensemble_size": show(spec.get("default_ensemble_size")),
        "spatial_divisor": show(spec.get("spatial_divisor")),
    }


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    """Create the output directory, refusing to mix a new bundle into other files."""
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigError(f"Bundle output {out_dir} exists and is not a directory")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise ConfigError(f"Bundle output directory {out_dir} is not empty; choose another or pass overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def _stripped_checkpoint(model: torch.nn.Module, spec: InferenceSpec) -> dict[str, Any]:
    """Build the bundle's checkpoint dict: weights, hyperparameters and the spec only."""
    checkpoint: dict[str, Any] = {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "hyper_parameters": dict(getattr(model, "hparams", {})),
        "pytorch-lightning_version": pl.__version__,
    }
    save_inference_metadata(checkpoint, spec)
    return checkpoint


def _verify(path: Path, reference: torch.nn.Module, spec: InferenceSpec) -> None:
    """Check that a written checkpoint reloads to the same model and runs.

    Raises
    ------
    ModelError
        Naming which check failed.
    """
    try:
        reloaded, reloaded_spec = load_model(path, device="cpu")
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Bundle verification failed: strict reload: {type(exc).__name__}: {exc}") from exc
    if reloaded_spec != spec:
        raise ModelError("Bundle verification failed: the reloaded inference spec differs from the source's")
    expected = reference.state_dict()
    got = reloaded.state_dict()
    if expected.keys() != got.keys() or any(not torch.equal(expected[k].cpu(), got[k]) for k in expected):
        raise ModelError("Bundle verification failed: the reloaded weights differ from the source's")

    channels = len(spec.standard_names)
    d = spec.spatial_divisor
    shape = (1, spec.input_steps, channels, d, d)
    x = torch.rand(shape, generator=torch.Generator().manual_seed(_VERIFY_SEED)) * 2 - 1
    try:
        with torch.no_grad():
            out = reloaded(x, forecast_steps=spec.forecast_steps, ensemble_size=1)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(
            f"Bundle verification failed: forward pass on input {tuple(shape)}: {type(exc).__name__}: {exc}"
        ) from exc
    expected_shape = (1, spec.forecast_steps, channels, d, d)
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected_shape:
        got_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
        raise ModelError(
            f"Bundle verification failed: forward pass on input {shape} gave {got_shape}, expected {expected_shape}"
        )
    if not torch.isfinite(out).all():
        raise ModelError(f"Bundle verification failed: forward pass on input {shape} gave non-finite values")
    logger.info(f"Verified: strict reload, identical weights, forward pass {shape} -> {expected_shape}")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON atomically, so a crash never leaves a half-written manifest."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(partial, path)


def _json_round_trip(data: Any) -> Any:
    """Return ``data`` as it reads back from JSON (tuples become lists, and so on)."""
    return json.loads(json.dumps(data))
