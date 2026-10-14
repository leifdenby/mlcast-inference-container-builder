"""Resolving the ``DEVICE`` setting to a torch device, failing hard when CUDA was required."""

from __future__ import annotations

import torch
from loguru import logger

from mlcast_icb.errors import DeviceError

#: Accepted values of the ``DEVICE`` setting.
DEVICES = ("auto", "cuda", "cpu")


def select_device(requested: str) -> torch.device:
    """Resolve a ``DEVICE`` setting to a concrete torch device.

    Parameters
    ----------
    requested : {'auto', 'cuda', 'cpu'}
        The requested device. ``'auto'`` prefers CUDA and falls back to CPU with a
        warning; ``'cuda'`` is a hard requirement.

    Returns
    -------
    torch.device
        The device to run on. Its name and memory are logged for CUDA.

    Raises
    ------
    DeviceError
        If ``'cuda'`` was requested but no CUDA device is visible. Silently running a
        nowcast on CPU would take minutes instead of seconds, which a pipeline would
        only notice as a missed deadline.
    """
    available = torch.cuda.is_available()
    if requested == "cuda":
        if not available:
            raise DeviceError(
                "DEVICE=cuda but no CUDA device is visible to this process. Check that the container was "
                "started with GPU access (e.g. `docker run --gpus all`) and that the NVIDIA driver is present, "
                f"or set DEVICE=cpu. torch {torch.__version__}, built for CUDA {torch.version.cuda}."
            )
        device = torch.device("cuda")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if available else "cpu")
        if not available:
            logger.warning("DEVICE=auto and no CUDA device is visible; running on CPU")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Using CUDA device {props.name} ({props.total_memory / 1024**3:.1f} GiB)")
    else:
        logger.info("Using CPU")
    return device
