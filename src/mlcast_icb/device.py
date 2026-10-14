"""Resolving the ``--device`` setting (``MLCAST_DEVICE``) to a torch device, failing hard when CUDA was required."""

from __future__ import annotations

import torch
from loguru import logger

from mlcast_icb.errors import DeviceError

#: Accepted values of the ``--device`` setting.
DEVICES = ("auto", "cuda", "cpu")


def select_device(requested: str) -> torch.device:
    """Resolve a ``--device`` setting to a concrete torch device.

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
                "The device is cuda but no CUDA device is visible to this process. Check that the container was "
                "started with GPU access (e.g. `docker run --gpus all`) and that the NVIDIA driver is present, "
                f"or set MLCAST_DEVICE=cpu (--device cpu). torch {torch.__version__}, built for CUDA "
                f"{torch.version.cuda}."
            )
        device = torch.device("cuda")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if available else "cpu")
        if not available:
            logger.warning("The device is auto and no CUDA device is visible; running on CPU")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Using CUDA device {props.name} ({props.total_memory / 1024**3:.1f} GiB)")
    else:
        logger.info("Using CPU")
    return device
