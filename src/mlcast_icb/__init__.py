"""Build ready-to-run inference container images from trained mlcast nowcasting models."""

from importlib.metadata import PackageNotFoundError, version

#: Prefix of the environment variables that mirror the options of ``run`` and ``serve``
#: (``--some-option`` is also ``MLCAST_SOME_OPTION``; see :mod:`mlcast_icb.envargs`).
ENV_PREFIX = "MLCAST"

try:
    __version__ = version("mlcast-inference-container-builder")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled source tree
    __version__ = "0.0.0"
