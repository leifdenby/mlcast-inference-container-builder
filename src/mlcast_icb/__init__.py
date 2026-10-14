"""Build ready-to-run inference container images from trained mlcast nowcasting models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlcast-inference-container-builder")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled source tree
    __version__ = "0.0.0"
