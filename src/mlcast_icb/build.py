"""Build model images: a model bundle as one layer on top of the runtime base image.

``mlcast-icb build`` turns a checkpoint (or an existing bundle) into an image:

1. make the bundle in a fresh build context with :func:`mlcast_icb.bundle.create_bundle`,
   which also verifies the model (strict reload, one CPU forward pass), so a broken model
   fails the build rather than the deployment;
2. copy ``docker/Dockerfile.model`` next to it;
3. run ``docker build`` (``docker buildx build --push`` with ``push``), passing the
   manifest's identity as build args, which the Dockerfile turns into image labels.
   If the base image is in the local image store and its mlcast version label differs
   from the bundle's, a warning is logged first.

With ``context_only`` it stops after step 2 and prints the build command, for CI systems
without a Docker daemon (kaniko, buildah).

The runtime base image is built once per builder release and torch variant from
``docker/Dockerfile.base`` (``mlcast-icb build-base`` from a source checkout) and named by
:func:`default_base_image`.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from loguru import logger

from mlcast_icb import __version__
from mlcast_icb.errors import ConfigError, MlcastIcbError

#: Registry and repository of the published runtime base images, tagged
#: ``<builder version>-<torch variant>`` (see :func:`default_base_image`).
RUNTIME_IMAGE_REPOSITORY = "ghcr.io/mlcast-community/mlcast-inference-runtime"

#: The torch variants a base image can be built for: the project's torch extras.
TORCH_VARIANTS = ("cpu", "gpu-cu126", "gpu-cu128", "gpu-cu130")

#: The variant ``build`` and ``build-base`` use unless told otherwise. CPU, because it
#: runs on any machine and platform; GPU deployments choose their CUDA version explicitly.
DEFAULT_TORCH_VARIANT = "cpu"

#: Prefix of the builder's own image labels.
LABEL_PREFIX = "io.github.mlcast-community"

#: File names inside a model image's build context.
MODEL_DOCKERFILE = "Dockerfile.model"
CONTEXT_BUNDLE_DIR = "bundle"

#: The base image's Dockerfile, relative to a source checkout.
BASE_DOCKERFILE = Path("docker") / "Dockerfile.base"

#: Characters a Docker tag may contain; anything else in a version becomes ``-``.
_TAG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


class DockerBuildError(MlcastIcbError):
    """``docker build`` ran and failed; its own output says why.

    Attributes
    ----------
    exit_code : int
        Always 1: the failure is in the image build, which Docker has already reported.
    """

    exit_code = 1


def image_tag_version(version: str = __version__) -> str:
    """Return a package version as a valid Docker tag component.

    Parameters
    ----------
    version : str, optional
        A PEP 440 version. Defaults to this builder's.

    Returns
    -------
    str
        The version with every character a tag cannot hold (the ``+`` of a local
        version, mainly) replaced by ``-``: ``0.1.dev4+g1a2b3c`` becomes ``0.1.dev4-g1a2b3c``.
    """
    return _TAG_UNSAFE.sub("-", version)


def default_base_image(torch_variant: str = DEFAULT_TORCH_VARIANT, version: str = __version__) -> str:
    """Return the runtime base image this builder version builds model images on.

    Parameters
    ----------
    torch_variant : str, optional
        One of :data:`TORCH_VARIANTS`.
    version : str, optional
        The builder version. Defaults to this builder's: a bundle is only guaranteed to
        load in the runtime of the builder (and so the mlcast) that made it.

    Returns
    -------
    str
        ``<RUNTIME_IMAGE_REPOSITORY>:<version>-<torch_variant>``.
    """
    _check_torch_variant(torch_variant)
    return f"{RUNTIME_IMAGE_REPOSITORY}:{image_tag_version(version)}-{torch_variant}"


def source_checkout() -> Path | None:
    """Return the source checkout this package runs from, or ``None`` when installed.

    Returns
    -------
    Path or None
        The directory holding ``docker/Dockerfile.base`` and ``uv.lock``, found from
        this file's location (``src/mlcast_icb/build.py``), as in an editable install.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / BASE_DOCKERFILE).is_file() and (root / "uv.lock").is_file():
        return root
    return None


def model_dockerfile() -> str:
    """Return the text of the model image's Dockerfile.

    Returns
    -------
    str
        ``docker/Dockerfile.model``, from the package (where the wheel carries it as
        ``mlcast_icb/docker/Dockerfile.model``) or from the source checkout.

    Raises
    ------
    ConfigError
        If neither has it: the installation is incomplete.
    """
    packaged = files("mlcast_icb").joinpath("docker", MODEL_DOCKERFILE)
    if packaged.is_file():
        return packaged.read_text()
    root = source_checkout()
    if root is not None and (root / "docker" / MODEL_DOCKERFILE).is_file():
        return (root / "docker" / MODEL_DOCKERFILE).read_text()
    raise ConfigError(f"This installation of the builder ({__version__}) lacks {MODEL_DOCKERFILE}; reinstall it")


def model_build_args(manifest: Mapping[str, Any], base_image: str) -> dict[str, str]:
    """Return the model Dockerfile's build args, read from a bundle's manifest.

    Parameters
    ----------
    manifest : mapping
        The bundle's ``manifest.json``.
    base_image : str
        The runtime base image to build on.

    Returns
    -------
    dict of str to str
        ``BASE_IMAGE`` and the ``MODEL_*`` args of ``docker/Dockerfile.model``, which
        become the image's labels. Unset manifest fields become empty strings.
    """

    def text(key: str) -> str:
        value = manifest.get(key)
        return "" if value is None else str(value)

    return {
        "BASE_IMAGE": base_image,
        "MODEL_NAME": text("name"),
        "MODEL_VERSION": text("model_version"),
        "MODEL_SHA256": text("model_sha256"),
        "MODEL_CREATED": text("created"),
        "MODEL_MLCAST_VERSION": text("mlcast_version"),
        "MODEL_BUILDER_VERSION": text("builder_version"),
    }


def write_context(
    context_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    bundle: str | Path | None = None,
    config: str | Path | None = None,
    timestep: str | None = None,
    name: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Write a model image's build context: ``Dockerfile.model`` and ``bundle/``.

    Parameters
    ----------
    context_dir : str or Path
        The directory to write; it must not exist or be empty.
    checkpoint : str or Path, optional
        A trained (or converted) checkpoint to bundle, verifying the model on the way.
    bundle : str or Path, optional
        An existing bundle to use instead of ``checkpoint``. It is checked (checksum,
        strict load) before it is copied.
    config, timestep, name, model_version : optional
        As for :func:`mlcast_icb.bundle.create_bundle`; with ``checkpoint`` only.

    Returns
    -------
    dict of str to Any
        The bundle's manifest.

    Raises
    ------
    ConfigError
        If not exactly one of ``checkpoint`` and ``bundle`` is given, if bundle options
        come with ``bundle``, or if ``context_dir`` is not empty.
    ModelError
        If the model cannot be bundled or the bundle is broken.
    """
    from mlcast_icb.bundle import MANIFEST_FILE, MODEL_FILE, create_bundle, load_bundle, read_manifest

    if (checkpoint is None) == (bundle is None):
        raise ConfigError("Give either a checkpoint or an existing bundle to build from, not both or neither")
    if bundle is not None and any(v is not None for v in (config, timestep, name, model_version)):
        raise ConfigError(
            "--config, --timestep, --name and --model-version apply when bundling a checkpoint; an existing bundle "
            "already records them"
        )
    context_dir = Path(context_dir)
    if context_dir.exists() and (not context_dir.is_dir() or any(context_dir.iterdir())):
        raise ConfigError(f"Build context {context_dir} must be a new or empty directory")
    context_dir.mkdir(parents=True, exist_ok=True)

    target = context_dir / CONTEXT_BUNDLE_DIR
    if checkpoint is not None:
        create_bundle(checkpoint, target, config_path=config, timestep=timestep, name=name, model_version=model_version)
    else:
        load_bundle(bundle, "cpu")  # checksum, strict load and spec agreement
        target.mkdir()
        for file in (MODEL_FILE, MANIFEST_FILE):
            shutil.copy2(Path(bundle) / file, target / file)
        logger.info(f"Copied verified bundle {bundle}")
    (context_dir / MODEL_DOCKERFILE).write_text(model_dockerfile())
    return read_manifest(target)


def docker_build_command(
    context_dir: str | Path,
    tag: str,
    build_args: Mapping[str, str],
    *,
    platform: str | None = None,
    push: bool = False,
    dockerfile: str = MODEL_DOCKERFILE,
) -> list[str]:
    """Return the command that builds an image from a context.

    Parameters
    ----------
    context_dir : str or Path
        The build context.
    tag : str
        The image reference to tag (and push) the image as.
    build_args : mapping of str to str
        ``--build-arg`` values.
    platform : str, optional
        Target platform, e.g. ``linux/amd64``. Defaults to the Docker host's.
    push : bool, optional
        Push the image instead of loading it into the local image store; this uses
        ``docker buildx build --push``.
    dockerfile : str, optional
        The Dockerfile, relative to ``context_dir``.

    Returns
    -------
    list of str
        The argv.
    """
    command = ["docker", "buildx", "build", "--push"] if push else ["docker", "build"]
    command += ["--file", str(Path(context_dir) / dockerfile), "--tag", tag]
    if platform:
        command += ["--platform", platform]
    for key, value in build_args.items():
        command += ["--build-arg", f"{key}={value}"]
    command.append(str(context_dir))
    return command


def check_docker(push: bool = False) -> None:
    """Check that a Docker client and daemon (and buildx, to push) are available.

    Parameters
    ----------
    push : bool, optional
        Also require ``docker buildx``.

    Raises
    ------
    ConfigError
        Naming what is missing and suggesting ``--context-only``.
    """
    hint = "Use --context-only DIR to write the build context and command for another builder (kaniko, buildah)."
    if shutil.which("docker") is None:
        raise ConfigError(f"No docker command on PATH, so the image cannot be built here. {hint}")
    result = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ConfigError(
            "The Docker daemon is not reachable" + (f" ({detail[-1]})" if detail else "") + f". Start it, or: {hint}"
        )
    if push:
        result = subprocess.run(["docker", "buildx", "version"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ConfigError(f"--push needs docker buildx, which is not installed. {hint}")


def run_docker(command: Sequence[str]) -> None:
    """Run a docker command, its output streaming to this process's.

    Parameters
    ----------
    command : sequence of str
        The argv.

    Raises
    ------
    DockerBuildError
        If it exits non-zero.
    """
    logger.info(f"Running: {shlex.join(command)}")
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        raise DockerBuildError(f"{command[0]} {command[1]} failed with exit code {result.returncode}; see its output")


def local_image_labels(image: str) -> dict[str, str] | None:
    """Return the labels of an image in the local Docker image store.

    Parameters
    ----------
    image : str
        The image reference.

    Returns
    -------
    dict of str to str, or None
        The image's labels (empty if it has none), or ``None`` if the image is not in
        the local store, or its labels cannot be read. Nothing is pulled.
    """
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        labels = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return None
    return labels if isinstance(labels, dict) else {}


def warn_on_mlcast_mismatch(manifest: Mapping[str, Any], base_image: str) -> None:
    """Warn when a bundle was made with another mlcast than the base image carries.

    Checked only when the base image is in the local image store; one that would be
    pulled (or a label that is missing) is skipped silently.

    Parameters
    ----------
    manifest : mapping
        The bundle's ``manifest.json``.
    base_image : str
        The runtime base image the model image is built on.
    """
    labels = local_image_labels(base_image)
    if labels is None:
        logger.debug(f"Base image {base_image} is not available locally; not comparing mlcast versions")
        return
    base_version = labels.get(f"{LABEL_PREFIX}.mlcast.version")
    bundle_version = manifest.get("mlcast_version")
    if not base_version or base_version == "unknown" or not bundle_version or base_version == bundle_version:
        return
    logger.warning(
        f"The bundle was made with mlcast {bundle_version}, but the base image {base_image} carries mlcast "
        f"{base_version}; the model may not load there. Bundle it with the builder release that matches the base "
        "image, or pick the base image of the builder that made the bundle."
    )


def build_image(
    tag: str,
    *,
    checkpoint: str | Path | None = None,
    bundle: str | Path | None = None,
    config: str | Path | None = None,
    timestep: str | None = None,
    name: str | None = None,
    model_version: str | None = None,
    base_image: str | None = None,
    torch_variant: str = DEFAULT_TORCH_VARIANT,
    platform: str | None = None,
    push: bool = False,
    context_only: str | Path | None = None,
) -> list[str]:
    """Build a model image, or write its build context.

    Parameters
    ----------
    tag : str
        The image reference, e.g. ``registry.example.com/nowcast/irene:1.0``.
    checkpoint, bundle, config, timestep, name, model_version : optional
        What to build from; see :func:`write_context`.
    base_image : str, optional
        The runtime base image. Defaults to :func:`default_base_image` for
        ``torch_variant``.
    torch_variant : str, optional
        Picks the default base image; ignored with ``base_image``.
    platform : str, optional
        Target platform, e.g. ``linux/amd64``.
    push : bool, optional
        Push the image (``docker buildx build --push``) instead of loading it locally.
    context_only : str or Path, optional
        Write the build context here and print the build command, without building.

    Returns
    -------
    list of str
        The build command (run, or to run).

    Raises
    ------
    ConfigError
        On bad options, or when Docker is unavailable (checked before the model is
        bundled, so a missing daemon fails fast).
    ModelError
        If the model cannot be bundled.
    DockerBuildError
        If ``docker build`` fails.
    """
    if base_image is None:
        base_image = default_base_image(torch_variant)
    if context_only is None:
        check_docker(push)

    def build_in(context: Path) -> list[str]:
        manifest = write_context(
            context,
            checkpoint=checkpoint,
            bundle=bundle,
            config=config,
            timestep=timestep,
            name=name,
            model_version=model_version,
        )
        command = docker_build_command(
            context, tag, model_build_args(manifest, base_image), platform=platform, push=push
        )
        if context_only is not None:
            logger.info(f"Wrote the build context {context}; build it with:")
            print(shlex.join(command))
        else:
            warn_on_mlcast_mismatch(manifest, base_image)
            run_docker(command)
            logger.info(f"Built {tag} on {base_image}" + (" and pushed it" if push else ""))
        return command

    if context_only is not None:
        return build_in(Path(context_only))
    with tempfile.TemporaryDirectory(prefix="mlcast-icb-build-") as tmp:
        return build_in(Path(tmp) / "context")


def lock_mlcast(lock_path: str | Path) -> tuple[str, str]:
    """Return the mlcast version and commit that a ``uv.lock`` pins.

    Parameters
    ----------
    lock_path : str or Path
        The lock file.

    Returns
    -------
    version : str
    commit : str
    """
    lock = tomllib.loads(Path(lock_path).read_text())
    package = next(p for p in lock["package"] if p["name"] == "mlcast")
    return package["version"], package["source"]["git"].partition("#")[2]


def build_base_command(
    root: str | Path,
    tag: str,
    torch_variant: str = DEFAULT_TORCH_VARIANT,
    *,
    platform: str | None = None,
    push: bool = False,
    builder_version: str = __version__,
    expected_version: str | None = None,
) -> list[str]:
    """Return the command that builds the runtime base image from a source checkout.

    Parameters
    ----------
    root : str or Path
        The checkout (the build context).
    tag : str
        The image reference.
    torch_variant : str, optional
        One of :data:`TORCH_VARIANTS`.
    platform : str, optional
        Target platform.
    push : bool, optional
        Push instead of loading locally.
    builder_version : str, optional
        Recorded in the image's labels.
    expected_version : str, optional
        Fail the build unless the package built from the checkout has this version
        (a release tag such as ``v0.1.0``).

    Returns
    -------
    list of str
        The argv.
    """
    _check_torch_variant(torch_variant)
    root = Path(root)
    mlcast_version, mlcast_commit = lock_mlcast(root / "uv.lock")
    build_args = {
        "TORCH_EXTRA": torch_variant,
        "BUILDER_VERSION": builder_version,
        "MLCAST_VERSION": mlcast_version,
        "MLCAST_COMMIT": mlcast_commit,
    }
    if expected_version:
        build_args["EXPECTED_VERSION"] = expected_version
    return docker_build_command(root, tag, build_args, platform=platform, push=push, dockerfile=str(BASE_DOCKERFILE))


def build_base(
    tag: str | None = None,
    torch_variant: str = DEFAULT_TORCH_VARIANT,
    *,
    platform: str | None = None,
    push: bool = False,
    expected_version: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Build the runtime base image from this package's source checkout.

    Parameters
    ----------
    tag : str, optional
        The image reference. Defaults to :func:`default_base_image`, the image
        ``mlcast-icb build`` uses by default, so a local base build is picked up as is.
    torch_variant, platform, push, expected_version : optional
        See :func:`build_base_command`.
    dry_run : bool, optional
        Print the command instead of running it.

    Returns
    -------
    list of str
        The build command.

    Raises
    ------
    ConfigError
        When not running from a source checkout, or Docker is unavailable.
    DockerBuildError
        If ``docker build`` fails.
    """
    root = source_checkout()
    if root is None:
        raise ConfigError(
            "build-base builds docker/Dockerfile.base from a source checkout (it needs the git history and uv.lock); "
            "run it from a clone of mlcast-inference-container-builder installed with `uv sync`"
        )
    tag = tag or default_base_image(torch_variant)
    command = build_base_command(
        root, tag, torch_variant, platform=platform, push=push, expected_version=expected_version
    )
    if dry_run:
        print(shlex.join(command))
        return command
    check_docker(push)
    run_docker(command)
    logger.info(f"Built the runtime base image {tag}" + (" and pushed it" if push else ""))
    return command


def _check_torch_variant(torch_variant: str) -> None:
    if torch_variant not in TORCH_VARIANTS:
        raise ConfigError(f"Unknown torch variant {torch_variant!r}; choose one of {', '.join(TORCH_VARIANTS)}")
