"""Model images: the build context, the docker commands, missing Docker, and (opt-in) real builds."""

from __future__ import annotations

import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from loguru import logger
from run_testdata import write_input

import mlcast_icb.build as build_module
from mlcast_icb import __version__
from mlcast_icb.__main__ import main
from mlcast_icb.build import (
    BASE_DOCKERFILE,
    CONTEXT_BUNDLE_DIR,
    MODEL_DOCKERFILE,
    RUNTIME_IMAGE_REPOSITORY,
    TORCH_VARIANTS,
    DockerBuildError,
    build_base_command,
    build_image,
    default_base_image,
    docker_build_command,
    image_tag_version,
    lock_mlcast,
    model_build_args,
    source_checkout,
    write_context,
)
from mlcast_icb.bundle import MANIFEST_FILE, MODEL_FILE, create_bundle
from mlcast_icb.errors import ConfigError

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle_dir(tiny_trained_model, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return create_bundle(
        tiny_trained_model.checkpoint_path,
        tmp_path_factory.mktemp("build_bundle") / "model",
        config_path=tiny_trained_model.config_path,
        timestep="PT5M",
        name="tiny",
        model_version="0.1",
    )


@pytest.fixture(scope="module")
def checkpoint_context(tiny_trained_model, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    context = tmp_path_factory.mktemp("build_context") / "context"
    manifest = write_context(
        context,
        checkpoint=tiny_trained_model.checkpoint_path,
        config=tiny_trained_model.config_path,
        timestep="PT5M",
        name="tiny",
        model_version="2.0",
    )
    return context, manifest


class FakeDocker:
    """Stands in for ``subprocess.run`` of the docker CLI, recording every command."""

    def __init__(self, daemon_up: bool = True, build_returncode: int = 0) -> None:
        self.daemon_up = daemon_up
        self.build_returncode = build_returncode
        #: Labels of the images in the local store, by reference; others are not present.
        self.local_images: dict[str, dict[str, str]] = {}
        self.calls: list[list[str]] = []
        self.context_files: list[str] = []

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[-1] in self.local_images:
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self.local_images[argv[-1]]) + "\n")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=f"Error: No such image: {argv[-1]}\n")
        if argv[:2] == ["docker", "version"]:
            if self.daemon_up:
                return subprocess.CompletedProcess(argv, 0, stdout="29.0.0\n", stderr="")
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock\n"
            )
        if argv[:3] == ["docker", "buildx", "version"]:
            return subprocess.CompletedProcess(argv, 0, stdout="v0.20\n", stderr="")
        # A build: record what the context held while docker could read it.
        context = Path(argv[-1])
        self.context_files = sorted(str(p.relative_to(context)) for p in context.rglob("*"))
        return subprocess.CompletedProcess(argv, self.build_returncode)


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(build_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(build_module.subprocess, "run", fake)
    return fake


# --- naming ---------------------------------------------------------------------------


def test_image_tag_version_replaces_what_a_tag_cannot_hold() -> None:
    assert image_tag_version("0.1.0") == "0.1.0"
    assert image_tag_version("0.1.dev4+g1a2b3c.d20261014") == "0.1.dev4-g1a2b3c.d20261014"


def test_default_base_image_is_per_builder_version_and_torch_variant() -> None:
    assert default_base_image("gpu-cu126", "0.2.0") == f"{RUNTIME_IMAGE_REPOSITORY}:0.2.0-gpu-cu126"
    assert default_base_image() == f"{RUNTIME_IMAGE_REPOSITORY}:{image_tag_version(__version__)}-cpu"
    assert re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", default_base_image().split(":")[-1])


def test_unknown_torch_variant_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="gpu-cu126"):
        default_base_image("cuda")


def test_base_dockerfile_has_a_stage_per_torch_variant() -> None:
    dockerfile = (REPO / BASE_DOCKERFILE).read_text()
    for variant in TORCH_VARIANTS:
        assert re.search(rf"^FROM \S+ AS os-{variant}$", dockerfile, re.MULTILINE), variant
    assert re.search(r"^FROM os-\$\{TORCH_EXTRA\} AS runtime$", dockerfile, re.MULTILINE)


# --- the build context ----------------------------------------------------------------


def test_context_holds_the_dockerfile_and_a_verified_bundle(checkpoint_context: tuple[Path, dict]) -> None:
    context, manifest = checkpoint_context
    files = sorted(str(p.relative_to(context)) for p in context.rglob("*"))
    assert files == [
        MODEL_DOCKERFILE,
        CONTEXT_BUNDLE_DIR,
        f"{CONTEXT_BUNDLE_DIR}/{MANIFEST_FILE}",
        f"{CONTEXT_BUNDLE_DIR}/{MODEL_FILE}",
    ]
    assert (context / MODEL_DOCKERFILE).read_text() == (REPO / "docker" / MODEL_DOCKERFILE).read_text()
    assert json.loads((context / CONTEXT_BUNDLE_DIR / MANIFEST_FILE).read_text()) == manifest
    assert (manifest["name"], manifest["model_version"], manifest["spec"]["timestep"]) == ("tiny", "2.0", "PT5M")


def test_model_dockerfile_adds_one_layer_with_the_bundle(checkpoint_context: tuple[Path, dict]) -> None:
    context, _ = checkpoint_context
    lines = [line for line in (context / MODEL_DOCKERFILE).read_text().splitlines() if line and line[0] != "#"]
    instructions = [line.split()[0] for line in lines if not line.startswith(" ")]
    assert instructions.count("FROM") == 1
    assert "FROM ${BASE_IMAGE}" in lines
    # LABEL and ARG add no layer; COPY is the only one.
    assert [i for i in instructions if i in {"RUN", "COPY", "ADD"}] == ["COPY"]
    assert f"COPY --chown=1000:1000 {CONTEXT_BUNDLE_DIR}/ /model/" in lines


def test_build_args_carry_the_manifest_into_every_label(checkpoint_context: tuple[Path, dict]) -> None:
    context, manifest = checkpoint_context
    args = model_build_args(manifest, "base:1")
    assert args == {
        "BASE_IMAGE": "base:1",
        "MODEL_NAME": "tiny",
        "MODEL_VERSION": "2.0",
        "MODEL_SHA256": manifest["model_sha256"],
        "MODEL_CREATED": manifest["created"],
        "MODEL_MLCAST_VERSION": manifest["mlcast_version"],
        "MODEL_BUILDER_VERSION": __version__,
    }
    dockerfile = (context / MODEL_DOCKERFILE).read_text()
    for name in args:
        assert re.search(rf"^ARG {name}\b", dockerfile, re.MULTILINE), name
        assert f"${{{name}}}" in dockerfile, f"{name} is not used in a label"


def test_unset_manifest_fields_become_empty_build_args() -> None:
    args = model_build_args({"name": None, "model_sha256": "ab"}, "base:1")
    assert (args["MODEL_NAME"], args["MODEL_VERSION"], args["MODEL_SHA256"]) == ("", "", "ab")


def test_context_from_an_existing_bundle_copies_it(bundle_dir: Path, tmp_path: Path) -> None:
    manifest = write_context(tmp_path / "ctx", bundle=bundle_dir)
    assert manifest["name"] == "tiny"
    for name in (MODEL_FILE, MANIFEST_FILE):
        assert (tmp_path / "ctx" / CONTEXT_BUNDLE_DIR / name).read_bytes() == (bundle_dir / name).read_bytes()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "either a checkpoint or an existing bundle"),
        ({"checkpoint": "m.ckpt", "bundle": "b"}, "either a checkpoint or an existing bundle"),
        ({"bundle": "b", "name": "x"}, "already records them"),
    ],
)
def test_context_options_are_checked(tmp_path: Path, kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        write_context(tmp_path / "ctx", **kwargs)


def test_context_must_be_new_or_empty(bundle_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "ctx").mkdir()
    (tmp_path / "ctx" / "stray").write_text("x")
    with pytest.raises(ConfigError, match="new or empty"):
        write_context(tmp_path / "ctx", bundle=bundle_dir)


# --- docker commands ------------------------------------------------------------------


def test_docker_build_command() -> None:
    assert docker_build_command("/ctx", "reg/m:1", {"A": "1", "B": "x y"}) == [
        "docker",
        "build",
        "--file",
        "/ctx/Dockerfile.model",
        "--tag",
        "reg/m:1",
        "--build-arg",
        "A=1",
        "--build-arg",
        "B=x y",
        "/ctx",
    ]


def test_push_and_platform_use_buildx() -> None:
    command = docker_build_command("/ctx", "reg/m:1", {}, platform="linux/amd64", push=True)
    assert command[:4] == ["docker", "buildx", "build", "--push"]
    assert command[command.index("--platform") + 1] == "linux/amd64"


def test_build_runs_docker_on_a_temporary_context(bundle_dir: Path, fake_docker: FakeDocker) -> None:
    command = build_image("reg/m:1", bundle=bundle_dir, base_image="base:1", platform="linux/amd64")
    assert fake_docker.calls[0][:2] == ["docker", "version"]
    assert fake_docker.calls[-1] == command
    assert command[:2] == ["docker", "build"]
    assert "BASE_IMAGE=base:1" in command and "MODEL_NAME=tiny" in command
    assert fake_docker.context_files == [
        MODEL_DOCKERFILE,
        CONTEXT_BUNDLE_DIR,
        f"{CONTEXT_BUNDLE_DIR}/{MANIFEST_FILE}",
        f"{CONTEXT_BUNDLE_DIR}/{MODEL_FILE}",
    ]
    assert not Path(command[-1]).exists(), "the temporary context is removed after the build"


def test_build_defaults_to_this_builders_base_image(bundle_dir: Path, fake_docker: FakeDocker) -> None:
    command = build_image("reg/m:1", bundle=bundle_dir, torch_variant="gpu-cu128")
    assert f"BASE_IMAGE={default_base_image('gpu-cu128')}" in command


@pytest.fixture
def warnings_logged() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    yield messages
    logger.remove(handler)


def test_build_warns_when_the_local_base_image_has_another_mlcast(
    bundle_dir: Path, fake_docker: FakeDocker, warnings_logged: list[str]
) -> None:
    fake_docker.local_images["base:1"] = {"io.github.mlcast-community.mlcast.version": "0.0.1"}
    command = build_image("reg/m:1", bundle=bundle_dir, base_image="base:1")
    assert ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", "base:1"] in fake_docker.calls
    assert fake_docker.calls[-1] == command, "a mismatch warns, it does not stop the build"
    assert len(warnings_logged) == 1
    assert "mlcast 0.0.1" in warnings_logged[0] and "base:1" in warnings_logged[0]


@pytest.mark.parametrize("labels", [None, {}, {"io.github.mlcast-community.mlcast.version": "unknown"}, "same"])
def test_build_does_not_warn_without_a_mismatch(
    labels, bundle_dir: Path, fake_docker: FakeDocker, warnings_logged: list[str]
) -> None:
    # Given a base image that is not local (None), has no mlcast label, an unknown one, or the bundle's
    if labels == "same":
        manifest = json.loads((bundle_dir / MANIFEST_FILE).read_text())
        labels = {"io.github.mlcast-community.mlcast.version": manifest["mlcast_version"]}
    if labels is not None:
        fake_docker.local_images["base:1"] = labels
    build_image("reg/m:1", bundle=bundle_dir, base_image="base:1")
    assert warnings_logged == []


def test_push_needs_buildx_and_pushes(bundle_dir: Path, fake_docker: FakeDocker) -> None:
    command = build_image("reg/m:1", bundle=bundle_dir, base_image="base:1", push=True)
    assert ["docker", "buildx", "version"] in fake_docker.calls
    assert command[:4] == ["docker", "buildx", "build", "--push"]


def test_failed_build_exits_1(bundle_dir: Path, fake_docker: FakeDocker) -> None:
    fake_docker.build_returncode = 17
    with pytest.raises(DockerBuildError, match="exit code 17") as excinfo:
        build_image("reg/m:1", bundle=bundle_dir, base_image="base:1")
    assert excinfo.value.exit_code == 1


def test_context_only_writes_the_context_and_prints_the_command(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_docker(*args, **kwargs):
        raise AssertionError("--context-only must not run docker")

    monkeypatch.setattr(build_module.subprocess, "run", no_docker)
    monkeypatch.setattr(build_module.shutil, "which", lambda name: None)
    command = build_image("reg/m:1", bundle=bundle_dir, base_image="base:1", context_only=tmp_path / "ctx")
    assert capsys.readouterr().out.strip() == shlex.join(command)
    assert command[-1] == str(tmp_path / "ctx")
    assert (tmp_path / "ctx" / CONTEXT_BUNDLE_DIR / MODEL_FILE).is_file()


def test_missing_docker_suggests_context_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_module.shutil, "which", lambda name: None)
    # Checked before bundling: the missing checkpoint is never reached.
    with pytest.raises(ConfigError, match="No docker command.*--context-only"):
        build_image("reg/m:1", checkpoint=tmp_path / "missing.ckpt", base_image="base:1")


def test_unreachable_daemon_suggests_context_only(bundle_dir: Path, fake_docker: FakeDocker) -> None:
    fake_docker.daemon_up = False
    with pytest.raises(ConfigError, match="daemon is not reachable.*Cannot connect.*--context-only"):
        build_image("reg/m:1", bundle=bundle_dir, base_image="base:1")
    assert len(fake_docker.calls) == 1


# --- the base image -------------------------------------------------------------------


def test_build_base_command_labels_the_locked_mlcast() -> None:
    root = source_checkout()
    assert root == REPO
    version, commit = lock_mlcast(REPO / "uv.lock")
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    command = build_base_command(REPO, "rt:1", "gpu-cu130", expected_version="v1.0.0")
    assert command[:2] == ["docker", "build"]
    assert command[command.index("--file") + 1] == str(REPO / BASE_DOCKERFILE)
    assert command[-1] == str(REPO)
    for arg in ("TORCH_EXTRA=gpu-cu130", f"MLCAST_VERSION={version}", f"MLCAST_COMMIT={commit}"):
        assert arg in command
    assert "EXPECTED_VERSION=v1.0.0" in command


def test_lockinfo_reads_the_same_mlcast_as_the_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    lockinfo = runpy.run_path(str(REPO / "docker" / "lockinfo.py"))
    url, commit, version = lockinfo["mlcast_source"](REPO / "uv.lock")
    assert (version, commit) == lock_mlcast(REPO / "uv.lock")
    assert url == "https://github.com/mlcast-community/mlcast"
    assert lockinfo["torch_index"](REPO / "pyproject.toml", "gpu-cu126") == "https://download.pytorch.org/whl/cu126"


def test_build_base_dry_run_prints_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["build-base", "--torch", "gpu-cu126", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "docker build" in printed and f"--tag {default_base_image('gpu-cu126')}" in printed


# --- command line ---------------------------------------------------------------------


def test_cli_build_context_only(bundle_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["build", "--bundle", str(bundle_dir), "--tag", "reg/m:1", "--context-only", str(tmp_path / "ctx")]
    assert main(args) == 0
    assert "docker build" in capsys.readouterr().out
    assert (tmp_path / "ctx" / MODEL_DOCKERFILE).is_file()


def test_cli_build_needs_a_source_and_a_tag(bundle_dir: Path) -> None:
    assert main(["build", "--tag", "reg/m:1"]) == 2
    assert main(["build", "--bundle", str(bundle_dir)]) == 2
    assert main(["build", "--bundle", str(bundle_dir), "--checkpoint", "m.ckpt", "--tag", "x"]) == 2


def test_cli_build_without_docker_exits_2(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(build_module.shutil, "which", lambda name: None)
    assert main(["build", "--bundle", str(bundle_dir), "--tag", "reg/m:1"]) == 2
    assert "--context-only" in capsys.readouterr().err


# --- real images (opt-in) -------------------------------------------------------------

#: Set to a cpu base image already built from this checkout (with the labels
#: ``build_base_command`` passes) to test on it instead of building one. CI builds it
#: beforehand with buildx and the GitHub Actions cache; the tests leave it in place.
PREBUILT_BASE_ENV = "ICB_TEST_BASE_IMAGE"
PREBUILT_BASE_IMAGE = os.environ.get(PREBUILT_BASE_ENV)
TEST_BASE_IMAGE = PREBUILT_BASE_IMAGE or "mlcast-icb-test-runtime:cpu"
TEST_MODEL_IMAGE = "mlcast-icb-test-model:tiny"


def docker(*args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, text=True, capture_output=True, **kwargs)


@pytest.fixture(scope="module")
def model_image(bundle_dir: Path) -> Iterator[str]:
    """The cpu runtime base built from this checkout, and the tiny model on top of it.

    Both tags are removed afterwards; the base's layers stay in Docker's build cache, so
    the next run rebuilds it in seconds unless uv.lock or the code changed. With
    ``ICB_TEST_BASE_IMAGE`` set, that base image is used as is and kept.
    """
    if shutil.which("docker") is None:
        pytest.fail("--run-docker needs the docker command")
    built = [TEST_MODEL_IMAGE]
    try:
        if PREBUILT_BASE_IMAGE is None:
            built.append(TEST_BASE_IMAGE)
            subprocess.run(build_base_command(REPO, TEST_BASE_IMAGE, "cpu"), check=True)
        build_image(TEST_MODEL_IMAGE, bundle=bundle_dir, base_image=TEST_BASE_IMAGE)
        yield TEST_MODEL_IMAGE
    finally:
        docker("image", "rm", *built, check=False)


@pytest.mark.docker
def test_image_labels_record_model_and_runtime(model_image: str, bundle_dir: Path) -> None:
    labels = json.loads(docker("image", "inspect", "--format", "{{json .Config.Labels}}", model_image).stdout)
    manifest = json.loads((bundle_dir / MANIFEST_FILE).read_text())
    version, commit = lock_mlcast(REPO / "uv.lock")
    assert labels["io.github.mlcast-community.model.sha256"] == manifest["model_sha256"]
    assert labels["io.github.mlcast-community.model.name"] == "tiny"
    assert labels["io.github.mlcast-community.mlcast.commit"] == commit
    assert labels["io.github.mlcast-community.mlcast.version"] == version
    assert labels["io.github.mlcast-community.torch.variant"] == "cpu"
    assert labels["org.opencontainers.image.base.name"] == TEST_BASE_IMAGE


@pytest.mark.docker
def test_image_info_verifies_the_bundle(model_image: str, bundle_dir: Path) -> None:
    result = docker("run", "--rm", model_image, "info", "--verify")
    assert json.loads(result.stdout) == json.loads((bundle_dir / MANIFEST_FILE).read_text())


@pytest.mark.docker
def test_image_without_a_subcommand_exits_2(model_image: str) -> None:
    result = docker("run", "--rm", model_image, check=False)
    assert result.returncode == 2
    assert "run" in result.stderr and "serve" in result.stderr


@pytest.mark.docker
def test_image_run_matches_a_local_run(model_image: str, bundle_dir: Path, tmp_path: Path) -> None:
    from mlcast_icb.run import main as run_main

    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o777)  # the container runs as uid 1000
    write_input(data / "in.zarr")
    # Configured by MLCAST_* variables and by options, mixed, as a container would be.
    env = {"MLCAST_INPUT_ZARR": "/data/in.zarr", "MLCAST_ENSEMBLE_SIZE": "2", "MLCAST_SEED": "7"}
    env_args = [arg for key, value in env.items() for arg in ("-e", f"{key}={value}")]
    result = docker(
        "run", "--rm", "-v", f"{data}:/data", *env_args, model_image, "run", "--output-zarr", "/data/out.zarr",
        check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "MLCAST_MODEL_DIR" in result.stderr and "/model (env)" in result.stderr  # from the image's ENV

    local_env = env | {
        "MLCAST_INPUT_ZARR": str(data / "in.zarr"),
        "MLCAST_OUTPUT_ZARR": str(tmp_path / "local.zarr"),
        "MLCAST_MODEL_DIR": str(bundle_dir),
        "MLCAST_DEVICE": "cpu",
    }
    assert run_main([], local_env) == 0
    in_docker = xr.open_zarr(data / "out.zarr")
    local = xr.open_zarr(tmp_path / "local.zarr")
    assert in_docker.rr.dims == ("ensemble_member", "time", "y", "x")
    np.testing.assert_allclose(in_docker.rr.values, local.rr.values, rtol=1e-5, atol=1e-6)
