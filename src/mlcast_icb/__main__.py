"""Command line: ``mlcast-icb {run,serve,bundle,info,build,build-base}``.

``run`` and ``serve`` are what an image executes. Each of their options ``--some-option``
can also be set with the environment variable ``MLCAST_SOME_OPTION`` (see
:mod:`mlcast_icb.options`), since a container is usually configured through its
environment; their arguments go to their own parsers. ``bundle``, ``info``, ``build`` and
``build-base`` are developer tools, configured through their options only: they read no
``MLCAST_*`` variable (except that ``info`` defaults to ``$MLCAST_MODEL_DIR``, the
image's bundle), and do not check them either.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from loguru import logger

from mlcast_icb import ENV_PREFIX, __version__
from mlcast_icb.errors import ConfigError, MlcastIcbError

#: Distribution name, as printed by ``--version``.
DISTRIBUTION = "mlcast-inference-container-builder"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="mlcast-icb",
        description="Run, serve or package a trained mlcast nowcasting model.",
    )
    parser.add_argument("--version", action="version", version=f"{DISTRIBUTION} {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Listed here for the help only: main() hands their arguments to their own parsers.
    sub.add_parser(
        "run",
        help="batch forecast, zarr in and out (options or MLCAST_* variables; see `run --help`)",
        add_help=False,
    )
    sub.add_parser(
        "serve",
        help="HTTP server, IRENE-compatible (options or MLCAST_* variables; see `serve --help`)",
        add_help=False,
    )

    bundle = sub.add_parser(
        "bundle",
        help="make a model bundle (model.ckpt + manifest.json) from a checkpoint",
        description="Strip a checkpoint to its weights, embed its inference spec, verify it with a strict reload "
        "and one CPU forward pass, and write model.ckpt and manifest.json.",
    )
    bundle.add_argument("--checkpoint", required=True, help="the trained (or converted) mlcast checkpoint")
    bundle.add_argument(
        "--config", help="the training config YAML; needed unless the checkpoint embeds an inference spec"
    )
    bundle.add_argument("--timestep", help="time between steps as an ISO 8601 duration, e.g. PT5M")
    bundle.add_argument("--out", required=True, help="output directory")
    bundle.add_argument("--name", help="model name, recorded in the manifest")
    bundle.add_argument("--model-version", help="model version, recorded in the manifest")
    bundle.add_argument("--overwrite", action="store_true", help="replace the contents of a non-empty --out")

    info = sub.add_parser("info", help="print a bundle's manifest")
    info.add_argument(
        "model_dir", nargs="?", help="bundle directory (default: $MLCAST_MODEL_DIR, or /model when that is unset)"
    )
    info.add_argument("--verify", action="store_true", help="also check the checksum and load the model on CPU")

    _add_build_parsers(sub)
    return parser


def _add_build_parsers(sub: argparse._SubParsersAction) -> None:
    """Add the ``build`` and ``build-base`` subcommands."""
    from mlcast_icb.build import DEFAULT_TORCH_VARIANT, RUNTIME_IMAGE_REPOSITORY, TORCH_VARIANTS

    build = sub.add_parser(
        "build",
        help="build a model image: a bundle on top of the runtime base image",
        description="Bundle a checkpoint (verifying it with a strict reload and one CPU forward pass), or take an "
        "existing bundle, and build an image that is the runtime base image plus the bundle at /model.",
    )
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="the trained (or converted) mlcast checkpoint to bundle")
    source.add_argument("--bundle", help="an existing bundle directory (from `mlcast-icb bundle`) to use as is")
    build.add_argument(
        "--config", help="the training config YAML; needed unless the checkpoint embeds an inference spec"
    )
    build.add_argument("--timestep", help="time between steps as an ISO 8601 duration, e.g. PT5M")
    build.add_argument("--name", help="model name, recorded in the manifest and the image labels")
    build.add_argument("--model-version", help="model version, recorded in the manifest and the image labels")
    build.add_argument("--tag", required=True, help="image reference to build, e.g. registry.example.com/irene:1.0")
    build.add_argument(
        "--base-image",
        help=f"runtime base image (default: {RUNTIME_IMAGE_REPOSITORY}:<builder version>-<torch variant>)",
    )
    build.add_argument(
        "--torch",
        choices=TORCH_VARIANTS,
        default=DEFAULT_TORCH_VARIANT,
        help="torch variant of the default base image (default: %(default)s)",
    )
    build.add_argument("--platform", help="target platform, e.g. linux/amd64 (default: the Docker host's)")
    build.add_argument("--push", action="store_true", help="push the image (docker buildx build --push)")
    build.add_argument(
        "--context-only",
        metavar="DIR",
        help="write the build context to DIR and print the build command, without building (no Docker needed)",
    )

    base = sub.add_parser(
        "build-base",
        help="build the runtime base image from this source checkout",
        description="Build docker/Dockerfile.base for one torch variant, tagged as `build` expects by default.",
    )
    base.add_argument(
        "--torch", choices=TORCH_VARIANTS, default=DEFAULT_TORCH_VARIANT, help="torch variant (default: %(default)s)"
    )
    base.add_argument("--tag", help="image reference (default: the one `build` uses for this variant)")
    base.add_argument("--platform", help="target platform, e.g. linux/amd64 (default: the Docker host's)")
    base.add_argument("--push", action="store_true", help="push the image (docker buildx build --push)")
    base.add_argument("--expected-version", help="fail unless the package builds as this version (a release tag)")
    base.add_argument("--dry-run", action="store_true", help="print the docker command instead of running it")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return a process exit code.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, without the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, a code from :mod:`mlcast_icb.errors` on a known failure, 2 for a
        usage error, 1 for anything unexpected.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        from mlcast_icb.run import main as run_main

        return run_main(argv[1:])
    if argv and argv[0] == "serve":
        from mlcast_icb.serve import main as serve_main

        return serve_main(argv[1:])

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return int(exc.code or 0)

    from mlcast_icb.logs import configure_logging

    configure_logging("INFO")
    if args.command is None:
        parser.print_help(sys.stderr)
        return ConfigError.exit_code
    try:
        if args.command == "bundle":
            return _bundle(args)
        if args.command == "build":
            return _build(args)
        if args.command == "build-base":
            return _build_base(args)
        return _info(args)
    except MlcastIcbError as exc:
        logger.error(f"{exc}")
        return exc.exit_code
    except Exception:
        logger.opt(exception=True).error("Unexpected failure")
        return 1


def _bundle(args: argparse.Namespace) -> int:
    """Create a bundle from the ``bundle`` subcommand's options."""
    from mlcast_icb.bundle import create_bundle, manifest_summary, read_manifest
    from mlcast_icb.logs import log_settings

    logger.info(f"{DISTRIBUTION} {__version__}: creating a model bundle")
    out = create_bundle(
        args.checkpoint,
        args.out,
        config_path=args.config,
        timestep=args.timestep,
        name=args.name,
        model_version=args.model_version,
        overwrite=args.overwrite,
    )
    log_settings(f"Model bundle {out}:", manifest_summary(read_manifest(out)))
    return 0


def _info(args: argparse.Namespace) -> int:
    """Print a bundle's manifest as JSON on stdout, optionally verifying the bundle."""
    from mlcast_icb.bundle import DEFAULT_MODEL_DIR, load_bundle, read_manifest

    model_dir = args.model_dir or os.environ.get(f"{ENV_PREFIX}_MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR
    if args.verify:
        _, _, manifest = load_bundle(model_dir, "cpu")
        logger.info(f"Bundle {model_dir} verified: checksum matches and the model loads")
    else:
        manifest = read_manifest(model_dir)
    print(json.dumps(manifest, indent=2))
    return 0


def _build(args: argparse.Namespace) -> int:
    """Build a model image (or write its context) from the ``build`` subcommand's options."""
    from mlcast_icb.build import build_image

    logger.info(f"{DISTRIBUTION} {__version__}: building a model image")
    build_image(
        args.tag,
        checkpoint=args.checkpoint,
        bundle=args.bundle,
        config=args.config,
        timestep=args.timestep,
        name=args.name,
        model_version=args.model_version,
        base_image=args.base_image,
        torch_variant=args.torch,
        platform=args.platform,
        push=args.push,
        context_only=args.context_only,
    )
    return 0


def _build_base(args: argparse.Namespace) -> int:
    """Build the runtime base image from the ``build-base`` subcommand's options."""
    from mlcast_icb.build import build_base

    build_base(
        args.tag,
        args.torch,
        platform=args.platform,
        push=args.push,
        expected_version=args.expected_version,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
