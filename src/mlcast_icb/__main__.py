"""Command line: ``mlcast-icb {run,serve,bundle,info}``.

``run`` and ``serve`` are what an image executes; they take no options, since a
container is configured through its environment. ``bundle`` and ``info`` are build-time
tools, configured through options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from loguru import logger

from mlcast_icb import __version__
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

    sub.add_parser(
        "run",
        help="batch forecast, zarr in and out (configured by environment variables)",
        description="Forecast from INPUT_ZARR into OUTPUT_ZARR. Configured entirely by environment variables; "
        "see the README.",
    )
    sub.add_parser("serve", help="HTTP server (not implemented yet)")

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
        "model_dir", nargs="?", help="bundle directory (default: $MODEL_DIR, or /model when that is unset)"
    )
    info.add_argument("--verify", action="store_true", help="also check the checksum and load the model on CPU")
    return parser


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
        usage error or ``serve`` (not implemented yet), 1 for anything unexpected.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return int(exc.code or 0)

    if args.command == "run":
        from mlcast_icb.run import main as run_main

        return run_main()

    from mlcast_icb.logs import configure_logging

    configure_logging(os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO")
    if args.command is None:
        parser.print_help(sys.stderr)
        return ConfigError.exit_code
    if args.command == "serve":
        logger.error("serve is not implemented yet; use run")
        return ConfigError.exit_code

    try:
        if args.command == "bundle":
            return _bundle(args)
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

    model_dir = args.model_dir or os.environ.get("MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR
    if args.verify:
        _, _, manifest = load_bundle(model_dir, "cpu")
        logger.info(f"Bundle {model_dir} verified: checksum matches and the model loads")
    else:
        manifest = read_manifest(model_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
