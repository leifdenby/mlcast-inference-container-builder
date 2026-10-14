"""The options of ``run`` and ``serve``, each mirrored by an ``MLCAST_*`` environment variable.

Both commands follow the convention of DMI's MLWP inference containers (see
:mod:`mlcast_icb.envargs`): every option ``--some-option`` can also be set with the
variable ``MLCAST_SOME_OPTION``; the command line wins over the variable, which wins
over the default. ``run`` and ``serve`` live in one image and share its environment, so
a variable of the other command is ignored (logged at DEBUG) rather than an error; a
``MLCAST_*`` variable that is an option of neither is an error (exit 2), with a
suggestion. The developer commands (``bundle``, ``info``, ``build``, ``build-base``) are
configured by their command line only, and their options have no variables.

argparse only collects the values, as strings: the commands validate them themselves
(:class:`~mlcast_icb.run.RunConfig`, :class:`~mlcast_icb.serve.ServeConfig`), so that
every problem is reported at once, naming the variable or the option it came from. This
module does not import the server (FastAPI, uvicorn), so ``run`` can learn ``serve``'s
variable names, and the other way round, without the ``serve`` extra.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd

from mlcast_icb import ENV_PREFIX, envargs
from mlcast_icb.bundle import DEFAULT_MODEL_DIR
from mlcast_icb.device import DEVICES
from mlcast_icb.errors import ConfigError
from mlcast_icb.io.storage import SIDES

#: The commands configured through options and ``MLCAST_*`` variables.
COMMANDS = ("run", "serve")

_DEVICE_METAVAR = "{" + ",".join(DEVICES) + "}"

_T = TypeVar("_T")


def run_parser() -> argparse.ArgumentParser:
    """Build the parser of ``mlcast-icb run``.

    Returns
    -------
    argparse.ArgumentParser
        Options only, every value a string (or ``None`` when not given); the
        ``--<side>-s3-anon`` flags are ``True``, ``False`` or ``None``.
    """
    parser = argparse.ArgumentParser(
        prog="mlcast-icb run",
        description="Forecast from the zarr store --input-zarr into the zarr store --output-zarr with the bundled "
        "model. Stores are local paths, s3:// or gs:// URIs.",
    )
    parser.add_argument("--input-zarr", metavar="URI", help="the input store (required)")
    parser.add_argument("--output-zarr", metavar="URI", help="the output store, replaced if it exists (required)")
    _add_model_dir(parser)
    parser.add_argument(
        "--forecast-reference-time",
        metavar="TIME",
        help="the last input time to forecast from, ISO 8601 (UTC if no time zone); default: the last time in "
        "the input",
    )
    parser.add_argument("--forecast-steps", metavar="N", help="number of lead times; default: the model's")
    parser.add_argument("--ensemble-size", metavar="N", help="number of ensemble members; default: the model's")
    parser.add_argument("--seed", metavar="N", help="seed for the ensemble noise, for a reproducible run")
    _add_device(parser)
    parser.add_argument(
        "--zarr-format",
        default="auto",
        metavar="{auto,2,3}",
        help="format of the output store; auto matches the input (default: %(default)s)",
    )
    _add_log_level(parser)
    for side in SIDES:
        parser.add_argument(
            f"--{side}-aws-profile",
            metavar="PROFILE",
            help=f"AWS profile (from ~/.aws) for the {side} store; default: the AWS configuration (AWS_PROFILE)",
        )
        parser.add_argument(
            f"--{side}-s3-endpoint-url",
            metavar="URL",
            help=f"S3 endpoint for the {side} store; default: the AWS configuration (AWS_ENDPOINT_URL, or the "
            "profile's endpoint_url)",
        )
        parser.add_argument(
            f"--{side}-s3-anon",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"unsigned (anonymous) requests to the {side} store; default: anonymous unless credentials are found",
        )
        parser.add_argument(
            f"--{side}-storage-options",
            metavar="JSON",
            help=f"further fsspec options for the {side} store, a JSON object merged last",
        )
    envargs.add_env_help(parser, ENV_PREFIX)
    return parser


def serve_parser() -> argparse.ArgumentParser:
    """Build the parser of ``mlcast-icb serve``.

    Returns
    -------
    argparse.ArgumentParser
        Options only, every value a string (or ``None`` when not given).
    """
    parser = argparse.ArgumentParser(
        prog="mlcast-icb serve",
        description="Serve forecasts over HTTP (POST /predict with a netCDF; IRENE-compatible). Needs the `serve` "
        "extra.",
    )
    _add_model_dir(parser)
    _add_device(parser)
    parser.add_argument("--host", default="0.0.0.0", help="address to bind (default: %(default)s)")
    parser.add_argument("--port", default="8000", metavar="N", help="port to bind (default: %(default)s)")
    parser.add_argument(
        "--max-upload-mb", default="100", metavar="N", help="largest accepted upload, in MiB (default: %(default)s)"
    )
    parser.add_argument(
        "--max-queue",
        default="2",
        metavar="N",
        help="predictions allowed to wait while one runs; more are answered 429 (default: %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        metavar="SECONDS",
        help="how long a prediction may wait for the model before it is answered 503; default: as long as it takes",
    )
    parser.add_argument(
        "--max-forecast-steps",
        default="48",
        metavar="N",
        help="upper limit of the forecast_steps parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--max-ensemble-size",
        default="10",
        metavar="N",
        help="upper limit of the ensemble_size parameter (default: %(default)s)",
    )
    _add_log_level(parser)
    envargs.add_env_help(parser, ENV_PREFIX)
    return parser


#: The parser of each command.
PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {"run": run_parser, "serve": serve_parser}


def _add_model_dir(parser: argparse.ArgumentParser) -> None:
    """Add ``--model-dir``, shared by ``run`` and ``serve``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command's parser, modified in place.
    """
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, metavar="DIR", help="the model bundle (default: %(default)s)"
    )


def _add_device(parser: argparse.ArgumentParser) -> None:
    """Add ``--device``, shared by ``run`` and ``serve``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command's parser, modified in place.
    """
    parser.add_argument(
        "--device",
        default="auto",
        metavar=_DEVICE_METAVAR,
        help="compute device; cuda fails (exit 4) without a GPU, auto falls back to the CPU (default: %(default)s; "
        "MLCAST_DEVICE=cuda in GPU images)",
    )


def _add_log_level(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level``, shared by ``run`` and ``serve``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command's parser, modified in place.
    """
    parser.add_argument("--log-level", default="INFO", metavar="LEVEL", help="loguru level (default: %(default)s)")


@dataclass(frozen=True)
class ParsedOptions:
    """A command's options, parsed from the command line and the environment.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command's parser.
    args : argparse.Namespace
        The values.
    sources : dict of str to str
        Where each value came from, by ``dest``: ``'cli'``, ``'env'`` or ``'default'``.
    ignored : list of str
        The ``MLCAST_*`` variables that are options of the other command, and so ignored.
    """

    parser: argparse.ArgumentParser
    args: argparse.Namespace
    sources: dict[str, str]
    ignored: list[str]

    def label(self, dest: str) -> str:
        """Name an option the way it was given, for an error message.

        Parameters
        ----------
        dest : str
            The option's ``dest``.

        Returns
        -------
        str
            The variable (``MLCAST_SEED``) for a value from the environment, the option
            (``--seed``) for one from the command line, and both when it was not given.
        """
        variable = envargs.env_var_names(self.parser, ENV_PREFIX)[dest]
        option = "--" + dest.replace("_", "-")
        source = self.sources.get(dest)
        if source == envargs.SOURCE_ENV:
            return variable
        if source == envargs.SOURCE_CLI:
            return option
        return f"{variable} ({option})"

    def config_lines(self, rendered: Mapping[str, str]) -> list[str]:
        """Render the effective configuration, one line per option, for the startup log.

        Parameters
        ----------
        rendered : mapping of str to str
            How to show each option's value, by ``dest``, secrets masked.

        Returns
        -------
        list of str
            ``'MLCAST_<OPTION> = <value> (<source>)'`` per option.
        """
        return envargs.config_lines(self.parser, argparse.Namespace(**rendered), self.sources, ENV_PREFIX)


def parse_options(
    command: str, argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> ParsedOptions:
    """Parse the options of ``run`` or ``serve`` from the command line and the ``MLCAST_*`` variables.

    Parameters
    ----------
    command : {'run', 'serve'}
        The command.
    argv : sequence of str, optional
        The command's arguments (after the command name). Defaults to none.
    env : mapping of str to str, optional
        The environment. Defaults to :data:`os.environ`.

    Returns
    -------
    ParsedOptions

    Raises
    ------
    SystemExit
        Exit code 2, after argparse has printed the usage and the error, for an unknown
        option, an ``MLCAST_*`` variable that is an option of neither command, or an
        invalid flag value; 0 after ``--help``.
    """
    parser = PARSERS[command]()
    also_known = {
        name
        for other in COMMANDS
        if other != command
        for name in envargs.env_var_names(PARSERS[other](), ENV_PREFIX).values()
    }
    args, sources = envargs.parse_args(parser, ENV_PREFIX, list(argv or ()), env, also_known)
    ignored = envargs.foreign_env_vars(parser, ENV_PREFIX, env, also_known)
    return ParsedOptions(parser=parser, args=args, sources=sources, ignored=ignored)


# --- validating the values ------------------------------------------------------------


def attempt(problems: list[str], parse: Callable[[], _T]) -> _T | None:
    """Run one validation step, recording a :class:`ConfigError` in ``problems`` instead of raising it."""
    try:
        return parse()
    except ConfigError as exc:
        problems.append(str(exc))
        return None


def blank_to_none(value: str | None) -> str | None:
    """Strip a value, and treat a blank one as not given."""
    value = None if value is None else value.strip()
    return value or None


def require(value: str | None, label: str) -> str:
    """Check that a required value is given.

    Raises
    ------
    ConfigError
        If it is unset or blank.
    """
    value = blank_to_none(value)
    if value is None:
        raise ConfigError(f"{label} is required but not set")
    return value


def get_int(value: str | None, label: str, minimum: int, maximum: int | None = None) -> int | None:
    """Parse an optional integer between ``minimum`` and ``maximum``; ``None`` when not given.

    Raises
    ------
    ConfigError
        If it is not an integer, or out of range.
    """
    raw = blank_to_none(value)
    if raw is None:
        return None
    try:
        number = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{label} must be an integer, got {raw!r}") from exc
    if number < minimum:
        raise ConfigError(f"{label} must be at least {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{label} must be at most {maximum}, got {number}")
    return number


def get_positive_float(value: str | None, label: str) -> float | None:
    """Parse an optional positive number of seconds; ``None`` when not given.

    Raises
    ------
    ConfigError
        If it is not a finite positive number.
    """
    raw = blank_to_none(value)
    if raw is None:
        return None
    try:
        number = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{label} must be a number of seconds, got {raw!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{label} must be a positive number of seconds, got {raw!r}")
    return number


def get_choice(value: str | None, label: str, choices: Sequence[str], default: str) -> str:
    """Parse a case-insensitive choice; ``default`` when not given.

    Raises
    ------
    ConfigError
        If it is not one of ``choices``.
    """
    raw = blank_to_none(value)
    if raw is None:
        return default
    if raw.lower() not in choices:
        raise ConfigError(f"{label} must be one of {', '.join(choices)}, got {raw!r}")
    return raw.lower()


def get_datetime(value: str | None, label: str) -> np.datetime64 | None:
    """Parse an ISO 8601 date-time as naive UTC (a time zone, if given, is converted); ``None`` when not given.

    Raises
    ------
    ConfigError
        If it is not a date-time.
    """
    raw = blank_to_none(value)
    if raw is None:
        return None
    message = f"{label} must be an ISO 8601 date-time such as '2026-05-01T12:00:00Z', got {raw!r}"
    try:
        stamp = pd.Timestamp(raw)
    except ValueError as exc:
        raise ConfigError(message) from exc
    if stamp is pd.NaT:
        raise ConfigError(message)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return np.datetime64(stamp.to_datetime64(), "ns")


def log_level(value: str | None) -> str:
    """Normalise a log level name; ``INFO`` when not given."""
    return (blank_to_none(value) or "INFO").upper()
