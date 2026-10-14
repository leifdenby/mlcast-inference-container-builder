"""Logging setup shared by every subcommand: loguru to stderr, one record per line."""

from __future__ import annotations

import sys

from loguru import logger

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)


def configure_logging(level: str) -> None:
    """Point loguru at stderr.

    Parameters
    ----------
    level : str
        A loguru level name. An unrecognised level warns and falls back to ``INFO``,
        rather than failing a run over a typo in a log setting.
    """
    logger.remove()
    try:
        logger.add(sys.stderr, level=level, format=LOG_FORMAT, colorize=sys.stderr.isatty())
    except ValueError:
        logger.add(sys.stderr, level="INFO", format=LOG_FORMAT, colorize=sys.stderr.isatty())
        logger.warning(f"LOG_LEVEL={level!r} is not a known level; using INFO")


def log_settings(headline: str, settings: dict[str, str]) -> None:
    """Log a headline and then one aligned ``name = value`` record per setting.

    Parameters
    ----------
    headline : str
        The first record, e.g. ``'Effective configuration (defaults included):'``.
    settings : dict of str to str
        The settings, in the order they should appear.

    Notes
    -----
    One record per setting rather than a single multi-line message: container log
    collectors split on newlines, so the continuation lines of one message would lose
    their timestamp and level, while separate records each keep theirs and stay
    greppable by name.
    """
    width = max((len(name) for name in settings), default=0)
    caller = logger.opt(depth=1)  # attribute the records to the module that logs the settings
    caller.info(headline)
    for name, value in settings.items():
        caller.info(f"  {name:<{width}} = {value}")
