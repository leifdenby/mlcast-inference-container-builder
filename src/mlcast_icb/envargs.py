"""Set the options of a command line interface from environment variables.

Copied from DMI's MLWP inference container for ANNA
(https://github.com/dmidk/mlwp-inference-container-anna, ``src/anna_inference/envargs.py``),
which defines this convention for DMI's inference containers, so that this container is
configured the same way, with its own prefix ``MLCAST``. Changes from the original:
numpy docstrings, type annotations and this repository's ruff formatting, and
``also_known`` / :func:`foreign_env_vars` for a program whose subcommands each have
their own options (see below). It stays standard-library only, so it can be compared
with, and copied back to, the original.

The convention, for the entrypoints of MLWP inference containers: each model has an
environment variable prefix (e.g. ``ANNA``), and every option ``--some-option`` of its
entrypoint can also be set with the variable ``{PREFIX}_SOME_OPTION``. An option given on
the command line wins over its variable, which wins over the option's default::

    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-duration", default="PT18H")
    envargs.add_env_help(parser, "ANNA")
    args, sources = envargs.parse_args(parser, "ANNA")
    for line in envargs.config_lines(parser, args, sources, "ANNA"):
        logger.info(line)

- The variables are turned into command line arguments placed before the real ones,
  so argparse checks their types and choices, and a required option can come from a
  variable alone.
- A flag (e.g. ``store_true``) takes ``true``/``false`` (also ``1``/``0``,
  ``yes``/``no``), and an option taking several values a space-separated list.
- A set variable with the prefix that matches no option is an error (a misspelling
  would otherwise have no effect), with a suggestion of close names. Variables without
  the prefix are ignored.
- An empty variable counts as unset.
- Only options are supported, no positional arguments or subcommands.

Subcommands: a program with several subcommands sharing one environment (one container
image whose entrypoint takes ``run`` or ``serve``) parses each subcommand's options with
that subcommand's own parser, and passes the variable names of the other subcommands as
``also_known``. Such a variable is then not an error, only ignored (see
:func:`foreign_env_vars`), so one environment can hold the configuration of every
subcommand, while a name that matches no option of any subcommand is still an error.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import os
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NoReturn

#: Accepted spellings of a true or false flag value (case-insensitive).
TRUE_VALUES = ("true", "1", "yes")
FALSE_VALUES = ("false", "0", "no")

#: Where an option's value came from, as :func:`parse_args` reports it.
SOURCE_CLI, SOURCE_ENV, SOURCE_DEFAULT = "cli", "env", "default"


def env_var_name(prefix: str, option: str) -> str:
    """Name the environment variable of an option.

    Parameters
    ----------
    prefix : str
        The program's variable prefix, e.g. ``'ANNA'``.
    option : str
        A long option, e.g. ``'--forecast-duration'``.

    Returns
    -------
    str
        E.g. ``'ANNA_FORECAST_DURATION'``.
    """
    return f"{prefix}_{option.lstrip('-').upper().replace('-', '_')}"


def _options(parser: argparse.ArgumentParser) -> list[tuple[argparse.Action, str]]:
    """List the options of a parser with their first long option string.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    Returns
    -------
    list of (argparse.Action, str)
        Every action except ``--help`` and ``--version``, with its first ``--`` option.

    Raises
    ------
    ValueError
        If the parser has a positional argument, a subcommand or a short-only option,
        which have no variable.
    """
    options = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        long_options = [o for o in action.option_strings if o.startswith("--")]
        if not long_options:
            raise ValueError(
                f"envargs only supports long options, not {action.dest!r} "
                "(a positional argument, subcommand or short-only option)"
            )
        options.append((action, long_options[0]))
    return options


def env_var_names(parser: argparse.ArgumentParser, prefix: str) -> dict[str, str]:
    """Name the environment variable of each option of a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.
    prefix : str
        The program's variable prefix.

    Returns
    -------
    dict of str to str
        The variable name of each option, by ``dest``, in the parser's order.
    """
    return {action.dest: env_var_name(prefix, option) for action, option in _options(parser)}


def add_env_help(parser: argparse.ArgumentParser, prefix: str) -> None:
    """Add ``[env: {PREFIX}_OPTION]`` to the help of each option, and the convention to the epilog.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, modified in place.
    prefix : str
        The program's variable prefix.
    """
    for action, option in _options(parser):
        if action.help == argparse.SUPPRESS:
            continue
        env_help = f"[env: {env_var_name(prefix, option)}]"
        action.help = f"{action.help} {env_help}" if action.help else env_help
    convention = (
        f"Every option --some-option can also be set with the environment "
        f"variable {prefix}_SOME_OPTION; an option on the command line wins "
        f"over its variable."
    )
    parser.epilog = f"{parser.epilog}\n\n{convention}" if parser.epilog else convention


def _parse_bool(parser: argparse.ArgumentParser, name: str, value: str) -> bool:
    """Parse a flag's variable.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, to report an error with.
    name : str
        The variable name, for the error.
    value : str
        Its value.

    Returns
    -------
    bool

    Raises
    ------
    SystemExit
        Exit code 2, through :meth:`argparse.ArgumentParser.error`, if the value is
        none of :data:`TRUE_VALUES` and :data:`FALSE_VALUES`.
    """
    if value.lower() in TRUE_VALUES:
        return True
    if value.lower() in FALSE_VALUES:
        return False
    _error(parser, f"{name}={value!r}: expected one of {', '.join(TRUE_VALUES + FALSE_VALUES)}")


def _check_value(parser: argparse.ArgumentParser, action: argparse.Action, name: str, value: str) -> None:
    """Check a value from a variable as argparse would, but name the variable in the error.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, to report an error with.
    action : argparse.Action
        The option the variable sets, with its ``type`` and ``choices``.
    name : str
        The variable name.
    value : str
        Its value (one of them, for an option taking several).

    Raises
    ------
    SystemExit
        Exit code 2 if ``action.type`` rejects the value or it is not one of
        ``action.choices``.
    """
    converted: Any = value
    if callable(action.type):
        try:
            converted = action.type(value)
        except (TypeError, ValueError, argparse.ArgumentTypeError) as ex:
            _error(parser, f"{name}={value!r}: invalid value ({ex})")
    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(map(repr, action.choices))
        _error(parser, f"{name}={value!r}: invalid choice (choose from {choices})")


def _env_tokens(
    parser: argparse.ArgumentParser, action: argparse.Action, option: str, name: str, value: str
) -> list[str]:
    """Turn one variable into command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, to report an error with.
    action : argparse.Action
        The option the variable sets.
    option : str
        Its long option string.
    name : str
        The variable name.
    value : str
        Its (non-empty) value.

    Returns
    -------
    list of str
        ``[option]`` or nothing for a flag, ``--no-<option>`` for a false
        ``BooleanOptionalAction``, ``['--option=value']`` for an option taking one
        value, and ``[option, *values]`` for one taking several.

    Raises
    ------
    SystemExit
        Exit code 2 for an invalid value.
    """
    if isinstance(action, argparse.BooleanOptionalAction):
        if _parse_bool(parser, name, value):
            return [option]
        return [f"--no-{option[2:]}"]
    if action.nargs == 0:
        # store_true, store_false, store_const, count: given or not
        return [option] if _parse_bool(parser, name, value) else []
    if action.nargs in (None, "?"):
        _check_value(parser, action, name, value)
        # `--option=value`, so a value starting with "-" isn't an option
        return [f"{option}={value}"]
    values = shlex.split(value)
    for v in values:
        _check_value(parser, action, name, v)
    return [option, *values]


def env_argv(
    parser: argparse.ArgumentParser,
    prefix: str,
    environ: Mapping[str, str] | None = None,
    also_known: Iterable[str] = (),
) -> tuple[list[str], set[str]]:
    """Turn the ``{prefix}_*`` variables into command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser whose options the variables set.
    prefix : str
        The program's variable prefix.
    environ : mapping of str to str, optional
        The environment. Defaults to :data:`os.environ`.
    also_known : iterable of str, optional
        Variable names that are not options of ``parser`` but not an error either
        (the options of another subcommand); they are left out.

    Returns
    -------
    argv : list of str
        The arguments, to be placed before the real command line.
    from_env : set of str
        The ``dest`` of each option set by a variable.

    Raises
    ------
    SystemExit
        Exit code 2 if a set ``{prefix}_*`` variable matches no option (nor
        ``also_known``), naming the closest option's variable, or a value is invalid.
    """
    environ = os.environ if environ is None else environ
    names = env_var_names(parser, prefix)
    actions = {action.dest: (action, option) for action, option in _options(parser)}
    legal = set(names.values()) | set(also_known)

    unknown = sorted(n for n in environ if n.startswith(f"{prefix}_") and n not in legal)
    if unknown:
        problems = []
        for n in unknown:
            close = difflib.get_close_matches(n, names.values(), n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            problems.append(f"{n}{hint}")
        _error(
            parser,
            f"unknown environment variable(s) {', '.join(problems)}: "
            f"{prefix}_* variables must match an option (see --help)",
        )

    argv: list[str] = []
    from_env: set[str] = set()
    for dest, name in names.items():
        # an empty variable counts as unset (e.g. `-e NAME=` in docker)
        if environ.get(name, "") != "":
            action, option = actions[dest]
            argv += _env_tokens(parser, action, option, name, environ[name])
            from_env.add(dest)
    return argv, from_env


def foreign_env_vars(
    parser: argparse.ArgumentParser,
    prefix: str,
    environ: Mapping[str, str] | None = None,
    also_known: Iterable[str] = (),
) -> list[str]:
    """List the set variables that :func:`parse_args` ignores as another subcommand's.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser being used.
    prefix : str
        The program's variable prefix.
    environ : mapping of str to str, optional
        The environment. Defaults to :data:`os.environ`.
    also_known : iterable of str, optional
        As given to :func:`parse_args`.

    Returns
    -------
    list of str
        The non-empty variables named in ``also_known`` that are not an option of
        ``parser``, sorted, e.g. to log that they are ignored.
    """
    environ = os.environ if environ is None else environ
    foreign = set(also_known) - set(env_var_names(parser, prefix).values())
    return sorted(n for n in environ if n in foreign and environ[n] != "")


def _cli_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    """Find the options given on the command line itself.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser; a copy without defaults or required options parses ``argv``.
    argv : sequence of str
        The command line alone, without the arguments from variables.

    Returns
    -------
    set of str
        The ``dest`` of each option in ``argv``.
    """
    cli_parser = copy.deepcopy(parser)
    for action in cli_parser._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    return set(vars(cli_parser.parse_args(list(argv))))


def parse_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    also_known: Iterable[str] = (),
) -> tuple[argparse.Namespace, dict[str, str]]:
    """Parse the command line and the ``{prefix}_*`` variables.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser; options only.
    prefix : str
        The program's variable prefix.
    argv : sequence of str, optional
        The command line. Defaults to ``sys.argv[1:]``.
    environ : mapping of str to str, optional
        The environment. Defaults to :data:`os.environ`.
    also_known : iterable of str, optional
        Variable names that are not options of ``parser`` but not an error either
        (the options of another subcommand); they are ignored.

    Returns
    -------
    args : argparse.Namespace
        The parsed options; the command line wins over a variable, which wins over the
        default.
    sources : dict of str to str
        Where each option's value came from, by ``dest``: :data:`SOURCE_CLI`,
        :data:`SOURCE_ENV` or :data:`SOURCE_DEFAULT`.

    Raises
    ------
    SystemExit
        Exit code 2 for an unknown variable or an invalid value, as argparse exits on a
        bad command line (and 0 after ``--help``).
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    from_env_argv, from_env = env_argv(parser, prefix, environ, also_known)
    args = parser.parse_args(from_env_argv + argv)

    from_cli = _cli_dests(parser, argv)
    sources = {}
    for dest in vars(args):
        if dest in from_cli:
            sources[dest] = SOURCE_CLI
        elif dest in from_env:
            sources[dest] = SOURCE_ENV
        else:
            sources[dest] = SOURCE_DEFAULT
    return args, sources


def config_lines(
    parser: argparse.ArgumentParser, args: argparse.Namespace, sources: Mapping[str, str], prefix: str
) -> list[str]:
    """Render the effective configuration, one line per option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.
    args : argparse.Namespace
        The values to show, by ``dest``: the parsed options, or a namespace of their
        renderings. Don't pass an option holding a secret as it is.
    sources : mapping of str to str
        Where each value came from, as :func:`parse_args` returns it.
    prefix : str
        The program's variable prefix.

    Returns
    -------
    list of str
        ``'{VARIABLE} = {value} ({source})'`` per option, the names aligned.
    """
    names = env_var_names(parser, prefix)
    width = max(map(len, names.values()), default=0)
    return [f"{name:<{width}} = {getattr(args, dest)} ({sources[dest]})" for dest, name in names.items()]


def _error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    """Report a usage error through the parser, which prints it and exits 2.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.
    message : str
        The error.

    Raises
    ------
    SystemExit
        Always, with exit code 2.
    """
    parser.error(message)
