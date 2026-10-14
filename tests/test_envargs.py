"""The generic CLI <-> ``{PREFIX}_*`` environment variable helper.

Ported from DMI's ANNA inference container (``tests/test_envargs.py`` in
dmidk/mlwp-inference-container-anna), plus the tests of ``also_known`` (subcommands).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import pytest

from mlcast_icb import envargs

PREFIX = "MODEL"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--mode", choices=["a", "b"], default="a")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction)
    parser.add_argument("--levels", nargs="+", type=int, default=[1000])
    return parser


def _parse(
    argv: Sequence[str], environ: Mapping[str, str], also_known: Sequence[str] = ()
) -> tuple[argparse.Namespace, dict[str, str]]:
    return envargs.parse_args(_parser(), PREFIX, argv, environ, also_known)


def test_env_var_name() -> None:
    assert envargs.env_var_name("ANNA", "--forecast-duration") == "ANNA_FORECAST_DURATION"


def test_env_sets_default_and_cli_wins() -> None:
    environ = {"MODEL_INPUT_PATH": "env.zarr", "MODEL_N_STEPS": "8"}
    args, sources = _parse([], environ)
    assert (args.input_path, args.n_steps) == ("env.zarr", 8)
    assert sources["input_path"] == sources["n_steps"] == "env"
    assert sources["mode"] == "default"

    args, sources = _parse(["--n-steps", "2"], environ)
    assert (args.input_path, args.n_steps) == ("env.zarr", 2)
    assert sources["n_steps"] == "cli"


def test_required_from_env_or_missing() -> None:
    args, _ = _parse([], {"MODEL_INPUT_PATH": "x"})
    assert args.input_path == "x"
    with pytest.raises(SystemExit) as ex:
        _parse([], {})
    assert ex.value.code == 2


def test_empty_variable_is_unset() -> None:
    args, sources = _parse(["--input-path", "x"], {"MODEL_N_STEPS": ""})
    assert args.n_steps == 4 and sources["n_steps"] == "default"


@pytest.mark.parametrize(
    ("value", "expected"), [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("no", False)]
)
def test_flags(value: str, expected: bool) -> None:
    args, _ = _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_DRY_RUN": value})
    assert args.dry_run is expected

    args, _ = _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_CACHE": value})
    assert args.cache is expected


def test_unset_boolean_optional_flag_stays_none() -> None:
    args, sources = _parse([], {"MODEL_INPUT_PATH": "x"})
    assert args.cache is None and sources["cache"] == "default"


def test_invalid_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_DRY_RUN": "maybe"})
    assert "MODEL_DRY_RUN='maybe'" in capsys.readouterr().err


def test_nargs_from_space_separated_value() -> None:
    args, _ = _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_LEVELS": "850 500"})
    assert args.levels == [850, 500]


def test_value_starting_with_a_dash_is_not_an_option() -> None:
    args, _ = _parse([], {"MODEL_INPUT_PATH": "-weird"})
    assert args.input_path == "-weird"


@pytest.mark.parametrize(("name", "value"), [("MODEL_N_STEPS", "many"), ("MODEL_MODE", "c")])
def test_type_and_choice_errors_name_the_variable(capsys: pytest.CaptureFixture[str], name: str, value: str) -> None:
    with pytest.raises(SystemExit) as ex:
        _parse([], {"MODEL_INPUT_PATH": "x", name: value})
    assert ex.value.code == 2
    assert f"{name}={value!r}" in capsys.readouterr().err


def test_unknown_prefixed_variable(capsys: pytest.CaptureFixture[str]) -> None:
    environ = {"MODEL_INPUT_PATH": "x", "MODEL_N_STEP": "8", "OTHER_X": "1"}
    with pytest.raises(SystemExit) as ex:
        _parse([], environ)
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "MODEL_N_STEP (did you mean MODEL_N_STEPS?)" in err
    assert "OTHER_X" not in err


def test_empty_unknown_variable_is_still_an_error() -> None:
    with pytest.raises(SystemExit):
        _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_N_STEP": ""})


def test_help_and_config_lines() -> None:
    parser = _parser()
    envargs.add_env_help(parser, PREFIX)
    assert "[env: MODEL_N_STEPS]" in parser.format_help()

    args, sources = envargs.parse_args(parser, PREFIX, ["--mode", "b"], {"MODEL_INPUT_PATH": "x"})
    lines = envargs.config_lines(parser, args, sources, PREFIX)
    assert any(line.split() == ["MODEL_INPUT_PATH", "=", "x", "(env)"] for line in lines)
    assert any(line.split() == ["MODEL_MODE", "=", "b", "(cli)"] for line in lines)
    assert any(line.split() == ["MODEL_N_STEPS", "=", "4", "(default)"] for line in lines)


def test_positional_arguments_unsupported() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    with pytest.raises(ValueError):
        envargs.parse_args(parser, PREFIX, ["x"], {})


# --- also_known: the variables of another subcommand ------------------------------------


def test_another_subcommands_variable_is_ignored_not_an_error() -> None:
    # Given a variable that is an option of another subcommand, not of this one
    environ = {"MODEL_INPUT_PATH": "x", "MODEL_PORT": "9000"}

    # When parsing with that subcommand's variables as also_known
    args, sources = _parse([], environ, also_known=["MODEL_PORT"])

    # Then it is neither an error nor an option, and it is reported as ignored
    assert not hasattr(args, "port")
    assert "port" not in sources
    assert envargs.foreign_env_vars(_parser(), PREFIX, environ, ["MODEL_PORT"]) == ["MODEL_PORT"]


def test_also_known_does_not_legalise_other_names(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ex:
        _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_PROT": "9000"}, also_known=["MODEL_PORT"])
    assert ex.value.code == 2
    assert "MODEL_PROT" in capsys.readouterr().err


def test_suggestions_come_from_this_parsers_options_only(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_N_STEP": "1"}, also_known=["MODEL_N_STEPZ"])
    assert "did you mean MODEL_N_STEPS?" in capsys.readouterr().err


def test_an_own_option_is_not_foreign_even_if_also_known() -> None:
    environ = {"MODEL_INPUT_PATH": "x", "MODEL_N_STEPS": "3", "MODEL_PORT": ""}
    args, _ = _parse([], environ, also_known=["MODEL_N_STEPS", "MODEL_PORT"])
    assert args.n_steps == 3
    # An empty variable is unset, so not reported either.
    assert envargs.foreign_env_vars(_parser(), PREFIX, environ, ["MODEL_N_STEPS", "MODEL_PORT"]) == []
