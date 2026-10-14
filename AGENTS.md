# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this project is

A tool that turns a trained mlcast nowcasting model (checkpoint + training config) into
an inference container image with two modes, `run` (zarr/S3 batch) and `serve` (HTTP).
See `PLAN.md` for the design and build order.

## Layout

```
src/mlcast_icb/
  upstream/     code meant to move into mlcast, one module per proposed PR (see its README.md)
tests/
  upstream/     one test_<module>.py per upstream module, portable to mlcast's tests/
```

## `mlcast_icb/upstream/` rules

- **The rest of the builder reaches mlcast only through `mlcast_icb.upstream`.** `run`,
  `serve` and `build` never import mlcast internals directly.
- **Upstream modules are written in mlcast's style**, as they would look inside mlcast:
  numpy docstrings, `jaxtyping` + `beartype` shape annotations, `loguru` for logging, no
  inline imports. They import only from mlcast and its dependencies, never from the rest
  of `mlcast_icb`.
- **Tests for `upstream/<module>.py` live in `tests/upstream/test_<module>.py`** and import
  only mlcast, the module under test and third-party libraries, so they can move into
  mlcast's `tests/` unchanged. Follow mlcast's test conventions (full type annotations on
  test parameters, Given/When/Then comments).
- **When the matching PR is merged** and the pinned mlcast includes it, the module becomes
  a one-line re-export from mlcast, and is deleted once nothing imports it.
- Keep `src/mlcast_icb/upstream/README.md` in sync when a module's API or target changes.

## Conventions

- Use `uv` for everything; see README.md for the commands. Pass
  `--extra cpu --extra remote --extra serve --extra dev` to `uv run` as well as `uv sync`;
  without them uv syncs the environment down and tests are skipped or fail.
- Never prefix shell commands with `cd ... &&`; pass absolute paths (`uv run --directory`,
  `git -C`). For ad-hoc Python, write a script file and run it rather than using heredocs
  or long `python -c` strings.
- Ruff settings match mlcast's (line length 120). Run `ruff format` rather than
  hand-wrapping.
- Versioning is `hatch-vcs` from the git tag; never hardcode a version.
- **mlcast is pinned to an exact commit** in `[tool.uv.sources]`. When bumping it, update
  the "Pinned mlcast version" table in README.md, re-run `uv lock`, and re-check the
  mlcast file/line references in `upstream/README.md`.
- **Never commit unless explicitly asked.**
