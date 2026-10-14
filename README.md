# mlcast-inference-container-builder

Turns a trained [mlcast](https://github.com/mlcast-community/mlcast) nowcasting model
(checkpoint + training config) into a ready-to-run inference container image. One image
offers two modes: `run` (batch, zarr in/out on S3) and `serve` (an HTTP server compatible
with the IRENE API).

Status: early. Only the first mlcast-facing module (`mlcast_icb.upstream.processing`)
exists so far.

- [PLAN.md](PLAN.md) — feasibility, design and build order.
- [src/mlcast_icb/upstream/README.md](src/mlcast_icb/upstream/README.md) — code written to
  move into mlcast, one proposed PR at a time.

## Development

```sh
uv sync --extra cpu --extra dev
uv run --extra cpu --extra dev pytest
uv run --extra cpu --extra dev ruff check .
uv run --extra cpu --extra dev ruff format --check .
```

torch is chosen by exactly one of the `cpu`, `gpu-cu126`, `gpu-cu128` and `gpu-cu130`
extras, mirroring mlcast's own extras. Pass the same `--extra` flags to `uv run`, or it
re-syncs without them.

## Pinned mlcast version

Each builder release pins one exact mlcast commit (`[tool.uv.sources]` in
`pyproject.toml`, resolved in `uv.lock`). Bumping mlcast means cutting a new builder
release.

| Builder version | mlcast commit |
| --- | --- |
| unreleased | [`5edce07`](https://github.com/mlcast-community/mlcast/commit/5edce0756410ecdd963b2a5ba85e3afc6a0a532b) (upstream `main`) |
