# mlcast-inference-container-builder

Turns a trained [mlcast](https://github.com/mlcast-community/mlcast) nowcasting model
(checkpoint + training config) into a ready-to-run inference container image. One image
offers two modes: `run` (batch, zarr in/out on S3) and `serve` (an HTTP server compatible
with the IRENE API).

Status: early. Model bundles (`bundle`, `info`) and the batch mode (`run`) work. `serve`
and the image build do not exist yet.

- [PLAN.md](PLAN.md) — feasibility, design and build order.
- [src/mlcast_icb/upstream/README.md](src/mlcast_icb/upstream/README.md) — code written to
  move into mlcast, one proposed PR at a time.

## Usage

One command, `mlcast-icb` (or `python -m mlcast_icb`), with four subcommands. `bundle`
and `info` run at build time, and `run` and `serve` run inside the image.

### `bundle`: package a trained model

```sh
mlcast-icb bundle --checkpoint model.ckpt --config config.yaml --timestep PT5M \
    --out bundle/ --name irene --model-version 1.0
```

This writes a **model bundle**, the directory an image carries at `/model`:

- `model.ckpt` holds the weights and the module's hyperparameters, with the inference spec
  embedded. Optimizer and scheduler state are dropped, so no training YAML is needed at
  run time.
- `manifest.json` holds the spec, the sha256 of `model.ckpt` and of the source checkpoint,
  the builder and mlcast versions, the creation time, and the optional `--name` and
  `--model-version`.

`--config` is needed unless the checkpoint already embeds a spec, as converted checkpoints
do. Training configs do not record the time between steps, so pass `--timestep`. Before
the manifest is written, the bundle is reloaded strictly and runs one forward pass on CPU.
A model that fails either check is never bundled, and the command exits 6.

### `info`: show what a bundle is

```sh
mlcast-icb info bundle/            # prints manifest.json; the default is $MODEL_DIR, then /model
mlcast-icb info bundle/ --verify   # also checks the checksum and loads the model
```

### `run`: batch forecast, zarr in and out (Mode 1)

`run` reads `INPUT_ZARR` and writes the forecast to `OUTPUT_ZARR`. It is configured only
through environment variables:

```sh
INPUT_ZARR=s3://radar/composite.zarr OUTPUT_ZARR=/data/forecast.zarr DEVICE=cuda mlcast-icb run
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `INPUT_ZARR` | **required** | Input store URI: a local path, `s3://…` or `gs://…` |
| `OUTPUT_ZARR` | **required** | Output store URI. Any existing store there is replaced |
| `MODEL_DIR` | `/model` | The model bundle |
| `FORECAST_REFERENCE_TIME` | last time in the input | The last input time, in ISO 8601 (`2026-05-01T12:00Z`). A time without a zone is read as UTC |
| `FORECAST_STEPS` | the model's | Number of steps to forecast |
| `ENSEMBLE_SIZE` | the model's | Number of ensemble members |
| `SEED` | unset | Seed for the ensemble noise. With a seed, runs are reproducible |
| `DEVICE` | `auto` | `auto`, `cuda` or `cpu`. With `cuda`, a run without a visible GPU exits 4 instead of falling back to CPU |
| `ZARR_FORMAT` | `auto` | `auto` (same as the input), `2` or `3` |
| `LOG_LEVEL` | `INFO` | loguru level |

The input and output stores are configured separately, through `SRC_*` and `DST_*`
variables: `*_AWS_PROFILE`, `*_AWS_ACCESS_KEY_ID`, `*_AWS_SECRET_ACCESS_KEY`,
`*_AWS_SESSION_TOKEN`, `*_S3_ANON`, `*_S3_ENDPOINT_URL` and `*_STORAGE_OPTIONS` (JSON).
Unprefixed `AWS_PROFILE` and `S3_ENDPOINT_URL` apply to both stores. Access is anonymous
unless credentials are given. This works the same way as in
[dummy-mlwp-model](https://github.com/dmidk/dummy-mlwp-model#remote-stores).

The run reads only the variables whose `standard_name` the model uses (plus their grid
mapping), and only the model's input window, ending at the reference time. It writes
one variable per input variable, with dimensions `(ensemble_member, time, y, x)`. The
`time` coordinate holds valid times, alongside `lead_time` and `forecast_reference_time`
coordinates. The model's name, version and sha256 are recorded in the store's attributes.

The log starts with the version. Next comes the effective configuration, one line per
setting with defaults included and credentials never shown, then the bundle's manifest.
At the end the log reports timings, and peak GPU memory on CUDA. A misspelt variable such
as `FORECAST_STEP` is logged as a warning that suggests the right name. All configuration
problems are reported together, in a single error.

| Exit code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Anything unexpected (traceback logged) |
| 2 | Configuration error |
| 3 | Input error: the store is missing, or does not match what the model expects |
| 4 | Device error: `DEVICE=cuda` with no usable GPU |
| 5 | Storage error: a store could not be reached, read or written |
| 6 | Model error: the bundle is missing or corrupt (rebuild the image) |

### `serve`

`serve` is not implemented yet and exits 2.

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
