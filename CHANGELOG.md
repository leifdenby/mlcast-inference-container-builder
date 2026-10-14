# CHANGELOG

## Unreleased

The first prototype: a trained mlcast model (checkpoint + training config) in, a
ready-to-run inference image out.

### Added

- Model loading from an mlcast checkpoint and its training config, with an inference spec
  (standard names, input/forecast steps, timestep, normalisation) embedded in the
  checkpoint, and a converter for FBK `convgru_ensemble` checkpoints such as IRENE
  (`mlcast_icb.upstream`, code proposed for mlcast PR by PR).
- An xarray-level `forecast()` with CF axis detection, padding to the network's divisor,
  clamping like training does, and seedable ensembles.
- `mlcast-icb bundle` and `info`: a verified model bundle (`model.ckpt` without optimizer
  state, plus `manifest.json`).
- `mlcast-icb run`: batch forecasts from zarr to zarr, local or on S3, with documented
  exit codes.
- `mlcast-icb serve`: an HTTP server compatible with the IRENE API, adding CF output,
  ensemble products, float16 packing, seeds, readiness and Prometheus metrics.
- `run` and `serve` are configured with the convention of DMI's MLWP inference containers
  (as in `dmidk/mlwp-inference-container-anna`, whose `envargs` module is copied as
  `mlcast_icb.envargs`): every option `--some-option` is also the environment variable
  `MLCAST_SOME_OPTION`; the command line wins over the variable, which wins over the
  default; the startup log shows every option with its variable, value and source; a set
  `MLCAST_*` variable that is an option of neither command is an error (exit 2) with a
  did-you-mean, while one of the other command is ignored. This replaces the unprefixed
  variables (`INPUT_ZARR`, `MODEL_DIR`, `DEVICE`, `PORT`, `MAX_QUEUE`, …) and the
  warn-only typo check.
- Per-store S3 options `--{input,output}-aws-profile`, `--{input,output}-s3-endpoint-url`,
  `--{input,output}-s3-anon` and `--{input,output}-storage-options`, replacing the
  `SRC_*`/`DST_*` variables. Credentials come only from the standard AWS configuration
  (`AWS_*` keys, `AWS_PROFILE`, a profile, an IAM role), so the per-store key variables
  and the unprefixed `S3_ENDPOINT_URL` are gone; `AWS_ENDPOINT_URL[_S3]` still apply
  through botocore. Images set `MLCAST_MODEL_DIR=/model`, and GPU images
  `MLCAST_DEVICE=cuda`.
- `mlcast-icb build-base` and `build`: the runtime base image per torch variant (`cpu`,
  `gpu-cu126`, `gpu-cu128`, `gpu-cu130`) from `uv.lock`, and model images as one layer on
  top of it. `build` warns when the bundle was made with another mlcast than the local
  base image carries.
- CI (lint, tests on Python 3.11 and the container's Python, the IRENE tests on main and
  weekly, the Docker tests when the image's inputs change), base image publishing to
  ghcr.io on version tags, and Dependabot for GitHub Actions, uv and the Docker bases.
