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
- `mlcast-icb run`: batch forecasts from zarr to zarr, local or on S3, configured through
  environment variables, with documented exit codes.
- `mlcast-icb serve`: an HTTP server compatible with the IRENE API, adding CF output,
  ensemble products, float16 packing, seeds, readiness and Prometheus metrics.
- `mlcast-icb build-base` and `build`: the runtime base image per torch variant (`cpu`,
  `gpu-cu126`, `gpu-cu128`, `gpu-cu130`) from `uv.lock`, and model images as one layer on
  top of it. `build` warns when the bundle was made with another mlcast than the local
  base image carries.
- CI (lint, tests on Python 3.11 and the container's Python, the IRENE tests on main and
  weekly, the Docker tests when the image's inputs change), base image publishing to
  ghcr.io on version tags, and Dependabot for GitHub Actions, uv and the Docker bases.
