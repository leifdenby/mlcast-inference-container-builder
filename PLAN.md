# mlcast-inference-container-builder — feasibility & plan

## Context
We want a Python tool that turns a trained mlcast-family nowcasting model into a
ready-to-run inference container image, with the user supplying only the checkpoint and the
training config. Two run modes, one image:
(1) batch: zarr in/out on S3, like `dummy-mlwp-model`; (2) an HTTP server, compatible
with and extending the IRENE server on Hugging Face (`it4lia/irene`, FBK), that takes a
netCDF and returns a forecast netCDF.

**Decisions:** scope is mlcast models only. One image carries both modes: build `run`
first, then `serve`. Upstream PRs to mlcast and other repos come **only after a full
working prototype**. Until then, all mlcast-facing code and fixes (e.g. PR H's padding)
live in `mlcast_icb/upstream/`, and the builder works with existing checkpoints. Each builder release pins one exact mlcast version.

## Verdict: both modes are feasible; they share one core
Both modes reduce to `xr.Dataset (past) -> model -> xr.Dataset (forecast)`. Only the I/O
frontend differs, and an image subcommand (`run` / `serve`) selects it. The real work is
**rebuilding the model from the checkpoint plus its config**.

## Model loading: mlcast checkpoint + Fiddle config only
- The checkpoint is not self-describing: `save_hyperparameters` keeps only 4 scalars
  (`mlcast/src/mlcast/nowcasting_module.py:61`), so the config YAML is required.
- Load the YAML with `mlcast.config.loader.load_yaml_config`, then build **only**
  `cfg.pl_module.network`. Saved YAMLs can reference `__main__` and removed modules, so
  never build the whole `Experiment`.
- Load the weights with `load_from_checkpoint(ckpt, network=net)`.
- Read `standard_names`, `input_steps` and `forecast_steps` from `cfg.data` as plain values.
- Normalisation comes from mlcast's `ReflectivityScaling`, keyed by standard name
  (`mlcast/src/mlcast/data/normalization.py:20`).
- The builder writes an **inference manifest** (network class + kwargs, standard names,
  input/forecast steps, timestep, default ensemble size, mlcast version, ckpt sha256).

## IRENE as the CI integration-test model
IRENE (`it4lia/irene`, BSD-2) is a `convgru_ensemble` checkpoint, not an mlcast one, so it
can't go through the builder as published. It can be converted once into an mlcast
ckpt + YAML, which is a test-fixture step, not a builder feature:
- Its `EncoderDecoder` has the same module tree as mlcast's `ConvGruModel`
  (`mlcast/src/mlcast/models/convgru.py:333`). Its hparams give
  `input_channels=1, num_blocks=5, noisy_decoder=true`, 6 input steps and 12 forecast steps.
- Normalisation is the same: Z=200·R^1.6, 0–60 dBZ mapped onto [-1, 1].
- `upstream/convert_convgru_ensemble.py` (PR G below) downloads the checkpoint, renames the
  `model.*` weight keys to `network.*`, and writes a matching Fiddle YAML (ConvGruModel,
  `rainfall_rate`, 6 in / 12 out).
- **Verified (2026-10-14):** after renaming the prefix, all 40 tensors load with
  `strict=True`, and the raw network output is bit-identical to FBK's code for the same
  seed. 513 of the 770 MB are Adam state; the weights are 256.7 MB. The differences that
  remain are in pre/post-processing (the floor value, float32 rounding, and the padding
  bug in PR H); see `src/mlcast_icb/upstream/README.md`, PR G.
- CI parity tests compare normalised outputs on a 256×256 crop. The full forecast on CPU
  takes ~0.5 s at 256×256 and ~8 s at 1400×1200.
- `sample_data.nc` is labelled `kg m-2 s-1` but holds mm/h, and its `crs` variable is
  missing; tests must override those attributes.
- Better long term: ask FBK to publish an mlcast-native export alongside the current file.
- CI tiers:
  - fast, on every PR: a tiny ConvGru trained for one step on synthetic data
  - slower, on main/tags: converted IRENE, with the 770 MB file cached between runs
- Parity check: our container's output against IRENE's own `predict()` on
  `examples/sample_data.nc`, with a fixed seed on CPU.

## Suggested mlcast changes that make the builder leaner
The builder works around each of these without them. Every one that lands upstream
deletes code from the builder, so they are ordered by how much each one removes.

1. **Self-describing checkpoints.** In `NowcastLightningModule.on_save_checkpoint`, store
   the serialized Fiddle config of `pl_module` (network included) and an inference spec.
   `from_checkpoint(path)` then rebuilds the model with no `network=` argument and no YAML.
   - Removes: the `--config` argument, loose YAML parsing, and the
     `__main__`/removed-module workarounds.
   - Today: `save_hyperparameters` keeps only 4 scalars (`nowcasting_module.py:61`),
     and `from_checkpoint` fails without `network` (`nowcasting_module.py:236-258`).
2. **An `InferenceSpec` in mlcast,** saved in the checkpoint. It is a dataclass holding:
   - `standard_names` with units, and the normalisation (`ReflectivityScaling` params) per channel
   - `input_steps`, `forecast_steps` and the timestep (e.g. `PT5M`)
   - the spatial divisor (`2**num_blocks`) and the default ensemble size
   - the mlcast version and git commit

   Today these live only in the data config (`cfg.data.dataset_factory`), or are implicit:
   normalisation is hard-coded by standard name in `data/normalization.py`.
   - Removes: the builder's own manifest format. The manifest becomes `InferenceSpec`.
3. **`mlcast export` CLI.** It writes a slim inference bundle: weights only, without
   optimizer or scheduler state, as safetensors or a stripped ckpt, plus the spec.
   - This makes images much smaller: IRENE's 770 MB ckpt very likely includes Adam state.
   - The builder's "build" step shrinks to "copy bundle onto base image".
   - It can validate at export time with a strict reload and a dummy forward pass, so a
     broken model never reaches the builder.
4. **xarray-level inference API:** `mlcast.inference.forecast(model, ds, forecast_steps=,
   ensemble_size=, seed=) -> xr.Dataset`. It covers:
   - CF axis detection (reuse `_detect_axes`, `source_data_datasets.py:84-127`)
   - selection by standard_name, unit conversion, padding to the divisor, and cropping back
   - CF output with time, `forecast_reference_time`, `lead_time` and the grid mapping carried through

   Removes: the builder's whole model/xarray layer. `run` and `serve` become pure I/O.
5. **One shared pre/post-processing function** for training and inference: NaN handling,
   normalisation, channel stacking, and output clamping. Today the two paths are written
   separately. They agree for ordinary inputs, but not at the edges:
   - NaN: `predict()` fills NaN with 0 mm/h before normalising (`nowcasting_module.py:294`).
     Training normalises first and then fills with -1 (`data/source_data_datasets.py:315`).
     0 mm/h normalises to exactly -1 (log10(0) → -inf, clipped to the 0 dBZ floor), so for
     NaN inputs the model sees the same values either way.
   - Negative values (found while implementing PR C): log10 of a negative rain rate is NaN,
     and `np.clip` keeps NaN. Training fills that NaN with -1, but `predict()` has already
     done its NaN fill, so **it feeds NaN into the network**. This is a real, if edge-case,
     bug in `predict()`.
   - The two would also diverge if `dbz_floor` were changed, or for
     `equivalent_reflectivity_factor` with a non-zero floor.
   - Default variable: `predict()` defaults to `rainfall_rate`, training to
     `rainfall_flux`. These give the same numbers, because flux is deliberately scaled with
     the rain-rate Z-R relation (with a warning, `normalization.py:96-120`).
   - Clamping is the one real difference. Training clamps network output to [-1, 1] before
     the loss (`nowcasting_module.py:126`), but `predict()` does not (`:305-310`).
     Unclamped outputs above 1 denormalise to more than 60 dBZ (above ~205 mm/h). Below -1
     they already become 0. Minor, but `forecast()` should clamp like training does.
6. **Seedable ensembles.** Accept a `torch.Generator` (or seed) in
   `ConvGruModel.forward`/`predict` for the noisy decoder, instead of the global RNG.
   - Removes: global-seed juggling around a lock in `serve`.
7. **A light `inference` dependency set.**
   - Move mlflow, tensorboard, ipykernel and plotting to a `train` extra.
   - Make sure `mlcast.inference` (and `mlcast.config.loader`) import without pulling in
     `mlcast.config.orchestrator`, the loggers or the samplers.
   - Result: a smaller base image, faster cold starts in mode 1, and fewer CVEs to track.
8. **Re-loadable training configs.** At the end of `train_from_config`
   (`config/orchestrator.py:123-151`), round-trip the saved YAML (`dump_yaml` → `load_yaml_config`).
   Fail early if it references `__main__` or anything unimportable, and write the mlcast version into it.
   - Only needed if item 1 is not adopted, since with 1 the YAML is no longer used for inference.
9. **Optional: a `convgru_ensemble` import utility** (`mlcast import-convgru-ensemble`),
   or an mlcast-native IRENE export published by FBK.
   - Moves the IRENE converter out of the builder's test fixtures.

### Where this code lives meanwhile: `mlcast_icb/upstream/`
Everything above that is code is written in its own subpackage, `mlcast_icb/upstream/`, as
it would look inside mlcast: mlcast's code style (numpy docstrings, jaxtyping/beartype),
importing only from mlcast. The rest of the builder (`run`, `serve`, `build`) never
touches mlcast internals directly. It imports only from `mlcast_icb.upstream`. When a PR
is merged and the pinned mlcast includes it, the matching module becomes a one-line
re-export (`from mlcast.inference import forecast`), and is deleted later.

`src/mlcast_icb/upstream/README.md` explains, PR by PR, what each module is for, where it
goes in mlcast, which mlcast code it replaces or touches, and the builder workaround it
makes unnecessary. The PRs below are independent unless a dependency is noted.

| PR | Items | Module in `upstream/` | Target in mlcast |
| --- | --- | --- | --- |
| **A. Self-describing checkpoints** | 1, 2 | `spec.py` (`InferenceSpec` dataclass, `spec_from_config(cfg)` reading `cfg.data` loosely); `checkpoint.py` (`load_model(ckpt, config=None)`, `save_inference_metadata(ckpt_dict, cfg)`) | `mlcast/inference/spec.py`; hooks `on_save_checkpoint`/`on_load_checkpoint` and a config-free `from_checkpoint` in `nowcasting_module.py`; `train_from_config` (`config/orchestrator.py`) hands the cfg to the module |
| **B. `mlcast export`** (needs A) | 3 | `export.py` (strip optimizer/scheduler state, write weights + spec, strict reload + dummy forward) | `mlcast/inference/export.py` + an `export` subcommand in `__main__.py` |
| **C. Shared pre/post-processing** (implemented) | 5 | `processing.py` (`preprocess(arrays, standard_names) -> np.ndarray (T, C, H, W)`, `postprocess(preds, standard_names) -> (B, E, T, C, H, W)`, with NaN fill and clamp) | `mlcast/data/processing.py`; `_build_sample` (`data/source_data_datasets.py:315`), both datasets' `__getitem__` (`:495-504`, `:635-644`) and `predict()` (`nowcasting_module.py:260`) call it |
| **D. xarray `forecast()` + seeding** (needs C) | 4, 6 | `forecast.py` (CF axis detection, selection by standard_name, padding/cropping, CF output coords, `seed=`) | `mlcast/inference/forecast.py`, reusing `_detect_axes` (`source_data_datasets.py:84-127`); `generator=` argument on `ConvGruModel.forward` (`models/convgru.py:362`). Until then `upstream/` uses `torch.random.fork_rng` + `manual_seed` |
| **E. Light inference install** | 7 | none (packaging only); a test in `upstream/` asserts that importing it does not pull in mlflow, tensorboard or `mlcast.config.orchestrator` | `pyproject.toml`: an `inference` core, with mlflow/tensorboard/ipykernel/plotting moved to a `train` extra; lazy imports where needed |
| **F. Re-loadable configs** (only without A) | 8 | `config_check.py` (round-trip `dump_yaml` → `load_yaml_config`, rejecting `__main__` references) | the end of `train_from_config` (`config/orchestrator.py:123-151`) |
| **G. convgru_ensemble import** (optional) | 9 | `convert_convgru_ensemble.py` (IRENE → mlcast ckpt + YAML) | `mlcast/inference/convert.py` + an `import-convgru-ensemble` CLI subcommand |

Suggested PR order: C (small, standalone), then A → B, then D, with E at any time. F is
dropped if A lands. Tests for each module live in `tests/upstream/test_<module>.py` and
are written to move into mlcast's `tests/` unchanged.

With 1–4 and 7 in place, the builder reduces to:
- a pinned base image (CUDA + torch + `mlcast[inference]`)
- `run` (zarr/S3 I/O, ported from dummy-mlwp) and `serve` (HTTP I/O), both calling
  `mlcast.inference.forecast`
- `mlcast-icb build bundle/ --tag …`, which is one `COPY` layer

Builder order: start against current mlcast, with all mlcast-facing code in
`mlcast_icb/upstream/`. Swap each module for a re-export once the pinned mlcast version
includes its PR.

## Mode 1 — batch, zarr on S3 (`run`): low risk
- Reuse `dummy-mlwp-model` patterns:
  - storage: SRC_/DST_ env vars, anonymous by default, credential masking (`src/dummy_mlwp/storage.py`)
  - env-var config with typo warnings (`envvars.py`) and exit codes (`errors.py`)
  - CF coordinate detection (`grid.py`)
  - forecastReferenceTime/leadTime output (`outputs.py`, `timeaxis.py`)
- Input follows the mlcast dataset spec. Variables are selected by CF `standard_name` via cf_xarray.
- Suited to K8s Jobs and Argo. The cost is a cold start (image pull + model load) on every run.

## Mode 2 — HTTP server (`serve`): IRENE-compatible, plus more
**Keep IRENE's API** so existing clients work unchanged:
- `GET /health`
- `GET /model/info`
- `POST /predict`: multipart netCDF upload. Query params `variable` (default `RR`),
  `forecast_steps`, `ensemble_size`. Returns `predictions.nc` with `precipitation_forecast(ensemble_member,
  forecast_step, y, x)` in mm/h, zlib compressed, plus the `X-Elapsed-Seconds` header.
- The same 413/422 validation errors.

**Add:**
1. **Correct concurrency.** IRENE's `async def` handlers run inference on the event loop,
   which blocks `/health` during a forecast. We use sync handlers in the threadpool, a single
   GPU lock, and a bounded queue that returns 429 when full, so health checks stay responsive.
2. **CF-complete output.**
   - Carry the input's x/y coordinates, `grid_mapping`/CRS and attributes through.
   - Add valid `time`, `forecast_reference_time` and `lead_time` coordinates, plus
     `standard_name`/`units` on the forecast variable.
   - A `format=irene` query param keeps the legacy bare layout.
3. **Smarter input handling.**
   - Find the variable by CF standard_name, falling back to `variable=`.
   - Check that the time spacing matches the model's 5-minute step.
   - Use the last `input_steps` timesteps (IRENE accepts any T ≥ 2).
   - Accept mm/h or kg m-2 s-1, converting by units.
   - Raw-body uploads (`Content-Type: application/x-netcdf`) work in addition to multipart,
     on both POST and PUT.
4. **Smaller responses on request.**
   - `products=` returns `members`, `mean`, `quantiles=0.1,0.5,0.9` and/or
     `exceedance=0.5,1,5` (mm/h probabilities) instead of every member.
   - `dtype=float16|float32` and `complevel=`.
5. **Reproducibility.** A `seed=` param for the noisy decoder. The response
   headers and attributes record model sha256, builder version and mlcast version.
6. **Operations.**
   - `/readyz`, separate from `/healthz`, reports ready only once the model is warm.
   - Prometheus `/metrics`, request IDs and structured logs.
   - The body-size limit and timeouts are configurable via env.
   - OpenAPI docs come with FastAPI.
7. **Later.**
   - An `output_uri=s3://…` param writes the result to S3 instead of returning it,
     bridging to mode 1's storage code.
   - Async `POST /jobs` + `GET /jobs/{id}` for long forecasts or large domains.

Implementation notes:
- Use FastAPI + uvicorn with one worker. Load the model once in `lifespan`, then
  warm it up with one dummy forward pass.
- Read the netCDF via `h5netcdf` from a spooled temp file, and stream the response from a
  temp file.
- Payload estimate: an Italian mosaic (~1400×1200, 6 steps) is ≈40 MB in. Ten members × 12
  steps is ≈800 MB uncompressed out, which is why `products=` and float16 matter.

## Build: "just give the checkpoint"
1. **Runtime base image per builder release.** It contains a CUDA base, torch, the pinned
   mlcast and this package's `run`/`serve`. It follows the `dummy-mlwp-model` Dockerfile
   recipe: uv, locked deps, torch in its own layer.
   - The pinned mlcast version lives in the builder's `uv.lock`. The README keeps a
     builder↔mlcast version table.
   - Bumping mlcast means cutting a new builder release.
2. **`mlcast-icb build --checkpoint m.ckpt --config config.yaml --tag reg/model:1.0`**
   - Write the manifest.
   - Smoke test on CPU: strict weight load, then one forward pass at the model's input shape.
     A failure stops the build, not the deployment.
   - The image is `FROM runtime-base` + one layer with `/model/{model.ckpt,manifest.json}`.
     It can be built with `docker build`, or by appending the layer without a Docker daemon.
   - Image labels record builder, mlcast and checkpoint versions.

## Package layout (new repo `mlcast-inference-container-builder`)
- `mlcast_icb/upstream/`: everything destined for mlcast (see the PR table), plus `README.md`.
  The manifest is the `InferenceSpec` from `upstream/spec.py`, written as `manifest.json`.
- `mlcast_icb/io/`: zarr/S3 (ported from dummy-mlwp `storage.py`), netCDF bytes
- `mlcast_icb/run.py` (mode 1), `mlcast_icb/serve.py` (mode 2)
- `mlcast_icb/build.py`, plus `Dockerfile.base` and the build-context template
- `tests/`:
  - a tiny ConvGru trained for one step with mlcast, giving a ckpt + YAML fixture
  - `tests/upstream/`: one test file per upstream module, written to be portable to mlcast
  - converted IRENE (`upstream/convert_convgru_ensemble.py`) as the slow integration test (770 MB)

## Verification
- Unit: build a manifest from the tiny fixture, then strict-load it.
- IRENE parity: our loaded model against IRENE's `predict()` on `examples/sample_data.nc`,
  with a fixed seed on CPU, should agree to float tolerance.
- `mlcast-icb build` on the converted IRENE ckpt + YAML, then:
  - `docker run … serve` and `curl -F file=@sample_data.nc localhost:8000/predict`. Check
    that the IRENE layout still works, and that the CF output has time, lead_time and CRS.
  - `docker run … run` against a local zarr, and against MinIO for S3.
- An IRENE API-compatibility test: replay IRENE's own `tests/test_serve.py` cases against our server.
