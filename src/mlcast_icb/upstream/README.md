# `mlcast_icb.upstream` — code meant to move into mlcast

This subpackage holds every piece of the inference container builder that would be
better placed in [mlcast](https://github.com/mlcast-community/mlcast) itself. It is
written as it would look inside mlcast — same code style (numpy docstrings,
jaxtyping/beartype), importing only from mlcast and its dependencies — so each module
can be lifted into a PR with minimal changes.

Rules for this subpackage:

- **The rest of the builder only reaches mlcast through here.** `run`, `serve` and
  `build` import from `mlcast_icb.upstream`, never from mlcast internals directly.
- **One module per proposed PR (or per part of one).** Tests for `upstream/<module>.py`
  live in `tests/upstream/test_<module>.py` and are written to move into mlcast's `tests/`
  unchanged.
- **When a PR is merged** and the builder's pinned mlcast includes it, the module here
  becomes a one-line re-export (e.g. `from mlcast.inference import forecast`). It is
  deleted once nothing in the builder imports it.

These PRs will be opened only once the builder works end to end as a prototype. Until
then, fixes such as PR H are applied here, inside the builder.

File and line references below are to mlcast at commit `5edce07`.

## Overview

| PR | Title | Modules here | Depends on | Builder workaround it removes |
| --- | --- | --- | --- | --- |
| H | Fix: pad with "no rain", not 30 dBZ | none (one-line fix in mlcast) | — | Padding in physical units before calling the network |
| C | Shared pre/post-processing | `processing.py` | — | Its own copy of the NaN/normalise/clamp logic |
| A | Self-describing checkpoints | `spec.py`, `checkpoint.py` | — | `--config` argument, loose YAML parsing, its own manifest format |
| B | `mlcast export` | `export.py` | A | Shipping full training checkpoints (optimizer state) in images |
| D | xarray `forecast()` + seedable ensembles | `forecast.py` | C | Its whole model/xarray layer; global-RNG seeding under a lock |
| E | Light inference install | none (packaging) | — | Training-only dependencies in the runtime image |
| F | Re-loadable training configs | `config_check.py` | — (dropped if A lands) | Workarounds for YAMLs that reference `__main__` or removed modules |
| G | `convgru_ensemble` checkpoint import | `convert_convgru_ensemble.py` | A (optional) | The IRENE conversion fixture |

Suggested order: **H** (a bug fix, one line), **C** (small, standalone), then **A → B**,
then **D**. **E** can go in at any time. **F** only matters if A is not adopted. **G** is
optional.

---

## PR H — Fix: `ConvGruModel` pads with 30 dBZ instead of "no rain"

**Module:** none. This is a fix in `mlcast/models/convgru.py:388`.

**Why.** `ConvGruModel.forward` pads H and W up to a multiple of `2**num_blocks` with
`value=0.0`. The input is already normalised at that point, so 0.0 is the middle of
[-1, 1]: 30 dBZ, about 2.7 mm/h. The padded strip therefore looks like a band of moderate
rain along the bottom and right edges, and the recurrent layers spread it into the domain.
Training never hits this, because 256×256 crops need no padding. Inference on a full
domain does.

Measured on IRENE's `sample_data.nc` (1400×1200, padded to 1408×1216), compared with
padding by 0 mm/h before normalising (which is what IRENE's own `predict()` does), the
forecast differs by up to **3.19 mm/h**.

**Change.** Pad with "no rain" in normalised space. The network does not know the scaling,
so either take a `pad_value` argument (default `-1.0`, set by the caller from
`scaling.norm_min`), or move padding out of the network into `preprocess` (PR C), which
knows the scaling.

**Tests.** A forecast on an odd-sized domain matches, inside the original domain, the same
forecast with the input pre-padded with 0 mm/h.

---

## PR C — Shared pre/post-processing

**Module:** `processing.py` → `mlcast/data/processing.py`

**Why.** Training and `predict()` each write their own preprocessing. They agree
numerically today, but only by coincidence of the current constants, and they already
differ in one place:

- *NaN handling.* `predict()` fills NaN with 0 before normalising
  (`nowcasting_module.py:294`); training normalises, then fills with -1
  (`data/source_data_datasets.py:315`). These agree because 0 mm/h normalises to exactly
  -1 (log10(0) → -inf, clipped to the 0 dBZ floor). They would diverge if `dbz_floor` were
  changed, or for `equivalent_reflectivity_factor` with a non-zero floor. They already
  diverge for negative rain rates (e.g. negative-valued no-data or noise): log10 of a
  negative number is NaN, which `np.clip` keeps, so training fills it with -1 but
  `predict()` (which filled NaN *before* normalising) passes NaN into the network.
- *Clamping.* Training clamps network output to [-1, 1] before the loss
  (`nowcasting_module.py:126`); `predict()` does not (`nowcasting_module.py:305-310`), so
  outputs above 1 denormalise to more than 60 dBZ (above ~205 mm/h).
- *Default variable.* `predict()` defaults to `rainfall_rate`, the training config to
  `rainfall_flux`. These are numerically identical today, because flux is scaled with the
  rain-rate Z-R relation (`data/normalization.py:96-120`), but the defaults should not
  differ.

**Proposed API.**

```python
def preprocess(
    arrays: Sequence[np.ndarray],          # one (T, H, W) array per channel, physical units
    standard_names: Sequence[str],
    scaling: ReflectivityScaling = DEFAULT_SCALING,
    return_valid_mask: bool = False,       # also return the (T, C, H, W) pre-fill validity mask
) -> np.ndarray:                           # (T, C, H, W) float32, normalised, NaN filled
    ...

def postprocess(
    preds: torch.Tensor,                   # (B, T, C*E, H, W) network output, members member-major
    standard_names: Sequence[str],
    scaling: ReflectivityScaling = DEFAULT_SCALING,
) -> np.ndarray:                           # (B, E, T, C, H, W), clamped, denormalised
    ...
```

The NaN fill value is defined once, as `scaling.norm_min`, rather than as a literal `-1.0`
in one place and a physical `0` in the other. Each channel uses the `scaling` instance's
own `normalization_registry` / `denormalization_registry`, so a custom scaling is honoured.

`ConvGruModel.forward` concatenates whole `(B, T, C, H, W)` member outputs on dim 2
(`models/convgru.py:397-403`), so index `e * C + c` is channel `c` of member `e`.
`postprocess` unpacks that into `(B, E, T, C, H, W)`; `predict()`'s `(E, T, H, W)` is
`out[0, :, :, 0]`. (`predict()` itself returns `(C*E, T, H, W)` and only matches its
docstring for `C = 1`.)

**Changes in mlcast.**

- New `mlcast/data/processing.py`.
- `SourceDataDatasetBase._build_sample`, `SourceDataIndexedDataset.__getitem__` and
  `SourceDataRandomSamplingDataset.__getitem__` (`data/source_data_datasets.py:283-332`,
  `:495-504`, `:635-644`) call `preprocess`. The validity mask comes from
  `return_valid_mask=True`: training computes it on the *normalised* data before filling,
  so values the conversion maps to NaN (negative rain rates) are masked as well as NaN
  inputs; a mask computed from the raw data would differ there.
- `NowcastLightningModule.predict` (`nowcasting_module.py:260-315`) calls `preprocess` and
  `postprocess`, and gains the clamp.

**Tests.** `preprocess` matches the current training path bit-for-bit on random data with
NaNs; `postprocess(preprocess(x))` round-trips within the [0, 60] dBZ range; `predict()`
output never exceeds the 60 dBZ rain rate.

---

## PR A — Self-describing checkpoints

**Modules:** `spec.py` → `mlcast/inference/spec.py`; `checkpoint.py` → hooks in
`mlcast/nowcasting_module.py`

**Why.** A checkpoint alone cannot rebuild its model.
`save_hyperparameters("ensemble_size", "loss_class", "loss_params", "masked_loss")`
(`nowcasting_module.py:61`) leaves out `network`, so `from_checkpoint`
(`nowcasting_module.py:236-258`) fails unless the caller already has the network. Inference
therefore needs the training YAML too, and saved YAMLs can reference classes that are no
longer importable (`__main__.CombinedDatasetFactory`, the removed
`mlcast.sampling.samplers`). The data-side facts inference needs (`standard_names`,
`input_steps`, `forecast_steps`, timestep) live only in `cfg.data`.

**Proposed API.**

```python
@dataclass(frozen=True)
class InferenceSpec:
    standard_names: list[str]
    units: list[str]
    scaling: ReflectivityScaling        # per model; per channel if that is ever needed
    input_steps: int
    forecast_steps: int
    timestep: str                       # ISO 8601 duration, e.g. "PT5M"
    spatial_divisor: int                # 2**num_blocks for ConvGru
    default_ensemble_size: int
    network_config: str                 # Fiddle YAML of cfg.pl_module.network
    mlcast_version: str
    mlcast_commit: str | None

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceSpec": ...

def spec_from_config(cfg: fdl.Config) -> InferenceSpec:
    """Read the spec from a training config, touching cfg.data only as plain values."""

# checkpoint.py
def save_inference_metadata(checkpoint: dict[str, Any], spec: InferenceSpec) -> None: ...
def load_model(checkpoint_path: str, config_path: str | None = None,
               device: str = "cpu") -> tuple[NowcastLightningModule, InferenceSpec]: ...
```

`load_model` uses the spec stored in the checkpoint when it is there, and otherwise falls
back to the YAML (building only `cfg.pl_module.network`, never the full `Experiment`). The
fallback keeps existing checkpoints working.

**Changes in mlcast.**

- New `mlcast/inference/spec.py`.
- `NowcastLightningModule` gains `on_save_checkpoint` (writes `checkpoint["mlcast_inference"]
  = spec.to_dict()`) and a `from_checkpoint` that rebuilds `network` from the stored
  config when it is not passed in.
- `train_from_config` (`config/orchestrator.py:123`) builds the spec from `cfg` and hands it
  to the module before `fit`.

**Tests.** Train one step on synthetic data; `from_checkpoint(path)` with no other
argument reloads strictly and gives identical output; an old-style checkpoint still loads
through `load_model(ckpt, config)`.

---

## PR B — `mlcast export` (needs A)

**Module:** `export.py` → `mlcast/inference/export.py` + `export` subcommand in
`mlcast/__main__.py` (next to `train`, `build-sampling-index`, `validate-sampling-index`,
`__main__.py:408-473`)

**Why.** Training checkpoints carry optimizer and scheduler state that inference never
uses. IRENE's 770 MB checkpoint very likely includes Adam's two moment buffers. Shipping
them makes images larger and slower to pull for no benefit. Exporting is also the natural
place to prove a model is loadable.

**Proposed API.**

```sh
mlcast export path/to/model.ckpt --out bundle/ [--config config.yaml]
```

`bundle/` contains `weights.safetensors` (or a stripped `.ckpt`) and `spec.json`
(`InferenceSpec.to_dict()`). Before writing, export reloads the weights strictly into a
freshly built network and runs one forward pass at `(1, input_steps, C, d, d)`, where `d`
is `spatial_divisor`. A model that cannot do this is never exported.

**Tests.** Export a tiny model; the bundle is much smaller than the checkpoint, reloads
strictly and gives identical output.

---

## PR D — xarray `forecast()` and seedable ensembles (needs C)

**Module:** `forecast.py` → `mlcast/inference/forecast.py`; plus a change in
`mlcast/models/convgru.py`

**Why.** The only inference entry point is `predict()`: one variable, a `(T, H, W)` tensor
in, a bare numpy array out, with no coordinates, times or metadata. Every consumer
(this builder, notebooks, verification tools) has to rebuild the xarray plumbing around it.
Separately, the noisy decoder draws from the global RNG (`torch.randn`,
`models/convgru.py:400` and `:405`), so a reproducible ensemble needs global seeding, which
a multi-threaded server can only do under a lock.

**Proposed API.**

```python
def forecast(
    model: NowcastLightningModule,
    spec: InferenceSpec,
    ds: xr.Dataset,
    forecast_steps: int | None = None,   # default: spec.forecast_steps
    ensemble_size: int | None = None,    # default: spec.default_ensemble_size
    seed: int | None = None,
) -> xr.Dataset:
    ...
```

It covers:

- finding time/y/x by CF attributes, reusing `_detect_axes`
  (`data/source_data_datasets.py:84-127`), and selecting variables by standard name;
- checking the time spacing against `spec.timestep`, and taking the last `input_steps`;
- unit checks/conversion; `preprocess`/`postprocess` from PR C;
- padding to `spec.spatial_divisor` and cropping back;
- CF output: dims `(ensemble_member, time, y, x)`, with valid `time`,
  `forecast_reference_time`, `lead_time`, the input's x/y coordinates and `grid_mapping`
  carried through, and `standard_name`/`units` on each variable.

In `ConvGruModel.forward` (`models/convgru.py:362`), add `generator: torch.Generator | None
= None` and pass it to the `torch.randn` calls. Until that lands, `upstream/forecast.py`
seeds inside `torch.random.fork_rng()` with `torch.manual_seed(seed)`.

**Tests.** Output has the expected dims, CF coordinates and attributes; the same seed gives
identical ensembles and different seeds give different ones; an odd-sized domain is padded
and cropped back to its input shape.

---

## PR E — Light inference install

**Module:** none (packaging). `tests/upstream/test_imports.py` asserts that importing
`mlcast_icb.upstream` does not import `mlflow`, `tensorboard` or
`mlcast.config.orchestrator`, so regressions are caught before the PR exists.

**Why.** Core dependencies include mlflow, tensorboard, ipykernel and their extras
(`pyproject.toml:34-55`). None are needed to run a model, and all of them end up in every
inference image, adding size, cold-start time and CVEs to track.

**Changes in mlcast.**

- Move mlflow, tensorboard, ipykernel and plotting libraries to a `train` extra (or keep
  them in core and add a slimmer `inference` install; whichever the maintainers prefer).
- Make sure `mlcast.inference`, `mlcast.data.normalization`, `mlcast.models` and
  `mlcast.config.loader` import without pulling in `mlcast.config.orchestrator`, the
  loggers or the samplers. Move imports inside functions where needed.

**Tests.** A CI job installs the inference set only and imports `mlcast.inference`.

---

## PR F — Re-loadable training configs (only if A is not adopted)

**Module:** `config_check.py` → the end of `train_from_config`
(`config/orchestrator.py:123-151`)

**Why.** The YAML written at training time (`dump_yaml`, `config/orchestrator.py:79`) is
not guaranteed to load again: it can reference objects defined in `__main__`, and nothing
records which mlcast version wrote it.

**Proposed change.** Right after writing the YAML, load it back with `load_yaml_config`,
fail early on any `__main__` reference or unimportable class, and store the mlcast version
in the YAML. If PR A lands, inference no longer needs the YAML, and this PR is optional.

---

## PR G — `convgru_ensemble` checkpoint import (optional)

**Module:** `convert_convgru_ensemble.py` → `mlcast/inference/convert.py` +
`import-convgru-ensemble` subcommand

**Why.** IRENE (`it4lia/irene` on Hugging Face) was trained with FBK's `convgru_ensemble`,
whose `EncoderDecoder` has the same module tree as mlcast's `ConvGruModel`
(`models/convgru.py:333`) and uses the same normalisation (Z=200·R^1.6, 0–60 dBZ →
[-1, 1]). A converter makes such models usable in mlcast and is how the builder gets a
real-world integration-test model.

**What it does.** Read the checkpoint's hyperparameters (`input_channels`, `num_blocks`,
`noisy_decoder`, `ensemble_size`, `forecast_steps`), rename `model.*` state-dict keys to
`network.*`, drop optimizer state, and write an mlcast checkpoint with an `InferenceSpec`
(with PR A) or a checkpoint plus a matching Fiddle YAML (without it). Strict loading is
checked as part of the conversion.

**Verified on IRENE** (mlcast `5edce07`):
- Removing the `model.` prefix (bare network) or replacing it with `network.` (Lightning
  module) is enough. All 40 tensors load with `strict=True`.
- The raw network gives bit-identical output to FBK's code for the same input and seed.
  Both draw one `torch.randn` per member, in the same order.
- Of the 770 MB checkpoint, 513 MB is Adam state; the weights are 256.7 MB (float32).

**Known differences from IRENE's own `predict()`** (not caused by the conversion):
- IRENE maps the 0 dBZ floor back to 0.0365 mm/h; mlcast maps it to exactly 0.
- IRENE computes `log10(200·R^1.6 + 1e-16)`, mlcast `log10(a) + 10b·log10(R)`. The
  normalised inputs differ by float32 rounding (2.4e-7).
- IRENE pads with 0 mm/h; mlcast pads with 30 dBZ (see PR H).

Parity tests should therefore compare normalised network outputs, on a 256×256 crop (no
padding). Outside the floor, end-to-end mm/h agree to 5e-4.

**Tests.** Convert a tiny synthetic `convgru_ensemble`-shaped checkpoint; the result loads
strictly and matches the original forward pass for the same seed. The full IRENE
conversion runs as a slow, opt-in test.

**Data caveat.** IRENE's `examples/sample_data.nc` labels `RR` as `units: kg m-2 s-1` with
`standard_name: tprate`, but the values are mm/h (max 111). It also points at a
`grid_mapping` variable `crs` that is missing from the file. Anything that trusts the
attributes will read it wrongly; tests should override them.
