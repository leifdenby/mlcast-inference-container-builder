"""Tests for the shared pre/post-processing (``processing.py``).

Written to move into mlcast's ``tests/data/test_processing.py`` unchanged, apart from the
import of the module under test.
"""

from collections.abc import Sequence

import numpy as np
import pytest
import torch

from mlcast.data.normalization import DEFAULT_SCALING, NORMALIZATION_REGISTRY, ReflectivityScaling
from mlcast.models.convgru import ConvGruModel
from mlcast.nowcasting_module import NowcastLightningModule
from mlcast_icb.upstream.processing import postprocess, preprocess

# Rain rate (mm/h) at the floor and ceiling of the default dBZ range (0 and 60 dBZ).
RR_FLOOR = float(DEFAULT_SCALING.reflectivity_to_rainfall_rate(np.array([1e-9]))[0])
RR_CEILING = float(DEFAULT_SCALING.reflectivity_to_rainfall_rate(np.array([DEFAULT_SCALING.dbz_ceiling]))[0])


class _RecordingNetwork(torch.nn.Module):
    """Stand-in network that records its input and returns a fixed output."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.seen_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, steps: int, ensemble_size: int = 1) -> torch.Tensor:
        self.seen_input = x.clone()
        return self.output


def _random_rain(rng: np.random.Generator, shape: tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    """Rain rates (mm/h) with zeros, NaNs and values beyond the 60 dBZ ceiling."""
    data = rng.exponential(scale=5.0, size=shape) * (rng.random(shape) < 0.6)
    data[rng.random(shape) < 0.01] = 500.0
    data[rng.random(shape) < 0.1] = np.nan
    return data.astype(dtype)


def _training_path(arrays: Sequence[np.ndarray], standard_names: Sequence[str]) -> np.ndarray:
    """Replicate the current training path: ``SourceDataIndexedDataset.__getitem__`` + ``_build_sample``."""
    channels = [NORMALIZATION_REGISTRY[name](array) for name, array in zip(standard_names, arrays, strict=True)]
    data = np.ascontiguousarray(np.swapaxes(np.stack(channels, axis=0), 0, 1), dtype=np.float32)
    return np.nan_to_num(data, nan=-1.0).astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preprocess_matches_training_path(dtype: type) -> None:
    """preprocess is bit-for-bit identical to the training data pipeline, NaNs and negatives included."""
    # Given rain rates and reflectivities with NaNs, zeros, negative and out-of-range values
    rng = np.random.default_rng(0)
    rain = _random_rain(rng, (5, 16, 12), dtype)
    rain[0, 0, :3] = -1.0
    reflectivity = rng.uniform(-20.0, 80.0, size=(5, 16, 12)).astype(dtype)
    reflectivity[rng.random(reflectivity.shape) < 0.1] = np.nan
    amount = _random_rain(rng, (5, 16, 12), dtype) / 12
    arrays = [rain, reflectivity, amount]
    standard_names = ["rainfall_rate", "equivalent_reflectivity_factor", "rainfall_amount"]

    # When preprocessing
    data, valid = preprocess(arrays, standard_names, return_valid_mask=True)

    # Then the result equals the training path exactly
    expected = _training_path(arrays, standard_names)
    assert data.dtype == np.float32
    assert data.shape == (5, 3, 16, 12)
    np.testing.assert_array_equal(data, expected)
    assert not np.isnan(data).any()
    # And the validity mask is what _build_sample computes before filling (NaN after normalising)
    channels = [NORMALIZATION_REGISTRY[name](array) for name, array in zip(standard_names, arrays, strict=True)]
    np.testing.assert_array_equal(valid, ~np.isnan(np.stack(channels, axis=1)))
    assert not valid[0, 0, 0, :3].any()


def test_preprocess_matches_predict_path() -> None:
    """preprocess gives the network the same input as NowcastLightningModule.predict."""
    # Given non-negative rain rates with NaNs, and a module whose network records its input
    rng = np.random.default_rng(1)
    rain = _random_rain(rng, (4, 8, 8))
    network = _RecordingNetwork(torch.zeros(1, 2, 1, 8, 8))
    module = NowcastLightningModule(network=network)

    # When predicting, and when preprocessing the same data
    module.predict(torch.from_numpy(rain), forecast_steps=2, ensemble_size=1, standard_name="rainfall_rate")
    data = preprocess([rain], ["rainfall_rate"])

    # Then the network input is identical
    assert network.seen_input is not None
    np.testing.assert_array_equal(network.seen_input.numpy(), data[np.newaxis])


def test_postprocess_matches_predict_in_range() -> None:
    """For in-range output, postprocess agrees with predict() for a single channel and ensemble."""
    # Given a network whose (B, T, E, H, W) output lies within [-1, 1]
    rng = np.random.default_rng(2)
    output = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(1, 3, 4, 8, 8)).astype(np.float32))
    module = NowcastLightningModule(network=_RecordingNetwork(output))
    past = torch.from_numpy(_random_rain(rng, (2, 8, 8)))

    # When predicting, and when postprocessing the same output
    expected = module.predict(past, forecast_steps=3, ensemble_size=4)
    out = postprocess(output, ["rainfall_rate"])

    # Then predict's (E, T, H, W) equals postprocess's (B, E, T, C, H, W) for B=0, C=0
    assert out.shape == (1, 4, 3, 1, 8, 8)
    np.testing.assert_array_equal(out[0, :, :, 0], expected)


def test_round_trip_within_dbz_range() -> None:
    """postprocess(preprocess(x)) recovers rain rates within the [0, 60] dBZ range."""
    # Given rain rates of exactly 0 or between the 0 and 60 dBZ rates
    rng = np.random.default_rng(3)
    rain = rng.uniform(RR_FLOOR * 1.01, RR_CEILING * 0.99, size=(6, 8, 8))
    rain[rng.random(rain.shape) < 0.3] = 0.0

    # When round-tripping through the network representation
    x = preprocess([rain], ["rainfall_rate"])
    out = postprocess(torch.from_numpy(x)[np.newaxis], ["rainfall_rate"])

    # Then the original rates come back, with exact zeros
    assert out.shape == (1, 1, 6, 1, 8, 8)
    np.testing.assert_allclose(out[0, 0, :, 0], rain, rtol=1e-5)
    np.testing.assert_array_equal(out[0, 0, :, 0] == 0, rain == 0)


def test_postprocess_clamps() -> None:
    """Output above norm_max gives at most the 60 dBZ rate; output below norm_min gives 0."""
    # Given network output beyond both ends of the normalised range
    preds = torch.tensor([1.0, 1.5, 10.0, -1.0, -1.5, -10.0]).reshape(1, 1, 1, 1, 6)

    # When postprocessing
    out = postprocess(preds, ["rainfall_rate"]).ravel()

    # Then values are clamped to the [0, 60] dBZ rain-rate range
    np.testing.assert_allclose(out[:3], RR_CEILING, rtol=1e-5)
    assert out.max() <= np.float32(RR_CEILING) * (1 + 1e-6)
    np.testing.assert_array_equal(out[3:], 0.0)


def test_custom_scaling_is_honoured() -> None:
    """A custom ReflectivityScaling sets the normalisation, the NaN fill value and the clamp."""
    # Given a scaling with a different dBZ range and normalised range
    scaling = ReflectivityScaling(dbz_floor=-10.0, dbz_ceiling=70.0, norm_min=0.0, norm_max=1.0)
    rain = np.array([[[np.nan, 0.0, 1.0, 10.0]]], dtype=np.float32)

    # When preprocessing with it
    x = preprocess([rain], ["rainfall_rate"], scaling=scaling)

    # Then NaN becomes norm_min and the rest follow the custom scaling
    assert x[0, 0, 0, 0] == scaling.norm_min
    np.testing.assert_array_equal(x[0, 0, 0, 1:], scaling.rainfall_rate_to_normalized(rain[0, 0, 1:]))
    assert not np.array_equal(x, preprocess([rain], ["rainfall_rate"]))

    # And when postprocessing output above the custom norm_max
    out = postprocess(torch.tensor([[[[[0.5, 2.0]]]]]), ["rainfall_rate"], scaling=scaling).ravel()

    # Then it is clamped to the custom 70 dBZ ceiling
    np.testing.assert_allclose(out[1], scaling.normalized_to_rainfall_rate(np.array([1.0]))[0], rtol=1e-5)
    np.testing.assert_allclose(out[0], scaling.normalized_to_rainfall_rate(np.array([0.5]))[0], rtol=1e-5)


def test_postprocess_ensemble_channel_layout() -> None:
    """postprocess unpacks member-major (B, T, E*C, H, W) output into (B, E, T, C, H, W)."""
    # Given output for 2 channels x 3 members, each (member, channel) slice a distinct constant
    standard_names = ["rainfall_rate", "equivalent_reflectivity_factor"]
    n_members, n_channels = 3, len(standard_names)
    values = torch.linspace(-0.9, 0.9, n_members * n_channels).reshape(n_members, n_channels)
    preds = torch.empty(2, 4, n_members * n_channels, 5, 6)
    for e in range(n_members):
        for c in range(n_channels):
            preds[:, :, e * n_channels + c] = values[e, c]

    # When postprocessing
    out = postprocess(preds, standard_names)

    # Then each (member, channel) is denormalised with its own channel's conversion
    assert out.shape == (2, n_members, 4, n_channels, 5, 6)
    for e in range(n_members):
        rr = DEFAULT_SCALING.normalized_to_rainfall_rate(values[e, 0].numpy().reshape(1))[0]
        dbz = DEFAULT_SCALING.denormalize_reflectivity(values[e, 1].numpy())
        np.testing.assert_array_equal(out[:, e, :, 0], rr)
        np.testing.assert_array_equal(out[:, e, :, 1], dbz)


def test_convgru_stacks_members_member_major() -> None:
    """ConvGruModel puts member e's C channels at indices e*C .. e*C+C-1, as postprocess assumes."""
    # Given a small noisy-decoder ConvGru with 2 channels
    torch.manual_seed(0)
    model = ConvGruModel(input_channels=2, num_blocks=1, noisy_decoder=True).eval()
    x = torch.randn(1, 2, 2, 4, 4)

    # When generating 3 members at once, and the same 3 noise draws one member at a time
    with torch.no_grad():
        torch.manual_seed(1)
        ensemble = model(x, steps=2, ensemble_size=3)
        torch.manual_seed(1)
        members = [model(x, steps=2, ensemble_size=1) for _ in range(3)]

    # Then the ensemble output is the members concatenated on dim 2
    torch.testing.assert_close(ensemble, torch.cat(members, dim=2))


def test_postprocess_rejects_mismatched_channels() -> None:
    """A channel count that is not a multiple of the number of standard names is an error."""
    with pytest.raises(ValueError, match="not a multiple"):
        postprocess(torch.zeros(1, 1, 3, 2, 2), ["rainfall_rate", "equivalent_reflectivity_factor"])


def test_unknown_standard_name_is_rejected() -> None:
    """An unregistered standard name is an error in both directions."""
    with pytest.raises(ValueError, match="air_temperature"):
        preprocess([np.zeros((1, 2, 2))], ["air_temperature"])
    with pytest.raises(ValueError, match="air_temperature"):
        postprocess(torch.zeros(1, 1, 1, 2, 2), ["air_temperature"])
