"""Tests for the inference spec (``spec.py``).

Written to move into mlcast's ``tests/inference/test_spec.py`` unchanged, apart from the
import of the module under test and the location of the test data.
"""

import dataclasses
import json
from pathlib import Path

import fiddle as fdl
import pytest
from loguru import logger

import mlcast
from mlcast.config import training_experiment
from mlcast.config.loader import load_yaml_config
from mlcast.data.normalization import DEFAULT_SCALING, ReflectivityScaling
from mlcast.models.convgru import ConvGruModel
from mlcast_icb.upstream.spec import (
    UNITS,
    InferenceSpec,
    LooseConfig,
    load_config_loose,
    loads_config_loose,
    spec_from_config,
    to_buildable,
)

# A training config written by mlcast that references __main__.CombinedDatasetFactory and
# the removed mlcast.sampling.samplers module, and combines two source datasets.
VERSION_24_CONFIG = Path(__file__).parent.parent / "data" / "mlcast_version_24_config.yaml"


def _spec(**overrides: object) -> InferenceSpec:
    fields = {
        "standard_names": ("rainfall_rate",),
        "units": ("mm h-1",),
        "scaling": DEFAULT_SCALING,
        "input_steps": 6,
        "forecast_steps": 12,
        "timestep": "PT5M",
        "spatial_divisor": 32,
        "default_ensemble_size": 10,
        "network_config": "!fdl.Config\n",
        "mlcast_version": "0.1.0",
        "mlcast_commit": None,
    }
    return InferenceSpec(**{**fields, **overrides})


def test_spec_from_trained_model_config(tiny_trained_model) -> None:
    """The spec of a config written by training holds the data and network facts."""
    # Given the config.yaml written while training a tiny model
    cfg = load_config_loose(tiny_trained_model.config_path)

    # When building the spec
    spec = spec_from_config(cfg)

    # Then it describes the model and its data
    assert spec.standard_names == ("rainfall_rate",)
    assert spec.units == ("mm h-1",)
    assert spec.scaling == DEFAULT_SCALING
    assert spec.input_steps == tiny_trained_model.input_steps
    assert spec.forecast_steps == tiny_trained_model.forecast_steps
    assert spec.timestep is None
    assert spec.spatial_divisor == 4
    assert spec.default_ensemble_size == 2
    assert spec.mlcast_version == mlcast.__version__
    # And its network config builds the network
    network = fdl.build(to_buildable(loads_config_loose(spec.network_config)))
    assert isinstance(network, ConvGruModel)
    assert network.num_blocks == 2


def test_spec_is_independent_of_how_the_config_was_loaded(tiny_trained_model) -> None:
    """Loose loading, mlcast's load_yaml_config and the in-memory config give the same spec."""
    # Given the same config loaded loosely, loaded with mlcast's loader, and rebuilt in memory
    loose = load_config_loose(tiny_trained_model.config_path)
    strict = load_yaml_config(tiny_trained_model.config_path)
    in_memory = training_experiment.as_buildable()
    in_memory.pl_module.network = fdl.Config(ConvGruModel, input_channels=1, num_blocks=2)
    in_memory.data.dataset_factory.standard_names = ["rainfall_rate"]
    in_memory.data.dataset_factory.input_steps = tiny_trained_model.input_steps
    in_memory.data.dataset_factory.forecast_steps = tiny_trained_model.forecast_steps

    # When building a spec from each
    specs = [spec_from_config(cfg, timestep="PT5M") for cfg in (loose, strict, in_memory)]

    # Then all three are equal
    assert specs[0] == specs[1] == specs[2]


def test_spec_round_trips_through_json(tiny_trained_model) -> None:
    """to_dict gives plain JSON, and from_dict restores an equal spec."""
    # Given a spec with a custom scaling and a timestep
    spec = dataclasses.replace(
        spec_from_config(load_config_loose(tiny_trained_model.config_path), timestep="PT5M"),
        scaling=ReflectivityScaling(dbz_floor=-10.0),
    )

    # When round-tripping it through JSON
    restored = InferenceSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    # Then it is unchanged
    assert restored == spec
    assert restored.scaling.dbz_floor == -10.0


def test_real_training_config_with_unimportable_classes() -> None:
    """A real training config that load_yaml_config cannot load still gives a spec."""
    # Given a config referencing __main__ and the removed mlcast.sampling.samplers
    # Then mlcast's own loader fails on it
    with pytest.raises((ImportError, AttributeError)):
        load_yaml_config(VERSION_24_CONFIG)
    # And its data section cannot be built
    cfg = load_config_loose(VERSION_24_CONFIG)
    with pytest.raises(ImportError, match="__main__"):
        to_buildable(cfg.data.dataset_factory)

    # When building the spec, capturing warnings
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        spec = spec_from_config(cfg, timestep="PT5M")
    finally:
        logger.remove(sink)

    # Then the step counts and network come from the config
    assert spec.input_steps == 6
    assert spec.forecast_steps == 12
    assert spec.spatial_divisor == 32
    assert spec.default_ensemble_size == 2
    network_cfg = loads_config_loose(spec.network_config)
    assert network_cfg.fn_or_cls == "mlcast.models.convgru.ConvGruModel"
    assert network_cfg.arguments == {"input_channels": 1, "noisy_decoder": False, "num_blocks": 5}
    # And the combined datasets' differing standard names give the first one's, with a warning
    assert spec.standard_names == ("rainfall_amount",)
    assert spec.units == ("kg m-2",)
    assert any("different standard names" in m for m in messages)


def _loose_experiment(factories: dict[str, LooseConfig]) -> LooseConfig:
    network = LooseConfig("mlcast.models.convgru.ConvGruModel", {"num_blocks": 3})
    pl_module = LooseConfig("mlcast.nowcasting_module.NowcastLightningModule", {"network": network})
    data = LooseConfig(
        "mlcast.data.source_data_datamodule.SourceDataDataModule",
        {"dataset_factory": LooseConfig("__main__.Combined", {"dataset_factories": factories}, partial=True)},
    )
    return LooseConfig("mlcast.config.base.Experiment", {"pl_module": pl_module, "data": data})


def _factory(**arguments: object) -> LooseConfig:
    return LooseConfig("mlcast.data.source_data_datasets.SourceDataIndexedDataset", arguments, partial=True)


def test_combined_datasets_must_agree_on_steps() -> None:
    """Combined dataset factories with different step counts are an error."""
    # Given two factories with different input_steps
    cfg = _loose_experiment(
        {
            "a": _factory(standard_names=["rainfall_rate"], input_steps=6, forecast_steps=12),
            "b": _factory(standard_names=["rainfall_rate"], input_steps=4, forecast_steps=12),
        }
    )

    # Then building the spec fails, naming the field
    with pytest.raises(ValueError, match="input_steps"):
        spec_from_config(cfg)


def test_network_defaults_are_used() -> None:
    """Unset network arguments take the class defaults, and ensemble size defaults to 1."""
    # Given a config whose network sets only num_blocks and whose module sets no ensemble size
    cfg = _loose_experiment(
        {"only": _factory(standard_names=["equivalent_reflectivity_factor"], input_steps=4, forecast_steps=2)}
    )

    # When building the spec
    spec = spec_from_config(cfg)

    # Then the divisor follows num_blocks, and the other defaults apply
    assert spec.spatial_divisor == 8
    assert spec.default_ensemble_size == 1
    assert spec.units == (UNITS["equivalent_reflectivity_factor"],)


def test_unknown_standard_name_is_rejected() -> None:
    """A standard name without known units is an error."""
    cfg = _loose_experiment({"only": _factory(standard_names=["air_temperature"], input_steps=1, forecast_steps=1)})
    with pytest.raises(ValueError, match="air_temperature"):
        spec_from_config(cfg)


def test_missing_data_fields_are_rejected() -> None:
    """A dataset factory without standard names or step counts is an error."""
    with pytest.raises(ValueError, match="forecast_steps"):
        spec_from_config(_loose_experiment({"only": _factory(standard_names=["rainfall_rate"], input_steps=1)}))
    with pytest.raises(ValueError, match="standard_names"):
        spec_from_config(_loose_experiment({"only": _factory(input_steps=1, forecast_steps=1)}))


@pytest.mark.parametrize("timestep", ["PT5M", "PT1H", "P1D", "PT2.5S", "PT1H30M", None])
def test_valid_timesteps(timestep: str | None) -> None:
    """ISO 8601 durations, or None, are accepted as timestep."""
    assert _spec(timestep=timestep).timestep == timestep


@pytest.mark.parametrize("timestep", ["5min", "PT", "P", "PT5", "300"])
def test_invalid_timesteps(timestep: str) -> None:
    """Anything but an ISO 8601 duration is rejected as timestep."""
    with pytest.raises(ValueError, match="ISO 8601"):
        _spec(timestep=timestep)


def test_invalid_specs_are_rejected() -> None:
    """Mismatched units, non-positive counts and malformed dictionaries are errors."""
    with pytest.raises(ValueError, match="units"):
        _spec(units=("mm h-1", "dBZ"))
    with pytest.raises(ValueError, match="input_steps"):
        _spec(input_steps=0)
    with pytest.raises(ValueError, match="newer"):
        InferenceSpec.from_dict({**_spec().to_dict(), "format_version": 999})
    with pytest.raises(ValueError, match="unknown"):
        InferenceSpec.from_dict({**_spec().to_dict(), "colour": "blue"})
