"""Inference spec: everything needed to run a trained model, apart from its weights.

A training checkpoint holds only the network weights and four scalar hyperparameters,
so on its own it cannot rebuild its network, and it does not record which variables,
how many input steps or which timestep the model was trained on. That information
lives in the training config (``cfg.pl_module.network`` and ``cfg.data``).
:class:`InferenceSpec` collects it in one JSON-serialisable place, so it can be stored
in the checkpoint itself (see ``checkpoint.py``) or written next to it.

Proposed for mlcast as ``mlcast/inference/spec.py`` (see ``README.md``, PR A).

Loading training configs loosely
--------------------------------
YAML configs written by training (``dump_yaml``, ``config/orchestrator.py:79``) are not
guaranteed to load again with ``mlcast.config.loader.load_yaml_config``: they can
reference classes defined in ``__main__`` (e.g. a ``CombinedDatasetFactory`` from a
training script) or in modules that no longer exist (``mlcast.sampling.samplers``).
``load_yaml_config`` imports every class in the file and fails on the first such
reference. Importing ``mlcast.config.loader`` also imports ``mlcast.config.orchestrator``
and with it mlflow and the other training loggers.

:func:`load_config_loose` therefore reads the YAML without importing anything: every
``!fdl.Config`` / ``!fdl.Partial`` node becomes a :class:`LooseConfig` that records the
class name and arguments as plain data. Only the subtree that is actually built
(``pl_module.network``) is resolved to a real ``fdl.Config`` with :func:`to_buildable`.
"""

import dataclasses
import importlib
import json
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import fiddle as fdl
import yaml
from fiddle.experimental.yaml_serialization import dump_yaml
from loguru import logger

import mlcast
from mlcast.data.normalization import DEFAULT_SCALING, ReflectivityScaling

#: Units of each supported CF standard name, as the normalisation in
#: ``mlcast.data.normalization`` expects them.
UNITS: dict[str, str] = {
    "rainfall_rate": "mm h-1",
    "rainfall_flux": "kg m-2 s-1",
    "rainfall_amount": "kg m-2",
    "equivalent_reflectivity_factor": "dBZ",
}

#: Version of the dictionary layout written by :meth:`InferenceSpec.to_dict`.
SPEC_FORMAT_VERSION = 1

# ISO 8601 duration with at least one component, e.g. "PT5M", "PT1H", "P1D", "PT2.5S".
_ISO_8601_DURATION = re.compile(r"^P(?!$)(\d+W)?(\d+D)?(T(?=\d)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?$")


# --------------------------------------------------------------------------------------
# Loose YAML loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LooseConfig:
    """A ``!fdl.Config`` or ``!fdl.Partial`` node read without importing its callable.

    Arguments are readable as attributes, like on a ``fdl.Config``, so code that only
    reads values (``cfg.data.dataset_factory.input_steps``) works on either.

    Parameters
    ----------
    fn_or_cls : str
        Fully qualified name of the callable, ``"module.QualName"``.
    arguments : dict of str to Any
        The node's arguments, with nested nodes as ``LooseConfig`` / ``LooseType``.
    partial : bool, optional
        ``True`` for a ``!fdl.Partial`` node. Default is ``False``.
    """

    fn_or_cls: str
    arguments: dict[str, Any]
    partial: bool = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self.arguments[name]
        except KeyError:
            raise AttributeError(f"{self.fn_or_cls} config has no argument {name!r}") from None


@dataclass(frozen=True)
class LooseType:
    """A ``!type`` node (a Python class used as a value) read without importing it.

    Parameters
    ----------
    name : str
        Fully qualified name of the class, ``"module.QualName"``.
    """

    name: str


class _LooseLoader(yaml.SafeLoader):
    """YAML loader that reads Fiddle tags as plain data instead of importing classes."""


def _construct_loose(loader: _LooseLoader, node: yaml.MappingNode, partial: bool) -> LooseConfig:
    mapping = loader.construct_mapping(node, deep=True)
    fn_or_cls_ref = mapping.pop("__fn_or_cls__")
    return LooseConfig(
        fn_or_cls=f"{fn_or_cls_ref['module']}.{fn_or_cls_ref['name']}",
        arguments=mapping,
        partial=partial,
    )


def _construct_loose_config(loader: _LooseLoader, node: yaml.MappingNode) -> LooseConfig:
    return _construct_loose(loader, node, partial=False)


def _construct_loose_partial(loader: _LooseLoader, node: yaml.MappingNode) -> LooseConfig:
    return _construct_loose(loader, node, partial=True)


def _construct_loose_type(loader: _LooseLoader, node: yaml.ScalarNode) -> LooseType:
    return LooseType(name=loader.construct_scalar(node))


def _construct_tagged_value(loader: _LooseLoader, node: yaml.MappingNode) -> None:
    raise NotImplementedError("fdl.TaggedValue nodes are not supported in mlcast configs.")


_LooseLoader.add_constructor("!fdl.Config", _construct_loose_config)
_LooseLoader.add_constructor("!fdl.Partial", _construct_loose_partial)
_LooseLoader.add_constructor("!type", _construct_loose_type)
_LooseLoader.add_constructor("!fdl.TaggedValue", _construct_tagged_value)


# The inverse of ``_LooseLoader``'s ``!type`` constructor. ``mlcast.config.orchestrator``
# registers the same representer on import; it is repeated here so that dumping a config
# that contains a class does not require importing the orchestrator (and the loggers).
def _type_representer(dumper: yaml.SafeDumper, data: type) -> yaml.ScalarNode:
    return dumper.represent_scalar("!type", f"{data.__module__}.{data.__qualname__}")


yaml.SafeDumper.add_representer(type, _type_representer)


def load_config_loose(path: str | Path) -> LooseConfig:
    """Read a Fiddle YAML config file as plain data, without importing any of its classes.

    Parameters
    ----------
    path : str or Path
        Path to a YAML file written by ``dump_yaml``.

    Returns
    -------
    cfg : LooseConfig
        The root node; see :func:`loads_config_loose`.
    """
    return loads_config_loose(Path(path).read_text())


def loads_config_loose(text: str) -> LooseConfig:
    """Read Fiddle YAML text as plain data, without importing any of its classes.

    Parameters
    ----------
    text : str
        YAML written by ``dump_yaml``.

    Returns
    -------
    cfg : LooseConfig
        The root node. Nested ``!fdl.Config`` / ``!fdl.Partial`` nodes are
        ``LooseConfig`` and ``!type`` values are ``LooseType``.

    Raises
    ------
    ValueError
        If the root of the YAML is not a ``!fdl.Config`` or ``!fdl.Partial`` node.
    """
    cfg = yaml.load(text, Loader=_LooseLoader)  # noqa: S506 — _LooseLoader is a SafeLoader that imports nothing
    if not isinstance(cfg, LooseConfig):
        raise ValueError(f"Expected a !fdl.Config or !fdl.Partial at the root of the YAML, got {type(cfg).__name__}")
    return cfg


def _resolve(qualified_name: str) -> Any:
    """Import ``"module.QualName"``, trying successively shorter module paths."""
    parts = qualified_name.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name, attrs = ".".join(parts[:split]), parts[split:]
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        try:
            for attr in attrs:
                obj = getattr(obj, attr)
        except AttributeError:
            break
        return obj
    raise ImportError(f"Cannot import {qualified_name!r}")


def to_buildable(value: Any) -> Any:
    """Resolve a loosely loaded value into real Fiddle configs, importing its classes.

    Parameters
    ----------
    value : Any
        A ``LooseConfig``, ``LooseType``, or a list/tuple/dict containing them, or a
        plain value (returned unchanged).

    Returns
    -------
    value : Any
        The same structure with every ``LooseConfig`` replaced by an ``fdl.Config`` /
        ``fdl.Partial`` and every ``LooseType`` by the class it names.

    Raises
    ------
    ImportError
        If a referenced class cannot be imported (e.g. it lives in ``__main__`` or in a
        module that no longer exists).
    """
    if isinstance(value, LooseConfig):
        buildable_cls = fdl.Partial if value.partial else fdl.Config
        try:
            fn_or_cls = _resolve(value.fn_or_cls)
        except ImportError as e:
            raise ImportError(
                f"Cannot import {value.fn_or_cls!r}, referenced by the config. Configs that refer to "
                "classes in __main__ or in removed modules can only be read loosely, not built."
            ) from e
        return buildable_cls(fn_or_cls, **{k: to_buildable(v) for k, v in value.arguments.items()})
    if isinstance(value, LooseType):
        return _resolve(value.name)
    if isinstance(value, dict):
        return {k: to_buildable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(to_buildable(v) for v in value)
    return value


def _arguments(node: Any) -> dict[str, Any]:
    """Return the explicitly set arguments of a config node, whichever way it was loaded."""
    if isinstance(node, LooseConfig):
        return node.arguments
    if isinstance(node, fdl.Buildable):
        return {str(k): v for k, v in fdl.ordered_arguments(node).items()}
    if isinstance(node, dict):
        return node
    raise TypeError(f"Expected a Fiddle config node, got {type(node).__name__}")


# --------------------------------------------------------------------------------------
# InferenceSpec
# --------------------------------------------------------------------------------------


def _mlcast_commit() -> str | None:
    """Return the git commit of the installed mlcast, if it can be determined."""
    try:
        direct_url = distribution("mlcast").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
    match = re.search(r"\+g([0-9a-f]{7,40})", mlcast.__version__)
    return match.group(1) if match else None


@dataclass(frozen=True)
class InferenceSpec:
    """Everything needed to run a trained model on new data, apart from its weights.

    Parameters
    ----------
    standard_names : tuple of str
        CF standard name of each input/output channel, in channel order.
    units : tuple of str
        Units of each channel, as the normalisation expects them (see ``UNITS``).
    scaling : ReflectivityScaling
        Z-R relation and normalisation range used for every channel.
    input_steps : int
        Number of past timesteps the model takes.
    forecast_steps : int
        Number of timesteps the model was trained to forecast.
    timestep : str or None
        Time between steps as an ISO 8601 duration (e.g. ``"PT5M"``), or ``None`` if
        unknown. Training configs do not record it.
    spatial_divisor : int
        Height and width are padded to a multiple of this (``2**num_blocks`` for
        ``ConvGruModel``).
    default_ensemble_size : int
        Ensemble size the model was trained with.
    network_config : str
        Fiddle YAML (``dump_yaml``) of the network config, ``cfg.pl_module.network``.
    mlcast_version : str
        Version of mlcast that created this spec.
    mlcast_commit : str or None
        Git commit of mlcast that created this spec, if known.
    """

    standard_names: tuple[str, ...]
    units: tuple[str, ...]
    scaling: ReflectivityScaling
    input_steps: int
    forecast_steps: int
    timestep: str | None
    spatial_divisor: int
    default_ensemble_size: int
    network_config: str
    mlcast_version: str
    mlcast_commit: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "standard_names", tuple(self.standard_names))
        object.__setattr__(self, "units", tuple(self.units))
        if len(self.standard_names) == 0:
            raise ValueError("At least one standard name is required")
        if len(self.units) != len(self.standard_names):
            raise ValueError(f"Got {len(self.units)} units for {len(self.standard_names)} standard names")
        for name in ("input_steps", "forecast_steps", "spatial_divisor", "default_ensemble_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.timestep is not None and not _ISO_8601_DURATION.match(self.timestep):
            raise ValueError(f"timestep must be an ISO 8601 duration such as 'PT5M', got {self.timestep!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-serialisable dictionary.

        Returns
        -------
        d : dict of str to Any
            All fields, with tuples as lists, ``scaling`` as a dictionary of its
            parameters, and a ``"format_version"`` key.
        """
        d = dataclasses.asdict(self)
        d["standard_names"] = list(self.standard_names)
        d["units"] = list(self.units)
        return {"format_version": SPEC_FORMAT_VERSION, **d}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceSpec":
        """Rebuild a spec from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict of str to Any
            As returned by :meth:`to_dict` (possibly after a JSON round trip).

        Returns
        -------
        spec : InferenceSpec

        Raises
        ------
        ValueError
            If the format version is newer than this code understands, or fields are
            missing or unknown.
        """
        d = dict(d)
        format_version = d.pop("format_version", SPEC_FORMAT_VERSION)
        if format_version > SPEC_FORMAT_VERSION:
            raise ValueError(
                f"Inference spec format version {format_version} is newer than the supported "
                f"version {SPEC_FORMAT_VERSION}; upgrade mlcast"
            )
        fields = {f.name for f in dataclasses.fields(cls)}
        missing, unknown = fields - set(d), set(d) - fields
        if missing or unknown:
            raise ValueError(f"Invalid inference spec: missing fields {sorted(missing)}, unknown {sorted(unknown)}")
        d["scaling"] = ReflectivityScaling(**d["scaling"])
        return cls(**d)


def _dataset_factories(data: Any) -> dict[str, dict[str, Any]]:
    """Return the arguments of each dataset factory in ``cfg.data``, by name.

    A combined factory (one with a ``dataset_factories`` mapping, as in configs that
    train on several source datasets) is expanded into its members.
    """
    factory_args = _arguments(_arguments(data)["dataset_factory"])
    members = factory_args.get("dataset_factories")
    if isinstance(members, dict):
        return {str(name): _arguments(member) for name, member in members.items()}
    return {"dataset_factory": factory_args}


def _common_value(factories: dict[str, dict[str, Any]], key: str) -> Any:
    """Return ``key`` from the dataset factories, which must all agree on it."""
    values = {name: args.get(key) for name, args in factories.items()}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Dataset factory {missing} in cfg.data does not set {key!r}")
    distinct = {json.dumps(value) for value in values.values()}
    if len(distinct) > 1:
        raise ValueError(f"Dataset factories in cfg.data disagree on {key!r}: {values}")
    return next(iter(values.values()))


def spec_from_config(cfg: Any, timestep: str | None = None) -> InferenceSpec:
    """Build an inference spec from a training config.

    ``cfg.data`` is only read as plain values, so it may reference classes that cannot
    be imported. ``cfg.pl_module.network`` is resolved to an ``fdl.Config`` (importing
    its class) to serialise it into ``network_config``, but not built.

    Parameters
    ----------
    cfg : fdl.Config or LooseConfig
        The training config, as built (``training_experiment.as_buildable()``), loaded by
        ``mlcast.config.loader.load_yaml_config``, or loaded by :func:`load_config_loose`.
    timestep : str or None, optional
        Time between steps as an ISO 8601 duration (e.g. ``"PT5M"``). Training configs
        do not record it, so it can only be given here. Default is ``None`` (unknown).

    Returns
    -------
    spec : InferenceSpec

    Raises
    ------
    ValueError
        If the dataset factories are missing ``standard_names``, ``input_steps`` or
        ``forecast_steps``, disagree on the step counts, or use a standard name without
        known units.
    ImportError
        If the network class cannot be imported.

    Notes
    -----
    When several dataset factories are combined and their ``standard_names`` differ
    (e.g. ``rainfall_amount`` for one source and ``rainfall_flux`` for another, which
    the network sees identically after normalisation), the first factory's names are
    used and a warning is logged.
    """
    pl_module_args = _arguments(_arguments(cfg)["pl_module"])
    network = pl_module_args["network"]
    network_buildable = to_buildable(network) if isinstance(network, LooseConfig) else network
    network_args = fdl.ordered_arguments(network_buildable, include_defaults=True)
    num_blocks = network_args.get("num_blocks")
    spatial_divisor = 2**num_blocks if isinstance(num_blocks, int) else 1

    factories = _dataset_factories(_arguments(cfg)["data"])
    input_steps = _common_value(factories, "input_steps")
    forecast_steps = _common_value(factories, "forecast_steps")
    names_by_factory = {name: args.get("standard_names") for name, args in factories.items()}
    first_name, standard_names = next(iter(names_by_factory.items()))
    if not standard_names:
        raise ValueError(f"Dataset factory {first_name!r} in cfg.data does not set 'standard_names'")
    if any(list(names) != list(standard_names) for names in names_by_factory.values() if names):
        logger.warning(
            "Dataset factories in cfg.data use different standard names {}; using {!r}'s {} for inference",
            names_by_factory,
            first_name,
            standard_names,
        )
    unknown = [name for name in standard_names if name not in UNITS]
    if unknown:
        raise ValueError(f"No units known for standard name(s) {unknown}; known: {sorted(UNITS)}")

    return InferenceSpec(
        standard_names=tuple(standard_names),
        units=tuple(UNITS[name] for name in standard_names),
        scaling=DEFAULT_SCALING,
        input_steps=int(input_steps),
        forecast_steps=int(forecast_steps),
        timestep=timestep,
        spatial_divisor=spatial_divisor,
        default_ensemble_size=int(pl_module_args.get("ensemble_size", 1)),
        network_config=dump_yaml(network_buildable),
        mlcast_version=mlcast.__version__,
        mlcast_commit=_mlcast_commit(),
    )
