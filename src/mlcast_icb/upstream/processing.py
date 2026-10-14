"""Shared pre- and post-processing for training and inference.

Converts physical fields to the normalised representation the network sees, and
network output back to physical units, in one place for both the training data
pipeline and inference, so the two cannot drift apart.

Proposed for mlcast as ``mlcast/data/processing.py`` (see ``README.md``, PR C).
``preprocess`` replaces the normalise/stack/NaN-fill steps in
``SourceDataIndexedDataset.__getitem__``, ``SourceDataRandomSamplingDataset.__getitem__``
and ``SourceDataDatasetBase._build_sample``, and the corresponding steps in
``NowcastLightningModule.predict``. ``postprocess`` replaces the denormalisation in
``predict`` and adds the clamp that training applies before the loss.

Channel layout
--------------
The network's output stacks ensemble members on the channel axis, member-major: for
``C`` channels and ``E`` members, index ``e * C + c`` on dim 2 is channel ``c`` of member
``e`` (``ConvGruModel.forward`` concatenates whole ``(B, T, C, H, W)`` member outputs
along dim 2).
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Float32, Shaped, jaxtyped

from mlcast.data.normalization import DEFAULT_SCALING, ReflectivityScaling


def _registry_lookup(
    registry: dict[str, Callable[[np.ndarray], np.ndarray]], standard_names: Sequence[str]
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Return the conversion function for each standard name.

    Parameters
    ----------
    registry : dict of str to Callable
        A ``ReflectivityScaling`` (de)normalisation registry.
    standard_names : Sequence[str]
        CF standard names, one per channel.

    Returns
    -------
    funcs : list of Callable
        One conversion function per standard name, in order.

    Raises
    ------
    ValueError
        If ``standard_names`` is empty or contains a name without a registered conversion.
    """
    if len(standard_names) == 0:
        raise ValueError("At least one standard name is required")
    unknown = [name for name in standard_names if name not in registry]
    if unknown:
        raise ValueError(f"No normalisation registered for standard name(s) {unknown}; known: {sorted(registry)}")
    return [registry[name] for name in standard_names]


@jaxtyped(typechecker=beartype)
def preprocess(
    arrays: Sequence[Shaped[np.ndarray, "time height width"]],
    standard_names: Sequence[str],
    scaling: ReflectivityScaling = DEFAULT_SCALING,
    return_valid_mask: bool = False,
) -> (
    Float32[np.ndarray, "time channels height width"]
    | tuple[Float32[np.ndarray, "time channels height width"], Bool[np.ndarray, "time channels height width"]]
):
    """Normalise physical fields and stack them into a network input array.

    Each channel is normalised with ``scaling.normalization_registry[standard_name]``
    in its own dtype, the channels are stacked on axis 1 and cast to ``float32``, and
    invalid values are filled with ``scaling.norm_min``.

    A value is invalid if it is NaN after normalisation: NaN in the input, and also
    values the conversion cannot map, such as negative rainfall rates (whose
    logarithm is NaN). Filling *after* normalising means invalid data is always
    treated as "no signal" (``norm_min``), independent of ``scaling``.

    Parameters
    ----------
    arrays : Sequence[np.ndarray]
        One ``(T, H, W)`` array per channel, in physical units matching its standard name
        (e.g. mm/h for ``rainfall_rate``). May contain NaN.
    standard_names : Sequence[str]
        CF standard name of each array, selecting its normalisation.
    scaling : ReflectivityScaling, optional
        Z-R relation and normalisation range. Default is ``DEFAULT_SCALING``.
    return_valid_mask : bool, optional
        If ``True``, also return a boolean mask that is ``True`` where the value was
        valid before filling (used for masked losses). Default is ``False``.

    Returns
    -------
    data : Float32[np.ndarray, "time channels height width"]
        Normalised, NaN-free input of shape ``(T, C, H, W)``.
    valid : Bool[np.ndarray, "time channels height width"]
        Only if ``return_valid_mask`` is ``True``: validity of each value before filling.

    Raises
    ------
    ValueError
        If the number of arrays and standard names differ, or a standard name is unknown.
    """
    if len(arrays) != len(standard_names):
        raise ValueError(f"Got {len(arrays)} arrays but {len(standard_names)} standard names")
    norm_funcs = _registry_lookup(scaling.normalization_registry, standard_names)

    channels = [norm_func(array) for norm_func, array in zip(norm_funcs, arrays, strict=True)]
    data = np.ascontiguousarray(np.stack(channels, axis=1), dtype=np.float32)

    valid = ~np.isnan(data) if return_valid_mask else None
    data = np.nan_to_num(data, nan=scaling.norm_min)

    if valid is not None:
        return data, valid
    return data


@jaxtyped(typechecker=beartype)
def postprocess(
    preds: Float[torch.Tensor, "batch time channels_x_members height width"],
    standard_names: Sequence[str],
    scaling: ReflectivityScaling = DEFAULT_SCALING,
) -> Float[np.ndarray, "batch members time channels height width"]:
    """Clamp network output to the normalised range and convert it to physical units.

    Output is clamped to ``[scaling.norm_min, scaling.norm_max]``, as training does
    before the loss, so no forecast exceeds ``scaling.dbz_ceiling``. Each channel is then
    denormalised with ``scaling.denormalization_registry[standard_name]``.

    Parameters
    ----------
    preds : Float[torch.Tensor, "batch time channels_x_members height width"]
        Network output of shape ``(B, T, C*E, H, W)``, ensemble members stacked
        member-major on dim 2 (index ``e * C + c`` is channel ``c`` of member ``e``).
    standard_names : Sequence[str]
        CF standard name of each of the ``C`` channels, selecting its denormalisation.
    scaling : ReflectivityScaling, optional
        Z-R relation and normalisation range. Default is ``DEFAULT_SCALING``.

    Returns
    -------
    out : Float[np.ndarray, "batch members time channels height width"]
        Forecast in physical units, of shape ``(B, E, T, C, H, W)``, in the dtype of
        ``preds``.

    Raises
    ------
    ValueError
        If dim 2 of ``preds`` is not a multiple of the number of standard names, or a
        standard name is unknown.
    """
    denorm_funcs = _registry_lookup(scaling.denormalization_registry, standard_names)
    batch, time, channels_x_members, height, width = preds.shape
    n_channels = len(standard_names)
    if channels_x_members % n_channels != 0:
        raise ValueError(
            f"Output has {channels_x_members} channels, which is not a multiple of the "
            f"{n_channels} standard names {list(standard_names)}"
        )
    n_members = channels_x_members // n_channels

    clamped = preds.detach().clamp(min=scaling.norm_min, max=scaling.norm_max).cpu().numpy()
    clamped = clamped.reshape(batch, time, n_members, n_channels, height, width).transpose(0, 2, 1, 3, 4, 5)

    out = np.empty_like(clamped)
    for c, denorm_func in enumerate(denorm_funcs):
        out[:, :, :, c] = denorm_func(clamped[:, :, :, c])
    return out
