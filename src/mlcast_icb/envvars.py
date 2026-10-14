"""A typo check for the environment variables a mode reads.

Configuration is environment variables only, and a variable nobody reads has no effect:
a misspelt ``DST_AWS_PROFIL`` would leave the output going to the default account with
no hint why. :func:`warn_unknown_env_vars` closes that gap by warning, at startup, about
set variables that look meant for this application but are not ones it reads.

Each mode (``run``, ``serve``) reads a different set of variables, so the known names are
passed in rather than listed here; :func:`~mlcast_icb.io.storage.storage_variables`
supplies the storage ones. A mode's list must name every variable it reads, or a
misspelling of a missing one goes unnoticed and the variable itself is reported.

It warns rather than fails. A container's environment is mostly not ours: Kubernetes
injects service-link variables for every Service in the namespace (``INPUT_SERVICE_HOST``,
``SRC_PORT_9000_TCP_ADDR``, ...), the CUDA base image sets ``NVIDIA_*`` and ``CUDA_*``,
and boto reads dozens of ``AWS_*`` variables of its own. Failing on an unrecognised name
would break deployments over variables that were never meant for this application.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher

from loguru import logger

from mlcast_icb.io.storage import SIDES

#: How similar (difflib ratio, 0 to 1) an unknown name must be to a known one to be
#: reported as a likely misspelling. 0.8 catches a dropped, doubled or swapped character
#: even in a short name (``DEVCIE`` scores 0.83 against ``DEVICE``) and near-misses such
#: as ``INPUT_VARS``, while ``AWS_REGION``, ``HOSTNAME`` and ``CUDA_VISIBLE_DEVICES``
#: stay well below it.
TYPO_CUTOFF = 0.8

#: Kubernetes service-link variables, injected for every Service in the namespace: for a
#: Service ``input``, ``INPUT_SERVICE_HOST``, ``INPUT_SERVICE_PORT[_<name>]``,
#: ``INPUT_PORT`` and ``INPUT_PORT_<n>_<proto>[_ADDR|_PORT|_PROTO]``. A Service named
#: ``src`` or ``input-zarr`` would otherwise look like one of ours.
_SERVICE_LINK = re.compile(
    r"[A-Z0-9_]+_(?:SERVICE_HOST|SERVICE_PORT(?:_[A-Z0-9_]+)?"
    r"|PORT(?:_[0-9]+_(?:TCP|UDP|SCTP)(?:_(?:ADDR|PORT|PROTO))?)?)"
)

#: Read by boto itself rather than by this application, and close enough to a storage
#: variable that the typo check would flag them. Only names that actually collide
#: belong here; any other library variable is already left alone.
_LIBRARY_VARIABLES = re.compile(r"AWS_SECRET_ACCESS_KEY|AWS_ENDPOINT_URL_[A-Z0-9_]+")


def warn_unknown_env_vars(known: Iterable[str], env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Warn about set variables that look meant for this application but are not read.

    A name is reported when it is not in ``known`` and either differs from a known name
    only in case, is a close misspelling of one (see :data:`TYPO_CUTOFF`), or carries a
    store prefix (``SRC_``, ``DST_``) that this application owns. Kubernetes
    service-link variables and the few boto variables that resemble ours are never
    reported.

    Parameters
    ----------
    known : iterable of str
        Every variable the calling mode reads, storage variables included.
    env : mapping of str to str, optional
        Environment to inspect. Defaults to :data:`os.environ`; tests pass a dict.

    Returns
    -------
    dict of str to str
        The warning logged for each reported name, in name order. Empty when nothing
        looks wrong.

    Notes
    -----
    This only ever warns: the environment of a container carries many variables that
    were never meant for this application. Only names are logged, never values, since
    a misspelled variable may well hold a secret.
    """
    registry = frozenset(known)
    env = os.environ if env is None else env
    warnings: dict[str, str] = {}
    for name in sorted(env):
        message = _diagnose(name, registry)
        if message is not None:
            logger.warning(message)
            warnings[name] = message
    return warnings


def _diagnose(name: str, registry: frozenset[str]) -> str | None:
    """Decide whether one variable name deserves a warning, and word it.

    Parameters
    ----------
    name : str
        The variable name, as set in the environment.
    registry : frozenset of str
        The names the calling mode reads.

    Returns
    -------
    str or None
        The warning, with a suggestion where there is one, or ``None`` when the name is
        known or plainly belongs to something else.
    """
    if name in registry or _SERVICE_LINK.fullmatch(name) or _LIBRARY_VARIABLES.fullmatch(name):
        return None

    unread = f"Environment variable {name} has no effect: this application does not read it."
    upper = name.upper()
    if upper in registry:
        return f"{unread} Did you mean {upper}? Names are case-sensitive."

    suggestions = _close_matches(upper, registry)
    if suggestions:
        return f"{unread} Did you mean {' or '.join(suggestions)}?"

    side = upper.partition("_")[0]
    if side in SIDES:
        return (
            f"{unread} The {side}_ prefix is reserved for the {SIDES[side]} store; "
            f"options without a dedicated variable go in {side}_STORAGE_OPTIONS."
        )
    return None


def _close_matches(name: str, registry: frozenset[str]) -> list[str]:
    """Find the known names a misspelled one was most likely meant to be.

    Parameters
    ----------
    name : str
        An unknown variable name, upper-cased.
    registry : frozenset of str
        The names the calling mode reads.

    Returns
    -------
    list of str
        The known names scoring highest against ``name``, provided that score reaches
        :data:`TYPO_CUTOFF`; empty otherwise. Usually one name, but every tie is kept,
        so ``STORAGE_OPTIONS`` suggests both the ``SRC_`` and ``DST_`` forms rather
        than an arbitrary one of them.
    """
    if not registry:
        return []
    scores = {known: SequenceMatcher(None, known, name).ratio() for known in registry}
    best = max(scores.values())
    if best < TYPO_CUTOFF:
        return []
    return sorted(known for known, score in scores.items() if score == best)
