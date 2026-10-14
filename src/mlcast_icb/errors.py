"""Error types, each mapped to a distinct process exit code.

An inference container runs unattended, inside a pipeline someone else wrote, so failures
must be unambiguous: the exit code says which layer rejected the run, and the message
says exactly what to change.

==== ===================================================================================
Code Meaning
==== ===================================================================================
0    Success
1    Anything unexpected: not one of these classes, so a bug, logged with its traceback
2    :class:`ConfigError` — the environment is wrong
3    :class:`InputError` — the input is missing or does not match what the model expects
4    :class:`DeviceError` — the requested compute device is unusable
5    :class:`StorageError` — a store could not be reached, read or written
6    :class:`ModelError` — the model bundle baked into the image is missing or corrupt
==== ===================================================================================

Codes 0-5 match ``dummy-mlwp-model``, so a pipeline built against the dummy model reads
these exit codes the same way. Code 6 is new: a broken model bundle is neither the
deployer's configuration (2) nor their data (3) but the image itself, and the fix is to
rebuild the image rather than to change anything about the run.
"""

from __future__ import annotations


class MlcastIcbError(Exception):
    """Base class for every error the builder's runtime raises deliberately.

    Attributes
    ----------
    exit_code : int
        Process exit code used when this error reaches the entrypoint.
    """

    exit_code = 1


class ConfigError(MlcastIcbError):
    """An environment variable is missing, malformed, or internally inconsistent.

    Attributes
    ----------
    exit_code : int
        Always 2.
    """

    exit_code = 2


class InputError(MlcastIcbError):
    """The input is missing, unreadable as data, or does not match what the model expects.

    Attributes
    ----------
    exit_code : int
        Always 3.
    """

    exit_code = 3


class PayloadTooLargeError(InputError):
    """An uploaded payload is larger than the configured limit.

    A separate class only so the HTTP server can answer 413 rather than 422; as a
    process exit it is an ordinary :class:`InputError`.

    Attributes
    ----------
    exit_code : int
        Always 3.
    """


class DeviceError(MlcastIcbError):
    """The requested compute device is unavailable or unusable.

    Attributes
    ----------
    exit_code : int
        Always 4.
    """

    exit_code = 4


class StorageError(MlcastIcbError):
    """A store could not be reached, read or written.

    This is the layer below the data — missing or rejected credentials, access denied,
    an unreachable endpoint, a destination that cannot be written — so it says nothing
    about the store's contents. An input store that simply does not exist stays an
    :class:`InputError`, since that is a wrong input URI rather than a failure to reach
    the store.

    Attributes
    ----------
    exit_code : int
        Always 5.
    """

    exit_code = 5


class ModelError(MlcastIcbError):
    """The model bundle in the image (checkpoint, manifest) is missing or cannot be loaded.

    Nothing the deployer sets at run time can fix this: the image has to be rebuilt.

    Attributes
    ----------
    exit_code : int
        Always 6.
    """

    exit_code = 6


def format_problems(headline: str, problems: list[str]) -> str:
    """Render a collected list of problems as a single multi-line message.

    Parameters
    ----------
    headline : str
        Opening line describing what was being checked.
    problems : list of str
        Individual problems, rendered as an indented bullet list beneath the headline.

    Returns
    -------
    str
        The headline followed by one indented line per problem.

    Examples
    --------
    >>> print(format_problems("Two problems:", ["first", "second"]))
    Two problems:
      - first
      - second
    """
    return "\n".join([headline] + [f"  - {p}" for p in problems])
