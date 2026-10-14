"""Per-side storage options, so input and output can live on different object stores.

The input and output stores are configured independently: a run can read from one S3
account and write to another, or read from a public bucket and write to an on-premise
S3-compatible host. Each side has its own options, as in DMI's other inference
containers: ``--input-aws-profile``, ``--input-s3-endpoint-url``, ``--input-s3-anon``
and ``--input-storage-options`` (or ``MLCAST_INPUT_AWS_PROFILE`` and so on), and the same
with ``output``. They are gathered in a :class:`StorageSettings` per side.

Three deliberate choices:

* **Otherwise, the standard AWS configuration applies.** A side without its own profile
  or endpoint is left to botocore, which reads ``AWS_PROFILE``, the ``AWS_*`` keys,
  ``AWS_ENDPOINT_URL[_S3]``, IAM role credentials, and a profile's ``endpoint_url`` in
  ``~/.aws/config``, for both sides alike. A side's own profile takes precedence over
  ``AWS_ACCESS_KEY_ID`` and friends for that side (botocore drops the environment's keys
  when a profile is given explicitly), and its own endpoint over ``AWS_ENDPOINT_URL``.
  Credentials themselves never have options of their own: they come from the
  environment, a profile, or a role.
* **Endpoints belong in the AWS config.** A profile in ``~/.aws/config`` can carry its
  own ``endpoint_url``, which is how one run reaches two different S3-compatible hosts
  with nothing but two profile names. ``--<side>-s3-endpoint-url`` stays available for
  deployments that cannot mount a config file.
* **Access is anonymous unless credentials are actually given.** Reading a public
  bucket is a common case, and defaulting to signed requests turns that into a
  confusing ``NoCredentialsError``. Naming a profile, having keys in the environment, or
  running under an IAM role all count as credentials and switch signing back on;
  ``--<side>-s3-anon`` decides explicitly.

When a store cannot be reached anyway, :func:`storage_error` turns the failure into a
:class:`~mlcast_icb.errors.StorageError` (exit 5) whose hint is built from these same
per-side options, and :func:`storage_exceptions` says which failures count.
:func:`storage_errors` wraps both in a context manager for the common case.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from mlcast_icb import ENV_PREFIX
from mlcast_icb.errors import ConfigError, StorageError, format_problems

#: The two independently configured stores; also the prefix of their options.
SIDES = ("input", "output")

#: Schemes for which the S3-specific options below are meaningful.
S3_SCHEMES = ("s3", "s3a")

#: Schemes that mean a path on this machine.
_LOCAL_SCHEMES = ("", "file", "local")

#: Exceptions from optional backend libraries that mean "the store could not be
#: reached", as ``(module, class)``. s3fs translates S3 error *responses* (403, 404, ...)
#: into ``OSError`` subclasses, but failures that never get a response — no credentials,
#: an unknown profile, an unreachable endpoint — surface as raw botocore errors, and a
#: download cut off mid-body as aiohttp's ``ClientPayloadError``.
_BACKEND_ERRORS = (
    ("botocore.exceptions", "BotoCoreError"),
    ("botocore.exceptions", "ClientError"),
    ("aiohttp", "ClientError"),
    ("gcsfs.retry", "HttpError"),
    ("google.auth.exceptions", "GoogleAuthError"),
)

#: Standard AWS variables that mean "this process already has AWS credentials": a
#: profile, keys, or the ambient role credentials used on ECS, EKS and EC2.
AMBIENT_CREDENTIAL_VARS = (
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)

#: Standard AWS variables that set the S3 endpoint, in botocore's order of precedence.
AMBIENT_ENDPOINT_VARS = ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL")

#: fsspec option names that carry a credential.
_CREDENTIAL_OPTIONS = ("profile", "key", "secret", "token", "username", "password")

#: Credential names s3fs also accepts inside ``client_kwargs``.
_CLIENT_CREDENTIAL_KWARGS = ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")

#: Substrings that mark an option name as secret when logging, matched case-insensitively
#: at any nesting depth. Deliberately broad: masking a harmless value costs a little
#: readability, logging a secret cannot be undone. Names that identify rather than
#: authenticate (``profile``, ``region_name``, ``endpoint_url``, ``project``,
#: ``client_email``) match none of these and stay visible.
_SECRET_NAME_PARTS = (
    "key",  # key, aws_access_key_id, private_key, account_key, SSECustomerKey
    "secret",  # secret, aws_secret_access_key, client_secret
    "token",  # token, aws_session_token, refresh_token, sas_token
    "password",
    "passwd",
    "passphrase",
    "credential",  # credential, credentials
    "auth",  # auth, Authorization, proxy_auth
    "connection_string",  # adlfs; embeds AccountKey=...
)

#: What a masked value is replaced with in the log.
_MASK = "***"


@dataclass(frozen=True)
class StorageSettings:
    """One side's storage options, as given on the command line or in the environment.

    Parameters
    ----------
    aws_profile : str, optional
        ``--<side>-aws-profile``: the AWS profile for this side, from ``~/.aws``.
    s3_endpoint_url : str, optional
        ``--<side>-s3-endpoint-url``: the S3 endpoint for this side.
    s3_anon : bool, optional
        ``--<side>-s3-anon``: whether to send unsigned requests; ``None`` decides
        automatically (anonymous unless there are credentials).
    storage_options : str, optional
        ``--<side>-storage-options``: a JSON object of further fsspec options, merged
        last, so it can override anything above.
    """

    aws_profile: str | None = None
    s3_endpoint_url: str | None = None
    s3_anon: bool | None = None
    storage_options: str | None = None


def side_variable(side: str, name: str) -> str:
    """Name the environment variable of one side's storage option.

    Parameters
    ----------
    side : {'input', 'output'}
        The side.
    name : str
        The option without the side, e.g. ``'AWS_PROFILE'``.

    Returns
    -------
    str
        E.g. ``'MLCAST_INPUT_AWS_PROFILE'``, the variable of ``--input-aws-profile``.
    """
    return f"{ENV_PREFIX}_{side.upper()}_{name}"


def storage_options(
    side: str, uri: str, settings: StorageSettings | None = None, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Build the fsspec storage options for one side.

    Parameters
    ----------
    side : {'input', 'output'}
        Which side is being configured, for the messages.
    uri : str
        The store URI, whose scheme decides which options apply.
    settings : StorageSettings, optional
        The side's own options. Defaults to none at all.
    env : mapping of str to str, optional
        The environment, for the standard AWS variables that count as credentials.
        Defaults to :data:`os.environ`; tests pass a dict.

    Returns
    -------
    dict
        Keyword arguments for the fsspec filesystem. Empty for a local path with no
        overrides, which leaves the existing behaviour untouched. For an S3 URI,
        always includes ``anon``, resolved as described in the module docstring.

    Raises
    ------
    ConfigError
        If the side's storage options are not a JSON object.
    """
    settings = settings or StorageSettings()
    env = os.environ if env is None else env
    scheme = urlsplit(uri).scheme.lower()
    options: dict[str, Any] = {}

    if scheme in S3_SCHEMES:
        if settings.aws_profile:
            options["profile"] = settings.aws_profile
        # Usually unset: the endpoint is better carried by the profile in ~/.aws/config.
        if settings.s3_endpoint_url:
            options["client_kwargs"] = {"endpoint_url": settings.s3_endpoint_url}
        if settings.s3_anon is not None:
            options["anon"] = settings.s3_anon
    elif settings.aws_profile or settings.s3_endpoint_url or settings.s3_anon is not None:
        names = ", ".join(side_variable(side, name) for name in ("AWS_PROFILE", "S3_ENDPOINT_URL", "S3_ANON"))
        logger.warning(f"Some of {names} are set but {uri} is not an S3 URI; ignoring them")

    # Merged last, so it can override anything above and reach options that have no
    # dedicated option of their own.
    options.update(_extra(side, settings.storage_options))

    if scheme in S3_SCHEMES and "anon" not in options:
        options["anon"] = not _has_credentials(env, options)
    return options


def log_storage_options(side: str, options: Mapping[str, Any]) -> None:
    """Log the storage options one side resolved to, with credentials masked.

    Parameters
    ----------
    side : {'input', 'output'}
        The side.
    options : mapping
        What :func:`storage_options` returned for it. Nothing is logged when empty.
    """
    if not options:
        return
    logger.info(f"The {side} storage options: {redact(options)}")
    if options.get("anon"):
        logger.info(
            f"The {side} store is using anonymous access; set {side_variable(side, 'AWS_PROFILE')} "
            "(or supply credentials) if the bucket is not public"
        )


def storage_exceptions() -> tuple[type[BaseException], ...]:
    """List the exception types that mean a store could not be reached, read or written.

    Returns
    -------
    tuple of type
        ``OSError`` — which covers local permission errors, connection and timeout
        errors, and the S3 error responses s3fs translates — plus the error base
        classes of whichever backend libraries are loaded (botocore, aiohttp, gcsfs,
        google-auth). Never ``Exception``: a programming error must stay an
        unexpected failure, with its traceback.

    Notes
    -----
    Backend classes are looked up in ``sys.modules`` rather than imported. An
    exception can only have been raised once its class's module is loaded, so this
    never imports a backend the run is not using, and costs nothing for local stores.
    Call it in the ``except`` clause itself, which is evaluated only when an exception
    is being matched.
    """
    found: list[type[BaseException]] = [OSError]
    for module_name, class_name in _BACKEND_ERRORS:
        cls = getattr(sys.modules.get(module_name), class_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


def storage_error(
    side: str,
    uri: str,
    options: Mapping[str, Any] | None,
    action: str,
    exc: BaseException,
    env: Mapping[str, str] | None = None,
) -> StorageError:
    """Describe a failure to reach one store, with a hint at what to check.

    Parameters
    ----------
    side : {'input', 'output'}
        Which store failed; also the prefix of the options that configure it.
    uri : str
        The store URI.
    options : mapping or None
        The storage options the store was opened with, to tailor the hint — for an S3
        URI, whether access was anonymous and which endpoint was used.
    action : str
        What was being attempted, e.g. ``'open'``, ``'read'``, ``'write'``.
    exc : BaseException
        The underlying error.
    env : mapping of str to str, optional
        The environment, for where the endpoint came from. Defaults to
        :data:`os.environ`.

    Returns
    -------
    StorageError
        An error naming the side, the URI and the underlying error, followed by what
        to check. The caller raises it ``from exc``.
    """
    env = os.environ if env is None else env
    headline = f"Could not {action} the {side} store {uri!r}: {_describe(exc)}"
    return StorageError(format_problems(headline, _hints(side, uri, options or {}, env)))


@contextmanager
def storage_errors(side: str, uri: str, options: Mapping[str, Any] | None, action: str) -> Iterator[None]:
    """Turn storage failures inside the block into a :class:`StorageError`.

    Parameters
    ----------
    side : {'input', 'output'}
        Which store is being used.
    uri : str
        The store URI.
    options : mapping or None
        The storage options in effect, for the hint.
    action : str
        What the block does, e.g. ``'read'`` or ``'write'``.

    Yields
    ------
    None

    Raises
    ------
    StorageError
        If the block raises one of :func:`storage_exceptions`, or an exception group
        made up only of them — zarr reports a failed read of part of a shard as a group
        (PEP 654) — chained from the original. Anything else propagates unchanged, so a
        programming error keeps its traceback.
    """
    try:
        yield
    except storage_exceptions() as exc:
        raise storage_error(side, uri, options, action, exc) from exc
    except BaseExceptionGroup as group:
        failures, rest = group.split(storage_exceptions())
        if failures is None or rest is not None:
            raise
        first: BaseException = failures
        while isinstance(first, BaseExceptionGroup):
            first = first.exceptions[0]
        raise storage_error(side, uri, options, action, first) from group


def _describe(exc: BaseException) -> str:
    """Render an exception, and its direct cause, for an error message.

    Parameters
    ----------
    exc : BaseException
        The exception to render.

    Returns
    -------
    str
        ``'Type: message'``, followed by the explicit cause when it adds anything.
        s3fs, for one, raises ``PermissionError('Access Denied')`` from a botocore
        ``ClientError`` whose message also names the S3 operation that was refused.
    """
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    cause = exc.__cause__
    if cause is not None and str(cause) and str(cause) not in text:
        text += f" (from {type(cause).__name__}: {cause})"
    return text


def _hints(side: str, uri: str, options: Mapping[str, Any], env: Mapping[str, str]) -> list[str]:
    """Suggest what to check after a store could not be reached.

    Parameters
    ----------
    side : {'input', 'output'}
        Which store failed.
    uri : str
        The store URI, whose scheme decides which hints apply.
    options : mapping
        The storage options in effect for that store.
    env : mapping of str to str
        The environment, for an endpoint set by a standard AWS variable.

    Returns
    -------
    list of str
        One or more hints. For S3 they say whether access was anonymous — the default
        without credentials, see the module docstring — and where the endpoint came
        from, naming the ``MLCAST_<SIDE>_*`` variables that change it.
    """
    scheme = urlsplit(uri).scheme.lower()
    if scheme in _LOCAL_SCHEMES:
        if side == "input":
            return [
                "check that the user running this process can read every file in the store; "
                "in a container, also check the volume mount"
            ]
        return [
            "check that the user running this process can create and write that path; "
            "in a container, also check that the volume is not mounted read-only"
        ]

    if scheme not in S3_SCHEMES:
        return [f"check the credentials for {scheme}:// and any {side_variable(side, 'STORAGE_OPTIONS')}"]

    profile_var = side_variable(side, "AWS_PROFILE")
    hints: list[str] = []
    if options.get("anon"):
        hints.append(
            f"the {side} store is using anonymous access, the default when no credentials are "
            f"found; if the bucket is not public, set {profile_var} (or AWS_PROFILE, or "
            f"AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)"
        )
    else:
        profile = options.get("profile")
        source = f"profile {profile!r}" if profile else "the credentials found"
        hints.append(
            f"the {side} store is signing requests with {source}; check they are valid and allowed "
            f"to {'read' if side == 'input' else 'write'} this bucket ({profile_var}, AWS_PROFILE, "
            f"AWS_ACCESS_KEY_ID, or {side_variable(side, 'S3_ANON')}=true for a public bucket)"
        )

    endpoint_var = side_variable(side, "S3_ENDPOINT_URL")
    client_kwargs = options.get("client_kwargs")
    endpoint = options.get("endpoint_url") or (
        client_kwargs.get("endpoint_url") if isinstance(client_kwargs, Mapping) else None
    )
    ambient = next((name for name in AMBIENT_ENDPOINT_VARS if env.get(name, "").strip()), None)
    if endpoint:
        hints.append(f"check that the endpoint {endpoint} is reachable from here")
    elif ambient:
        hints.append(
            f"check that the endpoint {env[ambient].strip()} (from {ambient}) is reachable from here; "
            f"{endpoint_var} overrides it for the {side} store"
        )
    else:
        hints.append(
            "no endpoint is set, so it comes from the profile's endpoint_url in ~/.aws/config, "
            f"or is AWS itself; {endpoint_var} sets it"
        )
    return hints


def _has_credentials(env: Mapping[str, str], options: Mapping[str, Any]) -> bool:
    """Decide whether a side has credentials, and so should sign its requests.

    Parameters
    ----------
    env : mapping of str to str
        The environment, for the standard AWS variables.
    options : mapping
        The storage options resolved so far, including the JSON escape hatch.

    Returns
    -------
    bool
        True when a profile, explicit keys, or ambient credentials (a profile, keys or
        role in the standard AWS variables) are present. False means the request
        should go out unsigned.
    """
    if any(options.get(name) for name in _CREDENTIAL_OPTIONS):
        return True
    client_kwargs = options.get("client_kwargs")
    if isinstance(client_kwargs, Mapping) and any(client_kwargs.get(name) for name in _CLIENT_CREDENTIAL_KWARGS):
        return True
    return any(env.get(name, "").strip() for name in AMBIENT_CREDENTIAL_VARS)


def _extra(side: str, raw: str | None) -> dict[str, Any]:
    """Parse the JSON escape hatch for options with no dedicated option.

    Parameters
    ----------
    side : {'input', 'output'}
        The side, for the error message.
    raw : str or None
        The ``--<side>-storage-options`` value.

    Returns
    -------
    dict
        The decoded object, or empty when unset.

    Raises
    ------
    ConfigError
        If the value is not valid JSON, or is JSON but not an object.
    """
    key = side_variable(side, "STORAGE_OPTIONS")
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{key} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{key} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def redact_json(raw: str | None) -> str | None:
    """Mask the credentials in a JSON object of storage options, for the log.

    Parameters
    ----------
    raw : str or None
        A ``--<side>-storage-options`` value as given.

    Returns
    -------
    str or None
        The object re-serialised by :func:`redact`; ``None`` when unset; and
        ``'*** (not a JSON object)'`` when it does not parse as one, since a malformed
        value may still hold a secret.
    """
    if raw is None or not raw.strip():
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return f"{_MASK} (not a JSON object)"
    return json.dumps(redact(parsed))


def redact(options: Mapping[str, Any]) -> dict[str, Any]:
    """Mask credential-bearing values so options can be logged safely.

    Parameters
    ----------
    options : mapping
        The storage options about to be logged.

    Returns
    -------
    dict
        A copy in which every value under a secret-looking name (see
        ``_SECRET_NAME_PARTS``) is replaced by ``'***'``, at any depth: nested mappings
        such as ``client_kwargs`` or ``config_kwargs`` and lists or tuples are walked,
        and the password in a URL such as ``https://user:pass@host`` is masked as well.
        Profile names, regions and endpoint hosts are kept, since seeing which account
        and host a run used is the whole point of logging this. ``options`` itself, and
        anything nested in it, is left unmodified.
    """
    return {name: _MASK if _is_secret_name(name) else _redact_value(value) for name, value in options.items()}


def _redact_value(value: Any) -> Any:
    """Redact one value whose own name is not secret, recursing into containers.

    Parameters
    ----------
    value : object
        A storage option value, or an item nested inside one.

    Returns
    -------
    object
        A redacted copy of a mapping, list or tuple; a string with any URL password
        masked; anything else unchanged.
    """
    if isinstance(value, Mapping):
        return redact(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, str):
        return _redact_url(value)
    return value


def _is_secret_name(name: Any) -> bool:
    """Report whether an option name looks like it carries a credential.

    Parameters
    ----------
    name : object
        A mapping key, normally a string.

    Returns
    -------
    bool
        True when the lower-cased name contains any of ``_SECRET_NAME_PARTS``.
    """
    lowered = str(name).lower()
    return any(part in lowered for part in _SECRET_NAME_PARTS)


def _redact_url(value: str) -> str:
    """Mask the password in a URL's userinfo, keeping the user name and host.

    Parameters
    ----------
    value : str
        Any string; only one that parses as a URL with a password is changed.

    Returns
    -------
    str
        ``scheme://user:***@host/...`` for a URL carrying a password, otherwise
        ``value`` unchanged.
    """
    if "://" not in value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if parts.password is None:
        return value
    userinfo, _, host = parts.netloc.rpartition("@")
    user = userinfo.partition(":")[0]
    return value.replace(parts.netloc, f"{user}:{_MASK}@{host}", 1)
