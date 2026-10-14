"""Per-side storage options, so input and output can live on different object stores.

The input and output stores are configured independently: a run can read from one S3
account and write to another, or read from a public bucket and write to an on-premise
S3-compatible host. Each side reads its own ``SRC_`` / ``DST_`` variables, falling back
to the unprefixed spellings when both sides share one configuration.

Two deliberate choices:

* **Endpoints belong in the AWS config.** A profile in ``~/.aws/config`` can carry its
  own ``endpoint_url``, which is how one run reaches two different S3-compatible hosts
  with nothing but two profile names. ``<side>_S3_ENDPOINT_URL`` stays available as an
  override for deployments that cannot mount a config file.
* **Access is anonymous unless credentials are actually given.** Reading a public
  bucket is a common case, and defaulting to signed requests turns that into a
  confusing ``NoCredentialsError``. Naming a profile, supplying keys, or running under
  an IAM role all count as credentials and switch signing back on.

When a store cannot be reached anyway, :func:`storage_error` turns the failure into a
:class:`~mlcast_icb.errors.StorageError` (exit 5) whose hint is built from these same
per-side options, and :func:`storage_exceptions` says which failures count.
:func:`storage_errors` wraps both in a context manager for the common case.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from mlcast_icb.errors import ConfigError, StorageError, format_problems

#: The two independently configured stores, by variable prefix.
SIDES = {"SRC": "input", "DST": "output"}

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

#: Environment variables that mean "this process already has AWS credentials", beyond a
#: profile or explicit keys: the ambient role credentials used on ECS, EKS and EC2.
AMBIENT_CREDENTIAL_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)

#: Storage variables read once per side, as ``<side>_<name>``.
SIDE_VARIABLES = (
    "AWS_PROFILE",
    "S3_ENDPOINT_URL",
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "S3_ANON",
    "STORAGE_OPTIONS",
)

#: Unprefixed storage variables that apply to both sides when the prefixed one is unset.
SHARED_VARIABLES = ("AWS_PROFILE", "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL", "S3_ANON")


def storage_variables(sides: tuple[str, ...] = tuple(SIDES)) -> frozenset[str]:
    """List every environment variable :func:`storage_options` may read.

    Parameters
    ----------
    sides : tuple of str, optional
        The sides in use, by prefix. Defaults to both ``SRC`` and ``DST``; a mode that
        only ever writes could pass ``("DST",)``.

    Returns
    -------
    frozenset of str
        The prefixed variables of each side, the shared unprefixed ones, and the ambient
        credential variables whose mere presence switches S3 requests to signed. Meant
        to be added to the known names given to
        :func:`~mlcast_icb.envvars.warn_unknown_env_vars`.
    """
    prefixed = {f"{side}_{name}" for side in sides for name in SIDE_VARIABLES}
    return frozenset(prefixed | set(SHARED_VARIABLES) | set(AMBIENT_CREDENTIAL_VARS))


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


def storage_options(env: Mapping[str, str], uri: str, side: str) -> dict[str, Any]:
    """Build the fsspec storage options for one side.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    uri : str
        The store URI, whose scheme decides which options apply.
    side : {'SRC', 'DST'}
        Which side is being configured; also the environment variable prefix.

    Returns
    -------
    dict
        Keyword arguments for the fsspec filesystem. Empty for a local path with no
        overrides, which leaves the existing behaviour untouched. For an S3 URI,
        always includes ``anon``, resolved as described in the module docstring.

    Raises
    ------
    ConfigError
        If ``<side>_STORAGE_OPTIONS`` is not a JSON object, or ``<side>_S3_ANON`` is
        not a boolean.
    """
    scheme = urlsplit(uri).scheme.lower()
    options: dict[str, Any] = {}

    if scheme in S3_SCHEMES:
        options.update(_s3_options(env, side))
    elif _s3_variables_set(env, side):
        logger.warning(
            f"{side}_AWS_PROFILE / {side}_S3_ENDPOINT_URL / {side}_S3_ANON are set but "
            f"{uri} is not an S3 URI; ignoring them"
        )

    # Merged last, so it can override anything above and reach options that have no
    # dedicated variable of their own.
    options.update(_extra(env, side))

    if scheme in S3_SCHEMES and "anon" not in options:
        options["anon"] = not _has_credentials(env, side, options)

    if options:
        logger.info(f"{side} storage options: {redact(options)}")
        if options.get("anon"):
            logger.info(
                f"{side} is using anonymous access; set {side}_AWS_PROFILE (or supply "
                f"credentials) if the bucket is not public"
            )
    return options


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
) -> StorageError:
    """Describe a failure to reach one store, with a hint at what to check.

    Parameters
    ----------
    side : {'SRC', 'DST'}
        Which store failed; also the prefix of the variables that configure it.
    uri : str
        The store URI.
    options : mapping or None
        The storage options the store was opened with, to tailor the hint — for an S3
        URI, whether access was anonymous and which endpoint was used.
    action : str
        What was being attempted, e.g. ``'open'``, ``'read'``, ``'write'``.
    exc : BaseException
        The underlying error.

    Returns
    -------
    StorageError
        An error naming the side, the URI and the underlying error, followed by what
        to check. The caller raises it ``from exc``.
    """
    headline = f"Could not {action} the {SIDES[side]} store {uri!r}: {_describe(exc)}"
    return StorageError(format_problems(headline, _hints(side, uri, options or {})))


@contextmanager
def storage_errors(side: str, uri: str, options: Mapping[str, Any] | None, action: str) -> Iterator[None]:
    """Turn storage failures inside the block into a :class:`StorageError`.

    Parameters
    ----------
    side : {'SRC', 'DST'}
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


def _hints(side: str, uri: str, options: Mapping[str, Any]) -> list[str]:
    """Suggest what to check after a store could not be reached.

    Parameters
    ----------
    side : {'SRC', 'DST'}
        Which store failed.
    uri : str
        The store URI, whose scheme decides which hints apply.
    options : mapping
        The storage options in effect for that store.

    Returns
    -------
    list of str
        One or more hints. For S3 they say whether access was anonymous — the default
        without credentials, see the module docstring — and where the endpoint came
        from, naming the ``<side>_*`` variables that change it.
    """
    scheme = urlsplit(uri).scheme.lower()
    if scheme in _LOCAL_SCHEMES:
        if side == "SRC":
            return [
                "check that the user running this process can read every file in the store; "
                "in a container, also check the volume mount"
            ]
        return [
            "check that the user running this process can create and write that path; "
            "in a container, also check that the volume is not mounted read-only"
        ]

    if scheme not in S3_SCHEMES:
        return [f"check the credentials for {scheme}:// and any {side}_STORAGE_OPTIONS"]

    hints: list[str] = []
    if options.get("anon"):
        hints.append(
            f"{side} is using anonymous access, the default when no credentials are "
            f"found; if the bucket is not public, set {side}_AWS_PROFILE (or "
            f"{side}_AWS_ACCESS_KEY_ID and {side}_AWS_SECRET_ACCESS_KEY)"
        )
    else:
        profile = options.get("profile")
        source = f"profile {profile!r}" if profile else "the credentials found"
        hints.append(
            f"{side} is signing requests with {source}; check they are valid and allowed "
            f"to {'read' if side == 'SRC' else 'write'} this bucket ({side}_AWS_PROFILE, "
            f"{side}_AWS_ACCESS_KEY_ID, or {side}_S3_ANON=true for a public bucket)"
        )

    client_kwargs = options.get("client_kwargs")
    endpoint = options.get("endpoint_url") or (
        client_kwargs.get("endpoint_url") if isinstance(client_kwargs, Mapping) else None
    )
    if endpoint:
        hints.append(f"check that the endpoint {endpoint} is reachable from here")
    else:
        hints.append(
            "no endpoint is set in the environment, so it comes from the profile's "
            f"endpoint_url in ~/.aws/config, or is AWS itself; {side}_S3_ENDPOINT_URL "
            "overrides it"
        )
    return hints


def _s3_options(env: Mapping[str, str], side: str) -> dict[str, Any]:
    """Collect the S3-specific options for one side.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.

    Returns
    -------
    dict
        Any of ``profile``, ``client_kwargs``, ``key``, ``secret``, ``token`` and
        ``anon`` that the environment specifies. ``anon`` appears only when set
        explicitly; otherwise it is derived later, once the JSON options are known.

    Raises
    ------
    ConfigError
        If ``<side>_S3_ANON`` is not a boolean.
    """
    options: dict[str, Any] = {}

    profile = _get(env, side, "AWS_PROFILE")
    if profile:
        options["profile"] = profile

    # Usually unset: the endpoint is better carried by the profile in ~/.aws/config.
    endpoint = _get(env, side, "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL")
    if endpoint:
        options["client_kwargs"] = {"endpoint_url": endpoint}

    for option, name in (
        ("key", "AWS_ACCESS_KEY_ID"),
        ("secret", "AWS_SECRET_ACCESS_KEY"),
        ("token", "AWS_SESSION_TOKEN"),
    ):
        # Only the prefixed spellings: unprefixed AWS_* keys are already ambient, and
        # boto picks them up itself.
        value = env.get(f"{side}_{name}", "").strip()
        if value:
            options[option] = value

    anon = _get(env, side, "S3_ANON")
    if anon is not None:
        options["anon"] = parse_bool(anon, f"{side}_S3_ANON")

    return options


def _s3_variables_set(env: Mapping[str, str], side: str) -> bool:
    """Report whether any S3-only variable is set for this side.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.

    Returns
    -------
    bool
        True when at least one prefixed S3 variable is set, which is worth warning
        about if the URI turns out not to be an S3 one.
    """
    names = ("AWS_PROFILE", "S3_ENDPOINT_URL", "S3_ANON", "AWS_ACCESS_KEY_ID")
    return any(env.get(f"{side}_{name}", "").strip() for name in names)


def _has_credentials(env: Mapping[str, str], side: str, options: Mapping[str, Any]) -> bool:
    """Decide whether this side has credentials, and so should sign its requests.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.
    options : mapping
        The storage options resolved so far, including the JSON escape hatch.

    Returns
    -------
    bool
        True when a profile, explicit keys, or ambient role credentials are present.
        False means the request should go out unsigned.
    """
    if any(options.get(name) for name in _CREDENTIAL_OPTIONS):
        return True
    client_kwargs = options.get("client_kwargs")
    if isinstance(client_kwargs, Mapping) and any(client_kwargs.get(name) for name in _CLIENT_CREDENTIAL_KWARGS):
        return True
    if _get(env, side, "AWS_PROFILE"):
        return True
    return any(env.get(name, "").strip() for name in AMBIENT_CREDENTIAL_VARS)


def _get(env: Mapping[str, str], side: str, *names: str) -> str | None:
    """Read one side-specific variable, falling back to the unprefixed spellings.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.
    *names : str
        Variable names without the prefix, in order of preference.

    Returns
    -------
    str or None
        The first of ``<side>_<name>`` (all names, in order) that is set, then the
        first bare ``<name>`` that is set — so a plain ``AWS_PROFILE`` still applies to
        both sides — otherwise ``None``.
    """
    for key in [f"{side}_{name}" for name in names] + list(names):
        value = env.get(key, "").strip()
        if value:
            return value
    return None


def _extra(env: Mapping[str, str], side: str) -> dict[str, Any]:
    """Parse the JSON escape hatch for options with no dedicated variable.

    Parameters
    ----------
    env : mapping of str to str
        Environment to read.
    side : str
        Variable prefix, ``'SRC'`` or ``'DST'``.

    Returns
    -------
    dict
        The decoded object, or empty when unset.

    Raises
    ------
    ConfigError
        If the value is not valid JSON, or is JSON but not an object.
    """
    key = f"{side}_STORAGE_OPTIONS"
    raw = env.get(key, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{key} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{key} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def parse_bool(value: str, key: str) -> bool:
    """Parse a boolean environment variable.

    Parameters
    ----------
    value : str
        The raw value.
    key : str
        Variable name, used in the error message.

    Returns
    -------
    bool
        The parsed value.

    Raises
    ------
    ConfigError
        If the value is not a recognised boolean spelling.
    """
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be a boolean such as 'true' or 'false', got {value!r}")


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
