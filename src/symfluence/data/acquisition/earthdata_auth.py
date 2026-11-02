# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NASA Earthdata authentication: the one place acquisition code resolves
Earthdata Login (URS) credentials, tokens, sessions and earthaccess logins.

Lookup order
------------
Username/password (:meth:`EarthdataAuth.credentials`), first complete pair wins:

1. **config** -- ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` keys
2. **environment** -- ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``
3. **.netrc** -- ``$NETRC``, then ``~/.netrc``, then ``~/_netrc``; machines
   :data:`EARTHDATA_NETRC_HOSTS` in order. ``earthaccess.login(persist=True)``
   writes here, so a persisted earthaccess login is found at this step.
4. **earthaccess** -- an earthaccess login already made in this process
   (``earthaccess.__auth__``). Never triggers a login or network call.

Bearer token (:meth:`EarthdataAuth.token`): config ``EARTHDATA_TOKEN``, then
environment ``EARTHDATA_TOKEN``.

Sessions (:meth:`EarthdataAuth.session`): optionally an earthaccess session
first, then a Bearer-token session, then a username/password session. The
returned :class:`EarthdataSession` re-applies the credentials when NASA's
OAuth flow redirects to URS (``requests`` strips ``Authorization`` on
cross-host redirects).

earthaccess (:meth:`EarthdataAuth.earthaccess_login`): credentials resolved
above are handed to earthaccess through its ``environment`` strategy, so
config-only and alternate-host ``.netrc`` credentials work there too; with
nothing resolved, earthaccess's own strategy (default ``"all"``) runs.

Secrets are never logged: log lines name the *source* only.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

from .utils import create_robust_session, netrc_candidate_paths, read_netrc_credentials

_module_logger = logging.getLogger(__name__)

URS_HOST = 'urs.earthdata.nasa.gov'

#: ``.netrc`` machines accepted for Earthdata Login, in lookup order.
EARTHDATA_NETRC_HOSTS: Tuple[str, ...] = (
    URS_HOST,
    'earthdata.nasa.gov',
    'appeears.earthdatacloud.nasa.gov',
    'opendap.earthdata.nasa.gov',
)

USERNAME_KEY = 'EARTHDATA_USERNAME'
PASSWORD_KEY = 'EARTHDATA_PASSWORD'
TOKEN_KEY = 'EARTHDATA_TOKEN'

# Redirect targets that get the Bearer token re-applied.
_TOKEN_HOST_SUFFIXES = ('.earthdata.nasa.gov', '.ornl.gov')

# earthaccess reads credentials from os.environ; serialise the brief window
# in which resolved credentials are exported for its "environment" strategy.
_EARTHACCESS_ENV_LOCK = threading.Lock()


class EarthdataAuthError(RuntimeError, PermissionError):
    """No usable NASA Earthdata credentials.

    Subclasses ``RuntimeError`` (what handlers raised before) and
    ``PermissionError`` (what the native backend maps to ``AuthRequired``).
    """


@dataclass(frozen=True)
class EarthdataCredentials:
    """A resolved username/password pair and where it came from."""

    username: str
    password: str = field(repr=False)
    source: str = ''


def missing_credentials_message(purpose: Optional[str] = None) -> str:
    """Actionable "credentials not found" text listing every source checked."""
    netrc_paths = ', '.join(str(p) for p in netrc_candidate_paths())
    what = f" for {purpose}" if purpose else ''
    return (
        f"NASA Earthdata credentials not found{what}. Checked, in order:\n"
        f"  1. config keys {TOKEN_KEY}, or {USERNAME_KEY} + {PASSWORD_KEY}\n"
        f"  2. environment variables {TOKEN_KEY}, or {USERNAME_KEY} + {PASSWORD_KEY}\n"
        f"  3. .netrc ({netrc_paths}), machine {URS_HOST}\n"
        "  4. an earthaccess login in this process\n"
        "Fix with one of:\n"
        f"  - export {TOKEN_KEY}=<token>  (generate at https://{URS_HOST} -> Generate Token)\n"
        f"  - add to ~/.netrc: machine {URS_HOST} login <user> password <pass>  (chmod 600)\n"
        "  - python -c \"import earthaccess; earthaccess.login(persist=True)\"\n"
        f"Register for a free account at https://{URS_HOST}/users/new"
    )


class EarthdataSession(requests.Session):
    """``requests.Session`` that survives NASA's URS OAuth redirect flow.

    Data hosts (GES DISC, ORNL DAAC, PO.DAAC, ...) redirect to
    ``urs.earthdata.nasa.gov`` to authenticate. ``requests`` strips the
    ``Authorization`` header on cross-host redirects, so this session
    re-applies a Bearer token on redirects to ``*.earthdata.nasa.gov`` /
    ``*.ornl.gov``, or basic credentials on redirects to URS.

    Args:
        token: Bearer token (sent to the origin host too).
        username, password: Basic credentials.
        origin_auth: Also send basic credentials to the origin host
            (``session.auth``), not only to URS.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        origin_auth: bool = False,
    ):
        super().__init__()
        self._token = token
        self._basic = (username, password) if username and password else None
        if token:
            self.headers.update({'Authorization': f'Bearer {token}'})
        elif self._basic and origin_auth:
            self.auth = self._basic

    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        host = urlparse(prepared_request.url).hostname or ''
        if self._token:
            if host == URS_HOST or host.endswith(_TOKEN_HOST_SUFFIXES):
                prepared_request.headers['Authorization'] = f'Bearer {self._token}'
        elif self._basic and host == URS_HOST:
            prepared_request.prepare_auth(self._basic)


@contextmanager
def _exported_env(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily set (and then restore) environment variables."""
    with _EARTHACCESS_ENV_LOCK:
        saved = {key: os.environ.get(key) for key in (USERNAME_KEY, PASSWORD_KEY, TOKEN_KEY)}
        for key in saved:
            os.environ.pop(key, None)
        os.environ.update(values)
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class EarthdataAuth:
    """Resolve NASA Earthdata credentials and build authenticated clients.

    Args:
        config: A ``SymfluenceConfig`` or flat config dict. Read with
            ``resolve_config_value(config, lambda: None, None, KEY)`` (the
            credential keys have no typed field).
        logger: Logger for source/diagnostic messages (never secrets).
        config_lookup: Alternative to ``config``: ``key -> value`` callable,
            e.g. a handler's ``self._get_config_value`` bound to ``dict_key``.
        netrc_hosts: ``.netrc`` machines to accept, in order.
    """

    def __init__(
        self,
        config: Any = None,
        logger: Optional[logging.Logger] = None,
        *,
        config_lookup: Optional[Callable[[str], Any]] = None,
        netrc_hosts: Sequence[str] = EARTHDATA_NETRC_HOSTS,
    ):
        self._config = config
        self._config_lookup = config_lookup
        self.logger = logger or _module_logger
        self.netrc_hosts = tuple(netrc_hosts)

    # ----- lookups -------------------------------------------------------

    def _config_value(self, key: str) -> Any:
        if self._config_lookup is not None:
            return self._config_lookup(key)
        if self._config is None:
            return None
        from symfluence.core.mixins.config import resolve_config_value
        return resolve_config_value(self._config, lambda: None, None, key)

    def resolve(self) -> Optional[EarthdataCredentials]:
        """Return the first complete username/password pair, or None."""
        username = self._config_value(USERNAME_KEY)
        password = self._config_value(PASSWORD_KEY)
        if username and password:
            return EarthdataCredentials(str(username), str(password), 'config')

        username = os.environ.get(USERNAME_KEY)
        password = os.environ.get(PASSWORD_KEY)
        if username and password:
            return EarthdataCredentials(username, password, 'environment')

        found = read_netrc_credentials(self.netrc_hosts, logger=self.logger)
        if found:
            username, password, where = found
            return EarthdataCredentials(username, password, f'.netrc ({where})')

        ea_auth = self._earthaccess_session_auth()
        if ea_auth is not None:
            return EarthdataCredentials(ea_auth[0], ea_auth[1], 'earthaccess login')
        return None

    def credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """``(username, password)``, or ``(None, None)`` if not found."""
        creds = self.resolve()
        if creds is None:
            return None, None
        self.logger.debug(f"Using Earthdata credentials from {creds.source}")
        return creds.username, creds.password

    def require_credentials(self, purpose: Optional[str] = None) -> Tuple[str, str]:
        """``(username, password)``; raise :class:`EarthdataAuthError` if missing."""
        username, password = self.credentials()
        if not username or not password:
            raise EarthdataAuthError(missing_credentials_message(purpose))
        return username, password

    def token(self) -> Optional[str]:
        """Bearer token from config, then environment; None if absent."""
        token = self._config_value(TOKEN_KEY)
        if token:
            self.logger.debug(f"Using Earthdata token from config ({TOKEN_KEY})")
            return str(token)
        token = os.environ.get(TOKEN_KEY)
        if token:
            self.logger.debug(f"Using Earthdata token from {TOKEN_KEY} env var")
            return token
        return None

    @staticmethod
    def _earthaccess_session_auth() -> Optional[Tuple[str, str]]:
        """Username/password of an in-process earthaccess login, if any."""
        earthaccess = sys.modules.get('earthaccess')
        if earthaccess is None:
            return None
        try:
            ea_auth = earthaccess.__auth__  # module __getattr__; no login side effect
        except AttributeError:
            return None
        if ea_auth is None or getattr(ea_auth, 'authenticated', False) is not True:
            return None
        username = getattr(ea_auth, 'username', None)
        password = getattr(ea_auth, 'password', None)
        if isinstance(username, str) and isinstance(password, str) and username and password:
            return username, password
        return None

    # ----- earthaccess ---------------------------------------------------

    def earthaccess_login(self, *, strategy: str = 'all', required: bool = False) -> Any:
        """Log in to earthaccess; return its ``Auth`` or None on failure.

        Resolved token/credentials are passed through earthaccess's
        ``environment`` strategy; otherwise ``strategy`` is used as given.

        Raises:
            EarthdataAuthError: if ``required`` and the login fails or
                earthaccess is not installed.
        """
        try:
            import earthaccess
        except ImportError as exc:
            if required:
                raise EarthdataAuthError(
                    "earthaccess is not installed (pip install earthaccess)"
                ) from exc
            return None

        token = self.token()
        exported: dict = {}
        if token:
            exported = {TOKEN_KEY: token}
        else:
            username, password = self.credentials()
            if username and password:
                exported = {USERNAME_KEY: username, PASSWORD_KEY: password}

        try:
            if exported:
                with _exported_env(exported):
                    ea_auth = earthaccess.login(strategy='environment')
            else:
                ea_auth = earthaccess.login(strategy=strategy)
        except Exception as exc:  # noqa: BLE001 — earthaccess raises its own exception types
            reason = f"earthaccess login failed ({type(exc).__name__})"
            if required:
                raise EarthdataAuthError(f"{reason}.\n{missing_credentials_message()}") from exc
            self.logger.warning(reason)
            return None

        if not ea_auth or getattr(ea_auth, 'authenticated', True) is False:
            if required:
                raise EarthdataAuthError(
                    f"earthaccess login was not authenticated.\n{missing_credentials_message()}"
                )
            self.logger.warning("earthaccess login was not authenticated")
            return None
        return ea_auth

    def earthaccess_session(self) -> Optional[requests.Session]:
        """An earthaccess HTTPS session, or None if earthaccess is unavailable/fails."""
        if self.earthaccess_login() is None:
            return None
        import earthaccess
        return earthaccess.get_requests_https_session()

    # ----- requests sessions --------------------------------------------

    def session(
        self,
        *,
        prefer_earthaccess: bool = False,
        origin_auth: bool = True,
        required: bool = False,
        purpose: Optional[str] = None,
        max_retries: Optional[int] = None,
        backoff_factor: float = 1.0,
    ) -> requests.Session:
        """An authenticated ``requests`` session.

        Order: earthaccess session (if ``prefer_earthaccess``), Bearer token,
        username/password. With nothing found, raise
        :class:`EarthdataAuthError` if ``required``, else warn and return an
        unauthenticated :class:`EarthdataSession`.

        Args:
            origin_auth: Send basic credentials to the origin host as well
                as to URS.
            max_retries, backoff_factor: Mount retrying adapters (see
                :func:`~symfluence.data.acquisition.utils.create_robust_session`).
        """
        if prefer_earthaccess:
            ea_session = self.earthaccess_session()
            if ea_session is not None:
                self.logger.debug("Using earthaccess session for Earthdata")
                return ea_session

        token = self.token()
        if token:
            session = EarthdataSession(token=token)
        else:
            username, password = self.credentials()
            if username and password:
                session = EarthdataSession(
                    username=username, password=password, origin_auth=origin_auth,
                )
            elif required:
                raise EarthdataAuthError(missing_credentials_message(purpose))
            else:
                self.logger.warning(missing_credentials_message(purpose))
                session = EarthdataSession()

        if max_retries is not None:
            create_robust_session(
                max_retries=max_retries, backoff_factor=backoff_factor, session=session,
            )
        return session


__all__ = [
    'EARTHDATA_NETRC_HOSTS',
    'URS_HOST',
    'EarthdataAuth',
    'EarthdataAuthError',
    'EarthdataCredentials',
    'EarthdataSession',
    'missing_credentials_message',
]
