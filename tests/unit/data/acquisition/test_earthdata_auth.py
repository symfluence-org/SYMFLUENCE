# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the shared NASA Earthdata auth helper (earthdata_auth.py).

Every test runs with HOME, ``Path.home()`` and ``NETRC`` pointed at a temp
directory and the Earthdata env vars cleared, so the real ~/.netrc and the
developer's environment are never read. No network calls are made.
"""
from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from symfluence.data.acquisition.earthdata_auth import (
    EarthdataAuth,
    EarthdataAuthError,
    EarthdataSession,
    missing_credentials_message,
)
from symfluence.data.acquisition.utils import read_netrc_credentials

pytestmark = [pytest.mark.unit, pytest.mark.acquisition]

SECRETS = ('cfg-secret', 'env-secret', 'netrc-secret', 'ea-secret', 'tok-secret', 'bad-secret')
_ENV_KEYS = ('EARTHDATA_USERNAME', 'EARTHDATA_PASSWORD', 'EARTHDATA_TOKEN', 'NETRC')


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HOME with no .netrc, no Earthdata env vars, no earthaccess state."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.delitem(sys.modules, 'earthaccess', raising=False)
    return tmp_path


def _write_netrc(path: Path, host: str = 'urs.earthdata.nasa.gov',
                 user: str = 'netrc_user', password: str = 'netrc-secret') -> Path:
    path.write_text(f"machine {host} login {user} password {password}\n")
    path.chmod(0o600)
    return path


def _fake_earthaccess(monkeypatch, *, authenticated=False, username=None, password=None,
                      login=None):
    """Install a stand-in ``earthaccess`` module in sys.modules."""
    module = types.ModuleType('earthaccess')
    module.__auth__ = types.SimpleNamespace(
        authenticated=authenticated, username=username, password=password,
    )
    module.login = login or MagicMock(return_value=types.SimpleNamespace(authenticated=True))
    module.get_requests_https_session = MagicMock(return_value=MagicMock(name='ea_session'))
    monkeypatch.setitem(sys.modules, 'earthaccess', module)
    return module


def _assert_no_secret(caplog):
    text = '\n'.join(r.getMessage() for r in caplog.records)
    for secret in SECRETS:
        assert secret not in text


# ----- each source alone ------------------------------------------------------

class TestSources:

    def test_config(self, home):
        auth = EarthdataAuth({'EARTHDATA_USERNAME': 'cfg_user', 'EARTHDATA_PASSWORD': 'cfg-secret'})
        assert auth.credentials() == ('cfg_user', 'cfg-secret')
        assert auth.resolve().source == 'config'

    def test_config_lookup_callable(self, home):
        values = {'EARTHDATA_USERNAME': 'cfg_user', 'EARTHDATA_PASSWORD': 'cfg-secret'}
        auth = EarthdataAuth(config_lookup=values.get)
        assert auth.credentials() == ('cfg_user', 'cfg-secret')

    def test_environment(self, home, monkeypatch):
        monkeypatch.setenv('EARTHDATA_USERNAME', 'env_user')
        monkeypatch.setenv('EARTHDATA_PASSWORD', 'env-secret')
        auth = EarthdataAuth()
        assert auth.credentials() == ('env_user', 'env-secret')
        assert auth.resolve().source == 'environment'

    def test_home_netrc(self, home):
        _write_netrc(home / '.netrc')
        assert EarthdataAuth().credentials() == ('netrc_user', 'netrc-secret')

    def test_netrc_env_var(self, home, tmp_path_factory, monkeypatch):
        other = tmp_path_factory.mktemp('elsewhere') / 'auth'
        _write_netrc(other)
        monkeypatch.setenv('NETRC', str(other))
        creds = EarthdataAuth().resolve()
        assert (creds.username, creds.password) == ('netrc_user', 'netrc-secret')
        assert str(other) in creds.source

    def test_windows_style_netrc(self, home):
        _write_netrc(home / '_netrc')
        assert EarthdataAuth().credentials() == ('netrc_user', 'netrc-secret')

    @pytest.mark.parametrize('host', [
        'earthdata.nasa.gov', 'appeears.earthdatacloud.nasa.gov', 'opendap.earthdata.nasa.gov',
    ])
    def test_netrc_alternate_hosts(self, home, host):
        _write_netrc(home / '.netrc', host=host)
        assert EarthdataAuth().credentials() == ('netrc_user', 'netrc-secret')

    def test_netrc_unrelated_host_ignored(self, home):
        _write_netrc(home / '.netrc', host='example.com')
        assert EarthdataAuth().credentials() == (None, None)

    def test_in_process_earthaccess_login(self, home, monkeypatch):
        _fake_earthaccess(monkeypatch, authenticated=True, username='ea_user', password='ea-secret')
        auth = EarthdataAuth()
        assert auth.credentials() == ('ea_user', 'ea-secret')
        assert auth.resolve().source == 'earthaccess login'

    def test_unauthenticated_earthaccess_ignored(self, home, monkeypatch):
        _fake_earthaccess(monkeypatch, authenticated=False, username='ea_user', password='ea-secret')
        assert EarthdataAuth().credentials() == (None, None)

    def test_partial_pairs_are_skipped(self, home, monkeypatch):
        monkeypatch.setenv('EARTHDATA_USERNAME', 'env_user')  # no password
        auth = EarthdataAuth({'EARTHDATA_PASSWORD': 'cfg-secret'})  # no username
        assert auth.credentials() == (None, None)

    def test_token_from_config(self, home):
        assert EarthdataAuth({'EARTHDATA_TOKEN': 'tok-secret'}).token() == 'tok-secret'

    def test_token_from_env(self, home, monkeypatch):
        monkeypatch.setenv('EARTHDATA_TOKEN', 'tok-secret')
        assert EarthdataAuth().token() == 'tok-secret'


# ----- precedence -------------------------------------------------------------

class TestPrecedence:

    @pytest.fixture
    def all_sources(self, home, monkeypatch):
        _write_netrc(home / '.netrc')
        monkeypatch.setenv('EARTHDATA_USERNAME', 'env_user')
        monkeypatch.setenv('EARTHDATA_PASSWORD', 'env-secret')
        _fake_earthaccess(monkeypatch, authenticated=True, username='ea_user', password='ea-secret')
        return home

    def test_config_beats_env(self, all_sources):
        auth = EarthdataAuth({'EARTHDATA_USERNAME': 'cfg_user', 'EARTHDATA_PASSWORD': 'cfg-secret'})
        assert auth.credentials() == ('cfg_user', 'cfg-secret')

    def test_env_beats_netrc(self, all_sources):
        assert EarthdataAuth().credentials() == ('env_user', 'env-secret')

    def test_netrc_beats_earthaccess(self, all_sources, monkeypatch):
        monkeypatch.delenv('EARTHDATA_USERNAME')
        assert EarthdataAuth().credentials() == ('netrc_user', 'netrc-secret')

    def test_netrc_env_var_file_before_home(self, home, tmp_path_factory, monkeypatch):
        _write_netrc(home / '.netrc', user='home_user')
        other = _write_netrc(tmp_path_factory.mktemp('n') / 'netrc', user='env_file_user')
        monkeypatch.setenv('NETRC', str(other))
        assert EarthdataAuth().credentials()[0] == 'env_file_user'

    def test_netrc_env_var_missing_file_falls_back_to_home(self, home, monkeypatch):
        _write_netrc(home / '.netrc')
        monkeypatch.setenv('NETRC', str(home / 'does-not-exist'))
        assert EarthdataAuth().credentials() == ('netrc_user', 'netrc-secret')

    def test_primary_host_beats_alternate(self, home):
        (home / '.netrc').write_text(
            "machine earthdata.nasa.gov login alt_user password x\n"
            "machine urs.earthdata.nasa.gov login urs_user password y\n"
        )
        assert EarthdataAuth().credentials()[0] == 'urs_user'

    def test_token_config_beats_env(self, home, monkeypatch):
        monkeypatch.setenv('EARTHDATA_TOKEN', 'env-token')
        assert EarthdataAuth({'EARTHDATA_TOKEN': 'cfg-token'}).token() == 'cfg-token'


# ----- missing credentials ----------------------------------------------------

class TestMissing:

    def test_credentials_none(self, home):
        assert EarthdataAuth().credentials() == (None, None)
        assert EarthdataAuth().token() is None

    def test_require_raises_informative_error(self, home):
        with pytest.raises(EarthdataAuthError) as exc_info:
            EarthdataAuth().require_credentials('MOD16 ET')
        err = exc_info.value
        assert isinstance(err, RuntimeError) and isinstance(err, PermissionError)
        msg = str(err)
        for needle in ('MOD16 ET', 'EARTHDATA_USERNAME', 'EARTHDATA_TOKEN',
                       'urs.earthdata.nasa.gov', str(home / '.netrc'), 'earthaccess'):
            assert needle in msg

    def test_required_session_raises(self, home):
        with pytest.raises(EarthdataAuthError):
            EarthdataAuth().session(required=True)

    def test_optional_session_warns_and_is_anonymous(self, home, caplog):
        with caplog.at_level(logging.WARNING):
            session = EarthdataAuth().session(purpose='Daymet')
        assert isinstance(session, EarthdataSession)
        assert 'Authorization' not in session.headers and session.auth is None
        assert 'Daymet' in caplog.text and 'EARTHDATA_TOKEN' in caplog.text

    def test_malformed_netrc_is_skipped(self, home):
        (home / '.netrc').write_text("machine urs.earthdata.nasa.gov login u password\n bad-secret ]]\n")
        assert EarthdataAuth().credentials() == (None, None)

    def test_message_lists_sources(self):
        msg = missing_credentials_message()
        assert msg.index('config') < msg.index('environment') < msg.index('.netrc') < msg.index('earthaccess login')


# ----- secrets never logged ---------------------------------------------------

class TestNoSecretsLogged:

    def test_every_source(self, home, monkeypatch, caplog):
        caplog.set_level(logging.DEBUG)
        log = logging.getLogger('test.earthdata')
        EarthdataAuth({'EARTHDATA_USERNAME': 'u', 'EARTHDATA_PASSWORD': 'cfg-secret',
                       'EARTHDATA_TOKEN': 'tok-secret'}, log).session()
        monkeypatch.setenv('EARTHDATA_USERNAME', 'u')
        monkeypatch.setenv('EARTHDATA_PASSWORD', 'env-secret')
        EarthdataAuth(logger=log).session()
        monkeypatch.delenv('EARTHDATA_USERNAME')
        _write_netrc(home / '.netrc')
        EarthdataAuth(logger=log).session()
        (home / '.netrc').write_text("machine urs.earthdata.nasa.gov password bad-secret ]] login\n")
        EarthdataAuth(logger=log).credentials()
        (home / '.netrc').unlink()
        _fake_earthaccess(monkeypatch, authenticated=True, username='u', password='ea-secret')
        EarthdataAuth(logger=log).credentials()
        assert caplog.records  # sources were logged ...
        _assert_no_secret(caplog)  # ... secrets were not

    def test_earthaccess_failure(self, home, monkeypatch, caplog):
        caplog.set_level(logging.DEBUG)
        monkeypatch.setenv('EARTHDATA_USERNAME', 'u')
        monkeypatch.setenv('EARTHDATA_PASSWORD', 'env-secret')
        _fake_earthaccess(monkeypatch, login=MagicMock(side_effect=ValueError('bad env-secret')))
        assert EarthdataAuth().earthaccess_login() is None
        assert 'earthaccess login failed' in caplog.text
        _assert_no_secret(caplog)

    def test_credentials_repr_hides_password(self, home):
        creds = EarthdataAuth({'EARTHDATA_USERNAME': 'u', 'EARTHDATA_PASSWORD': 'cfg-secret'}).resolve()
        assert 'cfg-secret' not in repr(creds)


# ----- sessions ---------------------------------------------------------------

def _redirect(session: requests.Session, url: str) -> requests.PreparedRequest:
    """Run the session's redirect auth hook for a hop to ``url``."""
    prepared = requests.Request('GET', url).prepare()
    response = MagicMock()
    response.request = requests.Request('GET', 'https://data.example.org/file').prepare()
    session.trust_env = False  # keep requests' own ~/.netrc lookup out of the test
    session.rebuild_auth(prepared, response)
    return prepared


class TestSessions:

    def test_token_session(self, home):
        session = EarthdataAuth({'EARTHDATA_TOKEN': 'tok-secret'}).session()
        assert session.headers['Authorization'] == 'Bearer tok-secret'
        for url in ('https://urs.earthdata.nasa.gov/oauth', 'https://opendap.earthdata.nasa.gov/x',
                    'https://thredds.daac.ornl.gov/x'):
            assert _redirect(session, url).headers['Authorization'] == 'Bearer tok-secret'
        assert 'Authorization' not in _redirect(session, 'https://elsewhere.example.com/').headers

    def test_token_preferred_over_credentials(self, home):
        auth = EarthdataAuth({'EARTHDATA_TOKEN': 'tok-secret', 'EARTHDATA_USERNAME': 'u',
                              'EARTHDATA_PASSWORD': 'cfg-secret'})
        session = auth.session()
        assert session.auth is None and 'Bearer' in session.headers['Authorization']

    def test_basic_session_reapplies_at_urs_only(self, home):
        _write_netrc(home / '.netrc')
        session = EarthdataAuth().session(origin_auth=False)
        assert session.auth is None
        assert _redirect(session, 'https://urs.earthdata.nasa.gov/oauth').headers['Authorization'].startswith('Basic ')
        assert 'Authorization' not in _redirect(session, 'https://goldsmr4.gesdisc.eosdis.nasa.gov/').headers

    def test_basic_session_origin_auth(self, home):
        _write_netrc(home / '.netrc')
        assert EarthdataAuth().session().auth == ('netrc_user', 'netrc-secret')

    def test_max_retries_mounts_adapter(self, home):
        session = EarthdataAuth({'EARTHDATA_TOKEN': 't'}).session(max_retries=3)
        assert session.get_adapter('https://x').max_retries.total == 3

    def test_prefer_earthaccess(self, home, monkeypatch):
        module = _fake_earthaccess(monkeypatch)
        session = EarthdataAuth().session(prefer_earthaccess=True)
        assert session is module.get_requests_https_session.return_value

    def test_prefer_earthaccess_falls_back_without_earthaccess(self, home, monkeypatch):
        monkeypatch.setitem(sys.modules, 'earthaccess', None)  # import -> ImportError
        session = EarthdataAuth({'EARTHDATA_TOKEN': 'tok-secret'}).session(prefer_earthaccess=True)
        assert isinstance(session, EarthdataSession)
        assert session.headers['Authorization'] == 'Bearer tok-secret'


# ----- earthaccess login --------------------------------------------------------

class TestEarthaccessLogin:

    def test_resolved_credentials_exported_then_restored(self, home, monkeypatch):
        seen = {}

        def login(strategy):
            seen['strategy'] = strategy
            seen['env'] = {k: os.environ.get(k) for k in _ENV_KEYS[:3]}
            return types.SimpleNamespace(authenticated=True)

        _fake_earthaccess(monkeypatch, login=login)
        auth = EarthdataAuth({'EARTHDATA_USERNAME': 'cfg_user', 'EARTHDATA_PASSWORD': 'cfg-secret'})
        assert auth.earthaccess_login() is not None
        assert seen['strategy'] == 'environment'
        assert seen['env'] == {'EARTHDATA_USERNAME': 'cfg_user', 'EARTHDATA_PASSWORD': 'cfg-secret',
                               'EARTHDATA_TOKEN': None}
        assert 'EARTHDATA_PASSWORD' not in os.environ

    def test_token_exported(self, home, monkeypatch):
        module = _fake_earthaccess(monkeypatch)
        monkeypatch.setenv('EARTHDATA_TOKEN', 'tok-secret')
        EarthdataAuth().earthaccess_login()
        module.login.assert_called_once_with(strategy='environment')
        assert os.environ['EARTHDATA_TOKEN'] == 'tok-secret'

    def test_nothing_resolved_uses_given_strategy(self, home, monkeypatch):
        module = _fake_earthaccess(monkeypatch)
        EarthdataAuth().earthaccess_login(strategy='netrc')
        module.login.assert_called_once_with(strategy='netrc')

    def test_required_failure_raises(self, home, monkeypatch):
        _fake_earthaccess(monkeypatch, login=MagicMock(return_value=types.SimpleNamespace(authenticated=False)))
        with pytest.raises(EarthdataAuthError):
            EarthdataAuth().earthaccess_login(required=True)

    def test_not_installed(self, home, monkeypatch):
        monkeypatch.setitem(sys.modules, 'earthaccess', None)
        assert EarthdataAuth().earthaccess_login() is None
        with pytest.raises(EarthdataAuthError, match='not installed'):
            EarthdataAuth().earthaccess_login(required=True)


# ----- the shared .netrc reader and handler wiring ---------------------------------

class TestNetrcReaderAndHandlers:

    def test_read_netrc_other_host(self, home):
        _write_netrc(home / '.netrc', host='ameriflux.lbl.gov', user='id', password='mail')
        found = read_netrc_credentials(['ameriflux.lbl.gov'])
        assert found[:2] == ('id', 'mail')
        assert read_netrc_credentials(['urs.earthdata.nasa.gov']) is None

    def test_no_netrc_file(self, home):
        assert read_netrc_credentials(['urs.earthdata.nasa.gov']) is None

    def test_weaker_overrides_removed(self):
        from symfluence.data.acquisition.base import BaseAcquisitionHandler
        from symfluence.data.acquisition.handlers.daymet import DaymetAcquirer
        from symfluence.data.acquisition.handlers.modis_et import MOD16ETAcquirer
        from symfluence.data.acquisition.handlers.modis_lai import MODISLAIAcquirer

        base = BaseAcquisitionHandler._get_earthdata_credentials
        for cls in (MOD16ETAcquirer, MODISLAIAcquirer, DaymetAcquirer):
            assert cls._get_earthdata_credentials is base

    def test_handler_reads_config_credentials(self, home):
        """modis_et's old override ignored config; the base one doesn't."""
        from symfluence.data.acquisition.handlers.modis_et import MOD16ETAcquirer

        handler = MOD16ETAcquirer.__new__(MOD16ETAcquirer)
        handler.logger = logging.getLogger('test')
        handler._get_config_value = lambda accessor, default=None, dict_key=None: {
            'EARTHDATA_USERNAME': 'cfg_user', 'EARTHDATA_PASSWORD': 'cfg-secret',
        }.get(dict_key, default)
        assert handler._get_earthdata_credentials() == ('cfg_user', 'cfg-secret')
