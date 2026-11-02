# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for registry-first streamflow dispatch in ObservedDataProcessor.

Covers the community-backend design: ``process_streamflow_data()`` consults
``R.observation_handlers`` before the hardcoded legacy provider branches when
either ``DATA_ACCESS: community`` is set or the provider is unknown to the
legacy dispatch. The default path (no DATA_ACCESS, built-in provider) must be
untouched.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.registries import R
from symfluence.data.acquisition.observed_processor import ObservedDataProcessor
from symfluence.data.observation.base import BaseObservationHandler
from symfluence.data.observation.registry import ObservationRegistry


def _base_config(tmp_path, provider: str, **extra) -> dict:
    config = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'DOMAIN_DEFINITION_METHOD': 'subset',
        'SUB_GRID_DISCRETIZATION': 'GRU',
        'EXPERIMENT_ID': 'test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': '3600',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'STREAMFLOW_DATA_PROVIDER': provider,
        'STREAMFLOW_RAW_PATH': 'default',
        'STREAMFLOW_PROCESSED_PATH': 'default',
        'STREAMFLOW_RAW_NAME': 'test_flow.csv',
    }
    config.update(extra)
    return config


class _FakeHandler(BaseObservationHandler):
    """Minimal observation handler recording acquire/process calls."""

    obs_type = 'streamflow'
    source_name = 'FAKE'

    acquire_calls: list = []
    process_calls: list = []

    def acquire(self) -> Path:
        type(self).acquire_calls.append(self)
        return Path('/tmp/fake_raw.csv')  # test stub path, never opened

    def process(self, input_path: Path) -> Path:
        type(self).process_calls.append(input_path)
        return Path('/tmp/fake_processed.csv')  # test stub path, never opened


@pytest.fixture
def fake_handler():
    """Provide a fresh fake handler class with call recording reset."""
    _FakeHandler.acquire_calls = []
    _FakeHandler.process_calls = []
    return _FakeHandler


@pytest.fixture
def no_observation_backends():
    """Strip the ObservationBackend tier so the handler tier is reachable.

    With the csfs plugin installed, ``R.observation_backends`` holds the
    CommunityObservationBackend, which serves community-mode primary
    streamflow BEFORE the registry-handler tier. Tests targeting the handler
    tier (the fallthrough) must run without it — and must hold either way.
    """
    removed: list[tuple[str, object]] = []
    for key in list(R.observation_backends.keys()):
        removed.append((key, R.observation_backends.get(key)))
        R.observation_backends.remove(key)

    yield

    for key, entry in removed:
        if R.observation_backends.get(key) is None:
            R.observation_backends.add(key, entry)


@pytest.fixture
def register_handler(fake_handler):
    """Register the fake handler under a key and restore registry state after."""
    injected: list[str] = []

    def _register(key: str):
        assert key not in R.observation_handlers, f"refusing to clobber existing handler {key!r}"
        R.observation_handlers.add(key)(fake_handler)
        injected.append(key)
        return fake_handler

    yield _register

    for key in injected:
        R.observation_handlers.remove(key)


def test_default_path_takes_legacy_branch_even_when_handler_registered(tmp_path):
    """Provider USGS without DATA_ACCESS: community must use the legacy branch.

    No registration needed: whether or not the real csfs plugin occupies
    'usgs' (it does when community-streamflow-service is installed), the
    registry must never be consulted on the default path.
    """
    logger = MagicMock()
    processor = ObservedDataProcessor(_base_config(tmp_path, 'USGS'), logger)

    with patch.object(ObservationRegistry, 'get_handler') as get_handler:
        processor.process_streamflow_data()

    get_handler.assert_not_called()
    # The dispatch note is per-item detail: emitted at DEBUG per the logging
    # level policy (ADR 0005), never at INFO.
    debug_messages = [str(call.args[0]) for call in logger.debug.call_args_list]
    assert any('USGS streamflow data handled by formalized observation handler' in m for m in debug_messages)
    logger.error.assert_not_called()


def test_community_backend_routes_legacy_provider_through_registry(
    tmp_path, register_handler, fake_handler, no_observation_backends
):
    """DATA_ACCESS: community sends a legacy provider to its registered handler.

    This exercises the registry-handler tier — the fallthrough below the
    ObservationBackend tier (stripped here; its routing is covered in
    tests/unit/data/backends/test_observation_seam.py). Adapts to the
    environment: when the real csfs plugin already occupies 'usgs'
    (community-streamflow-service installed), membership is satisfied and
    resolution is intercepted; otherwise the fake is registered outright
    (the CI case, no csfs installed).
    """
    logger = MagicMock()
    processor = ObservedDataProcessor(_base_config(tmp_path, 'USGS', DATA_ACCESS='community'), logger)

    if 'usgs' in R.observation_handlers:
        with patch.object(
            ObservationRegistry, 'get_handler',
            side_effect=lambda key, config, log: fake_handler(config, log),
        ) as get_handler:
            processor.process_streamflow_data()
        assert get_handler.call_args.args[0] == 'usgs'
    else:
        register_handler('usgs')
        processor.process_streamflow_data()

    assert len(fake_handler.acquire_calls) == 1
    assert fake_handler.process_calls == [Path('/tmp/fake_raw.csv')]
    info_messages = [str(call.args[0]) for call in logger.info.call_args_list]
    assert not any('handled by formalized observation handler' in m for m in info_messages)
    logger.error.assert_not_called()


def test_unknown_provider_uses_registry_without_data_access(tmp_path, register_handler, fake_handler):
    """A provider unknown to the legacy dispatch uses the registry even with default DATA_ACCESS.

    This is the CSFS-style plugin route; a synthetic key is used because the
    real ``csfs`` plugin may be installed (and registered) in the test env.
    """
    register_handler('fakepluginobs')
    logger = MagicMock()
    processor = ObservedDataProcessor(_base_config(tmp_path, 'FAKEPLUGINOBS'), logger)

    processor.process_streamflow_data()

    assert len(fake_handler.acquire_calls) == 1
    assert fake_handler.process_calls == [Path('/tmp/fake_raw.csv')]
    logger.error.assert_not_called()


def test_unknown_provider_without_handler_reports_unsupported(tmp_path):
    """Unknown provider with no registered handler hits the existing unsupported-provider error."""
    logger = MagicMock()
    processor = ObservedDataProcessor(_base_config(tmp_path, 'NOSUCH'), logger)

    processor.process_streamflow_data()

    error_messages = [str(call.args[0]) for call in logger.error.call_args_list]
    assert any('Unsupported streamflow data provider: NOSUCH' in m for m in error_messages)


def test_forcing_config_accepts_cfs_dataset():
    """The ForcingDatasetType Literal accepts the plugin-provided CFS dataset."""
    from symfluence.core.config.models.forcing import ForcingConfig

    config = ForcingConfig(FORCING_DATASET='CFS')
    assert config.dataset == 'CFS'
