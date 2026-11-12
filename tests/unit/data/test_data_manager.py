# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the DataManager facade (review item 17).

Scoped to construction + the delegation contract: the acquire_* methods forward to
the AcquisitionService. Real acquisition (network/I/O) is out of scope here. The
three services are patched so no downloads or heavy I/O occur.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data import data_manager as dm_mod
from symfluence.data.data_manager import DataManager


@pytest.fixture
def data_manager():
    cfg = SymfluenceConfig.from_minimal(
        domain_name="test_basin", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
    )
    logger = logging.getLogger("test.data_manager")
    # Patch the services so construction wires mocks instead of real acquisition.
    # _get_service logs service_class.__name__, so the class mocks need one.
    with patch.object(dm_mod, "AcquisitionService", MagicMock(__name__="AcquisitionService")), \
         patch.object(dm_mod, "EMEarthIntegrator", MagicMock(__name__="EMEarthIntegrator")), \
         patch.object(dm_mod, "VariableHandler", MagicMock(__name__="VariableHandler")):
        dm = DataManager(cfg, logger, reporting_manager=None)
    return dm


def test_construction_wires_services(data_manager):
    assert data_manager.acquisition_service is not None
    assert data_manager.em_earth_integrator is not None
    assert data_manager.variable_handler is not None


def test_acquire_attributes_delegates(data_manager):
    data_manager.acquire_attributes()
    data_manager.acquisition_service.acquire_attributes.assert_called_once()


def test_acquire_forcings_delegates(data_manager):
    data_manager.acquire_forcings()
    data_manager.acquisition_service.acquire_forcings.assert_called_once()


def test_acquire_observations_delegates(data_manager):
    data_manager.acquire_observations()
    data_manager.acquisition_service.acquire_observations.assert_called_once()


def test_get_status_returns_dict(data_manager):
    status = data_manager.get_status()
    assert isinstance(status, dict)
