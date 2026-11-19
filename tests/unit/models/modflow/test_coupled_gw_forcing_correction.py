# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION through the COUPLED_GW (land model + MODFLOW) parameter manager.

The coupled manager is a BaseParameterManager over the land model's parameters
plus MODFLOW's, so a plan built with the land model's adapter (HYPE here)
translates the common parameter to the land model's native ones, which the
coupled manager routes to the land model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R

LOG = logging.getLogger('test_coupled_gw_fc')


class _Land(BaseParameterManager):
    applied: List[Dict[str, Any]] = []

    def _get_parameter_names(self):
        return ['rrcs1', 'pcurain']

    def _load_parameter_bounds(self):
        return {'rrcs1': {'min': 0.01, 'max': 1.0}, 'pcurain': {'min': 0.0, 'max': 0.3}}

    def update_model_files(self, params):
        _Land.applied.append(dict(params))
        return True

    def get_initial_parameters(self):
        return {'rrcs1': 0.1, 'pcurain': 0.1}


@pytest.fixture
def land_model():
    R.parameter_managers.add('FCLAND')(_Land)
    _Land.applied = []
    yield 'FCLAND'
    R.parameter_managers.remove('FCLAND')


def test_coupled_gw_routes_translated_correction_to_land_model(tmp_path, land_model, monkeypatch):
    from symfluence.models.modflow.calibration.parameter_manager import CoupledGWParameterManager

    # the plan for COUPLED_GW uses the land model's adapter (HYPE-style here)
    monkeypatch.setitem(fc.ADAPTERS, land_model, fc.ADAPTERS['HYPE'])
    cfg = {'LAND_SURFACE_MODEL': land_model, 'MODFLOW_PARAMS_TO_CALIBRATE': 'K',
           'FORCING_CORRECTION_MODE': 'calibrated'}
    pm = CoupledGWParameterManager(cfg, LOG, tmp_path)
    plan = fc.build_plan(cfg, 'COUPLED_GW')
    assert plan.adapter_model == land_model
    assert pm.enable_forcing_correction(plan) == ['pcurain']
    assert pm.all_param_names == ['rrcs1', 'K', 'FC_PRECIP_MULT']
    params = pm.denormalize_parameters(np.array([0.5, 0.5, 1.0]))
    assert params['pcurain'] == pytest.approx(0.6) and params['pcusnow'] == pytest.approx(0.6)
    assert 'FC_PRECIP_MULT' not in params
    pm._update_modflow_files = lambda p: True
    assert pm.update_model_files(params)
    assert set(_Land.applied[-1]) == {'rrcs1', 'pcurain', 'pcusnow'}


def test_modflow_forward_applier_is_a_noop(tmp_path):
    from symfluence.models.modflow.runner import MODFLOWRunner

    applier = fc.find_forward_applier(MODFLOWRunner)
    assert applier is not None
    assert applier({}, LOG, tmp_path, {'native': {}}) == {'applied_by': 'land model (recharge)'}


def test_modflow_plan_is_passthrough():
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.2}
    plan = fc.build_plan(cfg, 'MODFLOW')
    assert plan.fixed == {} and plan.to_native({}) == {}


def test_coupled_gw_land_model_resolution():
    assert fc._coupled_land_model({'HYDROLOGICAL_MODEL': 'HYPE,MODFLOW'}) == 'HYPE'
    assert fc._coupled_land_model({'LAND_SURFACE_MODEL': 'mesh', 'HYDROLOGICAL_MODEL': 'HYPE'}) == 'MESH'
    assert fc._coupled_land_model({'HYDROLOGICAL_MODEL': 'MODFLOW'}) == 'SUMMA'
