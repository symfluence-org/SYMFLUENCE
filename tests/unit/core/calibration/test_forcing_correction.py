# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION section: plan, adapters, prior, water balance, hooks."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.exceptions import ConfigurationError

LOG = logging.getLogger('test_forcing_correction')


class _Manager(BaseParameterManager):
    """Minimal GR-like manager: X1, X2 plus its own PCORR."""

    def __init__(self, names: List[str], config: Dict[str, Any] | None = None):
        super().__init__(config or {}, LOG, Path('.'))
        self._names = names

    def _get_parameter_names(self):
        return list(self._names)

    def _load_parameter_bounds(self):
        return {'X1': {'min': 100.0, 'max': 1500.0}, 'X2': {'min': -5.0, 'max': 5.0},
                'PCORR': {'min': 0.5, 'max': 2.0}, 'TCORR': {'min': -3.0, 'max': 3.0},
                'pcurain': {'min': 0.0, 'max': 0.3}, 'pcusnow': {'min': 0.0, 'max': 0.6}}

    def update_model_files(self, params):
        return True

    def get_initial_parameters(self):
        return {'X1': 350.0, 'X2': 0.0, 'PCORR': 1.2}


# ---------------------------------------------------------------------------
# Back-compat: section absent
# ---------------------------------------------------------------------------

def test_absent_section_gives_no_plan():
    assert fc.build_plan({}, 'GR') is None
    assert fc.build_plan({'FORCING_CORRECTION_MODE': None}, 'GR') is None


def test_absent_section_leaves_model_parameters_untouched():
    pm = _Manager(['X1', 'X2', 'PCORR'])
    assert pm.forcing_correction is None
    assert pm.all_param_names == ['X1', 'X2', 'PCORR']
    params = pm.denormalize_parameters(np.array([0.0, 0.5, 1.0]))
    assert params == {'X1': 100.0, 'X2': 0.0, 'PCORR': 2.0}


def test_typed_config_section(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig
    cfg = SymfluenceConfig.from_minimal(
        't', 'GR', POUR_POINT_COORDS='64/-19', EXPERIMENT_TIME_START='2000-01-01 00:00',
        EXPERIMENT_TIME_END='2001-01-01 00:00')
    assert fc.build_plan(cfg, 'GR') is None
    cfg = SymfluenceConfig.from_minimal(
        't', 'GR', POUR_POINT_COORDS='64/-19', EXPERIMENT_TIME_START='2000-01-01 00:00',
        EXPERIMENT_TIME_END='2001-01-01 00:00', FORCING_CORRECTION_MODE='calibrated',
        FORCING_CORRECTION_QUANTITIES='precip,temp', FORCING_CORRECTION_PRIOR_WEIGHT=0.1)
    plan = fc.build_plan(cfg, 'GR')
    assert plan.calibrated == ['precip', 'temp']
    assert plan.bounds['precip'] == (0.7, 1.6)
    assert plan.prior_weight == 0.1


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_off_mode_drops_model_corrections_and_injects_nothing():
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'off'}, 'GR')
    pm = _Manager(['X1', 'PCORR', 'TCORR'])
    assert fc.conflicting_settings({}, pm.all_param_names, plan) == ['PCORR', 'TCORR']
    dropped = pm.enable_forcing_correction(plan)
    assert dropped == ['PCORR', 'TCORR']
    assert pm.all_param_names == ['X1']
    assert pm.denormalize_parameters(np.array([0.0])) == {'X1': 100.0}


def test_off_mode_all_parameters_dropped_stays_empty():
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'off'}, 'GR')
    pm = _Manager(['PCORR'])
    pm.enable_forcing_correction(plan)
    assert pm.all_param_names == []  # not reloaded from the model list


def test_fixed_mode_injects_native_values():
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.15,
           'FORCING_CORRECTION_TEMP_OFFSET': -0.5}
    plan = fc.build_plan(cfg, 'GR')
    assert plan.fixed == {'precip': 1.15, 'temp': -0.5}
    pm = _Manager(['X1', 'PCORR'])
    pm.enable_forcing_correction(plan)
    assert pm.all_param_names == ['X1']
    params = pm.denormalize_parameters(np.array([0.0]))
    assert params == {'X1': 100.0, 'PCORR': 1.15, 'TCORR': -0.5}
    assert plan.report()['applied'] == {'FC_PRECIP_MULT': 1.15, 'FC_TEMP_OFFSET': -0.5}


def test_fixed_mode_without_value_is_identity():
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'fixed'}, 'MESH')
    assert plan.fixed == {'precip': 1.0}
    assert plan.to_native({}) == {'PRECIP_MULT': 1.0}


def test_same_fixed_value_for_every_supported_model():
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.2}
    natives = {m: fc.build_plan(cfg, m).to_native({}) for m in fc.ADAPTERS}
    assert natives['GR'] == {'PCORR': 1.2}
    assert natives['SUMMA'] == {'precip_mult': 1.2}
    assert natives['MESH'] == {'PRECIP_MULT': 1.2}
    assert natives['WFLOW'] == {'PCORR': 1.2}
    assert natives['MHM'] == {'precipCorrectionFactor': 1.2}
    assert natives['HYPE'] == pytest.approx({'pcurain': 0.2, 'pcusnow': 0.2})


def test_hype_undercatch_with_snow_multiplier_round_trips():
    cfg = {'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_QUANTITIES': ['precip', 'snow']}
    plan = fc.build_plan(cfg, 'HYPE')
    native = plan.to_native({'FC_PRECIP_MULT': 1.1, 'FC_SNOW_MULT': 1.2})
    assert native == pytest.approx({'pcurain': 0.1, 'pcusnow': 1.1 * 1.2 - 1})
    assert plan.values_from_native(native) == pytest.approx({'precip': 1.1, 'snow': 1.2})
    assert set(plan.superseded_native_names()) == {'pcurain', 'pcusnow'}


def test_snow_parameter_superseded_only_when_snow_is_set():
    off = fc.build_plan({'FORCING_CORRECTION_MODE': 'off'}, 'SUMMA')
    assert off.superseded_native_names() == ['precip_mult']
    snow = fc.build_plan({'FORCING_CORRECTION_MODE': 'fixed',
                          'FORCING_CORRECTION_SNOW_MULTIPLIER': 1.3}, 'SUMMA')
    assert set(snow.superseded_native_names()) == {'precip_mult', 'frozenPrecipMultip'}
    assert snow.to_native({}) == {'precip_mult': 1.0, 'frozenPrecipMultip': 1.3}


def test_unsupported_quantity_raises():
    with pytest.raises(ConfigurationError, match='no mechanism'):
        fc.build_plan({'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_TEMP_OFFSET': 1.0}, 'WFLOW')
    with pytest.raises(ConfigurationError, match='no mechanism'):
        fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated'}, 'NOAHMP')
    # identity values and 'off' are fine anywhere
    assert fc.build_plan({'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_TEMP_OFFSET': 0.0},
                         'WFLOW').fixed == {'precip': 1.0}
    assert fc.build_plan({'FORCING_CORRECTION_MODE': 'off'}, 'NOAHMP').superseded_native_names() == []


def test_invalid_mode_and_bounds_raise():
    with pytest.raises(ConfigurationError):
        fc.build_plan({'FORCING_CORRECTION_MODE': 'sometimes'}, 'GR')
    with pytest.raises(ConfigurationError):
        fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated',
                       'FORCING_CORRECTION_PRECIP_BOUNDS': [0.0, 1.5]}, 'GR')


def test_calibrated_mode_common_name_and_bounds():
    cfg = {'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_QUANTITIES': 'precip,temp',
           'FORCING_CORRECTION_PRECIP_BOUNDS': [0.8, 1.5]}
    plan = fc.build_plan(cfg, 'GR')
    pm = _Manager(['X1', 'X2', 'PCORR'])
    pm.enable_forcing_correction(plan)
    assert pm.all_param_names == ['X1', 'X2', 'FC_PRECIP_MULT', 'FC_TEMP_OFFSET']
    bounds = pm.get_parameter_bounds()
    assert bounds['FC_PRECIP_MULT'] == {'min': 0.8, 'max': 1.5}
    assert bounds['FC_TEMP_OFFSET'] == {'min': -3.0, 'max': 3.0}
    assert 'PCORR' not in bounds
    params = pm.denormalize_parameters(np.array([0.0, 0.5, 1.0, 0.5]))
    assert params == pytest.approx({'X1': 100.0, 'X2': 0.0, 'PCORR': 1.5, 'TCORR': 0.0})
    # The model's initial PCORR seeds the common parameter
    x0 = pm.normalize_parameters(pm.get_initial_parameters())
    assert x0[2] == pytest.approx((1.2 - 0.8) / 0.7)
    assert x0[3] == pytest.approx(0.5)
    assert plan.common_values_from_native(params) == pytest.approx(
        {'FC_PRECIP_MULT': 1.5, 'FC_TEMP_OFFSET': 0.0})


def test_calibrated_and_fixed_quantities_combine():
    cfg = {'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_TEMP_OFFSET': 0.7}
    plan = fc.build_plan(cfg, 'GR')
    assert plan.calibrated == ['precip'] and plan.fixed == {'temp': 0.7}
    assert plan.to_native({'FC_PRECIP_MULT': 1.3}) == {'PCORR': 1.3, 'TCORR': 0.7}


# ---------------------------------------------------------------------------
# Prior penalty
# ---------------------------------------------------------------------------

def test_prior_penalty_lognormal_and_normal():
    cfg = {'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_QUANTITIES': ['precip', 'temp'],
           'FORCING_CORRECTION_PRIOR_WEIGHT': 0.5, 'FORCING_CORRECTION_PRIOR_PRECIP_MEDIAN': 1.1,
           'FORCING_CORRECTION_PRIOR_PRECIP_SIGMA': 0.1, 'FORCING_CORRECTION_PRIOR_TEMP_SIGMA': 2.0}
    plan = fc.build_plan(cfg, 'GR')
    assert plan.prior_penalty({'precip': 1.1, 'temp': 0.0}) == 0.0
    expected = 0.5 * 0.5 * ((math.log(1.3 / 1.1) / 0.1) ** 2 + (1.0 / 2.0) ** 2)
    assert plan.prior_penalty({'precip': 1.3, 'temp': 1.0}) == pytest.approx(expected)
    # symmetric in log space
    assert plan.prior_penalty({'precip': 1.1 * 1.2}) == pytest.approx(plan.prior_penalty({'precip': 1.1 / 1.2}))


def test_prior_off_by_default_and_not_in_fixed_mode():
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated'}, 'GR')
    assert plan.prior_penalty({'precip': 1.5}) == 0.0
    fixed = fc.build_plan({'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRIOR_WEIGHT': 1.0,
                           'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.5}, 'GR')
    assert fixed.prior_penalty({'precip': 1.5}) == 0.0


def test_apply_prior_to_score_from_task_dict():
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_PRIOR_WEIGHT': 1.0,
                          'FORCING_CORRECTION_PRIOR_PRECIP_SIGMA': 0.2}, 'MESH')
    data = plan.to_dict()
    assert fc.ForcingCorrectionPlan.from_dict(data) == plan
    score, penalty = fc.apply_prior_to_score(0.8, data, {'PRECIP_MULT': math.exp(0.2)})
    assert penalty == pytest.approx(0.5)
    assert score == pytest.approx(0.3)
    assert fc.apply_prior_to_score(0.8, None, {'PRECIP_MULT': 2.0}) == (0.8, 0.0)


# ---------------------------------------------------------------------------
# Water-balance estimator
# ---------------------------------------------------------------------------

def test_water_balance_with_given_et():
    est = fc.estimate_precip_multiplier(precip_mm_day=2.0, q_mm_day=1.5, et_mm_day=1.0)
    assert est['multiplier'] == pytest.approx(1.25)
    assert est['method'] == 'config_et'


def test_water_balance_budyko_closes_balance():
    est = fc.estimate_precip_multiplier(precip_mm_day=2.0, q_mm_day=1.8, pet_mm_day=1.0)
    m = est['multiplier']
    p = m * 2.0
    assert p - fc.budyko_et(p, 1.0) == pytest.approx(1.8, abs=1e-8)
    assert est['et_mm_day'] == pytest.approx(fc.budyko_et(p, 1.0))
    assert m > 1.0  # runoff ratio 0.9 with that PET needs more precipitation


def test_water_balance_rejects_bad_inputs():
    with pytest.raises(ValueError):
        fc.estimate_precip_multiplier(0.0, 1.0, et_mm_day=0.5)
    with pytest.raises(ValueError):
        fc.estimate_precip_multiplier(1.0, 1.0)


def _synthetic_project(tmp_path: Path, domain: str, p_mm_day: float, q_mm_day: float, area_km2: float):
    import xarray as xr
    times = pd.date_range('2001-01-01', '2004-12-31 23:00', freq='6h')
    forcing = tmp_path / 'data' / 'forcing' / 'basin_averaged_data'
    forcing.mkdir(parents=True)
    doy = times.dayofyear.values
    ds = xr.Dataset(
        {'pptrate': (('time', 'hru'), np.full((len(times), 2), p_mm_day / 86400.0)),
         'airtemp': (('time', 'hru'), (278.15 + 8 * np.sin(2 * np.pi * (doy - 100) / 365))[:, None]
                     * np.ones((1, 2))),
         'latitude': (('hru',), np.array([64.0, 64.2]))},
        coords={'time': times})
    ds.to_netcdf(forcing / f'{domain}_forcing.nc')
    obs = tmp_path / 'data' / 'observations' / 'streamflow' / 'preprocessed'
    obs.mkdir(parents=True)
    days = pd.date_range('2001-01-01', '2004-12-31', freq='D')
    q_cms = q_mm_day * area_km2 * 1e3 / 86400.0
    pd.DataFrame({'datetime': days, 'discharge_cms': q_cms}).to_csv(
        obs / f'{domain}_streamflow_processed.csv', index=False)


def test_water_balance_from_project_files(tmp_path):
    _synthetic_project(tmp_path, 'wb', p_mm_day=2.0, q_mm_day=1.5, area_km2=250.0)
    cfg = {'FIXED_CATCHMENT_AREA': 250.0e6, 'CALIBRATION_PERIOD': '2002-01-01, 2003-12-31',
           'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 'water_balance',
           'FORCING_CORRECTION_ET_MM_PER_YEAR': 365.25}
    inputs = fc.water_balance_inputs(cfg, tmp_path, 'wb')
    assert inputs['precip_mm_day'] == pytest.approx(2.0)
    assert inputs['q_mm_day'] == pytest.approx(1.5)
    assert inputs['latitude'] == pytest.approx(64.1)
    assert inputs['period'] == ['2002-01-01', '2003-12-31']
    assert inputs['pet_mm_day'] > 0
    plan = fc.build_plan(cfg, 'GR', project_dir=tmp_path, domain_name='wb')
    assert plan.fixed['precip'] == pytest.approx((1.5 + 1.0) / 2.0)
    assert plan.report()['water_balance_estimate']['method'] == 'config_et'
    # Without ET: Budyko on Oudin PET
    cfg.pop('FORCING_CORRECTION_ET_MM_PER_YEAR')
    budyko = fc.build_plan(cfg, 'GR', project_dir=tmp_path, domain_name='wb')
    assert budyko.estimate['method'] == 'budyko' and budyko.fixed['precip'] > 0.75


def test_water_balance_without_data_raises(tmp_path):
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 'water_balance'}
    with pytest.raises(ConfigurationError):
        fc.build_plan(cfg, 'GR', project_dir=tmp_path, domain_name='none')


# ---------------------------------------------------------------------------
# Worker / task builder / optimizer hooks
# ---------------------------------------------------------------------------

class _Worker(BaseWorker):
    def __init__(self):
        super().__init__(config={'OPTIMIZATION_METRIC': 'KGE'})
        self.seen: Dict[str, Any] = {}

    def apply_parameters(self, params, settings_dir, **kwargs):
        self.seen['apply'] = (dict(params), dict(kwargs))
        return True

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        self.seen['run'] = dict(kwargs)
        return True

    def calculate_metrics(self, output_dir, config, **kwargs):
        return {'KGE': 0.8}


def _task(extra):
    return WorkerTask(individual_id=0, params={'PCORR': math.exp(0.2)}, proc_id=0,
                      config={'OPTIMIZATION_METRIC': 'KGE'}, settings_dir=Path('.'),
                      output_dir=Path('.'), additional_data=extra)


def test_worker_without_plan_unchanged():
    w = _Worker()
    result = w.evaluate(_task({'foo': 1}))
    assert result.score == pytest.approx(0.8)
    assert fc.PENALTY_METRIC not in result.metrics
    assert w.seen['apply'][1]['foo'] == 1


def test_worker_subtracts_prior_and_hides_plan_from_model():
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_PRIOR_WEIGHT': 1.0,
                          'FORCING_CORRECTION_PRIOR_PRECIP_SIGMA': 0.2}, 'GR')
    w = _Worker()
    result = w.evaluate(_task({fc.TASK_KEY: plan.to_dict(), 'foo': 1}))
    assert result.score == pytest.approx(0.3)
    assert result.metrics['KGE'] == pytest.approx(0.8)
    assert result.metrics[fc.PENALTY_METRIC] == pytest.approx(0.5)
    assert fc.TASK_KEY not in w.seen['apply'][1] and fc.TASK_KEY not in w.seen['run']


def test_task_builder_carries_plan_only_with_prior(tmp_path):
    from symfluence.core.calibration.optimizers.evaluators.task_builder import TaskBuilder

    def builder(plan):
        pm = SimpleNamespace(forcing_correction=plan, original_depths=None)
        return TaskBuilder({}, tmp_path, 'd', tmp_path, tmp_path, {}, 1, 'KGE', pm, LOG)

    params = {'PCORR': 1.0}
    assert fc.TASK_KEY not in builder(None).build_task(0, params, 0, 'e')
    no_prior = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated'}, 'GR')
    assert fc.TASK_KEY not in builder(no_prior).build_task(0, params, 0, 'e')
    prior = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated',
                           'FORCING_CORRECTION_PRIOR_WEIGHT': 0.1}, 'GR')
    assert builder(prior).build_task(0, params, 0, 'e')[fc.TASK_KEY] == prior.to_dict()


def _optimizer_stub(tmp_path, config, names):
    from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
    stub = SimpleNamespace(
        config=config, logger=LOG, project_dir=tmp_path, domain_name='d', results_dir=tmp_path,
        param_manager=_Manager(names), _get_model_name=lambda: 'GR')
    stub._setup_forcing_correction = BaseModelOptimizer._setup_forcing_correction.__get__(stub)
    stub._write_forcing_correction_report = \
        BaseModelOptimizer._write_forcing_correction_report.__get__(stub)
    return stub


def test_optimizer_setup_warns_reports_and_back_compat(tmp_path, caplog):
    import json
    legacy = _optimizer_stub(tmp_path, {}, ['X1', 'PCORR'])
    legacy._setup_forcing_correction()
    assert legacy.param_manager.all_param_names == ['X1', 'PCORR']
    assert not (tmp_path / fc.REPORT_NAME).exists()

    stub = _optimizer_stub(tmp_path, {'FORCING_CORRECTION_MODE': 'calibrated'}, ['X1', 'PCORR'])
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        stub._setup_forcing_correction()
    assert 'supersedes per-model forcing corrections' in caplog.text and 'PCORR' in caplog.text
    assert stub.param_manager.all_param_names == ['X1', 'FC_PRECIP_MULT']
    report = json.loads((tmp_path / fc.REPORT_NAME).read_text())
    assert report['mode'] == 'calibrated' and report['calibrated'] == ['FC_PRECIP_MULT']
    stub._write_forcing_correction_report({'X1': 300.0, 'PCORR': 1.25})
    report = json.loads((tmp_path / fc.REPORT_NAME).read_text())
    assert report['applied'] == {'FC_PRECIP_MULT': 1.25}
    assert report['applied_native'] == {'PCORR': 1.25}


def test_supported_models_table_matches_adapters():
    rows = {r['model']: r for r in fc.supported_models_table()}
    assert set(rows) == {'GR', 'SUMMA', 'MESH', 'WFLOW', 'MHM', 'HYPE'}
    assert rows['GR']['temp'] == 'TCORR'
