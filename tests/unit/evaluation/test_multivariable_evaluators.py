"""Model-agnostic GLACIER_MB / BASIN_* evaluators and bias-sensitive objective defaults."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import symfluence  # noqa: F401
from symfluence.core.metrics.metric_transformer import MetricTransformer
from symfluence.core.modeling.evaluators.multivariable import (
    BasinETEvaluator,
    BasinSCAEvaluator,
    GlacierMassBalanceEvaluator,
    glacier_mb_metrics_for_run,
    load_mass_balance_observations,
    requested_glacier_mb_metrics,
)
from symfluence.optimization.objectives import ObjectiveRegistry

pytestmark = pytest.mark.unit
LOG = logging.getLogger('test')


def _summa_glacier_output(sim_dir: Path, rate_mm_per_yr: float, area_km2: float = 10.0):
    """Daily SUMMA output whose glacier loses ``rate`` mm w.e./yr over ``area``."""
    dates = pd.date_range('2000-01-01', '2010-12-31', freq='D')
    years = np.asarray((dates - dates[0]).days) / 365.25
    storage_m3 = 5e9 + rate_mm_per_yr / 1000.0 * area_km2 * 1e6 * years
    ds = xr.Dataset({
        'basin__GlacierStorage': (('time', 'gru'), (storage_m3 / 1e9)[:, None]),
        'basin__GlacierArea': (('time', 'gru'), np.full((len(dates), 1), area_km2 * 1e6)),
    }, coords={'time': dates})
    sim_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(sim_dir / 'run_day.nc')


def _config(tmp_path, obs_path, **extra):
    cfg = {
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'GLACIER_MB_OBS_PATH': str(obs_path),
        'CALIBRATION_PERIOD': '2000-01-01, 2010-12-31',
        'EVALUATION_PERIOD': '',
        'CALIBRATION_TIMESTEP': 'native',
    }
    cfg.update(extra)
    return cfg


# ----------------------------------------------------------------------
# observation formats
# ----------------------------------------------------------------------
def test_load_hugonnet_period_format(tmp_path):
    p = tmp_path / 'hugonnet.csv'
    pd.DataFrame({'rgiid': ['RGI60-06.00001'], 'period': ['2000-01-01_2010-01-01'],
                  'dmdtda': [-0.8], 'err_dmdtda': [0.1]}).to_csv(p, index=False)
    obs = load_mass_balance_observations(p)
    assert obs.loc[0, 'start'] == pd.Timestamp('2000-01-01')
    assert obs.loc[0, 'rate'] == pytest.approx(-800.0)
    assert obs.loc[0, 'sigma'] == pytest.approx(100.0)


def test_load_lamah_year_format(tmp_path):
    p = tmp_path / 'glacier_timeseries.csv'
    p.write_text('YYYY;annual_net_MB;g_area_dyn\n2005;-1.2;9.5\n2006;-0.4;9.4\n')
    obs = load_mass_balance_observations(p)
    assert list(obs['end']) == [pd.Timestamp(2005, 9, 30), pd.Timestamp(2006, 9, 30)]
    assert obs.loc[0, 'start'] == pd.Timestamp(2004, 9, 30)
    assert obs.loc[1, 'area_km2'] == pytest.approx(9.4)
    assert np.isnan(obs.loc[0, 'sigma'])


def test_load_start_end_mm_units(tmp_path):
    p = tmp_path / 'mb.csv'
    pd.DataFrame({'start_date': ['2001-10-01'], 'end_date': ['2002-05-01'],
                  'mb_rate': [1500.0]}).to_csv(p, index=False)
    obs = load_mass_balance_observations(p, units='mm_we_per_yr')
    assert obs.loc[0, 'rate'] == pytest.approx(1500.0)


# ----------------------------------------------------------------------
# GLACIER_MB evaluator
# ----------------------------------------------------------------------
def _hugonnet(tmp_path, rate_mwe, err=None):
    p = tmp_path / 'obs.csv'
    row = {'period': ['2000-01-01_2010-01-01'], 'dmdtda': [rate_mwe]}
    if err is not None:
        row['err_dmdtda'] = [err]
    pd.DataFrame(row).to_csv(p, index=False)
    return p


def test_glacier_mb_perfect_match(tmp_path):
    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -800.0)
    ev = GlacierMassBalanceEvaluator(_config(tmp_path, _hugonnet(tmp_path, -0.8, 0.1)), tmp_path, LOG)
    m = ev.calculate_metrics(sim)
    assert m['GMB_BIAS'] == pytest.approx(0.0, abs=1.0)
    assert m['GMB_SCORE'] == pytest.approx(1.0, abs=0.01)
    assert m['Calib_GMB_N'] == 1


def test_glacier_mb_bias_uses_uncertainty(tmp_path):
    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -500.0)
    ev = GlacierMassBalanceEvaluator(_config(tmp_path, _hugonnet(tmp_path, -0.8, 0.1)), tmp_path, LOG)
    m = ev.calculate_metrics(sim)
    assert m['GMB_BIAS'] == pytest.approx(300.0, rel=0.01)       # less negative than observed
    assert m['GMB_REL_BIAS'] == pytest.approx(300.0 / 800.0, rel=0.01)
    assert m['GMB_SCORE'] == pytest.approx(1.0 - 3.0, rel=0.02)    # 300 / 100 sigma
    assert m['GMB_WITHIN_UNCERTAINTY'] == 0.0


def test_glacier_mb_default_scale_without_uncertainty(tmp_path):
    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -500.0)
    ev = GlacierMassBalanceEvaluator(_config(tmp_path, _hugonnet(tmp_path, -0.8)), tmp_path, LOG)
    assert ev.calculate_metrics(sim)['GMB_SCORE'] == pytest.approx(1.0 - 0.3, abs=0.01)


def test_glacier_mb_observed_area_operator(tmp_path):
    """With an observed area, storage change is divided by it (model area 10, obs 20 km2)."""
    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -1000.0, area_km2=10.0)
    p = tmp_path / 'obs.csv'
    p.write_text('year;annual_net_MB;g_area_dyn\n2005;-0.5;20.0\n')
    ev = GlacierMassBalanceEvaluator(_config(tmp_path, p), tmp_path, LOG)
    m = ev.calculate_metrics(sim)
    assert m['GMB_SIM_MEAN'] == pytest.approx(-500.0, rel=0.01)
    assert m['GMB_BIAS'] == pytest.approx(0.0, abs=5.0)


def test_glacier_mb_periods_outside_calibration_are_ignored(tmp_path):
    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -800.0)
    cfg = _config(tmp_path, _hugonnet(tmp_path, -0.8), CALIBRATION_PERIOD='2003-01-01, 2006-12-31')
    assert GlacierMassBalanceEvaluator(cfg, tmp_path, LOG).calculate_metrics(sim) is None


def test_glacier_mb_unsupported_model_returns_none(tmp_path):
    sim = tmp_path / 'sim'
    sim.mkdir()
    cfg = _config(tmp_path, _hugonnet(tmp_path, -0.8), HYDROLOGICAL_MODEL='MHM')
    assert GlacierMassBalanceEvaluator(cfg, tmp_path, LOG).calculate_metrics(sim) is None


def test_glacier_mb_from_series(tmp_path):
    """MultivariateTarget passes the cumulative specific balance series."""
    idx = pd.date_range('2000-01-01', '2010-01-01', freq='D')
    series = pd.Series(-800.0 * (idx - idx[0]).days / 365.25, index=idx)
    ev = GlacierMassBalanceEvaluator(_config(tmp_path, _hugonnet(tmp_path, -0.8)), tmp_path, LOG)
    assert ev.calculate_metrics(series)['GMB_BIAS'] == pytest.approx(0.0, abs=1.0)


# ----------------------------------------------------------------------
# SCA / ET: bias-sensitive scores
# ----------------------------------------------------------------------
def _summa_snow_et(sim_dir: Path, sca: np.ndarray, et_mm: np.ndarray, dates):
    sim_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset({
        'scalarGroundSnowFraction': (('time', 'hru'), sca[:, None]),
        'scalarTotalET': (('time', 'hru'), (-et_mm / 86400.0)[:, None]),
    }, coords={'time': dates}).to_netcdf(sim_dir / 'run_day.nc')


@pytest.fixture
def biased_snow_et(tmp_path):
    dates = pd.date_range('2001-01-01', periods=365, freq='D')
    season = 0.5 + 0.4 * np.cos(2 * np.pi * np.arange(365) / 365)
    obs_sca = season * 0.6
    obs_et = 1.0 + np.sin(2 * np.pi * np.arange(365) / 365) * 0.5 + 0.6
    _summa_snow_et(tmp_path / 'sim', season, obs_et * 1.6, dates)      # perfect timing, biased
    pd.DataFrame({'date': dates, 'fsca': obs_sca * 100}).to_csv(tmp_path / 'sca.csv', index=False)
    pd.DataFrame({'date': dates, 'et': obs_et}).to_csv(tmp_path / 'et.csv', index=False)
    cfg = {'HYDROLOGICAL_MODEL': 'SUMMA', 'BASIN_SCA_OBS_PATH': str(tmp_path / 'sca.csv'),
           'BASIN_ET_OBS_PATH': str(tmp_path / 'et.csv'), 'CALIBRATION_PERIOD': '2001-01-01, 2001-12-31',
           'EVALUATION_PERIOD': '', 'CALIBRATION_TIMESTEP': 'native'}
    return cfg, tmp_path


def test_sca_and_et_scores_penalise_bias(biased_snow_et):
    cfg, root = biased_snow_et
    sca = BasinSCAEvaluator(cfg, root, LOG).calculate_metrics(root / 'sim')
    et = BasinETEvaluator(cfg, root, LOG).calculate_metrics(root / 'sim')
    for m in (sca, et):
        assert m['correlation'] == pytest.approx(1.0, abs=1e-6)   # correlation sees nothing wrong
        assert m['KGE'] < 0.5                                       # KGE does
        assert m['beta'] > 1.5
    assert sca['beta'] == pytest.approx(1 / 0.6, rel=1e-3)          # percent obs rescaled to 0-1


def test_multivariate_defaults_penalise_bias_for_sca_and_et():
    results = {'SCA': {'KGE': 0.3, 'correlation': 1.0}, 'ET': {'KGE': 0.4, 'correlation': 1.0}}
    obj = ObjectiveRegistry.get_objective('MULTIVARIATE', {'OBJECTIVE_WEIGHTS': {'SCA': 1, 'ET': 1}}, LOG)
    # Old default (correlation) would give 0.0; KGE gives 0.5*0.7 + 0.5*0.6
    assert obj.calculate(results) == pytest.approx(0.65)


def test_multivariate_glacier_mb_term_and_minimise_metrics():
    results = {'STREAMFLOW': {'KGE': 0.8}, 'GLACIER_MB': {'GMB_SCORE': 0.5, 'GMB_ABS_BIAS': 250.0}}
    obj = ObjectiveRegistry.get_objective(
        'MULTIVARIATE', {'OBJECTIVE_WEIGHTS': {'STREAMFLOW': 0.5, 'GLACIER_MB': 0.5}}, LOG)
    assert obj.calculate(results) == pytest.approx(0.5 * 0.2 + 0.5 * 0.5)
    obj = ObjectiveRegistry.get_objective(
        'MULTIVARIATE', {'OBJECTIVE_WEIGHTS': {'GLACIER_MB': 1.0},
                         'OBJECTIVE_METRICS': {'glacier_mb': 'GMB_ABS_BIAS'}}, LOG)
    assert obj.calculate(results) == pytest.approx(250.0)       # minimised: |value|, not 1 - value


def test_multivariate_none_result_is_penalised_not_crash():
    obj = ObjectiveRegistry.get_objective('MULTIVARIATE', {'OBJECTIVE_WEIGHTS': {'GLACIER_MB': 1.0}}, LOG)
    assert obj.calculate({'GLACIER_MB': None}) == pytest.approx(2.0)


# ----------------------------------------------------------------------
# NSGA-II pairing / worker hook
# ----------------------------------------------------------------------
def test_gmb_metric_directions_registered():
    assert MetricTransformer.get_direction('GMB_SCORE') == 'maximize'
    assert MetricTransformer.get_direction('GMB_ABS_BIAS') == 'minimize'
    assert MetricTransformer.transform_for_maximization('GMB_ABS_BIAS', 200.0) == pytest.approx(-200.0)


def test_requested_glacier_mb_metrics_from_nsga2_and_composite():
    cfg = {'OPTIMIZATION_METRIC': 'KGE', 'NSGA2_PRIMARY_METRIC': 'KGE',
           'NSGA2_SECONDARY_METRIC': 'gmb_score', 'COMPOSITE_METRIC': {'KGE': 0.7, 'GMB_ABS_BIAS': 0.3}}
    assert requested_glacier_mb_metrics(cfg) == ['GMB_SCORE', 'GMB_ABS_BIAS']
    assert requested_glacier_mb_metrics({'OPTIMIZATION_METRIC': 'KGE'}) == []


def test_worker_hook_scores_run_and_fails_closed(tmp_path):
    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -800.0)
    cfg = _config(tmp_path, _hugonnet(tmp_path, -0.8, 0.1), NSGA2_SECONDARY_METRIC='GMB_SCORE')
    m = glacier_mb_metrics_for_run(cfg, [tmp_path / 'missing', sim], tmp_path, LOG)
    assert m['GMB_SCORE'] == pytest.approx(1.0, abs=0.01)
    failed = glacier_mb_metrics_for_run(cfg, [tmp_path / 'missing'], tmp_path, LOG)
    assert np.isnan(failed['GMB_SCORE'])


def test_population_objective_extraction_pairs_kge_and_gmb():
    from symfluence.core.calibration.optimizers.evaluators.population_evaluator import PopulationEvaluator
    pe = PopulationEvaluator.__new__(PopulationEvaluator)
    pe.DEFAULT_PENALTY_SCORE = -999.0
    metrics = {'KGE': 0.7, 'GMB_SCORE': 0.4}
    assert pe._objective_from_metrics('KGE', metrics, {}) == pytest.approx(0.7)
    assert pe._objective_from_metrics('GMB_SCORE', metrics, {}) == pytest.approx(0.4)
    assert pe._objective_from_metrics('GMB_SCORE', {'GMB_SCORE': float('nan')}, {}) == -999.0


def test_base_worker_adds_gmb_metrics_for_composite_objective(tmp_path):
    """A worker with no glacier code of its own can be scored on KGE + glacier MB."""
    from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask

    sim = tmp_path / 'sim'
    _summa_glacier_output(sim, -500.0)
    cfg = _config(tmp_path, _hugonnet(tmp_path, -0.8, 0.1),
                  OPTIMIZATION_METRIC='COMPOSITE', COMPOSITE_METRIC={'KGE': 0.5, 'GMB_SCORE': 0.5})

    class _Worker(BaseWorker):
        def apply_parameters(self, params, settings_dir, **kwargs):
            return True

        def run_model(self, config, settings_dir, output_dir, **kwargs):
            return True

        def calculate_metrics(self, output_dir, config, **kwargs):
            return {'KGE': 0.8}

    worker = _Worker(cfg, LOG)
    task = WorkerTask(individual_id=0, params={}, proc_id=0, config=cfg,
                      settings_dir=tmp_path, output_dir=sim)
    result = worker._evaluate_once(task)
    assert result.metrics['GMB_SCORE'] == pytest.approx(-2.0, abs=0.05)
    assert result.score == pytest.approx(0.5 * 0.8 + 0.5 * -2.0, abs=0.05)
