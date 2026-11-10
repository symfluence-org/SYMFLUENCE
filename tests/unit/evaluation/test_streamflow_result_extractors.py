"""StreamflowEvaluator: result-extractor path, CF units and results-CSV fallback (#342)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.core.modeling.evaluators.streamflow_units import classify_streamflow_units

DATES = pd.date_range('2010-01-01', periods=5)
LOGGER = logging.getLogger('test-streamflow-extractors')


def _evaluator(tmp_path, model, **extra):
    config = dict(HYDROLOGICAL_MODEL=model, EXPERIMENT_ID='run_1', DOMAIN_NAME='test',
                  CALIBRATION_PERIOD='2010-01-01, 2010-01-05', CALIBRATION_TIMESTEP='native',
                  SYMFLUENCE_DATA_DIR=str(tmp_path), **extra)
    return StreamflowEvaluator(config, tmp_path, LOGGER)


def _sim_dir(tmp_path, model):
    d = tmp_path / 'simulations' / 'run_1' / model
    d.mkdir(parents=True)
    return d


def _extract(evaluator, sim_dir):
    files = evaluator.get_simulation_files(sim_dir)
    assert files, f"no files found in {sim_dir}"
    return evaluator.extract_simulated_data(files)


@pytest.mark.parametrize('units', ['m3 s-1', 'm3/s', 'm^3 s^-1', 'm3.s-1', 'm³/s', 'cms', 'm**3/s'])
def test_volumetric_spellings(units):
    parsed = classify_streamflow_units(units)
    assert parsed.is_volumetric and parsed.factor == pytest.approx(1.0)


@pytest.mark.parametrize('units,factor', [
    ('m s-1', 1.0), ('m/s', 1.0), ('mm/s', 1e-3), ('mm/h', 1e-3 / 3600),
    ('mm/day', 1e-3 / 86400), ('mm d-1', 1e-3 / 86400), ('kg m-2 s-1', 1e-3),
])
def test_depth_rate_spellings(units, factor):
    parsed = classify_streamflow_units(units)
    assert parsed.is_depth_rate and parsed.factor == pytest.approx(factor)


@pytest.mark.parametrize('units', ['', None, 'mm timestep-1', 'furlongs/fortnight', 'K'])
def test_unknown_units(units):
    assert classify_streamflow_units(units).kind == 'unknown'


def test_lumped_lstm_predicted_streamflow(tmp_path):
    """The SUMMA-centric path raised 'No suitable streamflow variable' here."""
    sim_dir = _sim_dir(tmp_path, 'LSTM')
    q = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    ds = xr.Dataset({'predicted_streamflow': ('time', q)}, coords={'time': DATES})
    ds.predicted_streamflow.attrs['units'] = 'm3 s-1'
    ds.to_netcdf(sim_dir / 'run_1_LSTM_output.nc')
    ds.to_netcdf(sim_dir / 'run_1_timestep.nc')  # the copy LSTM also writes
    result = _extract(_evaluator(tmp_path, 'LSTM', FIXED_CATCHMENT_AREA=1e9), sim_dir)
    np.testing.assert_allclose(result.values, q)


def test_distributed_lstm_volumetric_runoff_is_summed_not_area_scaled(tmp_path):
    """Per-GRU averageRoutedRunoff in m3 s-1: 10+10 m³/s, not 2e7."""
    sim_dir = _sim_dir(tmp_path, 'LSTM')
    ds = xr.Dataset({'averageRoutedRunoff': (('time', 'gru'), np.full((5, 2), 10.0))},
                    coords={'time': DATES, 'gru': [0, 1]})
    ds.averageRoutedRunoff.attrs['units'] = 'm3 s-1'
    ds.to_netcdf(sim_dir / 'run_1_LSTM_output.nc')
    ds.to_netcdf(sim_dir / 'run_1_timestep.nc')
    result = _extract(_evaluator(tmp_path, 'LSTM', FIXED_CATCHMENT_AREA=1e6), sim_dir)
    np.testing.assert_allclose(result.values, 20.0)


def test_cf_streamflow_variable_m3_s_minus_1(tmp_path):
    sim_dir = _sim_dir(tmp_path, 'XMODEL')
    ds = xr.Dataset({'streamflow': ('time', np.arange(1.0, 6.0))}, coords={'time': DATES})
    ds.streamflow.attrs['units'] = 'm3 s-1'
    ds.to_netcdf(sim_dir / 'run_1_timestep.nc')
    result = _extract(_evaluator(tmp_path, 'XMODEL', FIXED_CATCHMENT_AREA=1e9), sim_dir)
    np.testing.assert_allclose(result.values, np.arange(1.0, 6.0))


def test_summa_depth_rate_still_multiplied_by_area(tmp_path):
    sim_dir = _sim_dir(tmp_path, 'SUMMA')
    ds = xr.Dataset({'averageRoutedRunoff': (('time', 'gru'), np.full((5, 1), 1e-8))},
                    coords={'time': DATES})
    ds.averageRoutedRunoff.attrs['units'] = 'm s-1'
    ds.to_netcdf(sim_dir / 'run_1_timestep.nc')
    result = _extract(_evaluator(tmp_path, 'SUMMA', FIXED_CATCHMENT_AREA=1e9), sim_dir)
    np.testing.assert_allclose(result.values, 10.0)  # 1e-8 m/s × 1e9 m²


def test_summa_style_mm_per_day_converted_to_m_per_s_first(tmp_path):
    sim_dir = _sim_dir(tmp_path, 'XMODEL')
    ds = xr.Dataset({'averageRoutedRunoff': ('time', np.full(5, 1.0))}, coords={'time': DATES})
    ds.averageRoutedRunoff.attrs['units'] = 'mm/day'
    ds.to_netcdf(sim_dir / 'run_1_timestep.nc')
    result = _extract(_evaluator(tmp_path, 'XMODEL', FIXED_CATCHMENT_AREA=8.64e7), sim_dir)
    np.testing.assert_allclose(result.values, 1.0)  # 1 mm/day over 86.4 km²


def _write_results_csv(tmp_path, column, values):
    (tmp_path / 'results').mkdir(exist_ok=True)
    pd.DataFrame({column: values}, index=pd.Index(DATES, name='datetime')).to_csv(
        tmp_path / 'results' / 'run_1_results.csv')


def test_results_csv_fallback_when_output_unreadable(tmp_path):
    """Distributed GNN per-GRU output has no evaluator branch; the CSV does."""
    sim_dir = _sim_dir(tmp_path, 'GNN')
    ds = xr.Dataset({'predicted_streamflow': (('time', 'gru'), np.ones((5, 3)))},
                    coords={'time': DATES, 'gru': [0, 1, 2]})
    ds.predicted_streamflow.attrs['units'] = 'm3 s-1'
    ds.to_netcdf(sim_dir / 'run_1_GNN_output.nc')
    _write_results_csv(tmp_path, 'GNN_discharge_cms', [4.0, 4.5, 5.0, 5.5, 6.0])
    result = _extract(_evaluator(tmp_path, 'GNN'), sim_dir)
    np.testing.assert_allclose(result.values, [4.0, 4.5, 5.0, 5.5, 6.0])


def test_results_csv_used_when_no_output_file(tmp_path):
    sim_dir = _sim_dir(tmp_path, 'LSTM')
    _write_results_csv(tmp_path, 'LSTM_discharge_cms', [1.0, 2.0, 3.0, 4.0, 5.0])
    result = _extract(_evaluator(tmp_path, 'LSTM'), sim_dir)
    np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_results_csv_never_used_for_calibration_trial_dirs(tmp_path):
    """A worker trial dir must not be scored against the experiment's own run."""
    trial_dir = tmp_path / 'optimization' / 'GNN' / 'process_0'
    trial_dir.mkdir(parents=True)
    ds = xr.Dataset({'predicted_streamflow': (('time', 'gru'), np.ones((5, 3)))},
                    coords={'time': DATES, 'gru': [0, 1, 2]})
    ds.to_netcdf(trial_dir / 'run_1_GNN_output.nc')
    _write_results_csv(tmp_path, 'GNN_discharge_cms', [4.0] * 5)
    evaluator = _evaluator(tmp_path, 'GNN')
    files = evaluator.get_simulation_files(trial_dir)
    with pytest.raises(ValueError):
        evaluator.extract_simulated_data(files)
