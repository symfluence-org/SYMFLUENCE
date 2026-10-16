"""Native outlet CSV must win over snow/ice state NetCDF in final scoring."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.models.wflow.calibration.worker import WflowWorker


@pytest.mark.parametrize('parameter,expected', [
    ('river_water__volume_flow_rate', False),
    ('soil_surface_water__net_runoff_volume_flux', True),
])
def test_q_units_ignore_other_diagnostic_columns(tmp_path, parameter, expected):
    (tmp_path / 'wflow_sbm.toml').write_text(
        f'[[output.csv.column]]\nheader="Q"\nparameter="{parameter}"\n'
        '[[output.csv.column]]\nheader="runoff_diagnostic"\n'
        'parameter="soil_surface_water__net_runoff_volume_flux"\n')
    worker = WflowWorker({}, logging.getLogger('units'))
    assert worker._is_runoff_flux_output(tmp_path, {}) is expected


def test_native_final_windows_match_worker_and_ignore_cryosphere(tmp_path):
    dates = pd.date_range('2013-01-01', periods=120)
    observed = pd.Series(20 + 8 * np.sin(np.arange(120) * .23), index=dates)
    simulated = observed.copy()
    simulated.iloc[60:] *= 1.4
    pd.DataFrame({'time': dates + pd.Timedelta(days=1), 'Q': simulated.values,
                  'WB_max': 0.}).to_csv(tmp_path / 'output.csv', index=False)
    xr.Dataset({'snow': ('time', np.ones(120))}, coords={'time': dates}).to_netcdf(tmp_path / 'cryosphere.nc')
    observed.to_frame('discharge_cms').to_csv(tmp_path / 'obs.csv', index_label='datetime')
    config = dict(HYDROLOGICAL_MODEL='WFLOW', WFLOW_SPATIAL_MODE='distributed',
                  CALIBRATION_PERIOD='2013-01-01,2013-03-01',
                  EVALUATION_PERIOD='2013-03-02,2013-04-30', CALIBRATION_TIMESTEP='daily',
                  OBSERVATIONS_PATH=str(tmp_path / 'obs.csv'), DOMAIN_NAME='test',
                  SYMFLUENCE_DATA_DIR=str(tmp_path))
    logger = logging.getLogger('wflow-test')
    evaluator = StreamflowEvaluator(config, tmp_path, logger)
    paths = evaluator.get_simulation_files(tmp_path)
    assert paths == [tmp_path / 'output.csv']
    series = evaluator.extract_simulated_data(paths)
    pd.testing.assert_index_equal(series.index, dates, check_names=False)
    np.testing.assert_allclose(series.values, simulated.values, rtol=1e-14)
    metrics = evaluator.calculate_metrics(tmp_path, calibration_only=False)
    worker = WflowWorker(config, logger)
    for prefix, period in [('Calib', config['CALIBRATION_PERIOD']), ('Eval', config['EVALUATION_PERIOD'])]:
        worker_metrics = worker.calculate_metrics(tmp_path, dict(config, CALIBRATION_PERIOD=period))
        assert metrics[prefix + '_KGE'] == pytest.approx(worker_metrics['kge'], abs=1e-10)
    assert metrics['Calib_KGE'] == pytest.approx(1.)
    assert metrics['Eval_KGE'] < .5
    (tmp_path / 'output.csv').unlink()
    assert evaluator.get_simulation_files(tmp_path) == []
