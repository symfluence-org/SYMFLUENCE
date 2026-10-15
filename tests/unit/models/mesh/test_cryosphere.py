"""Physical objective contracts: endpoints, support, periods and failure modes."""
from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from symfluence.models.mesh.calibration.worker import MESHWorker
from symfluence.models.mesh.cryosphere import calculate_constraints


@pytest.fixture
def case(tmp_path):
    root = tmp_path/'model'
    obs = tmp_path/'observations'
    root.mkdir()
    obs.mkdir()
    hours = pd.date_range('2012-09-30', '2013-09-30 23:00', freq='h')
    days = pd.date_range('2012-09-30', '2013-09-30')
    units = [{'surface': 'land', 'area_m2': 3e6}, {'surface': 'ice', 'area_m2': 1e6}]
    (root/'spatial_units.json').write_text(json.dumps({'units': units}))
    (root/'MESH_input_run_options.ini').write_text('TIMESTEPFLAG 60\n2012 274 0 0\n2013 274 0 0\n')
    (root/'MESH_glacier_storage.ini').write_text('1\n1021 10000\n')
    # Bulk loses 2 m over the water year; retained snow gains 1 m, net -1 m.
    end = np.full(len(hours), 10000.)
    end[-1] = 8000.
    pd.DataFrame({'step': np.arange(1, len(hours)+1), 'tile': 2, 'mid': 1021,
                  'start_kgm2': np.r_[10000., end[:-1]], 'end_kgm2': end,
                  'residual_kgm2': 0.}).to_csv(root/'MESH_glacier_ledger.csv', index=False)
    for name in ['SNO_H', 'LQWSSNO_H']:
        values = np.zeros((len(hours), 3))
        if name == 'SNO_H':
            values[-1, 1] = 1000.
        pd.DataFrame(values, index=hours).to_csv(root/f'{name}.csv', header=False)
    pd.DataFrame(np.tile([.4, 1., 0.], (len(days), 1)), index=days).to_csv(root/'FSNO_D.csv', header=False)
    pd.DataFrame({'YYYY': days.year, 'MM': days.month, 'DD': days.day,
                  'fsca_outside_glaciers': 40.}).to_csv(obs/'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';', index=False)
    pd.DataFrame({'annual_net_MB': [-1.], 'g_area_dyn': [1.]}, index=[2013]).to_csv(obs/'glacier_timeseries.csv', sep=';')
    return root, obs


def test_endpoint_snow_and_land_only_support_are_included(case):
    root, obs = case
    result = calculate_constraints(root, obs, '2012-10-01, 2013-09-30')
    assert result['snow_score'] == pytest.approx(1.)
    assert result['glacier_score'] == pytest.approx(1.)
    # Favorable reported area must not conceal a bad fixed-area comparison.
    pd.DataFrame({'annual_net_MB': [-.5], 'g_area_dyn': [2.]}, index=[2013]).to_csv(obs/'glacier_timeseries.csv', sep=';')
    result = calculate_constraints(root, obs, '2012-10-01, 2013-09-30')
    assert result['glacier_mb_rmse'] == pytest.approx(.5)
    assert result['glacier_reported_area_rmse'] == pytest.approx(0.)


@pytest.mark.parametrize('failure', ['truncated', 'missing_snow', 'missing_glacier', 'wrong_step', 'partial_year'])
def test_missing_constraint_evidence_rejects_trial(case, failure):
    root, obs = case
    period = '2012-10-01, 2013-09-30'
    if failure == 'truncated':
        p = root/'SNO_H.csv'
        p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
    elif failure == 'missing_snow':
        p = obs/'modis_fractional_snow_cover_and_glacier_albedo.csv'
        f = pd.read_csv(p, sep=';')
        f[f.MM < 8].to_csv(p, sep=';', index=False)
    elif failure == 'missing_glacier':
        pd.DataFrame({'annual_net_MB': [np.nan], 'g_area_dyn': [1.]}, index=[2013]).to_csv(obs/'glacier_timeseries.csv', sep=';')
    elif failure == 'wrong_step':
        p = root/'MESH_input_run_options.ini'
        p.write_text(p.read_text().replace('TIMESTEPFLAG 60', 'TIMESTEPFLAG 30'))
    else:
        period = '2013-01-01, 2013-09-30'
    with pytest.raises(ValueError):
        calculate_constraints(root, obs, period)


def test_composite_cannot_silently_become_discharge_only():
    worker = MESHWorker(logger=Mock())
    config = {'model': {'mesh': {'cryosphere_constraints': True}},
              'optimization': {'metric': 'COMPOSITE', 'composite_metric': {'kge': .6, 'snow_score': .2, 'glacier_score': .2}}}
    assert worker._extract_primary_score({'kge': .9}, config) == worker.penalty_score
    result = worker._extract_primary_score({'kge': .9, 'snow_score': -.5, 'glacier_score': .5}, config)
    assert result == pytest.approx(.54)
    config['optimization']['composite_metric'] = {'kge': 1.}
    assert worker._extract_primary_score({'kge': .9, 'snow_score': 1., 'glacier_score': 1.}, config) == worker.penalty_score


def test_holdout_observations_cannot_change_calibration_score(case):
    root, obs = case
    expected = calculate_constraints(root, obs, '2012-10-01, 2013-09-30')
    path = obs/'modis_fractional_snow_cover_and_glacier_albedo.csv'
    data = pd.read_csv(path, sep=';')
    future = data.copy()
    future.YYYY += 2
    future.fsca_outside_glaciers = 100.
    pd.concat([data, future]).to_csv(path, sep=';', index=False)
    path = obs/'glacier_timeseries.csv'
    data = pd.read_csv(path, sep=';', index_col=0)
    data.loc[2015] = [-99., 1.]
    data.to_csv(path, sep=';')
    assert calculate_constraints(root, obs, '2012-10-01, 2013-09-30') == expected
