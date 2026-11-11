"""NGEN forcing: hourly resampling of coarser sources and id-based assignment."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.ngen.preprocessor import NgenPreProcessor

PRECIP_COL = 'atmosphere_water__liquid_equivalent_precipitation_rate'


def _pre(tmp_path=None):
    pre = object.__new__(NgenPreProcessor)
    pre.logger = logging.getLogger('test_ngen_forcing')
    pre._include_noah = True
    pre._include_snow17 = False
    pre._include_sacsma = False
    if tmp_path is not None:
        pre.forcing_dir = tmp_path
    return pre


def _forcing(freq, periods, hru_ids=(1,)):
    times = pd.date_range('2020-01-01', periods=periods, freq=freq)
    n = len(hru_ids)
    scale = np.arange(1, n + 1, dtype=float)[None, :]
    ramp = np.arange(periods, dtype=float)[:, None]
    return xr.Dataset(
        {
            'precipitation_flux': (('time', 'hru'), np.full((periods, n), 1e-4) * scale),
            'air_temperature': (('time', 'hru'), 280.0 + ramp * np.ones((1, n))),
            'wind_speed': (('time', 'hru'), np.full((periods, n), 3.0)),
            'hruId': (('hru',), np.asarray(hru_ids)),
        },
        coords={'time': times},
    )


@pytest.mark.parametrize('freq,periods', [('3h', 8), ('6h', 4), ('30min', 48), ('h', 24)])
def test_resample_to_hourly_conserves_precipitation_and_leaves_no_gaps(freq, periods):
    """One day at 1e-4 mm/s is 8.64 mm whatever the source step."""
    ds, step = _pre()._resample_forcing_to_hourly(_forcing(freq, periods))
    assert step == int(pd.Timedelta(freq if freq != 'h' else '1h').total_seconds())
    assert ds.sizes['time'] == 24
    assert not np.isnan(ds['precipitation_flux'].values).any()
    assert not np.isnan(ds['air_temperature'].values).any()
    np.testing.assert_allclose(ds['precipitation_flux'].values, 1e-4)
    np.testing.assert_allclose(float((ds['precipitation_flux'] * 3600).sum()), 8.64)


def test_coarse_state_variables_are_interpolated():
    ds, _ = _pre()._resample_forcing_to_hourly(_forcing('3h', 8))
    np.testing.assert_allclose(
        ds['air_temperature'].values[:4, 0], [280.0, 280.0 + 1 / 3, 280.0 + 2 / 3, 281.0]
    )


def test_forcing_is_assigned_to_catchments_by_id_not_position(tmp_path):
    pre = _pre(tmp_path)
    forcing = _forcing('h', 6, hru_ids=(30, 10, 20))   # precip scale 1, 2, 3
    cat_ids = ['cat-10', 'cat-20', 'cat-30']
    aligned = pre._align_forcing_to_catchments(forcing, cat_ids)
    assert aligned['hruId'].values.tolist() == [10, 20, 30]

    csv_dir = pre._write_csv_forcing_files(aligned, cat_ids)
    for cat_id, scale in zip(cat_ids, (2.0, 3.0, 1.0)):
        df = pd.read_csv(csv_dir / f'{cat_id}_forcing.csv')
        np.testing.assert_allclose(df[PRECIP_COL], 1e-4 * scale)

    nc = pre._create_ngen_forcing_dataset(aligned, cat_ids)
    np.testing.assert_allclose(nc['precip_rate'].values[:, 0], [2e-4, 3e-4, 1e-4], rtol=1e-6)


def test_float_and_string_ids_match():
    pre = _pre()
    forcing = _forcing('h', 3, hru_ids=(2.0, 1.0))
    aligned = pre._align_forcing_to_catchments(forcing, ['cat-1', 'cat-2.0'])
    assert aligned['hruId'].values.tolist() == [1.0, 2.0]


def test_unmatched_catchment_raises():
    forcing = _forcing('h', 3, hru_ids=(1, 2, 3))
    with pytest.raises(ModelExecutionError, match="cat-9"):
        _pre()._align_forcing_to_catchments(forcing, ['cat-1', 'cat-2', 'cat-9'])


def test_forcing_without_ids_and_different_count_raises():
    forcing = _forcing('h', 3, hru_ids=(1, 2, 3)).drop_vars('hruId')
    with pytest.raises(ModelExecutionError, match='no HRU ids'):
        _pre()._align_forcing_to_catchments(forcing, ['cat-1', 'cat-2'])


def test_single_hru_forcing_serves_every_catchment(tmp_path):
    pre = _pre(tmp_path)
    forcing = _forcing('h', 3, hru_ids=(1,))
    cat_ids = ['cat-5', 'cat-6']
    aligned = pre._align_forcing_to_catchments(forcing, cat_ids)
    csv_dir = pre._write_csv_forcing_files(aligned, cat_ids)
    for cat_id in cat_ids:
        np.testing.assert_allclose(pd.read_csv(csv_dir / f'{cat_id}_forcing.csv')[PRECIP_COL], 1e-4)
