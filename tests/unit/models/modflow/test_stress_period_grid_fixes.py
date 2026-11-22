"""MODFLOW: daily expansion of stress-period flows, basin main zone, lumped grid area and layers."""
from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.modflow.basin_model import (
    WEIGHTS_FILE,
    MF6BasinGrid,
    MF6BasinParams,
    main_zone_per_cell,
    write_parameter_packages,
)
from symfluence.models.modflow.config_schema import MODFLOWConfig
from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler, periods_to_daily
from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

# ---------------------------------------------------------------------------
# 1. Stress-period drain flow spread over every day it covers
# ---------------------------------------------------------------------------


def _per_period(start, perlen, values):
    idx = pd.Timestamp(start) + pd.to_timedelta(np.arange(len(values)) * perlen, unit='D')
    return pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(idx), name='drain_m3d')


def test_periods_to_daily_identity_for_daily_periods():
    s = _per_period('2000-01-01', 1.0, [1.0, 2.0, 3.0])
    out = periods_to_daily(s, 1.0)
    np.testing.assert_array_equal(out.to_numpy(), s.to_numpy())
    assert list(out.index) == list(s.index)


def test_periods_to_daily_weekly_step_fill_and_partial_last_period():
    s = _per_period('2000-01-01', 7.0, [10.0, 20.0, 30.0])
    out = periods_to_daily(s, 7.0)
    assert len(out) == 21
    assert list(out.index) == list(pd.date_range('2000-01-01', periods=21, freq='D'))
    np.testing.assert_array_equal(out.to_numpy(), np.repeat([10.0, 20.0, 30.0], 7))


def test_periods_to_daily_fractional_periods_split_days():
    s = _per_period('2000-01-01', 2.5, [1.0, 3.0])
    out = periods_to_daily(s, 2.5)
    # days 0,1 -> 1; day 2 is half period 1, half period 2; days 3,4 -> 3
    np.testing.assert_allclose(out.to_numpy(), [1.0, 1.0, 2.0, 3.0, 3.0])
    # volume is conserved
    assert out.sum() == pytest.approx(1.0 * 2.5 + 3.0 * 2.5)


def test_periods_to_daily_keeps_time_of_day_start_aligned_to_dates():
    # EXPERIMENT_TIME_START with an hour: periods start at 01:00, days still map 1:1
    s = _per_period('2000-01-01 01:00', 1.0, [1.0, 2.0])
    out = periods_to_daily(s, 1.0)
    assert list(out.index) == list(pd.date_range('2000-01-01', periods=2, freq='D'))
    np.testing.assert_array_equal(out.to_numpy(), [1.0, 2.0])


def test_periods_to_daily_infers_period_length():
    s = _per_period('2000-01-01', 7.0, [10.0, 20.0])
    assert len(periods_to_daily(s)) == 14


def _surface(n_days, start='2000-01-01'):
    return pd.Series(np.full(n_days, 1e-8), index=pd.date_range(start, periods=n_days, freq='D'))


@pytest.mark.parametrize('perlen', [1.0, 7.0])
def test_combine_flows_covers_every_land_model_day(perlen):
    n_days = 30   # with 7-day periods the 5th period is only partly inside the record
    nper = int(math.ceil(n_days / perlen))
    drain = _per_period('2000-01-01', perlen, 86400.0 * np.arange(1, nper + 1))   # m3/d -> 1, 2, ... m3/s
    total = SUMMAToMODFLOWCoupler({}).combine_flows(_surface(n_days), drain, 1e6,
                                                    stress_period_length=perlen)
    assert len(total) == n_days
    assert not total.isna().any()
    assert list(total.index) == list(pd.date_range('2000-01-01', periods=n_days, freq='D'))
    expected_base = (np.arange(n_days) // int(perlen)) + 1.0
    np.testing.assert_allclose(total.to_numpy(), expected_base + 1e-8 * 1e6)


def test_combine_flows_daily_unchanged_from_previous_alignment():
    drain = _per_period('2000-01-01', 1.0, [86400.0, 2 * 86400.0, 3 * 86400.0])
    surface = _surface(5)
    total = SUMMAToMODFLOWCoupler({}).combine_flows(surface, drain, 1e6)
    # Only days MODFLOW simulated, exactly surface + drain/86400.
    assert list(total.index) == list(pd.date_range('2000-01-01', periods=3, freq='D'))
    np.testing.assert_array_equal(total.to_numpy(), np.array([1.0, 2.0, 3.0]) + 1e-8 * 1e6)


# ---------------------------------------------------------------------------
# 2. Basin mode: a cell's main zone is the one covering most of it
# ---------------------------------------------------------------------------


def test_main_zone_ranks_by_overlap_not_multiplier():
    w = pd.DataFrame({'zone_id': [1, 2], 'row': [1, 1], 'col': [1, 1],
                      'mult': [0.6, 0.5],                  # small zone 1 rescaled above zone 2
                      'coverage': [1.0, 1.0], 'overlap_m2': [1e5, 9e5]})
    assert main_zone_per_cell(w).zone_id.tolist() == [2]
    # Old weight files without overlap_m2 still work (ranking by mult).
    assert main_zone_per_cell(w.drop(columns='overlap_m2')).zone_id.tolist() == [1]


def _two_cell_grid(lake_zones):
    return MF6BasinGrid(
        nrow=1, ncol=2, cell_size=1000.0, origin=(0.0, 1000.0), crs='EPSG:32627',
        active=np.array([[True, True]]), top=np.array([[100.0, 110.0]]),
        basin_frac=np.ones((1, 2)),
        river=pd.DataFrame({'row': [1], 'col': [1], 'length': [1000.0], 'strtop': [95.0], 'seg': [1]}),
        lake_zones=lake_zones)


def test_seepl_uses_zone_with_largest_overlap(tmp_path):
    # Cell (1,1): lake zone 1 holds 10 % but has the larger mult; zone 2 holds 90 %.
    # Cell (1,2): lake zone 1 holds all of it.
    pd.DataFrame({'zone_id': [1, 2, 1], 'row': [1, 1, 1], 'col': [1, 1, 2],
                  'mult': [0.6, 0.5, 1.0], 'coverage': [1.0, 1.0, 1.0],
                  'overlap_m2': [1e5, 9e5, 1e6]}).to_csv(tmp_path / WEIGHTS_FILE, index=False)
    write_parameter_packages(tmp_path, _two_cell_grid([1]), MF6BasinParams())

    def cells(name):
        text = (tmp_path / name).read_text().split('BEGIN PERIOD 1\n')[1]
        return {(int(t[1]), int(t[2])) for t in (line.split() for line in text.splitlines()) if t and t[0] == '1'}

    assert cells('gwf.seepl.drn') == {(1, 2)}
    assert cells('gwf.seep.drn') == {(1, 1)}


# ---------------------------------------------------------------------------
# 3 & 4. Lumped grid: area equals the catchment; one layer only
# ---------------------------------------------------------------------------


def _preprocessor(tmp_path, area_km2=100.0, **mf):
    cfg = SimpleNamespace(
        domain=SimpleNamespace(name='t', experiment_id='e', catchment_area=area_km2,
                               time_start='2000-01-01', time_end='2000-01-31'),
        model=SimpleNamespace(modflow=MODFLOWConfig(**mf)),
        system=SimpleNamespace(data_dir=str(tmp_path)),
    )
    return MODFLOWPreProcessor(cfg, logging.getLogger('test_modflow'))


def _dis(tmp_path):
    text = (tmp_path / 'domain_t' / 'settings' / 'MODFLOW' / 'gwf.dis').read_text().split()
    get = {k: text[text.index(k) + 1] for k in ('NLAY', 'NROW', 'NCOL')}
    delr = float(text[text.index('DELR') + 2])
    delc = float(text[text.index('DELC') + 2])
    return {k: int(v) for k, v in get.items()}, delr, delc


@pytest.mark.parametrize('nrow,ncol', [(1, 1), (2, 3), (4, 4)])
def test_lumped_grid_area_equals_catchment(tmp_path, nrow, ncol):
    _preprocessor(tmp_path, MODFLOW_NROW=nrow, MODFLOW_NCOL=ncol).run_preprocessing()
    dims, delr, delc = _dis(tmp_path)
    assert (dims['NROW'], dims['NCOL']) == (nrow, ncol)
    assert nrow * ncol * delr * delc == pytest.approx(100.0e6, rel=1e-9)


def test_lumped_grid_overrides_mismatched_cell_size(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger='test_modflow'):
        _preprocessor(tmp_path, MODFLOW_NROW=2, MODFLOW_NCOL=2, MODFLOW_CELL_SIZE=1000.0).run_preprocessing()
    _, delr, delc = _dis(tmp_path)
    assert 4 * delr * delc == pytest.approx(100.0e6, rel=1e-9)
    assert 'MODFLOW_CELL_SIZE' in caplog.text


def test_lumped_rejects_multiple_layers(tmp_path):
    with pytest.raises(ConfigurationError, match='MODFLOW_NLAY'):
        _preprocessor(tmp_path, MODFLOW_NLAY=2).run_preprocessing()


def test_lumped_default_single_layer(tmp_path):
    _preprocessor(tmp_path).run_preprocessing()
    dims, _, _ = _dis(tmp_path)
    assert dims['NLAY'] == 1
