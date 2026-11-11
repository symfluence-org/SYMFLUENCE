"""Built-in t-route Muskingum-Cunge: time step from data, HRUs matched by id."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ConfigurationError, ModelExecutionError
from symfluence.models.troute.config_schema import TRouteConfig
from symfluence.models.troute.runner import TRouteRunner
from symfluence.models.troute.topology_generator import TRouteTopologyGenerator


def _runner(troute_cfg=None):
    runner = TRouteRunner.__new__(TRouteRunner)
    runner.logger = logging.getLogger('test_troute')
    runner.config = SimpleNamespace(model=SimpleNamespace(troute=troute_cfg or TRouteConfig()))

    def get_value(accessor, default=None, **kwargs):
        try:
            value = accessor()
        except AttributeError:
            value = None
        return default if value is None else value

    runner._get_config_value = get_value
    return runner


def _topology(path, n, length, hru_ids=None, hru_to_seg=None, areas=None):
    seg = np.arange(1, n + 1)
    data = {
        'comid': ('link', seg), 'to_node': ('link', np.r_[seg[1:], 0]),
        'length': ('link', np.full(n, float(length))), 'slope': ('link', np.full(n, 0.001)),
        'n': ('link', np.full(n, 0.035)), 'channel_width': ('link', np.full(n, 20.0)),
        'link_id_hru': ('nhru', seg if hru_to_seg is None else np.asarray(hru_to_seg)),
        'hru_area_m2': ('nhru', np.full(n, 1e7) if areas is None else np.asarray(areas)),
    }
    if hru_ids is not None:
        data['hruId'] = ('nhru', np.asarray(hru_ids))
    xr.Dataset(data).to_netcdf(path)


def _route(runner, tmp_path, runoff, topo):
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    runner._run_builtin_muskingum_cunge(runoff, topo, out)
    with xr.open_dataset(out / 'troute_output.nc') as ds:
        return ds.flow.values.copy(), dict(ds.attrs)


def _daily_pulse(tmp_path, n=10):
    times = pd.date_range('2000-01-01', periods=60, freq='D')
    q = np.full((60, n), 1e-9)
    q[:, 0] = 5e-7
    q[10, 0] = 5e-5
    xr.Dataset({'q_lateral': (('time', 'gru'), q), 'gruId': ('gru', np.arange(1, n + 1))},
               coords={'time': times}).to_netcdf(tmp_path / 'r.nc')
    return tmp_path / 'r.nc'


def test_dt_comes_from_the_runoff_time_axis_not_the_3600_default(tmp_path):
    """Daily runoff through 50 km of channel arrives within a day, not 24x later."""
    _topology(tmp_path / 't.nc', 10, 5000.0, hru_ids=np.arange(1, 11))
    flow, attrs = _route(_runner(), tmp_path, _daily_pulse(tmp_path), tmp_path / 't.nc')
    assert attrs['dt_seconds'] == 86400.0
    assert int(flow[:, -1].argmax()) in (10, 11)


def test_explicit_dt_that_disagrees_with_the_data_is_an_error(tmp_path):
    _topology(tmp_path / 't.nc', 10, 5000.0, hru_ids=np.arange(1, 11))
    runoff = _daily_pulse(tmp_path)
    with pytest.raises(ConfigurationError, match='SETTINGS_TROUTE_DT_SECONDS'):
        _route(_runner(TRouteConfig(SETTINGS_TROUTE_DT_SECONDS=3600)), tmp_path, runoff, tmp_path / 't.nc')
    # ...and one that agrees is accepted.
    _, attrs = _route(_runner(TRouteConfig(SETTINGS_TROUTE_DT_SECONDS=86400)), tmp_path, runoff, tmp_path / 't.nc')
    assert attrs['dt_seconds'] == 86400.0


def test_irregular_time_axis_is_refused(tmp_path):
    _topology(tmp_path / 't.nc', 2, 1000.0, hru_ids=[1, 2])
    times = pd.to_datetime(['2000-01-01', '2000-01-02', '2000-01-05', '2000-01-06'])
    xr.Dataset({'q_lateral': (('time', 'gru'), np.zeros((4, 2))), 'gruId': ('gru', [1, 2])},
               coords={'time': times}).to_netcdf(tmp_path / 'r.nc')
    with pytest.raises(ModelExecutionError, match='not regular'):
        _route(_runner(), tmp_path, tmp_path / 'r.nc', tmp_path / 't.nc')


def _two_hru_runoff(tmp_path, ids):
    times = pd.date_range('2000-01-01', periods=48, freq='h')
    q = np.zeros((48, 2))
    q[:, 0] = 1e-7  # column 0 carries the runoff
    xr.Dataset({'q_lateral': (('time', 'gru'), q), 'gruId': ('gru', ids)},
               coords={'time': times}).to_netcdf(tmp_path / 'r2.nc')
    return tmp_path / 'r2.nc'


def test_runoff_columns_are_matched_to_topology_hrus_by_id(tmp_path):
    """Column 0 is gruId 2 (the 900 km2 HRU): outlet must see ~90 m3/s, not 0.1."""
    _topology(tmp_path / 't2.nc', 2, 1000.0, hru_ids=[1, 2], hru_to_seg=[1, 2], areas=[1e6, 9e8])
    flow, _ = _route(_runner(), tmp_path, _two_hru_runoff(tmp_path, [2, 1]), tmp_path / 't2.nc')
    assert flow[-1, -1] == pytest.approx(90.0, rel=0.05)


def test_unmatched_runoff_ids_raise(tmp_path):
    _topology(tmp_path / 't2.nc', 2, 1000.0, hru_ids=[1, 2], areas=[1e6, 9e8])
    with pytest.raises(ModelExecutionError, match='do not match'):
        _route(_runner(), tmp_path, _two_hru_runoff(tmp_path, [2, 7]), tmp_path / 't2.nc')


def test_topology_without_hru_ids_is_refused_rather_than_matched_by_position(tmp_path):
    _topology(tmp_path / 't2.nc', 2, 1000.0, areas=[1e6, 9e8])
    with pytest.raises(ModelExecutionError, match="no 'hruId'"):
        _route(_runner(), tmp_path, _two_hru_runoff(tmp_path, [2, 1]), tmp_path / 't2.nc')


def test_lumped_runoff_feeds_every_topology_hru_by_its_own_area(tmp_path):
    """One runoff column, HRUs listed in a different order than the segments."""
    _topology(tmp_path / 't3.nc', 2, 1000.0, hru_ids=[20, 10], hru_to_seg=[2, 1], areas=[9e8, 1e6])
    times = pd.date_range('2000-01-01', periods=48, freq='h')
    xr.Dataset({'q_lateral': (('time', 'gru'), np.full((48, 1), 1e-7)), 'gruId': ('gru', [1])},
               coords={'time': times}).to_netcdf(tmp_path / 'r3.nc')
    flow, _ = _route(_runner(), tmp_path, tmp_path / 'r3.nc', tmp_path / 't3.nc')
    # headwater segment 1 only receives the 1 km2 HRU
    assert flow[-1, 0] == pytest.approx(0.1, rel=0.05)
    assert flow[-1, 1] == pytest.approx(90.1, rel=0.05)


def test_topology_file_carries_hru_ids(tmp_path):
    from symfluence.core.modeling.utilities.base_topology_generator import TopologyData

    pp = SimpleNamespace(
        logger=logging.getLogger('test_troute'), config=SimpleNamespace(),
        _get_config_value=lambda accessor, default=None, **kw: default,
    )
    data = TopologyData(
        seg_ids=np.array([1, 2]), down_seg_ids=np.array([2, 0]), slopes=np.array([.01, .01]),
        lengths=np.array([100., 100.]), hru_ids=np.array([65, 7]),
        hru_to_seg_ids=np.array([2, 1]), hru_areas=np.array([1e6, 2e6]),
    )
    TRouteTopologyGenerator(pp).write_topology_file(data, tmp_path / 'topo.nc')
    with xr.open_dataset(tmp_path / 'topo.nc') as ds:
        np.testing.assert_array_equal(ds.hruId.values, [65, 7])
        np.testing.assert_array_equal(ds.link_id_hru.values, [2, 1])
