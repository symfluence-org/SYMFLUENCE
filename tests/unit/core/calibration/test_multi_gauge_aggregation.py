"""Multi-gauge metrics: filter cache key, NaN-safe weights, one open per evaluation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.calibration.multi_gauge.metrics import MultiGaugeMetrics

IDS = [101, 102, 103]
SEGMENTS = [1001, 1002, 1003]


@pytest.fixture
def setup(tmp_path):
    MultiGaugeMetrics._filter_cache = None
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    obs_dir = tmp_path / 'obs'
    obs_dir.mkdir()
    sim = np.zeros((len(dates), len(IDS)))
    for i, gauge_id in enumerate(IDS):
        rng = np.random.default_rng(gauge_id)
        flow = 5.0 + 3.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + rng.normal(0, 0.5, len(dates))
        flow = np.maximum(flow, 0.1)
        pd.DataFrame({'YYYY': dates.year, 'MM': dates.month, 'DD': dates.day, 'qobs': flow}).to_csv(
            obs_dir / f'ID_{gauge_id}.csv', sep=';', index=False)
        # gauge 103 is simulated badly, so its weight matters
        sim[:, i] = flow if gauge_id != 103 else flow[::-1] * 3.0
    sim_path = tmp_path / 'mizu.nc'
    xr.Dataset(
        {'IRFroutedRunoff': (('time', 'seg'), sim), 'reachID': ('seg', SEGMENTS)},
        coords={'time': dates},
    ).to_netcdf(sim_path)

    def make(areas, distances=(0.001, 0.01, 0.4)):
        mapping = tmp_path / f'mapping_{len(list(tmp_path.glob("mapping_*")))}.csv'
        pd.DataFrame({
            'id': IDS, 'name': ['A', 'B', 'C'], 'nearest_segment': SEGMENTS,
            'distance_to_segment': list(distances), 'area_calc': list(areas),
        }).to_csv(mapping, index=False)
        return MultiGaugeMetrics(mapping, obs_dir, logging.getLogger('test_mg'))

    return make, sim_path


def test_one_nan_area_does_not_make_the_weighted_aggregate_nan(setup):
    make, sim_path = setup
    result = make([100.0, np.nan, 300.0]).calculate_multi_gauge_metrics(
        sim_path, aggregation='weighted', min_gauges=1)
    assert np.isfinite(result['kge'])
    assert result['n_valid_gauges'] == 3
    assert result['n_weighted_gauges'] == 2
    per = result['per_gauge']
    expected = (100.0 * per[101]['kge'] + 300.0 * per[103]['kge']) / 400.0
    assert result['kge'] == pytest.approx(expected)


def test_weighted_aggregate_with_no_usable_weight_is_a_failure(setup):
    make, sim_path = setup
    result = make([np.nan, np.nan, 0.0]).calculate_multi_gauge_metrics(
        sim_path, aggregation='weighted', min_gauges=1)
    assert result['kge'] == -9999.0


def test_filter_cache_is_not_shared_across_different_filter_configs(setup):
    make, _ = setup
    metrics = make([100.0, 200.0, 300.0])
    strict = metrics._apply_quality_filters(IDS, '2020-01-01', '2020-12-31', {'max_distance': 0.005})
    assert strict == [101]
    loose = metrics._apply_quality_filters(IDS, '2020-01-01', '2020-12-31', {'max_distance': 0.05})
    assert loose == [101, 102]
    # same question again -> cached answer
    assert metrics._apply_quality_filters(IDS, '2020-01-01', '2020-12-31', {'max_distance': 0.05}) == [101, 102]


def test_filter_cache_is_not_shared_across_mappings_or_periods(setup):
    make, _ = setup
    near = make([100.0, 200.0, 300.0], distances=(0.001, 0.001, 0.001))
    far = make([100.0, 200.0, 300.0], distances=(0.9, 0.9, 0.001))
    config = {'max_distance': 0.05}
    assert near._apply_quality_filters(IDS, None, None, config) == IDS
    assert far._apply_quality_filters(IDS, None, None, config) == [103]
    key_a = MultiGaugeMetrics._filter_cache[0]
    far._apply_quality_filters(IDS, '2020-06-01', None, config)
    assert MultiGaugeMetrics._filter_cache[0] != key_a


def test_simulation_file_is_opened_once_per_evaluation(setup, monkeypatch):
    make, sim_path = setup
    opened = []
    real_open = xr.open_dataset

    def counting_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(xr, 'open_dataset', counting_open)
    metrics = make([100.0, 200.0, 300.0])
    result = metrics.calculate_multi_gauge_metrics(sim_path, min_gauges=1)
    assert result['n_valid_gauges'] == 3
    assert opened.count(str(sim_path)) == 1
    assert metrics._sim_handle is None  # closed after the evaluation

    # a standalone extraction still works and does not leak the handle
    series = metrics._extract_simulated_at_segment(sim_path, 1001)
    assert series is not None and len(series) == 366
    assert metrics._sim_handle is None
