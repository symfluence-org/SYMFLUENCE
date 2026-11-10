"""
Unit Tests for MultiGaugeMetrics.

Tests multi-gauge calibration metrics:
- Initialization and gauge mapping loading
- Observed streamflow loading and caching
- KGE calculation
- Quality filters
- Aggregation methods
- Configuration helper
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Mark all tests in this module
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def gauge_mapping_csv(tmp_path):
    """Create a gauge-segment mapping CSV file."""
    df = pd.DataFrame({
        'id': [101, 102, 103, 104, 105],
        'name': ['Gauge_A', 'Gauge_B', 'Gauge_C', 'Gauge_D', 'Gauge_E'],
        'nearest_segment': [1001, 1002, 1003, 1004, 1005],
        'distance_to_segment': [0.001, 0.01, 0.05, 0.1, 0.5],
        'area_calc': [100.0, 200.0, 500.0, 1000.0, 2000.0],
    })
    path = tmp_path / "gauge_segment_mapping.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def obs_data_dir(tmp_path):
    """Create observation data directory with sample streamflow files."""
    obs_dir = tmp_path / "observations"
    obs_dir.mkdir()

    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

    for gauge_id in [101, 102, 103, 104, 105]:
        np.random.seed(gauge_id)  # Reproducible
        flow = 5.0 + 3.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + \
               np.random.normal(0, 0.5, len(dates))
        flow = np.maximum(flow, 0.1)

        df = pd.DataFrame({
            'YYYY': dates.year,
            'MM': dates.month,
            'DD': dates.day,
            'qobs': flow,
        })
        df.to_csv(obs_dir / f"ID_{gauge_id}.csv", sep=';', index=False)

    return obs_dir


@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_multi_gauge')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def metrics_instance(gauge_mapping_csv, obs_data_dir, test_logger):
    """Create a MultiGaugeMetrics instance."""
    from symfluence.models.fuse.calibration.multi_gauge_metrics import MultiGaugeMetrics

    # Reset class-level cache before each test
    MultiGaugeMetrics._filter_cache = None

    return MultiGaugeMetrics(
        gauge_segment_mapping_path=gauge_mapping_csv,
        obs_data_dir=obs_data_dir,
        logger=test_logger,
    )


# =============================================================================
# Initialization Tests
# =============================================================================

class TestMultiGaugeMetricsInit:
    """Test initialization."""

    def test_loads_gauge_mapping(self, metrics_instance):
        """Should load gauge mapping from CSV."""
        assert len(metrics_instance.gauge_mapping) == 5

    def test_gauge_mapping_has_required_columns(self, metrics_instance):
        """Mapping should have id and nearest_segment columns."""
        assert 'id' in metrics_instance.gauge_mapping.columns
        assert 'nearest_segment' in metrics_instance.gauge_mapping.columns

    def test_raises_on_missing_mapping(self, tmp_path, test_logger):
        """Should raise FileNotFoundError if mapping file missing."""
        from symfluence.models.fuse.calibration.multi_gauge_metrics import MultiGaugeMetrics

        with pytest.raises(FileNotFoundError):
            MultiGaugeMetrics(
                gauge_segment_mapping_path=tmp_path / "nonexistent.csv",
                obs_data_dir=tmp_path,
                logger=test_logger,
            )

    def test_empty_obs_cache_on_init(self, metrics_instance):
        """Observation cache should be empty initially."""
        assert len(metrics_instance._obs_cache) == 0


# =============================================================================
# Observed Streamflow Loading Tests
# =============================================================================

class TestLoadObservedStreamflow:
    """Tests for _load_observed_streamflow method."""

    def test_loads_valid_gauge(self, metrics_instance):
        """Should load observation data for a valid gauge."""
        obs = metrics_instance._load_observed_streamflow(101)

        assert obs is not None
        assert len(obs) > 0
        assert isinstance(obs.index, pd.DatetimeIndex)

    def test_caches_loaded_data(self, metrics_instance):
        """Should cache loaded data for reuse."""
        metrics_instance._load_observed_streamflow(101)

        assert 101 in metrics_instance._obs_cache

    def test_uses_cache_on_second_call(self, metrics_instance):
        """Second call should use cache."""
        obs1 = metrics_instance._load_observed_streamflow(101)
        obs2 = metrics_instance._load_observed_streamflow(101)

        assert obs1 is obs2  # Same object from cache

    def test_returns_none_for_missing_gauge(self, metrics_instance):
        """Should return None for non-existent gauge file."""
        obs = metrics_instance._load_observed_streamflow(9999)

        assert obs is None

    def test_date_filtering(self, metrics_instance):
        """Should filter by date range when specified."""
        obs = metrics_instance._load_observed_streamflow(
            101, start_date='2020-06-01', end_date='2020-06-30'
        )

        assert obs is not None
        assert obs.index.min() >= pd.Timestamp('2020-06-01')
        assert obs.index.max() <= pd.Timestamp('2020-06-30')


# =============================================================================
# KGE Calculation Tests
# =============================================================================

class TestCalculateKGE:
    """Tests for _calculate_kge method."""

    def test_perfect_kge(self, metrics_instance):
        """Identical series should give KGE = 1.0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        sim = obs.copy()

        kge = metrics_instance._calculate_kge(obs, sim)

        assert kge == pytest.approx(1.0)

    def test_kge_with_bias(self, metrics_instance):
        """Constant bias should reduce KGE."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        sim = obs * 2.0  # Double the magnitude

        kge = metrics_instance._calculate_kge(obs, sim)

        assert kge < 1.0

    def test_kge_empty_arrays(self, metrics_instance):
        """Should return sentinel value for empty arrays."""
        kge = metrics_instance._calculate_kge(np.array([]), np.array([]))

        assert kge == -9999.0

    def test_kge_insufficient_data(self, metrics_instance):
        """Should return sentinel for fewer than 10 data points."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])

        kge = metrics_instance._calculate_kge(obs, sim)

        assert kge == -9999.0

    def test_kge_handles_nan(self, metrics_instance):
        """Should handle NaN values in arrays."""
        # Need enough non-NaN overlapping points (>=10 after NaN removal)
        obs = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan, 11.0, 12.0, 13.0, 14.0])
        sim = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

        kge = metrics_instance._calculate_kge(obs, sim)

        # After removing NaN: 11 valid points remain, should be valid
        assert kge > -9998

    def test_kge_range(self, metrics_instance):
        """KGE should be reasonable for typical hydrological data."""
        np.random.seed(42)
        obs = np.random.uniform(1.0, 10.0, 100)
        sim = obs + np.random.normal(0, 0.5, 100)

        kge = metrics_instance._calculate_kge(obs, sim)

        assert -1.0 < kge <= 1.0

    def test_kge_matches_core_metrics(self, metrics_instance):
        """Per-gauge KGE uses the shared symfluence.core.metrics definition."""
        from symfluence.core import metrics as core_metrics
        rng = np.random.default_rng(0)
        obs = rng.gamma(2.0, 3.0, 200)
        sim = 0.8 * obs + rng.normal(0, 1, 200)
        assert metrics_instance._calculate_kge(obs, sim) == pytest.approx(core_metrics.kge(obs, sim))

    def test_kge_constant_obs_is_invalid(self, metrics_instance):
        """Constant observations make KGE undefined, not a finite -0.42."""
        obs = np.full(50, 2.0)
        sim = np.linspace(1, 3, 50)
        assert metrics_instance._calculate_kge(obs, sim) == -9999.0

    def test_kge_negative_obs_mean_keeps_bias_term(self, metrics_instance):
        """A negative observed mean must not force beta to 0."""
        rng = np.random.default_rng(1)
        obs = rng.normal(-5.0, 1.0, 100)
        kge = metrics_instance._calculate_kge(obs, obs.copy())
        assert kge == pytest.approx(1.0)


# =============================================================================
# Quality Filter Tests
# =============================================================================

class TestApplyQualityFilters:
    """Tests for _apply_quality_filters method."""

    def test_no_filters_returns_all(self, metrics_instance):
        """Should return all gauges when no filters configured."""
        gauge_ids = [101, 102, 103]
        filter_config = {}

        result = metrics_instance._apply_quality_filters(
            gauge_ids, '2020-01-01', '2020-12-31', filter_config
        )

        assert result == gauge_ids

    def test_distance_filter_excludes_far_gauges(self, metrics_instance):
        """Distance filter should exclude gauges too far from segment."""
        gauge_ids = [101, 102, 103, 104, 105]
        filter_config = {'max_distance': 0.05}

        result = metrics_instance._apply_quality_filters(
            gauge_ids, '2020-01-01', '2020-12-31', filter_config
        )

        # Gauge 104 (0.1) and 105 (0.5) exceed max_distance
        assert 101 in result
        assert 102 in result
        assert 104 not in result
        assert 105 not in result

    def test_filter_caching(self, metrics_instance):
        """Repeated calls with same gauge set should use cache."""
        from symfluence.models.fuse.calibration.multi_gauge_metrics import MultiGaugeMetrics

        gauge_ids = [101, 102, 103]
        filter_config = {'max_distance': 0.5}

        result1 = metrics_instance._apply_quality_filters(
            gauge_ids, '2020-01-01', '2020-12-31', filter_config
        )

        # Cache should now be set
        assert MultiGaugeMetrics._filter_cache is not None

        result2 = metrics_instance._apply_quality_filters(
            gauge_ids, '2020-01-01', '2020-12-31', filter_config
        )

        assert result1 == result2


# =============================================================================
# Get Available Gauges Tests
# =============================================================================

class TestGetAvailableGauges:
    """Tests for get_available_gauges method."""

    def test_returns_gauges_with_sufficient_data(self, metrics_instance):
        """Should return gauges meeting minimum data point threshold."""
        available = metrics_instance.get_available_gauges(
            start_date='2020-01-01',
            end_date='2020-12-31',
            min_data_points=100,
        )

        # All 5 gauges have 366 days of data in 2020
        assert len(available) == 5

    def test_high_threshold_filters_gauges(self, metrics_instance):
        """Very high threshold should filter some gauges."""
        available = metrics_instance.get_available_gauges(
            start_date='2020-06-01',
            end_date='2020-06-30',
            min_data_points=100,  # Only 30 days available
        )

        assert len(available) == 0


# =============================================================================
# Configuration Helper Tests
# =============================================================================

class TestCreateMultiGaugeConfig:
    """Tests for create_multi_gauge_config function."""

    def test_returns_dict(self):
        """Should return a configuration dictionary."""
        from symfluence.models.fuse.calibration.multi_gauge_metrics import (
            create_multi_gauge_config,
        )

        config = create_multi_gauge_config(
            gauge_segment_mapping_path="/path/to/mapping.csv",
            obs_data_dir="/path/to/obs",
        )

        assert isinstance(config, dict)
        assert config['MULTI_GAUGE_CALIBRATION'] is True
        assert config['GAUGE_SEGMENT_MAPPING'] == "/path/to/mapping.csv"

    def test_custom_aggregation(self):
        """Should accept custom aggregation method."""
        from symfluence.models.fuse.calibration.multi_gauge_metrics import (
            create_multi_gauge_config,
        )

        config = create_multi_gauge_config(
            gauge_segment_mapping_path="/path/to/mapping.csv",
            obs_data_dir="/path/to/obs",
            aggregation='median',
            min_gauges=3,
        )

        assert config['MULTI_GAUGE_AGGREGATION'] == 'median'
        assert config['MULTI_GAUGE_MIN_GAUGES'] == 3

    def test_gauge_ids_optional(self):
        """gauge_ids should default to None (use all)."""
        from symfluence.models.fuse.calibration.multi_gauge_metrics import (
            create_multi_gauge_config,
        )

        config = create_multi_gauge_config(
            gauge_segment_mapping_path="/path/to/mapping.csv",
            obs_data_dir="/path/to/obs",
        )

        assert config['MULTI_GAUGE_IDS'] is None
