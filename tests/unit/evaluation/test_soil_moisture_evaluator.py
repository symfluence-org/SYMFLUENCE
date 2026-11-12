"""
Tests for SoilMoistureEvaluator.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.mixins.config import SCHEMA_DEFAULT, fallback_default
from symfluence.evaluation.evaluators.soil_moisture import SoilMoistureEvaluator


def _make_config_value_getter(config_dict):
    """Create a mock _get_config_value that uses config_dict values."""
    call_count = [0]
    init_values = [
        '2010-01-01, 2015-12-31',
        '2016-01-01, 2018-12-31',
        'daily',
        config_dict.get('OPTIMIZATION_TARGET', 'sm_point'),
    ]

    def mock_get_config_value(typed_accessor, default=SCHEMA_DEFAULT, dict_key=None):
        if call_count[0] < len(init_values):
            result = init_values[call_count[0]]
            call_count[0] += 1
            return result
        if dict_key and dict_key in config_dict:
            return config_dict[dict_key]
        return fallback_default(default, dict_key)

    return mock_get_config_value


@pytest.fixture
def sm_evaluator(mock_config, tmp_path, mock_logger):
    """Create a SoilMoistureEvaluator for testing."""
    config_dict = {
        'OPTIMIZATION_TARGET': 'sm_point',
        'SM_TARGET_DEPTH': 'auto',
        'SM_DEPTH_TOLERANCE': 0.05,
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'CALIBRATION_TIMESTEP': 'daily',
    }
    mock_config.to_dict.side_effect = lambda flatten=True: config_dict

    with patch.object(SoilMoistureEvaluator, '_get_config_value',
                     side_effect=_make_config_value_getter(config_dict)):
        evaluator = SoilMoistureEvaluator(mock_config, tmp_path, mock_logger)
        evaluator.config_dict = config_dict
        return evaluator


class TestSoilMoistureEvaluatorInit:
    """Test SoilMoistureEvaluator initialization."""

    def test_init_sm_point(self, mock_config, tmp_path, mock_logger):
        """Test initialization with point soil moisture target."""
        config_dict = {'OPTIMIZATION_TARGET': 'sm_point'}
        with patch.object(SoilMoistureEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = SoilMoistureEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.optimization_target == 'sm_point'
            assert hasattr(evaluator, 'target_depth')

    def test_init_sm_smap(self, mock_config, tmp_path, mock_logger):
        """Test initialization with SMAP target."""
        config_dict = {
            'OPTIMIZATION_TARGET': 'sm_smap',
            'SMAP_LAYER': 'surface_sm',
        }
        with patch.object(SoilMoistureEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = SoilMoistureEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.optimization_target == 'sm_smap'


class TestPointSoilMoistureExtraction:
    """Test point soil moisture extraction."""

    def test_extracts_single_layer(self, sm_evaluator, tmp_path):
        """Test extraction with automatic layer selection."""
        ds = xr.Dataset({
            'mLayerVolFracLiq': (['time', 'hru', 'midToto'],
                                 np.random.uniform(0.1, 0.4, (100, 1, 5))),
            'mLayerDepth': (['hru', 'midToto'], [[0.1, 0.2, 0.3, 0.5, 1.0]]),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
            'midToto': range(5),
        })

        file_path = tmp_path / 'sm_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = sm_evaluator._extract_point_soil_moisture(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100
        # Volumetric fraction should be between 0 and 1
        assert 0 <= result.min() <= result.max() <= 1

    def test_raises_for_missing_variable(self, sm_evaluator, tmp_path):
        """Test raises error when mLayerVolFracLiq not found."""
        ds = xr.Dataset({
            'some_variable': (['time', 'hru'], np.random.rand(10, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'empty_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            with pytest.raises(ValueError, match="mLayerVolFracLiq variable not found"):
                sm_evaluator._extract_point_soil_moisture(ds_loaded)


class TestFindTargetLayer:
    """Test target layer finding."""

    def test_auto_selects_shallowest(self, sm_evaluator, tmp_path):
        """Test auto mode selects shallowest layer (index 0)."""
        sm_evaluator.target_depth = 'auto'

        layer_depths = xr.DataArray(
            [0.1, 0.2, 0.3, 0.5, 1.0],
            dims=['midToto'],
        )

        result = sm_evaluator._find_target_layer(layer_depths)

        assert result == 0

    def test_finds_closest_layer(self, sm_evaluator, tmp_path):
        """Test finding layer closest to target depth."""
        sm_evaluator.target_depth = 0.5

        # Layer depths: cumulative midpoints would be approx 0.05, 0.2, 0.45, 0.85, 1.6
        layer_depths = xr.DataArray(
            [0.1, 0.2, 0.3, 0.5, 1.0],
            dims=['midToto'],
        )

        result = sm_evaluator._find_target_layer(layer_depths)

        # Should find the layer closest to 0.5m depth
        assert result in [2, 3]  # Either of these could be closest


class TestSMAPExtraction:
    """Test SMAP soil moisture extraction."""

    def test_extracts_surface_sm(self, mock_config, tmp_path, mock_logger):
        """Test SMAP surface soil moisture extraction."""
        config_dict = {
            'OPTIMIZATION_TARGET': 'sm_smap',
            'SMAP_LAYER': 'surface_sm',
        }
        with patch.object(SoilMoistureEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = SoilMoistureEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator.smap_layer = 'surface_sm'

            ds = xr.Dataset({
                'mLayerVolFracLiq': (['time', 'hru', 'midToto'],
                                     np.random.uniform(0.1, 0.4, (100, 1, 5))),
                'mLayerDepth': (['hru', 'midToto'], [[0.1, 0.2, 0.3, 0.5, 1.0]]),
            },
            coords={
                'time': pd.date_range('2010-01-01', periods=100),
                'hru': [0],
                'midToto': range(5),
            })

            file_path = tmp_path / 'sm_output.nc'
            ds.to_netcdf(file_path)

            with xr.open_dataset(file_path) as ds_loaded:
                result = evaluator._extract_smap_soil_moisture(ds_loaded)

            assert isinstance(result, pd.Series)
            assert len(result) == 100


class TestDepthWeightedMean:
    """Test depth-weighted mean calculation."""

    def test_calculates_weighted_mean(self, sm_evaluator, tmp_path):
        """Test depth-weighted mean calculation."""
        # Create data with known values
        sm_data = xr.DataArray(
            [[[0.3, 0.25, 0.2, 0.15, 0.1]]],  # SM values decreasing with depth
            dims=['time', 'hru', 'midToto'],
            coords={
                'time': [pd.Timestamp('2010-01-01')],
                'hru': [0],
                'midToto': range(5),
            }
        )

        layer_depths = xr.DataArray(
            [[0.1, 0.2, 0.3, 0.4, 0.5]],  # Layer thicknesses
            dims=['hru', 'midToto'],
        )

        # Collapse HRU first
        sm_data = sm_data.isel(hru=0)

        result = sm_evaluator._depth_weighted_mean(
            sm_data, layer_depths,
            target_depth_m=0.3,  # 30 cm target
            layer_dim='midToto'
        )

        # Should weight by fraction of layer in target depth
        assert isinstance(result, xr.DataArray)


class TestObservedDataPath:
    """Test observed data path resolution."""

    def test_sm_point_path(self, sm_evaluator, tmp_path):
        """Test point SM observed data path."""
        sm_evaluator._project_dir = tmp_path
        sm_evaluator.domain_name = 'test_basin'
        sm_evaluator.optimization_target = 'sm_point'

        result = sm_evaluator.get_observed_data_path()

        assert 'point' in str(result)

    def test_sm_smap_path(self, sm_evaluator, tmp_path):
        """Test SMAP observed data path."""
        sm_evaluator._project_dir = tmp_path
        sm_evaluator.domain_name = 'test_basin'
        sm_evaluator.optimization_target = 'sm_smap'

        result = sm_evaluator.get_observed_data_path()

        assert 'smap' in str(result)


class TestObservedDataColumn:
    """Test observed data column selection."""

    def test_finds_sm_column_with_depth(self, sm_evaluator):
        """Test finding SM column with depth."""
        sm_evaluator.optimization_target = 'sm_point'
        sm_evaluator.target_depth = '0.1'
        columns = ['DateTime', 'sm_0.1', 'sm_0.5']

        result = sm_evaluator._get_observed_data_column(columns)

        assert result == 'sm_0.1'

    def test_finds_smap_column(self, sm_evaluator):
        """Test finding SMAP column."""
        sm_evaluator.optimization_target = 'sm_smap'
        sm_evaluator.smap_layer = 'surface_sm'
        columns = ['DateTime', 'surface_sm', 'rootzone_sm']

        result = sm_evaluator._get_observed_data_column(columns)

        assert result == 'surface_sm'


class TestNeedsRouting:
    """Test routing requirement."""

    def test_sm_never_needs_routing(self, sm_evaluator):
        """Test that soil moisture evaluation never needs routing."""
        assert sm_evaluator.needs_routing() is False
