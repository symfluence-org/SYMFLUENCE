"""
Tests for StreamflowEvaluator.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator


def _make_config_value_getter(config_dict):
    """Create a mock _get_config_value that uses config_dict values."""
    call_count = [0]
    init_values = [
        '2010-01-01, 2015-12-31',
        '2016-01-01, 2018-12-31',
        'daily',
    ]

    def mock_get_config_value(typed_accessor, default=None, dict_key=None):
        if call_count[0] < len(init_values):
            result = init_values[call_count[0]]
            call_count[0] += 1
            return result
        if dict_key and dict_key in config_dict:
            return config_dict[dict_key]
        return default

    return mock_get_config_value


@pytest.fixture
def streamflow_evaluator(mock_config, tmp_path, mock_logger):
    """Create a StreamflowEvaluator for testing."""
    config_dict = {
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'CALIBRATION_TIMESTEP': 'daily',
        'OPTIMIZATION_TARGET': 'streamflow',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'REQUIRE_EXPLICIT_CATCHMENT_AREA': False,
        'FIXED_CATCHMENT_AREA': None,
    }
    mock_config.to_dict.side_effect = lambda flatten=True: config_dict
    mock_config.domain.definition_method = 'lumped'
    mock_config.domain.delineation.routing = 'lumped'

    with patch.object(StreamflowEvaluator, '_get_config_value',
                     side_effect=_make_config_value_getter(config_dict)):
        evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
        evaluator.config_dict = config_dict
        return evaluator


class TestStreamflowEvaluatorInit:
    """Test StreamflowEvaluator initialization."""

    def test_basic_initialization(self, streamflow_evaluator):
        """Test basic initialization."""
        assert streamflow_evaluator is not None

    def test_optimization_target(self, streamflow_evaluator):
        """Test that streamflow evaluator doesn't override target."""
        # StreamflowEvaluator should inherit from ModelEvaluator without
        # setting a specific optimization_target in __init__
        assert hasattr(streamflow_evaluator, 'calibration_period')


class TestMizuRouteDetection:
    """Test mizuRoute output detection."""

    def test_detects_mizuroute_output(self, streamflow_evaluator, tmp_path):
        """Test detection of mizuRoute output file."""
        # Create mock mizuRoute output
        ds = xr.Dataset({
            'IRFroutedRunoff': (['time', 'seg'], np.random.rand(10, 5)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'seg': range(5),
        })

        file_path = tmp_path / 'mizuroute_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._is_mizuroute_output(file_path)
        assert result is True

    def test_detects_summa_output(self, streamflow_evaluator, tmp_path):
        """Test detection of SUMMA (non-mizuRoute) output file."""
        ds = xr.Dataset({
            'averageRoutedRunoff': (['time', 'hru'], np.random.rand(10, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'summa_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._is_mizuroute_output(file_path)
        assert result is False


class TestExtractMizuRouteStreamflow:
    """Test mizuRoute streamflow extraction."""

    def test_extracts_irf_runoff(self, streamflow_evaluator, tmp_path):
        """Test extraction of IRFroutedRunoff."""
        # Create mock data with outlet having highest mean
        data = np.zeros((10, 5))
        data[:, 4] = np.random.rand(10) + 10  # Outlet has highest values

        ds = xr.Dataset({
            'IRFroutedRunoff': (['time', 'seg'], data),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'seg': range(5),
        })

        file_path = tmp_path / 'mizuroute_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._extract_mizuroute_streamflow(file_path)

        assert isinstance(result, pd.Series)
        assert len(result) == 10

    def test_extracts_kwt_runoff(self, streamflow_evaluator, tmp_path):
        """Test extraction of KWTroutedRunoff as fallback."""
        data = np.zeros((10, 5))
        data[:, 4] = np.random.rand(10) + 10

        ds = xr.Dataset({
            'KWTroutedRunoff': (['time', 'seg'], data),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'seg': range(5),
        })

        file_path = tmp_path / 'mizuroute_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._extract_mizuroute_streamflow(file_path)

        assert isinstance(result, pd.Series)
        assert len(result) == 10

    def test_raises_for_missing_variable(self, streamflow_evaluator, tmp_path):
        """Test raises error when no suitable variable found."""
        ds = xr.Dataset({
            'some_other_variable': (['time', 'seg'], np.random.rand(10, 5)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'seg': range(5),
        })

        file_path = tmp_path / 'empty_output.nc'
        ds.to_netcdf(file_path)

        with pytest.raises(ValueError, match="No suitable streamflow variable"):
            streamflow_evaluator._extract_mizuroute_streamflow(file_path)


class TestExtractSummaStreamflow:
    """Test SUMMA streamflow extraction."""

    def test_extracts_routed_runoff(self, streamflow_evaluator, tmp_path):
        """Test extraction of averageRoutedRunoff."""
        ds = xr.Dataset({
            'averageRoutedRunoff': (['time', 'hru'], np.random.uniform(1e-9, 1e-8, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })
        ds['averageRoutedRunoff'].attrs['units'] = 'm s-1'

        file_path = tmp_path / 'summa_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._extract_summa_streamflow(file_path)

        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_converts_mass_flux_to_volume(self, streamflow_evaluator, tmp_path):
        """Test conversion of mass flux (kg m-2 s-1) to volume flux."""
        # Use values indicating mass flux (> 1e-6)
        ds = xr.Dataset({
            'averageRoutedRunoff': (['time', 'hru'], np.full((100, 1), 1e-4)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })
        ds['averageRoutedRunoff'].attrs['units'] = 'kg m-2 s-1'

        file_path = tmp_path / 'summa_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._extract_summa_streamflow(file_path)

        # Values should be divided by 1000 (water density)
        assert result.mean() == pytest.approx(1e-4 / 1000 * 1e6, rel=0.1)

    def test_extracts_precomputed_streamflow(self, streamflow_evaluator, tmp_path):
        """Test extraction of pre-computed streamflow variable."""
        ds = xr.Dataset({
            'streamflow': (['time'], np.random.uniform(10, 100, 100)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
        })
        ds['streamflow'].attrs['units'] = 'm3/s'

        file_path = tmp_path / 'model_output.nc'
        ds.to_netcdf(file_path)

        result = streamflow_evaluator._extract_summa_streamflow(file_path)

        assert isinstance(result, pd.Series)
        assert len(result) == 100


class TestExtractFuseStreamflow:
    """FUSE extraction must never guess a unit conversion silently."""

    @staticmethod
    def _write_fuse(tmp_path, values, units=None, var_name='q_routed'):
        ds = xr.Dataset({
            var_name: (['time'], np.asarray(values, dtype=float)),
        }, coords={'time': pd.date_range('2010-01-01', periods=len(values))})
        if units is not None:
            ds[var_name].attrs['units'] = units
        path = tmp_path / 'fuse_output.nc'
        ds.to_netcdf(path)
        return path

    def test_m3s_units_pass_through(self, streamflow_evaluator, tmp_path):
        path = self._write_fuse(tmp_path, [5.0, 6.0], units='m3/s')
        result = streamflow_evaluator._extract_fuse_streamflow(path)
        assert result.tolist() == [5.0, 6.0]

    def test_mm_per_day_is_converted_with_area(self, streamflow_evaluator, tmp_path):
        streamflow_evaluator.config_dict = {'FIXED_CATCHMENT_AREA': 8.64e7}
        path = self._write_fuse(tmp_path, [1.0], units='mm/day')
        result = streamflow_evaluator._extract_fuse_streamflow(path)
        # 1 mm/day over 86.4 km² -> 1 m³/s
        assert result.iloc[0] == pytest.approx(1.0)

    def test_unrecognized_units_raise(self, streamflow_evaluator, tmp_path):
        from symfluence.core.exceptions import EvaluationError
        path = self._write_fuse(tmp_path, [1.0, 2.0], units='furlongs/fortnight')
        with pytest.raises(EvaluationError, match="Unrecognized streamflow units"):
            streamflow_evaluator._extract_fuse_streamflow(path)

    def test_mm_per_timestep_daily_converts_like_mm_per_day(
        self, streamflow_evaluator, tmp_path
    ):
        """FUSE's native units string 'mm timestep-1' on a daily time axis."""
        streamflow_evaluator.config_dict = {'FIXED_CATCHMENT_AREA': 8.64e7}
        path = self._write_fuse(tmp_path, [1.0, 1.0], units='mm timestep-1')
        result = streamflow_evaluator._extract_fuse_streamflow(path)
        # 1 mm/day over 86.4 km² -> 1 m³/s
        assert result.iloc[0] == pytest.approx(1.0)

    def test_mm_per_timestep_hourly_uses_time_axis_spacing(
        self, streamflow_evaluator, tmp_path
    ):
        streamflow_evaluator.config_dict = {'FIXED_CATCHMENT_AREA': 8.64e7}
        ds = xr.Dataset({
            'q_instnt': (['time'], np.asarray([1.0, 1.0], dtype=float)),
        }, coords={'time': pd.date_range('2010-01-01', periods=2, freq='h')})
        ds['q_instnt'].attrs['units'] = 'mm timestep-1'
        path = tmp_path / 'fuse_hourly.nc'
        ds.to_netcdf(path)
        result = streamflow_evaluator._extract_fuse_streamflow(path)
        # 1 mm/hour over 86.4 km² -> 24 m³/s
        assert result.iloc[0] == pytest.approx(24.0)

    def test_mm_per_hour_converts_as_hourly_not_daily(
        self, streamflow_evaluator, tmp_path
    ):
        """'mm' alone used to route into the mm/day branch; mm/h is hourly."""
        streamflow_evaluator.config_dict = {'FIXED_CATCHMENT_AREA': 8.64e7}
        path = self._write_fuse(tmp_path, [1.0], units='mm/h')
        result = streamflow_evaluator._extract_fuse_streamflow(path)
        # 1 mm/h over 86.4 km² -> 24 m³/s (not the 1 m³/s a daily reading gives)
        assert result.iloc[0] == pytest.approx(24.0)

    def test_cf_volumetric_units_pass_through(self, streamflow_evaluator, tmp_path):
        path = self._write_fuse(tmp_path, [5.0, 6.0], units='m3 s-1')
        result = streamflow_evaluator._extract_fuse_streamflow(path)
        assert result.tolist() == [5.0, 6.0]

    def test_missing_units_on_native_variable_warns_and_converts(
        self, streamflow_evaluator, tmp_path, caplog
    ):
        streamflow_evaluator.config_dict = {'FIXED_CATCHMENT_AREA': 8.64e7}
        path = self._write_fuse(tmp_path, [1.0], units=None, var_name='q_routed')
        with caplog.at_level('WARNING'):
            result = streamflow_evaluator._extract_fuse_streamflow(path)
        assert result.iloc[0] == pytest.approx(1.0)
        assert any('no units attribute' in r.message for r in caplog.records)

    def test_missing_units_on_generic_variable_raises(
        self, streamflow_evaluator, tmp_path
    ):
        from symfluence.core.exceptions import EvaluationError
        path = self._write_fuse(tmp_path, [1.0], units=None, var_name='total_discharge')
        with pytest.raises(EvaluationError, match="Unrecognized streamflow units"):
            streamflow_evaluator._extract_fuse_streamflow(path)


class TestGetCatchmentArea:
    """Test catchment area determination."""

    def test_uses_fixed_area_from_config(self, streamflow_evaluator):
        """Test using fixed catchment area from config."""
        streamflow_evaluator.config_dict = {'FIXED_CATCHMENT_AREA': 1e8}

        result = streamflow_evaluator._get_catchment_area()

        assert result == 1e8

    def test_reads_from_summa_attributes(self, streamflow_evaluator, tmp_path):
        """SUMMA attributes.nc is consulted when SUMMA is the active model."""
        # Create attributes file
        attrs_dir = tmp_path / 'settings' / 'SUMMA'
        attrs_dir.mkdir(parents=True)

        ds = xr.Dataset({
            'HRUarea': (['hru'], [5e6, 3e6]),  # 8 km² total
        },
        coords={'hru': [0, 1]})

        attrs_file = attrs_dir / 'attributes.nc'
        ds.to_netcdf(attrs_file)

        streamflow_evaluator._project_dir = tmp_path
        streamflow_evaluator.config_dict = {'HYDROLOGICAL_MODEL': 'SUMMA'}

        result = streamflow_evaluator._get_catchment_area()

        assert result == pytest.approx(8e6)

    def test_summa_attributes_ignored_for_other_models(
        self, streamflow_evaluator, tmp_path
    ):
        """A non-SUMMA run never reads SUMMA's settings files for area."""
        attrs_dir = tmp_path / 'settings' / 'SUMMA'
        attrs_dir.mkdir(parents=True)
        ds = xr.Dataset(
            {'HRUarea': (['hru'], [5e6, 3e6])}, coords={'hru': [0, 1]}
        )
        ds.to_netcdf(attrs_dir / 'attributes.nc')

        streamflow_evaluator._project_dir = tmp_path
        streamflow_evaluator.config_dict = {
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FIXED_CATCHMENT_AREA': None,
            'REQUIRE_EXPLICIT_CATCHMENT_AREA': False,
        }

        result = streamflow_evaluator._get_catchment_area()

        assert result == pytest.approx(1e6)

    def test_default_fallback(self, streamflow_evaluator, tmp_path):
        """Test default fallback to 1 km²."""
        streamflow_evaluator._project_dir = tmp_path
        streamflow_evaluator.config_dict = {
            'REQUIRE_EXPLICIT_CATCHMENT_AREA': False,
            'FIXED_CATCHMENT_AREA': None,
        }

        result = streamflow_evaluator._get_catchment_area()

        assert result == 1e6


class TestObservedDataPath:
    """Test observed data path resolution."""

    def test_default_path(self, streamflow_evaluator, tmp_path):
        """Test default observed data path."""
        streamflow_evaluator._project_dir = tmp_path
        streamflow_evaluator.domain_name = 'test_basin'
        # Explicitly set observations path to None/default so it uses the default path
        streamflow_evaluator.config_dict = {'OBSERVATIONS_PATH': None}
        streamflow_evaluator.config.observations.streamflow_path = None

        result = streamflow_evaluator.get_observed_data_path()

        expected = tmp_path / 'data' / 'observations' / 'streamflow' / 'preprocessed' / 'test_basin_streamflow_processed.csv'
        assert result == expected

    def test_override_path_from_config(self, streamflow_evaluator, tmp_path):
        """Test path override from config."""
        custom_path = tmp_path / 'custom' / 'obs.csv'
        streamflow_evaluator.config_dict = {'OBSERVATIONS_PATH': str(custom_path)}

        result = streamflow_evaluator.get_observed_data_path()

        assert result == custom_path


class TestObservedDataColumn:
    """Test observed data column selection."""

    def test_finds_flow_column(self, streamflow_evaluator):
        """Test finding streamflow column by keyword."""
        columns = ['DateTime', 'flow_m3s', 'temperature']

        result = streamflow_evaluator._get_observed_data_column(columns)

        assert result == 'flow_m3s'

    def test_finds_discharge_column(self, streamflow_evaluator):
        """Test finding discharge column."""
        columns = ['DateTime', 'discharge', 'precip']

        result = streamflow_evaluator._get_observed_data_column(columns)

        assert result == 'discharge'

    def test_finds_q_column(self, streamflow_evaluator):
        """Test finding Q_ prefixed column."""
        columns = ['DateTime', 'q_mm', 'temp']

        result = streamflow_evaluator._get_observed_data_column(columns)

        assert result == 'q_mm'

    def test_returns_none_for_no_match(self, streamflow_evaluator):
        """Test returns None when no matching column found."""
        columns = ['DateTime', 'temperature', 'pressure']

        result = streamflow_evaluator._get_observed_data_column(columns)

        assert result is None


class TestNeedsRouting:
    """Test routing requirement determination."""

    def test_lumped_no_routing(self, mock_config, tmp_path, mock_logger):
        """Test that lumped domain doesn't need routing by default."""
        config_dict = {'ROUTING_DELINEATION': 'lumped'}
        with patch.object(StreamflowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.needs_routing() is False

    def test_distributed_needs_routing(self, mock_config, tmp_path, mock_logger):
        """Test that distributed domain needs routing."""
        config_dict = {'DOMAIN_DEFINITION_METHOD': 'distributed'}
        mock_config.domain.definition_method = 'distributed'
        mock_config.domain.delineation.routing = 'distributed'
        with patch.object(StreamflowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            # Ensure the config property returns the mock
            evaluator._config = mock_config

        # Call needs_routing() outside the patch context - should use the mock config's attributes
        assert evaluator.needs_routing() is True

    def test_lumped_with_river_network_needs_routing(self, mock_config, tmp_path, mock_logger):
        """Test that lumped domain with river_network routing needs routing."""
        config_dict = {'ROUTING_DELINEATION': 'river_network'}
        with patch.object(StreamflowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.needs_routing() is True


class TestCatchmentAreaValidation:
    """Test catchment area validation method."""

    def test_validates_normal_area(self, streamflow_evaluator):
        """Test validation of normal catchment area."""
        result = streamflow_evaluator._validate_catchment_area(1e8, 'test')

        assert result == 1e8

    def test_rejects_zero_area(self, streamflow_evaluator):
        """Test rejection of zero area."""
        result = streamflow_evaluator._validate_catchment_area(0.0, 'test')

        assert result is None

    def test_rejects_negative_area(self, streamflow_evaluator):
        """Test rejection of negative area."""
        result = streamflow_evaluator._validate_catchment_area(-1e6, 'test')

        assert result is None

    def test_rejects_too_small_area(self, streamflow_evaluator):
        """Test rejection of area below minimum threshold."""
        # MIN_CATCHMENT_AREA = 1e3 (0.001 km²)
        result = streamflow_evaluator._validate_catchment_area(500, 'test')

        assert result is None

    def test_rejects_too_large_area(self, streamflow_evaluator):
        """Test rejection of area above maximum threshold."""
        # MAX_CATCHMENT_AREA = 1e12 (1,000,000 km²)
        result = streamflow_evaluator._validate_catchment_area(1e13, 'test')

        assert result is None

    def test_accepts_minimum_area(self, streamflow_evaluator):
        """Test acceptance of area at minimum threshold."""
        result = streamflow_evaluator._validate_catchment_area(1e3, 'test')

        assert result == 1e3

    def test_accepts_maximum_area(self, streamflow_evaluator):
        """Test acceptance of area at maximum threshold."""
        result = streamflow_evaluator._validate_catchment_area(1e12, 'test')

        assert result == 1e12


class TestRequireExplicitCatchmentArea:
    """Test REQUIRE_EXPLICIT_CATCHMENT_AREA config option."""

    def test_raises_when_explicit_required_and_detection_fails(
        self, mock_config, tmp_path, mock_logger
    ):
        """Test that ValueError is raised when explicit area required but not found."""
        config_dict = {
            'REQUIRE_EXPLICIT_CATCHMENT_AREA': True,
            'FIXED_CATCHMENT_AREA': None,
        }

        with patch.object(StreamflowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator._project_dir = tmp_path  # Empty directory, no files to detect

            with pytest.raises(ValueError, match="REQUIRE_EXPLICIT_CATCHMENT_AREA"):
                evaluator._get_catchment_area()

    def test_uses_default_when_explicit_not_required(
        self, mock_config, tmp_path, mock_logger
    ):
        """Test fallback to default when explicit not required."""
        config_dict = {
            'REQUIRE_EXPLICIT_CATCHMENT_AREA': False,
            'FIXED_CATCHMENT_AREA': None,
        }

        with patch.object(StreamflowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator._project_dir = tmp_path

            result = evaluator._get_catchment_area()

            assert result == 1e6  # Default 1 km²

    def test_fixed_area_overrides_explicit_requirement(
        self, mock_config, tmp_path, mock_logger
    ):
        """Test that fixed area works even with explicit requirement."""
        config_dict = {
            'REQUIRE_EXPLICIT_CATCHMENT_AREA': True,
            'FIXED_CATCHMENT_AREA': 5e7,
        }

        with patch.object(StreamflowEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = StreamflowEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator._project_dir = tmp_path

            result = evaluator._get_catchment_area()

            assert result == 5e7
