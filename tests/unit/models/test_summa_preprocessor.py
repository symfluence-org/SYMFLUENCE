"""
Unit tests for SUMMA preprocessor.

Tests SUMMA-specific preprocessing functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from symfluence.models.summa import SummaPreProcessor


class TestSummaPreProcessorInitialization:
    """Test SUMMA preprocessor initialization."""

    def test_initialization_with_valid_config(self, summa_config, mock_logger, setup_test_directories):
        """Test SUMMA preprocessor initializes correctly."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        assert preprocessor.model_name == "SUMMA"
        assert preprocessor.domain_name == summa_config.domain.name
        assert preprocessor.forcing_dataset == summa_config.forcing.dataset.lower()

    def test_summa_specific_paths(self, summa_config, mock_logger, setup_test_directories):
        """Test SUMMA-specific path initialization."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Check SUMMA-specific paths (forcing now under data/)
        assert preprocessor.forcing_summa_path == preprocessor.project_dir / 'data' / 'forcing' / 'SUMMA_input'
        assert preprocessor.dem_path.exists() or preprocessor.dem_path.name.endswith('.tif')

    def test_summa_configuration_attributes(self, summa_config, mock_logger, setup_test_directories):
        """Test SUMMA configuration attributes are set."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)
        preprocessor.hruId = summa_config.paths.catchment_hruid
        preprocessor.gruId = summa_config.paths.catchment_gruid

        assert preprocessor.hruId == summa_config.paths.catchment_hruid
        assert preprocessor.gruId == summa_config.paths.catchment_gruid
        assert preprocessor.data_step == summa_config.forcing.time_step_size
        assert preprocessor.forcing_measurement_height == float(summa_config.forcing.measurement_height)

    def test_uses_base_class_forcing_paths(self, summa_config, mock_logger, setup_test_directories):
        """Test that SUMMA uses base class forcing paths."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # These should come from BaseModelPreProcessor
        assert hasattr(preprocessor, 'merged_forcing_path')
        assert hasattr(preprocessor, 'shapefile_path')
        assert hasattr(preprocessor, 'intersect_path')


class TestSummaPathResolution:
    """Test SUMMA path resolution methods."""

    def test_dem_path_default(self, summa_config, mock_logger, setup_test_directories):
        """Test DEM path with default name."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        expected_name = f"domain_{summa_config.domain.name}_elv.tif"
        assert preprocessor.dem_path.name == expected_name

    def test_dem_path_custom(self, summa_config, mock_logger, setup_test_directories):
        """Test DEM path with custom name."""
        # Note: SymfluenceConfig is frozen, but we can re-init if needed.
        # However, for unit tests on preprocessor, we can often just mock the config.
        # But here let's create a new config with the override.
        from symfluence.core.config.models import SymfluenceConfig
        overrides = summa_config.to_dict(flatten=True)
        overrides['DEM_NAME'] = 'custom_dem.tif'
        custom_config = SymfluenceConfig(**overrides)

        preprocessor = SummaPreProcessor(custom_config, mock_logger)

        assert preprocessor.dem_path.name == 'custom_dem.tif'

    def test_catchment_path_default(self, summa_config, mock_logger, setup_test_directories):
        """Test catchment path with defaults."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        discretization = summa_config.domain.discretization
        expected_name = f"{summa_config.domain.name}_HRUs_{discretization}.shp"
        assert preprocessor.catchment_name == expected_name

    def test_river_network_path_default(self, summa_config, mock_logger, setup_test_directories):
        """Test river network path with defaults."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        expected_name = f"{summa_config.domain.name}_riverNetwork_lumped.shp" # Base method suffix
        assert preprocessor.river_network_name == expected_name


class TestSummaCopyBaseSettings:
    """Test SUMMA base settings copying."""

    def test_copy_base_settings_uses_correct_source(self, summa_config, mock_logger, setup_test_directories):
        """Test that copy_base_settings uses correct source directory."""
        # Create a config with overridden code_dir for this test
        from symfluence.core.config.models import SymfluenceConfig
        overrides = summa_config.to_dict(flatten=True)
        overrides['SYMFLUENCE_CODE_DIR'] = str(setup_test_directories['code_dir'])
        custom_config = SymfluenceConfig(**overrides)

        preprocessor = SummaPreProcessor(custom_config, mock_logger)

        # Call copy_base_settings
        preprocessor.copy_base_settings()

        # Verify file was copied
        settings_path = preprocessor.setup_dir
        assert settings_path.exists()


class TestSummaPreprocessingWorkflow:
    """Test SUMMA preprocessing workflow."""

    @patch.object(SummaPreProcessor, 'apply_datastep_and_lapse_rate')
    @patch.object(SummaPreProcessor, 'copy_base_settings')
    @patch.object(SummaPreProcessor, 'create_file_manager')
    @patch.object(SummaPreProcessor, 'create_forcing_file_list')
    @patch.object(SummaPreProcessor, 'create_initial_conditions')
    @patch.object(SummaPreProcessor, 'create_trial_parameters')
    @patch.object(SummaPreProcessor, 'create_attributes_file')
    def test_run_preprocessing_calls_all_steps(
        self,
        mock_attrs,
        mock_params,
        mock_initial,
        mock_forcing_list,
        mock_file_mgr,
        mock_copy,
        mock_lapse,
        summa_config,
        mock_logger,
        setup_test_directories
    ):
        """Test that run_preprocessing calls all required steps in order."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Run preprocessing
        preprocessor.run_preprocessing()

        # Verify all steps were called
        mock_lapse.assert_called_once()
        mock_copy.assert_called_once()
        mock_file_mgr.assert_called_once()
        mock_forcing_list.assert_called_once()
        mock_initial.assert_called_once()
        mock_params.assert_called_once()
        mock_attrs.assert_called_once()


class TestSummaTimestepHandling:
    """Test SUMMA timestep configuration."""

    def test_data_step_set_from_config(self, summa_config, mock_logger, setup_test_directories):
        """Test that data_step is set from forcing timestep size."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        assert preprocessor.data_step == summa_config.forcing.time_step_size

    def test_uses_base_class_timestep_config(self, summa_config, mock_logger, setup_test_directories):
        """Test that SUMMA can use base class timestep config."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Should have access to base class method
        timestep_config = preprocessor.get_timestep_config()

        assert timestep_config['timestep_seconds'] == summa_config.forcing.time_step_size
        assert 'time_label' in timestep_config


class TestSummaRegistration:
    """Test SUMMA model registration."""

    def test_summa_registered_as_preprocessor(self):
        """Test that SUMMA is registered in the model registry."""
        from symfluence.models.registry import ModelRegistry

        # SUMMA should be registered
        assert 'SUMMA' in ModelRegistry._preprocessors


class TestSummaElevationCorrectionSkip:
    """Test SUMMA skips lapse when model-agnostic correction already applied."""

    def test_skips_lapse_when_already_corrected(self, summa_config, mock_logger, setup_test_directories):
        """Verify topology is not loaded when elevation_corrected attribute is present."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)
        fp = preprocessor.forcing_processor

        # Create a fake basin-averaged NetCDF with elevation_corrected=1
        basin_dir = fp.forcing_basin_path
        basin_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset({
            'air_temperature': (['time', 'hru'], np.full((2, 1), 280.0)),
            'hruId': (['hru'], np.array([1001], dtype=np.int32)),
        }, coords={'time': [0.0, 1.0], 'hru': [0]})
        ds.attrs['elevation_corrected'] = 1
        nc_path = basin_dir / f'{preprocessor.domain_name}_ERA5_2020-01-01.nc'
        ds.to_netcdf(nc_path)

        with patch.object(fp, '_load_topology_data') as mock_topo, \
             patch.object(fp, '_process_forcing_batches'):
            preprocessor.apply_datastep_and_lapse_rate()
            # Topology should NOT have been loaded
            mock_topo.assert_not_called()

    def test_applies_lapse_when_not_corrected(self, summa_config, mock_logger, setup_test_directories):
        """Verify existing behavior unchanged when attribute is absent."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)
        fp = preprocessor.forcing_processor

        # Create a fake basin-averaged NetCDF WITHOUT elevation_corrected
        basin_dir = fp.forcing_basin_path
        basin_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset({
            'air_temperature': (['time', 'hru'], np.full((2, 1), 280.0)),
            'hruId': (['hru'], np.array([1001], dtype=np.int32)),
        }, coords={'time': [0.0, 1.0], 'hru': [0]})
        nc_path = basin_dir / f'{preprocessor.domain_name}_ERA5_2020-01-01.nc'
        ds.to_netcdf(nc_path)

        with patch.object(fp, '_find_intersection_file') as mock_find, \
             patch.object(fp, '_load_topology_data') as mock_topo, \
             patch.object(fp, '_precalculate_lapse_corrections') as mock_lapse, \
             patch.object(fp, '_process_forcing_batches'):
            mock_find.return_value = MagicMock()
            mock_topo.return_value = MagicMock()
            mock_lapse.return_value = (MagicMock(), 0.0065)

            preprocessor.apply_datastep_and_lapse_rate()
            # Topology SHOULD have been loaded
            mock_topo.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
