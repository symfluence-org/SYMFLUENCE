"""
Tests for MESH preprocessor.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestMESHPreprocessorInitialization:
    """Tests for MESH preprocessor initialization."""

    def test_preprocessor_can_be_imported(self):
        """Test that MESHPreProcessor can be imported."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor
        assert MESHPreProcessor is not None

    def test_preprocessor_initialization(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test preprocessor initializes with config."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        assert preprocessor is not None
        assert preprocessor.domain_name == 'test_domain'

    def test_preprocessor_model_name(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test preprocessor returns correct model name."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        assert preprocessor._get_model_name() == 'MESH'


class TestMESHSpatialMode:
    """Tests for MESH spatial mode detection."""

    def test_distributed_mode_from_config(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test distributed mode is set from config."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        mode = preprocessor._get_spatial_mode()
        assert mode == 'distributed'

    def test_auto_mode_resolves_to_lumped(self, temp_dir, mock_logger):
        """Test auto mode resolves to lumped for lumped domains."""
        from symfluence.core.config.models import SymfluenceConfig
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'mesh_test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'MESH',
            'FORCING_DATASET': 'ERA5',
            'MESH_SPATIAL_MODE': 'auto',
        }
        config = SymfluenceConfig(**config_dict)

        # Set up directories
        data_dir = config.system.data_dir
        domain_dir = data_dir / f"domain_{config.domain.name}"
        (domain_dir / 'forcing' / 'MESH_input').mkdir(parents=True, exist_ok=True)
        (domain_dir / 'shapefiles' / 'river_basins').mkdir(parents=True, exist_ok=True)
        (domain_dir / 'shapefiles' / 'river_network').mkdir(parents=True, exist_ok=True)

        preprocessor = MESHPreProcessor(config, mock_logger)
        mode = preprocessor._get_spatial_mode()
        assert mode == 'lumped'

    def test_auto_mode_resolves_to_distributed(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test auto mode resolves to distributed for delineated domains."""
        from symfluence.core.config.models import SymfluenceConfig
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        # Create config with auto mode
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(mesh_config.system.data_dir),
            'SYMFLUENCE_CODE_DIR': str(mesh_config.system.code_dir),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'mesh_test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'delineate',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'MESH',
            'FORCING_DATASET': 'ERA5',
            'MESH_SPATIAL_MODE': 'auto',
        }
        config = SymfluenceConfig(**config_dict)

        preprocessor = MESHPreProcessor(config, mock_logger)
        mode = preprocessor._get_spatial_mode()
        assert mode == 'distributed'


class TestMESHComponentProperties:
    """Tests for MESH preprocessor lazy-initialized components."""

    def test_meshflow_manager_property(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test meshflow manager property creates manager."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        # Initialize the meshflow config first
        preprocessor._meshflow_config = {'test': 'config'}

        # Accessing the property should create the manager
        with patch('symfluence.models.mesh.preprocessor.MESHFlowManager') as mock_manager:
            mock_manager.return_value = MagicMock()
            manager = preprocessor.meshflow_manager
            assert manager is not None
            mock_manager.assert_called_once()

    def test_drainage_database_property(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test drainage database property creates database manager."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)

        with patch('symfluence.models.mesh.preprocessor.MESHDrainageDatabase') as mock_db:
            mock_db.return_value = MagicMock()
            db = preprocessor.drainage_database
            assert db is not None
            mock_db.assert_called_once()

    def test_parameter_fixer_property(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test parameter fixer property creates fixer."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)

        with patch('symfluence.models.mesh.preprocessor.MESHParameterFixer') as mock_fixer:
            mock_fixer.return_value = MagicMock()
            fixer = preprocessor.parameter_fixer
            assert fixer is not None
            mock_fixer.assert_called_once()


class TestMESHConfigCreation:
    """Tests for MESH meshflow configuration creation."""

    def test_create_meshflow_config(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test meshflow configuration dictionary creation."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)

        # Mock the data_preprocessor to avoid file I/O
        with patch.object(preprocessor, '_data_preprocessor') as mock_data_proc:
            mock_data_proc.detect_gru_classes.return_value = [10]

            config = preprocessor._create_meshflow_config()

            assert config is not None
            assert 'riv' in config
            assert 'cat' in config
            assert 'forcing_vars' in config
            assert 'settings' in config

    def test_meshflow_config_contains_required_keys(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test meshflow config contains all required keys."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)

        with patch.object(preprocessor, '_data_preprocessor') as mock_data_proc:
            mock_data_proc.detect_gru_classes.return_value = [10]

            config = preprocessor._create_meshflow_config()

            required_keys = [
                'riv', 'cat', 'landcover', 'forcing_files', 'forcing_vars',
                'forcing_units', 'main_id', 'ds_main_id', 'landcover_classes',
                'ddb_vars', 'settings'
            ]
            for key in required_keys:
                assert key in config, f"Missing required key: {key}"


class TestMESHForcingSelection:
    """Tests for discretization-aware forcing selection handed to meshflow.

    Regression guard for the native-Windows MESH exp-02 failure: when a
    basin_averaged_data store holds two remapped forcing files of the SAME
    domain (an untokened legacy remap beside a namespaced ``grus`` remap, both
    overlapping in time), meshflow's ``core.forcing_files == 'single'`` merge
    globbed both and ``xr.combine_by_coords`` died with "Could not find any
    dimension coordinates to use to order the Dataset objects for
    concatenation". ``_select_meshflow_forcing`` must narrow the input to the
    single file matching the run's discretization so the merge never sees the
    ambiguous pair.
    """

    def _make_ds(self):
        import numpy as np
        import xarray as xr

        t = np.array(['2002-01-01', '2002-01-02'], dtype='datetime64[ns]')
        return xr.Dataset(
            {'air_temperature': (('time', 'hru'), np.ones((2, 1)))},
            coords={'time': t, 'hru': [1]},
        )

    def test_two_mismatched_remaps_would_break_combine(self, temp_dir):
        """Document the bug: combining both remaps raises the exact ValueError."""
        import xarray as xr

        # Two coherent hru=1 datasets that share time/hru coords: exactly the
        # situation combine_by_coords cannot order.
        ds1 = self._make_ds()
        ds2 = self._make_ds()
        with pytest.raises(ValueError, match="dimension coordinates"):
            xr.combine_by_coords(
                [ds1, ds2], data_vars='minimal', coords='minimal', compat='override'
            )

    def test_selects_single_discretization_file(
        self, mesh_config, mock_logger, setup_mesh_directories, temp_dir
    ):
        """Two remaps present -> returns only the file matching discretization 'GRUs'."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        forcing_dir = temp_dir / 'basin_averaged_data'
        forcing_dir.mkdir(parents=True, exist_ok=True)
        # Untokened legacy remap (digit-initial hash -> no token) + grus remap.
        legacy = forcing_dir / 'test_domain_ERA5_remapped_4ae454551262b9b7.nc'
        grus = forcing_dir / 'test_domain_ERA5_remapped_grus_4ae454551262b9b7.nc'
        self._make_ds().to_netcdf(legacy)
        self._make_ds().to_netcdf(grus)

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        result = preprocessor._select_meshflow_forcing(
            forcing_dir, str(forcing_dir / '*.nc')
        )

        # A single explicit file path (not the ambiguous glob) is returned.
        assert result == str(grus)

        # And that single file opens as one coherent dataset (no concat needed).
        import glob as _glob

        import xarray as xr
        matched = _glob.glob(result)
        assert len(matched) == 1
        with xr.open_dataset(matched[0]) as ds:
            assert ds.sizes['hru'] == 1

    def test_single_file_returns_glob_unchanged(
        self, mesh_config, mock_logger, setup_mesh_directories, temp_dir
    ):
        """One file present -> nothing to disambiguate, glob passes through."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        forcing_dir = temp_dir / 'basin_averaged_data'
        forcing_dir.mkdir(parents=True, exist_ok=True)
        only = forcing_dir / 'test_domain_ERA5_remapped_grus_4ae454551262b9b7.nc'
        self._make_ds().to_netcdf(only)

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        glob_str = str(forcing_dir / '*.nc')
        result = preprocessor._select_meshflow_forcing(forcing_dir, glob_str)

        assert result == glob_str

    def test_no_token_match_returns_glob_unchanged(
        self, mesh_config, mock_logger, setup_mesh_directories, temp_dir
    ):
        """Two same-discretization time chunks (no token reduction) -> glob unchanged.

        A single-discretization / pre-namespacing store whose files all lack the
        run's token must pass through so meshflow can order them by time.
        """
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        forcing_dir = temp_dir / 'basin_averaged_data'
        forcing_dir.mkdir(parents=True, exist_ok=True)
        # Legacy date-tag-only remaps: neither carries the 'grus' token.
        a = forcing_dir / 'test_domain_RDRS_remapped_2002-01-01-00-00-00.nc'
        b = forcing_dir / 'test_domain_RDRS_remapped_2002-02-01-00-00-00.nc'
        self._make_ds().to_netcdf(a)
        self._make_ds().to_netcdf(b)

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        glob_str = str(forcing_dir / '*.nc')
        result = preprocessor._select_meshflow_forcing(forcing_dir, glob_str)

        assert result == glob_str


class TestMESHPreprocessorHelpers:
    """Tests for MESH preprocessor helper methods."""

    def test_create_minimal_landcover_csv(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test minimal landcover CSV creation."""
        from symfluence.models.mesh.preprocessor import MESHPreProcessor

        preprocessor = MESHPreProcessor(mesh_config, mock_logger)
        csv_path = setup_mesh_directories['attributes_dir'] / 'test_landcover.csv'

        preprocessor._create_minimal_landcover_csv(csv_path)

        assert csv_path.exists()
        content = csv_path.read_text()
        assert 'GRU_ID' in content
        assert 'IGBP_10' in content
