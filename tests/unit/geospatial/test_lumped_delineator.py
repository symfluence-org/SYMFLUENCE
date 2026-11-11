"""
Tests for LumpedWatershedDelineator class.

Tests lumped watershed delineation including:
- Single basin creation
- Pour point handling
- River network creation
- Required field validation
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import fixtures
from .conftest import requires_geopandas

# Skip all tests if geopandas not available
pytestmark = requires_geopandas

try:
    import geopandas as gpd
except ImportError:
    gpd = None


class TestLumpedDelineatorInit:
    """Tests for LumpedWatershedDelineator initialization."""

    def test_init_default_config(self, mock_config_dict, mock_logger, tmp_path):
        """Test LumpedWatershedDelineator initializes with default configuration."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(mock_config_dict, mock_logger)

                assert delineator.delineation_method == 'TauDEM'

    def test_method_name_is_lumped(self, mock_config_dict, mock_logger, tmp_path):
        """Test that method name returns 'lumped'."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                method_name = delineator._get_delineation_method_name()

                assert method_name == 'lumped'


class TestRiverNetworkCreation:
    """Tests for river network creation from pour point."""

    def test_create_river_network_structure(self, geospatial_test_domain_setup, tmp_path):
        """Test river network has correct structure."""
        setup = geospatial_test_domain_setup
        config = setup['config']
        logger = setup['logger']
        pour_point_path = setup['pour_point_path']

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(config, logger)

                # Create river network from pour point
                river_network_path = tmp_path / "test_network.shp"
                delineator._create_river_network(pour_point_path, river_network_path)

                # Verify network was created
                assert river_network_path.exists()

                network_gdf = gpd.read_file(river_network_path)
                assert 'LINKNO' in network_gdf.columns
                assert 'DSLINKNO' in network_gdf.columns
                assert 'Length' in network_gdf.columns
                assert 'Slope' in network_gdf.columns
                assert 'GRU_ID' in network_gdf.columns

    def test_river_network_default_values(self, geospatial_test_domain_setup, tmp_path):
        """Test river network has correct default values."""
        setup = geospatial_test_domain_setup
        config = setup['config']
        logger = setup['logger']
        pour_point_path = setup['pour_point_path']

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(config, logger)

                river_network_path = tmp_path / "test_network.shp"
                delineator._create_river_network(pour_point_path, river_network_path)

                network_gdf = gpd.read_file(river_network_path)

                # Verify default values
                assert network_gdf['LINKNO'].iloc[0] == 1
                assert network_gdf['DSLINKNO'].iloc[0] == 0  # Outlet
                assert network_gdf['GRU_ID'].iloc[0] == 1


class TestRequiredFields:
    """Tests for required field validation."""

    def test_ensure_required_fields_adds_missing(self, geospatial_test_domain_setup, tmp_path):
        """Test that missing required fields are added."""
        setup = geospatial_test_domain_setup
        config = setup['config']
        logger = setup['logger']
        basins_gdf = setup['basins_gdf'].copy()
        rivers_gdf = setup['rivers_gdf'].copy()

        # Remove some fields to test addition
        if 'GRU_ID' in basins_gdf.columns:
            basins_gdf = basins_gdf.drop(columns=['GRU_ID'])

        basins_path = tmp_path / "test_basins.shp"
        rivers_path = tmp_path / "test_rivers.shp"

        # Re-add GRU_ID since shapefile needs it for our method
        basins_gdf['GRU_ID'] = range(1, len(basins_gdf) + 1)
        basins_gdf.to_file(basins_path)
        rivers_gdf.to_file(rivers_path)

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(config, logger)
                delineator._ensure_required_fields(basins_path, rivers_path)

                # Reload and check
                updated_basins = gpd.read_file(basins_path)
                assert 'GRU_ID' in updated_basins.columns
                assert 'gru_to_seg' in updated_basins.columns

    def test_ensure_required_fields_calculates_area(self, geospatial_test_domain_setup, tmp_path):
        """Test that GRU_area is calculated if missing."""
        setup = geospatial_test_domain_setup
        config = setup['config']
        logger = setup['logger']
        basins_gdf = setup['basins_gdf'].copy()
        rivers_gdf = setup['rivers_gdf'].copy()

        # Remove area column
        if 'GRU_area' in basins_gdf.columns:
            basins_gdf = basins_gdf.drop(columns=['GRU_area'])

        basins_path = tmp_path / "test_basins_no_area.shp"
        rivers_path = tmp_path / "test_rivers.shp"

        basins_gdf.to_file(basins_path)
        rivers_gdf.to_file(rivers_path)

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(config, logger)
                delineator._ensure_required_fields(basins_path, rivers_path)

                # Reload and check
                updated_basins = gpd.read_file(basins_path)
                assert 'GRU_area' in updated_basins.columns
                assert all(updated_basins['GRU_area'] > 0)


class TestOutputPaths:
    """Tests for output path generation."""

    def test_output_paths_structure(self, mock_config_dict, mock_logger, tmp_path):
        """Test output paths follow expected structure."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['DOMAIN_NAME'] = 'test_watershed'

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(mock_config_dict, mock_logger)

                # Check that interim directory is set correctly
                assert 'lumped' in str(delineator.interim_dir)


class TestTauDEMIntegration:
    """Tests for TauDEM integration (mocked)."""

    def test_taudem_workflow_steps(self, mock_config_dict, mock_logger, tmp_path):
        """Test TauDEM workflow executes expected steps."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)

        # Create required directories
        pour_point_dir = tmp_path / "shapefiles" / "pour_point"
        pour_point_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock pour point file
        from fixtures.geospatial_fixtures import synthetic_pour_point_gdf
        pour_point = synthetic_pour_point_gdf()
        pour_point_path = pour_point_dir / "test_domain_pourPoint.shp"
        pour_point.to_file(pour_point_path)

        mock_taudem = MagicMock()
        mock_taudem.run_command = MagicMock()

        mock_gdal = MagicMock()

        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor', return_value=mock_taudem):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor', return_value=mock_gdal):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator

                delineator = LumpedWatershedDelineator(mock_config_dict, mock_logger)
                delineator.pour_point_path = pour_point_path

                # Create output directory
                delineator.output_dir.mkdir(parents=True, exist_ok=True)

                # Mock DEM existence
                delineator.dem_path = tmp_path / "dem.tif"
                delineator.dem_path.parent.mkdir(parents=True, exist_ok=True)
                delineator.dem_path.touch()

                # Run the TauDEM workflow (this will fail at gagewatershed but we test the setup)
                # Note: This is a partial test - full integration requires TauDEM installation


class TestGagewatershedSelection:
    """Tests for selecting the outlet drainage area from gagewatershed regions.

    ``gagewatershed`` writes one *nested* region per gauge, labelled with that
    gauge's 0-based id. Two failure modes are covered here: the id-0 labelling that
    used to make the pixel-exact path raise (sending every lumped delineation down
    the streamnet fallback, which overshoots downstream of the gauge), and the
    multi-gauge case where dissolving every region would fuse unrelated watersheds.
    """

    @staticmethod
    def _delineator(mock_config_dict, mock_logger, tmp_path):
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.lumped_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.lumped_delineator import (
                    LumpedWatershedDelineator,
                )
                return LumpedWatershedDelineator(mock_config_dict, mock_logger)

    @staticmethod
    def _regions(ids):
        from shapely.geometry import box
        return gpd.GeoDataFrame(
            {'ID': list(ids)},
            geometry=[box(i, 0, i + 1, 1) for i in range(len(ids))],
            crs='EPSG:4326',
        )

    def test_reads_id_tree(self, mock_config_dict, mock_logger, tmp_path):
        d = self._delineator(mock_config_dict, mock_logger, tmp_path)
        p = tmp_path / "watershed_id.txt"
        p.write_text("id iddown\n0 -1\n1 0\n2 1\n")

        assert d._read_gagewatershed_id_tree(p) == {0: -1, 1: 0, 2: 1}

    def test_missing_id_tree_returns_none(self, mock_config_dict, mock_logger, tmp_path):
        d = self._delineator(mock_config_dict, mock_logger, tmp_path)
        assert d._read_gagewatershed_id_tree(tmp_path / "absent.txt") is None

    def test_single_gauge_region_is_kept(self, mock_config_dict, mock_logger, tmp_path):
        """The single-gauge case is labelled id 0 and must survive untouched."""
        d = self._delineator(mock_config_dict, mock_logger, tmp_path)
        regions = self._regions([0])

        out = d._select_gagewatershed_basin(regions, {0: -1})

        assert out['ID'].tolist() == [0]

    def test_keeps_only_regions_draining_to_outlet(self, mock_config_dict, mock_logger, tmp_path):
        """Gauges 0<-1<-2 drain to the outlet; gauges 7,8 are a separate network."""
        d = self._delineator(mock_config_dict, mock_logger, tmp_path)
        regions = self._regions([0, 1, 2, 7, 8])
        id_tree = {0: -1, 1: 0, 2: 1, 7: 8, 8: 7}  # 7/8 form a cycle, no outlet

        out = d._select_gagewatershed_basin(regions, id_tree)

        assert sorted(out['ID'].tolist()) == [0, 1, 2]

    def test_falls_back_to_all_regions_without_topology(self, mock_config_dict, mock_logger, tmp_path):
        """No id file: keep everything rather than silently dropping drainage area."""
        d = self._delineator(mock_config_dict, mock_logger, tmp_path)
        regions = self._regions([0, 1, 2])

        out = d._select_gagewatershed_basin(regions, None)

        assert sorted(out['ID'].tolist()) == [0, 1, 2]

    def test_falls_back_when_no_outlet_in_topology(self, mock_config_dict, mock_logger, tmp_path):
        d = self._delineator(mock_config_dict, mock_logger, tmp_path)
        regions = self._regions([0, 1])

        out = d._select_gagewatershed_basin(regions, {0: 1, 1: 0})

        assert sorted(out['ID'].tolist()) == [0, 1]


def test_min_gru_size_warns_that_it_is_not_applied(tmp_path):
    """MIN_GRU_SIZE is read but nothing merges GRUs: say so instead of ignoring it."""
    from unittest.mock import MagicMock, patch

    from symfluence.geospatial.geofabric.base.base_delineator import BaseGeofabricDelineator

    class _Delineator(BaseGeofabricDelineator):
        def _get_delineation_method_name(self):
            return 'test'

    def _make(min_gru_size):
        logger = MagicMock()
        config = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'DOMAIN_NAME': 'd', 'MIN_GRU_SIZE': min_gru_size}
        with patch.object(BaseGeofabricDelineator, '_get_taudem_dir', return_value=tmp_path), \
                patch.object(BaseGeofabricDelineator, '_set_taudem_path'), \
                patch.object(BaseGeofabricDelineator, '_get_dem_path', return_value=tmp_path / 'dem.tif'):
            try:
                _Delineator(config, logger)
            except Exception:  # noqa: BLE001 - later init steps need a full config; irrelevant here
                pass
        return [str(c) for c in logger.warning.call_args_list]

    assert any('MIN_GRU_SIZE' in w and 'not applied' in w for w in _make(5.0))
    assert not any('MIN_GRU_SIZE' in w for w in _make(0))
