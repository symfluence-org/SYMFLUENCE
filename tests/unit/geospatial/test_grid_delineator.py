"""
Tests for GridDelineator class.

Tests grid-based domain delineation including:
- Grid creation from bounding box
- Watershed clipping
- D8 flow direction extraction
- Native grid creation (Phase 2)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

# Import fixtures
from .conftest import requires_geopandas, requires_rasterio

# Skip all tests if geopandas not available
pytestmark = requires_geopandas

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:
    gpd = None


class TestGridDelineatorInit:
    """Tests for GridDelineator initialization."""

    def test_init_default_config(self, mock_config_dict, mock_logger, tmp_path):
        """Test GridDelineator initializes with default configuration."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)

                assert delineator.grid_cell_size == 1000.0
                assert delineator.clip_to_watershed is True
                assert delineator.grid_source == 'generate'
                assert delineator.native_grid_dataset == 'era5'

    def test_init_custom_grid_size(self, mock_config_dict, mock_logger, tmp_path):
        """Test GridDelineator respects custom grid cell size."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 500.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)

                assert delineator.grid_cell_size == 500.0


class TestGridCreation:
    """Tests for grid creation from bounding box."""

    def test_create_grid_from_bbox_valid(self, mock_config_dict, mock_logger, tmp_path):
        """Test grid creation with valid bounding box."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['BOUNDING_BOX_COORDS'] = '47.0/8.0/46.0/9.0'
        mock_config_dict['GRID_CELL_SIZE'] = 10000.0  # Large cells for fast test

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                assert grid_gdf is not None
                assert len(grid_gdf) > 0
                assert 'GRU_ID' in grid_gdf.columns
                assert 'GRU_area' in grid_gdf.columns
                assert grid_gdf.crs.to_epsg() == 4326

    def test_create_grid_from_bbox_invalid_format(self, mock_config_dict, mock_logger, tmp_path):
        """Test grid creation fails gracefully with invalid bbox format."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['BOUNDING_BOX_COORDS'] = 'invalid/format'

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                assert grid_gdf is None

    def test_create_grid_from_bbox_missing_coords(self, mock_config_dict, mock_logger, tmp_path):
        """Test grid creation fails gracefully with missing bbox coords."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['BOUNDING_BOX_COORDS'] = ''

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                assert grid_gdf is None

    def test_grid_has_sequential_ids(self, mock_config_dict, mock_logger, tmp_path):
        """Test that grid cells have sequential GRU_IDs starting from 1."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 20000.0  # Large cells

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                if grid_gdf is not None and len(grid_gdf) > 0:
                    ids = sorted(grid_gdf['GRU_ID'].tolist())
                    expected_ids = list(range(1, len(grid_gdf) + 1))
                    assert ids == expected_ids


class TestGridAttributes:
    """Tests for grid attribute calculation."""

    def test_add_grid_attributes_centroids(self, mock_config_dict, mock_logger, tmp_path):
        """Test that grid attributes include centroid coordinates."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 20000.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                if grid_gdf is not None:
                    grid_gdf = delineator._add_grid_attributes(grid_gdf)

                    assert 'center_lon' in grid_gdf.columns
                    assert 'center_lat' in grid_gdf.columns
                    assert 'gru_to_seg' in grid_gdf.columns


class TestRiverNetworkFromGrid:
    """Tests for river network creation from grid topology."""

    def test_create_river_network_from_grid(self, mock_config_dict, mock_logger, tmp_path):
        """Test river network creation from grid cells."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_CELL_SIZE'] = 20000.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                grid_gdf = delineator._create_grid_from_bbox()

                if grid_gdf is not None:
                    # Add required columns for network creation
                    grid_gdf['downstream'] = 0  # All outlets for simplicity

                    network_gdf = delineator._create_river_network_from_grid(grid_gdf)

                    assert network_gdf is not None
                    assert len(network_gdf) == len(grid_gdf)
                    assert 'LINKNO' in network_gdf.columns
                    assert 'DSLINKNO' in network_gdf.columns


class TestNativeGridImplementation:
    """Tests for native grid creation from forcing data (Phase 2)."""

    def test_native_grid_era5_structure(self, geospatial_test_domain_setup):
        """Test native grid creation reads ERA5-like forcing data."""
        setup = geospatial_test_domain_setup
        config = setup['config']
        logger = setup['logger']

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(config, logger)

                # Note: This test will exercise the _create_native_grid method
                # After Phase 2 implementation, this should return a valid GeoDataFrame
                result = delineator._create_native_grid()

                # For now (stub), it returns None and logs a warning
                # After implementation: assert result is not None

    def test_native_grid_coord_detection(self, geospatial_test_domain_setup, tmp_path):
        """Test coordinate name detection for various forcing datasets."""
        import xarray as xr

        # Create test datasets with different coordinate conventions
        coord_conventions = [
            {'lat_name': 'latitude', 'lon_name': 'longitude'},  # ERA5
            {'lat_name': 'lat', 'lon_name': 'lon'},  # CERRA/CARRA
            {'lat_name': 'y', 'lon_name': 'x'},  # Some grids
        ]

        for conv in coord_conventions:
            lat = np.arange(46.0, 47.0, 0.25)
            lon = np.arange(8.0, 9.0, 0.25)

            ds = xr.Dataset(
                {'temp': ([conv['lat_name'], conv['lon_name']], np.random.rand(len(lat), len(lon)))},
                coords={conv['lat_name']: lat, conv['lon_name']: lon}
            )

            test_path = tmp_path / f"test_{conv['lat_name']}.nc"
            ds.to_netcdf(test_path)

            # Verify the file was created with expected coordinates
            with xr.open_dataset(test_path) as loaded:
                assert conv['lat_name'] in loaded.coords
                assert conv['lon_name'] in loaded.coords


class TestMethodSuffix:
    """Tests for method suffix generation."""

    def test_method_suffix_generate(self, mock_config_dict, mock_logger, tmp_path):
        """Test method suffix for generated grid."""
        mock_config_dict['PROJECT_DIR'] = str(tmp_path)
        mock_config_dict['GRID_SOURCE'] = 'generate'
        mock_config_dict['GRID_CELL_SIZE'] = 1000.0

        with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.TauDEMExecutor'):
            with patch('symfluence.geospatial.geofabric.delineators.grid_delineator.GDALProcessor'):
                from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

                delineator = GridDelineator(mock_config_dict, mock_logger)
                suffix = delineator._get_delineation_method_name()

                # Suffix should include 'distributed' and cell size
                assert 'distributed' in suffix.lower() or '1000' in suffix


class TestD8Offsets:
    """Tests for D8 flow direction offset constants."""

    def test_d8_offsets_complete(self):
        """Test that all 8 D8 directions are defined."""
        from symfluence.geospatial.geofabric.delineators.grid_delineator import D8_OFFSETS

        assert len(D8_OFFSETS) == 8
        assert all(d in D8_OFFSETS for d in range(1, 9))

    def test_d8_offsets_values(self):
        """Test D8 offset values are valid (row, col) tuples."""
        from symfluence.geospatial.geofabric.delineators.grid_delineator import D8_OFFSETS

        for direction, (drow, dcol) in D8_OFFSETS.items():
            assert isinstance(drow, int)
            assert isinstance(dcol, int)
            assert -1 <= drow <= 1
            assert -1 <= dcol <= 1
            # Can't have both zero (that would be no movement)
            assert not (drow == 0 and dcol == 0)


def _bare_grid_delineator(tmp_path, bbox="64.10/-19.10/64.00/-19.00"):
    """GridDelineator without running __init__ (no TauDEM / project needed)."""
    from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator

    g = object.__new__(GridDelineator)
    g.logger = MagicMock()
    g.grid_cell_size = 1000.0
    g.native_grid_dataset = 'era5'
    g._get_config_value = lambda *a, **k: bbox
    return g


class TestD8TopologyDirection:
    """The grid ``row`` index increases northward; D8 codes must respect that."""

    @pytest.mark.parametrize("code,sign_dx,sign_dy", [
        (1, 1, 0), (2, 1, 1), (3, 0, 1), (4, -1, 1),
        (5, -1, 0), (6, -1, -1), (7, 0, -1), (8, 1, -1),
    ])
    def test_downstream_cell_lies_in_d8_direction(self, tmp_path, code, sign_dx, sign_dy):
        import rasterio
        from rasterio.transform import from_origin

        g = _bare_grid_delineator(tmp_path)
        grid = g._create_grid_from_bbox()
        cent = grid.to_crs(grid.estimate_utm_crs()).geometry.centroid

        d8_path = tmp_path / f"d8_{code}.tif"
        with rasterio.open(
            d8_path, 'w', driver='GTiff', height=800, width=800, count=1, dtype='int16',
            crs='EPSG:4326', transform=from_origin(-19.2, 64.2, 0.0005, 0.0005), nodata=0,
        ) as dst:
            dst.write(np.full((800, 800), code, dtype='int16'), 1)

        out = g._extract_grid_d8_topology(grid.copy(), d8_path)
        x = dict(zip(out['GRU_ID'], cent.x))
        y = dict(zip(out['GRU_ID'], cent.y))
        linked = out[out['downstream'] > 0]
        assert len(linked) > 0
        dx = np.mean([x[d] - x[i] for i, d in zip(linked['GRU_ID'], linked['downstream'])])
        dy = np.mean([y[d] - y[i] for i, d in zip(linked['GRU_ID'], linked['downstream'])])
        assert np.sign(round(dx / 1000.0)) == sign_dx
        assert np.sign(round(dy / 1000.0)) == sign_dy


class TestTopologyCycles:
    """Cycles in the dominant-direction topology are broken, not written."""

    def _grid(self, downstream, elev):
        from shapely.geometry import box
        cells = []
        gid = 1
        for row in range(2):
            for col in range(2):
                cells.append({'geometry': box(col, row, col + 1, row + 1),
                              'GRU_ID': gid, 'row': row, 'col': col})
                gid += 1
        gdf = gpd.GeoDataFrame(cells, crs='EPSG:4326')
        gdf['downstream'] = downstream
        gdf['elev_mean'] = elev
        return gdf

    def test_two_cell_cycle_redirected_to_lower_neighbour(self, tmp_path):
        g = _bare_grid_delineator(tmp_path)
        # 1 <-> 2 form a loop; 3 is lower than both and is an outlet
        gdf = self._grid([2, 1, 0, 3], [100.0, 90.0, 50.0, 120.0])
        out = g._resolve_topology_cycles(gdf)
        down = dict(zip(out['GRU_ID'], out['downstream']))
        assert down == {1: 2, 2: 3, 3: 0, 4: 3}
        g.logger.warning.assert_called_once()

    def test_cycle_without_lower_neighbour_becomes_outlet(self, tmp_path):
        g = _bare_grid_delineator(tmp_path)
        gdf = self._grid([2, 1, 0, 3], [100.0, 90.0, 150.0, 160.0])
        out = g._resolve_topology_cycles(gdf)
        down = dict(zip(out['GRU_ID'], out['downstream']))
        assert down == {1: 2, 2: 0, 3: 0, 4: 3}

    def test_acyclic_topology_untouched(self, tmp_path):
        g = _bare_grid_delineator(tmp_path)
        gdf = self._grid([2, 4, 1, 0], [3.0, 2.0, 4.0, 1.0])
        out = g._resolve_topology_cycles(gdf)
        assert list(out['downstream']) == [2, 4, 1, 0]
        g.logger.warning.assert_not_called()


class TestNativeGridCoordinates:
    def _delineator(self, tmp_path, ds, bbox):
        g = _bare_grid_delineator(tmp_path, bbox=bbox)
        path = tmp_path / "era5_test.nc"
        ds.to_netcdf(path)
        g._locate_forcing_file = lambda: path
        return g

    def test_descending_latitude_keeps_file_index(self, tmp_path):
        import xarray as xr
        lat = np.array([47.0, 46.75, 46.5, 46.25, 46.0])   # ERA5 order: north first
        lon = np.array([8.0, 8.25, 8.5])
        ds = xr.Dataset({'t': (('latitude', 'longitude'), np.zeros((5, 3)))},
                        coords={'latitude': lat, 'longitude': lon})
        g = self._delineator(tmp_path, ds, "46.3/8.2/46.2/8.3")
        grid = g._create_native_grid()
        assert grid is not None and len(grid) > 0
        for _, cell in grid.iterrows():
            assert lat[int(cell['native_lat_idx'])] == pytest.approx(cell['center_lat'])
            assert lon[int(cell['native_lon_idx'])] == pytest.approx(cell['center_lon'])
        # row index increases northward
        by_row = grid.groupby('row')['center_lat'].first()
        assert by_row.is_monotonic_increasing

    def test_two_dimensional_coordinates_raise(self, tmp_path):
        import xarray as xr

        from symfluence.core.exceptions import DiscretizationError
        lat2d, lon2d = np.meshgrid(np.linspace(63, 65, 4), np.linspace(-20, -18, 5), indexing='ij')
        ds = xr.Dataset({'t': (('y', 'x'), np.zeros((4, 5)))},
                        coords={'latitude': (('y', 'x'), lat2d), 'longitude': (('y', 'x'), lon2d)})
        g = self._delineator(tmp_path, ds, "64.5/-19.5/63.5/-18.5")
        with pytest.raises(DiscretizationError, match="1-D"):
            g._create_native_grid()
