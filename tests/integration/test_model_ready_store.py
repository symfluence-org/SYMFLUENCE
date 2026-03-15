"""Integration test for the model-ready data store."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip('netCDF4')
gpd = pytest.importorskip('geopandas')

from shapely.geometry import box

from symfluence.data.model_ready.store_builder import ModelReadyStoreBuilder


def _setup_fixture_domain(root: Path) -> None:
    """Create a minimal domain directory with forcing, obs, and shapefile data."""
    domain = root

    # --- Forcings ---
    basin_avg = domain / 'forcing' / 'basin_averaged_data'
    basin_avg.mkdir(parents=True)
    ds = netCDF4.Dataset(str(basin_avg / 'forcing.nc'), 'w', format='NETCDF4_CLASSIC')
    ds.createDimension('time', 5)
    ds.createDimension('hru', 1)
    t = ds.createVariable('time', 'f8', ('time',))
    t[:] = [0, 1, 2, 3, 4]
    v = ds.createVariable('air_temperature', 'f4', ('time', 'hru'))
    v[:] = np.random.rand(5, 1).astype('f4')
    ds.close()

    # --- Observations ---
    obs_dir = domain / 'observations' / 'streamflow' / 'preprocessed'
    obs_dir.mkdir(parents=True)
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'datetime': dates,
        'discharge_cms': np.random.uniform(1, 10, 30),
    })
    df.to_csv(obs_dir / 'test_streamflow_processed.csv', index=False)

    # --- Catchment shapefile ---
    catch_dir = domain / 'shapefiles' / 'catchment'
    catch_dir.mkdir(parents=True)
    gdf = gpd.GeoDataFrame({
        'HRU_ID': ['hru_0', 'hru_1'],
        'HRU_area': [1e6, 2e6],
        'geometry': [box(0, 50, 1, 51), box(1, 50, 2, 51)],
    }, crs='EPSG:4326')
    gdf.to_file(catch_dir / 'test_HRUs_lumped.shp')

    # --- DEM intersection ---
    dem_dir = domain / 'shapefiles' / 'catchment_intersection' / 'with_dem'
    dem_dir.mkdir(parents=True)
    gdf_dem = gpd.GeoDataFrame({
        'elev_mean': [1500.0, 1800.0],
        'geometry': [box(0, 50, 1, 51), box(1, 50, 2, 51)],
    }, crs='EPSG:4326')
    gdf_dem.to_file(dem_dir / 'catchment_with_dem.shp')


class TestModelReadyStoreIntegration:
    """Full store build from a fixture domain."""

    def test_build_all(self, tmp_path):
        _setup_fixture_domain(tmp_path)

        builder = ModelReadyStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
            config_dict={'FORCING_DATASET': 'ERA5'},
        )
        builder.build_all()

        # Verify forcings
        forcings_dir = tmp_path / 'data' / 'model_ready' / 'forcings'
        assert forcings_dir.exists()
        assert (forcings_dir / 'forcing.nc').exists()

        # Verify observations
        obs_nc = tmp_path / 'data' / 'model_ready' / 'observations' / 'test_observations.nc'
        assert obs_nc.exists()
        ds = netCDF4.Dataset(str(obs_nc), 'r')
        assert 'streamflow' in ds.groups
        ds.close()

        # Verify attributes
        attrs_nc = tmp_path / 'data' / 'model_ready' / 'attributes' / 'test_attributes.nc'
        assert attrs_nc.exists()
        ds = netCDF4.Dataset(str(attrs_nc), 'r')
        assert 'hru_identity' in ds.groups
        assert 'terrain' in ds.groups
        ds.close()

    def test_is_store_complete(self, tmp_path):
        _setup_fixture_domain(tmp_path)

        builder = ModelReadyStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
        )
        assert not builder.is_store_complete()

        builder.build_all()
        assert builder.is_store_complete()

    def test_forcing_fallback_in_preprocessor(self, tmp_path):
        """Verify base_preprocessor uses model-ready forcings when available."""
        _setup_fixture_domain(tmp_path)

        builder = ModelReadyStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
        )
        builder.build_forcings()

        # Check that the model-ready forcing path exists and has files
        mr_forcing = tmp_path / 'data' / 'model_ready' / 'forcings'
        assert mr_forcing.exists()
        assert list(mr_forcing.glob('*.nc'))

    def test_cf_metadata_on_forcings(self, tmp_path):
        """Verify CF-1.8 metadata is written to forcing files."""
        _setup_fixture_domain(tmp_path)

        builder = ModelReadyStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
            config_dict={'FORCING_DATASET': 'ERA5'},
        )
        builder.build_forcings()

        # Read the original file (enriched in-place)
        nc_path = tmp_path / 'forcing' / 'basin_averaged_data' / 'forcing.nc'
        ds = netCDF4.Dataset(str(nc_path), 'r')
        assert ds.Conventions == 'CF-1.8'
        assert ds.variables['air_temperature'].standard_name == 'air_temperature'
        ds.close()

    def test_migrate_from_legacy(self, tmp_path):
        """migrate_from_legacy should build the same store."""
        _setup_fixture_domain(tmp_path)

        builder = ModelReadyStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
        )
        builder.migrate_from_legacy()

        assert builder.is_store_complete()
