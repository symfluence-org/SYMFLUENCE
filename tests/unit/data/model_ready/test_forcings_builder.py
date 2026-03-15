"""Tests for ForcingsStoreBuilder."""

import os
from pathlib import Path

import pytest

from symfluence.data.model_ready.forcings_builder import ForcingsStoreBuilder

# Only run metadata-enrichment tests when netCDF4 is available
netCDF4 = pytest.importorskip('netCDF4')


def _create_dummy_nc(path: Path, var_name: str = 'air_temperature') -> None:
    """Create a minimal NetCDF file for testing."""
    import numpy as np
    ds = netCDF4.Dataset(str(path), 'w', format='NETCDF4_CLASSIC')
    ds.createDimension('time', 3)
    ds.createDimension('hru', 1)
    t = ds.createVariable('time', 'f8', ('time',))
    t[:] = [0, 1, 2]
    v = ds.createVariable(var_name, 'f4', ('time', 'hru'))
    v[:] = np.random.rand(3, 1).astype('f4')
    ds.close()


class TestForcingsStoreBuilder:
    """Tests for symlink creation, copy mode, and metadata enrichment."""

    def test_build_creates_symlinks(self, tmp_path):
        src = tmp_path / 'forcing' / 'basin_averaged_data'
        src.mkdir(parents=True)
        _create_dummy_nc(src / 'forcing.nc')

        builder = ForcingsStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
            forcing_dataset='ERA5',
            strategy='symlink',
        )
        result = builder.build()
        assert result is not None

        link = tmp_path / 'data' / 'model_ready' / 'forcings' / 'forcing.nc'
        assert link.exists()
        assert link.is_symlink()

    def test_build_creates_copies(self, tmp_path):
        src = tmp_path / 'forcing' / 'basin_averaged_data'
        src.mkdir(parents=True)
        _create_dummy_nc(src / 'forcing.nc')

        builder = ForcingsStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
            forcing_dataset='ERA5',
            strategy='copy',
        )
        result = builder.build()
        assert result is not None

        copy_path = tmp_path / 'data' / 'model_ready' / 'forcings' / 'forcing.nc'
        assert copy_path.exists()
        assert not copy_path.is_symlink()

    def test_build_skips_missing_source(self, tmp_path):
        builder = ForcingsStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
        )
        result = builder.build()
        assert result is None

    def test_build_skips_empty_dir(self, tmp_path):
        src = tmp_path / 'forcing' / 'basin_averaged_data'
        src.mkdir(parents=True)
        builder = ForcingsStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
        )
        result = builder.build()
        assert result is None

    def test_metadata_enrichment(self, tmp_path):
        src = tmp_path / 'forcing' / 'basin_averaged_data'
        src.mkdir(parents=True)
        nc_path = src / 'forcing.nc'
        _create_dummy_nc(nc_path, var_name='air_temperature')

        builder = ForcingsStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
            forcing_dataset='ERA5',
        )
        builder.build()

        # Read enriched metadata from original file
        ds = netCDF4.Dataset(str(nc_path), 'r')
        assert ds.Conventions == 'CF-1.8'
        assert 'test' in ds.domain_name

        v = ds.variables['air_temperature']
        assert v.standard_name == 'air_temperature'
        assert v.units == 'K'
        assert v.source_source == 'ERA5'
        ds.close()

    def test_metadata_idempotent(self, tmp_path):
        """Building twice should not duplicate attributes."""
        src = tmp_path / 'forcing' / 'basin_averaged_data'
        src.mkdir(parents=True)
        nc_path = src / 'forcing.nc'
        _create_dummy_nc(nc_path, var_name='air_temperature')

        builder = ForcingsStoreBuilder(
            project_dir=tmp_path,
            domain_name='test',
            forcing_dataset='ERA5',
        )
        builder.build()
        builder.build()  # Second build

        ds = netCDF4.Dataset(str(nc_path), 'r')
        assert ds.Conventions == 'CF-1.8'
        ds.close()

    def test_rebuild_replaces_symlinks(self, tmp_path):
        src = tmp_path / 'forcing' / 'basin_averaged_data'
        src.mkdir(parents=True)
        _create_dummy_nc(src / 'a.nc')

        builder = ForcingsStoreBuilder(
            project_dir=tmp_path, domain_name='t',
        )
        builder.build()
        builder.build()  # Should not fail on existing symlink

        link = tmp_path / 'data' / 'model_ready' / 'forcings' / 'a.nc'
        assert link.exists()
