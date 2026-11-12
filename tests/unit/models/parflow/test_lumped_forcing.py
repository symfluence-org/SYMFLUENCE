# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Lumped ParFlow / CLM-ParFlow forcing uses the domain's own elevation bands and latitude."""
from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import ConfigurationError, ShapefileError
from symfluence.models.clmparflow.preprocessor import CLMParFlowPreProcessor
from symfluence.models.parflow import lumped_forcing
from symfluence.models.parflow.lumped_forcing import (
    LumpedForcingMixin,
    SnowBands,
    bands_from_elevations,
    load_cached_snow_setup,
    resolve_snow_bands,
    update_snow17_boundary_values,
)
from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

LOG = logging.getLogger('test_parflow_lumped')


def _config(tmp_path, model='PARFLOW'):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-01-06 00:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": model,
        "FORCING_DATASET": "ERA5",
    })


def _write_dem_and_catchment(tmp_path):
    """10x10 DEM, 100..199 m west half inside the catchment, 5000 m outside."""
    dem = np.full((10, 10), 5000.0, dtype='float32')
    dem[:, :5] = np.arange(100, 150, dtype='float32').reshape(10, 5)
    dem_path = tmp_path / 'dem_elv.tif'
    with rasterio.open(dem_path, 'w', driver='GTiff', height=10, width=10, count=1, dtype='float32',
                       crs='EPSG:4326', transform=from_origin(-20.0, 65.0, 0.01, 0.01)) as dst:
        dst.write(dem, 1)
    shp = tmp_path / 'catchment.shp'
    gpd.GeoDataFrame({'GRU_ID': [1]}, geometry=[box(-20.0, 64.9, -19.95, 65.0)], crs='EPSG:4326').to_file(shp)
    return dem_path, shp, dem[:, :5]


def test_bands_come_from_the_domain_dem(tmp_path):
    """Regression: bands 1521..2614 m around 2099 m (Bow at Banff) were applied to every domain."""
    dem_path, shp, inside = _write_dem_and_catchment(tmp_path)
    bands = resolve_snow_bands(dem_path, shp, None, LOG)
    assert bands.elevations == pytest.approx(np.percentile(inside, [10, 30, 50, 70, 90]))
    assert bands.fractions == pytest.approx([0.2] * 5)
    assert bands.reference_elev == pytest.approx(inside.mean())
    assert max(bands.elevations) < 200  # nothing from outside the outline, nothing from Banff
    offsets = bands.temperature_offsets(0.0065)
    assert offsets[0] > 0 > offsets[-1]


def test_single_band_without_elevation_distribution(tmp_path):
    bands = resolve_snow_bands(None, tmp_path / 'missing.shp', 640.0, LOG)
    assert bands.elevations == (640.0,) and bands.fractions == (1.0,)
    assert bands.temperature_offsets(0.0065) == [0.0]
    # Unknown mean elevation still means "no lapse adjustment", not an invented elevation.
    assert SnowBands.single(None, 'x').temperature_offsets(0.0065) == [0.0]


def test_bands_need_valid_elevations():
    with pytest.raises(ConfigurationError):
        bands_from_elevations(np.array([]), 'empty')


def test_no_basin_constants_left_on_the_preprocessors():
    for cls in (ParFlowPreProcessor, CLMParFlowPreProcessor):
        assert issubclass(cls, LumpedForcingMixin)
        assert not hasattr(cls, 'ELEVATION_BANDS') and not hasattr(cls, 'BASIN_MEAN_ELEV')
        # one implementation, not a copy per model
        assert cls._apply_snow_model is LumpedForcingMixin._apply_snow_model
        assert cls._prepare_daily_rainfall is LumpedForcingMixin._prepare_daily_rainfall
        assert cls._get_time_info is LumpedForcingMixin._get_time_info


def test_parflow_latitude_unknown_raises(tmp_path):
    """Regression: fell back to 51.36 (Banff) when nothing gave a latitude."""
    pp = ParFlowPreProcessor(_config(tmp_path), Mock())
    with pytest.raises(ShapefileError, match='latitude'):
        pp._get_latitude()


def test_parflow_latitude_from_forcing(tmp_path):
    from symfluence.core.mixins.project import resolve_forcing_basin_path
    pp = ParFlowPreProcessor(_config(tmp_path), Mock())
    fdir = resolve_forcing_basin_path(pp.project_dir)
    fdir.mkdir(parents=True)
    xr.Dataset({'latitude': (('hru',), [64.2]), 'longitude': (('hru',), [-19.1])}).to_netcdf(
        fdir / 'ERA5_remapped.nc')
    assert pp._get_latitude() == pytest.approx(64.2)


def _write_forcing(pp, n=120):
    from symfluence.core.mixins.project import resolve_forcing_basin_path
    fdir = resolve_forcing_basin_path(pp.project_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-01", periods=n, freq='h')
    xr.Dataset(
        {"precipitation_flux": (("time", "hru"), np.full((n, 1), 1e-4)),
         "air_temperature": (("time", "hru"), np.full((n, 1), 274.15))},
        coords={"time": times, "hru": [0]},
    ).to_netcdf(fdir / "ERA5_remapped.nc")


@pytest.mark.parametrize('cls,model', [(ParFlowPreProcessor, 'PARFLOW'), (CLMParFlowPreProcessor, 'CLMPARFLOW')])
def test_cache_records_bands_and_calibration_reuses_them(tmp_path, cls, model):
    pp = cls(_config(tmp_path, model), Mock())
    _write_forcing(pp)
    out = tmp_path / 'out'
    out.mkdir()
    bands = SnowBands((100.0, 900.0), (0.5, 0.5), 500.0, 'test')
    with patch.object(pp, '_get_latitude', return_value=64.0), patch.object(pp, '_get_snow_bands', return_value=bands):
        days = pp._prepare_daily_rainfall(out)
    assert days is not None and len(days) == 5
    cache_path = out / 'hourly_forcing_cache.npz'
    cached_bands, latitude = load_cached_snow_setup(np.load(cache_path), cache_path)
    assert cached_bands.elevations == (100.0, 900.0) and cached_bands.reference_elev == 500.0
    assert latitude == 64.0
    if cls is CLMParFlowPreProcessor:
        assert (out / 'forcing.1d').exists()

    # Calibration re-runs Snow-17 with the cached geometry: with default
    # Snow-17 parameters it reproduces the preprocessor's daily values.
    seen = {}
    real = lumped_forcing.band_rain_plus_melt

    def spy(ppt, temp, doy, used_bands, used_lat, *args, **kwargs):
        seen['bands'], seen['lat'] = used_bands, used_lat
        return real(ppt, temp, doy, used_bands, used_lat, *args, **kwargs)

    entries = {f'Patch.z_upper.BCPressure.{label}.Value': 'x' for label, _, _ in days}
    with patch.object(lumped_forcing, 'band_rain_plus_melt', spy):
        update_snow17_boundary_values(entries, {}, cache_path)
    assert seen['bands'].elevations == (100.0, 900.0) and seen['lat'] == 64.0
    for label, _, net_m_hr in days:
        assert float(entries[f'Patch.z_upper.BCPressure.{label}.Value']) == pytest.approx(-net_m_hr, rel=1e-5, abs=1e-12)


def test_cache_without_bands_is_rejected(tmp_path):
    """A cache from the old preprocessor embeds another basin's bands."""
    cache_path = tmp_path / 'hourly_forcing_cache.npz'
    times = pd.date_range('2005-01-01', periods=24, freq='h').values
    np.savez_compressed(cache_path, ppt_mm_hr=np.zeros(24), pet_mm_hr=np.zeros(24), temp_c=np.zeros(24),
                        times=times.astype('int64'))
    with pytest.raises(ConfigurationError, match='Re-run model preprocessing'):
        update_snow17_boundary_values({}, {'SNOW17_SCF': 1.0}, cache_path)


def test_preprocessor_resolves_bands_from_project_dem(tmp_path):
    pp = ParFlowPreProcessor(_config(tmp_path), Mock())
    dem_path, shp, inside = _write_dem_and_catchment(tmp_path)
    with patch.object(pp, '_dem_path', return_value=dem_path), \
            patch.object(pp, 'get_catchment_path', return_value=shp), \
            patch.object(pp, 'get_catchment_location', return_value={'elev': None}):
        bands = pp._get_snow_bands()
    assert bands.reference_elev == pytest.approx(inside.mean())


def test_lumped_readers_select_this_runs_forcing(tmp_path):
    """Issue #339: only the forcing remapped onto this run's HRUs is read."""
    from symfluence.models.parflow.lumped_forcing import location_from_forcing, prepare_daily_rainfall

    def write(name, lat, temp):
        times = pd.date_range("2005-01-01", periods=48, freq='h')
        xr.Dataset(
            {"precipitation_flux": (("time", "hru"), np.full((48, 1), 1e-4)),
             "air_temperature": (("time", "hru"), np.full((48, 1), temp)),
             "latitude": (("hru",), [lat]), "longitude": (("hru",), [-19.0])},
            coords={"time": times, "hru": [0]},
        ).to_netcdf(tmp_path / name)

    write('a_ERA5_remapped_elevation.nc', 10.0, 250.0)   # another HRU set, sorts first
    write('b_ERA5_remapped_lumped.nc', 64.0, 283.15)
    selected = []

    def selector(files):
        selected.append(list(files))
        return [f for f in files if 'lumped' in f.name]

    assert location_from_forcing(tmp_path, selector) == (64.0, -19.0)
    assert location_from_forcing(tmp_path)[0] == 10.0
    out = tmp_path / 'out'
    out.mkdir()
    days = prepare_daily_rainfall(
        forcing_dir=tmp_path, output_dir=out, start='2005-01-01', end='2005-01-03',
        latitude=lambda: 64.0, snow_bands=lambda: SnowBands.single(500.0, 'test'),
        logger=LOG, model_label='ParFlow', forcing_selector=selector)
    assert days is not None and len(days) == 2
    cache = np.load(out / 'hourly_forcing_cache.npz')
    assert cache['temp_c'] == pytest.approx(10.0)
    assert len(selected) == 2 and all(len(files) == 2 for files in selected)


def test_preprocessors_pass_their_run_forcing_selector(tmp_path):
    for cls, model in ((ParFlowPreProcessor, 'PARFLOW'), (CLMParFlowPreProcessor, 'CLMPARFLOW')):
        pp = cls(_config(tmp_path, model), Mock())
        with patch.object(lumped_forcing, 'prepare_daily_rainfall', return_value=None) as prep, \
                patch.object(pp, '_get_forcing_dir', return_value=tmp_path):
            pp._prepare_daily_rainfall(tmp_path)
        assert prep.call_args.kwargs['forcing_selector'] == pp._select_run_forcing


@pytest.mark.parametrize('lat', [78.2, -75.0, 66.0, 45.0])
def test_oudin_is_finite_inside_the_polar_circles(lat):
    """Polar day and night used to give NaN (unclipped arccos argument)."""
    from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
    from symfluence.models.parflow.lumped_forcing import oudin_pet_mm_hr

    dates = pd.date_range('2021-01-01', '2021-12-31', freq='D')
    temp_k = np.full(len(dates), 283.15)
    pet = oudin_pet_mm_hr(temp_k, dates, lat)
    assert np.all(np.isfinite(pet)) and np.all(pet >= 0)
    np.testing.assert_allclose(
        pet * 24.0,
        PETCalculatorMixin.oudin_pet_numpy(temp_k - 273.15, np.asarray(dates.dayofyear), lat))
    if lat > 70:
        assert pet[0] == 0.0 and pet[180] > 0.0  # polar night, polar day
