"""
Tests for the ARCO-ERA5 acquisition and preprocessing pipeline.

Verifies that the full flow — from raw ARCO-style ERA5 data through
era5_to_summa_schema() and ERA5Handler.process_dataset() — produces
correct output variables, units, coordinate conventions, and value ranges.
"""

import logging

import numpy as np
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.era5_processing import (
    ARCO_VARIABLE_NAMES,
    era5_to_summa_schema,
)
from symfluence.data.preprocessing.dataset_handlers.era5_utils import ERA5Handler

EXPECTED_VARS = {'airtemp', 'airpres', 'windspd', 'spechum', 'pptrate', 'SWRadAtm', 'LWRadAtm'}


def _make_arco_era5_dataset(n_time=25, lats=None, lons=None):
    """
    Create a synthetic dataset mimicking raw ARCO-ERA5 Zarr output.

    Uses 0-360 longitude convention (ARCO native) and physically
    plausible values for all variables.
    """
    if lats is None:
        lats = np.array([51.5, 51.25, 51.0])  # descending, like ERA5
    if lons is None:
        lons = np.array([243.5, 243.75, 244.0, 244.25])  # 0-360 convention

    times = np.arange(
        np.datetime64('2002-09-01T00:00'),
        np.datetime64('2002-09-01T00:00') + np.timedelta64(n_time, 'h'),
        np.timedelta64(1, 'h'),
    )
    shape = (len(times), len(lats), len(lons))
    rng = np.random.default_rng(42)

    # Instantaneous variables — physically plausible ranges
    temp = rng.uniform(270, 300, shape).astype('float32')         # K
    dewpoint = temp - rng.uniform(2, 10, shape).astype('float32') # K, always below temp
    pressure = rng.uniform(85000, 102000, shape).astype('float32')  # Pa
    u_wind = rng.uniform(-10, 10, shape).astype('float32')        # m/s
    v_wind = rng.uniform(-10, 10, shape).astype('float32')        # m/s

    # Accumulated variables — cumulative within each day, reset daily
    # Precipitation: accumulated in meters (small values)
    precip_hourly = rng.uniform(0, 0.001, shape).astype('float32')
    precip = np.cumsum(precip_hourly, axis=0).astype('float32')

    # Radiation: accumulated in J/m² (positive, increasing)
    sw_hourly = rng.uniform(0, 500 * 3600, shape).astype('float32')
    sw_rad = np.cumsum(sw_hourly, axis=0).astype('float32')

    lw_hourly = rng.uniform(200 * 3600, 400 * 3600, shape).astype('float32')
    lw_rad = np.cumsum(lw_hourly, axis=0).astype('float32')

    ds = xr.Dataset(
        {
            ARCO_VARIABLE_NAMES['temperature']: (['time', 'latitude', 'longitude'], temp),
            ARCO_VARIABLE_NAMES['dewpoint']: (['time', 'latitude', 'longitude'], dewpoint),
            ARCO_VARIABLE_NAMES['pressure']: (['time', 'latitude', 'longitude'], pressure),
            ARCO_VARIABLE_NAMES['wind_u']: (['time', 'latitude', 'longitude'], u_wind),
            ARCO_VARIABLE_NAMES['wind_v']: (['time', 'latitude', 'longitude'], v_wind),
            ARCO_VARIABLE_NAMES['precipitation']: (['time', 'latitude', 'longitude'], precip),
            ARCO_VARIABLE_NAMES['sw_radiation']: (['time', 'latitude', 'longitude'], sw_rad),
            ARCO_VARIABLE_NAMES['lw_radiation']: (['time', 'latitude', 'longitude'], lw_rad),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        },
    )
    return ds


class TestEra5ToSummaSchema:
    """Tests for the ARCO → SUMMA variable conversion."""

    def test_produces_all_expected_variables(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert set(result.data_vars) == EXPECTED_VARS

    def test_preserves_spatial_coordinates(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert 'latitude' in result.coords
        assert 'longitude' in result.coords
        assert result.sizes['latitude'] == 3
        assert result.sizes['longitude'] == 4

    def test_time_dimension_sliced_for_arco(self):
        """ARCO slices off the first timestep due to de-accumulation."""
        n_time = 25
        ds = _make_arco_era5_dataset(n_time=n_time)
        result = era5_to_summa_schema(ds, source='arco')
        assert result.sizes['time'] == n_time - 1

    def test_temperature_in_kelvin(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert result['airtemp'].min() > 200  # sanity: Kelvin
        assert result['airtemp'].max() < 350

    def test_pressure_in_pascals(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert result['airpres'].min() > 50000
        assert result['airpres'].max() < 110000

    def test_wind_speed_non_negative(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert float(result['windspd'].min()) >= 0

    def test_specific_humidity_in_range(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert float(result['spechum'].min()) >= 0
        assert float(result['spechum'].max()) < 0.05  # max ~40 g/kg

    def test_precipitation_rate_non_negative(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert float(result['pptrate'].min()) >= 0

    def test_radiation_non_negative(self):
        ds = _make_arco_era5_dataset()
        result = era5_to_summa_schema(ds, source='arco')
        assert float(result['SWRadAtm'].min()) >= 0
        assert float(result['LWRadAtm'].min()) >= 0

    def test_retains_arco_longitude_convention(self):
        """era5_to_summa_schema does NOT convert longitude — that's process_dataset's job."""
        lons = np.array([243.5, 243.75, 244.0])
        ds = _make_arco_era5_dataset(lons=lons)
        result = era5_to_summa_schema(ds, source='arco')
        assert float(result.longitude.max()) > 180


class TestERA5HandlerLongitudeNormalization:
    """Tests that process_dataset converts 0-360 → -180/+180."""

    @pytest.fixture()
    def handler(self):
        return ERA5Handler(
            config={'DOMAIN_NAME': 'test', 'FORCING_DATASET': 'ERA5'},
            logger=logging.getLogger('test'),
            project_dir='/tmp',
        )

    def test_converts_0_360_to_negative_180(self, handler):
        """Longitudes >180 should be converted to negative values."""
        ds = xr.Dataset(
            {'airtemp': (['time', 'latitude', 'longitude'], np.ones((2, 3, 4)))},
            coords={
                'time': [0, 1],
                'latitude': [51.5, 51.25, 51.0],
                'longitude': [243.5, 243.75, 244.0, 244.25],
            },
        )
        result = handler.process_dataset(ds)
        assert float(result.longitude.max()) <= 180
        assert float(result.longitude.min()) >= -180
        np.testing.assert_allclose(
            result.longitude.values,
            [-116.5, -116.25, -116.0, -115.75],
        )

    def test_preserves_already_negative_longitudes(self, handler):
        """Longitudes already in -180/+180 should be unchanged."""
        lons = np.array([-116.5, -116.25, -116.0, -115.75])
        ds = xr.Dataset(
            {'airtemp': (['time', 'latitude', 'longitude'], np.ones((2, 3, 4)))},
            coords={
                'time': [0, 1],
                'latitude': [51.5, 51.25, 51.0],
                'longitude': lons,
            },
        )
        result = handler.process_dataset(ds)
        np.testing.assert_array_equal(result.longitude.values, lons)

    def test_sorts_longitude_after_conversion(self, handler):
        """After 0-360 → -180/+180 conversion, longitudes must be ascending."""
        ds = xr.Dataset(
            {'airtemp': (['time', 'latitude', 'longitude'], np.ones((2, 2, 5)))},
            coords={
                'time': [0, 1],
                'latitude': [51.0, 50.0],
                'longitude': [243.0, 243.5, 244.0, 244.5, 245.0],
            },
        )
        result = handler.process_dataset(ds)
        lons = result.longitude.values
        assert all(lons[i] < lons[i + 1] for i in range(len(lons) - 1))

    def test_data_values_follow_coordinate_reorder(self, handler):
        """Ensure data values stay aligned with their coordinates after sort."""
        # Each longitude gets a unique value so we can track reordering
        data = np.array([[[[1, 2, 3]], [[1, 2, 3]]]])  # (1, 2, 1, 3) -> time, lat, lat, lon
        ds = xr.Dataset(
            {'airtemp': (['time', 'latitude', 'longitude'],
                         np.array([[[10, 20, 30], [40, 50, 60]]]))},
            coords={
                'time': [0],
                'latitude': [51.0, 50.0],
                'longitude': [350.0, 10.0, 20.0],  # wraps: 350 → -10
            },
        )
        result = handler.process_dataset(ds)
        # After conversion: -10, 10, 20 — the 350→-10 column should come first
        assert float(result.longitude[0]) == -10.0
        assert int(result['airtemp'].sel(time=0, latitude=51.0, longitude=-10.0)) == 10
        assert int(result['airtemp'].sel(time=0, latitude=51.0, longitude=10.0)) == 20
        assert int(result['airtemp'].sel(time=0, latitude=51.0, longitude=20.0)) == 30


class TestFullARCOPipeline:
    """End-to-end: ARCO acquisition output → era5_to_summa_schema → process_dataset."""

    @pytest.fixture()
    def handler(self):
        return ERA5Handler(
            config={'DOMAIN_NAME': 'test', 'FORCING_DATASET': 'ERA5'},
            logger=logging.getLogger('test'),
            project_dir='/tmp',
        )

    def test_full_pipeline_produces_negative_longitudes(self, handler):
        """Simulates what happens when ARCO data flows through the full pipeline."""
        # Step 1: Raw ARCO data with 0-360 longitudes
        ds_raw = _make_arco_era5_dataset()
        assert float(ds_raw.longitude.max()) > 180

        # Step 2: era5_to_summa_schema (acquisition-time conversion)
        ds_converted = era5_to_summa_schema(ds_raw, source='arco')
        assert set(ds_converted.data_vars) == EXPECTED_VARS

        # Step 3: process_dataset (preprocessing-time standardization)
        ds_final = handler.process_dataset(ds_converted)

        # Longitudes must be in -180/+180
        assert float(ds_final.longitude.max()) <= 180
        assert float(ds_final.longitude.min()) >= -180

        # Expected conversion: 243.5→-116.5, 243.75→-116.25, etc.
        expected_lons = np.array([-116.5, -116.25, -116.0, -115.75])
        np.testing.assert_allclose(ds_final.longitude.values, expected_lons)

        # All variables still present and valid
        assert set(ds_final.data_vars) == EXPECTED_VARS
        for var in ds_final.data_vars:
            assert not np.any(np.isnan(ds_final[var].values)), f"NaN found in {var}"
