"""
Unit tests for newly added observation handlers.

Tests import, registration, and basic functionality for:
- ERA5-Land
- MSWEP
- MODIS LST
- MODIS LAI
- GRDC
- OpenET
- Sentinel-1 SM
- Daymet
- VIIRS Snow
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Test marker
pytestmark = [pytest.mark.unit, pytest.mark.data]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration for testing handlers."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-10 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 3600,
        'BOUNDING_BOX_COORDS': '41/-106/39/-104',
    }


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_handlers")


# =============================================================================
# Import and Registration Tests
# =============================================================================

class TestHandlerImports:
    """Test that all new handlers can be imported."""

    def test_era5_land_import(self):
        """Test ERA5-Land handler imports."""
        from symfluence.data.acquisition.handlers.era5_land import ERA5LandAcquirer
        from symfluence.data.observation.handlers.era5_land import ERA5LandHandler
        assert ERA5LandHandler is not None
        assert ERA5LandAcquirer is not None

    def test_mswep_import(self):
        """Test MSWEP handler imports."""
        from symfluence.data.acquisition.handlers.mswep import MSWEPAcquirer
        from symfluence.data.observation.handlers.mswep import MSWEPHandler
        assert MSWEPHandler is not None
        assert MSWEPAcquirer is not None

    def test_modis_lst_import(self):
        """Test MODIS LST handler imports."""
        from symfluence.data.acquisition.handlers.modis_lst import MODISLSTAcquirer
        from symfluence.data.observation.handlers.modis_lst import MODISLSTHandler
        assert MODISLSTHandler is not None
        assert MODISLSTAcquirer is not None

    def test_modis_lai_import(self):
        """Test MODIS LAI handler imports."""
        from symfluence.data.acquisition.handlers.modis_lai import MODISLAIAcquirer
        from symfluence.data.observation.handlers.modis_lai import MODISLAIHandler
        assert MODISLAIHandler is not None
        assert MODISLAIAcquirer is not None

    def test_grdc_import(self):
        """Test GRDC handler imports."""
        from symfluence.data.acquisition.handlers.grdc import GRDCAcquirer
        from symfluence.data.observation.handlers.grdc import GRDCHandler
        assert GRDCHandler is not None
        assert GRDCAcquirer is not None

    def test_openet_import(self):
        """Test OpenET handler imports."""
        from symfluence.data.acquisition.handlers.openet import OpenETAcquirer
        from symfluence.data.observation.handlers.openet import OpenETHandler
        assert OpenETHandler is not None
        assert OpenETAcquirer is not None

    def test_sentinel1_sm_import(self):
        """Test Sentinel-1 SM handler imports."""
        from symfluence.data.acquisition.handlers.sentinel1_sm import Sentinel1SMAcquirer
        from symfluence.data.observation.handlers.sentinel1_sm import Sentinel1SMHandler
        assert Sentinel1SMHandler is not None
        assert Sentinel1SMAcquirer is not None

    def test_daymet_import(self):
        """Test Daymet handler imports."""
        from symfluence.data.acquisition.handlers.daymet import DaymetAcquirer
        from symfluence.data.observation.handlers.daymet import DaymetHandler
        assert DaymetHandler is not None
        assert DaymetAcquirer is not None

    def test_viirs_snow_import(self):
        """Test VIIRS Snow handler imports."""
        from symfluence.data.acquisition.handlers.viirs_snow import VIIRSSnowAcquirer
        from symfluence.data.observation.handlers.viirs_snow import VIIRSSnowHandler
        assert VIIRSSnowHandler is not None
        assert VIIRSSnowAcquirer is not None


class TestHandlerRegistration:
    """Test that all new handlers are properly registered."""

    def test_observation_registry_new_handlers(self):
        """Test new handlers are registered in ObservationRegistry."""
        # Ensure handlers are imported (triggers registration)
        import symfluence.data.observation.handlers
        from symfluence.data.observation.registry import ObservationRegistry

        # Check each new handler is registered
        new_handlers = [
            'era5_land', 'era5land',
            'mswep',
            'modis_lst', 'mod11',
            'modis_lai', 'mcd15', 'lai',
            'grdc',
            'openet',
            'sentinel1_sm', 's1_sm',
            'daymet',
            'viirs_snow', 'vnp10',
        ]

        for handler_name in new_handlers:
            assert ObservationRegistry.is_registered(handler_name), \
                f"Handler '{handler_name}' not registered in ObservationRegistry"

    def test_acquisition_registry_new_handlers(self):
        """Test new handlers are registered in AcquisitionRegistry."""
        # Ensure handlers are imported
        import symfluence.data.acquisition.handlers
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        # Check each new acquisition handler is registered
        new_handlers = [
            'era5_land', 'era5-land',
            'mswep',
            'modis_lst', 'mod11',
            'modis_lai', 'mcd15',
            'grdc',
            'openet',
            'sentinel1_sm', 's1_sm',
            'daymet',
            'viirs_snow', 'vnp10',
        ]

        for handler_name in new_handlers:
            assert AcquisitionRegistry.is_registered(handler_name), \
                f"Handler '{handler_name}' not registered in AcquisitionRegistry"


# =============================================================================
# Handler Instantiation Tests
# =============================================================================

class TestHandlerInstantiation:
    """Test that handlers can be instantiated with mock config."""

    def test_era5_land_handler_init(self, mock_config, logger):
        """Test ERA5-Land handler initialization."""
        from symfluence.data.observation.handlers.era5_land import ERA5LandHandler
        handler = ERA5LandHandler(mock_config, logger)
        assert handler.obs_type == "reanalysis"
        assert handler.source_name == "ECMWF_ERA5_LAND"

    def test_mswep_handler_init(self, mock_config, logger):
        """Test MSWEP handler initialization."""
        from symfluence.data.observation.handlers.mswep import MSWEPHandler
        handler = MSWEPHandler(mock_config, logger)
        assert handler.obs_type == "precipitation"
        assert handler.source_name == "GLOH2O_MSWEP"

    def test_modis_lst_handler_init(self, mock_config, logger):
        """Test MODIS LST handler initialization."""
        from symfluence.data.observation.handlers.modis_lst import MODISLSTHandler
        handler = MODISLSTHandler(mock_config, logger)
        assert handler.obs_type == "lst"
        assert handler.source_name == "NASA_MODIS"

    def test_modis_lai_handler_init(self, mock_config, logger):
        """Test MODIS LAI handler initialization."""
        from symfluence.data.observation.handlers.modis_lai import MODISLAIHandler
        handler = MODISLAIHandler(mock_config, logger)
        assert handler.obs_type == "lai"
        assert handler.source_name == "NASA_MODIS"

    def test_grdc_handler_init(self, mock_config, logger):
        """Test GRDC handler initialization."""
        from symfluence.data.observation.handlers.grdc import GRDCHandler
        handler = GRDCHandler(mock_config, logger)
        assert handler.obs_type == "streamflow"
        assert handler.source_name == "GRDC"

    def test_openet_handler_init(self, mock_config, logger):
        """Test OpenET handler initialization."""
        from symfluence.data.observation.handlers.openet import OpenETHandler
        handler = OpenETHandler(mock_config, logger)
        assert handler.obs_type == "et"
        assert handler.source_name == "OpenET"

    def test_sentinel1_sm_handler_init(self, mock_config, logger):
        """Test Sentinel-1 SM handler initialization."""
        from symfluence.data.observation.handlers.sentinel1_sm import Sentinel1SMHandler
        handler = Sentinel1SMHandler(mock_config, logger)
        assert handler.obs_type == "soil_moisture"
        assert handler.source_name == "Sentinel-1"

    def test_daymet_handler_init(self, mock_config, logger):
        """Test Daymet handler initialization."""
        from symfluence.data.observation.handlers.daymet import DaymetHandler
        handler = DaymetHandler(mock_config, logger)
        assert handler.obs_type == "climate"
        assert handler.source_name == "ORNL_Daymet"

    def test_viirs_snow_handler_init(self, mock_config, logger):
        """Test VIIRS Snow handler initialization."""
        from symfluence.data.observation.handlers.viirs_snow import VIIRSSnowHandler
        handler = VIIRSSnowHandler(mock_config, logger)
        assert handler.obs_type == "snow_cover"
        assert handler.source_name == "NASA_VIIRS"

    def test_modis_et_default_dir_resolution(self, mock_config, logger):
        """MOD16_ET_DIR='default' should resolve into the project observations tree."""
        from symfluence.data.observation.handlers.modis_et import MODISETHandler

        handler = MODISETHandler(mock_config, logger)
        resolved = handler._resolve_mod16_dir()

        expected = (
            Path(mock_config['SYMFLUENCE_DATA_DIR'])
            / "domain_test_domain"
            / "data"
            / "observations"
            / "et"
            / "modis"
        )
        assert resolved == expected
        assert resolved != Path("default")

    def test_modis_et_custom_dir_resolution(self, mock_config, logger, tmp_path):
        """Custom MOD16_ET_DIR should be preserved verbatim."""
        from symfluence.data.observation.handlers.modis_et import MODISETHandler

        custom_dir = tmp_path / "custom_mod16"
        mock_config['MOD16_ET_DIR'] = str(custom_dir)
        handler = MODISETHandler(mock_config, logger)

        assert handler._resolve_mod16_dir() == custom_dir


# =============================================================================
# Processing Tests with Mock Data
# =============================================================================

class TestERA5LandProcessing:
    """Test ERA5-Land data processing."""

    def test_process_netcdf(self, mock_config, logger, tmp_path):
        """Test processing ERA5-Land NetCDF data."""
        from symfluence.data.observation.handlers.era5_land import ERA5LandHandler

        # Create mock NetCDF
        era5_dir = tmp_path / "domain_test_domain" / "observations" / "era5_land"
        era5_dir.mkdir(parents=True)
        mock_file = era5_dir / "era5_land_20200101_20200110_daily.nc"

        times = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        ds = xr.Dataset(
            data_vars={
                'tp': (('time', 'lat', 'lon'), np.random.rand(len(times), 3, 3) * 0.01),
                't2m': (('time', 'lat', 'lon'), np.random.rand(len(times), 3, 3) * 10 + 270),
            },
            coords={
                'time': times,
                'lat': [40, 40.5, 41],
                'lon': [-106, -105.5, -105],
            }
        )
        ds.to_netcdf(mock_file, engine='scipy')

        mock_config['ERA5_LAND_DIR'] = str(era5_dir)
        handler = ERA5LandHandler(mock_config, logger)

        result = handler.process(era5_dir)
        assert result.exists()


class TestMSWEPProcessing:
    """Test MSWEP data processing."""

    def test_process_netcdf(self, mock_config, logger, tmp_path):
        """Test processing MSWEP NetCDF data."""
        from symfluence.data.observation.handlers.mswep import MSWEPHandler

        # Create mock NetCDF
        mswep_dir = tmp_path / "domain_test_domain" / "observations" / "precipitation" / "mswep"
        mswep_dir.mkdir(parents=True)
        mock_file = mswep_dir / "mswep_2020001.nc"

        times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        ds = xr.Dataset(
            data_vars={
                'precipitation': (('time', 'lat', 'lon'), np.random.rand(len(times), 3, 3) * 10),
            },
            coords={
                'time': times,
                'lat': [40, 40.5, 41],
                'lon': [-106, -105.5, -105],
            }
        )
        ds.to_netcdf(mock_file, engine='scipy')

        mock_config['MSWEP_DIR'] = str(mswep_dir)
        handler = MSWEPHandler(mock_config, logger)

        result = handler.process(mswep_dir)
        assert result.exists()


class TestMODISLSTProcessing:
    """Test MODIS LST data processing."""

    def test_process_netcdf(self, mock_config, logger, tmp_path):
        """Test processing MODIS LST NetCDF data."""
        from symfluence.data.observation.handlers.modis_lst import MODISLSTHandler

        # Create mock NetCDF
        lst_dir = tmp_path / "domain_test_domain" / "observations" / "temperature" / "modis_lst"
        lst_dir.mkdir(parents=True)
        mock_file = lst_dir / "MOD11A1_LST_test.nc"

        times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        # LST values in scaled DN (will be multiplied by 0.02 to get K)
        ds = xr.Dataset(
            data_vars={
                'LST_Day_1km': (('time', 'lat', 'lon'), np.random.randint(13000, 16000, (len(times), 3, 3)).astype(np.float64)),
                'LST_Night_1km': (('time', 'lat', 'lon'), np.random.randint(12000, 15000, (len(times), 3, 3)).astype(np.float64)),
            },
            coords={
                'time': times,
                'lat': [40, 40.5, 41],
                'lon': [-106, -105.5, -105],
            }
        )
        ds.to_netcdf(mock_file, engine='scipy')

        mock_config['MODIS_LST_DIR'] = str(lst_dir)
        handler = MODISLSTHandler(mock_config, logger)

        result = handler.process(lst_dir)
        assert result.exists()


class TestMODISLAIProcessing:
    """Test MODIS LAI data processing."""

    def test_process_netcdf(self, mock_config, logger, tmp_path):
        """Test processing MODIS LAI NetCDF data."""
        from symfluence.data.observation.handlers.modis_lai import MODISLAIHandler

        # Create mock NetCDF
        lai_dir = tmp_path / "domain_test_domain" / "observations" / "vegetation" / "modis_lai"
        lai_dir.mkdir(parents=True)
        mock_file = lai_dir / "MCD15A2H_LAI_test.nc"

        # 8-day composite
        times = pd.date_range('2020-01-01', '2020-01-25', freq='8D')
        # LAI values in scaled DN (0-100, will be multiplied by 0.1)
        ds = xr.Dataset(
            data_vars={
                'Lai_500m': (('time', 'lat', 'lon'), np.random.randint(0, 70, (len(times), 3, 3)).astype(np.float64)),
                'Fpar_500m': (('time', 'lat', 'lon'), np.random.randint(0, 100, (len(times), 3, 3)).astype(np.float64)),
            },
            coords={
                'time': times,
                'lat': [40, 40.5, 41],
                'lon': [-106, -105.5, -105],
            }
        )
        ds.to_netcdf(mock_file, engine='scipy')

        mock_config['MODIS_LAI_DIR'] = str(lai_dir)
        handler = MODISLAIHandler(mock_config, logger)

        result = handler.process(lai_dir)
        assert result.exists()


class TestGRDCProcessing:
    """Test GRDC data processing."""

    def test_process_csv(self, mock_config, logger, tmp_path):
        """Test processing GRDC CSV data."""
        from symfluence.data.observation.handlers.grdc import GRDCHandler

        # Create mock CSV
        grdc_dir = tmp_path / "domain_test_domain" / "observations" / "streamflow" / "grdc"
        grdc_dir.mkdir(parents=True)
        mock_file = grdc_dir / "grdc_1234567_daily.csv"

        # GRDC format CSV
        data = """YYYY-MM-DD,Value
2020-01-01,100.5
2020-01-02,105.2
2020-01-03,98.7
2020-01-04,102.3
2020-01-05,110.0
"""
        mock_file.write_text(data)

        mock_config['GRDC_DATA_DIR'] = str(grdc_dir)
        mock_config['GRDC_STATION_IDS'] = '1234567'
        handler = GRDCHandler(mock_config, logger)

        result = handler.process(grdc_dir)
        assert result.exists()


class TestOpenETProcessing:
    """Test OpenET data processing."""

    def test_process_csv(self, mock_config, logger, tmp_path):
        """Test processing OpenET CSV data."""
        from symfluence.data.observation.handlers.openet import OpenETHandler

        # Create mock CSV
        openet_dir = tmp_path / "domain_test_domain" / "observations" / "et" / "openet"
        openet_dir.mkdir(parents=True)
        mock_file = openet_dir / "openet_ensemble_20200101_20200110_monthly.csv"

        data = """date,et_mm
2020-01-01,50.2
2020-01-02,48.5
2020-01-03,52.1
2020-01-04,49.8
2020-01-05,51.3
"""
        mock_file.write_text(data)

        mock_config['OPENET_DIR'] = str(openet_dir)
        handler = OpenETHandler(mock_config, logger)

        result = handler.process(openet_dir)
        assert result.exists()


class TestDaymetProcessing:
    """Test Daymet data processing."""

    def test_process_netcdf(self, mock_config, logger, tmp_path):
        """Test processing Daymet NetCDF data."""
        from symfluence.data.observation.handlers.daymet import DaymetHandler

        # Create mock NetCDF
        daymet_dir = tmp_path / "domain_test_domain" / "observations" / "climate" / "daymet"
        daymet_dir.mkdir(parents=True)
        mock_file = daymet_dir / "daymet_tmax_2020.nc"

        times = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        ds = xr.Dataset(
            data_vars={
                'tmax': (('time', 'lat', 'lon'), np.random.rand(len(times), 3, 3) * 10 + 5),
                'tmin': (('time', 'lat', 'lon'), np.random.rand(len(times), 3, 3) * 10 - 5),
                'prcp': (('time', 'lat', 'lon'), np.random.rand(len(times), 3, 3) * 10),
            },
            coords={
                'time': times,
                'lat': [40, 40.5, 41],
                'lon': [-106, -105.5, -105],
            }
        )
        ds.to_netcdf(mock_file, engine='scipy')

        mock_config['DAYMET_DIR'] = str(daymet_dir)
        handler = DaymetHandler(mock_config, logger)

        result = handler.process(daymet_dir)
        assert result.exists()


class TestVIIRSSnowProcessing:
    """Test VIIRS Snow data processing."""

    def test_process_netcdf(self, mock_config, logger, tmp_path):
        """Test processing VIIRS Snow NetCDF data."""
        from symfluence.data.observation.handlers.viirs_snow import VIIRSSnowHandler

        # Create mock NetCDF
        viirs_dir = tmp_path / "domain_test_domain" / "observations" / "snow" / "viirs"
        viirs_dir.mkdir(parents=True)
        mock_file = viirs_dir / "VNP10A1F_Snow_test.nc"

        times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
        # NDSI Snow Cover values (0-100)
        ds = xr.Dataset(
            data_vars={
                'CGF_NDSI_Snow_Cover': (('time', 'lat', 'lon'), np.random.randint(0, 100, (len(times), 3, 3)).astype(np.float64)),
            },
            coords={
                'time': times,
                'lat': [40, 40.5, 41],
                'lon': [-106, -105.5, -105],
            }
        )
        ds.to_netcdf(mock_file, engine='scipy')

        mock_config['VIIRS_SNOW_DIR'] = str(viirs_dir)
        handler = VIIRSSnowHandler(mock_config, logger)

        result = handler.process(viirs_dir)
        assert result.exists()


# =============================================================================
# Comprehensive Module Import Test
# =============================================================================

def test_all_handlers_in_init():
    """Test all handlers are exported from __init__.py."""
    from symfluence.data.observation.handlers import (
        DaymetHandler,
        ERA5LandHandler,
        GRDCHandler,
        MODISLAIHandler,
        MODISLSTHandler,
        MSWEPHandler,
        OpenETHandler,
        Sentinel1SMHandler,
        VIIRSSnowHandler,
    )

    # All imports should succeed
    assert ERA5LandHandler is not None
    assert MSWEPHandler is not None
    assert MODISLSTHandler is not None
    assert MODISLAIHandler is not None
    assert GRDCHandler is not None
    assert OpenETHandler is not None
    assert Sentinel1SMHandler is not None
    assert DaymetHandler is not None
    assert VIIRSSnowHandler is not None
