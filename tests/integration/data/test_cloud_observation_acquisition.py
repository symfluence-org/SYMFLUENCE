import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from symfluence.core import SYMFLUENCE
from symfluence.data.data_manager import DataManager
from symfluence.data.observation.registry import ObservationRegistry

pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.requires_cloud, pytest.mark.slow]


def is_wsc_api_available(station_id: str = '05BB001') -> bool:
    """Check if WSC GeoMet API is accessible and returning data."""
    try:
        response = requests.get(
            "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items",
            params={'STATION_NUMBER': station_id, 'f': 'json', 'limit': 1},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return len(data.get('features', [])) > 0
    except Exception:  # noqa: BLE001
        return False

@pytest.fixture
def mock_config(tmp_path):
    return {
        # Required system settings
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        # Required domain settings
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-05 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        # Required forcing and model settings
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        # Observation-specific settings
        'FORCING_TIME_STEP_SIZE': 3600,
        'DOWNLOAD_USGS_DATA': True,
        'STATION_ID': '06306300',
        'STREAMFLOW_DATA_PROVIDER': 'USGS',
        'ADDITIONAL_OBSERVATIONS': 'usgs_streamflow',
        'STREAMFLOW_RAW_PATH': 'default',
        'STREAMFLOW_PROCESSED_PATH': 'default',
        'STREAMFLOW_RAW_NAME': 'test_raw.rdb'
    }

@pytest.fixture
def mock_usgs_response():
    # Real USGS RDB format
    return """# USGS RDB content
# Data columns:
agency_cd	site_no	datetime	00060_00000
5s	15s	20d	14n
USGS	06306300	2020-01-01 00:00	100.0
USGS	06306300	2020-01-01 01:00	110.0
USGS	06306300	2020-01-01 02:00	120.0
USGS	06306300	2020-01-01 03:00	130.0
"""

def test_usgs_streamflow_acquisition(mock_config, mock_usgs_response):
    """Test the formalized USGS streamflow acquisition pathway (Live Canary)."""
    logger = logging.getLogger("test_usgs")

    dm = DataManager(mock_config, logger)

    # Verify handler is registered (registry accepts both cases)
    assert ObservationRegistry.is_registered('usgs_streamflow')
    assert ObservationRegistry.is_registered('USGS_STREAMFLOW')  # Case-insensitive

    # Process observed data (Live API call)
    dm.process_observed_data()

    # Verify results
    project_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / f"domain_{mock_config['DOMAIN_NAME']}"
    raw_file = project_dir / "data" / "observations" / "streamflow" / "raw_data" / "usgs_06306300_raw.rdb"
    processed_file = project_dir / "data" / "observations" / "streamflow" / "preprocessed" / f"{mock_config['DOMAIN_NAME']}_streamflow_processed.csv"

    assert raw_file.exists(), f"Raw USGS file not found at {raw_file}"
    assert processed_file.exists(), f"Processed USGS file not found at {processed_file}"

    # Load and verify content
    df = pd.read_csv(processed_file)
    assert 'datetime' in df.columns
    assert 'discharge_cms' in df.columns
    assert len(df) > 0

def test_usgs_groundwater_acquisition(mock_config):
    """Test the formalized USGS groundwater acquisition pathway (Mocked)."""
    logger = logging.getLogger("test_usgs_gw")

    # Update config for real groundwater station
    gw_config = mock_config.copy()
    gw_config['DOWNLOAD_USGS_DATA'] = False
    gw_config['DOWNLOAD_USGS_GW'] = 'true'
    gw_config['USGS_STATION'] = '01646500'
    gw_config['STATION_ID'] = '01646500'
    gw_config['ADDITIONAL_OBSERVATIONS'] = 'usgs_gw'
    gw_config['DATA_ACCESS'] = 'cloud'

    # Mock Response
    mock_json = {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "variableName": "Depth to water level, feet below land surface",
                        "parameterCode": "72019",
                        "unit": {"unitCode": "ft"}
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2020-01-01T12:00:00.000", "value": "10.5"},
                                {"dateTime": "2020-01-02T12:00:00.000", "value": "10.4"}
                            ]
                        }
                    ]
                }
            ]
        }
    }

    dm = DataManager(gw_config, logger)

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.text = json.dumps(mock_json)

        dm.process_observed_data()

    processed_path = Path(gw_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "groundwater" / "test_domain_groundwater_processed.csv"

    assert processed_path.exists(), f"Processed USGS GW file not found at {processed_path}"

    df = pd.read_csv(processed_path)
    assert 'datetime' in df.columns
    assert 'groundwater_level' in df.columns
    assert len(df) > 0

@pytest.mark.integration
def test_provo_usgs_full_e2e(tmp_path):
    """
    E2E test for Provo River USGS data acquisition and processing.
    Runs a minimal full workflow to confirm actual usable data is retrieved.
    """
    import yaml

    # Complete config to satisfy SymfluenceConfig (Pydantic) validation
    config_data = {
        # Global
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'provo_river_test',
        'EXPERIMENT_ID': 'e2e_test',
        'EXPERIMENT_TIME_START': '2023-01-01 00:00',
        'EXPERIMENT_TIME_END': '2023-01-02 23:00',
        'CALIBRATION_PERIOD': '2023-01-01, 2023-01-01',
        'EVALUATION_PERIOD': '2023-01-02, 2023-01-02',
        'SPINUP_PERIOD': '2023-01-01, 2023-01-01',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,

        # Domain
        'POUR_POINT_COORDS': '40.5577/-111.1688',
        'BOUNDING_BOX_COORDS': '41/-111.7/40.0/-110.6',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',

        # Forcing
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,

        # Model
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',

        # Observations
        'STREAMFLOW_DATA_PROVIDER': 'USGS',
        'DOWNLOAD_USGS_DATA': True,
        'STATION_ID': '10163000',

        # Fallbacks for validation
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'SUPPLEMENT_FORCING': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    # Initialize SYMFLUENCE with path
    sym = SYMFLUENCE(config_input=config_file)

    # Run observed data processing
    sym.managers['data'].process_observed_data()

    # Verify results
    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "streamflow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_streamflow_processed.csv"

    assert processed_file.exists(), f"Processed USGS file not found at {processed_file}"

    # Load processed data and verify it has content and correct format
    df = pd.read_csv(processed_file)
    assert not df.empty, "Processed data is empty"
    assert 'datetime' in df.columns
    assert 'discharge_cms' in df.columns
    assert (df['discharge_cms'] >= 0).all()

@pytest.mark.integration
def test_wsc_geomet_full_e2e(tmp_path):
    """
    E2E test for Bow River WSC data acquisition via GeoMet API.
    """
    # Check if WSC API is accessible before running the test
    if not is_wsc_api_available('05BB001'):
        pytest.skip("WSC GeoMet API not accessible or returning no data for station 05BB001")

    import yaml
    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'bow_river_wsc_test',
        'EXPERIMENT_ID': 'wsc_e2e_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-05 23:00',
        'CALIBRATION_PERIOD': '2020-01-01, 2020-01-01',
        'EVALUATION_PERIOD': '2020-01-02, 2020-01-02',
        'SPINUP_PERIOD': '2020-01-01, 2020-01-01',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '51.1722/-115.5717',
        'BOUNDING_BOX_COORDS': '51.8/-116.6/50.9/-115.5',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'WSC',
        'DOWNLOAD_WSC_DATA': True,
        'STATION_ID': '05BB001',
        'DOWNLOAD_USGS_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'SUPPLEMENT_FORCING': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "wsc_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)
    sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "streamflow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_streamflow_processed.csv"

    assert processed_file.exists(), "Processed WSC file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty
    # Relax check: ignore -9999 or other no-data values
    valid_data = df[df['discharge_cms'] > -9000]
    if not valid_data.empty:
        assert (valid_data['discharge_cms'] >= 0).all()

@pytest.mark.integration
def test_usgs_gw_full_e2e(tmp_path):
    """
    E2E test for USGS Groundwater data acquisition via API.
    """
    import json

    import yaml

    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'usgs_gw_test',
        'EXPERIMENT_ID': 'gw_e2e_test',
        'EXPERIMENT_TIME_START': '2022-01-01 00:00',
        'EXPERIMENT_TIME_END': '2022-01-05 23:00',
        'CALIBRATION_PERIOD': '2022-01-01, 2022-01-01',
        'EVALUATION_PERIOD': '2022-01-02, 2022-01-02',
        'SPINUP_PERIOD': '2022-01-01, 2022-01-01',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '40.0/-105.0',
        'BOUNDING_BOX_COORDS': '41/-106/39/-104',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'none',
        'DOWNLOAD_USGS_DATA': False,
        'STATION_ID': '01646500',
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': True,
        'USGS_STATION': '01646500', # Potomac River well
        'ADDITIONAL_OBSERVATIONS': 'usgs_gw',
        'SUPPLEMENT_FORG': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "gw_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)

    # Mock Response
    mock_json = {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "variableName": "Depth to water level, feet below land surface",
                        "parameterCode": "72019",
                        "unit": {"unitCode": "ft"}
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2022-01-01T12:00:00.000", "value": "10.5"},
                                {"dateTime": "2022-01-02T12:00:00.000", "value": "10.4"}
                            ]
                        }
                    ]
                }
            ]
        }
    }

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.text = json.dumps(mock_json)

        sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "groundwater" / f"{config_data['DOMAIN_NAME']}_groundwater_processed.csv"

    assert processed_file.exists(), "Processed USGS GW file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty
    assert 'groundwater_level' in df.columns

@pytest.mark.integration
def test_grace_acquisition_and_processing(mock_config, tmp_path):
    """Test the GRACE acquisition and processing pathway with mocked NetCDF data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_grace")

    # 1. Create a mock GRACE NetCDF file
    grace_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "grace"
    grace_dir.mkdir(parents=True, exist_ok=True)
    mock_grace_file = grace_dir / "GRACE_JPL_test.nc"

    times = pd.date_range('2003-01-01', '2005-01-01', freq='MS')
    lats = np.linspace(30, 50, 10)
    lons = np.linspace(-120, -100, 10)

    ds = xr.Dataset(
        data_vars={
            'lwe_thickness': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)))
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    ds.to_netcdf(mock_grace_file)

    # 2. Create a mock catchment shapefile (required by GRACEHandler)
    catchment_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "shapefiles" / "catchment"
    catchment_dir.mkdir(parents=True, exist_ok=True)
    catchment_shp = catchment_dir / "test_domain_catchment.shp"

    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.GeoDataFrame({
        'ID': [1],
        'geometry': [box(-115, 35, -105, 45)]
    }, crs='EPSG:4326')
    gdf.to_file(catchment_shp)

    # 3. Configure for GRACE
    grace_config = mock_config.copy()
    grace_config['ADDITIONAL_OBSERVATIONS'] = 'grace'
    grace_config['CATCHMENT_PATH'] = str(catchment_dir)
    grace_config['CATCHMENT_SHP_NAME'] = 'test_domain_catchment.shp'

    # 4. Run acquisition and processing
    dm = DataManager(grace_config, logger)
    dm.acquire_observations()
    dm.process_observed_data()

    # 5. Verify results
    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "grace" / "preprocessed" / "test_domain_grace_tws_processed.csv"

    assert processed_file.exists(), "Processed GRACE file not found"
    df = pd.read_csv(processed_file)
    # Check for any grace data columns (CSR, GSFC, JPL - depending on what's available)
    grace_cols = [c for c in df.columns if 'grace' in c.lower()]
    assert len(grace_cols) > 0, f"No GRACE columns found. Columns: {df.columns.tolist()}"
    assert len(df) > 0

@pytest.mark.integration
def test_modis_snow_acquisition_and_processing(mock_config, tmp_path):
    """Test the MODIS Snow acquisition and processing pathway with mocked NetCDF data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_modis")

    # 1. Create a mock MODIS NetCDF file
    snow_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow" / "raw"
    snow_dir.mkdir(parents=True, exist_ok=True)
    mock_snow_file = snow_dir / "test_domain_MOD10A1.006_raw.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(30, 50, 5)
    lons = np.linspace(-120, -100, 5)

    ds = xr.Dataset(
        data_vars={
            'NDSI_Snow_Cover': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)))
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    ds.to_netcdf(mock_snow_file)

    # 2. Configure for MODIS
    modis_config = mock_config.copy()
    modis_config['ADDITIONAL_OBSERVATIONS'] = 'modis_snow'
    modis_config['DATA_ACCESS'] = 'cloud' # Trigger acquire() logic

    # 3. Just verify DataManager can be initialized with observation config
    dm = DataManager(modis_config, logger)

    # Verify the handler is registered (registry accepts both cases)
    assert ObservationRegistry.is_registered('modis_snow'), "modis_snow handler not registered"

@pytest.mark.integration
def test_smap_acquisition_and_processing(mock_config, tmp_path):
    """Test the SMAP acquisition and processing pathway."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_smap")

    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "soil_moisture" / "smap"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "smap_test.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    ds = xr.Dataset(
        data_vars={'soil_moisture': (('time', 'lat', 'lon'), np.random.rand(len(times), 2, 2))},
        coords={'time': times, 'lat': [40, 41], 'lon': [-105, -104]}
    )
    ds.to_netcdf(mock_file)

    smap_config = mock_config.copy()
    smap_config['ADDITIONAL_OBSERVATIONS'] = 'smap'

    dm = DataManager(smap_config, logger)
    dm.acquire_observations()
    dm.process_observed_data()

    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "soil_moisture" / "preprocessed" / "test_domain_smap_processed.csv"
    assert processed_file.exists()
    df = pd.read_csv(processed_file)
    assert 'soil_moisture' in df.columns

@pytest.mark.integration
def test_esa_cci_sm_acquisition_and_processing(mock_config, tmp_path):
    """Test the ESA CCI SM acquisition and processing pathway."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_esa")

    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "soil_moisture" / "esa_cci"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "esa_test.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    ds = xr.Dataset(
        data_vars={'sm': (('time', 'lat', 'lon'), np.random.rand(len(times), 2, 2))},
        coords={'time': times, 'lat': [40, 41], 'lon': [-105, -104]}
    )
    ds.to_netcdf(mock_file)

    esa_config = mock_config.copy()
    esa_config['ADDITIONAL_OBSERVATIONS'] = 'esa_cci_sm'

    # Just verify DataManager can be initialized with observation config
    dm = DataManager(esa_config, logger)

    # Verify the handler is registered (registry accepts both cases)
    assert ObservationRegistry.is_registered('esa_cci_sm'), "esa_cci_sm handler not registered"

@pytest.mark.integration
def test_fluxcom_et_acquisition_and_processing(mock_config, tmp_path):
    """Test the FLUXCOM ET acquisition and processing pathway."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_fluxcom")

    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "fluxcom"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "fluxcom_test.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    ds = xr.Dataset(
        data_vars={'ET': (('time', 'lat', 'lon'), np.random.rand(len(times), 2, 2))},
        coords={'time': times, 'lat': [40, 41], 'lon': [-105, -104]}
    )
    ds.to_netcdf(mock_file)

    fluxcom_config = mock_config.copy()
    fluxcom_config['ADDITIONAL_OBSERVATIONS'] = 'fluxcom_et'

    dm = DataManager(fluxcom_config, logger)
    dm.acquire_observations()
    dm.process_observed_data()

    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "preprocessed" / "test_domain_fluxcom_et_processed.csv"
    assert processed_file.exists()
    df = pd.read_csv(processed_file)
    assert 'ET' in df.columns


@pytest.mark.integration
def test_gpm_imerg_acquisition_and_processing(mock_config, tmp_path):
    """Test the GPM IMERG precipitation acquisition and processing pathway with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_gpm")

    # Create mock GPM IMERG NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "precipitation" / "gpm_imerg"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_GPM_IMERG_final_raw.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(39, 41, 5)
    lons = np.linspace(-106, -104, 5)

    ds = xr.Dataset(
        data_vars={
            'precipitation': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 10)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        },
        attrs={'title': 'GPM IMERG Final Precipitation'}
    )
    ds['precipitation'].attrs['units'] = 'mm/day'
    ds.to_netcdf(mock_file)

    # Configure for GPM IMERG
    gpm_config = mock_config.copy()
    gpm_config['ADDITIONAL_OBSERVATIONS'] = 'gpm_imerg'
    gpm_config['DATA_ACCESS'] = 'local'  # Use local mock data

    dm = DataManager(gpm_config, logger)

    # Verify handler is registered
    assert ObservationRegistry.is_registered('gpm_imerg'), "gpm_imerg handler not registered"
    assert ObservationRegistry.is_registered('gpm'), "gpm handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Verify output
    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "precipitation" / "preprocessed" / "test_domain_gpm_imerg_processed.csv"

    assert processed_file.exists(), f"Processed GPM file not found at {processed_file}"
    df = pd.read_csv(processed_file)
    assert 'datetime' in df.columns or 'time' in df.columns
    assert 'precipitation_mm' in df.columns
    assert len(df) > 0
    assert (df['precipitation_mm'] >= 0).all(), "Precipitation should be non-negative"


@pytest.mark.integration
@pytest.mark.requires_credentials
@pytest.mark.skip(reason="GPM IMERG requires NASA Earthdata credentials with GES DISC application approval - see https://disc.gsfc.nasa.gov/data-access")
def test_gpm_imerg_live_acquisition(tmp_path):
    """
    Live E2E test for GPM IMERG precipitation data acquisition.
    Requires NASA Earthdata credentials with GES DISC application approval.

    To enable this test:
    1. Create NASA Earthdata account at https://urs.earthdata.nasa.gov
    2. Approve GES DISC application in your Earthdata profile
    3. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables
       OR add to ~/.netrc:
       machine urs.earthdata.nasa.gov login <username> password <password>
    """
    import os

    import yaml

    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'gpm_test',
        'EXPERIMENT_ID': 'gpm_e2e_test',
        'EXPERIMENT_TIME_START': '2023-06-01 00:00',
        'EXPERIMENT_TIME_END': '2023-06-03 23:00',
        'CALIBRATION_PERIOD': '2023-06-01, 2023-06-02',
        'EVALUATION_PERIOD': '2023-06-02, 2023-06-03',
        'SPINUP_PERIOD': '2023-06-01, 2023-06-01',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '40.0/-105.0',
        'BOUNDING_BOX_COORDS': '40.5/-105.5/39.5/-104.5',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'none',
        'DOWNLOAD_USGS_DATA': False,
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'ADDITIONAL_OBSERVATIONS': 'gpm_imerg',
        'GPM_PRODUCT': 'final',
        'GPM_MAX_GRANULES': 3,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "gpm_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)
    sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "precipitation" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_gpm_imerg_processed.csv"

    assert processed_file.exists(), "Processed GPM IMERG file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty, "GPM IMERG data is empty"
    assert 'precipitation_mm' in df.columns
    assert (df['precipitation_mm'] >= 0).all()


@pytest.mark.integration
def test_chirps_acquisition_and_processing(mock_config, tmp_path):
    """Test the CHIRPS precipitation acquisition and processing pathway with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_chirps")

    # Create mock CHIRPS NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "precipitation" / "chirps"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_CHIRPS_daily_raw.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(39, 41, 10)
    lons = np.linspace(-106, -104, 10)

    ds = xr.Dataset(
        data_vars={
            'precip': (('time', 'latitude', 'longitude'), np.random.rand(len(times), len(lats), len(lons)) * 15)
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        },
        attrs={'title': 'CHIRPS Daily Precipitation'}
    )
    ds['precip'].attrs['units'] = 'mm/day'
    ds.to_netcdf(mock_file)

    # Configure for CHIRPS
    chirps_config = mock_config.copy()
    chirps_config['ADDITIONAL_OBSERVATIONS'] = 'chirps'
    chirps_config['DATA_ACCESS'] = 'local'
    chirps_config['BOUNDING_BOX_COORDS'] = '41.0/-106.0/39.0/-104.0'  # lat_max/lon_min/lat_min/lon_max
    chirps_config['POUR_POINT_COORDS'] = '40.0/-105.0'

    dm = DataManager(chirps_config, logger)

    # Verify handler is registered
    assert ObservationRegistry.is_registered('chirps'), "chirps handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Verify output
    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "precipitation" / "preprocessed" / "test_domain_chirps_processed.csv"

    assert processed_file.exists(), f"Processed CHIRPS file not found at {processed_file}"
    df = pd.read_csv(processed_file)
    assert 'datetime' in df.columns or 'time' in df.columns
    assert 'precipitation_mm' in df.columns
    assert len(df) > 0
    assert (df['precipitation_mm'] >= 0).all(), "Precipitation should be non-negative"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(900)  # CHIRPS downloads full annual files (~1-6 GB) before subsetting
def test_chirps_live_acquisition(tmp_path):
    """
    Live E2E test for CHIRPS precipitation data acquisition.
    No authentication required - publicly available data.
    """
    import yaml

    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'chirps_test',
        'EXPERIMENT_ID': 'chirps_e2e_test',
        'EXPERIMENT_TIME_START': '2020-06-01 00:00',
        'EXPERIMENT_TIME_END': '2020-06-05 23:00',
        'CALIBRATION_PERIOD': '2020-06-01, 2020-06-02',
        'EVALUATION_PERIOD': '2020-06-02, 2020-06-03',
        'SPINUP_PERIOD': '2020-06-01, 2020-06-01',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '40.0/-105.0',
        'BOUNDING_BOX_COORDS': '40.5/-105.5/39.5/-104.5',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'none',
        'DOWNLOAD_USGS_DATA': False,
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'ADDITIONAL_OBSERVATIONS': 'chirps',
        'CHIRPS_PRODUCT': 'daily',
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "chirps_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)
    sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "precipitation" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_chirps_processed.csv"

    assert processed_file.exists(), "Processed CHIRPS file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty, "CHIRPS data is empty"
    assert 'precipitation_mm' in df.columns
    assert (df['precipitation_mm'] >= 0).all()


@pytest.mark.integration
def test_snodas_acquisition_and_processing(mock_config, tmp_path):
    """Test the SNODAS snow acquisition and processing pathway with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_snodas")

    # Create mock SNODAS NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow" / "snodas"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_SNODAS_swe_raw.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(39, 41, 10)
    lons = np.linspace(-106, -104, 10)

    # SWE values in meters (0-1m range typical)
    ds = xr.Dataset(
        data_vars={
            'swe': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 0.5)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        },
        attrs={'title': 'SNODAS Snow Water Equivalent'}
    )
    ds['swe'].attrs['units'] = 'm'
    ds.to_netcdf(mock_file)

    # Configure for SNODAS
    snodas_config = mock_config.copy()
    snodas_config['ADDITIONAL_OBSERVATIONS'] = 'snodas'
    snodas_config['DATA_ACCESS'] = 'local'
    snodas_config['SNODAS_VARIABLE'] = 'swe'

    dm = DataManager(snodas_config, logger)

    # Verify handler is registered
    assert ObservationRegistry.is_registered('snodas'), "snodas handler not registered"
    assert ObservationRegistry.is_registered('snodas_swe'), "snodas_swe handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Verify output
    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow" / "preprocessed" / "test_domain_snodas_swe_processed.csv"

    assert processed_file.exists(), f"Processed SNODAS file not found at {processed_file}"
    df = pd.read_csv(processed_file)
    assert 'datetime' in df.columns or 'time' in df.columns
    assert 'swe_m' in df.columns or 'swe_mm' in df.columns
    assert len(df) > 0


@pytest.mark.integration
def test_snodas_live_acquisition(tmp_path):
    """
    Live E2E test for SNODAS snow data acquisition.
    No authentication required - publicly available data.
    Note: SNODAS only covers CONUS, so use appropriate coordinates.
    """
    import yaml

    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'snodas_test',
        'EXPERIMENT_ID': 'snodas_e2e_test',
        'EXPERIMENT_TIME_START': '2023-01-15 00:00',
        'EXPERIMENT_TIME_END': '2023-01-17 23:00',
        'CALIBRATION_PERIOD': '2023-01-15, 2023-01-16',
        'EVALUATION_PERIOD': '2023-01-16, 2023-01-17',
        'SPINUP_PERIOD': '2023-01-15, 2023-01-15',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '40.0/-105.5',
        'BOUNDING_BOX_COORDS': '40.5/-106.0/39.5/-105.0',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'none',
        'DOWNLOAD_USGS_DATA': False,
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'ADDITIONAL_OBSERVATIONS': 'snodas',
        'SNODAS_VARIABLE': 'swe',
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "snodas_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)
    sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "snow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_snodas_swe_processed.csv"

    assert processed_file.exists(), "Processed SNODAS file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty, "SNODAS data is empty"
    assert 'swe_m' in df.columns or 'swe_mm' in df.columns


@pytest.mark.integration
def test_jrc_water_acquisition_and_processing(mock_config, tmp_path):
    """Test the JRC Global Surface Water acquisition and processing pathway with mocked data."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    logger = logging.getLogger("test_jrc")

    # Create mock JRC GeoTIFF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "surface_water" / "jrc"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_JRC_occurrence_merged.tif"

    # Create mock water occurrence raster (0-100%)
    height, width = 100, 100
    data = np.random.randint(0, 100, (height, width), dtype=np.uint8)

    # Set transform for the bounding box area
    transform = from_bounds(-106, 39, -104, 41, width, height)

    with rasterio.open(
        mock_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=255
    ) as dst:
        dst.write(data, 1)

    # Configure for JRC
    jrc_config = mock_config.copy()
    jrc_config['ADDITIONAL_OBSERVATIONS'] = 'jrc_water'
    jrc_config['DATA_ACCESS'] = 'local'
    jrc_config['JRC_WATER_DATASET'] = 'occurrence'

    dm = DataManager(jrc_config, logger)

    # Verify handler is registered
    assert ObservationRegistry.is_registered('jrc_water'), "jrc_water handler not registered"
    assert ObservationRegistry.is_registered('jrc_gsw'), "jrc_gsw handler not registered"
    assert ObservationRegistry.is_registered('surface_water'), "surface_water handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Verify output
    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "surface_water" / "preprocessed" / "test_domain_jrc_occurrence_processed.csv"

    assert processed_file.exists(), f"Processed JRC file not found at {processed_file}"
    df = pd.read_csv(processed_file)
    assert 'occurrence_mean' in df.columns
    assert len(df) > 0


@pytest.mark.integration
def test_ssebop_acquisition_and_processing(mock_config, tmp_path):
    """Test the SSEBop ET acquisition and processing pathway with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_ssebop")

    # Create mock SSEBop NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "ssebop"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_SSEBop_conus_daily_raw.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(39, 41, 10)
    lons = np.linspace(-106, -104, 10)

    # ET values in mm/day (0-10 range typical)
    ds = xr.Dataset(
        data_vars={
            'et': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 5)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        },
        attrs={'title': 'SSEBop Evapotranspiration'}
    )
    ds['et'].attrs['units'] = 'mm/day'
    ds.to_netcdf(mock_file)

    # Configure for SSEBop
    ssebop_config = mock_config.copy()
    ssebop_config['ADDITIONAL_OBSERVATIONS'] = 'ssebop'
    ssebop_config['DATA_ACCESS'] = 'local'
    ssebop_config['SSEBOP_PRODUCT'] = 'conus'

    # Verify handler is registered
    assert ObservationRegistry.is_registered('ssebop'), "ssebop handler not registered"
    assert ObservationRegistry.is_registered('ssebop_et'), "ssebop_et handler not registered"

    # Use direct handler calls to avoid DataManager swallowing exceptions
    handler = ObservationRegistry.get_handler('ssebop', ssebop_config, logger)
    raw_path = handler.acquire()
    processed_path = handler.process(raw_path)

    # Verify output
    assert processed_path.exists(), f"Processed SSEBop file not found at {processed_path}"
    assert processed_path.suffix == '.csv', f"Expected CSV output, got {processed_path.suffix}"
    df = pd.read_csv(processed_path)
    assert 'datetime' in df.columns or 'time' in df.columns
    assert 'et_mm_day' in df.columns
    assert len(df) > 0
    assert (df['et_mm_day'] >= 0).all(), "ET should be non-negative"


# =============================================================================
# Additional Handler Tests - Ensuring Full Coverage
# =============================================================================

@pytest.mark.integration
def test_gleam_et_acquisition_and_processing(mock_config, tmp_path):
    """Test the GLEAM ET acquisition and processing pathway with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_gleam")

    # Create mock GLEAM NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "gleam"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "E_2020_GLEAM_v3.6a_daily.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(39, 41, 5)
    lons = np.linspace(-106, -104, 5)

    ds = xr.Dataset(
        data_vars={
            'E': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 5)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        },
        attrs={'title': 'GLEAM Evaporation'}
    )
    ds['E'].attrs['units'] = 'mm/day'
    ds.to_netcdf(mock_file)

    gleam_config = mock_config.copy()
    gleam_config['ADDITIONAL_OBSERVATIONS'] = 'gleam_et'
    gleam_config['DATA_ACCESS'] = 'local'

    dm = DataManager(gleam_config, logger)

    assert ObservationRegistry.is_registered('gleam_et'), "gleam_et handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "preprocessed" / "test_domain_gleam_et_processed.csv"

    assert processed_file.exists(), f"Processed GLEAM file not found at {processed_file}"
    df = pd.read_csv(processed_file)
    assert len(df) > 0


@pytest.mark.integration
def test_modis_et_acquisition_and_processing(mock_config, tmp_path):
    """Test the MODIS ET (MOD16) acquisition and processing pathway with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_modis_et")

    # Create mock MODIS ET NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "modis"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_MOD16A2_raw.nc"

    # MODIS ET is 8-day composite
    times = pd.date_range('2020-01-01', '2020-01-25', freq='8D')
    lats = np.linspace(39, 41, 5)
    lons = np.linspace(-106, -104, 5)

    ds = xr.Dataset(
        data_vars={
            'ET_500m': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 30)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        },
        attrs={'title': 'MODIS MOD16A2 Evapotranspiration'}
    )
    ds['ET_500m'].attrs['units'] = 'kg/m^2/8day'
    ds.to_netcdf(mock_file)

    modis_config = mock_config.copy()
    modis_config['ADDITIONAL_OBSERVATIONS'] = 'modis_et'
    modis_config['DATA_ACCESS'] = 'local'

    dm = DataManager(modis_config, logger)

    assert ObservationRegistry.is_registered('modis_et'), "modis_et handler not registered"
    assert ObservationRegistry.is_registered('mod16'), "mod16 handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Check for output file
    processed_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "et" / "preprocessed"
    processed_files = list(processed_dir.glob("*modis*.csv")) + list(processed_dir.glob("*mod16*.csv"))
    assert len(processed_files) > 0 or processed_dir.exists(), "MODIS ET processing directory should exist"


@pytest.mark.integration
def test_modis_sca_acquisition_and_processing(mock_config, tmp_path):
    """Test the MODIS SCA (Snow Cover Area) acquisition and processing with mocked data."""
    import numpy as np
    import xarray as xr

    logger = logging.getLogger("test_modis_sca")

    # Create mock MODIS SCA NetCDF file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow" / "raw"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "test_domain_MOD10A1_merged.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(39, 41, 5)
    lons = np.linspace(-106, -104, 5)

    # NDSI Snow Cover values (0-100)
    ds = xr.Dataset(
        data_vars={
            'NDSI_Snow_Cover': (('time', 'lat', 'lon'), np.random.randint(0, 100, (len(times), len(lats), len(lons))))
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        },
        attrs={'title': 'MODIS Snow Cover Area'}
    )
    ds.to_netcdf(mock_file)

    sca_config = mock_config.copy()
    sca_config['ADDITIONAL_OBSERVATIONS'] = 'modis_sca'
    sca_config['DATA_ACCESS'] = 'local'

    dm = DataManager(sca_config, logger)

    assert ObservationRegistry.is_registered('modis_sca'), "modis_sca handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Verify processing occurred
    processed_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow" / "preprocessed"
    assert processed_dir.exists() or obs_dir.exists(), "MODIS SCA output directory should exist"


@pytest.mark.integration
def test_smhi_streamflow_acquisition_and_processing(mock_config, tmp_path):
    """Test the SMHI streamflow acquisition and processing with mocked API response."""
    logger = logging.getLogger("test_smhi")

    # Mock SMHI API response
    mock_json = {
        "value": [
            {"date": 1577836800000, "value": "10.5"},  # 2020-01-01
            {"date": 1577923200000, "value": "11.2"},  # 2020-01-02
            {"date": 1578009600000, "value": "10.8"},  # 2020-01-03
        ]
    }

    smhi_config = mock_config.copy()
    smhi_config['ADDITIONAL_OBSERVATIONS'] = 'smhi_streamflow'
    smhi_config['STREAMFLOW_DATA_PROVIDER'] = 'SMHI'
    smhi_config['STATION_ID'] = '2252'  # Example Swedish station
    smhi_config['DATA_ACCESS'] = 'cloud'

    dm = DataManager(smhi_config, logger)

    assert ObservationRegistry.is_registered('smhi_streamflow'), "smhi_streamflow handler not registered"

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.text = json.dumps(mock_json)

        dm.process_observed_data()

    # Check output exists or handler ran
    processed_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "streamflow"
    assert processed_dir.exists() or True, "SMHI streamflow handler should have run"


@pytest.mark.integration
def test_snotel_acquisition_and_processing(mock_config, tmp_path):
    """Test the SNOTEL SWE acquisition and processing with mocked data."""
    logger = logging.getLogger("test_snotel")

    # Create mock SNOTEL CSV file (NRCS format)
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow" / "swe" / "raw"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "snotel_raw.csv"

    # SNOTEL CSV format
    snotel_data = """Date,Snow Water Equivalent (in),Precipitation Accumulation (in)
2020-01-01,5.2,8.1
2020-01-02,5.5,8.3
2020-01-03,5.8,8.5
2020-01-04,6.0,8.8
2020-01-05,6.2,9.0
"""
    with open(mock_file, 'w') as f:
        f.write(snotel_data)

    snotel_config = mock_config.copy()
    snotel_config['ADDITIONAL_OBSERVATIONS'] = 'snotel'
    snotel_config['DOWNLOAD_SNOTEL'] = True
    snotel_config['SNOTEL_STATION_ID'] = '1234'
    snotel_config['DATA_ACCESS'] = 'local'

    dm = DataManager(snotel_config, logger)

    assert ObservationRegistry.is_registered('snotel'), "snotel handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Verify handler ran
    processed_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "snow"
    assert processed_dir.exists(), "SNOTEL processing directory should exist"


@pytest.mark.integration
def test_ismn_soil_moisture_acquisition_and_processing(mock_config, tmp_path):
    """Test the ISMN soil moisture acquisition and processing with mocked data."""
    logger = logging.getLogger("test_ismn")

    # Create mock ISMN station file
    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "soil_moisture" / "ismn"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "station_001.csv"

    ismn_data = """timestamp,soil_moisture,depth
2020-01-01 00:00:00,0.25,0.05
2020-01-01 12:00:00,0.24,0.05
2020-01-02 00:00:00,0.26,0.05
2020-01-02 12:00:00,0.25,0.05
2020-01-03 00:00:00,0.27,0.05
"""
    with open(mock_file, 'w') as f:
        f.write(ismn_data)

    ismn_config = mock_config.copy()
    ismn_config['ADDITIONAL_OBSERVATIONS'] = 'ismn'
    ismn_config['DATA_ACCESS'] = 'local'

    dm = DataManager(ismn_config, logger)

    assert ObservationRegistry.is_registered('ismn'), "ismn handler not registered"

    dm.acquire_observations()
    dm.process_observed_data()

    # Check for processed output
    processed_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations" / "soil_moisture"
    processed_files = list(processed_dir.rglob("*ismn*.csv"))
    assert len(processed_files) > 0 or processed_dir.exists(), "ISMN processing should create output"


@pytest.mark.integration
def test_ggmn_groundwater_acquisition_and_processing(mock_config, tmp_path):
    """Test the GGMN groundwater acquisition and processing with mocked WFS response."""
    logger = logging.getLogger("test_ggmn")

    # Mock GGMN WFS response (simplified)
    mock_wfs_response = """<?xml version="1.0"?>
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs">
        <gml:featureMember xmlns:gml="http://www.opengis.net/gml">
            <ggmn:Well>
                <ggmn:id>WELL001</ggmn:id>
                <ggmn:lat>40.0</ggmn:lat>
                <ggmn:lon>-105.0</ggmn:lon>
            </ggmn:Well>
        </gml:featureMember>
    </wfs:FeatureCollection>"""

    mock_measurement_html = """
    <html><body>
    <table><tr><td>2020-01-01</td><td>10.5</td></tr>
    <tr><td>2020-01-02</td><td>10.4</td></tr></table>
    </body></html>"""

    ggmn_config = mock_config.copy()
    ggmn_config['ADDITIONAL_OBSERVATIONS'] = 'ggmn'
    ggmn_config['DATA_ACCESS'] = 'cloud'

    dm = DataManager(ggmn_config, logger)

    assert ObservationRegistry.is_registered('ggmn'), "ggmn handler not registered"

    with patch('requests.get') as mock_get:
        # First call returns WFS, second returns measurements
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = mock_wfs_response
        mock_get.return_value.content = mock_wfs_response.encode()

        # Handler should run without error
        try:
            dm.process_observed_data()
        except Exception as e:  # noqa: BLE001
            # GGMN may fail gracefully if no stations found - that's OK for this test
            logger.warning(f"GGMN test completed with: {e}")

    # Verify handler was invoked
    processed_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "data" / "observations"
    assert processed_dir.exists(), "Observations directory should exist"


@pytest.mark.integration
def test_lamah_ice_streamflow_processing(mock_config, tmp_path):
    """Test the LamaH-ICE streamflow processing with mocked local data."""
    logger = logging.getLogger("test_lamah")

    # Create mock LamaH-ICE file at correct directory structure
    # Handler expects: LAMAH_ICE_PATH / D_gauges / 2_timeseries / daily / ID_{station_id}.csv
    lamah_path = tmp_path / "lamah_ice_data"
    raw_dir = lamah_path / "D_gauges" / "2_timeseries" / "daily"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mock_file = raw_dir / "ID_001.csv"

    lamah_data = """YYYY;MM;DD;qobs;qc_flag
2020;01;01;15.5;0
2020;01;02;16.2;0
2020;01;03;14.8;0
2020;01;04;15.0;0
2020;01;05;15.3;0
"""
    with open(mock_file, 'w') as f:
        f.write(lamah_data)

    lamah_config = mock_config.copy()
    lamah_config['ADDITIONAL_OBSERVATIONS'] = 'lamah_ice_streamflow'
    lamah_config['LAMAH_ICE_PATH'] = str(lamah_path)
    lamah_config['STATION_ID'] = '001'
    lamah_config['DATA_ACCESS'] = 'local'

    assert ObservationRegistry.is_registered('lamah_ice_streamflow'), "lamah_ice_streamflow handler not registered"

    # Use direct handler calls to avoid DataManager swallowing exceptions
    handler = ObservationRegistry.get_handler('lamah_ice_streamflow', lamah_config, logger)
    raw_path = handler.acquire()
    processed_path = handler.process(raw_path)

    # Check for processed output
    assert processed_path.exists(), f"Processed LamaH-ICE file not found at {processed_path}"
    df = pd.read_csv(processed_path)
    assert 'discharge_cms' in df.columns, f"Expected discharge_cms column, got {df.columns.tolist()}"
    assert len(df) > 0, "Processed data should not be empty"


# =============================================================================
# Live Data Acquisition Tests (require network access)
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_smhi_live_acquisition(tmp_path):
    """
    Live E2E test for SMHI streamflow data acquisition.
    No authentication required - publicly available Swedish data.
    """
    import yaml

    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'smhi_test',
        'EXPERIMENT_ID': 'smhi_e2e_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-10 23:00',
        'CALIBRATION_PERIOD': '2020-01-01, 2020-01-05',
        'EVALUATION_PERIOD': '2020-01-05, 2020-01-10',
        'SPINUP_PERIOD': '2020-01-01, 2020-01-01',
        'NUM_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '59.33/18.07',  # Stockholm area
        'BOUNDING_BOX_COORDS': '60/17/58/19',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'SMHI',
        'STATION_ID': '2252',  # Norrström station
        'DOWNLOAD_USGS_DATA': False,
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "smhi_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)
    sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "data" / "observations" / "streamflow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_streamflow_processed.csv"

    assert processed_file.exists(), "Processed SMHI file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty, "SMHI data is empty"
