# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Cloud Data Utilities for SYMFLUENCE

Registry-based access to cloud-hosted forcing and attribute datasets.

This module provides a facade for downloading data from various cloud providers
(NASA, NOAA, Copernicus, etc.) without exposing the complexities of individual
dataset APIs. It uses the AcquisitionRegistry plugin pattern to dynamically
select and instantiate appropriate handlers based on configuration.

The CloudForcingDownloader is the main entry point for users and calibration
workflows. Individual download_* convenience methods provide backward compatibility
and direct access to specific datasets.

Supported Datasets:
    Forcing:
        - ERA5, AORC, HRRR, CONUS404, NEX-GDDP, EM-Earth
    Attributes:
        - SoilGrids, MODIS Landcover, USGS NLCD
        - DEM: Copernicus GLO-30, Copernicus GLO-90, FABDEM, NASADEM,
               SRTM GL1, ETOPO 2022, Mapzen Terrain, ALOS AW3D30
        - Glacier data (RGI)
    Observations:
        - GRACE, MODIS (snow/ET), USGS streamflow, SMAP soil moisture, ISMN

Example:
    >>> from pathlib import Path
    >>> config = {'FORCING_DATASET': 'ERA5', 'SUPPLEMENT_FORCING': False}
    >>> downloader = CloudForcingDownloader(config, logger)
    >>> output = downloader.download_forcing_data(Path('./data'))
"""
from pathlib import Path
from typing import Dict

from symfluence.core.mixins import ConfigMixin
from symfluence.data.acquisition.registry import AcquisitionRegistry


class CloudForcingDownloader(ConfigMixin):
    """Main entry point for cloud data acquisition using the AcquisitionRegistry.

    Provides a registry-based facade for downloading climate, weather, and
    hydrological data from cloud providers. Uses the plugin pattern to dynamically
    select handlers based on dataset name from configuration.

    This class handles:
    - Main dataset download (ERA5, AORC, etc.)
    - Optional supplemental datasets (EM-Earth as backup)
    - Error handling and logging
    - Backward-compatible convenience methods for specific datasets

    Args:
        config: Configuration dictionary containing:
            - FORCING_DATASET (str): Main dataset to download (case-insensitive)
            - SUPPLEMENT_FORCING (bool): Whether to also download EM-Earth backup
        logger: Python logger instance for status messages

    Attributes:
        dataset_name: Uppercase version of FORCING_DATASET from config
        supplement_data: Whether supplemental data should be acquired
    """

    def __init__(self, config: Dict, logger):
        """Initialize the cloud downloader with configuration.

        Args:
            config: Configuration dict with FORCING_DATASET and SUPPLEMENT_FORCING keys
            logger: Logger instance for diagnostic messages
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.dataset_name = config.get('FORCING_DATASET', '').upper()
        self.supplement_data = config.get('SUPPLEMENT_FORCING', False)

    def download_forcing_data(self, output_dir: Path) -> Path:
        """Download forcing data based on configured dataset.

        Downloads the primary forcing dataset (ERA5, AORC, HRRR, etc.) specified
        in configuration. Optionally supplements with EM-Earth data before
        downloading the main dataset (useful as backup for data gaps).

        The actual download implementation is delegated to the registered handler
        for the dataset type, allowing for dataset-specific logic (authentication,
        data format handling, spatial/temporal filtering).

        Args:
            output_dir: Directory where downloaded data will be saved.

        Returns:
            Path: Location of downloaded/processed forcing data file(s).

        Raises:
            ValueError: If configured dataset is not registered in AcquisitionRegistry
            IOError: If download fails (network, permissions, storage issues)

        Note:
            - Supplemental EM-Earth data (if enabled) is downloaded first as backup
            - Main dataset download is wrapped with error handling
            - Dataset name is case-insensitive (normalized to uppercase)
        """
        self.logger.info(f"Starting cloud data acquisition for {self.dataset_name}")

        # Supplemental data handling (kept from original behavior).
        # EM-Earth is used as a backup/gap-filling dataset for missing forcing data.
        if self.supplement_data:
            self.logger.info('Supplementing data, downloading EM-Earth')
            em_handler = AcquisitionRegistry.get_handler('EM-EARTH', self.config, self.logger)
            em_handler.download(output_dir)

        # Main dataset download via registry-based handler lookup.
        # The registry returns a handler class specific to the dataset type,
        # which knows how to download and process data from that source.
        try:
            handler = AcquisitionRegistry.get_handler(self.dataset_name, self.config, self.logger)
            return handler.download(output_dir)
        except ValueError as e:
            self.logger.error(str(e))
            raise

    # Legacy convenience methods for backward compatibility.
    # These methods allow calling specific dataset downloads without
    # reconfiguring the CloudForcingDownloader.

    def download_soilgrids_soilclasses(self) -> Path:
        """Download SoilGrids soil class data for domain.

        Acquires global soil classification data from SoilGrids v2.0 and extracts
        values for the domain bounding box.

        Returns:
            Path: Location of downloaded soil class raster file.
        """
        handler = AcquisitionRegistry.get_handler('SOILGRIDS', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_global_soilclasses(self) -> Path:
        """Alias for download_soilgrids_soilclasses for backward compatibility.

        Use download_soilgrids_soilclasses() for new code.

        Returns:
            Path: Location of downloaded soil class file.
        """
        return self.download_soilgrids_soilclasses()

    def download_modis_landcover(self) -> Path:
        """Download MODIS land cover classification for domain.

        Acquires MODIS MCD12Q1 land cover data and clips to domain bounding box.
        Uses IGBP classification scheme.

        Returns:
            Path: Location of downloaded land cover raster file.
        """
        handler = AcquisitionRegistry.get_handler('MODIS_LANDCOVER', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_usgs_landcover(self) -> Path:
        """Download USGS National Land Cover Database (NLCD) for domain.

        Acquires NLCD data (US only) and clips to domain bounding box.
        Note: Only covers contiguous United States.

        Returns:
            Path: Location of downloaded land cover raster file.

        Raises:
            ValueError: If domain is outside NLCD coverage area
        """
        handler = AcquisitionRegistry.get_handler('USGS_NLCD', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_copernicus_dem(self) -> Path:
        """Download Copernicus DEM 30m elevation data for domain.

        Acquires elevation data from Copernicus Digital Elevation Model and
        clips to domain bounding box.

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('COPDEM30', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_fabdem(self) -> Path:
        """Download FABDEM (Forest And Buildings removed DEM) elevation data.

        Acquires elevation data with forest canopy and building artifacts removed,
        useful for hydrologic modeling. Clips to domain bounding box.

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('FABDEM', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_nasadem_local(self) -> Path:
        """Download NASADEM elevation data for local use.

        Acquires NASA's SRTM-derived DEM at 30m resolution (where available)
        and clips to domain bounding box.

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('NASADEM_LOCAL', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_copernicus_dem_90(self) -> Path:
        """Download Copernicus DEM GLO-90 (90m) elevation data for domain.

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('COPDEM90', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_srtm_dem(self) -> Path:
        """Download SRTM GL1 (30m) elevation data for domain.

        Coverage: 60N to 56S latitude.

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('SRTM', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_etopo_dem(self) -> Path:
        """Download ETOPO 2022 global relief model elevation data for domain.

        Configurable resolution via ETOPO_RESOLUTION (15s/30s/60s) and
        variant via ETOPO_VARIANT (surface/bedrock).

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('ETOPO2022', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_mapzen_dem(self) -> Path:
        """Download Mapzen terrain tile elevation data for domain.

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('MAPZEN', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_alos_dem(self) -> Path:
        """Download ALOS AW3D30 (30m) elevation data for domain.

        Requires optional packages: planetary-computer, pystac-client.
        Install with: pip install symfluence[alos]

        Returns:
            Path: Location of downloaded elevation raster file.
        """
        handler = AcquisitionRegistry.get_handler('ALOS', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))

    def download_glacier_data(self) -> Path:
        """Download glacier extent data from RGI (Randolph Glacier Inventory).

        Acquires glacier extent vector data (version 6.0) and clips to domain.
        Used for snow/ice processes in hydrologic modeling.

        Returns:
            Path: Location of downloaded glacier shapefile.
        """
        handler = AcquisitionRegistry.get_handler('GLACIER', self.config, self.logger)
        return handler.download(Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')))


def check_cloud_access_availability(dataset_name: str, logger) -> bool:
    """Check if a dataset is available for cloud access via the registry.

    Queries the AcquisitionRegistry to determine if a given dataset name
    has a registered handler. Useful for validating configuration before
    attempting downloads.

    Args:
        dataset_name: Case-insensitive dataset identifier (e.g., 'ERA5')
        logger: Logger instance for status messages

    Returns:
        bool: True if dataset is registered and available, False otherwise

    Example:
        >>> if check_cloud_access_availability('ERA5', logger):
        ...     downloader = CloudForcingDownloader(config, logger)
    """
    if AcquisitionRegistry.is_registered(dataset_name):
        logger.info(f"✓ {dataset_name} supports cloud data access")
        return True
    else:
        logger.warning(f"✗ {dataset_name} does not support cloud access.")
        return False


# Variable name mappings for dataset handlers.
# These map dataset-specific variable names to SYMFLUENCE standard variable names
# used throughout the preprocessing pipeline. This allows flexible input datasets
# while maintaining consistent internal variable naming.

def get_aorc_variable_mapping() -> Dict[str, str]:
    """Get variable name mapping for AORC forcing dataset.

    Maps AORC variable names to SYMFLUENCE standard variables.

    AORC (Analysis of Record for Calibration) is a gridded meteorological
    dataset derived from NOAA observations.

    Returns:
        dict: Mapping from AORC names to standard names
              (e.g., 'APCP_surface' -> 'precipitation_flux')
    """
    return {
        'APCP_surface': 'precipitation_flux', 'TMP_2maboveground': 'air_temperature', 'SPFH_2maboveground': 'specific_humidity',
        'PRES_surface': 'surface_air_pressure', 'DLWRF_surface': 'surface_downwelling_longwave_flux', 'DSWRF_surface': 'surface_downwelling_shortwave_flux',
        'UGRD_10maboveground': 'wind_u', 'VGRD_10maboveground': 'wind_v'
    }

def get_era5_variable_mapping() -> Dict[str, str]:
    """Get variable name mapping for ERA5 forcing dataset.

    Maps ERA5 variable names to SYMFLUENCE standard variables.

    ERA5 is the ECMWF reanalysis dataset, covering 1940-present at
    31 km resolution.

    Returns:
        dict: Mapping from ERA5 names to standard names
              (e.g., 't2m' -> 'air_temperature')
    """
    return {
        't2m': 'air_temperature', 'u10': 'wind_u', 'v10': 'wind_v', 'sp': 'surface_air_pressure',
        'd2m': 'dewpoint', 'q': 'specific_humidity', 'tp': 'precipitation_flux', 'ssrd': 'surface_downwelling_shortwave_flux', 'strd': 'surface_downwelling_longwave_flux'
    }

def get_emearth_variable_mapping() -> Dict[str, str]:
    """Get variable name mapping for EM-Earth forcing dataset.

    Maps EM-Earth variable names to SYMFLUENCE standard variables.

    EM-Earth is a global gridded climate dataset combining satellite
    observations and reanalysis data.

    Returns:
        dict: Mapping from EM-Earth names to standard names
    """
    return {"prcp": "precipitation_flux", "prcp_corrected": "precipitation_flux", "tmean": "air_temperature", "trange": "temp_range", "tdew": "dewpoint"}

def get_hrrr_variable_mapping() -> Dict[str, str]:
    """Get variable name mapping for HRRR forcing dataset.

    Maps HRRR variable names to SYMFLUENCE standard variables.

    HRRR (High-Resolution Rapid Refresh) is a NOAA hourly forecast
    dataset at 3 km resolution over North America.

    Returns:
        dict: Mapping from HRRR names to standard names
    """
    return {
        'TMP': 'air_temperature', 'SPFH': 'specific_humidity', 'PRES': 'surface_air_pressure', 'UGRD': 'wind_u',
        'VGRD': 'wind_v', 'DSWRF': 'surface_downwelling_shortwave_flux', 'DLWRF': 'surface_downwelling_longwave_flux', 'APCP': 'precipitation_flux'
    }

def get_conus404_variable_mapping() -> Dict[str, str]:
    """Get variable name mapping for CONUS404 forcing dataset.

    Maps CONUS404 variable names to SYMFLUENCE standard variables.

    CONUS404 is a 4 km WRF-downscaled dataset covering continental US
    from NCAR.

    Returns:
        dict: Mapping from CONUS404 names to standard names
    """
    return {
        'T2': 'air_temperature', 'Q2': 'specific_humidity', 'PSFC': 'surface_air_pressure', 'U10': 'wind_u',
        'V10': 'wind_v', 'GLW': 'surface_downwelling_longwave_flux', 'SWDOWN': 'surface_downwelling_shortwave_flux', 'RAINRATE': 'precipitation_flux'
    }
