# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Acquisition Service

Unified facade for all data acquisition workflows in SYMFLUENCE. Coordinates
downloading and processing of geospatial attributes, forcing data, and
observations from diverse sources (cloud, HPC, local). Acts as high-level
orchestrator delegating to specialized acquisition handlers and cloud
downloaders.

Architecture:
    AcquisitionService provides two parallel acquisition paths:

    1. CLOUD Mode (CloudForcingDownloader):
       - Cloud-based data providers with direct HTTP/S3 access
       - DEM sources: Copernicus GLO-30/90, FABDEM, NASADEM, SRTM, ETOPO, Mapzen, ALOS
       - Soil class: SoilGrids via WCS subsetting
       - Land cover: MODIS Landcover (multi-year mode), USGS NLCD
       - Forcing: ERA5 (CDS), CARRA/CERRA (CDS), AORC (AWS/GCS), NEX-GDDP (Zenodo)
       - Observations: USGS, WSC, SMHI, SNOTEL, GRACE, MODIS snow/ET

    2. MAF Mode (gistoolRunner, datatoolRunner):
       - HPC-based data access via external MAF tools on supercomputers
       - gistool: MERIT-Hydro elevation, MODIS landcover, SoilGrids soil class
       - datatool: ERA5, RDRS, CASR forcing data with Slurm job monitoring
       - Configuration: Generates MAF JSON configs and executes MAF scheduler
       - Output: Same directory structure as CLOUD mode

Data Acquisition Workflows:
    1. Attribute Acquisition (acquire_attributes)
       - DEM/elevation: Multiple sources with fallback logic
       - Soil classification: SoilGrids primary, gistool fallback
       - Land cover: MODIS or USGS depending on availability
       - Output: GeoTIFF rasters at project_dir/attributes/{type}/

    2. Forcing Data Download (acquire_forcings)
       - Datasets: ERA5, CARRA, CERRA, AORC, NEX-GDDP
       - Mode selection: CLOUD vs MAF based on config.domain.data_access
       - Caching: RawForcingCache with automatic TTL/checksum validation
       - Unit conversion: Via VariableHandler for dataset-specific mappings
       - Output: NetCDF at project_dir/forcing/{dataset}_raw/

    3. Observation Data Retrieval (acquire_observations)
       - Streamflow: USGS (NWIS), WSC (Canada), SMHI (Nordic)
       - Gridded: GRACE, MODIS Snow, MODIS ET, FLUXNET
       - Point sensors: SNOTEL (NOAA snow/precip/temp)
       - Output: CSV at project_dir/observations/{type}/processed/

    4. EM-Earth Supplementary Data (acquire_em_earth_forcings)
       - Gridded ERA5 re-analysis supplementing point/coarse data
       - Subsetting: Via bounding box
       - Averaging: Spatial mean over domain
       - Output: NetCDF at project_dir/forcing/em_earth_supplementary/

Configuration Parameters:
    Data Source Selection:
        domain.data_access: 'CLOUD' or 'MAF' (default: 'MAF')
        domain.dem_source: 'merit_hydro', 'copernicus', 'copdem90', 'fabdem', 'nasadem', 'srtm', 'etopo', 'mapzen', 'alos'
        domain.land_class_source: 'modis', 'usgs_nlcd' (cloud only)
        domain.bounding_box_coords: 'lat_min/lon_min/lat_max/lon_max'

    Download Flags:
        domain.download_dem: Enable DEM acquisition (default: True)
        domain.download_soil: Enable soil class acquisition (default: True)
        domain.download_landcover: Enable land cover acquisition (default: True)

    Observation Sources:
        optimization.observation_variables: List of variables to download
        evaluation.targets: Evaluation targets (e.g., 'streamflow')

    MAF Configuration:
        domain.hpc_account: HPC account for job submission
        domain.hpc_cache_dir: HPC cache directory
        domain.hpc_job_timeout: Max seconds to wait for jobs

Caching and Error Handling:
    Raw Forcing Cache:
    - RawForcingCache manages downloaded forcing files
    - TTL: Files cached for configurable duration (default: 30 days)
    - Validation: Checksum-based integrity checking
    - Fallback: Automatic re-download if cache corrupted

    Error Recovery:
    - Network failures: Retry with exponential backoff
    - Partial downloads: Cleanup and retry
    - Missing data: Warn and continue with available sources
    - Configuration errors: Validate early and report clearly

Examples:
    >>> # Create service and run all acquisitions
    >>> from symfluence.data.acquisition.acquisition_service import AcquisitionService
    >>> acq = AcquisitionService(config, logger, reporting_manager=reporter)
    >>> acq.acquire_attributes()
    >>> acq.acquire_forcings()
    >>> acq.acquire_observations()
    >>> acq.acquire_em_earth_forcings()

    >>> # Cloud-only mode (faster for small domains)
    >>> # Set config.domain.data_access = 'CLOUD'
    >>> acq.acquire_attributes()

    >>> # MAF mode (for large domains on HPC)
    >>> # Set config.domain.data_access = 'MAF'
    >>> acq.acquire_attributes()

References:
    - MERIT-Hydro: Yamazaki et al. (2019) Global Hydrology, Earth System Science
    - Copernicus DEM: https://copernicus-dem-30m.s3.amazonaws.com/
    - FABDEM: Hawker et al. (2022) Scientific Data
    - SoilGrids: Poggio et al. (2021) Scientific Data
    - MODIS: Justice et al. (2002) Remote Sensing Reviews
"""

import concurrent.futures
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr

from symfluence.core.mixins import ConfigurableMixin
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.data.acquisition.cloud_downloader import CloudForcingDownloader, check_cloud_access_availability
from symfluence.data.acquisition.maf_pipeline import datatoolRunner, gistoolRunner
from symfluence.data.cache import RawForcingCache
from symfluence.data.utils.variable_utils import VariableHandler
from symfluence.geospatial.raster_utils import calculate_landcover_mode

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class AcquisitionService(ConfigurableMixin):
    """Unified data acquisition service for all SYMFLUENCE data needs.

    High-level facade orchestrating geospatial attributes, forcing data, and
    observation data acquisition from multiple sources (cloud, HPC, local).
    Provides flexible acquisition modes (CLOUD vs MAF) and handles caching,
    error recovery, and visualization.

    Acquisition Modes:
        CLOUD Mode:
        - Direct HTTP/S3 access to cloud providers
        - Faster for small domains, requires internet access
        - DEM sources: Copernicus GLO-30/90, FABDEM, NASADEM, SRTM, ETOPO, Mapzen, ALOS
        - Forcing: ERA5 (CDS), CARRA/CERRA, AORC, NEX-GDDP
        - Suitable for research, testing, small basins

        MAF Mode:
        - HPC-based via external MAF tools (gistool, datatool)
        - Better for large domains, requires HPC access
        - Same output format as CLOUD mode
        - Handles job queuing and monitoring via Slurm
        - Suitable for operational, large-scale applications

    Data Acquisition Methods:
        acquire_attributes(): Geospatial attributes (DEM, soil, landcover)
        acquire_forcings(): Meteorological forcing data (ERA5, CARRA, etc.)
        acquire_observations(): Validation data (streamflow, GRACE, SNOTEL, etc.)
        acquire_em_earth_forcings(): Supplementary forcing from EM-Earth

    Key Features:
        - Multi-source geospatial data with automatic fallbacks
        - Caching with TTL and checksum-based validation
        - Parallel downloading where supported
        - Progress visualization via reporting_manager
        - Comprehensive error handling and logging
        - Configuration-driven mode selection

    Attributes:
        config: Typed SymfluenceConfig instance
        logger: Logger for acquisition progress tracking
        data_dir: Root data directory (from config.system.data_dir)
        domain_name: Domain identifier (from config.domain.name)
        project_dir: Project-specific directory (data_dir/domain_{domain_name})
        reporting_manager: Optional visualization manager
        variable_handler: VariableHandler for dataset-specific unit conversion

    Configuration:
        domain.data_access: 'CLOUD' or 'MAF' (default: 'MAF')
        domain.dem_source: DEM provider ('merit_hydro', 'copernicus', 'copdem90', 'fabdem', 'nasadem', 'srtm', 'etopo', 'mapzen', 'alos')
        domain.land_class_source: Land cover provider ('modis', 'usgs_nlcd')
        domain.download_dem: Enable DEM acquisition (default: True)
        domain.download_soil: Enable soil class (default: True)
        domain.download_landcover: Enable land cover (default: True)

    Examples:
        >>> # Create service with config and logger
        >>> acq = AcquisitionService(config, logger, reporting_manager=reporter)

        >>> # Run complete acquisition workflow
        >>> acq.acquire_attributes()   # DEM, soil, landcover
        >>> acq.acquire_forcings()     # ERA5, CARRA, etc.
        >>> acq.acquire_observations() # Streamflow, GRACE, etc.
        >>> acq.acquire_em_earth_forcings()  # Supplementary data

        >>> # Cloud-only mode (small domain)
        >>> config.domain.data_access = 'CLOUD'
        >>> acq.acquire_attributes()

        >>> # MAF mode (large domain on HPC)
        >>> config.domain.data_access = 'MAF'
        >>> acq.acquire_forcings()

    See Also:
        CloudForcingDownloader: Cloud-based data source handlers
        gistoolRunner: HPC geospatial data extraction
        datatoolRunner: HPC forcing data extraction
        RawForcingCache: Forcing data caching system
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        reporting_manager: Any = None
    ):
        # Set up typed config via ConfigurableMixin
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        # Backward compatibility alias
        self.config = self._config

        self.logger = logger
        self.reporting_manager = reporting_manager
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name)
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        self._auto_bbox_logged = False

    def _resolve_bounding_box(self, purpose: str) -> str:
        """Resolve BOUNDING_BOX_COORDS with auto-derivation for point domains.

        If the user set ``BOUNDING_BOX_COORDS`` explicitly, that value wins.
        Otherwise, for point domains (``DOMAIN_DEFINITION_METHOD: point``) with
        ``POUR_POINT_COORDS`` set, a small square bbox is derived from the
        point using ``POINT_BUFFER_DISTANCE`` (defaults to 0.01°, ~1 km).

        Args:
            purpose: Short label ("attributes" or "forcing") used in error
                and log messages so users know which call site failed.

        Returns:
            The resolved bbox string in "north/west/south/east" order.

        Raises:
            ValueError: If no bbox is provided and auto-derivation does not
                apply (non-point domain, or point domain without coords).
        """
        bbox_str = self._get_config_value(
            lambda: self.config.domain.bounding_box_coords, default=None
        )
        if bbox_str:
            return bbox_str

        definition_method = str(self._get_config_value(
            lambda: self.config.domain.definition_method, default=''
        )).lower()
        pour_point = self._get_config_value(
            lambda: self.config.domain.pour_point_coords, default=None
        )

        if definition_method == 'point' and pour_point:
            try:
                lat_str, lon_str = str(pour_point).split('/')
                lat, lon = float(lat_str), float(lon_str)
            except (ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Cannot auto-derive BOUNDING_BOX_COORDS for {purpose}: "
                    f"POUR_POINT_COORDS='{pour_point}' is not in 'lat/lon' format."
                ) from exc

            buffer = self._get_config_value(
                lambda: self.config.domain.delineation.point_buffer_distance,
                default=None,
                dict_key='POINT_BUFFER_DISTANCE',
            )
            if buffer is None:
                buffer = 0.01  # ~1 km at the equator
            buffer = float(buffer)

            derived = f"{lat + buffer}/{lon - buffer}/{lat - buffer}/{lon + buffer}"
            if not self._auto_bbox_logged:
                self.logger.info(
                    f"BOUNDING_BOX_COORDS not set; auto-derived {derived} "
                    f"from POUR_POINT_COORDS={pour_point} with buffer={buffer}° "
                    f"(point domain). Override by setting BOUNDING_BOX_COORDS "
                    f"explicitly or POINT_BUFFER_DISTANCE."
                )
                self._auto_bbox_logged = True
            return derived

        raise ValueError(
            f"BOUNDING_BOX_COORDS is required for cloud-based {purpose} "
            f"acquisition (DATA_ACCESS: CLOUD) but was not set. Add "
            f"BOUNDING_BOX_COORDS: 'north/west/south/east' to your "
            f"configuration file (e.g. '44.5/-87.9/44.2/-87.5'). "
            f"For point domains, setting POUR_POINT_COORDS alone is enough — "
            f"the bbox is auto-derived using POINT_BUFFER_DISTANCE (default 0.01°)."
        )

    def _run_parallel_tasks(
        self,
        tasks: List[Tuple[str, Callable]],
        desc: str = "Acquiring",
    ) -> Dict[str, Any]:
        """Run acquisition tasks concurrently using ThreadPoolExecutor.

        Args:
            tasks: List of (name, callable) tuples.
            desc: Description for logging.

        Returns:
            Dict mapping task name to result (or exception).
        """
        max_workers = self._get_config_value(
            lambda: self.config.data.max_acquisition_workers, default=3,
        )

        # On macOS, HDF5/netCDF4 have thread-safety issues that cause
        # segfaults when multiple threads perform xarray operations
        # concurrently.  Fall back to serial execution.
        if sys.platform == 'darwin':
            max_workers = 1

        max_workers = min(max_workers, len(tasks))

        results: Dict[str, Any] = {}

        if max_workers <= 1:
            self.logger.info(f"{desc}: {len(tasks)} tasks (serial)")
            for name, func in tasks:
                try:
                    self.logger.info(f"Starting: {name}")
                    results[name] = func()
                    self.logger.info(f"Completed: {name}")
                except (OSError, FileNotFoundError, KeyError, ValueError,
                        TypeError, RuntimeError, ImportError,
                        AttributeError, IndexError) as exc:
                    results[name] = exc
                    self.logger.warning(f"Failed: {name}: {exc}")
            return results

        self.logger.info(
            f"{desc}: {len(tasks)} tasks with {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name: Dict[concurrent.futures.Future, str] = {}
            for name, func in tasks:
                self.logger.info(f"Submitting: {name}")
                future_to_name[executor.submit(func)] = name

            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    self.logger.info(f"Completed: {name}")
                except (OSError, FileNotFoundError, KeyError, ValueError,
                        TypeError, RuntimeError, ImportError,
                        AttributeError, IndexError) as exc:
                    results[name] = exc
                    self.logger.warning(f"Failed: {name}: {exc}")

        return results

    def acquire_attributes(self):
        """Acquire geospatial attributes including DEM, soil, and land cover data."""
        self.logger.info("Starting attribute acquisition")

        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='MAF').upper()
        dem_source = self._get_config_value(lambda: self.config.domain.dem_source, default='merit_hydro').lower()

        dem_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'elevation' / 'dem'
        soilclass_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'soilclass'
        landclass_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'landclass'

        for dir_path in [dem_dir, soilclass_dir, landclass_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        if data_access == 'CLOUD':
            self.logger.info(f"Cloud data access enabled for attributes (DEM_SOURCE: {dem_source})")

            bbox_str = self._resolve_bounding_box("attributes")

            try:
                downloader = CloudForcingDownloader(self.config, self.logger)
                attr_tasks: List[Tuple[str, Callable]] = []

                # --- DEM task ---
                if self._get_config_value(lambda: self.config.domain.download_dem, default=True):
                    def _acquire_dem():
                        if dem_source == 'copernicus':
                            return downloader.download_copernicus_dem()
                        elif dem_source == 'fabdem':
                            return downloader.download_fabdem()
                        elif dem_source == 'nasadem':
                            if self._get_config_value(lambda: self.config.data.geospatial.nasadem.local_dir, dict_key='NASADEM_LOCAL_DIR'):
                                return downloader.download_nasadem_local()
                            raise ValueError("DEM_SOURCE set to 'nasadem' but NASADEM_LOCAL_DIR not configured.")
                        elif dem_source in ('copdem90', 'copernicus_90'):
                            return downloader.download_copernicus_dem_90()
                        elif dem_source == 'srtm':
                            return downloader.download_srtm_dem()
                        elif dem_source == 'etopo':
                            return downloader.download_etopo_dem()
                        elif dem_source == 'mapzen':
                            return downloader.download_mapzen_dem()
                        elif dem_source == 'alos':
                            return downloader.download_alos_dem()
                        elif dem_source == 'merit_hydro':
                            gr = gistoolRunner(self.config, self.logger)
                            bbox = bbox_str.split('/')
                            latlims = f"{bbox[0]},{bbox[2]}"
                            lonlims = f"{bbox[1]},{bbox[3]}"
                            self._acquire_elevation_data(gr, dem_dir, latlims, lonlims)
                            return dem_dir / f"domain_{self.domain_name}_elv.tif"
                        else:
                            # Surface common misconfigurations with an actionable
                            # hint instead of just "unsupported". MERIT-Hydro in
                            # particular looks superficially right — it's in many
                            # of our HPC paper configs — but it's only reachable
                            # via the MAF gistool path, not cloud. If the user
                            # set it with DATA_ACCESS=cloud they need a
                            # cloud-reachable source.
                            lower = str(dem_source).lower()
                            accepted_cloud = [
                                'copernicus', 'copdem90', 'copernicus_90',
                                'fabdem', 'nasadem', 'srtm', 'etopo',
                                'mapzen', 'alos',
                            ]
                            hint = ""
                            if 'merit' in lower:
                                hint = (
                                    " MERIT-Hydro is only available via the MAF "
                                    "gistool path (DATA_ACCESS: hpc). For "
                                    "DATA_ACCESS: cloud, use 'copernicus' "
                                    "(the default) or one of: "
                                    f"{', '.join(accepted_cloud)}."
                                )
                            else:
                                hint = (
                                    f" Accepted cloud DEM sources: "
                                    f"{', '.join(accepted_cloud)}."
                                )
                            raise ValueError(
                                f"Unsupported DEM_SOURCE: '{dem_source}' for "
                                f"DATA_ACCESS: cloud.{hint}"
                            )
                    attr_tasks.append(('DEM', _acquire_dem))
                else:
                    self.logger.info("Skipping DEM acquisition (DOWNLOAD_DEM is False)")

                # --- Soil task ---
                if self._get_config_value(lambda: self.config.domain.download_soil, default=True):
                    attr_tasks.append(('soil', downloader.download_global_soilclasses))
                else:
                    self.logger.info("Skipping soil class acquisition (DOWNLOAD_SOIL is False)")

                # --- Landcover task ---
                if self._get_config_value(lambda: self.config.domain.download_landcover, default=True):
                    land_source = self._get_config_value(lambda: self.config.domain.land_class_source, default='modis').lower()
                    def _acquire_landcover():
                        if land_source == 'modis':
                            return downloader.download_modis_landcover()
                        elif land_source == 'usgs_nlcd':
                            return downloader.download_usgs_landcover()
                        raise ValueError(f"Unsupported LAND_CLASS_SOURCE: '{land_source}'. Supported: 'modis', 'usgs_nlcd'.")
                    attr_tasks.append(('landcover', _acquire_landcover))
                else:
                    self.logger.info("Skipping land cover acquisition (DOWNLOAD_LAND_COVER is False)")

                # --- Glacier task (optional, failure is non-fatal) ---
                if self._get_config_value(lambda: self.config.data.download_glacier_data, default=False):
                    attr_tasks.append(('glacier', downloader.download_glacier_data))

                # Run attribute downloads concurrently
                if attr_tasks:
                    results = self._run_parallel_tasks(attr_tasks, desc="Acquiring attributes")

                    # Re-raise failures for required attributes; glacier is optional
                    for name, result in results.items():
                        if isinstance(result, Exception):
                            if name == 'glacier':
                                self.logger.warning(f"Glacier data acquisition failed: {result}")
                            else:
                                self.logger.error(f"Error during cloud attribute acquisition ({name}): {result}")
                                raise result

                    # Visualization after all downloads complete
                    if self.reporting_manager:
                        elev_file = results.get('DEM')
                        if elev_file and not isinstance(elev_file, Exception) and Path(elev_file).exists():
                            self.reporting_manager.visualize_spatial_coverage(elev_file, 'elevation', 'acquisition')

                        soil_file = results.get('soil')
                        if soil_file and not isinstance(soil_file, Exception) and Path(soil_file).exists():
                            self.reporting_manager.visualize_spatial_coverage(soil_file, 'soil_class', 'acquisition')

                        lc_file = results.get('landcover')
                        if lc_file and not isinstance(lc_file, Exception) and Path(lc_file).exists():
                            self.reporting_manager.visualize_spatial_coverage(lc_file, 'land_class', 'acquisition')

            except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
                self.logger.error(f"Error during cloud attribute acquisition: {e}")
                raise
            except (ImportError, AttributeError, IndexError) as e:
                self.logger.error(f"Error during cloud attribute acquisition: {e}")
                raise

        else:
            self.logger.info("Using traditional MAF attribute acquisition workflow")
            gr = gistoolRunner(self.config, self.logger)
            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords).split('/')
            latlims = f"{bbox[0]},{bbox[2]}"
            lonlims = f"{bbox[1]},{bbox[3]}"

            try:
                self._acquire_elevation_data(gr, dem_dir, latlims, lonlims)
                self._acquire_landcover_data(gr, landclass_dir, latlims, lonlims)
                self._acquire_soilclass_data(gr, soilclass_dir, latlims, lonlims)
                self.logger.info("Attribute acquisition completed successfully")

                if self.reporting_manager:
                    # Attempt to visualize acquired files
                    try:
                        dem_file = dem_dir / f"domain_{self.domain_name}_elv.tif"
                        if dem_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(dem_file, 'elevation', 'acquisition')

                        land_file = landclass_dir / f"domain_{self.domain_name}_land_classes.tif"
                        if land_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(land_file, 'land_class', 'acquisition')

                        soil_file = soilclass_dir / f"domain_{self.domain_name}_soil_classes.tif"
                        if soil_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(soil_file, 'soil_class', 'acquisition')
                    except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e_viz:
                        self.logger.warning(f"Failed to visualize MAF attributes: {e_viz}")
                    except (ImportError, AttributeError, IndexError) as e_viz:
                        self.logger.warning(f"Failed to visualize MAF attributes: {e_viz}")

            except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
                self.logger.error(f"Error during attribute acquisition: {e}")
                raise
            except (ImportError, AttributeError, IndexError) as e:
                self.logger.error(f"Error during attribute acquisition: {e}")
                raise

    def _acquire_elevation_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        self.logger.info("Acquiring elevation data")
        gistool_command = gistool_runner.create_gistool_command(
            dataset='MERIT-Hydro',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables='elv'
        )
        gistool_runner.execute_gistool_command(gistool_command)

    def _acquire_landcover_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        self.logger.info("Acquiring land cover data")
        start_year = 2001
        end_year = 2020
        modis_var = "MCD12Q1.006"

        gistool_command = gistool_runner.create_gistool_command(
            dataset='MODIS',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables=modis_var,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-01-01"
        )
        gistool_runner.execute_gistool_command(gistool_command)

        land_name = self._get_config_value(lambda: self.config.domain.land_class_name, default='default')
        if land_name == 'default':
            land_name = f"domain_{self.domain_name}_land_classes.tif"

        if start_year != end_year:
            input_dir = output_dir / modis_var
            output_file = output_dir / land_name
            self.logger.info("Calculating land cover mode across years")
            calculate_landcover_mode(input_dir, output_file, start_year, end_year, self.domain_name)

    def _acquire_soilclass_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        self.logger.info("Acquiring soil class data")
        gistool_command = gistool_runner.create_gistool_command(
            dataset='soil_class',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables='soil_classes'
        )
        gistool_runner.execute_gistool_command(gistool_command)

    def _expected_forcing_times(self, dataset: str) -> Optional[pd.DatetimeIndex]:
        resolution_hours = {
            "CARRA": 1,
            "CERRA": 3,
        }
        dataset_key = dataset.upper()
        if dataset_key not in resolution_hours:
            return None

        start = pd.to_datetime(self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START'))
        end = pd.to_datetime(self._get_config_value(lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END'))
        if pd.isna(start) or pd.isna(end) or end < start:
            return None

        freq = f"{resolution_hours[dataset_key]}h"
        return pd.date_range(start, end, freq=freq)

    def _cached_forcing_has_expected_times(
        self, cached_file: Path, expected_times: pd.DatetimeIndex
    ) -> bool:
        try:
            with xr.open_dataset(cached_file) as ds:
                if "time" not in ds:
                    return False
                actual_times = pd.to_datetime(ds["time"].values)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.logger.warning(f"Failed to validate cached forcing file {cached_file}: {exc}")
            return False

        if len(actual_times) < len(expected_times):
            return False

        return actual_times[0] <= expected_times[0] and actual_times[-1] >= expected_times[-1]

    def acquire_forcings(self):
        """Acquire forcing data for the model simulation."""
        self.logger.info("Starting forcing data acquisition")

        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='MAF').upper()
        forcing_dataset = self._get_config_value(lambda: self.config.forcing.dataset, default='').upper()

        # Compute effective forcing time step locally (never mutate shared config).
        # CARRA/CERRA default to 10800s when the user hasn't explicitly changed
        # from the generic default (3600s).
        configured_ts = self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            dict_key='FORCING_TIME_STEP_SIZE',
        )
        _GENERIC_DEFAULT = 3600
        if forcing_dataset in {"CARRA", "CERRA"} and (not configured_ts or configured_ts == _GENERIC_DEFAULT):
            self._effective_forcing_time_step = 10800
            self.logger.info(
                f"Using effective FORCING_TIME_STEP_SIZE=10800s for {forcing_dataset} "
                f"(configured value was {configured_ts or 'unset'})"
            )
        else:
            self._effective_forcing_time_step = configured_ts or _GENERIC_DEFAULT

        if data_access == 'CLOUD':
            self.logger.info(f"Cloud data access enabled for {forcing_dataset}")

            if not check_cloud_access_availability(forcing_dataset, self.logger):
                raise ValueError(f"Dataset '{forcing_dataset}' does not support DATA_ACCESS: cloud.")

            bbox = self._resolve_bounding_box("forcing")

            raw_data_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data'
            raw_data_dir.mkdir(parents=True, exist_ok=True)

            # Initialize cache
            cache_root = self.data_dir / 'cache' / 'raw_forcing'
            cache = RawForcingCache(
                cache_root=cache_root,
                max_size_gb=self._get_config_value(lambda: self.config.data.forcing_cache_size_gb, default=3.0, dict_key='FORCING_CACHE_SIZE_GB'),
                ttl_days=self._get_config_value(lambda: self.config.data.forcing_cache_ttl_days, default=30, dict_key='FORCING_CACHE_TTL_DAYS'),
                enable_checksum=self._get_config_value(lambda: self.config.data.forcing_cache_checksum, default=True, dict_key='FORCING_CACHE_CHECKSUM')
            )

            # Generate cache key (reuse the bbox resolved above so a point
            # domain's auto-derived bbox ends up in the cache key too).
            time_start = self._get_config_value(lambda: self.config.domain.time_start)
            time_end = self._get_config_value(lambda: self.config.domain.time_end)

            # Check for dataset-specific variable configuration (e.g., HRRR_VARS, AORC_VARS)
            # Fall back to generic FORCING_VARIABLES if not found
            dataset_vars_key = f"{forcing_dataset.upper()}_VARS"
            variables = self._get_config_value(lambda: None, default=None, dict_key=dataset_vars_key)
            if variables is None:
                variables = self._get_config_value(lambda: self.config.forcing.variables, dict_key='FORCING_VARIABLES')

            cache_key = cache.generate_cache_key(
                dataset=forcing_dataset,
                bbox=bbox,
                time_start=time_start,
                time_end=time_end,
                variables=variables if isinstance(variables, list) else None
            )

            # Check cache first
            cached_file = cache.get(cache_key)
            if cached_file and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
                expected_times = self._expected_forcing_times(forcing_dataset)
                if expected_times is not None and not self._cached_forcing_has_expected_times(
                    cached_file, expected_times
                ):
                    self.logger.warning(
                        f"Cached forcing data {cached_file} does not cover the requested time range; "
                        "re-downloading from source."
                    )
                    cached_file = None

            if cached_file and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
                self.logger.info(f"✓ Using cached forcing data: {cache_key}")
                # Copy from cache to project directory
                import shutil
                output_file = raw_data_dir / cached_file.name
                shutil.copy(cached_file, output_file)
                self.logger.info(f"✓ Copied cached file to: {output_file}")
            else:
                # Cache miss - download from source
                if cached_file:
                    self.logger.info("FORCE_DOWNLOAD enabled - skipping cache")
                else:
                    self.logger.info("Cache miss - downloading from source")

                try:
                    downloader = CloudForcingDownloader(self.config, self.logger)
                    output_file = downloader.download_forcing_data(raw_data_dir)
                    self.logger.info(f"✓ Cloud forcing data acquisition completed: {output_file}")

                    # Handle case where output is a directory (e.g. non-aggregated files)
                    if output_file.is_dir():
                        self.logger.info("Output is a directory - skipping single-file caching and visualization")

                        # Find a sample file for visualization
                        sample_files = list(output_file.glob("*.nc"))
                        if sample_files:
                            sample_file = sample_files[0]
                            if self.reporting_manager:
                                self.reporting_manager.visualize_spatial_coverage(sample_file, 'forcing_sample', 'acquisition')

                        self.logger.warning("Caching is not currently supported for non-aggregated forcing files. Skipping cache.")
                    else:
                        if self.reporting_manager and output_file and output_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(output_file, 'forcing_sample', 'acquisition')

                        # Store in cache
                        try:
                            cache.put(
                                cache_key=cache_key,
                                file_path=output_file,
                                metadata={
                                    'dataset': forcing_dataset,
                                    'bbox': bbox,
                                    'time_range': f"{time_start} to {time_end}",
                                    'variables': variables if isinstance(variables, list) else str(variables),
                                    'domain_name': self.domain_name
                                }
                            )
                        except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as cache_error:
                            self.logger.warning(f"Failed to cache downloaded file: {cache_error}")
                            # Don't fail the acquisition if caching fails
                        except (ImportError, AttributeError, IndexError) as cache_error:
                            self.logger.warning(f"Failed to cache downloaded file: {cache_error}")

                except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
                    self.logger.error(f"Error during cloud data acquisition: {e}")
                    raise
                except (ImportError, AttributeError, IndexError) as e:
                    self.logger.error(f"Error during cloud data acquisition: {e}")
                    raise

        else:
            self.logger.info("Using traditional MAF data acquisition workflow")

            if forcing_dataset not in datatoolRunner.supported_datasets():
                supported = ', '.join(sorted(datatoolRunner.supported_datasets()))
                raise ValueError(
                    f"Dataset '{forcing_dataset}' is not supported with DATA_ACCESS: MAF. "
                    f"Supported datatool datasets: {supported}. "
                    f"Try DATA_ACCESS: cloud for this dataset instead."
                )

            dr = datatoolRunner(self.config, self.logger)
            raw_data_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data'
            raw_data_dir.mkdir(parents=True, exist_ok=True)

            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords).split('/')
            latlims = f"{bbox[2]},{bbox[0]}"
            lonlims = f"{bbox[1]},{bbox[3]}"

            variables = self._get_config_value(lambda: self.config.forcing.variables, default='default')
            if variables == 'default':
                variables = self.variable_handler.get_dataset_variables(
                    dataset=self._get_config_value(lambda: self.config.forcing.dataset)
                )

            try:
                datatool_command = dr.create_datatool_command(
                    dataset=self._get_config_value(lambda: self.config.forcing.dataset),
                    output_dir=raw_data_dir,
                    lat_lims=latlims,
                    lon_lims=lonlims,
                    variables=variables,
                    start_date=self._get_config_value(lambda: self.config.domain.time_start),
                    end_date=self._get_config_value(lambda: self.config.domain.time_end)
                )
                dr.execute_datatool_command(datatool_command)
                self.logger.info("Primary forcing data acquisition completed successfully")

                if self.reporting_manager:
                    # Find a sample forcing file
                    sample_files = list(raw_data_dir.glob("*.nc"))
                    if sample_files:
                        self.reporting_manager.visualize_spatial_coverage(sample_files[0], 'forcing_sample', 'acquisition')

            except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
                self.logger.error(f"Error during forcing data acquisition: {e}")
                raise
            except (ImportError, AttributeError, IndexError) as e:
                self.logger.error(f"Error during forcing data acquisition: {e}")
                raise

        if self._get_config_value(lambda: self.config.forcing.supplement, default=False):
            self.logger.info("SUPPLEMENT_FORCING enabled - acquiring EM-Earth data")
            self.acquire_em_earth_forcings()

    def acquire_observations(self):
        """
        Acquire additional observations based on configuration.
        This handles registry-based observations (GRACE, MODIS, etc.)
        that require an 'acquire' step before processing.
        """
        from symfluence.core.registries import R

        additional_obs = self._get_config_value(lambda: self.config.data.additional_observations) or []
        if isinstance(additional_obs, str):
            additional_obs = [o.strip() for o in additional_obs.split(',')]
        elif additional_obs is None:
            additional_obs = []

        # Track which observations are PRIMARY (configured via
        # streamflow_data_provider — failure here breaks downstream
        # calibration / benchmarking and must NOT be silently swallowed
        # as a warning). All other observations remain best-effort.
        # Co-author NB/NV reported a calibration crash where the workflow
        # said the obs step was complete (exit 0) but no streamflow file
        # had been written — root cause: WSC handler failed silently
        # because HYDAT was not installed.
        primary_obs: set = set()

        # Auto-detect observation types based on config flags (matching process_observed_data logic)
        streamflow_provider = (self._get_config_value(lambda: self.config.data.streamflow_data_provider) or '').upper()
        if streamflow_provider == 'USGS' and 'USGS_STREAMFLOW' not in additional_obs:
            additional_obs.append('USGS_STREAMFLOW')
            primary_obs.add('USGS_STREAMFLOW')
        elif streamflow_provider == 'WSC' and 'WSC_STREAMFLOW' not in additional_obs:
            additional_obs.append('WSC_STREAMFLOW')
            primary_obs.add('WSC_STREAMFLOW')
        elif streamflow_provider == 'SMHI' and 'SMHI_STREAMFLOW' not in additional_obs:
            additional_obs.append('SMHI_STREAMFLOW')
            primary_obs.add('SMHI_STREAMFLOW')
        elif streamflow_provider == 'LAMAH_ICE' and 'LAMAH_ICE_STREAMFLOW' not in additional_obs:
            additional_obs.append('LAMAH_ICE_STREAMFLOW')
            primary_obs.add('LAMAH_ICE_STREAMFLOW')
        elif streamflow_provider:
            # Provider was configured but doesn't match a known shortcut —
            # mark whatever already in additional_obs that matches as primary.
            for obs in additional_obs:
                if 'STREAMFLOW' in str(obs).upper():
                    primary_obs.add(str(obs).upper())

        # Check for USGS Groundwater download
        download_usgs_gw = self._get_config_value(lambda: self.config.evaluation.usgs_gw.download, default=False, dict_key='DOWNLOAD_USGS_GW')
        if isinstance(download_usgs_gw, str):
            download_usgs_gw = download_usgs_gw.lower() == 'true'
        if download_usgs_gw and 'USGS_GW' not in additional_obs:
            additional_obs.append('USGS_GW')

        # Check for MODIS Snow
        if self._get_config_value(lambda: self.config.evaluation.modis_snow.download, default=False, dict_key='DOWNLOAD_MODIS_SNOW') and 'MODIS_SNOW' not in additional_obs:
            additional_obs.append('MODIS_SNOW')

        # Check for SNOTEL
        download_snotel = self._get_config_value(lambda: self.config.evaluation.snotel.download, default=False, dict_key='DOWNLOAD_SNOTEL')
        if isinstance(download_snotel, str):
            download_snotel = download_snotel.lower() == 'true'
        if download_snotel and 'SNOTEL' not in additional_obs:
            additional_obs.append('SNOTEL')

        # Check for GRACE
        if self._get_config_value(lambda: self.config.evaluation.grace.download, default=False, dict_key='DOWNLOAD_GRACE') and 'GRACE' not in additional_obs:
            additional_obs.append('GRACE')

        # Check for MOD16 ET (based on ET_OBS_SOURCE or OPTIMIZATION_TARGET)
        et_obs_source = str(self._get_config_value(lambda: self.config.evaluation.et_obs_source, default='', dict_key='ET_OBS_SOURCE')).lower()
        optimization_target = str(self._get_config_value(lambda: self.config.optimization.target, default='', dict_key='OPTIMIZATION_TARGET')).lower()
        if et_obs_source in ('mod16', 'modis', 'modis_et', 'mod16a2'):
            if 'MODIS_ET' not in additional_obs and 'MOD16' not in additional_obs:
                additional_obs.append('MODIS_ET')
        elif optimization_target == 'et' and not et_obs_source:
            # Default to MOD16 if ET calibration without explicit source
            if 'MODIS_ET' not in additional_obs:
                additional_obs.append('MODIS_ET')

        # Check for FLUXNET data (based on config flags or ET_OBS_SOURCE)
        if self._get_config_value(lambda: self.config.evaluation.fluxnet.download, default=False, dict_key='DOWNLOAD_FLUXNET') or et_obs_source == 'fluxnet':
            if 'FLUXNET' not in additional_obs and 'FLUXNET_ET' not in additional_obs:
                additional_obs.append('FLUXNET_ET')

        # Check for multi-source ET (both FLUXNET and MOD16)
        if self._get_config_value(lambda: self.config.evaluation.multi_source_et, default=False, dict_key='MULTI_SOURCE_ET'):
            if 'FLUXNET_ET' not in additional_obs and 'FLUXNET' not in additional_obs:
                additional_obs.append('FLUXNET_ET')
            if 'MODIS_ET' not in additional_obs and 'MOD16' not in additional_obs:
                additional_obs.append('MODIS_ET')

        if not additional_obs:
            return

        self.logger.info(f"Acquiring additional observations: {additional_obs}")

        # Build task list for parallel execution
        tasks: List[Tuple[str, Callable]] = []
        for obs_type in additional_obs:
            if obs_type in R.observation_handlers:
                handler_cls = R.observation_handlers.get(obs_type)
                if handler_cls:
                    handler = handler_cls(self.config, self.logger)
                    tasks.append((obs_type, handler.acquire))
            else:
                self.logger.debug(f"Skipping acquisition for {obs_type}: no registry handler")

        if tasks:
            results = self._run_parallel_tasks(tasks, desc="Acquiring observations")
            primary_failures: List[Tuple[str, Exception]] = []
            for name, result in results.items():
                if isinstance(result, Exception):
                    if name.upper() in primary_obs:
                        # Primary streamflow provider failure — never silent.
                        primary_failures.append((name, result))
                        self.logger.error(
                            f"Failed to acquire primary streamflow observation "
                            f"{name}: {result}"
                        )
                    else:
                        # Optional observation (GRACE, SNOTEL, etc.) — best-effort.
                        self.logger.warning(
                            f"Failed to acquire optional observation {name}: {result}"
                        )

            if primary_failures:
                # Surface a single actionable error covering all failed primaries.
                # Downstream calibration / benchmarking depend on this file; if
                # it's missing they'll crash later with confusing 'No such file'
                # errors. Fail here instead so the user knows where to look.
                names = ', '.join(name for name, _ in primary_failures)
                first_msg = str(primary_failures[0][1])
                hint = ""
                if 'HYDAT' in first_msg or 'WSC' in names.upper():
                    hint = (
                        " Hint: WSC streamflow requires the HYDAT SQLite "
                        "database (Hydat.sqlite3) to be installed and "
                        "DATATOOL_DATASET_ROOT pointed at its parent. "
                        "See https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/"
                    )
                raise ValueError(
                    f"Primary streamflow observation acquisition failed for: "
                    f"{names}. Downstream calibration / benchmarking / decision "
                    f"analyses will fail without this data, so the workflow stops "
                    f"here rather than producing silent zeros.{hint} "
                    f"Original error: {first_msg}"
                )

    def acquire_em_earth_forcings(self):
        """Acquire EM-Earth precipitation and temperature data."""
        self.logger.info("Starting EM-Earth forcing data acquisition")

        try:
            em_earth_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data_em_earth'
            em_earth_dir.mkdir(parents=True, exist_ok=True)

            em_region = self._get_config_value(lambda: self.config.forcing.em_earth.region, default='NorthAmerica', dict_key='EM_EARTH_REGION')
            em_earth_prcp_dir = self._get_config_value(lambda: self.config.forcing.em_earth.prcp_dir, default=f"/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/prcp/{em_region}", dict_key='EM_EARTH_PRCP_DIR')
            em_earth_tmean_dir = self._get_config_value(lambda: self.config.forcing.em_earth.tmean_dir, default=f"/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/tmean/{em_region}", dict_key='EM_EARTH_TMEAN_DIR')

            if not Path(em_earth_prcp_dir).exists():
                raise FileNotFoundError(f"EM-Earth precipitation directory not found: {em_earth_prcp_dir}")
            if not Path(em_earth_tmean_dir).exists():
                raise FileNotFoundError(f"EM-Earth temperature directory not found: {em_earth_tmean_dir}")

            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords)
            bbox_parts = bbox.split('/')
            lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min

            self.logger.info(f"Watershed bounding box: {bbox}")
            self.logger.info(f"Watershed size: {lat_range:.4f}° x {lon_range:.4f}°")

            min_bbox_size = self._get_config_value(lambda: self.config.forcing.em_earth.min_bbox_size, default=0.1, dict_key='EM_EARTH_MIN_BBOX_SIZE')
            if lat_range < min_bbox_size or lon_range < min_bbox_size:
                self.logger.warning("Very small watershed detected. EM-Earth processing will use spatial averaging.")

            try:
                start_date = datetime.strptime(self._get_config_value(lambda: self.config.domain.time_start), '%Y-%m-%d %H:%M')
                end_date = datetime.strptime(self._get_config_value(lambda: self.config.domain.time_end), '%Y-%m-%d %H:%M')
            except ValueError as e:
                raise ValueError(f"Invalid date format in configuration: {str(e)}") from e

            self.logger.info(f"Processing EM-Earth data for period: {start_date} to {end_date}")

            year_months = self._generate_year_month_list(start_date, end_date)

            if not year_months:
                raise ValueError("No valid year-month combinations found for the specified time period")

            # Build month-processing tasks for parallel execution
            month_tasks: List[Tuple[str, Callable]] = []
            for year_month in year_months:
                def _make_month_task(ym=year_month):
                    return self._process_em_earth_month(
                        ym, em_earth_prcp_dir, em_earth_tmean_dir, em_earth_dir, bbox
                    )
                month_tasks.append((year_month, _make_month_task))

            results = self._run_parallel_tasks(month_tasks, desc="Processing EM-Earth months")

            processed_files = []
            failed_months = []
            for year_month in year_months:
                result = results.get(year_month)
                if isinstance(result, Exception):
                    failed_months.append(year_month)
                elif result is not None:
                    processed_files.append(result)
                else:
                    failed_months.append(year_month)

            if not processed_files:
                raise ValueError("No EM-Earth data files were successfully processed")

            success_rate = len(processed_files) / len(year_months) * 100
            self.logger.info(f"EM-Earth forcing data acquisition completed. Success rate: {success_rate:.1f}%")

            if failed_months and success_rate < 50:
                raise ValueError(f"EM-Earth processing success rate too low ({success_rate:.1f}%).")

            if self.reporting_manager and processed_files:
                # Visualize one sample file
                self.reporting_manager.visualize_spatial_coverage(processed_files[0], 'em_earth_sample', 'acquisition')

        except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.error(f"Error during EM-Earth forcing data acquisition: {e}")
            raise
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.error(f"Error during EM-Earth forcing data acquisition: {e}")
            raise

    def _generate_year_month_list(self, start_date: datetime, end_date: datetime) -> List[str]:
        year_months = []
        current_date = start_date.replace(day=1)

        while current_date <= end_date:
            year_month = current_date.strftime('%Y%m')
            year_months.append(year_month)
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        return year_months

    def _process_em_earth_month(self, year_month: str, prcp_dir: str, tmean_dir: str,
                               output_dir: Path, bbox: str) -> Optional[Path]:
        em_region = self._get_config_value(lambda: self.config.forcing.em_earth.region, default='NorthAmerica', dict_key='EM_EARTH_REGION')

        prcp_pattern = f"EM_Earth_deterministic_hourly_{em_region}_{year_month}.nc"
        tmean_pattern = f"EM_Earth_deterministic_hourly_{em_region}_{year_month}.nc"

        prcp_file = Path(prcp_dir) / prcp_pattern
        tmean_file = Path(tmean_dir) / tmean_pattern

        if not prcp_file.exists():
            self.logger.warning(f"EM-Earth precipitation file not found: {prcp_file}")
            return None
        if not tmean_file.exists():
            self.logger.warning(f"EM-Earth temperature file not found: {tmean_file}")
            return None

        output_file = output_dir / f"watershed_subset_{year_month}.nc"

        if output_file.exists() and not self._get_config_value(lambda: self.config.system.force_run_all_steps, default=False, dict_key='FORCE_RUN_ALL_STEPS'):
            self.logger.info(f"EM-Earth file already exists, skipping: {output_file}")
            return output_file

        try:
            self._process_em_earth_data(str(prcp_file), str(tmean_file), str(output_file), bbox)
            return output_file
        except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.error(f"Error processing EM-Earth data for {year_month}: {str(e)}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.error(f"Error processing EM-Earth data for {year_month}: {e}")
            return None

    def _process_em_earth_data(self, prcp_file: str, tmean_file: str, output_file: str, bbox: str):
        """Process EM-Earth precipitation and temperature data for a specific bounding box."""
        import xarray as xr

        bbox_parts = bbox.split('/')
        if len(bbox_parts) != 4:
            raise ValueError(f"Invalid bounding box format: {bbox}. Expected lat_max/lon_min/lat_min/lon_max")

        lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        min_bbox_size = 0.1
        original_bbox = (lat_min, lat_max, lon_min, lon_max)

        if lat_range < min_bbox_size or lon_range < min_bbox_size:
            self.logger.warning(f"Very small watershed detected (lat: {lat_range:.4f}°, lon: {lon_range:.4f}°)")

            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2

            lat_min_extract = lat_center - min_bbox_size/2
            lat_max_extract = lat_center + min_bbox_size/2
            lon_min_extract = lon_center - min_bbox_size/2
            lon_max_extract = lon_center + min_bbox_size/2
        else:
            lat_min_extract, lat_max_extract = lat_min, lat_max
            lon_min_extract, lon_max_extract = lon_min, lon_max

        try:
            prcp_ds = xr.open_dataset(prcp_file)
            tmean_ds = xr.open_dataset(tmean_file)
        except (OSError, FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
            raise ValueError(f"Error opening EM-Earth files: {str(e)}") from e

        try:
            if lon_min_extract > lon_max_extract:
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_extract) & (prcp_ds.lat <= lat_max_extract) &
                    ((prcp_ds.lon >= lon_min_extract) | (prcp_ds.lon <= lon_max_extract)), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_extract) & (tmean_ds.lat <= lat_max_extract) &
                    ((tmean_ds.lon >= lon_min_extract) | (tmean_ds.lon <= lon_max_extract)), drop=True
                )
            else:
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_extract) & (prcp_ds.lat <= lat_max_extract) &
                    (prcp_ds.lon >= lon_min_extract) & (prcp_ds.lon <= lon_max_extract), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_extract) & (tmean_ds.lat <= lat_max_extract) &
                    (tmean_ds.lon >= lon_min_extract) & (tmean_ds.lon <= lon_max_extract), drop=True
                )

            if prcp_subset.sizes.get('lat', 0) == 0 or prcp_subset.sizes.get('lon', 0) == 0:
                self.logger.warning("No precipitation data found with initial expansion, trying larger expansion")
                larger_expand = 0.2
                lat_center = (original_bbox[0] + original_bbox[1]) / 2
                lon_center = (original_bbox[2] + original_bbox[3]) / 2

                lat_min_large = lat_center - larger_expand
                lat_max_large = lat_center + larger_expand
                lon_min_large = lon_center - larger_expand
                lon_max_large = lon_center + larger_expand

                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_large) & (prcp_ds.lat <= lat_max_large) &
                    (prcp_ds.lon >= lon_min_large) & (prcp_ds.lon <= lon_max_large), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_large) & (tmean_ds.lat <= lat_max_large) &
                    (tmean_ds.lon >= lon_min_large) & (tmean_ds.lon <= lon_max_large), drop=True
                )

            if prcp_subset.sizes.get('lat', 0) == 0 or prcp_subset.sizes.get('lon', 0) == 0:
                raise ValueError("No precipitation data found within the expanded bounding box.")
            if tmean_subset.sizes.get('lat', 0) == 0 or tmean_subset.sizes.get('lon', 0) == 0:
                raise ValueError("No temperature data found within the expanded bounding box.")

        except (OSError, FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
            raise ValueError(f"Error subsetting EM-Earth data: {str(e)}") from e

        if (lat_min_extract, lat_max_extract, lon_min_extract, lon_max_extract) != original_bbox:
            self.logger.info("Computing spatial average over expanded area to represent the small watershed")
            prcp_subset = prcp_subset.mean(dim=['lat', 'lon'], skipna=True, keep_attrs=True)
            tmean_subset = tmean_subset.mean(dim=['lat', 'lon'], skipna=True, keep_attrs=True)

            prcp_subset = prcp_subset.expand_dims({'lat': [original_bbox[0] + (original_bbox[1] - original_bbox[0])/2]})
            prcp_subset = prcp_subset.expand_dims({'lon': [original_bbox[2] + (original_bbox[3] - original_bbox[2])/2]})
            tmean_subset = tmean_subset.expand_dims({'lat': [original_bbox[0] + (original_bbox[1] - original_bbox[0])/2]})
            tmean_subset = tmean_subset.expand_dims({'lon': [original_bbox[2] + (original_bbox[3] - original_bbox[2])/2]})

        try:
            merged_ds = xr.Dataset()
            merged_ds = merged_ds.assign_coords({
                'lat': prcp_subset.lat,
                'lon': prcp_subset.lon,
                'time': prcp_subset.time
            })

            for var in prcp_subset.data_vars:
                if 'prcp' in var:
                    merged_ds[var] = prcp_subset[var]

            for var in tmean_subset.data_vars:
                if 'tmean' in var or 'temp' in var:
                    if tmean_subset.sizes.get('lat', 0) < 2 or tmean_subset.sizes.get('lon', 0) < 2:
                        temp_interp = tmean_subset[var].interp(
                            lat=prcp_subset.lat,
                            lon=prcp_subset.lon,
                            method='nearest'
                        )
                    else:
                        temp_interp = tmean_subset[var].interp(
                            lat=prcp_subset.lat,
                            lon=prcp_subset.lon,
                            method='linear'
                        )
                    merged_ds[var] = temp_interp

            is_small_watershed = lat_range < min_bbox_size or lon_range < min_bbox_size
            is_spatially_averaged = (lat_min_extract, lat_max_extract, lon_min_extract, lon_max_extract) != original_bbox

            merged_ds.attrs.update({
                'small_watershed_processing': int(is_small_watershed),
                'spatial_averaging_applied': int(is_spatially_averaged),
                'subset_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            merged_ds.to_netcdf(output_file)

        except (OSError, FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
            raise ValueError(f"Error merging EM-Earth datasets: {str(e)}") from e

        finally:
            prcp_ds.close()
            tmean_ds.close()
