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
from __future__ import annotations

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
from symfluence.core.modeling.variable_utils import VariableHandler
from symfluence.data.acquisition.cloud_downloader import CloudForcingDownloader, check_cloud_access_availability
from symfluence.data.acquisition.maf_pipeline import datatoolRunner, gistoolRunner
from symfluence.data.acquisition.registry import AcquisitionRegistry
from symfluence.data.acquisition.request_planning import expected_forcing_times, forcing_request_facts
from symfluence.data.cache import RawForcingCache
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

        # Point domains may be configured with only POUR_POINT_COORDS; the shared
        # helper on ConfigMixin derives the square extent (single source of truth,
        # also used by the point delineator and acquisition handlers).
        derived = self._resolve_point_bbox()
        if derived:
            if not self._auto_bbox_logged:
                self.logger.info(
                    f"BOUNDING_BOX_COORDS not set; auto-derived {derived} from "
                    f"POUR_POINT_COORDS (point domain). Override by setting "
                    f"BOUNDING_BOX_COORDS explicitly or POINT_BUFFER_DISTANCE."
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
                    self.logger.debug(f"Starting: {name}")
                    results[name] = func()
                    self.logger.debug(f"Completed: {name}")
                except (OSError, FileNotFoundError, KeyError, ValueError,
                        TypeError, RuntimeError, ImportError,
                        AttributeError, IndexError) as exc:
                    results[name] = exc
                    self.logger.warning(f"Failed: {name}: {exc}")
            self._log_task_summary(desc, results)
            return results

        self.logger.info(
            f"{desc}: {len(tasks)} tasks with {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name: Dict[concurrent.futures.Future, str] = {}
            for name, func in tasks:
                self.logger.debug(f"Submitting: {name}")
                future_to_name[executor.submit(func)] = name

            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    self.logger.debug(f"Completed: {name}")
                except (OSError, FileNotFoundError, KeyError, ValueError,
                        TypeError, RuntimeError, ImportError,
                        AttributeError, IndexError) as exc:
                    results[name] = exc
                    self.logger.warning(f"Failed: {name}: {exc}")

        self._log_task_summary(desc, results)
        return results

    def _log_task_summary(self, desc: str, results: Dict[str, Any]) -> None:
        """Log a one-line INFO summary of a task batch (failures already warned)."""
        n_failed = sum(1 for value in results.values() if isinstance(value, Exception))
        n_ok = len(results) - n_failed
        if n_failed:
            self.logger.info(f"{desc}: {n_ok}/{len(results)} tasks succeeded, {n_failed} failed")
        else:
            self.logger.info(f"{desc}: all {len(results)} tasks completed")

    def acquire_attributes(self):
        """Acquire geospatial attributes including DEM, soil, and land cover data."""
        self.logger.info("Starting attribute acquisition")

        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='MAF').upper()
        dem_source = self._get_config_value(lambda: self.config.domain.dem_source, default='copdem90').lower()

        dem_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'elevation' / 'dem'
        soilclass_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'soilclass'
        landclass_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'landclass'

        for dir_path in [dem_dir, soilclass_dir, landclass_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # COMMUNITY follows the CLOUD pathway for attribute acquisition (see
        # acquire_forcings): rasters still come from the registry handlers;
        # the community attribute layer (CAS) plugs in downstream as an
        # attribute processor, not here.
        if data_access in ('CLOUD', 'COMMUNITY'):
            self.logger.info(f"{data_access.capitalize()} data access enabled for attributes (DEM_SOURCE: {dem_source})")

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
                if self._get_config_value(lambda: self.config.data.download_glacier_data, default=False, dict_key='DOWNLOAD_GLACIER_DATA'):
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

            self._acquire_profile_datasets()

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

            self._acquire_profile_datasets()

    def _acquire_profile_datasets(self):
        """Acquire extended datasets based on ATTRIBUTE_PROFILE config.

        Reads the profile name from config and downloads all datasets
        listed in the profile that haven't been individually disabled
        via DOWNLOAD_* override flags.  Failures are non-fatal by
        default (warns and continues).
        """
        from symfluence.data.acquisition.attribute_profiles import PROFILES

        profile_name = self._get_config_value(
            lambda: self.config.domain.attribute_profile,
            default='core',
            dict_key='ATTRIBUTE_PROFILE',
        )
        if isinstance(profile_name, str):
            profile_name = profile_name.lower().strip()

        profile_datasets = PROFILES.get(profile_name, [])
        if not profile_datasets:
            return

        self.logger.info(
            f"Attribute profile '{profile_name}': "
            f"{len(profile_datasets)} extended datasets to acquire"
        )

        tasks: List[Tuple[str, Callable]] = []
        for ds in profile_datasets:
            if ds.config_override_key:
                override = self._get_config_value(
                    lambda: None,
                    default=True,
                    dict_key=ds.config_override_key,
                )
                if isinstance(override, str):
                    override = override.lower() not in ('false', '0', 'no')
                if not override:
                    self.logger.info(
                        f"Skipping {ds.description} "
                        f"({ds.config_override_key} is False)"
                    )
                    continue

            def _make_task(dataset=ds):
                handler = AcquisitionRegistry.get_handler(
                    dataset.handler_name, self.config, self.logger,
                )
                out_dir = (
                    resolve_data_subdir(self.project_dir, 'attributes')
                    / dataset.output_subdir
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                return handler.download(out_dir)

            tasks.append((ds.description, _make_task))

        if not tasks:
            return

        results = self._run_parallel_tasks(
            tasks, desc=f"Acquiring '{profile_name}' profile datasets",
        )

        for name, result in results.items():
            if isinstance(result, Exception):
                ds_info = next(
                    (d for d in profile_datasets if d.description == name),
                    None,
                )
                if ds_info and ds_info.fatal:
                    self.logger.error(
                        f"Required profile dataset failed: {name}: {result}"
                    )
                    raise result
                self.logger.warning(
                    f"Optional profile dataset failed (non-fatal): "
                    f"{name}: {result}"
                )
            else:
                self.logger.info(f"Profile dataset acquired: {name}")

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
        start = self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')
        end = self._get_config_value(lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END')
        return expected_forcing_times(dataset, start, end)

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

    def _forcing_request_facts(self, forcing_dataset: str):
        """Window/variables for a forcing request, resolved exactly once.

        Returns ``(window, variables)`` where ``window`` is an ISO-string
        ``(start, end)`` tuple (or ``None``) and ``variables`` a frozenset of
        requested names (or ``None`` = dataset default).
        """
        time_start = self._get_config_value(lambda: self.config.domain.time_start)
        time_end = self._get_config_value(lambda: self.config.domain.time_end)
        dataset_vars_key = f"{forcing_dataset.upper()}_VARS"
        variables = self._get_config_value(lambda: None, default=None, dict_key=dataset_vars_key)
        defaults = None
        if variables is None:
            defaults = self._get_config_value(lambda: self.config.forcing.variables, dict_key='FORCING_VARIABLES')
        return forcing_request_facts(time_start, time_end, variables, defaults)

    def _maybe_select_protocol_backend(self, forcing_dataset: str):
        """Consult the AcquisitionBackend selector — only once it can matter.

        Returns ``None`` (meaning "use the existing dispatch unchanged")
        unless BOTH hold: (a) at least one non-native backend is registered
        under the protocol (e.g. the cfs plugin's CommunityForcingBackend),
        and (b) the selector resolves the request to that non-native backend.
        Selection honours ``DATA_ACCESS`` priority, per-dataset
        ``<DATASET>_BACKEND`` pins, capability variable/window checks, and the
        parity-gate policy (ungraded datasets refused unless
        ``ALLOW_UNGATED_BACKENDS``). Without a non-native backend installed,
        cloud/community/MAF behavior is bit-identical to the legacy dispatch.
        """
        from symfluence.core.registries import R

        non_native = [k for k in R.acquisition_backends.keys() if k != 'native']
        if not non_native:
            return None

        from symfluence.data.backends.errors import AcquisitionError
        from symfluence.data.backends.selection import select_backend

        window, variables = self._forcing_request_facts(forcing_dataset)
        try:
            backend = select_backend(
                forcing_dataset, self.config,
                variables=variables, window=window, logger=self.logger,
            )
        except AcquisitionError as exc:
            self.logger.info(
                f"Acquisition-backend selection declined for {forcing_dataset} ({exc}); "
                f"using the existing dispatch."
            )
            return None
        if getattr(backend, 'name', 'native') == 'native':
            return None  # native == the existing dispatch below, untouched
        return backend

    def _protocol_backend_cache(self):
        """The raw-forcing cache, configured exactly like the legacy path's."""
        return RawForcingCache(
            cache_root=self.data_dir / 'cache' / 'raw_forcing',
            max_size_gb=self._get_config_value(
                lambda: self.config.data.forcing_cache_size_gb,
                default=3.0, dict_key='FORCING_CACHE_SIZE_GB'),
            ttl_days=self._get_config_value(
                lambda: self.config.data.forcing_cache_ttl_days,
                default=30, dict_key='FORCING_CACHE_TTL_DAYS'),
            enable_checksum=self._get_config_value(
                lambda: self.config.data.forcing_cache_checksum,
                default=True, dict_key='FORCING_CACHE_CHECKSUM'),
        )

    @staticmethod
    def _declared_schema_for(backend, forcing_dataset: str) -> str:
        """The schema the backend declares for a dataset (capability lookup)."""
        wanted = forcing_dataset.lower()
        try:
            for cap in backend.capabilities():
                if cap.dataset_id.lower() == wanted:
                    return str(cap.schema)
        except (AttributeError, TypeError, ValueError, KeyError, RuntimeError, ImportError):
            pass  # capability probing must never break acquisition; key degrades to 'unknown'
        return 'unknown'

    def _restore_protocol_cache_hit(self, meta: dict, raw_data_dir: Path, cached_file: Path) -> bool:
        """Restore a protocol-path cache entry: raw file + sidecar manifest.

        Returns False when the entry predates manifest-aware caching (treated
        as a miss: without the manifest, downstream schema dispatch would
        silently fall back to the native heuristics path).
        """
        import shutil

        from symfluence.data.backends.contract import (
            MANIFEST_FILENAME,
            validate_manifest,
        )

        payload = meta.get('manifest')
        if not isinstance(payload, dict):
            return False
        dest = raw_data_dir / str(meta.get('original_name') or cached_file.name)
        shutil.copy(cached_file, dest)
        payload = dict(payload)
        payload['paths'] = [str(dest)]
        try:
            validate_manifest(payload)
        except ValueError as exc:
            self.logger.warning(f"Cached acquisition manifest invalid ({exc}); re-acquiring.")
            dest.unlink(missing_ok=True)
            return False
        import json as _json
        (raw_data_dir / MANIFEST_FILENAME).write_text(
            _json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
        self.logger.info(f"✓ Restored cached protocol-backend forcing to: {dest}")
        return True

    def _acquire_forcing_via_protocol_backend(self, backend, forcing_dataset: str):
        """Acquire forcing through a protocol backend (non-native).

        Builds the :class:`AcquisitionRequest` from config exactly as the
        legacy path resolves bbox/window/variables and lets the backend
        deliver into ``raw_data/``. The declared-schema result (and its
        sidecar manifest, written by the backend) drives downstream dispatch.

        Cache: keyed structurally on (dataset, bbox, window, variables,
        schema, backend) — the schema/backend salt is part of the key, so
        protocol-path entries never collide with the legacy path's
        ``DATA_ACCESS``-salted entries. The sidecar manifest is stored in the
        cache metadata and restored on hit, keeping schema dispatch intact
        for resumed runs.
        """
        import json as _json

        from symfluence.data.backends.contract import (
            AcquisitionRequest,
            CredentialContext,
            build_manifest,
        )

        bbox_str = self._resolve_bounding_box("forcing")
        north, west, south, east = (float(part) for part in bbox_str.split('/'))

        raw_data_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data'
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        window, variables = self._forcing_request_facts(forcing_dataset)
        schema = self._declared_schema_for(backend, forcing_dataset)

        cache = self._protocol_backend_cache()
        cache_key = cache.generate_cache_key(
            dataset=forcing_dataset,
            bbox=bbox_str,
            time_start=window[0] if window else '',
            time_end=window[1] if window else '',
            variables=sorted(variables) if variables else None,
            backend=f"{getattr(backend, 'name', 'backend')}:{schema}",
        )

        force_download = self._get_config_value(
            lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD')
        if not force_download:
            cached_file = cache.get(cache_key)
            if cached_file is not None:
                meta_path = cache.cache_root / f"{cache_key}.meta.json"
                try:
                    meta = _json.loads(meta_path.read_text(encoding='utf-8'))
                except (OSError, _json.JSONDecodeError):
                    meta = {}
                if self._restore_protocol_cache_hit(meta, raw_data_dir, cached_file):
                    return None
                self.logger.info(
                    "Cached protocol-backend entry lacks a usable manifest; re-acquiring."
                )

        request = AcquisitionRequest(
            dataset_id=forcing_dataset,
            bbox=(south, west, north, east),
            window=window,
            variables=variables,
            target_dir=raw_data_dir,
            credentials=CredentialContext(),
        )
        result = backend.acquire(request)
        self.logger.info(
            f"✓ Forcing data acquired via '{backend.name}' backend "
            f"(schema={result.schema}, {len(result.paths)} file(s))"
        )
        for warning in result.warnings:
            self.logger.warning(f"Backend '{backend.name}' warning: {warning}")

        # Single-file results are cached with their manifest; multi-file
        # results (e.g. NEX-GDDP per-scenario sets) skip caching, like the
        # legacy path skips directory outputs.
        if len(result.paths) == 1 and str(result.paths[0]).endswith('.nc'):
            try:
                cache.put(cache_key, result.paths[0], metadata={
                    'dataset': forcing_dataset,
                    'bbox': bbox_str,
                    'time_start': window[0] if window else '',
                    'time_end': window[1] if window else '',
                    'backend': getattr(backend, 'name', 'backend'),
                    'schema': str(result.schema),
                    'original_name': Path(result.paths[0]).name,
                    'manifest': build_manifest(result),
                })
            except (OSError, ValueError) as exc:
                self.logger.warning(f"Failed to cache protocol-backend forcing output: {exc}")
        return result

    def acquire_forcings(self):
        """Acquire forcing data for the model simulation."""
        self.logger.info("Starting forcing data acquisition")

        # If forcing_path points to existing data, symlink into raw_data and skip download
        forcing_path = self._get_config_value(
            lambda: self.config.paths.forcing_path, default=None, dict_key='FORCING_PATH')
        if forcing_path and forcing_path != 'default':
            forcing_path = Path(forcing_path)
            if forcing_path.exists():
                nc_files = list(forcing_path.glob('*.nc')) + list(forcing_path.glob('*.nc4'))
                csv_files = list(forcing_path.glob('*.csv'))
                if nc_files or csv_files:
                    raw_data_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data'
                    raw_data_dir.mkdir(parents=True, exist_ok=True)
                    existing_raw = list(raw_data_dir.glob('*.nc')) + list(raw_data_dir.glob('*.csv'))
                    if not existing_raw:
                        raw_data_dir.symlink_to(forcing_path) if not raw_data_dir.exists() else None
                        for f in (nc_files + csv_files):
                            link = raw_data_dir / f.name
                            if not link.exists():
                                link.symlink_to(f)
                        self.logger.info(
                            f"✓ Using pre-staged forcing data from forcing_path: {forcing_path} "
                            f"({len(nc_files)} .nc, {len(csv_files)} .csv files symlinked to raw_data/)"
                        )
                    else:
                        self.logger.info(
                            f"✓ Forcing data already exists in raw_data/ ({len(existing_raw)} files), "
                            f"skipping acquisition"
                        )
                    return

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

        # COMMUNITY consults the AcquisitionBackend selector first (below);
        # datasets no registered non-native backend claims fall through to the
        # same registry-handler pathway as CLOUD, so behavior without protocol
        # backends installed is identical to CLOUD.
        if data_access in ('CLOUD', 'COMMUNITY'):
            self.logger.info(f"{data_access.capitalize()} data access enabled for {forcing_dataset}")

            # ── AcquisitionBackend protocol selection ─────────────────────
            # Consulted only when a NON-native backend has registered under
            # the protocol (e.g. the cfs plugin's CommunityForcingBackend).
            # DATA_ACCESS: community routes eligible datasets through the
            # protocol path; cloud/MAF and <DATASET>_BACKEND: native keep the
            # legacy dispatch below byte-for-byte.
            protocol_backend = self._maybe_select_protocol_backend(forcing_dataset)
            if protocol_backend is not None:
                self._acquire_forcing_via_protocol_backend(protocol_backend, forcing_dataset)
                if self._get_config_value(lambda: self.config.forcing.supplement, default=False):
                    self.logger.info("SUPPLEMENT_FORCING enabled - acquiring EM-Earth data")
                    self.acquire_em_earth_forcings()
                return

            if not check_cloud_access_availability(forcing_dataset, self.logger):
                raise ValueError(
                    f"Dataset '{forcing_dataset}' has no registered acquisition handler "
                    f"(DATA_ACCESS: {data_access.lower()})."
                )

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

            cache_backend = data_access
            if forcing_dataset == 'CARRA':
                source = self._get_config_value(lambda: self.config.forcing.carra_source, default='cds', dict_key='CARRA_SOURCE')
                if source != 'cds':
                    cache_backend = f'{data_access}:{source}'
            cache_key = cache.generate_cache_key(
                dataset=forcing_dataset,
                bbox=bbox,
                time_start=time_start,
                time_end=time_end,
                variables=variables if isinstance(variables, list) else None,
                backend=cache_backend,
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

    def _observation_window(self):
        """The (start, end) ISO window for observation requests, or None."""
        time_start = self._get_config_value(lambda: self.config.domain.time_start)
        time_end = self._get_config_value(lambda: self.config.domain.time_end)
        return (str(time_start), str(time_end)) if time_start and time_end else None

    def _community_observation_backend(self, provider: str, kind: str):
        """Return the community ObservationBackend serving (provider, kind), else None.

        The single seam through which BOTH streamflow (CSFS) and non-streamflow
        kinds (COS: TWS/SWE/ET/...) are routed to the community tier instead of
        the native ``evaluation.<kind>.download`` handlers. Only under
        ``DATA_ACCESS: community`` with a backend registered that actually admits
        the provider for this kind (license/parity gate passes, no native pin);
        returns None for every other mode, leaving the legacy native path intact.
        """
        if not provider:
            return None
        data_access = str(self._get_config_value(
            lambda: self.config.domain.data_access, default='MAF', dict_key='DATA_ACCESS')).lower()
        if data_access != 'community':
            return None
        from symfluence.core.registries import R
        if not R.observation_backends.keys():
            return None
        from symfluence.data.backends.errors import AcquisitionError
        from symfluence.data.backends.selection import select_observation_backend
        try:
            return select_observation_backend(
                provider, self.config, kind=kind,
                window=self._observation_window(), logger=self.logger,
            )
        except AcquisitionError:
            return None

    def _community_serves_streamflow(self, provider: str) -> bool:
        """True if a community ObservationBackend serves *provider*'s streamflow.

        Mirrors ``DataManager._observation_backend_serves``. When True, the native
        ``<PROVIDER>_STREAMFLOW`` download is skipped here — ObservedDataProcessor's
        backend tier fetches it instead — so the provider is not double-fetched
        (native raw + community re-fetch).
        """
        return self._community_observation_backend(provider, 'streamflow') is not None

    #: Non-streamflow additional-obs keys that, under DATA_ACCESS: community, are
    #: served by an ObservationBackend (e.g. COS) instead of the native
    #: ``evaluation.<kind>.download`` handler. Maps the native key -> a full adapter
    #: spec ``(cos_provider, kind, output_path_helper, out_column, value_scale)``:
    #:
    #:  * ``output_path_helper`` — the name of a ``data.observation.paths`` helper
    #:    returning the FIRST file the matching evaluator searches, so the canonical
    #:    write is picked up transparently;
    #:  * ``out_column`` — the column name that evaluator reads the observed value
    #:    from;
    #:  * ``value_scale`` — factor applied to the COS canonical value to match the
    #:    evaluator's expected magnitude/unit. Identity for kinds whose canonical
    #:    unit already matches; ``1/25.4`` for SWE because the snow evaluator
    #:    force-converts inches->mm by magnitude (threshold 250), so COS mm is
    #:    written as inches to reconstruct the correct mm downstream.
    #:
    #: The community backend acquires AND reduces in one call;
    #: :meth:`_route_community_nonstreamflow_obs` then writes the canonical file and
    #: the native handler is skipped. Gridded COS connectors (grace, mod16_et,
    #: modis_sca) reduce a SUPPLIED granule (live-fetch is not wired in COS v0.1.0):
    #: only GRACE has a pre-staged-mascon convention here, so the other gridded
    #: providers decline and fall back to native; the point/tower connectors
    #: (snotel, usgs_gw, fluxnet_et) fetch live.
    _COMMUNITY_NONSTREAMFLOW_OBS = {
        'GRACE':      ('grace',      'tws',         'tws_default_observation_path',         'grace_jpl_anomaly', 1.0),
        'SNOTEL':     ('snotel',     'swe',         'swe_default_observation_path',         'swe',               1.0 / 25.4),
        'MODIS_SNOW': ('modis_sca',  'snow_cover',  'snow_cover_default_observation_path',  'sca',               1.0),
        'MODIS_ET':   ('mod16_et',   'et',          'modis_et_default_observation_path',    'et_mm_day',         1.0),
        'FLUXNET_ET': ('fluxnet_et', 'et',          'fluxnet_et_default_observation_path',  'et_mm_day',         1.0),
        'USGS_GW':    ('usgs_gw',    'groundwater', 'groundwater_default_observation_path', 'depth',             1.0),
        'SMAP':       ('smap_sm',    'soil_moisture', 'smap_default_observation_path',        'soil_moisture',     1.0),
        'CHIRPS':     ('chirps_precip', 'precipitation', 'chirps_default_observation_path',    'precipitation_mm',   1.0),
    }

    def _route_community_nonstreamflow_obs(self, additional_obs) -> set:
        """Acquire community-served non-streamflow observations via the backend tier.

        For each additional-obs key whose (provider, kind) a community backend
        serves, acquire through the backend and write the canonical processed
        file the evaluators read; return the set of keys handled so the caller
        drops them from the native task list. Best-effort: a community failure
        logs a warning and leaves the native key in place (graceful fallback).
        No-op unless ``DATA_ACCESS: community`` with a backend registered.
        """
        handled: set = set()
        for obs_type in list(additional_obs):
            spec = self._COMMUNITY_NONSTREAMFLOW_OBS.get(str(obs_type).upper())
            if not spec:
                continue
            provider, kind = spec[0], spec[1]
            backend = self._community_observation_backend(provider, kind)
            if backend is None:
                continue
            try:
                if self._acquire_observation_via_backend(backend, spec):
                    handled.add(obs_type)
                    self.logger.info(
                        f"Observation '{obs_type}' ({kind}) served by the "
                        f"'{backend.name}' community backend; skipping native download."
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort; fall back to native
                self.logger.warning(
                    f"Community acquisition of {obs_type} via '{backend.name}' failed "
                    f"({exc}); falling back to the native handler."
                )
        return handled

    def _acquire_observation_via_backend(self, backend, spec: tuple) -> bool:
        """Acquire one (provider, kind) via *backend*, write its canonical file.

        *spec* is a :data:`_COMMUNITY_NONSTREAMFLOW_OBS` entry
        ``(provider, kind, path_helper, out_column, value_scale)``.
        """
        from symfluence.data.backends.contract import ObservationRequest

        provider, kind, path_helper, out_column, value_scale = spec
        domain_name = self._get_config_value(
            lambda: self.config.domain.name, default='domain', dict_key='DOMAIN_NAME')
        raw_dir = resolve_data_subdir(self.project_dir, 'observations') / kind / 'community_raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Spatial-reduction context for gridded products (basin-mean needs a bbox).
        # The backend reads these from request.options; translate the SYMFLUENCE bbox
        # "lat_max/lon_min/lat_min/lon_max" to the (lat_min, lon_min, lat_max,
        # lon_max) order COS expects.
        options: Dict[str, Any] = {'domain_name': domain_name}
        bbox = self._get_config_value(
            lambda: self.config.domain.bounding_box_coords, dict_key='BOUNDING_BOX_COORDS')
        if bbox:
            try:
                lat_max, lon_min, lat_min, lon_max = (float(x) for x in str(bbox).split('/'))
                options['bbox'] = (lat_min, lon_min, lat_max, lon_max)
            except (ValueError, TypeError):
                self.logger.debug(f"Could not parse BOUNDING_BOX_COORDS {bbox!r} for community obs reduction")
        connector_config = self._community_connector_config(provider)
        if connector_config:
            options['connector_config'] = connector_config
        request = ObservationRequest(
            provider_id=provider, station_ids=self._community_station_ids(provider), kind=kind,
            window=self._observation_window(), target_dir=raw_dir, options=options,
        )
        result = backend.acquire(request)
        return self._write_community_obs_output(
            result, domain_name, path_helper=path_helper,
            out_column=out_column, value_scale=value_scale,
        )

    def _community_connector_config(self, provider: str) -> Dict[str, Any]:
        """COS connector config for a community provider (a supplied granule, if any).

        Gridded COS connectors reduce a supplied NetCDF (Earthdata live-fetch is not
        wired in COS v0.1.0). Only GRACE has a pre-staged-mascon convention here, so
        the other gridded providers (mod16_et, modis_sca) return ``{}`` and decline
        -> fall back to native. The point/tower connectors fetch live (no file).
        """
        if provider == 'grace':
            return self._community_grace_connector_config()
        if provider == 'smap_sm':
            return self._community_staged_connector_config('soil_moisture', ('smap', 'raw_data'))
        if provider == 'chirps_precip':
            return self._community_staged_connector_config('precipitation', ('chirps', 'raw_data'))
        return {}

    def _community_staged_connector_config(self, family: str, subdirs: tuple) -> Dict[str, Any]:
        """Return a staged NetCDF for a COS gridded connector, if one exists."""
        root = resolve_data_subdir(self.project_dir, 'observations') / family
        candidates = [root]
        candidates.extend(root / part for part in subdirs)
        for directory in candidates:
            if directory.exists():
                files = sorted(directory.glob('*.nc'))
                if files:
                    return {'nc_path': str(files[0])}
        return {}

    def _community_station_ids(self, provider: str) -> tuple:
        """Explicit station ids for a point/tower community provider.

        Point networks (SNOTEL, USGS groundwater) and flux towers (FLUXNET) select
        by station, read from the SAME config the native handler uses — so a domain
        that already names its station is served by COS without extra config.
        Gridded providers (grace/tws, mod16_et, modis_sca) select by bbox and get
        no station ids. A comma-separated config value yields multiple stations.
        """
        raw: Any = None
        if provider == 'snotel':
            raw = (self._get_config_value(lambda: self.config.evaluation.snotel.station, dict_key='SNOTEL_STATION')
                   or self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID'))
        elif provider == 'fluxnet_et':
            raw = self._get_config_value(lambda: self.config.evaluation.fluxnet.station, dict_key='FLUXNET_STATION')
        elif provider == 'usgs_gw':
            raw = (self._get_config_value(lambda: self.config.data.usgs_site_code, dict_key='USGS_SITE_CODE')
                   or self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID'))
        if not raw:
            return ()
        if isinstance(raw, (list, tuple)):
            return tuple(str(s).strip() for s in raw if str(s).strip())
        return tuple(s.strip() for s in str(raw).split(',') if s.strip())

    def _community_grace_connector_config(self) -> Dict[str, Any]:
        """COS connector config for GRACE: a downloaded mascon NetCDF to reduce.

        COS's GRACE connector reduces a supplied NetCDF (its Earthdata live fetch
        is not yet implemented in v0.1.0), so we point it at an already-downloaded
        mascon under ``observations/grace/`` — preferring the JPL RL06 product the
        config selects. Returns ``{}`` when none is present, in which case COS
        declines and the routing falls back to the native GRACE handler.
        """
        grace_dir = resolve_data_subdir(self.project_dir, 'observations') / 'grace'
        if not grace_dir.exists():
            return {}
        ncs = sorted(grace_dir.glob('*.nc'))
        if not ncs:
            return {}
        preferred = [p for p in ncs if 'JPL' in p.name.upper()] or ncs
        return {'nc_path': str(preferred[0])}

    def _write_community_obs_output(
        self, result, domain_name: str, *, path_helper: str, out_column: str, value_scale: float = 1.0,
    ) -> bool:
        """Adapt a community observation delivery (OBS_CSV_V1) to its canonical file.

        COS delivers a per-site OBS_CSV_V1 table (``datetime, value, quality_flag``);
        each evaluator reads a specific file + column. We take the value column,
        scale it to the evaluator's expected unit (``value_scale``), rename it to
        ``out_column``, aggregate to one series per ``datetime`` (basin-representative
        mean across delivered sites), and write the first path the evaluator searches
        (``path_helper`` in :mod:`symfluence.data.observation.paths`). Returns False
        when nothing usable was delivered, so the routing falls back to native.

        Unit notes: TWS is correlation-scored (scale-invariant — COS mm vs native cm
        does not affect it); SWE is written in inches (COS mm * 1/25.4) because the
        snow evaluator force-converts inches->mm by magnitude; the rest already match
        the evaluator's expected unit.
        """
        import pandas as pd

        from symfluence.data.observation import paths as _obs_paths

        if not result.paths:
            return False
        # OBS_CSV_V1 always delivers a 'value' column; the kind-specific names cover
        # backends that label it (e.g. the GRACE 'tws_anomaly_mm' delivery).
        value_candidates = (
            'value', 'tws_anomaly_mm', 'tws_anomaly', 'swe_mm', 'et_mm_day',
            'sca_fraction', 'sca', 'groundwater_level',
        )
        frames = []
        for path in result.paths:
            df = pd.read_csv(path)
            if 'datetime' not in df.columns:
                continue
            value_col = next((c for c in value_candidates if c in df.columns), None)
            if value_col is None:
                continue
            frames.append(df[['datetime', value_col]].rename(columns={value_col: out_column}))
        if not frames:
            return False
        out = pd.concat(frames).groupby('datetime', as_index=False)[out_column].mean()
        if value_scale != 1.0:
            out[out_column] = out[out_column] * value_scale
        out_path = getattr(_obs_paths, path_helper)(self.project_dir, domain_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        self.logger.info(
            f"Wrote canonical community observations to {out_path} ({len(out)} rows, col '{out_column}')"
        )
        return True

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
        # Under DATA_ACCESS: community, when a community backend (e.g. CSFS) serves
        # this provider, ObservedDataProcessor's backend tier fetches it. Do NOT
        # also queue the native <PROVIDER>_STREAMFLOW download here — that produced
        # a redundant double-fetch (native raw downloaded, then re-fetched via the
        # backend, which is what is actually used). Mirrors the same skip in
        # DataManager.process_observed_data's additional_obs assembly.
        if self._community_serves_streamflow(streamflow_provider):
            self.logger.info(
                f"Streamflow provider '{streamflow_provider}' is community-served; "
                "skipping the native observation download (handled by the backend tier)."
            )
        elif streamflow_provider == 'USGS' and 'USGS_STREAMFLOW' not in additional_obs:
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

        # Under DATA_ACCESS: community, route non-streamflow kinds (GRACE TWS, …)
        # through the ObservationBackend tier (COS) instead of the native
        # evaluation.<kind>.download handler, mirroring the streamflow routing.
        # Handled keys are dropped from the native task list below; anything the
        # community tier declines or fails on falls back to its native handler.
        community_handled = self._route_community_nonstreamflow_obs(additional_obs)
        if community_handled:
            additional_obs = [o for o in additional_obs if o not in community_handled]
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
