# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Data Manager

Facade that coordinates acquisition, observation processing, and model-agnostic
preprocessing. Keeps orchestration thin while services handle the heavy
lifting. See docs under ``docs/source/configuration`` and ``docs/source/data``
for full workflows.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.core.exceptions import DataAcquisitionError, symfluence_error_handler
from symfluence.core.registries import R
from symfluence.data.acquisition.acquisition_service import AcquisitionService
from symfluence.data.acquisition.observed_processor import ObservedDataProcessor
from symfluence.data.preprocessing.em_earth_integrator import EMEarthIntegrator
from symfluence.data.preprocessing.forcing_resampler import ForcingResampler
from symfluence.data.preprocessing.geospatial_statistics import GeospatialStatistics
from symfluence.data.utils.variable_utils import VariableHandler

if TYPE_CHECKING:
    pass


class DataManager(BaseManager):
    """Facade that orchestrates acquisition, preprocessing, and observation handling.

    Delegates to acquisition/preprocessing services and registries; keeps
    runtime imports slim. Detailed behaviour lives in the docs.
    """

    def _initialize_services(self) -> None:
        """Initialize data management services."""
        self.acquisition_service = self._get_service(
            AcquisitionService,
            self.config,
            self.logger,
            self.reporting_manager
        )
        self.em_earth_integrator = self._get_service(
            EMEarthIntegrator,
            self.config,
            self.logger
        )
        self.variable_handler = self._get_service(
            VariableHandler,
            self.config_dict,
            self.logger,
            'ERA5',
            'SUMMA'
        )

    def acquire_attributes(self):
        """
        Acquire geospatial attributes (DEM, soil classes, land cover) for the domain.

        Downloads and processes required geospatial data layers including elevation,
        soil classification, and land cover data from configured data sources.
        """
        self.acquisition_service.acquire_attributes()

        # Generate attribute acquisition diagnostics
        if self.reporting_manager:
            with symfluence_error_handler(
                "generating attribute diagnostics",
                self.logger,
                reraise=False,
                error_type=DataAcquisitionError
            ):
                domain_name = self._get_config_value(
                    lambda: self.config.domain.name,
                    'domain'
                )
                dem_path = self.project_attributes_dir / 'elevation' / 'dem' / f"{domain_name}_elv.tif"
                soil_path = self.project_attributes_dir / 'soilclass' / f"{domain_name}_soilclass.tif"
                land_path = self.project_attributes_dir / 'landclass' / f"{domain_name}_landclass.tif"

                # Try alternative paths if standard ones don't exist
                if not dem_path.exists():
                    dem_files = list((self.project_attributes_dir / 'elevation').rglob("*.tif"))
                    dem_path = dem_files[0] if dem_files else None
                if not soil_path.exists():
                    soil_files = list((self.project_attributes_dir / 'soilclass').rglob("*.tif"))
                    soil_path = soil_files[0] if soil_files else None
                if not land_path.exists():
                    land_files = list((self.project_attributes_dir / 'landclass').rglob("*.tif"))
                    land_path = land_files[0] if land_files else None

                self.reporting_manager.diagnostic_attributes(
                    dem_path=dem_path,
                    soil_path=soil_path,
                    land_path=land_path
                )

    def acquire_forcings(self):
        """
        Acquire meteorological forcing data for the simulation period.

        Downloads forcing variables (precipitation, temperature, radiation, etc.)
        from the configured forcing dataset (ERA5, RDRS, CARRA, etc.) for the
        specified temporal domain.
        """
        self.acquisition_service.acquire_forcings()

        # Generate raw forcing diagnostics
        if self.reporting_manager:
            with symfluence_error_handler(
                "generating raw forcing diagnostics",
                self.logger,
                reraise=False,
                error_type=DataAcquisitionError
            ):
                # Check for merged or raw forcing files
                merged_dir = self.project_forcing_dir / 'merged_data'
                raw_dir = self.project_forcing_dir / 'raw_data'
                forcing_dir = merged_dir if merged_dir.exists() else raw_dir

                if forcing_dir.exists():
                    forcing_files = list(forcing_dir.glob("*.nc"))
                    if forcing_files:
                        domain_shp = self.project_dir / 'shapefiles' / 'river_basins'
                        domain_files = list(domain_shp.glob("*.shp")) if domain_shp.exists() else []
                        self.reporting_manager.diagnostic_forcing_raw(
                            forcing_nc=forcing_files[0],
                            domain_shp=domain_files[0] if domain_files else None
                        )

    def acquire_observations(self):
        """
        Acquire observational data for model calibration and validation.

        Downloads streamflow observations, snow measurements, and other validation
        data from configured observation sources (USGS, WSC, SNOTEL, etc.).
        """
        self.acquisition_service.acquire_observations()

    def acquire_em_earth_forcings(self):
        """
        Acquire EM-Earth supplementary forcing data.

        Downloads and processes EM-Earth reanalysis data for gap-filling or
        supplementing primary forcing datasets.
        """
        self.acquisition_service.acquire_em_earth_forcings()

    def process_observed_data(self):
        """
        Process observed data including streamflow and additional variables.

        Raises:
            DataAcquisitionError: If data processing fails
        """
        self.logger.info("Processing observed data")
        self.acquire_observations()

        with symfluence_error_handler(
            "observed data processing",
            self.logger,
            error_type=DataAcquisitionError
        ):
            # 1. Parse observations to process
            additional_obs = self._get_config_value(
                lambda: self.config.data.additional_observations,
                []
            )
            if additional_obs is None:
                additional_obs = []
            elif isinstance(additional_obs, str):
                additional_obs = [o.strip() for o in additional_obs.split(',')]

            # 2. Check for primary streamflow provider and handle USGS/WSC migration
            streamflow_provider = str(self._get_config_value(
                lambda: self.config.data.streamflow_data_provider,
                ''
            )).upper()
            if streamflow_provider == 'USGS' and 'usgs_streamflow' not in [o.lower() for o in additional_obs]:
                # Automatically add usgs_streamflow if it's the primary provider but not in additional_obs
                additional_obs.append('usgs_streamflow')
            elif streamflow_provider == 'WSC' and 'wsc_streamflow' not in [o.lower() for o in additional_obs]:
                additional_obs.append('wsc_streamflow')
            elif streamflow_provider == 'SMHI' and 'smhi_streamflow' not in [o.lower() for o in additional_obs]:
                additional_obs.append('smhi_streamflow')
            elif streamflow_provider == 'LAMAH_ICE' and 'lamah_ice_streamflow' not in [o.lower() for o in additional_obs]:
                additional_obs.append('lamah_ice_streamflow')
            elif streamflow_provider == 'DGA' and 'dga_streamflow' not in [o.lower() for o in additional_obs]:
                additional_obs.append('dga_streamflow')

            # Check for USGS Groundwater download and ensure it's in additional_obs
            download_usgs_gw = self._get_config_value(
                lambda: self.config.evaluation.usgs_gw.download,
                False
            )
            if isinstance(download_usgs_gw, str):
                download_usgs_gw = download_usgs_gw.lower() == 'true'

            if download_usgs_gw and 'usgs_gw' not in [o.lower() for o in additional_obs]:
                additional_obs.append('usgs_gw')

            # Check for GRACE TWS and ensure it's in additional_obs
            download_grace = self._get_config_value(
                lambda: self.config.evaluation.grace.download,
                False
            )
            if isinstance(download_grace, str):
                download_grace = download_grace.lower() == 'true'

            if download_grace and 'grace' not in [o.lower() for o in additional_obs]:
                additional_obs.append('grace')

            # Check for MODIS Snow and ensure it's in additional_obs
            download_modis_snow = self._get_config_value(
                lambda: self.config.evaluation.modis_snow.download,
                False
            )
            if download_modis_snow and 'modis_snow' not in [o.lower() for o in additional_obs]:
                additional_obs.append('modis_snow')

            # Check for SNOTEL download and ensure it's in additional_obs
            download_snotel = self._get_config_value(
                lambda: self.config.evaluation.snotel.download,
                False
            )
            if isinstance(download_snotel, str):
                download_snotel = download_snotel.lower() == 'true'

            if download_snotel and 'snotel' not in [o.lower() for o in additional_obs]:
                additional_obs.append('snotel')

            # Check for ISMN download and ensure it's in additional_obs
            download_ismn = self._get_config_value(
                lambda: self.config.data.download_ismn,
                False
            )
            if isinstance(download_ismn, str):
                download_ismn = download_ismn.lower() == 'true'

            if download_ismn and 'ismn' not in [o.lower() for o in additional_obs]:
                additional_obs.append('ismn')

            # 3. Traditional streamflow processing (for providers not yet migrated)
            observed_data_processor = ObservedDataProcessor(self.config, self.logger)

            # Only run traditional if NOT using the formalized handlers
            # Note: Registry uses lowercase keys, so we check with case-insensitive comparison
            formalized_providers = ['usgs_streamflow', 'wsc_streamflow', 'smhi_streamflow', 'lamah_ice_streamflow', 'dga_streamflow']
            additional_obs_lower = [o.lower() for o in additional_obs]
            is_formalized = any(obs in additional_obs_lower for obs in formalized_providers)

            if not is_formalized:
                observed_data_processor.process_streamflow_data()

            observed_data_processor.process_fluxnet_data()

            # 4. Registry-based additional observations (GRACE, MODIS, USGS, etc.)

            for obs_type in additional_obs:
                try:
                    if obs_type in R.observation_handlers:
                        self.logger.info(f"Processing registry-based observation: {obs_type}")
                        handler_cls = R.observation_handlers.get(obs_type)
                        handler = handler_cls(self.config, self.logger) if handler_cls else None
                        raw_path = handler.acquire()
                        processed_path = handler.process(raw_path)

                        # Visualize processed data
                        if self.reporting_manager and processed_path and processed_path.exists():
                            if processed_path.suffix == '.csv':
                                df = pd.read_csv(processed_path)
                                # Assuming first numeric column is the value
                                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                                if not numeric_cols.empty:
                                    self.reporting_manager.visualize_data_distribution(
                                        df[numeric_cols[0]],
                                        variable_name=f"{obs_type}_{numeric_cols[0]}",
                                        stage='preprocessing'
                                    )
                            elif processed_path.suffix in ['.tif', '.nc']:
                                self.reporting_manager.visualize_spatial_coverage(
                                    processed_path,
                                    variable_name=obs_type,
                                    stage='preprocessing'
                                )

                    else:
                        self.logger.warning(f"Observation type {obs_type} requested but no handler registered.")
                except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
                    self.logger.warning(f"Failed to process additional observation {obs_type}: {e}")
                except Exception as e:  # noqa: BLE001 — must-not-raise contract
                    self.logger.exception(f"Unexpected failure processing additional observation {obs_type}: {e}")

            # Generate diagnostic plots for streamflow observations
            if self.reporting_manager:
                with symfluence_error_handler(
                    "generating observation diagnostics",
                    self.logger,
                    reraise=False,
                    error_type=DataAcquisitionError
                ):
                    obs_dir = self.project_observations_dir / "streamflow" / "preprocessed"
                    if obs_dir.exists():
                        obs_files = list(obs_dir.glob("*.csv"))
                        if obs_files:
                            obs_df = pd.read_csv(obs_files[0], parse_dates=True)
                            self.reporting_manager.diagnostic_observations(
                                obs_df=obs_df,
                                obs_type='streamflow'
                            )

            self.logger.info("Observed data processing completed successfully")

    def run_model_agnostic_preprocessing(self):
        """
        Run model-agnostic preprocessing including basin averaging and resampling.

        Raises:
            DataAcquisitionError: If preprocessing fails
        """
        # Create required directories
        basin_averaged_data = self.project_forcing_dir / 'basin_averaged_data'
        catchment_intersection_dir = self.project_dir / 'shapefiles' / 'catchment_intersection'

        basin_averaged_data.mkdir(parents=True, exist_ok=True)
        catchment_intersection_dir.mkdir(parents=True, exist_ok=True)

        with symfluence_error_handler(
            "model-agnostic preprocessing",
            self.logger,
            error_type=DataAcquisitionError
        ):
            # Run geospatial statistics
            self.logger.debug("Running geospatial statistics")
            gs = GeospatialStatistics(self.config, self.logger)
            gs.run_statistics()

            # Run forcing resampling
            self.logger.debug("Running forcing resampling")
            fr = ForcingResampler(self.config, self.logger)
            fr.run_resampling()

            # Apply model-agnostic elevation corrections
            from symfluence.data.preprocessing import ElevationCorrectionProcessor
            if ElevationCorrectionProcessor is not None:
                elev_proc = ElevationCorrectionProcessor(self.config, self.logger)
                elev_proc.apply()

            # Visualize preprocessed forcing if available
            if self.reporting_manager:
                with symfluence_error_handler(
                    "visualizing preprocessed forcing",
                    self.logger,
                    reraise=False,
                    error_type=DataAcquisitionError
                ):
                    # Check for basin averaged files
                    basin_files = list(basin_averaged_data.glob("*.nc"))
                    if basin_files:
                        self.reporting_manager.visualize_spatial_coverage(basin_files[0], 'forcing_processed', 'preprocessing')

                # Visualize raw vs remapped forcing comparison
                with symfluence_error_handler(
                    "visualizing forcing comparison",
                    self.logger,
                    reraise=False,
                    error_type=DataAcquisitionError
                ):
                    self._visualize_forcing_comparison(basin_averaged_data)

                # Generate forcing remapping diagnostics
                with symfluence_error_handler(
                    "generating forcing remapping diagnostics",
                    self.logger,
                    reraise=False,
                    error_type=DataAcquisitionError
                ):
                    raw_forcing_dir = self.project_forcing_dir / 'merged_data'
                    if not raw_forcing_dir.exists():
                        raw_forcing_dir = self.project_forcing_dir / 'raw_data'
                    raw_files = list(raw_forcing_dir.glob("*.nc")) if raw_forcing_dir.exists() else []
                    basin_files = list(basin_averaged_data.glob("*.nc"))
                    if raw_files and basin_files:
                        hru_shp = self._find_hru_shapefile()
                        self.reporting_manager.diagnostic_forcing_remapped(
                            raw_nc=raw_files[0],
                            remapped_nc=basin_files[0],
                            hru_shp=hru_shp
                        )

            # Integrate EM-Earth data if supplementation is enabled
            supplement_forcing = self._get_config_value(
                lambda: self.config.forcing.supplement,
                False
            )
            if supplement_forcing:
                self.logger.debug("Integrating EM-Earth data")
                self.em_earth_integrator.integrate_em_earth_data()

            self.logger.info("Model-agnostic preprocessing completed successfully")

    def build_model_ready_store(self):
        """Build or refresh the model-ready data store.

        Creates CF-1.8 compliant NetCDF files for forcings, observations,
        and attributes in ``data/model_ready/``.
        """
        from symfluence.data.model_ready.store_builder import ModelReadyStoreBuilder

        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            'domain'
        )

        builder = ModelReadyStoreBuilder(
            project_dir=self.project_dir,
            domain_name=domain_name,
            config=self.config,
        )
        builder.build_all()

    def validate_data_directories(self) -> bool:
        """Validate that required data directories exist.

        .. deprecated::
            Use :meth:`validate_readiness` instead.
        """
        import warnings
        warnings.warn(
            "validate_data_directories() is deprecated, use validate_readiness()",
            DeprecationWarning,
            stacklevel=2,
        )
        readiness = self.validate_readiness()
        return readiness.get('data_directories', False)

    def validate_readiness(self) -> Dict[str, bool]:
        """
        Validate that this manager is ready for execution.

        Checks whether required data directories exist.

        Returns:
            Dict mapping check names to pass/fail booleans.
        """
        required_dirs = [
            self.project_attributes_dir,
            self.project_forcing_dir,
            self.project_observations_dir,
            self.project_dir / 'shapefiles'
        ]
        all_exist = True
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.logger.warning(f"Required directory does not exist: {dir_path}")
                all_exist = False
        return {'data_directories': all_exist}

    def _visualize_forcing_comparison(self, basin_averaged_data: Path) -> None:
        """
        Visualize raw vs. remapped forcing comparison.

        Args:
            basin_averaged_data: Path to basin averaged data directory
        """
        if not self.reporting_manager:
            return

        # Find remapped file (basin averaged)
        remapped_files = list(basin_averaged_data.glob("*.nc"))
        if not remapped_files:
            self.logger.debug("No remapped forcing files found for comparison visualization")
            return
        remapped_forcing_file = remapped_files[0]

        # Find raw forcing file (check merged_data first, then raw_data)
        raw_forcing_dir = self.project_forcing_dir / 'merged_data'
        if not raw_forcing_dir.exists() or not list(raw_forcing_dir.glob("*.nc")):
            raw_forcing_dir = self.project_forcing_dir / 'raw_data'

        raw_files = list(raw_forcing_dir.glob("*.nc")) if raw_forcing_dir.exists() else []
        if not raw_files:
            self.logger.debug("No raw forcing files found for comparison visualization")
            return
        raw_forcing_file = raw_files[0]

        # Find forcing grid shapefile
        forcing_grid_shp = self._find_forcing_shapefile()
        if forcing_grid_shp is None:
            self.logger.debug("Forcing grid shapefile not found for comparison visualization")
            return

        # Find HRU shapefile
        hru_shp = self._find_hru_shapefile()
        if hru_shp is None:
            self.logger.debug("HRU shapefile not found for comparison visualization")
            return

        # Call visualization
        self.reporting_manager.visualize_forcing_comparison(
            raw_forcing_file=raw_forcing_file,
            remapped_forcing_file=remapped_forcing_file,
            forcing_grid_shp=forcing_grid_shp,
            hru_shp=hru_shp
        )

    def _find_hru_shapefile(self) -> Optional[Path]:
        """
        Find the HRU/catchment shapefile.

        Returns:
            Path to HRU shapefile, or None if not found
        """
        catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
        if not catchment_dir.exists():
            return None

        # Try to find HRU shapefile based on common naming patterns
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            'domain'
        )

        # Try explicit config value first
        catchment_name = self._get_config_value(
            lambda: self.config.paths.catchment_name,
            'default'
        )
        if catchment_name != 'default':
            explicit_path = catchment_dir / catchment_name
            if explicit_path.exists():
                return explicit_path

        # Search for HRU shapefiles with common patterns
        patterns = [
            f"{domain_name}_HRUs_*.shp",
            f"{domain_name}_catchment*.shp",
            "*HRU*.shp",
            "*catchment*.shp",
            "*.shp"  # Fallback to any shapefile
        ]

        for pattern in patterns:
            matches = list(catchment_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _find_forcing_shapefile(self) -> Optional[Path]:
        """
        Find the forcing grid shapefile.

        Returns:
            Path to forcing shapefile, or None if not found
        """
        forcing_shp_dir = self.project_dir / 'shapefiles' / 'forcing'
        if not forcing_shp_dir.exists():
            return None

        # Try explicit config value first
        forcing_dataset = self._get_config_value(
            lambda: self.config.forcing.dataset,
            'ERA5'
        )
        expected_path = forcing_shp_dir / f"forcing_{forcing_dataset}.shp"
        if expected_path.exists():
            return expected_path

        # Search for any forcing shapefile (handles cases like 'local' dataset)
        patterns = [
            "forcing_*.shp",
            "*.shp"  # Fallback to any shapefile
        ]

        for pattern in patterns:
            matches = list(forcing_shp_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def get_data_status(self) -> Dict[str, Any]:
        """Get status of data acquisition and preprocessing."""
        status = {
            'project_dir': str(self.project_dir),
            'attributes_acquired': (self.project_attributes_dir / 'elevation' / 'dem').exists(),
            'forcings_acquired': (self.project_forcing_dir / 'raw_data').exists(),
            'forcings_preprocessed': (self.project_forcing_dir / 'basin_averaged_data').exists(),
            'observed_data_processed': (self.project_observations_dir / 'streamflow' / 'preprocessed').exists(),
        }

        status['dem_exists'] = (self.project_attributes_dir / 'elevation' / 'dem').exists()
        status['soilclass_exists'] = (self.project_attributes_dir / 'soilclass').exists()
        status['landclass_exists'] = (self.project_attributes_dir / 'landclass').exists()

        supplement_forcing = self._get_config_value(
            lambda: self.config.forcing.supplement,
            False
        )
        if supplement_forcing:
            status['em_earth_acquired'] = (self.project_forcing_dir / 'raw_data_em_earth').exists()
            status['em_earth_integrated'] = (self.project_forcing_dir / 'em_earth_remapped').exists()
        else:
            status['em_earth_acquired'] = False
            status['em_earth_integrated'] = False

        return status
