# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA model preprocessor.

This module contains the main SummaPreProcessor class that orchestrates
the preprocessing workflow for SUMMA model runs.
"""

# Standard library imports
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Third-party imports
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigValidationError
from symfluence.core.modeling.base import BaseModelPreProcessor
from symfluence.core.registries import R

# Local imports
from .attributes_manager import SummaAttributesManager
from .config_manager import SummaConfigManager
from .forcing_processor import SummaForcingProcessor
from .glacier_manager import GlacierAttributesManager


@R.preprocessors.add('SUMMA')
class SummaPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Preprocessor for the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    Handles complete preprocessing workflow for SUMMA including forcing data processing,
    spatial attributes generation, configuration file setup, and initial conditions.

    Key Operations:
        - Process forcing data into SUMMA-compatible NetCDF format
        - Generate spatial attribute files (HRU characteristics, elevation bands, etc.)
        - Create SUMMA configuration files (file_manager, decision files, parameter files)
        - Set up initial conditions and cold state files
        - Handle glacier attributes for glacierized catchments (if applicable)
        - Configure elevation band discretization for distributed modeling

    Workflow Steps:
        1. Initialize paths and load configuration
        2. Process forcing data (temperature, precipitation, radiation, etc.)
        3. Generate attribute files from spatial data (DEM, land cover, soil)
        4. Create SUMMA-specific configuration files
        5. Set up trial parameter files
        6. Configure output specifications

    Supported Discretizations:
        - Lumped: Single HRU per catchment
        - Distributed: Multiple HRUs with elevation bands
        - Glacier: Special handling for glacier mass balance

    Inherits from:
        BaseModelPreProcessor: Common preprocessing patterns and utilities
        ObservationLoaderMixin: Observation data loading capabilities

    Example:
        >>> from symfluence.models.summa.preprocessor import SummaPreProcessor
        >>> preprocessor = SummaPreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates SUMMA input files in: project_dir/forcing/SUMMA_input/
    """


    MODEL_NAME = "SUMMA"
    def __init__(self, config, logger):
        """
        Initialize the SummaPreProcessor.

        Args:
            config: SymfluenceConfig instance
            logger: Logger object for recording processing information
        """
        # Initialize base class (handles standard paths and common setup)
        super().__init__(config, logger)

        # SUMMA-specific paths (base class now handles shapefile_path, merged_forcing_path, intersect_path)
        self.dem_path = self.get_dem_path()
        self.forcing_summa_path = self.project_forcing_dir / 'SUMMA_input'

        # Catchment and river network (use backward-compatible path resolution)
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

        self.river_network_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        self.river_network_name = self._get_config_value(
            lambda: self.config.paths.river_network_name
        )
        if self.river_network_name == 'default' or self.river_network_name is None:
            self.river_network_name = self.get_river_network_path().name

        # SUMMA-specific configuration using typed config
        self.hruId = self._get_config_value(
            lambda: self.config.paths.catchment_hruid
        )
        self.gruId = self._get_config_value(
            lambda: self.config.paths.catchment_gruid
        )
        self.data_step = self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            self.forcing_time_step_size
        )
        self.coldstate_name = self._get_config_value(
            lambda: self.config.model.summa.coldstate if self.config.model.summa else None
        )
        self.parameter_name = self._get_config_value(
            lambda: self.config.model.summa.trialparams if self.config.model.summa else None
        )
        self.attribute_name = self._get_config_value(
            lambda: self.config.model.summa.attributes if self.config.model.summa else None
        )
        self.forcing_measurement_height = float(
            self._get_config_value(
                lambda: self.config.forcing.measurement_height,
                3.0  # Default measurement height
            )
        )

        # Initialize forcing processor
        self.forcing_processor = SummaForcingProcessor(
            config=self.config_dict,
            logger=self.logger,
            forcing_basin_path=self.forcing_basin_path,
            forcing_summa_path=self.forcing_summa_path,
            intersect_path=self.intersect_path,
            catchment_path=self.catchment_path,
            project_dir=self.project_dir,
            setup_dir=self.setup_dir,
            domain_name=self.domain_name,
            forcing_dataset=self.forcing_dataset,
            data_step=self.data_step,
            gruId=self.gruId,
            hruId=self.hruId,
            catchment_name=self.catchment_name
        )

        # Initialize configuration manager
        self.config_manager = SummaConfigManager(
            config=self.config,
            logger=self.logger,
            project_dir=self.project_dir,
            setup_dir=self.setup_dir,
            forcing_summa_path=self.forcing_summa_path,
            catchment_path=self.catchment_path,
            catchment_name=self.catchment_name,
            dem_path=self.dem_path,
            hruId=self.hruId,
            gruId=self.gruId,
            data_step=self.data_step,
            coldstate_name=self.coldstate_name,
            parameter_name=self.parameter_name,
            attribute_name=self.attribute_name,
            forcing_measurement_height=self.forcing_measurement_height,
            filter_forcing_hru_ids_callback=self._filter_forcing_hru_ids,
            get_base_settings_source_dir_callback=self.get_base_settings_source_dir,
            get_default_path_callback=self._get_default_path,
            get_simulation_times_callback=self._get_simulation_times
        )

        # Initialize attributes manager
        self.attributes_manager = SummaAttributesManager(
            config=self.config_dict,
            logger=self.logger,
            catchment_path=self.catchment_path,
            catchment_name=self.catchment_name,
            dem_path=self.dem_path,
            forcing_summa_path=self.forcing_summa_path,
            setup_dir=self.setup_dir,
            project_dir=self.project_dir,
            hruId=self.hruId,
            gruId=self.gruId,
            attribute_name=self.attribute_name,
            forcing_measurement_height=self.forcing_measurement_height,
            get_default_path_callback=self._get_default_path
        )

        # Lazy-initialize glacier attributes manager (only when needed)
        self._glacier_manager = None

    @property
    def glacier_manager(self) -> Optional[GlacierAttributesManager]:
        """Get glacier manager, initializing lazily if needed."""
        if self._glacier_manager is None:
            try:
                self._glacier_manager = GlacierAttributesManager(
                    config=self.config_dict,
                    logger=self.logger,
                    domain_name=self.domain_name,
                    dem_path=self.dem_path,
                    project_dir=self.project_dir
                )
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.debug(f"Glacier manager initialization skipped: {e}", exc_info=True)
                return None
        return self._glacier_manager

    def run_preprocessing(self):
        """
        Run the complete SUMMA spatial preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting SUMMA spatial preprocessing")
        return self.run_preprocessing_template()

    def _pre_setup(self) -> None:
        """SUMMA-specific pre-setup: apply lapse rate corrections (template hook).

        If the model-agnostic ``ElevationCorrectionProcessor`` has already
        corrected the basin-averaged files (``elevation_corrected`` NetCDF
        attribute), the SUMMA forcing processor skips its own lapse rate
        calculation while still performing SUMMA-specific fixes (time
        coordinate format, NaN interpolation, data validation).
        """
        self.apply_datastep_and_lapse_rate()

    def _prepare_forcing(self) -> None:
        """SUMMA-specific forcing preparation (template hook)."""
        self.create_forcing_file_list()

    def _create_model_configs(self) -> None:
        """SUMMA-specific configuration file creation (template hook)."""
        self.create_file_manager()
        self.create_point_file_manager_lists()
        self.create_attributes_file()
        self.create_initial_conditions()
        self.create_trial_parameters()

        # Check if glacier preprocessing is enabled
        glacier_mode = self._is_glacier_mode_enabled()
        if glacier_mode:
            self.run_glacier_preprocessing()

    def _is_glacier_mode_enabled(self) -> bool:
        """Check if glacier mode is enabled based on config, file manager name, or data presence."""
        # Check explicit config flag
        glacier_mode = self._get_config_value(
            lambda: self.config.model.summa.glacier_mode if self.config.model.summa else None,
            default=False
        )
        if glacier_mode:
            self.logger.debug("Glacier mode enabled via SETTINGS_SUMMA_GLACIER_MODE")
            return True

        # Check file manager name
        filemanager_name = self._get_config_value(
            lambda: self.config.model.summa.filemanager if self.config.model.summa else None,
            default='fileManager.txt'
        )
        if 'glac' in filemanager_name.lower():
            self.logger.debug("Glacier mode enabled via fileManager name")
            return True

        # Check for glacier raster data (only if glacier manager can be initialized)
        glacier_dir = self.project_attributes_dir / 'glaciers'
        if self.glacier_manager is not None and self.glacier_manager.has_glacier_data(glacier_dir):
            self.logger.info("Glacier raster data detected, enabling glacier mode")
            return True

        return False

    def run_glacier_preprocessing(self) -> None:
        """
        Run glacier-specific preprocessing for SUMMA.

        This method creates glacier-specific files required for SUMMA glacier simulations:
        - Glacier attributes file (attributes_glac.nc)
        - Glacier initial conditions (coldState_glac.nc)

        The glacier preprocessing is triggered when SETTINGS_SUMMA_GLACIER_MODE is True
        or when the file manager name contains 'glac'.
        """
        self.logger.info("Running glacier preprocessing for SUMMA")

        glacier = self.config.model.summa
        if glacier.glacier_outline_path or glacier.glacier_thickness_dir:
            if not (glacier.glacier_outline_path and glacier.glacier_thickness_dir):
                raise ValueError('Explicit glacier geometry requires both outlines and thickness rasters')
            from .glacier_geometry import prepare_glacier_geometry
            # These are the freshly generated inputs already named in the file
            # manager. Fail visibly rather than silently running without ice.
            prepare_glacier_geometry(
                self.setup_dir / self.attribute_name,
                self.setup_dir / self.coldstate_name,
                self.catchment_path / self.catchment_name,
                self.dem_path,
                Path(glacier.glacier_outline_path),
                Path(glacier.glacier_thickness_dir),
                glacier.glacier_grid_resolution,
                self.hruId,
            )
            return

        try:
            # Check if glacier files already exist (user-provided)
            glacier_attr_path = self.setup_dir / 'attributes_glac.nc'
            glacier_cold_path = self.setup_dir / 'coldState_glac.nc'

            if glacier_attr_path.exists() and glacier_cold_path.exists():
                self.logger.info("Glacier files already exist, skipping generation")
                self._log_glacier_files_status()
                return

            # Create glacier files from base files if they don't exist
            self._create_glacier_files()

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Glacier preprocessing failed: {e}", exc_info=True)
            self.logger.info("Continuing without glacier-specific files - using base SUMMA configuration")
            # Don't raise - allow the workflow to continue without glacier files

    def _create_glacier_files(self) -> None:
        """
        Create glacier-specific SUMMA files from raster data or base files.

        If glacier raster data exists in attributes/glaciers/, processes the
        rasters into proper glacier NetCDF files. Otherwise, falls back to
        copying base attributes and coldState files.
        """
        import shutil

        glacier_dir = self.project_attributes_dir / 'glaciers'
        base_attr = self.setup_dir / self.attribute_name

        # Check if we can process raster data (glacier manager must be available)
        can_process_raster = (
            self.glacier_manager is not None and
            self.glacier_manager.has_glacier_data(glacier_dir) and
            base_attr.exists()
        )

        if can_process_raster:
            self.logger.info("Processing glacier attributes from raster data")
            success = self.glacier_manager.process_glacier_attributes(
                glacier_dir=glacier_dir,
                settings_dir=self.setup_dir,
                base_attributes_file=base_attr
            )
            if success:
                self._log_glacier_files_status()
                return
            else:
                self.logger.warning("Glacier raster processing failed, falling back to base files")

        # Fallback: Copy base files to glacier files
        base_cold = self.setup_dir / self.coldstate_name

        glacier_attr_name = self._get_config_value(
            lambda: self.config.model.summa.glacier_attributes if self.config.model.summa else None,
            default='attributes_glac.nc'
        )
        glacier_cold_name = self._get_config_value(
            lambda: self.config.model.summa.glacier_coldstate if self.config.model.summa else None,
            default='coldState_glac.nc'
        )

        glacier_attr_path = self.setup_dir / glacier_attr_name
        glacier_cold_path = self.setup_dir / glacier_cold_name

        # Copy base files to glacier files if they don't exist
        if not glacier_attr_path.exists() and base_attr.exists():
            shutil.copy2(base_attr, glacier_attr_path)
            self.logger.info(f"Created {glacier_attr_name} from {self.attribute_name}")

        if not glacier_cold_path.exists() and base_cold.exists():
            shutil.copy2(base_cold, glacier_cold_path)
            self.logger.info(f"Created {glacier_cold_name} from {self.coldstate_name}")

        self._log_glacier_files_status()

    def _log_glacier_files_status(self) -> None:
        """Log the status of glacier files."""
        glacier_files = [
            'attributes_glac.nc',
            'coldState_glac.nc',
        ]
        for gf in glacier_files:
            path = self.setup_dir / gf
            status = "EXISTS" if path.exists() else "MISSING"
            self.logger.info(f"Glacier file {gf}: {status}")

    def copy_base_settings(self):  # type: ignore[override]
        """
        Copy SUMMA base settings from the source directory to the project's settings directory.

        Delegates to the configuration manager.
        """
        self.config_manager.copy_base_settings()


    def create_file_manager(self):
        """
        Create the SUMMA file manager configuration file.

        Delegates to the configuration manager.
        """
        self.config_manager.create_file_manager()

    def create_point_file_manager_lists(self):
        """
        Create file manager list files for point simulations.

        For point domain simulations, the runner expects list files that point to
        individual file managers. For simple point simulations, we create lists
        that point to the main file manager.
        """
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method,
            default=''
        )

        if domain_method != 'point':
            # Only create list files for point simulations
            return

        self.logger.info("Creating file manager lists for point simulation")

        # Get the file manager path
        filemanager_name = self._get_config_value(
            lambda: self.config.model.summa.filemanager if self.config.model.summa else None,
            default='fileManager.txt'
        )
        filemanager_path = self.setup_dir / filemanager_name

        # Create list files pointing to the main file manager
        fm_list_path = self.setup_dir / 'list_fileManager.txt'
        fm_ic_list_path = self.setup_dir / 'list_fileManager_IC.txt'

        # For a simple point simulation, both lists point to the same file manager
        with open(fm_list_path, 'w', encoding='utf-8') as f:
            f.write(f"{filemanager_path}\n")

        with open(fm_ic_list_path, 'w', encoding='utf-8') as f:
            f.write(f"{filemanager_path}\n")

        self.logger.info(f"Created file manager lists at {self.setup_dir}")

    def apply_datastep_and_lapse_rate(self):
        """
        Apply temperature lapse rate corrections to forcing data.

        Delegates to the forcing processor for actual implementation.
        """
        self.forcing_processor.apply_datastep_and_lapse_rate()
        if self.data_step != self.forcing_processor.data_step:
            self.data_step = self.forcing_processor.data_step
            self.config_manager.data_step = self.forcing_processor.data_step
            self.logger.info(f"Updated SUMMA data step to {self.data_step}s based on forcing data")

    def create_forcing_file_list(self):
        """
        Create a list of forcing files for SUMMA.

        Delegates to the forcing processor for actual implementation.
        """
        self.forcing_processor.create_forcing_file_list()

    def _filter_forcing_hru_ids(self, forcing_hru_ids):
        """
        Filter forcing HRU IDs against catchment shapefile.

        Delegates to the forcing processor for actual implementation.

        Args:
            forcing_hru_ids: List or array of HRU IDs from forcing data

        Returns:
            Filtered list of HRU IDs
        """
        return self.forcing_processor._filter_forcing_hru_ids(forcing_hru_ids)



    def create_initial_conditions(self):
        """
        Create the initial conditions (cold state) file for SUMMA.

        Delegates to the configuration manager.
        """
        self.config_manager.create_initial_conditions()


    def create_trial_parameters(self):
        """
        Create the trial parameters file for SUMMA.

        Delegates to the configuration manager.
        """
        self.config_manager.create_trial_parameters()



    def create_attributes_file(self):
        """
        Create the attributes file for SUMMA.

        Delegates to the attributes manager for actual implementation.
        """
        self.attributes_manager.create_attributes_file()

    def _get_simulation_times(self) -> tuple[str, str]:
        """
        Get the simulation start and end times from config or calculate defaults.

        Returns:
            tuple[str, str]: A tuple containing the simulation start and end times.

        Raises:
            ValueError: If the time format in the configuration is invalid.
        """
        # Use typed config
        sim_start = self._get_config_value(
            lambda: self.config.domain.time_start
        )
        sim_end = self._get_config_value(
            lambda: self.config.domain.time_end
        )

        if sim_start == 'default' or sim_end == 'default':
            start_year = sim_start.split('-')[0] if sim_start != 'default' else None
            end_year = sim_end.split('-')[0] if sim_end != 'default' else None
            if not start_year or not end_year:
                raise ConfigValidationError("EXPERIMENT_TIME_START or EXPERIMENT_TIME_END is missing from configuration")
            sim_start = f"{start_year}-01-01 01:00" if sim_start == 'default' else sim_start
            sim_end = f"{end_year}-12-31 22:00" if sim_end == 'default' else sim_end

        forcing_times = self._get_forcing_times()
        if forcing_times:
            sim_start_dt = datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            sim_end_dt = datetime.strptime(sim_end, "%Y-%m-%d %H:%M")

            start_floor = max((t for t in forcing_times if t <= sim_start_dt), default=forcing_times[0])
            end_ceil = min((t for t in forcing_times if t >= sim_end_dt), default=forcing_times[-1])

            if start_floor != sim_start_dt:
                self.logger.info(
                    f"Adjusting SUMMA start time to forcing timestep: {start_floor}"
                )
            if end_ceil != sim_end_dt:
                self.logger.info(
                    f"Adjusting SUMMA end time to forcing timestep: {end_ceil}"
                )

            if start_floor > end_ceil:
                self.logger.warning(
                    "Forcing timesteps do not cover requested range; using full forcing span."
                )
                start_floor = forcing_times[0]
                end_ceil = forcing_times[-1]

            sim_start = start_floor.strftime("%Y-%m-%d %H:%M")
            sim_end = end_ceil.strftime("%Y-%m-%d %H:%M")

        # Validate time format
        try:
            datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            datetime.strptime(sim_end, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ConfigValidationError("Invalid time format in configuration. Expected 'YYYY-MM-DD HH:MM'") from None

        return sim_start, sim_end

    def _get_forcing_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        forcing_times = self._get_forcing_times()
        if not forcing_times:
            return None

        return (forcing_times[0], forcing_times[-1])

    def _get_forcing_times(self) -> list[datetime]:
        forcing_dir = self.forcing_summa_path
        if not forcing_dir.exists():
            return []

        forcing_files = sorted(forcing_dir.glob("*.nc"))
        if not forcing_files:
            return []

        unique_times: set[datetime] = set()
        for forcing_file in forcing_files:
            try:
                with xr.open_dataset(forcing_file) as ds:
                    if "time" not in ds:
                        continue
                    time_raw = ds["time"].values
                    if len(time_raw) > 0 and hasattr(time_raw[0], 'strftime') and not isinstance(time_raw[0], datetime):
                        times = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in time_raw])
                    else:
                        times = pd.to_datetime(time_raw)
            except Exception as exc:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Failed to read forcing times from {forcing_file}: {exc}", exc_info=True)
                continue

            if len(times) == 0:
                continue
            unique_times.update(times.to_pydatetime())

        if not unique_times:
            return []

        return sorted(unique_times)
