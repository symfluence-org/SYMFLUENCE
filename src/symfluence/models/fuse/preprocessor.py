# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE model preprocessor for data preparation and configuration.

Prepares forcing data, calculates PET, generates FUSE input files,
and handles synthetic data generation for the FUSE hydrological model.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.core.exceptions import FileOperationError, ModelExecutionError
from symfluence.core.geometry_utils import GeospatialUtilsMixin
from symfluence.core.modeling.base import BaseModelPreProcessor
from symfluence.core.modeling.mixins import DatasetBuilderMixin, PETCalculatorMixin, SpatialModeDetectionMixin
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.modeling.variable_utils import VariableHandler
from symfluence.core.registries import R

from .elevation_band_manager import FuseElevationBandManager
from .forcing_processor import FuseForcingProcessor
from .synthetic_data_generator import FuseSyntheticDataGenerator


@R.preprocessors.add('FUSE')
class FUSEPreProcessor(BaseModelPreProcessor, PETCalculatorMixin, GeospatialUtilsMixin, DatasetBuilderMixin, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Preprocessor for the FUSE (Framework for Understanding Structural Errors) model.

    Handles complete preprocessing workflow for FUSE including forcing data processing,
    PET calculation, elevation band discretization, and configuration file generation.
    FUSE is a modular modeling framework that allows testing different model structures.

    Key Operations:
        - Process forcing data (precipitation, temperature, PET)
        - Calculate potential evapotranspiration using multiple methods
        - Generate elevation band discretization for distributed modeling
        - Create FUSE control and parameter files
        - Generate synthetic forcing data (for testing/calibration)
        - Calculate catchment attributes (area, elevation statistics)
        - Handle both lumped and distributed spatial configurations

    Workflow Steps:
        1. Initialize paths and load configuration
        2. Process forcing data and calculate PET
        3. Generate elevation bands (if distributed mode)
        4. Create FUSE control file (fm_control.txt)
        5. Set up parameter files with model structure selections
        6. Configure input/output file specifications
        7. Prepare observation data for evaluation

    Supported Spatial Modes:
        - Lumped: Single catchment unit
        - Distributed: Multiple elevation bands with area weighting
        - Synthetic: Generate idealized forcing for testing

    PET Calculation Methods:
        - Hamon: Temperature-based method
        - Priestley-Taylor: Radiation-based method
        - Penman-Monteith: Energy balance method (requires additional variables)

    Inherits from:
        BaseModelPreProcessor: Common preprocessing patterns and utilities
        PETCalculatorMixin: Potential evapotranspiration calculation methods
        GeospatialUtilsMixin: Spatial operations (centroid, area calculation)
        ObservationLoaderMixin: Observation data loading capabilities
        DatasetBuilderMixin: NetCDF dataset construction utilities

    Attributes:
        config (SymfluenceConfig): Typed configuration object
        logger: Logger object for recording processing information
        project_dir (Path): Directory for the current project
        forcing_fuse_path (Path): Directory for FUSE input files
        catchment_path (Path): Path to catchment shapefile
        forcing_processor (FuseForcingProcessor): Handles forcing data transformation
        elevation_band_manager (FuseElevationBandManager): Manages elevation discretization
        synthetic_generator (FuseSyntheticDataGenerator): Generates synthetic forcing

    Example:
        >>> from symfluence.models.fuse.preprocessor import FUSEPreProcessor
        >>> preprocessor = FUSEPreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates FUSE input files in: project_dir/forcing/FUSE_input/
        # Generates: input_info.txt, elev_bands.txt, fm_control.txt
    """

    MODEL_NAME = "FUSE"
    _fuse_file_id: Optional[str] = None

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the FUSE preprocessor.

        Sets up FUSE-specific paths, forcing processors, elevation band manager,
        and synthetic data generator. The preprocessor handles complete FUSE
        input preparation including forcing data, PET calculation, and
        configuration file generation.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                FUSE model settings, paths, PET method, and spatial mode options.
            logger: Logger instance for status messages and debugging output.

        Note:
            Creates three delegate objects for modular processing:
            - forcing_processor: Handles forcing data transformation
            - elevation_band_manager: Manages elevation discretization
            - synthetic_data_generator: Creates synthetic forcing for testing
        """
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        # FUSE-specific paths
        self.forcing_fuse_path = self.project_forcing_dir / 'FUSE_input'

        # Setup catchment path using base class helper
        self.catchment_path = self.get_catchment_path()
        self.catchment_name = self.catchment_path.name

        # Initialize forcing processor (use base class methods for callbacks)
        self.forcing_processor = FuseForcingProcessor(
            config=self.config_dict,
            logger=self.logger,
            project_dir=self.project_dir,
            forcing_basin_path=self.forcing_basin_path,
            forcing_fuse_path=self.forcing_fuse_path,
            catchment_path=self.catchment_path,
            domain_name=self.domain_name,
            calculate_pet_callback=self._calculate_pet,
            calculate_catchment_centroid_callback=self.calculate_catchment_centroid,
            get_simulation_time_window_callback=self.get_simulation_time_window,
            subset_to_simulation_time_callback=self.subset_to_simulation_time
        )

        # Initialize elevation band manager
        self.elevation_band_manager = FuseElevationBandManager(
            config=self.config_dict,
            logger=self.logger,
            project_dir=self.project_dir,
            forcing_fuse_path=self.forcing_fuse_path,
            catchment_path=self.catchment_path,
            domain_name=self.domain_name,
            calculate_catchment_centroid_callback=self.calculate_catchment_centroid
        )

        # Initialize synthetic data generator
        self.synthetic_data_generator = FuseSyntheticDataGenerator(logger=self.logger)

    def _get_fuse_file_id(self) -> str:
        """Get a short file ID for FUSE outputs and settings.

        FUSE Fortran uses a CHARACTER(LEN=6) buffer for FMODEL_ID,
        so the ID must be kept to 6 chars max to avoid truncation.
        """
        if hasattr(self, '_fuse_file_id') and self._fuse_file_id is not None:
            return self._fuse_file_id
        fuse_id = self._get_config_value(
            lambda: self.config.model.fuse.file_id,
            default=None
        )
        if not fuse_id:
            fuse_id = self.experiment_id if self.experiment_id else 'fuse'
        if len(fuse_id) > 6:
            import hashlib
            fuse_id = hashlib.md5(fuse_id.encode(), usedforsecurity=False).hexdigest()[:6]
        self._fuse_file_id = fuse_id
        return fuse_id

    def _get_timestep_config(self):
        """
        Get FUSE-specific timestep configuration.

        FUSE only supports daily timesteps, so this method forces daily
        resolution regardless of the base config. This differs from
        the base class get_timestep_config() which respects the config.

        Returns:
            dict: Configuration with resample_freq, time_units, time_unit,
                conversion_factor, time_label, and timestep_seconds
        """
        ts_config = self.get_timestep_config()
        if ts_config['time_unit'] == 'h':
            self.logger.debug("FUSE uses daily timesteps; overriding hourly config")
            return {
                'resample_freq': 'D',
                'time_units': 'days since 1970-01-01',
                'time_unit': 'D',
                'conversion_factor': UnitConversion.MM_DAY_TO_CMS,
                'time_label': 'daily',
                'timestep_seconds': 86400
            }
        return ts_config

    def run_preprocessing(self):
        """
        Run the complete FUSE preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.debug("Starting FUSE preprocessing")
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """FUSE-specific forcing data preparation (template hook)."""
        self.prepare_forcing_data()

    def _create_model_configs(self) -> None:
        """FUSE-specific configuration file creation (template hook)."""
        self.create_elevation_bands()
        self.update_input_info()
        self.create_filemanager()

    def create_directories(self, additional_dirs: Optional[List[Path]] = None):
        """Create necessary directories for FUSE setup."""
        # FUSE-specific directories (setup_dir and forcing_dir handled by base)
        fuse_dirs = [self.forcing_fuse_path]
        if additional_dirs:
            fuse_dirs.extend(additional_dirs)
        super().create_directories(additional_dirs=fuse_dirs)

    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[List[str]] = None):
        """
        Copy FUSE base settings from the source directory to the project's settings directory.
        Updates the model ID in the decisions file name to match the experiment ID.
        """
        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)

        self.logger.debug("Copying FUSE base settings")

        from symfluence.core.modeling.base_settings import get_base_settings_dir
        base_settings_path = get_base_settings_dir('FUSE')
        settings_path = self._get_default_path('SETTINGS_FUSE_PATH', 'settings/FUSE')

        try:
            settings_path.mkdir(parents=True, exist_ok=True)

            fuse_id = self._get_fuse_file_id()
            decision_file_path = None
            for file in os.listdir(base_settings_path):
                source_file = base_settings_path / file

                # Handle the decisions file specially
                if 'fuse_zDecisions_' in file:
                    # Create new filename with experiment ID
                    new_filename = file.replace('902', fuse_id)
                    dest_file = settings_path / new_filename
                    decision_file_path = dest_file
                    self.logger.debug(f"Renaming decisions file from {file} to {new_filename}")
                else:
                    dest_file = settings_path / file

                copyfile(source_file, dest_file)
                self.logger.debug(f"Copied {source_file} to {dest_file}")

            # Apply FUSE_DECISION_OPTIONS from config to update model decisions
            decision_options = self._get_config_value(
                lambda: self.config.model.fuse.decision_options,
                default={}
            )
            snow_model = self._get_config_value(
                lambda: self.config.model.fuse.snow_model,
                default=None
            )

            if decision_file_path and decision_file_path.exists() and (decision_options or snow_model):
                try:
                    with open(decision_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    with open(decision_file_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()

                # Build a map of decision key -> value to apply
                decisions_to_apply = {}
                for key, values in decision_options.items():
                    if isinstance(values, list) and values:
                        decisions_to_apply[key] = values[0]
                    elif isinstance(values, str):
                        decisions_to_apply[key] = values

                # FUSE_SNOW_MODEL as fallback for SNOWM if not in FUSE_DECISION_OPTIONS
                if snow_model and 'SNOWM' not in decisions_to_apply:
                    decisions_to_apply['SNOWM'] = snow_model

                # Update decision lines (lines 2-10, 0-indexed 1-9)
                for i in range(1, min(10, len(lines))):
                    parts = lines[i].split()
                    if len(parts) >= 2:
                        decision_key = parts[1]
                        if decision_key in decisions_to_apply:
                            new_value = decisions_to_apply[decision_key]
                            rest_of_line = ' '.join(parts[1:])
                            lines[i] = f"{new_value:<10} {rest_of_line}\n"
                            self.logger.debug(f"Updated FUSE decision {decision_key} to {new_value}")

                with open(decision_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                if decisions_to_apply:
                    self.logger.info(f"Applied FUSE decision options: {decisions_to_apply}")

            self.logger.info(f"FUSE base settings copied to {settings_path}")

        except FileNotFoundError as e:
            self.logger.error(f"Source file or directory not found: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission error when copying files: {e}")
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error copying base settings: {e}")
            raise

    def generate_synthetic_hydrograph(self, ds, area_km2, mean_temp_threshold=0.0):
        """Generate synthetic hydrograph - delegates to synthetic data generator"""
        return self.synthetic_data_generator.generate_synthetic_hydrograph(ds, area_km2, mean_temp_threshold)

    def prepare_forcing_data(self):
        """Prepare forcing data for lumped, semi-distributed, or distributed FUSE."""
        try:
            t0 = time.time()
            ts_config = self._get_timestep_config()
            spatial_mode = self._get_config_value(
                lambda: self.config.model.fuse.spatial_mode,
                default=None
            ) or self._infer_spatial_mode_from_domain()
            subcatchment_dim = self._get_config_value(
                lambda: self.config.model.fuse.subcatchment_dim,
                default='longitude'
            )

            ds = self._load_forcing_dataset(ts_config)
            ds = self._organize_and_resample(ds, spatial_mode, subcatchment_dim, ts_config)
            obs_ds, pet = self._compute_observations_and_pet(ds, spatial_mode, ts_config)
            output_file = self._save_fuse_forcing(ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config)

            self.logger.info(f"PERF: Total prepare_forcing_data took {time.time() - t0:.2f}s")
            return output_file

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Helpers for prepare_forcing_data
    # ------------------------------------------------------------------

    def _load_forcing_dataset(self, ts_config):
        """Open forcing files, apply variable handler, and subset to sim time."""
        self.logger.debug(f"Using {ts_config['time_label']} timestep (resample freq: {ts_config['resample_freq']})")

        forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
        if not forcing_files:
            raise FileOperationError("No forcing files found in basin-averaged data directory")

        variable_handler = VariableHandler(
            config=self.config_dict, logger=self.logger, dataset='CFIF', model='FUSE'
        )
        self.logger.debug(f"FUSE forcing_files count: {len(forcing_files)}")

        t1 = time.time()
        if len(forcing_files) == 1:
            ds = xr.open_dataset(forcing_files[0])
        else:
            ds = xr.open_mfdataset(
                forcing_files, data_vars='minimal', combine='nested', concat_dim='time', coords='minimal', compat='override'
            ).sortby('time')
        self.logger.info(f"PERF: Opening dataset took {time.time() - t1:.2f}s")

        t2 = time.time()
        ds = variable_handler.process_forcing_data(ds)
        self.logger.info(f"PERF: process_forcing_data took {time.time() - t2:.2f}s")

        t3 = time.time()
        ds = self.subset_to_simulation_time(ds, "Forcing")
        self.logger.info(f"PERF: subset_to_simulation_time took {time.time() - t3:.2f}s")

        return ds

    def _organize_and_resample(self, ds, spatial_mode, subcatchment_dim, ts_config):
        """Apply spatial organisation and resample to target timestep."""
        self.logger.debug(f"Preparing FUSE forcing data in {spatial_mode} mode")

        t4 = time.time()
        if spatial_mode == SpatialMode.LUMPED:
            ds = self._prepare_lumped_forcing(ds)
        elif spatial_mode == SpatialMode.SEMI_DISTRIBUTED:
            ds = self._prepare_semi_distributed_forcing(ds, subcatchment_dim)
        elif spatial_mode == SpatialMode.DISTRIBUTED:
            ds = self._prepare_distributed_forcing(ds)
        else:
            raise ModelExecutionError(f"Unknown FUSE spatial mode: {spatial_mode}")
        self.logger.info(f"PERF: Spatial prep ({spatial_mode}) took {time.time() - t4:.2f}s")

        t5 = time.time()
        self.logger.debug(f"Resampling data to {ts_config['time_label']} resolution")
        ds = ds.resample(time=ts_config['resample_freq']).mean()
        self.logger.info(f"PERF: Resampling took {time.time() - t5:.2f}s")

        return ds

    def _compute_observations_and_pet(self, ds, spatial_mode, ts_config):
        """Load streamflow observations and calculate PET."""
        t6 = time.time()
        time_window = self.get_simulation_time_window()
        obs_ds = self._load_streamflow_observations(spatial_mode, ts_config, time_window)
        self.logger.info(f"PERF: Loading observations took {time.time() - t6:.2f}s")

        pet_method = self._get_config_value(
            lambda: self.config.forcing.pet_method,
            default='oudin'
        ).lower()
        self.logger.debug(f"Using PET method: {pet_method}")

        t7 = time.time()
        if spatial_mode == SpatialMode.LUMPED:
            catchment = gpd.read_file(self.catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
            pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
        else:
            pet = self._calculate_distributed_pet(ds, spatial_mode, pet_method)

        pet = pet.resample(time=ts_config['resample_freq']).mean()
        self.logger.debug(f"PET data resampled to {ts_config['time_label']} resolution")
        self.logger.info(f"PERF: PET calculation took {time.time() - t7:.2f}s")

        return obs_ds, pet

    def _save_fuse_forcing(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config):
        """Create FUSE dataset, capture dates, materialise, and write NetCDF."""
        t8 = time.time()
        fuse_forcing = self._create_fuse_forcing_dataset(
            ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config
        )
        self.logger.info(f"PERF: _create_fuse_forcing_dataset took {time.time() - t8:.2f}s")

        # Capture actual simulation dates (FUSE uses days since 1970-01-01)
        try:
            t9 = time.time()
            time_vals = fuse_forcing['time'].values
            if len(time_vals) > 0:
                dates = pd.to_datetime(time_vals, unit='D', origin='1970-01-01')
                self.actual_start_time = dates.min().to_pydatetime()
                self.actual_end_time = dates.max().to_pydatetime()
                self.logger.info(f"Captured actual simulation dates: {self.actual_start_time} to {self.actual_end_time}")
            self.logger.info(f"PERF: Date capture took {time.time() - t9:.2f}s")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not capture actual dates from FUSE forcing: {e}", exc_info=True)

        try:
            if hasattr(ds, "close"):
                ds.close()
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(f"Could not close forcing dataset: {e}", exc_info=True)

        self.logger.info("Materializing FUSE forcing dataset before write")
        fuse_forcing = fuse_forcing.load()

        t10 = time.time()
        output_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        encoding = self._get_encoding_dict(fuse_forcing)
        fuse_forcing.to_netcdf(output_file, unlimited_dims=['time'],
                               encoding=encoding, format='NETCDF4')

        self.logger.info(f"PERF: Saving NetCDF took {time.time() - t10:.2f}s")
        self.logger.debug(f"FUSE forcing data saved: {output_file}")
        return output_file


    def _calculate_pet(self, temp_data: xr.DataArray, lat: float, method: str = 'oudin') -> xr.DataArray:
        """
        Calculate PET using the specified method.

        Args:
            temp_data (xr.DataArray): Temperature data
            lat (float): Latitude of the catchment centroid
            method (str): PET method ('oudin', 'hamon', or 'hargreaves')

        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        method = method.lower()

        if method == 'oudin':
            return self.calculate_pet_oudin(temp_data, lat)
        elif method == 'hamon':
            return self.calculate_pet_hamon(temp_data, lat)
        elif method == 'hargreaves':
            return self.calculate_pet_hargreaves(temp_data, lat)
        else:
            self.logger.warning(f"Unknown PET method '{method}', defaulting to Oudin")
            return self.calculate_pet_oudin(temp_data, lat)

    def _add_forcing_variables(self, fuse_forcing, ds, pet, obs_ds, spatial_dims, n_subcatchments, ts_config=None):
        """
        Add forcing variables to the FUSE dataset using efficient xarray broadcasting.

        Processes precipitation, temperature, PET, and streamflow observations
        and adds them to the FUSE forcing dataset with proper coordinate alignment,
        NaN handling, and encoding for FUSE compatibility.

        Args:
            fuse_forcing: Target xarray Dataset with coordinate structure defined.
            ds: Source forcing dataset containing 'precip' and 'temp' variables.
            pet: Potential evapotranspiration DataArray.
            obs_ds: Observation dataset with 'q_obs' or None for synthetic generation.
            spatial_dims: Tuple of dimension names, e.g., ('time', 'latitude', 'longitude').
            n_subcatchments: Number of subcatchments for distributed mode.
            ts_config: Timestep configuration dict (uses _get_timestep_config if None).

        Returns:
            dict: Encoding dictionary for NetCDF output with compression settings.

        Note:
            FUSE interprets -9999 as actual negative values, not missing data.
            This method fills NaN values with interpolated/mean data to avoid
            FUSE misinterpreting missing data.
        """

        if ts_config is None:
            ts_config = self._get_timestep_config()

        time_label = ts_config['time_label']
        unit_str = f"mm/{time_label.replace('-', ' ')}"

        # Determine target spatial dimension (latitude or longitude based on config)
        subcatchment_dim = self._get_config_value(
            lambda: self.config.model.fuse.subcatchment_dim,
            default='longitude'
        )

        # Alignment time index (already in fuse_forcing.time but we need it as datetime for reindexing)
        # FUSE uses numeric time, but we'll use datetime for internal alignment
        time_index = pd.to_datetime('1970-01-01') + pd.to_timedelta(fuse_forcing.time.values, unit='D')

        def process_var(da, name):
            pv_start = time.time()
            # Ensure data has datetime index for alignment
            if not pd.api.types.is_datetime64_any_dtype(da.time.dtype):
                da = da.assign_coords(time=pd.to_datetime('1970-01-01') + pd.to_timedelta(da.time.values, unit='D'))

            # Align to the expected time index
            da_aligned = da.reindex(time=time_index, method='nearest', tolerance='1D').ffill(dim='time')

            # Switch back to numeric time to match fuse_forcing exactly (avoids DTypePromotionError)
            # Drop time first to ensure clean replacement of coordinate and index
            da_aligned = da_aligned.drop_vars('time').assign_coords(time=fuse_forcing.time)

            # CRITICAL FIX: Map 'hru' dimension to the target spatial dimension (latitude/longitude)
            # This ensures FUSE gets 3D (time, lat, lon) not 4D (time, lat, lon, hru)
            if 'hru' in da_aligned.dims:
                # Remove target dimension if it exists as a coordinate to avoid conflict
                if subcatchment_dim in da_aligned.coords:
                    da_aligned = da_aligned.drop_vars(subcatchment_dim)

                # Rename hru to the target spatial dimension
                da_aligned = da_aligned.rename({'hru': subcatchment_dim})
                # Assign the correct coordinate values from fuse_forcing
                da_aligned = da_aligned.assign_coords({subcatchment_dim: fuse_forcing[subcatchment_dim].values})

            # Broadcast to spatial dimensions (now should only add the singleton dimension)
            da_broadcasted = da_aligned.broadcast_like(fuse_forcing)

            # CRITICAL: Handle NaN values properly for FUSE
            # FUSE interprets -9999 as actual negative values, not as missing data
            # We need to fill NaN values with interpolated/mean data
            t_copy = time.time()
            da_values = da_broadcasted.values.copy()
            self.logger.debug(f"PERF: [{name}] da.values copy took {time.time() - t_copy:.4f}s")

            nan_mask = np.isnan(da_values)

            if np.any(nan_mask):
                # Find subcatchments with all-NaN data
                all_nan_mask = np.all(nan_mask, axis=0)  # Shape: (lat, lon) or similar

                if np.any(all_nan_mask):
                    self.logger.warning(f"Variable '{name}' has {np.sum(all_nan_mask)} subcatchments with all NaN values - filling with spatial mean")

                    # Compute mean across valid subcatchments for each timestep
                    valid_data = np.where(nan_mask, np.nan, da_values)
                    spatial_mean = np.nanmean(valid_data, axis=(1, 2), keepdims=True)

                    # Fill NaN values with spatial mean
                    da_values = np.where(nan_mask, spatial_mean, da_values)

                    # If still NaN (all data missing for a timestep), use temporal mean
                    still_nan = np.isnan(da_values)
                    if np.any(still_nan):
                        temporal_mean = np.nanmean(da_values)
                        da_values = np.where(still_nan, temporal_mean, da_values)
                else:
                    # Scattered NaN values - fill with forward fill then backward fill
                    self.logger.debug(f"Variable '{name}' has {np.sum(nan_mask)} scattered NaN values - filling with interpolation")
                    da_values = np.where(nan_mask, np.nan, da_values)
                    # Use xarray for interpolation
                    da_temp = xr.DataArray(da_values, dims=da_broadcasted.dims, coords=da_broadcasted.coords)
                    da_temp = da_temp.ffill(dim='time').bfill(dim='time')
                    da_values = da_temp.values

            # Ensure non-negative for precipitation (floating-point precision can cause tiny negative values)
            if name == 'precip':
                da_values = np.maximum(da_values, 0.0)

            self.logger.debug(f"PERF: [{name}] process_var total took {time.time() - pv_start:.4f}s")
            result = xr.DataArray(da_values, dims=da_broadcasted.dims, coords=da_broadcasted.coords).astype('float32')
            return result.transpose(*spatial_dims)

        # Map variables
        var_map = {
            'precip': (ds['precip'], 'precipitation', unit_str, f'Mean {time_label} precipitation'),
            'temp': (ds['temp'], 'temperature', 'degC', f'Mean {time_label} temperature'),
            'pet': (pet, 'pet', unit_str, f'Mean {time_label} pet')
        }

        if obs_ds is not None and 'q_obs' in obs_ds:
            var_map['q_obs'] = (obs_ds['q_obs'], 'streamflow', unit_str, f'Mean observed {time_label} discharge')
        else:
            self.logger.warning(
                "Streamflow observations not found — writing dummy q_obs (-9999) to FUSE forcing file. "
                "Observations will be needed for calibration."
            )
            dummy_q = xr.full_like(ds['precip'], -9999.0)
            dummy_q.name = 'q_obs'
            var_map['q_obs'] = (dummy_q, 'streamflow', unit_str, 'Dummy observed discharge (missing)')

        # Process and add to dataset
        encoding = {}
        for var_name, (da, standard_name, units, long_name) in var_map.items():
            fuse_forcing[var_name] = process_var(da, var_name)
            fuse_forcing[var_name].attrs = {'units': units, 'long_name': long_name}
            encoding[var_name] = {'_FillValue': -9999.0, 'dtype': 'float32', 'zlib': False}

        # Ensure coordinates also have strict encoding for FUSE compatibility
        for coord in fuse_forcing.coords:
            encoding[coord] = {'_FillValue': None, 'dtype': 'float64'}

        return encoding

    def _generate_distributed_synthetic_hydrograph(self, ds, n_subcatchments, time_length):
        """
        Generate distributed synthetic hydrograph for testing and calibration.

        Creates synthetic streamflow data when observed data is unavailable,
        useful for model testing and structure exploration. Delegates to
        FuseSyntheticDataGenerator for the actual computation.

        Args:
            ds: Source forcing dataset with precipitation and temperature.
            n_subcatchments: Number of subcatchments to generate data for.
            time_length: Number of timesteps in the output.

        Returns:
            numpy.ndarray: Synthetic streamflow array of shape (time_length, n_subcatchments).
        """
        return self.synthetic_data_generator.generate_distributed_synthetic_hydrograph(ds, n_subcatchments, time_length)

    def _prepare_lumped_forcing(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare lumped forcing data - delegates to forcing processor"""
        return self.forcing_processor._prepare_lumped_forcing(ds)

    def _prepare_semi_distributed_forcing(self, ds: xr.Dataset, subcatchment_dim: str) -> xr.Dataset:
        """Prepare semi-distributed forcing data - delegates to forcing processor"""
        return self.forcing_processor._prepare_semi_distributed_forcing(ds, subcatchment_dim)

    def _prepare_distributed_forcing(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare fully distributed forcing data - delegates to forcing processor"""
        return self.forcing_processor._prepare_distributed_forcing(ds)

    def _load_subcatchment_data(self) -> np.ndarray:
        """Load subcatchment information - delegates to forcing processor"""
        return self.forcing_processor._load_subcatchment_data()

    def _create_fuse_forcing_dataset(
        self,
        ds: xr.Dataset,
        pet: xr.DataArray,
        obs_ds: Optional[xr.Dataset],
        spatial_mode: str,
        subcatchment_dim: str,
        ts_config: Optional[Dict[str, Any]] = None
    ) -> xr.Dataset:
        """
        Create the final FUSE forcing dataset with proper coordinate structure.

        Routes to the appropriate dataset creation method based on spatial mode.
        Lumped mode creates a single-point dataset while distributed modes
        create multi-dimensional datasets with subcatchment coordinates.

        Args:
            ds: Processed forcing dataset with 'precip' and 'temp' variables.
            pet: Calculated potential evapotranspiration DataArray.
            obs_ds: Observation dataset with 'q_obs' or None for synthetic data.
            spatial_mode: One of 'lumped', 'semi_distributed', or 'distributed'.
            subcatchment_dim: Dimension for subcatchments ('latitude' or 'longitude').
            ts_config: Timestep configuration dict (uses _get_timestep_config if None).

        Returns:
            xr.Dataset: Complete FUSE forcing dataset ready for model input.
        """

        if ts_config is None:
            ts_config = self._get_timestep_config()

        if spatial_mode == SpatialMode.LUMPED:
            return self._create_lumped_dataset(ds, pet, obs_ds, ts_config)
        else:
            return self._create_distributed_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config)

    def _create_distributed_dataset(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim, ts_config=None):
        """
        Create distributed or semi-distributed FUSE forcing dataset.

        Builds a multi-dimensional forcing dataset for distributed FUSE runs
        where subcatchments are represented along either latitude or longitude
        dimension. Uses efficient xarray operations for broadcasting.

        Args:
            ds: Processed forcing dataset with 'precip' and 'temp' variables.
            pet: Calculated potential evapotranspiration DataArray.
            obs_ds: Observation dataset with 'q_obs' or None for synthetic data.
            spatial_mode: Either 'semi_distributed' or 'distributed'.
            subcatchment_dim: Dimension for subcatchments ('latitude' or 'longitude').
            ts_config: Timestep configuration dict (uses _get_timestep_config if None).

        Returns:
            xr.Dataset: FUSE forcing dataset with dimensions (time, latitude, longitude)
                where one spatial dimension contains subcatchment IDs and the other
                contains the catchment centroid coordinate.
        """

        if ts_config is None:
            ts_config = self._get_timestep_config()

        # Get spatial information
        subcatchments = self._load_subcatchment_data()

        # Get reference coordinates
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)

        # Convert to numeric time values for FUSE
        time_numeric = ((pd.to_datetime(ds.time.values) - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values

        # Create coordinate system
        # CRITICAL: Order must be time, lat, lon for FUSE to read correct dimensions
        if subcatchment_dim == 'latitude':
            coords = {
                'time': ('time', time_numeric),
                'latitude': ('latitude', subcatchments.astype(float)),
                'longitude': ('longitude', [float(mean_lon)])
            }
            spatial_dims = ('time', 'latitude', 'longitude')
        else:  # longitude
            coords = {
                'time': ('time', time_numeric),
                'longitude': ('longitude', subcatchments.astype(float)),
                'latitude': ('latitude', [float(mean_lat)])
            }
            spatial_dims = ('time', 'latitude', 'longitude')

        # Create dataset
        fuse_forcing = xr.Dataset(coords=coords)

        # Add coordinate attributes
        fuse_forcing.longitude.attrs = {
            'units': 'degreesE' if subcatchment_dim != 'longitude' else 'subcatchment_id',
            'long_name': 'longitude' if subcatchment_dim != 'longitude' else 'subcatchment identifier'
        }
        fuse_forcing.latitude.attrs = {
            'units': 'degreesN' if subcatchment_dim != 'latitude' else 'subcatchment_id',
            'long_name': 'latitude' if subcatchment_dim != 'latitude' else 'subcatchment identifier'
        }
        fuse_forcing.time.attrs = {
            'units': 'days since 1970-01-01',
            'long_name': 'time'
        }

        # Add data variables
        self._add_forcing_variables(fuse_forcing, ds, pet, obs_ds, spatial_dims, len(subcatchments), ts_config)

        return fuse_forcing


    def create_filemanager(self):
        """
        Create FUSE file manager file by modifying template with project-specific settings.
        """
        self.logger.info("Creating FUSE file manager file")

        # Define source and destination paths — re-copy from resources if missing
        template_path = self.setup_dir / 'fm_catch.txt'
        if not template_path.exists():
            self.logger.warning(f"Template {template_path} missing, re-copying from resources")
            self.setup_dir.mkdir(parents=True, exist_ok=True)
            self.copy_base_settings()

        # Define the paths to replace
        fuse_id = self._get_fuse_file_id()
        settings = {
            'SETNGS_PATH': str(self.project_dir / 'settings' / 'FUSE') + '/',
            'INPUT_PATH': str(self.project_forcing_dir / 'FUSE_input') + '/',
            'OUTPUT_PATH': str(self.project_dir / 'simulations' / self.experiment_id / 'FUSE') + '/',
            'MAXN': str(self._get_config_value(
                lambda: self.config.optimization.number_of_iterations,
                default=100
            )),
            'FMODEL_ID': fuse_id,
            'M_DECISIONS': f"fuse_zDecisions_{fuse_id}.txt"
        }

        # Get and format dates from forcing data if available, else config
        start_time = datetime.strptime(self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01 00:00'
        ), '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self._get_config_value(
            lambda: self.config.domain.time_end,
            default='2010-12-31 00:00'
        ), '%Y-%m-%d %H:%M')
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"

        if hasattr(self, 'actual_start_time') and hasattr(self, 'actual_end_time'):
            self.logger.info(f"Using captured dates from preprocessing: {self.actual_start_time} to {self.actual_end_time}")
            start_time = self.actual_start_time
            end_time = self.actual_end_time
        else:
            self.logger.info(f"Checking forcing file for dates: {forcing_file}")
            if forcing_file.exists():
                try:
                    self.logger.info("Forcing file exists, attempting to read time range...")
                    with xr.open_dataset(forcing_file) as ds:
                        time_vals = pd.to_datetime(ds.time.values)

                    if len(time_vals) > 0:
                        start_time = time_vals.min().to_pydatetime()
                        end_time = time_vals.max().to_pydatetime()
                        self.logger.info(f"Updated simulation dates from forcing file: {start_time} to {end_time}")
                    else:
                        self.logger.warning("Forcing file has no time values!")
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Unable to read forcing time range from {forcing_file}: {e}", exc_info=True)
            else:
                self.logger.warning(f"Forcing file not found at {forcing_file}, using config dates.")

        cal_period = self._get_config_value(
            lambda: self.config.domain.calibration_period,
            default=None
        )
        if cal_period:
            cal_start_time = datetime.strptime(cal_period.split(',')[0], '%Y-%m-%d')
            cal_end_time = datetime.strptime(cal_period.split(',')[1].strip(), '%Y-%m-%d')
            cal_start_time = max(cal_start_time, start_time)
            cal_end_time = min(cal_end_time, end_time)
        else:
            cal_start_time = start_time + timedelta(days=1)
            cal_end_time = end_time
            self.logger.info(
                f"No calibration_period configured; defaulting eval period to "
                f"{cal_start_time.strftime('%Y-%m-%d')} – {cal_end_time.strftime('%Y-%m-%d')} "
                f"(sim window with 1-day spinup)"
            )

        date_settings = {
            'date_start_sim': start_time.strftime('%Y-%m-%d'),
            'date_end_sim': end_time.strftime('%Y-%m-%d'),
            'date_start_eval': cal_start_time.strftime('%Y-%m-%d'),
            'date_end_eval': cal_end_time.strftime('%Y-%m-%d')
        }

        try:
            # Read the template file with encoding fallback
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(
                    f"UTF-8 decode error reading {template_path}, falling back to latin-1"
                )
                with open(template_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Process each line
            modified_lines = []
            for line in lines:
                line_modified = line

                # Replace paths
                for path_key, new_path in settings.items():
                    if path_key in line:
                        # Find the content between quotes and replace it
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > 0:
                            line_modified = line[:start] + new_path + line[end:]
                            self.logger.debug(f"Updated {path_key} path to: {new_path}")

                # Replace dates
                for date_key, new_date in date_settings.items():
                    if date_key in line:
                        # Find the content between quotes and replace it
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > 0:
                            line_modified = line[:start] + new_date + line[end:]
                            self.logger.debug(f"Updated {date_key} to: {new_date}")

                modified_lines.append(line_modified)

            # Write the modified file
            with open(template_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)

            self.logger.info(f"FUSE file manager created at: {template_path}")


        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error creating FUSE file manager: {str(e)}")
            raise

    def update_input_info(self):
        """Update FUSE input_info.txt based on the configured timestep."""
        input_info_path = self.setup_dir / 'input_info.txt'
        if not input_info_path.exists():
            self.logger.warning(f"FUSE input_info.txt not found at {input_info_path}")
            return

        ts_config = self._get_timestep_config()
        if ts_config['time_unit'] == 'h':
            deltim = "0.0416667"
            unit_str = "mm/h"
        else:
            deltim = "1.0"
            unit_str = "mm/d"

        unit_keys = {
            '<units_aprecip>': unit_str,
            '<units_potevap>': unit_str,
            '<units_q>': unit_str,
        }

        try:
            with open(input_info_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(input_info_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        updated_lines = []
        for line in lines:
            line_updated = line
            if '<deltim>' in line:
                line_updated = re.sub(r"(<deltim>\\s+)(\\S+)", rf"\\1{deltim}", line)
            for key, value in unit_keys.items():
                if key in line:
                    line_updated = re.sub(rf"({re.escape(key)}\\s+)(\\S+)", rf"\\1{value}", line_updated)
            updated_lines.append(line_updated)

        with open(input_info_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

        self.logger.info("Updated FUSE input_info.txt for forcing timestep")



    # Removed: _get_catchment_centroid() - now inherited from GeospatialUtilsMixin

    def create_elevation_bands(self):
        """Create elevation bands netCDF file - delegates to elevation band manager"""
        return self.elevation_band_manager.create_elevation_bands()

    # NOTE: _create_distributed_elevation_bands() and _create_lumped_elevation_bands() have been
    # removed as they are now fully handled by FuseElevationBandManager. This eliminates ~120 lines
    # of duplicated code.

    def _load_streamflow_observations(
        self,
        spatial_mode: str,
        ts_config: Optional[Dict[str, Any]] = None,
        time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    ) -> Optional[xr.Dataset]:
        """
        Load streamflow observations for FUSE forcing data.

        Uses ObservationLoaderMixin for standardized observation loading.

        Args:
            spatial_mode: Spatial mode ('lumped', 'semi_distributed', 'distributed')
            ts_config: Timestep configuration from _get_timestep_config()
            time_window: Optional (start, end) timestamp tuple for filtering

        Returns:
            Dataset containing observed streamflow or None if not available
        """
        # Get timestep config if not provided
        if ts_config is None:
            ts_config = self._get_timestep_config()

        # Determine target units based on timestep
        if ts_config['time_unit'] == 'h':
            target_units = 'mm_per_hour'
        elif ts_config['time_unit'] == 'D':
            target_units = 'mm_per_day'
        else:
            target_units = 'mm_per_timestep'

        # Use ObservationLoaderMixin to load observations
        obs_ds = self.load_streamflow_observations(
            output_format='xarray',
            target_units=target_units,
            resample_freq=ts_config['resample_freq'],
            time_slice=time_window,
            return_none_on_error=True
        )

        return obs_ds  # type: ignore[return-value]

    def _calculate_distributed_pet(self, ds, spatial_mode, pet_method='oudin'):
        """Calculate PET for distributed/semi-distributed modes - delegates to forcing processor"""
        return self.forcing_processor._calculate_distributed_pet(ds, spatial_mode, pet_method)

    def _get_encoding_dict(self, fuse_forcing):
        """Get encoding dictionary for netCDF output - delegates to forcing processor"""
        return self.forcing_processor.get_encoding_dict(fuse_forcing)

    def _create_lumped_dataset(self, ds, pet, obs_ds, ts_config=None):
        """
        Create lumped FUSE forcing dataset with configurable timestep using xarray native alignment.
        """
        if ts_config is None:
            ts_config = self._get_timestep_config()

        # Get catchment centroid for coordinates
        catchment = gpd.read_file(self.catchment_path)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)

        # Align all datasets to a common time period using xarray.align
        # First, ensure all have compatible time coordinates
        to_align = [ds, pet]
        if obs_ds is not None:
            # Handle obs_ds which might have numeric time
            if obs_ds.time.dtype.kind in ['i', 'u', 'f']:
                obs_ds = obs_ds.assign_coords(time=pd.to_datetime('1970-01-01') + pd.to_timedelta(obs_ds.time.values, unit='D'))
            to_align.append(obs_ds)

        # Keep the forcing clock even when discharge observations end earlier.
        # Missing observations remain NaN and must not truncate a transfer run.
        aligned = xr.align(*to_align, join='left')
        ds_a, pet_a = aligned[0], aligned[1]
        obs_ds_a = aligned[2] if obs_ds is not None else None

        if len(ds_a.time) == 0:
            msg = "Overlap between forcing, PET, and observations is empty."
            for i, data in enumerate(to_align):
                if hasattr(data, 'time') and len(data.time) > 0:
                    msg += f" Dataset {i} range: {data.time.min().values} to {data.time.max().values}."
                else:
                    msg += f" Dataset {i} is empty or has no time."
            raise ModelExecutionError(msg)

        self.logger.info(f"Aligned data to overlapping period: {ds_a.time.min().values} to {ds_a.time.max().values}")

        # Create coordinates for FUSE
        time_numeric = ((pd.to_datetime(ds_a.time.values) - pd.Timestamp('1970-01-01')).total_seconds() / 86400).values
        # CRITICAL: Order must be time, lat, lon for FUSE to read correct dimensions
        coords = {
            'time': ('time', time_numeric),
            'latitude': ('latitude', [float(mean_lat)]),
            'longitude': ('longitude', [float(mean_lon)])
        }

        fuse_forcing = xr.Dataset(coords=coords)
        fuse_forcing.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        fuse_forcing.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'}
        fuse_forcing.time.attrs = {'units': 'days since 1970-01-01', 'long_name': 'time'}

        # Determine unit string for variables
        time_label = ts_config['time_label']
        unit_str = f"mm/{time_label.replace('-', ' ')}"

        # Core meteorological variables
        var_map = {
            'precip': (ds_a['precip'], 'precipitation', unit_str, f'Mean {time_label} precipitation'),
            'temp': (ds_a['temp'], 'temperature', 'degC', f'Mean {time_label} temperature'),
            'pet': (pet_a, 'pet', unit_str, f'Mean {time_label} pet')
        }

        if obs_ds_a is not None and 'q_obs' in obs_ds_a:
            var_map['q_obs'] = (obs_ds_a['q_obs'], 'streamflow', unit_str, f'Mean observed {time_label} discharge')
        else:
            self.logger.warning(
                "Streamflow observations not found — writing dummy q_obs (-9999) to FUSE forcing file. "
                "Observations will be needed for calibration."
            )
            dummy_q = xr.full_like(ds_a['precip'], -9999.0)
            dummy_q.name = 'q_obs'
            var_map['q_obs'] = (dummy_q, 'streamflow', unit_str, 'Dummy observed discharge (missing)')

        # Add variables with broadcasting
        for var_name, (da, _, units, long_name) in var_map.items():
            # Ensure da has same time coord type and values as fuse_forcing to avoid DTypePromotionError
            # Drop time first to ensure clean replacement of coordinate and index
            da_aligned = da.drop_vars('time').assign_coords(time=fuse_forcing.time)

            # Broadcast to (time, lat, lon) where lat=1, lon=1
            fuse_forcing[var_name] = da_aligned.broadcast_like(fuse_forcing).fillna(-9999.0).astype('float32')
            fuse_forcing[var_name].attrs = {'units': units, 'long_name': long_name}

        return fuse_forcing

    def _map_hrus_to_subcatchments(self, ds, subcatchments):
        """Map HRU data to subcatchments - delegates to forcing processor"""
        return self.forcing_processor._map_hrus_to_subcatchments(ds, subcatchments)

    def _replicate_to_subcatchments(self, ds, n_subcatchments):
        """Replicate lumped data to subcatchments - delegates to forcing processor"""
        return self.forcing_processor._replicate_to_subcatchments(ds, n_subcatchments)

    def _create_distributed_from_catchment(self, ds):
        """Create HRU-level data from catchment data - delegates to forcing processor"""
        return self.forcing_processor._create_distributed_from_catchment(ds)

    def _get_base_file_path(self, file_type, file_def_path, file_name):  # type: ignore[override]
        val = self._get_config_value(lambda: getattr(self.config.paths, file_type.lower(), None), default='default')
        if val == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(val)
