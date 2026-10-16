# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR model preprocessor.

Handles data preparation, PET calculation, snow module setup, and file organization.
Supports both lumped and distributed spatial modes.
Uses shared utilities for forcing data processing and data quality handling.
"""

# Optional R/rpy2 support - only needed for GR models
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.core.geometry_utils import GeospatialUtilsMixin
from symfluence.core.modeling.base import BaseModelPreProcessor
from symfluence.core.modeling.mixins import DatasetBuilderMixin, PETCalculatorMixin, SpatialModeDetectionMixin
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.modeling.utilities import ForcingDataProcessor
from symfluence.core.modeling.variable_utils import VariableHandler
from symfluence.core.registries import R

from .r_environment import configure_r_dll_search, verify_gr_r_runtime

# Must run before anything imports rpy2 — see r_environment for why.
configure_r_dll_search()

HAS_RPY2 = find_spec("rpy2") is not None


@R.preprocessors.add('GR')
class GRPreProcessor(BaseModelPreProcessor, PETCalculatorMixin, GeospatialUtilsMixin, DatasetBuilderMixin, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Preprocessor for the GR family of models (GR4J, GR5J, GR6J).

    Handles complete preprocessing workflow for GR models including forcing data processing,
    PET calculation, optional snow module setup, and preparation of input data for R-based
    execution. GR models are parsimonious lumped/semi-distributed rainfall-runoff models
    developed by INRAE (France).

    Key Operations:
        - Process forcing data (precipitation, temperature, PET)
        - Calculate potential evapotranspiration using multiple methods
        - Prepare input data for GR model execution via rpy2/R interface
        - Handle both lumped and distributed (HRU-based) spatial configurations
        - Configure optional snow module (CemaNeige) for snowmelt modeling
        - Quality control and gap-filling of forcing data
        - Load and align observation data for calibration/evaluation

    Workflow Steps:
        1. Initialize paths and validate R/rpy2 installation
        2. Determine spatial mode (lumped vs. distributed)
        3. Process forcing data and calculate PET
        4. Aggregate forcing data by spatial units (if distributed)
        5. Handle data quality issues (gaps, outliers)
        6. Prepare input matrices for R model execution
        7. Configure snow module parameters (if enabled)
        8. Save processed data in R-compatible format

    Supported Spatial Modes:
        - Lumped: Single catchment-averaged inputs
        - Distributed: Multiple HRUs with area-weighted aggregation
        - Auto: Automatically detect based on domain definition method

    GR Model Variants:
        - GR4J: 4-parameter daily model (basic rainfall-runoff)
        - GR5J: 5-parameter daily model (includes groundwater exchange)
        - GR6J: 6-parameter daily model (includes parallel routing)
        - CemaNeige: Optional 2-parameter snow module

    PET Calculation Methods:
        - Oudin: Simple temperature-based method (GR model default)
        - Hamon: Temperature and daylight-based
        - Priestley-Taylor: Radiation-based method

    Inherits from:
        BaseModelPreProcessor: Common preprocessing patterns and utilities
        PETCalculatorMixin: Potential evapotranspiration calculation methods
        GeospatialUtilsMixin: Spatial operations (area calculation, centroid)
        ObservationLoaderMixin: Observation data loading capabilities
        DatasetBuilderMixin: NetCDF dataset construction utilities

    Attributes:
        config (SymfluenceConfig): Typed configuration object
        logger: Logger object for recording processing information
        project_dir (Path): Directory for the current project
        forcing_gr_path (Path): Directory for GR input files
        catchment_path (Path): Path to catchment shapefile
        spatial_mode (str): Spatial configuration ('lumped' or 'distributed')
        forcing_processor (ForcingDataProcessor): Handles forcing data transformation
        quality_handler (DataQualityHandler): Handles data quality control

    Requirements:
        - R programming language (>= 4.0)
        - rpy2 Python package (R-Python interface)
        - airGR R package (GR model implementations)

    Example:
        >>> from symfluence.models.gr.preprocessor import GRPreProcessor
        >>> preprocessor = GRPreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates GR input files in: project_dir/forcing/GR_input/
        # Generates: forcing_data.csv, catchment_attributes.csv

    Raises:
        ImportError: If R or rpy2 is not installed
        ModelExecutionError: If preprocessing fails
    """

    MODEL_NAME = "GR"

    def __init__(self, config, logger):
        if not HAS_RPY2:
            raise ImportError(
                "GR models require R and rpy2. "
                "Please install R and rpy2, or use a different model. "
                "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
            )

        # Initialize base class (handles typed config validation)
        super().__init__(config, logger)

        # GR-specific paths
        self.forcing_gr_path = self.project_forcing_dir / 'GR_input'

        # GR-specific catchment configuration (use backward-compatible path resolution)
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

        # Resolve spatial mode using mixin
        self.spatial_mode = self.detect_spatial_mode('GR')

    def run_preprocessing(self):
        """
        Run the complete GR preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ImportError: If rpy2 or the embedded R cannot be started.
            ModelExecutionError: If the embedded R cannot load airGR.
        """
        # GR preprocessing can take the better part of an hour, and the model
        # it feeds is useless without a working R. Check the R stack up front
        # rather than discovering it after all that work has been thrown away.
        verify_gr_r_runtime()

        self.logger.info(f"Starting GR preprocessing in {self.spatial_mode} mode")
        # GR does not ship base settings; avoid noisy warning.
        self.copy_base_settings = lambda *args, **kwargs: None
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """GR-specific forcing data preparation (template hook)."""
        self.prepare_forcing_data()

    def prepare_forcing_data(self):
        """
        Prepare forcing data with support for lumped and distributed modes.

        Uses shared ForcingDataProcessor for loading and subsetting.
        """
        try:
            # Use shared ForcingDataProcessor for loading
            fdp = ForcingDataProcessor(self.config, self.logger)
            ds = fdp.load_forcing_data(self.forcing_basin_path)

            # Subset to simulation window using base class method
            ds = self.subset_to_simulation_time(ds, "Forcing")

            # Basin-averaged forcing data is already in CFIF format (from model-agnostic preprocessing)
            variable_handler = VariableHandler(
                config=self.config_dict,
                logger=self.logger,
                dataset='CFIF',
                model='GR'
            )

            # Process variables
            ds_variable_handler = variable_handler.process_forcing_data(ds)
            ds = ds_variable_handler

            # Handle spatial organization based on mode
            if self.spatial_mode == SpatialMode.LUMPED:

                self.logger.info("Preparing lumped forcing data")
                # Apply area-weighted aggregation if HRU dimension exists
                if 'hru' in ds.dims:
                    ds = self._apply_area_weighted_aggregation(ds)
                return self._prepare_lumped_forcing(ds)
            elif self.spatial_mode == SpatialMode.DISTRIBUTED:
                self.logger.info("Preparing distributed forcing data")
                return self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown GR spatial mode: {self.spatial_mode}")

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise

    def _apply_area_weighted_aggregation(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply area-weighted spatial aggregation for lumped mode.

        For spatially heterogeneous catchments, area-weighted averaging provides
        more physically accurate basin-average forcing than simple mean.

        Args:
            ds: Input forcing dataset with 'hru' dimension

        Returns:
            Aggregated forcing dataset (no 'hru' dimension)
        """
        try:
            # Load catchment to get HRU areas
            catchment = gpd.read_file(self.get_catchment_path())
            n_hrus = len(catchment)

            # Get area for each HRU - check various column naming conventions
            area_col = None
            for col in ['area', 'area_km2', 'HRU_area', 'GRU_area', 'Area', 'AREA']:
                if col in catchment.columns:
                    area_col = col
                    break

            if area_col:
                areas = catchment[area_col].values
                # Convert m² to km² if values are large (> 1e6 suggests m²)
                if areas.mean() > 1e6:
                    areas = areas / 1e6
                    self.logger.debug(f"Using '{area_col}' column for area weights (converted m² to km²)")
                else:
                    self.logger.debug(f"Using '{area_col}' column for area weights")
            else:
                # Calculate from geometry - need to project to equal-area CRS first
                catchment_projected = catchment.to_crs('EPSG:6933')  # Equal-area projection
                areas = catchment_projected.geometry.area.values / 1e6  # m² to km²
                self.logger.debug("Calculated areas from projected geometry (m² to km²)")

            # Verify dimensions match
            if len(areas) != ds.sizes.get('hru', 0):
                self.logger.warning(
                    f"Area count ({len(areas)}) doesn't match HRU count ({ds.sizes.get('hru', 0)}). "
                    f"Falling back to simple mean."
                )
                return ds.mean(dim='hru')

            # Normalize to get weights
            total_area = areas.sum()
            if total_area <= 0:
                self.logger.warning(f"Total area is {total_area}. Falling back to simple mean.")
                return ds.mean(dim='hru')

            weights = areas / total_area
            weight_da = xr.DataArray(weights, dims=['hru'])

            # Apply area-weighted averaging
            ds_lumped = (ds * weight_da).sum(dim='hru')

            self.logger.info(
                f"Applied area-weighted aggregation "
                f"({n_hrus} HRUs, total area: {total_area:.2f} km²)"
            )

            return ds_lumped

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(
                f"Failed to apply area-weighted aggregation: {e}. "
                f"Falling back to simple mean."
            , exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return ds.mean(dim='hru')

    def _prepare_lumped_forcing(self, ds: xr.Dataset) -> Path:
        """Prepare lumped forcing data using shared utilities."""
        # Check if we have enough dates for inference
        if len(ds.time) < 3:
            self.logger.warning("Fewer than 3 time steps in forcing. Cannot infer frequency for GR.")
            # Skip resampling
        else:
            # Use shared ForcingDataProcessor for resampling
            fdp = ForcingDataProcessor(self.config, self.logger)
            ds = fdp.resample_to_frequency(ds, target_freq='D', method='mean')

        # Unit conversions already handled by VariableHandler.process_forcing_data()

        # Load streamflow observations
        obs_path = self.project_observations_dir / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"

        # Read observations
        if obs_path.exists():
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_daily = obs_df.resample('D').mean()
        else:
            # Check if fallback to dummy observations is allowed
            allow_dummy = self._get_config_value(
                lambda: self.config.model.gr.allow_dummy_observations, False
            )
            if allow_dummy:
                self.logger.warning(
                    f"No streamflow observations at {obs_path}, using dummy zeros "
                    "(GR_ALLOW_DUMMY_OBSERVATIONS=True)"
                )
                obs_daily = pd.DataFrame(
                    {'discharge_cms': [0.0] * len(ds.time)},
                    index=pd.to_datetime(ds.time.values)
                )
            else:
                raise FileNotFoundError(
                    f"No streamflow observations found at {obs_path}. "
                    f"Set GR_ALLOW_DUMMY_OBSERVATIONS=True to use zero-filled dummy data."
                )

        # Get area from river basins shapefile
        basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        basin_name = self._get_config_value(lambda: self.config.paths.river_basins_name, default=None)
        if basin_name == 'default' or basin_name is None:
            method_suffix = self._get_method_suffix()
            basin_name = f"{self.domain_name}_riverBasins_{method_suffix}.shp"
        basin_path = basin_dir / basin_name

        if basin_path.exists():
            basin_gdf = gpd.read_file(basin_path)
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        else:
            # Check if fallback to default area is allowed
            allow_default = self._get_config_value(
                lambda: self.config.model.gr.allow_default_area, False
            )
            if allow_default:
                self.logger.warning(
                    f"Basin shapefile not found at {basin_path}, using area=1.0 km² "
                    "(GR_ALLOW_DEFAULT_AREA=True)"
                )
                area_km2 = 1.0
            else:
                raise FileNotFoundError(
                    f"Basin shapefile not found at {basin_path}. "
                    f"Set GR_ALLOW_DEFAULT_AREA=True to use 1.0 km² default."
                )

        self.logger.info(f"Total catchment area from GRU_area: {area_km2:.2f} km2")

        # Convert units from cms to mm/day for GR input
        obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * UnitConversion.MM_DAY_TO_CMS

        # Create observation dataset
        obs_ds = xr.Dataset(
            {'q_obs': ('time', obs_daily['discharge_mmday'].values)},
            coords={'time': obs_daily.index.values}
        )

        # Read catchment and get centroid (using inherited GeospatialUtilsMixin)
        catchment_path = self.get_catchment_path()
        if catchment_path.exists():
            catchment = gpd.read_file(catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        else:
            self.logger.warning(f"Catchment shapefile not found at {catchment_path}, using default latitude")
            mean_lat = 45.0

        # Calculate PET using GR variable name 'T' (mapped by VariableHandler)
        pet = self.calculate_pet_oudin(ds['T'], mean_lat)

        # Observations may end before forcing (e.g. a forecast/transfer run).
        # Preserve the simulation clock and leave unavailable observations NaN.
        start_time = ds.time.min().values
        end_time = ds.time.max().values

        # Create explicit time index
        time_index = pd.date_range(start=start_time, end=end_time, freq='D')

        # Select and align data
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)

        # Create GR forcing data using GR variable names (P, T after VariableHandler processing)
        gr_forcing = pd.DataFrame({
            'time': time_index.strftime('%Y-%m-%d'),
            'pr': ds['P'].values,  # GR uses 'P' for precipitation
            'temp': ds['T'].values,  # GR uses 'T' for temperature
            'pet': pet.values,
            'q_obs': obs_ds['q_obs'].values
        })

        # Save to CSV
        output_file = self.forcing_gr_path / f"{self.domain_name}_input.csv"
        gr_forcing.to_csv(output_file, index=False)

        self.logger.info(f"Lumped forcing data saved to: {output_file}")
        return output_file

    def _prepare_distributed_forcing(self, ds: xr.Dataset) -> Path:
        """Prepare distributed forcing data for each HRU using shared utilities."""

        # Load catchment to get HRU information
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)

        # Check if we have HRU dimension in forcing data
        if 'hru' not in ds.dims:
            self.logger.warning("No HRU dimension found in forcing data, creating distributed data from lumped")
            # Replicate lumped data to all HRUs
            n_hrus = len(catchment)
            ds = ds.expand_dims(hru=n_hrus)

        # Use shared ForcingDataProcessor for resampling
        fdp = ForcingDataProcessor(self.config, self.logger)
        ds = fdp.resample_to_frequency(ds, target_freq='D', method='mean')

        # Unit conversions already handled by VariableHandler.process_forcing_data()

        # Load streamflow observations (at outlet)
        obs_path = self.project_observations_dir / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"

        if obs_path.exists():
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_daily = obs_df.resample('D').mean()

            # Get area for unit conversion
            basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
            basin_name = self._get_config_value(lambda: self.config.paths.river_basins_name, default=None)
            if basin_name == 'default' or basin_name is None:
                method_suffix = self._get_method_suffix()
                basin_name = f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            basin_path = basin_dir / basin_name
            basin_gdf = gpd.read_file(basin_path)

            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * UnitConversion.MM_DAY_TO_CMS
        else:
            self.logger.warning("No streamflow observations found")
            obs_daily = None

        # Calculate PET for each HRU using its centroid latitude
        self.logger.info("Calculating PET for each HRU")

        # Ensure catchment has proper CRS
        if catchment.crs is None:
            catchment.set_crs(epsg=4326, inplace=True)

        # Get centroids for each HRU avoiding geographic CRS warning
        if catchment.crs.is_geographic:
            hru_centroids = catchment.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
        else:
            hru_centroids = catchment.geometry.centroid.to_crs(epsg=4326)

        hru_lats = hru_centroids.y.values

        # Calculate PET for each HRU using GR variable name 'T'
        pet_data = []
        for i, lat in enumerate(hru_lats):
            temp_hru = ds['T'].isel(hru=i)
            pet_hru = self.calculate_pet_oudin(temp_hru, lat)
            pet_data.append(pet_hru.values)

        # Stack PET data
        pet_array = np.stack(pet_data, axis=1)  # shape: (time, hru)
        pet = xr.DataArray(
            pet_array,
            dims=['time', 'hru'],
            coords={'time': ds.time, 'hru': ds.hru},
            attrs={
                'units': 'mm/day',
                'long_name': 'Potential evapotranspiration (Oudin formula)',
                'standard_name': 'water_potential_evaporation_flux'
            }
        )

        # Find overlapping time period
        start_time = ds.time.min().values
        end_time = ds.time.max().values

        if obs_daily is not None:
            start_time = max(start_time, obs_daily.index.min())
            end_time = min(end_time, obs_daily.index.max())

        time_index = pd.date_range(start=start_time, end=end_time, freq='D')

        # Select and align data
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)

        if obs_daily is not None:
            obs_daily = obs_daily.reindex(time_index)

        # Save distributed forcing as NetCDF (one file with all HRUs)
        output_file = self.forcing_gr_path / f"{self.domain_name}_input_distributed.nc"

        # Create output dataset using GR variable names (P, T after VariableHandler processing)
        gr_forcing = xr.Dataset({
            'pr': ds['P'],  # GR uses 'P' for precipitation
            'temp': ds['T'],  # GR uses 'T' for temperature
            'pet': pet
        })

        if obs_daily is not None:
            gr_forcing['q_obs'] = xr.DataArray(
                obs_daily['discharge_mmday'].values,
                dims=['time'],
                coords={'time': time_index}
            )

        # Add HRU metadata
        gr_forcing['hru_id'] = xr.DataArray(
            catchment['GRU_ID'].values if 'GRU_ID' in catchment.columns else np.arange(len(catchment)),
            dims=['hru'],
            attrs={'long_name': 'HRU identifier'}
        )

        gr_forcing['hru_lat'] = xr.DataArray(
            hru_lats,
            dims=['hru'],
            attrs={'long_name': 'HRU centroid latitude', 'units': 'degrees_north'}
        )

        # Save to NetCDF
        encoding = {var: {'zlib': True, 'complevel': 4} for var in gr_forcing.data_vars}
        gr_forcing.to_netcdf(output_file, encoding=encoding)

        self.logger.info(f"Distributed forcing data saved to: {output_file}")
        self.logger.info(f"Number of HRUs: {len(ds.hru)}")

        return output_file
