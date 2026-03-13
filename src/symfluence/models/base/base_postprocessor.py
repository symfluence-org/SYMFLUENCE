# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base class for model postprocessors.

Provides shared infrastructure for all model postprocessing modules including:
- Configuration management
- Path resolution with default fallbacks
- Results directory creation
- Common data extraction patterns
- Unit conversion utilities
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.core.path_resolver import PathResolverMixin
from symfluence.core.validation import validate_config_keys
from symfluence.models.mixins import ModelComponentMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelPostProcessor(ABC, ModelComponentMixin, PathResolverMixin):  # type: ignore[misc]
    """
    Abstract base class for all model postprocessors.

    Provides common initialization, path management, and utility methods
    that are shared across different hydrological model postprocessors.

    All model postprocessors should inherit from this class to ensure
    consistent behavior and reduce code duplication.

    Attributes:
        config: SymfluenceConfig instance
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'GR')
        sim_dir: Directory for model simulation outputs
        results_dir: Directory for processed results (CSV, plots)
        experiment_id: Current experiment identifier

    Example:
        >>> @ModelRegistry.register_postprocessor('MYMODEL')
        >>> class MyModelPostprocessor(BaseModelPostProcessor):
        ...     def _get_model_name(self) -> str:
        ...         return "MYMODEL"
        ...
        ...     def extract_streamflow(self) -> Optional[Path]:
        ...         # Model-specific extraction logic
        ...         pass
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize base model postprocessor.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
            reporting_manager: ReportingManager instance

        Raises:
            ConfigurationError: If required configuration keys are missing
        """
        # Common initialization via mixin
        self._init_model_component(config, logger, reporting_manager)

        # Postprocessor-specific: standard output directories
        # experiment_id is available via ConfigMixin property
        self.sim_dir = self._get_simulation_dir()
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Allow subclasses to perform custom setup
        self._setup_model_specific_paths()

    def _validate_required_config(self) -> None:
        """
        Validate that all required configuration keys are present.

        Subclasses can override to add model-specific required keys.
        """
        required_keys = [
            'SYMFLUENCE_DATA_DIR',
            'DOMAIN_NAME',
        ]
        validate_config_keys(
            self.config_dict,
            required_keys,
            f"{self._get_model_name() if hasattr(self, '_get_model_name') else 'Model'} postprocessor initialization"
        )

    @abstractmethod
    def _get_model_name(self) -> str:
        """
        Return the name of the model.

        Must be implemented by subclasses.

        Returns:
            Model name (e.g., 'SUMMA', 'FUSE', 'GR', 'HYPE')

        Example:
            >>> def _get_model_name(self) -> str:
            ...     return "SUMMA"
        """
        pass

    def _get_simulation_dir(self) -> Path:
        """
        Get simulation output directory for this model.

        Default implementation uses standard path structure:
        project_dir/simulations/{experiment_id}/{model_name}

        Subclasses can override for custom behavior (e.g., HYPE).

        Returns:
            Path to simulation directory

        Example:
            >>> # Default behavior:
            >>> self.sim_dir
            Path('data/domain_mybasin/simulations/exp001/SUMMA')

            >>> # Override for custom paths:
            >>> def _get_simulation_dir(self) -> Path:
            ...     return self.project_dir / "custom" / "path"
        """
        return (self.project_dir / 'simulations' /
                self.experiment_id / self.model_name)

    def _setup_model_specific_paths(self) -> None:
        """
        Hook for subclasses to set up model-specific paths.

        Called after base paths are initialized but before any operations.
        Override this method to add model-specific path attributes.

        Example:
            >>> def _setup_model_specific_paths(self):
            ...     self.routing_dir = self.sim_dir / 'routing'
            ...     self.spatial_mode = self.config_dict.get('GR_SPATIAL_MODE', 'lumped')
        """
        pass

    @abstractmethod
    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from model output.

        Must be implemented by subclasses to handle model-specific
        output formats and extraction logic.

        Returns:
            Path to extracted streamflow CSV file, or None if extraction fails

        Raises:
            ModelExecutionError: If extraction fails critically

        Example:
            >>> def extract_streamflow(self) -> Optional[Path]:
            ...     sim_file = self.sim_dir / "output.nc"
            ...     q_sim = self.read_netcdf_streamflow(sim_file, 'discharge')
            ...     q_cms = self.convert_mm_per_day_to_cms(q_sim)
            ...     return self.save_streamflow_to_results(q_cms)
        """
        pass

    # -------------------------------------------------------------------------
    # Common Helper Methods
    # -------------------------------------------------------------------------

    def get_catchment_area_km2(self) -> float:
        """
        Get total catchment area in km².

        Reads from river basins shapefile GRU_area field.

        Returns:
            Total catchment area in km²

        Raises:
            FileNotFoundError: If river basins shapefile not found
            ValueError: If GRU_area field not found or calculation fails

        Example:
            >>> area = self.get_catchment_area_km2()
            >>> print(f"Catchment area: {area:.2f} km²")
        """
        import geopandas as gpd

        basin_name = self._get_config_value(lambda: self.config.paths.river_basins_name)
        if basin_name == 'default' or basin_name is None:
            basin_name = f"{self.domain_name}_riverBasins_{self.domain_definition_method}.shp"

        basin_path = self._get_file_path(
            path_key='RIVER_BASINS_PATH',
            name_key='RIVER_BASINS_NAME',
            default_subpath='shapefiles/river_basins',
            default_name=basin_name
        )

        if not basin_path.exists():
            raise FileNotFoundError(f"River basins shapefile not found: {basin_path}")

        basin_gdf = gpd.read_file(basin_path)

        if 'GRU_area' not in basin_gdf.columns:
            raise ValueError(f"GRU_area field not found in {basin_path}")

        # GRU_area is in m², convert to km²
        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area: {area_km2:.2f} km²")

        return area_km2

    def convert_mm_per_day_to_cms(
        self,
        series: pd.Series,
        catchment_area_km2: Optional[float] = None
    ) -> pd.Series:
        """
        Convert streamflow from mm/day to cubic meters per second (cms).

        Uses the standard conversion: Q(cms) = Q(mm/day) * Area(km²) / 86.4

        Args:
            series: Streamflow in mm/day
            catchment_area_km2: Catchment area (auto-fetched if None)

        Returns:
            Streamflow in cms

        Example:
            >>> q_mm_day = pd.Series([10, 15, 20])
            >>> q_cms = self.convert_mm_per_day_to_cms(q_mm_day)
        """
        if catchment_area_km2 is None:
            catchment_area_km2 = self.get_catchment_area_km2()

        return series * catchment_area_km2 / UnitConversion.MM_DAY_TO_CMS

    def convert_cms_to_mm_per_day(
        self,
        series: pd.Series,
        catchment_area_km2: Optional[float] = None
    ) -> pd.Series:
        """
        Convert streamflow from cms to mm/day.

        Uses the standard conversion: Q(mm/day) = Q(cms) * 86.4 / Area(km²)

        Args:
            series: Streamflow in cms
            catchment_area_km2: Catchment area (auto-fetched if None)

        Returns:
            Streamflow in mm/day

        Example:
            >>> q_cms = pd.Series([100, 150, 200])
            >>> q_mm_day = self.convert_cms_to_mm_per_day(q_cms)
        """
        if catchment_area_km2 is None:
            catchment_area_km2 = self.get_catchment_area_km2()

        return series * UnitConversion.MM_DAY_TO_CMS / catchment_area_km2

    def save_streamflow_to_results(
        self,
        streamflow: pd.Series,
        model_column_name: Optional[str] = None,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save streamflow to results CSV, appending to existing file if present.

        Standard pattern used across all postprocessors: read existing results,
        add/update model column, save back to CSV.

        Args:
            streamflow: Time series of streamflow (index = datetime)
            model_column_name: Column name (default: {MODEL}_discharge_cms)
            output_file: Output file path (default: {experiment_id}_results.csv)

        Returns:
            Path to saved CSV file

        Example:
            >>> q_sim = pd.Series([100, 150, 200], index=pd.date_range('2020-01-01', periods=3))
            >>> output = self.save_streamflow_to_results(q_sim)
            >>> print(f"Saved to: {output}")
        """
        if model_column_name is None:
            model_column_name = f"{self.model_name}_discharge_cms"

        if output_file is None:
            output_file = self.results_dir / f"{self.experiment_id}_results.csv"

        # Read existing results if present
        if output_file.exists():
            results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        else:
            results_df = pd.DataFrame(index=streamflow.index)

        # Add/update model column
        results_df[model_column_name] = streamflow

        # Save with proper formatting
        results_df.to_csv(output_file)
        self.logger.info(f"Streamflow saved to: {output_file}")

        # Also write to the model-output data store as CF-compliant NetCDF
        self._save_to_model_output_store(results_df)

        # Automatically visualize results if reporting manager is available
        self.visualize_streamflow_results()

        return output_file

    def visualize_streamflow_results(self) -> None:
        """
        Create standardized streamflow plots using ReportingManager.

        This method is called automatically after saving results.
        It delegates to reporting_manager.visualize_timeseries_results(),
        which handles:
        - Loading the consolidated results CSV
        - Loading observations
        - Aligning time series
        - Calculating metrics
        - Generating comparison and diagnostic plots
        """
        if self.reporting_manager and self.reporting_manager.is_visualization_enabled():
            try:
                self.logger.info(f"Creating standardized streamflow plots for {self.model_name}...")
                self.reporting_manager.visualize_timeseries_results()
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.error(f"Error creating streamflow plots: {str(e)}")
        else:
            self.logger.debug("Skipping visualization (manager not available or visualization disabled)")

    def _save_to_model_output_store(self, results_df: pd.DataFrame) -> None:
        """Write postprocessed results to the model-output data store as CF-compliant NetCDF.

        Creates ``data/model_output/{domain}_{experiment}_results.nc`` alongside
        the model-ready *input* store, embedding provenance metadata (model name,
        experiment, framework version, creation timestamp) directly in the file.
        """
        try:
            from symfluence.data.model_ready.cf_conventions import (
                CF_STANDARD_NAMES,
                build_global_attrs,
            )

            output_store = self.project_dir / "data" / "model_output"
            output_store.mkdir(parents=True, exist_ok=True)
            nc_path = output_store / f"{self.experiment_id}_results.nc"

            # Build xarray Dataset from the results DataFrame
            ds = xr.Dataset()

            for col in results_df.columns:
                da = xr.DataArray(
                    results_df[col].values,
                    dims=["time"],
                    coords={"time": results_df.index.values},
                )
                # Attach CF attributes when available
                base_var = col.replace(f"{self.model_name}_", "").replace("_cms", "")
                if "discharge" in base_var:
                    cf_key = "discharge_cms"
                else:
                    cf_key = base_var
                if cf_key in CF_STANDARD_NAMES:
                    da.attrs.update(CF_STANDARD_NAMES[cf_key])
                da.attrs["column_name"] = col
                ds[col] = da

            # Global attributes with provenance
            ds.attrs.update(build_global_attrs(
                domain_name=self.domain_name,
                title=f"SYMFLUENCE model output — {self.experiment_id}",
                history=f"Postprocessed by {self.model_name} postprocessor",
            ))
            ds.attrs["model_name"] = self.model_name
            ds.attrs["experiment_id"] = self.experiment_id

            ds.to_netcdf(nc_path)
            self.logger.info(f"Model output NetCDF saved to: {nc_path}")

        except Exception as e:  # noqa: BLE001 — non-critical; CSV is the primary output
            self.logger.debug(f"Could not write model-output NetCDF: {e}")

    def read_netcdf_streamflow(
        self,
        file_path: Path,
        variable_name: str,
        **selection_kwargs
    ) -> pd.Series:
        """
        Read streamflow from NetCDF file with flexible selection.

        Common pattern for FUSE, SUMMA, FLASH models that output NetCDF.

        Args:
            file_path: Path to NetCDF file
            variable_name: Name of streamflow variable
            **selection_kwargs: Keyword arguments for .sel() or .isel()

        Returns:
            Streamflow as pandas Series

        Raises:
            FileNotFoundError: If NetCDF file not found
            ValueError: If variable not found in file

        Example:
            >>> # Select by integer index
            >>> q = self.read_netcdf_streamflow(
            ...     path, 'q_routed',
            ...     param_set=0, latitude=0, longitude=0
            ... )
            >>> # Select by label
            >>> q = self.read_netcdf_streamflow(
            ...     path, 'discharge',
            ...     reach_id=123456
            ... )
        """
        if not file_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {file_path}")

        ds = xr.open_dataset(file_path)

        if variable_name not in ds:
            raise ValueError(f"Variable '{variable_name}' not found in {file_path}")

        # Extract data array
        data = ds[variable_name]

        # Apply selections if provided
        if selection_kwargs:
            # Try isel first (integer indexing)
            try:
                data = data.isel(**selection_kwargs)
            except (TypeError, KeyError):
                # Fall back to sel (label-based indexing)
                data = data.sel(**selection_kwargs)

        # Convert to pandas Series
        series = cast(pd.Series, data.to_pandas())

        ds.close()
        return series
