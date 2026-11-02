# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base class for model preprocessors.

Provides shared infrastructure for all model preprocessing modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation
- Common directory structure
- Settings file copying
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr

from symfluence.core.constants import ModelDefaults, UnitConversion
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.mixins import ShapefileAccessMixin
from symfluence.core.modeling.mixins import ModelComponentMixin, ObservationLoaderMixin
from symfluence.core.modeling.utilities.periods import to_pydatetime
from symfluence.core.path_resolver import PathResolverMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelPreProcessor(ABC, ModelComponentMixin, PathResolverMixin, ShapefileAccessMixin, ObservationLoaderMixin):  # type: ignore[misc]
    """
    Abstract base class for all model preprocessors.

    Provides common initialization, path management, and utility methods
    that are shared across different hydrological model preprocessors.

    Subclasses should set ``MODEL_NAME`` as a class variable. The legacy
    ``_get_model_name()`` method is still supported as a fallback.

    Inherits:
        PathResolverMixin: Path resolution with typed config access
        ShapefileAccessMixin: Shapefile column name properties

    Attributes:
        config: SymfluenceConfig instance
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'GR')
        setup_dir: Directory for model setup files
        forcing_dir: Directory for model-specific forcing inputs
        forcing_basin_path: Directory for basin-averaged forcing data
    """

    MODEL_NAME: str = ""

    def __init__(self, config: Union['SymfluenceConfig', Dict[str, Any]], logger: logging.Logger):
        """
        Initialize base model preprocessor.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance

        Raises:
            ConfigurationError: If required configuration keys are missing
        """
        # Common initialization via mixin (no reporting_manager for preprocessor)
        self._init_model_component(config, logger)

        # Preprocessor-specific paths
        self.setup_dir: Path = self.project_dir / "settings" / self.model_name
        self.forcing_dir: Path = self.project_forcing_dir / f"{self.model_name}_input"

        # Common forcing paths — prefer model-ready store, fall back to legacy
        _model_ready_forcing = self.project_dir / 'data' / 'model_ready' / 'forcings'
        if _model_ready_forcing.exists() and any(_model_ready_forcing.glob('*.nc')):
            self.forcing_basin_path = _model_ready_forcing
        else:
            self.forcing_basin_path = self.project_forcing_dir / 'basin_averaged_data'
        self.forcing_raw_path = self._get_default_path('FORCING_RAW_PATH', 'forcing/raw_data')
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_data')

        # Common shapefile paths
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        self.intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'

        # Common configuration - direct typed access
        # Properties forcing_dataset and forcing_time_step_size are available via ConfigMixin

    def _validate_required_config(self) -> None:
        """
        Validate that all required configuration keys are present.

        Subclasses can override to add model-specific required keys.

        Raises:
            ConfigurationError: If required keys are missing
        """
        required_keys = [
            'SYMFLUENCE_DATA_DIR',
            'DOMAIN_NAME',
            'FORCING_DATASET',
        ]
        self.validate_config(required_keys, f"{self.MODEL_NAME or self.__class__.__name__} preprocessing")

    def _get_base_file_path(self, file_type: str, path_key: str,
                       name_key: str, default_name: str) -> Path:
        """
        Resolve complete file path from config or defaults.

        Args:
            file_type: Description of file type (for logging)
            path_key: Config key for directory path
            name_key: Config key for file name
            default_name: Default file name if config value is 'default'

        Returns:
            Complete file path
        """
        # Get directory path
        dir_path = self._get_config_value(lambda: None, dict_key=path_key)
        if dir_path == 'default' or dir_path is None:
            self.logger.warning(f"No {file_type} path specified, path resolution may fail")
            dir_path = self.project_dir
        else:
            dir_path = Path(dir_path)

        # Get file name
        file_name = self._get_config_value(lambda: None, dict_key=name_key)
        if file_name == 'default' or file_name is None:
            file_name = default_name

        return dir_path / file_name

    def create_directories(self, additional_dirs: Optional[List[Path]] = None):
        """
        Create necessary directories for model setup.

        Creates standard directories (setup_dir, forcing_dir) plus any
        additional directories specified by the model.

        Args:
            additional_dirs: Optional list of additional directories to create

        Raises:
            FileOperationError: If directory creation fails
        """
        # Standard directories
        dirs_to_create = [
            self.setup_dir,
            self.forcing_dir
        ]

        # Add model-specific directories
        if additional_dirs:
            dirs_to_create.extend(additional_dirs)

        # Create all directories using FileUtilsMixin's ensure_dir
        for dir_path in dirs_to_create:
            self.ensure_dir(dir_path)

    def copy_base_settings(self, source_dir: Optional[Path] = None,
                          file_patterns: Optional[List[str]] = None):
        """
        Copy base settings files from source to setup directory.

        Args:
            source_dir: Source directory containing base settings.
                       If None, uses default location based on model name.
            file_patterns: List of file patterns to copy (e.g., ['*.txt', '*.nc']).
                          If None, copies all files.

        Raises:
            FileOperationError: If settings cannot be copied
        """
        if source_dir is None:
            # Try to find base settings in config, then fall back to SYMFLUENCE_CODE_DIR
            try:
                base_settings_key = f'SETTINGS_{self.model_name}_BASE_PATH'
                source_dir = Path(self._get_config_value(lambda: None, default='default', dict_key=base_settings_key))
                if source_dir == Path('default') or not source_dir.exists():
                    fallback_dir = self.get_base_settings_source_dir()
                    if fallback_dir.exists():
                        source_dir = fallback_dir
                    else:
                        self.logger.warning(
                            f"Base settings directory not found for {self.model_name}. "
                            f"Skipping settings copy."
                        )
                        return
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Could not locate base settings: {e}", exc_info=True)
                return

        if source_dir is not None and not source_dir.exists():
            self.logger.warning(
                f"Base settings source directory does not exist: {source_dir}. "
                f"Skipping settings copy."
            )
            return

        self.logger.info(f"Copying base settings from {source_dir} to {self.setup_dir}")

        # Use FileUtilsMixin's copy methods
        if file_patterns is None:
            self.copy_tree(source_dir, self.setup_dir)
        else:
            self.ensure_dir(self.setup_dir)
            for pattern in file_patterns:
                for file_path in source_dir.glob(pattern):
                    if file_path.is_file():
                        self.copy_file(file_path, self.setup_dir / file_path.name)

    def get_catchment_path(self) -> Path:
        """
        Get path to catchment shapefile with backward compatibility.

        Checks for existing file in the legacy location first, then returns
        the new organized path if not found.

        The new path structure:
            shapefiles/catchment/{domain_definition_method}/{experiment_id}/

        Returns:
            Path to catchment shapefile
        """
        return self._get_catchment_file_path()

    def get_catchment_location(self) -> Dict[str, Any]:
        """Return the catchment centroid, area and mean elevation.

        Reads the catchment shapefile and returns ``{'lat', 'lon', 'area_m2',
        'elev'}`` where lat/lon are the whole-catchment centroid in decimal
        degrees, ``area_m2`` is the projected (UTM) area, and ``elev`` is the
        area-weighted mean of the ``elev_mean`` attribute (``None`` if the
        shapefile has no such column).

        Raises rather than falling back to a default location: a silent
        fallback builds a model for the wrong place and still "succeeds".

        Raises:
            FileNotFoundError: If the catchment shapefile does not exist.
            ValueError: If the shapefile contains no features.
        """
        import geopandas as gpd

        from symfluence.core.geometry_utils import (
            calculate_catchment_area_km2,
            calculate_catchment_centroid,
        )

        catchment_path = self.get_catchment_path()
        if not catchment_path.exists():
            raise FileNotFoundError(
                f"Catchment shapefile not found: {catchment_path}. Run domain "
                "definition/discretisation before preprocessing this model."
            )
        gdf = gpd.read_file(catchment_path)
        if gdf.empty:
            raise ValueError(f"Catchment shapefile has no features: {catchment_path}")
        if gdf.crs is None:
            self.logger.warning(f"Catchment CRS not defined in {catchment_path}; assuming EPSG:4326")
            gdf = gdf.set_crs(epsg=4326)

        lon, lat = calculate_catchment_centroid(gdf, self.logger)
        area_m2 = calculate_catchment_area_km2(gdf, self.logger) * 1e6

        elev: Optional[float] = None
        if 'elev_mean' in gdf.columns:
            elevs = pd.to_numeric(gdf['elev_mean'], errors='coerce')
            weights = gdf.to_crs(gdf.estimate_utm_crs()).geometry.area
            valid = elevs.notna()
            if valid.any():
                elev = float((elevs[valid] * weights[valid]).sum() / weights[valid].sum())

        return {'lat': float(lat), 'lon': float(lon), 'area_m2': float(area_m2), 'elev': elev}

    def get_river_network_path(self) -> Path:
        """
        Get path to river network shapefile.

        Returns:
            Path to river network shapefile
        """
        river_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        river_name = self._get_config_value(lambda: self.config.paths.river_network_name)

        if river_name == 'default' or river_name is None:
            # Use the standardized method suffix for the filename
            method_suffix = self._get_method_suffix()
            river_name = f"{self.domain_name}_riverNetwork_{method_suffix}.shp"

        return river_path / river_name

    def _is_lumped(self) -> bool:
        """
        Check if domain is configured as lumped.

        Returns:
            True if lumped, False if distributed
        """
        return self.domain_definition_method == 'lumped'

    def get_dem_path(self) -> Path:
        """
        Get path to DEM file.

        Returns:
            Path to DEM file
        """
        dem_name = self._get_config_value(lambda: self.config.paths.dem_name)
        if dem_name == "default" or dem_name is None:
            dem_name = f"domain_{self.domain_name}_elv.tif"
        return self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")

    def get_timestep_config(self) -> Dict[str, Any]:
        """
        Get timestep configuration based on FORCING_TIME_STEP_SIZE.

        Provides standardized configuration for time-related parameters used
        across different models for data processing and unit conversions.

        Returns:
            Dict with keys:
                - resample_freq: Pandas resample frequency string (e.g., 'h', 'D')
                - time_units: NetCDF time units string
                - time_unit: Pandas timedelta unit
                - conversion_factor: Factor to convert from cms to mm/timestep
                - time_label: Human-readable label
                - timestep_seconds: Timestep in seconds
        """
        timestep_seconds = self.forcing_time_step_size

        if timestep_seconds == ModelDefaults.DEFAULT_TIMESTEP_HOURLY:  # Hourly
            return {
                'resample_freq': 'h',
                'time_units': 'hours since 1970-01-01',
                'time_unit': 'h',
                'conversion_factor': UnitConversion.MM_HOUR_TO_CMS,  # cms to mm/hour
                'time_label': 'hourly',
                'timestep_seconds': ModelDefaults.DEFAULT_TIMESTEP_HOURLY
            }
        elif timestep_seconds == ModelDefaults.DEFAULT_TIMESTEP_DAILY:  # Daily
            return {
                'resample_freq': 'D',
                'time_units': 'days since 1970-01-01',
                'time_unit': 'D',
                'conversion_factor': UnitConversion.MM_DAY_TO_CMS,  # cms to mm/day
                'time_label': 'daily',
                'timestep_seconds': ModelDefaults.DEFAULT_TIMESTEP_DAILY
            }
        else:
            # Generic case - calculate based on seconds
            hours = timestep_seconds / UnitConversion.SECONDS_PER_HOUR
            if hours < UnitConversion.HOURS_PER_DAY:
                return {
                    'resample_freq': f'{int(hours)}h',
                    'time_units': 'hours since 1970-01-01',
                    'time_unit': 'h',
                    'conversion_factor': UnitConversion.MM_HOUR_TO_CMS * hours,
                    'time_label': f'{int(hours)}-hourly',
                    'timestep_seconds': timestep_seconds
                }
            else:
                days = timestep_seconds / UnitConversion.SECONDS_PER_DAY
                return {
                    'resample_freq': f'{int(days)}D',
                    'time_units': 'days since 1970-01-01',
                    'time_unit': 'D',
                    'conversion_factor': UnitConversion.MM_DAY_TO_CMS * days,
                    'time_label': f'{int(days)}-daily',
                    'timestep_seconds': timestep_seconds
                }

    def get_base_settings_source_dir(self) -> Path:
        """
        Get the source directory for base settings files.

        Uses importlib.resources to load base settings from package data,
        working in both development and installed modes. If
        SYMFLUENCE_CODE_DIR is configured, prefer the local copy
        (src-based layout).

        Returns:
            Path to base settings directory for this model
        """
        from symfluence.core.modeling.base_settings import get_base_settings_dir

        # Check for code_dir in typed config
        code_dir_value = self._get_config_value(
            lambda: self.config.system.code_dir,
            default=None
        )
        if code_dir_value:
            code_dir = Path(code_dir_value)
            # Registry anchor -> src-layout path of the model package's
            # base_settings/ (e.g. symfluence.models.summa -> src/symfluence/
            # models/summa/base_settings).
            try:
                from symfluence.core.registries import R

                anchor = R.base_settings.get(self.model_name)
            except Exception:  # noqa: BLE001 — registry unavailable in stripped contexts
                anchor = None
            if anchor:
                pkg_settings = code_dir / "src" / Path(*anchor.split(".")) / "base_settings"
                if pkg_settings.exists():
                    return pkg_settings
            # Legacy central layout.
            base_settings_src = code_dir / "src" / "symfluence" / "resources" / "base_settings" / self.model_name
            if base_settings_src.exists():
                return base_settings_src

        return get_base_settings_dir(self.model_name)

    # =========================================================================
    # Time Window Utilities
    # =========================================================================

    def get_simulation_time_window(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Get simulation start/end times from typed config.

        Returns:
            Tuple of (start_time, end_time) as pandas Timestamps, or None if
            time window cannot be determined.
        """
        start_raw = self.config.domain.time_start
        end_raw = self.config.domain.time_end

        if not start_raw or not end_raw:
            return None

        try:
            return pd.to_datetime(start_raw), pd.to_datetime(end_raw)
        except Exception as exc:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Unable to parse simulation time window: {exc}", exc_info=True)
            return None

    #: ``(start, end)`` used by :meth:`_get_simulation_dates` when
    #: ``EXPERIMENT_TIME_START``/``_END`` are unset. ``None``: a missing value raises.
    SIMULATION_DATE_DEFAULTS: Optional[Tuple[str, str]] = None

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Simulation start/end (``EXPERIMENT_TIME_START``/``_END``) as ``datetime``.

        Strings are parsed with ``pd.Timestamp``, so a time-of-day is kept.
        Unset values fall back to ``SIMULATION_DATE_DEFAULTS``; with no
        default a missing value raises ``ValueError``.
        """
        default_start, default_end = self.SIMULATION_DATE_DEFAULTS or (None, None)
        start = self._get_config_value(lambda: self.config.domain.time_start, default=default_start)
        end = self._get_config_value(lambda: self.config.domain.time_end, default=default_end)
        return to_pydatetime(start), to_pydatetime(end)

    def subset_to_simulation_time(
        self,
        ds: xr.Dataset,
        label: str = "Data"
    ) -> xr.Dataset:
        """
        Subset dataset to the configured simulation time window.

        Args:
            ds: Dataset with a 'time' coordinate
            label: Label for logging messages (e.g., "Forcing", "Observations")

        Returns:
            Dataset subset to simulation window, or original if subsetting fails
        """
        time_window = self.get_simulation_time_window()
        if time_window is None or "time" not in ds.coords:
            return ds

        start_time, end_time = time_window
        try:
            subset = ds.sel(time=slice(start_time, end_time))
        except Exception as exc:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Unable to subset {label} to simulation window: {exc}", exc_info=True)
            return ds

        if len(subset.time) == 0:
            self.logger.warning(
                f"{label} has no records in simulation window; using full dataset"
            )
            return ds

        self.logger.info(
            f"{label} subset to simulation window: "
            f"{subset.time.min().values} to {subset.time.max().values}"
        )
        return subset

    def align_datasets_to_period(
        self,
        datasets: Dict[str, xr.Dataset],
        start_time: Union[datetime, pd.Timestamp],
        end_time: Union[datetime, pd.Timestamp],
        freq: str = 'D'
    ) -> Tuple[Dict[str, xr.Dataset], pd.DatetimeIndex]:
        """
        Align multiple datasets to a common time period with reindexing.

        This is useful when combining forcing data, observations, and other
        time series that may have slightly different time ranges.

        Args:
            datasets: Dict mapping names to xr.Dataset objects
            start_time: Start of the alignment period
            end_time: End of the alignment period
            freq: Pandas frequency string for the time index (e.g., 'D', 'h')

        Returns:
            Tuple of (aligned_datasets dict, time_index)
        """
        time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        aligned = {}

        for name, ds in datasets.items():
            try:
                aligned[name] = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
            except Exception as exc:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Could not align {name} to time period: {exc}", exc_info=True)
                aligned[name] = ds

        return aligned, time_index

    # =========================================================================
    # Template Method Pattern for Preprocessing
    # =========================================================================

    def run_preprocessing_template(self) -> bool:
        """
        Template method for preprocessing workflow.

        Provides a standard preprocessing workflow that models can use by
        overriding the hook methods. This ensures consistent error handling
        and logging across all model preprocessors.

        The workflow is:
        1. _pre_setup() - Model-specific pre-setup (optional)
        2. create_directories() - Create necessary directories
        3. copy_base_settings() - Copy base settings files
        4. _prepare_forcing() - Prepare forcing data (model-specific)
        5. _create_model_configs() - Create model config files (model-specific)
        6. _post_setup() - Model-specific post-setup (optional)

        Returns:
            True if preprocessing completed successfully

        Raises:
            ModelExecutionError: If any step fails
        """
        with symfluence_error_handler(
            f"{self.model_name} preprocessing",
            self.logger,
            error_type=ModelExecutionError
        ):
            self._pre_setup()
            self.create_directories()
            self.copy_base_settings()
            self._prepare_forcing()
            self._create_model_configs()
            self._post_setup()
            self.logger.info(f"{self.model_name} preprocessing completed successfully")
            return True

    def _pre_setup(self) -> None:
        """
        Hook for model-specific pre-setup tasks.

        Override in subclass to perform any setup needed before directory
        creation and settings copy. Default implementation does nothing.
        """
        pass

    def _prepare_forcing(self) -> None:
        """
        Hook for model-specific forcing data preparation.

        Override in subclass to implement forcing data processing.
        Default implementation does nothing.
        """
        pass

    def _create_model_configs(self) -> None:
        """
        Hook for model-specific configuration file creation.

        Override in subclass to create model-specific config files
        (e.g., file managers, parameter files). Default implementation
        does nothing.
        """
        pass

    def _post_setup(self) -> None:
        """
        Hook for model-specific post-setup tasks.

        Override in subclass to perform any cleanup or finalization
        after main preprocessing. Default implementation does nothing.
        """
        pass

    def _get_model_name(self) -> str:
        """
        Get the name of the model.

        Prefers the ``MODEL_NAME`` class variable. Subclasses that still
        override this method will continue to work.

        Returns:
            Model name string (e.g., 'SUMMA', 'FUSE', 'GR')
        """
        if self.MODEL_NAME:
            return self.MODEL_NAME
        raise NotImplementedError(
            f"{self.__class__.__name__} must set MODEL_NAME class variable "
            "or implement _get_model_name()"
        )

    @abstractmethod
    def run_preprocessing(self) -> bool:
        """
        Run model-specific preprocessing.

        Returns:
            True if preprocessing completed successfully
        """
        pass
