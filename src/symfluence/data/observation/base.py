# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base Observation Handler for SYMFLUENCE

Provides the abstract base class and utilities for observation data handlers,
including standardized error handling, output contracts, and common operations.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pandas as pd
import xarray as xr

from symfluence.core import ConfigurableMixin
from symfluence.core.exceptions import DataAcquisitionError
from symfluence.geospatial.coordinate_utils import CoordinateUtilsMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


# =============================================================================
# Custom Exceptions
# =============================================================================

class ObservationError(DataAcquisitionError):
    """Base exception for observation handling errors."""
    pass


class ObservationAcquisitionError(ObservationError):
    """Error during data acquisition phase."""
    pass


class ObservationProcessingError(ObservationError):
    """Error during data processing phase."""
    pass


class ObservationValidationError(ObservationError):
    """Error during data validation phase."""
    pass


# =============================================================================
# Output Data Contract
# =============================================================================

@dataclass
class ObservationMetadata:
    """Metadata for processed observation output."""
    source: str                    # e.g., "USGS_NWIS", "NASA_GRACE"
    variable: str                  # e.g., "streamflow", "tws_anomaly"
    units: str                     # e.g., "m3/s", "mm"
    temporal_resolution: str       # e.g., "daily", "monthly"
    spatial_aggregation: str       # e.g., "point", "basin_mean"
    processing_date: str           # ISO format
    quality_flags: Optional[Dict] = None  # QC info if available
    domain_name: Optional[str] = None
    station_id: Optional[str] = None
    data_range_start: Optional[str] = None
    data_range_end: Optional[str] = None
    n_records: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# Standard column names by observation type
STANDARD_COLUMNS: Dict[str, List[str]] = {
    'streamflow': ['datetime', 'value', 'quality_flag'],
    'soil_moisture': ['datetime', 'value', 'depth_m', 'quality_flag'],
    'snow_cover': ['datetime', 'sca_fraction', 'quality_flag'],
    'swe': ['datetime', 'swe_mm', 'quality_flag'],
    'et': ['datetime', 'et_mm_day', 'quality_flag'],
    'tws': ['datetime', 'tws_anomaly_mm', 'uncertainty_mm'],
    'lst': ['datetime', 'lst_k', 'quality_flag'],
    'lai': ['datetime', 'lai', 'quality_flag'],
    'precipitation': ['datetime', 'precip_mm', 'quality_flag'],
    'groundwater': ['datetime', 'groundwater_level', 'quality_flag'],
}


# =============================================================================
# Base Handler Class
# =============================================================================

class BaseObservationHandler(ABC, ConfigurableMixin, CoordinateUtilsMixin):
    """Abstract base class for observation data handlers.

    This class defines the interface for acquiring and processing observational data
    (e.g., GRACE water storage, MODIS snow cover, streamflow observations). Subclasses
    implement handlers for specific data sources.

    The handler is responsible for two main tasks:
    1. **Acquisition**: Downloading or locating raw data files from remote or local sources
    2. **Processing**: Converting raw data into SYMFLUENCE-standard formats (e.g., gridded
       NetCDF with standardized variable names and spatial/temporal coordinates)

    Class Attributes:
        obs_type (str): Observation type identifier (e.g., "streamflow", "snow_cover")
        source_name (str): Data source identifier (e.g., "USGS", "MODIS")

    Attributes:
        bbox (dict): Bounding box coordinates for spatial filtering (parsed from config)
        start_date (pd.Timestamp): Experiment start date for temporal filtering
        end_date (pd.Timestamp): Experiment end date for temporal filtering
        logger: Logger instance for diagnostic and error messages
    """

    # Class attributes to be overridden by subclasses
    obs_type: str = ""           # e.g., "streamflow", "snow_cover"
    source_name: str = ""        # e.g., "USGS", "MODIS"

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger
    ):
        """Initialize the observation handler.

        Args:
            config: SYMFLUENCE configuration object or dict. If dict, will be converted
                to SymfluenceConfig for type safety and validation.
            logger: Python logger instance for recording acquisition/processing events.

        Raises:
            ValueError: If config dict cannot be converted to SymfluenceConfig.
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        # Standard attributes via typed config access
        self.bbox = self._parse_bbox(self._get_config_value(
            lambda: self.config.domain.bounding_box_coords, default=None))
        self.start_date = pd.to_datetime(self._get_config_value(
            lambda: self.config.domain.time_start, default=None))
        self.end_date = pd.to_datetime(self._get_config_value(
            lambda: self.config.domain.time_end, default=None))

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def acquire(self) -> Path:
        """Acquire raw data from the source (download or locate local files).

        Subclasses must implement this method to retrieve raw observational data
        from their respective data source (e.g., GRACE server, USGS database).
        Implementations should handle authentication, error handling, and logging.

        Returns:
            Path: Local filesystem path to the acquired raw data file(s).

        Raises:
            IOError: If data cannot be retrieved from the source.
            ValueError: If data for the specified spatial/temporal bounds doesn't exist.
        """
        pass

    @abstractmethod
    def process(self, input_path: Path) -> Path:
        """Process raw data into SYMFLUENCE-standard formats.

        Subclasses must implement this method to transform raw data into a
        standardized format (typically gridded NetCDF) with:
        - Standardized variable names (e.g., 'SWE', 'ET', 'streamflow')
        - Proper spatial coordinates (lat/lon or projected CRS)
        - Proper temporal coordinates (datetime)
        - Required metadata (units, source, processing steps)

        Args:
            input_path: Path to raw data file(s) from acquire().

        Returns:
            Path: Local filesystem path to processed data file in standard format.

        Raises:
            IOError: If input file cannot be read.
            ValueError: If data cannot be processed into standard format.
        """
        pass

    # =========================================================================
    # Output Contract Properties
    # =========================================================================

    @property
    def standard_columns(self) -> List[str]:
        """Get standard columns for this observation type."""
        return STANDARD_COLUMNS.get(self.obs_type, ['datetime', 'value'])

    # =========================================================================
    # Safe Execution Methods (Error Handling)
    # =========================================================================

    def safe_acquire(self) -> Optional[Path]:
        """Acquire with standardized error handling.

        Returns:
            Path to acquired data, or None if acquisition fails.

        Raises:
            ObservationAcquisitionError: If acquisition fails with details.
        """
        try:
            return self.acquire()
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            source = self.source_name or self.__class__.__name__
            self.logger.error(f"{source} acquisition failed: {e}")
            raise ObservationAcquisitionError(f"{source}: {e}") from e

    def safe_process(self, input_path: Path) -> Optional[Path]:
        """Process with standardized error handling.

        Args:
            input_path: Path to raw data to process.

        Returns:
            Path to processed data, or None if processing fails.

        Raises:
            ObservationProcessingError: If processing fails with details.
        """
        try:
            return self.process(input_path)
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            source = self.source_name or self.__class__.__name__
            self.logger.error(f"{source} processing failed: {e}")
            raise ObservationProcessingError(f"{source}: {e}") from e

    # =========================================================================
    # Utility Methods - Path Construction
    # =========================================================================

    def _get_observation_dir(self, obs_type: str, stage: str = 'preprocessed') -> Path:
        """Construct standardized observation directory path.

        Args:
            obs_type: Observation type (e.g., 'streamflow', 'snow', 'grace')
            stage: Processing stage ('raw_data', 'preprocessed', 'processed')

        Returns:
            Path to observation directory (created if it doesn't exist)
        """
        output_dir = self.project_observations_dir / obs_type / stage
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_processed_data(
        self,
        df: pd.DataFrame,
        obs_type: str,
        suffix: str = 'processed',
        index_label: str = 'datetime'
    ) -> Path:
        """Save processed DataFrame to standard location with logging.

        Args:
            df: Processed DataFrame to save
            obs_type: Observation type for directory structure
            suffix: File name suffix (e.g., 'processed', 'raw')
            index_label: Label for the index column in CSV

        Returns:
            Path to saved file
        """
        output_dir = self._get_observation_dir(obs_type)
        output_file = output_dir / f"{self.domain_name}_{obs_type}_{suffix}.csv"
        df.to_csv(output_file, index_label=index_label)
        self.logger.info(f"{obs_type} processing complete: {output_file}")
        return output_file

    def _resolve_catchment_shapefile(self, required: bool = False) -> Optional[Path]:
        """Locate the catchment/basin polygon for spatial masking or averaging.

        Domain discretization writes the catchment under an experiment-scoped
        nested layout
        (``shapefiles/catchment/{definition_method}/{experiment_id}/
        {domain}_HRUs_{discretization}.shp``) and may upper-case the
        discretization suffix (GRUs -> GRUS), so a flat
        ``{domain}_catchment.shp`` lookup never matches. Resolution order:

        1. Explicit ``CATCHMENT_PATH`` / ``CATCHMENT_SHP_NAME`` (config.paths).
        2. Canonical nested catchment layout (definition_method/experiment_id),
           then the legacy flat ``shapefiles/catchment`` directory.
        3. Any ``{domain}_HRUs_*.shp`` anywhere under ``shapefiles/catchment``.
        4. The ``shapefiles/river_basins`` outline.

        Args:
            required: When True, raise FileNotFoundError if nothing is found;
                otherwise return None.

        Returns:
            Path to the resolved shapefile, or None when not found and not
            required.
        """
        # 1. Explicit user-provided path/name takes precedence. These keys live
        #    on config.paths (NOT config.domain — a common source of silent
        #    fall-through to defaults).
        catchment_path_cfg = self._get_config_value(
            lambda: self.config.paths.catchment_path, default='default', dict_key='CATCHMENT_PATH'
        )
        catchment_name_cfg = self._get_config_value(
            lambda: self.config.paths.catchment_name, default='default', dict_key='CATCHMENT_SHP_NAME'
        )
        if catchment_path_cfg and catchment_path_cfg != 'default':
            base_dir = Path(catchment_path_cfg)
            if catchment_name_cfg and catchment_name_cfg != 'default':
                explicit = base_dir / catchment_name_cfg
                if explicit.exists():
                    return explicit
            found = sorted(base_dir.rglob("*.shp"))
            if found:
                return found[0]

        catchment_dir = self.project_shapefiles_dir / "catchment"

        # 2. Canonical nested layout, then legacy flat directory.
        nested_dir = catchment_dir / self.domain_definition_method / self.experiment_id
        for candidate_dir in (nested_dir, catchment_dir):
            if not candidate_dir.exists():
                continue
            matches = (
                sorted(candidate_dir.glob(f"{self.domain_name}_HRUs_*.shp"))
                or sorted(candidate_dir.glob(f"{self.domain_name}*.shp"))
            )
            if not matches and candidate_dir == nested_dir:
                # The experiment subdir is catchment-specific; any .shp there is ours.
                matches = sorted(candidate_dir.glob("*.shp"))
            if matches:
                return matches[0]

        # 3. Deep search anywhere under catchment/ (handles casing/name drift).
        if catchment_dir.exists():
            deep = sorted(catchment_dir.rglob(f"{self.domain_name}_HRUs_*.shp"))
            if deep:
                return deep[0]

        # 4. River basins outline as a last resort.
        river_basins_dir = self.project_shapefiles_dir / "river_basins"
        if river_basins_dir.exists():
            rb_matches = (
                sorted(river_basins_dir.glob(f"{self.domain_name}_riverBasins_*.shp"))
                or sorted(river_basins_dir.glob("*.shp"))
            )
            if rb_matches:
                return rb_matches[0]

        if required:
            raise FileNotFoundError(
                f"Catchment/basin shapefile not found. Searched '{catchment_dir}' "
                f"(incl. {self.domain_definition_method}/{self.experiment_id}) and "
                f"river_basins. Run define_domain + discretize_domain first, or set "
                f"CATCHMENT_PATH/CATCHMENT_SHP_NAME explicitly."
            )
        return None

    def _load_catchment_gdf(self, required: bool = False):
        """Load the resolved catchment/basin shapefile as a GeoDataFrame.

        Returns None (with a warning) when the shapefile cannot be resolved and
        ``required`` is False; raises FileNotFoundError when ``required``.
        """
        basin_shp = self._resolve_catchment_shapefile(required=required)
        if basin_shp is None:
            self.logger.warning(
                "Catchment shapefile not found; spatial masking unavailable. "
                "Run define_domain + discretize_domain, or set CATCHMENT_PATH."
            )
            return None
        import geopandas as gpd  # lazy: heavy dependency
        self.logger.debug(f"Using catchment shapefile: {basin_shp}")
        return gpd.read_file(basin_shp)

    # =========================================================================
    # Utility Methods - Data Loading
    # =========================================================================

    def _open_dataset(self, path: Path, **kwargs) -> xr.Dataset:
        """Open NetCDF/HDF5 with engine fallback.

        Args:
            path: Path to dataset file
            **kwargs: Additional arguments passed to xr.open_dataset

        Returns:
            Opened xarray Dataset

        Raises:
            IOError: If file cannot be opened with any available engine
        """
        engines = ['netcdf4', 'h5netcdf', 'scipy']
        last_error = None

        for engine in engines:
            try:
                return xr.open_dataset(path, engine=engine, **kwargs)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                last_error = e
                continue

        raise IOError(f"Failed to open {path} with any engine: {last_error}")

    def _find_data_files(
        self,
        directory: Path,
        patterns: List[str],
        recursive: bool = True
    ) -> List[Path]:
        """Find data files matching patterns with fallback.

        Args:
            directory: Directory to search in
            patterns: List of glob patterns to try in order
            recursive: Whether to search recursively

        Returns:
            Sorted list of matching file paths
        """
        files: List[Path] = []
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
            if files:
                break
        return sorted(files)

    # =========================================================================
    # Utility Methods - Validation
    # =========================================================================

    def _validate_output(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate processed output meets requirements.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid, False otherwise
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            self.logger.warning(f"Missing required columns: {missing}")
            return False
        if df.empty:
            self.logger.warning("Output DataFrame is empty")
            return False
        return True

    # =========================================================================
    # Utility Methods - Metadata
    # =========================================================================

    def _create_metadata(self, **kwargs) -> ObservationMetadata:
        """Create metadata for this handler's output.

        Args:
            **kwargs: Override default metadata fields

        Returns:
            ObservationMetadata instance
        """
        defaults = {
            'source': self.source_name or self.__class__.__name__,
            'variable': self.obs_type or 'unknown',
            'units': 'unknown',
            'temporal_resolution': 'unknown',
            'spatial_aggregation': 'unknown',
            'processing_date': datetime.now().isoformat(),
            'domain_name': self.domain_name,
        }
        defaults.update(kwargs)
        return ObservationMetadata(**defaults)

    def _save_with_metadata(
        self,
        df: pd.DataFrame,
        metadata: ObservationMetadata,
        obs_type: Optional[str] = None
    ) -> Path:
        """Save data with accompanying metadata JSON.

        Args:
            df: DataFrame to save
            metadata: Metadata to save alongside data
            obs_type: Override observation type for path

        Returns:
            Path to saved data file
        """
        obs_type = obs_type or self.obs_type or 'unknown'

        # Update metadata with data statistics
        if not df.empty and df.index.name or hasattr(df.index, 'min'):
            metadata.n_records = len(df)
            try:
                metadata.data_range_start = str(df.index.min())
                metadata.data_range_end = str(df.index.max())
            except (AttributeError, TypeError):
                pass

        # Save data
        output_path = self._save_processed_data(df, obs_type)

        # Save metadata alongside
        meta_path = output_path.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        self.logger.debug(f"Metadata saved: {meta_path}")
        return output_path

    # =========================================================================
    # Utility Methods - Config Access
    # =========================================================================
    #
    # Observation handlers have two methods for config access:
    #
    # 1. _get_config_value(typed_accessor, default, dict_key) - from ConfigMixin
    #    Tries typed config first, then dict key fallback.
    #    Best when: You have a primary typed config path.
    #    Example:
    #        station_id = self._get_config_value(
    #            lambda: self.config.evaluation.streamflow.station_id,
    #            default=None,
    #            dict_key='STATION_ID'
    #        )
    #
    # 2. _config_value(*dict_keys, typed_path, default) - defined below
    #    Tries typed config first, then multiple dict keys in order.
    #    Best when: You need to check multiple possible config locations.
    #    Example:
    #        station_id = self._config_value(
    #            'STATION_ID', 'USGS_SITE_CODE', 'STREAMFLOW_STATION',
    #            typed_path=lambda: self.config.evaluation.streamflow.station_id,
    #            default=None
    #        )
    #
    # AVOID: Direct access to self.config_dict.get() without fallbacks.
    # =========================================================================

    def _config_value(
        self,
        *keys: str,
        typed_path: Optional[Callable] = None,
        default: Any = None
    ) -> Any:
        """
        Unified config access with fallback chain.

        Args:
            *keys: Config dict keys to try in order
            typed_path: Lambda to access typed config (e.g., lambda: self.config.evaluation.grace.path)
            default: Default value if all lookups fail

        Returns:
            Configuration value or default

        Example:
            station_id = self._config_value(
                'STATION_ID', 'USGS_SITE_CODE',
                typed_path=lambda: self.config.evaluation.streamflow.station_id,
                default=None
            )
        """
        # Try typed config first (if available)
        if typed_path is not None and self.config is not None:
            try:
                value = typed_path()
                if value is not None:
                    return value
            except (AttributeError, KeyError, TypeError):
                pass

        # Try dict keys in order
        for key in keys:
            value = self.config_dict.get(key)
            if value is not None:
                return value

        return default

    # =========================================================================
    # Utility Methods - Column/Coordinate Finding
    # =========================================================================

    def _find_col(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """
        Find a column name from a priority-ordered list of candidates.

        Candidates are tried in order, so earlier candidates win. An exact
        (case-insensitive) match on any candidate beats a substring match:
        first every candidate is tried as an exact name, then every candidate
        as a substring, in candidate order (file column order breaks ties).
        This keeps e.g. ``['discharge_vol', 'discharge_spec', 'discharge']``
        from returning ``discharge_spec`` because it sits first in the file,
        and ``'date'`` from grabbing ``update_date`` over ``date``.

        Args:
            columns: List of column names to search
            candidates: Candidate names, most preferred first

        Returns:
            Best matching column name, or None if no match found

        Example:
            datetime_col = self._find_col(df.columns, ['datetime', 'date_time', 'dateTime'])
        """
        cols = [str(c) for c in columns]
        lowered = [c.lower() for c in cols]
        for cand in candidates:
            target = cand.lower()
            for col, low in zip(cols, lowered):
                if low == target:
                    return col
        for cand in candidates:
            target = cand.lower()
            for col, low in zip(cols, lowered):
                if target in low:
                    return col
        return None

    def _daily_streamflow_to_target(self, series: 'pd.Series') -> 'pd.Series':
        """Put a DAILY-MEAN discharge series on the forcing timestep, honestly.

        Daily-mean archives (WSC daily means, SMHI, CAMELS, DGA, ANA, Hub'Eau
        QmJ) must not be linearly interpolated onto an hourly grid: that
        changes each day's mean when the series is aggregated back to daily
        and, with a gap limit, invents observations inside data gaps.

        - Sub-daily target: each hour carries its own day's mean (step
          function within the day); no interpolation across days and no gap
          filling (a missing day stays missing for all its hours), so the
          daily aggregate of the output equals the source daily mean.
        - Daily or coarser target: plain mean resample.

        The LamaH-Ice handler avoids the same problem by keeping its daily
        series native; both preserve the daily calibration target.
        """
        import pandas as pd

        series = series.sort_index()
        daily = series.resample('D').mean()
        freq = self._get_resample_freq()
        step = pd.to_timedelta(1, unit='D') if freq == 'D' else pd.to_timedelta(
            freq if freq[0].isdigit() else f'1{freq}')
        if step >= pd.Timedelta(days=1) or daily.empty:
            return daily if freq == 'D' else daily.resample(freq).mean()
        sub_index = pd.date_range(
            daily.index[0], daily.index[-1] + pd.Timedelta(days=1) - step, freq=step
        )
        # Every calendar day is present in ``daily`` (NaN when missing), so a
        # forward reindex never carries a value past its own day.
        out = daily.reindex(sub_index, method='ffill')
        out.index.name = series.index.name
        out.name = series.name
        return out

    def _get_resample_freq(self) -> str:
        """
        Get pandas resampling frequency based on configured timestep.

        Uses the forcing timestep size from config to determine appropriate
        resampling frequency for observation data alignment.

        Returns:
            Pandas frequency string ('h' for hourly, 'D' for daily, or '{n}s' for seconds)
        """
        from symfluence.core.constants import ModelDefaults

        timestep_size = self._config_value(
            'FORCING_TIME_STEP_SIZE', 'TIME_STEP_SIZE',
            typed_path=lambda: self.config.forcing.time_step_size,
            default=3600
        )
        timestep_size = int(timestep_size)

        if timestep_size <= 10800:  # 3 hours or less
            return 'h'
        elif timestep_size == ModelDefaults.DEFAULT_TIMESTEP_DAILY or timestep_size == 86400:
            return 'D'
        else:
            return f'{timestep_size}s'
