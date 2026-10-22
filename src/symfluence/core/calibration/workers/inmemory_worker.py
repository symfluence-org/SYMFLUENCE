# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Base class for in-memory hydrological model workers.

Provides shared functionality for models that run simulations in-memory
rather than through external executables (jFUSE, cFUSE, HBV, etc.).

This base class consolidates:
- Forcing data loading (NetCDF/CSV)
- Observation loading and unit conversion
- Catchment area retrieval with caching
- Output file saving (CSV + NetCDF)
- Metric calculation with alignment
- Unit conversion helpers (mm/day ↔ m³/s)
- Native gradient support hooks

Architecture:
    InMemoryModelWorker inherits from BaseWorker and provides common
    infrastructure for in-memory model evaluation. Concrete workers
    (JFUSEWorker, CFUSEWorker, HBVWorker) inherit from this class
    and implement model-specific simulation logic.

    BaseWorker
        └── InMemoryModelWorker  (common forcing/obs/metrics/gradients)
                ├── JFUSEWorker   (JAX-based FUSE)
                ├── CFUSEWorker   (C++/PyTorch FUSE)
                └── HBVWorker     (JAX-based HBV)
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.logging_utils import log_once
from symfluence.core.metrics import kge, nse
from symfluence.core.mixins.project import resolve_data_subdir

# Optional dependencies
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


class InMemoryModelWorker(BaseWorker):
    """Base class for in-memory hydrological model workers.

    Provides common infrastructure for models that run simulations entirely
    in-memory (no external executable), including:
    - Data loading (forcing and observations)
    - Unit conversion (mm/day ↔ m³/s)
    - Output file generation
    - Metric calculation
    - Native gradient support hooks

    Subclasses must implement:
    - _get_model_name(): Returns model identifier (e.g., 'HBV', 'JFUSE')
    - _get_forcing_subdir(): Returns forcing subdirectory name
    - _get_forcing_variable_map(): Maps standard names to model-specific names
    - _run_simulation(): Executes the model simulation

    Optionally override:
    - _initialize_model(): Setup model-specific components
    - _create_loss_function(): For native gradient support
    """

    # Conversion constant: mm/day to m³/s requires area in km²
    # Q(m³/s) = runoff(mm/day) × area(km²) / 86.4
    MM_DAY_TO_CMS_FACTOR = 86.4

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the in-memory model worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Data storage
        self._forcing: Optional[Dict[str, np.ndarray]] = None
        self._observations: Optional[np.ndarray] = None
        self._time_index: Optional[pd.DatetimeIndex] = None

        # Cached values
        self._catchment_area_km2: Optional[float] = None
        self._initialized: bool = False

        # Simulation results
        self._last_runoff: Optional[np.ndarray] = None
        self._current_params: Optional[Dict[str, float]] = None

        # Configuration
        self._warmup_days: int = self._get_warmup_days_config()

    @property
    def warmup_days(self) -> int:
        """Number of warmup days to skip in evaluation."""
        return self._warmup_days

    @warmup_days.setter
    def warmup_days(self, value: int) -> None:
        """Set warmup days."""
        self._warmup_days = value

    def warmup_steps(self) -> int:
        """Number of leading timesteps excluded as warmup.

        Defaults to ``warmup_days``, which is correct for daily models.
        Models running on a sub-daily timestep override this to convert
        days into timesteps.
        """
        return self.warmup_days

    def get_calibration_slice(self) -> Optional[Tuple[int, int]]:
        """Start/end indices of the calibration period in post-warmup arrays.

        Differentiable loss functions must score the same window the
        optimizer is reported against. Without this, a gradient-based
        optimizer trains on everything after warmup — including the
        held-out evaluation period — while the final evaluation reports
        calibration-period metrics, so the two numbers disagree and the
        split-sample test is no longer honest.

        Returns:
            ``(start, end)`` for slicing arrays that already have warmup
            removed, or None when no calibration period is configured or
            it does not overlap the simulated record.
        """
        cal_period = self._cfg(
            'CALIBRATION_PERIOD',
            self._cfg('EXPERIMENT_CALIBRATION_PERIOD', '')
        )
        if not cal_period or self._time_index is None:
            return None

        try:
            dates = [d.strip() for d in str(cal_period).split(',')]
            if len(dates) < 2:
                return None

            start_date = pd.Timestamp(dates[0])
            end_date = pd.Timestamp(dates[1])

            time_after_warmup = self._time_index[self.warmup_steps():]
            if not isinstance(time_after_warmup, pd.DatetimeIndex):
                time_after_warmup = pd.DatetimeIndex(time_after_warmup)

            cal_mask = (time_after_warmup >= start_date) & (time_after_warmup <= end_date)
            indices = np.where(cal_mask)[0]
            if len(indices) == 0:
                return None

            return int(indices[0]), int(indices[-1] + 1)
        except (ValueError, TypeError):
            return None

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model identifier (e.g., 'HBV', 'JFUSE', 'CFUSE').

        Used for file naming and logging.
        """
        pass

    @abstractmethod
    def _get_forcing_subdir(self) -> str:
        """Return the forcing subdirectory name (e.g., 'HBV_input', 'JFUSE_input')."""
        pass

    @abstractmethod
    def _get_forcing_variable_map(self) -> Dict[str, str]:
        """Return mapping from standard names to model-specific variable names.

        Standard names: 'precip', 'temp', 'pet'
        Returns dict like {'precip': 'pr', 'temp': 'temp', 'pet': 'pet'}
        """
        pass

    @abstractmethod
    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run model simulation and return runoff.

        Args:
            forcing: Dictionary with 'precip', 'temp', 'pet' arrays
            params: Parameter dictionary
            **kwargs: Model-specific arguments

        Returns:
            Runoff array in mm/day (full length including warmup)
        """
        pass

    # =========================================================================
    # Optional hooks for subclass customization
    # =========================================================================

    def _get_warmup_days_config(self) -> int:
        """Get warmup days from config. Override for model-specific key."""
        model_key = f"{self._get_model_name().upper()}_WARMUP_DAYS"
        return self._cfg(model_key, self._cfg('WARMUP_DAYS', 365))

    def _initialize_model(self) -> bool:
        """Initialize model-specific components. Override if needed.

        Returns:
            True if initialization successful
        """
        return True

    def _get_observation_unit_factor(self) -> float:
        """Get factor to convert observations to mm/day.

        Default assumes observations are in m³/s.
        Override if observations are in different units.

        Returns:
            Factor to multiply observations by
        """
        # Default: observations in m³/s, convert to mm/day
        # Q(mm/day) = Q(m³/s) × 86.4 / area(km²)
        area_km2 = self.get_catchment_area()
        return self.MM_DAY_TO_CMS_FACTOR / area_km2

    # =========================================================================
    # Data Loading
    # =========================================================================

    def initialize(self, task: Optional[WorkerTask] = None) -> bool:
        """Initialize model and load data.

        Args:
            task: Optional task containing paths

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Initialize model components
            if not self._initialize_model():
                return False

            # Load forcing data
            if not self._load_forcing(task):
                return False

            # Load observations
            if not self._load_observations(task):
                self.logger.warning("No observations loaded - calibration will fail")

            self._initialized = True
            n_timesteps = len(self._forcing['precip']) if self._forcing else 0
            self.logger.info(
                f"{self._get_model_name()} worker initialized: "
                f"{n_timesteps} timesteps"
            )
            return True

        except (FileNotFoundError, IOError, ImportError, ValueError) as e:
            self.logger.error(f"Failed to initialize {self._get_model_name()} worker: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _get_forcing_dir(self, task: Optional[WorkerTask] = None) -> Path:
        """Get path to forcing directory.

        Args:
            task: Optional task containing paths

        Returns:
            Path to forcing directory
        """
        if task and task.settings_dir:
            # Look relative to settings dir
            parent = task.settings_dir.parent.parent if task.settings_dir.parent else task.settings_dir
            forcing_dir = resolve_data_subdir(parent, 'forcing') / self._get_forcing_subdir()
            if forcing_dir.exists():
                return forcing_dir

        # Fall back to config-based path
        data_dir = Path(self._cfg('SYMFLUENCE_DATA_DIR', self._cfg('ROOT_PATH', '.')))
        domain_name = self._cfg('DOMAIN_NAME', 'domain')
        domain_dir = data_dir / f"domain_{domain_name}"
        return resolve_data_subdir(domain_dir, 'forcing') / self._get_forcing_subdir()

    def _load_forcing(self, task: Optional[WorkerTask] = None) -> bool:
        """Load forcing data from NetCDF or CSV.

        Args:
            task: Optional task containing paths

        Returns:
            True if loading successful
        """
        if self._forcing is not None:
            return True

        forcing_dir = self._get_forcing_dir(task)
        domain_name = self._cfg('DOMAIN_NAME', 'domain')
        model_name = self._get_model_name().lower()
        var_map = self._get_forcing_variable_map()

        # Try NetCDF first
        nc_patterns = [
            forcing_dir / f"{domain_name}_{model_name}_forcing.nc",
            forcing_dir / f"{domain_name}_forcing.nc",
        ]

        for nc_file in nc_patterns:
            if nc_file.exists() and HAS_XARRAY:
                try:
                    ds = xr.open_dataset(nc_file)
                    self._forcing = {}

                    for std_name, var_name in var_map.items():
                        if var_name in ds.variables:
                            self._forcing[std_name] = ds[var_name].values.flatten()
                        elif std_name in ds.variables:
                            self._forcing[std_name] = ds[std_name].values.flatten()
                        else:
                            log_once(
                                self.logger,
                                logging.WARNING,
                                key=f'forcing-var-missing-{nc_file}-{var_name}',
                                message=f"Variable {var_name} not found in {nc_file}",
                            )
                            continue

                    if 'time' in ds.coords:
                        self._time_index = pd.to_datetime(ds.time.values)

                    ds.close()

                    if len(self._forcing) >= 3:
                        self.logger.debug(f"Loaded forcing from {nc_file}")
                        return True
                except (OSError, RuntimeError, KeyError) as e:
                    log_once(
                        self.logger,
                        logging.WARNING,
                        key=f'forcing-load-error-{nc_file}',
                        message=f"Error loading {nc_file}: {e}",
                    )

        # Try CSV
        csv_patterns = [
            forcing_dir / f"{domain_name}_{model_name}_forcing.csv",
            forcing_dir / f"{domain_name}_forcing.csv",
        ]

        for csv_file in csv_patterns:
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    self._forcing = {}

                    for std_name, var_name in var_map.items():
                        if var_name in df.columns:
                            self._forcing[std_name] = df[var_name].values
                        elif std_name in df.columns:
                            self._forcing[std_name] = df[std_name].values

                    if 'time' in df.columns:
                        self._time_index = pd.to_datetime(df['time'])
                    elif 'datetime' in df.columns:
                        self._time_index = pd.to_datetime(df['datetime'])

                    if len(self._forcing) >= 3:
                        self.logger.debug(f"Loaded forcing from {csv_file}")
                        return True
                except (FileNotFoundError, ValueError, KeyError) as e:
                    log_once(
                        self.logger,
                        logging.WARNING,
                        key=f'forcing-load-error-{csv_file}',
                        message=f"Error loading {csv_file}: {e}",
                    )

        # Static misconfiguration: the same directory is missing on every
        # evaluation, so emit the ERROR once and demote repeats to DEBUG.
        log_once(
            self.logger,
            logging.ERROR,
            key=f'no-forcing-{forcing_dir}',
            message=(
                f"No forcing file found in {forcing_dir} - "
                f"all evaluations from this worker will return the penalty score "
                f"until the forcing data is provided"
            ),
        )
        return False

    def _load_observations(self, task: Optional[WorkerTask] = None) -> bool:
        """Load observation data.

        Args:
            task: Optional task containing paths

        Returns:
            True if loading successful
        """
        if self._observations is not None:
            return True

        domain_name = self._cfg('DOMAIN_NAME', 'domain')
        data_dir = Path(self._cfg('SYMFLUENCE_DATA_DIR', self._cfg('ROOT_PATH', '.')))
        project_dir = data_dir / f"domain_{domain_name}"

        # Try observations from forcing directory first
        forcing_dir = self._get_forcing_dir(task)
        obs_patterns = [
            forcing_dir / f"{domain_name}_observations.csv",
            resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv",
        ]

        for obs_file in obs_patterns:
            if obs_file.exists():
                try:
                    obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

                    # Ensure DatetimeIndex
                    if not isinstance(obs_df.index, pd.DatetimeIndex):
                        obs_df.index = pd.to_datetime(obs_df.index)

                    # Get first column (discharge)
                    obs_cms = obs_df.iloc[:, 0]

                    # Resample to daily if sub-daily
                    if len(obs_cms) > 1:
                        time_diff = obs_cms.index[1] - obs_cms.index[0]
                        if time_diff < pd.Timedelta(days=1):
                            self.logger.debug(f"Resampling observations from {time_diff} to daily")
                            obs_cms = obs_cms.resample('D').mean().dropna()

                    # Convert m³/s to mm/day
                    conversion_factor = self._get_observation_unit_factor()
                    obs_mm_day = obs_cms * conversion_factor

                    # Align with forcing time if available
                    if self._time_index is not None:
                        forcing_dates = pd.to_datetime(self._time_index).normalize()
                        obs_aligned = obs_mm_day.reindex(forcing_dates)
                        n_valid = (~obs_aligned.isna()).sum()
                        self.logger.debug(f"Aligned observations: {n_valid}/{len(forcing_dates)} valid")
                        self._observations = obs_aligned.values
                    else:
                        self._observations = obs_mm_day.values

                    self.logger.debug(f"Loaded observations from {obs_file}")
                    return True

                except (FileNotFoundError, ValueError, KeyError) as e:
                    log_once(
                        self.logger,
                        logging.WARNING,
                        key=f'obs-load-error-{obs_file}',
                        message=f"Error loading {obs_file}: {e}",
                    )

        log_once(
            self.logger,
            logging.WARNING,
            key='no-observation-file',
            message="No observation file found",
        )
        return False

    # =========================================================================
    # Catchment Area
    # =========================================================================

    def get_catchment_area(self) -> float:
        """Get catchment area in km².

        Tries multiple sources in order:
        1. Cached value
        2. Shapefile
        3. Config value
        4. Default fallback

        Returns:
            Catchment area in km²
        """
        if self._catchment_area_km2 is not None:
            return self._catchment_area_km2

        # Try shapefile
        try:
            import geopandas as gpd
            data_dir = Path(self._cfg('SYMFLUENCE_DATA_DIR', self._cfg('ROOT_PATH', '.')))
            domain_name = self._cfg('DOMAIN_NAME', 'domain')
            catchment_dir = data_dir / f"domain_{domain_name}" / 'shapefiles' / 'catchment'

            for pattern in ['*_HRUs_*.shp', '*_catchment*.shp', '*.shp']:
                shp_files = list(catchment_dir.rglob(pattern))
                if shp_files:
                    gdf = gpd.read_file(shp_files[0])
                    # Prefer canonical m² columns. LamaH-Ice shapefiles ship an
                    # `area_calc` column already in km², which the legacy
                    # first-match logic blindly divided by 1e6, producing a
                    # 1,000,000× too-small area and KGE = 1 − √2 across every
                    # j-stack run.
                    preferred = ['GRU_area', 'HRU_area', 'AREA_M2', 'area_m2']
                    chosen = next((c for c in preferred if c in gdf.columns), None)
                    if chosen is None:
                        area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                        chosen = area_cols[0] if area_cols else None
                    if chosen:
                        total_area_m2 = gdf[chosen].sum()
                        self._catchment_area_km2 = float(total_area_m2) / 1e6
                        self.logger.info(f"Catchment area from shapefile column '{chosen}': {self._catchment_area_km2:.2f} km²")
                        return self._catchment_area_km2
        except (FileNotFoundError, ValueError, AttributeError) as e:
            self.logger.debug(f"Could not read catchment area from shapefile: {e}")

        # Try config
        area_km2 = self._cfg('CATCHMENT_AREA_KM2')
        if area_km2 is None:
            area_km2 = self._cfg('CATCHMENT_AREA')
        if area_km2 is None:
            domain_config = self._cfg('DOMAIN', {})
            if isinstance(domain_config, dict):
                area_km2 = domain_config.get('catchment_area_km2')
                if area_km2 is None:
                    area_km2 = domain_config.get('catchment_area')

        if area_km2 is not None:
            self._catchment_area_km2 = float(area_km2)
            self.logger.info(f"Catchment area from config: {self._catchment_area_km2:.2f} km²")
            return self._catchment_area_km2

        # Default fallback
        self.logger.warning("Could not determine catchment area, using default 1000 km²")
        self._catchment_area_km2 = 1000.0
        return self._catchment_area_km2

    def get_catchment_area_m2(self) -> float:
        """Get catchment area in m²."""
        return self.get_catchment_area() * 1e6

    # =========================================================================
    # Unit Conversion
    # =========================================================================

    def runoff_to_streamflow(self, runoff_mm_day: np.ndarray) -> np.ndarray:
        """Convert runoff from mm/day to m³/s.

        Args:
            runoff_mm_day: Runoff array in mm/day

        Returns:
            Streamflow array in m³/s
        """
        area_km2 = self.get_catchment_area()
        return runoff_mm_day * area_km2 / self.MM_DAY_TO_CMS_FACTOR

    def streamflow_to_runoff(self, streamflow_cms: np.ndarray) -> np.ndarray:
        """Convert streamflow from m³/s to mm/day.

        Args:
            streamflow_cms: Streamflow array in m³/s

        Returns:
            Runoff array in mm/day
        """
        area_km2 = self.get_catchment_area()
        return streamflow_cms * self.MM_DAY_TO_CMS_FACTOR / area_km2

    # =========================================================================
    # Output File Saving
    # =========================================================================

    def save_output_files(
        self,
        runoff_mm_day: np.ndarray,
        output_dir: Path,
        time_index: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """Save simulation output to CSV and NetCDF files.

        Args:
            runoff_mm_day: Runoff array in mm/day
            output_dir: Directory to save files
            time_index: Optional time index (uses stored index if None)
        """
        if not HAS_XARRAY:
            self.logger.warning("xarray not available, skipping NetCDF output")
            return

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            domain_name = self._cfg('DOMAIN_NAME', 'domain')
            model_name = self._get_model_name().lower()

            # Use provided or stored time index
            if time_index is None:
                time_index = self._time_index
            if time_index is None:
                time_index = pd.date_range(start='2000-01-01', periods=len(runoff_mm_day), freq='D')

            # Ensure lengths match
            if len(time_index) != len(runoff_mm_day):
                self.logger.warning(
                    f"Time index length ({len(time_index)}) != runoff length ({len(runoff_mm_day)}). "
                    "Creating synthetic time index."
                )
                time_index = pd.date_range(start='2000-01-01', periods=len(runoff_mm_day), freq='D')

            # Convert to streamflow
            streamflow_cms = self.runoff_to_streamflow(runoff_mm_day)

            # Save CSV
            results_df = pd.DataFrame({
                'datetime': time_index,
                'streamflow_mm_day': runoff_mm_day,
                'streamflow_cms': streamflow_cms,
            })
            csv_file = output_dir / f"{domain_name}_{model_name}_output.csv"
            results_df.to_csv(csv_file, index=False)

            # Save NetCDF
            ds = xr.Dataset(
                data_vars={
                    'streamflow': (['time'], streamflow_cms.astype(np.float32)),
                    'runoff': (['time'], runoff_mm_day.astype(np.float32)),
                },
                coords={'time': time_index},
                attrs={
                    'model': self._get_model_name(),
                    'catchment_area_km2': self.get_catchment_area(),
                }
            )
            ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
            ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

            nc_file = output_dir / f"{domain_name}_{model_name}_output.nc"
            ds.to_netcdf(nc_file)
            ds.close()

            self.logger.debug(f"Saved {model_name} output to: {output_dir}")

        except (OSError, RuntimeError, IOError) as e:
            self.logger.warning(f"Failed to save output files: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    # =========================================================================
    # Metric Calculation
    # =========================================================================

    def calculate_streamflow_metrics(
        self,
        sim_mm_day: np.ndarray,
        obs_mm_day: np.ndarray,
        skip_warmup: bool = True,
        time_index: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """Calculate standard streamflow metrics.

        Args:
            sim_mm_day: Simulated runoff in mm/day
            obs_mm_day: Observed runoff in mm/day
            skip_warmup: Whether to skip warmup period
            time_index: Optional time index for period filtering

        Returns:
            Dictionary with 'kge', 'nse', 'n_points', and optionally 'error'
        """
        try:
            # Get time index for filtering
            if time_index is None:
                time_index = self._time_index

            # Skip warmup if requested
            if skip_warmup:
                sim = sim_mm_day[self.warmup_days:]
                obs = obs_mm_day[self.warmup_days:]
                if time_index is not None and len(time_index) > self.warmup_days:
                    time_index = time_index[self.warmup_days:]
            else:
                sim = sim_mm_day
                obs = obs_mm_day

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]
            if time_index is not None and len(time_index) > min_len:
                time_index = time_index[:min_len]

            # Filter to calibration period if specified
            cal_period = self._cfg('CALIBRATION_PERIOD', '')
            if cal_period and time_index is not None:
                try:
                    dates = [d.strip() for d in cal_period.split(',')]
                    if len(dates) >= 2:
                        start_date = pd.Timestamp(dates[0])
                        end_date = pd.Timestamp(dates[1])

                        # Ensure time_index is DatetimeIndex
                        if not isinstance(time_index, pd.DatetimeIndex):
                            time_index = pd.DatetimeIndex(time_index)

                        # Create mask for calibration period
                        cal_mask = (time_index >= start_date) & (time_index <= end_date)
                        sim = sim[cal_mask]
                        obs = obs[cal_mask]
                        self.logger.debug(
                            f"Filtered to calibration period {start_date} - {end_date}: "
                            f"{cal_mask.sum()} points"
                        )
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Could not filter to calibration period: {e}")

            # Remove NaN
            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim = sim[valid_mask]
            obs = obs[valid_mask]

            if len(sim) < 10:
                return {
                    'kge': self.penalty_score,
                    'nse': self.penalty_score,
                    'n_points': len(sim),
                    'error': 'Insufficient valid data points'
                }

            # Calculate metrics
            kge_val = float(kge(obs, sim, transfo=1))
            nse_val = float(nse(obs, sim, transfo=1))

            # Handle NaN
            if np.isnan(kge_val):
                kge_val = self.penalty_score
            if np.isnan(nse_val):
                nse_val = self.penalty_score

            return {
                'kge': kge_val,
                'nse': nse_val,
                'n_points': len(sim),
                'mean_sim': float(np.mean(sim)),
                'mean_obs': float(np.mean(obs)),
            }

        except (ValueError, ZeroDivisionError, KeyError) as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {
                'kge': self.penalty_score,
                'nse': self.penalty_score,
                'error': str(e)
            }

    # =========================================================================
    # BaseWorker Implementation
    # =========================================================================

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """Apply parameters for simulation.

        For in-memory models, parameters are passed directly to simulation,
        so this just stores them.

        Args:
            params: Parameter values
            settings_dir: Settings directory (unused)
            **kwargs: Additional arguments

        Returns:
            True (always succeeds for in-memory models)
        """
        # Initialize if needed
        task = kwargs.get('task')
        if not self._initialized:
            if not self.initialize(task):
                return False

        self._current_params = params
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """Run model simulation.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if simulation successful
        """
        try:
            # Ensure initialized
            if not self._initialized:
                if not self.initialize():
                    return False

            # Get parameters
            params = kwargs.pop('params', self._current_params)
            if params is None:
                self.logger.error("No parameters provided for simulation")
                return False

            # Run simulation
            self._last_runoff = self._run_simulation(
                self._forcing,
                params,
                **kwargs
            )

            # Save output if requested
            save_output = kwargs.get('save_output', False)
            if save_output and output_dir:
                # Adjust time index for warmup
                time_index = self._time_index
                if time_index is not None and len(time_index) > self.warmup_days:
                    time_index = time_index[self.warmup_days:]

                runoff_after_warmup = self._last_runoff[self.warmup_days:]
                self.save_output_files(runoff_after_warmup, output_dir, time_index)

            return True

        except (ValueError, RuntimeError, IOError) as e:
            self.logger.error(f"Error running {self._get_model_name()}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate metrics from simulation results.

        Args:
            output_dir: Output directory (unused - results in memory)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        if self._last_runoff is None:
            return {'kge': self.penalty_score, 'error': 'No simulation results'}

        if self._observations is None:
            return {'kge': self.penalty_score, 'error': 'No observations'}

        return self.calculate_streamflow_metrics(
            self._last_runoff,
            self._observations,
            skip_warmup=True
        )

    # =========================================================================
    # Native Gradient Support
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """Check if native gradient computation is available.

        Default: Returns True if JAX is installed.
        Override in subclass for model-specific requirements.

        Returns:
            True if native gradients supported
        """
        return HAS_JAX

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """Compute gradient of loss with respect to parameters.

        Default implementation uses finite differences.
        Subclasses can override with autodiff for efficiency.

        Args:
            params: Current parameter values
            metric: Metric to compute gradient for

        Returns:
            Dictionary of parameter gradients, or None if unavailable
        """
        # Default: return None (use finite differences in optimizer)
        return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Evaluate loss and compute gradient in single pass.

        Default implementation evaluates loss only.
        Subclasses with autodiff should override for efficiency.

        Args:
            params: Parameter values
            metric: Metric to evaluate

        Returns:
            Tuple of (loss_value, gradient_dict or None)
        """
        # Default: compute loss only
        loss = self._evaluate_loss(params, metric)
        return loss, self.compute_gradient(params, metric)

    def _evaluate_loss(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> float:
        """Evaluate loss for given parameters.

        Args:
            params: Parameter values
            metric: Metric to evaluate

        Returns:
            Loss value (negative of metric for minimization)
        """
        try:
            # Run simulation
            runoff = self._run_simulation(self._forcing, params)

            # Calculate metric
            metrics = self.calculate_streamflow_metrics(
                runoff,
                self._observations,
                skip_warmup=True
            )

            # Get requested metric
            metric_lower = metric.lower()
            if metric_lower in metrics:
                val = metrics[metric_lower]
            elif 'kge' in metrics:
                val = metrics['kge']
            else:
                return self.penalty_score

            # Return negative for minimization
            return -val if not np.isnan(val) else self.penalty_score

        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Error evaluating loss: {e}")
            return self.penalty_score
