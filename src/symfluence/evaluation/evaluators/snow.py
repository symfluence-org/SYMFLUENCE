#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Snow (SWE/SCA) Evaluator.

Evaluates snow water equivalent (SWE) and snow-covered area (SCA) from model outputs.
Supports multi-target calibration with automatic target selection.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConverter
from symfluence.data.observation.paths import first_existing_path, snow_observation_candidates
from symfluence.evaluation.output_file_locator import OutputFileLocator
from symfluence.evaluation.registry import EvaluationRegistry

from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('SNOW')
class SnowEvaluator(ModelEvaluator):
    """Snow evaluator for SWE and SCA calibration.

    Evaluates simulated snow using one of two metrics:
    1. SWE (Snow Water Equivalent): Mass of water in snowpack (kg/m²)
       - Directly comparable to observations (in-situ, satellite-derived)
       - Continuous variable (0 to ~1000 kg/m²)
    2. SCA (Snow-Covered Area): Fraction of basin covered by snow (%)
       - Derived from satellite observations (MODIS, Landsat)
       - Discontinuous (0 to 100%, or binary presence/absence)

    Multi-target Support:
        Can override target via kwargs (for multivariate calibration).
        Supports SWE, SCA, snow_depth targets.

    Target Resolution Priority:
        1. kwargs target override (for multivariate mode)
        2. config.optimization.target (typed config)
        3. CALIBRATION_VARIABLE in dict config (default: swe)
        4. Pattern matching: if 'swe'/'snow' in name → swe; if 'sca' → sca

    Output Variables:
        - scalarSWE: SUMMA SWE output (kg/m²)
        - scalarSCA: SUMMA fractional SCA (0-1)
        - spatial dimensions: HRU, GRU, point (averaged/selected)

    Attributes:
        optimization_target: 'swe', 'sca', or 'snow_depth'
        variable_name: Same as optimization_target
    """

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger, **kwargs):
        """Initialize snow evaluator with target determination.

        Determines whether to evaluate SWE or SCA via multiple configuration sources.

        Args:
            config: Typed configuration object
            project_dir: Project root directory
            logger: Logger instance
            **kwargs: Optional target override (target='swe' or target='sca')
        """
        # Allow target override from kwargs (for multivariate calibration)
        self._target_override = kwargs.get('target')
        super().__init__(config, project_dir, logger)

        # Determine variable target: swe or sca
        self.optimization_target = self._target_override
        if self.optimization_target:
            self.optimization_target = self.optimization_target.lower()

        if not self.optimization_target:
            # Get from typed config, with fallback to flat config keys and CALIBRATION_VARIABLE
            opt_target = self._get_config_value(
                lambda: self.config.optimization.target,
                default=None,
                dict_key='OPTIMIZATION_TARGET'
            )

            calib_var = self._get_config_value(
                lambda: self.config.optimization.calibration_variable,
                default='swe',
                dict_key='CALIBRATION_VARIABLE'
            )
            self.logger.debug(f"Snow evaluator init: target={opt_target}, calib_var={calib_var}")
            self.optimization_target = (opt_target or calib_var).lower()

        if self.optimization_target not in ['swe', 'sca', 'snow_depth']:
            # Check if OPTIMIZATION_TARGET contains swe/sca/snow_depth keywords
            opt_target = self._get_config_value(
                lambda: self.config.optimization.target,
                default='',
                dict_key='OPTIMIZATION_TARGET'
            ).lower()
            if opt_target in ['swe', 'sca', 'snow_depth']:
                self.optimization_target = opt_target
            else:
                # Fall back to CALIBRATION_VARIABLE
                calibration_var = self._get_config_value(
                    lambda: self.config.optimization.calibration_variable,
                    default='',
                    dict_key='CALIBRATION_VARIABLE'
                ).lower()
                if 'swe' in calibration_var or 'snow' in calibration_var:
                    self.optimization_target = 'swe'
                elif 'sca' in calibration_var:
                    self.optimization_target = 'sca'

        self.logger.debug(f"Snow evaluator initialized with target: {self.optimization_target}")
        self.variable_name = self.optimization_target

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate snow output files containing SWE and/or SCA variables.

        Args:
            sim_dir: Directory containing simulation outputs

        Returns:
            List[Path]: Paths to snow output files (typically NetCDF)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_snow_files(sim_dir)

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract specified snow variable (SWE or SCA) from simulation output.

        Dispatches to _extract_swe_data() or _extract_sca_data() based on
        optimization_target determined during initialization.

        Args:
            sim_files: List of simulation output files
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Time series of selected snow variable

        Raises:
            Exception: If file cannot be read or variable not found
        """
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'swe':
                    return self._extract_swe_data(ds)
                elif self.optimization_target == 'sca':
                    return self._extract_sca_data(ds)
                elif self.optimization_target == 'snow_depth':
                    return self._extract_snow_depth_data(ds)
                else:
                    return self._extract_swe_data(ds)
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting snow data from {sim_file}: {str(e)}")
            raise

    def _extract_swe_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract Snow Water Equivalent (SWE) from SUMMA output.

        SWE is the mass of water contained in the snowpack (kg/m²).
        This method:
        1. Loads scalarSWE from NetCDF
        2. Collapses spatial dimensions (HRU/GRU) to basin scale
        3. Returns time series in kg/m²

        Spatial Aggregation:
            - Single HRU/GRU: selects that unit (isel)
            - Multiple units: averages across (mean)
            - Any other dimensions: selects first

        Args:
            ds: xarray Dataset with scalarSWE variable

        Returns:
            pd.Series: Time series of basin-scale SWE (kg/m²)

        Raises:
            ValueError: If scalarSWE not found in dataset
        """
        if 'scalarSWE' not in ds.variables:
            raise ValueError("scalarSWE variable not found")

        # Use base class method for spatial dimension collapse
        sim_data = self._collapse_spatial_dims(ds['scalarSWE'], aggregate='mean')

        if len(sim_data) > 0:
            self.logger.debug(f"SWE extraction: min={sim_data.min():.3f}, max={sim_data.max():.3f}, mean={sim_data.mean():.3f} kg/m² (n={len(sim_data)})")

        return sim_data

    def _extract_sca_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract Snow-Covered Area (SCA) from SUMMA output.

        SCA represents the fraction of the basin covered by snow (0-1 or 0-100%).
        This method handles two extraction strategies:

        1. Direct SCA variable (scalarGroundSnowFraction):
           - Returns fractional snow cover directly (0-1)
           - Preferred when available

        2. Derived from SWE (scalarSWE):
           - Converts SWE to binary snow presence using threshold
           - SWE > threshold → snow present (1.0)
           - SWE ≤ threshold → no snow (0.0)

        SCA from SWE Threshold: 1.0 kg/m²
        --------------------------------
        Physical reasoning for 1.0 kg/m² threshold:
          - 1 kg/m² = 1 mm water equivalent of snow
          - At typical snow density (100-300 kg/m³), this is 3-10 mm snow depth
          - This is the minimum detectable snow for most satellite sensors
          - MODIS snow detection limit is ~1-2 cm, roughly 2-5 kg/m² SWE
          - Using 1.0 kg/m² is conservative (captures thin snow cover)
          - Helps match satellite-derived SCA observations

        Note: This binary conversion loses information about snow depth.
        For continuous SCA, prefer scalarGroundSnowFraction if available.

        Args:
            ds: xarray Dataset with snow variables

        Returns:
            pd.Series: Time series of fractional SCA (0-1)

        Raises:
            ValueError: If no suitable SCA variable found
        """
        # SCA threshold for deriving snow cover from SWE (kg/m²)
        # 1.0 kg/m² ≈ 1 mm SWE ≈ minimum satellite-detectable snow
        SCA_SWE_THRESHOLD = 1.0

        sca_vars = ['scalarGroundSnowFraction', 'scalarSWE']
        for var_name in sca_vars:
            if var_name in ds.variables:
                # Use base class method for spatial dimension collapse
                sim_data = self._collapse_spatial_dims(ds[var_name], aggregate='mean')

                if var_name == 'scalarSWE':
                    # Convert SWE to binary snow presence
                    self.logger.debug(
                        f"Deriving SCA from SWE using threshold {SCA_SWE_THRESHOLD} kg/m²"
                    )
                    sim_data = (sim_data > SCA_SWE_THRESHOLD).astype(float)

                return sim_data
        raise ValueError("No suitable SCA variable found")

    def _extract_snow_depth_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract snow depth from SUMMA output.

        Snow depth represents the physical thickness of the snowpack (meters).
        This method handles two extraction strategies:

        1. Direct snow depth variable (scalarSnowDepth):
           - Returns snow depth directly in meters
           - Preferred when available

        2. Derived from SWE (scalarSWE):
           - Estimates depth using assumed snow density
           - depth = SWE / density
           - Default density: 250 kg/m³ (typical settled snow)

        Snow Density for SWE-to-Depth Conversion: 250 kg/m³
        ---------------------------------------------------
        Physical reasoning for 250 kg/m³ default:
          - Fresh snow: 50-100 kg/m³
          - Settled snow: 200-400 kg/m³
          - Old/compacted snow: 400-550 kg/m³
          - 250 kg/m³ represents typical mid-season settled snow
          - Conversion: depth_m = SWE_kg_m2 / 250 = SWE_mm / 250

        Note: Snow density varies significantly with climate, age, and
        metamorphism. For accurate depth, prefer scalarSnowDepth when available.

        Args:
            ds: xarray Dataset with snow variables

        Returns:
            pd.Series: Time series of snow depth (meters)

        Raises:
            ValueError: If no suitable snow depth variable found
        """
        # Default snow density for SWE-to-depth conversion (kg/m³)
        # 250 kg/m³ = typical settled seasonal snow
        DEFAULT_SNOW_DENSITY = 250.0

        if 'scalarSnowDepth' in ds.variables:
            sim_data = self._collapse_spatial_dims(ds['scalarSnowDepth'], aggregate='mean')
            self.logger.debug(f"Extracted snow depth directly: mean={sim_data.mean():.3f} m")
            return sim_data

        elif 'scalarSWE' in ds.variables:
            # Derive depth from SWE using assumed density
            swe_data = self._collapse_spatial_dims(ds['scalarSWE'], aggregate='mean')
            # SWE in kg/m² (= mm), density in kg/m³ → depth in m
            # depth = SWE / density = (kg/m²) / (kg/m³) = m
            snow_depth = swe_data / DEFAULT_SNOW_DENSITY
            self.logger.debug(
                f"Derived snow depth from SWE using density {DEFAULT_SNOW_DENSITY} kg/m³: "
                f"mean={snow_depth.mean():.3f} m"
            )
            return snow_depth

        raise ValueError("No suitable snow depth variable found (need scalarSnowDepth or scalarSWE)")

    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None,
                         mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True, **kwargs) -> Optional[Dict[str, float]]:
        """
        Calculate performance metrics for simulated snow data.

        Args:
            sim: Either a Path to simulation directory or a pre-loaded pd.Series
            obs: Optional pre-loaded pd.Series of observations. If None, loads from file.
            mizuroute_dir: mizuRoute simulation directory (if needed and sim is Path)
            calibration_only: If True, only use calibration period
        """
        simulated_data = sim
        # Ensure we are using the correct target if provided in kwargs
        if 'target' in kwargs:
            self.optimization_target = kwargs['target'].lower()
            self.variable_name = self.optimization_target

        # Call base class with proper signature: sim, obs=None, mizuroute_dir=None, calibration_only=True
        return super().calculate_metrics(
            sim=simulated_data,
            obs=obs,
            mizuroute_dir=mizuroute_dir,
            calibration_only=calibration_only
        )

    def get_observed_data_path(self) -> Path:
        """Get path to preprocessed observed snow data."""
        return first_existing_path(
            snow_observation_candidates(
                self.project_dir,
                self.domain_name,
                self.optimization_target,
            )
        )

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'swe':
            # Check for exact match first
            for col in columns:
                if col.lower() in ['swe', 'swe_mm']:
                    return col
            # Then check for patterns
            for col in columns:
                if any(term in col.lower() for term in ['swe', 'swe_mm', 'snow_water_equivalent', 'value', 'water_equiv']):
                    return col
        elif self.optimization_target == 'sca':
            for col in columns:
                if any(term in col.lower() for term in ['snow_cover_ratio', 'sca', 'snow_cover']):
                    return col
        elif self.optimization_target == 'snow_depth':
            # Check for exact match first
            for col in columns:
                if col.lower() in ['snow_depth', 'depth', 'snowdepth']:
                    return col
            # Then check for patterns
            for col in columns:
                if any(term in col.lower() for term in ['snow_depth', 'depth_m', 'depth_cm', 'hs', 'snowdepth']):
                    return col
        return None

    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.warning(f"Snow observation file not found: {obs_path}")
                return None

            obs_df = pd.read_csv(obs_path)

            self.logger.debug(f"Loading snow observations for target: {self.optimization_target}")
            self.logger.debug(f"Available columns: {list(obs_df.columns)}")

            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)

            self.logger.debug(f"Identified columns - date: {date_col}, data: {data_col}")

            if not date_col or not data_col:
                self.logger.warning(f"Could not find required columns in {obs_path}. Need Date and data column.")
                return None

            # Try default parsing, then dayfirst=True; use whichever preserves more rows
            # Handles DD/MM/YYYY formatted CSVs (e.g. SNOTEL processed data)
            dt_default = pd.to_datetime(obs_df[date_col], errors='coerce')
            dt_dayfirst = pd.to_datetime(obs_df[date_col], dayfirst=True, errors='coerce')
            if dt_dayfirst.notna().sum() > dt_default.notna().sum():
                obs_df['DateTime'] = dt_dayfirst
                self.logger.debug("Using dayfirst=True for date parsing (more valid dates)")
            else:
                obs_df['DateTime'] = dt_default
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)

            obs_series = obs_df[data_col].copy()
            missing_indicators = ['', ' ', 'NA', 'na', 'N/A', 'n/a', 'NULL', 'null', '-', '--', '---', 'missing', 'Missing', 'MISSING']
            for indicator in missing_indicators:
                obs_series = obs_series.replace(indicator, np.nan)

            obs_series = pd.to_numeric(obs_series, errors='coerce')

            if self.optimization_target == 'swe':
                # Convert if data is likely in inches (common for NRCS SNOTEL data)
                # Uses centralized UnitConverter for consistent unit handling
                obs_series = UnitConverter.swe_inches_to_mm(
                    obs_series,
                    auto_detect=True,
                    logger=self.logger
                )
                obs_series = obs_series[obs_series >= 0]
            elif self.optimization_target == 'snow_depth':
                # Snow depth observations may be in cm or m
                # If max > 50, assume cm and convert to m
                DEPTH_UNIT_THRESHOLD = 50  # meters - if max > 50, likely in cm
                if obs_series.max() > DEPTH_UNIT_THRESHOLD:
                    self.logger.debug(
                        f"Snow depth max={obs_series.max():.1f} > {DEPTH_UNIT_THRESHOLD}: "
                        "assuming cm, converting to m"
                    )
                    obs_series = obs_series / 100.0  # cm to m
                obs_series = obs_series[obs_series >= 0]

            return obs_series.dropna()
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error loading observed snow data: {str(e)}")
            return None

    def _convert_swe_units(self, obs_swe: pd.Series) -> pd.Series:
        """Convert SWE units from inches to kg/m² (mm water equivalent).

        Uses centralized UnitConverter for consistent unit handling.
        This method provides explicit (non-auto-detect) conversion.

        Physical basis:
          - 1 inch = 25.4 mm (exact definition)
          - SWE in mm = SWE in inches × 25.4
          - 1 mm SWE = 1 kg/m² (water density = 1000 kg/m³)

        Args:
            obs_swe: SWE observations in inches

        Returns:
            pd.Series: SWE observations in kg/m² (mm water equivalent)
        """
        return UnitConverter.swe_inches_to_mm(
            obs_swe,
            auto_detect=False,
            logger=self.logger
        )

    def needs_routing(self) -> bool:
        return False
