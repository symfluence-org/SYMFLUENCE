# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Physical constants and unit conversion factors for SYMFLUENCE.

Centralizes all hardcoded constants to eliminate duplication and
improve maintainability across the codebase.
"""
from __future__ import annotations

import logging
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd


class UnitConversion:
    """
    Unit conversion factors for hydrological calculations.

    All factors are scientifically derived and documented to provide
    a single source of truth for unit conversions throughout SYMFLUENCE.
    """

    # Time constants
    SECONDS_PER_HOUR = 3600
    """Seconds in one hour."""

    SECONDS_PER_DAY = 86400
    """Seconds in one day (24 hours × 3600 seconds)."""

    HOURS_PER_DAY = 24
    """Hours in one day."""

    DAYS_PER_YEAR = 365.25
    """Average days per year accounting for leap years."""

    # Streamflow conversions
    MM_DAY_TO_CMS = SECONDS_PER_DAY / 1000.0
    """
    Convert mm/day to m³/s (cms) per km² of catchment area.

    Formula: Q(cms) = Q(mm/day) * Area(km²) / MM_DAY_TO_CMS

    Derivation:
        1 mm/day over 1 km² =
        (0.001 m) × (1,000,000 m²) / (86,400 s) =
        1000 m³ / 86,400 s =
        0.01157 m³/s

    Therefore: 86.4 = 86,400 / 1000
    """

    MM_HOUR_TO_CMS = SECONDS_PER_HOUR / 1000.0
    """
    Convert mm/hour to m³/s per km² of catchment area.

    Formula: Q(cms) = Q(mm/hour) * Area(km²) / MM_HOUR_TO_CMS

    Derivation:
        1 mm/hour over 1 km² =
        (0.001 m) × (1,000,000 m²) / (3,600 s) =
        1000 m³ / 3,600 s =
        0.278 m³/s

    Therefore: 3.6 = 3,600 / 1000
    """

    CFS_TO_CMS = 0.028316846592
    """
    Convert cubic feet per second to cubic meters per second.

    1 cubic foot = 0.028316846592 cubic meters (exact)
    """

    # Area conversions
    M2_TO_KM2 = 1e-6
    """Convert square meters to square kilometers (m² / 1,000,000)."""

    KM2_TO_M2 = 1e6
    """Convert square kilometers to square meters (km² × 1,000,000)."""

    HA_TO_KM2 = 0.01
    """Convert hectares to square kilometers (1 ha = 0.01 km²)."""

    # Pressure conversions
    PA_TO_KPA = 0.001
    """Convert Pascals to kiloPascals."""

    # Length conversions
    FEET_TO_METERS = 0.3048
    """
    Convert feet to meters.

    1 foot = 0.3048 meters (exact, international foot)
    """

    METERS_TO_FEET = 3.28084
    """
    Convert meters to feet.

    1 meter = 3.28084 feet (approximate)
    """

    @classmethod
    def mm_per_timestep_to_cms_factor(
        cls,
        timestep_seconds: int
    ) -> float:
        """
        Get conversion factor for mm/timestep to cms.

        This is the divisor to convert mm per timestep to m³/s per km².

        Args:
            timestep_seconds: Model timestep in seconds

        Returns:
            Conversion factor (timestep_seconds / 1000)
        """
        return timestep_seconds / 1000.0


class PhysicalConstants:
    """
    Physical constants for hydrological and meteorological calculations.

    Values are from standard reference sources and widely accepted
    approximations used in hydrological modeling.
    """

    # Temperature conversions
    KELVIN_OFFSET = 273.15
    """Offset to convert Celsius to Kelvin (T_K = T_C + 273.15)."""

    # Water properties
    WATER_DENSITY = 1000.0
    """Water density in kg/m³ at 4°C (maximum density)."""

    LATENT_HEAT_VAPORIZATION = 2.45e6
    """
    Latent heat of vaporization of water in J/kg at 20°C.

    Used in evapotranspiration calculations. Value varies with temperature:
    - 0°C: 2.501 × 10⁶ J/kg
    - 20°C: 2.453 × 10⁶ J/kg
    - 100°C: 2.257 × 10⁶ J/kg
    """

    SPECIFIC_HEAT_WATER = 4186.0
    """Specific heat capacity of water in J/(kg·K) at 15°C."""

    # Atmospheric constants
    STEFAN_BOLTZMANN = 5.67e-8
    """Stefan-Boltzmann constant in W/(m²·K⁴) for radiation calculations."""

    GAS_CONSTANT_DRY_AIR = 287.05
    """Specific gas constant for dry air in J/(kg·K)."""

    # Earth constants
    GRAVITY = 9.80665
    """
    Standard acceleration due to gravity in m/s².

    This is the internationally adopted standard value. Actual gravity
    varies slightly with latitude and elevation.
    """

    EARTH_RADIUS_KM = 6371.0
    """Mean radius of Earth in kilometers."""

    MOLAR_MASS_DRY_AIR = 0.0289644
    """Molar mass of dry air in kg/mol."""

    UNIVERSAL_GAS_CONSTANT = 8.31447
    """Universal gas constant in J/(mol·K)."""

    STANDARD_LAPSE_RATE = 0.0065
    """Standard environmental lapse rate in K/m."""


class ModelDefaults:
    """Default configuration values used across models."""

    # Timesteps
    DEFAULT_TIMESTEP_HOURLY = 3600
    """Default hourly timestep in seconds."""

    DEFAULT_TIMESTEP_DAILY = 86400
    """Default daily timestep in seconds."""

    # Spatial configuration
    DEFAULT_DISCRETIZATION = 'lumped'
    """Default spatial discretization method."""

    # Temporal configuration
    DEFAULT_SPINUP_DAYS = 365
    """Default spin-up period in days for model initialization."""

    # Numerical precision
    DEFAULT_TOLERANCE = 1e-6
    """Default numerical tolerance for convergence checks."""

    # Optimization/Calibration
    PENALTY_SCORE = -9999.0
    """
    Standard penalty score for failed model evaluations during optimization.

    Used when a model run fails, produces invalid output, or encounters errors.
    This value ensures failed runs are strongly penalized in optimization but
    still distinguishable from uninitialized values.
    """


# Export convenience dictionary for backward compatibility
UNIT_CONVERSIONS: Dict[str, float] = {
    'mm_day_to_cms': UnitConversion.MM_DAY_TO_CMS,
    'mm_hour_to_cms': UnitConversion.MM_HOUR_TO_CMS,
    'cfs_to_cms': UnitConversion.CFS_TO_CMS,
    'm2_to_km2': UnitConversion.M2_TO_KM2,
    'km2_to_m2': UnitConversion.KM2_TO_M2,
    'seconds_per_day': UnitConversion.SECONDS_PER_DAY,
}
"""
Convenience dictionary for accessing unit conversion factors.

Provided for backward compatibility and convenience. Prefer using
UnitConversion class directly for better IDE support and documentation.
"""


class UnitConverter:
    """Centralized unit conversion utilities for hydrological data.

    Provides static methods for common unit conversions with optional
    auto-detection of input units based on data magnitude heuristics.

    This class consolidates scattered conversion logic from evaluators
    to ensure consistent unit handling across the evaluation suite.

    Thresholds and Constants:
        MASS_FLUX_THRESHOLD: 1e-6 m/s - values above indicate mass flux units
        SWE_UNIT_THRESHOLD: 250 - SWE values above indicate mm units
        SECONDS_PER_DAY: 86400 - conversion factor for daily rates
        INCHES_TO_MM: 25.4 - standard conversion factor
    """

    MASS_FLUX_THRESHOLD = 1e-6  # m/s threshold for detecting mass flux
    SWE_UNIT_THRESHOLD = 250    # values below likely in inches
    SECONDS_PER_DAY = 86400
    INCHES_TO_MM = 25.4

    @classmethod
    def et_mass_flux_to_mm_day(
        cls,
        data: 'pd.Series',
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert ET from kg m⁻² s⁻¹ to mm/day.

        SUMMA outputs evapotranspiration in mass flux units (kg m⁻² s⁻¹).
        This converts to the more common mm/day representation.

        Physical basis:
            - 1 kg m⁻² s⁻¹ = 1 mm/s (assuming water density = 1000 kg/m³)
            - 1 mm/s × 86400 s/day = 86400 mm/day

        Args:
            data: ET time series in kg m⁻² s⁻¹
            logger: Optional logger for debug messages

        Returns:
            ET time series in mm/day
        """
        if logger:
            logger.debug("Converting ET from kg m⁻² s⁻¹ to mm/day")
        return data * cls.SECONDS_PER_DAY

    @classmethod
    def swe_inches_to_mm(
        cls,
        data: 'pd.Series',
        auto_detect: bool = True,
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert SWE from inches to mm with optional auto-detection.

        SNOTEL and some other data sources report SWE in inches.
        This method can auto-detect based on data magnitude.

        Auto-detection logic:
            If max(data) < SWE_UNIT_THRESHOLD (250), assume inches
            Rationale: 250 inches = 6350 mm is an extremely high SWE value

        Args:
            data: SWE time series (potentially in inches)
            auto_detect: If True, detect units from data magnitude
            logger: Optional logger for debug messages

        Returns:
            SWE time series in mm
        """
        import numpy as np

        if auto_detect:
            max_val = np.nanmax(data)
            if max_val < cls.SWE_UNIT_THRESHOLD:
                if logger:
                    logger.debug(
                        f"SWE max={max_val:.1f} < {cls.SWE_UNIT_THRESHOLD}: "
                        f"assuming inches, converting to mm"
                    )
                return data * cls.INCHES_TO_MM
            else:
                if logger:
                    logger.debug(
                        f"SWE max={max_val:.1f} >= {cls.SWE_UNIT_THRESHOLD}: "
                        f"assuming already in mm"
                    )
                return data
        else:
            if logger:
                logger.debug("Converting SWE from inches to mm (forced)")
            return data * cls.INCHES_TO_MM

    @staticmethod
    def swe_column_units(column_name: str) -> Optional[str]:
        """Units encoded in a processed SWE column name, or None if absent.

        Observation handlers write the unit into the column name
        (``swe_mm`` for CanSWE/NorSWE, ``swe_in`` for SNOTEL) so readers do
        not have to guess it from the data magnitude.
        """
        name = str(column_name).strip().lower()
        if name.endswith(('_mm', '(mm)', '_kg_m2', '_kgm-2')):
            return 'mm'
        if name.endswith(('_in', '_inch', '_inches', '(in)')):
            return 'in'
        return None

    @classmethod
    def swe_series_to_mm(
        cls,
        data: 'pd.Series',
        column_name: str,
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert a processed SWE column to mm using the unit in its name.

        Only legacy files whose column carries no unit (e.g. a bare ``swe``
        written by older SNOTEL processing) fall back to magnitude detection,
        with a warning: shallow-snow mm series (max < 250 mm) are otherwise
        indistinguishable from inches and would be inflated 25.4x.
        """
        units = cls.swe_column_units(column_name)
        if units == 'mm':
            return data
        if units == 'in':
            return data * cls.INCHES_TO_MM
        if logger:
            logger.warning(
                f"SWE column '{column_name}' does not state its units; guessing "
                f"from magnitude (max < {cls.SWE_UNIT_THRESHOLD} -> inches). "
                "Re-run observation processing to write a unit-explicit column "
                "(swe_mm / swe_in)."
            )
        return cls.swe_inches_to_mm(data, auto_detect=True, logger=logger)

    @classmethod
    def detect_and_convert_mass_flux(
        cls,
        data: 'pd.Series',
        logger: Optional[logging.Logger] = None
    ) -> Tuple['pd.Series', bool]:
        """Detect mass flux and convert to volume flux if needed.

        SUMMA may output runoff in mass flux (kg m⁻² s⁻¹) incorrectly
        labeled as volume flux (m s⁻¹). This detects and corrects.

        Detection logic:
            If mean(data) > MASS_FLUX_THRESHOLD (1e-6 m/s), likely mass flux
            Rationale: 1e-6 m/s = 86.4 mm/day runoff (extremely high)

        Args:
            data: Runoff time series (possibly mislabeled units)
            logger: Optional logger for debug messages

        Returns:
            Tuple of (converted_data, was_converted)
        """
        import numpy as np

        mean_val = np.nanmean(data)
        if mean_val > cls.MASS_FLUX_THRESHOLD:
            if logger:
                logger.debug(
                    f"Mean runoff {mean_val:.2e} > {cls.MASS_FLUX_THRESHOLD:.0e} m/s: "
                    f"detected mass flux, dividing by 1000"
                )
            return data / 1000.0, True
        return data, False

    @classmethod
    def streamflow_mm_day_to_cms(
        cls,
        data: 'pd.Series',
        catchment_area_m2: float,
        logger: Optional[logging.Logger] = None
    ) -> 'pd.Series':
        """Convert mm/day to m³/s given catchment area.

        Converts per-unit-area runoff depth to volumetric discharge.

        Formula:
            Q(m³/s) = depth(mm/day) × area(m²) / (1000 mm/m × 86400 s/day)

        Args:
            data: Runoff time series in mm/day
            catchment_area_m2: Catchment area in square meters
            logger: Optional logger for debug messages

        Returns:
            Discharge time series in m³/s
        """
        if logger:
            logger.debug(
                f"Converting mm/day to m³/s with area={catchment_area_m2:.0f} m²"
            )
        return data * catchment_area_m2 / (1000.0 * cls.SECONDS_PER_DAY)


class UnitDetectionThresholds:
    """
    Heuristic thresholds for automatic unit detection.

    These thresholds are used to infer input data units based on typical
    value ranges. While convenient, explicit unit specification is always
    preferred when available.

    Warning:
        These heuristics may fail for unusual data ranges. Always verify
        unit conversions are applied correctly for your specific data.
    """

    # Area unit detection
    AREA_KM2_VS_M2 = 1000.0
    """
    Threshold for detecting km² vs m² area units.

    If mean(area) < 1000, values are assumed to be in km² and converted to m².
    Rationale: A catchment of 1000 m² (0.001 km²) is unrealistically small,
    while 1000 km² is a reasonable catchment size.
    """

    # Temperature unit detection
    TEMP_KELVIN_VS_CELSIUS = 100.0
    """
    Threshold for detecting Kelvin vs Celsius temperature units.

    If mean(temp) > 100, values are assumed to be in Kelvin and converted to °C.
    Rationale: Mean temperatures on Earth never exceed 100°C, but 100K (-173°C)
    is well below any terrestrial temperature, so values > 100 must be Kelvin.
    """

    # Flux rate detection
    FLUX_RATE_MM_S_VS_MM_DAY = 0.01
    """
    Threshold for detecting mm/s vs mm/day flux rates (PET, evaporation).

    If mean(|flux|) < 0.01, values are assumed to be in mm/s (or similar small
    rate units) and converted to mm/day.
    Rationale: Typical daily PET is 1-10 mm/day. A mean of 0.01 mm/day is
    unrealistically low, suggesting the data is in mm/s (× 86400 for mm/day).
    """


class SupportedModels:
    """Validity check for model names, derived from the component registry.

    A model is "supported" once it has been registered with the framework via
    an entry point / ``model_manifest`` (RTI review item 18). Registration is
    therefore the whitelist: this no longer hardcodes a model list. The former
    sibling lists (models with a forcing adapter / plotter / preset) were
    likewise hardcoded and drifted from reality; those call sites now read the
    registry instead (``R.plotters`` for already-registered plotters;
    ``R.forcing_adapters`` / ``R.presets`` for capabilities a package declares
    with ``model_manifest(forcing_adapter_module=..., init_preset_module=...)``
    and a consumer drains with ``Registry.load_modules()``).
    """

    #: Models whose "calibration" is internal training (gradient descent during
    #: the run step), not an external DDS/PSO parameter search. They register no
    #: optimizer/worker and have no calibrated parameters, so both the calibration
    #: and sensitivity-analysis paths skip them rather than reporting a failure.
    #:
    #: Declared by the owning package via ``model_manifest(self_training=True)``
    #: rather than hardcoded here: core cannot know which models an installed
    #: plugin trains internally, and a hardcoded list could only ever describe
    #: the in-tree suite.
    class _SelfTraining(AbstractSet):
        """Live set view over the models that declared ``self_training``.

        A real ``collections.abc.Set`` rather than a hand-rolled stand-in, so
        set algebra (``-``, ``&``, ``|``) works against it as callers expect.
        Deliberately UNHASHABLE: the contents come from plugin registration, so
        a hash taken before discovery would not match one taken after.
        """

        @classmethod
        def _from_iterable(cls, iterable):
            # Set operations build their result through this hook, which
            # defaults to cls(iterable); this class is a live view and takes no
            # arguments, so results come back as plain frozensets.
            return frozenset(iterable)

        def _resolved(self) -> frozenset:
            from symfluence.core.registry import self_training_models

            return self_training_models()

        def __contains__(self, item: object) -> bool:
            return isinstance(item, str) and item.upper() in self._resolved()

        def __iter__(self):
            return iter(self._resolved())

        def __len__(self) -> int:
            return len(self._resolved())

        def __repr__(self) -> str:
            return f"SELF_TRAINING({sorted(self._resolved())})"

    SELF_TRAINING = _SelfTraining()

    @classmethod
    def is_valid(cls, model_name: str) -> bool:
        """Return True if *model_name* is a model registered with the framework."""
        from symfluence.core.registries import R

        key = model_name.upper()
        return key in R.runners or key in R.config_schemas


class ConfigKeys:
    """
    Configuration key constants for standardized config access.

    Centralizes all configuration key strings used throughout SYMFLUENCE
    to eliminate magic strings and improve maintainability. These keys
    correspond to the flattened dictionary representation of SymfluenceConfig.

    Usage:
        from symfluence.core.constants import ConfigKeys

        value = self._get_config_value(
            lambda: self.config.domain.name,
            default='unnamed',
            dict_key=ConfigKeys.DOMAIN_NAME
        )
    """

    # System configuration
    SYMFLUENCE_DATA_DIR = 'SYMFLUENCE_DATA_DIR'
    """Root data directory for all SYMFLUENCE data."""

    # Domain configuration
    DOMAIN_NAME = 'DOMAIN_NAME'
    """Name of the study domain."""

    EXPERIMENT_ID = 'EXPERIMENT_ID'
    """Unique identifier for the experiment/run."""

    DOMAIN_DEFINITION_METHOD = 'DOMAIN_DEFINITION_METHOD'
    """Method used to define the domain (delineate, subset, lumped)."""

    SUB_GRID_DISCRETIZATION = 'SUB_GRID_DISCRETIZATION'
    """Sub-grid discretization method (lumped, elevation, point)."""

    ELEVATION_BAND_SIZE = 'ELEVATION_BAND_SIZE'
    """Elevation band size for discretization in meters."""

    # Path configuration
    OBSERVATIONS_PATH = 'OBSERVATIONS_PATH'
    """Path to streamflow observations file."""

    RIVER_BASINS_NAME = 'RIVER_BASINS_NAME'
    """Name of the river basins shapefile."""

    RIVER_BASINS_PATH = 'RIVER_BASINS_PATH'
    """Path to the river basins shapefile."""

    RIVER_NETWORK_SHP_NAME = 'RIVER_NETWORK_SHP_NAME'
    """Name of the river network shapefile."""

    RIVER_NETWORK_SHP_SEGID = 'RIVER_NETWORK_SHP_SEGID'
    """Segment ID field in river network shapefile."""

    POUR_POINT_SHP_NAME = 'POUR_POINT_SHP_NAME'
    """Name of the pour point shapefile."""

    CATCHMENT_SHP_NAME = 'CATCHMENT_SHP_NAME'
    """Name of the catchment shapefile."""

    CATCHMENT_SHP_HRUID = 'CATCHMENT_SHP_HRUID'
    """HRU ID field in catchment shapefile."""

    # Optimization configuration
    OPTIMIZATION_METRIC = 'OPTIMIZATION_METRIC'
    """Metric used for optimization (KGE, NSE, RMSE, etc.)."""

    OPTIMIZATION_TARGET = 'OPTIMIZATION_TARGET'
    """Target variable for optimization (streamflow, swe, et)."""

    # Time configuration
    SPINUP_PERIOD = 'SPINUP_PERIOD'
    """Spin-up period string (e.g., '1981-01-01,1982-01-01')."""

    CALIBRATION_PERIOD = 'CALIBRATION_PERIOD'
    """Calibration period string."""

    EVALUATION_PERIOD = 'EVALUATION_PERIOD'
    """Evaluation period string."""

    # Spatial configuration
    SIM_REACH_ID = 'SIM_REACH_ID'
    """Simulated reach ID for routing comparison."""

    POUR_POINT_COORDS = 'POUR_POINT_COORDS'
    """Pour point coordinates [lat, lon]."""

    POUR_POINT_ADDITIONAL_COORDS = 'POUR_POINT_ADDITIONAL_COORDS'
    """Additional interior outlet coordinates (semidistributed/distributed only).

    One or more 'lat/lon' pairs written into the outlets shapefile alongside
    POUR_POINT_COORDS so TauDEM breaks the stream network at each, forcing
    subbasin/GRU boundaries to align with interior gauges.
    """

    # Forcing configuration
    FORCING_DATASET = 'FORCING_DATASET'
    """Name of the forcing dataset (ERA5, RDRS, etc.)."""

    # Shapefile attribute fields
    RIVER_BASIN_SHP_AREA = 'RIVER_BASIN_SHP_AREA'
    """Area field name in river basin shapefile."""


class DiscretizationMethods:
    """
    Constants for sub-grid discretization methods.

    These define how the domain is divided into hydrological response units (HRUs)
    for spatial representation in models.
    """

    LUMPED = 'lumped'
    """Single HRU representing the entire catchment."""

    POINT = 'point'
    """Point-based discretization for specific locations."""

    ELEVATION = 'elevation'
    """Elevation band-based discretization."""

    ALL: Tuple[str, ...] = (LUMPED, POINT, ELEVATION)
    """All supported discretization methods."""

    @classmethod
    def is_valid(cls, method: str) -> bool:
        """Check if a discretization method is valid."""
        return method.lower() in cls.ALL


class DelineationMethods:
    """
    Constants for domain delineation methods.

    These define how the study domain boundary is determined.
    """

    DELINEATE = 'delineate'
    """Delineate watershed from DEM using pour point."""

    SUBSET = 'subset'
    """Subset from existing geospatial data."""

    LUMPED = 'lumped'
    """Use predefined lumped basin boundary."""

    ALL: Tuple[str, ...] = (DELINEATE, SUBSET, LUMPED)
    """All supported delineation methods."""

    @classmethod
    def is_valid(cls, method: str) -> bool:
        """Check if a delineation method is valid."""
        return method.lower() in cls.ALL
