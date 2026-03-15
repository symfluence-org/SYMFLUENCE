# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CF-Intermediate Format (CFIF) Variable Definitions.

This module defines the standard variable names, units, and attributes
for the model-neutral intermediate format used in SYMFLUENCE.

Variable Naming Convention:
    - Uses CF standard names with underscores (e.g., 'air_temperature')
    - Short enough for practical use, long enough for clarity
    - Maps directly to CF standard_name attributes

Standard Units (SI-based, CF-compliant):
    - Temperature: K (Kelvin)
    - Precipitation: kg m-2 s-1 (mass flux)
    - Pressure: Pa (Pascals)
    - Radiation: W m-2 (Watts per square meter)
    - Humidity: kg kg-1 (specific) or % (relative)
    - Wind: m s-1 (meters per second)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Core CFIF variable definitions
# Each entry contains: cf_standard_name, units, long_name, description
CFIF_VARIABLES: Dict[str, Dict[str, str]] = {
    # Temperature
    'air_temperature': {
        'cf_standard_name': 'air_temperature',
        'units': 'K',
        'long_name': 'Near-surface air temperature',
        'description': 'Temperature of air at 2m above surface',
    },

    # Precipitation
    'precipitation_flux': {
        'cf_standard_name': 'precipitation_flux',
        'units': 'kg m-2 s-1',
        'long_name': 'Precipitation rate',
        'description': 'Total precipitation rate (liquid + solid)',
    },

    # Radiation
    'surface_downwelling_shortwave_flux': {
        'cf_standard_name': 'surface_downwelling_shortwave_flux_in_air',
        'units': 'W m-2',
        'long_name': 'Downward shortwave radiation',
        'description': 'Incoming solar radiation at surface',
    },
    'surface_downwelling_longwave_flux': {
        'cf_standard_name': 'surface_downwelling_longwave_flux_in_air',
        'units': 'W m-2',
        'long_name': 'Downward longwave radiation',
        'description': 'Incoming thermal radiation at surface',
    },

    # Humidity
    'specific_humidity': {
        'cf_standard_name': 'specific_humidity',
        'units': 'kg kg-1',
        'long_name': 'Specific humidity',
        'description': 'Mass of water vapor per mass of moist air',
    },
    'relative_humidity': {
        'cf_standard_name': 'relative_humidity',
        'units': '%',
        'long_name': 'Relative humidity',
        'description': 'Ratio of actual to saturation vapor pressure',
    },

    # Wind
    'wind_speed': {
        'cf_standard_name': 'wind_speed',
        'units': 'm s-1',
        'long_name': 'Wind speed',
        'description': 'Magnitude of wind velocity at 10m',
    },
    'eastward_wind': {
        'cf_standard_name': 'eastward_wind',
        'units': 'm s-1',
        'long_name': 'Eastward wind component',
        'description': 'U-component of wind at 10m',
    },
    'northward_wind': {
        'cf_standard_name': 'northward_wind',
        'units': 'm s-1',
        'long_name': 'Northward wind component',
        'description': 'V-component of wind at 10m',
    },

    # Pressure
    'surface_air_pressure': {
        'cf_standard_name': 'surface_air_pressure',
        'units': 'Pa',
        'long_name': 'Surface air pressure',
        'description': 'Atmospheric pressure at surface',
    },

    # Additional variables for specific models
    'snowfall_flux': {
        'cf_standard_name': 'snowfall_flux',
        'units': 'kg m-2 s-1',
        'long_name': 'Snowfall rate',
        'description': 'Solid precipitation rate',
    },
    'rainfall_flux': {
        'cf_standard_name': 'rainfall_flux',
        'units': 'kg m-2 s-1',
        'long_name': 'Rainfall rate',
        'description': 'Liquid precipitation rate',
    },
    'air_temperature_min': {
        'cf_standard_name': 'air_temperature',
        'units': 'K',
        'long_name': 'Minimum air temperature',
        'description': 'Daily minimum temperature',
        'cell_methods': 'time: minimum',
    },
    'air_temperature_max': {
        'cf_standard_name': 'air_temperature',
        'units': 'K',
        'long_name': 'Maximum air temperature',
        'description': 'Daily maximum temperature',
        'cell_methods': 'time: maximum',
    },
    'potential_evapotranspiration': {
        'cf_standard_name': 'water_potential_evaporation_flux',
        'units': 'kg m-2 s-1',
        'long_name': 'Potential evapotranspiration',
        'description': 'Reference evapotranspiration rate',
    },
}


# Mapping from legacy SUMMA variable names to CFIF names
SUMMA_TO_CFIF_MAPPING: Dict[str, str] = {
    'airtemp': 'air_temperature',
    'pptrate': 'precipitation_flux',
    'SWRadAtm': 'surface_downwelling_shortwave_flux',
    'LWRadAtm': 'surface_downwelling_longwave_flux',
    'spechum': 'specific_humidity',
    'relhum': 'relative_humidity',
    'windspd': 'wind_speed',
    'windspd_u': 'eastward_wind',
    'windspd_v': 'northward_wind',
    'airpres': 'surface_air_pressure',
}

# Reverse mapping: CFIF names to SUMMA names
CFIF_TO_SUMMA_MAPPING: Dict[str, str] = {
    v: k for k, v in SUMMA_TO_CFIF_MAPPING.items()
}


def normalize_to_cfif(ds):
    """
    Rename any legacy SUMMA-style variable names in a dataset to CFIF names.

    This is a backward-compatibility helper for reading pre-existing forcing
    files that use the legacy naming convention. Only renames variables that
    have a known mapping and whose CFIF counterpart is not already present.

    Args:
        ds: xarray Dataset (may contain SUMMA-style or CFIF variable names)

    Returns:
        Dataset with CFIF variable names (unchanged if already CFIF-named)
    """
    renames = {k: v for k, v in SUMMA_TO_CFIF_MAPPING.items()
               if k in ds and v not in ds}
    return ds.rename(renames) if renames else ds


def get_cfif_variable(name: str) -> Optional[Dict[str, str]]:
    """
    Get variable definition by CFIF name.

    Args:
        name: CFIF variable name (e.g., 'air_temperature')

    Returns:
        Dict with variable attributes, or None if not found
    """
    return CFIF_VARIABLES.get(name)


def get_cfif_standard_name(name: str) -> Optional[str]:
    """
    Get CF standard_name for a CFIF variable.

    Args:
        name: CFIF variable name

    Returns:
        CF standard_name string, or None if not found
    """
    var_info = get_cfif_variable(name)
    return var_info.get('cf_standard_name') if var_info else None


def get_cfif_units(name: str) -> Optional[str]:
    """
    Get standard units for a CFIF variable.

    Args:
        name: CFIF variable name

    Returns:
        Units string, or None if not found
    """
    var_info = get_cfif_variable(name)
    return var_info.get('units') if var_info else None


def validate_cfif_dataset(ds, required_vars: Optional[List[str]] = None) -> List[str]:
    """
    Validate that a dataset conforms to CFIF conventions.

    Args:
        ds: xarray Dataset to validate
        required_vars: List of required CFIF variable names.
                      If None, validates against all present CFIF variables.

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    # Check required variables
    if required_vars:
        for var in required_vars:
            if var not in ds.data_vars:
                issues.append(f"Missing required variable: {var}")

    # Validate present CFIF variables
    for var in ds.data_vars:
        if var in CFIF_VARIABLES:
            expected = CFIF_VARIABLES[var]

            # Check units
            if 'units' in ds[var].attrs:
                if ds[var].attrs['units'] != expected['units']:
                    issues.append(
                        f"Variable {var}: expected units '{expected['units']}', "
                        f"got '{ds[var].attrs['units']}'"
                    )
            else:
                issues.append(f"Variable {var}: missing 'units' attribute")

    return issues
