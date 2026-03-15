# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CF-Intermediate Format (CFIF) for SYMFLUENCE.

This package provides a model-neutral, CF-compliant intermediate format
for forcing data. CFIF serves as the standard format between raw data
acquisition/preprocessing and model-specific adapters.

Benefits:
    - Model-agnostic: Not tied to any specific hydrological model
    - CF-compliant: Uses CF standard names and conventions
    - Extensible: Easy to add new variables as needed
    - Consistent: Single source of truth for variable definitions

Usage:
    >>> from symfluence.data.preprocessing.cfif import CFIF_VARIABLES, get_cfif_variable
    >>> var_info = get_cfif_variable('air_temperature')
    >>> print(var_info['units'])  # 'K'

Architecture:
    1. Raw data (ERA5, CONUS404, etc.) → Dataset handlers → CFIF format
    2. CFIF format → Model adapters → Model-specific format (SUMMA, HYPE, etc.)
"""

from .units import (
    UnitConverter,
    convert_humidity,
    convert_precipitation,
    convert_pressure,
    convert_radiation,
    convert_temperature,
    convert_wind,
)
from .variables import (
    CFIF_TO_SUMMA_MAPPING,
    CFIF_VARIABLES,
    SUMMA_TO_CFIF_MAPPING,
    get_cfif_standard_name,
    get_cfif_units,
    get_cfif_variable,
    normalize_to_cfif,
    validate_cfif_dataset,
)

__all__ = [
    # Variable definitions
    'CFIF_VARIABLES',
    'SUMMA_TO_CFIF_MAPPING',
    'CFIF_TO_SUMMA_MAPPING',
    'get_cfif_variable',
    'get_cfif_standard_name',
    'get_cfif_units',
    'normalize_to_cfif',
    'validate_cfif_dataset',
    # Unit conversions
    'UnitConverter',
    'convert_temperature',
    'convert_precipitation',
    'convert_pressure',
    'convert_radiation',
    'convert_humidity',
    'convert_wind',
]
