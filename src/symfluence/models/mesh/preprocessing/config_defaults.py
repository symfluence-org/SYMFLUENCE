# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH Configuration Defaults

Default variable mappings, units, and parameter values for MESH model.
"""

from typing import Any, Dict, Union

# ---------------------------------------------------------------------------
# Shared spatial-mode predicates
# ---------------------------------------------------------------------------
# These centralise the scattered force_single_gru / elevation-band guards
# that were previously duplicated across drainage_database, preprocessor,
# and gru_count_manager.  Import them anywhere MESH preprocessing needs to
# ask "should I collapse GRUs?" or "is this elevation-band mode?".


def _cfg_get(config: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """Get a config value from either a flat dict or a typed config object."""
    if isinstance(config, dict):
        return config.get(key, default)
    # Typed config: try known nested paths
    _TYPED_PATHS = {
        'SUB_GRID_DISCRETIZATION': lambda c: c.model.mesh.sub_grid_discretization,
        'MESH_FORCE_SINGLE_GRU': lambda c: c.model.mesh.force_single_gru,
        'MESH_SPATIAL_MODE': lambda c: c.model.mesh.spatial_mode,
        'DOMAIN_DEFINITION_METHOD': lambda c: c.domain.definition_method,
    }
    accessor = _TYPED_PATHS.get(key)
    if accessor:
        try:
            val = accessor(config)
            if val is not None:
                return val
        except (AttributeError, KeyError, TypeError):
            pass
    # Fall back to config_dict if available
    cd = config.to_dict(flatten=True) if hasattr(config, 'to_dict') else (config if isinstance(config, dict) else {})
    if cd:
        return cd.get(key, default)
    return default


def is_elevation_band_mode(config_dict: Union[Dict[str, Any], Any]) -> bool:
    """Check if elevation band discretization is enabled.

    When True, GRU collapse and zero-fraction trimming must be skipped
    to preserve the multi-subbasin elevation band structure.
    """
    sub_grid = _cfg_get(config_dict, 'SUB_GRID_DISCRETIZATION', 'GRUS')
    return isinstance(sub_grid, str) and sub_grid.lower() == 'elevation'


def should_force_single_gru(config_dict: Union[Dict[str, Any], Any]) -> bool:
    """Determine whether to collapse GRUs to a single class.

    Decision logic (evaluated in order):
    1. Always *False* for elevation-band discretization.
    2. If ``MESH_FORCE_SINGLE_GRU`` is explicitly set (and ≠ "default"),
       honour the boolean value.
    3. Otherwise auto-enable for lumped / point spatial modes.
    """
    if is_elevation_band_mode(config_dict):
        return False

    raw = _cfg_get(config_dict, 'MESH_FORCE_SINGLE_GRU', None)
    if raw is not None and raw != 'default':
        return _as_bool_value(raw)

    spatial_mode = str(_cfg_get(config_dict, 'MESH_SPATIAL_MODE', 'auto')).lower()
    domain_method = str(_cfg_get(config_dict, 'DOMAIN_DEFINITION_METHOD', '')).lower()
    return spatial_mode in ('lumped', 'point') or domain_method in ('lumped', 'point')


def _as_bool_value(value: Any, default: bool = False) -> bool:
    """Parse a truthy / falsey config value."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ('true', '1', 'yes', 'y', 'on'):
            return True
        if val in ('false', '0', 'no', 'n', 'off'):
            return False
    return default


class MESHConfigDefaults:
    """
    Provides default configuration values for MESH preprocessing.

    Contains mappings for forcing variables, landcover classes,
    drainage database variables, and GRU parameters.
    """

    # meshflow expects: standard_name -> actual_file_variable_name
    FORCING_VARS: Dict[str, str] = {
        "air_pressure": "surface_air_pressure",
        "specific_humidity": "specific_humidity",
        "air_temperature": "air_temperature",
        "wind_speed": "wind_speed",
        "precipitation": "precipitation_flux",
        "shortwave_radiation": "surface_downwelling_shortwave_flux",
        "longwave_radiation": "surface_downwelling_longwave_flux",
    }

    # Units from source data
    FORCING_UNITS: Dict[str, str] = {
        "air_pressure": 'Pa',
        "specific_humidity": 'kg/kg',
        "air_temperature": 'K',
        "wind_speed": 'm/s',
        "precipitation": 'm/s',
        "shortwave_radiation": 'W/m^2',
        "longwave_radiation": 'W/m^2',
    }

    # Target units for MESH
    FORCING_TO_UNITS: Dict[str, str] = {
        "air_pressure": 'Pa',
        "specific_humidity": 'kg/kg',
        "air_temperature": 'K',
        "wind_speed": 'm/s',
        "precipitation": 'mm/s',
        "shortwave_radiation": 'W/m^2',
        "longwave_radiation": 'W/m^2',
    }

    # NALCMS 2020 landcover classes (integer keys for meshflow compatibility)
    LANDCOVER_CLASSES: Dict[int, str] = {
        1: 'Temperate or sub-polar needleleaf forest',
        2: 'Sub-polar taiga needleleaf forest',
        3: 'Tropical or sub-tropical broadleaf evergreen forest',
        4: 'Tropical or sub-tropical broadleaf deciduous forest',
        5: 'Temperate or sub-polar broadleaf deciduous forest',
        6: 'Mixed forest',
        7: 'Tropical or sub-tropical shrubland',
        8: 'Temperate or sub-polar shrubland',
        9: 'Tropical or sub-tropical grassland',
        10: 'Temperate or sub-polar grassland',
        11: 'Sub-polar or polar shrubland-lichen-moss',
        12: 'Sub-polar or polar grassland-lichen-moss',
        13: 'Sub-polar or polar barren-lichen-moss',
        14: 'Wetland',
        15: 'Cropland',
        16: 'Barren lands',
        17: 'Urban',
        18: 'Water',
        19: 'Snow and Ice',
    }

    # ddb_vars maps standard names -> input shapefile column names
    DDB_VARS: Dict[str, str] = {
        'river_slope': 'Slope',
        'river_length': 'Length',
        'river_class': 'strmOrder',
    }

    DDB_UNITS: Dict[str, str] = {
        'river_slope': 'm/m',
        'river_length': 'm',
        'rank': 'dimensionless',
        'next': 'dimensionless',
        'gru': 'dimensionless',
        'subbasin_area': 'm^2',
    }

    DDB_MIN_VALUES: Dict[str, float] = {
        'river_slope': 1e-6,
        'river_length': 1e-3,
        'subbasin_area': 1e-3,
    }

    # Full NALCMS to CLASS parameter type mapping
    FULL_GRU_MAPPING: Dict[int, str] = {
        0: 'grass',       # Unknown -> conservative default (grassland)
        1: 'needleleaf',  # Temperate or sub-polar needleleaf forest
        2: 'needleleaf',  # Sub-polar taiga needleleaf forest
        3: 'broadleaf',   # Tropical or sub-tropical broadleaf evergreen forest
        4: 'broadleaf',   # Tropical or sub-tropical broadleaf deciduous forest
        5: 'broadleaf',   # Temperate or sub-polar broadleaf deciduous forest
        6: 'broadleaf',   # Mixed forest
        7: 'grass',       # Tropical or sub-tropical shrubland
        8: 'grass',       # Temperate or sub-polar shrubland
        9: 'grass',       # Tropical or sub-tropical grassland
        10: 'grass',      # Temperate or sub-polar grassland
        11: 'grass',      # Sub-polar or polar shrubland-lichen-moss
        12: 'grass',      # Sub-polar or polar grassland-lichen-moss
        13: 'barrenland', # Sub-polar or polar barren-lichen-moss
        14: 'wetland',    # Wetland (distinct from open water)
        15: 'crops',      # Cropland
        16: 'barrenland', # Barren lands
        17: 'urban',      # Urban
        18: 'water',      # Water
        19: 'barrenland', # Snow and Ice (glacier surfaces, not open water)
    }

    # MESH variable name mappings
    MESH_VAR_NAMES: Dict[str, str] = {
        'air_pressure': 'PRES',
        'specific_humidity': 'QA',
        'air_temperature': 'TA',
        'wind_speed': 'UV',
        'precipitation': 'PRE',
        'shortwave_radiation': 'FSIN',
        'longwave_radiation': 'FLIN',
    }

    # MESH 1.5 variable mapping to file names
    VAR_TO_FILE: Dict[str, str] = {
        'FSIN': 'basin_shortwave.nc',
        'FLIN': 'basin_longwave.nc',
        'PRES': 'basin_pres.nc',
        'TA': 'basin_temperature.nc',
        'QA': 'basin_humidity.nc',
        'UV': 'basin_wind.nc',
        'PRE': 'basin_rain.nc',
    }

    @classmethod
    def get_var_long_name(cls, var: str) -> str:
        """Get long name for MESH variable."""
        names = {
            'FSIN': 'downward shortwave radiation',
            'FLIN': 'downward longwave radiation',
            'PRES': 'air pressure',
            'TA': 'air temperature',
            'QA': 'specific humidity',
            'UV': 'wind speed',
            'PRE': 'precipitation rate',
        }
        return names.get(var, var)

    @classmethod
    def get_var_units(cls, var: str) -> str:
        """Get units for MESH variable."""
        units = {
            'FSIN': 'W m-2',
            'FLIN': 'W m-2',
            'PRES': 'Pa',
            'TA': 'K',
            'QA': 'kg kg-1',
            'UV': 'm s-1',
            'PRE': 'kg m-2 s-1',
        }
        return units.get(var, '1')

    # Unit conversion factors: (source_unit, target_unit) -> multiplier
    # MESH expects: PRE in kg m-2 s-1 (equivalent to mm/s for water)
    UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
        'precipitation': {
            'm/s_to_kg m-2 s-1': 1000.0,  # m/s -> mm/s -> kg/m²/s (water density ~1000 kg/m³)
            'mm/s_to_kg m-2 s-1': 1.0,    # mm/s ≈ kg/m²/s for water
            'mm/h_to_kg m-2 s-1': 1.0/3600.0,  # mm/hr -> mm/s
            'm/h_to_kg m-2 s-1': 1000.0/3600.0,
        },
        'air_temperature': {
            'C_to_K': 273.15,  # additive, not multiplicative
            'K_to_K': 1.0,
        },
    }

    @classmethod
    def convert_forcing_data(cls, data, standard_name: str, source_units: str, logger=None) -> tuple:
        """
        Convert forcing data from source units to MESH-expected units.

        Args:
            data: numpy array of values
            standard_name: standard variable name (e.g., 'precipitation')
            source_units: units of source data
            logger: optional logger for warnings

        Returns:
            tuple: (converted_data, target_units)
        """

        mesh_var = cls.MESH_VAR_NAMES.get(standard_name, standard_name)
        target_units = cls.get_var_units(mesh_var)

        # Normalize source units string
        source_units_norm = source_units.replace(' ', '').replace('^', '').lower()

        if standard_name == 'precipitation':
            # Precipitation: convert to kg m-2 s-1
            if source_units_norm in ['m/s', 'ms-1', 'ms^-1']:
                converted = data * 1000.0
                if logger:
                    logger.info("Converted precipitation: m/s -> kg m-2 s-1 (×1000)")
            elif source_units_norm in ['mm/s', 'mms-1', 'kgm-2s-1', 'kg/m2/s']:
                converted = data  # Already correct
            elif source_units_norm in ['mm/h', 'mm/hr', 'mmh-1']:
                converted = data / 3600.0
                if logger:
                    logger.info("Converted precipitation: mm/h -> kg m-2 s-1 (÷3600)")
            elif source_units_norm in ['mm/d', 'mm/day', 'mmd-1']:
                converted = data / 86400.0
                if logger:
                    logger.info("Converted precipitation: mm/d -> kg m-2 s-1 (÷86400)")
            else:
                converted = data
                if logger:
                    logger.warning(
                        f"Unknown precipitation units '{source_units}', assuming already in kg m-2 s-1"
                    )
            return converted, target_units

        elif standard_name == 'air_temperature':
            # Temperature: convert to K
            if source_units_norm in ['c', 'degc', 'celsius', '°c']:
                converted = data + 273.15
                if logger:
                    logger.info("Converted temperature: °C -> K (+273.15)")
            elif source_units_norm in ['k', 'kelvin']:
                converted = data  # Already in K
            else:
                converted = data
                if logger:
                    logger.warning(f"Unknown temperature units '{source_units}', assuming K")
            return converted, target_units

        # For other variables, check if units need conversion
        # Most others (Pa, kg/kg, m/s, W/m²) are typically already correct
        return data, target_units

    @classmethod
    def get_recommended_spinup_days(cls, latitude: float = None, elevation: float = None) -> int:
        """
        Get climate-appropriate spinup period for MESH simulations.

        Spinup requirements vary by climate:
        - Arctic/permafrost (lat >= 60°): 3 years - deep soil thermal equilibration
        - Alpine (elev >= 2500m): 3 years - glacier/snowpack equilibration
        - Boreal (50° <= lat < 60°): 2 years - seasonal snow equilibration
        - Temperate (lat < 50°): 1 year - adequate for most soil moisture

        Args:
            latitude: Domain centroid latitude in degrees (optional)
            elevation: Mean domain elevation in meters (optional)

        Returns:
            Recommended spinup period in days
        """
        # Check for high-altitude alpine conditions first
        if elevation is not None and elevation >= 2500:
            return 1095  # 3 years

        # Then check latitude-based climate zones
        if latitude is not None:
            abs_lat = abs(latitude)
            if abs_lat >= 60:
                return 1095  # 3 years - Arctic/permafrost
            elif abs_lat >= 50:
                return 730   # 2 years - Boreal
            else:
                return 365   # 1 year - Temperate

        # Default: conservative 2-year spinup
        return 730

    @classmethod
    def get_gru_mapping_for_classes(cls, detected_classes: list) -> Dict[int, str]:
        """
        Get GRU mapping filtered to only include detected classes.

        Args:
            detected_classes: List of GRU class numbers present in data

        Returns:
            Filtered GRU mapping dictionary
        """
        if detected_classes:
            return {k: cls.FULL_GRU_MAPPING.get(k, 'needleleaf')
                    for k in detected_classes}
        return cls.FULL_GRU_MAPPING

    @classmethod
    def get_default_settings(
        cls,
        forcing_start_date: str,
        sim_start_date: str,
        sim_end_date: str,
        gru_mapping: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Build default meshflow settings dictionary.

        Args:
            forcing_start_date: Start date for forcing data
            sim_start_date: Simulation start date
            sim_end_date: Simulation end date
            gru_mapping: GRU class to type mapping

        Returns:
            Settings dictionary for meshflow
        """
        return {
            'core': {
                'forcing_files': 'single',
                'forcing_start_date': forcing_start_date,
                'simulation_start_date': sim_start_date,
                'simulation_end_date': sim_end_date,
                'forcing_time_zone': 'UTC',
                'output_path': 'results',
            },
            'class_params': {
                'measurement_heights': {
                    'wind_speed': 10.0,
                    'specific_humidity': 2.0,
                    'air_temperature': 2.0,
                    'roughness_length': 50.0,
                },
                'copyright': {
                    'author': 'University of Calgary',
                    'location': 'SYMFLUENCE',
                },
                'grus': gru_mapping,
            },
            'hydrology_params': {
                'routing': [
                    {
                        'r2n': 0.4,
                        'r1n': 0.02,
                        'pwr': 2.37,
                        'flz': 0.001,
                    },
                ],
                'hydrology': {},
            },
            'run_options': {
                'flags': {
                    'etc': {
                        'RUNMODE': 'runrte',
                    },
                },
            },
        }
