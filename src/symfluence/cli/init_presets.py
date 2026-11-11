# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Preset definitions for SYMFLUENCE --init command.

This module defines templates for common SYMFLUENCE configurations that users can
use as starting points for their projects. Each preset includes model-specific
settings and sensible defaults.

Architecture:
    Model-specific presets are now defined in their respective model directories
    (e.g., models/summa/init_preset.py, models/fuse/init_preset.py) and registered
    with the PresetRegistry. This module maintains backward compatibility by
    aggregating presets from both the registry and legacy definitions.
"""
from __future__ import annotations

from symfluence.core.registries import R

# Legacy preset definitions (kept for backward compatibility)
# New presets should be defined in their model directories using PresetRegistry
_LEGACY_PRESETS = {
    'fuse-provo': {
        'description': 'FUSE model for Provo River, Utah with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'DOMAIN_NAME': 'provo_river',
            'EXPERIMENT_ID': 'run_1',
            'EXPERIMENT_TIME_START': '1999-10-01 00:00',
            'EXPERIMENT_TIME_END': '2002-09-30 23:00',
            'CALIBRATION_PERIOD': '2000-10-01, 2001-09-30',
            'EVALUATION_PERIOD': '2001-10-01, 2002-09-30',
            'SPINUP_PERIOD': '1999-10-01, 2000-09-30',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Geospatial settings
            'POUR_POINT_COORDS': '40.5577278126503/-111.168783841314',
            'BOUNDING_BOX_COORDS': '41/-111.74291666666666/40.0/-110.6208333333333',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'ROUTING_DELINEATION': 'lumped',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 2500,
            'LUMPED_WATERSHED_METHOD': 'TauDEM',

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FUSE_SPATIAL_MODE': 'lumped',
            'FUSE_EXE': 'fuse.exe',
            'FUSE_INSTALL_PATH': 'default',
            'SETTINGS_FUSE_PATH': 'default',
            'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
            'EXPERIMENT_OUTPUT_FUSE': 'default',
            'ROUTING_MODEL': 'none',

            # FUSE calibration parameters
            'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',

            # Evaluation settings
            'STREAMFLOW_DATA_PROVIDER': 'USGS',
            'DOWNLOAD_USGS_DATA': True,
            'STATION_ID': '10154200',
            'SIM_REACH_ID': 1,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 200,
            'OPTIMIZATION_METRIC': 'KGE',
            'DDS_R': 0.2,
            'RANDOM_SEED': 42,
        },
        'fuse_decisions': {
            'RFERR': ['multiplc_e'],
            'ARCH1': ['tension1_1'],
            'ARCH2': ['fixedsiz_2'],
            'QSURF': ['arno_x_vic'],
            'QPERC': ['perc_w2sat'],
            'ESOIL': ['rootweight'],
            'QINTF': ['intflwnone'],
            'Q_TDH': ['rout_gamma'],
            'SNOWM': ['temp_index']
        }
    },

    'summa-basic': {
        'description': 'Generic SUMMA distributed setup with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'EXPERIMENT_ID': 'run_1',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Geospatial settings
            'DOMAIN_DEFINITION_METHOD': 'delineate',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'ROUTING_DELINEATION': 'distributed',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 1000,
            'ELEVATION_BAND_SIZE': 200,
            'MIN_HRU_SIZE': 0,

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'ROUTING_MODEL': 'mizuRoute',
            'SUMMA_INSTALL_PATH': 'default',
            'SUMMA_EXE': 'summa_sundials.exe',
            'SETTINGS_SUMMA_PATH': 'default',
            'SETTINGS_SUMMA_FILEMANAGER': 'fileManager.txt',
            'SETTINGS_SUMMA_CONNECT_HRUS': 'yes',
            'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': False,
            'EXPERIMENT_OUTPUT_SUMMA': 'default',

            # mizuRoute settings
            'MIZUROUTE_INSTALL_PATH': 'default',
            'MIZUROUTE_EXE': 'mizuRoute.exe',
            'SETTINGS_MIZU_PATH': 'default',
            'SETTINGS_MIZU_WITHIN_BASIN': 0,
            'SETTINGS_MIZU_ROUTING_DT': 3600,
            'SETTINGS_MIZU_NEEDS_REMAP': 'no',
            'EXPERIMENT_OUTPUT_MIZUROUTE': 'default',

            # Calibration parameters
            'PARAMS_TO_CALIBRATE': 'k_soil,theta_sat,aquiferBaseflowExp,aquiferBaseflowRate',
            'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaScale,routingGammaShape',

            # Evaluation settings
            'DOWNLOAD_USGS_DATA': True,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 200,
            'OPTIMIZATION_METRIC': 'KGE',
            'DDS_R': 0.2,
            'RANDOM_SEED': 42,
        },
        'summa_decisions': {
            'snowIncept': ['lightSnow'],
            'compaction': ['consettl'],
            'snowLayers': ['CLM_2010'],
            'alb_method': ['conDecay'],
            'thCondSnow': ['tyen1965']
        }
    },

    'fuse-basic': {
        'description': 'Generic FUSE lumped setup with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'EXPERIMENT_ID': 'run_1',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Geospatial settings
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'ROUTING_DELINEATION': 'lumped',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 1000,
            'LUMPED_WATERSHED_METHOD': 'TauDEM',

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FUSE_SPATIAL_MODE': 'lumped',
            'FUSE_EXE': 'fuse.exe',
            'FUSE_INSTALL_PATH': 'default',
            'SETTINGS_FUSE_PATH': 'default',
            'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
            'EXPERIMENT_OUTPUT_FUSE': 'default',
            'ROUTING_MODEL': 'none',

            # FUSE calibration parameters
            'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1',

            # Evaluation settings
            'DOWNLOAD_USGS_DATA': True,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 200,
            'OPTIMIZATION_METRIC': 'KGE',
            'DDS_R': 0.2,
            'RANDOM_SEED': 42,
        },
        'fuse_decisions': {
            'RFERR': ['multiplc_e'],
            'ARCH1': ['tension1_1'],
            'ARCH2': ['fixedsiz_2'],
            'QSURF': ['arno_x_vic'],
            'QPERC': ['perc_w2sat'],
            'ESOIL': ['rootweight'],
            'QINTF': ['intflwnone'],
            'Q_TDH': ['rout_gamma'],
            'SNOWM': ['temp_index']
        }
    }
}

# Backward compatibility: expose PRESETS as the merged result
# This allows existing code that accesses init_presets.PRESETS to continue working
def _get_merged_presets():
    """Get all presets from both registry and legacy definitions."""
    # Ensure declared preset modules are imported to trigger registration
    R.presets.load_modules()

    # Start with registry presets (from unified registry)
    merged = {}
    for k in R.presets.keys():
        val = R.presets.get(k)
        # Handle both callable loaders and direct dict presets
        merged[k] = val() if callable(val) else val

    # Add legacy presets that aren't in the registry (for backward compat)
    for name, preset in _LEGACY_PRESETS.items():
        if name not in merged:
            merged[name] = preset

    return merged


# For backward compatibility, PRESETS is still available at module level
# Note: This is computed lazily via load_presets() for better performance
# Code that accesses init_presets.PRESETS directly should use load_presets() instead
PRESETS = _LEGACY_PRESETS  # Start with legacy, full merge happens via load_presets()


def load_presets():
    """
    Load and return preset dictionary.

    This function aggregates presets from:
    1. Model-specific presets registered via PresetRegistry
    2. Legacy presets defined in this module

    Returns:
        dict: Dictionary of all available presets
    """
    return _get_merged_presets()


def get_preset(name):
    """
    Get specific preset by name.

    Checks both the PresetRegistry (model-specific presets) and
    legacy presets for backward compatibility.

    Args:
        name (str): Preset name

    Returns:
        dict: Preset configuration

    Raises:
        ValueError: If preset name is unknown
    """
    presets = _get_merged_presets()
    # Normalize: lowercase, underscores to hyphens (e.g. bow_river -> bow-river)
    normalized = name.lower().replace('_', '-')
    if normalized in presets:
        return presets[normalized]
    if name in presets:
        return presets[name]
    available = ', '.join(sorted(presets.keys()))
    raise ValueError(
        f"Unknown preset: {name}. Available presets: {available}"
    )


def list_preset_names():
    """
    Return list of available preset names.

    Includes both registry-based and legacy presets.

    Returns:
        list: List of preset names
    """
    return sorted(_get_merged_presets().keys())


def validate_preset(preset):
    """
    Validate preset structure.

    Args:
        preset (dict): Preset to validate

    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []

    # Check required keys
    required_keys = ['description', 'base_template', 'settings']
    for key in required_keys:
        if key not in preset:
            errors.append(f"Missing required key: {key}")

    # Check settings is a dict
    if 'settings' in preset and not isinstance(preset['settings'], dict):
        errors.append("'settings' must be a dictionary")

    # Check model-specific decisions if present
    if 'fuse_decisions' in preset and not isinstance(preset['fuse_decisions'], dict):
        errors.append("'fuse_decisions' must be a dictionary")

    if 'summa_decisions' in preset and not isinstance(preset['summa_decisions'], dict):
        errors.append("'summa_decisions' must be a dictionary")

    return (len(errors) == 0, errors)
