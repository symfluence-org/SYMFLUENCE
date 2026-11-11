# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA initialization presets.

This module registers SUMMA-specific presets with the unified registry,
keeping model-specific configuration within the model directory.
"""
from __future__ import annotations

from symfluence.core.registries import R


@R.presets.add('bow-river')
def bow_river_preset():
    """Bow River at Banff lumped SUMMA setup with ERA5 forcing and DDS calibration."""
    return {
        'description': 'Bow River at Banff (WSC 05BB001) lumped SUMMA setup with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'DOMAIN_NAME': 'Bow_at_Banff',
            'EXPERIMENT_ID': 'run_1',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Temporal settings
            'EXPERIMENT_TIME_START': '2004-01-01 01:00',
            'EXPERIMENT_TIME_END': '2007-12-31 23:00',
            'SPINUP_PERIOD': '2004-01-01, 2005-09-30',
            'CALIBRATION_PERIOD': '2005-10-01, 2006-09-30',
            'EVALUATION_PERIOD': '2006-10-01, 2007-09-30',

            # Geospatial settings
            'POUR_POINT_COORDS': '51.1722/-115.5717',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'ROUTING_DELINEATION': 'lumped',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 1000,
            'LUMPED_WATERSHED_METHOD': 'TauDEM',
            'DEM_SOURCE': 'copdem90',

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'SUMMA_INSTALL_PATH': 'default',
            'SUMMA_EXE': 'summa_sundials.exe',
            'SETTINGS_SUMMA_PATH': 'default',
            'SETTINGS_SUMMA_FILEMANAGER': 'fileManager.txt',
            'SETTINGS_SUMMA_CONNECT_HRUS': 'yes',
            'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': False,
            'EXPERIMENT_OUTPUT_SUMMA': 'default',
            'ROUTING_MODEL': 'none',

            # Calibration parameters
            'PARAMS_TO_CALIBRATE': 'k_soil,theta_sat,aquiferBaseflowExp,aquiferBaseflowRate,qSurfScale,summerLAI,frozenPrecipMultip,Fcapil,tempCritRain,heightCanopyTop,heightCanopyBottom,windReductionParam,vGn_n',
            'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaScale,routingGammaShape',

            # Evaluation settings — WSC (Water Survey of Canada) station
            'STREAMFLOW_DATA_PROVIDER': 'WSC',
            'DOWNLOAD_WSC_DATA': True,
            'STATION_ID': '05BB001',
            'SIM_REACH_ID': 1,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 100,
            'OPTIMIZATION_METRIC': 'KGE',
            'CALIBRATION_TIMESTEP': 'hourly',
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
    }


@R.presets.add('summa-basic')
def summa_basic_preset():
    """Generic SUMMA distributed setup with ERA5 forcing."""
    return {
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
    }
