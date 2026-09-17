# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for SUMMA.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, PositiveInt

from symfluence.core.config.models.base import FROZEN_CONFIG


class SUMMAConfig(BaseModel):
    """SUMMA hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='SUMMA_INSTALL_PATH')
    exe: str = Field(default='summa_sundials.exe', alias='SUMMA_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_SUMMA_PATH')
    filemanager: str = Field(default='fileManager.txt', alias='SETTINGS_SUMMA_FILEMANAGER')
    forcing_list: str = Field(default='forcingFileList.txt', alias='SETTINGS_SUMMA_FORCING_LIST')
    coldstate: str = Field(default='coldState.nc', alias='SETTINGS_SUMMA_COLDSTATE')
    trialparams: str = Field(default='trialParams.nc', alias='SETTINGS_SUMMA_TRIALPARAMS')
    attributes: str = Field(default='attributes.nc', alias='SETTINGS_SUMMA_ATTRIBUTES')
    output: str = Field(default='outputControl.txt', alias='SETTINGS_SUMMA_OUTPUT')
    additional_outputs: Dict[str, PositiveInt] = Field(
        default_factory=dict, alias='SUMMA_ADDITIONAL_OUTPUTS',
        description='Additional SUMMA output variables mapped to output frequency codes (1 = timestep, 24 = daily).')
    basin_params_file: str = Field(default='basinParamInfo.txt', alias='SETTINGS_SUMMA_BASIN_PARAMS_FILE')
    local_params_file: str = Field(default='localParamInfo.txt', alias='SETTINGS_SUMMA_LOCAL_PARAMS_FILE')
    connect_hrus: bool = Field(default=True, alias='SETTINGS_SUMMA_CONNECT_HRUS')
    trialparam_n: int = Field(default=0, alias='SETTINGS_SUMMA_TRIALPARAM_N')
    trialparam_1: Optional[str] = Field(default=None, alias='SETTINGS_SUMMA_TRIALPARAM_1')
    use_parallel: bool = Field(default=False, alias='SETTINGS_SUMMA_USE_PARALLEL_SUMMA')
    parallel_backend: Literal['slurm', 'local'] = Field(
        default='slurm',
        alias='SETTINGS_SUMMA_PARALLEL_BACKEND'
    )
    cpus_per_task: int = Field(
        default=32, alias='SETTINGS_SUMMA_CPUS_PER_TASK', ge=1, le=256
    )
    time_limit: str = Field(default='01:00:00', alias='SETTINGS_SUMMA_TIME_LIMIT')
    mem: Union[int, str] = Field(default='5G', alias='SETTINGS_SUMMA_MEM')  # SLURM-style memory spec like "12G"
    gru_count: int = Field(default=85, alias='SETTINGS_SUMMA_GRU_COUNT')
    gru_per_job: int = Field(default=5, alias='SETTINGS_SUMMA_GRU_PER_JOB')
    parallel_path: str = Field(default='default', alias='SETTINGS_SUMMA_PARALLEL_PATH')
    parallel_exe: str = Field(default='summa_actors.exe', alias='SETTINGS_SUMMA_PARALLEL_EXE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_SUMMA')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_SUMMA')
    params_to_calibrate: str = Field(
        default='k_soil,aquiferBaseflowRate,albedoMax,snowfrz_scale',
        alias='PARAMS_TO_CALIBRATE'
    )
    basin_params_to_calibrate: str = Field(
        default='routingGammaShape,routingGammaScale',
        alias='BASIN_PARAMS_TO_CALIBRATE'
    )
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='SUMMA_DECISION_OPTIONS')
    calibrate_depth: bool = Field(default=False, alias='CALIBRATE_DEPTH')
    depth_total_mult_bounds: Optional[List[float]] = Field(default=None, alias='DEPTH_TOTAL_MULT_BOUNDS')
    depth_shape_factor_bounds: Optional[List[float]] = Field(default=None, alias='DEPTH_SHAPE_FACTOR_BOUNDS')
    # Glacier-related settings
    glacier_mode: bool = Field(default=False, alias='SETTINGS_SUMMA_GLACIER_MODE')
    glacier_attributes: str = Field(default='attributes_glac.nc', alias='SETTINGS_SUMMA_GLACIER_ATTRIBUTES')
    glacier_coldstate: str = Field(default='coldState_glac.nc', alias='SETTINGS_SUMMA_GLACIER_COLDSTATE')
    # Execution settings
    timeout: int = Field(default=7200, alias='SUMMA_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    backup_settings: str = Field(default='no', alias='EXPERIMENT_BACKUP_SETTINGS')
    monitor_slurm_job: bool = Field(default=True, alias='MONITOR_SLURM_JOB')
    soilprofile: str = Field(default='FA', alias='SETTINGS_SUMMA_SOILPROFILE')
    init_matric_head: float = Field(default=-1.0, alias='SUMMA_INIT_MATRIC_HEAD')
    # Regionalization settings
    parameter_regionalization: str = Field(default='lumped', alias='PARAMETER_REGIONALIZATION')
    transfer_function_attributes_path: Optional[str] = Field(default=None, alias='TRANSFER_FUNCTION_ATTRIBUTES')
    transfer_function_param_config: Optional[Dict[str, Any]] = Field(default=None, alias='TRANSFER_FUNCTION_PARAM_CONFIG')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
SUMMAConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_SUMMA_PATH': 'settings_path',
    'SETTINGS_SUMMA_FILEMANAGER': 'filemanager',
    'SETTINGS_SUMMA_FORCING_LIST': 'forcing_list',
    'SETTINGS_SUMMA_COLDSTATE': 'coldstate',
    'SETTINGS_SUMMA_TRIALPARAMS': 'trialparams',
    'SETTINGS_SUMMA_ATTRIBUTES': 'attributes',
    'SETTINGS_SUMMA_OUTPUT': 'output',
    'SETTINGS_SUMMA_BASIN_PARAMS_FILE': 'basin_params_file',
    'SETTINGS_SUMMA_LOCAL_PARAMS_FILE': 'local_params_file',
    'SETTINGS_SUMMA_CONNECT_HRUS': 'connect_hrus',
    'SETTINGS_SUMMA_TRIALPARAM_N': 'trialparam_n',
    'SETTINGS_SUMMA_TRIALPARAM_1': 'trialparam_1',
    'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': 'use_parallel',
    'SETTINGS_SUMMA_PARALLEL_BACKEND': 'parallel_backend',
    'SETTINGS_SUMMA_CPUS_PER_TASK': 'cpus_per_task',
    'SETTINGS_SUMMA_TIME_LIMIT': 'time_limit',
    'SETTINGS_SUMMA_MEM': 'mem',
    'SETTINGS_SUMMA_GRU_COUNT': 'gru_count',
    'SETTINGS_SUMMA_GRU_PER_JOB': 'gru_per_job',
    'SETTINGS_SUMMA_PARALLEL_PATH': 'parallel_path',
    'SETTINGS_SUMMA_PARALLEL_EXE': 'parallel_exe',
    'SETTINGS_SUMMA_GLACIER_MODE': 'glacier_mode',
    'SETTINGS_SUMMA_GLACIER_ATTRIBUTES': 'glacier_attributes',
    'SETTINGS_SUMMA_GLACIER_COLDSTATE': 'glacier_coldstate',
    'SETTINGS_SUMMA_SOILPROFILE': 'soilprofile',
}
