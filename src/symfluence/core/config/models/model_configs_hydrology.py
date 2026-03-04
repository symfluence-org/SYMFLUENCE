# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Hydrological model configuration classes."""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .base import FROZEN_CONFIG
from .model_config_types import SpatialModeType
from .model_configs_ml_fire import WMFireConfig


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
    basin_params_file: str = Field(default='basinParamInfo.txt', alias='SETTINGS_SUMMA_BASIN_PARAMS_FILE')
    local_params_file: str = Field(default='localParamInfo.txt', alias='SETTINGS_SUMMA_LOCAL_PARAMS_FILE')
    connect_hrus: bool = Field(default=True, alias='SETTINGS_SUMMA_CONNECT_HRUS')
    trialparam_n: int = Field(default=0, alias='SETTINGS_SUMMA_TRIALPARAM_N')
    trialparam_1: Optional[str] = Field(default=None, alias='SETTINGS_SUMMA_TRIALPARAM_1')
    use_parallel: bool = Field(default=False, alias='SETTINGS_SUMMA_USE_PARALLEL_SUMMA')
    cpus_per_task: int = Field(default=32, alias='SETTINGS_SUMMA_CPUS_PER_TASK', ge=1, le=256)
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
    init_grid_file: str = Field(default='coldState_glacSurfTopo.nc', alias='SETTINGS_SUMMA_INIT_GRID_FILE')
    attrib_grid_file: str = Field(default='attributes_glacBedTopo.nc', alias='SETTINGS_SUMMA_ATTRIB_GRID_FILE')


class FUSEConfig(BaseModel):
    """FUSE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='FUSE_INSTALL_PATH')
    exe: str = Field(default='fuse.exe', alias='FUSE_EXE')
    routing_integration: str = Field(default='default', alias='FUSE_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_FUSE_PATH')
    filemanager: str = Field(default='default', alias='SETTINGS_FUSE_FILEMANAGER')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='FUSE_SPATIAL_MODE')
    subcatchment_dim: str = Field(default='longitude', alias='FUSE_SUBCATCHMENT_DIM')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_FUSE')
    params_to_calibrate: str = Field(
        default='MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',
        alias='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
    )
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='FUSE_DECISION_OPTIONS')
    # Additional FUSE settings
    file_id: Optional[str] = Field(default=None, alias='FUSE_FILE_ID')
    n_elevation_bands: int = Field(default=1, alias='FUSE_N_ELEVATION_BANDS', ge=1)
    timeout: int = Field(default=3600, alias='FUSE_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    run_internal_calibration: bool = Field(default=True, alias='FUSE_RUN_INTERNAL_CALIBRATION')
    output_timestep_seconds: int = Field(default=86400, alias='FUSE_OUTPUT_TIMESTEP_SECONDS')
    snow_model: Optional[str] = Field(default=None, alias='FUSE_SNOW_MODEL')
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='FUSE_PARAM_BOUNDS')
    parameter_regionalization: str = Field(default='lumped', alias='PARAMETER_REGIONALIZATION')
    use_transfer_functions: bool = Field(default=False, alias='USE_TRANSFER_FUNCTIONS')
    transfer_function_coeff_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='TRANSFER_FUNCTION_COEFF_BOUNDS')
    solution_method: Optional[int] = Field(
        default=None, alias='FUSE_SOLUTION_METHOD',
        description='Numerical solver: 0=explicit Euler (fast), 1=implicit Euler (stable). Default: 0'
    )
    timestep_type: Optional[int] = Field(
        default=None, alias='FUSE_TIMESTEP_TYPE',
        description='Timestep control: 0=fixed, 1=adaptive. Default: 0'
    )


class GRConfig(BaseModel):
    """GR (GR4J/GR5J) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GR_INSTALL_PATH')
    exe: str = Field(default='GR.r', alias='GR_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='GR_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='GR_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_GR_PATH')
    control: str = Field(default='default', alias='SETTINGS_GR_CONTROL')
    params_to_calibrate: str = Field(
        default='X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff',
        alias='GR_PARAMS_TO_CALIBRATE'
    )
    # Fallback behavior control - default to False to prevent silent data corruption
    allow_dummy_observations: bool = Field(
        default=False,
        alias='GR_ALLOW_DUMMY_OBSERVATIONS',
        description='If True, use zero-filled dummy observations when no streamflow data found'
    )
    allow_default_area: bool = Field(
        default=False,
        alias='GR_ALLOW_DEFAULT_AREA',
        description='If True, use 1.0 km² default area when basin shapefile not found'
    )
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='GR_PARAM_BOUNDS')
    gr4j_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='GR4J_PARAM_BOUNDS')
    initial_params: str = Field(default='default', alias='GR_INITIAL_PARAMS')
    default_params: Optional[List[float]] = Field(default=None, alias='GR_DEFAULT_PARAMS')


class HBVConfig(BaseModel):
    """HBV-96 hydrological model configuration"""
    model_config = FROZEN_CONFIG

    spatial_mode: SpatialModeType = Field(default='auto', alias='HBV_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='HBV_ROUTING_INTEGRATION')
    backend: Literal['jax', 'numpy'] = Field(default='jax', alias='HBV_BACKEND')
    use_gpu: bool = Field(default=False, alias='HBV_USE_GPU')
    jit_compile: bool = Field(default=True, alias='HBV_JIT_COMPILE')
    warmup_days: int = Field(default=365, alias='HBV_WARMUP_DAYS', ge=0)
    timestep_hours: int = Field(default=24, alias='HBV_TIMESTEP_HOURS', ge=1, le=24)
    params_to_calibrate: str = Field(
        default='default',  # 'default' triggers use of all available HBV parameters
        alias='HBV_PARAMS_TO_CALIBRATE',
        description="Parameters to calibrate. Use 'default' for all parameters, or specify comma-separated list."
    )
    use_gradient_calibration: bool = Field(default=True, alias='HBV_USE_GRADIENT_CALIBRATION')
    calibration_metric: Literal['KGE', 'NSE'] = Field(default='KGE', alias='HBV_CALIBRATION_METRIC')
    # Initial state values
    initial_snow: float = Field(default=0.0, alias='HBV_INITIAL_SNOW', ge=0.0)
    initial_sm: float = Field(default=150.0, alias='HBV_INITIAL_SM', ge=0.0)
    initial_suz: float = Field(default=10.0, alias='HBV_INITIAL_SUZ', ge=0.0)
    initial_slz: float = Field(default=10.0, alias='HBV_INITIAL_SLZ', ge=0.0)
    # PET configuration
    pet_method: Literal['input', 'hamon', 'thornthwaite'] = Field(default='input', alias='HBV_PET_METHOD')
    latitude: Optional[float] = Field(default=None, alias='HBV_LATITUDE', ge=-90.0, le=90.0)
    allow_unit_heuristics: bool = Field(
        default=False,
        alias='HBV_ALLOW_UNIT_HEURISTICS',
        description='Allow magnitude-based unit detection for precip/PET when units are missing or ambiguous'
    )
    # Output configuration
    save_states: bool = Field(default=False, alias='HBV_SAVE_STATES')
    output_frequency: Literal['daily', 'timestep'] = Field(default='daily', alias='HBV_OUTPUT_FREQUENCY')
    # Default parameter values
    default_tt: float = Field(default=0.0, alias='HBV_DEFAULT_TT')
    default_cfmax: float = Field(default=3.5, alias='HBV_DEFAULT_CFMAX')
    default_sfcf: float = Field(default=0.9, alias='HBV_DEFAULT_SFCF')
    default_cfr: float = Field(default=0.05, alias='HBV_DEFAULT_CFR')
    default_cwh: float = Field(default=0.1, alias='HBV_DEFAULT_CWH')
    default_fc: float = Field(default=250.0, alias='HBV_DEFAULT_FC')
    default_lp: float = Field(default=0.7, alias='HBV_DEFAULT_LP')
    default_beta: float = Field(default=2.5, alias='HBV_DEFAULT_BETA')
    default_k0: float = Field(default=0.3, alias='HBV_DEFAULT_K0')
    default_k1: float = Field(default=0.1, alias='HBV_DEFAULT_K1')
    default_k2: float = Field(default=0.01, alias='HBV_DEFAULT_K2')
    default_uzl: float = Field(default=30.0, alias='HBV_DEFAULT_UZL')
    default_perc: float = Field(default=2.5, alias='HBV_DEFAULT_PERC')
    default_maxbas: float = Field(default=2.5, alias='HBV_DEFAULT_MAXBAS')


class HECHMSConfig(BaseModel):
    """HEC-HMS hydrological model configuration (native Python/JAX)"""
    model_config = FROZEN_CONFIG

    backend: Literal['jax', 'numpy'] = Field(default='jax', alias='HECHMS_BACKEND')
    warmup_days: int = Field(default=365, alias='HECHMS_WARMUP_DAYS', ge=0)
    params_to_calibrate: str = Field(
        default='default',
        alias='HECHMS_PARAMS_TO_CALIBRATE',
        description="Parameters to calibrate. 'default' for all, or comma-separated list."
    )
    calibration_metric: Literal['KGE', 'NSE'] = Field(default='KGE', alias='HECHMS_CALIBRATION_METRIC')
    pet_method: Literal['input', 'oudin', 'hamon'] = Field(default='input', alias='HECHMS_PET_METHOD')
    latitude: Optional[float] = Field(default=None, alias='HECHMS_LATITUDE', ge=-90.0, le=90.0)
    # Initial state
    initial_snow_swe: float = Field(default=0.0, alias='HECHMS_INITIAL_SNOW_SWE', ge=0.0)
    initial_gw_storage: float = Field(default=10.0, alias='HECHMS_INITIAL_GW_STORAGE', ge=0.0)
    # Default parameter values (14 params)
    default_px_temp: float = Field(default=1.0, alias='HECHMS_DEFAULT_PX_TEMP')
    default_base_temp: float = Field(default=0.0, alias='HECHMS_DEFAULT_BASE_TEMP')
    default_ati_meltrate_coeff: float = Field(default=0.98, alias='HECHMS_DEFAULT_ATI_MELTRATE_COEFF')
    default_meltrate_max: float = Field(default=5.0, alias='HECHMS_DEFAULT_MELTRATE_MAX')
    default_meltrate_min: float = Field(default=1.0, alias='HECHMS_DEFAULT_MELTRATE_MIN')
    default_cold_limit: float = Field(default=10.0, alias='HECHMS_DEFAULT_COLD_LIMIT')
    default_ati_cold_rate_coeff: float = Field(default=0.1, alias='HECHMS_DEFAULT_ATI_COLD_RATE_COEFF')
    default_water_capacity: float = Field(default=0.05, alias='HECHMS_DEFAULT_WATER_CAPACITY')
    default_cn: float = Field(default=65.0, alias='HECHMS_DEFAULT_CN')
    default_initial_abstraction_ratio: float = Field(default=0.2, alias='HECHMS_DEFAULT_INITIAL_ABSTRACTION_RATIO')
    default_tc: float = Field(default=3.0, alias='HECHMS_DEFAULT_TC')
    default_r_coeff: float = Field(default=5.0, alias='HECHMS_DEFAULT_R_COEFF')
    default_gw_storage_coeff: float = Field(default=30.0, alias='HECHMS_DEFAULT_GW_STORAGE_COEFF')
    default_deep_perc_fraction: float = Field(default=0.1, alias='HECHMS_DEFAULT_DEEP_PERC_FRACTION')


class TOPMODELConfig(BaseModel):
    """TOPMODEL (Beven & Kirkby 1979) hydrological model configuration (native Python/JAX)"""
    model_config = FROZEN_CONFIG

    backend: Literal['jax', 'numpy'] = Field(default='jax', alias='TOPMODEL_BACKEND')
    warmup_days: int = Field(default=365, alias='TOPMODEL_WARMUP_DAYS', ge=0)
    params_to_calibrate: str = Field(
        default='default',
        alias='TOPMODEL_PARAMS_TO_CALIBRATE',
        description="Parameters to calibrate. 'default' for all, or comma-separated list."
    )
    calibration_metric: Literal['KGE', 'NSE'] = Field(default='KGE', alias='TOPMODEL_CALIBRATION_METRIC')
    pet_method: Literal['input', 'oudin', 'hamon'] = Field(default='input', alias='TOPMODEL_PET_METHOD')
    latitude: Optional[float] = Field(default=None, alias='TOPMODEL_LATITUDE', ge=-90.0, le=90.0)
    # Default parameter values (11 params)
    default_m: float = Field(default=0.05, alias='TOPMODEL_DEFAULT_M')
    default_lnTe: float = Field(default=1.0, alias='TOPMODEL_DEFAULT_LNTE')
    default_Srmax: float = Field(default=0.05, alias='TOPMODEL_DEFAULT_SRMAX')
    default_Sr0: float = Field(default=0.01, alias='TOPMODEL_DEFAULT_SR0')
    default_td: float = Field(default=5.0, alias='TOPMODEL_DEFAULT_TD')
    default_k_route: float = Field(default=48.0, alias='TOPMODEL_DEFAULT_K_ROUTE')
    default_DDF: float = Field(default=3.5, alias='TOPMODEL_DEFAULT_DDF')
    default_T_melt: float = Field(default=0.0, alias='TOPMODEL_DEFAULT_T_MELT')
    default_T_snow: float = Field(default=1.0, alias='TOPMODEL_DEFAULT_T_SNOW')
    default_ti_std: float = Field(default=4.0, alias='TOPMODEL_DEFAULT_TI_STD')
    default_S0: float = Field(default=0.5, alias='TOPMODEL_DEFAULT_S0')


class HYPEConfig(BaseModel):
    """HYPE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='HYPE_INSTALL_PATH')
    exe: str = Field(default='hype', alias='HYPE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_HYPE_PATH')
    info_file: str = Field(default='info.txt', alias='SETTINGS_HYPE_INFO')
    params_to_calibrate: str = Field(
        default='ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp,wcwp,wcfc,wcep,srrcs',
        alias='HYPE_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='HYPE_SPINUP_DAYS')


class NGENConfig(BaseModel):
    """NGEN (Next Generation Water Resources Modeling Framework) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='NGEN_INSTALL_PATH')
    exe: str = Field(default='ngen', alias='NGEN_EXE')
    modules_to_calibrate: str = Field(default='CFE', alias='NGEN_MODULES_TO_CALIBRATE')
    cfe_params_to_calibrate: str = Field(
        default='maxsmc,satdk,bb,slop',
        alias='NGEN_CFE_PARAMS_TO_CALIBRATE'
    )
    noah_params_to_calibrate: str = Field(
        default='refkdt,slope,smcmax,dksat',
        alias='NGEN_NOAH_PARAMS_TO_CALIBRATE'
    )
    pet_params_to_calibrate: str = Field(
        default='wind_speed_measurement_height_m',
        alias='NGEN_PET_PARAMS_TO_CALIBRATE'
    )
    active_catchment_id: Optional[str] = Field(default=None, alias='NGEN_ACTIVE_CATCHMENT_ID')
    # Parameter bounds overrides (per-module)
    cfe_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_CFE_PARAM_BOUNDS')
    noah_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_NOAH_PARAM_BOUNDS')
    pet_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_PET_PARAM_BOUNDS')
    # Module selection (replaces individual ENABLE_* flags)
    modules_selected: str = Field(default='SLOTH,PET,CFE', alias='NGEN_MODULES_SELECTED')
    noah_et_fallback: str = Field(default='ETRAN', alias='NGEN_NOAH_ET_FALLBACK')

    @model_validator(mode='before')
    @classmethod
    def _migrate_enable_flags(cls, values: Any) -> Any:
        """Auto-migrate deprecated ENABLE_* flags to NGEN_MODULES_SELECTED."""
        if not isinstance(values, dict):
            return values

        # Check for any legacy ENABLE_* keys
        enable_keys = {
            'ENABLE_SLOTH': ('SLOTH', True),
            'ENABLE_PET': ('PET', True),
            'ENABLE_NOAH': ('NOAH', False),
            'ENABLE_CFE': ('CFE', True),
        }
        found_legacy = {k: v for k, v in enable_keys.items() if k in values}

        if not found_legacy:
            return values

        # Only migrate if NGEN_MODULES_SELECTED is not already explicitly set
        if 'NGEN_MODULES_SELECTED' in values or 'modules_selected' in values:
            # Remove stale legacy keys so they don't cause Pydantic errors
            for k in found_legacy:
                values.pop(k, None)
            return values

        # Build modules list from legacy flags
        modules = []
        for key, (mod_name, default_on) in enable_keys.items():
            raw = values.get(key, default_on)
            # Handle string booleans from YAML
            if isinstance(raw, str):
                enabled = raw.lower() in ('true', '1', 'yes')
            else:
                enabled = bool(raw)
            if enabled:
                modules.append(mod_name)

        values['NGEN_MODULES_SELECTED'] = ','.join(modules)

        # Remove legacy keys
        for k in found_legacy:
            values.pop(k, None)

        warnings.warn(
            "ENABLE_SLOTH/ENABLE_PET/ENABLE_NOAH/ENABLE_CFE are deprecated. "
            f"Use NGEN_MODULES_SELECTED: '{values['NGEN_MODULES_SELECTED']}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return values

    @model_validator(mode='after')
    def _validate_calibrate_subset(self) -> 'NGENConfig':
        """Ensure modules_to_calibrate is a subset of modules_selected."""
        selected = {m.strip().upper() for m in self.modules_selected.split(',') if m.strip()}
        calibrate = {m.strip().upper() for m in self.modules_to_calibrate.split(',') if m.strip()}
        not_selected = calibrate - selected
        if not_selected:
            raise ValueError(
                f"NGEN_MODULES_TO_CALIBRATE contains modules not in NGEN_MODULES_SELECTED: "
                f"{not_selected}. Either add them to NGEN_MODULES_SELECTED or remove them "
                f"from NGEN_MODULES_TO_CALIBRATE."
            )
        return self
    run_troute: bool = Field(default=True, alias='NGEN_RUN_TROUTE')


class MESHConfig(BaseModel):
    """MESH (Modélisation Environnementale-Surface Hydrology) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MESH_INSTALL_PATH')
    exe: str = Field(default='mesh.exe', alias='MESH_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='MESH_SPATIAL_MODE')
    settings_path: str = Field(default='default', alias='SETTINGS_MESH_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MESH')
    forcing_path: str = Field(default='default', alias='MESH_FORCING_PATH')
    forcing_vars: str = Field(default='default', alias='MESH_FORCING_VARS')
    forcing_units: str = Field(default='default', alias='MESH_FORCING_UNITS')
    forcing_to_units: str = Field(default='default', alias='MESH_FORCING_TO_UNITS')
    landcover_stats_path: str = Field(default='default', alias='MESH_LANDCOVER_STATS_PATH')
    landcover_stats_dir: str = Field(default='default', alias='MESH_LANDCOVER_STATS_DIR')
    landcover_stats_file: str = Field(default='default', alias='MESH_LANDCOVER_STATS_FILE')
    main_id: str = Field(default='default', alias='MESH_MAIN_ID')
    ds_main_id: str = Field(default='default', alias='MESH_DS_MAIN_ID')
    landcover_classes: str = Field(default='default', alias='MESH_LANDCOVER_CLASSES')
    ddb_vars: str = Field(default='default', alias='MESH_DDB_VARS')
    ddb_units: str = Field(default='default', alias='MESH_DDB_UNITS')
    ddb_to_units: str = Field(default='default', alias='MESH_DDB_TO_UNITS')
    ddb_min_values: str = Field(default='default', alias='MESH_DDB_MIN_VALUES')
    gru_dim: str = Field(default='default', alias='MESH_GRU_DIM')
    hru_dim: str = Field(default='default', alias='MESH_HRU_DIM')
    outlet_value: str = Field(default='default', alias='MESH_OUTLET_VALUE')
    # Additional MESH settings
    input_file: str = Field(default='default', alias='SETTINGS_MESH_INPUT')
    params_to_calibrate: str = Field(
        default='ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR',
        alias='MESH_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='MESH_SPINUP_DAYS')
    gru_min_total: float = Field(default=0.0, alias='MESH_GRU_MIN_TOTAL')
    # Lumped mode enforcement settings
    force_single_gru: bool = Field(default=True, alias='MESH_FORCE_SINGLE_GRU')
    apply_params_all_grus: bool = Field(default=True, alias='MESH_APPLY_PARAMS_ALL_GRUS')
    use_landcover_multipliers: bool = Field(default=True, alias='MESH_USE_LANDCOVER_MULTIPLIERS')
    enable_frozen_soil: bool = Field(default=True, alias='MESH_ENABLE_FROZEN_SOIL')
    daily_tolerance_days: int = Field(default=1, alias='MESH_DAILY_TOLERANCE_DAYS')



class RHESSysConfig(BaseModel):
    """RHESSys (Regional Hydro-Ecologic Simulation System) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='RHESSYS_INSTALL_PATH')
    exe: str = Field(default='rhessys', alias='RHESSYS_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_RHESSYS_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_RHESSYS')
    forcing_path: str = Field(default='default', alias='FORCING_RHESSYS_PATH')
    world_template: str = Field(default='world.template', alias='RHESSYS_WORLD_TEMPLATE')
    flow_template: str = Field(default='flow.template', alias='RHESSYS_FLOW_TEMPLATE')
    # LNA/TWI controls (optional; None disables caps)
    lna_area_cap_m2: Optional[float] = Field(default=None, alias='RHESSYS_LNA_AREA_CAP_M2')
    lna_min: Optional[float] = Field(default=None, alias='RHESSYS_LNA_MIN')
    lna_max: Optional[float] = Field(default=None, alias='RHESSYS_LNA_MAX')
    params_to_calibrate: str = Field(
        default=(
            'sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,porosity_decay,'
            'soil_depth,snow_melt_Tcoef,max_snow_temp,min_rain_temp,theta_mean_std_p1'
        ),
        alias='RHESSYS_PARAMS_TO_CALIBRATE'
    )
    skip_calibration: bool = Field(default=True, alias='RHESSYS_SKIP_CALIBRATION')
    # WMFire integration (wildfire spread module)
    use_wmfire: bool = Field(default=False, alias='RHESSYS_USE_WMFIRE')
    wmfire_install_path: str = Field(default='installs/wmfire/lib', alias='WMFIRE_INSTALL_PATH')
    wmfire_lib: str = Field(default='libwmfire.so', alias='WMFIRE_LIB')
    wmfire: Optional[WMFireConfig] = Field(default=None, description='Enhanced WMFire configuration')
    # Legacy VMFire aliases
    use_vmfire: bool = Field(default=False, alias='RHESSYS_USE_VMFIRE')
    vmfire_install_path: str = Field(default='installs/wmfire/lib', alias='VMFIRE_INSTALL_PATH')
    # Execution settings
    timeout: int = Field(default=7200, alias='RHESSYS_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    # Grow mode for Farquhar photosynthesis and transpiration (default True)
    use_grow_mode: bool = Field(default=True, alias='RHESSYS_USE_GROW_MODE')


class VICConfig(BaseModel):
    """VIC (Variable Infiltration Capacity) model configuration.

    VIC is a large-scale, semi-distributed hydrological model that solves
    full water and energy balances. It uses a grid-based structure and is
    typically applied to large river basins.

    Reference:
        Liang, X., D. P. Lettenmaier, E. F. Wood, and S. J. Burges, 1994:
        A simple hydrologically based model of land surface water and energy
        fluxes for general circulation models. J. Geophys. Res., 99(D7), 14415-14428.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='VIC_INSTALL_PATH')
    exe: str = Field(default='vic_image.exe', alias='VIC_EXE')
    driver: Literal['image', 'classic'] = Field(default='image', alias='VIC_DRIVER')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_VIC_PATH')
    global_param_file: str = Field(default='vic_global.txt', alias='VIC_GLOBAL_PARAM_FILE')
    domain_file: str = Field(default='vic_domain.nc', alias='VIC_DOMAIN_FILE')
    params_file: str = Field(default='vic_params.nc', alias='VIC_PARAMS_FILE')

    # Spatial mode
    spatial_mode: SpatialModeType = Field(default='auto', alias='VIC_SPATIAL_MODE')

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_VIC')
    output_prefix: str = Field(default='vic_output', alias='VIC_OUTPUT_PREFIX')

    # Calibration
    params_to_calibrate: Optional[str] = Field(
        default='infilt,Ds,Dsmax,Ws,c,depth1,depth2,depth3,expt,expt_increase,Ksat,Ksat_decay,Wcr_FRACT,Wpwp_ratio,snow_rough,max_snow_albedo,min_rain_temp,max_snow_temp,elev_offset',
        alias='VIC_PARAMS_TO_CALIBRATE'
    )

    # Model options
    full_energy: bool = Field(default=True, alias='VIC_FULL_ENERGY')
    frozen_soil: bool = Field(default=True, alias='VIC_FROZEN_SOIL')
    snow_band: bool = Field(default=False, alias='VIC_SNOW_BAND')
    n_snow_bands: int = Field(default=10, alias='VIC_N_SNOW_BANDS', ge=1, le=25)
    pfactor_per_km: float = Field(default=0.0005, alias='VIC_PFACTOR_PER_KM', ge=0.0, le=0.01)

    # Timing
    model_steps_per_day: int = Field(default=24, alias='VIC_STEPS_PER_DAY', ge=1, le=48)

    # Execution
    timeout: int = Field(default=7200, alias='VIC_TIMEOUT', ge=60, le=86400)


class CLMConfig(BaseModel):
    """CLM (Community Land Model / CTSM 5.x) configuration.

    CLM5 is the land component of CESM, providing comprehensive
    biogeophysics, biogeochemistry, hydrology, snow, and vegetation
    dynamics. It is the most physics-heavy LSM in the ensemble.

    Reference:
        Lawrence, D. M., et al. (2019): The Community Land Model version 5.
        JAMES, 11, 4245-4287.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='CLM_INSTALL_PATH')
    exe: str = Field(default='cesm.exe', alias='CLM_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_CLM_PATH')
    compset: str = Field(default='I2000Clm50SpGs', alias='CLM_COMPSET')
    params_file: str = Field(default='clm5_params.nc', alias='CLM_PARAMS_FILE')
    surfdata_file: str = Field(default='surfdata_clm.nc', alias='CLM_SURFDATA_FILE')
    domain_file: str = Field(default='domain.nc', alias='CLM_DOMAIN_FILE')

    # Spatial mode
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CLM_SPATIAL_MODE')

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CLM')
    hist_nhtfrq: int = Field(default=-24, alias='CLM_HIST_NHTFRQ')
    hist_mfilt: int = Field(default=365, alias='CLM_HIST_MFILT')

    # Calibration
    params_to_calibrate: Optional[str] = Field(
        default=None,
        alias='CLM_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='CLM_TIMEOUT', ge=60, le=86400)
    warmup_days: int = Field(default=365, alias='CLM_WARMUP_DAYS', ge=0, le=3650)



class SWATConfig(BaseModel):
    """SWAT (Soil and Water Assessment Tool) model configuration.

    SWAT is a river basin scale model developed by USDA-ARS to predict
    the impact of land management on water, sediment, and agricultural
    chemical yields.

    Reference:
        Arnold, J.G., et al. (1998): Large area hydrologic modeling and
        assessment Part I: Model development. JAWRA, 34(1), 73-89.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='SWAT_INSTALL_PATH')
    exe: str = Field(default='swat_rel.exe', alias='SWAT_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_SWAT_PATH')
    txtinout_dir: str = Field(default='TxtInOut', alias='SWAT_TXTINOUT_DIR')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='SWAT_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_SWAT')
    params_to_calibrate: str = Field(
        default='CN2,ALPHA_BF,GW_DELAY,GWQMN,GW_REVAP,ESCO,SOL_AWC,SOL_K,SURLAG,SFTMP,SMTMP,SMFMX,SMFMN,TIMP',
        alias='SWAT_PARAMS_TO_CALIBRATE'
    )
    warmup_years: int = Field(default=2, alias='SWAT_WARMUP_YEARS', ge=0, le=10)
    timeout: int = Field(default=3600, alias='SWAT_TIMEOUT', ge=60, le=86400)
    plaps: float = Field(default=0.0, alias='SWAT_PLAPS')
    tlaps: float = Field(default=0.0, alias='SWAT_TLAPS')


class MHMConfig(BaseModel):
    """mHM (mesoscale Hydrological Model) configuration.

    mHM is a spatially distributed hydrological model developed at the
    Helmholtz Centre for Environmental Research (UFZ). It uses multiscale
    parameter regionalization (MPR) for parameter transfer.

    Reference:
        Samaniego, L., et al. (2010): Multiscale parameter regionalization
        of a grid-based hydrologic model at the mesoscale. Water Resources
        Research, 46, W05523.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MHM_INSTALL_PATH')
    exe: str = Field(default='mhm', alias='MHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MHM_PATH')
    namelist_file: str = Field(default='mhm.nml', alias='MHM_NAMELIST_FILE')
    routing_namelist: str = Field(default='mrm.nml', alias='MHM_ROUTING_NAMELIST')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MHM')
    params_to_calibrate: str = Field(
        default='canopyInterceptionFactor,snowTreshholdTemperature,degreeDayFactor_forest,degreeDayFactor_pervious,PTF_Ks_constant,interflowRecession_slope,rechargeCoefficient,GeoParam(1,:),infiltrationShapeFactor,rootFractionCoefficient_pervious,interflowStorageCapacityFactor,slowInterflowRecession_Ks,muskingumTravelTime_constant,orgMatterContent_forest',
        alias='MHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='MHM_TIMEOUT', ge=60, le=86400)


class CRHMConfig(BaseModel):
    """CRHM (Cold Regions Hydrological Model) configuration.

    CRHM is a physically-based, object-oriented hydrological model
    designed specifically for cold-region processes including blowing
    snow, energy-balance snowmelt, and frozen soil infiltration.

    Reference:
        Pomeroy, J.W., et al. (2007): The Cold Regions Hydrological Model:
        a platform for basing process representation and model structure on
        physical evidence. Hydrological Processes, 21(19), 2650-2667.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='CRHM_INSTALL_PATH')
    exe: str = Field(default='crhm', alias='CRHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_CRHM_PATH')
    project_file: str = Field(default='model.prj', alias='CRHM_PROJECT_FILE')
    observation_file: str = Field(default='forcing.obs', alias='CRHM_OBSERVATION_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CRHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CRHM')
    params_to_calibrate: str = Field(
        default='basin_area,Ht,Asnow,inhibit_evap,Ksat,soil_rechr_max,soil_moist_max,soil_gw_K,Sdmax,fetch',
        alias='CRHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='CRHM_TIMEOUT', ge=60, le=86400)


class WRFHydroConfig(BaseModel):
    """WRF-Hydro (NCAR) coupled atmosphere-hydrology model configuration.

    WRF-Hydro is NCAR's community hydrological modeling system and forms
    the backbone of the US National Water Model. It couples the Noah-MP
    land surface model with terrain-following routing.

    Reference:
        Gochis, D.J., et al. (2020): The WRF-Hydro modeling system technical
        description, (Version 5.1.1). NCAR Technical Note.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WRFHYDRO_INSTALL_PATH')
    exe: str = Field(default='wrf_hydro.exe', alias='WRFHYDRO_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WRFHYDRO_PATH')
    namelist_file: str = Field(default='namelist.hrldas', alias='WRFHYDRO_NAMELIST_FILE')
    hydro_namelist: str = Field(default='hydro.namelist', alias='WRFHYDRO_HYDRO_NAMELIST')
    spatial_mode: SpatialModeType = Field(default='distributed', alias='WRFHYDRO_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WRFHYDRO')
    params_to_calibrate: str = Field(
        default='REFKDT,SLOPE,OVROUGHRTFAC,RETDEPRTFAC,LKSATFAC,BEXP,DKSAT,SMCMAX',
        alias='WRFHYDRO_PARAMS_TO_CALIBRATE'
    )
    lsm: str = Field(default='noahmp', alias='WRFHYDRO_LSM')
    routing_option: str = Field(default='gridded', alias='WRFHYDRO_ROUTING_OPTION')
    channel_routing: str = Field(default='diffusive_wave', alias='WRFHYDRO_CHANNEL_ROUTING')
    restart_frequency: str = Field(default='monthly', alias='WRFHYDRO_RESTART_FREQUENCY')
    timeout: int = Field(default=7200, alias='WRFHYDRO_TIMEOUT', ge=60, le=86400)


class PRMSConfig(BaseModel):
    """PRMS (Precipitation-Runoff Modeling System) configuration.

    PRMS is a deterministic, distributed-parameter, physical-process
    watershed model developed by the USGS for simulating the effects
    of precipitation, climate, and land use on streamflow.

    Reference:
        Markstrom, S.L., et al. (2015): PRMS-IV, the Precipitation-Runoff
        Modeling System, Version 4. USGS Techniques and Methods 6-B7.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='PRMS_INSTALL_PATH')
    exe: str = Field(default='prms', alias='PRMS_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_PRMS_PATH')
    control_file: str = Field(default='control.dat', alias='PRMS_CONTROL_FILE')
    parameter_file: str = Field(default='params.dat', alias='PRMS_PARAMETER_FILE')
    data_file: str = Field(default='data.dat', alias='PRMS_DATA_FILE')
    spatial_mode: SpatialModeType = Field(default='semi_distributed', alias='PRMS_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PRMS')
    params_to_calibrate: str = Field(
        default='soil_moist_max,soil_rechr_max,tmax_allrain,tmax_allsnow,hru_percent_imperv,carea_max,smidx_coef,slowcoef_lin,gwflow_coef,ssr2gw_rate',
        alias='PRMS_PARAMS_TO_CALIBRATE'
    )
    model_mode: str = Field(default='DAILY', alias='PRMS_MODEL_MODE')
    timeout: int = Field(default=3600, alias='PRMS_TIMEOUT', ge=60, le=86400)


class GSFLOWConfig(BaseModel):
    """GSFLOW (coupled PRMS + MODFLOW-NWT) configuration.

    GSFLOW is a USGS coupled groundwater–surface-water model that integrates
    PRMS (surface/soil) with MODFLOW-NWT (saturated zone) via SFR and UZF
    packages for bidirectional exchange.

    Reference:
        Markstrom, S.L., et al. (2008): GSFLOW—Coupled Ground-Water and
        Surface-Water Flow Model Based on the Integration of the
        Precipitation-Runoff Modeling System (PRMS) and the Modular
        Ground-Water Flow Model (MODFLOW-2005). USGS Techniques and
        Methods 6-D1.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GSFLOW_INSTALL_PATH')
    exe: str = Field(default='gsflow', alias='GSFLOW_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_GSFLOW_PATH')
    control_file: str = Field(default='gsflow.control', alias='GSFLOW_CONTROL_FILE')
    parameter_file: str = Field(default='params.dat', alias='GSFLOW_PARAMETER_FILE')
    modflow_nam_file: str = Field(default='modflow.nam', alias='GSFLOW_MODFLOW_NAM_FILE')
    spatial_mode: SpatialModeType = Field(default='semi_distributed', alias='GSFLOW_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_GSFLOW')
    params_to_calibrate: str = Field(
        default='soil_moist_max,soil_rechr_max,ssr2gw_rate,gwflow_coef,gw_seep_coef,K,SY,slowcoef_lin,carea_max,smidx_coef',
        alias='GSFLOW_PARAMS_TO_CALIBRATE'
    )
    gsflow_mode: str = Field(default='COUPLED', alias='GSFLOW_MODE')
    timeout: int = Field(default=7200, alias='GSFLOW_TIMEOUT', ge=60, le=86400)


class WATFLOODConfig(BaseModel):
    """WATFLOOD (Kouwen) distributed flood forecasting model configuration.

    WATFLOOD is a physically-based, distributed hydrological model using
    Grouped Response Units (GRUs) on a regular grid with internal channel
    routing. It is optimized for flood forecasting with simplified energy
    balance requiring only precipitation and temperature forcing.

    Reference:
        Kouwen, N. (2018): WATFLOOD/WATROUTE Hydrological Model Routing
        & Flood Forecasting System. University of Waterloo.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WATFLOOD_INSTALL_PATH')
    exe: str = Field(default='watflood', alias='WATFLOOD_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WATFLOOD_PATH')
    shed_file: str = Field(default='bow_shd.r2c', alias='WATFLOOD_SHED_FILE')
    par_file: str = Field(default='bow.par', alias='WATFLOOD_PAR_FILE')
    event_file: str = Field(default='event.evt', alias='WATFLOOD_EVENT_FILE')
    spatial_mode: SpatialModeType = Field(default='distributed', alias='WATFLOOD_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WATFLOOD')
    params_to_calibrate: str = Field(
        default='FLZCOEF,PWR,R2N,AK,AKF,REESSION,RETN,AK2,AK2FS,R3,DS,FPET,FTALL,FM,BASE,SUBLIM_FACTOR',
        alias='WATFLOOD_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='WATFLOOD_TIMEOUT', ge=60, le=86400)


class WflowConfig(BaseModel):
    """Wflow (wflow_sbm) distributed hydrological model configuration.

    Wflow is a distributed hydrological model developed by Deltares,
    implemented in Julia. The wflow_sbm concept uses topography-driven
    subsurface flow with a kinematic wave for overland and river routing.

    Reference:
        van Verseveld et al. (2024): Wflow_sbm v0.7.3, a spatially
        distributed hydrological model. Geosci. Model Dev., 17, 3021-3043.
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WFLOW_INSTALL_PATH')
    exe: str = Field(default='wflow_cli', alias='WFLOW_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WFLOW_PATH')
    config_file: str = Field(default='wflow_sbm.toml', alias='WFLOW_CONFIG_FILE')
    staticmaps_file: str = Field(default='wflow_staticmaps.nc', alias='WFLOW_STATICMAPS_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='WFLOW_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WFLOW')
    output_file: str = Field(default='output.nc', alias='WFLOW_OUTPUT_FILE')
    pet_method: str = Field(default='oudin', alias='WFLOW_PET_METHOD')
    params_to_calibrate: str = Field(
        default='KsatVer,f,SoilThickness,InfiltCapPath,RootingDepth,KsatHorFrac,n_river,PathFrac,thetaS,thetaR,Cfmax,TT,TTI,TTM,WHC',
        alias='WFLOW_PARAMS_TO_CALIBRATE',
    )
    timeout: int = Field(default=7200, alias='WFLOW_TIMEOUT', ge=60, le=86400)


__all__ = [
    'SUMMAConfig',
    'FUSEConfig',
    'GRConfig',
    'HBVConfig',
    'HECHMSConfig',
    'TOPMODELConfig',
    'HYPEConfig',
    'NGENConfig',
    'MESHConfig',
    'RHESSysConfig',
    'VICConfig',
    'CLMConfig',
    'SWATConfig',
    'MHMConfig',
    'CRHMConfig',
    'WRFHydroConfig',
    'PRMSConfig',
    'GSFLOWConfig',
    'WATFLOODConfig',
    'WflowConfig',
]
