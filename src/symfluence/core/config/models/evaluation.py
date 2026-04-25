# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Evaluation configuration models.

Contains configuration classes for observation data sources:
StreamflowConfig, SNOTELConfig, FluxNetConfig, USGSGWConfig, SMAPConfig,
GRACEConfig, MODISSnowConfig, AttributesConfig, and the parent EvaluationConfig.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG

# Supported merge strategies for multi-source data
MergeStrategyType = Literal['max', 'min', 'mean', 'priority']


class StreamflowConfig(BaseModel):
    """Streamflow observation data settings"""
    model_config = FROZEN_CONFIG

    data_provider: Optional[str] = Field(default=None, alias='STREAMFLOW_DATA_PROVIDER')
    download_usgs: bool = Field(default=False, alias='DOWNLOAD_USGS_DATA')
    download_wsc: bool = Field(default=False, alias='DOWNLOAD_WSC_DATA')
    station_id: Optional[str] = Field(default=None, alias='STATION_ID')
    raw_path: str = Field(default='default', alias='STREAMFLOW_RAW_PATH')
    raw_name: str = Field(default='default', alias='STREAMFLOW_RAW_NAME')
    processed_path: str = Field(default='default', alias='STREAMFLOW_PROCESSED_PATH')
    hydat_path: str = Field(default='default', alias='HYDAT_PATH')


class SNOTELConfig(BaseModel):
    """SNOTEL observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_SNOTEL')
    station: Optional[str] = Field(default=None, alias='SNOTEL_STATION')
    path: Optional[str] = Field(default=None, alias='SNOTEL_PATH')


class FluxNetConfig(BaseModel):
    """FluxNet observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_FLUXNET')
    station: Optional[str] = Field(default=None, alias='FLUXNET_STATION')
    path: Optional[str] = Field(default=None, alias='FLUXNET_PATH')


class USGSGWConfig(BaseModel):
    """USGS groundwater observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_USGS_GW')
    station: Optional[str] = Field(default=None, alias='USGS_STATION')


class SMAPConfig(BaseModel):
    """SMAP soil moisture observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_SMAP')
    product: str = Field(default='SPL4SMGP', alias='SMAP_PRODUCT')
    path: str = Field(default='default', alias='SMAP_PATH')
    max_granules: Optional[int] = Field(default=None, alias='SMAP_MAX_GRANULES')
    use_opendap: bool = Field(default=False, alias='SMAP_USE_OPENDAP')
    surface_depth_m: float = Field(default=0.05, alias='SMAP_SURFACE_DEPTH_M')
    rootzone_depth_m: float = Field(default=1.0, alias='SMAP_ROOTZONE_DEPTH_M')


class ISMNConfig(BaseModel):
    """ISMN soil moisture observation data settings"""
    model_config = FROZEN_CONFIG

    # Note: download flag uses DataConfig.download_ismn (alias='DOWNLOAD_ISMN')
    # This field is for internal use only, no alias to avoid conflict
    download: bool = Field(default=False)
    path: str = Field(default='default', alias='ISMN_PATH')
    api_base: str = Field(default='https://ismn.earth/dataviewer', alias='ISMN_API_BASE')
    metadata_url: Optional[str] = Field(default='https://ismn.earth/static/dataviewer/network_station_details.json', alias='ISMN_METADATA_URL')
    variable_list_url: Optional[str] = Field(default=None, alias='ISMN_VARIABLE_LIST_URL')
    data_url_template: Optional[str] = Field(default=None, alias='ISMN_DATA_URL_TEMPLATE')
    max_stations: int = Field(default=3, alias='ISMN_MAX_STATIONS')
    search_radius_km: Optional[float] = Field(default=None, alias='ISMN_SEARCH_RADIUS_KM')
    target_depth_m: float = Field(default=0.05, alias='ISMN_TARGET_DEPTH_M')
    temporal_aggregation: str = Field(default='daily_mean', alias='ISMN_TEMPORAL_AGGREGATION')


class GRACEConfig(BaseModel):
    """GRACE terrestrial water storage observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_GRACE')
    product: str = Field(default='RL06', alias='GRACE_PRODUCT')
    path: str = Field(default='default', alias='GRACE_PATH')
    data_dir: str = Field(default='default', alias='GRACE_DATA_DIR')


class MODISSnowConfig(BaseModel):
    """MODIS snow cover observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_MODIS_SNOW')
    product: str = Field(default='MOD10A1.061', alias='MODIS_SNOW_PRODUCT')
    path: str = Field(default='default', alias='MODIS_SNOW_PATH')
    data_dir: str = Field(default='default', alias='MODIS_SNOW_DIR')
    min_pixels: int = Field(default=100, alias='MODIS_MIN_PIXELS')

    # Merged SCA settings (MOD10A1 + MYD10A1)
    merge: bool = Field(default=True, alias='MODIS_SCA_MERGE')
    products: list = Field(default=['MOD10A1.061', 'MYD10A1.061'], alias='MODIS_SCA_PRODUCTS')
    merge_strategy: MergeStrategyType = Field(default='max', alias='MODIS_SCA_MERGE_STRATEGY')
    cloud_filter: bool = Field(default=True, alias='MODIS_SCA_CLOUD_FILTER')
    min_valid_ratio: float = Field(default=0.1, alias='MODIS_SCA_MIN_VALID_RATIO')
    normalize: bool = Field(default=True, alias='MODIS_SCA_NORMALIZE')
    use_catchment_mask: bool = Field(default=False, alias='MODIS_SCA_USE_CATCHMENT_MASK')


class MODISETConfig(BaseModel):
    """MODIS evapotranspiration (MOD16) observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_MODIS_ET')
    product: str = Field(default='MOD16A2.061', alias='MODIS_ET_PRODUCT')
    path: str = Field(default='default', alias='MODIS_ET_PATH')
    data_dir: str = Field(default='default', alias='MOD16_ET_DIR')


class AttributesConfig(BaseModel):
    """Catchment attributes data settings"""
    model_config = FROZEN_CONFIG

    data_dir: str = Field(default='default', alias='ATTRIBUTES_DATA_DIR')
    soilgrids_path: str = Field(default='default', alias='ATTRIBUTES_SOILGRIDS_PATH')
    pelletier_path: str = Field(default='default', alias='ATTRIBUTES_PELLETIER_PATH')
    merit_path: str = Field(default='default', alias='ATTRIBUTES_MERIT_PATH')
    modis_path: str = Field(default='default', alias='ATTRIBUTES_MODIS_PATH')
    glclu_path: str = Field(default='default', alias='ATTRIBUTES_GLCLU_PATH')
    forest_height_path: str = Field(default='default', alias='ATTRIBUTES_FOREST_HEIGHT_PATH')
    worldclim_path: str = Field(default='default', alias='ATTRIBUTES_WORLDCLIM_PATH')
    glim_path: str = Field(default='default', alias='ATTRIBUTES_GLIM_PATH')
    groundwater_path: str = Field(default='default', alias='ATTRIBUTES_GROUNDWATER_PATH')
    streamflow_path: str = Field(default='default', alias='ATTRIBUTES_STREAMFLOW_PATH')
    glwd_path: str = Field(default='default', alias='ATTRIBUTES_GLWD_PATH')
    hydrolakes_path: str = Field(default='default', alias='ATTRIBUTES_HYDROLAKES_PATH')
    output_dir: str = Field(default='default', alias='ATTRIBUTES_OUTPUT_DIR')


class SMHIConfig(BaseModel):
    """SMHI (Swedish Meteorological and Hydrological Institute) streamflow data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_SMHI_DATA')
    station_id: Optional[str] = Field(default=None, alias='SMHI_STATION_ID')
    path: str = Field(default='default', alias='SMHI_PATH')


class LAMAHICEConfig(BaseModel):
    """LamaH-Ice (Iceland basins) streamflow data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_LAMAH_ICE_DATA')
    path: Optional[str] = Field(default=None, alias='LAMAH_ICE_PATH')
    station_id: Optional[str] = Field(default=None, alias='LAMAH_ICE_STATION_ID')
    domain_id: Optional[Union[str, int]] = Field(default=None, alias='LAMAH_ICE_DOMAIN_ID')


class GlacierConfig(BaseModel):
    """Glacier observation data settings"""
    model_config = FROZEN_CONFIG

    download: bool = Field(default=False, alias='DOWNLOAD_GLACIER_DATA')
    path: str = Field(default='default', alias='GLACIER_PATH')
    source: str = Field(default='RGI', alias='GLACIER_SOURCE')


class TWSEvalConfig(BaseModel):
    """TWS (Total Water Storage) evaluator settings"""
    model_config = FROZEN_CONFIG

    grace_column: str = Field(default='grace_jpl_anomaly', alias='TWS_GRACE_COLUMN')
    anomaly_baseline: str = Field(default='overlap', alias='TWS_ANOMALY_BASELINE')
    unit_conversion: float = Field(default=1.0, alias='TWS_UNIT_CONVERSION')
    detrend: bool = Field(default=False, alias='TWS_DETREND')
    scale_to_obs: bool = Field(default=False, alias='TWS_SCALE_TO_OBS')
    storage_components: str = Field(default='', alias='TWS_STORAGE_COMPONENTS')
    obs_path: Optional[str] = Field(default=None, alias='TWS_OBS_PATH')


class ETEvalConfig(BaseModel):
    """ET (Evapotranspiration) evaluator settings"""
    model_config = FROZEN_CONFIG

    obs_source: str = Field(default='', alias='ET_OBS_SOURCE')
    temporal_aggregation: str = Field(default='daily_mean', alias='ET_TEMPORAL_AGGREGATION')
    use_quality_control: bool = Field(default=True, alias='ET_USE_QUALITY_CONTROL')
    max_quality_flag: int = Field(default=2, alias='ET_MAX_QUALITY_FLAG')
    modis_max_qc: int = Field(default=0, alias='ET_MODIS_MAX_QC')
    gleam_max_relative_uncertainty: float = Field(default=0.5, alias='ET_GLEAM_MAX_RELATIVE_UNCERTAINTY')
    obs_path: Optional[str] = Field(default=None, alias='ET_OBS_PATH')


class EvaluationConfig(BaseModel):
    """Evaluation data and analysis configuration"""
    model_config = FROZEN_CONFIG

    evaluation_data: Optional[List[str]] = Field(default=None, alias='EVALUATION_DATA')
    analyses: Optional[List[str]] = Field(default=None, alias='ANALYSES')
    sim_reach_id: Optional[int] = Field(default=None, alias='SIM_REACH_ID')
    evaluation_variable: str = Field(default='', alias='EVALUATION_VARIABLE')
    spinup_years: int = Field(default=0, alias='EVALUATION_SPINUP_YEARS')

    # Observation data sources
    streamflow: Optional[StreamflowConfig] = Field(default_factory=StreamflowConfig)
    snotel: Optional[SNOTELConfig] = Field(default_factory=SNOTELConfig)
    fluxnet: Optional[FluxNetConfig] = Field(default_factory=FluxNetConfig)
    usgs_gw: Optional[USGSGWConfig] = Field(default_factory=USGSGWConfig)
    smap: Optional[SMAPConfig] = Field(default_factory=SMAPConfig)
    ismn: Optional[ISMNConfig] = Field(default_factory=ISMNConfig)
    grace: Optional[GRACEConfig] = Field(default_factory=GRACEConfig)
    modis_snow: Optional[MODISSnowConfig] = Field(default_factory=MODISSnowConfig)
    modis_et: Optional[MODISETConfig] = Field(default_factory=MODISETConfig)
    attributes: Optional[AttributesConfig] = Field(default_factory=AttributesConfig)
    smhi: Optional[SMHIConfig] = Field(default_factory=SMHIConfig)
    lamah_ice: Optional[LAMAHICEConfig] = Field(default_factory=LAMAHICEConfig)
    glacier: Optional[GlacierConfig] = Field(default_factory=GlacierConfig)
    hru_gauge_mapping: Optional[Dict[str, Any]] = Field(default_factory=dict, alias='HRU_GAUGE_MAPPING')
    # Evaluator-specific settings
    tws: Optional[TWSEvalConfig] = Field(default_factory=TWSEvalConfig)
    et: Optional[ETEvalConfig] = Field(default_factory=ETEvalConfig)

    @field_validator('evaluation_data', 'analyses', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
