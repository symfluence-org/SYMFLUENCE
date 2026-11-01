# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data management configuration models.

Contains DataConfig for high-level data acquisition and processing settings,
including geospatial data acquisition configuration.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG


class SoilGridsConfig(BaseModel):
    """SoilGrids soil classification data acquisition settings."""
    model_config = FROZEN_CONFIG

    layer: str = Field(default='wrb_0-5cm_mode', alias='SOILGRIDS_LAYER')
    wcs_map: str = Field(default='/map/wcs/soilgrids.map', alias='SOILGRIDS_WCS_MAP')
    coverage_id: Optional[str] = Field(default=None, alias='SOILGRIDS_COVERAGE_ID')
    hs_cache_dir: str = Field(default='default', alias='SOILGRIDS_HS_CACHE_DIR')
    hs_resource_id: str = Field(default='1361509511e44adfba814f6950c6e742', alias='SOILGRIDS_HS_RESOURCE_ID')
    hs_api_url: Optional[str] = Field(default=None, alias='SOILGRIDS_HS_API_URL')


class MODISLandcoverConfig(BaseModel):
    """MODIS land cover acquisition settings."""
    model_config = FROZEN_CONFIG

    years: Optional[List[int]] = Field(default=None, alias='MODIS_LANDCOVER_YEARS')
    start_year: Optional[int] = Field(default=None, alias='MODIS_LANDCOVER_START_YEAR')
    end_year: Optional[int] = Field(default=None, alias='MODIS_LANDCOVER_END_YEAR')
    year: Optional[int] = Field(default=None, alias='LANDCOVER_YEAR')
    base_url: str = Field(default='https://zenodo.org/records/8367523/files', alias='MODIS_LANDCOVER_BASE_URL')
    cache_dir: str = Field(default='default', alias='MODIS_LANDCOVER_CACHE_DIR')
    local_file: Optional[str] = Field(default=None, alias='LANDCOVER_LOCAL_FILE')

    @field_validator('years', mode='before')
    @classmethod
    def validate_years(cls, v):
        """Convert years to list of ints."""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [int(y) for y in v]
        if isinstance(v, str):
            return [int(y.strip()) for y in v.split(',') if y.strip()]
        return [int(v)]


class NLCDConfig(BaseModel):
    """USGS NLCD land cover acquisition settings."""
    model_config = FROZEN_CONFIG

    coverage_id: str = Field(default='NLCD_2019_Land_Cover_L48', alias='NLCD_COVERAGE_ID')


class NASADEMConfig(BaseModel):
    """NASADEM local data settings."""
    model_config = FROZEN_CONFIG

    local_dir: Optional[str] = Field(default=None, alias='NASADEM_LOCAL_DIR')


class GeospatialConfig(BaseModel):
    """Geospatial data acquisition configuration."""
    model_config = FROZEN_CONFIG

    soilgrids: Optional[SoilGridsConfig] = Field(default_factory=SoilGridsConfig)
    modis_landcover: Optional[MODISLandcoverConfig] = Field(default_factory=MODISLandcoverConfig)
    nlcd: Optional[NLCDConfig] = Field(default_factory=NLCDConfig)
    nasadem: Optional[NASADEMConfig] = Field(default_factory=NASADEMConfig)


class DataConfig(BaseModel):
    """Configuration for data acquisition and processing"""
    model_config = FROZEN_CONFIG

    # High-level acquisition flags
    additional_observations: Optional[List[str]] = Field(default=None, alias='ADDITIONAL_OBSERVATIONS')
    force_download: bool = Field(default=False, alias='FORCE_DOWNLOAD')

    # Geofabric acquisition settings
    hydrosheds_level: Optional[int] = Field(
        default=None, alias='HYDROSHEDS_LEVEL',
        description='HydroBASINS level to acquire (default 12)'
    )
    tdx_source: Optional[str] = Field(
        default=None, alias='TDX_SOURCE',
        description="TDX-Hydro download source (e.g. 'geoglows')"
    )

    # Streamflow provider
    streamflow_data_provider: Optional[str] = Field(default=None, alias='STREAMFLOW_DATA_PROVIDER')

    # Streamflow station IDs for different providers
    usgs_site_code: Optional[str] = Field(default=None, alias='USGS_SITE_CODE')
    download_usgs_data: bool = Field(default=False, alias='DOWNLOAD_USGS_DATA')
    streamflow_station_id: Optional[str] = Field(default=None, alias='STREAMFLOW_STATION_ID')

    # Note: Dataset download flags consolidated to EvaluationConfig sections:
    # - download_usgs_gw -> evaluation.usgs_gw.download
    # - download_modis_snow -> evaluation.modis_snow.download
    # - download_snotel -> evaluation.snotel.download
    # - download_smhi_data -> evaluation.smhi.download
    # - download_lamah_ice_data -> evaluation.lamah_ice.download
    # - download_glacier_data -> evaluation.glacier.download
    # - lamah_ice_path -> evaluation.lamah_ice.path
    download_ismn: bool = Field(default=False, alias='DOWNLOAD_ISMN')

    # Model-ready data store
    build_model_ready_store: bool = Field(default=True, alias='BUILD_MODEL_READY_STORE')
    model_ready_forcing_strategy: str = Field(default='symlink', alias='MODEL_READY_FORCING_STRATEGY')

    # Geospatial processing settings
    elev_chunk_size: int = Field(default=10_000, alias='ELEV_CHUNK_SIZE')
    elev_tile_target: int = Field(default=50_000, alias='ELEV_TILE_TARGET')

    # Concurrent acquisition
    max_acquisition_workers: int = Field(default=3, alias='MAX_ACQUISITION_WORKERS', ge=1, le=8)

    # Attribute cleanup / consolidation. No in-tree consumer since the legacy
    # acquisition.cleanup_processor was removed; kept so existing configs validate.
    unify_soil: bool = Field(default=False, alias='UNIFY_SOIL')
    minimum_land_fraction: float = Field(default=0.01, alias='MINIMUM_LAND_FRACTION', ge=0, le=1.0)
    num_land_cover: int = Field(default=20, alias='NUM_LAND_COVER', ge=1)

    # MAF / datatool job submission account. No in-tree consumer since the legacy
    # acquisition.maf_processor was removed; kept so existing configs validate.
    tool_account: Optional[str] = Field(default=None, alias='TOOL_ACCOUNT')

    # Geospatial data acquisition settings
    geospatial: Optional[GeospatialConfig] = Field(default_factory=GeospatialConfig)

    @field_validator('additional_observations', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
