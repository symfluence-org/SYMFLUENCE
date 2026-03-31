# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcing configuration models.

Contains NexConfig, EMEarthConfig, and ForcingConfig for meteorological forcing data.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG

# Supported forcing dataset types
ForcingDatasetType = Literal[
    'NLDAS', 'NLDAS2', 'NEX-GDDP', 'ERA5', 'EM-EARTH', 'RDRS', 'CASR', 'CARRA', 'CERRA',
    'MSWEP', 'AORC', 'CONUS404', 'HRRR', 'DAYMET', 'NWM3_RETROSPECTIVE', 'local'
]

# Supported PET calculation methods
PETMethodType = Literal['oudin', 'hargreaves', 'priestley_taylor', 'penman', 'fao56']


class NexConfig(BaseModel):
    """NASA NEX-GDDP climate projection settings"""
    model_config = FROZEN_CONFIG

    models: Optional[List[str]] = Field(default=None, alias='NEX_MODELS')
    scenarios: Optional[List[str]] = Field(default=None, alias='NEX_SCENARIOS')
    ensembles: Optional[List[str]] = Field(default=None, alias='NEX_ENSEMBLES')
    variables: Optional[List[str]] = Field(default=None, alias='NEX_VARIABLES')

    @field_validator('models', 'scenarios', 'ensembles', 'variables', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class EMEarthConfig(BaseModel):
    """EM-Earth ensemble meteorological forcing settings"""
    model_config = FROZEN_CONFIG

    region: str = Field(default='NorthAmerica', alias='EM_EARTH_REGION')
    prcp_dir: Optional[str] = Field(default=None, alias='EM_EARTH_PRCP_DIR')
    tmean_dir: Optional[str] = Field(default=None, alias='EM_EARTH_TMEAN_DIR')
    min_bbox_size: float = Field(default=0.1, alias='EM_EARTH_MIN_BBOX_SIZE')
    max_expansion: float = Field(default=0.2, alias='EM_EARTH_MAX_EXPANSION')
    prcp_var: str = Field(default='prcp', alias='EM_PRCP')
    data_type: str = Field(default='deterministic', alias='EM_EARTH_DATA_TYPE')


class ERA5Config(BaseModel):
    """ERA5 reanalysis forcing settings"""
    model_config = FROZEN_CONFIG

    use_cds: Optional[bool] = Field(default=None, alias='ERA5_USE_CDS')
    zarr_path: str = Field(
        default='gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
        alias='ERA5_ZARR_PATH'
    )
    time_step_hours: int = Field(default=1, alias='ERA5_TIME_STEP_HOURS')
    variables: Optional[List[str]] = Field(
        default=None,
        alias='ERA5_VARS'
    )

    @field_validator('variables', mode='before')
    @classmethod
    def validate_variables(cls, v):
        """Normalize string lists"""
        if v is None:
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class LapseRateConfig(BaseModel):
    """Configuration for elevation-dependent variable corrections.

    Per-variable toggles control which corrections are applied beyond
    the master ``apply_lapse_rate`` / ``lapse_rate`` temperature toggle.
    All optional corrections default to OFF.
    """
    model_config = FROZEN_CONFIG

    # Per-variable toggles (all default OFF except temperature via master toggle)
    pressure: bool = False
    humidity: bool = False
    longwave: bool = False
    shortwave: bool = False
    precipitation: bool = False
    wind: bool = False

    # Configurable gradient parameters
    lw_gradient: float = -0.029       # W/m² per m elevation (-2.9 W/m² per 100m)
    sw_gradient: float = 0.00002      # fractional per m (+2% per km)
    precip_gradient: float = 0.00004  # fractional per m (+4% per 100m)
    wind_gradient: float = 0.0        # disabled even when wind=True


class ForcingConfig(BaseModel):
    """Meteorological forcing configuration"""
    model_config = FROZEN_CONFIG

    # Required dataset
    dataset: ForcingDatasetType = Field(alias='FORCING_DATASET')

    # Forcing settings
    time_step_size: int = Field(default=3600, alias='FORCING_TIME_STEP_SIZE', ge=60, le=86400)
    variables: str = Field(default='default', alias='FORCING_VARIABLES')
    measurement_height: float = Field(default=2.0, alias='FORCING_MEASUREMENT_HEIGHT', gt=0)
    apply_lapse_rate: bool = Field(default=True, alias='APPLY_LAPSE_RATE')
    lapse_rate: float = Field(default=0.0065, alias='LAPSE_RATE')
    shape_lat_name: str = Field(default='lat', alias='FORCING_SHAPE_LAT_NAME')
    shape_lon_name: str = Field(default='lon', alias='FORCING_SHAPE_LON_NAME')
    pet_method: PETMethodType = Field(default='oudin', alias='PET_METHOD')
    supplement: bool = Field(default=False, alias='SUPPLEMENT_FORCING')
    keep_raw: bool = Field(default=True, alias='KEEP_RAW_FORCING')

    # ERA5-specific settings (legacy, prefer using era5 subsection)
    era5_use_cds: Optional[bool] = Field(default=None, alias='ERA5_USE_CDS')

    # Elevation correction settings
    lapse: Optional[LapseRateConfig] = Field(default=None)

    # Dataset-specific settings
    nex: Optional[NexConfig] = Field(default=None)
    em_earth: Optional[EMEarthConfig] = Field(default=None)
    era5: Optional[ERA5Config] = Field(default_factory=ERA5Config)

    @field_validator('variables', mode='before')
    @classmethod
    def normalize_variables(cls, v):
        """Convert list or other types to comma-separated string for variables"""
        if isinstance(v, list):
            return ','.join(str(item).strip() for item in v)
        return str(v) if v is not None else 'default'
