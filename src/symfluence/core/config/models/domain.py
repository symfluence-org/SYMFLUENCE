# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Domain configuration models.

Contains DelineationConfig and DomainConfig for spatial extent, timing, and discretization.
"""

import warnings
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG


class DelineationConfig(BaseModel):
    """Watershed delineation settings"""
    model_config = FROZEN_CONFIG

    routing: str = Field(default='lumped', alias='ROUTING_DELINEATION')
    geofabric_type: str = Field(default='na', alias='GEOFABRIC_TYPE')
    method: str = Field(default='stream_threshold', alias='DELINEATION_METHOD')
    curvature_threshold: float = Field(default=0.0, alias='CURVATURE_THRESHOLD')
    min_source_threshold: int = Field(default=100, alias='MIN_SOURCE_THRESHOLD')
    stream_threshold: float = Field(default=5000.0, alias='STREAM_THRESHOLD')
    slope_area_threshold: float = Field(default=100.0, alias='SLOPE_AREA_THRESHOLD')
    slope_area_exponent: float = Field(default=2.0, alias='SLOPE_AREA_EXPONENT')
    area_exponent: float = Field(default=1.0, alias='AREA_EXPONENT')
    multi_scale_thresholds: Optional[str] = Field(default=None, alias='MULTI_SCALE_THRESHOLDS')
    use_drop_analysis: bool = Field(default=False, alias='USE_DROP_ANALYSIS')
    drop_analysis_min_threshold: int = Field(default=100, alias='DROP_ANALYSIS_MIN_THRESHOLD')
    drop_analysis_max_threshold: int = Field(default=10000, alias='DROP_ANALYSIS_MAX_THRESHOLD')
    drop_analysis_num_thresholds: int = Field(default=10, alias='DROP_ANALYSIS_NUM_THRESHOLDS')
    drop_analysis_log_spacing: bool = Field(default=True, alias='DROP_ANALYSIS_LOG_SPACING')
    lumped_watershed_method: str = Field(default='TauDEM', alias='LUMPED_WATERSHED_METHOD')
    cleanup_intermediate_files: bool = Field(default=False, alias='CLEANUP_INTERMEDIATE_FILES')
    delineate_coastal_watersheds: bool = Field(default=False, alias='DELINEATE_COASTAL_WATERSHEDS')
    delineate_by_pourpoint: bool = Field(default=True, alias='DELINEATE_BY_POURPOINT')
    move_outlets_max_distance: float = Field(default=200.0, alias='MOVE_OUTLETS_MAX_DISTANCE')
    geofabric_bbox_padding: float = Field(default=0.02, alias='GEOFABRIC_BBOX_PADDING')
    point_buffer_distance: Optional[float] = Field(default=None, alias='POINT_BUFFER_DISTANCE')
    max_retries: int = Field(default=3, alias='MAX_RETRIES')
    retry_delay: float = Field(default=5.0, alias='RETRY_DELAY')

    # DEM conditioning (stream burning)
    dem_conditioning_method: str = Field(default='none', alias='DEM_CONDITIONING_METHOD')
    stream_burn_depth: float = Field(default=5.0, alias='STREAM_BURN_DEPTH')
    stream_burn_source: str = Field(default='auto', alias='STREAM_BURN_SOURCE')
    stream_burn_custom_path: str = Field(default='default', alias='STREAM_BURN_CUSTOM_PATH')

    @field_validator('dem_conditioning_method', mode='before')
    @classmethod
    def normalize_conditioning_method(cls, v):
        """Normalize and validate DEM conditioning method."""
        if not isinstance(v, str):
            return v
        v = v.lower().strip()
        valid = {'none', 'burn_streams'}
        if v not in valid:
            raise ValueError(
                f"DEM_CONDITIONING_METHOD must be one of {valid}, got '{v}'"
            )
        return v

    @field_validator('stream_burn_depth')
    @classmethod
    def validate_burn_depth(cls, v):
        """Ensure burn depth is positive."""
        if v <= 0:
            raise ValueError(f"STREAM_BURN_DEPTH must be positive, got {v}")
        return v

    @field_validator('multi_scale_thresholds', mode='before')
    @classmethod
    def normalize_multi_scale_thresholds(cls, v):
        """Convert list to comma-separated string"""
        if isinstance(v, list):
            return ','.join(str(x) for x in v)
        return v

    @field_validator('geofabric_type', mode='before')
    @classmethod
    def normalize_geofabric_type(cls, v):
        """Normalize geofabric type aliases to canonical names."""
        if not isinstance(v, str):
            return v
        alias_map = {
            'merit_basins': 'merit',
            'geoglows': 'tdx',
            'tdx_hydro': 'tdx',
            'nws_hydrofabric': 'nws',
            'nextgen': 'nws',
            'hydrobasins': 'hydrosheds',
        }
        return alias_map.get(v.lower(), v.lower())

    @field_validator('stream_threshold', 'slope_area_threshold')
    @classmethod
    def validate_positive_thresholds(cls, v, info):
        """Ensure thresholds are non-negative"""
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {v}")
        return v


class DomainConfig(BaseModel):
    """Domain definition: spatial extent, timing, discretization"""
    model_config = FROZEN_CONFIG

    # Required identification
    name: str = Field(alias='DOMAIN_NAME')
    experiment_id: str = Field(alias='EXPERIMENT_ID')

    # Required timing
    time_start: str = Field(alias='EXPERIMENT_TIME_START')
    time_end: str = Field(alias='EXPERIMENT_TIME_END')

    # Optional time periods
    calibration_period: Optional[str] = Field(default=None, alias='CALIBRATION_PERIOD')
    calibration_start_date: Optional[str] = Field(default=None, alias='CALIBRATION_START_DATE')
    calibration_end_date: Optional[str] = Field(default=None, alias='CALIBRATION_END_DATE')
    evaluation_period: Optional[str] = Field(default=None, alias='EVALUATION_PERIOD')
    spinup_period: Optional[str] = Field(default=None, alias='SPINUP_PERIOD')

    # Required spatial definition
    definition_method: Literal['point', 'lumped', 'semidistributed', 'distributed'] = Field(alias='DOMAIN_DEFINITION_METHOD')
    discretization: str = Field(alias='SUB_GRID_DISCRETIZATION')

    # Subsetting option (for lumped, semidistributed, distributed)
    subset_from_geofabric: bool = Field(default=False, alias='SUBSET_FROM_GEOFABRIC')

    # Grid source (for distributed only)
    grid_source: Literal['generate', 'native'] = Field(default='generate', alias='GRID_SOURCE')

    # Native grid dataset identifier (for distributed + native)
    native_grid_dataset: str = Field(default='era5', alias='NATIVE_GRID_DATASET')

    # Grid-based distributed mode settings
    grid_cell_size: float = Field(default=1000.0, alias='GRID_CELL_SIZE')  # meters
    clip_grid_to_watershed: bool = Field(default=True, alias='CLIP_GRID_TO_WATERSHED')

    # Optional spatial coordinates
    pour_point_coords: Optional[str] = Field(default=None, alias='POUR_POINT_COORDS')
    bounding_box_coords: Optional[str] = Field(default=None, alias='BOUNDING_BOX_COORDS')

    # Delineation settings (nested)
    delineation: DelineationConfig = Field(default_factory=DelineationConfig)

    # Discretization settings
    min_gru_size: float = Field(default=0.0, alias='MIN_GRU_SIZE')
    min_hru_size: float = Field(default=0.0, alias='MIN_HRU_SIZE')
    elevation_band_size: float = Field(default=200.0, alias='ELEVATION_BAND_SIZE')
    radiation_class_number: int = Field(default=1, alias='RADIATION_CLASS_NUMBER')
    aspect_class_number: int = Field(default=1, alias='ASPECT_CLASS_NUMBER')
    aspect_path: str = Field(default='default', alias='ASPECT_PATH')

    # Catchment area (km2) - used for unit conversions (e.g., CLM QRUNOFF mm/s -> m3/s)
    catchment_area: Optional[float] = Field(default=None, alias='CATCHMENT_AREA')

    # Data access
    data_access: str = Field(default='cloud', alias='DATA_ACCESS')
    download_dem: bool = Field(default=True, alias='DOWNLOAD_DEM')
    download_soil: bool = Field(default=True, alias='DOWNLOAD_SOIL')
    download_landcover: bool = Field(default=True, alias='DOWNLOAD_LAND_COVER')
    dem_source: str = Field(default='copernicus', alias='DEM_SOURCE')  # Copernicus DEM is free and open
    land_class_source: str = Field(default='modis', alias='LAND_CLASS_SOURCE')
    land_class_name: str = Field(default='default', alias='LAND_CLASS_NAME')
    soilgrids_layer: str = Field(default='wrb_0-5cm_mode', alias='SOILGRIDS_LAYER')

    @field_validator('min_gru_size', 'min_hru_size', 'elevation_band_size')
    @classmethod
    def validate_positive_thresholds(cls, v, info):
        """Ensure thresholds are non-negative"""
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {v}")
        return v

    @field_validator('radiation_class_number', 'aspect_class_number')
    @classmethod
    def validate_positive_integers(cls, v, info):
        """Ensure positive integers"""
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1, got {v}")
        return v

    @field_validator('definition_method', mode='before')
    @classmethod
    def normalize_definition_method(cls, v):
        """Map legacy method names to new values for backwards compatibility."""
        legacy_mapping = {
            'delineate': 'semidistributed',
            'distribute': 'distributed',
            'discretized': 'semidistributed',  # deprecated
            'subset': 'semidistributed',  # now use subset_from_geofabric=True
            # Spelling variants written by several paper configs
            'semi_distributed': 'semidistributed',
            'semi-distributed': 'semidistributed',
        }
        if v in legacy_mapping:
            if v == 'discretized':
                warnings.warn(
                    f"definition_method '{v}' is deprecated, use 'semidistributed' instead",
                    DeprecationWarning,
                    stacklevel=2
                )
            return legacy_mapping[v]
        return v
