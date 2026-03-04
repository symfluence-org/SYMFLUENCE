# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model configuration entrypoint and compatibility exports.

This module preserves the historical import path
``symfluence.core.config.models.model_configs`` while delegating concrete
model-specific class definitions to focused modules.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .base import FROZEN_CONFIG
from .model_config_types import SpatialModeType
from .model_configs_hydrology import (
    CLMConfig,
    CRHMConfig,
    FUSEConfig,
    GRConfig,
    GSFLOWConfig,
    HBVConfig,
    HECHMSConfig,
    HYPEConfig,
    MESHConfig,
    MHMConfig,
    NGENConfig,
    PRMSConfig,
    RHESSysConfig,
    SUMMAConfig,
    SWATConfig,
    TOPMODELConfig,
    VICConfig,
    WATFLOODConfig,
    WflowConfig,
    WRFHydroConfig,
)
from .model_configs_integrated import (
    CLMParFlowConfig,
    HydroGeoSphereConfig,
    MODFLOWConfig,
    ParFlowConfig,
    PIHMConfig,
)
from .model_configs_ml_fire import GNNConfig, IGNACIOConfig, LSTMConfig, WMFireConfig
from .model_configs_routing import DRouteConfig, MizuRouteConfig, TRouteConfig

ConfigRegistryEntry = tuple[str, type[BaseModel]]

# Keep registry coverage aligned with historical behavior from the previous
# if/elif implementation to avoid implicit behavior changes.
HYDROLOGICAL_MODEL_REGISTRY: dict[str, ConfigRegistryEntry] = {
    'SUMMA': ('summa', SUMMAConfig),
    'FUSE': ('fuse', FUSEConfig),
    'GR': ('gr', GRConfig),
    'HBV': ('hbv', HBVConfig),
    'HECHMS': ('hechms', HECHMSConfig),
    'TOPMODEL': ('topmodel', TOPMODELConfig),
    'HYPE': ('hype', HYPEConfig),
    'NGEN': ('ngen', NGENConfig),
    'MESH': ('mesh', MESHConfig),
    'LSTM': ('lstm', LSTMConfig),
    'RHESSYS': ('rhessys', RHESSysConfig),
    'GNN': ('gnn', GNNConfig),
    'VIC': ('vic', VICConfig),
    'CLM': ('clm', CLMConfig),
    'MODFLOW': ('modflow', MODFLOWConfig),
    'PARFLOW': ('parflow', ParFlowConfig),
    'CLMPARFLOW': ('clmparflow', CLMParFlowConfig),
    'SWAT': ('swat', SWATConfig),
    'MHM': ('mhm', MHMConfig),
    'CRHM': ('crhm', CRHMConfig),
    'WRFHYDRO': ('wrfhydro', WRFHydroConfig),
    'PRMS': ('prms', PRMSConfig),
    'PIHM': ('pihm', PIHMConfig),
    'HYDROGEOSPHERE': ('hydrogeosphere', HydroGeoSphereConfig),
    'GSFLOW': ('gsflow', GSFLOWConfig),
    'WATFLOOD': ('watflood', WATFLOODConfig),
    'WFLOW': ('wflow', WflowConfig),
}

ROUTING_MODEL_REGISTRY: dict[str, ConfigRegistryEntry] = {
    'MIZUROUTE': ('mizuroute', MizuRouteConfig),
    'DROUTE': ('droute', DRouteConfig),
}

GROUNDWATER_MODEL_REGISTRY: dict[str, ConfigRegistryEntry] = {
    'MODFLOW': ('modflow', MODFLOWConfig),
    'PARFLOW': ('parflow', ParFlowConfig),
}

FIRE_MODEL_REGISTRY: dict[str, ConfigRegistryEntry] = {
    'IGNACIO': ('ignacio', IGNACIOConfig),
}


class ModelConfig(BaseModel):
    """Hydrological model configuration"""
    model_config = FROZEN_CONFIG

    # Required model selection
    hydrological_model: Union[str, List[str]] = Field(alias='HYDROLOGICAL_MODEL')
    routing_model: Optional[str] = Field(default=None, alias='ROUTING_MODEL')
    groundwater_model: Optional[str] = Field(default=None, alias='GROUNDWATER_MODEL')
    fire_model: Optional[str] = Field(default=None, alias='FIRE_MODEL')
    # Coupling settings
    coupling_mode: str = Field(default='sequential', alias='COUPLING_MODE')
    conservation_mode: Optional[str] = Field(default=None, alias='CONSERVATION_MODE')
    topology_file: Optional[str] = Field(default=None, alias='TOPOLOGY_FILE')
    snow_module: str = Field(default='', alias='SNOW_MODULE')

    # Model-specific configurations (optional, validated only if model is selected)
    summa: Optional[SUMMAConfig] = Field(default=None)
    fuse: Optional[FUSEConfig] = Field(default=None)
    gr: Optional[GRConfig] = Field(default=None)
    hbv: Optional[HBVConfig] = Field(default=None)
    hechms: Optional[HECHMSConfig] = Field(default=None)
    topmodel: Optional[TOPMODELConfig] = Field(default=None)
    hype: Optional[HYPEConfig] = Field(default=None)
    ngen: Optional[NGENConfig] = Field(default=None)
    mesh: Optional[MESHConfig] = Field(default=None)
    mizuroute: Optional[MizuRouteConfig] = Field(default=None)
    droute: Optional[DRouteConfig] = Field(default=None)
    troute: Optional[TRouteConfig] = Field(default=None)
    lstm: Optional[LSTMConfig] = Field(default=None, alias='lstm')
    rhessys: Optional[RHESSysConfig] = Field(default=None)
    gnn: Optional[GNNConfig] = Field(default=None)
    ignacio: Optional[IGNACIOConfig] = Field(default=None)
    vic: Optional[VICConfig] = Field(default=None)
    clm: Optional[CLMConfig] = Field(default=None)
    modflow: Optional[MODFLOWConfig] = Field(default=None)
    parflow: Optional[ParFlowConfig] = Field(default=None)
    clmparflow: Optional[CLMParFlowConfig] = Field(default=None)
    swat: Optional[SWATConfig] = Field(default=None)
    mhm: Optional[MHMConfig] = Field(default=None)
    crhm: Optional[CRHMConfig] = Field(default=None)
    wrfhydro: Optional[WRFHydroConfig] = Field(default=None)
    prms: Optional[PRMSConfig] = Field(default=None)
    pihm: Optional[PIHMConfig] = Field(default=None)
    hydrogeosphere: Optional[HydroGeoSphereConfig] = Field(default=None)
    gsflow: Optional[GSFLOWConfig] = Field(default=None)
    watflood: Optional[WATFLOODConfig] = Field(default=None)
    wflow: Optional[WflowConfig] = Field(default=None)

    @field_validator('hydrological_model')
    @classmethod
    def validate_hydrological_model(cls, v):
        """Normalize model list to comma-separated string"""
        if isinstance(v, list):
            return ",".join(str(i).strip() for i in v)
        return v

    @model_validator(mode='before')
    @classmethod
    def auto_populate_model_configs(cls, values):
        """Auto-populate model-specific configs when model is selected."""
        if not isinstance(values, dict):
            return values

        # Get hydrological_model from values (check both alias and field name)
        hydrological_model = values.get('HYDROLOGICAL_MODEL') or values.get('hydrological_model')
        if not hydrological_model:
            return values

        # Parse models from hydrological_model string
        if isinstance(hydrological_model, list):
            models = [str(m).strip().upper() for m in hydrological_model]
        else:
            models = [m.strip().upper() for m in str(hydrological_model).split(',')]

        # Auto-create model configs using registry lookup.
        for model_name in models:
            registry_entry = HYDROLOGICAL_MODEL_REGISTRY.get(model_name)
            if registry_entry is None:
                continue
            field_name, config_cls = registry_entry
            if values.get(field_name) is None:
                values[field_name] = config_cls()

        # Auto-create routing model config if needed
        routing_model = values.get('ROUTING_MODEL') or values.get('routing_model')
        if routing_model:
            routing_entry = ROUTING_MODEL_REGISTRY.get(str(routing_model).upper())
            if routing_entry is not None:
                field_name, config_cls = routing_entry
                if values.get(field_name) is None:
                    values[field_name] = config_cls()

        # Auto-create groundwater model config if needed
        gw_model = values.get('GROUNDWATER_MODEL') or values.get('groundwater_model')
        if gw_model:
            gw_entry = GROUNDWATER_MODEL_REGISTRY.get(str(gw_model).upper())
            if gw_entry is not None:
                field_name, config_cls = gw_entry
                if values.get(field_name) is None:
                    values[field_name] = config_cls()

        # Auto-create fire model config if needed
        fire_model = values.get('FIRE_MODEL') or values.get('fire_model')
        if fire_model:
            fire_entry = FIRE_MODEL_REGISTRY.get(str(fire_model).upper())
            if fire_entry is not None:
                field_name, config_cls = fire_entry
                if values.get(field_name) is None:
                    values[field_name] = config_cls()

        # Forward deprecated ENABLE_* keys into the ngen sub-dict so
        # NGENConfig._migrate_enable_flags can pick them up (handles flat
        # configs where these land at the ModelConfig level).
        _enable_keys = ('ENABLE_SLOTH', 'ENABLE_PET', 'ENABLE_NOAH', 'ENABLE_CFE')
        found = {k: values[k] for k in _enable_keys if k in values}
        if found:
            ngen_dict = values.get('ngen')
            if ngen_dict is None:
                ngen_dict = {}
                values['ngen'] = ngen_dict
            if isinstance(ngen_dict, dict):
                for k, v in found.items():
                    ngen_dict.setdefault(k, v)
            # Remove from top level so Pydantic doesn't reject them
            for k in found:
                values.pop(k, None)

        return values

__all__ = [
    'SpatialModeType',
    'ModelConfig',
    'SUMMAConfig',
    'FUSEConfig',
    'GRConfig',
    'HBVConfig',
    'HECHMSConfig',
    'TOPMODELConfig',
    'HYPEConfig',
    'NGENConfig',
    'MESHConfig',
    'MizuRouteConfig',
    'DRouteConfig',
    'TRouteConfig',
    'LSTMConfig',
    'WMFireConfig',
    'RHESSysConfig',
    'VICConfig',
    'CLMConfig',
    'MODFLOWConfig',
    'ParFlowConfig',
    'CLMParFlowConfig',
    'PIHMConfig',
    'HydroGeoSphereConfig',
    'GNNConfig',
    'IGNACIOConfig',
    'SWATConfig',
    'MHMConfig',
    'CRHMConfig',
    'WRFHydroConfig',
    'PRMSConfig',
    'GSFLOWConfig',
    'WATFLOODConfig',
    'WflowConfig',
]
