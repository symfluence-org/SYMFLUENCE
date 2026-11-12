# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Hierarchical configuration models for SYMFLUENCE.

This package defines a nested configuration structure that organizes the 346+
configuration parameters into logical sections while maintaining backward
compatibility through factory methods and dict-like access.

Key design features:
- Type-safe hierarchical structure (config.domain.name vs config['DOMAIN_NAME'])
- Factory methods: from_preset(), from_minimal(), from_file()
- Backward compatibility: to_dict(), get(), __getitem__()
- All validation logic preserved from original flat model
- Immutable configs (frozen=True) to prevent mutation bugs
"""

# Root config - the main entry point
# FEWS adapter config
from __future__ import annotations

from symfluence.core.config.models.fews import FEWSConfig

# Data configs
from .data import (
    DataConfig,
    GeospatialConfig,
    MODISLandcoverConfig,
    NASADEMConfig,
    NLCDConfig,
    SoilGridsConfig,
)

# Domain configs
from .domain import DelineationConfig, DomainConfig

# Evaluation configs
from .evaluation import (
    AttributesConfig,
    EvaluationConfig,
    FluxNetConfig,
    GRACEConfig,
    HubeauConfig,
    ISMNConfig,
    KiWISConfig,
    MODISETConfig,
    MODISSnowConfig,
    SMAPConfig,
    SNOTELConfig,
    StreamflowConfig,
    USGSGWConfig,
)

# Forcing configs
from .forcing import EMEarthConfig, ERA5Config, ForcingConfig, LapseRateConfig, NexConfig

# Model configs
from .model_configs import ModelConfig

# Per-model schema classes (SUMMAConfig, FUSEConfig, ...) live with their
# model packages and resolve lazily through R.config_schemas; importing them
# eagerly here would force registry resolution before model plugins load.
_LAZY_MODEL_SCHEMA_NAMES = frozenset({
    'FUSEConfig', 'GNNConfig', 'GRConfig', 'HYPEConfig', 'LSTMConfig',
    'MESHConfig', 'MizuRouteConfig', 'NGENConfig', 'RHESSysConfig',
    'SUMMAConfig',
})


def __getattr__(name: str):
    if name in _LAZY_MODEL_SCHEMA_NAMES:
        from . import model_configs

        value = getattr(model_configs, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Optimization configs
from .optimization import (
    DDSConfig,
    DEConfig,
    EmulationConfig,
    NSGA2Config,
    OptimizationConfig,
    PSOConfig,
    SCEUAConfig,
)

# Paths configs
from .paths import PathsConfig, ShapefilePathConfig
from .root import SymfluenceConfig

# State management and data assimilation configs
from .state_config import DataAssimilationConfig, EnKFConfig, StateConfig

# System config
from .system import SystemConfig

__all__ = [
    *sorted(_LAZY_MODEL_SCHEMA_NAMES),
    # Root
    "SymfluenceConfig",
    # System
    "SystemConfig",
    # Domain
    "DomainConfig",
    "DelineationConfig",
    # Data
    "DataConfig",
    "GeospatialConfig",
    "SoilGridsConfig",
    "MODISLandcoverConfig",
    "NLCDConfig",
    "NASADEMConfig",
    # Forcing
    "ForcingConfig",
    "LapseRateConfig",
    "NexConfig",
    "EMEarthConfig",
    "ERA5Config",
    # Models
    "ModelConfig",
    # Optimization
    "OptimizationConfig",
    "PSOConfig",
    "DEConfig",
    "DDSConfig",
    "SCEUAConfig",
    "NSGA2Config",
    "EmulationConfig",
    # Evaluation
    "EvaluationConfig",
    "StreamflowConfig",
    "HubeauConfig",
    "KiWISConfig",
    "SNOTELConfig",
    "FluxNetConfig",
    "USGSGWConfig",
    "SMAPConfig",
    "ISMNConfig",
    "GRACEConfig",
    "MODISSnowConfig",
    "MODISETConfig",
    "AttributesConfig",
    # Paths
    "PathsConfig",
    "ShapefilePathConfig",
    # State management and data assimilation
    "StateConfig",
    "EnKFConfig",
    "DataAssimilationConfig",
    # FEWS
    "FEWSConfig",
]
