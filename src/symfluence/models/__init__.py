# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# src/symfluence.models/__init__.py
"""Hydrological model utilities.

This module provides:
- ModelRegistry: Central registry for model runners/preprocessors/postprocessors
- Execution Framework: Unified subprocess/SLURM execution (execution submodule)
- Config Schemas: Type-safe configuration contracts (config submodule)
- Templates: Base classes for new model implementations (templates submodule)
"""

from .registry import ModelRegistry

# Import execution framework components
try:
    from .execution import (
        ExecutionResult,
        ModelExecutor,
        RoutingConfig,
        SlurmJobConfig,
        SpatialMode,
        SpatialOrchestrator,
    )
except ImportError:
    pass  # Optional - may not be needed by all users

# Import config schema components
try:
    from .config import (
        ModelConfigSchema,
        get_model_schema,
        validate_model_config,
    )
except ImportError:
    pass  # Optional

# Import template components
try:
    from .templates import (
        ModelRunResult,
        UnifiedModelRunner,
    )
except ImportError:
    pass  # Optional

# Import all models to register them
import logging
import warnings

logger = logging.getLogger(__name__)

# Suppress experimental module warnings and missing optional dependency warnings
warnings.filterwarnings('ignore', message='.*is an EXPERIMENTAL module.*')
warnings.filterwarnings('ignore', message='.*import failed.*')

# Import from modular packages (preferred).
# Use `except Exception` (not ImportError) because model imports may  # noqa: BLE001 — comment only
# transitively trigger threading._register_atexit() which raises
# RuntimeError on daemon threads (e.g. Panel's Tornado IO loop).
_model_names = [
    'summa', 'fuse', 'ngen', 'mizuroute', 'troute',
    'hype', 'mesh', 'lstm', 'gr', 'gnn', 'rhessys',
    'ignacio', 'vic', 'clm',
    'modflow', 'parflow', 'clmparflow',
    'swat', 'mhm', 'crhm', 'wrfhydro', 'prms',
    'pihm', 'wmfire',
    'gsflow', 'watflood', 'wflow',
]

import importlib as _importlib

for _model_name in _model_names:
    try:
        _importlib.import_module(f'.{_model_name}', __name__)
    except Exception as _e:  # noqa: BLE001 — optional dependency
        logger.debug(f"Could not import {_model_name}: {_e}")

del _model_names, _model_name, _importlib
try:
    del _e
except NameError:
    pass


__all__ = [
    # Core
    "ModelRegistry",
    # Execution Framework
    "ModelExecutor",
    "ExecutionResult",
    "SlurmJobConfig",
    "SpatialOrchestrator",
    "SpatialMode",
    "RoutingConfig",
    # Config Schemas
    "ModelConfigSchema",
    "get_model_schema",
    "validate_model_config",
    # Templates
    "UnifiedModelRunner",
    "ModelRunResult",
]
