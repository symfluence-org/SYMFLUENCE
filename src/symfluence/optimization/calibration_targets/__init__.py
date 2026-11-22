# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Calibration Targets

Calibration target classes that handle data loading, processing, and metric
calculation for specific variables during the optimization/calibration process.

Each calibration target is responsible for:
- Loading observed data for a specific variable
- Extracting simulated data from model outputs
- Calculating objective metrics for calibration

Base calibration targets (aliases from evaluation.evaluators):
- CalibrationTarget: Base class for all calibration targets
- ETTarget: Evapotranspiration calibration target
- StreamflowTarget: Streamflow calibration target (generic/SUMMA)
- SoilMoistureTarget: Soil moisture calibration target
- SnowTarget: Snow calibration target
- GroundwaterTarget: Groundwater calibration target
- TWSTarget: Terrestrial water storage calibration target
- MultivariateTarget: Multivariate calibration combining multiple variables

Model-specific calibration targets (``SUMMAStreamflowTarget``,
``GRStreamflowTarget``, ...) are resolved lazily (PEP 562) from their
canonical homes under ``symfluence.models.<model>.calibration.targets`` —
this package must not import the models layer at module level. Prefer the
registry (``R.calibration_targets``) or ``create_calibration_target()``.

Factory function for registry-based target creation:
- create_calibration_target(): Creates targets using registry with fallback
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from .base import (
    BasinETTarget,
    BasinSCATarget,
    BasinSWETarget,
    CalibrationTarget,
    ETTarget,
    GlacierMBTarget,
    GroundwaterTarget,
    MultivariateTarget,
    SnowTarget,
    SoilMoistureTarget,
    StreamflowTarget,
    TWSTarget,
)

# =========================================================================
# Lazy access to model-specific targets (canonical home: the model package)
# =========================================================================

_MODEL_TARGET_EXPORTS: Dict[str, str] = {
    'SUMMAStreamflowTarget': 'symfluence.models.summa.calibration.targets',
    'SUMMASnowTarget': 'symfluence.models.summa.calibration.targets',
    'SUMMAETTarget': 'symfluence.models.summa.calibration.targets',
    'GRStreamflowTarget': 'symfluence.models.gr.calibration.targets',
    'HYPEStreamflowTarget': 'symfluence.models.hype.calibration.targets',
    'RHESSysStreamflowTarget': 'symfluence.models.rhessys.calibration.targets',
    'NgenStreamflowTarget': 'symfluence.models.ngen.calibration.targets',
    'FUSEStreamflowTarget': 'symfluence.models.fuse.calibration.targets',
    'FUSESnowTarget': 'symfluence.models.fuse.calibration.targets',
}


def __getattr__(name: str):
    module_path = _MODEL_TARGET_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    target = getattr(importlib.import_module(module_path), name)
    globals()[name] = target
    return target


# =========================================================================
# Default target mapping (model-agnostic targets)
# =========================================================================

_DEFAULT_TARGETS: Dict[str, Type[CalibrationTarget]] = {
    'streamflow': StreamflowTarget,
    'et': ETTarget,
    'evapotranspiration': ETTarget,
    'snow': SnowTarget,
    'swe': SnowTarget,
    'sca': SnowTarget,
    'groundwater': GroundwaterTarget,
    'gw': GroundwaterTarget,
    'soil_moisture': SoilMoistureTarget,
    'sm': SoilMoistureTarget,
    'sm_point': SoilMoistureTarget,
    'sm_smap': SoilMoistureTarget,
    'sm_ismn': SoilMoistureTarget,
    'sm_esa': SoilMoistureTarget,
    'tws': TWSTarget,
    'stor_grace': TWSTarget,
    'stor_mb': TWSTarget,
    'multivariate': MultivariateTarget,
    'glacier_mb': GlacierMBTarget,
    'basin_swe': BasinSWETarget,
    'basin_sca': BasinSCATarget,
    'basin_et': BasinETTarget,
}

# Model-specific target fallbacks as exported names, imported only if the
# registry lookup misses (e.g. the model package's decorator has not fired).
_MODEL_SPECIFIC_TARGETS: Dict[str, Dict[str, str]] = {
    'SUMMA': {
        'streamflow': 'SUMMAStreamflowTarget',
        'snow': 'SUMMASnowTarget',
        'et': 'SUMMAETTarget',
    },
    'FUSE': {
        'streamflow': 'FUSEStreamflowTarget',
        'snow': 'FUSESnowTarget',
    },
    'NGEN': {
        'streamflow': 'NgenStreamflowTarget',
    },
    'GR': {
        'streamflow': 'GRStreamflowTarget',
    },
    'HYPE': {
        'streamflow': 'HYPEStreamflowTarget',
    },
    'RHESSYS': {
        'streamflow': 'RHESSysStreamflowTarget',
    },
}


def _resolve_model_target(model_key: str, target_key: str) -> Optional[Type[CalibrationTarget]]:
    """Resolve a model-specific fallback target class, importing lazily."""
    export_name = _MODEL_SPECIFIC_TARGETS.get(model_key, {}).get(target_key)
    if export_name is None:
        return None
    try:
        return __getattr__(export_name)
    except (ImportError, AttributeError):
        return None


def create_calibration_target(
    model_name: str,
    target_type: str,
    config: Dict[str, Any],
    project_dir: Path,
    logger: logging.Logger
) -> CalibrationTarget:
    """
    Factory function to create calibration targets using registry with fallback.

    This function provides a centralized way to create calibration targets:
    1. First checks OptimizerRegistry for registered model-specific targets
    2. Falls back to model-specific target mappings (lazily imported)
    3. Falls back to default (model-agnostic) targets

    Args:
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'NGEN')
        target_type: Type of calibration target (e.g., 'streamflow', 'snow', 'et')
        config: Configuration dictionary
        project_dir: Path to project directory
        logger: Logger instance

    Returns:
        Instantiated calibration target

    Raises:
        ValueError: If no suitable target class is found

    Example:
        >>> target = create_calibration_target(
        ...     model_name='SUMMA',
        ...     target_type='streamflow',
        ...     config=config,
        ...     project_dir=project_dir,
        ...     logger=logger
        ... )
    """
    from symfluence.core.registries import R

    model_key = model_name.upper()
    target_key = target_type.lower()

    # 1. Try registry first (for dynamically registered targets)
    target_cls = R.calibration_targets.get(f"{model_key}_{target_key.upper()}")

    # 2. Try model-specific mapping (lazy import from the model package)
    if target_cls is None:
        target_cls = _resolve_model_target(model_key, target_key)

    # 3. Fall back to default targets
    if target_cls is None:
        target_cls = _DEFAULT_TARGETS.get(target_key)

    if target_cls is None:
        available = list(_DEFAULT_TARGETS.keys())
        raise ValueError(
            f"No calibration target found for model='{model_name}', type='{target_type}'. "
            f"Available target types: {available}"
        )

    logger.debug(f"Creating calibration target: {target_cls.__name__} for {model_name}/{target_type}")
    return target_cls(config, project_dir, logger)


def get_available_target_types(model_name: Optional[str] = None) -> list:
    """
    Get available calibration target types.

    Args:
        model_name: Optional model name to get model-specific targets

    Returns:
        List of available target type names
    """
    targets = set(_DEFAULT_TARGETS.keys())

    if model_name:
        model_key = model_name.upper()
        if model_key in _MODEL_SPECIFIC_TARGETS:
            targets.update(_MODEL_SPECIFIC_TARGETS[model_key].keys())

    return sorted(targets)


__all__ = [
    # Base targets
    'CalibrationTarget',
    'ETTarget',
    'StreamflowTarget',
    'SoilMoistureTarget',
    'SnowTarget',
    'GroundwaterTarget',
    'TWSTarget',
    'MultivariateTarget',
    'GlacierMBTarget',
    'BasinSWETarget',
    'BasinSCATarget',
    'BasinETTarget',
    # SUMMA targets
    'SUMMAStreamflowTarget',
    'SUMMASnowTarget',
    'SUMMAETTarget',
    # Other model-specific targets
    'GRStreamflowTarget',
    'HYPEStreamflowTarget',
    'RHESSysStreamflowTarget',
    'NgenStreamflowTarget',
    'FUSEStreamflowTarget',
    'FUSESnowTarget',
    # Factory functions
    'create_calibration_target',
    'get_available_target_types',
]
