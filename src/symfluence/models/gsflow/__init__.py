# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GSFLOW (coupled PRMS + MODFLOW-NWT) Hydrological Model.

GSFLOW is a USGS coupled groundwater-surface-water model that integrates
PRMS (surface/soil processes) with MODFLOW-NWT (saturated zone) via SFR
and UZF packages for bidirectional exchange.

Supports three operation modes:
- PRMS: Surface processes only
- MODFLOW: Groundwater only
- COUPLED: Full bidirectional PRMS↔MODFLOW-NWT exchange (default)

References:
    Markstrom, S.L., et al. (2008): GSFLOW---Coupled Ground-Water and
    Surface-Water Flow Model. USGS Techniques and Methods 6-D1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'GSFLOWPreProcessor': ('.preprocessor', 'GSFLOWPreProcessor'),
    'GSFLOWRunner': ('.runner', 'GSFLOWRunner'),
    'GSFLOWResultExtractor': ('.extractor', 'GSFLOWResultExtractor'),
    'GSFLOWPostProcessor': ('.postprocessor', 'GSFLOWPostProcessor'),
    'GSFLOWModelOptimizer': ('.calibration', 'GSFLOWModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for GSFLOW module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['GSFLOWConfigAdapter'])


__all__ = [
    "GSFLOWPreProcessor",
    "GSFLOWRunner",
    "GSFLOWResultExtractor",
    "GSFLOWPostProcessor",
    "GSFLOWConfigAdapter",
]

# Register all GSFLOW components via unified registry
from symfluence.core.registry import model_manifest

from .config import GSFLOWConfigAdapter


def register() -> None:
    """Register GSFLOW components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "GSFLOW",
        config_adapter=GSFLOWConfigAdapter,
        build_instructions_module="symfluence.models.gsflow.build_instructions",
    )
    base = 'symfluence.models.gsflow'
    R.preprocessors.add_lazy("GSFLOW", f"{base}.preprocessor.GSFLOWPreProcessor")
    R.runners.add_lazy("GSFLOW", f"{base}.runner.GSFLOWRunner")
    R.postprocessors.add_lazy("GSFLOW", f"{base}.postprocessor.GSFLOWPostProcessor")
    R.result_extractors.add_lazy("GSFLOW", f"{base}.extractor.GSFLOWResultExtractor")
    R.output_adapters.add_lazy("GSFLOW", "symfluence.models.prms.output_adapter.GSFLOWOutputAdapter")
    R.optimizers.add_lazy("GSFLOW", f"{base}.calibration.optimizer.GSFLOWModelOptimizer")
    R.workers.add_lazy("GSFLOW", f"{base}.calibration.worker.GSFLOWWorker")
    R.parameter_managers.add_lazy("GSFLOW", f"{base}.calibration.parameter_manager.GSFLOWParameterManager")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "GSFLOW",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED},
            default_mode=SpatialMode.SEMI_DISTRIBUTED,
            requires_routing={
                SpatialMode.SEMI_DISTRIBUTED: False,  # Internal SFR routing
                SpatialMode.LUMPED: False,
            },
            warning_message=(
                "GSFLOW couples PRMS surface processes with MODFLOW-NWT groundwater. "
                "Internal SFR/UZF packages handle GW-SW exchange."
            ),
        ),
    )

    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .calibration import GSFLOWModelOptimizer
    from .extractor import GSFLOWResultExtractor
    from .postprocessor import GSFLOWPostProcessor
    from .preprocessor import GSFLOWPreProcessor
    from .runner import GSFLOWRunner
