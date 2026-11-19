# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SUMMA (Structure for Unifying Multiple Modeling Alternatives) Hydrological Model.

This module implements SUMMA, a unified framework for process-based hydrological
modeling that enables systematic exploration of different model representations.
SUMMA allows users to select from multiple physically-based options for each
hydrological process, generating 200+ unique model configurations.

Model Architecture:
    SUMMA uses a layered approach with configurable process representations:

    1. **Canopy Processes**: Interception, throughfall, canopy snow
       - Options: Big-leaf, two-stream radiation, CLM-style

    2. **Snow Processes**: Accumulation, metamorphism, melt
       - Options: Temperature index, energy balance, layered snow

    3. **Soil Processes**: Infiltration, percolation, drainage
       - Options: Richards equation, simplified bucket, Green-Ampt

    4. **Groundwater**: Baseflow generation, aquifer dynamics
       - Options: TOPMODEL, linear reservoir, power-law

    5. **Runoff Generation**: Surface and subsurface routing
       - Options: Saturation excess, infiltration excess, variable area

Design Rationale:
    SUMMA addresses model structural uncertainty systematically:
    - Most models hard-code process representations
    - SUMMA exposes alternatives as runtime decisions
    - Enables hypothesis testing across process formulations
    - Reduces need for multiple model codebases
    - Supports ensemble modeling with structural uncertainty

Spatial Structure:
    - GRU (Grouped Response Unit): Routing unit containing multiple HRUs
    - HRU (Hydrologic Response Unit): Fundamental computational unit
    - Layers: Vertical discretization for snow and soil

Key Components:
    SummaPreProcessor: Forcing preparation, attributes, trial parameters
    SummaRunner: Model execution with parallel support (summa_actors)
    SUMMAPostProcessor: Output extraction and NetCDF processing
    SummaStructureAnalyzer: Decision ensemble analysis
    SummaForcingProcessor: Forcing file preparation
    SummaConfigManager: Configuration file generation
    SummaAttributesManager: HRU attribute management

Configuration Parameters:
    SETTINGS_SUMMA_CONNECT_HRUS: Enable lateral HRU connectivity (default: True)
    SUMMA_DECISION_OPTIONS: Dictionary of decision choices for ensemble runs
    SETTINGS_SUMMA_GLACIER_MODE: Enable glacier dynamics (default: False)
    SETTINGS_SUMMA_USE_PARALLEL_SUMMA: Use parallel execution (default: False)
    SETTINGS_SUMMA_PARALLEL_BACKEND: Parallel backend (default: 'slurm')
    PARAMS_TO_CALIBRATE: Local parameters
        (default: 'albedo_max,albedo_min,canopy_capacity,slow_drainage')
    BASIN_PARAMS_TO_CALIBRATE: Basin-scale routing parameters
        (default: 'routingGammaShape,routingGammaScale')

Typical Workflow:
    1. Initialize SummaPreProcessor with configuration
    2. Process forcing data via SummaForcingProcessor
    3. Generate attributes and trial parameters via managers
    4. Create file manager and decision files
    5. Execute SUMMA (serial or parallel) via SummaRunner
    6. Extract results and analyze decisions via SUMMAPostProcessor

Limitations and Considerations:
    - Requires SUMMA executable (compiled with Sundials solver recommended)
    - Decision ensemble runs multiply computational cost
    - Glacier mode requires additional attribute preparation
    - Large domains benefit from parallel execution (summa_actors)
    - Some decision combinations may be incompatible or unstable
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution, analysis, and plotting classes pull the
# geospatial/matplotlib stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'SummaAttributesManager': ('.attributes_manager', 'SummaAttributesManager'),
    'SummaConfigManager': ('.config_manager', 'SummaConfigManager'),
    'SummaForcingProcessor': ('.forcing_processor', 'SummaForcingProcessor'),
    'SUMMAPostProcessor': ('.postprocessor', 'SUMMAPostProcessor'),
    'SummaPreProcessor': ('.preprocessor', 'SummaPreProcessor'),
    'SummaRunner': ('.runner', 'SummaRunner'),
    'SummaStructureAnalyzer': ('.structure_analyzer', 'SummaStructureAnalyzer'),
    'visualize_summa': ('.visualizer', 'visualize_summa'),
    'SUMMAResultExtractor': ('.extractor', 'SUMMAResultExtractor'),
    'SUMMAPlotter': ('.plotter', 'SUMMAPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for SUMMA module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['SUMMAConfigAdapter'])


__all__ = [
    'SummaPreProcessor',
    'SummaRunner',
    'SUMMAPostProcessor',
    'SummaStructureAnalyzer',
    'SummaForcingProcessor',
    'SummaConfigManager',
    'SummaAttributesManager'
]

# Register all SUMMA components via unified registry
from symfluence.core.registry import model_manifest

from .config import SUMMAConfigAdapter


def register() -> None:
    """Register SUMMA components with the unified registry.

    Execution, extraction, analysis, and plotting classes are registered
    lazily — imported on first registry access rather than at
    plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "SUMMA",
        config_adapter=SUMMAConfigAdapter,
        build_instructions_module="symfluence.models.summa.build_instructions",
    )
    base = 'symfluence.models.summa'
    R.preprocessors.add_lazy("SUMMA", f"{base}.preprocessor.SummaPreProcessor")
    R.base_settings.add("SUMMA", "symfluence.models.summa")
    R.runners.add_lazy("SUMMA", f"{base}.runner.SummaRunner", runner_method='run_summa')
    R.postprocessors.add_lazy("SUMMA", f"{base}.postprocessor.SUMMAPostProcessor")
    R.visualizers.add_lazy("SUMMA", f"{base}.visualizer.visualize_summa")
    R.result_extractors.add_lazy("SUMMA", f"{base}.extractor.SUMMAResultExtractor")
    R.output_adapters.add_lazy("SUMMA", f"{base}.output_adapter.SUMMAOutputAdapter")
    R.decision_analyzers.add_lazy("SUMMA", f"{base}.structure_analyzer.SummaStructureAnalyzer")
    R.plotters.add_lazy("SUMMA", f"{base}.plotter.SUMMAPlotter")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "SUMMA",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.DISTRIBUTED,
            requires_routing={
                SpatialMode.DISTRIBUTED: True,
                SpatialMode.SEMI_DISTRIBUTED: True,
                SpatialMode.LUMPED: False,
            },
            warning_message=None,
        ),
    )


if TYPE_CHECKING:
    from .attributes_manager import SummaAttributesManager
    from .config_manager import SummaConfigManager
    from .extractor import SUMMAResultExtractor
    from .forcing_processor import SummaForcingProcessor
    from .plotter import SUMMAPlotter
    from .postprocessor import SUMMAPostProcessor
    from .preprocessor import SummaPreProcessor
    from .runner import SummaRunner
    from .structure_analyzer import SummaStructureAnalyzer
    from .visualizer import visualize_summa
