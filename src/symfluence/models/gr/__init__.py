# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GR (Génie Rural) Family of Conceptual Hydrological Models.

This module implements the GR family of rainfall-runoff models developed by INRAE
(formerly Cemagref/IRSTEA). The GR models are parsimonious conceptual models known
for excellent performance relative to their simplicity.

Model Variants:
    GR4J: 4-parameter daily model (production store, routing store, unit hydrographs)
    GR5J: 5-parameter version with inter-catchment groundwater exchange
    GR6J: 6-parameter version with additional exponential store

    All variants can be coupled with CemaNeige, a 2-parameter snow module using
    degree-day snowmelt with cold content accounting.

Model Architecture:
    1. **Interception**: Neutralization of P and PE via threshold
    2. **Production Store**: Soil moisture accounting (parameter X1 = capacity)
    3. **Percolation**: Leakage from production store to routing
    4. **Unit Hydrographs**: Two UHs split flow (90%/10%) with X4 time base
    5. **Routing Store**: Nonlinear reservoir (X3 = capacity) with exchange (X2)
    6. **CemaNeige** (optional): Degree-day snow with CTG and Kf parameters

Design Rationale:
    The GR approach emphasizes parsimony and robustness:
    - Few parameters reduce equifinality and overfitting risk
    - Proven performance across diverse climates and catchment sizes
    - R-based implementation via airGR package for reliability
    - CemaNeige provides simple but effective snow representation

Spatial Modes:
    lumped: Single GRU for entire catchment
    distributed: Multiple subcatchments with optional mizuRoute routing
    auto: Automatically detect based on domain configuration

Key Components:
    GRPreProcessor: Forcing preparation, catchment setup, R script generation
    GRRunner: Model execution via Rscript with airGR/airGRdatassim packages
    GRPostProcessor: Output parsing, result formatting, metric calculation

Configuration Parameters:
    GR_SPATIAL_MODE: Spatial setup ('auto', 'lumped', 'distributed')
    GR_ROUTING_INTEGRATION: Routing model ('none', 'mizuroute')
    GR_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff')
        X1: Production store capacity (mm)
        X2: Groundwater exchange coefficient (mm/day)
        X3: Routing store capacity (mm)
        X4: Unit hydrograph time base (days)
        CTG: CemaNeige degree-day factor
        Kf: CemaNeige cold content factor

Typical Workflow:
    1. Initialize GRPreProcessor with configuration
    2. Process forcing data (precipitation, temperature, PET)
    3. Generate R control script and input CSV files
    4. Execute GR via GRRunner (calls Rscript)
    5. Parse outputs and format results via GRPostProcessor

Limitations and Considerations:
    - Requires R with airGR/airGRdatassim packages installed
    - CemaNeige is daily timestep only (no sub-daily snow)
    - Parameter bounds should respect physical constraints (X1, X3 > 0)
    - For distributed mode, subcatchment areas needed for routing
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and extraction classes pull the
# geospatial/observation stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'GRPostProcessor': ('.postprocessor', 'GRPostProcessor'),
    'GRPreProcessor': ('.preprocessor', 'GRPreProcessor'),
    'GRRunner': ('.runner', 'GRRunner'),
    'visualize_gr': ('.visualizer', 'visualize_gr'),
    'GRResultExtractor': ('.extractor', 'GRResultExtractor'),
}


def __getattr__(name: str):
    """Lazy import handler for GR module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['GRConfigAdapter'])


__all__ = [
    'GRPreProcessor',
    'GRRunner',
    'GRPostProcessor',
    'visualize_gr'
]


# Register all GR components via unified registry
from symfluence.core.registry import model_manifest

from .config import GRConfigAdapter


def register() -> None:
    """Register GR components with the unified registry.

    Execution and extraction classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "GR",
        config_adapter=GRConfigAdapter,
    )
    base = 'symfluence.models.gr'
    R.preprocessors.add_lazy("GR", f"{base}.preprocessor.GRPreProcessor")
    R.runners.add_lazy("GR", f"{base}.runner.GRRunner", runner_method='run_gr')
    R.postprocessors.add_lazy("GR", f"{base}.postprocessor.GRPostProcessor")
    R.visualizers.add_lazy("GR", f"{base}.visualizer.visualize_gr")
    R.result_extractors.add_lazy("GR", f"{base}.extractor.GRResultExtractor")
    R.output_adapters.add_lazy("GR", f"{base}.output_adapter.GROutputAdapter")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "GR",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.LUMPED,
            requires_routing={
                SpatialMode.DISTRIBUTED: True,
                SpatialMode.SEMI_DISTRIBUTED: True,
                SpatialMode.LUMPED: False,
            },
            warning_message=None,
        ),
    )

    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .extractor import GRResultExtractor
    from .postprocessor import GRPostProcessor
    from .preprocessor import GRPreProcessor
    from .runner import GRRunner
    from .visualizer import visualize_gr
