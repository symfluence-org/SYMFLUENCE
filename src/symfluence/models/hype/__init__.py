# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE (HYdrological Predictions for the Environment) Model.

This module implements the HYPE semi-distributed process-based hydrological model
developed by SMHI (Swedish Meteorological and Hydrological Institute). HYPE is
designed for large-scale operational hydrological prediction and has been applied
from catchment to continental scales (e.g., E-HYPE covering all of Europe).

Model Architecture:
    1. **Spatial Discretization**: Subbasins containing Soil-Land Classes (SLCs)
       that combine soil type and land use for parameter regionalization
    2. **Snow Processes**: Degree-day snowmelt with liquid water refreezing
    3. **Soil Moisture**: Multi-layer soil model with infiltration and percolation
    4. **Evapotranspiration**: Penman-Monteith or simpler temperature-based methods
    5. **Groundwater**: Upper and lower groundwater boxes with regional flow
    6. **Routing**: Internal subbasin routing with river delay and dampening

Design Rationale:
    HYPE addresses large-scale operational prediction needs:
    - SLC-based parameterization enables parameter transfer to ungauged basins
    - Process-based structure supports scenario analysis (land use, climate)
    - Proven operational use in national flood forecasting services
    - Supports multiple output types (water balance, nutrients, loads)

Spatial Structure:
    - Subbasins: Hydrological response units for routing
    - SLCs: Soil-land class combinations within each subbasin
    - Outlets: Defined pour points for streamflow comparison

Key Components:
    HYPEPreProcessor: Orchestrates preprocessing pipeline
    HYPERunner: Model execution and simulation management
    HYPEPostProcessor: Output extraction and analysis
    HYPEForcingProcessor: Forcing data conversion (hourly to daily aggregation)
    HYPEConfigManager: Configuration file generation (info.txt, par.txt, filedir.txt)
    HYPEGeoDataManager: Geographic data files (GeoData.txt, GeoClass.txt, ForcKey.txt)

Configuration Parameters:
    HYPE_SPINUP_DAYS: Model spinup period in days (default: from SPINUP_PERIOD, else 365)
    SETTINGS_HYPE_INFO: Info file name (default: 'info.txt')
    HYPE_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp')
        ttmp: Temperature threshold for snow/rain
        cmlt: Degree-day snowmelt factor
        cevp: Evapotranspiration coefficient
        lp: Soil moisture threshold for ET reduction
        rrcs1/rrcs2: Recession coefficients for upper/lower response
        rcgrw: Regional groundwater flow coefficient
        rivvel: River routing velocity
        damp: River routing dampening

Typical Workflow:
    1. Initialize HYPEPreProcessor with configuration
    2. Process forcing data via HYPEForcingProcessor (temporal aggregation)
    3. Generate geographic data files via HYPEGeoDataManager
    4. Create configuration files via HYPEConfigManager
    5. Execute HYPE via HYPERunner
    6. Extract results via HYPEPostProcessor

Limitations and Considerations:
    - Requires HYPE executable (compiled from source or from SMHI)
    - SLC delineation requires soil and land use spatial data
    - Daily timestep is standard; sub-daily requires special configuration
    - Spinup period needed to initialize soil moisture and groundwater states
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution, extraction, and plotting classes pull the
# geospatial/matplotlib stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'HYPEConfigManager': ('.config_manager', 'HYPEConfigManager'),
    'HYPEForcingProcessor': ('.forcing_processor', 'HYPEForcingProcessor'),
    'HYPEGeoDataManager': ('.geodata_manager', 'HYPEGeoDataManager'),
    'HYPEPostProcessor': ('.postprocessor', 'HYPEPostProcessor'),
    'HYPEPreProcessor': ('.preprocessor', 'HYPEPreProcessor'),
    'HYPERunner': ('.runner', 'HYPERunner'),
    'visualize_hype': ('.visualizer', 'visualize_hype'),
    'HYPEResultExtractor': ('.extractor', 'HYPEResultExtractor'),
    'HYPEPlotter': ('.plotter', 'HYPEPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for HYPE module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['HYPEConfigAdapter'])


__all__ = [
    'HYPEPreProcessor',
    'HYPERunner',
    'HYPEPostProcessor',
    'visualize_hype',
    'HYPEForcingProcessor',
    'HYPEConfigManager',
    'HYPEGeoDataManager',
]

# Register all HYPE components via unified registry
from symfluence.core.registry import model_manifest

from .config import HYPEConfigAdapter


def register() -> None:
    """Register HYPE components with the unified registry.

    Execution, extraction, and plotting classes are registered lazily —
    imported on first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "HYPE",
        config_adapter=HYPEConfigAdapter,
        build_instructions_module="symfluence.models.hype.build_instructions",
    )
    base = 'symfluence.models.hype'
    R.preprocessors.add_lazy("HYPE", f"{base}.preprocessor.HYPEPreProcessor")
    R.runners.add_lazy("HYPE", f"{base}.runner.HYPERunner")
    R.postprocessors.add_lazy("HYPE", f"{base}.postprocessor.HYPEPostProcessor")
    R.visualizers.add_lazy("HYPE", f"{base}.visualizer.visualize_hype")
    R.result_extractors.add_lazy("HYPE", f"{base}.extractor.HYPEResultExtractor")
    R.output_adapters.add_lazy("HYPE", f"{base}.output_adapter.HYPEOutputAdapter")
    R.plotters.add_lazy("HYPE", f"{base}.plotter.HYPEPlotter")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "HYPE",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.SEMI_DISTRIBUTED,
            requires_routing={
                # HYPE has internal routing.
                SpatialMode.DISTRIBUTED: False,
                SpatialMode.SEMI_DISTRIBUTED: False,
                SpatialMode.LUMPED: False,
            },
            warning_message=None,
        ),
    )

    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .config_manager import HYPEConfigManager
    from .extractor import HYPEResultExtractor
    from .forcing_processor import HYPEForcingProcessor
    from .geodata_manager import HYPEGeoDataManager
    from .plotter import HYPEPlotter
    from .postprocessor import HYPEPostProcessor
    from .preprocessor import HYPEPreProcessor
    from .runner import HYPERunner
    from .visualizer import visualize_hype
