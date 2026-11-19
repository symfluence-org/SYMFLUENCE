# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mHM (mesoscale Hydrological Model).

This module implements mHM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install mhm`
- Preprocessing (forcing, morphology, namelists)
- Model execution (Fortran binary)
- Result extraction
- Calibration support

mHM is a spatially distributed hydrological model developed at the
Helmholtz Centre for Environmental Research (UFZ). It uses Multiscale
Parameter Regionalization (MPR) for parameter transfer across scales.

Model Architecture:
    mHM uses a grid-based structure with:

    1. **Forcing Files**: NetCDF grids with meteorological data
       - Precipitation (pre), temperature (tavg), PET (pet)

    2. **Morphological Inputs**: DEM, soil, land cover grids
       - Used by MPR for parameter regionalization

    3. **Namelist Files**: Fortran namelists controlling simulation
       - mhm.nml: Main model configuration
       - mrm.nml: Routing (mRM) configuration

    4. **Output Files**: NetCDF files with results
       - discharge_*.nc: Simulated discharge [m3/s]
       - mHM_Fluxes_States_*.nc: Fluxes and states

Design Rationale:
    mHM is well-suited for:
    - Mesoscale hydrological modeling
    - Parameter regionalization studies
    - Multi-basin hydrological assessment
    - Spatially distributed process modeling

Key Components:
    MHMPreProcessor: Forcing, morphology, and namelist generation
    MHMRunner: Model execution with Fortran binary
    MHMResultExtractor: Output extraction and analysis

Configuration Parameters:
    MHM_INSTALL_PATH: Path to mHM installation
    MHM_EXE: Executable name (default: mhm)
    MHM_NAMELIST_FILE: Main namelist (default: mhm.nml)
    MHM_ROUTING_NAMELIST: Routing namelist (default: mrm.nml)
    MHM_SPATIAL_MODE: 'lumped' or 'distributed'
    MHM_PARAMS_TO_CALIBRATE: Calibration parameters

Typical Workflow:
    1. Prepare forcing data (precipitation, temperature, PET)
    2. Generate morphological inputs (DEM, soil, land cover)
    3. Create Fortran namelists (mhm.nml, mrm.nml)
    4. Run mHM binary from settings directory
    5. Extract and analyze discharge and fluxes/states

References:
    Samaniego, L., et al. (2010): Multiscale parameter regionalization
    of a grid-based hydrologic model at the mesoscale. Water Resources
    Research, 46, W05523.

    Kumar, R., et al. (2013): Toward computationally efficient large-scale
    hydrologic predictions with a multiscale regionalization scheme. Water
    Resources Research, 49, 5700-5714.

    https://git.ufz.de/mhm/mhm
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'MHMPreProcessor': ('.preprocessor', 'MHMPreProcessor'),
    'MHMRunner': ('.runner', 'MHMRunner'),
    'MHMResultExtractor': ('.extractor', 'MHMResultExtractor'),
    'MHMPostProcessor': ('.postprocessor', 'MHMPostProcessor'),
    'MHMModelOptimizer': ('.calibration', 'MHMModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for mHM module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['MHMConfigAdapter'])


__all__ = [
    "MHMPreProcessor",
    "MHMRunner",
    "MHMResultExtractor",
    "MHMPostProcessor",
    "MHMConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import MHMConfigAdapter


def register() -> None:
    """Register MHM components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "MHM",
        config_adapter=MHMConfigAdapter,
        build_instructions_module="symfluence.models.mhm.build_instructions",
    )
    base = 'symfluence.models.mhm'
    R.preprocessors.add_lazy("MHM", f"{base}.preprocessor.MHMPreProcessor")
    R.runners.add_lazy("MHM", f"{base}.runner.MHMRunner")
    R.postprocessors.add_lazy("MHM", f"{base}.postprocessor.MHMPostProcessor")
    R.result_extractors.add_lazy("MHM", f"{base}.extractor.MHMResultExtractor")
    R.output_adapters.add_lazy("MHM", f"{base}.output_adapter.MHMOutputAdapter")
    R.optimizers.add_lazy("MHM", f"{base}.calibration.optimizer.MHMModelOptimizer")
    R.workers.add_lazy("MHM", f"{base}.calibration.worker.MHMWorker")
    R.parameter_managers.add_lazy("MHM", f"{base}.calibration.parameter_manager.MHMParameterManager")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "MHM",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED},
            default_mode=SpatialMode.LUMPED,
            requires_routing={SpatialMode.LUMPED: False},
            warning_message=(
                "mHM is a mesoscale hydrological model. Lumped mode uses "
                "a single-cell domain with multiscale parameter regionalization."
            ),
        ),
    )


if TYPE_CHECKING:
    from .calibration import MHMModelOptimizer
    from .extractor import MHMResultExtractor
    from .postprocessor import MHMPostProcessor
    from .preprocessor import MHMPreProcessor
    from .runner import MHMRunner
