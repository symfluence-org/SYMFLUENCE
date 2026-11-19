# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PRMS (Precipitation-Runoff Modeling System) Hydrological Model.

This module implements PRMS support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install prms`
- Preprocessing (control file, parameter file, data file)
- Model execution (prms -C control.dat)
- Result extraction (statvar output for streamflow, ET, SWE)
- Calibration support (10 parameters: soil_moist_max, soil_rechr_max, etc.)

PRMS is a deterministic, distributed-parameter, physical-process watershed
model developed by the USGS. It simulates the effects of various
combinations of precipitation, climate, and land use on streamflow,
sediment yields, and general basin hydrology.

Model Architecture:
    PRMS uses an HRU-based semi-distributed structure with:

    1. **Control File**: Master simulation control
       - Simulation dates, module selection, output options
       - File paths for parameter and data files

    2. **Parameter File**: HRU and segment parameters
       - HRU definitions (area, elevation, slope, aspect)
       - Soil parameters (soil_moist_max, soil_rechr_max, etc.)
       - Monthly parameters (jh_coef, rain_adj, snow_adj)

    3. **Data File**: Forcing time series
       - Precipitation, temperature (min/max), solar radiation
       - One column per HRU or station

    4. **Output Files**: Statistic variable output
       - statvar.dat: Selected output variables (streamflow, ET, SWE)
       - CSV or NetCDF format depending on configuration

Design Rationale:
    PRMS is well-suited for:
    - Watershed-scale hydrological modeling
    - Water resource assessment and planning
    - Climate change impact studies
    - National Hydrologic Model (NHM) applications

Key Components:
    PRMSPreProcessor: Control, parameter, and data file generation
    PRMSRunner: Model execution with prms executable
    PRMSResultExtractor: Statvar output extraction
    PRMSPostProcessor: Streamflow extraction and unit handling
    PRMSConfigAdapter: Configuration schema and validation

Configuration Parameters:
    PRMS_INSTALL_PATH: Path to PRMS installation
    PRMS_EXE: Executable name (default: prms)
    PRMS_SPATIAL_MODE: 'semi_distributed' (default)
    PRMS_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'soil_moist_max,soil_rechr_max,tmax_allrain,tmax_allsnow,
                   hru_percent_imperv,carea_max,smidx_coef,slowcoef_lin,
                   gwflow_coef,ssr2gw_rate')

Typical Workflow:
    1. Define HRU structure from catchment delineation
    2. Generate parameter file with soil/vegetation properties
    3. Prepare data file with forcing time series
    4. Create control file with simulation options
    5. Run prms -C control.dat
    6. Extract streamflow from statvar output

References:
    Markstrom, S.L., Regan, R.S., Hay, L.E., Viger, R.J., Webb, R.M.T.,
    Payn, R.A., and LaFontaine, J.H. (2015): PRMS-IV, the Precipitation-
    Runoff Modeling System, Version 4. USGS Techniques and Methods 6-B7.

    https://github.com/nhm-usgs/prms
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'PRMSPreProcessor': ('.preprocessor', 'PRMSPreProcessor'),
    'PRMSRunner': ('.runner', 'PRMSRunner'),
    'PRMSResultExtractor': ('.extractor', 'PRMSResultExtractor'),
    'PRMSPostProcessor': ('.postprocessor', 'PRMSPostProcessor'),
    'PRMSModelOptimizer': ('.calibration', 'PRMSModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for PRMS module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['PRMSConfigAdapter'])


__all__ = [
    "PRMSPreProcessor",
    "PRMSRunner",
    "PRMSResultExtractor",
    "PRMSPostProcessor",
    "PRMSConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import PRMSConfigAdapter


def register() -> None:
    """Register PRMS components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "PRMS",
        config_adapter=PRMSConfigAdapter,
        build_instructions_module="symfluence.models.prms.build_instructions",
    )
    base = 'symfluence.models.prms'
    R.preprocessors.add_lazy("PRMS", f"{base}.preprocessor.PRMSPreProcessor")
    R.runners.add_lazy("PRMS", f"{base}.runner.PRMSRunner")
    R.postprocessors.add_lazy("PRMS", f"{base}.postprocessor.PRMSPostProcessor")
    R.result_extractors.add_lazy("PRMS", f"{base}.extractor.PRMSResultExtractor")
    R.output_adapters.add_lazy("PRMS", f"{base}.output_adapter.PRMSOutputAdapter")
    R.optimizers.add_lazy("PRMS", f"{base}.calibration.optimizer.PRMSModelOptimizer")
    R.workers.add_lazy("PRMS", f"{base}.calibration.worker.PRMSWorker")
    R.parameter_managers.add_lazy("PRMS", f"{base}.calibration.parameter_manager.PRMSParameterManager")


if TYPE_CHECKING:
    from .calibration import PRMSModelOptimizer
    from .extractor import PRMSResultExtractor
    from .postprocessor import PRMSPostProcessor
    from .preprocessor import PRMSPreProcessor
    from .runner import PRMSRunner
