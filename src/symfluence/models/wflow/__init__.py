# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow (wflow_sbm) Distributed Hydrological Model."""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'WflowPreProcessor': ('.preprocessor', 'WflowPreProcessor'),
    'WflowRunner': ('.runner', 'WflowRunner'),
    'WflowResultExtractor': ('.extractor', 'WflowResultExtractor'),
    'WflowPostProcessor': ('.postprocessor', 'WflowPostProcessor'),
    'WflowModelOptimizer': ('.calibration', 'WflowModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for Wflow module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['WflowConfigAdapter'])


__all__ = [
    "WflowPreProcessor",
    "WflowRunner",
    "WflowResultExtractor",
    "WflowPostProcessor",
    "WflowConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import WflowConfigAdapter


def register() -> None:
    """Register WFLOW components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "WFLOW",
        config_adapter=WflowConfigAdapter,
        build_instructions_module="symfluence.models.wflow.build_instructions",
    )
    base = 'symfluence.models.wflow'
    R.preprocessors.add_lazy("WFLOW", f"{base}.preprocessor.WflowPreProcessor")
    R.runners.add_lazy("WFLOW", f"{base}.runner.WflowRunner")
    R.postprocessors.add_lazy("WFLOW", f"{base}.postprocessor.WflowPostProcessor")
    R.result_extractors.add_lazy("WFLOW", f"{base}.extractor.WflowResultExtractor")
    R.output_adapters.add_lazy("WFLOW", f"{base}.output_adapter.WflowOutputAdapter")
    R.optimizers.add_lazy("WFLOW", f"{base}.calibration.optimizer.WflowModelOptimizer")
    R.workers.add_lazy("WFLOW", f"{base}.calibration.worker.WflowWorker")
    R.parameter_managers.add_lazy("WFLOW", f"{base}.calibration.parameter_manager.WflowParameterManager")


if TYPE_CHECKING:
    from .calibration import WflowModelOptimizer
    from .extractor import WflowResultExtractor
    from .postprocessor import WflowPostProcessor
    from .preprocessor import WflowPreProcessor
    from .runner import WflowRunner
