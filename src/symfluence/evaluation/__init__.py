# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Evaluation module for SYMFLUENCE model performance assessment.

This package provides tools for evaluating hydrological model outputs against
observational data, including streamflow, snow, soil moisture, and other
environmental variables.

Key components:
    EvaluationRegistry: Central registry for evaluation configurations
    AnalysisRegistry: Registry for analysis types and methods
    BaseStructureEnsembleAnalyzer: Multi-model ensemble analysis
    OutputFileLocator: Utility for locating model output files
    likelihood: Gaussian log-likelihood with observation uncertainty support

Example:
    >>> from symfluence.evaluation import EvaluationRegistry
    >>> registry = EvaluationRegistry()
    >>> registry.register_evaluator('streamflow', streamflow_evaluator)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy re-exports (PEP 562): the evaluators subpackage pulls the observation
# and geospatial stacks, and this package is reached from optimizer/worker
# base modules at `import symfluence`. Evaluator lookups still work without
# importing anything here — the registry bootstrap installs a seeder on
# R.evaluators that imports `.evaluators` on first access.
_LAZY_IMPORTS = {
    'evaluators': ('.evaluators', None),
    'AnalysisRegistry': ('.analysis_registry', 'AnalysisRegistry'),
    'KoopmanAnalyzer': ('.koopman_analysis', 'KoopmanAnalyzer'),
    'MetricTransformer': ('symfluence.core.metrics.metric_transformer', 'MetricTransformer'),
    'OutputFileLocator': ('symfluence.core.modeling.output_file_locator', 'OutputFileLocator'),
    'get_output_file_locator': ('symfluence.core.modeling.output_file_locator', 'get_output_file_locator'),
    'EvaluationRegistry': ('symfluence.core.modeling.evaluation_registry', 'EvaluationRegistry'),
    'BaseStructureEnsembleAnalyzer': ('symfluence.core.modeling.structure_ensemble', 'BaseStructureEnsembleAnalyzer'),
}


def __getattr__(name: str):
    """Lazy import handler for evaluation re-exports."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY_IMPORTS.keys())


if TYPE_CHECKING:
    from symfluence.core.metrics.metric_transformer import MetricTransformer
    from symfluence.core.modeling.evaluation_registry import EvaluationRegistry
    from symfluence.core.modeling.output_file_locator import OutputFileLocator, get_output_file_locator
    from symfluence.core.modeling.structure_ensemble import BaseStructureEnsembleAnalyzer

    from . import evaluators
    from .analysis_registry import AnalysisRegistry
    from .koopman_analysis import KoopmanAnalyzer

__all__ = [
    "EvaluationRegistry",
    "AnalysisRegistry",
    "evaluators",
    "BaseStructureEnsembleAnalyzer",
    "OutputFileLocator",
    "get_output_file_locator",
    "MetricTransformer",
    "KoopmanAnalyzer",
]
