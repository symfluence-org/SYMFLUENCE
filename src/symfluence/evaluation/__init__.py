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
from . import evaluators
from .analysis_registry import AnalysisRegistry
from .koopman_analysis import KoopmanAnalyzer
from .metric_transformer import MetricTransformer
from .output_file_locator import OutputFileLocator, get_output_file_locator
from .registry import EvaluationRegistry
from .structure_ensemble import BaseStructureEnsembleAnalyzer

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
