# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Identity guards for Stage 1 contract promotions."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_core_metrics_facade_exports_promoted_classes():
    from symfluence.core.metrics import MetricTransformer, StreamflowMetrics
    from symfluence.core.metrics.metric_transformer import MetricTransformer as CanonicalTransformer
    from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics as CanonicalStreamflowMetrics

    assert MetricTransformer is CanonicalTransformer
    assert StreamflowMetrics is CanonicalStreamflowMetrics


def test_netcdf_utility_compatibility_path_preserves_identity():
    from symfluence.core.modeling.netcdf_utils import create_minimal_encoding as canonical_minimal
    from symfluence.core.modeling.netcdf_utils import create_netcdf_encoding as canonical_encoding
    from symfluence.data.utils.netcdf_utils import create_minimal_encoding, create_netcdf_encoding

    assert create_minimal_encoding is canonical_minimal
    assert create_netcdf_encoding is canonical_encoding


def test_model_ready_attribute_compatibility_paths_preserve_identity():
    from symfluence.core.modeling.model_ready import AttributesReader as CanonicalReader
    from symfluence.core.modeling.model_ready import open_canonical_attributes as canonical_open
    from symfluence.core.modeling.model_ready import resolve_model_ready_path as canonical_resolve
    from symfluence.data.model_ready import (
        AttributesReader,
        open_canonical_attributes,
        resolve_model_ready_path,
    )

    assert AttributesReader is CanonicalReader
    assert open_canonical_attributes is canonical_open
    assert resolve_model_ready_path is canonical_resolve


def test_evaluation_contract_compatibility_paths_preserve_identity():
    from symfluence.core.modeling.evaluation_registry import EvaluationRegistry as CanonicalRegistry
    from symfluence.core.modeling.structure_ensemble import (
        BaseStructureEnsembleAnalyzer as CanonicalAnalyzer,
    )
    from symfluence.evaluation import BaseStructureEnsembleAnalyzer, EvaluationRegistry

    assert EvaluationRegistry is CanonicalRegistry
    assert BaseStructureEnsembleAnalyzer is CanonicalAnalyzer


def test_coupling_capability_compatibility_path_preserves_identity():
    from symfluence.core.modeling.coupling import INSTALL_SUGGESTION as canonical_suggestion
    from symfluence.core.modeling.coupling import is_dcoupler_available as canonical_available
    from symfluence.coupling import INSTALL_SUGGESTION, is_dcoupler_available

    assert INSTALL_SUGGESTION == canonical_suggestion
    assert is_dcoupler_available is canonical_available


def test_calibration_target_facade_resolves_legacy_classes():
    from symfluence.core.calibration.targets import resolve_calibration_target
    from symfluence.optimization.calibration_targets import SnowTarget, StreamflowTarget, TWSTarget

    assert resolve_calibration_target('discharge') is StreamflowTarget
    assert resolve_calibration_target('swe') is SnowTarget
    assert resolve_calibration_target('grace_tws') is TWSTarget
    assert resolve_calibration_target('unknown') is StreamflowTarget


def test_reporting_contract_compatibility_paths_preserve_identity():
    from symfluence.core.reporting import BasePlotter as CanonicalPlotter
    from symfluence.core.reporting import PlotConfig as CanonicalConfig
    from symfluence.core.reporting import calculate_metrics as canonical_metrics
    from symfluence.core.reporting import resolve_default_name as canonical_resolve
    from symfluence.reporting.config.plot_config import PlotConfig
    from symfluence.reporting.core.base_plotter import BasePlotter
    from symfluence.reporting.core.plot_utils import calculate_metrics
    from symfluence.reporting.core.shapefile_helper import resolve_default_name

    assert BasePlotter is CanonicalPlotter
    assert PlotConfig is CanonicalConfig
    assert calculate_metrics is canonical_metrics
    assert resolve_default_name is canonical_resolve


def test_evaluator_contract_compatibility_paths_preserve_identity():
    from symfluence.core.modeling.evaluators import ETEvaluator as CanonicalET
    from symfluence.core.modeling.evaluators import ModelEvaluator as CanonicalBase
    from symfluence.core.modeling.evaluators import StreamflowEvaluator as CanonicalStreamflow
    from symfluence.core.modeling.evaluators import TWSEvaluator as CanonicalTWS
    from symfluence.evaluation.evaluators import (
        ETEvaluator,
        ModelEvaluator,
        StreamflowEvaluator,
        TWSEvaluator,
    )

    assert ETEvaluator is CanonicalET
    assert ModelEvaluator is CanonicalBase
    assert StreamflowEvaluator is CanonicalStreamflow
    assert TWSEvaluator is CanonicalTWS


def test_observation_path_compatibility_preserves_identity():
    from symfluence.core.modeling.observation_paths import (
        streamflow_observation_candidates as canonical_candidates,
    )
    from symfluence.data.observation.paths import streamflow_observation_candidates

    assert streamflow_observation_candidates is canonical_candidates


def test_likelihood_compatibility_path_preserves_identity():
    from symfluence.core.metrics.likelihood import gaussian_log_likelihood as canonical
    from symfluence.evaluation.likelihood import gaussian_log_likelihood

    assert gaussian_log_likelihood is canonical


def test_regionalization_compatibility_paths_preserve_identity():
    from symfluence.core.calibration.regionalization.strategies import (
        RegionalizationFactory as CanonicalFactory,
    )
    from symfluence.core.calibration.regionalization.transfer_functions import (
        LinearTF as CanonicalLinearTF,
    )
    from symfluence.optimization.regionalization import LinearTF, RegionalizationFactory

    assert RegionalizationFactory is CanonicalFactory
    assert LinearTF is CanonicalLinearTF
