# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Multi-criteria objective function for multi-variable model calibration.

Combines multiple variables and metrics into a single weighted scalar
objective for multi-criteria optimization.

This module implements the MULTIVARIATE objective that allows simultaneous
calibration against multiple data streams (streamflow, snow, evapotranspiration, etc.)
with user-defined weights to reflect relative importance.
"""
from __future__ import annotations

from typing import Dict

from symfluence.core.metrics.metric_transformer import MetricTransformer
from symfluence.core.registries import R

from .base import BaseObjective

#: Default primary metric per variable. All are bias-sensitive: KGE contains
#: the mean ratio (beta) and the variability ratio (alpha), so a run that gets
#: the timing right but the magnitude wrong is penalised. (Correlation, the
#: old default for SCA and ET, is blind to bias and cannot constrain the water
#: balance.) GLACIER_MB uses the period-mean rate score from the GLACIER_MB
#: evaluator.
DEFAULT_METRICS = {
    'STREAMFLOW': 'kge',
    'TWS': 'nse',
    'SWE': 'kge',
    'SCA': 'kge',
    'SNOW': 'kge',
    'ET': 'kge',
    'BASIN_SWE': 'kge',
    'BASIN_SCA': 'kge',
    'BASIN_ET': 'kge',
    'GLACIER_MB': 'GMB_SCORE',
}


@R.objectives.add('MULTIVARIATE')
class MultivariateObjective(BaseObjective):
    """Weighted multi-variable objective function.

    Combines multiple variables and metrics into a single scalar objective
    for multi-criteria calibration. Supports flexible metric selection per
    variable and user-defined weighting for multi-criteria optimization.

    This objective is the primary method for calibrating models against
    multiple observational datasets simultaneously (e.g., streamflow, snow cover,
    evapotranspiration, groundwater anomalies).

    Configuration Requirements:
        OBJECTIVE_WEIGHTS (dict): Variable-specific weights. Example:
            {'STREAMFLOW': 0.7, 'SWE': 0.2, 'ET': 0.1}
            Weights are automatically normalized to sum to 1.0.

        OBJECTIVE_METRICS (dict, optional): Primary metric per variable
            (keys case-insensitive), overriding :data:`DEFAULT_METRICS`
            (KGE for streamflow, SWE, SCA and ET; NSE for TWS; GMB_SCORE for
            GLACIER_MB). Valid metrics depend on the evaluator: 'kge', 'nse',
            'rmse', 'pbias', 'correlation', 'GMB_SCORE', 'GMB_ABS_BIAS', ...
            Correlation is accepted but is bias-blind.

    Algorithm:
        1. Retrieve weights from config (default: STREAMFLOW=1.0)
        2. Normalize weights so they sum to 1.0
        3. For each variable in weights:
           - Look up primary metric value (with case-insensitive fallback)
           - If metric not found, assign penalty value (-10.0)
           - If variable not in results, assign penalty (2.0)
        4. Sum weighted losses: w_i * (1 - metric_i) for maximised metrics,
           w_i * |metric_i| for minimised ones (RMSE, PBIAS, GMB_ABS_BIAS)
        5. Return composite score (to be minimized)

    Metric Value Resolution (robust lookup):
        The method handles varying metric case conventions by trying:
        1. Exact match (e.g., 'kge')
        2. Uppercase match (e.g., 'KGE')
        3. Lowercase match
        4. Aliases (e.g., 'corr' -> 'correlation' or 'r')

    Score Transformation:
        Most metrics (KGE, NSE) are naturally maximized (range 0-1).
        This objective transforms them for minimization: score = 1 - metric
        This makes optimization algorithms (which minimize) move toward 1.0 metric values.

    Example Configuration:
        OBJECTIVE_FUNCTION: MULTIVARIATE
        OBJECTIVE_WEIGHTS:
            STREAMFLOW: 0.7
            SWE: 0.2
            ET: 0.1
        OBJECTIVE_METRICS:
            STREAMFLOW: kge
            SWE: nse
            ET: correlation
    """
    def calculate(self, evaluation_results: Dict[str, Dict[str, float]]) -> float:
        """Calculate weighted multi-variable objective score.

        Transforms multi-dimensional evaluation results into a scalar objective
        by: (1) weighting each variable by user-specified importance, (2) selecting
        the primary metric for each variable, (3) transforming metrics to minimize,
        and (4) summing weighted contributions.

        Args:
            evaluation_results: Nested dict structure {variable: {metric: value}}.
                Example: {'STREAMFLOW': {'kge': 0.85, 'nse': 0.82}, 'SWE': {'nse': 0.72}}

        Returns:
            float: Composite objective score. Lower values indicate better overall
            model performance. Typical range 0.0 (perfect) to 2.0+ (poor).

        Note:
            - Variables in evaluation_results but not in weights are ignored
            - Variables in weights but missing (or None) in evaluation_results incur 2.0 penalty
            - Missing metrics for a variable incur -10.0 penalty (high cost to encourage
              fixing evaluation issues)
            - Maximised metrics contribute (1 - value), minimised ones |value|
              (direction from the metrics registry)
        """
        # Get weights from config (default: streamflow only if not specified)
        weights = self._get_config_value(
            lambda: self.config.optimization.objective_weights,
            default={'STREAMFLOW': 1.0},
            dict_key='OBJECTIVE_WEIGHTS'
        )

        # Normalize weights to sum to 1.0. This ensures the objective value is
        # scale-independent: changing all weights by a constant factor doesn't
        # change optimization behavior.
        total_weight = sum(weights.values())
        norm_weights = {k.upper(): v/total_weight for k, v in weights.items()}

        # Primary metric per variable from config. User entries (any key case)
        # override DEFAULT_METRICS per variable.
        configured = self._get_config_value(
            lambda: self.config.optimization.objective_metrics,
            default=None,
            dict_key='OBJECTIVE_METRICS'
        ) or {}
        metrics = dict(DEFAULT_METRICS)
        metrics.update({str(k).upper(): str(v) for k, v in configured.items()})

        composite_score = 0.0

        for var, weight in norm_weights.items():
            if var in evaluation_results and evaluation_results[var]:
                metric_name = metrics.get(var, 'kge')

                # Robust lookup for metric value. Different evaluators may use
                # different case conventions (kge vs KGE vs KGE_prime), so we
                # search systematically: exact -> upper -> lower -> aliases
                val = None

                # 1. Try exact match (most common convention)
                if metric_name in evaluation_results[var]:
                    val = evaluation_results[var][metric_name]
                # 2. Try uppercase (common in evaluators: KGE, NSE)
                elif metric_name.upper() in evaluation_results[var]:
                    val = evaluation_results[var][metric_name.upper()]
                # 3. Try lowercase
                elif metric_name.lower() in evaluation_results[var]:
                    val = evaluation_results[var][metric_name.lower()]
                # 4. Handle common aliases (correlation vs corr vs r)
                elif metric_name.lower() == 'corr':
                    # Try 'correlation' and 'r'
                    if 'correlation' in evaluation_results[var]:
                        val = evaluation_results[var]['correlation']
                    elif 'r' in evaluation_results[var]:
                        val = evaluation_results[var]['r']

                if val is None:
                    # Metric not found for this variable. This indicates a problem
                    # (data issues, mismatched evaluator output). Use high penalty
                    # to encourage fixing the underlying evaluation issue.
                    score = -10.0
                else:
                    score = val

                # Transform to a loss: 1 - metric for maximised scores whose
                # optimum is 1 (KGE, NSE, GMB_SCORE, correlation), |metric| for
                # minimised errors (RMSE, PBIAS, GMB_ABS_BIAS), whose optimum is 0.
                if val is not None and MetricTransformer.get_direction(metric_name) == 'minimize':
                    composite_score += weight * abs(score)
                else:
                    composite_score += weight * (1.0 - score)
            else:
                # Variable not in evaluation results (missing observations for this stream).
                # Assign moderate penalty to reduce contribution without completely
                # killing convergence (unlike -10.0 penalty for missing metrics).
                composite_score += weight * 2.0

        return composite_score
