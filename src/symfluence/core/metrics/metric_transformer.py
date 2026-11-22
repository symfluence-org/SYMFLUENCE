#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Metric Transformer for Optimization

This module provides utilities to transform metric values for consistent optimization
direction handling. All metrics are normalized to maximization convention, meaning
higher transformed values are always better.

For example:
- KGE (maximize): 0.8 -> 0.8 (unchanged)
- RMSE (minimize): 10.0 -> -10.0 (negated)
- PBIAS (minimize, signed): -15% -> -15 (use absolute value then negate)
- PBIAS (minimize, signed): +15% -> -15 (both +15% and -15% bias are equally bad)

Usage
-----
>>> from symfluence.core.metrics.metric_transformer import MetricTransformer
>>> score = MetricTransformer.transform_for_maximization('RMSE', 10.0)
>>> # score = -10.0 (now lower RMSE gives higher transformed score)

>>> # Check if metric should be minimized
>>> if MetricTransformer.get_direction('MAE') == 'minimize':
...     print("MAE is a minimization metric")
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .metrics import canonicalize_metric_name, get_metric_info


class MetricTransformer:
    """Utility class for transforming metrics to consistent optimization direction.

    All optimization algorithms in symfluence assume maximization. This class
    transforms metric values so that higher values always indicate better
    performance, regardless of the metric's natural direction.

    Transformation Rules:
        - Maximize metrics (KGE, NSE, R2, correlation): No transformation
        - Minimize metrics (RMSE, MAE, NRMSE, MARE): Negate value
        - Signed minimize metrics (PBIAS, bias): Negate absolute value
          This ensures that both positive and negative bias are penalized equally.

    Attributes:
        SIGNED_MINIMIZE_METRICS: Set of metrics where sign indicates direction
            of bias but magnitude should be minimized (e.g., PBIAS can be +10%
            or -10%, both equally bad).

    Example:
        >>> # For RMSE = 5.0 (lower is better)
        >>> MetricTransformer.transform_for_maximization('RMSE', 5.0)
        -5.0  # Now optimizer can maximize this

        >>> # For PBIAS = -20% (closer to 0 is better)
        >>> MetricTransformer.transform_for_maximization('PBIAS', -20.0)
        -20.0  # abs(-20) = 20, then negate = -20

        >>> # For PBIAS = +20% (equally bad as -20%)
        >>> MetricTransformer.transform_for_maximization('PBIAS', 20.0)
        -20.0  # Both result in same transformed value

    See Also:
        get_metric_info: Get metadata about a metric including direction
        METRIC_REGISTRY: Registry of all available metrics and their properties
    """

    # Metrics where sign indicates direction but magnitude should be minimized
    # Both +10% and -10% PBIAS are equally bad, so we use -abs(value)
    # Matched on the canonical name, so any capitalisation (PBias, Bias) counts.
    SIGNED_MINIMIZE_METRICS = {'PBIAS', 'pbias', 'bias', 'Bias', 'BIAS'}

    # Suffixes that indicate a flow transformation was applied
    # These don't change the optimization direction of the base metric
    TRANSFORM_SUFFIXES = ('_LOG', '_INV', '_SQRT', '_BOX_COX', '_log', '_inv', '_sqrt', '_box_cox')

    @classmethod
    def get_direction(cls, metric_name: str) -> str:
        """Get the optimization direction for a metric.

        Handles transformed metric names (e.g., KGE_LOG, RMSE_INV) by inferring
        the direction from the base metric. The flow transformation doesn't change
        whether a metric should be maximized or minimized.

        Args:
            metric_name: Name of the metric (case-insensitive for common metrics)

        Returns:
            'maximize' or 'minimize'. Defaults to 'maximize' if metric not found.

        Example:
            >>> MetricTransformer.get_direction('KGE')
            'maximize'
            >>> MetricTransformer.get_direction('KGE_LOG')
            'maximize'
            >>> MetricTransformer.get_direction('RMSE')
            'minimize'
            >>> MetricTransformer.get_direction('RMSE_LOG')
            'minimize'
        """
        info = get_metric_info(metric_name)
        if info is not None:
            return info.direction

        # For transformed metrics (e.g., KGE_LOG, RMSE_INV), strip the suffix
        # and look up the base metric's direction
        for suffix in cls.TRANSFORM_SUFFIXES:
            if metric_name.endswith(suffix):
                base_name = metric_name[:-len(suffix)]
                base_info = get_metric_info(base_name)
                if base_info is not None:
                    return base_info.direction

        # Default to maximize for unknown metrics (safer for fitness tracking)
        return 'maximize'

    @classmethod
    def is_minimize_metric(cls, metric_name: str) -> bool:
        """Check if a metric should be minimized.

        Args:
            metric_name: Name of the metric

        Returns:
            True if the metric should be minimized, False otherwise.
        """
        return cls.get_direction(metric_name) == 'minimize'

    @classmethod
    def transform_for_maximization(
        cls,
        metric_name: str,
        value: Optional[float]
    ) -> Optional[float]:
        """Transform a metric value to maximization convention.

        After transformation, higher values always indicate better performance.
        This allows optimization algorithms to always maximize without needing
        to know the original direction of each metric.

        Args:
            metric_name: Name of the metric (e.g., 'KGE', 'RMSE', 'PBIAS')
            value: Raw metric value to transform

        Returns:
            Transformed value suitable for maximization, or the original value
            if it's None or NaN (these are passed through unchanged).

        Transformation Logic:
            - If value is None or NaN: Return as-is (penalty handling elsewhere)
            - If metric direction is 'maximize': Return value unchanged
            - If metric is in SIGNED_MINIMIZE_METRICS: Return -abs(value)
            - If metric direction is 'minimize': Return -value

        Example:
            >>> # KGE: maximize, no change
            >>> MetricTransformer.transform_for_maximization('KGE', 0.85)
            0.85

            >>> # RMSE: minimize, negate
            >>> MetricTransformer.transform_for_maximization('RMSE', 10.0)
            -10.0

            >>> # PBIAS: minimize signed, negate absolute
            >>> MetricTransformer.transform_for_maximization('PBIAS', -15.0)
            -15.0
            >>> MetricTransformer.transform_for_maximization('PBIAS', 15.0)
            -15.0
        """
        # Pass through None and NaN unchanged
        if value is None:
            return value
        if isinstance(value, float) and np.isnan(value):
            return value

        direction = cls.get_direction(metric_name)

        # Maximize metrics: no transformation needed
        if direction == 'maximize':
            return value

        # Signed minimize metrics: use negative absolute value
        # This ensures +10% PBIAS and -10% PBIAS are treated equally
        if canonicalize_metric_name(metric_name) in {'PBIAS', 'bias'}:
            return -abs(value)

        # Regular minimize metrics: simple negation
        return -value

    @classmethod
    def inverse_transform(
        cls,
        metric_name: str,
        transformed_value: Optional[float]
    ) -> Optional[float]:
        """Convert a transformed value back to original metric space.

        This is the inverse of transform_for_maximization(). Useful for
        displaying results in their natural units.

        Args:
            metric_name: Name of the metric
            transformed_value: Value that was previously transformed

        Returns:
            Original metric value

        Note:
            For signed minimize metrics (PBIAS), the sign information is lost
            during transformation, so inverse_transform returns the absolute
            value negated (original sign cannot be recovered).
        """
        if transformed_value is None:
            return transformed_value
        if isinstance(transformed_value, float) and np.isnan(transformed_value):
            return transformed_value

        direction = cls.get_direction(metric_name)

        if direction == 'maximize':
            return transformed_value

        # For minimize metrics, negate back
        # Note: For signed metrics, we lose the original sign
        return -transformed_value

    @classmethod
    def transform_objectives(
        cls,
        objective_names: list,
        values: list
    ) -> list:
        """Transform a list of objective values for multi-objective optimization.

        Useful for NSGA-II and other multi-objective algorithms where each
        objective may have different optimization directions.

        Args:
            objective_names: List of metric names (e.g., ['KGE', 'RMSE'])
            values: List of corresponding metric values

        Returns:
            List of transformed values, all in maximization convention

        Example:
            >>> names = ['KGE', 'RMSE']
            >>> values = [0.8, 10.0]
            >>> MetricTransformer.transform_objectives(names, values)
            [0.8, -10.0]
        """
        if len(objective_names) != len(values):
            raise ValueError(
                f"Length mismatch: {len(objective_names)} names vs {len(values)} values"
            )

        return [
            cls.transform_for_maximization(name, val)
            for name, val in zip(objective_names, values)
        ]
