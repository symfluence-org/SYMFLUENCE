# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Residual analysis panel for model comparison visualizations.

Renders residual/bias analysis visualizations.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from symfluence.reporting.panels.base_panel import BasePanel


class ResidualAnalysisPanel(BasePanel):
    """Panel for residual/bias analysis visualization.

    Renders bar plot showing monthly bias (percentage) for model vs observations.

    Data Requirements:
        - results_df: DataFrame with DateTimeIndex and model columns
        - obs_series: Observation time series
        - model_cols: List of model column names (uses first one)

    Example:
        panel = ResidualAnalysisPanel(plot_config, logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': ['SUMMA_discharge']
        })
    """

    MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render residual analysis bar plot.

        Args:
            ax: Matplotlib axis
            data: Dictionary containing:
                - results_df: DataFrame with DateTimeIndex and model columns
                - obs_series: Observation Series
                - model_cols: List of model column names
        """
        results_df = data.get('results_df')
        obs_series = data.get('obs_series')
        model_cols = data.get('model_cols', [])

        if obs_series is None or results_df is None or not model_cols:
            ax.text(0.5, 0.5, 'No data for residual analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            return

        col = model_cols[0]
        if col not in results_df.columns:
            ax.text(0.5, 0.5, 'Model column not found',
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            return

        # Get months from index
        months = results_df.index.month
        obs_values = obs_series.values
        sim_values = results_df[col].values

        # Calculate monthly bias
        monthly_bias = self._calculate_monthly_bias(obs_values, sim_values, months)

        # Create bar plot
        positions = np.arange(1, 13)
        colors = [self._get_color(0) if b >= 0 else '#d62728' for b in monthly_bias]
        ax.bar(positions, monthly_bias, color=colors, alpha=0.7)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(self.MONTH_LABELS)

        model_name = self._extract_model_name(col)
        self._apply_styling(
            ax,
            xlabel='Month',
            ylabel='Bias (%)',
            title=f'Monthly Bias - {model_name}',
            legend=False
        )

    def _calculate_monthly_bias(
        self,
        obs_values: np.ndarray,
        sim_values: np.ndarray,
        months: np.ndarray
    ) -> List[float]:
        """Calculate monthly percentage bias.

        Args:
            obs_values: Observation values
            sim_values: Simulated values
            months: Array of month numbers (1-12)

        Returns:
            List of monthly bias percentages
        """
        monthly_bias = []

        for m in range(1, 13):
            mask = (months == m) & ~np.isnan(obs_values) & ~np.isnan(sim_values)
            if mask.sum() > 0:
                obs_m = obs_values[mask]
                sim_m = sim_values[mask]
                mean_obs = np.mean(obs_m)
                if mean_obs != 0:
                    bias = ((np.mean(sim_m) - mean_obs) / mean_obs) * 100
                else:
                    bias = 0
                monthly_bias.append(bias)
            else:
                monthly_bias.append(0)

        return monthly_bias
