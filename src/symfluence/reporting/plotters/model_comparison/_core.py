# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Core plotting helpers and panel wiring for model comparison plots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.reporting.core.plot_utils import (
    calculate_metrics,
)
from symfluence.reporting.panels import (
    FDCPanel,
    MetricsTablePanel,
    MonthlyBoxplotPanel,
    MultiScatterPanel,
    ResidualAnalysisPanel,
    TimeSeriesPanel,
)


class ModelComparisonCoreMixin:
    """Shared plotting primitives for model comparison visualizations."""

    MODEL_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    @property
    def _ts_panel(self) -> TimeSeriesPanel:
        if getattr(self, "_cached_ts_panel", None) is None:
            self._cached_ts_panel = TimeSeriesPanel(self.plot_config, self.logger)
        return self._cached_ts_panel

    @property
    def _fdc_panel(self) -> FDCPanel:
        if getattr(self, "_cached_fdc_panel", None) is None:
            self._cached_fdc_panel = FDCPanel(self.plot_config, self.logger)
        return self._cached_fdc_panel

    @property
    def _metrics_panel(self) -> MetricsTablePanel:
        if getattr(self, "_cached_metrics_panel", None) is None:
            self._cached_metrics_panel = MetricsTablePanel(self.plot_config, self.logger)
        return self._cached_metrics_panel

    @property
    def _scatter_panel(self) -> MultiScatterPanel:
        if getattr(self, "_cached_scatter_panel", None) is None:
            self._cached_scatter_panel = MultiScatterPanel(self.plot_config, self.logger)
        return self._cached_scatter_panel

    @property
    def _monthly_panel(self) -> MonthlyBoxplotPanel:
        if getattr(self, "_cached_monthly_panel", None) is None:
            self._cached_monthly_panel = MonthlyBoxplotPanel(self.plot_config, self.logger)
        return self._cached_monthly_panel

    @property
    def _residual_panel(self) -> ResidualAnalysisPanel:
        if getattr(self, "_cached_residual_panel", None) is None:
            self._cached_residual_panel = ResidualAnalysisPanel(self.plot_config, self.logger)
        return self._cached_residual_panel

    @staticmethod
    def _model_name_from_column(column: str) -> str:
        return column.replace("_discharge_cms", "").replace("_discharge", "")

    @staticmethod
    def _find_discharge_columns(results_df: pd.DataFrame) -> List[str]:
        return [
            column
            for column in results_df.columns
            if "discharge" in column.lower() and "obs" not in column.lower()
        ]

    @staticmethod
    def _align_valid_pairs(
        obs_values: Any,
        sim_values: Any,
        min_points: int = 1,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return NaN-free (obs, sim) arrays paired by timestamp.

        When both inputs are pandas Series they are joined on their index, so
        a simulation table and an observation series covering different
        periods (e.g. the default and the calibrated run) are compared only on
        matching dates, never element-by-element. Plain arrays keep positional
        pairing and must already be aligned.
        """
        if isinstance(obs_values, pd.Series) and isinstance(sim_values, pd.Series):
            obs_s = obs_values[~obs_values.index.duplicated(keep="first")]
            sim_s = sim_values[~sim_values.index.duplicated(keep="first")]
            joined = pd.concat([obs_s, sim_s], axis=1, join="inner")
            obs_arr = pd.to_numeric(joined.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            sim_arr = pd.to_numeric(joined.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        else:
            obs_arr = np.asarray(obs_values, dtype=float)
            sim_arr = np.asarray(sim_values, dtype=float)
        valid_mask = ~(np.isnan(obs_arr) | np.isnan(sim_arr))
        obs_clean = obs_arr[valid_mask]
        sim_clean = sim_arr[valid_mask]
        if len(obs_clean) < min_points:
            return None
        return obs_clean, sim_clean

    def _calculate_all_metrics(
        self,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        metrics_dict: Dict[str, Dict[str, float]] = {}

        if obs_series is None:
            return metrics_dict

        for col in model_cols:
            aligned = self._align_valid_pairs(obs_series, results_df[col], min_points=10)
            if aligned is None:
                continue
            obs_clean, sim_clean = aligned

            metrics = calculate_metrics(obs_clean, sim_clean)

            mean_obs = np.mean(obs_clean)
            mean_sim = np.mean(sim_clean)
            bias = ((mean_sim - mean_obs) / mean_obs) * 100 if mean_obs != 0 else np.nan
            metrics["Bias%"] = bias

            model_name = self._model_name_from_column(col)
            metrics_dict[model_name] = metrics

        return metrics_dict

    def _plot_metrics_table(self, ax: Any, metrics_dict: Dict[str, Dict[str, float]]) -> None:
        ax.axis("off")

        if not metrics_dict:
            ax.text(0.5, 0.5, "No metrics available", transform=ax.transAxes, ha="center", va="center", fontsize=12)
            return

        headers = ["Model", "KGE", "NSE", "RMSE", "Bias%"]
        cell_data = []

        for model_name, metrics in metrics_dict.items():
            row = [
                model_name,
                f"{metrics.get('KGE', np.nan):.3f}",
                f"{metrics.get('NSE', np.nan):.3f}",
                f"{metrics.get('RMSE', np.nan):.2f}",
                f"{metrics.get('Bias%', np.nan):+.1f}%",
            ]
            cell_data.append(row)

        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colColours=["#f0f0f0"] * len(headers),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title("Performance Metrics", fontsize=12, fontweight="bold", pad=10)
