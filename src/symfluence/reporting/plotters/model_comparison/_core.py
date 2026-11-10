# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Core plotting helpers and panel wiring for model comparison plots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.reporting.core.plot_utils import (
    calculate_flow_duration_curve,
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
        if not hasattr(self, "__ts_panel"):
            self.__ts_panel = TimeSeriesPanel(self.plot_config, self.logger)
        return self.__ts_panel

    @property
    def _fdc_panel(self) -> FDCPanel:
        if not hasattr(self, "__fdc_panel"):
            self.__fdc_panel = FDCPanel(self.plot_config, self.logger)
        return self.__fdc_panel

    @property
    def _metrics_panel(self) -> MetricsTablePanel:
        if not hasattr(self, "__metrics_panel"):
            self.__metrics_panel = MetricsTablePanel(self.plot_config, self.logger)
        return self.__metrics_panel

    @property
    def _scatter_panel(self) -> MultiScatterPanel:
        if not hasattr(self, "__scatter_panel"):
            self.__scatter_panel = MultiScatterPanel(self.plot_config, self.logger)
        return self.__scatter_panel

    @property
    def _monthly_panel(self) -> MonthlyBoxplotPanel:
        if not hasattr(self, "__monthly_panel"):
            self.__monthly_panel = MonthlyBoxplotPanel(self.plot_config, self.logger)
        return self.__monthly_panel

    @property
    def _residual_panel(self) -> ResidualAnalysisPanel:
        if not hasattr(self, "__residual_panel"):
            self.__residual_panel = ResidualAnalysisPanel(self.plot_config, self.logger)
        return self.__residual_panel

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

    def _plot_timeseries_panel(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str],
    ) -> None:
        if obs_series is not None:
            ax.plot(
                obs_series.index,
                obs_series.values,
                color="black",
                linewidth=1.5,
                label="Observed",
                zorder=10,
            )

        for i, col in enumerate(model_cols):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            model_name = self._model_name_from_column(col)
            ax.plot(
                results_df.index,
                results_df[col],
                color=color,
                linewidth=1.0,
                alpha=0.8,
                label=model_name,
            )

        self._apply_standard_styling(
            ax,
            xlabel="Date",
            ylabel="Discharge (m³/s)",
            title="Time Series Comparison",
            legend=True,
            legend_loc="upper right",
        )
        self._format_date_axis(ax, format_type="full")

    def _plot_fdc_panel(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str],
    ) -> None:
        if obs_series is not None:
            exc_obs, flows_obs = calculate_flow_duration_curve(obs_series.values)
            if len(exc_obs) > 0:
                ax.plot(
                    exc_obs * 100,
                    flows_obs,
                    color="black",
                    linewidth=2,
                    label="Observed",
                    zorder=10,
                )

        for i, col in enumerate(model_cols):
            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            exc, flows = calculate_flow_duration_curve(results_df[col].values)
            if len(exc) > 0:
                model_name = self._model_name_from_column(col)
                ax.plot(exc * 100, flows, color=color, linewidth=1.5, alpha=0.8, label=model_name)

        ax.set_yscale("log")
        ax.set_xlim([0, 100])

        self._apply_standard_styling(
            ax,
            xlabel="Exceedance Probability (%)",
            ylabel="Discharge (m³/s)",
            title="Flow Duration Curves",
            legend=True,
            legend_loc="upper right",
        )

    def _plot_scatter_panels(
        self,
        axes: List[Any],
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str],
    ) -> None:
        if obs_series is None:
            return

        for i, (ax, col) in enumerate(zip(axes, model_cols[: len(axes)])):
            aligned = self._align_valid_pairs(obs_series, results_df[col], min_points=10)
            if aligned is None:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
                continue
            obs_clean, sim_clean = aligned

            color = self.MODEL_COLORS[i % len(self.MODEL_COLORS)]
            ax.scatter(obs_clean, sim_clean, c=color, alpha=0.3, s=10, edgecolors="none")

            max_val = max(np.max(obs_clean), np.max(sim_clean))
            min_val = min(np.min(obs_clean), np.min(sim_clean))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="1:1 line")

            correlation = np.corrcoef(obs_clean, sim_clean)[0, 1]
            r_squared = correlation**2

            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            model_name = self._model_name_from_column(col)
            self._apply_standard_styling(
                ax,
                xlabel="Observed (m³/s)",
                ylabel="Simulated (m³/s)",
                title=model_name,
                legend=False,
            )

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

    def _plot_monthly_boxplots(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str],
    ) -> None:
        _plt, _ = self._setup_matplotlib()

        months = results_df.index.month
        month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

        positions = np.arange(1, 13)

        if obs_series is not None:
            obs_months = pd.DatetimeIndex(obs_series.index).month
            obs_monthly = [obs_series[obs_months == m].dropna().values for m in range(1, 13)]
            bp_obs = ax.boxplot(obs_monthly, positions=positions - 0.2, widths=0.15, patch_artist=True)
            for patch in bp_obs["boxes"]:
                patch.set_facecolor("black")
                patch.set_alpha(0.5)

        if model_cols:
            col = model_cols[0]
            model_monthly = [results_df[col][months == m].dropna().values for m in range(1, 13)]
            bp_model = ax.boxplot(model_monthly, positions=positions + 0.2, widths=0.15, patch_artist=True)
            color = self.MODEL_COLORS[0]
            for patch in bp_model["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(month_names)

        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor="black", alpha=0.5, label="Observed")]
        if model_cols:
            model_name = self._model_name_from_column(model_cols[0])
            legend_elements.append(Patch(facecolor=self.MODEL_COLORS[0], alpha=0.5, label=model_name))
        ax.legend(handles=legend_elements, loc="upper right")

        self._apply_standard_styling(
            ax,
            xlabel="Month",
            ylabel="Discharge (m³/s)",
            title="Monthly Distribution",
            legend=False,
        )

    def _plot_residual_analysis(
        self,
        ax: Any,
        results_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        model_cols: List[str],
    ) -> None:
        if obs_series is None or not model_cols:
            ax.text(0.5, 0.5, "No data for residual analysis", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            return

        months = results_df.index.month
        month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

        col = model_cols[0]
        # Pair observations with the simulation by timestamp
        obs_values = obs_series[~obs_series.index.duplicated(keep="first")].reindex(results_df.index).to_numpy(dtype=float)
        sim_values = results_df[col].to_numpy(dtype=float)

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

        positions = np.arange(1, 13)
        colors = [self.MODEL_COLORS[0] if b >= 0 else "#d62728" for b in monthly_bias]
        ax.bar(positions, monthly_bias, color=colors, alpha=0.7)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(month_names)

        model_name = self._model_name_from_column(col)
        self._apply_standard_styling(
            ax,
            xlabel="Month",
            ylabel="Bias (%)",
            title=f"Monthly Bias - {model_name}",
            legend=False,
        )
