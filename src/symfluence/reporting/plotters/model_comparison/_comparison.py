# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Default-vs-calibrated comparison plotting helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.plot_utils import calculate_flow_duration_curve


class ModelComparisonDefaultVsCalibratedMixin:
    """Plot default vs calibrated run comparisons."""

    @BasePlotter._plot_safe("creating default vs calibrated comparison")
    def plot_default_vs_calibrated_comparison(
        self,
        experiment_id: str = "default",
    ) -> Optional[str]:
        """Create a multi-panel default-vs-calibrated comparison plot."""
        default_df, obs_series = self._collect_model_data(experiment_id, context="run_model")
        calibrated_df, _ = self._collect_model_data(experiment_id, context="calibrate_model")

        if default_df is None or calibrated_df is None:
            self.logger.warning("Could not load both default and calibrated results for comparison")
            return None

        default_cols = self._find_discharge_columns(default_df)
        calibrated_cols = self._find_discharge_columns(calibrated_df)

        if not default_cols or not calibrated_cols:
            self.logger.warning("No model discharge columns found")
            return None

        default_metrics = self._calculate_all_metrics(default_df, obs_series, default_cols)
        calibrated_metrics = self._calculate_all_metrics(calibrated_df, obs_series, calibrated_cols)

        plt, _ = self._setup_matplotlib()
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

        fig.suptitle(
            f"Default vs Calibrated Model Comparison\n{experiment_id}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        ax_ts = fig.add_subplot(gs[0, :])
        self._plot_comparison_timeseries(
            ax_ts,
            default_df,
            calibrated_df,
            obs_series,
            default_cols[0],
            calibrated_cols[0],
        )

        ax_fdc = fig.add_subplot(gs[1, 0])
        self._plot_comparison_fdc(
            ax_fdc,
            default_df,
            calibrated_df,
            obs_series,
            default_cols[0],
            calibrated_cols[0],
        )

        ax_metrics = fig.add_subplot(gs[1, 1])
        self._plot_comparison_metrics_table(
            ax_metrics,
            default_metrics,
            calibrated_metrics,
            default_cols[0],
            calibrated_cols[0],
        )

        ax_scatter_default = fig.add_subplot(gs[2, 0])
        self._plot_single_scatter(
            ax_scatter_default,
            default_df[default_cols[0]],
            obs_series,
            "Default Run",
            self.MODEL_COLORS[0],
        )

        ax_scatter_calib = fig.add_subplot(gs[2, 1])
        self._plot_single_scatter(
            ax_scatter_calib,
            calibrated_df[calibrated_cols[0]],
            obs_series,
            "Calibrated Run",
            self.MODEL_COLORS[1],
        )

        output_dir = self._ensure_output_dir("model_comparison")
        plot_path = output_dir / f"{experiment_id}_default_vs_calibrated.png"
        return self._save_and_close(fig, plot_path)

    def _plot_comparison_timeseries(
        self,
        ax: Any,
        default_df: pd.DataFrame,
        calibrated_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        default_col: str,
        calibrated_col: str,
    ) -> None:
        if obs_series is not None:
            ax.plot(obs_series.index, obs_series.values, color="black", linewidth=1.5, label="Observed", zorder=10)

        ax.plot(
            default_df.index,
            default_df[default_col],
            color=self.MODEL_COLORS[0],
            linewidth=1.0,
            alpha=0.8,
            label="Default",
            linestyle="--",
        )

        ax.plot(
            calibrated_df.index,
            calibrated_df[calibrated_col],
            color=self.MODEL_COLORS[1],
            linewidth=1.0,
            alpha=0.9,
            label="Calibrated",
        )

        self._apply_standard_styling(
            ax,
            xlabel="Date",
            ylabel="Discharge (m³/s)",
            title="Time Series: Default vs Calibrated",
            legend=True,
            legend_loc="upper right",
        )
        self._format_date_axis(ax, format_type="full")

    def _plot_comparison_fdc(
        self,
        ax: Any,
        default_df: pd.DataFrame,
        calibrated_df: pd.DataFrame,
        obs_series: Optional[pd.Series],
        default_col: str,
        calibrated_col: str,
    ) -> None:
        if obs_series is not None:
            exc_obs, flows_obs = calculate_flow_duration_curve(obs_series.values)
            if len(exc_obs) > 0:
                ax.plot(exc_obs * 100, flows_obs, color="black", linewidth=2, label="Observed", zorder=10)

        exc_def, flows_def = calculate_flow_duration_curve(default_df[default_col].values)
        if len(exc_def) > 0:
            ax.plot(
                exc_def * 100,
                flows_def,
                color=self.MODEL_COLORS[0],
                linewidth=1.5,
                alpha=0.8,
                label="Default",
                linestyle="--",
            )

        exc_cal, flows_cal = calculate_flow_duration_curve(calibrated_df[calibrated_col].values)
        if len(exc_cal) > 0:
            ax.plot(
                exc_cal * 100,
                flows_cal,
                color=self.MODEL_COLORS[1],
                linewidth=1.5,
                alpha=0.9,
                label="Calibrated",
            )

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

    def _plot_comparison_metrics_table(
        self,
        ax: Any,
        default_metrics: Dict[str, Dict[str, float]],
        calibrated_metrics: Dict[str, Dict[str, float]],
        default_col: str,
        calibrated_col: str,
    ) -> None:
        ax.axis("off")

        default_name = self._model_name_from_column(default_col)
        calibrated_name = self._model_name_from_column(calibrated_col)

        def_metrics = default_metrics.get(default_name, {})
        cal_metrics = calibrated_metrics.get(calibrated_name, {})

        headers = ["Metric", "Default", "Calibrated", "Change"]
        metrics_to_show = ["KGE", "NSE", "RMSE", "Bias%"]
        cell_data = []

        for metric in metrics_to_show:
            def_val = def_metrics.get(metric, np.nan)
            cal_val = cal_metrics.get(metric, np.nan)

            if not np.isnan(def_val) and not np.isnan(cal_val):
                if metric in ["KGE", "NSE"]:
                    change = cal_val - def_val
                    change_str = f"{change:+.3f}" if change != 0 else "0.000"
                elif metric == "RMSE":
                    change = def_val - cal_val
                    change_str = f"{change:+.2f}" if change != 0 else "0.00"
                else:
                    change = abs(def_val) - abs(cal_val)
                    change_str = f"{change:+.1f}%" if change != 0 else "0.0%"
            else:
                change_str = "N/A"

            if metric == "Bias%":
                row = [metric, f"{def_val:+.1f}%", f"{cal_val:+.1f}%", change_str]
            elif metric == "RMSE":
                row = [metric, f"{def_val:.2f}", f"{cal_val:.2f}", change_str]
            else:
                row = [metric, f"{def_val:.3f}", f"{cal_val:.3f}", change_str]

            cell_data.append(row)

        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colColours=["#f0f0f0"] * len(headers),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 1.8)

        ax.set_title("Performance Comparison\n(+ = improvement)", fontsize=12, fontweight="bold", pad=10)

    def _plot_single_scatter(
        self,
        ax: Any,
        sim_series: pd.Series,
        obs_series: Optional[pd.Series],
        label: str,
        color: str,
    ) -> None:
        if obs_series is None:
            ax.text(0.5, 0.5, "No observations", transform=ax.transAxes, ha="center", va="center")
            return

        aligned = self._align_valid_pairs(obs_series, sim_series, min_points=10)
        if aligned is None:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            return
        obs_clean, sim_clean = aligned

        ax.scatter(obs_clean, sim_clean, c=color, alpha=0.3, s=15, edgecolors="none")

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

        self._apply_standard_styling(
            ax,
            xlabel="Observed (m³/s)",
            ylabel="Simulated (m³/s)",
            title=label,
            legend=False,
        )
