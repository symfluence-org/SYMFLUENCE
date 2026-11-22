# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""One-page run summary figure: ``results/<experiment_id>/summary.png``.

Hydrograph (calibration / evaluation / test periods shaded), flow duration
curve and the metrics table from ``metrics.json``, built from the existing
reporting panels. Written by default after postprocessing; set
``SUMMARY_REPORT: false`` to skip it.

Uses matplotlib's object API with the Agg canvas, so it never touches the
pyplot state machine or needs a display.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from symfluence.core.metrics.run_metrics import SCORED_PERIODS
from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG
from symfluence.reporting.panels import FDCPanel, MetricsTablePanel, TimeSeriesPanel

__all__ = ['plot_run_summary']

_PERIOD_COLOURS = {'calibration': '#1f77b4', 'evaluation': '#ff7f0e', 'test': '#2ca02c'}
_PERIOD_SHORT = {'calibration': 'cal', 'evaluation': 'eval', 'test': 'test', 'full': 'all'}
_MAX_TABLE_ROWS = 10


def _table_rows(doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """MetricsTablePanel rows: one per model and scored period."""
    rows: Dict[str, Dict[str, Any]] = {}
    for model, entry in (doc.get('models') or {}).items():
        metrics = entry.get('metrics') or {}
        periods = [p for p in SCORED_PERIODS if p in metrics] or (['full'] if 'full' in metrics else [])
        for period in periods:
            m = metrics[period]
            rows[f"{model} ({_PERIOD_SHORT[period]})"] = {
                'KGE': m.get('KGE'), 'NSE': m.get('NSE'),
                'RMSE': m.get('RMSE'), 'Bias%': m.get('PBIAS'),
            }
            if len(rows) >= _MAX_TABLE_ROWS:
                return rows
    return rows


def plot_run_summary(
    results_df: pd.DataFrame,
    obs: Optional[pd.Series],
    doc: Mapping[str, Any],
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Render the summary page to ``output_path`` (PNG) and return the path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    from matplotlib.figure import Figure

    logger = logger or logging.getLogger(__name__)
    model_cols = [c for c in results_df.columns if 'discharge' in str(c).lower()]
    plot_df = results_df[model_cols].apply(pd.to_numeric, errors='coerce')
    plot_df.index = pd.DatetimeIndex(pd.to_datetime(plot_df.index))
    obs_aligned: Optional[pd.Series] = None
    if doc.get('resolution') != 'native':
        plot_df = plot_df.resample('D').mean()
    if obs is not None:
        o = pd.to_numeric(obs, errors='coerce')
        o.index = pd.DatetimeIndex(pd.to_datetime(o.index))
        o = o[~o.index.duplicated(keep='first')].sort_index()
        if doc.get('resolution') != 'native':
            o = o.resample('D').mean()
        obs_aligned = o.reindex(plot_df.index)

    fig = Figure(figsize=(13, 9), constrained_layout=True)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])
    ax_ts = fig.add_subplot(grid[0, :])
    ax_fdc = fig.add_subplot(grid[1, 0])
    ax_tab = fig.add_subplot(grid[1, 1])

    data = {'results_df': plot_df, 'obs_series': obs_aligned, 'model_cols': model_cols}
    TimeSeriesPanel(DEFAULT_PLOT_CONFIG, logger).render(ax_ts, data)
    locator = AutoDateLocator()
    ax_ts.xaxis.set_major_locator(locator)
    ax_ts.xaxis.set_major_formatter(AutoDateFormatter(locator))

    periods = doc.get('periods') or {}
    spinup = periods.get('spinup')
    if spinup:
        ax_ts.axvspan(pd.Timestamp(spinup[0]), pd.Timestamp(spinup[1]),
                      color='#7f7f7f', alpha=0.12, label='spin-up (not scored)')
    for name in SCORED_PERIODS:
        span = periods.get(name)
        if span:
            ax_ts.axvspan(pd.Timestamp(span[0]), pd.Timestamp(span[1]),
                          color=_PERIOD_COLOURS[name], alpha=0.08, label=f"{name} period")
    if any(periods.get(n) for n in SCORED_PERIODS):
        ax_ts.legend(loc='upper right', framealpha=0.9, fontsize=8)
    title = f"{doc.get('domain', '')} / {doc.get('experiment_id', '')}"
    ax_ts.set_title(f"Hydrograph - {title}", fontsize=12, fontweight='bold')

    fdc_obs = obs_aligned.dropna() if obs_aligned is not None else None
    FDCPanel(DEFAULT_PLOT_CONFIG, logger).render(
        ax_fdc, {'results_df': plot_df, 'obs_series': fdc_obs, 'model_cols': model_cols}
    )
    MetricsTablePanel(DEFAULT_PLOT_CONFIG, logger).render(ax_tab, {'metrics_dict': _table_rows(doc)})
    warnings = list(doc.get('warnings') or [])
    if warnings:
        ax_tab.text(0.0, 0.02, f"{len(warnings)} warning(s) - see metrics.json",
                    transform=ax_tab.transAxes, fontsize=8, color='#b00020')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    return output_path
