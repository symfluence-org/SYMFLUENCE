# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODFLOW 6 Coupling Diagnostics & Visualization

Generates a multi-panel overview figure showing how SUMMA+MODFLOW coupling
works: flow separation hydrograph, groundwater head, recharge vs baseflow,
flow duration curves, water balance, and performance metrics.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.registries import R
from symfluence.core.reporting.base_plotter import BasePlotter
from symfluence.core.reporting.plot_utils import (
    calculate_flow_duration_curve,
    calculate_metrics,
)

logger = logging.getLogger(__name__)


@R.plotters.add('MODFLOW')
class MODFLOWPlotter(BasePlotter):
    """Plotter for MODFLOW coupling diagnostics.

    Generates a 3x3 GridSpec figure with:
    - Flow separation hydrograph (obs, surface, baseflow, total)
    - Performance metrics table
    - Groundwater head time series
    - Recharge vs baseflow dual-axis plot
    - Flow duration curves by component
    - Water balance summary
    """

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Entry point for BasePlotter interface.

        Delegates to plot_coupling_results with experiment_id from kwargs or config.
        """
        experiment_id = kwargs.get(
            'experiment_id',
            self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='run_1',
                dict_key='EXPERIMENT_ID',
            ),
        )
        return self.plot_coupling_results(experiment_id)

    def plot_coupling_results(self, experiment_id: str) -> Optional[str]:
        """Main entry: collect data, build 6-panel figure, save PNG.

        Args:
            experiment_id: Experiment identifier for locating output files.

        Returns:
            Path to saved PNG or None on failure.
        """
        try:
            data = self._collect_coupling_data(experiment_id)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not collect MODFLOW coupling data: {e}", exc_info=True)
            return None

        plt, _ = self._setup_matplotlib()
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        # Row 0: flow separation (cols 0-1) + metrics table (col 2)
        ax_hydro = fig.add_subplot(gs[0, 0:2])
        ax_metrics = fig.add_subplot(gs[0, 2])

        # Row 1: groundwater head (col 0) + recharge vs baseflow (cols 1-2)
        ax_head = fig.add_subplot(gs[1, 0])
        ax_rch = fig.add_subplot(gs[1, 1:3])

        # Row 2: FDC (col 0) + water balance (cols 1-2)
        ax_fdc = fig.add_subplot(gs[2, 0])
        ax_wb = fig.add_subplot(gs[2, 1:3])

        # -- Panel 1: Flow separation hydrograph --
        self._plot_flow_separation(
            ax_hydro,
            data.get('obs'),
            data['surface_m3s'],
            data['baseflow_m3s'],
            data['total_m3s'],
        )

        # -- Panel 2: Metrics table --
        self._plot_metrics_table(ax_metrics, data.get('obs'), data['total_m3s'])

        # -- Panel 3: Groundwater head --
        self._plot_groundwater_head(
            ax_head,
            data['head'],
            data.get('top'),
            data.get('drain_elev'),
            data.get('bot'),
        )

        # -- Panel 4: Recharge vs baseflow --
        self._plot_recharge_vs_baseflow(
            ax_rch,
            data['recharge_m_d'],
            data['drain_m3_d'],
        )

        # -- Panel 5: Flow duration curves --
        self._plot_fdc_components(
            ax_fdc,
            data.get('obs'),
            data['total_m3s'],
            data['surface_m3s'],
            data['baseflow_m3s'],
        )

        # -- Panel 6: Water balance --
        total_rch = data['recharge_m_d'].sum()
        total_drain = data['drain_m3_d'].sum()
        sy = data.get('sy', 0.15)
        head_series = data['head']
        if len(head_series) >= 2:
            delta_h = head_series.iloc[-1] - head_series.iloc[0]
        else:
            delta_h = 0.0
        cell_size = data.get('cell_size', 1000.0)
        delta_storage = sy * delta_h * cell_size * cell_size  # m3

        surface_total = data['surface_m3s'].sum() * 86400.0  # m3 (daily values)
        baseflow_total = data['baseflow_m3s'].sum() * 86400.0
        flow_total = surface_total + baseflow_total
        surface_frac = surface_total / flow_total if flow_total > 0 else 0.5
        baseflow_frac = 1.0 - surface_frac

        self._plot_water_balance(
            ax_wb,
            total_rch,
            total_drain,
            delta_storage,
            surface_frac,
            baseflow_frac,
        )

        fig.suptitle(
            f'MODFLOW Coupling Overview — {experiment_id}',
            fontsize=14,
            fontweight='bold',
            y=0.98,
        )

        output_dir = self._ensure_output_dir('modflow_coupling')
        output_path = output_dir / f'{experiment_id}_coupling_overview.png'
        return self._save_and_close(fig, output_path)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_coupling_data(self, experiment_id: str) -> Dict[str, Any]:
        """Load coupling data from output directories.

        Collects observed streamflow, SUMMA surface runoff / recharge,
        MODFLOW head and drain discharge, and aquifer geometry from config.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Dictionary with all data needed for the 6-panel figure.
        """
        from symfluence.models.modflow.extractor import MODFLOWResultExtractor

        sim_base = self.project_dir / 'simulations' / experiment_id
        modflow_dir = sim_base / 'MODFLOW'
        summa_dir = sim_base / 'SUMMA'

        sp_length = float(self._get_config_value(
            lambda: self.config.model.modflow.stress_period_length,
            default=1.0,
            dict_key='MODFLOW_STRESS_PERIOD_LENGTH',
        ))
        start_date = str(self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01',
            dict_key='EXPERIMENT_TIME_START',
        ))
        from symfluence.core.metrics.streamflow_metrics import require_catchment_area_km2
        area_km2 = require_catchment_area_km2(
            self.config_dict, self.project_dir, self.domain_name,
            configured_km2=self._get_config_value(
                lambda: self.config.domain.catchment_area,
                default=None,
                dict_key='CATCHMENT_AREA',
            ),
        )
        area_m2 = area_km2 * 1e6

        top = float(self._get_config_value(
            lambda: self.config.model.modflow.top, default=1500.0,
            dict_key='MODFLOW_TOP',
        ))
        bot = float(self._get_config_value(
            lambda: self.config.model.modflow.bot, default=1400.0,
            dict_key='MODFLOW_BOT',
        ))
        # Unset means mid-aquifer, as in the preprocessor's DRN package.
        drain_elev = self._get_config_value(
            lambda: self.config.model.modflow.drain_elevation, default=None,
            dict_key='MODFLOW_DRAIN_ELEVATION',
        )
        drain_elev = (top + bot) / 2.0 if drain_elev is None else float(drain_elev)
        sy = float(self._get_config_value(
            lambda: self.config.model.modflow.sy, default=0.15,
            dict_key='MODFLOW_SY',
        ))
        cell_size = float(self._get_config_value(
            lambda: self.config.model.modflow.cell_size, default=1000.0,
            dict_key='MODFLOW_CELL_SIZE',
        ))

        extractor = MODFLOWResultExtractor()

        # Extract MODFLOW head
        head = extractor.extract_variable(
            modflow_dir, 'head',
            stress_period_length=sp_length, start_date=start_date,
        )

        # Extract MODFLOW drain discharge (m3/d)
        drain_m3_d = extractor.extract_variable(
            modflow_dir, 'drain_discharge',
            stress_period_length=sp_length, start_date=start_date,
        )
        baseflow_m3s = drain_m3_d / 86400.0

        # Extract SUMMA recharge and surface runoff
        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler
        coupler = SUMMAToMODFLOWCoupler(self.config_dict, self.logger)

        recharge_m_d = coupler.extract_recharge_from_summa(summa_dir)
        surface_kg_m2_s = coupler.extract_surface_runoff(summa_dir)
        surface_m3s = surface_kg_m2_s * area_m2 / 1000.0

        # Total simulated streamflow
        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        total_m3s = surface_m3s.loc[common_idx] + baseflow_m3s.loc[common_idx]
        total_m3s.name = 'total_m3s'

        # Load observed streamflow (best-effort)
        obs = self._load_observed_streamflow(experiment_id)

        return {
            'obs': obs,
            'surface_m3s': surface_m3s,
            'baseflow_m3s': baseflow_m3s,
            'total_m3s': total_m3s,
            'head': head,
            'recharge_m_d': recharge_m_d,
            'drain_m3_d': drain_m3_d,
            'top': top,
            'bot': bot,
            'drain_elev': drain_elev,
            'sy': sy,
            'cell_size': cell_size,
        }

    def _load_observed_streamflow(self, experiment_id: str) -> Optional[pd.Series]:
        """Try to load observed streamflow from results CSV.

        Returns:
            Observed series (m3/s) or None if not available.
        """
        results_dir = self.project_dir / 'results'
        for pattern in ['*streamflow*.csv', '*obs*.csv', '*results*.csv']:
            csvs = sorted(results_dir.glob(pattern))
            for csv_path in csvs:
                try:
                    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
                    for col in df.columns:
                        if 'obs' in col.lower():
                            return df[col].dropna()
                except Exception:  # noqa: BLE001 — model execution resilience
                    continue
        return None

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _plot_flow_separation(
        self,
        ax: Any,
        obs: Optional[pd.Series],
        surface_m3s: pd.Series,
        baseflow_m3s: pd.Series,
        total_m3s: pd.Series,
    ) -> None:
        """Stacked area: surface (orange) + baseflow (teal), obs line, total line."""
        common = surface_m3s.index.intersection(baseflow_m3s.index)
        surf = surface_m3s.reindex(common).fillna(0.0)
        base = baseflow_m3s.reindex(common).fillna(0.0)
        total = total_m3s.reindex(common).fillna(0.0)

        ax.fill_between(common, 0, base.values, color='#2ca89a', alpha=0.6, label='Baseflow')
        ax.fill_between(common, base.values, base.values + surf.values,
                        color='#e8873a', alpha=0.6, label='Surface runoff')
        ax.plot(common, total.values, color='#1f77b4', linewidth=1.2, label='Total simulated')

        if obs is not None and len(obs) > 0:
            ax.plot(obs.index, obs.values, color='black', linewidth=1.0, label='Observed')

        self._apply_standard_styling(
            ax, ylabel='Discharge (m\u00b3/s)',
            title='Flow Separation Hydrograph', legend_loc='upper right',
        )
        self._format_date_axis(ax, format_type='month', rotation=30)

    def _plot_metrics_table(
        self,
        ax: Any,
        obs: Optional[pd.Series],
        sim: pd.Series,
    ) -> None:
        """Render a metrics table in the given axis."""
        ax.axis('off')

        if obs is None or len(obs) == 0:
            ax.text(
                0.5, 0.5, 'No observed data\navailable for metrics',
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
            )
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            return

        # Align
        common = obs.index.intersection(sim.index)
        if len(common) == 0:
            ax.text(
                0.5, 0.5, 'No overlapping dates\nfor metric calculation',
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
            )
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            return

        metrics = calculate_metrics(obs.loc[common].values, sim.loc[common].values)

        # Compute volume bias
        obs_sum = obs.loc[common].sum()
        if obs_sum > 0:
            bias_pct = (sim.loc[common].sum() - obs_sum) / obs_sum * 100
        else:
            bias_pct = np.nan
        metrics['Bias%'] = bias_pct

        # Build table
        display_keys = ['KGE', 'KGEp', 'NSE', 'RMSE', 'MAE', 'Bias%']
        cell_text = []
        for k in display_keys:
            val = metrics.get(k, np.nan)
            if k == 'Bias%':
                fmt = f'{val:+.1f}%' if not np.isnan(val) else 'N/A'
            else:
                fmt = f'{val:.3f}' if not np.isnan(val) else 'N/A'
            cell_text.append([k, fmt])

        table = ax.table(
            cellText=cell_text,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0', '#f0f0f0'],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.4)

        # Color-code KGE/NSE rows
        _color_map = {0: 'KGE', 2: 'NSE'}
        for row_idx, key in _color_map.items():
            try:
                val = metrics[key]
                cell = table[(row_idx + 1, 1)]
                if val >= 0.7:
                    cell.set_facecolor('#90EE90')
                elif val >= 0.5:
                    cell.set_facecolor('#FFFFE0')
                else:
                    cell.set_facecolor('#FFB6C1')
            except (KeyError, ValueError):
                pass

        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)

    def _plot_groundwater_head(
        self,
        ax: Any,
        head_series: pd.Series,
        top: Optional[float],
        drain_elev: Optional[float],
        bot: Optional[float],
    ) -> None:
        """Head time series with horizontal reference lines for aquifer geometry."""
        ax.plot(head_series.index, head_series.values, color='#1f77b4', linewidth=1.2,
                label='Head')

        if top is not None:
            ax.axhline(top, color='brown', linestyle='--', linewidth=0.8, label=f'Top ({top:.0f} m)')
        if drain_elev is not None:
            ax.axhline(drain_elev, color='#2ca89a', linestyle='-.', linewidth=0.8,
                        label=f'Drain elev ({drain_elev:.0f} m)')
        if bot is not None:
            ax.axhline(bot, color='grey', linestyle='--', linewidth=0.8, label=f'Bottom ({bot:.0f} m)')

        self._apply_standard_styling(
            ax, ylabel='Head (m)', title='Groundwater Head',
            legend_loc='lower left',
        )
        self._format_date_axis(ax, format_type='month', rotation=30)

    def _plot_recharge_vs_baseflow(
        self,
        ax: Any,
        recharge_m_d: pd.Series,
        drain_m3_d: pd.Series,
    ) -> None:
        """Dual y-axis: recharge bars (left) + drain discharge line (right)."""
        color_rch = '#6baed6'
        color_drn = '#e6550d'

        ax.bar(recharge_m_d.index, recharge_m_d.values, width=1.0,
               color=color_rch, alpha=0.7, label='Recharge (m/d)')
        ax.set_ylabel('Recharge (m/d)', color=color_rch,
                       fontsize=self.plot_config.FONT_SIZE_MEDIUM)
        ax.tick_params(axis='y', labelcolor=color_rch)

        ax2 = ax.twinx()
        ax2.plot(drain_m3_d.index, drain_m3_d.values, color=color_drn,
                 linewidth=1.2, label='Drain discharge (m\u00b3/d)')
        ax2.set_ylabel('Drain discharge (m\u00b3/d)', color=color_drn,
                        fontsize=self.plot_config.FONT_SIZE_MEDIUM)
        ax2.tick_params(axis='y', labelcolor=color_drn)

        ax.set_title('Recharge vs Baseflow', fontsize=self.plot_config.FONT_SIZE_TITLE)
        self._add_grid(ax)
        self._set_background_color(ax)
        self._format_date_axis(ax, format_type='month', rotation=30)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper right',
                  fontsize=self.plot_config.LEGEND_FONT_SIZE,
                  framealpha=self.plot_config.LEGEND_FRAMEALPHA)

    def _plot_fdc_components(
        self,
        ax: Any,
        obs: Optional[pd.Series],
        total: pd.Series,
        surface: pd.Series,
        baseflow: pd.Series,
    ) -> None:
        """4-curve FDC on log-log scale."""
        curves = [
            (total, '#1f77b4', 'Total sim', 1.2),
            (surface, '#e8873a', 'Surface', 0.9),
            (baseflow, '#2ca89a', 'Baseflow', 0.9),
        ]

        if obs is not None and len(obs) > 0:
            exc, srt = calculate_flow_duration_curve(obs.values)
            if len(exc) > 0:
                ax.plot(exc * 100, srt, color='black', linewidth=1.0, label='Observed')

        for series, color, label, lw in curves:
            vals = series.values[~np.isnan(series.values)]
            if len(vals) == 0:
                continue
            exc, srt = calculate_flow_duration_curve(vals)
            if len(exc) > 0:
                ax.plot(exc * 100, srt, color=color, linewidth=lw, label=label)

        ax.set_yscale('log')
        ax.set_xlim(0, 100)
        self._apply_standard_styling(
            ax, xlabel='Exceedance (%)', ylabel='Discharge (m\u00b3/s)',
            title='Flow Duration Curves', legend_loc='upper right',
        )

    def _plot_water_balance(
        self,
        ax: Any,
        recharge_total: float,
        drain_total: float,
        delta_storage: float,
        surface_frac: float,
        baseflow_frac: float,
    ) -> None:
        """Horizontal bar chart of cumulative fluxes + inset pie for flow split."""
        labels = [
            f'Recharge ({recharge_total:.1f} m)',
            f'Drain ({drain_total:.1f} m\u00b3)',
            f'\u0394Storage ({delta_storage:.1f} m\u00b3)',
        ]
        values = [recharge_total, drain_total, abs(delta_storage)]
        colors = ['#6baed6', '#e6550d', '#74c476']

        bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.5)

        # Value annotations
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=9)

        ax.set_xlabel('Cumulative volume', fontsize=self.plot_config.FONT_SIZE_MEDIUM)
        ax.set_title('Water Balance', fontsize=self.plot_config.FONT_SIZE_TITLE)
        self._add_grid(ax, alpha=0.3)
        self._set_background_color(ax)

        # Inset pie for surface vs baseflow split
        inset_ax = ax.inset_axes([0.65, 0.15, 0.30, 0.65])
        wedges, texts, autotexts = inset_ax.pie(
            [surface_frac, baseflow_frac],
            labels=['Surface', 'Baseflow'],
            colors=['#e8873a', '#2ca89a'],
            autopct='%1.0f%%',
            startangle=90,
            textprops={'fontsize': 8},
        )
        for t in autotexts:
            t.set_fontsize(8)
        inset_ax.set_title('Flow split', fontsize=9)
