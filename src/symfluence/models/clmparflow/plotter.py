# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Diagnostics & Visualization

Generates a multi-panel overview figure showing CLMParFlow results:
flow separation hydrograph, pressure head, flow duration curves,
and performance metrics.
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


@R.plotters.add('CLMPARFLOW')
class CLMParFlowPlotter(BasePlotter):
    """Plotter for CLMParFlow diagnostics.

    Generates a 2x2 figure with:
    - Flow separation hydrograph (obs, overland, subsurface, total)
    - Performance metrics table
    - Pressure head time series
    - Flow duration curves
    """

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Entry point for BasePlotter interface."""
        experiment_id = kwargs.get(
            'experiment_id',
            self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='run_1',
                dict_key='EXPERIMENT_ID',
            ),
        )
        return self.plot_results(experiment_id)

    def plot_results(self, experiment_id: str) -> Optional[str]:
        """Main entry: collect data, build 4-panel figure, save PNG."""
        try:
            data = self._collect_data(experiment_id)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not collect CLMParFlow data: {e}", exc_info=True)
            return None

        plt, _ = self._setup_matplotlib()
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

        ax_hydro = fig.add_subplot(gs[0, 0])
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_press = fig.add_subplot(gs[1, 0])
        ax_fdc = fig.add_subplot(gs[1, 1])

        # Panel 1: Flow hydrograph
        self._plot_flow_hydrograph(ax_hydro, data)

        # Panel 2: Metrics table
        self._plot_metrics_table(ax_metrics, data.get('obs'), data['total_m3s'])

        # Panel 3: Pressure head
        self._plot_pressure_head(ax_press, data['pressure'], data.get('top'), data.get('bot'))

        # Panel 4: FDC
        self._plot_fdc(ax_fdc, data)

        fig.suptitle(
            f'CLMParFlow Overview \u2014 {experiment_id}',
            fontsize=14, fontweight='bold', y=0.98,
        )

        output_dir = self._ensure_output_dir('clmparflow')
        output_path = output_dir / f'{experiment_id}_clmparflow_overview.png'
        return self._save_and_close(fig, output_path)

    def _collect_data(self, experiment_id: str) -> Dict[str, Any]:
        """Load CLMParFlow data from output directories."""
        from .extractor import CLMParFlowResultExtractor

        sim_base = self.project_dir / 'simulations' / experiment_id
        clmpf_dir = sim_base / 'CLMPARFLOW'

        timestep_hours = float(self._get_config_value(
            lambda: self.config.model.clmparflow.timestep_hours,
            default=1.0, dict_key='CLMPARFLOW_TIMESTEP_HOURS',
        ))
        start_date = str(self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01', dict_key='EXPERIMENT_TIME_START',
        ))
        k_sat = float(self._get_config_value(
            lambda: self.config.model.clmparflow.k_sat,
            default=5.0, dict_key='CLMPARFLOW_K_SAT',
        ))
        dx = float(self._get_config_value(
            lambda: self.config.model.clmparflow.dx,
            default=1000.0, dict_key='CLMPARFLOW_DX',
        ))
        dy = float(self._get_config_value(
            lambda: self.config.model.clmparflow.dy,
            default=1000.0, dict_key='CLMPARFLOW_DY',
        ))
        dz = float(self._get_config_value(
            lambda: self.config.model.clmparflow.dz,
            default=2.0, dict_key='CLMPARFLOW_DZ',
        ))
        top = float(self._get_config_value(
            lambda: self.config.model.clmparflow.top,
            default=2.0, dict_key='CLMPARFLOW_TOP',
        ))
        bot = float(self._get_config_value(
            lambda: self.config.model.clmparflow.bot,
            default=0.0, dict_key='CLMPARFLOW_BOT',
        ))

        from symfluence.models.parflow.extractor import resolve_dump_interval_hours
        common_kwargs = dict(
            timestep_hours=timestep_hours,
            dump_interval_hours=resolve_dump_interval_hours(self.config, 'clmparflow'),
            start_date=start_date,
            k_sat=k_sat, dx=dx, dy=dy, dz=dz,
        )

        extractor = CLMParFlowResultExtractor()

        pressure = extractor.extract_variable(clmpf_dir, 'pressure', **common_kwargs)
        overland_m3s = extractor.extract_variable(clmpf_dir, 'overland_flow', **common_kwargs)
        subsurface_m3hr = extractor.extract_variable(clmpf_dir, 'subsurface_drainage', **common_kwargs)
        subsurface_m3s = subsurface_m3hr / 3600.0

        common_idx = overland_m3s.index.intersection(subsurface_m3s.index)
        total_m3s = overland_m3s.loc[common_idx] + subsurface_m3s.loc[common_idx]
        total_m3s.name = 'total_m3s'

        obs = self._load_observed_streamflow(experiment_id)

        return {
            'obs': obs,
            'overland_m3s': overland_m3s,
            'subsurface_m3s': subsurface_m3s,
            'total_m3s': total_m3s,
            'pressure': pressure,
            'top': top,
            'bot': bot,
        }

    def _load_observed_streamflow(self, experiment_id: str) -> Optional[pd.Series]:
        """Try to load observed streamflow from results CSV."""
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

    def _plot_flow_hydrograph(self, ax: Any, data: Dict) -> None:
        """Stacked area: overland + subsurface, obs line, total line."""
        common = data['overland_m3s'].index.intersection(data['subsurface_m3s'].index)
        surf = data['overland_m3s'].reindex(common).fillna(0.0)
        sub = data['subsurface_m3s'].reindex(common).fillna(0.0)
        total = data['total_m3s'].reindex(common).fillna(0.0)

        ax.fill_between(common, 0, sub.values, color='#2ca89a', alpha=0.6, label='Subsurface')
        ax.fill_between(common, sub.values, sub.values + surf.values,
                        color='#e8873a', alpha=0.6, label='Overland flow')
        ax.plot(common, total.values, color='#1f77b4', linewidth=1.2, label='Total simulated')

        if data.get('obs') is not None and len(data['obs']) > 0:
            ax.plot(data['obs'].index, data['obs'].values, color='black', linewidth=1.0,
                    label='Observed')

        self._apply_standard_styling(
            ax, ylabel='Discharge (m\u00b3/s)',
            title='Flow Separation Hydrograph', legend_loc='upper right',
        )
        self._format_date_axis(ax, format_type='month', rotation=30)

    def _plot_metrics_table(self, ax: Any, obs: Optional[pd.Series], sim: pd.Series) -> None:
        """Render a metrics table."""
        ax.axis('off')

        if obs is None or len(obs) == 0:
            ax.text(0.5, 0.5, 'No observed data\navailable for metrics',
                    transform=ax.transAxes, ha='center', va='center', fontsize=11)
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            return

        common = obs.index.intersection(sim.index)
        if len(common) == 0:
            ax.text(0.5, 0.5, 'No overlapping dates\nfor metric calculation',
                    transform=ax.transAxes, ha='center', va='center', fontsize=11)
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            return

        metrics = calculate_metrics(obs.loc[common].values, sim.loc[common].values)

        display_keys = ['KGE', 'KGEp', 'NSE', 'RMSE', 'MAE']
        cell_text = []
        for k in display_keys:
            val = metrics.get(k, np.nan)
            fmt = f'{val:.3f}' if not np.isnan(val) else 'N/A'
            cell_text.append([k, fmt])

        table = ax.table(
            cellText=cell_text, colLabels=['Metric', 'Value'],
            cellLoc='center', loc='center', colColours=['#f0f0f0', '#f0f0f0'],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.4)
        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)

    def _plot_pressure_head(self, ax: Any, pressure: pd.Series,
                            top: Optional[float], bot: Optional[float]) -> None:
        """Pressure head time series."""
        ax.plot(pressure.index, pressure.values, color='#1f77b4', linewidth=1.2)
        if top is not None:
            ax.axhline(top, color='brown', linestyle='--', linewidth=0.8, label=f'Surface ({top:.0f} m)')
        if bot is not None:
            ax.axhline(bot, color='grey', linestyle='--', linewidth=0.8, label=f'Bottom ({bot:.0f} m)')
        self._apply_standard_styling(
            ax, ylabel='Pressure head (m)', title='Pressure Head', legend_loc='lower left',
        )
        self._format_date_axis(ax, format_type='month', rotation=30)

    def _plot_fdc(self, ax: Any, data: Dict) -> None:
        """Flow duration curves."""
        total = data['total_m3s']
        vals = total.values[~np.isnan(total.values)]
        if len(vals) > 0:
            exc, srt = calculate_flow_duration_curve(vals)
            if len(exc) > 0:
                ax.plot(exc * 100, srt, color='#1f77b4', linewidth=1.2, label='CLMParFlow')

        if data.get('obs') is not None and len(data['obs']) > 0:
            exc_obs, srt_obs = calculate_flow_duration_curve(data['obs'].values)
            if len(exc_obs) > 0:
                ax.plot(exc_obs * 100, srt_obs, color='black', linewidth=1.0, label='Observed')

        ax.set_yscale('log')
        ax.set_xlim(0, 100)
        self._apply_standard_styling(
            ax, xlabel='Exceedance (%)', ylabel='Discharge (m\u00b3/s)',
            title='Flow Duration Curve', legend_loc='upper right',
        )
