# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Coupling Diagnostics & Visualization

Generates a multi-panel overview figure showing SUMMA+PIHM coupling:
flow separation hydrograph, groundwater head, recharge vs baseflow,
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
)

logger = logging.getLogger(__name__)


@R.plotters.add('PIHM')
class PIHMPlotter(BasePlotter):
    """Plotter for PIHM coupling diagnostics."""

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
        return self.plot_coupling_results(experiment_id)

    def plot_coupling_results(self, experiment_id: str) -> Optional[str]:
        """Main entry: collect data, build figure, save PNG."""
        try:
            data = self._collect_coupling_data(experiment_id)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not collect PIHM coupling data: {e}", exc_info=True)
            return None

        plt, _ = self._setup_matplotlib()
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

        ax_hydro = fig.add_subplot(gs[0, :])
        ax_head = fig.add_subplot(gs[1, 0])
        ax_fdc = fig.add_subplot(gs[1, 1])

        # Flow separation hydrograph
        self._plot_flow_separation(
            ax_hydro, data.get('obs'), data['surface_m3s'],
            data['baseflow_m3s'], data['total_m3s'],
        )

        # Groundwater head
        if 'head' in data and len(data['head']) > 0:
            ax_head.plot(data['head'].index, data['head'].values,
                         color='#1f77b4', linewidth=1.2, label='GW Head')
            self._apply_standard_styling(
                ax_head, ylabel='Head (m)', title='Groundwater Head',
                legend_loc='lower left',
            )
            self._format_date_axis(ax_head, format_type='month', rotation=30)
        else:
            ax_head.text(0.5, 0.5, 'No GW head data', transform=ax_head.transAxes,
                         ha='center', va='center')
            ax_head.set_title('Groundwater Head')

        # Flow duration curves
        self._plot_fdc(ax_fdc, data.get('obs'), data['total_m3s'],
                       data['surface_m3s'], data['baseflow_m3s'])

        fig.suptitle(
            f'PIHM Coupling Overview — {experiment_id}',
            fontsize=14, fontweight='bold', y=0.98,
        )

        output_dir = self._ensure_output_dir('pihm_coupling')
        output_path = output_dir / f'{experiment_id}_coupling_overview.png'
        return self._save_and_close(fig, output_path)

    def _collect_coupling_data(self, experiment_id: str) -> Dict[str, Any]:
        """Load coupling data from output directories."""
        from symfluence.models.pihm.coupling import SUMAToPIHMCoupler
        from symfluence.models.pihm.extractor import PIHMResultExtractor

        sim_base = self.project_dir / 'simulations' / experiment_id
        pihm_dir = sim_base / 'PIHM'
        summa_dir = sim_base / 'SUMMA'

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

        extractor = PIHMResultExtractor()

        river_flux = extractor.extract_variable(
            pihm_dir, 'river_flux', start_date=start_date,
        )
        baseflow_m3s = river_flux

        try:
            head = extractor.extract_variable(
                pihm_dir, 'groundwater_head', start_date=start_date,
            )
        except Exception:  # noqa: BLE001 — model execution resilience
            head = pd.Series(dtype=float)

        coupler = SUMAToPIHMCoupler(self.config_dict, self.logger)
        surface_kg_m2_s = coupler.extract_surface_runoff(summa_dir)
        surface_m3s = surface_kg_m2_s * area_m2 / 1000.0

        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        total_m3s = surface_m3s.loc[common_idx] + baseflow_m3s.loc[common_idx]
        total_m3s.name = 'total_m3s'

        obs = self._load_observed_streamflow(experiment_id)

        return {
            'obs': obs,
            'surface_m3s': surface_m3s,
            'baseflow_m3s': baseflow_m3s,
            'total_m3s': total_m3s,
            'head': head,
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

    def _plot_flow_separation(self, ax, obs, surface_m3s, baseflow_m3s, total_m3s):
        """Stacked area: surface + baseflow, obs line, total line."""
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

    def _plot_fdc(self, ax, obs, total, surface, baseflow):
        """Flow duration curves."""
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
