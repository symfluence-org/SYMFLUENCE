# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
T-Route Routing Diagnostics & Visualization

Generates a 2x2 multi-panel overview figure showing t-route routing results:
routed vs observed hydrograph, performance metrics table, flow duration
curves, and routing attenuation (lateral inflow vs routed discharge).
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


@R.plotters.add('TROUTE')
class TRoutePlotter(BasePlotter):
    """Plotter for t-route routing diagnostics.

    Generates a 2x2 GridSpec figure with:
    - Panel 1: Routed vs observed hydrograph
    - Panel 2: Performance metrics table (KGE, NSE, RMSE, Bias%)
    - Panel 3: Flow duration curves (observed vs routed)
    - Panel 4: Routing attenuation (lateral inflow vs routed discharge)
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
        return self.plot_routing_results(experiment_id)

    def plot_routing_results(self, experiment_id: str) -> Optional[str]:
        """Main entry: collect data, build 4-panel figure, save PNG.

        Args:
            experiment_id: Experiment identifier for locating output files.

        Returns:
            Path to saved PNG or None on failure.
        """
        try:
            data = self._collect_routing_data(experiment_id)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not collect t-route routing data: {e}", exc_info=True)
            return None

        plt, _ = self._setup_matplotlib()
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

        ax_hydro = fig.add_subplot(gs[0, 0])
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_fdc = fig.add_subplot(gs[1, 0])
        ax_atten = fig.add_subplot(gs[1, 1])

        # Panel 1: Routed vs observed hydrograph
        self._plot_hydrograph(
            ax_hydro,
            data.get('obs'),
            data['routed_m3s'],
        )

        # Panel 2: Performance metrics table
        self._plot_metrics_table(ax_metrics, data.get('obs'), data['routed_m3s'])

        # Panel 3: Flow duration curves
        self._plot_fdc(ax_fdc, data.get('obs'), data['routed_m3s'])

        # Panel 4: Routing attenuation
        self._plot_attenuation(
            ax_atten,
            data.get('lateral_inflow_m3s'),
            data['routed_m3s'],
        )

        fig.suptitle(
            f'T-Route Routing Overview \u2014 {experiment_id}',
            fontsize=14,
            fontweight='bold',
            y=0.98,
        )

        output_dir = self._ensure_output_dir('troute_routing')
        output_path = output_dir / f'{experiment_id}_routing_overview.png'
        return self._save_and_close(fig, output_path)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_routing_data(self, experiment_id: str) -> Dict[str, Any]:
        """Load routing data from output directories."""

        sim_base = self.project_dir / 'simulations' / experiment_id

        output_dir_cfg = self._get_config_value(
            lambda: self.config.model.troute.experiment_output,
            dict_key='EXPERIMENT_OUTPUT_TROUTE',
        )
        if output_dir_cfg != 'default':
            output_dir = self.project_dir / output_dir_cfg
        else:
            # Try TRoute first (runner's _get_model_name returns "TRoute"), then TROUTE
            output_dir = sim_base / 'TRoute'
            if not output_dir.exists():
                output_dir = sim_base / 'TROUTE'

        # Load routed discharge from flowveldepth output
        routed_m3s = self._load_routed_discharge(output_dir)

        # Load lateral inflow if available
        lateral_inflow_m3s = self._load_lateral_inflow(output_dir)

        # Load observed streamflow (best-effort)
        obs = self._load_observed_streamflow(experiment_id)

        return {
            'obs': obs,
            'routed_m3s': routed_m3s,
            'lateral_inflow_m3s': lateral_inflow_m3s,
        }

    def _load_routed_discharge(self, output_dir) -> pd.Series:
        """Load routed discharge from t-route output files."""
        from pathlib import Path

        import xarray as xr

        output_dir = Path(output_dir)

        # Check for built-in output (troute_output.nc) and nwm_routing output (flowveldepth)
        nc_candidates = (
            sorted(output_dir.glob('troute_output.nc'))
            + sorted(output_dir.glob('*flowveldepth*.nc'))
            + sorted(output_dir.glob('nex-troute-out.nc'))
            + sorted(output_dir.glob('*.nc'))
        )

        for nc_file in nc_candidates:
            ds = xr.open_dataset(nc_file)
            for var in ['flow', 'streamflow', 'discharge', 'q_lateral']:
                if var in ds:
                    var_data = ds[var]
                    # Select outlet reach (highest mean flow)
                    if 'feature_id' in var_data.dims:
                        feature_means = var_data.mean(dim='time').values
                        outlet_idx = int(np.argmax(feature_means))
                        series = var_data.isel(feature_id=outlet_idx).to_series()
                    elif 'seg' in var_data.dims:
                        seg_means = var_data.mean(dim='time').values
                        outlet_idx = int(np.argmax(seg_means))
                        series = var_data.isel(seg=outlet_idx).to_series()
                    else:
                        series_candidate = var_data.squeeze().to_pandas()
                        if isinstance(series_candidate, pd.Series):
                            series = series_candidate
                        elif isinstance(series_candidate, pd.DataFrame):
                            if series_candidate.empty:
                                continue
                            series = series_candidate.iloc[:, 0]
                        else:
                            series = pd.Series(series_candidate)

                    # Resample to daily
                    series.index = pd.to_datetime(series.index)
                    series = series.resample('D').mean()
                    ds.close()
                    return series
            ds.close()

        # Fallback: CSV output
        csv_files = sorted(output_dir.glob('*flowveldepth*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[-1], parse_dates=[0], index_col=0)
            for col in df.columns:
                if 'flow' in col.lower() or 'q' in col.lower():
                    return df[col].dropna()

        raise FileNotFoundError(
            f"No t-route output found in {output_dir}"
        )

    def _load_lateral_inflow(self, output_dir) -> Optional[pd.Series]:
        """Load total lateral inflow from SUMMA output (sum across HRUs)."""
        from pathlib import Path

        import xarray as xr

        output_dir = Path(output_dir)

        # Check t-route output dir for qlateral files
        nc_files = sorted(output_dir.glob('*qlateral*.nc'))
        if nc_files:
            ds = xr.open_dataset(nc_files[-1])
            for var in ['qlateral', 'q_lateral', 'lateral_inflow']:
                if var in ds:
                    series = ds[var].to_series()
                    ds.close()
                    return series
            ds.close()

        # Load from SUMMA output (upstream model) and sum HRU runoff
        sim_base = output_dir.parent  # simulations/{exp_id}
        summa_dir = sim_base / 'SUMMA'
        if summa_dir.exists():
            summa_files = sorted(summa_dir.glob('*_timestep.nc'))
            if summa_files:
                try:
                    ds = xr.open_dataset(summa_files[0])
                    for var in ['q_lateral', 'averageRoutedRunoff']:
                        if var in ds:
                            var_data = ds[var]
                            # Sum across all HRUs to get total lateral inflow
                            spatial_dims = [d for d in var_data.dims if d != 'time']
                            if spatial_dims:
                                if 'HRUarea' in ds:
                                    total = (var_data * ds['HRUarea']).sum(dim=spatial_dims)
                                else:
                                    total = var_data.sum(dim=spatial_dims)
                            else:
                                total = var_data.squeeze()
                            series = total.to_pandas()
                            series.index = pd.to_datetime(series.index)
                            series = series.resample('D').mean()
                            ds.close()
                            return series
                    ds.close()
                except Exception:  # noqa: BLE001 — model execution resilience
                    pass

        return None

    def _load_observed_streamflow(self, experiment_id: str) -> Optional[pd.Series]:
        """Try to load observed streamflow from results CSV or observations dir."""
        # Check auto-generated results first
        search_dirs = [
            self.project_dir / 'results',
            self.project_observations_dir / 'streamflow',
            self.project_observations_dir,
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in [f'{experiment_id}_results.csv', '*streamflow*.csv', '*obs*.csv', '*results*.csv']:
                csvs = sorted(search_dir.glob(pattern))
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

    def _plot_hydrograph(
        self,
        ax: Any,
        obs: Optional[pd.Series],
        routed: pd.Series,
    ) -> None:
        """Routed discharge line with optional observed overlay."""
        ax.plot(routed.index, routed.values, color='#1f77b4', linewidth=1.2,
                label='Routed (t-route)')

        if obs is not None and len(obs) > 0:
            ax.plot(obs.index, obs.values, color='black', linewidth=1.0,
                    label='Observed')

        self._apply_standard_styling(
            ax, ylabel='Discharge (m\u00b3/s)',
            title='Routed vs Observed Hydrograph', legend_loc='upper right',
        )
        self._format_date_axis(ax, format_type='month', rotation=30)

    def _plot_metrics_table(
        self,
        ax: Any,
        obs: Optional[pd.Series],
        sim: pd.Series,
    ) -> None:
        """Render a performance metrics table."""
        ax.axis('off')

        if obs is None or len(obs) == 0:
            ax.text(
                0.5, 0.5, 'No observed data\navailable for metrics',
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
            )
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            return

        common = obs.index.intersection(sim.index)
        if len(common) == 0:
            ax.text(
                0.5, 0.5, 'No overlapping dates\nfor metric calculation',
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
            )
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            return

        metrics = calculate_metrics(obs.loc[common].values, sim.loc[common].values)

        obs_sum = obs.loc[common].sum()
        if obs_sum > 0:
            bias_pct = (sim.loc[common].sum() - obs_sum) / obs_sum * 100
        else:
            bias_pct = np.nan
        metrics['Bias%'] = bias_pct

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

    def _plot_fdc(
        self,
        ax: Any,
        obs: Optional[pd.Series],
        routed: pd.Series,
    ) -> None:
        """Flow duration curves for observed and routed discharge."""
        if obs is not None and len(obs) > 0:
            exc, srt = calculate_flow_duration_curve(obs.values)
            if len(exc) > 0:
                ax.plot(exc * 100, srt, color='black', linewidth=1.0,
                        label='Observed')

        vals = routed.values[~np.isnan(routed.values)]
        if len(vals) > 0:
            exc, srt = calculate_flow_duration_curve(vals)
            if len(exc) > 0:
                ax.plot(exc * 100, srt, color='#1f77b4', linewidth=1.2,
                        label='Routed (t-route)')

        ax.set_yscale('log')
        ax.set_xlim(0, 100)
        self._apply_standard_styling(
            ax, xlabel='Exceedance (%)', ylabel='Discharge (m\u00b3/s)',
            title='Flow Duration Curves', legend_loc='upper right',
        )

    def _plot_attenuation(
        self,
        ax: Any,
        lateral_inflow: Optional[pd.Series],
        routed: pd.Series,
    ) -> None:
        """Compare lateral inflow vs routed discharge to show routing attenuation."""
        if lateral_inflow is not None and len(lateral_inflow) > 0:
            common = lateral_inflow.index.intersection(routed.index)
            lat = lateral_inflow.reindex(common).fillna(0.0)
            rout = routed.reindex(common).fillna(0.0)

            ax.fill_between(
                common, lat.values, rout.values,
                alpha=0.3, color='#ff7f0e', label='Attenuation',
            )
            ax.plot(common, lat.values, color='#ff7f0e', linewidth=1.0,
                    linestyle='--', label='Lateral inflow')
            ax.plot(common, rout.values, color='#1f77b4', linewidth=1.2,
                    label='Routed discharge')

            self._apply_standard_styling(
                ax, ylabel='Discharge (m\u00b3/s)',
                title='Routing Attenuation', legend_loc='upper right',
            )
            self._format_date_axis(ax, format_type='month', rotation=30)
        else:
            ax.text(
                0.5, 0.5, 'No lateral inflow data\navailable',
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
            )
            ax.set_title('Routing Attenuation', fontsize=12, fontweight='bold')
            self._set_background_color(ax)
