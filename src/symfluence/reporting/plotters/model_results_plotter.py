# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model-specific result plotting.

This module provides specialized visualization for different hydrological
model outputs including SUMMA, FUSE, NGEN, LSTM, and HYPE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.constants import ConfigKeys
from symfluence.core.path_resolver import find_basin_shapefile
from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.shapefile_helper import resolve_default_name


class ModelResultsPlotter(BasePlotter):
    """
    Plotter for model-specific outputs.

    Handles visualization of outputs from:
    - SUMMA
    - FUSE
    - NGEN
    - LSTM
    - HYPE
    """

    @BasePlotter._plot_safe("plot_fuse_streamflow")
    def plot_fuse_streamflow(
        self,
        model_outputs: List[Tuple[str, str]],
        obs_files: List[Tuple[str, str]]
    ) -> Optional[str]:
        """
        Visualize FUSE simulated streamflow against observations.

        Args:
            model_outputs: List of tuples (model_name, output_file)
            obs_files: List of tuples (obs_name, obs_file)

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import geopandas as gpd
        import xarray as xr

        from symfluence.core.constants import UnitConversion
        from symfluence.reporting.core.plot_utils import calculate_metrics

        plot_dir = self._ensure_output_dir('results')
        exp_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='FUSE', dict_key=ConfigKeys.EXPERIMENT_ID)
        plot_filename = plot_dir / f"{exp_id}_FUSE_streamflow_comparison.png"

        fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)

        # Handle observations
        obs_dfs = []
        for _, obs_file in obs_files:
            df = pd.read_csv(obs_file, parse_dates=['datetime'])
            df.set_index('datetime', inplace=True)
            obs_dfs.append(df)

        # Handle FUSE output
        for model_name, output_file in model_outputs:
            if model_name.upper() == 'FUSE':
                with xr.open_dataset(output_file) as ds:
                    sim_flow = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0).to_series()

                    # Unit conversion (mm/day to cms)
                    basin_name = resolve_default_name(
                        self.config,
                        'RIVER_BASINS_NAME',
                        '{domain}_riverBasins_delineate.shp'
                    )

                    basin_path = self.project_dir / 'shapefiles' / 'river_basins' / basin_name
                    if not basin_path.exists():
                        basin_path = Path(self._get_config_value(lambda: self.config.paths.river_basins_path, default='', dict_key=ConfigKeys.RIVER_BASINS_PATH))

                    if basin_path.exists():
                        basin_gdf = gpd.read_file(basin_path)
                        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
                        sim_flow = sim_flow * area_km2 / UnitConversion.MM_DAY_TO_CMS

                    if obs_dfs:
                        start_date = max(sim_flow.index.min(), obs_dfs[0].index.min())
                        end_date = min(sim_flow.index.max(), obs_dfs[0].index.max())

                        sim_plot = sim_flow.loc[start_date:end_date]
                        obs_plot = obs_dfs[0]['discharge_cms'].loc[start_date:end_date]

                        ax.plot(sim_plot.index, sim_plot, label='FUSE', color=self.plot_config.COLOR_SIMULATED_PRIMARY)
                        ax.plot(obs_plot.index, obs_plot, label='Observed', color=self.plot_config.COLOR_OBSERVED)

                        metrics = calculate_metrics(obs_plot.values, sim_plot.values)
                        self._add_metrics_text(ax, metrics)

        self._apply_standard_styling(
            ax, xlabel='Date', ylabel='Streamflow (m³/s)',
            title='FUSE Streamflow Comparison', legend=True
        )
        self._format_date_axis(ax, format_type='month')

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    @BasePlotter._plot_safe("plot_summa_outputs")
    def plot_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """
        Create spatial and temporal visualizations for SUMMA output variables.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping variable names to plot paths
        """
        plt, _ = self._setup_matplotlib()
        import geopandas as gpd
        import xarray as xr
        from matplotlib import gridspec

        plot_paths: Dict[str, str] = {}

        summa_file = self.get_simulation_dir('SUMMA', experiment_id) / f"{experiment_id}_day.nc"
        if not summa_file.exists():
            return {}

        plot_dir = self._ensure_output_dir('summa_outputs', experiment_id)
        ds = xr.open_dataset(summa_file)

        hru_shp = find_basin_shapefile(
            self.project_dir / 'shapefiles', self.domain_name,
            self.domain_definition_method, self.experiment_id,
            include_river_basins=False,
        )
        hru_gdf = gpd.read_file(hru_shp) if hru_shp is not None else None

        skip_vars = {'hru', 'time', 'gru', 'dateId', 'latitude', 'longitude', 'hruId', 'gruId'}

        for var_name in ds.data_vars:
            if var_name in skip_vars or 'time' not in ds[var_name].dims:
                continue

            fig = plt.figure(figsize=self.plot_config.FIGURE_SIZE_MEDIUM_TALL)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])
            ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

            var_mean = ds[var_name].mean(dim='time').compute()
            if hru_gdf is not None:
                plot_gdf = hru_gdf.copy()
                plot_gdf['value'] = var_mean.values
                plot_gdf = plot_gdf.to_crs(epsg=3857)
                vmin, vmax = np.percentile(var_mean.values, [2, 98])
                plot_gdf.plot(column='value', ax=ax1, vmin=vmin, vmax=vmax, cmap='RdYlBu', legend=True)
                ax1.set_axis_off()

            mean_ts = ds[var_name].mean(dim='hru').compute()
            ax2.plot(mean_ts.time, mean_ts, color=self.plot_config.COLOR_SIMULATED_PRIMARY)
            self._apply_standard_styling(ax2, xlabel='Date', ylabel=var_name, title=f'Mean Time Series: {var_name}', legend=False)
            self._format_date_axis(ax2)

            plot_file = plot_dir / f'{var_name}.png'
            self._save_and_close(fig, plot_file)
            plot_paths[str(var_name)] = str(plot_file)

        ds.close()

        return plot_paths

    @BasePlotter._plot_safe("plot_ngen_results")
    def plot_ngen_results(
        self,
        sim_df: pd.DataFrame,
        obs_df: Optional[pd.DataFrame],
        experiment_id: str,
        results_dir: Path
    ) -> Optional[str]:
        """
        Visualize NGen streamflow plots.

        Args:
            sim_df: Simulated streamflow dataframe
            obs_df: Observed streamflow dataframe (optional)
            experiment_id: Experiment identifier
            results_dir: Results directory

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_flow_duration_curve, calculate_metrics

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.plot_config.FIGURE_SIZE_LARGE)
        ax1.plot(sim_df['datetime'], sim_df['streamflow_cms'], label='NGEN Simulated', color=self.plot_config.COLOR_SIMULATED_PRIMARY)

        if obs_df is not None:
            ax1.plot(obs_df['datetime'], obs_df['streamflow_cms'], label='Observed', color=self.plot_config.COLOR_OBSERVED, alpha=0.7)
            merged = pd.merge(sim_df, obs_df, on='datetime', suffixes=('_sim', '_obs'))
            if not merged.empty:
                self._add_metrics_text(ax1, calculate_metrics(merged['streamflow_cms_obs'].values, merged['streamflow_cms_sim'].values))

        self._apply_standard_styling(ax1, ylabel='Streamflow (cms)', title=f'NGEN Streamflow - {experiment_id}')
        self._format_date_axis(ax1, format_type='month')

        exc_sim, flows_sim = calculate_flow_duration_curve(sim_df['streamflow_cms'].values)
        ax2.semilogy(exc_sim, flows_sim, label='NGEN Simulated', color=self.plot_config.COLOR_SIMULATED_PRIMARY)

        if obs_df is not None:
            exc_obs, flows_obs = calculate_flow_duration_curve(obs_df['streamflow_cms'].values)
            ax2.semilogy(exc_obs, flows_obs, label='Observed', color=self.plot_config.COLOR_OBSERVED)

        self._apply_standard_styling(ax2, xlabel='Exceedance Probability (%)', ylabel='Streamflow (cms)', title='Flow Duration Curve')

        plot_file = self._ensure_output_dir('results') / f"ngen_streamflow_{experiment_id}.png"
        return self._save_and_close(fig, plot_file)

    @BasePlotter._plot_safe("plot_lstm_results")
    def plot_lstm_results(
        self,
        results_df: pd.DataFrame,
        obs_streamflow: pd.DataFrame,
        obs_snow: pd.DataFrame,
        use_snow: bool,
        output_dir: Path,
        experiment_id: str
    ) -> Optional[str]:
        """
        Visualize LSTM simulation results.

        Args:
            results_df: Simulation results dataframe
            obs_streamflow: Observed streamflow dataframe
            obs_snow: Observed snow dataframe
            use_snow: Whether snow metrics/plots are required
            output_dir: Output directory
            experiment_id: Experiment identifier

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from matplotlib.gridspec import GridSpec

        from symfluence.reporting.core.plot_utils import calculate_metrics

        sim_dates, sim_q = results_df.index, results_df['predicted_streamflow']
        obs_q = obs_streamflow.reindex(sim_dates)['streamflow']

        fig = plt.figure(figsize=self.plot_config.FIGURE_SIZE_LARGE)
        gs = GridSpec(2 if use_snow else 1, 1)
        ax1 = fig.add_subplot(gs[0])

        ax1.plot(sim_dates, sim_q, label='LSTM simulated', color='blue')
        ax1.plot(sim_dates, obs_q, label='Observed', color='red')
        self._add_metrics_text(ax1, calculate_metrics(obs_q.values, sim_q.values), label="Streamflow")
        self._apply_standard_styling(ax1, ylabel='Streamflow (m³/s)', title='Observed vs Simulated Streamflow')
        self._format_date_axis(ax1)

        if use_snow and not obs_snow.empty and 'predicted_SWE' in results_df.columns:
            ax2 = fig.add_subplot(gs[1])
            sim_swe, obs_swe = results_df['predicted_SWE'], obs_snow.reindex(sim_dates)['snw']
            ax2.plot(sim_dates, sim_swe, label='LSTM simulated', color='blue')
            ax2.plot(sim_dates, obs_swe, label='Observed', color='red')
            self._add_metrics_text(ax2, calculate_metrics(obs_swe.values, sim_swe.values), label="SWE")
            self._apply_standard_styling(ax2, ylabel='SWE (mm)', title='Observed vs Simulated SWE')
            self._format_date_axis(ax2)

        plot_file = self._ensure_output_dir('results') / f"{experiment_id}_LSTM_results.png"
        return self._save_and_close(fig, plot_file)

    @BasePlotter._plot_safe("plot_hype_results")
    def plot_hype_results(
        self,
        sim_flow: pd.DataFrame,
        obs_flow: pd.DataFrame,
        outlet_id: str,
        domain_name: str,
        experiment_id: str,
        project_dir: Path
    ) -> Optional[str]:
        """
        Visualize HYPE streamflow comparison.

        Args:
            sim_flow: Simulated streamflow dataframe
            obs_flow: Observed streamflow dataframe
            outlet_id: Outlet ID
            domain_name: Domain name
            experiment_id: Experiment identifier
            project_dir: Project directory

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)
        ax.plot(sim_flow.index, sim_flow['HYPE_discharge_cms'], label='Simulated', color='blue')
        ax.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red')
        self._apply_standard_styling(ax, ylabel='Discharge (m³/s)', title=f'Streamflow Comparison - {domain_name}\nOutlet ID: {outlet_id}')
        self._format_date_axis(ax)

        plot_file = self._ensure_output_dir("results") / f"{experiment_id}_HYPE_comparison.png"
        return self._save_and_close(fig, plot_file)

    @BasePlotter._plot_safe("plot_timeseries_results")
    def plot_timeseries_results(
        self,
        df: pd.DataFrame,
        experiment_id: str,
        domain_name: str
    ) -> Optional[str]:
        """
        Create timeseries comparison plot from consolidated results DataFrame.

        Args:
            df: Consolidated results DataFrame
            experiment_id: Experiment identifier
            domain_name: Domain name

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_metrics

        plot_dir = self._ensure_output_dir('results')
        plot_file = plot_dir / f'{experiment_id}_timeseries_comparison.png'

        fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_LARGE)

        # Find models in columns
        models = [c.replace('_discharge_cms', '') for c in df.columns if '_discharge_cms' in c]

        # Plot models
        for i, model in enumerate(models):
            col = f"{model}_discharge_cms"
            color = self.plot_config.get_color_from_palette(i)
            style = self.plot_config.get_line_style(i)

            metrics = calculate_metrics(df['Observed'].values, df[col].values)
            kge = metrics.get('KGE', np.nan)
            label = f'{model} (KGE: {kge:.3f})'

            ax.plot(df.index, df[col], label=label, color=color, linestyle=style, alpha=0.6)

        # Plot Observed on top
        ax.plot(df.index, df['Observed'], color=self.plot_config.COLOR_OBSERVED,
               label='Observed', linewidth=self.plot_config.LINE_WIDTH_OBSERVED, zorder=10)

        self._apply_standard_styling(
            ax, ylabel='Discharge (m³/s)',
            title=f'Streamflow Comparison - {domain_name}',
            legend=True
        )
        self._format_date_axis(ax)

        plt.tight_layout()
        return self._save_and_close(fig, plot_file)

    @BasePlotter._plot_safe("plot_diagnostics")
    def plot_diagnostics(
        self,
        df: pd.DataFrame,
        experiment_id: str,
        domain_name: str
    ) -> Optional[str]:
        """
        Create diagnostic plots (scatter and FDC) for each model.

        Args:
            df: Consolidated results DataFrame
            experiment_id: Experiment identifier
            domain_name: Domain name

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_flow_duration_curve, calculate_metrics

        plot_dir = self._ensure_output_dir('results')
        plot_file = plot_dir / f'{experiment_id}_diagnostic_plots.png'

        models = [c.replace('_discharge_cms', '') for c in df.columns if '_discharge_cms' in c]
        n_models = len(models)
        if n_models == 0:
            return None

        fig = plt.figure(figsize=(15, 5 * n_models))
        gs = plt.GridSpec(n_models, 2)

        for i, model in enumerate(models):
            col = f"{model}_discharge_cms"
            color = self.plot_config.get_color_from_palette(i)

            # Scatter plot
            ax_scatter = fig.add_subplot(gs[i, 0])
            ax_scatter.scatter(df['Observed'], df[col], alpha=0.5, s=10, color=color)

            # 1:1 line
            max_val = max(df['Observed'].max(), df[col].max())
            ax_scatter.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

            metrics = calculate_metrics(df['Observed'].values, df[col].values)
            self._add_metrics_text(ax_scatter, metrics)

            self._apply_standard_styling(ax_scatter, xlabel='Observed', ylabel='Simulated', title=f'{model} - Scatter')

            # FDC
            ax_fdc = fig.add_subplot(gs[i, 1])
            exc_obs, f_obs = calculate_flow_duration_curve(df['Observed'].values)
            exc_sim, f_sim = calculate_flow_duration_curve(df[col].values)

            ax_fdc.plot(exc_obs, f_obs, 'k-', label='Observed')
            ax_fdc.plot(exc_sim, f_sim, color=color, label=model)

            ax_fdc.set_xscale('log')
            ax_fdc.set_yscale('log')
            self._apply_standard_styling(ax_fdc, xlabel='Exceedance', ylabel='Discharge', title=f'{model} - FDC', legend=True)

        plt.tight_layout()
        return self._save_and_close(fig, plot_file)

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        if 'experiment_id' in kwargs:
            results = self.plot_summa_outputs(kwargs['experiment_id'])
            return str(list(results.values())[0]) if results else None
        return None
