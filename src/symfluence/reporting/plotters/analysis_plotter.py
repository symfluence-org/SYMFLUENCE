# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Analysis visualization plotter.

Handles plotting of sensitivity analysis, decision impacts, and threshold analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.constants import ConfigKeys, UnitConversion, UnitConverter
from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.panels import (
    FDCPanel,
    ScatterPanel,
    TimeSeriesPanel,
)


class AnalysisPlotter(BasePlotter):
    """
    Plotter for analysis visualizations.

    Handles:
    - Sensitivity analysis results
    - Decision impact analysis
    - Hydrograph comparisons with highlighting
    - Drop/threshold analysis
    """

    # -------------------------------------------------------------------------
    # Lazy-loaded panel properties
    # -------------------------------------------------------------------------
    @property
    def _scatter_panel(self) -> ScatterPanel:
        """Lazy-loaded scatter panel."""
        if not hasattr(self, '__scatter_panel'):
            self.__scatter_panel = ScatterPanel(self.plot_config, self.logger)
        return self.__scatter_panel

    @property
    def _fdc_panel(self) -> FDCPanel:
        """Lazy-loaded FDC panel."""
        if not hasattr(self, '__fdc_panel'):
            self.__fdc_panel = FDCPanel(self.plot_config, self.logger)
        return self.__fdc_panel

    @property
    def _ts_panel(self) -> TimeSeriesPanel:
        """Lazy-loaded time series panel."""
        if not hasattr(self, '__ts_panel'):
            self.__ts_panel = TimeSeriesPanel(self.plot_config, self.logger)
        return self.__ts_panel

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------
    @BasePlotter._plot_safe("creating sensitivity plot")
    def plot_sensitivity_analysis(
        self,
        sensitivity_data: Any,
        output_file: Path,
        plot_type: str = 'single'
    ) -> Optional[str]:
        """
        Visualize sensitivity analysis results.

        Args:
            sensitivity_data: Data to plot (Series or DataFrame)
            output_file: Path to save the plot
            plot_type: 'single' for one method, 'comparison' for multiple

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        output_file.parent.mkdir(parents=True, exist_ok=True)

        if plot_type == 'single':
            fig, ax = plt.subplots(
                figsize=self.plot_config.FIGURE_SIZE_SMALL
            )
            sensitivity_data.plot(kind='bar', ax=ax)

            self._apply_standard_styling(
                ax,
                xlabel="Parameters",
                ylabel="Sensitivity",
                title="Parameter Sensitivity Analysis",
                legend=False
            )

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        elif plot_type == 'comparison':
            fig, ax = plt.subplots(
                figsize=self.plot_config.FIGURE_SIZE_MEDIUM_TALL
            )
            sensitivity_data.plot(kind='bar', ax=ax)

            self._apply_standard_styling(
                ax,
                xlabel="Parameters",
                ylabel="Sensitivity",
                title="Sensitivity Analysis Comparison",
                legend=True,
                legend_loc='upper left'
            )

            ax.legend(
                title="Method",
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )

            plt.tight_layout()

        else:
            self.logger.warning(f"Unknown plot_type '{plot_type}', using 'single'")
            return self.plot_sensitivity_analysis(sensitivity_data, output_file, 'single')

        return self._save_and_close(fig, output_file)

    @BasePlotter._plot_safe("creating decision impact plots")
    def plot_decision_impacts(
        self,
        results_file: Path,
        output_folder: Path
    ) -> Optional[Dict[str, str]]:
        """
        Visualize decision analysis impacts.

        Args:
            results_file: Path to the CSV results file
            output_folder: Folder to save plots

        Returns:
            Dictionary mapping metric names to plot paths, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        output_folder.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(results_file)
        metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']

        # Identify decision columns (exclude Iteration and metrics)
        decisions = [col for col in df.columns if col not in ['Iteration'] + metrics]

        plot_paths = {}

        for metric in metrics:
            if metric not in df.columns:
                continue

            fig, axes = plt.subplots(
                len(decisions), 1,
                figsize=(12, 6 * len(decisions))
            )

            # Handle single decision case
            if len(decisions) == 1:
                axes = [axes]

            for i, decision in enumerate(decisions):
                impact = df.groupby(decision)[metric].mean().sort_values(ascending=False)
                impact.plot(kind='bar', ax=axes[i])

                axes[i].set_title(f'Impact of {decision} on {metric}')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=self.plot_config.GRID_ALPHA)

            plt.tight_layout()
            output_path = output_folder / f'{metric}_decision_impacts.png'
            self._save_and_close(fig, output_path)
            plot_paths[metric] = str(output_path)

        self.logger.info("Decision impact plots saved")
        return plot_paths

    @BasePlotter._plot_safe("creating hydrograph plot")
    def plot_hydrographs_with_highlight(
        self,
        results_file: Path,
        simulation_results: Dict,
        observed_streamflow: Any,
        decision_options: Dict,
        output_folder: Path,
        metric: str = 'kge'
    ) -> Optional[str]:
        """
        Visualize hydrographs with top performers highlighted.

        Args:
            results_file: Path to results CSV
            simulation_results: Dictionary of simulation results
            observed_streamflow: Observed streamflow series
            decision_options: Dictionary of decision options
            output_folder: Output folder
            metric: Metric to use for highlighting

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        output_folder.mkdir(parents=True, exist_ok=True)

        # Read results file
        results_df = pd.read_csv(results_file)

        # Calculate threshold for top 5%
        if metric in ['mae', 'rmse']:  # Lower is better
            threshold = results_df[metric].quantile(0.05)
            top_combinations = results_df[results_df[metric] <= threshold]
        else:  # Higher is better
            threshold = results_df[metric].quantile(0.95)
            top_combinations = results_df[results_df[metric] >= threshold]

        # Find overlapping period
        start_date = observed_streamflow.index.min()
        end_date = observed_streamflow.index.max()

        for sim in simulation_results.values():
            start_date = max(start_date, sim.index.min())
            end_date = min(end_date, sim.index.max())

        # Calculate y-axis limit from top 5%
        max_top5 = 0
        for _, row in top_combinations.iterrows():
            combo = tuple(row[list(decision_options.keys())])
            if combo in simulation_results:
                sim = simulation_results[combo]
                sim_overlap = sim.loc[start_date:end_date]
                max_top5 = max(max_top5, sim_overlap.max())

        # Create plot
        fig, ax = plt.subplots(
            figsize=self.plot_config.FIGURE_SIZE_MEDIUM
        )

        ax.set_title(
            f'Hydrograph Comparison ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})\n'
            f'Top 5% combinations by {metric} metric highlighted',
            fontsize=self.plot_config.FONT_SIZE_TITLE,
            pad=20
        )

        ax.set_ylim(0, max_top5 * 1.1)

        # Plot top 5%
        for _, row in top_combinations.iterrows():
            combo = tuple(row[list(decision_options.keys())])
            if combo in simulation_results:
                sim = simulation_results[combo]
                sim_overlap = sim.loc[start_date:end_date]
                ax.plot(
                    sim_overlap.index,
                    sim_overlap.values,
                    color=self.plot_config.COLOR_SIMULATED_PRIMARY,
                    alpha=self.plot_config.ALPHA_FAINT,
                    linewidth=self.plot_config.LINE_WIDTH_THIN
                )

        # Add legend
        ax.plot(
            [], [],
            color=self.plot_config.COLOR_SIMULATED_PRIMARY,
            alpha=self.plot_config.ALPHA_FAINT,
            label=f'Top 5% by {metric}'
        )

        self._apply_standard_styling(
            ax,
            xlabel='Date',
            ylabel='Streamflow (m³/s)',
            legend=True
        )

        plt.tight_layout()

        # Save plot
        plot_file = output_folder / f'hydrograph_comparison_{metric}.png'
        saved_path = self._save_and_close(fig, plot_file)

        # Save summary CSV
        summary_file = output_folder / f'top_combinations_{metric}.csv'
        top_combinations.to_csv(summary_file, index=False)
        self.logger.info(f"Top combinations saved to: {summary_file}")

        return saved_path

    @BasePlotter._plot_safe("creating drop analysis plot")
    def plot_drop_analysis(
        self,
        drop_data: List[Dict],
        optimal_threshold: float,
        project_dir: Path
    ) -> Optional[str]:
        """
        Visualize drop analysis for stream threshold selection.

        Args:
            drop_data: List of dictionaries with threshold and drop statistics
            optimal_threshold: The selected optimal threshold
            project_dir: Project directory for saving the plot

        Returns:
            Path to saved plot, or None if failed
        """
        # Handle empty data gracefully
        if not drop_data:
            self.logger.warning("Empty drop_data provided, skipping plot")
            return None

        plt, _ = self._setup_matplotlib()

        thresholds = [d['threshold'] for d in drop_data]
        mean_drops = [d['mean_drop'] for d in drop_data]

        fig, ax = plt.subplots(
            figsize=self.plot_config.FIGURE_SIZE_SMALL
        )

        ax.loglog(
            thresholds,
            mean_drops,
            'bo-',
            linewidth=self.plot_config.LINE_WIDTH_THICK,
            markersize=self.plot_config.MARKER_SIZE_LARGE,
            label='Mean Drop'
        )

        ax.axvline(
            optimal_threshold,
            color=self.plot_config.COLOR_VALIDATION,
            linestyle='--',
            linewidth=self.plot_config.LINE_WIDTH_THICK,
            label=f'Optimal Threshold = {optimal_threshold:.0f}'
        )

        self._apply_standard_styling(
            ax,
            xlabel='Contributing Area Threshold (cells)',
            ylabel='Mean Stream Drop (m)',
            title='Drop Analysis for Stream Threshold Selection',
            legend=True
        )

        # Save plot
        plot_path = self._ensure_output_dir("drop_analysis") / "drop_analysis.png"
        return self._save_and_close(fig, plot_path)

    @BasePlotter._plot_safe("plot_streamflow_comparison")
    def plot_streamflow_comparison(
        self,
        model_outputs: List[Tuple[str, str]],
        obs_files: List[Tuple[str, str]],
        lumped: bool = False,
        spinup_percent: Optional[float] = None
    ) -> Optional[str]:
        """
        Visualize streamflow comparison between multiple models and observations.

        Args:
            model_outputs: List of tuples (model_name, file_path)
            obs_files: List of tuples (obs_name, file_path)
            lumped: Whether these are lumped watershed models
            spinup_percent: Percentage of data to skip as spinup

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import xarray as xr

        from symfluence.reporting.core.plot_utils import (
            align_timeseries,
            calculate_flow_duration_curve,
            calculate_metrics,
        )

        spinup_percent = spinup_percent if spinup_percent is not None else self.plot_config.SPINUP_PERCENT_DEFAULT

        plot_dir = self._ensure_output_dir('results')
        plot_filename = plot_dir / 'streamflow_comparison.png'

        # Load observations
        obs_data = []
        for obs_name, obs_file in obs_files:
            try:
                df = pd.read_csv(obs_file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                # Resample to hourly if needed, or daily
                df = df['discharge_cms'].resample('h').mean()
                obs_data.append((obs_name, df))
            except Exception as e:  # noqa: BLE001 — reporting resilience
                self.logger.warning(f"Could not read observation file {obs_file}: {str(e)}")

        if not obs_data:
            self.logger.error("No observation data could be loaded")
            return None

        # Load simulations
        sim_data = []
        for sim_name, sim_file in model_outputs:
            try:
                ds = xr.open_dataset(sim_file)

                if lumped:
                    if 'averageRoutedRunoff' in ds:
                        runoff = ds['averageRoutedRunoff'].to_series()
                        sim_data.append((sim_name, runoff))
                else:
                    if 'IRFroutedRunoff' in ds:
                        runoff = ds['IRFroutedRunoff'].to_series()
                        sim_data.append((sim_name, runoff))
                    elif 'averageRoutedRunoff' in ds:
                        runoff = ds['averageRoutedRunoff'].to_series()
                        sim_data.append((sim_name, runoff))
            except Exception as e:  # noqa: BLE001 — reporting resilience
                self.logger.warning(f"Could not read simulation file {sim_file}: {str(e)}")

        if not sim_data:
            self.logger.error("No simulation data could be loaded")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.plot_config.FIGURE_SIZE_XLARGE_TALL
        )

        # Plot time series
        for obs_name, obs in obs_data:
            ax1.plot(
                obs.index, obs,
                label=f'Observed ({obs_name})',
                color=self.plot_config.COLOR_OBSERVED,
                linewidth=self.plot_config.LINE_WIDTH_OBSERVED,
                zorder=5
            )

        for i, (sim_name, sim) in enumerate(sim_data):
            color = self.plot_config.get_color_from_palette(i)
            style = self.plot_config.get_line_style(i)

            # Align and calculate metrics
            aligned_obs, aligned_sim = align_timeseries(
                obs_data[0][1], sim, spinup_percent=spinup_percent
            )

            if not aligned_sim.empty:
                ax1.plot(
                    aligned_sim.index, aligned_sim,
                    label=f'Simulated ({sim_name})',
                    color=color,
                    linestyle=style,
                    linewidth=self.plot_config.LINE_WIDTH_DEFAULT
                )

                metrics = calculate_metrics(aligned_obs.values, aligned_sim.values)
                self._add_metrics_text(
                    ax1, metrics,
                    position=(0.02, 0.98 - 0.15 * i),
                    label=sim_name
                )

        self._apply_standard_styling(
            ax1,
            xlabel='Date',
            ylabel='Streamflow (m³/s)',
            title=f'Streamflow Comparison (after {spinup_percent}% spinup)',
            legend=True
        )
        self._format_date_axis(ax1)

        # Plot FDC
        for obs_name, obs in obs_data:
            exc, flows = calculate_flow_duration_curve(obs.values)
            ax2.plot(
                exc, flows,
                label=f'Observed ({obs_name})',
                color=self.plot_config.COLOR_OBSERVED,
                linewidth=self.plot_config.LINE_WIDTH_OBSERVED
            )

        for i, (sim_name, sim) in enumerate(sim_data):
            color = self.plot_config.get_color_from_palette(i)
            style = self.plot_config.get_line_style(i)
            exc, flows = calculate_flow_duration_curve(sim.values)
            ax2.plot(
                exc, flows,
                label=f'Simulated ({sim_name})',
                color=color,
                linestyle=style,
                linewidth=self.plot_config.LINE_WIDTH_DEFAULT
            )

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        self._apply_standard_styling(
            ax2,
            xlabel='Exceedance Probability',
            ylabel='Streamflow (m³/s)',
            title='Flow Duration Curve',
            legend=True
        )

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

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
                    # Get q_routed
                    sim_flow = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0).to_series()

                    # Unit conversion (mm/day to cms)
                    basin_name = self._get_config_value(lambda: self.config.paths.river_basins_name, default='default', dict_key=ConfigKeys.RIVER_BASINS_NAME)
                    if basin_name == 'default':
                        basin_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_riverBasins_delineate.shp"

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

    @BasePlotter._plot_safe("loading SWE observations")
    def _load_swe_observations(self) -> Optional[pd.Series]:
        """
        Load SWE observations for comparison with scalarSWE.

        Searches common locations for processed SWE observation files.
        Returns observations converted to mm for comparison with SUMMA output.
        """
        domain_name = self._get_config_value(
            lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME
        )

        # Search paths for SWE observations (primary: processed/, fallback: preprocessed/)
        search_paths = [
            self.project_observations_dir / "snow" / "swe" / "processed" / f"{domain_name}_swe_processed.csv",
            self.project_observations_dir / "snow" / "processed" / f"{domain_name}_swe_processed.csv",
            self.project_observations_dir / "snow" / "swe" / "preprocessed" / f"{domain_name}_swe_processed.csv",
            self.project_observations_dir / "snow" / "preprocessed" / f"{domain_name}_snow_processed.csv",
        ]

        obs_path = None
        for path in search_paths:
            if path.exists():
                obs_path = path
                break

        if obs_path is None:
            return None

        df = pd.read_csv(obs_path)

        # Find date column
        date_col = None
        for col in df.columns:
            if col.lower() in ('date', 'datetime', 'time', 'timestamp'):
                date_col = col
                break
        if date_col is None and df.columns[0]:
            date_col = df.columns[0]

        # Try multiple date formats
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except (ValueError, TypeError):
            try:
                # Try day-first format (DD/MM/YYYY)
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
            except (ValueError, TypeError):
                try:
                    # Try mixed format inference
                    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse dates in {obs_path}")
                    return None

        df = df.set_index(date_col).sort_index()

        # Find SWE column
        swe_col = None
        for col in df.columns:
            if col.lower() in ('swe', 'snw', 'snow_water_equivalent'):
                swe_col = col
                break
        if swe_col is None:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                swe_col = numeric_cols[0]

        if swe_col is None:
            return None

        obs_series = df[swe_col].astype(float)

        return UnitConverter.swe_inches_to_mm(
            obs_series,
            auto_detect=True,
            logger=self.logger,
        )

    @BasePlotter._plot_safe("loading energy flux observations")
    def _load_energy_flux_observations(self, flux_type: str = 'LE') -> Optional[pd.Series]:
        """
        Load energy flux observations for comparison.

        Args:
            flux_type: 'LE' for latent heat, 'H' for sensible heat

        Returns:
            Observations in W/m² for comparison with SUMMA output.
        """
        domain_name = self._get_config_value(
            lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME
        )

        # Search paths for FLUXNET/energy flux observations
        search_paths = [
            self.project_observations_dir / "energy_fluxes" / "processed" / f"{domain_name}_fluxnet_processed.csv",
            self.project_observations_dir / "et" / "preprocessed" / f"{domain_name}_fluxnet_et_processed.csv",
            self.project_observations_dir / "energy_fluxes" / "processed",
        ]

        obs_path = None
        for path in search_paths:
            if path.is_file() and path.exists():
                obs_path = path
                break
            elif path.is_dir() and path.exists():
                # Search for any CSV in directory
                csvs = list(path.glob("*.csv"))
                if csvs:
                    obs_path = csvs[0]
                    break

        if obs_path is None:
            return None

        df = pd.read_csv(obs_path)

        # Find timestamp column
        ts_col = None
        for col in df.columns:
            if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                ts_col = col
                break

        if ts_col is None:
            return None

        # Parse timestamp (handle FLUXNET format YYYYMMDDHHMM)
        try:
            df['datetime'] = pd.to_datetime(df[ts_col])
        except (ValueError, TypeError):
            try:
                df['datetime'] = pd.to_datetime(df[ts_col].astype(str), format='%Y%m%d%H%M')
            except (ValueError, TypeError):
                df['datetime'] = pd.to_datetime(df[ts_col].astype(str).str[:8], format='%Y%m%d')

        df = df.set_index('datetime').sort_index()

        # Find the appropriate flux column
        if flux_type == 'LE':
            col_candidates = ['LE_F_MDS', 'LE', 'LE_CORR', 'latent_heat', 'scalarLatHeatTotal']
        else:  # H
            col_candidates = ['H_F_MDS', 'H', 'H_CORR', 'sensible_heat', 'scalarSenHeatTotal']

        flux_col = None
        for col in col_candidates:
            if col in df.columns:
                flux_col = col
                break

        if flux_col is None:
            return None

        obs_series = df[flux_col].astype(float)

        # Resample to daily if higher frequency (handles hourly, 30-min, etc.)
        if len(obs_series) > 0:
            freq = pd.infer_freq(obs_series.index)
            # Check for sub-daily frequencies: H (hourly), T/min (minutes), S (seconds)
            is_sub_daily = freq and any(x in str(freq) for x in ['H', 'T', 'min', 'S', 'h'])
            # Also check by counting: if >365 entries per year, likely sub-daily
            if not is_sub_daily and len(obs_series) > 0:
                days_span = (obs_series.index.max() - obs_series.index.min()).days + 1
                if days_span > 0 and len(obs_series) / days_span > 1.5:
                    is_sub_daily = True
            if is_sub_daily:
                obs_series = obs_series.resample('D').mean()

        return obs_series

    def _get_variable_display_info(self, var_name: str) -> Dict[str, str]:
        """Get display-friendly names and units for SUMMA variables."""
        var_info = {
            'scalarSWE': {
                'title': 'Snow Water Equivalent',
                'short_name': 'SWE',
                'unit': 'mm',
                'cmap': 'Blues',
                'obs_label': 'SNOTEL Observed',
            },
            'scalarLatHeatTotal': {
                'title': 'Latent Heat Flux',
                'short_name': 'LE',
                'unit': 'W/m²',
                'cmap': 'YlOrRd',
                'obs_label': 'FLUXNET Observed',
            },
            'scalarSenHeatTotal': {
                'title': 'Sensible Heat Flux',
                'short_name': 'H',
                'unit': 'W/m²',
                'cmap': 'RdYlBu_r',
                'obs_label': 'FLUXNET Observed',
            },
            'scalarSnowDepth': {
                'title': 'Snow Depth',
                'short_name': 'Snow Depth',
                'unit': 'm',
                'cmap': 'Blues',
                'obs_label': 'Observed',
            },
            'scalarTotalRunoff': {
                'title': 'Total Runoff',
                'short_name': 'Runoff',
                'unit': 'mm/day',
                'cmap': 'Blues',
                'obs_label': 'Observed',
            },
        }
        return var_info.get(var_name, {
            'title': var_name,
            'short_name': var_name,
            'unit': '',
            'cmap': 'viridis',
            'obs_label': 'Observed',
        })

    @BasePlotter._plot_safe("plot_summa_outputs")
    def plot_summa_outputs(
        self,
        experiment_id: str,
        *,
        summa_file: Optional[Path] = None,
        output_suffix: str = "",
    ) -> Dict[str, str]:
        """
        Create professional visualizations for SUMMA output variables.

        Auto-detects and overlays observations for key variables:
        - scalarSWE: Snow water equivalent (from SNOTEL/snow observations)
        - scalarLatHeatTotal: Latent heat flux (from FLUXNET)
        - scalarSenHeatTotal: Sensible heat flux (from FLUXNET)

        When observations are available, creates a comprehensive 4-panel layout:
        - Spatial map with colorbar
        - Time series comparison with metrics
        - Scatter plot with 1:1 line and correlation
        - Monthly boxplot comparison

        Visual styling matches Camille's model comparison overview for consistency.
        """
        plt, _ = self._setup_matplotlib()
        import matplotlib.dates as mdates
        import xarray as xr
        from matplotlib import gridspec
        from matplotlib.patches import Patch

        from symfluence.reporting.core.plot_utils import calculate_metrics

        plot_paths: Dict[str, str] = {}

        # Professional color palette
        COLOR_OBS = '#2c3e50'       # Dark blue-gray for observations
        COLOR_SIM = '#e74c3c'       # Professional red for simulations
        COLOR_ACCENT = '#3498db'    # Blue accent

        # Define observation loaders for supported variables
        obs_loaders = {
            'scalarSWE': self._load_swe_observations,
            'scalarLatHeatTotal': lambda: self._load_energy_flux_observations('LE'),
            'scalarSenHeatTotal': lambda: self._load_energy_flux_observations('H'),
        }

        if summa_file is None:
            summa_file = self.project_dir / "simulations" / experiment_id / "SUMMA" / f"{experiment_id}_day.nc"
        if not summa_file.exists():
            return {}

        plot_dir = self._ensure_output_dir('summa_outputs', experiment_id)
        ds = xr.open_dataset(summa_file)

        # Get domain name for title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='Domain',
            dict_key=ConfigKeys.DOMAIN_NAME
        )

        skip_vars = {'hru', 'time', 'gru', 'dateId', 'latitude', 'longitude', 'hruId', 'gruId'}

        for var_name in ds.data_vars:
            var_name_str = str(var_name)
            if var_name_str in skip_vars or 'time' not in ds[var_name].dims:
                continue

            # Get display info for this variable
            var_info = self._get_variable_display_info(var_name_str)

            # Check if we have observations for this variable
            obs_data = None
            if var_name_str in obs_loaders:
                obs_data = obs_loaders[var_name_str]()

            # Determine figure layout based on whether we have observations
            has_obs = obs_data is not None and len(obs_data) > 0

            if has_obs:
                # Professional 4-panel layout matching Camille's style
                fig = plt.figure(figsize=(16, 12), facecolor='white')
                gs = gridspec.GridSpec(
                    3, 3,
                    height_ratios=[0.08, 1.2, 1],
                    width_ratios=[1.2, 1, 0.8],
                    hspace=0.35, wspace=0.3
                )

                # Title spanning full width
                ax_title = fig.add_subplot(gs[0, :])
                ax_title.axis('off')
                ax_title.text(0.5, 0.5,
                             f"{var_info['title']} Evaluation — {domain_name}",
                             ha='center', va='center',
                             fontsize=18, fontweight='bold',
                             transform=ax_title.transAxes)
                ax_title.text(0.5, -0.3,
                             f"Experiment: {experiment_id}",
                             ha='center', va='center',
                             fontsize=11, color='#7f8c8d',
                             transform=ax_title.transAxes)

                # Panel 1: Time series (row 1, cols 0-1)
                ax_ts = fig.add_subplot(gs[1, 0:2])

                # Panel 2: Metrics box (row 1, col 2)
                ax_metrics = fig.add_subplot(gs[1, 2])

                # Panel 3: Scatter plot (row 2, col 0)
                ax_scatter = fig.add_subplot(gs[2, 0])

                # Panel 4: Monthly boxplot (row 2, cols 1-2)
                ax_monthly = fig.add_subplot(gs[2, 1:3])

            else:
                # Simpler 2-panel layout for sim-only
                fig = plt.figure(figsize=(14, 8), facecolor='white')
                gs = gridspec.GridSpec(2, 2, height_ratios=[0.08, 1], hspace=0.25)

                ax_title = fig.add_subplot(gs[0, :])
                ax_title.axis('off')
                ax_title.text(0.5, 0.5,
                             f"{var_info['title']} — {domain_name}",
                             ha='center', va='center',
                             fontsize=16, fontweight='bold')

                ax_ts = fig.add_subplot(gs[1, :])
                ax_scatter = None
                ax_metrics = None
                ax_monthly = None

            # Extract time series data - handle multiple dimensions
            var_data = ds[var_name]

            # Get all dimensions except 'time'
            dims_to_mean = [d for d in var_data.dims if d != 'time']

            if dims_to_mean:
                mean_ts = var_data.mean(dim=dims_to_mean).compute()
            else:
                mean_ts = var_data.compute()

            # Ensure 1D array
            ts_values = mean_ts.values
            if ts_values.ndim > 1:
                ts_values = ts_values.squeeze()
            if ts_values.ndim > 1:
                # If still multi-dimensional, take mean across remaining dims
                ts_values = np.nanmean(ts_values, axis=tuple(range(1, ts_values.ndim)))

            sim_series = pd.Series(ts_values, index=pd.to_datetime(mean_ts.time.values))

            # SUMMA uses opposite sign convention for latent heat:
            # SUMMA: negative = energy leaving surface (evaporation)
            # FLUXNET: positive = energy leaving surface (evaporation)
            # Negate SUMMA LE for comparison with observations
            if var_name == 'scalarLatHeatTotal' and has_obs:
                sim_series = -sim_series

            if has_obs:
                # Align observations and simulations
                common_idx = sim_series.index.intersection(obs_data.index)

                if len(common_idx) > 10:
                    sim_aligned = sim_series.loc[common_idx].dropna()
                    obs_aligned = obs_data.loc[common_idx].dropna()

                    # Re-align after dropping NaNs
                    common_idx = sim_aligned.index.intersection(obs_aligned.index)
                    sim_aligned = sim_aligned.loc[common_idx]
                    obs_aligned = obs_aligned.loc[common_idx]

                    # ===== TIME SERIES PANEL =====
                    ax_ts.fill_between(obs_aligned.index, 0, obs_aligned.values,
                                      alpha=0.15, color=COLOR_OBS, label='_nolegend_')
                    ax_ts.plot(obs_aligned.index, obs_aligned.values,
                              color=COLOR_OBS, linewidth=1.8, label=var_info['obs_label'],
                              alpha=0.9)
                    ax_ts.plot(sim_aligned.index, sim_aligned.values,
                              color=COLOR_SIM, linewidth=1.5, label='SUMMA Simulated',
                              alpha=0.9)

                    ax_ts.set_xlabel('Date', fontsize=11, fontweight='medium')
                    ax_ts.set_ylabel(f"{var_info['short_name']} ({var_info['unit']})",
                                    fontsize=11, fontweight='medium')
                    ax_ts.set_title('Time Series Comparison', fontsize=12,
                                   fontweight='bold', pad=10)

                    ax_ts.legend(loc='upper right', frameon=True, fancybox=True,
                                shadow=False, fontsize=10, framealpha=0.95)
                    ax_ts.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax_ts.set_facecolor('#fafafa')

                    # Format date axis
                    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha='right')

                    # Set y-axis to start at 0 for SWE
                    if var_name == 'scalarSWE':
                        ax_ts.set_ylim(bottom=0)

                    # ===== METRICS PANEL =====
                    metrics = calculate_metrics(obs_aligned.values, sim_aligned.values)
                    ax_metrics.axis('off')
                    ax_metrics.set_facecolor('#f8f9fa')

                    # Create metrics display
                    metrics_display = [
                        ('KGE', metrics.get('KGE', np.nan), '#27ae60' if metrics.get('KGE', 0) > 0.5 else '#e74c3c'),
                        ('NSE', metrics.get('NSE', np.nan), '#27ae60' if metrics.get('NSE', 0) > 0.5 else '#e74c3c'),
                        ('RMSE', metrics.get('RMSE', np.nan), '#3498db'),
                        ('Bias %', metrics.get('PBIAS', np.nan), '#9b59b6'),
                        ('MAE', metrics.get('MAE', np.nan), '#3498db'),
                    ]

                    ax_metrics.text(0.5, 0.95, 'Performance Metrics',
                                   ha='center', va='top', fontsize=13, fontweight='bold',
                                   transform=ax_metrics.transAxes)

                    for i, (name, value, color) in enumerate(metrics_display):
                        y_pos = 0.78 - i * 0.15
                        if not np.isnan(value):
                            ax_metrics.text(0.15, y_pos, f'{name}:', ha='left', va='center',
                                           fontsize=11, fontweight='medium',
                                           transform=ax_metrics.transAxes)
                            ax_metrics.text(0.85, y_pos, f'{value:.3f}', ha='right', va='center',
                                           fontsize=12, fontweight='bold', color=color,
                                           transform=ax_metrics.transAxes)

                    # Add colored background box
                    from matplotlib.patches import FancyBboxPatch
                    bbox = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                         boxstyle="round,pad=0.02,rounding_size=0.02",
                                         facecolor='#f8f9fa', edgecolor='#bdc3c7',
                                         linewidth=1.5, transform=ax_metrics.transAxes,
                                         clip_on=False)
                    ax_metrics.add_patch(bbox)

                    # ===== SCATTER PLOT PANEL =====
                    ax_scatter.scatter(obs_aligned.values, sim_aligned.values,
                                      alpha=0.5, s=25, c=COLOR_ACCENT, edgecolors='white',
                                      linewidth=0.3)

                    # 1:1 line
                    min_val = min(obs_aligned.min(), sim_aligned.min())
                    max_val = max(obs_aligned.max(), sim_aligned.max())
                    margin = (max_val - min_val) * 0.05
                    ax_scatter.plot([min_val - margin, max_val + margin],
                                   [min_val - margin, max_val + margin],
                                   'k--', linewidth=1.5, alpha=0.7, label='1:1 Line')

                    # Correlation
                    valid_mask = ~(np.isnan(obs_aligned.values) | np.isnan(sim_aligned.values))
                    if valid_mask.sum() > 2:
                        corr = np.corrcoef(obs_aligned.values[valid_mask],
                                          sim_aligned.values[valid_mask])[0, 1]
                        ax_scatter.text(0.05, 0.95, f'r = {corr:.3f}',
                                       transform=ax_scatter.transAxes,
                                       fontsize=12, fontweight='bold', va='top',
                                       bbox=dict(boxstyle='round,pad=0.4',
                                                facecolor='white', edgecolor='#bdc3c7',
                                                alpha=0.9))

                    ax_scatter.set_xlabel(f'Observed ({var_info["unit"]})',
                                         fontsize=11, fontweight='medium')
                    ax_scatter.set_ylabel(f'Simulated ({var_info["unit"]})',
                                         fontsize=11, fontweight='medium')
                    ax_scatter.set_title('Observed vs Simulated', fontsize=12,
                                        fontweight='bold', pad=10)
                    ax_scatter.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax_scatter.set_facecolor('#fafafa')
                    ax_scatter.set_aspect('equal', adjustable='box')
                    ax_scatter.set_xlim(min_val - margin, max_val + margin)
                    ax_scatter.set_ylim(min_val - margin, max_val + margin)

                    # ===== MONTHLY BOXPLOT PANEL =====
                    # Create monthly data
                    obs_monthly = obs_aligned.groupby(obs_aligned.index.month)
                    sim_monthly = sim_aligned.groupby(sim_aligned.index.month)

                    months = range(1, 13)
                    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
                    positions_obs = np.arange(1, 13) - 0.2
                    positions_sim = np.arange(1, 13) + 0.2

                    # Observed boxplots
                    obs_data_monthly = [obs_monthly.get_group(m).values
                                       if m in obs_monthly.groups else [] for m in months]
                    bp_obs = ax_monthly.boxplot(obs_data_monthly, positions=positions_obs,
                                               widths=0.35, patch_artist=True,
                                               showfliers=False)
                    for patch in bp_obs['boxes']:
                        patch.set_facecolor(COLOR_OBS)
                        patch.set_alpha(0.6)
                    for median in bp_obs['medians']:
                        median.set_color('white')
                        median.set_linewidth(1.5)

                    # Simulated boxplots
                    sim_data_monthly: List[Any] = [sim_monthly.get_group(m).values
                                                   if m in sim_monthly.groups else [] for m in months]
                    bp_sim = ax_monthly.boxplot(sim_data_monthly, positions=positions_sim,
                                               widths=0.35, patch_artist=True,
                                               showfliers=False)
                    for patch in bp_sim['boxes']:
                        patch.set_facecolor(COLOR_SIM)
                        patch.set_alpha(0.6)
                    for median in bp_sim['medians']:
                        median.set_color('white')
                        median.set_linewidth(1.5)

                    ax_monthly.set_xticks(range(1, 13))
                    ax_monthly.set_xticklabels(month_names)
                    ax_monthly.set_xlabel('Month', fontsize=11, fontweight='medium')
                    ax_monthly.set_ylabel(f"{var_info['short_name']} ({var_info['unit']})",
                                         fontsize=11, fontweight='medium')
                    ax_monthly.set_title('Monthly Distribution', fontsize=12,
                                        fontweight='bold', pad=10)
                    ax_monthly.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
                    ax_monthly.set_facecolor('#fafafa')

                    # Legend for boxplots
                    legend_elements = [
                        Patch(facecolor=COLOR_OBS, alpha=0.6, label='Observed'),
                        Patch(facecolor=COLOR_SIM, alpha=0.6, label='Simulated')
                    ]
                    ax_monthly.legend(handles=legend_elements, loc='upper right',
                                     frameon=True, fancybox=True, fontsize=10)

                else:
                    # Not enough overlap
                    ax_ts.plot(sim_series.index, sim_series.values,
                              color=COLOR_SIM, linewidth=1.5)
                    ax_ts.set_xlabel('Date', fontsize=11)
                    ax_ts.set_ylabel(f"{var_info['short_name']} ({var_info['unit']})", fontsize=11)
                    ax_ts.set_title(f'{var_info["title"]} Time Series', fontsize=12, fontweight='bold')
                    ax_ts.grid(True, alpha=0.3)
                    ax_ts.set_facecolor('#fafafa')

                    if ax_scatter:
                        ax_scatter.text(0.5, 0.5, 'Insufficient\ndata overlap',
                                      transform=ax_scatter.transAxes, ha='center', va='center',
                                      fontsize=12, color='#7f8c8d')
                        ax_scatter.set_facecolor('#f8f9fa')
                    if ax_metrics:
                        ax_metrics.axis('off')
                    if ax_monthly:
                        ax_monthly.axis('off')
            else:
                # Simulation-only plot
                ax_ts.plot(sim_series.index, sim_series.values,
                          color=COLOR_SIM, linewidth=1.5, label='SUMMA Simulated')
                ax_ts.fill_between(sim_series.index, 0, sim_series.values,
                                  alpha=0.15, color=COLOR_SIM)
                ax_ts.set_xlabel('Date', fontsize=11, fontweight='medium')
                ax_ts.set_ylabel(f"{var_info['short_name']} ({var_info['unit']})",
                                fontsize=11, fontweight='medium')
                ax_ts.set_title(f'{var_info["title"]} Time Series', fontsize=12,
                               fontweight='bold', pad=10)
                ax_ts.legend(loc='upper right', frameon=True, fontsize=10)
                ax_ts.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax_ts.set_facecolor('#fafafa')

                # Format date axis
                ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Use subplots_adjust for GridSpec layouts (tight_layout not always compatible)
            try:
                if has_obs:
                    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)
                else:
                    plt.tight_layout()
            except (ValueError, RuntimeError):
                # Layout adjustment not critical - may fail with complex GridSpec
                pass

            suffix = f"_{output_suffix}" if output_suffix else ""
            plot_file = plot_dir / f'{var_name}{suffix}.png'
            self._save_and_close(fig, plot_file)
            plot_paths[str(var_name)] = str(plot_file)
        ds.close()
        return plot_paths

    @BasePlotter._plot_safe("plot_ngen_results")
    def plot_ngen_results(self, sim_df: pd.DataFrame, obs_df: Optional[pd.DataFrame], experiment_id: str, results_dir: Path) -> Optional[str]:
        """Visualize NGen streamflow plots."""
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
    def plot_lstm_results(self, results_df: pd.DataFrame, obs_streamflow: pd.DataFrame, obs_snow: pd.DataFrame, use_snow: bool, output_dir: Path, experiment_id: str) -> Optional[str]:
        """Visualize LSTM simulation results."""
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
    def plot_hype_results(self, sim_flow: pd.DataFrame, obs_flow: pd.DataFrame, outlet_id: str, domain_name: str, experiment_id: str, project_dir: Path) -> Optional[str]:
        """Visualize HYPE streamflow comparison."""
        plt, _ = self._setup_matplotlib()
        fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)
        ax.plot(sim_flow.index, sim_flow['HYPE_discharge_cms'], label='Simulated', color='blue')
        ax.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red')
        self._apply_standard_styling(ax, ylabel='Discharge (m³/s)', title=f'Streamflow Comparison - {domain_name}\nOutlet ID: {outlet_id}')
        self._format_date_axis(ax)
        plot_file = self._ensure_output_dir("results") / f"{experiment_id}_HYPE_comparison.png"
        return self._save_and_close(fig, plot_file)

    @BasePlotter._plot_safe("plot_timeseries_results")
    def plot_timeseries_results(self, df: pd.DataFrame, experiment_id: str, domain_name: str) -> Optional[str]:
        """Create timeseries comparison plot from consolidated results DataFrame."""
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

            # Plot with KGE in label
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
    def plot_diagnostics(self, df: pd.DataFrame, experiment_id: str, domain_name: str) -> Optional[str]:
        """Create diagnostic plots (scatter and FDC) for each model.

        Uses ScatterPanel for scatter plots. FDC uses log-log scale (different
        from ModelComparisonPlotter's log-linear FDC).
        """
        plt, _ = self._setup_matplotlib()
        from symfluence.reporting.core.plot_utils import calculate_flow_duration_curve

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

            # Scatter plot - use panel
            ax_scatter = fig.add_subplot(gs[i, 0])
            self._scatter_panel.render(ax_scatter, {
                'obs_values': df['Observed'].values,
                'sim_values': df[col].values,
                'model_name': f'{model} - Scatter',
                'color_index': i
            })

            # FDC - custom log-log implementation (different from panel's log-linear)
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
        if 'sensitivity_data' in kwargs and 'output_file' in kwargs:
            return self.plot_sensitivity_analysis(
                kwargs['sensitivity_data'],
                kwargs['output_file'],
                kwargs.get('plot_type', 'single')
            )
        elif 'drop_data' in kwargs and 'optimal_threshold' in kwargs:
            return self.plot_drop_analysis(
                kwargs['drop_data'],
                kwargs['optimal_threshold'],
                kwargs.get('project_dir', Path('.'))
            )
        return None
