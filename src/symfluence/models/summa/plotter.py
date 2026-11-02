# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Model Plotter

Model-specific visualization for SUMMA outputs including spatial and temporal
plots of hydrological variables.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from symfluence.core.path_resolver import find_basin_shapefile
from symfluence.core.registries import R
from symfluence.core.reporting.base_plotter import BasePlotter


@R.plotters.add('SUMMA')
class SUMMAPlotter(BasePlotter):
    """
    Plotter for SUMMA model outputs.

    Handles visualization of SUMMA simulation results including:
    - Spatial distributions of variables across HRUs
    - Time series of domain-averaged variables
    - Combined spatial-temporal visualizations
    """

    def plot_outputs(self, experiment_id: str) -> Dict[str, str]:
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
        try:
            summa_file = self.get_simulation_dir('SUMMA', experiment_id) / f"{experiment_id}_day.nc"
            if not summa_file.exists():
                self.logger.warning(f"SUMMA output file not found: {summa_file}")
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

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error in SUMMAPlotter.plot_outputs: {str(e)}", exc_info=True)

        return plot_paths

    def plot(self, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates based on provided kwargs.
        """
        if 'experiment_id' in kwargs:
            results = self.plot_outputs(kwargs['experiment_id'])
            return str(list(results.values())[0]) if results else None
        return None
