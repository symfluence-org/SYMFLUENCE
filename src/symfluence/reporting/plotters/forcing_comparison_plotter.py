# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcing comparison plotter for raw vs. remapped data visualization.

Provides side-by-side map visualization comparing raw gridded forcing data
with HRU-remapped forcing data.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from symfluence.core.constants import ConfigKeys
from symfluence.reporting.core.base_plotter import BasePlotter


class ForcingComparisonPlotter(BasePlotter):
    """
    Plotter for comparing raw forcing data with HRU-remapped forcing data.

    Creates side-by-side map visualizations showing:
    - Left panel: Raw gridded forcing (grid cells colored by value)
    - Right panel: Remapped HRU-averaged forcing (polygons colored by value)
    - Shared colorbar for comparison
    - Statistics boxes showing min/max/mean/std for each panel
    """

    # Variable-specific colormap mappings
    VARIABLE_COLORMAPS: Dict[str, str] = {
        'precipitation_flux': 'Blues',
        'precipitation': 'Blues',
        'pr': 'Blues',
        'prcp': 'Blues',
        'air_temperature': 'RdYlBu_r',
        'temperature': 'RdYlBu_r',
        'tas': 'RdYlBu_r',
        't2m': 'RdYlBu_r',
        'temp': 'RdYlBu_r',
        'lwradatm': 'YlOrRd',
        'swradatm': 'YlOrRd',
        'rlds': 'YlOrRd',
        'rsds': 'YlOrRd',
        'longwave': 'YlOrRd',
        'shortwave': 'YlOrRd',
        'specific_humidity': 'YlGnBu',
        'huss': 'YlGnBu',
        'wind_speed': 'Purples',
        'wind': 'Purples',
        'sfcwind': 'Purples',
        'surface_air_pressure': 'Greys',
        'pressure': 'Greys',
        'ps': 'Greys',
    }

    @BasePlotter._plot_safe("plot_raw_vs_remapped")
    def plot_raw_vs_remapped(
        self,
        raw_forcing_file: Path,
        remapped_forcing_file: Path,
        forcing_grid_shp: Path,
        hru_shp: Path,
        variable: str = 'precipitation_flux',
        time_index: int = 0
    ) -> Optional[str]:
        """
        Create side-by-side map comparing raw and remapped forcing data.

        Args:
            raw_forcing_file: Path to raw NetCDF forcing file
            remapped_forcing_file: Path to remapped NetCDF forcing file
            forcing_grid_shp: Path to forcing grid shapefile
            hru_shp: Path to HRU/catchment shapefile
            variable: Variable to visualize (default: 'precipitation_flux')
            time_index: Time index to visualize (default: 0)

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import geopandas as gpd
        import xarray as xr
        from matplotlib.colors import Normalize
        from matplotlib.gridspec import GridSpec

        # Check for point-scale domain (skip visualization)
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method,
            dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD
        )
        if domain_method == 'point':
            self.logger.info("Skipping raw vs remapped visualization for point-scale domain")
            return None

        # Check if shapefiles exist
        if not forcing_grid_shp.exists():
            self.logger.warning(f"Forcing grid shapefile not found: {forcing_grid_shp}")
            return None

        if not hru_shp.exists():
            self.logger.warning(f"HRU shapefile not found: {hru_shp}")
            return None

        # Setup output directory
        plot_dir = self._ensure_output_dir('agnostic_preprocessing')
        plot_filename = plot_dir / 'raw_vs_remap.png'

        # Load NetCDF data
        self.logger.debug(f"Loading raw forcing from {raw_forcing_file}")
        raw_ds = xr.open_dataset(raw_forcing_file)

        self.logger.debug(f"Loading remapped forcing from {remapped_forcing_file}")
        remapped_ds = xr.open_dataset(remapped_forcing_file)

        # Find the variable in the datasets
        raw_var, raw_data = self._extract_variable_data(raw_ds, variable, time_index)
        remap_var, remap_data = self._extract_variable_data(remapped_ds, variable, time_index)

        if raw_data is None or remap_data is None:
            self.logger.warning(f"Could not find variable {variable} in forcing files")
            raw_ds.close()
            remapped_ds.close()
            return None

        # Load shapefiles
        self.logger.debug(f"Loading forcing grid shapefile from {forcing_grid_shp}")
        forcing_gdf = gpd.read_file(forcing_grid_shp)

        self.logger.debug(f"Loading HRU shapefile from {hru_shp}")
        hru_gdf = gpd.read_file(hru_shp)

        # Reproject to Web Mercator for consistent visualization
        forcing_gdf_web = forcing_gdf.to_crs(epsg=3857)
        hru_gdf_web = hru_gdf.to_crs(epsg=3857)

        # Match raw data to forcing grid cells
        raw_gdf = self._match_raw_to_grid(raw_data, forcing_gdf_web, raw_ds)

        # Match remapped data to HRUs
        remap_gdf = self._match_remapped_to_hru(remap_data, hru_gdf_web, remapped_ds)

        if raw_gdf is None or remap_gdf is None:
            self.logger.warning("Could not match data to shapefiles")
            raw_ds.close()
            remapped_ds.close()
            return None

        # Calculate shared color scale
        raw_valid = raw_gdf['value'].dropna()
        remap_valid = remap_gdf['value'].dropna()

        vmin = min(raw_valid.min(), remap_valid.min())
        vmax = max(raw_valid.max(), remap_valid.max())

        # Get colormap
        cmap = self._get_colormap(variable)

        # Create figure with GridSpec (1x3: raw, remapped, colorbar)
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

        ax_raw = fig.add_subplot(gs[0])
        ax_remap = fig.add_subplot(gs[1])
        ax_cbar = fig.add_subplot(gs[2])

        # Create shared normalization
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot raw data (left panel)
        raw_gdf.plot(
            column='value',
            ax=ax_raw,
            cmap=cmap,
            norm=norm,
            legend=False,
            edgecolor='gray',
            linewidth=0.3
        )
        ax_raw.set_title('Raw Gridded Forcing', fontsize=self.plot_config.FONT_SIZE_TITLE, fontweight='bold')
        ax_raw.set_axis_off()

        # Plot remapped data (right panel)
        remap_gdf.plot(
            column='value',
            ax=ax_remap,
            cmap=cmap,
            norm=norm,
            legend=False,
            edgecolor='gray',
            linewidth=0.3
        )
        ax_remap.set_title('HRU-Remapped Forcing', fontsize=self.plot_config.FONT_SIZE_TITLE, fontweight='bold')
        ax_remap.set_axis_off()

        # Add shared colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax_cbar)

        # Get variable units if available
        units = self._get_variable_units(raw_ds, raw_var)
        cbar.set_label(f'{raw_var} [{units}]' if units else raw_var, fontsize=self.plot_config.FONT_SIZE_MEDIUM)

        # Add statistics boxes
        raw_stats = self._compute_stats(raw_valid)
        remap_stats = self._compute_stats(remap_valid)

        self._add_stats_box(ax_raw, raw_stats, position=(0.02, 0.02))
        self._add_stats_box(ax_remap, remap_stats, position=(0.02, 0.02))

        # Add main title
        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)
        forcing_dataset = self._get_config_value(
            lambda: self.config.forcing.dataset,
            dict_key=ConfigKeys.FORCING_DATASET
        )
        fig.suptitle(
            f'Forcing Comparison: {domain_name} - {forcing_dataset}',
            fontsize=self.plot_config.FONT_SIZE_TITLE + 2,
            fontweight='bold',
            y=0.98
        )

        # Clean up
        raw_ds.close()
        remapped_ds.close()

        # Save and close
        return self._save_and_close(fig, plot_filename)

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates to plot_raw_vs_remapped().
        """
        return self.plot_raw_vs_remapped(*args, **kwargs)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_colormap(self, variable: str) -> str:
        """
        Get appropriate colormap for the given variable.

        Args:
            variable: Variable name

        Returns:
            Colormap name
        """
        var_lower = variable.lower()
        for key, cmap in self.VARIABLE_COLORMAPS.items():
            if key in var_lower:
                return cmap
        return 'viridis'  # Default

    def _extract_variable_data(
        self,
        ds: Any,
        variable: str,
        time_index: int
    ) -> tuple:
        """
        Extract variable data from dataset at specified time index.

        Args:
            ds: xarray Dataset
            variable: Target variable name
            time_index: Time index to extract

        Returns:
            Tuple of (found_variable_name, data_array) or (None, None) if not found
        """
        # Try exact match first
        if variable in ds.data_vars:
            data = ds[variable]
            if 'time' in data.dims:
                data = data.isel(time=time_index)
            return variable, data.values

        # Try case-insensitive match
        for var in ds.data_vars:
            if var.lower() == variable.lower():
                data = ds[var]
                if 'time' in data.dims:
                    data = data.isel(time=time_index)
                return var, data.values

        # Try partial match
        for var in ds.data_vars:
            if variable.lower() in var.lower():
                data = ds[var]
                if 'time' in data.dims:
                    data = data.isel(time=time_index)
                return var, data.values

        # Auto-detect: use first non-coordinate variable
        coord_names = {'time', 'lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'hru', 'gru', 'hruId', 'gruId'}
        for var in ds.data_vars:
            if var.lower() not in [c.lower() for c in coord_names]:
                data = ds[var]
                if 'time' in data.dims:
                    data = data.isel(time=time_index)
                self.logger.info(f"Auto-detected variable: {var}")
                return var, data.values

        return None, None

    @BasePlotter._plot_safe("matching raw data to grid")
    def _match_raw_to_grid(
        self,
        raw_data: np.ndarray,
        forcing_gdf: Any,
        ds: Any
    ) -> Optional[Any]:
        """
        Match raw gridded data to forcing grid shapefile.

        Args:
            raw_data: 2D numpy array of raw data (lat, lon)
            forcing_gdf: GeoDataFrame of forcing grid
            ds: xarray Dataset (for coordinate info)

        Returns:
            GeoDataFrame with 'value' column, or None if failed
        """
        # Get coordinates from dataset
        lat_name = None
        lon_name = None
        for name in ['lat', 'latitude', 'y', 'rlat']:
            if name in ds.coords or name in ds.dims:
                lat_name = name
                break
        for name in ['lon', 'longitude', 'x', 'rlon']:
            if name in ds.coords or name in ds.dims:
                lon_name = name
                break

        if lat_name is None or lon_name is None:
            self.logger.warning("Could not find lat/lon coordinates in raw dataset")
            # Fall back: assume grid cells are in order
            if len(forcing_gdf) == raw_data.size:
                forcing_gdf = forcing_gdf.copy()
                forcing_gdf['value'] = raw_data.flatten()
                return forcing_gdf
            return None

        lats = ds[lat_name].values
        lons = ds[lon_name].values

        # Create meshgrid if 1D
        if lats.ndim == 1 and lons.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lon_grid, lat_grid = lons, lats

        # Flatten data and coordinates
        flat_data = raw_data.flatten()
        flat_lat = lat_grid.flatten()
        flat_lon = lon_grid.flatten()

        # Match to grid cells by centroid
        forcing_gdf = forcing_gdf.copy()
        centroids = forcing_gdf.geometry.centroid

        # Convert centroids back to geographic coordinates for matching
        centroids_geo = centroids.to_crs(epsg=4326)

        # Simple nearest-neighbor matching
        values = []
        for cent in centroids_geo:
            dist = np.sqrt((flat_lon - cent.x)**2 + (flat_lat - cent.y)**2)
            nearest_idx = np.argmin(dist)
            values.append(flat_data[nearest_idx])

        forcing_gdf['value'] = values
        return forcing_gdf

    @BasePlotter._plot_safe("matching remapped data to HRUs")
    def _match_remapped_to_hru(
        self,
        remap_data: np.ndarray,
        hru_gdf: Any,
        ds: Any
    ) -> Optional[Any]:
        """
        Match remapped data to HRU shapefile.

        Args:
            remap_data: 1D numpy array of remapped data (HRU dimension)
            hru_gdf: GeoDataFrame of HRUs
            ds: xarray Dataset (for HRU ID info)

        Returns:
            GeoDataFrame with 'value' column, or None if failed
        """
        hru_gdf = hru_gdf.copy()

        # Check if data length matches HRU count
        if len(remap_data.flatten()) == len(hru_gdf):
            hru_gdf['value'] = remap_data.flatten()
            return hru_gdf

        # Try to match by HRU ID
        hru_id_names = ['hru', 'hruId', 'HRU_ID', 'gruId', 'gru', 'GRU_ID', 'COMID', 'cat']
        ds_hru_ids = None
        for name in hru_id_names:
            if name in ds.dims or name in ds.coords:
                ds_hru_ids = ds[name].values if name in ds.coords else None
                break

        gdf_hru_col = None
        for name in ['HRU_ID', 'hru_id', 'hruId', 'COMID', 'cat', 'gruId', 'GRU_ID']:
            if name in hru_gdf.columns:
                gdf_hru_col = name
                break

        if ds_hru_ids is not None and gdf_hru_col is not None:
            # Match by ID
            id_to_value = dict(zip(ds_hru_ids, remap_data.flatten()))
            hru_gdf['value'] = hru_gdf[gdf_hru_col].map(id_to_value)
            return hru_gdf

        # Fallback: assume order matches
        if len(remap_data.flatten()) <= len(hru_gdf):
            hru_gdf['value'] = np.nan
            hru_gdf.iloc[:len(remap_data.flatten()), hru_gdf.columns.get_loc('value')] = remap_data.flatten()
            return hru_gdf

        self.logger.warning(f"Remapped data length ({len(remap_data.flatten())}) doesn't match HRU count ({len(hru_gdf)})")
        return None

    def _compute_stats(self, data: Any) -> Dict[str, float]:
        """
        Compute statistics for data.

        Args:
            data: Pandas Series or numpy array

        Returns:
            Dictionary with min, max, mean, std
        """
        return {
            'Min': float(np.nanmin(data)),
            'Max': float(np.nanmax(data)),
            'Mean': float(np.nanmean(data)),
            'Std': float(np.nanstd(data))
        }

    def _add_stats_box(
        self,
        ax: Any,
        stats: Dict[str, float],
        position: tuple = (0.02, 0.02)
    ) -> None:
        """
        Add statistics box to plot.

        Args:
            ax: Matplotlib axis
            stats: Dictionary of statistics
            position: (x, y) position in axis coordinates
        """
        plt, _ = self._setup_matplotlib()

        stats_text = "Statistics:\n"
        stats_text += f"Min: {stats['Min']:.4g}\n"
        stats_text += f"Max: {stats['Max']:.4g}\n"
        stats_text += f"Mean: {stats['Mean']:.4g}\n"
        stats_text += f"Std: {stats['Std']:.4g}"

        ax.text(
            position[0], position[1],
            stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            fontsize=self.plot_config.FONT_SIZE_SMALL,
            bbox=dict(
                facecolor='white',
                alpha=0.85,
                edgecolor='gray',
                boxstyle='round,pad=0.3'
            )
        )

    def _get_variable_units(self, ds: Any, variable: str) -> Optional[str]:
        """
        Get units for a variable from the dataset.

        Args:
            ds: xarray Dataset
            variable: Variable name

        Returns:
            Units string or None
        """
        if variable in ds.data_vars:
            attrs = ds[variable].attrs
            for attr in ['units', 'unit', 'Units', 'UNITS']:
                if attr in attrs:
                    return attrs[attr]
        return None
