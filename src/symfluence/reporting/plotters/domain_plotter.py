# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Domain visualization plotter.

Handles plotting of domain boundaries, discretization, and spatial features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from symfluence.core.constants import ConfigKeys
from symfluence.reporting.core.base_plotter import BasePlotter


class DomainPlotter(BasePlotter):
    """
    Plotter for domain and discretization visualizations.

    Handles:
    - Domain boundary maps with basemap
    - Discretized domain (HRU) visualizations
    - River networks and pour points
    - Elevation bands and land class distributions
    """

    @BasePlotter._plot_safe("plot_domain")
    def plot_domain(self) -> Optional[str]:
        """
        Create a map visualization of the delineated domain with optional basemap.

        Plots:
        - Catchment boundary
        - River network (if not lumped)
        - Pour point

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from matplotlib.lines import Line2D

        # Setup plot directory
        plot_dir = self._ensure_output_dir('domain')
        plot_filename = plot_dir / 'domain_map.png'

        # Load shapefiles
        catchment_gdf = self._load_catchment_shapefile()
        pour_point_gdf = self._load_pour_point_shapefile()

        # Check if we need river network
        domain_method = self._get_config_value(lambda: self.config.domain.definition_method, dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD)
        load_river_network = domain_method not in ['lumped', 'point']

        river_gdf = None
        if load_river_network:
            river_gdf = self._load_river_network_shapefile()

        # Reproject to Web Mercator for basemap
        catchment_gdf_web = catchment_gdf.to_crs(epsg=3857)
        pour_point_gdf_web = pour_point_gdf.to_crs(epsg=3857)
        river_gdf_web = river_gdf.to_crs(epsg=3857) if river_gdf is not None else None

        # Create figure
        fig, ax = plt.subplots(
            figsize=self.plot_config.FIGURE_SIZE_XLARGE
        )

        # Calculate bounds with buffer
        bounds = catchment_gdf_web.total_bounds
        buffer_x = (bounds[2] - bounds[0]) * 0.1
        buffer_y = (bounds[3] - bounds[1]) * 0.1
        plot_bounds = [
            bounds[0] - buffer_x,
            bounds[1] - buffer_y,
            bounds[2] + buffer_x,
            bounds[3] + buffer_y
        ]

        # Set map extent
        ax.set_xlim([plot_bounds[0], plot_bounds[2]])
        ax.set_ylim([plot_bounds[1], plot_bounds[3]])

        # Plot catchment boundary
        catchment_gdf_web.boundary.plot(
            ax=ax,
            linewidth=self.plot_config.LINE_WIDTH_THICK,
            color=self.plot_config.COLOR_BOUNDARY,
            label='Catchment Boundary',
            zorder=2
        )

        # Plot river network if available
        if river_gdf_web is not None:
            self._plot_river_network(ax, river_gdf_web)

        # Plot pour point
        pour_point_gdf_web.plot(
            ax=ax,
            color=self.plot_config.COLOR_POUR_POINT,
            marker='*',
            markersize=200,
            label='Pour Point',
            zorder=4
        )

        # Add north arrow (using our utility from base class)
        self._add_north_arrow_spatial(ax)

        # Add title
        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)
        plt.title(
            f'Delineated Domain: {domain_name}',
            fontsize=self.plot_config.FONT_SIZE_TITLE,
            pad=20,
            fontweight='bold'
        )

        # Create custom legend
        legend_elements = [
            Line2D([0], [0], color=self.plot_config.COLOR_BOUNDARY,
                  linewidth=2, label='Catchment Boundary'),
            Line2D([0], [0], color=self.plot_config.COLOR_POUR_POINT,
                  marker='*', label='Pour Point', markersize=15, linewidth=0)
        ]

        if river_gdf_web is not None:
            legend_elements.insert(1, Line2D([0], [0],
                color=self.plot_config.COLOR_RIVER,
                linewidth=2, label='River Network'))

        ax.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            facecolor='white',
            framealpha=0.9
        )

        # Remove axes
        ax.set_axis_off()

        # Add domain info box
        self._add_domain_info_box(ax, catchment_gdf_web)

        # Save and close
        return self._save_and_close(fig, plot_filename)

    @BasePlotter._plot_safe("plot_discretized_domain")
    def plot_discretized_domain(self, discretization_method: str) -> Optional[str]:
        """
        Create a map visualization of the discretized domain (HRUs).

        Args:
            discretization_method: Method used for discretization
                                 (e.g., 'elevation', 'landclass', 'soilclass')

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        # Setup plot directory
        plot_dir = self._ensure_output_dir('discretization')
        plot_filename = plot_dir / f'domain_discretization_{discretization_method}.png'

        # Load HRU shapefile
        hru_gdf = self._load_hru_shapefile(discretization_method)

        # Load river network and pour point
        domain_method = self._get_config_value(lambda: self.config.domain.definition_method, dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD)
        load_river_network = domain_method != 'lumped'

        river_gdf = None
        if load_river_network:
            river_gdf = self._load_river_network_shapefile()

        pour_point_gdf = self._load_pour_point_shapefile()

        # Create figure with high DPI
        fig, ax = plt.subplots(
            figsize=self.plot_config.FIGURE_SIZE_XLARGE,
            dpi=self.plot_config.DPI_DEFAULT
        )

        # Reproject to Web Mercator
        hru_gdf_web = hru_gdf.to_crs(epsg=3857)
        river_gdf_web = river_gdf.to_crs(epsg=3857) if river_gdf is not None else None
        pour_point_gdf_web = pour_point_gdf.to_crs(epsg=3857)

        # Calculate bounds with buffer
        bounds = hru_gdf_web.total_bounds
        buffer_x = (bounds[2] - bounds[0]) * 0.1
        buffer_y = (bounds[3] - bounds[1]) * 0.1
        plot_bounds = [
            bounds[0] - buffer_x,
            bounds[1] - buffer_y,
            bounds[2] + buffer_x,
            bounds[3] + buffer_y
        ]

        # Set map extent
        ax.set_xlim([plot_bounds[0], plot_bounds[2]])
        ax.set_ylim([plot_bounds[1], plot_bounds[3]])

        # Setup colormap and plot HRUs
        class_col, legend_title, legend_labels, colors = self._setup_hru_colormap(
            hru_gdf, discretization_method
        )

        unique_classes = sorted(hru_gdf[class_col].unique())
        n_classes = len(unique_classes)

        cmap = ListedColormap(colors)
        norm = plt.Normalize(
            vmin=min(unique_classes) - 0.5,
            vmax=max(unique_classes) + 0.5
        )

        hru_gdf_web.plot(
            column=class_col,
            ax=ax,
            cmap=cmap,
            norm=norm,
            alpha=self.plot_config.ALPHA_LIGHT,
            legend=False
        )

        # Create custom legend
        legend_elements = [
            Patch(
                facecolor=colors[i],
                alpha=self.plot_config.ALPHA_LIGHT,
                label=legend_labels[i]
            )
            for i in range(n_classes)
        ]

        # Handle large number of classes
        if n_classes > 20:
            # Save separate legend file
            self._save_separate_legend(
                legend_elements, legend_title, plot_filename
            )
            ax.text(
                0.98, 0.02, 'See separate legend file',
                transform=ax.transAxes,
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8)
            )
        else:
            ax.legend(
                handles=legend_elements,
                title=legend_title,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                frameon=True,
                fancybox=True,
                shadow=True
            )

        # Plot HRU boundaries
        hru_gdf_web.boundary.plot(
            ax=ax,
            linewidth=0.5,
            color=self.plot_config.COLOR_BOUNDARY,
            alpha=0.5
        )

        # Plot river network if available
        if river_gdf_web is not None:
            self._plot_river_network(ax, river_gdf_web)

        # Plot pour point
        pour_point_gdf_web.plot(
            ax=ax,
            color=self.plot_config.COLOR_POUR_POINT,
            marker='*',
            markersize=200,
            label='Pour Point',
            zorder=4
        )

        # Add north arrow
        self._add_north_arrow_spatial(ax)

        # Add title
        plt.title(
            f'Domain Discretization: {discretization_method.title()}',
            fontsize=self.plot_config.FONT_SIZE_TITLE,
            pad=20,
            fontweight='bold'
        )

        # Add info box
        self._add_discretization_info_box(ax, hru_gdf_web)

        # Remove axes
        ax.set_axis_off()

        # Save and close
        return self._save_and_close(fig, plot_filename)

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates to plot_domain() by default.
        """
        return self.plot_domain()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _load_catchment_shapefile(self) -> Any:
        """Load catchment/river basin shapefile."""
        import geopandas as gpd
        catchment_name = self._get_config_value(lambda: self.config.paths.river_basins_name, dict_key=ConfigKeys.RIVER_BASINS_NAME)
        if catchment_name == 'default':
            catchment_name = (
                f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_riverBasins_"
                f"{self._get_config_value(lambda: self.config.domain.definition_method, dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD)}.shp"
            )
        catchment_path = self._get_file_path(
            'RIVER_BASINS_PATH', 'shapefiles/river_basins', catchment_name
        )
        return gpd.read_file(catchment_path)

    def _load_river_network_shapefile(self) -> Any:
        """Load river network shapefile."""
        import geopandas as gpd
        river_name = self._get_config_value(lambda: self.config.paths.river_network_name, dict_key=ConfigKeys.RIVER_NETWORK_SHP_NAME)
        if river_name == 'default':
            domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)
            domain_def_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD
            )
            river_name = f"{domain_name}_riverNetwork_{domain_def_method}.shp"
        river_path = self._get_file_path(
            'RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network', river_name
        )
        return gpd.read_file(river_path)

    def _load_pour_point_shapefile(self) -> Any:
        """Load pour point shapefile."""
        import geopandas as gpd
        pour_point_name = self._get_config_value(lambda: self.config.paths.pour_point_name, dict_key=ConfigKeys.POUR_POINT_SHP_NAME)
        if pour_point_name == 'default':
            pour_point_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_pourPoint.shp"
        pour_point_path = self._get_file_path(
            'POUR_POINT_SHP_PATH', 'shapefiles/pour_point', pour_point_name
        )
        return gpd.read_file(pour_point_path)

    def _load_hru_shapefile(self, discretization_method: str) -> Any:
        """Load HRU shapefile for discretized domain."""
        import geopandas as gpd
        catchment_name = self._get_config_value(lambda: self.config.paths.catchment_name, dict_key=ConfigKeys.CATCHMENT_SHP_NAME)
        if catchment_name == 'default':
            catchment_name = (
                f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_HRUs_{discretization_method}.shp"
            )
        catchment_path = self._get_hru_file_path(catchment_name)
        return gpd.read_file(catchment_path)

    def _get_hru_file_path(self, file_name: str) -> Path:
        """
        Get HRU shapefile path with backward compatibility.

        The new path structure includes domain_definition_method and experiment_id:
            shapefiles/catchment/{domain_definition_method}/{experiment_id}/

        For backward compatibility:
            - If the file exists at the old path (shapefiles/catchment/),
              returns the old path
            - Otherwise returns the new organized path
        """
        # Check if explicit path is set in config
        catchment_path_config = self._get_config_value(lambda: None, default=None, dict_key='CATCHMENT_PATH')
        if catchment_path_config and catchment_path_config != 'default':
            return Path(catchment_path_config) / file_name

        # Get config values for new path structure
        domain_def_method = self._get_config_value(
            lambda: self.config.domain.definition_method,
            dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD
        )
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            dict_key=ConfigKeys.EXPERIMENT_ID
        )

        old_path = self.project_dir / 'shapefiles' / 'catchment' / file_name
        new_path = self.project_dir / 'shapefiles' / 'catchment' / domain_def_method / experiment_id / file_name

        # Check old path first for backward compatibility
        if old_path.exists():
            return old_path

        return new_path

    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Get file path from config or use default."""
        file_type_val = self._get_config_value(lambda: None, default='default', dict_key=file_type)
        if file_type_val == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(file_type_val)

    def _plot_river_network(self, ax: Any, river_gdf: Any) -> None:
        """Plot river network with variable width based on stream order."""
        if 'StreamOrde' in river_gdf.columns:
            # Variable width based on stream order
            min_order = river_gdf['StreamOrde'].min()
            max_order = river_gdf['StreamOrde'].max()
            river_gdf['line_width'] = river_gdf['StreamOrde'].apply(
                lambda x: 0.5 + 2 * (x - min_order) / (max_order - min_order)
            )

            for idx, row in river_gdf.iterrows():
                ax.plot(
                    row.geometry.xy[0],
                    row.geometry.xy[1],
                    color=self.plot_config.COLOR_RIVER,
                    linewidth=row['line_width'],
                    zorder=3,
                    alpha=0.8
                )
        else:
            # Fixed width
            river_gdf.plot(
                ax=ax,
                color=self.plot_config.COLOR_RIVER,
                linewidth=1,
                label='River Network',
                zorder=3,
                alpha=0.8
            )

    def _add_north_arrow_spatial(self, ax: Any) -> None:
        """Add north arrow to spatial plot."""
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        arrow_x = xmin + (xmax - xmin) * 0.05
        arrow_y_top = ymax - (ymax - ymin) * 0.05
        arrow_y_bottom = arrow_y_top - (ymax - ymin) * 0.05

        ax.annotate(
            'N',
            xy=(arrow_x, arrow_y_top),
            xytext=(arrow_x, arrow_y_bottom),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=5)
        )

    def _add_domain_info_box(self, ax: Any, catchment_gdf: Any) -> None:
        """Add information box with domain statistics."""
        from symfluence.core.geometry_utils import feature_areas_m2
        area_km2 = feature_areas_m2(catchment_gdf).sum() / 1e6

        info_text = "Domain Statistics:\n"
        info_text += f"Area: {area_km2:.1f} km²"

        if 'elev_mean' in catchment_gdf.columns:
            min_elev = catchment_gdf['elev_mean'].min()
            max_elev = catchment_gdf['elev_mean'].max()
            info_text += f"\nElevation Range: {min_elev:.0f} - {max_elev:.0f} m"

        ax.text(
            0.02, 0.02, info_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10),
            fontsize=10,
            verticalalignment='bottom'
        )

    def _add_discretization_info_box(self, ax: Any, hru_gdf: Any) -> None:
        """Add information box with discretization statistics."""
        n_hrus = len(hru_gdf)
        from symfluence.core.geometry_utils import feature_areas_m2
        total_area = feature_areas_m2(hru_gdf).sum() / 1e6
        mean_area = total_area / n_hrus

        info_text = "Discretization Statistics:\n"
        info_text += f"Number of HRUs: {n_hrus}\n"
        info_text += f"Total Area: {total_area:.1f} km²\n"
        info_text += f"Mean HRU Area: {mean_area:.1f} km²"

        if 'elev_mean' in hru_gdf.columns:
            min_elev = hru_gdf['elev_mean'].min()
            max_elev = hru_gdf['elev_mean'].max()
            info_text += f"\nElevation Range: {min_elev:.0f} - {max_elev:.0f} m"

        ax.text(
            0.02, 0.02, info_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10),
            fontsize=10,
            verticalalignment='bottom'
        )

    def _setup_hru_colormap(
        self,
        hru_gdf: Any,
        discretization_method: str
    ) -> tuple:
        """
        Setup colormap and labels for HRU visualization.

        Returns:
            Tuple of (class_column, legend_title, legend_labels, colors)
        """
        plt, _ = self._setup_matplotlib()

        # Define mappings
        class_mappings = {
            'elevation': {'col': 'elevClass', 'title': 'Elevation Classes', 'cm': 'terrain'},
            'soilclass': {'col': 'soilClass', 'title': 'Soil Classes', 'cm': 'Set3'},
            'landclass': {'col': 'landClass', 'title': 'Land Use Classes', 'cm': 'Set2'},
            'radiation': {'col': 'radiationClass', 'title': 'Radiation Classes', 'cm': 'YlOrRd'},
            'grus': {'col': 'GRU_ID', 'title': 'GRU Units', 'cm': 'viridis'},
            'default': {'col': 'HRU_ID', 'title': 'HRU Classes', 'cm': 'tab20'}
        }

        mapping = class_mappings.get(discretization_method.lower(), class_mappings['default'])
        class_col = mapping['col']
        legend_title = mapping['title']

        unique_classes = sorted(hru_gdf[class_col].unique())
        n_classes = len(unique_classes)

        # Create labels
        if discretization_method.lower() == 'elevation' and 'elev_mean' in hru_gdf.columns:
            # Elevation-specific labels
            elev_range = hru_gdf['elev_mean'].agg(['min', 'max'])
            min_elev = int(elev_range['min'])
            max_elev = int(elev_range['max'])
            band_size = int(self._get_config_value(lambda: self.config.domain.elevation_band_size, default=400, dict_key=ConfigKeys.ELEVATION_BAND_SIZE))

            legend_labels = []
            for cls in unique_classes:
                lower = min_elev + ((cls - 1) * band_size)
                upper = min_elev + (cls * band_size)
                if upper > max_elev:
                    upper = max_elev
                legend_labels.append(f'{lower}-{upper}m')
        else:
            legend_labels = [f'Class {i}' for i in unique_classes]

        # Get colors - use continuous colormap for large number of classes
        base_cmap = plt.get_cmap(mapping['cm'])

        # For large datasets (e.g., GRUs), always use continuous colormap
        if n_classes > 100 or discretization_method.lower() == 'grus':
            # Use continuous colormap that can handle any number of classes
            colors = [base_cmap(i / n_classes) for i in range(n_classes)]
        elif n_classes > base_cmap.N:
            # Need more colors from discrete colormaps
            additional_cmaps = ['Set3', 'Set2', 'Set1', 'Paired', 'tab20']
            all_colors = []

            all_colors.extend([base_cmap(i) for i in np.linspace(0, 1, base_cmap.N)])

            for cmap_name in additional_cmaps:
                if len(all_colors) >= n_classes:
                    break
                cmap = plt.get_cmap(cmap_name)
                all_colors.extend([cmap(i) for i in np.linspace(0, 1, cmap.N)])

            colors = all_colors[:n_classes]
        else:
            colors = [base_cmap(i) for i in np.linspace(0, 1, n_classes)]

        return class_col, legend_title, legend_labels, colors

    def _save_separate_legend(
        self,
        legend_elements: list,
        legend_title: str,
        plot_filename: Path
    ) -> None:
        """Save legend to separate file for large number of classes."""
        plt, _ = self._setup_matplotlib()

        legend_fig = plt.figure(figsize=(2, 6))
        legend_ax = legend_fig.add_axes([0, 0, 1, 1])
        legend_ax.set_axis_off()

        legend_ax.legend(
            handles=legend_elements,
            title=legend_title,
            loc='center',
            bbox_to_anchor=(0.5, 0.5),
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=1,
            mode="expand"
        )

        legend_filename = plot_filename.parent / f'{plot_filename.stem}_legend.png'
        legend_fig.savefig(legend_filename, bbox_inches='tight', dpi=300)
        plt.close(legend_fig)
        self.logger.info(f"Separate legend saved to {legend_filename}")
