# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Workflow diagnostic plotter for SYMFLUENCE.

Generates validation-focused diagnostic plots at the end of each workflow step
to help users verify data correctness for subsequent steps.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.constants import ConfigKeys
from symfluence.reporting.core.base_plotter import BasePlotter


class WorkflowDiagnosticPlotter(BasePlotter):
    """
    Plotter for workflow step validation diagnostics.

    Generates diagnostic plots focused on data validation rather than
    results presentation. Each plot helps verify that the output of
    a workflow step is suitable for subsequent steps.

    Output Location:
        {project_dir}/reporting/workflow_diagnostics/{step_name}/
    """

    def _ensure_diagnostic_dir(self, step_name: str) -> Path:
        """
        Ensure diagnostic output directory exists for a workflow step.

        Args:
            step_name: Name of the workflow step (e.g., 'domain_definition')

        Returns:
            Path to the diagnostic output directory
        """
        diagnostic_dir = self.project_dir / "reporting" / "workflow_diagnostics" / step_name
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        return diagnostic_dir

    def _get_timestamp(self) -> str:
        """Get timestamp string for unique filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Not typically called directly - use specific diagnostic methods instead.
        """
        return None

    # =========================================================================
    # Domain Definition Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("domain definition diagnostics")
    def plot_domain_definition_diagnostic(
        self,
        basin_gdf: Any,
        dem_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate diagnostic plots for domain definition step.

        Creates a multi-panel figure with:
        - Panel 1: Basin boundary with DEM coverage overlay
        - Panel 2: DEM nodata analysis
        - Panel 3: Elevation histogram

        Args:
            basin_gdf: GeoDataFrame of basin boundaries
            dem_path: Path to DEM raster file

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        output_dir = self._ensure_diagnostic_dir('domain_definition')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_domain_diagnostic.png'

        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Basin boundary
        ax1 = axes[0]
        basin_gdf.plot(ax=ax1, facecolor='lightblue', edgecolor='darkblue', linewidth=2)
        ax1.set_title('Basin Boundary', fontsize=self.plot_config.FONT_SIZE_TITLE)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')

        # Calculate and display area
        if basin_gdf.crs and basin_gdf.crs.is_projected:
            area_km2 = basin_gdf.geometry.area.sum() / 1e6
        else:
            # Approximate area for geographic CRS
            basin_projected = basin_gdf.to_crs(epsg=3857)
            area_km2 = basin_projected.geometry.area.sum() / 1e6

        ax1.text(
            0.02, 0.98,
            f'Area: {area_km2:.1f} km²\nPolygons: {len(basin_gdf)}',
            transform=ax1.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        # Panel 2 & 3: DEM analysis (if available)
        if dem_path and dem_path.exists():
            try:
                import rasterio

                with rasterio.open(dem_path) as src:
                    dem_data = src.read(1)
                    nodata = src.nodata

                    # Panel 2: NoData analysis
                    ax2 = axes[1]
                    if nodata is not None:
                        nodata_mask = dem_data == nodata
                        valid_mask = ~nodata_mask
                        nodata_pct = (nodata_mask.sum() / nodata_mask.size) * 100

                        # Create visualization
                        display_data = np.where(valid_mask, dem_data, np.nan)
                        im = ax2.imshow(display_data, cmap='terrain', aspect='auto')
                        plt.colorbar(im, ax=ax2, label='Elevation (m)')
                        ax2.set_title(f'DEM Coverage\nNoData: {nodata_pct:.1f}%',
                                    fontsize=self.plot_config.FONT_SIZE_TITLE)
                    else:
                        im = ax2.imshow(dem_data, cmap='terrain', aspect='auto')
                        plt.colorbar(im, ax=ax2, label='Elevation (m)')
                        ax2.set_title('DEM Coverage', fontsize=self.plot_config.FONT_SIZE_TITLE)

                    ax2.set_xlabel('Column')
                    ax2.set_ylabel('Row')

                    # Panel 3: Elevation histogram
                    ax3 = axes[2]
                    valid_elevations = dem_data[dem_data != nodata] if nodata else dem_data.flatten()
                    valid_elevations = valid_elevations[~np.isnan(valid_elevations)]

                    if len(valid_elevations) > 0:
                        ax3.hist(valid_elevations, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
                        ax3.axvline(np.mean(valid_elevations), color='red', linestyle='--',
                                   label=f'Mean: {np.mean(valid_elevations):.0f}m')
                        ax3.axvline(np.median(valid_elevations), color='orange', linestyle='--',
                                   label=f'Median: {np.median(valid_elevations):.0f}m')
                        ax3.set_xlabel('Elevation (m)')
                        ax3.set_ylabel('Frequency')
                        ax3.set_title('Elevation Distribution', fontsize=self.plot_config.FONT_SIZE_TITLE)
                        ax3.legend(loc='upper right', fontsize=8)

                        # Add statistics
                        stats_text = (f'Min: {np.min(valid_elevations):.0f}m\n'
                                    f'Max: {np.max(valid_elevations):.0f}m\n'
                                    f'Std: {np.std(valid_elevations):.0f}m')
                        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                                verticalalignment='top', horizontalalignment='right',
                                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
            except Exception as e:  # noqa: BLE001 — reporting resilience
                self.logger.warning(f"Could not read DEM for diagnostics: {e}")
                axes[1].text(0.5, 0.5, 'DEM not available', ha='center', va='center',
                           transform=axes[1].transAxes)
                axes[2].text(0.5, 0.5, 'DEM not available', ha='center', va='center',
                           transform=axes[2].transAxes)
        else:
            axes[1].text(0.5, 0.5, 'DEM not provided', ha='center', va='center',
                       transform=axes[1].transAxes)
            axes[1].set_title('DEM Coverage')
            axes[2].text(0.5, 0.5, 'DEM not provided', ha='center', va='center',
                       transform=axes[2].transAxes)
            axes[2].set_title('Elevation Distribution')

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Domain Definition Diagnostics: {domain_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Discretization Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("discretization diagnostics")
    def plot_discretization_diagnostic(
        self,
        hru_gdf: Any,
        method: str
    ) -> Optional[str]:
        """
        Generate diagnostic plots for discretization step.

        Creates a multi-panel figure with:
        - Panel 1: HRU spatial distribution
        - Panel 2: HRU area distribution
        - Panel 3: HRU count by class (if applicable)

        Args:
            hru_gdf: GeoDataFrame of HRU polygons
            method: Discretization method used

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        output_dir = self._ensure_diagnostic_dir('discretization')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_hru_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Calculate areas (in km²)
        if hru_gdf.crs and hru_gdf.crs.is_projected:
            areas_km2 = hru_gdf.geometry.area / 1e6
        else:
            hru_projected = hru_gdf.to_crs(epsg=3857)
            areas_km2 = hru_projected.geometry.area / 1e6

        # Panel 1: HRU spatial distribution
        ax1 = axes[0]
        hru_gdf.plot(ax=ax1, column=hru_gdf.index, cmap='tab20', edgecolor='black', linewidth=0.5)
        ax1.set_title(f'HRU Distribution ({len(hru_gdf)} HRUs)', fontsize=self.plot_config.FONT_SIZE_TITLE)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')

        # Panel 2: Area distribution
        ax2 = axes[1]
        ax2.hist(areas_km2, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax2.axvline(areas_km2.mean(), color='red', linestyle='--', label=f'Mean: {areas_km2.mean():.2f} km²')
        ax2.axvline(areas_km2.median(), color='orange', linestyle='--', label=f'Median: {areas_km2.median():.2f} km²')
        ax2.set_xlabel('HRU Area (km²)')
        ax2.set_ylabel('Count')
        ax2.set_title('HRU Area Distribution', fontsize=self.plot_config.FONT_SIZE_TITLE)
        ax2.legend(loc='upper right', fontsize=8)

        # Add warning indicators for tiny/huge HRUs
        tiny_threshold = 0.1  # km²
        huge_threshold = areas_km2.mean() * 10
        n_tiny = (areas_km2 < tiny_threshold).sum()
        n_huge = (areas_km2 > huge_threshold).sum()

        warning_text = f'Min: {areas_km2.min():.3f} km²\nMax: {areas_km2.max():.2f} km²'
        if n_tiny > 0:
            warning_text += f'\nTiny HRUs (<{tiny_threshold} km²): {n_tiny}'
        if n_huge > 0:
            warning_text += f'\nLarge HRUs (>{huge_threshold:.0f} km²): {n_huge}'

        ax2.text(0.98, 0.75, warning_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(facecolor='yellow' if (n_tiny > 0 or n_huge > 0) else 'white',
                                     alpha=0.8, edgecolor='none'))

        # Panel 3: Count by class
        ax3 = axes[2]

        # Try to find a class column
        class_columns = ['elevClass', 'landClass', 'soilClass', 'HRU_ID', 'gruId']
        class_col = None
        for col in class_columns:
            if col in hru_gdf.columns:
                class_col = col
                break

        if class_col:
            class_counts = hru_gdf[class_col].value_counts().sort_index()
            ax3.bar(range(len(class_counts)), class_counts.values, color='steelblue', edgecolor='white')
            ax3.set_xlabel(f'{class_col}')
            ax3.set_ylabel('Count')
            ax3.set_title(f'HRUs per {class_col}', fontsize=self.plot_config.FONT_SIZE_TITLE)

            # Add labels if not too many
            if len(class_counts) <= 15:
                ax3.set_xticks(range(len(class_counts)))
                ax3.set_xticklabels(class_counts.index, rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, f'Method: {method}\nNo class column found',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('HRU Classes', fontsize=self.plot_config.FONT_SIZE_TITLE)

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Discretization Diagnostics: {domain_name} ({method})',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Observation Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("observation diagnostics")
    def plot_observations_diagnostic(
        self,
        obs_df: Any,
        obs_type: str
    ) -> Optional[str]:
        """
        Generate diagnostic plots for observation processing step.

        Creates a multi-panel figure with:
        - Panel 1: Time series with gaps highlighted
        - Panel 2: Value distribution with outliers
        - Panel 3: Data availability timeline

        Args:
            obs_df: DataFrame of observations (must have datetime index or column)
            obs_type: Type of observations (e.g., 'streamflow', 'swe')

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import pandas as pd
        output_dir = self._ensure_diagnostic_dir('observations')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_{obs_type}_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Ensure datetime index
        if not isinstance(obs_df.index, pd.DatetimeIndex):
            datetime_cols = [c for c in obs_df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if datetime_cols:
                obs_df = obs_df.set_index(pd.to_datetime(obs_df[datetime_cols[0]]))

        # Find the value column
        value_cols = [c for c in obs_df.columns if c.lower() not in ['date', 'datetime', 'time']]
        value_col = value_cols[0] if value_cols else obs_df.columns[0]
        values = obs_df[value_col]

        # Panel 1: Time series with gaps
        ax1 = axes[0]
        ax1.plot(obs_df.index, values, 'b-', linewidth=0.5, alpha=0.7)
        ax1.scatter(obs_df.index[values.isna()], [values.mean()] * values.isna().sum(),
                   color='red', s=5, alpha=0.5, label='Missing')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(value_col)
        ax1.set_title(f'{obs_type.title()} Time Series', fontsize=self.plot_config.FONT_SIZE_TITLE)
        ax1.tick_params(axis='x', rotation=45)

        # Add gap statistics
        n_missing = values.isna().sum()
        n_total = len(values)
        gap_pct = (n_missing / n_total) * 100 if n_total > 0 else 0
        ax1.text(0.02, 0.98, f'Missing: {n_missing}/{n_total} ({gap_pct:.1f}%)',
                transform=ax1.transAxes, verticalalignment='top',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        # Panel 2: Value distribution with outliers
        ax2 = axes[1]
        valid_values = values.dropna()

        if len(valid_values) > 0:
            ax2.hist(valid_values, bins=50, color='steelblue', edgecolor='white', alpha=0.7)

            # Calculate and mark outliers (using IQR method)
            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            n_outliers = ((valid_values < lower_bound) | (valid_values > upper_bound)).sum()

            ax2.axvline(valid_values.mean(), color='red', linestyle='--', label=f'Mean: {valid_values.mean():.2f}')
            ax2.axvline(lower_bound, color='orange', linestyle=':', alpha=0.7)
            ax2.axvline(upper_bound, color='orange', linestyle=':', alpha=0.7, label='IQR bounds')

            stats_text = (f'Min: {valid_values.min():.2f}\n'
                         f'Max: {valid_values.max():.2f}\n'
                         f'Std: {valid_values.std():.2f}\n'
                         f'Outliers: {n_outliers}')
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        ax2.set_xlabel(value_col)
        ax2.set_ylabel('Frequency')
        ax2.set_title('Value Distribution', fontsize=self.plot_config.FONT_SIZE_TITLE)
        ax2.legend(loc='upper left', fontsize=8)

        # Panel 3: Data availability by year/month
        ax3 = axes[2]
        if isinstance(obs_df.index, pd.DatetimeIndex):
            # Create availability matrix by year and month
            obs_df_copy = obs_df.copy()
            obs_df_copy['year'] = obs_df_copy.index.year
            obs_df_copy['month'] = obs_df_copy.index.month
            obs_df_copy['available'] = ~values.isna()

            availability = obs_df_copy.groupby(['year', 'month'])['available'].mean().unstack()

            if not availability.empty:
                im = ax3.imshow(availability.values, aspect='auto', cmap='RdYlGn',
                               vmin=0, vmax=1)
                plt.colorbar(im, ax=ax3, label='Data Availability')
                ax3.set_xlabel('Month')
                ax3.set_ylabel('Year')
                ax3.set_title('Monthly Data Availability', fontsize=self.plot_config.FONT_SIZE_TITLE)

                # Set tick labels
                if len(availability.index) <= 20:
                    ax3.set_yticks(range(len(availability.index)))
                    ax3.set_yticklabels(availability.index)
                ax3.set_xticks(range(12))
                ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        else:
            ax3.text(0.5, 0.5, 'Datetime index required', ha='center', va='center',
                    transform=ax3.transAxes)

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Observation Diagnostics: {domain_name} - {obs_type}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Forcing Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("forcing raw diagnostics")
    def plot_forcing_raw_diagnostic(
        self,
        forcing_nc: Path,
        domain_shp: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate diagnostic plots for raw forcing acquisition step.

        Creates a multi-panel figure with:
        - Panel 1: Spatial coverage map
        - Panel 2: Variable completeness
        - Panel 3: Temporal extent

        Args:
            forcing_nc: Path to raw forcing NetCDF file
            domain_shp: Optional path to domain shapefile for overlay

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import xarray as xr

        output_dir = self._ensure_diagnostic_dir('forcing_raw')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_forcing_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Open the NetCDF file
        ds = xr.open_dataset(forcing_nc)

        # Panel 1: Spatial coverage (first variable, first timestep)
        ax1 = axes[0]
        data_vars = [v for v in ds.data_vars if len(ds[v].dims) >= 2]
        if data_vars:
            var_name = data_vars[0]
            data = ds[var_name]
            if 'time' in data.dims:
                data = data.isel(time=0)
            if len(data.dims) >= 2:
                img = np.asarray(data.values)
                if img.ndim == 1:
                    img = img[np.newaxis, :]
                ax1.imshow(img, cmap='viridis', aspect='auto')
                ax1.set_title(f'Spatial Coverage ({var_name})', fontsize=self.plot_config.FONT_SIZE_TITLE)
        else:
            ax1.text(0.5, 0.5, 'No spatial data found', ha='center', va='center',
                    transform=ax1.transAxes)
            ax1.set_title('Spatial Coverage')

        # Panel 2: Variable completeness
        ax2 = axes[1]
        var_names = list(ds.data_vars)
        completeness = []
        for var in var_names:
            data = ds[var].values
            valid_pct = (1 - np.isnan(data).sum() / data.size) * 100 if data.size > 0 else 0
            completeness.append(valid_pct)

        colors = ['green' if c >= 95 else 'yellow' if c >= 80 else 'red' for c in completeness]
        ax2.barh(range(len(var_names)), completeness, color=colors, edgecolor='white')
        ax2.set_yticks(range(len(var_names)))
        ax2.set_yticklabels(var_names)
        ax2.set_xlabel('Data Completeness (%)')
        ax2.set_xlim(0, 100)
        ax2.axvline(95, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(80, color='orange', linestyle='--', alpha=0.5)
        ax2.set_title('Variable Completeness', fontsize=self.plot_config.FONT_SIZE_TITLE)

        # Panel 3: Temporal extent
        ax3 = axes[2]
        if 'time' in ds.dims:
            times = ds['time'].values
            n_timesteps = len(times)

            # Create timeline bar
            ax3.barh([0], [1], color='steelblue', height=0.5)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(-0.5, 0.5)
            ax3.set_yticks([])

            info_text = (f'Start: {str(times[0])[:10]}\n'
                       f'End: {str(times[-1])[:10]}\n'
                       f'Timesteps: {n_timesteps}')
            ax3.text(0.5, 0.7, info_text, ha='center', va='bottom',
                    transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Temporal Extent', fontsize=self.plot_config.FONT_SIZE_TITLE)
        else:
            ax3.text(0.5, 0.5, 'No time dimension', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Temporal Extent')

        ds.close()

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Raw Forcing Diagnostics: {domain_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    @BasePlotter._plot_safe("forcing remapping diagnostics")
    def plot_forcing_remapped_diagnostic(
        self,
        raw_nc: Path,
        remapped_nc: Path,
        hru_shp: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate diagnostic plots for forcing remapping step.

        Creates a multi-panel figure with:
        - Panel 1: Raw vs remapped spatial comparison
        - Panel 2: Conservation check (total precipitation)
        - Panel 3: Per-HRU coverage/values

        Args:
            raw_nc: Path to raw forcing NetCDF file
            remapped_nc: Path to remapped forcing NetCDF file
            hru_shp: Optional path to HRU shapefile

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import xarray as xr

        output_dir = self._ensure_diagnostic_dir('forcing_remapped')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_remapping_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Open both datasets
        ds_raw = xr.open_dataset(raw_nc)
        ds_remap = xr.open_dataset(remapped_nc)

        # Find a common variable (preferably precipitation)
        raw_vars = list(ds_raw.data_vars)
        remap_vars = list(ds_remap.data_vars)
        common_vars = set(raw_vars) & set(remap_vars)

        # Prefer precipitation
        ppt_vars = [v for v in common_vars if 'ppt' in v.lower() or 'precip' in v.lower() or 'pr' in v.lower()]
        var_name = ppt_vars[0] if ppt_vars else (list(common_vars)[0] if common_vars else None)

        # Panel 1: Raw data snapshot
        ax1 = axes[0]
        if var_name and var_name in ds_raw:
            raw_data = ds_raw[var_name]
            if 'time' in raw_data.dims:
                raw_data = raw_data.isel(time=0)
            if len(raw_data.dims) >= 2:
                img = np.asarray(raw_data.values)
                if img.ndim == 1:
                    img = img[np.newaxis, :]
                ax1.imshow(img, cmap='Blues', aspect='auto')
            ax1.set_title(f'Raw {var_name}', fontsize=self.plot_config.FONT_SIZE_TITLE)
        else:
            ax1.text(0.5, 0.5, 'No comparable variable', ha='center', va='center',
                    transform=ax1.transAxes)
            ax1.set_title('Raw Forcing')

        # Panel 2: Conservation check
        ax2 = axes[1]
        if var_name:
            # Calculate totals
            raw_total = ds_raw[var_name].sum().values if var_name in ds_raw else 0
            remap_total = ds_remap[var_name].sum().values if var_name in ds_remap else 0

            ax2.bar(['Raw', 'Remapped'], [raw_total, remap_total],
                    color=['steelblue', 'coral'], edgecolor='white')

            # Calculate difference
            if raw_total > 0:
                diff_pct = ((remap_total - raw_total) / raw_total) * 100
                color = 'green' if abs(diff_pct) < 5 else 'orange' if abs(diff_pct) < 10 else 'red'
                ax2.text(0.5, 0.95, f'Difference: {diff_pct:+.2f}%',
                        transform=ax2.transAxes, ha='center', va='top',
                        fontsize=11, color=color, fontweight='bold')

            ax2.set_ylabel(f'Total {var_name}')
            ax2.set_title('Conservation Check', fontsize=self.plot_config.FONT_SIZE_TITLE)
        else:
            ax2.text(0.5, 0.5, 'No variable for comparison', ha='center', va='center',
                    transform=ax2.transAxes)
            ax2.set_title('Conservation Check')

        # Panel 3: Remapped data per HRU
        ax3 = axes[2]
        if var_name and var_name in ds_remap:
            remap_data = ds_remap[var_name]
            if 'time' in remap_data.dims:
                # Average over time
                remap_mean = remap_data.mean(dim='time')
            else:
                remap_mean = remap_data

            # If it's 1D (per HRU), plot as bar
            if len(remap_mean.dims) == 1:
                values = remap_mean.values
                ax3.bar(range(len(values)), values, color='steelblue', edgecolor='white')
                ax3.set_xlabel('HRU Index')
                ax3.set_ylabel(f'Mean {var_name}')
            else:
                img = np.asarray(remap_mean.values)
                if img.ndim == 1:
                    img = img[np.newaxis, :]
                ax3.imshow(img, cmap='Blues', aspect='auto')
            ax3.set_title('Remapped Per-HRU', fontsize=self.plot_config.FONT_SIZE_TITLE)
        else:
            ax3.text(0.5, 0.5, 'No remapped data', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Remapped Data')

        ds_raw.close()
        ds_remap.close()

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Forcing Remapping Diagnostics: {domain_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Model Preprocessing Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("model preprocessing diagnostics")
    def plot_model_preprocessing_diagnostic(
        self,
        input_dir: Path,
        model_name: str
    ) -> Optional[str]:
        """
        Generate diagnostic plots for model preprocessing step.

        Creates a multi-panel figure with:
        - Panel 1: File inventory (required vs present)
        - Panel 2: Variable inventory from NetCDF files
        - Panel 3: Config file validation summary

        Args:
            input_dir: Path to model input directory
            model_name: Name of the model

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        output_dir = self._ensure_diagnostic_dir('model_preprocessing')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_{model_name}_input_diagnostic.png'

        # Routing models keep their files in settings/, not forcing/
        if model_name.upper() == 'MIZUROUTE':
            settings_candidate = input_dir.parent.parent / 'settings' / 'mizuRoute'
            if settings_candidate.exists():
                input_dir = settings_candidate

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: File inventory
        ax1 = axes[0]
        if input_dir.exists():
            files = list(input_dir.glob('*'))
            file_types: Dict[str, int] = {}
            for f in files:
                ext = f.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1

            if file_types:
                labels = list(file_types.keys())
                sizes = list(file_types.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                ax1.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors)
                ax1.set_title(f'Files in {model_name}_input', fontsize=self.plot_config.FONT_SIZE_TITLE)

                # Add total count
                ax1.text(0.5, -0.1, f'Total: {len(files)} files',
                        transform=ax1.transAxes, ha='center', fontsize=10)
            else:
                ax1.text(0.5, 0.5, 'No files found', ha='center', va='center',
                        transform=ax1.transAxes)
                ax1.set_title('File Inventory')
        else:
            ax1.text(0.5, 0.5, f'Directory not found:\n{input_dir}',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('File Inventory')

        # Panel 2: Variable inventory from NetCDF files
        ax2 = axes[1]
        nc_files = list(input_dir.glob('*.nc')) if input_dir.exists() else []
        if nc_files:
            try:
                import xarray as xr
                all_vars: List[str] = []
                for nc_file in nc_files[:5]:  # Limit to first 5 files
                    try:
                        ds = xr.open_dataset(nc_file)
                        all_vars.extend(str(v) for v in ds.data_vars)
                        ds.close()
                    except (OSError, ValueError, KeyError) as e:
                        self.logger.debug(f"Could not read {nc_file.name} for variable inventory: {e}")

                if all_vars:
                    var_counts: Dict[str, int] = {}
                    for v in all_vars:
                        var_counts[v] = var_counts.get(v, 0) + 1

                    # Sort by frequency
                    sorted_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                    var_names = [v[0] for v in sorted_vars]
                    var_freq = [v[1] for v in sorted_vars]

                    ax2.barh(range(len(var_names)), var_freq, color='steelblue', edgecolor='white')
                    ax2.set_yticks(range(len(var_names)))
                    ax2.set_yticklabels(var_names)
                    ax2.set_xlabel('Files containing variable')
                    ax2.set_title('Variable Inventory', fontsize=self.plot_config.FONT_SIZE_TITLE)
                else:
                    ax2.text(0.5, 0.5, 'No variables found', ha='center', va='center',
                            transform=ax2.transAxes)
                    ax2.set_title('Variable Inventory')
            except ImportError:
                ax2.text(0.5, 0.5, 'xarray not available', ha='center', va='center',
                        transform=ax2.transAxes)
                ax2.set_title('Variable Inventory')
        else:
            ax2.text(0.5, 0.5, 'No NetCDF files', ha='center', va='center',
                    transform=ax2.transAxes)
            ax2.set_title('Variable Inventory')

        # Panel 3: Validation summary
        ax3 = axes[2]
        ax3.axis('off')

        # Create validation summary text
        validation_items = []

        # Check for common required files based on model.
        # SUMMA forcing is split into monthly files in the input_dir;
        # attributes.nc and coldState.nc live in settings/SUMMA/.
        settings_dir = input_dir.parent.parent / 'settings' / model_name.upper()
        if model_name.upper() == 'SUMMA':
            checks = [
                ('forcing (*.nc)', any(input_dir.glob('*.nc'))),
                ('attributes.nc', (settings_dir / 'attributes.nc').exists()),
                ('coldState.nc', (settings_dir / 'coldState.nc').exists()),
            ]
        elif model_name.upper() == 'FUSE':
            checks = [
                ('forcing (*.nc)', any(input_dir.glob('*.nc'))),
                ('elev_bands.nc', any(input_dir.glob('*elev_bands*'))),
            ]
        elif model_name.upper() == 'HYPE':
            checks = [
                ('Qobs.txt', (input_dir / 'Qobs.txt').exists()),
                ('Pobs.txt', (input_dir / 'Pobs.txt').exists()),
            ]
        elif model_name.upper() == 'MIZUROUTE':
            checks = [
                ('topology.nc', any(input_dir.glob('*topology*'))),
                ('control file', any(input_dir.glob('*.control'))),
            ]
        elif model_name.upper() == 'MESH':
            checks = [
                ('MESH_drainage_database.r2c', any(input_dir.glob('*drainage_database*'))),
            ]
        else:
            checks = []

        if checks:
            for label, found in checks:
                status = '  [OK]' if found else ' [MISSING]'
                validation_items.append(f'{label}: {status}')
        else:
            validation_items.append(f'No specific requirements for {model_name}')

        # Add file size info
        if input_dir.exists():
            total_size = sum(f.stat().st_size for f in input_dir.glob('**/*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            validation_items.append(f'\nTotal size: {size_mb:.1f} MB')

        validation_text = '\n'.join(validation_items)
        ax3.text(0.1, 0.9, f'Validation Summary:\n\n{validation_text}',
                transform=ax3.transAxes, verticalalignment='top',
                fontsize=10, family='monospace',
                bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
        ax3.set_title('Configuration Validation', fontsize=self.plot_config.FONT_SIZE_TITLE)

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Model Preprocessing Diagnostics: {domain_name} - {model_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Model Output Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("model output diagnostics")
    def plot_model_output_diagnostic(
        self,
        output_nc: Path,
        model_name: str
    ) -> Optional[str]:
        """
        Generate diagnostic plots for model output step.

        Creates a multi-panel figure with:
        - Panel 1: Output variable ranges
        - Panel 2: NaN/missing data heatmap
        - Panel 3: Key variable time series

        Args:
            output_nc: Path to model output NetCDF file
            model_name: Name of the model

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        import xarray as xr

        output_dir = self._ensure_diagnostic_dir('model_output')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_{model_name}_output_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Open the output file
        ds = xr.open_dataset(output_nc)
        data_vars = [str(v) for v in ds.data_vars]

        # Panel 1: Variable ranges
        ax1 = axes[0]
        if data_vars:
            var_stats = []
            for var in data_vars[:20]:  # Limit to 20 variables
                data = ds[var].values.flatten()
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    var_stats.append({
                        'name': var[:15],
                        'min': np.min(valid_data),
                        'max': np.max(valid_data),
                        'mean': np.mean(valid_data)
                    })

            if var_stats:
                names = [v['name'] for v in var_stats]
                mins = [v['min'] for v in var_stats]
                maxs = [v['max'] for v in var_stats]

                y_pos = range(len(names))
                ax1.barh(y_pos, maxs, color='lightcoral', alpha=0.7, label='Max')
                ax1.barh(y_pos, mins, color='steelblue', alpha=0.7, label='Min')
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(names)
                ax1.set_xlabel('Value')
                ax1.legend(loc='lower right', fontsize=8)
            ax1.set_title('Variable Ranges', fontsize=self.plot_config.FONT_SIZE_TITLE)
        else:
            ax1.text(0.5, 0.5, 'No variables', ha='center', va='center',
                    transform=ax1.transAxes)
            ax1.set_title('Variable Ranges')

        # Panel 2: NaN heatmap
        ax2 = axes[1]
        nan_percentages = []
        var_names_short = []
        for var in data_vars[:20]:
            data = ds[var].values
            nan_pct = (np.isnan(data).sum() / data.size) * 100 if data.size > 0 else 0
            nan_percentages.append(nan_pct)
            var_names_short.append(var[:12])

        if nan_percentages:
            colors = ['green' if p == 0 else 'yellow' if p < 5 else 'orange' if p < 20 else 'red'
                     for p in nan_percentages]
            ax2.barh(range(len(var_names_short)), nan_percentages, color=colors, edgecolor='white')
            ax2.set_yticks(range(len(var_names_short)))
            ax2.set_yticklabels(var_names_short)
            ax2.set_xlabel('NaN Percentage (%)')
            ax2.set_xlim(0, max(100, max(nan_percentages) * 1.1))
            ax2.axvline(5, color='orange', linestyle='--', alpha=0.5)
            ax2.axvline(20, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Missing Data Analysis', fontsize=self.plot_config.FONT_SIZE_TITLE)

        # Panel 3: Key variable time series
        ax3 = axes[2]
        # Try to find a key output variable (discharge, runoff, etc.)
        key_vars = ['scalarTotalRunoff', 'averageRoutedRunoff', 'discharge', 'runoff', 'q']
        plot_var = None
        for kv in key_vars:
            matches = [v for v in data_vars if kv.lower() in v.lower()]
            if matches:
                plot_var = matches[0]
                break

        if plot_var is None and data_vars:
            plot_var = data_vars[0]

        if plot_var and 'time' in ds[plot_var].dims:
            ts_data = ds[plot_var]
            # Average over non-time dimensions
            for dim in ts_data.dims:
                if dim != 'time':
                    ts_data = ts_data.mean(dim=dim)

            ax3.plot(ts_data['time'].values, ts_data.values, color='steelblue', linewidth=0.5)
            ax3.set_title(f'{plot_var} Time Series', fontsize=self.plot_config.FONT_SIZE_TITLE)
            ax3.set_xlabel('Time')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No time series available', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Output Time Series')

        ds.close()

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Model Output Diagnostics: {domain_name} - {model_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Attribute Acquisition Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("attributes diagnostics")
    def plot_attributes_diagnostic(
        self,
        dem_path: Optional[Path] = None,
        soil_path: Optional[Path] = None,
        land_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate diagnostic plots for attribute acquisition step.

        Creates a multi-panel figure with:
        - Panel 1: DEM coverage and statistics
        - Panel 2: Soil class distribution
        - Panel 3: Land class distribution

        Args:
            dem_path: Path to DEM raster file
            soil_path: Path to soil class raster file
            land_path: Path to land class raster file

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        output_dir = self._ensure_diagnostic_dir('attributes')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_attributes_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: DEM
        ax1 = axes[0]
        if dem_path and dem_path.exists():
            try:
                import rasterio

                with rasterio.open(dem_path) as src:
                    dem_data = src.read(1)
                    nodata = src.nodata

                    # Mask nodata
                    if nodata is not None:
                        dem_masked = np.where(dem_data != nodata, dem_data, np.nan)
                    else:
                        dem_masked = dem_data.astype(float)

                    im = ax1.imshow(dem_masked, cmap='terrain', aspect='auto')
                    plt.colorbar(im, ax=ax1, label='Elevation (m)')

                    # Statistics
                    valid = dem_masked[~np.isnan(dem_masked)]
                    if len(valid) > 0:
                        stats_text = (f'Min: {np.min(valid):.0f}m\n'
                                    f'Max: {np.max(valid):.0f}m\n'
                                    f'Mean: {np.mean(valid):.0f}m\n'
                                    f'Range: {np.max(valid)-np.min(valid):.0f}m')
                        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                                verticalalignment='top', fontsize=9,
                                bbox=dict(facecolor='white', alpha=0.8))

                    ax1.set_title('Digital Elevation Model', fontsize=self.plot_config.FONT_SIZE_TITLE)
            except Exception as e:  # noqa: BLE001 — reporting resilience
                ax1.text(0.5, 0.5, f'Error loading DEM:\n{str(e)[:50]}',
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('DEM')
        else:
            ax1.text(0.5, 0.5, 'DEM not available', ha='center', va='center',
                    transform=ax1.transAxes)
            ax1.set_title('Digital Elevation Model')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')

        # Panel 2: Soil classes
        ax2 = axes[1]
        if soil_path and soil_path.exists():
            try:
                import rasterio

                with rasterio.open(soil_path) as src:
                    soil_data = src.read(1)
                    nodata = src.nodata

                    if nodata is not None:
                        valid_soil = soil_data[soil_data != nodata]
                    else:
                        valid_soil = soil_data.flatten()

                    if len(valid_soil) > 0:
                        unique, counts = np.unique(valid_soil, return_counts=True)
                        # Limit to top 15 classes
                        if len(unique) > 15:
                            top_idx = np.argsort(counts)[-15:]
                            unique = unique[top_idx]
                            counts = counts[top_idx]

                        ax2.barh(range(len(unique)), counts, color='sandybrown', edgecolor='white')
                        ax2.set_yticks(range(len(unique)))
                        ax2.set_yticklabels([f'Class {int(u)}' for u in unique])
                        ax2.set_xlabel('Pixel Count')

                        # Add percentage labels
                        total = counts.sum()
                        for i, (u, c) in enumerate(zip(unique, counts)):
                            pct = (c / total) * 100
                            ax2.text(c + total*0.01, i, f'{pct:.1f}%', va='center', fontsize=8)

                    ax2.set_title('Soil Class Distribution', fontsize=self.plot_config.FONT_SIZE_TITLE)
            except Exception as e:  # noqa: BLE001 — reporting resilience
                ax2.text(0.5, 0.5, f'Error loading soil:\n{str(e)[:50]}',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Soil Classes')
        else:
            ax2.text(0.5, 0.5, 'Soil data not available', ha='center', va='center',
                    transform=ax2.transAxes)
            ax2.set_title('Soil Class Distribution')

        # Panel 3: Land classes
        ax3 = axes[2]
        if land_path and land_path.exists():
            try:
                import rasterio

                with rasterio.open(land_path) as src:
                    land_data = src.read(1)
                    nodata = src.nodata

                    if nodata is not None:
                        valid_land = land_data[land_data != nodata]
                    else:
                        valid_land = land_data.flatten()

                    if len(valid_land) > 0:
                        unique, counts = np.unique(valid_land, return_counts=True)
                        # Limit to top 15 classes
                        if len(unique) > 15:
                            top_idx = np.argsort(counts)[-15:]
                            unique = unique[top_idx]
                            counts = counts[top_idx]

                        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
                        ax3.barh(range(len(unique)), counts, color=colors, edgecolor='white')
                        ax3.set_yticks(range(len(unique)))
                        ax3.set_yticklabels([f'Class {int(u)}' for u in unique])
                        ax3.set_xlabel('Pixel Count')

                        # Add percentage labels
                        total = counts.sum()
                        for i, (u, c) in enumerate(zip(unique, counts)):
                            pct = (c / total) * 100
                            ax3.text(c + total*0.01, i, f'{pct:.1f}%', va='center', fontsize=8)

                    ax3.set_title('Land Class Distribution', fontsize=self.plot_config.FONT_SIZE_TITLE)
            except Exception as e:  # noqa: BLE001 — reporting resilience
                ax3.text(0.5, 0.5, f'Error loading land:\n{str(e)[:50]}',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Land Classes')
        else:
            ax3.text(0.5, 0.5, 'Land data not available', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Land Class Distribution')

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Attribute Acquisition Diagnostics: {domain_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)

    # =========================================================================
    # Calibration Diagnostics
    # =========================================================================

    @BasePlotter._plot_safe("calibration diagnostics")
    def plot_calibration_diagnostic(
        self,
        history: Optional[List[Dict]] = None,
        best_params: Optional[Dict[str, float]] = None,
        obs_vs_sim: Optional[Dict[str, Any]] = None,
        model_name: str = 'Unknown'
    ) -> Optional[str]:
        """
        Generate diagnostic plots for calibration step.

        Creates a multi-panel figure with:
        - Panel 1: Optimization convergence (objective function over iterations)
        - Panel 2: Parameter evolution or final values
        - Panel 3: Observed vs simulated comparison (if available)

        Args:
            history: List of optimization history dictionaries with 'iteration', 'objective', etc.
            best_params: Dictionary of best parameter values
            obs_vs_sim: Dictionary with 'observed' and 'simulated' arrays/series
            model_name: Name of the model being calibrated

        Returns:
            Path to saved diagnostic plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()
        output_dir = self._ensure_diagnostic_dir('calibration')
        timestamp = self._get_timestamp()
        plot_filename = output_dir / f'{timestamp}_{model_name}_calibration_diagnostic.png'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Convergence plot
        ax1 = axes[0]
        if history and len(history) > 0:
            # Extract iterations and objective values
            iterations: List[int] = []
            objectives = []
            for h in history:
                if isinstance(h, dict):
                    it = h.get('iteration', h.get('gen', h.get('step', len(iterations))))
                    obj = h.get('objective', h.get('best_fitness', h.get('kge', h.get('value', None))))
                    if obj is not None:
                        iterations.append(it)
                        objectives.append(obj)

            if iterations and objectives:
                ax1.plot(iterations, objectives, 'b-', linewidth=1.5, marker='o', markersize=3)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Objective Function')

                # Add best value annotation
                best_idx = np.argmax(objectives) if objectives[0] < objectives[-1] else np.argmin(objectives)
                best_val = objectives[best_idx]
                ax1.axhline(best_val, color='green', linestyle='--', alpha=0.7,
                           label=f'Best: {best_val:.4f}')
                ax1.legend(loc='best', fontsize=9)

                # Calculate improvement
                if len(objectives) > 1:
                    improvement = abs(objectives[-1] - objectives[0])
                    ax1.text(0.98, 0.02, f'Improvement: {improvement:.4f}',
                            transform=ax1.transAxes, ha='right', va='bottom',
                            fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax1.text(0.5, 0.5, 'No valid objective data', ha='center', va='center',
                        transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'No optimization history', ha='center', va='center',
                    transform=ax1.transAxes)
        ax1.set_title('Optimization Convergence', fontsize=self.plot_config.FONT_SIZE_TITLE)
        self._add_grid(ax1)

        # Panel 2: Parameter values
        ax2 = axes[1]
        if best_params and len(best_params) > 0:
            param_names = list(best_params.keys())[:15]  # Limit to 15 params
            param_values = [best_params[p] for p in param_names]

            # Normalize for visualization (0-1 scale for comparison)
            if len(param_values) > 0:
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(param_names)))
                ax2.barh(range(len(param_names)), param_values, color=colors, edgecolor='white')
                ax2.set_yticks(range(len(param_names)))
                ax2.set_yticklabels([p[:20] for p in param_names])  # Truncate long names
                ax2.set_xlabel('Parameter Value')

                # Add value labels
                for i, v in enumerate(param_values):
                    ax2.text(v, i, f' {v:.3g}', va='center', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No parameter data', ha='center', va='center',
                    transform=ax2.transAxes)
        ax2.set_title('Calibrated Parameters', fontsize=self.plot_config.FONT_SIZE_TITLE)

        # Panel 3: Observed vs Simulated
        ax3 = axes[2]
        if obs_vs_sim and 'observed' in obs_vs_sim and 'simulated' in obs_vs_sim:
            obs = np.array(obs_vs_sim['observed']).flatten()
            sim = np.array(obs_vs_sim['simulated']).flatten()

            # Remove NaN pairs
            valid_mask = ~(np.isnan(obs) | np.isnan(sim))
            obs_valid = obs[valid_mask]
            sim_valid = sim[valid_mask]

            if len(obs_valid) > 0:
                # Scatter plot
                ax3.scatter(obs_valid, sim_valid, alpha=0.5, s=10, c='steelblue')

                # 1:1 line
                min_val = min(obs_valid.min(), sim_valid.min())
                max_val = max(obs_valid.max(), sim_valid.max())
                ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='1:1 line')

                ax3.set_xlabel('Observed')
                ax3.set_ylabel('Simulated')
                ax3.legend(loc='upper left', fontsize=9)

                # Calculate metrics
                correlation = np.corrcoef(obs_valid, sim_valid)[0, 1]
                bias = np.mean(sim_valid - obs_valid)
                rmse = np.sqrt(np.mean((sim_valid - obs_valid)**2))

                metrics_text = f'r = {correlation:.3f}\nBias = {bias:.3f}\nRMSE = {rmse:.3f}'
                ax3.text(0.98, 0.02, metrics_text, transform=ax3.transAxes,
                        ha='right', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax3.text(0.5, 0.5, 'No valid data pairs', ha='center', va='center',
                        transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Obs vs Sim data\nnot available', ha='center', va='center',
                    transform=ax3.transAxes)
        ax3.set_title('Observed vs Simulated', fontsize=self.plot_config.FONT_SIZE_TITLE)
        self._add_grid(ax3)

        # Add overall title
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME
        )
        fig.suptitle(f'Calibration Diagnostics: {domain_name} - {model_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        return self._save_and_close(fig, plot_filename)
