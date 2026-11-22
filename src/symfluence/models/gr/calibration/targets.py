#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
GR Calibration Targets

Provides calibration target classes for GR4J/GR6J hydrological models.
Handles GR-specific output formats (lumped CSV, distributed NetCDF) and
unit conversions (mm/day → m³/s).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, cast

import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.core.modeling.evaluators import StreamflowEvaluator
from symfluence.core.modeling.output_file_locator import OutputFileLocator
from symfluence.core.registries import R

if TYPE_CHECKING:
    pass


@R.calibration_targets.add('GR_STREAMFLOW')
class GRStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for GR4J and GR6J conceptual rainfall-runoff models.

    GR models are lumped or semi-distributed conceptual models that output
    streamflow at daily timesteps. This target handles:
    1. Lumped mode: CSV output with daily mean discharge
    2. Distributed mode: NetCDF output with per-HRU runoff requiring aggregation

    GR Output Characteristics:
        - Lumped: Single point output (q_sim column in mm/day or m³/s)
        - Distributed: Per-HRU or per-GRU output (requires spatial aggregation)
        - Timestep: Always daily (requires daily resampling of observations)
        - Units: Typically mm/day for runoff, converted to m³/s for comparison

    Unit Conversions:
        GR outputs in mm/day (depth). Conversion to m³/s:
        - mm/day → m/day (divide by 1000)
        - m/day → m³/day (multiply by catchment area in m²)
        - m³/day → m³/s (divide by 86400)
        - Or use constant: MM_DAY_TO_CMS from UnitConversion

    Output Format Detection:
        - .csv files: Lumped GR output (single location)
        - .nc files: Check for mizuRoute routing (IRFroutedRunoff) or GR distributed (q_routed)
    """

    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed streamflow and resample to daily frequency.

        GR models output daily mean discharge, so observations must be aggregated
        to daily timestep for proper comparison.

        Returns:
            Optional[pd.Series]: Daily aggregated streamflow (m³/s) or None if load fails
        """
        try:
            obs_path = self.get_observed_data_path()
            obs_series = self._load_observed_data_from_path(obs_path)

            if obs_series is not None:
                # Resample to daily frequency (mean) to match GR output frequency
                obs_daily = obs_series.resample('D').mean()
                self.logger.debug(f"Resampled observations from {len(obs_series)} to {len(obs_daily)} daily values")
                return obs_daily

            return obs_series

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error loading observed data: {str(e)}", exc_info=True)
            return None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate GR model output files (CSV for lumped, NetCDF for distributed).

        Args:
            sim_dir: Directory containing GR simulation outputs

        Returns:
            List[Path]: Path(s) to GR output file(s)
        """
        locator = OutputFileLocator(self.logger)
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default=None
        )
        return locator.find_gr_output(sim_dir, self.domain_name, experiment_id)

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract GR streamflow from lumped (CSV) or distributed (NetCDF) output.

        Format Detection and Extraction:
        - .csv: Lumped GR output → calls _extract_lumped_gr_streamflow()
        - .nc with mizuRoute vars: Routed output → calls parent _extract_mizuroute_streamflow()
        - .nc with GR vars: Distributed GR output → calls _extract_distributed_gr_streamflow()

        Args:
            sim_files: List of simulation output file(s)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Daily streamflow (m³/s)
        """
        sim_file = sim_files[0]
        self.logger.debug(f"Extracting simulated streamflow from: {sim_file}")

        if sim_file.suffix == '.csv':
            return self._extract_lumped_gr_streamflow(sim_file)
        else:
            # Check if it's mizuRoute output or GR distributed output
            if self._is_mizuroute_output(sim_file):
                sim_data = self._extract_mizuroute_streamflow(sim_file)

                # Check units and convert if needed (mm/day -> cms)
                try:
                    with xr.open_dataset(sim_file) as ds:
                        streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
                        var_name = next((v for v in streamflow_vars if v in ds.variables), None)

                        if var_name:
                            units = ds[var_name].attrs.get('units', '').lower()
                            if 'm3' in units or 'cms' in units:
                                return sim_data
                            else:
                                self.logger.info(f"Converting mizuRoute output units ({units}) to cms")
                                area_m2 = self._get_catchment_area()
                                area_km2 = area_m2 / 1e6
                                return sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
                except Exception as e:  # noqa: BLE001 — calibration resilience
                    self.logger.warning(f"Could not determine units from mizuRoute output: {e}", exc_info=True)

                return sim_data
            else:
                return self._extract_distributed_gr_streamflow(sim_file)

    def _extract_lumped_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR lumped CSV output with unit conversion.

        Args:
            sim_file: Path to GR CSV output file

        Returns:
            pd.Series: Basin-scale daily streamflow (m³/s)
        """
        df_sim = pd.read_csv(sim_file, index_col='datetime', parse_dates=True)
        # GR4J output is in mm/day. Convert to cms.
        area_m2 = self._get_catchment_area()
        area_km2 = area_m2 / 1e6
        simulated_streamflow = df_sim['q_sim'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
        return simulated_streamflow

    def _extract_distributed_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR distributed NetCDF output with aggregation.

        Args:
            sim_file: Path to GR distributed NetCDF output

        Returns:
            pd.Series: Basin-scale daily streamflow (m³/s)

        Raises:
            ValueError: If configured variable not found in file
        """
        with xr.open_dataset(sim_file) as ds:
            routing_var = self._get_config_value(
                lambda: self.config.model.mizuroute.routing_var,
                dict_key='SETTINGS_MIZU_ROUTING_VAR',
            )
            if routing_var in ('default', None, ''):
                routing_var = 'q_routed'

            if routing_var in ds.variables:
                var = ds[routing_var]
                sim_data = var.mean(dim='gru').to_pandas()

                units = var.attrs.get('units', 'mm/d').lower()
                area_m2 = self._get_catchment_area()

                if 'm/s' in units or 'm s-1' in units:
                    self.logger.info("Extracting GR distributed output in m/s")
                    return sim_data * area_m2
                else:
                    self.logger.info("Extracting GR distributed output in mm/day")
                    area_km2 = area_m2 / 1e6
                    return sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
            else:
                self.logger.warning(f"Variable '{routing_var}' not found in {sim_file}. Trying fallback 'q_routed'.")
                if 'q_routed' in ds.variables:
                    var = ds['q_routed']
                    sim_data = cast(pd.Series, var.mean(dim='gru').to_pandas())
                    area_m2 = self._get_catchment_area()
                    return sim_data * area_m2 if 'm/s' in var.attrs.get('units', '').lower() else sim_data * (area_m2 / 1e6) / UnitConversion.MM_DAY_TO_CMS

                raise ValueError(f"Neither '{routing_var}' nor 'q_routed' found in {sim_file}. Available: {list(ds.variables)}")

    def _get_catchment_area(self) -> float:
        """Determine catchment area with GR-specific priority strategy.

        GR catchment area resolution priority:
        1. River basin shapefile: GRU_area column or geometry
        2. Catchment/HRU shapefile: GRU_area or geometry
        3. Fallback to parent class method

        Returns:
            float: Catchment area in m² (square meters)
        """
        # Priority 1: Try river basins shapefile
        try:
            import geopandas as gpd
            basin_dir = self.project_dir / 'shapefiles' / 'river_basins'
            if basin_dir.exists():
                basin_files = list(basin_dir.glob(f"{self.domain_name}*.shp"))
                if basin_files:
                    basin_path = basin_files[0]
                    gdf = gpd.read_file(basin_path)
                    if 'GRU_area' in gdf.columns:
                        area_m2 = gdf['GRU_area'].sum()
                        if 0 < area_m2 < 1e12:
                            self.logger.debug(f"Catchment area from river_basins GRU_area: {area_m2:.2f} m2")
                            return float(area_m2)

                    if gdf.crs and not gdf.crs.is_geographic:
                        area_m2 = gdf.geometry.area.sum()
                    else:
                        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                        area_m2 = gdf_utm.geometry.area.sum()
                    self.logger.debug(f"Catchment area from river_basins geometry: {area_m2:.2f} m2")
                    return float(area_m2)
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.debug(f"Error calculating area from river_basins: {e}", exc_info=True)

        # Priority 2: Try catchment/HRU shapefile
        try:
            import geopandas as gpd
            project_dir = self.project_dir

            c_path = self._get_config_value(lambda: self.config.paths.catchment_path, dict_key='CATCHMENT_PATH')
            if c_path == 'default' or not c_path:
                catchment_path = project_dir / 'shapefiles' / 'catchment'
            else:
                catchment_path = Path(c_path)

            discretization = self._get_config_value(
                lambda: self.config.domain.discretization,
                default='elevation'
            )

            c_name = self._get_config_value(lambda: self.config.paths.catchment_name, dict_key='CATCHMENT_SHP_NAME')
            if not c_name or c_name == 'default':
                catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
            else:
                catchment_name = c_name

            catchment_file = catchment_path / catchment_name
            if catchment_file.exists():
                gdf = gpd.read_file(catchment_file)
                if 'GRU_area' in gdf.columns:
                    area_m2 = gdf['GRU_area'].sum()
                    if 0 < area_m2 < 1e12:
                        self.logger.debug(f"Catchment area from GRU_area: {area_m2:.2f} m2")
                        return float(area_m2)

                if gdf.crs and not gdf.crs.is_geographic:
                    area_m2 = gdf.geometry.area.sum()
                else:
                    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                    area_m2 = gdf_utm.geometry.area.sum()

                self.logger.debug(f"Catchment area calculated from geometry: {area_m2:.2f} m2")
                return float(area_m2)
            else:
                self.logger.debug(f"Catchment file not found: {catchment_file}")
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.debug(f"Error calculating area from shapefile: {e}", exc_info=True)

        # Fallback to base logic
        area_m2 = super()._get_catchment_area()
        self.logger.debug(f"Catchment area from base fallback: {area_m2:.2f} m2")
        return area_m2


__all__ = ['GRStreamflowTarget']
