#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
NextGen (ngen) Calibration Targets

Provides calibration target classes for ngen model outputs.
Currently supports streamflow calibration with plans for snow, ET, etc.

Note: This module has been refactored to use the centralized evaluators in
symfluence.core.modeling.evaluators.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.logging_utils import log_once
from symfluence.core.modeling.evaluators import StreamflowEvaluator
from symfluence.core.path_resolver import find_basin_shapefile
from symfluence.core.registries import R


@R.calibration_targets.add('NGEN_STREAMFLOW')
class NgenStreamflowTarget(StreamflowEvaluator):
    """NextGen-specific streamflow evaluator that handles nexus-style outputs."""

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        self.station_id = self._get_config_value(lambda: None, default=None, dict_key='STATION_ID')
        # Cache for nexus areas to avoid repeated file reads during calibration
        self._nexus_areas_cache: Optional[Dict[str, float]] = None
        # Get configurable timestep (default 3600s = 1 hour)
        self._timestep_seconds = self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            default=3600,
            dict_key='FORCING_TIME_STEP_SIZE'
        )
        # Get warm-up period in days (default 30 days for model spin-up)
        self._warmup_days = self._get_config_value(
            lambda: None,
            default=30,
            dict_key='CALIBRATION_WARMUP_DAYS'
        )

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """
        Find ngen output files for calibration.

        Priority order:
        1. T-Route routed outputs (if available) - proper accumulated flow at outlet
        2. Single nexus output (if CALIBRATION_NEXUS_ID specified)
        3. All nexus outputs summed (fallback - approximates basin total)
        """
        # Detect lumped domain (single nexus) to avoid misleading routing warnings
        is_lumped = False
        try:
            import json
            nexus_file = self.project_dir / 'settings' / 'NGEN' / 'nexus.geojson'
            if nexus_file.exists():
                nexus_data = json.loads(nexus_file.read_text(encoding='utf-8'))
                num_nexuses = len(nexus_data.get('features', []))
                if num_nexuses == 1:
                    is_lumped = True
        except Exception:  # noqa: BLE001 — calibration resilience
            is_lumped = False

        # Check for t-route outputs first (NetCDF format)
        troute_dir = sim_dir / "troute_output"
        if troute_dir.exists():
            # Look for t-route NetCDF output
            troute_nc_files = list(troute_dir.glob("*.nc")) + list(troute_dir.glob("*.parquet"))
            if troute_nc_files:
                self.logger.debug(f"Found t-route routing outputs: {len(troute_nc_files)} files")
                self.logger.debug("Using routed flow from t-route (proper accumulated upstream flow)")
                return troute_nc_files

        # Fallback to raw nexus outputs
        files = list(sim_dir.glob('nex-*_output.csv'))
        if not files:
            # Try recursive search if not found in top level
            files = list(sim_dir.glob('**/nex-*_output.csv'))

        # Filter by CALIBRATION_NEXUS_ID if configured
        target_nexus = self._get_config_value(lambda: self.config.model.ngen.calibration_nexus_id, default=None)
        if target_nexus:
            # Normalize ID (ensure it has nex- prefix if file has it)
            # Assuming config might say "nex-57" or just "57"
            target_files = [f for f in files if f.stem == f"{target_nexus}_output" or f.stem == target_nexus]

            if target_files:
                self.logger.debug(f"Using calibration nexus: {target_nexus}")
                if is_lumped:
                    self.logger.debug(
                        "Lumped domain (single nexus). Using raw nexus output at outlet."
                    )
                else:
                    self.logger.debug(
                        f"Using raw nexus output for {target_nexus} (local catchment runoff)."
                    )
                return target_files
            else:
                log_once(
                    self.logger, logging.WARNING,
                    key=f'ngen-nexus-id-missing-{target_nexus}',
                    message=(
                        f"Configured CALIBRATION_NEXUS_ID '{target_nexus}' not found in output files. "
                        f"Available: {[f.stem for f in files[:10]]}. "
                        f"Falling back to summing ALL nexus outputs"
                    ),
                )
                return files

        # If no CALIBRATION_NEXUS_ID specified, sum all nexuses
        # This is only valid for single-nexus (lumped) domains
        if len(files) > 1:
            log_once(
                self.logger, logging.WARNING,
                key='ngen-multi-nexus-sum',
                message=(
                    f"CALIBRATION_NEXUS_ID not specified for multi-nexus domain ({len(files)} nexuses). "
                    f"Summing raw nexus outputs is NOT equivalent to routed outlet flow! "
                    f"For scientifically valid calibration of distributed domains, either: "
                    f"(1) Set CALIBRATION_NEXUS_ID to the outlet nexus, or "
                    f"(2) Enable t-route routing (NGEN_RUN_TROUTE: True)"
                ),
            )
        else:
            self.logger.debug("Single nexus domain - using raw nexus output as outlet flow")
        return files

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """
        Extract streamflow from NGEN outputs.

        Handles both:
        - T-Route routed outputs (NetCDF/Parquet format)
        - Raw NGEN nexus outputs (CSV format)

        Applies warm-up period filtering if configured.
        """
        if not sim_files:
            return pd.Series(dtype=float)

        # Check if we have t-route outputs (NetCDF or Parquet)
        first_file = sim_files[0]
        if first_file.suffix in ['.nc', '.parquet']:
            flow_series = self._extract_troute_data(sim_files)
        else:
            flow_series = self._extract_nexus_data(sim_files)

        # Apply warm-up period filter to exclude model spin-up from calibration
        if self._warmup_days > 0 and not flow_series.empty:
            warmup_timesteps = int((self._warmup_days * 24 * 3600) / self._timestep_seconds)
            if len(flow_series) > warmup_timesteps:
                original_len = len(flow_series)
                flow_series = flow_series.iloc[warmup_timesteps:]
                self.logger.debug(
                    f"Applied {self._warmup_days}-day warm-up filter: "
                    f"{original_len} -> {len(flow_series)} timesteps"
                )
            else:
                log_once(
                    self.logger, logging.WARNING,
                    key='ngen-warmup-exceeds-sim',
                    message=(
                        f"Warm-up period ({self._warmup_days} days = {warmup_timesteps} timesteps) "
                        f"exceeds simulation length ({len(flow_series)} timesteps). "
                        f"No warm-up filtering applied."
                    ),
                )

        return flow_series

    def _extract_troute_data(self, troute_files: List[Path]) -> pd.Series:
        """Extract routed streamflow from t-route NetCDF/Parquet outputs."""
        import xarray as xr

        # Get target nexus ID (outlet)
        target_nexus = self._get_config_value(lambda: self.config.model.ngen.calibration_nexus_id, default=None)

        try:
            # Read t-route output (typically NetCDF with time and feature_id dimensions)
            for troute_file in troute_files:
                if troute_file.suffix == '.nc':
                    ds = xr.open_dataset(troute_file)

                    # T-Route outputs streamflow by feature_id (nexus)
                    # Look for streamflow variable (usually 'streamflow' or 'q_out')
                    flow_vars = [v for v in ds.data_vars if 'flow' in v.lower() or 'q' in v.lower()]

                    if not flow_vars:
                        log_once(self.logger, logging.WARNING, key=f'ngen-troute-no-flow-var-{troute_file}',
                                 message=f"No streamflow variable found in {troute_file}")
                        continue

                    flow_var = flow_vars[0]
                    self.logger.debug(f"Using t-route variable: {flow_var}")

                    # Extract flow at target nexus
                    if 'feature_id' in ds.dims:
                        feature_ids = ds['feature_id'].values
                        if target_nexus:
                            nexus_id_str = target_nexus.replace('nex-', '')
                            flow_data = ds[flow_var].sel(feature_id=nexus_id_str)
                        else:
                            # Auto-detect: use first feature_id if not configured
                            if len(feature_ids) > 0:
                                autodetect_id = str(feature_ids[0])
                                flow_data = ds[flow_var].sel(feature_id=feature_ids[0])
                                log_once(
                                    self.logger, logging.WARNING,
                                    key='ngen-troute-autodetect-nexus',
                                    message=(
                                        "CALIBRATION_NEXUS_ID not set. Using first feature_id from t-route output "
                                        f"({autodetect_id}). Set CALIBRATION_NEXUS_ID to avoid mis-targeting."
                                    ),
                                )
                            else:
                                log_once(self.logger, logging.WARNING, key='ngen-troute-no-feature-id',
                                          message="T-route output has no feature_id entries; using full series.")
                                flow_data = ds[flow_var]
                    else:
                        # Single location or need to select differently
                        flow_data = ds[flow_var]

                    # Convert to pandas Series
                    flow_series = flow_data.to_series()
                    flow_series.name = f'{target_nexus}_routed'

                    ds.close()
                    self.logger.debug(f"Extracted routed flow from t-route: {len(flow_series)} timesteps")
                    return flow_series.sort_index()

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading t-route outputs: {e}", exc_info=True)
            log_once(self.logger, logging.WARNING, key='ngen-troute-fallback-raw',
                                message="Falling back to raw nexus outputs")

        return pd.Series(dtype=float)

    def _get_nexus_areas(self) -> Dict[str, float]:
        """Load catchment areas mapped to nexus IDs from GeoJSON (cached)."""
        # Return cached areas if available (avoids repeated file reads during calibration)
        if self._nexus_areas_cache is not None:
            return self._nexus_areas_cache

        try:
            import json

            # Try to find catchments GeoJSON file
            ngen_settings = self.project_dir / 'settings' / 'NGEN'
            geojson_files = list(ngen_settings.glob('*catchments*.geojson'))

            if not geojson_files:
                log_once(self.logger, logging.WARNING, key='ngen-no-catchments-geojson',
                                message="No catchments GeoJSON found for area conversion")
                self._nexus_areas_cache = {}
                return {}

            geojson_path = geojson_files[0]
            self.logger.debug(f"Reading catchment areas from {geojson_path}")

            # Load GeoJSON and create nexus-area mapping
            with open(geojson_path, encoding='utf-8') as f:
                geojson_data = json.load(f)

            nexus_areas = {}  # Map of nexus_id -> area_km2
            for feature in geojson_data.get('features', []):
                props = feature.get('properties', {})
                props.get('id', '')
                toid = props.get('toid', '')  # This is the nexus ID
                area_km2 = props.get('areasqkm', 0)

                # Validate area data
                if not toid:
                    continue
                if area_km2 is None or not pd.notna(area_km2) or area_km2 <= 0:
                    self.logger.debug(f"Invalid or missing area for {toid}: {area_km2}")
                    continue

                # Normalize nexus ID (ensure it has nex- prefix)
                if not toid.startswith('nex-'):
                    toid = f'nex-{toid}'
                nexus_areas[toid] = float(area_km2)

            self.logger.debug(f"Loaded {len(nexus_areas)} catchment areas for unit conversion")
            self._nexus_areas_cache = nexus_areas
            return nexus_areas
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error loading catchment areas: {e}", exc_info=True)
            self._nexus_areas_cache = {}
            return {}

    def _extract_nexus_data(self, nexus_files: List[Path]) -> pd.Series:
        """Extract streamflow from raw NGEN nexus CSV outputs."""
        all_streamflow = []
        nexus_areas = self._get_nexus_areas()
        timestep_seconds = self._timestep_seconds  # Use configurable timestep

        for nexus_file in nexus_files:
            try:
                # ngen output format: index, datetime, flow
                # Check for headerless format (common in NGEN)
                df = pd.read_csv(nexus_file)

                # Check for standard NGEN headerless format (index, time, flow)
                is_headerless = False
                if len(df.columns) == 3:
                    try:
                        # Try parsing the FIRST row's second column as date
                        pd.to_datetime(df.columns[1])
                        is_headerless = True
                    except (ValueError, TypeError):
                        pass

                if is_headerless:
                    # Reload with header=None
                    df = pd.read_csv(nexus_file, header=None, names=['index', 'datetime', 'flow'])

                if df.empty:
                    continue

                # Standardize columns if not headerless but weird
                if 'time' in df.columns:
                    df = df.rename(columns={'time': 'datetime'})
                if 'Time' in df.columns:
                    df = df.rename(columns={'Time': 'datetime'})

                # Find flow column
                if 'flow' not in df.columns:
                    for col in ['Flow', 'Q_OUT', 'streamflow', 'discharge', 'q_cms']:
                        if col in df.columns:
                            df = df.rename(columns={col: 'flow'})
                            break

                if 'datetime' not in df.columns or 'flow' not in df.columns:
                    log_once(self.logger, logging.WARNING, key=f'ngen-nexus-columns-{nexus_file.name}',
                             message=f"Could not identify datetime/flow columns in {nexus_file}. Columns: {df.columns.tolist()}")
                    continue

                index = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None)
                flow_values = df['flow'].values

                # Convert depth (meters) to volumetric flow (m³/s)
                # flow_depth (m) * area (km²) * (1e6 m²/km²) / timestep (s) = m³/s
                nexus_id = nexus_file.stem.replace('_output', '')

                # Check config to skip conversion - this is the recommended explicit approach
                is_flow_already = self._get_config_value(lambda: self.config.model.ngen.csv_output_is_flow, default=False)

                if is_flow_already:
                    # Explicit config: trust user setting, no conversion needed
                    self.logger.debug(f"No conversion for {nexus_id} (NGEN_CSV_OUTPUT_IS_FLOW=True)")
                elif nexus_id in nexus_areas and nexus_areas[nexus_id] > 0:
                    area_m2 = nexus_areas[nexus_id] * 1e6  # km² to m²

                    # Heuristic-based unit detection - not recommended, may produce wrong results
                    # NOTE: CFE with output_variable_units="m3/s" outputs flow directly
                    # This heuristic is a fallback when NGEN_CSV_OUTPUT_IS_FLOW is not set
                    potential_flow = (flow_values * area_m2) / timestep_seconds
                    mean_raw = np.mean(flow_values)
                    mean_converted = np.mean(potential_flow)

                    # Calculate conversion factor
                    conversion_factor = mean_converted / mean_raw if mean_raw > 0 else 1

                    # Skip conversion if values suggest output is already in flow units
                    if mean_converted > 100000 or conversion_factor > 100:
                        log_once(
                            self.logger, logging.WARNING,
                            key=f'ngen-unit-heuristic-{nexus_id}',
                            message=(
                                f"Unit heuristic for {nexus_id}: output appears to already be in m³/s "
                                f"(raw mean: {mean_raw:.4f}, conversion would multiply by {conversion_factor:.1f}x). "
                                f"For explicit control, set NGEN_CSV_OUTPUT_IS_FLOW: True in config."
                            ),
                        )
                        # Don't convert
                    else:
                        flow_values = potential_flow
                        log_once(
                            self.logger, logging.INFO,
                            key=f'ngen-depth-to-flow-{nexus_id}',
                            message=(
                                f"Converted {nexus_id} from depth to flow using area {nexus_areas[nexus_id]:.2f} km². "
                                f"If output was already in m³/s, set NGEN_CSV_OUTPUT_IS_FLOW: True to disable conversion."
                            ),
                        )
                else:
                    self.logger.debug(f"No conversion for {nexus_id} (no area found), assuming m³/s")

                s = pd.Series(
                    flow_values,
                    index=index,
                    name=nexus_file.stem
                )
                all_streamflow.append(s)
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.error(f"Error reading {nexus_file}: {e}", exc_info=True)
                continue

        if not all_streamflow:
            return pd.Series(dtype=float)

        # Handle single vs multiple nexus outputs
        if len(all_streamflow) == 1:
            self.logger.debug(f"Using single nexus output: {all_streamflow[0].name}")
            return all_streamflow[0].sort_index()
        else:
            # Sum all nexus outputs for total catchment outflow
            self.logger.debug(f"Summing {len(all_streamflow)} nexus outputs for basin total")
            combined = pd.concat(all_streamflow, axis=1).sum(axis=1)
            combined.name = 'basin_total'
            return combined.sort_index()

    def calculate_metrics(self, sim: Optional[Any] = None, obs: Optional[pd.Series] = None,
                         mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True, **kwargs) -> Optional[Dict[str, float]]:
        """
        Standardized metrics calculation for NextGen.

        Args:
            sim: Path to simulation directory or pre-loaded pd.Series.
            obs: Optional pre-loaded pd.Series of observations.
            mizuroute_dir: Optional mizuRoute directory (unused for NGEN).
            calibration_only: Whether to calculate only calibration metrics.
        """
        experiment_id = kwargs.get('experiment_id')
        output_dir = kwargs.get('output_dir')

        if sim is None:
            # Determine simulation directory
            if output_dir is not None:
                sim = Path(output_dir)
            else:
                exp_id = experiment_id or self._get_config_value(lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID')
                sim = self.project_dir / 'simulations' / exp_id / 'NGEN'

        # Use base class method with our specialized data extraction
        return super().calculate_metrics(
            sim=sim,
            obs=obs,
            mizuroute_dir=mizuroute_dir,
            calibration_only=calibration_only
        )

    def _get_catchment_area(self) -> float:
        """Detailed catchment area calculation for NextGen."""
        import geopandas as gpd

        cfg_area = self._get_config_value(lambda: None, default=None, dict_key='CATCHMENT_AREA_KM2')
        if cfg_area:
            return float(cfg_area)

        # Shared finder handles the nested layout and GRUs/GRUS casing drift
        # (the old glob "*HRUs_GRUs.shp" was case-sensitive and non-recursive).
        shp_path = find_basin_shapefile(
            self.project_dir / "shapefiles",
            self.domain_name,
            self.domain_definition_method,
            self.experiment_id,
            include_river_basins=False,
            logger=self.logger,
        )
        if not shp_path:
            return 100.0

        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None or gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:5070")

            area_km2 = float(gdf.geometry.area.sum() / 1e6)
            return area_km2
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error calculating catchment area from {shp_path}: {e}", exc_info=True)
            return 100.0
