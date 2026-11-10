#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Metrics Calculation for SUMMA Workers

This module contains functions for calculating calibration metrics
in worker processes, supporting multi-target optimization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.logging_utils import log_once
from symfluence.core.metrics import (
    kge as calc_kge,
)
from symfluence.core.metrics import (
    mae as calc_mae,
)
from symfluence.core.metrics import (
    nse as calc_nse,
)
from symfluence.core.metrics import (
    pbias as calc_pbias,
)
from symfluence.core.metrics import (
    rmse as calc_rmse,
)
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.modeling.utilities.periods import parse_period


def resample_to_timestep(data: pd.Series, target_timestep: str, logger) -> pd.Series:
    """
    Resample time series data to target timestep

    Args:
        data: Time series data with DatetimeIndex
        target_timestep: Target timestep ('native', 'hourly', or 'daily')
        logger: Logger instance

    Returns:
        Resampled time series
    """
    if target_timestep == 'native' or data is None or len(data) == 0:
        return data

    try:
        # Determine current timestep
        time_diff = data.index[1] - data.index[0] if len(data) > 1 else pd.Timedelta(hours=1)

        # Check if already at target timestep
        if target_timestep == 'hourly' and pd.Timedelta(minutes=45) <= time_diff <= pd.Timedelta(minutes=75):
            logger.debug("Data already at hourly timestep")
            return data
        elif target_timestep == 'daily' and pd.Timedelta(hours=20) <= time_diff <= pd.Timedelta(hours=28):
            logger.debug("Data already at daily timestep")
            return data

        # Perform resampling
        if target_timestep == 'hourly':
            if time_diff < pd.Timedelta(hours=1):
                # Upsampling: sub-hourly to hourly (mean aggregation)
                logger.debug(f"Aggregating {time_diff} data to hourly using mean")
                resampled = data.resample('h').mean()
            elif time_diff > pd.Timedelta(hours=1):
                # Downsampling: daily/coarser to hourly (interpolation)
                logger.debug(f"Interpolating {time_diff} data to hourly")
                resampled = data.resample('h').asfreq()
                resampled = resampled.interpolate(method='time', limit_direction='both')
            else:
                resampled = data

        elif target_timestep == 'daily':
            if time_diff < pd.Timedelta(days=1):
                # Upsampling: hourly/sub-daily to daily (mean aggregation)
                logger.debug(f"Aggregating {time_diff} data to daily using mean")
                resampled = data.resample('D').mean()
            elif time_diff > pd.Timedelta(days=1):
                # Downsampling: weekly/monthly to daily (interpolation)
                logger.debug(f"Interpolating {time_diff} data to daily")
                resampled = data.resample('D').asfreq()
                resampled = resampled.interpolate(method='time', limit_direction='both')
            else:
                resampled = data
        else:
            resampled = data

        # Remove any NaN values introduced by resampling at edges
        resampled = resampled.dropna()

        logger.debug(f"Resampled from {len(data)} to {len(resampled)} points (target: {target_timestep})")

        return resampled

    except (ValueError, KeyError, ZeroDivisionError, FileNotFoundError) as e:
        logger.debug(f"Error resampling to {target_timestep}: {str(e)}")
        logger.debug("Returning original data without resampling")
        return data


def _get_catchment_area_worker(config: Dict, logger) -> float:
    """Get actual catchment area for unit conversion (worker version)"""
    try:
        # Priority 0: Manual override
        fixed_area = config.get('FIXED_CATCHMENT_AREA')
        if fixed_area:
            logger.debug(f"Using fixed catchment area from config: {fixed_area} m²")
            return float(fixed_area)

        domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        project_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, None, 'SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}"

        # Priority 1: Try SUMMA attributes file first (most reliable)
        attrs_file = project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
        if attrs_file.exists():
            try:
                import xarray as xr
                with xr.open_dataset(attrs_file) as attrs:
                    if 'HRUarea' in attrs.data_vars:
                        catchment_area_m2 = float(attrs['HRUarea'].values.sum())
                        if 0 < catchment_area_m2 < 1e12:  # Reasonable area check
                            logger.debug(f"Using catchment area from SUMMA attributes: {catchment_area_m2:.0f} m²")
                            return catchment_area_m2
                    else:
                        logger.debug("HRUarea not found in SUMMA attributes file")
            except (ValueError, KeyError, FileNotFoundError) as e:
                logger.debug(f"Error reading SUMMA attributes file: {str(e)}")
        else:
            logger.debug(f"SUMMA attributes file not found: {attrs_file}")

        # Priority 2: Try basin shapefile
        basin_path = project_dir / "shapefiles" / "river_basins"
        basin_files = list(basin_path.glob("*.shp"))

        if basin_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(basin_files[0])
                area_col = resolve_config_value(config, lambda: config.paths.river_basin_area, 'GRU_area', 'RIVER_BASIN_SHP_AREA')

                logger.debug(f"Found basin shapefile: {basin_files[0]}")
                logger.debug(f"Looking for area column: {area_col}")
                logger.debug(f"Available columns: {list(gdf.columns)}")

                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    logger.debug(f"Total area from column: {total_area}")

                    if 0 < total_area < 1e12:  # Reasonable area check
                        logger.debug(f"Using catchment area from shapefile: {total_area:.0f} m²")
                        return total_area

                # Fallback: calculate from geometry
                if gdf.crs and gdf.crs.is_geographic:
                    # Reproject to UTM for area calculation
                    # Use geopandas estimate_utm_crs() for automatic hemisphere detection
                    utm_crs = gdf.estimate_utm_crs()
                    gdf = gdf.to_crs(utm_crs)

                geom_area = gdf.geometry.area.sum()
                logger.debug(f"Using catchment area from geometry: {geom_area:.0f} m²")
                return geom_area

            except ImportError:
                log_once(logger, logging.WARNING, key='area-geopandas-unavailable',
                         message="geopandas not available for area calculation")
            except (ValueError, KeyError, FileNotFoundError) as e:
                log_once(logger, logging.WARNING, key='area-basin-shp-error',
                         message=f"Error reading basin shapefile: {str(e)}")

        # Priority 3: Try catchment shapefile
        catchment_path = project_dir / "shapefiles" / "catchment"
        catchment_files = list(catchment_path.glob("*.shp"))

        if catchment_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(catchment_files[0])
                area_col = resolve_config_value(config, lambda: config.paths.catchment_area, 'HRU_area', 'CATCHMENT_SHP_AREA')

                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:
                        logger.debug(f"Using catchment area from catchment shapefile: {total_area:.0f} m²")
                        return total_area

            except (ValueError, KeyError, FileNotFoundError) as e:
                log_once(logger, logging.WARNING, key='area-catchment-shp-error',
                         message=f"Error reading catchment shapefile: {str(e)}")

    except (ValueError, KeyError, ZeroDivisionError, FileNotFoundError) as e:
        log_once(logger, logging.WARNING, key='area-calc-error',
                 message=f"Could not calculate catchment area: {str(e)}")

    # Fallback
    log_once(logger, logging.WARNING, key='area-default',
             message="Using default catchment area: 1,000,000 m²")
    return 1e6


def _calculate_metrics_with_target(summa_dir: Path, mizuroute_dir: Path, config: Dict, logger, project_dir: str = None) -> Optional[Dict[Any, Any]]:
    """
    Calculate metrics using proper CalibrationTarget classes.

    This replaces the inline streamflow-only calculation to support all calibration targets
    including streamflow, SWE, SCA, ET, soil moisture, etc.

    Args:
        summa_dir: SUMMA simulation directory
        mizuroute_dir: mizuRoute simulation directory
        config: Configuration dictionary
        logger: Logger instance
        project_dir: Project directory path (if None, reconstructs from config)
    """
    try:
        from pathlib import Path as PathType

        from symfluence.core.calibration.targets import create_calibration_target

        # Use provided project_dir, or reconstruct from config if not provided
        if project_dir is None:
            project_dir = PathType(config.get('SYMFLUENCE_DATA_DIR', '.')) / f"domain_{resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')}"
        else:
            project_dir = PathType(project_dir)

        # Determine the calibration target type
        calibration_var = resolve_config_value(config, lambda: config.optimization.calibration_variable, 'streamflow', 'CALIBRATION_VARIABLE')
        optimization_target = resolve_config_value(config, lambda: config.optimization.target, calibration_var, 'OPTIMIZATION_TARGET').lower()

        logger.debug(f"[METRICS_CALC] Config keys: {list(config.keys())[:10]}")
        logger.debug(f"[METRICS_CALC] OPTIMIZATION_TARGET in config: {resolve_config_value(config, lambda: config.optimization.target, None, 'OPTIMIZATION_TARGET')}")
        logger.debug(f"[METRICS_CALC] CALIBRATION_VARIABLE in config: {resolve_config_value(config, lambda: config.optimization.calibration_variable, None, 'CALIBRATION_VARIABLE')}")
        logger.debug(f"Creating calibration target for: {optimization_target}")

        known_targets = {
            'streamflow', 'flow', 'discharge', 'swe', 'sca', 'snow_depth', 'snow',
            'gw_depth', 'gw_grace', 'groundwater', 'gw', 'et', 'latent_heat',
            'evapotranspiration', 'sm_point', 'sm_smap', 'sm_esa', 'sm_ismn',
            'soil_moisture', 'sm', 'tws', 'grace', 'grace_tws', 'total_storage',
            'stor_grace', 'stor_mb', 'multivariate',
        }
        if optimization_target not in known_targets:
            log_once(logger, logging.WARNING, key=f'unknown-opt-target-{optimization_target}',
                     message=f"Unknown optimization target '{optimization_target}', defaulting to streamflow")
        target = create_calibration_target(optimization_target, config, project_dir, logger)

        # Validate mizuRoute output exists when routing is required for streamflow
        if optimization_target in ['streamflow', 'flow', 'discharge']:
            domain_method = resolve_config_value(config, lambda: config.domain.definition_method, 'lumped', 'DOMAIN_DEFINITION_METHOD')
            routing_delineation = resolve_config_value(config, lambda: config.domain.delineation.routing, 'lumped', 'ROUTING_DELINEATION')

            # Check if routing is required (non-lumped domain or river_network routing)
            needs_routing = (
                domain_method not in ['point', 'lumped'] or
                (domain_method == 'lumped' and routing_delineation == 'river_network')
            )

            if needs_routing:
                if mizuroute_dir is None:
                    logger.error("Routing required but mizuroute_dir is None")
                    return None

                mizuroute_dir_path = PathType(mizuroute_dir) if not isinstance(mizuroute_dir, PathType) else mizuroute_dir
                mizu_files = list(mizuroute_dir_path.glob("*.nc")) if mizuroute_dir_path.exists() else []

                if not mizu_files:
                    logger.error(
                        f"Routing required for {domain_method} domain but no mizuRoute output files found in {mizuroute_dir}. "
                        "Cannot calculate streamflow metrics without routed output."
                    )
                    return None

                logger.debug(f"Found {len(mizu_files)} mizuRoute output files for metrics calculation")

        # Calculate metrics using the target
        logger.debug(f"Calculating metrics from {summa_dir}")

        # DIAGNOSTIC: Check what files exist and their SWE values
        day_files = list(Path(summa_dir).glob("*_day.nc"))
        if day_files:
            import xarray as xr
            with xr.open_dataset(day_files[0]) as ds:
                if 'scalarSWE' in ds:
                    swe_raw = ds['scalarSWE']
                    if swe_raw.size > 0:
                        logger.debug(
                            "WORKER DIAG SWE in file: min=%.3f, max=%.3f kg/m²",
                            float(swe_raw.min()),
                            float(swe_raw.max()),
                        )
                    else:
                        logger.debug("WORKER DIAG SWE in file: array is empty")

        metrics = target.calculate_metrics(
            summa_dir,
            mizuroute_dir=mizuroute_dir,
            calibration_only=True
        )

        if metrics:
            logger.debug(f"Metrics calculated successfully: {list(metrics.keys())}")
            return metrics
        else:
            log_once(logger, logging.WARNING, key='empty-target-metrics',
                     message="Calibration target returned empty metrics")
            return None

    except (ValueError, KeyError, ZeroDivisionError, FileNotFoundError) as e:
        import traceback
        # First occurrence carries the traceback at ERROR; identical repeats
        # (same exception message every evaluation) drop to DEBUG.
        log_once(
            logger, logging.ERROR, key=f'target-metrics-error-{e}',
            message=(
                f"Error calculating metrics with calibration target: {str(e)}\n"
                f"{traceback.format_exc()}"
            ),
        )
        return None


def _calculate_metrics_inline_worker(summa_dir: Path, mizuroute_dir: Path, config: Dict, logger) -> Optional[Dict[Any, Any]]:
    """
    Calculate metrics inline without using CalibrationTarget classes (STREAMFLOW ONLY).

    DEPRECATED: This function is kept for backward compatibility but only supports streamflow.
    Use _calculate_metrics_with_target instead for multi-target support.
    """
    try:
        logger.debug("Starting inline metrics calculation")
        logger.debug(f"SUMMA dir: {summa_dir}")
        logger.debug(f"mizuRoute dir: {mizuroute_dir}")
        logger.debug(f"SUMMA dir exists: {summa_dir.exists()}")
        logger.debug(f"mizuRoute dir exists: {mizuroute_dir.exists() if mizuroute_dir else 'None'}")

        source = _select_inline_simulation_source(summa_dir, mizuroute_dir, config, logger)
        if source is None:
            return None

        sim_file, use_mizuroute, catchment_area = source
        logger.debug(f"Using simulation file: {sim_file}")
        logger.debug(f"Extracting simulated streamflow (use_mizuroute={use_mizuroute})...")

        sim_data = _extract_inline_simulated_streamflow(
            sim_file, use_mizuroute, summa_dir, config, catchment_area, logger
        )
        if sim_data is None:
            return None

        obs_data = _load_inline_observations(config, logger)
        if obs_data is None:
            return None

        obs_period, sim_period = _filter_inline_calibration_period(
            obs_data, sim_data, resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD'), logger
        )
        aligned = _align_inline_timeseries(obs_period, sim_period, config, logger)
        if aligned is None:
            return None

        obs_valid, sim_valid = aligned
        if len(obs_valid) < 10:
            return None

        return _compute_inline_metrics(obs_valid, sim_valid, logger)

    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        return None
    except (ValueError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Error in inline metrics calculation: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


def _select_inline_simulation_source(
    summa_dir: Path,
    mizuroute_dir: Path,
    config: Dict,
    logger,
) -> Optional[Tuple[Path, bool, float]]:
    if mizuroute_dir and mizuroute_dir.exists():
        mizu_files = list(mizuroute_dir.glob("*.nc"))
        logger.debug(f"Found {len(mizu_files)} mizuRoute .nc files")
        for file in mizu_files[:3]:
            logger.debug(f"mizuRoute file: {file.name}")
        if mizu_files:
            logger.debug("Using mizuRoute files (already in m³/s)")
            return mizu_files[0], True, 0.0

    domain_method = resolve_config_value(config, lambda: config.domain.definition_method, 'lumped', 'DOMAIN_DEFINITION_METHOD')
    routing_delineation = resolve_config_value(config, lambda: config.domain.delineation.routing, 'lumped', 'ROUTING_DELINEATION')
    if domain_method not in ['lumped', 'point'] or routing_delineation != 'lumped':
        logger.error(
            "No mizuRoute output found and domain/routing not both lumped "
            f"(Domain: {domain_method}, Routing: {routing_delineation}). "
            "Cannot fall back to SUMMA runoff."
        )
        return None

    summa_files = list(summa_dir.glob("*timestep.nc"))
    logger.debug(f"Found {len(summa_files)} SUMMA timestep files")
    if not summa_files:
        logger.debug("No simulation files found")
        return None

    try:
        catchment_area = _get_catchment_area_worker(config, logger)
        logger.debug(f"Got catchment area = {catchment_area:.2e} m²")
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.debug(f"Error getting catchment area: {str(e)}")
        catchment_area = 1e6

    logger.debug("Using SUMMA files (lumped/point domain and routing, need m/s to m³/s conversion)")
    return summa_files[0], False, catchment_area


def _extract_inline_simulated_streamflow(
    sim_file: Path,
    use_mizuroute: bool,
    summa_dir: Path,
    config: Dict,
    catchment_area: float,
    logger,
) -> Optional[pd.Series]:
    import xarray as xr

    try:
        with xr.open_dataset(sim_file) as ds:
            if use_mizuroute:
                return _extract_inline_mizuroute_streamflow(ds, logger, config, sim_file)
            return _extract_inline_summa_streamflow(ds, summa_dir, config, catchment_area, logger)
    except (ValueError, KeyError, ZeroDivisionError, FileNotFoundError) as e:
        logger.error(f"Exception extracting simulated streamflow: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def _extract_inline_mizuroute_streamflow(ds, logger, config=None, sim_file=None) -> Optional[pd.Series]:
    routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
    routing_var = next((name for name in routing_vars if name in ds.variables), None)
    if routing_var is None:
        return None

    var = ds[routing_var]
    try:
        if 'seg' not in var.dims and 'reachID' not in var.dims:
            return None
        # Shared reach rule (SIM_REACH_ID, topology outlet, else highest mean)
        from symfluence.core.modeling.utilities.reach_selection import (
            select_outlet_series,
            sim_reach_id_from_config,
            topology_candidates_from_config,
        )
        sim_data = select_outlet_series(
            ds, var, sim_reach_id_from_config(config),
            topology_candidates_from_config(config, near=sim_file), logger,
        )
        logger.debug(
            f"Extracted {routing_var} (mizuRoute), mean = {float(sim_data.mean().item()):.2f} m³/s"
        )
        return sim_data
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.debug(f"Error extracting outlet segment from {routing_var}: {str(e)}")
        return None


def _extract_inline_summa_streamflow(
    ds,
    summa_dir: Path,
    config: Dict,
    catchment_area: float,
    logger,
) -> Optional[pd.Series]:
    summa_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
    for var_name in summa_vars:
        if var_name not in ds.variables:
            continue

        var = ds[var_name]
        units = var.attrs.get('units', 'unknown')
        logger.debug(f"Worker found streamflow variable {var_name} with units: '{units}'")

        if _inline_is_mass_flux(var, logger, var_name):
            logger.debug(f"Worker: Converting {var_name} from mass flux to volume flux (dividing by 1000)")
            var = var / 1000.0

        sim_data = _try_inline_area_weighted_aggregation(var, summa_dir, config, logger, var_name)
        if sim_data is None:
            sim_data = _inline_extract_first_series(var)
            logger.debug(f"Converting {var_name} using catchment area = {catchment_area:.2e} m²")
            logger.debug(f"Pre-conversion mean = {sim_data.mean():.2e}")
            sim_data = sim_data * catchment_area
            logger.debug(f"Post-conversion mean = {float(sim_data.mean().item()):.2f} m³/s")

        return sim_data

    logger.debug("sim_data is None after trying all SUMMA variables")
    return None


def _inline_is_mass_flux(var, logger, var_name: str) -> bool:
    if 'units' in var.attrs and 'kg' in var.attrs['units'] and 's-1' in var.attrs['units']:
        return True
    if float(var.mean().item()) > 1e-6:
        logger.debug(
            f"Worker: Variable {var_name} mean ({float(var.mean().item()):.2e}) is unreasonably high. "
            "Assuming mislabeled mass flux."
        )
        return True
    return False


def _try_inline_area_weighted_aggregation(
    var,
    summa_dir: Path,
    config: Dict,
    logger,
    var_name: str,
) -> Optional[pd.Series]:
    import xarray as xr

    if len(var.shape) <= 1:
        return None

    try:
        possible_attr_paths = [
            summa_dir.parent.parent.parent / 'settings' / 'SUMMA' / 'attributes.nc',
            Path(resolve_config_value(config, lambda: config.system.data_dir, None, 'SYMFLUENCE_DATA_DIR')) / f"domain_{resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'attributes.nc'
        ]
        attrs_file = next((path for path in possible_attr_paths if path.exists()), None)
        if attrs_file is None:
            return None

        with xr.open_dataset(attrs_file) as attrs:
            # Same rule as the evaluator and postprocessor: HRU areas, or GRU
            # areas (HRUarea summed by hru2gruId when nHRU != nGRU).
            from symfluence.core.modeling.utilities.area_weighting import area_weighted_discharge
            weighted = area_weighted_discharge(var, attrs)
            if weighted is not None:
                logger.debug(f"Worker: Performing area-weighted aggregation for {var_name}")
                return weighted.to_pandas()
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.debug(f"Aggregation failed, falling back: {e}")

    return None


def _inline_extract_first_series(var) -> pd.Series:
    if len(var.shape) <= 1:
        return var.to_pandas()
    if 'hru' in var.dims:
        return var.isel(hru=0).to_pandas()
    if 'gru' in var.dims:
        return var.isel(gru=0).to_pandas()

    non_time_dims = [dim for dim in var.dims if dim != 'time']
    if non_time_dims:
        return var.isel({non_time_dims[0]: 0}).to_pandas()
    return var.to_pandas()


def _load_inline_observations(config: Dict, logger) -> Optional[pd.Series]:
    try:
        domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        project_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, None, 'SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}"
        obs_path = resolve_data_subdir(project_dir, 'observations') / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"

        logger.debug(f"Looking for observations at: {obs_path}")
        if not obs_path.exists():
            logger.debug("Observation file does not exist")
            return None

        obs_df = pd.read_csv(obs_path)
        date_col = next((col for col in obs_df.columns if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)
        flow_col = next((col for col in obs_df.columns if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow'])), None)

        logger.debug(f"Found date_col={date_col}, flow_col={flow_col}")
        if not date_col or not flow_col:
            logger.debug("Missing date or flow column in observations")
            return None

        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
        obs_df.set_index('DateTime', inplace=True)
        obs_data = obs_df[flow_col]
        logger.debug(f"Loaded {len(obs_data)} observation points")
        return obs_data
    except (ValueError, KeyError, ZeroDivisionError, FileNotFoundError) as e:
        logger.error(f"Exception loading observations: {str(e)}")
        return None


def _filter_inline_calibration_period(
    obs_data: pd.Series,
    sim_data: pd.Series,
    cal_period: str,
    logger,
) -> Tuple[pd.Series, pd.Series]:
    if not cal_period:
        return obs_data, sim_data

    try:
        dates = parse_period(cal_period, allow_extra=True)
        if dates is None:
            return obs_data, sim_data

        start_date, end_date = dates
        obs_period = obs_data[(obs_data.index >= start_date) & (obs_data.index <= end_date)]

        sim_time_diff = sim_data.index[1] - sim_data.index[0] if len(sim_data) > 1 else pd.Timedelta(hours=1)
        if pd.Timedelta(minutes=45) <= sim_time_diff <= pd.Timedelta(minutes=75):
            sim_data.index = sim_data.index.round('h')

        sim_period = sim_data[(sim_data.index >= start_date) & (sim_data.index <= end_date)]
        logger.debug(f"After period filtering, sim_period mean = {float(sim_period.mean().item()):.2f}")
        return obs_period, sim_period
    except (KeyError, ValueError):
        return obs_data, sim_data


def _align_inline_timeseries(
    obs_period: pd.Series,
    sim_period: pd.Series,
    config: Dict,
    logger,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    if obs_period.index.tz is not None and sim_period.index.tz is None:
        sim_period.index = sim_period.index.tz_localize(obs_period.index.tz)
    elif obs_period.index.tz is None and sim_period.index.tz is not None:
        obs_period.index = obs_period.index.tz_localize(sim_period.index.tz)

    calibration_timestep = resolve_config_value(config, lambda: config.optimization.calibration_timestep, 'native', 'CALIBRATION_TIMESTEP').lower()
    if calibration_timestep != 'native':
        obs_period = resample_to_timestep(obs_period, calibration_timestep, logger)
        sim_period = resample_to_timestep(sim_period, calibration_timestep, logger)
        logger.debug(f"After resampling, sim_period mean = {float(sim_period.mean().item()):.2f}")

    common_idx = obs_period.index.intersection(sim_period.index)
    logger.debug(f"obs_period has {len(obs_period)} points, sim_period has {len(sim_period)} points")
    logger.debug(f"obs_period type: {type(obs_period)}, sim_period type: {type(sim_period)}")
    logger.debug(f"Common index has {len(common_idx)} points")
    if len(common_idx) == 0:
        logger.debug("No common timesteps between sim and obs")
        return None

    if isinstance(obs_period, pd.DataFrame):
        obs_period = obs_period.squeeze()
    if isinstance(sim_period, pd.DataFrame):
        sim_period = sim_period.squeeze()

    obs_common = pd.to_numeric(obs_period.loc[common_idx], errors='coerce')
    sim_common = pd.to_numeric(sim_period.loc[common_idx], errors='coerce')  # type: ignore[call-overload]
    logger.debug(f"After intersection, sim_common mean = {float(sim_common.mean().item()):.2f}")

    valid = ~(obs_common.isna() | sim_common.isna() | (obs_common < -900) | (sim_common < -900))
    return obs_common[valid], sim_common[valid]


def _compute_inline_metrics(obs_valid: pd.Series, sim_valid: pd.Series, logger) -> Optional[Dict[Any, Any]]:
    try:
        nse_val = calc_nse(obs_valid.values, sim_valid.values)
        kge_result = calc_kge(obs_valid.values, sim_valid.values, return_components=True)
        rmse_val = calc_rmse(obs_valid.values, sim_valid.values)
        mae_val = calc_mae(obs_valid.values, sim_valid.values)
        pbias_val = calc_pbias(obs_valid.values, sim_valid.values)

        kge_val = kge_result['KGE'] if isinstance(kge_result, dict) else kge_result
        r_val = kge_result['r'] if isinstance(kge_result, dict) else obs_valid.corr(sim_valid)
        alpha_val = kge_result['alpha'] if isinstance(kge_result, dict) else np.nan
        beta_val = kge_result['beta'] if isinstance(kge_result, dict) else np.nan

        logger.debug(
            f"Final KGE = {kge_val:.4f} (obs_mean={obs_valid.mean():.2f}, "
            f"sim_mean={float(sim_valid.mean().item()):.2f})"
        )

        return {
            'Calib_NSE': nse_val, 'Calib_KGE': kge_val, 'Calib_RMSE': rmse_val,
            'Calib_MAE': mae_val, 'Calib_PBIAS': pbias_val,
            'Calib_r': r_val, 'Calib_alpha': alpha_val, 'Calib_beta': beta_val,
            'Calib_correlation': r_val,
            'NSE': nse_val, 'KGE': kge_val, 'RMSE': rmse_val, 'MAE': mae_val,
            'PBIAS': pbias_val, 'correlation': r_val
        }
    except (KeyError, ValueError):
        return None


def _calculate_multitarget_objectives(task: Dict, summa_dir: str, mizuroute_dir: str,
                                       config: Dict, project_dir: str, logger) -> List[float]:
    """
    Calculate objectives for multi-target optimization in worker process.

    This function should be called from _evaluate_parameters_worker when
    multi_target_mode is True.

    Parameters
    ----------
    task : Dict
        Task dictionary containing:
        - multi_target_mode: bool
        - primary_target_type: str
        - secondary_target_type: str
        - primary_metric: str
        - secondary_metric: str
    summa_dir : str
        Path to SUMMA simulation directory
    mizuroute_dir : str
        Path to mizuRoute simulation directory
    config : Dict
        Configuration dictionary
    project_dir : str
        Project directory path
    logger : Logger
        Logger instance

    Returns
    -------
    List[float]
        [objective1, objective2] values
    """
    from pathlib import Path

    from symfluence.core.calibration.targets import create_calibration_target

    # Convert to Path objects for safety
    summa_dir = Path(summa_dir)
    mizuroute_dir = Path(mizuroute_dir) if mizuroute_dir else None

    project_path = Path(project_dir)

    def create_target(target_type: str):
        """Create calibration target by type name."""
        return create_calibration_target(target_type, config, project_path, logger)

    def extract_metric(metrics: Dict, metric_name: str) -> float:
        """Extract specific metric from metrics dictionary."""
        if not metrics:
            return -1.0

        # Try exact match
        if metric_name in metrics:
            val = metrics[metric_name]
            return val if val is not None and not np.isnan(val) else -1.0

        # Try with Calib_ prefix
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics:
            val = metrics[calib_key]
            return val if val is not None and not np.isnan(val) else -1.0

        # Try suffix match
        for key, value in metrics.items():
            if key.endswith(f"_{metric_name}"):
                return value if value is not None and not np.isnan(value) else -1.0

        return -1.0

    try:
        if task.get('multi_target_mode', False):
            # Multi-target mode: use two different calibration targets
            primary_target_type = task.get('primary_target_type', 'streamflow')
            secondary_target_type = task.get('secondary_target_type', 'gw_depth')
            primary_metric = task.get('primary_metric', 'KGE')
            secondary_metric = task.get('secondary_metric', 'KGE')

            # Create targets
            primary_target = create_target(primary_target_type)
            secondary_target = create_target(secondary_target_type)

            # Debug: Log target types being used
            logger.debug(f"[MULTI-TARGET DEBUG] Primary target: {primary_target_type} -> {type(primary_target).__name__}")
            logger.debug(f"[MULTI-TARGET DEBUG] Secondary target: {secondary_target_type} -> {type(secondary_target).__name__}")
            logger.debug(f"[MULTI-TARGET DEBUG] Secondary has calculate_metrics: {hasattr(secondary_target, 'calculate_metrics')}")

            # Check if TWSEvaluator's calculate_metrics is being used
            if 'tws' in secondary_target_type.lower() or 'grace' in secondary_target_type.lower() or 'stor_grace' in secondary_target_type.lower():
                from symfluence.core.modeling.evaluators.tws import TWSEvaluator
                is_tws = isinstance(secondary_target, TWSEvaluator)
                logger.debug(f"[MULTI-TARGET DEBUG] Secondary is TWSEvaluator instance: {is_tws}")
                logger.debug(f"[MULTI-TARGET DEBUG] Secondary MRO: {[c.__name__ for c in type(secondary_target).__mro__]}")

            # Calculate metrics
            primary_metrics = primary_target.calculate_metrics(
                summa_dir,
                mizuroute_dir=mizuroute_dir
            )
            secondary_metrics = secondary_target.calculate_metrics(
                summa_dir,
                mizuroute_dir=mizuroute_dir
            )

            obj1 = extract_metric(primary_metrics, primary_metric)
            obj2 = extract_metric(secondary_metrics, secondary_metric)

        else:
            # Default mode: NSE and KGE from same target
            target_type = task.get('calibration_variable', 'streamflow')
            target = create_target(target_type)

            metrics = target.calculate_metrics(
                summa_dir,
                mizuroute_dir=mizuroute_dir
            )

            obj1 = extract_metric(metrics, 'NSE')
            obj2 = extract_metric(metrics, 'KGE')

        return [obj1, obj2]

    except (ValueError, KeyError, ZeroDivisionError, FileNotFoundError) as e:
        log_once(logger, logging.WARNING, key='multi-target-objective-failed',
                 message=f"Multi-target objective calculation failed: {e}")
        return [-1.0, -1.0]
