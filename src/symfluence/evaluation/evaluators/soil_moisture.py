#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Soil Moisture Evaluator.

Evaluates simulated soil moisture from SUMMA against observations from multiple sources:
satellite remote sensing (SMAP, ESA CCI), tower networks (ISMN), and point measurements.

Supported Observation Sources:
    - SMAP (Soil Moisture Active/Passive): NASA satellite, ~3 km resolution, surface & rootzone
    - ESA CCI (Climate Change Initiative): ESA satellite, ~25 km resolution, surface layer only
    - ISMN (Int'l Soil Moisture Network): Tower observations, point-scale, multiple depths
    - Point observations: Generic tower/station measurements at specified depths

Model Output (SUMMA):
    - Variable: mLayerVolFracLiq (volumetric liquid water fraction, 0-1)
    - Dimensions: (time, hru/gru, mLayerDepth)
    - Layer depths: mLayerDepth array specifying soil layer thicknesses

Spatial and Depth Handling:
    - Spatial: Collapse HRU/GRU via mean (multi-HRU) or isel (single HRU)
    - Depth: Target layer selection or depth-weighted averaging to observation depth
    - SMAP surface: Typically 0-5 cm depth from model
    - SMAP rootzone: Typically 0-100 cm (depth-weighted average)
    - ESA CCI surface: 0-5 cm depth
    - ISMN/Point: Target-specific depth with ±5cm tolerance (configurable)

Configuration:
    SM_TARGET_DEPTH: Target depth for point observations ('auto' or float meters)
    SM_DEPTH_TOLERANCE: Acceptable depth difference for matching layers (default: 0.05 m)
    SMAP_LAYER: 'surface_sm' or 'rootzone_sm' (default: 'surface_sm')
    SM_TEMPORAL_AGGREGATION: 'daily_mean' (default) or none
    SM_USE_QUALITY_CONTROL: Enable QC filtering (default: True)
    SM_MIN_VALID_PIXELS: Minimum valid pixels for SMAP QC (default: 10)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.data.observation.paths import first_existing_path, soil_moisture_observation_candidates
from symfluence.evaluation.output_file_locator import OutputFileLocator
from symfluence.evaluation.registry import EvaluationRegistry

from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('SOIL_MOISTURE')
class SoilMoistureEvaluator(ModelEvaluator):
    """Soil moisture evaluator supporting multiple observation sources.

    Comprehensive soil moisture evaluation framework supporting four distinct
    observation sources with source-specific depth handling and extraction logic.

    Supported Targets:
        - 'sm_point': Generic point observations at specified depth (towers, in-situ)
        - 'sm_smap': NASA SMAP satellite data (surface or rootzone)
        - 'sm_ismn': ISMN tower network (multiple heights, standardized)
        - 'sm_esa': ESA CCI satellite (surface layer, ~25 km resolution)

    Source-Specific Characteristics:
        SMAP:
            - Resolution: ~3 km (descending passes)
            - Depth: Surface (0-5 cm) or rootzone (0-100 cm)
            - Method: Depth-weighted averaging for rootzone
            - QC: Minimum valid pixel count threshold
        ESA CCI:
            - Resolution: ~25 km (coarser than SMAP)
            - Depth: Surface only (0-5 cm assumed)
            - Method: Single depth-weighted average
            - Coverage: 1978-present (longest record)
        ISMN:
            - Resolution: Point-scale (tower location)
            - Depth: Multiple sensors per tower (multi-layer observations)
            - Method: Target depth with tolerance matching
            - Coverage: 600+ stations, varies by site
        Point observations:
            - Custom stations or tower data
            - User-specified target depth
            - Flexible date formats and column naming

    Depth Matching Strategy (Point/ISMN):
        1. If target_depth = 'auto': Select shallowest available layer
        2. Else: Find layer closest to target_depth (within ±50 cm tolerance)
        3. Fallback: Use layer 0 (shallowest) if no close match
        4. Layer identification: Cumulative depth to layer midpoint

    Depth-Weighted Averaging (SMAP/ESA):
        Computes weighted mean of soil layers within target depth range:
        - Weights: Fraction of layer thickness in target range
        - Sum normalized: weights / sum(weights) = 1.0
        - Handles variable layer thicknesses

    Configuration:
        SM_TARGET_DEPTH: Point target depth (default: 'auto', float meters)
        SM_DEPTH_TOLERANCE: Acceptable depth tolerance (default: 0.05 m)
        SMAP_LAYER: 'surface_sm' or 'rootzone_sm' (default: 'surface_sm')
        SM_TEMPORAL_AGGREGATION: 'daily_mean' or none (default: 'daily_mean')
        SM_USE_QUALITY_CONTROL: Enable/disable QC (default: True)
        SM_MIN_VALID_PIXELS: SMAP QC threshold (default: 10)

    Attributes:
        optimization_target: 'sm_point', 'sm_smap', 'sm_ismn', or 'sm_esa'
        variable_name: Same as optimization_target
        target_depth: Target depth for point/ISMN (float or 'auto')
        smap_layer: 'surface_sm' or 'rootzone_sm' for SMAP
        use_quality_control: Enable/disable QC filtering
        min_valid_pixels: SMAP minimum valid pixel threshold
    """

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger):
        """Initialize soil moisture evaluator with source-specific depth configuration.

        Determines observation source (point, SMAP, ISMN, ESA CCI) and configures
        target depth, temporal aggregation, and quality control settings specific
        to each source.

        Target Resolution Priority:
            1. config.optimization.target (typed config)
            2. EVALUATION_VARIABLE (dict config, if contains 'sm_' or 'soil')
            3. Default: 'streamflow' (will be overridden if EVALUATION_VARIABLE matches)

        Source-Specific Configuration:
            sm_point:
                - SM_TARGET_DEPTH: Target depth ('auto' or float meters)
                - SM_DEPTH_TOLERANCE: Acceptable depth difference (default: 0.05 m)
                - Uses _extract_point_soil_moisture()
            sm_smap:
                - SMAP_LAYER: 'surface_sm' (0-5 cm) or 'rootzone_sm' (0-100 cm)
                - SM_TEMPORAL_AGGREGATION: temporal aggregation method
                - Uses _depth_weighted_mean() for layer integration
            sm_ismn:
                - SM_TARGET_DEPTH: Target depth from ISMN config or config_dict
                - SM_TEMPORAL_AGGREGATION: From config.evaluation.ismn or default
                - Uses _extract_point_soil_moisture() with depth matching
            sm_esa:
                - ESA_SURFACE_DEPTH_M: Surface layer depth (default: 0.05 m)
                - SM_TEMPORAL_AGGREGATION: temporal aggregation method
                - Uses depth-weighted averaging for ESA surface layer

        Quality Control:
            SM_USE_QUALITY_CONTROL: Enable/disable QC (default: True)
            SM_MIN_VALID_PIXELS: SMAP minimum valid pixels (default: 10)
            - Filters SMAP obs where valid_px < threshold

        Args:
            config: Typed configuration object (SymfluenceConfig)
            project_dir: Project root directory
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger)

        # Get optimization target from typed config
        self.optimization_target = self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow',
            dict_key='OPTIMIZATION_TARGET'
        )

        # Check if target is a valid soil moisture target, otherwise look at EVALUATION_VARIABLE
        if self.optimization_target not in ['sm_point', 'sm_smap', 'sm_esa', 'sm_ismn']:
            eval_var = self._get_config_value(
                lambda: None, default='', dict_key='EVALUATION_VARIABLE'
            )
            if any(x in eval_var for x in ['sm_', 'soil']):
                self.optimization_target = eval_var

        self.variable_name = self.optimization_target

        if self.optimization_target == 'sm_point':
            self.target_depth = self._get_config_value(
                lambda: self.config.evaluation.soil_moisture.target_depth_m,
                default='auto',
                dict_key='SM_TARGET_DEPTH'
            )
            self.depth_tolerance = self._get_config_value(
                lambda: self.config.evaluation.soil_moisture.depth_tolerance_m,
                default=0.05,
                dict_key='SM_DEPTH_TOLERANCE'
            )
        elif self.optimization_target == 'sm_smap':
            self.smap_layer = self._get_config_value(
                lambda: self.config.evaluation.smap.layer,
                default='surface_sm',
                dict_key='SMAP_LAYER'
            )
            self.temporal_aggregation = self._get_config_value(
                lambda: self.config.evaluation.soil_moisture.temporal_aggregation,
                default='daily_mean',
                dict_key='SM_TEMPORAL_AGGREGATION'
            )
        elif self.optimization_target == 'sm_ismn':
            self.target_depth = self._get_config_value(
                lambda: self.config.evaluation.ismn.target_depth_m,
                default='auto',
                dict_key='ISMN_TARGET_DEPTH_M'
            )
            self.temporal_aggregation = self._get_config_value(
                lambda: self.config.evaluation.ismn.temporal_aggregation,
                default='daily_mean',
                dict_key='ISMN_TEMPORAL_AGGREGATION'
            )
        elif self.optimization_target == 'sm_esa':
            self.esa_surface_depth = float(self._get_config_value(
                lambda: self.config.evaluation.esa_cci.surface_depth_m,
                default=0.05,
                dict_key='ESA_SURFACE_DEPTH_M'
            ))
            self.temporal_aggregation = self._get_config_value(
                lambda: self.config.evaluation.soil_moisture.temporal_aggregation,
                default='daily_mean',
                dict_key='SM_TEMPORAL_AGGREGATION'
            )

        self.use_quality_control = self._get_config_value(
            lambda: self.config.evaluation.soil_moisture.use_quality_control,
            default=True,
            dict_key='SM_USE_QUALITY_CONTROL'
        )
        self.min_valid_pixels = self._get_config_value(
            lambda: self.config.evaluation.smap.min_valid_pixels,
            default=10,
            dict_key='SM_MIN_VALID_PIXELS'
        )

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate SUMMA output files containing soil moisture variables.

        Searches for NetCDF files with mLayerVolFracLiq (volumetric soil water fraction)
        and mLayerDepth (soil layer thicknesses).

        Args:
            sim_dir: Directory containing SUMMA simulation output

        Returns:
            List[Path]: Paths to soil moisture output files (NetCDF)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_soil_moisture_files(sim_dir)

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract source-specific soil moisture from SUMMA output.

        Dispatches to extraction method based on optimization_target:
        - sm_point/sm_ismn: Single layer at target depth
        - sm_smap: Surface (0-5 cm) or rootzone (0-100 cm) depth-weighted mean
        - sm_esa: Surface layer (0-5 cm) depth-weighted mean

        Args:
            sim_files: List of SUMMA output files (NetCDF)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Time series of volumetric soil moisture (0-1 fraction)

        Raises:
            Exception: If extraction method raises error
        """
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'sm_point':
                    return self._extract_point_soil_moisture(ds)
                elif self.optimization_target == 'sm_smap':
                    return self._extract_smap_soil_moisture(ds)
                elif self.optimization_target == 'sm_ismn':
                    return self._extract_point_soil_moisture(ds)
                elif self.optimization_target == 'sm_esa':
                    return self._extract_esa_soil_moisture(ds)
                else:
                    return self._extract_point_soil_moisture(ds)
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting soil moisture data from {sim_file}: {str(e)}")
            raise

    def _extract_point_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        if 'mLayerVolFracLiq' not in ds.variables:
            raise ValueError("mLayerVolFracLiq variable not found")
        soil_moisture_var = ds['mLayerVolFracLiq']
        layer_depths = ds['mLayerDepth']

        # Find layer dimension
        layer_dims = [dim for dim in soil_moisture_var.dims if 'mid' in dim.lower() or 'layer' in dim.lower()]
        if not layer_dims:
            raise ValueError("Layer dimension not found in soil moisture output")
        layer_dim = layer_dims[0]

        # Collapse spatial dimensions on depths for layer selection
        depths_xr = layer_depths
        for dim in ['hru', 'gru']:
            if dim in depths_xr.dims:
                if depths_xr.sizes[dim] == 1:
                    depths_xr = depths_xr.isel({dim: 0})
                else:
                    depths_xr = depths_xr.mean(dim=dim)

        target_layer_idx = self._find_target_layer(depths_xr)

        # Select target layer first, then use base class method for remaining collapse
        sim_xr = soil_moisture_var.isel({layer_dim: target_layer_idx})
        sim_data = self._collapse_spatial_dims(sim_xr, aggregate='mean')
        return sim_data

    def _find_target_layer(self, layer_depths: xr.DataArray) -> int:
        try:
            if self.target_depth == 'auto':
                return 0
            try:
                target_depth_m = float(self.target_depth)
            except (ValueError, TypeError):
                return 0

            if 'time' in layer_depths.dims:
                depths_sample = layer_depths.isel(time=0).values
            else:
                depths_sample = layer_depths.values

            cumulative_depths = np.cumsum(depths_sample) - depths_sample / 2
            depth_differences = np.abs(cumulative_depths - target_depth_m)
            best_layer_idx = np.argmin(depth_differences)
            return int(best_layer_idx)
        except (ValueError, TypeError, KeyError, IndexError) as e:
            self.logger.debug(f"Could not determine best layer index, defaulting to 0: {e}")
            return 0

    def _extract_smap_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        soil_moisture_var = ds['mLayerVolFracLiq']
        layer_depths = ds['mLayerDepth'] if 'mLayerDepth' in ds.variables else None

        # Collapse spatial dimensions first (preserving layer dimension)
        sim_xr = soil_moisture_var
        for dim in ['hru', 'gru']:
            if dim in sim_xr.dims:
                if sim_xr.sizes[dim] == 1:
                    sim_xr = sim_xr.isel({dim: 0})
                else:
                    sim_xr = sim_xr.mean(dim=dim)

        layer_dims = [dim for dim in sim_xr.dims if 'mid' in dim.lower() or 'layer' in dim.lower()]
        if not layer_dims:
            raise ValueError("Layer dimension not found in simulated soil moisture output")
        layer_dim = layer_dims[0]

        if self.smap_layer == 'surface_sm':
            surface_depth = float(self._get_config_value(
                lambda: self.config.evaluation.smap.surface_depth_m,
                default=0.05,
                dict_key='SMAP_SURFACE_DEPTH_M'
            ))
            if layer_depths is None:
                sim_xr = sim_xr.isel({layer_dim: 0})
            else:
                sim_xr = self._depth_weighted_mean(sim_xr, layer_depths, surface_depth, layer_dim)
        elif self.smap_layer == 'rootzone_sm':
            rootzone_depth = float(self._get_config_value(
                lambda: self.config.evaluation.smap.rootzone_depth_m,
                default=1.0,
                dict_key='SMAP_ROOTZONE_DEPTH_M'
            ))
            if layer_depths is None:
                sim_xr = sim_xr.isel({layer_dim: slice(0, 3)}).mean(dim=layer_dim)
            else:
                sim_xr = self._depth_weighted_mean(sim_xr, layer_depths, rootzone_depth, layer_dim)
        else:
            raise ValueError(f"Unknown SMAP layer: {self.smap_layer}")

        # Use base class method for any remaining spatial dimension collapse
        sim_data = self._collapse_spatial_dims(sim_xr, aggregate='mean')
        return sim_data

    def _depth_weighted_mean(
        self,
        sim_xr: xr.DataArray,
        layer_depths: xr.DataArray,
        target_depth_m: float,
        layer_dim: str,
    ) -> xr.DataArray:
        """Compute depth-weighted mean of soil moisture over target depth range.

        Averages soil moisture from multiple layers within a specified depth range,
        weighting each layer by the fraction of its thickness within the target depth.

        Algorithm:
            1. Extract layer thickness array and collapse spatial dims
            2. Compute cumulative depth to each layer bottom
            3. For each layer: weight = min(thickness, remaining_depth) / target_depth
            4. Normalize weights: weights /= sum(weights) → sum = 1.0
            5. Compute weighted mean: sm_weighted = sum(sm_layer * weight_layer)

        Example (rootzone 0-100 cm):
            Layers: [0-10cm (SM=0.30), 10-40cm (SM=0.25), 40-100cm (SM=0.20), 100-200cm (SM=0.15)]
            Weights: [10/100, 30/100, 60/100, 0/100] = [0.1, 0.3, 0.6, 0.0]
            Result: sm_rootzone = 0.30*0.1 + 0.25*0.3 + 0.20*0.6 + 0.15*0.0 = 0.21

        Handles:
            - Variable layer thicknesses (not uniform)
            - Missing layer depth info (defaults to thickness=0)
            - Target depth exceeding available soil (uses available layers)
            - Time-varying layer thicknesses (uses first time step)

        Args:
            sim_xr: Soil moisture array (time × layer_dim × ...)
            layer_depths: Layer thickness array (layer_dim × ...)
            target_depth_m: Target depth in meters (e.g., 0.05 for SMAP surface, 1.0 for rootzone)
            layer_dim: Name of layer dimension ('mLayerDepth', etc.)

        Returns:
            xr.DataArray: Depth-weighted soil moisture time series
        """
        depth_xr = layer_depths
        for dim in ['hru', 'gru']:
            if dim in depth_xr.dims:
                if depth_xr.sizes[dim] == 1:
                    depth_xr = depth_xr.isel({dim: 0})
                else:
                    depth_xr = depth_xr.mean(dim=dim)
        if 'time' in depth_xr.dims:
            depth_xr = depth_xr.isel(time=0)
        other_dims = [dim for dim in depth_xr.dims if dim != layer_dim]
        if other_dims:
            depth_xr = depth_xr.isel({dim: 0 for dim in other_dims})

        depth_vals = np.asarray(depth_xr.values).astype(float).ravel()
        n_layers = sim_xr.sizes[layer_dim]
        if depth_vals.size < n_layers:
            depth_vals = np.pad(depth_vals, (0, n_layers - depth_vals.size), constant_values=0.0)
        depth_vals = depth_vals[:n_layers]

        weights = np.zeros(n_layers, dtype=float)
        remaining = target_depth_m
        for i, thickness in enumerate(depth_vals):
            if remaining <= 0:
                break
            thickness = float(thickness) if np.isfinite(thickness) else 0.0
            if thickness <= 0:
                continue
            use = min(thickness, remaining)
            weights[i] = use
            remaining -= use

        if weights.sum() <= 0:
            return sim_xr.isel({layer_dim: 0})

        weights /= weights.sum()
        weight_da = xr.DataArray(weights, dims=[layer_dim], coords={layer_dim: sim_xr[layer_dim]})
        return (sim_xr * weight_da).sum(dim=layer_dim)

    def _extract_esa_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        soil_moisture_var = ds['mLayerVolFracLiq']
        layer_depths = ds['mLayerDepth'] if 'mLayerDepth' in ds.variables else None

        # Collapse spatial dimensions first (preserving layer dimension)
        sim_xr = soil_moisture_var
        for dim in ['hru', 'gru']:
            if dim in sim_xr.dims:
                if sim_xr.sizes[dim] == 1:
                    sim_xr = sim_xr.isel({dim: 0})
                else:
                    sim_xr = sim_xr.mean(dim=dim)

        layer_dims = [dim for dim in sim_xr.dims if 'mid' in dim.lower() or 'layer' in dim.lower()]
        if not layer_dims:
            raise ValueError("Layer dimension not found in simulated soil moisture output")
        layer_dim = layer_dims[0]

        if layer_depths is None:
            sim_xr = sim_xr.isel({layer_dim: 0})
        else:
            sim_xr = self._depth_weighted_mean(sim_xr, layer_depths, self.esa_surface_depth, layer_dim)

        # Use base class method for any remaining spatial dimension collapse
        sim_data = self._collapse_spatial_dims(sim_xr, aggregate='mean')
        return sim_data

    def get_observed_data_path(self) -> Path:
        return first_existing_path(
            soil_moisture_observation_candidates(
                self.project_dir,
                self.domain_name,
                self.optimization_target,
            )
        )

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'sm_point':
            if self.target_depth == 'auto':
                depth_columns = [col for col in columns if col.startswith('sm_')]
                if depth_columns:
                    depths = []
                    for col in depth_columns:
                        try:
                            depth_str = col.split('_')[1]
                            depths.append((float(depth_str), col))
                        except (IndexError, ValueError):
                            continue  # Skip columns with unparseable depth
                    if depths:
                        depths.sort()
                        self.target_depth = str(depths[0][0])
                        return depths[0][1]
            else:
                target_depth_str = str(self.target_depth)
                for col in columns:
                    if col.startswith('sm_') and target_depth_str in col:
                        return col
        elif self.optimization_target == 'sm_smap':
            if self.smap_layer in columns:
                return self.smap_layer
            for col in columns:
                if 'surface_sm' in col.lower() or 'rootzone_sm' in col.lower():
                    return col
        elif self.optimization_target == 'sm_ismn':
            target_depth_str = str(self.target_depth)
            for col in columns:
                if col.startswith('sm_') and target_depth_str in col:
                    return col
            for col in columns:
                if col.startswith('sm_'):
                    return col
        elif self.optimization_target == 'sm_esa':
            for col in columns:
                if any(term in col.lower() for term in ['esa', 'soil_moisture', 'sm']):
                    return col
        return None

    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                return None

            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)

            if not date_col or not data_col:
                return None

            if self.optimization_target == 'sm_esa':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
                if obs_df['DateTime'].isna().all():
                    obs_df['DateTime'] = pd.to_datetime(
                        obs_df[date_col],
                        format='%d/%m/%Y',
                        errors='coerce',
                    )
            else:
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')

            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)

            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')

            if self.optimization_target == 'sm_smap' and self.use_quality_control:
                if 'valid_px' in obs_df.columns:
                    valid_pixels = pd.to_numeric(obs_df['valid_px'], errors='coerce')
                    quality_mask = valid_pixels >= self.min_valid_pixels
                    obs_series = obs_series[quality_mask]

            obs_series = obs_series.dropna()

            if hasattr(self, 'temporal_aggregation') and self.temporal_aggregation == 'daily_mean':
                obs_series = obs_series.resample('D').mean().dropna()

            return obs_series
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error loading observed soil moisture data: {str(e)}")
            return None

    def needs_routing(self) -> bool:
        """Determine if soil moisture evaluation requires streamflow routing.

        Soil moisture is measured at point-scale (towers, satellites) and does not
        require streamflow routing models. Storage is evaluated directly without
        downstream propagation.

        Returns:
            bool: False (soil moisture evaluator never requires routing)
        """
        return False
