#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
FUSE Calibration Targets

Provides calibration target classes specifically designed for FUSE model output.
Handles both lumped FUSE output and distributed FUSE + mizuRoute output.

Note: This module has been refactored to use the centralized evaluators in
symfluence.core.modeling.evaluators.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.core.mixins.config import resolve_config_value

# Suppress xarray FutureWarning about timedelta64 decoding
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')

from symfluence.core.constants import UnitConversion
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.evaluators import SnowEvaluator, StreamflowEvaluator
from symfluence.core.registries import R


@R.calibration_targets.add('FUSE_STREAMFLOW')
class FUSEStreamflowTarget(StreamflowEvaluator):
    """FUSE-specific streamflow evaluator handling lumped/distributed modes."""

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        self.is_distributed = resolve_config_value(
            config,
            lambda: config.model.fuse.spatial_mode,
            dict_key='FUSE_SPATIAL_MODE',
        ).lower() == 'distributed'

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Find FUSE or mizuRoute output files."""
        if self.is_distributed:
            # For distributed mode, need mizuRoute routed output (not raw FUSE)
            # mizuRoute files typically have proc_ prefix
            fuse_suffixes = ('runs_def.nc', 'runs_pre.nc', 'runs_best.nc', 'runs_sce.nc')

            # First: look for mizuRoute-specific patterns
            for pattern in [f"proc_*{self.experiment_id}*.nc", "proc_*.nc"]:
                files = list(sim_dir.glob(pattern))
                if files:
                    return [max(files, key=lambda x: x.stat().st_mtime)]

            # Second: any .nc file that is NOT a FUSE output
            all_nc = list(sim_dir.glob("*.nc"))
            non_fuse = [f for f in all_nc
                        if not any(f.name.endswith(s) for s in fuse_suffixes)]
            if non_fuse:
                return [max(non_fuse, key=lambda x: x.stat().st_mtime)]

            # Last resort: any nc file
            if all_nc:
                return [max(all_nc, key=lambda x: x.stat().st_mtime)]
            return []
        else:
            # Look for lumped FUSE files
            search_dirs = [sim_dir, sim_dir / "output"]
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for pattern in ['runs_def.nc', 'runs_pre.nc', 'runs_best.nc']:
                    files = list(search_dir.glob(f"*_{pattern}"))
                    if files:
                        return [files[0]]
            return []

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow with appropriate unit conversion for FUSE."""
        if not sim_files:
            return pd.Series(dtype=float)

        sim_file = sim_files[0]

        if self.is_distributed:
            # Use parent's auto-detection (mizuRoute → VIC → SUMMA fallback)
            # rather than assuming file is always mizuRoute format
            return super().extract_simulated_data(sim_files)
        else:
            # Lumped FUSE output is in mm/day, needs conversion to cms
            with xr.open_dataset(sim_file) as ds:
                if 'q_routed' in ds.variables:
                    sim_var = 'q_routed'
                elif 'q_instnt' in ds.variables:
                    sim_var = 'q_instnt'
                else:
                    raise ModelExecutionError("No runoff variable found in FUSE output")

                # Handle variable dimensions - param_set may or may not exist
                var = ds[sim_var]
                isel_kwargs = {}
                if 'param_set' in var.dims:
                    isel_kwargs['param_set'] = 0
                if 'latitude' in var.dims:
                    isel_kwargs['latitude'] = 0
                if 'longitude' in var.dims:
                    isel_kwargs['longitude'] = 0
                simulated = var.isel(**isel_kwargs) if isel_kwargs else var
                sim_df = cast(pd.Series, simulated.to_pandas())

                # Convert mm/day to cms: Q(cms) = Q(mm/day) * Area(km2) / 86.4
                area_km2 = self._get_catchment_area()
                return sim_df * area_km2 / UnitConversion.MM_DAY_TO_CMS

    def _get_catchment_area(self) -> float:
        """FUSE-specific area calculation logic or fallback."""
        # Simple fallback for now, could be more detailed like the original
        return super()._get_catchment_area() / 1e6 # Base class returns m2


@R.calibration_targets.add('FUSE_SNOW')
class FUSESnowTarget(SnowEvaluator):
    """FUSE-specific snow evaluator."""

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        self.is_distributed = resolve_config_value(
            config,
            lambda: config.model.fuse.spatial_mode,
            dict_key='FUSE_SPATIAL_MODE',
        ).lower() == 'distributed'

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Snow variables are always in FUSE output, not mizuRoute."""
        for pattern in ['runs_def.nc', 'runs_best.nc', 'runs_sce.nc']:
            files = list(sim_dir.glob(f"*_{pattern}"))
            if files:
                return [files[0]]
        return []

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract SWE with FUSE-specific variable naming."""
        if not sim_files:
            return pd.Series(dtype=float)

        with xr.open_dataset(sim_files[0]) as ds:
            # Primary candidates for SWE variables in FUSE
            swe_candidates = ['swe_tot', 'swe_z01', 'swe', 'SWE', 'snowpack']
            sim_var = next((c for c in swe_candidates if c in ds.variables), None)

            if sim_var is None:
                raise ModelExecutionError("No SWE variable found in FUSE output")

            if self.is_distributed:
                simulated = ds[sim_var].isel(param_set=0) if 'param_set' in ds[sim_var].dims else ds[sim_var]
                spatial_dims = [dim for dim in simulated.dims if dim != 'time']
                if spatial_dims:
                    simulated = simulated.mean(dim=spatial_dims)
            else:
                # Handle variable dimensions - param_set may or may not exist
                var = ds[sim_var]
                isel_kwargs = {}
                if 'param_set' in var.dims:
                    isel_kwargs['param_set'] = 0
                if 'latitude' in var.dims:
                    isel_kwargs['latitude'] = 0
                if 'longitude' in var.dims:
                    isel_kwargs['longitude'] = 0
                simulated = var.isel(**isel_kwargs) if isel_kwargs else var

            return cast(pd.Series, simulated.to_pandas())
