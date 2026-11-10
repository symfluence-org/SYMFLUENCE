# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC Result Extractor.

Handles extraction of simulation results from VIC model outputs.
VIC outputs are in NetCDF format with standard variable names.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor

logger = logging.getLogger(__name__)


RUNOFF_VARS = ('OUT_RUNOFF', 'RUNOFF')
BASEFLOW_VARS = ('OUT_BASEFLOW', 'BASEFLOW')


def load_vic_cell_area(domain_file: Optional[Path]):
    """Grid-cell areas (m2) from a VIC domain file, or None if unavailable.

    ``area * frac`` when a land fraction is present, so partial cells weigh
    by their land area.
    """
    if domain_file is None or not Path(domain_file).exists():
        return None
    import xarray as xr
    try:
        with xr.open_dataset(domain_file) as dom:
            if 'area' not in dom.data_vars:
                return None
            area = dom['area'].load()
            if 'frac' in dom.data_vars:
                area = area * dom['frac'].load()
            return area
    except (OSError, ValueError, KeyError) as e:
        logger.debug(f"Could not read VIC cell areas from {domain_file}: {e}")
        return None


def basin_mean_vic_runoff(ds, cell_area=None, include_baseflow: bool = True):
    """Basin-average VIC runoff (+ baseflow) depth per output step.

    The single aggregation rule shared by the calibration worker, the
    postprocessor and the extractor. VIC writes runoff and baseflow as a
    depth (mm per output step) for each grid cell, so the basin value is the
    area-weighted *mean* over active (non-NaN) cells -- summing depths over
    cells inflates discharge by the number of cells.

    Args:
        ds: VIC output dataset.
        cell_area: Optional per-cell areas (DataArray or array with the
            output's spatial shape). Falls back to an ``area`` variable in
            ``ds``; equal weights when neither is available.
        include_baseflow: Add OUT_BASEFLOW when present.

    Returns:
        xarray.DataArray over ``time`` (mm per step).
    """
    import numpy as np
    import xarray as xr

    runoff = next((ds[v] for v in RUNOFF_VARS if v in ds.data_vars), None)
    if runoff is None:
        raise ModelExecutionError(
            f"No VIC runoff variable ({', '.join(RUNOFF_VARS)}) in output; "
            f"available: {list(ds.data_vars)}"
        )
    total = runoff
    if include_baseflow:
        baseflow = next((ds[v] for v in BASEFLOW_VARS if v in ds.data_vars), None)
        if baseflow is not None:
            total = total + baseflow

    spatial_dims = [d for d in total.dims if d != 'time']
    if not spatial_dims:
        return total

    if cell_area is None and 'area' in ds.data_vars:
        cell_area = ds['area']
    spatial_shape = tuple(total.sizes[d] for d in spatial_dims)
    if cell_area is not None:
        values = np.asarray(getattr(cell_area, 'values', cell_area), dtype=float)
        if values.shape != spatial_shape:
            logger.warning(
                f"VIC cell-area shape {values.shape} does not match output grid "
                f"{spatial_shape}; using equal cell weights"
            )
            values = np.ones(spatial_shape)
    else:
        values = np.ones(spatial_shape)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    weights = xr.DataArray(values, dims=spatial_dims)

    valid = total.notnull()
    weight_sum = weights.where(valid, 0.0).sum(dim=spatial_dims)
    weighted = (total.fillna(0.0) * weights).sum(dim=spatial_dims)
    return weighted / weight_sum.where(weight_sum > 0)


class VICResultExtractor(ModelResultExtractor):
    """VIC-specific result extraction.

    Handles VIC model's output characteristics:
    - File formats: NetCDF (.nc)
    - Variable naming: OUT_RUNOFF, OUT_BASEFLOW, OUT_EVAP, etc.
    - Spatial aggregation: Grid cells to basin
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for VIC outputs."""
        return {
            'streamflow': [
                '*fluxes*.nc',
                'vic_output*.nc',
                '*runoff*.nc',
            ],
            'et': [
                '*fluxes*.nc',
                'vic_output*.nc',
            ],
            'snow': [
                '*fluxes*.nc',
                'vic_output*.nc',
            ],
            'soil_moisture': [
                '*fluxes*.nc',
                'vic_output*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get VIC variable names for different types.

        VIC output variables follow the OUT_* naming convention:
        - OUT_RUNOFF: Surface runoff [mm]
        - OUT_BASEFLOW: Baseflow [mm]
        - OUT_EVAP: Evaporation [mm]
        - OUT_SWE: Snow water equivalent [mm]
        """
        variable_mapping = {
            'streamflow': [
                'OUT_RUNOFF',
                'OUT_BASEFLOW',
                'RUNOFF',
                'BASEFLOW',
            ],
            'runoff': [
                'OUT_RUNOFF',
                'RUNOFF',
            ],
            'baseflow': [
                'OUT_BASEFLOW',
                'BASEFLOW',
            ],
            'et': [
                'OUT_EVAP',
                'OUT_EVAP_CANOP',
                'OUT_TRANSP_VEG',
                'EVAP',
            ],
            'snow': [
                'OUT_SWE',
                'OUT_SNOW_DEPTH',
                'SWE',
            ],
            'soil_moisture': [
                'OUT_SOIL_MOIST',
                'OUT_SOIL_WET',
                'SOIL_MOIST',
            ],
            'precipitation': [
                'OUT_PREC',
                'PREC',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from VIC output.

        Args:
            output_file: Path to VIC output file (NetCDF)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - catchment_area: Catchment area in m² for unit conversion
                - aggregate: Whether to aggregate grid cells (default: True)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        aggregate = kwargs.get('aggregate', True)
        catchment_area = kwargs.get('catchment_area')

        try:
            ds = xr.open_dataset(output_file)

            # Find the variable
            found_var = None
            for var_name in var_names:
                if var_name in ds.data_vars:
                    found_var = var_name
                    break

            if found_var is None:
                available = list(ds.data_vars)
                ds.close()
                raise ValueError(
                    f"No suitable variable found in {output_file}. "
                    f"Tried: {var_names}. Available: {available}"
                )

            data = ds[found_var]

            # Aggregate spatially if needed
            if aggregate and len(data.dims) > 1:
                # Sum over spatial dimensions (typical for runoff/baseflow)
                spatial_dims = [d for d in data.dims if d not in ['time']]
                if variable_type == 'streamflow':
                    data = basin_mean_vic_runoff(ds, kwargs.get('cell_area'))
                elif variable_type in ['runoff', 'baseflow']:
                    # Depths: area-weighted basin mean, never a sum over cells
                    cell_area = kwargs.get('cell_area')
                    if cell_area is None and 'area' in ds.data_vars:
                        cell_area = ds['area']
                    data = basin_mean_vic_runoff(
                        xr.Dataset({'OUT_RUNOFF': data}), cell_area,
                        include_baseflow=False,
                    )
                else:
                    # For other variables, take mean
                    data = data.mean(dim=spatial_dims)

            # Convert to pandas Series
            if 'time' in data.dims:
                series = data.to_series()
            else:
                # Single value case
                series = pd.Series([float(data.values)], index=[pd.Timestamp.now()])

            # Unit conversion for streamflow
            if variable_type == 'streamflow' and catchment_area:
                # VIC outputs in mm/timestep
                # Convert to m³/s: mm * area_m² / 1000 / seconds_per_day
                seconds_per_day = 86400
                series = series * catchment_area / 1000 / seconds_per_day

            ds.close()
            return series

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(
                f"Error reading VIC output file {output_file}: {str(e)}"
            ) from e

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract total streamflow (runoff + baseflow) from VIC output.

        Args:
            output_dir: Directory containing VIC outputs
            catchment_area: Catchment area in m² for conversion to m³/s
            **kwargs: Additional options

        Returns:
            Time series of streamflow
        """
        import xarray as xr

        # Find output file
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(
                f"No VIC output file found in {output_dir} matching {patterns}"
            )

        with xr.open_dataset(output_file) as ds:
            total_flow = basin_mean_vic_runoff(ds, kwargs.get('cell_area'))
            series = total_flow.to_series()

        # Unit conversion
        if catchment_area:
            # mm -> m³/s
            series = series * catchment_area / 1000 / 86400

        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion."""
        return variable_type in ['streamflow', 'runoff', 'baseflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow', 'runoff', 'baseflow', 'et', 'precipitation']:
            return 'mean'  # Area-weighted mean of per-cell depths
        else:
            return 'mean'  # Average for states like SWE, soil moisture

    def aggregate_to_basin(
        self,
        output_file: Path,
        variable_type: str,
        cell_areas: Optional[Dict] = None
    ) -> pd.Series:
        """
        Aggregate grid cell outputs to basin level.

        Args:
            output_file: Path to VIC output file
            variable_type: Type of variable
            cell_areas: Optional dict mapping cell indices to areas

        Returns:
            Basin-aggregated time series
        """
        import xarray as xr

        ds = xr.open_dataset(output_file)
        var_names = self.get_variable_names(variable_type)

        found_var = None
        for var in var_names:
            if var in ds.data_vars:
                found_var = var
                break

        if found_var is None:
            ds.close()
            raise ValueError(f"Variable type {variable_type} not found")

        data = ds[found_var]

        # Determine aggregation method
        method = self.get_spatial_aggregation_method(variable_type)
        spatial_dims = [d for d in data.dims if d not in ['time']]

        if method == 'sum':
            if cell_areas:
                # Area-weighted sum
                weights = xr.DataArray(
                    list(cell_areas.values()),
                    dims=['cell']
                )
                aggregated = (data * weights).sum(dim=spatial_dims)
            else:
                aggregated = data.sum(dim=spatial_dims)
        else:
            if cell_areas:
                # Area-weighted mean
                total_area = sum(cell_areas.values())
                weights = xr.DataArray(
                    [a / total_area for a in cell_areas.values()],
                    dims=['cell']
                )
                aggregated = (data * weights).sum(dim=spatial_dims)
            else:
                aggregated = data.mean(dim=spatial_dims)

        series = aggregated.to_series()
        ds.close()

        return series
