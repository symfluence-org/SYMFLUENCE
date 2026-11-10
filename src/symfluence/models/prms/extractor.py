# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Result Extractor.

Handles extraction of simulation results from PRMS model outputs.
PRMS outputs are typically in statvar text or NetCDF format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor


class PRMSResultExtractor(ModelResultExtractor):
    """PRMS-specific result extraction.

    Handles PRMS model's output characteristics:
    - File formats: statvar text (.dat), CSV, or NetCDF (.nc)
    - Variable naming: seg_outflow, hru_actet, soil_moist, pkwater_equiv
    - HRU-based spatial structure
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for PRMS outputs."""
        return {
            'streamflow': [
                'statvar.dat',
                'statvar*.csv',
                'statvar*.nc',
                '*_statvar*',
            ],
            'et': [
                'statvar.dat',
                'statvar*.csv',
                'statvar*.nc',
            ],
            'soil_moisture': [
                'statvar.dat',
                'statvar*.csv',
                'statvar*.nc',
            ],
            'snow': [
                'statvar.dat',
                'statvar*.csv',
                'statvar*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get PRMS variable names for different types.

        PRMS output variables:
        - seg_outflow: Segment outflow (streamflow) [cfs or cms]
        - hru_actet: Actual ET [inches]
        - soil_moist: Soil moisture storage [inches]
        - pkwater_equiv: Snowpack water equivalent (SWE) [inches]
        """
        variable_mapping = {
            'streamflow': [
                'seg_outflow',
                'basin_cfs',
                'basin_cms',
                'streamflow_cfs',
            ],
            'et': [
                'hru_actet',
                'potet',
                'basin_actet',
                'hru_intcpevap',
            ],
            'soil_moisture': [
                'soil_moist',
                'soil_rechr',
                'ssres_stor',
                'basin_soil_moist',
            ],
            'snow': [
                'pkwater_equiv',
                'snowcov_area',
                'snowmelt',
                'basin_pweqv',
            ],
            'precipitation': [
                'hru_ppt',
                'hru_rain',
                'hru_snow',
                'basin_ppt',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from PRMS output.

        Args:
            output_file: Path to PRMS output file
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        var_names = self.get_variable_names(variable_type)

        if variable_type == 'streamflow':
            # Shared reader: correct column from header counts, cfs -> m3/s
            from .statvar import read_netcdf_streamflow, read_statvar_streamflow
            if output_file.suffix == '.nc':
                return read_netcdf_streamflow(output_file)
            return read_statvar_streamflow(output_file)

        if output_file.suffix == '.nc':
            return self._extract_from_netcdf(output_file, var_names)
        else:
            return self._extract_from_statvar(output_file, var_names)

    def _extract_from_netcdf(self, output_file: Path, var_names: List[str]) -> pd.Series:
        """Extract variable from NetCDF output."""
        import xarray as xr

        ds = xr.open_dataset(output_file)

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

        # Aggregate over HRU/segment dimensions
        spatial_dims = [d for d in data.dims if d not in ['time']]
        if spatial_dims:
            data = data.sum(dim=spatial_dims)

        series = data.to_series() if 'time' in data.dims else pd.Series(
            [float(data.values)], index=[pd.Timestamp.now()]
        )

        ds.close()
        return series

    def _extract_from_statvar(self, output_file: Path, var_names: List[str]) -> pd.Series:
        """Extract a variable from a statvar text file (mean over its elements)."""
        from .statvar import parse_statvar

        header, df = parse_statvar(output_file)
        counts = dict(header)
        for name in var_names:
            if name not in counts:
                continue
            if counts[name] == 1:
                return df[name].rename(name)
            cols = [f"{name}_{i + 1}" for i in range(counts[name])]
            return df[cols].mean(axis=1).rename(name)
        raise ModelExecutionError(
            f"None of {var_names} in {output_file}; header has {list(counts)}"
        )

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from PRMS output.

        basin_cfs / seg_outflow are converted from cfs to m3/s.

        Args:
            output_dir: Directory containing PRMS outputs
            catchment_area: Not used (streamflow already in cms)
            **kwargs: Additional options

        Returns:
            Time series of streamflow in m3/s
        """
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(
                f"No PRMS statvar output found in {output_dir}"
            )

        return self.extract_variable(output_file, 'streamflow')

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        PRMS seg_outflow is already in cms. Other variables may be in
        inches and require conversion.
        """
        return variable_type not in ['streamflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow']:
            return 'sum'
        elif variable_type in ['et', 'precipitation']:
            return 'mean'  # Average over HRUs
        else:
            return 'mean'  # Average for states like SWE, soil moisture
