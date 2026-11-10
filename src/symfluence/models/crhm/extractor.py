# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Result Extractor.

Handles extraction of simulation results from CRHM model outputs.
CRHM outputs are in CSV format with standard variable names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.modeling.base import ModelResultExtractor


def basinflow_to_cms(volume: pd.Series) -> pd.Series:
    """CRHM basinflow (m3 per interval, end-labelled) -> m3/s, start-labelled.

    Shared by the calibration worker and the postprocessor. The interval is
    inferred from the first two timestamps (1 h when undeterminable). CRHM
    stamps each value at the *end* of the interval it covers, so labels are
    moved back one interval before any daily resampling -- otherwise the last
    hour of each day is attributed to the next day.
    """
    series = volume.astype(float).copy()
    series.index = pd.DatetimeIndex(series.index)
    interval_seconds = 3600.0
    if len(series) > 1:
        dt = (series.index[1] - series.index[0]).total_seconds()
        if dt > 0:
            interval_seconds = dt
    series = series / interval_seconds
    series.index = series.index - pd.Timedelta(seconds=interval_seconds)
    return series


class CRHMResultExtractor(ModelResultExtractor):
    """CRHM-specific result extraction.

    Handles CRHM model's output characteristics:
    - File formats: CSV text files
    - Variable naming: flow, SWE, soil_moist, etc.
    - Spatial aggregation: HRU-based to basin
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for CRHM outputs."""
        return {
            'streamflow': [
                '*output*.csv',
                '*flow*.csv',
                '*.csv',
            ],
            'swe': [
                '*output*.csv',
                '*snow*.csv',
                '*.csv',
            ],
            'et': [
                '*output*.csv',
                '*evap*.csv',
                '*.csv',
            ],
            'soil_moisture': [
                '*output*.csv',
                '*soil*.csv',
                '*.csv',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get CRHM variable names for different types.

        CRHM output variables:
        - flow: Streamflow at basin outlet [m3/s]
        - SWE: Snow water equivalent [mm]
        - soil_moist: Soil moisture content [mm]
        - evap: Evapotranspiration [mm]
        """
        variable_mapping = {
            'streamflow': [
                'flow',
                'Flow',
                'discharge',
                'Discharge',
                'Q',
                'flow_cms',
                'basinflow',
            ],
            'swe': [
                'SWE',
                'swe',
                'snowpack',
                'snow_water_equivalent',
            ],
            'et': [
                'evap',
                'Evap',
                'ET',
                'et',
                'evaporation',
                'hru_actet',
            ],
            'soil_moisture': [
                'soil_moist',
                'soil_moisture',
                'Soil_moist',
                'soil_rechr',
            ],
            'snowdepth': [
                'snowdepth',
                'snow_depth',
                'Snowdepth',
            ],
            'runoff': [
                'runoff',
                'Runoff',
                'hru_runoff',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from CRHM CSV output.

        Args:
            output_file: Path to CRHM output file (CSV)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - catchment_area: Catchment area in m2 for unit conversion
                - aggregate: Whether to aggregate HRUs (default: True)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        var_names = self.get_variable_names(variable_type)

        try:
            # CRHM (Fortran) output is Latin-1 encoded (unit headers contain the
            # degree symbol 0xba), so decode as latin-1 rather than UTF-8.
            df = pd.read_csv(output_file, parse_dates=True, index_col=0, encoding='latin-1')

            # Find the variable column
            found_col = None
            for var_name in var_names:
                if var_name in df.columns:
                    found_col = var_name
                    break
                # Also check for HRU-specific columns (e.g., flow(1), SWE(1))
                hru_cols = [c for c in df.columns if c.startswith(var_name)]
                if hru_cols:
                    found_col = hru_cols[0]
                    break

            if found_col is None:
                available = list(df.columns)
                raise ValueError(
                    f"No suitable variable found in {output_file}. "
                    f"Tried: {var_names}. Available: {available}"
                )

            series = df[found_col]

            # If multiple HRU columns exist and aggregation requested
            aggregate = kwargs.get('aggregate', True)
            if aggregate:
                base_name = found_col.split('(')[0] if '(' in found_col else found_col
                hru_cols = [c for c in df.columns if c.startswith(base_name)]
                if len(hru_cols) > 1:
                    if variable_type in ['streamflow', 'runoff']:
                        series = df[hru_cols].sum(axis=1)
                    else:
                        series = df[hru_cols].mean(axis=1)

            return series

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(
                f"Error reading CRHM output file {output_file}: {str(e)}"
            ) from e

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from CRHM output.

        Args:
            output_dir: Directory containing CRHM outputs
            catchment_area: Catchment area in m2 (unused, CRHM outputs m3/s)
            **kwargs: Additional options

        Returns:
            Time series of streamflow in m3/s
        """
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
                f"No CRHM output file found in {output_dir} matching {patterns}"
            )

        series = self.extract_variable(output_file, 'streamflow')

        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        CRHM typically outputs flow in m3/s, so no conversion needed
        for streamflow. SWE, soil moisture are in mm.
        """
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow', 'runoff']:
            return 'sum'  # Sum HRU contributions
        else:
            return 'mean'  # Average for states like SWE, soil moisture

    def aggregate_to_basin(
        self,
        output_file: Path,
        variable_type: str,
        cell_areas: Optional[Dict] = None
    ) -> pd.Series:
        """
        Aggregate HRU outputs to basin level.

        Args:
            output_file: Path to CRHM output file
            variable_type: Type of variable
            cell_areas: Optional dict mapping HRU indices to areas

        Returns:
            Basin-aggregated time series
        """
        # CRHM (Fortran) output is Latin-1 encoded (unit headers contain the
        # degree symbol 0xba), so decode as latin-1 rather than UTF-8.
        df = pd.read_csv(output_file, parse_dates=True, index_col=0, encoding='latin-1')
        var_names = self.get_variable_names(variable_type)

        # Find matching columns
        found_cols = []
        for var_name in var_names:
            cols = [c for c in df.columns if c.startswith(var_name)]
            if cols:
                found_cols = cols
                break

        if not found_cols:
            raise ValueError(f"Variable type {variable_type} not found")

        method = self.get_spatial_aggregation_method(variable_type)

        if method == 'sum':
            if cell_areas and len(found_cols) == len(cell_areas):
                # Area-weighted sum
                weights = list(cell_areas.values())
                total_area = sum(weights)
                aggregated = sum(
                    df[col] * w / total_area for col, w in zip(found_cols, weights)
                ) * total_area
            else:
                aggregated = df[found_cols].sum(axis=1)
        else:
            if cell_areas and len(found_cols) == len(cell_areas):
                weights = list(cell_areas.values())
                total_area = sum(weights)
                aggregated = sum(
                    df[col] * w / total_area for col, w in zip(found_cols, weights)
                )
            else:
                aggregated = df[found_cols].mean(axis=1)

        return aggregated
