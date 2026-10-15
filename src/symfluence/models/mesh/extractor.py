# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH Result Extractor.

Handles extraction of simulation results from MESH (Modélisation
Environmentale Communautaire - Surface and Hydrology) model outputs.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from symfluence.core.modeling.base import ModelResultExtractor


class MESHResultExtractor(ModelResultExtractor):
    """MESH-specific result extraction.

    Handles MESH's unique output characteristics:
    - File format: CSV files (MESH_output_streamflow.csv)
    - Variable naming: QOSIM* (simulated streamflow), ET*, SNOW*, etc.
    - Time format: Julian day (DAY) and YEAR columns
    - Spatial: Multiple subbasins (QOSIM1, QOSIM2, etc.)
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for MESH outputs."""
        return {
            'streamflow': [
                'MESH_spatial_streamflow.csv',
                'MESH_output_streamflow.csv',
                'MESH_output_streamflow_ts.csv',
                'forcing/MESH_input/MESH_output_streamflow.csv',
                'forcing/MESH_input/results/MESH_output_streamflow.csv',
                'results/MESH_output_streamflow.csv',
                # Lumped mode output - GRU water balance (hourly, has ROF)
                'GRU_water_balance.csv',
                'forcing/MESH_input/GRU_water_balance.csv',
            ],
            'runoff': [
                # For lumped mode, runoff is in GRU water balance
                'GRU_water_balance.csv',
                'forcing/MESH_input/GRU_water_balance.csv',
            ],
            'et': [
                'GRU_water_balance.csv',
                'forcing/MESH_input/GRU_water_balance.csv',
            ],
            'snow': [
                'GRU_water_balance.csv',
                'forcing/MESH_input/GRU_water_balance.csv',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get MESH variable names for different types."""
        variable_mapping = {
            # Simulated streamflow (routed) or runoff (lumped)
            'streamflow': ['QOSIM', 'ROF', 'RFF'],
            'runoff': ['ROF', 'RFF', 'RFFACC'],  # Total runoff from water balance
            'et': ['EVAP', 'ET', 'ETACC', 'EVAPOTRANSPIRATION'],
            'snow': ['SNO', 'SNOW', 'SWE', 'SNOWPACK'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from MESH CSV output.

        Args:
            output_file: Path to MESH CSV output
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - subbasin_index: Which subbasin to extract (default: 0 for outlet)
                - start_date: Simulation start date for GRU files (default: 2001-01-01)
                - timestep_hours: Timestep in hours for GRU files (default: 1)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found
        """
        if output_file.suffix != '.csv':
            raise ValueError(f"MESH extractor only supports CSV files, got {output_file.suffix}")

        try:
            # Check if this is a GRU_water_balance.csv file (different format)
            if 'GRU_water_balance' in output_file.name:
                return self._extract_from_gru_water_balance(
                    output_file, variable_type, **kwargs
                )

            # Check if this is Basin_average_water_balance.csv (lumped noroute mode)
            if 'Basin_average_water_balance' in output_file.name:
                return self._extract_from_basin_avg_water_balance(
                    output_file, variable_type, **kwargs
                )

            # Read standard MESH CSV file
            df = pd.read_csv(output_file, skipinitialspace=True)

            # Convert DAY and YEAR to datetime
            df['datetime'] = df.apply(self._julian_to_datetime, axis=1)

            # Find the variable column
            if variable_type == 'streamflow':
                return self._extract_streamflow(df, **kwargs)
            else:
                return self._extract_generic_variable(df, variable_type)

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Failed to extract {variable_type} from {output_file}: {e}") from e

    def _extract_from_gru_water_balance(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from GRU_water_balance.csv file.

        This file has hourly data with columns:
        IHOUR,IMIN,IDAY,IYEAR,PRE,EVAP,ROF,ROFO,ROFS,ROFB,SCAN,RCAN,SNO,...

        The timestamp columns are often malformed, so we reconstruct
        timestamps from the row index based on the simulation start date.

        Args:
            output_file: Path to GRU_water_balance.csv
            variable_type: Type of variable to extract
            **kwargs: Options including start_date, timestep_hours, aggregate

        Returns:
            Daily aggregated time series
        """
        # Read the GRU water balance file
        df = pd.read_csv(output_file, skipinitialspace=True)

        # Get simulation parameters
        start_date = kwargs.get('start_date', datetime(2001, 1, 1))
        timestep_hours = kwargs.get('timestep_hours', 1)
        aggregate = kwargs.get('aggregate', 'daily')

        # Reconstruct timestamps from row index
        n_rows = len(df)
        timestamps = [
            start_date + timedelta(hours=i * timestep_hours)
            for i in range(n_rows)
        ]
        df['datetime'] = timestamps

        # Map variable type to column name
        var_mapping = {
            'streamflow': 'ROF',  # Total runoff (mm/timestep)
            'runoff': 'ROF',
            'et': 'EVAP',
            'snow': 'SNO',
            'precipitation': 'PRE',
        }

        col_name = var_mapping.get(variable_type, variable_type.upper())

        # Find the column
        if col_name not in df.columns:
            # Try to find a similar column
            for alt_name in self.get_variable_names(variable_type):
                if alt_name in df.columns:
                    col_name = alt_name
                    break
            else:
                raise ValueError(
                    f"Column '{col_name}' not found in GRU_water_balance.csv. "
                    f"Available: {list(df.columns)}"
                )

        # Create time series
        series = pd.Series(
            df[col_name].values,
            index=pd.DatetimeIndex(df['datetime']),
            name=col_name
        )

        # Aggregate to daily if requested (default for calibration)
        if aggregate == 'daily':
            # Sum values within each day (since values are mm/timestep)
            daily_series = series.resample('D').sum()
            daily_series.name = col_name
            return daily_series

        return series

    def _extract_from_basin_avg_water_balance(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from Basin_average_water_balance.csv file.

        This file has daily data with columns:
        YEAR, JDAY, PRECACC, ETACC, RFFACC, OVRFLWACC, LATFLWACC, LKGACC, ...

        Args:
            output_file: Path to Basin_average_water_balance.csv
            variable_type: Type of variable to extract
            **kwargs: Options

        Returns:
            Daily time series
        """
        # Read the Basin average water balance file
        df = pd.read_csv(output_file, skipinitialspace=True)

        if df.empty or len(df) <= 1:
            raise ValueError("Basin_average_water_balance.csv has no data rows")

        # Convert YEAR and JDAY to datetime
        df['datetime'] = df.apply(self._julian_to_datetime, axis=1)

        # Map variable type to column name
        var_mapping = {
            'et': ['ET', 'ETACC'],
            'snow': ['SNO'],
            'precipitation': ['PREC', 'PRECACC'],
        }

        # For runoff/streamflow: compute total discharge from the basin water
        # balance. The correct total depends on the run mode, distinguished by
        # whether the lower-zone baseflow (LKG) is active:
        #
        #  * Routing mode (runrte / run_def, wf_lzs active, LKG != 0): the
        #    basin-average RFF ALREADY reports total runoff — surface +
        #    interflow + lower-zone baseflow — i.e. RFF == OVRFLW + LATFLW + LKG
        #    (verified against the MESH Basin_average_water_balance to < 1e-5
        #    mm/day). RFF alone is therefore the total streamflow; adding LKG
        #    again double-counts baseflow.
        #  * noroute mode (LKG == 0): RFF carries only the surface term
        #    (OVRFLW + LATFLW) while the deep soil drainage leaves via DRAINSOL,
        #    which RFF excludes. Add DRAINSOL as the baseflow contribution.
        if variable_type in ('streamflow', 'runoff'):
            df.columns = df.columns.str.strip()
            if 'RFF' in df.columns:
                total = df['RFF'].copy()
                lkg_active = 'LKG' in df.columns and df['LKG'].abs().sum() > 0
                if not lkg_active and 'DRAINSOL' in df.columns:
                    # noroute: RFF is surface-only, add deep drainage as baseflow
                    total = total + df['DRAINSOL']
                return pd.Series(
                    total.values,
                    index=pd.DatetimeIndex(df['datetime']),
                    name='total_runoff'
                )
            elif 'RFFACC' in df.columns:
                # RFFACC is typically cumulative; difference it to get daily runoff
                series = pd.Series(
                    df['RFFACC'].astype(float).values,
                    index=pd.DatetimeIndex(df['datetime']),
                    name='RFFACC'
                )
                diff = series.diff()
                if not diff.empty:
                    diff.iloc[0] = series.iloc[0]
                return diff
            raise ValueError("No RFF or RFFACC column found")

        col_candidates = var_mapping.get(variable_type, [variable_type.upper()])

        # Find the column
        col_name = None
        for candidate in col_candidates:
            if candidate in df.columns:
                col_name = candidate
                break

        if col_name is None:
            raise ValueError(
                f"No column for '{variable_type}' found in Basin_average_water_balance.csv. "
                f"Tried: {col_candidates}. Available: {list(df.columns)}"
            )

        # Create time series
        series = pd.Series(
            df[col_name].values,
            index=pd.DatetimeIndex(df['datetime']),
            name=col_name
        )

        return series

    def _julian_to_datetime(self, row) -> datetime:
        """Convert Julian day and year to datetime.

        Args:
            row: DataFrame row with DAY/JDAY and YEAR columns

        Returns:
            datetime object
        """
        year = int(row['YEAR'])
        # Handle both DAY and JDAY column names
        if 'DAY' in row.index:
            day = int(row['DAY'])
        elif 'JDAY' in row.index:
            day = int(row['JDAY'])
        else:
            raise ValueError("Neither DAY nor JDAY column found in row")
        return datetime(year, 1, 1) + timedelta(days=day - 1)

    def _extract_streamflow(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Extract streamflow from MESH output.

        Args:
            df: DataFrame with MESH output
            **kwargs: Optional subbasin_index

        Returns:
            Streamflow time series
        """
        # Find QOSIM columns (simulated streamflow) - for routed output
        streamflow_cols = [col for col in df.columns if col.startswith('QOSIM')]

        if streamflow_cols:
            # Select subbasin (default to first - usually the outlet)
            subbasin_index = kwargs.get('subbasin_index', 0)
            if subbasin_index >= len(streamflow_cols):
                subbasin_index = 0

            selected_col = streamflow_cols[subbasin_index]

            # Create time series
            return pd.Series(
                df[selected_col].values,
                index=df['datetime'],
                name=selected_col
            )

        # For lumped mode, look for RFF (runoff) from basin average water balance
        # Strip whitespace from column names (MESH adds extra spaces)
        df.columns = df.columns.str.strip()

        if 'RFF' in df.columns:
            return pd.Series(
                df['RFF'].values,
                index=df['datetime'],
                name='RFF'
            )

        # Fallback: look for any runoff-like columns
        runoff_cols = [col for col in df.columns
                       if col not in ['YEAR', 'DAY', 'JDAY', 'datetime']
                       and 'ACC' not in col]  # Skip accumulated columns

        if runoff_cols:
            # For lumped mode, use the first runoff column
            subbasin_index = kwargs.get('subbasin_index', 0)
            if subbasin_index >= len(runoff_cols):
                subbasin_index = 0

            selected_col = runoff_cols[subbasin_index]

            return pd.Series(
                df[selected_col].values,
                index=df['datetime'],
                name='RFF'  # Total runoff
            )

        raise ValueError("No simulated streamflow (QOSIM*) or runoff (RFF) columns found in MESH output")

    def _extract_generic_variable(self, df: pd.DataFrame, variable_type: str) -> pd.Series:
        """Extract generic variable from MESH output.

        Args:
            df: DataFrame with MESH output
            variable_type: Type of variable

        Returns:
            Variable time series
        """
        var_names = self.get_variable_names(variable_type)

        # Find matching column
        for var_name in var_names:
            matching_cols = [col for col in df.columns if var_name in col.upper()]
            if matching_cols:
                selected_col = matching_cols[0]
                return pd.Series(
                    df[selected_col].values,
                    index=df['datetime'],
                    name=selected_col
                )

        raise ValueError(
            f"No suitable variable found for '{variable_type}'. "
            f"Tried: {var_names}"
        )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """MESH outputs are typically in standard units (m³/s for streamflow)."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """MESH outputs by subbasin/GRU."""
        return 'selection'  # Select specific subbasin/GRU

    def extract_water_balance_from_echo_print(self, echo_print_path: Path) -> Dict[str, float]:
        """Extract water balance totals from MESH echo_print.txt file.

        Useful for lumped mode validation when daily output is not available.

        Args:
            echo_print_path: Path to MESH_output_echo_print.txt

        Returns:
            Dictionary with water balance components (in mm):
            - precipitation: Total precipitation
            - evapotranspiration: Total ET
            - runoff: Total runoff
            - storage_change: Change in storage
        """

        if not echo_print_path.exists():
            raise ValueError(f"Echo print file not found: {echo_print_path}")

        with open(echo_print_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse water balance section
        balance = {}

        # Total precipitation
        match = re.search(r'Total precipitation\s+=\s+([\d.]+)', content)
        if match:
            balance['precipitation'] = float(match.group(1))

        # Total evapotranspiration
        match = re.search(r'Total evapotranspiration\s+=\s+([\d.]+)', content)
        if match:
            balance['evapotranspiration'] = float(match.group(1))

        # Total runoff
        match = re.search(r'Total runoff\s+=\s+([\d.]+)', content)
        if match:
            balance['runoff'] = float(match.group(1))

        # Change in storage
        match = re.search(r'Change in storage\s+=\s+([\d.]+)', content)
        if match:
            balance['storage_change'] = float(match.group(1))

        return balance
