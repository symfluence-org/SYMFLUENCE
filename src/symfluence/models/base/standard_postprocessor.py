# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Standard Model Postprocessor

Provides a simplified base class for models with standard streamflow extraction patterns.
Most models can inherit from this and only define configuration attributes, reducing
boilerplate from 50-150 lines to 10-20 lines.

Usage:
    @ModelRegistry.register_postprocessor('MYMODEL')
    class MyModelPostprocessor(StandardModelPostprocessor):
        model_name = "MYMODEL"
        output_file_pattern = "{domain}_{experiment}_output.nc"
        streamflow_variable = "discharge"
        streamflow_unit = "mm_per_day"  # or "cms"
        netcdf_selections = {"hru": 0}  # Optional dimension selections
"""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import xarray as xr

from .base_postprocessor import BaseModelPostProcessor


class StandardModelPostprocessor(BaseModelPostProcessor):
    """
    Simplified postprocessor for models with standard extraction patterns.

    Subclasses only need to define class attributes for configuration.
    Override methods only when custom logic is needed.

    Class Attributes:
        model_name: Name of the model (e.g., "FUSE", "SUMMA")
        output_file_pattern: Pattern for output file name, supports:
            - {domain}: domain_name
            - {experiment}: experiment_id
            - {start_date}: EXPERIMENT_TIME_START date portion
        streamflow_variable: Name of streamflow variable in output file
        streamflow_unit: Unit of streamflow - "mm_per_day" or "cms"
        netcdf_selections: Dict of dimension selections for xr.isel()
        text_file_separator: Separator for text files (default: ",")
        text_file_skiprows: Rows to skip in text files (default: 0)
        text_file_date_column: Name of date column (default: "DATE")
        text_file_flow_column: Name or config key for flow column
        output_dir_override: Override for output directory (default: None, uses sim_dir)
        use_routing_output: Whether to read from mizuRoute output (default: False)

        # Dynamic outlet selection (Phase 1.1 extension)
        outlet_selection_method: str - "config", "highest_discharge", or "pattern"
        outlet_column_pattern: str - Regex pattern for column matching (e.g., r"QOSIM\\d+")

        # Custom date parsing
        date_parser_type: str - "standard" or "julian" (DAY+YEAR columns)

        # Multi-file aggregation (for NGEN-like models)
        output_file_glob: str - Glob pattern for multiple files (e.g., "nex-*_output.csv")
        aggregation_method: str - "sum", "mean", or "concat"

    Example:
        >>> class FUSEPostprocessor(StandardModelPostprocessor):
        ...     model_name = "FUSE"
        ...     output_file_pattern = "{domain}_{experiment}_runs_best.nc"
        ...     streamflow_variable = "q_routed"
        ...     streamflow_unit = "mm_per_day"
        ...     netcdf_selections = {"param_set": 0, "latitude": 0, "longitude": 0}
    """

    # Required: subclass must define
    model_name: str = None

    # NetCDF configuration
    output_file_pattern: str = "{domain}_{experiment}_output.nc"
    streamflow_variable: str = "discharge"
    streamflow_unit: str = "cms"  # "mm_per_day" or "cms"
    netcdf_selections: Dict[str, Any] = {}

    # Text file configuration (for models that output CSV/TSV)
    text_file_separator: str = ","
    text_file_skiprows: int = 0
    text_file_date_column: str = "DATE"
    text_file_flow_column: str = None  # Column name or None to use config

    # Output directory override
    output_dir_override: str = None  # e.g., "mizuRoute" for routing output

    # Routing integration
    use_routing_output: bool = False
    routing_variable: str = "IRFroutedRunoff"
    routing_file_pattern: str = "{experiment}.h.{start_date}-03600.nc"

    # Resampling (e.g., hourly to daily)
    resample_frequency: str = None  # e.g., "D" for daily

    # Dynamic outlet selection (Phase 1.1 extension)
    outlet_selection_method: str = "config"  # "config", "highest_discharge", "pattern"
    outlet_column_pattern: str = None  # Regex pattern for column matching

    # Custom date parsing
    date_parser_type: str = "standard"  # "standard" or "julian" (DAY+YEAR columns)

    # Multi-file aggregation (for NGEN-like models)
    output_file_glob: str = None  # Glob pattern for multiple files
    aggregation_method: str = "sum"  # "sum", "mean", or "concat"

    def _get_model_name(self) -> str:
        """Return the model name from class attribute."""
        if self.model_name is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'model_name' class attribute"
            )
        return self.model_name

    def _get_output_dir(self) -> Path:
        """Get the output directory for streamflow extraction."""
        if self.output_dir_override:
            return self.project_dir / 'simulations' / self.experiment_id / self.output_dir_override
        if self.use_routing_output:
            return self.project_dir / 'simulations' / self.experiment_id / 'mizuRoute'
        return self.sim_dir

    def _format_pattern(self, pattern: str) -> str:
        """Format a file pattern with available substitutions."""
        start_time = self.time_start or ''
        start_date = start_time.split()[0] if start_time else ''

        return pattern.format(
            domain=self.domain_name,
            experiment=self.experiment_id,
            start_date=start_date,
            model=self.model_name.lower()
        )

    def _get_output_file(self) -> Path:
        """Get the path to the output file."""
        output_dir = self._get_output_dir()

        if self.use_routing_output:
            pattern = self.routing_file_pattern
        else:
            pattern = self.output_file_pattern

        filename = self._format_pattern(pattern)
        return output_dir / filename

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow using standard patterns.

        Automatically handles:
        - NetCDF files with configurable selections
        - Text files with configurable parsing
        - Multi-file aggregation (when output_file_glob is set)
        - Unit conversion (mm/day to cms)
        - Resampling (e.g., hourly to daily)

        Override this method only for complex extraction logic.

        Returns:
            Path to saved results CSV, or None if extraction fails
        """
        try:
            self.logger.info(f"Extracting {self.model_name} streamflow results")

            # Check if multi-file extraction is configured
            if self.output_file_glob:
                streamflow = self._extract_from_multiple_files()
            else:
                # Single file extraction
                output_file = self._get_output_file()

                if not output_file.exists():
                    self.logger.error(f"{self.model_name} output file not found: {output_file}")
                    return None

                # Determine file type and extract accordingly
                suffix = output_file.suffix.lower()

                if suffix in ('.nc', '.nc4', '.netcdf'):
                    streamflow = self._extract_from_netcdf(output_file)
                elif suffix in ('.csv', '.txt', '.tsv'):
                    streamflow = self._extract_from_text(output_file)
                else:
                    self.logger.error(f"Unsupported file format: {suffix}")
                    return None

            if streamflow is None:
                return None

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            # Apply unit conversion if needed
            if self.streamflow_unit == "mm_per_day":
                streamflow = self.convert_mm_per_day_to_cms(streamflow)

            # Save to standard results format
            return self.save_streamflow_to_results(streamflow)

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting {self.model_name} streamflow: {str(e)}")
            raise

    def _extract_from_netcdf(self, file_path: Path) -> Optional[pd.Series]:
        """
        Extract streamflow from NetCDF file.

        Uses class attributes for variable name and dimension selections.
        """
        variable = self.routing_variable if self.use_routing_output else self.streamflow_variable
        selections = self._get_netcdf_selections()

        try:
            return self.read_netcdf_streamflow(file_path, variable, **selections)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error reading NetCDF: {e}")
            return None

    def _get_netcdf_selections(self) -> Dict[str, Any]:
        """
        Get NetCDF dimension selections.

        Override this method to add dynamic selections based on config.
        """
        selections = dict(self.netcdf_selections)  # Copy class attribute

        # Handle routing output reach selection
        if self.use_routing_output:
            sim_reach_id = self._get_config_value(lambda: self.config.evaluation.streamflow.sim_reach_id)
            if sim_reach_id:
                # This will be handled specially in extract_streamflow for routing
                pass

        return selections

    def _extract_from_text(self, file_path: Path) -> Optional[pd.Series]:
        """
        Extract streamflow from text/CSV file.

        Uses class attributes for separator, skiprows, column names, and date parsing.
        Supports both standard datetime columns and Julian day format (DAY+YEAR).
        """
        try:
            df = pd.read_csv(
                file_path,
                sep=self.text_file_separator,
                skiprows=self.text_file_skiprows,
                skipinitialspace=True  # Handle leading spaces in columns
            )

            # Handle date parsing based on date_parser_type
            if self.date_parser_type == "julian":
                # Parse Julian day format (DAY, YEAR columns)
                df = self._parse_julian_date(df)
            elif self.text_file_date_column in df.columns:
                # Standard datetime column
                df[self.text_file_date_column] = pd.to_datetime(df[self.text_file_date_column])
                df.set_index(self.text_file_date_column, inplace=True)

            # Get flow column
            flow_column = self._get_flow_column(df)
            if flow_column is None or flow_column not in df.columns:
                self.logger.error(f"Flow column '{flow_column}' not found in {file_path}")
                self.logger.debug(f"Available columns: {df.columns.tolist()}")
                return None

            return df[flow_column]

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error reading text file: {e}")
            return None

    def _get_flow_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Get the flow column name.

        Uses outlet_selection_method to determine how to select the column:
        - "config": Use config value (default behavior)
        - "highest_discharge": Select column with highest mean discharge
        - "pattern": Select first column matching outlet_column_pattern regex
        """
        # First try the configured column name
        flow_column = None
        if self.text_file_flow_column:
            # Check if it's a config key or direct column name
            if self.text_file_flow_column.startswith('config:'):
                config_key = self.text_file_flow_column[7:]  # Remove 'config:' prefix
                flow_column = str(self._get_config_value(lambda: None, dict_key=config_key))
            else:
                flow_column = self.text_file_flow_column

            # Check if column exists in dataframe
            if flow_column and flow_column in df.columns:
                return flow_column

        # If config column not found, use fallback based on outlet_selection_method
        if self.outlet_selection_method == "highest_discharge":
            return self._select_outlet_by_highest_discharge(df)
        elif self.outlet_selection_method == "pattern":
            return self._select_columns_by_pattern(df)

        return flow_column

    def _select_outlet_by_highest_discharge(self, df: pd.DataFrame) -> Optional[str]:
        """
        Select column with highest mean discharge (HYPE pattern).

        This is useful when the configured outlet ID is not found in the output,
        as the column with highest mean flow typically represents the watershed outlet.

        Args:
            df: DataFrame with potential discharge columns

        Returns:
            Column name with highest mean discharge, or None if no numeric columns
        """
        # Filter to only numeric columns (exclude date columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            self.logger.warning("No numeric columns found for outlet selection")
            return None

        # Find column with highest mean
        col_means = df[numeric_cols].mean()
        outlet_col = str(col_means.idxmax())

        self.logger.info(
            f"Auto-selected outlet column '{outlet_col}' "
            f"(highest mean flow: {col_means.max():.2f})"
        )
        return outlet_col

    def _select_columns_by_pattern(self, df: pd.DataFrame) -> Optional[str]:
        """
        Select first column matching outlet_column_pattern regex.

        Args:
            df: DataFrame with columns to search

        Returns:
            First matching column name, or None if no match
        """
        if not self.outlet_column_pattern:
            self.logger.warning("outlet_column_pattern not set for pattern selection")
            return None

        pattern = re.compile(self.outlet_column_pattern)
        matching_cols = [col for col in df.columns if pattern.match(str(col))]

        if not matching_cols:
            self.logger.warning(
                f"No columns matching pattern '{self.outlet_column_pattern}' found. "
                f"Available columns: {df.columns.tolist()}"
            )
            return None

        # Return first match (could be extended to select by other criteria)
        selected_col = matching_cols[0]
        self.logger.info(
            f"Selected column '{selected_col}' matching pattern '{self.outlet_column_pattern}'"
        )
        return selected_col

    def _parse_julian_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse Julian day format (DAY, YEAR columns) to datetime index.

        MESH and some other models output dates as separate DAY (day of year)
        and YEAR columns instead of a single datetime column.

        Args:
            df: DataFrame with DAY and YEAR columns

        Returns:
            DataFrame with datetime index set
        """
        if 'DAY' not in df.columns or 'YEAR' not in df.columns:
            self.logger.warning(
                "Julian date parsing requested but DAY/YEAR columns not found. "
                f"Available columns: {df.columns.tolist()}"
            )
            return df

        def julian_to_datetime(row):
            return datetime(int(row['YEAR']), 1, 1) + timedelta(days=int(row['DAY']) - 1)

        df['datetime'] = df.apply(julian_to_datetime, axis=1)
        df.set_index('datetime', inplace=True)

        # Drop the original DAY/YEAR columns to clean up
        df = df.drop(columns=['DAY', 'YEAR'], errors='ignore')

        return df

    def _extract_from_multiple_files(self) -> Optional[pd.Series]:
        """
        Aggregate streamflow from multiple output files (NGEN pattern).

        Uses output_file_glob to find files and aggregation_method to combine them.

        Returns:
            Aggregated streamflow series, or None if extraction fails
        """
        if not self.output_file_glob:
            self.logger.error("output_file_glob not set for multi-file extraction")
            return None

        output_dir = self._get_output_dir()
        output_files = list(output_dir.glob(self.output_file_glob))

        if not output_files:
            self.logger.error(
                f"No files matching '{self.output_file_glob}' found in {output_dir}"
            )
            return None

        self.logger.info(f"Found {len(output_files)} files matching '{self.output_file_glob}'")

        all_streamflow: List[pd.DataFrame] = []

        for file_path in output_files:
            try:
                file_id = file_path.stem.replace('_output', '')
                df = self._read_single_output_file(file_path)

                if df is not None and 'streamflow' in df.columns:
                    df['file_id'] = file_id
                    all_streamflow.append(df)

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Error processing {file_path}: {e}")
                continue

        if not all_streamflow:
            self.logger.error("No streamflow data extracted from any files")
            return None

        # Combine all data
        combined = pd.concat(all_streamflow, ignore_index=True)

        # Aggregate based on method
        if self.aggregation_method == "sum":
            aggregated = combined.groupby('datetime')['streamflow'].sum()
        elif self.aggregation_method == "mean":
            aggregated = combined.groupby('datetime')['streamflow'].mean()
        else:  # concat - return first file's data (or could be extended)
            aggregated = all_streamflow[0].set_index('datetime')['streamflow']

        self.logger.info(
            f"Aggregated {len(output_files)} files using '{self.aggregation_method}' method"
        )
        return aggregated

    def _read_single_output_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read a single output file for multi-file aggregation.

        Override this method for custom file reading logic (e.g., headerless CSVs).

        Args:
            file_path: Path to the output file

        Returns:
            DataFrame with 'datetime' and 'streamflow' columns, or None if failed
        """
        try:
            df = pd.read_csv(file_path, sep=self.text_file_separator)

            # Try to find time column
            time_col = None
            for col_name in ['time', 'Time', 'datetime', 'DATE', 'date']:
                if col_name in df.columns:
                    time_col = col_name
                    break

            # Try to find flow column
            flow_col = None
            for col_name in ['flow', 'Flow', 'Q_OUT', 'streamflow', 'discharge']:
                if col_name in df.columns:
                    flow_col = col_name
                    break

            if time_col is None or flow_col is None:
                self.logger.warning(
                    f"Could not find time/flow columns in {file_path}. "
                    f"Columns: {df.columns.tolist()}"
                )
                return None

            # Create standardized output
            result = pd.DataFrame({
                'datetime': pd.to_datetime(df[time_col]),
                'streamflow': df[flow_col]
            })

            return result

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error reading {file_path}: {e}")
            return None


class RoutedModelPostprocessor(StandardModelPostprocessor):
    """
    Specialized postprocessor for models that use mizuRoute routing.

    Handles the common pattern of reading from mizuRoute output files
    and selecting the outlet reach.

    Example:
        >>> class SUMMAPostprocessor(RoutedModelPostprocessor):
        ...     model_name = "SUMMA"
    """

    use_routing_output: bool = True
    routing_variable: str = "IRFroutedRunoff"
    routing_file_pattern: str = "{experiment}.h.{start_date}-03600.nc"
    resample_frequency: str = "D"  # Hourly to daily for routing output
    streamflow_unit: str = "cms"   # Routing output is already in cms

    def _setup_model_specific_paths(self) -> None:
        """Set up routing-specific paths."""
        self.mizuroute_dir = self.project_dir / 'simulations' / self.experiment_id / 'mizuRoute'

    def _extract_from_netcdf(self, file_path: Path) -> Optional[pd.Series]:
        """
        Extract routed streamflow from mizuRoute output.

        Handles reach selection based on SIM_REACH_ID config.
        """
        try:
            ds = xr.open_dataset(file_path, engine='netcdf4')

            # Get reach selection
            sim_reach_id = self._get_config_value(lambda: self.config.evaluation.sim_reach_id)

            if sim_reach_id is not None:
                sim_reach_id = int(sim_reach_id)
                if 'reachID' in ds:
                    segment_mask = ds['reachID'].values == sim_reach_id
                    if segment_mask.any():
                        idx = int(segment_mask.argmax())
                        ds_selected = ds.isel(seg=idx)
                    else:
                        self.logger.warning(
                            f"SIM_REACH_ID={sim_reach_id} not found in reachID; "
                            f"falling back to outlet (last segment)"
                        )
                        ds_selected = ds.isel(seg=-1)
                else:
                    ds_selected = ds.isel(seg=-1)
            else:
                ds_selected = ds.isel(seg=-1)

            # Extract routing variable
            streamflow = cast(pd.Series, ds_selected[self.routing_variable].to_pandas())

            # Round index to hour for proper resampling
            if hasattr(streamflow.index, 'round'):
                streamflow.index = streamflow.index.round(freq='h')

            ds.close()
            return streamflow

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error reading routing output: {e}")
            return None
