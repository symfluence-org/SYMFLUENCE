# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS model postprocessor.

Handles extraction and processing of PRMS model simulation results.
Uses StandardModelPostProcessor for reduced boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('PRMS')
class PRMSPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the PRMS model.

    PRMS outputs streamflow as seg_outflow in the statvar output file.
    Units are cubic meters per second (cms). The seg_outflow variable
    represents segment outflow at the basin outlet.

    Attributes:
        model_name: "PRMS"
        output_file_pattern: "statvar*"
        streamflow_variable: "seg_outflow"
        streamflow_unit: "cms"
    """

    model_name = "PRMS"

    output_file_pattern = "statvar*"

    streamflow_variable = "seg_outflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "PRMS"

    def _setup_model_specific_paths(self) -> None:
        """Set up PRMS-specific paths."""
        self.prms_output_dir = self.get_simulation_dir('PRMS')
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """PRMS outputs to standard simulation directory."""
        return self.get_simulation_dir('PRMS')

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from PRMS statvar output.

        PRMS statvar files contain seg_outflow in cms (m3/s).
        Supports both CSV and space-delimited formats.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from PRMS statvar output")

        output_dir = self._get_output_dir()

        # Find statvar file (check both output dir and settings dir)
        output_file = None
        search_dirs = [output_dir, self.project_dir / "settings" / "PRMS"]
        for search_dir in search_dirs:
            for pattern in ['statvar.dat', 'statvar*.csv', 'statvar*.nc']:
                matches = list(search_dir.glob(pattern))
                if matches:
                    output_file = matches[0]
                    break
            if output_file:
                break

        if output_file is None:
            self.logger.error("PRMS statvar output not found")
            return None

        try:

            if output_file.suffix == '.nc':
                return self._extract_from_netcdf(output_file)

            # Parse statvar text file
            streamflow = self._parse_statvar_file(output_file)

            if streamflow is None or streamflow.empty:
                self.logger.error("No streamflow data found in PRMS statvar output")
                return None

            # Streamflow is already in cms, no conversion needed

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='PRMS_discharge_cms'
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Error extracting PRMS streamflow: {str(e)}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_statvar_file(self, statvar_path: Path):
        """Parse PRMS statvar.dat file and extract streamflow.

        PRMS statvar format:
            Line 1: number of output variables (integer)
            Lines 2..N+1: variable_name count (e.g. ``basin_cfs 1``)
            Lines N+2..: step year month day hour min sec val1 val2 ...
        """
        import pandas as pd

        try:
            lines = statvar_path.read_text().strip().split('\n')
            if not lines:
                return None

            # Parse header: first line is variable count
            try:
                num_vars = int(lines[0].strip())
            except ValueError:
                num_vars = 0

            # Read variable names from header
            var_names = []
            if num_vars > 0:
                for i in range(1, min(num_vars + 1, len(lines))):
                    parts = lines[i].strip().split()
                    if parts:
                        var_names.append(parts[0])
                data_start = num_vars + 1
            else:
                data_start = 0

            # Find the streamflow variable column index
            flow_var = None
            flow_col_offset = 0
            for var in ['basin_cfs', 'seg_outflow', 'streamflow']:
                if var in var_names:
                    flow_var = var
                    flow_col_offset = var_names.index(var)
                    break
            if flow_var is None and var_names:
                flow_var = var_names[0]
                flow_col_offset = 0

            # Data columns: step year month day hour min sec var1 var2 ...
            # So first variable is at index 7
            flow_col = 7 + flow_col_offset

            dates = []
            values = []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) > flow_col:
                    try:
                        year = int(parts[1])
                        month = int(parts[2])
                        day = int(parts[3])
                        val = float(parts[flow_col])
                        dates.append(pd.Timestamp(year=year, month=month, day=day))
                        # basin_cfs is in cubic feet per second; convert to cms
                        if flow_var == 'basin_cfs':
                            val *= 0.0283168
                        values.append(val)
                    except (ValueError, IndexError):
                        continue

            if dates:
                return pd.Series(values, index=dates, name='PRMS_discharge_cms')

            return None

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Error parsing statvar file: {e}", exc_info=True)
            return None

    def _extract_from_netcdf(self, nc_path: Path):
        """Extract streamflow from NetCDF statvar output."""
        import xarray as xr

        ds = xr.open_dataset(nc_path)

        for var in ['seg_outflow', 'streamflow', 'basin_cfs']:
            if var in ds.data_vars:
                flow = ds[var]
                if 'nsegment' in flow.dims:
                    flow = flow.isel(nsegment=-1)
                series = flow.to_series()
                ds.close()
                return series

        ds.close()
        return None
