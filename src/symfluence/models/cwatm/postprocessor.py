# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM model postprocessor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('CWATM')
class CWatMPostProcessor(StandardModelPostProcessor):
    """Postprocessor for CWatM.

    CWatM outputs discharge in NetCDF or TSS format.
    The ``discharge`` variable is already in m3/s.
    """

    model_name = "CWATM"
    output_file_pattern = "discharge_daily*"
    streamflow_variable = "discharge"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "CWATM"

    def _setup_model_specific_paths(self) -> None:
        self.cwatm_output_dir = (
            self.get_simulation_dir('CWATM')
        )
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        return self.get_simulation_dir('CWATM')

    def extract_streamflow(self) -> Optional[Path]:
        import pandas as pd
        import xarray as xr

        self.logger.info("Extracting streamflow from CWatM outputs")
        output_dir = self._get_output_dir()

        # Try TSS file first (time series at gauge)
        for tss_pattern in ['discharge_daily.tss', 'discharge*.tss']:
            matches = list(output_dir.glob(tss_pattern))
            if matches:
                try:
                    lines = matches[0].read_text().strip().split('\n')
                    data_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts and parts[0].replace('.', '').replace('-', '').isdigit():
                            data_lines.append(float(parts[-1]))
                    if data_lines:
                        streamflow = pd.Series(data_lines, name='discharge')
                        streamflow.index.name = 'datetime'
                        return self.save_streamflow_to_results(
                            streamflow, model_column_name='CWATM_discharge_cms'
                        )
                except (OSError, ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to read TSS: {e}")

        # Fall back to NetCDF
        nc_file = None
        for pattern in ['discharge_daily.nc', 'discharge*.nc', '*discharge*.nc']:
            matches = list(output_dir.glob(pattern))
            if matches:
                nc_file = matches[0]
                break

        if nc_file is None:
            self.logger.error(f"CWatM discharge output not found in {output_dir}")
            return None

        try:
            ds = xr.open_dataset(nc_file)
            q_var = None
            for var in ['discharge', 'Qsim', 'Q']:
                if var in ds.data_vars:
                    q_var = ds[var]
                    break
            if q_var is None:
                self.logger.error(f"No discharge variable in {nc_file}")
                ds.close()
                return None

            spatial_dims = [d for d in q_var.dims if d not in ['time']]
            if spatial_dims:
                streamflow = q_var.max(dim=spatial_dims)
            else:
                streamflow = q_var

            streamflow = streamflow.to_series()
            ds.close()

            if not isinstance(streamflow.index, pd.DatetimeIndex):
                streamflow.index = pd.to_datetime(streamflow.index)
            streamflow.index.name = 'datetime'

            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow, model_column_name='CWATM_discharge_cms'
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            import traceback
            self.logger.error(f"Error extracting CWatM streamflow: {e}")
            self.logger.debug(traceback.format_exc())
            return None
