# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB model postprocessor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('PCRGLOBWB')
class PCRGLOBWBPostProcessor(StandardModelPostProcessor):
    """Postprocessor for the PCR-GLOBWB 2.0 model.

    PCR-GLOBWB outputs discharge in NetCDF format with naming convention
    ``{var}_{aggregation}_output.nc``.  The ``discharge`` variable is
    already in m3/s so no unit conversion is required.
    """

    model_name = "PCRGLOBWB"
    output_file_pattern = "discharge_*_output.nc"
    streamflow_variable = "discharge"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "PCRGLOBWB"

    def _setup_model_specific_paths(self) -> None:
        self.pcrglobwb_output_dir = (
            self.get_simulation_dir('PCRGLOBWB')
        )
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        return self.get_simulation_dir('PCRGLOBWB')

    def extract_streamflow(self) -> Optional[Path]:
        import pandas as pd
        import xarray as xr

        self.logger.info("Extracting streamflow from PCR-GLOBWB outputs")
        output_dir = self._get_output_dir()

        # PCR-GLOBWB writes output to a netcdf/ subdirectory
        search_dirs = [
            output_dir / 'netcdf',
            output_dir,
            self.project_dir / 'settings' / 'PCRGLOBWB' / 'output',
        ]

        nc_file = None
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in ['discharge_dailyTot_output.nc', 'discharge_*_output.nc']:
                matches = list(search_dir.glob(pattern))
                if matches:
                    nc_file = matches[0]
                    break
            if nc_file is not None:
                break

        if nc_file is None:
            self.logger.error(
                f"PCR-GLOBWB discharge output not found in {search_dirs}"
            )
            return None

        try:
            ds = xr.open_dataset(nc_file)
            q_var = None
            for var in ['discharge', 'Qsim', 'Q']:
                if var in ds.data_vars:
                    q_var = ds[var]
                    break
            if q_var is None:
                self.logger.error(
                    f"No discharge variable found in {nc_file}. "
                    f"Available: {list(ds.data_vars)}"
                )
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
                streamflow, model_column_name='PCRGLOBWB_discharge_cms'
            )
        except Exception as e:  # noqa: BLE001
            import traceback
            self.logger.error(f"Error extracting PCR-GLOBWB streamflow: {e}")
            self.logger.debug(traceback.format_exc())
            return None
