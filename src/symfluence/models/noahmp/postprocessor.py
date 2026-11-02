# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP model postprocessor.

noah-owp-modular writes a single ``output.nc`` containing all water and
energy variables on the model time axis.  Runoff (surface + subsurface)
is the closest analogue to streamflow for a 1-D column model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('NOAHMP')
class NoahMPPostProcessor(StandardModelPostProcessor):
    """Postprocessor for standalone Noah-MP.

    Extracts total runoff (SFCRNOFF + UGDRNOFF) from the output NetCDF,
    converting from kg/m2 to mm.
    """

    model_name = "NOAHMP"
    output_file_pattern = "output*.nc"
    streamflow_variable = "total_runoff"
    streamflow_unit = "mm"

    def _get_model_name(self) -> str:
        return "NOAHMP"

    def _setup_model_specific_paths(self) -> None:
        self.noahmp_output_dir = (
            self.get_simulation_dir('NOAHMP')
        )
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        return self.get_simulation_dir('NOAHMP')

    def extract_streamflow(self) -> Optional[Path]:
        import pandas as pd
        import xarray as xr

        self.logger.info("Extracting runoff from Noah-MP outputs")
        # The namelist uses a relative output_filename, so the model writes
        # output.nc into its working directory (settings/NOAHMP) for the
        # standalone run_model step. Search both the simulations output dir and
        # the settings dir.
        search_dirs = [
            self._get_output_dir(),
            self.project_dir / 'settings' / 'NOAHMP',
        ]

        nc_file = None
        for output_dir in search_dirs:
            for pattern in ['output.nc', 'output*.nc', '*output*.nc']:
                matches = list(output_dir.glob(pattern))
                if matches:
                    nc_file = matches[0]
                    break
            if nc_file is not None:
                break

        if nc_file is None:
            self.logger.error(
                f"Noah-MP output not found in any of {search_dirs}"
            )
            return None

        try:
            ds = xr.open_dataset(nc_file)

            runoff = None
            if 'SFCRNOFF' in ds.data_vars and 'UGDRNOFF' in ds.data_vars:
                runoff = ds['SFCRNOFF'] + ds['UGDRNOFF']
            else:
                for var in ['SFCRNOFF', 'UGDRNOFF', 'QINSUR', 'RUNSRF', 'RUNSUB']:
                    if var in ds.data_vars:
                        runoff = ds[var]
                        break

            if runoff is None:
                self.logger.error(f"No runoff variable in {nc_file}")
                ds.close()
                return None

            spatial_dims = [d for d in runoff.dims if d not in ['time']]
            if spatial_dims:
                runoff = runoff.mean(dim=spatial_dims)

            streamflow = runoff.to_series()
            ds.close()

            if not isinstance(streamflow.index, pd.DatetimeIndex):
                streamflow.index = pd.to_datetime(streamflow.index)
            streamflow.index.name = 'datetime'

            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow, model_column_name='NOAHMP_runoff_mm'
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            import traceback
            self.logger.error(f"Error extracting Noah-MP runoff: {e}")
            self.logger.debug(traceback.format_exc())
            return None
