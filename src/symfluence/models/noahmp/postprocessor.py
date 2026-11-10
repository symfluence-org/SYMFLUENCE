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

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('NOAHMP')
class NoahMPPostProcessor(StandardModelPostProcessor):
    """Postprocessor for standalone Noah-MP.

    Extracts total runoff (SFCRNOFF + UGDRNOFF) from the output NetCDF and
    converts it to daily discharge (m3/s) exactly as the calibration worker
    does, writing the standard ``NOAHMP_discharge_cms`` column.
    """

    model_name = "NOAHMP"
    output_file_pattern = "output*.nc"
    streamflow_variable = "total_runoff"
    streamflow_unit = "cms"

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
            # Same conversion as the calibration worker: SFCRNOFF + UGDRNOFF
            # (mm/step) -> daily m3/s with the shapefile area, then the
            # calibrated Nash cascade (settings/NOAHMP/route_k.txt).
            from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

            from .extractor import noahmp_daily_discharge_cms, read_route_k

            area_km2 = StreamflowMetrics().get_catchment_area(
                self.config_dict, self.project_dir, self.domain_name, source='shapefile'
            )
            streamflow = noahmp_daily_discharge_cms(
                nc_file, area_km2 * 1e6,
                read_route_k(self.project_dir / 'settings' / 'NOAHMP'),
            )
            streamflow.index.name = 'datetime'

            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow, model_column_name='NOAHMP_discharge_cms'
            )
        except (OSError, ValueError, KeyError, TypeError, ModelExecutionError) as e:
            import traceback
            self.logger.error(f"Error extracting Noah-MP runoff: {e}")
            self.logger.debug(traceback.format_exc())
            return None
