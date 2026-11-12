# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC model postprocessor.

Handles extraction and processing of VIC model simulation results.
Uses StandardModelPostProcessor for reduced boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('VIC')
class VICPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the VIC model.

    VIC outputs streamflow as the sum of OUT_RUNOFF (surface runoff)
    and OUT_BASEFLOW (baseflow) in NetCDF format. Units are mm/day
    and require conversion to m³/s using catchment area.

    Attributes:
        model_name: "VIC"
        output_file_pattern: "vic_output*.nc"
        streamflow_variable: "OUT_RUNOFF"
        streamflow_unit: "mm_per_day"
    """

    model_name = "VIC"

    output_file_pattern = "vic_output*.nc"

    streamflow_variable = "OUT_RUNOFF"
    streamflow_unit = "mm_per_day"

    def _get_model_name(self) -> str:
        return "VIC"

    def _setup_model_specific_paths(self) -> None:
        """Set up VIC-specific paths."""
        self.vic_output_dir = self.get_simulation_dir('VIC')
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """VIC outputs to standard simulation directory."""
        return self.get_simulation_dir('VIC')

    def _get_domain_file(self) -> Path:
        """VIC domain file (cell areas) written by the preprocessor."""
        domain_file = self._get_config_value(lambda: self.config.model.vic.domain_file, dict_key='VIC_DOMAIN_FILE')
        return self.project_dir / 'settings' / 'VIC' / 'parameters' / domain_file

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from VIC outputs.

        VIC total streamflow = OUT_RUNOFF + OUT_BASEFLOW.
        Sums both components, takes the area-weighted basin mean over grid
        cells (shared with the calibration worker) and converts mm/day to m³/s.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from VIC outputs")

        output_dir = self._get_output_dir()

        # Find output file
        output_file = None
        for pattern in ['vic_output*.nc', '*fluxes*.nc', '*runoff*.nc']:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            self.logger.error(f"VIC output not found in {output_dir}")
            return None

        try:
            import xarray as xr

            from .extractor import basin_mean_vic_runoff, load_vic_cell_area

            cell_area = load_vic_cell_area(self._get_domain_file())
            with xr.open_dataset(output_file) as ds:
                try:
                    total_runoff = basin_mean_vic_runoff(ds, cell_area)
                except ModelExecutionError as e:
                    self.logger.error(f"{e}")
                    return None
                streamflow = total_runoff.to_series()

            # Convert mm/day to m³/s
            streamflow = self.convert_mm_per_day_to_cms(streamflow)

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='VIC_discharge_cms'
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Error extracting VIC streamflow: {str(e)}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return None
