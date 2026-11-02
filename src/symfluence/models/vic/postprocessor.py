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

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from VIC outputs.

        VIC total streamflow = OUT_RUNOFF + OUT_BASEFLOW.
        Sums both components and converts from mm/day to m³/s.

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

            ds = xr.open_dataset(output_file)

            # Find runoff variable
            runoff = None
            for var in ['OUT_RUNOFF', 'RUNOFF']:
                if var in ds.data_vars:
                    runoff = ds[var]
                    break

            if runoff is None:
                self.logger.error("No runoff variable found in VIC output")
                ds.close()
                return None

            # Aggregate spatially if needed
            spatial_dims = [d for d in runoff.dims if d not in ['time']]
            if spatial_dims:
                total_runoff = runoff.sum(dim=spatial_dims)
            else:
                total_runoff = runoff

            # Add baseflow
            for var in ['OUT_BASEFLOW', 'BASEFLOW']:
                if var in ds.data_vars:
                    baseflow = ds[var]
                    if spatial_dims:
                        baseflow = baseflow.sum(dim=spatial_dims)
                    total_runoff = total_runoff + baseflow
                    break

            streamflow = total_runoff.to_series()
            ds.close()

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
