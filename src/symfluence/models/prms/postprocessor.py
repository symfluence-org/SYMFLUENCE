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

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('PRMS')
class PRMSPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the PRMS model.

    PRMS outputs streamflow as basin_cfs / seg_outflow (cfs) in the statvar
    output file; both are converted to m3/s by the shared reader in
    :mod:`symfluence.models.prms.statvar`.

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

        basin_cfs / seg_outflow (cfs) are converted to m3/s.
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
                streamflow = self._extract_from_netcdf(output_file)
            else:
                # Parse statvar text file
                streamflow = self._parse_statvar_file(output_file)

            if streamflow is None or streamflow.empty:
                self.logger.error("No streamflow data found in PRMS statvar output")
                return None

            # Already converted to m3/s by the shared statvar reader

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
        """Basin streamflow (m3/s) from a PRMS statvar file.

        Delegates to :func:`symfluence.models.prms.statvar.read_statvar_streamflow`,
        the same parser the calibration worker uses (header element counts,
        cfs -> m3/s for basin_cfs and seg_outflow).
        """
        from .statvar import read_statvar_streamflow
        try:
            return read_statvar_streamflow(statvar_path).rename('PRMS_discharge_cms')
        except (OSError, ModelExecutionError) as e:
            self.logger.warning(f"Error parsing statvar file: {e}", exc_info=True)
            return None

    def _extract_from_netcdf(self, nc_path: Path):
        """Extract streamflow (m3/s) from NetCDF statvar output."""
        from .statvar import read_netcdf_streamflow
        try:
            return read_netcdf_streamflow(nc_path).rename('PRMS_discharge_cms')
        except ModelExecutionError as e:
            self.logger.warning(f"{e}")
            return None
