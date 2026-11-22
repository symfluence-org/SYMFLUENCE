# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM model postprocessor.

Handles extraction and processing of mHM model simulation results.
Uses StandardModelPostProcessor for reduced boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('MHM')
class MHMPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the mHM model.

    mHM outputs discharge in NetCDF format (discharge_*.nc) with
    units of m3/s. Fluxes and states are stored in separate
    mHM_Fluxes_States_*.nc files.

    Attributes:
        model_name: "MHM"
        output_file_pattern: "discharge_*.nc"
        streamflow_variable: "Qsim"
        streamflow_unit: "m3_per_s"
    """

    model_name = "MHM"

    output_file_pattern = "discharge_*.nc"

    streamflow_variable = "Qsim"
    streamflow_unit = "m3_per_s"

    def _get_model_name(self) -> str:
        return "MHM"

    def _setup_model_specific_paths(self) -> None:
        """Set up mHM-specific paths."""
        self.mhm_output_dir = self.project_dir / 'simulations' / self.experiment_id / 'MHM'
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """mHM outputs to standard simulation directory."""
        return self.project_dir / 'simulations' / self.experiment_id / 'MHM'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from mHM outputs.

        mHM discharge is already in m3/s in the discharge_*.nc file.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from mHM outputs")

        output_dir = self._get_output_dir()

        # Find discharge output file
        output_file = None
        # Search in output directory and settings output directory
        search_dirs = [output_dir]
        mhm_settings_dir = self.project_dir / 'settings' / 'MHM'
        for subdir in ['output', '']:
            candidate = mhm_settings_dir / subdir if subdir else mhm_settings_dir
            if candidate.exists():
                search_dirs.append(candidate)

        for search_dir in search_dirs:
            for pattern in ['discharge_*.nc', '*discharge*.nc', 'Qsim*.nc']:
                matches = list(search_dir.glob(pattern))
                if matches:
                    output_file = matches[0]
                    break
            if output_file is not None:
                break

        if output_file is None:
            self.logger.error(f"mHM discharge output not found in {output_dir}")
            return None

        try:
            from .extractor import read_mhm_discharge

            streamflow = read_mhm_discharge(output_file)

            # mHM discharge is already in m3/s, no conversion needed

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='MHM_discharge_cms'
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Error extracting mHM streamflow: {str(e)}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return None
