# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH model postprocessor.

Handles extraction and processing of MESH model simulation results.
Migrated to use StandardModelPostProcessor for reduced boilerplate (Phase 1.3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('MESH')
class MESHPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the MESH model.

    Handles extraction and processing of MESH model simulation results.
    Uses StandardModelPostProcessor with configuration-based extraction.

    MESH outputs streamflow to MESH_output_streamflow.csv with columns:
    DAY, YEAR, QOMEAS1, QOSIM1, QOMEAS2, QOSIM2, ...

    Special handling:
    - Julian date format (DAY + YEAR columns)
    - Column pattern matching for QOSIM* columns
    - Output directory is forcing/MESH_input (not standard sim_dir)

    Attributes:
        model_name: "MESH"
        output_file_pattern: "MESH_output_streamflow.csv"
        date_parser_type: "julian" for DAY+YEAR columns
        outlet_column_pattern: r"QOSIM\\d+" to match QOSIM1, QOSIM2, etc.
    """

    # Model identification
    model_name = "MESH"

    # Output file configuration
    output_file_pattern = "MESH_output_streamflow.csv"

    # Text file parsing
    text_file_separator = ","
    text_file_skiprows = 0  # MESH has a proper header

    # Julian date parsing (DAY + YEAR columns)
    date_parser_type = "julian"

    # Column pattern matching for QOSIM columns
    outlet_column_pattern = r"QOSIM\d+"
    outlet_selection_method = "pattern"

    # Streamflow is already in cms from MESH
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "MESH"

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_basin_path = self.project_forcing_dir / 'basin_averaged_data'
        self.forcing_mesh_path = self.project_forcing_dir / 'MESH_input'
        # Catchment paths (use backward-compatible path resolution)
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """
        Override: MESH outputs to forcing directory, not simulation directory.

        Returns:
            Path to MESH output directory (forcing/MESH_input)
        """
        return self.project_forcing_dir / 'MESH_input'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from MESH outputs.

        Handles multiple MESH output formats:
        - MESH_output_streamflow.csv (routed mode with QOSIM columns)
        - results/Basin_average_water_balance.csv (lumped/noroute mode with RFF)
        - MESH_output_streamflow_ts.csv (alternative timestep output)

        Returns:
            Optional[Path]: Path to processed streamflow file, or None if extraction fails
        """
        self.logger.info("Extracting streamflow from MESH outputs")

        output_dir = self._get_output_dir()

        # Try output files in priority order
        candidates = [
            output_dir / self.output_file_pattern,
            output_dir / 'results' / 'Basin_average_water_balance.csv',
            output_dir / 'MESH_output_streamflow_ts.csv',
        ]

        mesh_output_file = None
        for candidate in candidates:
            if candidate.exists():
                mesh_output_file = candidate
                break

        if mesh_output_file is None:
            self.logger.warning(
                f"MESH streamflow output not found at {output_dir}")
            return None

        try:
            # Basin_average_water_balance needs the extractor (RFF + unit conversion)
            if 'Basin_average_water_balance' in mesh_output_file.name:
                from .extractor import MESHResultExtractor
                extractor = MESHResultExtractor('MESH')

                # Parse start date from run options
                # Shared strict parser: raises instead of assuming a start date
                from .runner import parse_run_options_dates
                start_date, _ = parse_run_options_dates(output_dir / 'MESH_input_run_options.ini')

                streamflow = extractor.extract_variable(
                    mesh_output_file, 'runoff',
                    start_date=start_date, aggregate='daily'
                )

                # Convert mm/day to m³/s using basin area
                import xarray as xr
                ddb_path = output_dir / 'MESH_drainage_database.nc'
                if ddb_path.exists():
                    with xr.open_dataset(ddb_path) as ds:
                        area_m2 = float(ds['GridArea'].values.sum()) if 'GridArea' in ds else float(ds['DA'].values.sum())
                    streamflow = streamflow * (area_m2 * 0.001 / 86400)
                    self.logger.info(f"Converted runoff to discharge using area={area_m2/1e6:.1f} km²")

                return self.save_streamflow_to_results(
                    streamflow,
                    model_column_name='MESH_discharge_cms'
                )

            # Standard streamflow CSV — use text extraction
            streamflow = self._extract_from_text(mesh_output_file)

            if streamflow is None:
                return None

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            # Apply unit conversion if needed
            if self.streamflow_unit == "mm_per_day":
                streamflow = self.convert_mm_per_day_to_cms(streamflow)

            # Use inherited save method
            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='MESH_discharge_cms'
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Error extracting MESH streamflow: {str(e)}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return None
