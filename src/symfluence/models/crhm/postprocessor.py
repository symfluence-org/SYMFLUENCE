# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM model postprocessor.

Handles extraction and processing of CRHM model simulation results.
Uses StandardModelPostProcessor for reduced boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('CRHM')
class CRHMPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the CRHM model.

    CRHM outputs a tab-separated text file with:
    - Row 1: column headers (time, SWE(1), basinflow(1), ...)
    - Row 2: units (units, (mm), (m^3/int), ...)
    - Row 3+: ISO datetime + numeric values

    ``basinflow`` is in m^3 per interval; we convert to m^3/s.
    """

    model_name = "CRHM"

    output_file_pattern = "crhm_output.txt"

    streamflow_variable = "basinflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "CRHM"

    def _setup_model_specific_paths(self) -> None:
        """Set up CRHM-specific paths."""
        self.crhm_output_dir = self.get_simulation_dir('CRHM')
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """CRHM outputs to standard simulation directory."""
        return self.get_simulation_dir('CRHM')

    def _find_output_file(self) -> Optional[Path]:
        """Locate the CRHM output file."""
        output_dir = self._get_output_dir()

        # Search patterns in priority order: known name, then txt, then csv
        search_dirs = [output_dir, self.project_dir / 'settings' / 'CRHM']
        patterns = ['crhm_output.txt', '*output*.txt', '*.txt', '*output*.csv', '*.csv']

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from CRHM output.

        CRHM writes a tab-separated file with a units row.  The
        ``basinflow`` column contains basin outflow in m^3 per
        time-interval.  We infer the interval from consecutive
        timestamps and convert to m^3/s.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from CRHM outputs")

        output_file = self._find_output_file()
        if output_file is None:
            self.logger.error(
                f"CRHM output not found in {self._get_output_dir()}"
            )
            return None

        self.logger.info(f"Reading CRHM output: {output_file}")

        try:
            import pandas as pd

            # Read tab-separated file, skip the units row (row index 1).
            # CRHM (Fortran) writes its output in Latin-1, not UTF-8 — unit
            # headers contain the degree symbol (0xba), which raises a
            # UnicodeDecodeError under pandas' default UTF-8 decoding.
            df = pd.read_csv(
                output_file, sep='\t', skiprows=[1],
                index_col=0, parse_dates=True, encoding='latin-1'
            )

            # Find basinflow column (may include HRU index, e.g. "basinflow(1)")
            flow_col = None
            for col in df.columns:
                if col.lower().startswith('basinflow'):
                    flow_col = col
                    break

            # Fallback: any column with 'flow' in the name
            if flow_col is None:
                flow_cols = [c for c in df.columns if 'flow' in c.lower()]
                if flow_cols:
                    flow_col = flow_cols[0]

            if flow_col is None:
                self.logger.error(
                    f"No flow variable found in CRHM output. "
                    f"Available columns: {list(df.columns)}"
                )
                return None

            self.logger.info(f"Using flow column: {flow_col}")
            # m^3/interval (end-labelled) -> m^3/s (start-labelled); the
            # same conversion the calibration worker applies.
            from .extractor import basinflow_to_cms
            streamflow = basinflow_to_cms(df[flow_col])

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='CRHM_discharge_cms'
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Error extracting CRHM streamflow: {str(e)}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return None
