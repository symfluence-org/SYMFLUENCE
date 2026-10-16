#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
HYPE Calibration Targets

Provides calibration target classes for HYPE hydrological model.
Handles HYPE output formats (timeCOUT.txt for direct, NetCDF for routed)
with automatic outlet subbasin selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from symfluence.core.registries import R
from symfluence.evaluation.evaluators import StreamflowEvaluator
from symfluence.evaluation.output_file_locator import OutputFileLocator

if TYPE_CHECKING:
    pass


@R.calibration_targets.add('HYPE_STREAMFLOW')
class HYPEStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for HYPE model outputs.

    HYPE is a semi-distributed hydrological model that produces streamflow
    for multiple subbasins. This target handles:
    1. timeCOUT.txt: Direct HYPE output with all subbasins
    2. NetCDF: mizuRoute-routed output (if coupled)

    Key Features:
        - Automatic outlet subbasin selection (highest mean discharge)
        - Handles multiple subbasin columns with intelligent detection
        - Tab-separated timeCOUT.txt parsing
        - Output already in m³/s (no unit conversion needed)

    Output Format:
        HYPE timeCOUT.txt contains:
        - DATE column (YYYY-MM-DD format)
        - Multiple subbasin columns (numeric IDs or names)
        - Values in m³/s (discharge at each subbasin outlet)

    Outlet Selection Strategy:
        When multiple subbasins exist, selects outlet by finding the
        subbasin with highest mean discharge (typically largest/downstream basin).
    """

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate HYPE output files (timeCOUT.txt or mizuRoute NetCDF).

        Args:
            sim_dir: Directory containing HYPE simulation outputs

        Returns:
            List[Path]: Paths to HYPE streamflow output file(s)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_hype_output(sim_dir, 'streamflow')

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract HYPE streamflow from direct output or routed NetCDF.

        Format Detection:
        - .nc files: mizuRoute routed output (calls parent method)
        - .txt files: HYPE timeCOUT.txt direct output

        Args:
            sim_files: List of simulation output file(s)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Daily streamflow time series (m³/s)
        """
        sim_file = sim_files[0]
        self.logger.info(f"Extracting HYPE streamflow from: {sim_file}")

        if sim_file.suffix == '.nc':
            return self._extract_mizuroute_streamflow(sim_file)

        return self._extract_hype_streamflow(sim_file)  # type: ignore[return-value]

    def _extract_hype_streamflow(self, sim_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from HYPE timeCOUT.txt with automatic outlet selection.

        HYPE timeCOUT.txt contains discharge (m³/s) for all subbasins at each timestep.
        This method:
        1. Parses tab-separated file with DATE column
        2. Detects all subbasin columns (numeric IDs or names)
        3. Selects outlet subbasin (highest mean discharge)
        4. Returns time series already in m³/s (no conversion needed)

        Args:
            sim_file: Path to HYPE timeCOUT.txt file

        Returns:
            pd.Series: Time series of outlet discharge (m³/s)

        Raises:
            Exception: If file cannot be read or no subbasins found
        """
        try:
            # Read timeCOUT.txt (tab-separated, skip first comment line)
            df = pd.read_csv(sim_file, sep='\t', skiprows=1)

            # Parse dates
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
                df = df.set_index('DATE')
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')

            # Get all subbasin columns (numeric column names)
            subbasin_cols = [col for col in df.columns if col not in ['DATE', 'time']]

            if len(subbasin_cols) == 0:
                raise ValueError(f"No subbasin columns found in {sim_file}")

            # Auto-select outlet subbasin (highest mean flow)
            if len(subbasin_cols) > 1:
                subbasin_means = df[subbasin_cols].mean(numeric_only=True)
                outlet_col = str(subbasin_means.idxmax())
                self.logger.info(f"Auto-selected HYPE outlet subbasin {outlet_col} (mean flow: {subbasin_means[outlet_col]:.2f} m3/s)")
                self.logger.debug(f"All subbasin mean flows: {dict(sorted(subbasin_means.items(), key=lambda x: x[1], reverse=True)[:5])}")
                streamflow = df[outlet_col]
            else:
                # Single subbasin - use it directly
                outlet_col = subbasin_cols[0]
                self.logger.info(f"Using single HYPE subbasin: {outlet_col}")
                streamflow = df[outlet_col]

            # Convert to numeric and handle errors
            streamflow = pd.to_numeric(streamflow, errors='coerce')

            # HYPE timeCOUT.txt is already in m3/s - no unit conversion needed
            return streamflow

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting HYPE streamflow from {sim_file}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise


__all__ = ['HYPEStreamflowTarget']
