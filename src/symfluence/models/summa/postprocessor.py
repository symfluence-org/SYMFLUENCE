# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA postprocessor module.

Handles both routed (mizuRoute) and non-routed (native SUMMA) output.
For lumped domains without routing, reads averageRoutedRunoff from SUMMA
output and converts from m/s to m³/s using HRU area.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from symfluence.core.modeling.base import RoutedModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('SUMMA')
class SUMMAPostProcessor(RoutedModelPostProcessor):
    """
    Postprocessor for SUMMA model outputs.

    When mizuRoute routing output exists, extracts routed streamflow
    (IRFroutedRunoff).  Otherwise falls back to SUMMA's native
    averageRoutedRunoff, converting from m/s depth to m³/s discharge.
    """

    model_name = "SUMMA"

    def extract_streamflow(self) -> Optional[Path]:
        routing_file = self._get_output_file()

        if routing_file.exists():
            return super().extract_streamflow()

        self.logger.info(
            "No mizuRoute output found — extracting from SUMMA native output"
        )
        return self._extract_native_summa_streamflow()

    def _extract_native_summa_streamflow(self) -> Optional[Path]:
        summa_dir = (
            self.project_dir / 'simulations' / self.experiment_id / 'SUMMA'
        )
        candidates = sorted(summa_dir.glob(f"{self.experiment_id}*.nc"))
        # Restart files contain state snapshots, not runoff time series, and
        # sort before the timestep output in ordinary experiment directories.
        candidates = [
            p for p in candidates
            if 'day' not in p.stem
            and not p.stem[len(self.experiment_id):].lower().startswith('_restart')
        ]

        if not candidates:
            self.logger.error(f"No SUMMA output files found in {summa_dir}")
            return None

        try:
            with xr.open_dataset(candidates[0]) as ds:
                if 'averageRoutedRunoff' in ds:
                    var_name = 'averageRoutedRunoff'
                elif 'scalarTotalRunoff' in ds:
                    var_name = 'scalarTotalRunoff'
                else:
                    self.logger.error(
                        f"No runoff variable in {candidates[0]}. "
                        f"Available: {list(ds.data_vars)}"
                    )
                    return None

                var = ds[var_name]

                # Collapse HRU dimension (area-weighted if possible)
                if 'hru' in var.dims and var.sizes['hru'] > 1:
                    if 'HRUarea' in ds:
                        weights = ds['HRUarea'] / ds['HRUarea'].sum()
                        series_m_per_s = (var * weights).sum(dim='hru').to_pandas()
                    else:
                        series_m_per_s = var.mean(dim='hru').to_pandas()
                else:
                    series_m_per_s = var.squeeze().to_pandas()

                # Convert m/s depth to m³/s discharge
                if 'HRUarea' in ds:
                    basin_area_m2 = float(ds['HRUarea'].sum())
                    streamflow = series_m_per_s * basin_area_m2
                else:
                    attrs_file = (
                        self.project_dir / 'settings' / 'SUMMA'
                        / 'attributes.nc'
                    )
                    if attrs_file.exists():
                        with xr.open_dataset(attrs_file) as attrs:
                            basin_area_m2 = float(attrs['HRUarea'].sum())
                        streamflow = series_m_per_s * basin_area_m2
                    else:
                        self.logger.warning(
                            "HRUarea not available — streamflow units "
                            "will be m/s depth, not m³/s"
                        )
                        streamflow = series_m_per_s

            streamflow.index = pd.to_datetime(streamflow.index).round(freq='h')

            # Resample to daily (same as routed path)
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(streamflow)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error extracting SUMMA native streamflow: {e}", exc_info=True)
            return None
