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


def _native_runoff_to_discharge(var, ds, attrs_file: Path, logger) -> Optional[pd.Series]:
    """Per-HRU/GRU runoff (m/s) -> basin discharge (m3/s).

    Area-weighted sum over HRUs or GRUs with the rule shared with the
    evaluator and the calibration worker (GRU areas = HRUarea summed by
    hru2gruId). Areas come from the output file when it carries them,
    else ``attrs_file`` (settings/SUMMA/attributes.nc).
    """
    from symfluence.core.modeling.utilities.area_weighting import area_weighted_discharge

    spatial_dims = [d for d in var.dims if d != 'time']

    if spatial_dims:
        discharge = None
        if 'HRUarea' in ds or 'GRUarea' in ds:
            discharge = area_weighted_discharge(var, ds)
        if discharge is None and attrs_file.exists():
            with xr.open_dataset(attrs_file) as attrs:
                discharge = area_weighted_discharge(var, attrs.load())
        if discharge is not None:
            return discharge.to_pandas()
        if any(var.sizes[d] > 1 for d in spatial_dims):
            logger.error(
                f"Cannot aggregate SUMMA runoff over {dict(var.sizes)} without "
                f"matching HRU/GRU areas (output file or {attrs_file})"
            )
            return None

    series_m_per_s = var.squeeze().to_pandas()
    if 'HRUarea' in ds:
        return series_m_per_s * float(ds['HRUarea'].sum())
    if attrs_file.exists():
        with xr.open_dataset(attrs_file) as attrs:
            return series_m_per_s * float(attrs['HRUarea'].sum())
    logger.warning(
        "HRUarea not available — streamflow units will be m/s depth, not m³/s"
    )
    return series_m_per_s


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
        # Match the output-type suffix after the experiment id, not a
        # substring of the whole stem: an experiment id containing "day"
        # (e.g. "run_daymet") must not drop every file.
        def _tail(p: Path) -> str:
            return p.stem[len(self.experiment_id):].lower()

        candidates = [
            p for p in candidates
            if not _tail(p).endswith('_day')
            and not _tail(p).startswith('_restart')
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
                streamflow = _native_runoff_to_discharge(
                    var, ds,
                    self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc',
                    self.logger,
                )
                if streamflow is None:
                    return None

            streamflow.index = pd.to_datetime(streamflow.index).round(freq='h')

            # Resample to daily (same as routed path)
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(streamflow)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error extracting SUMMA native streamflow: {e}", exc_info=True)
            return None
