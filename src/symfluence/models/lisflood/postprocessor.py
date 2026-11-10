# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD model postprocessor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.modeling.utilities.periods import to_pydatetime
from symfluence.core.registries import R


@R.postprocessors.add("LISFLOOD")
class LisfloodPostProcessor(StandardModelPostProcessor):
    """Postprocessor for the LISFLOOD model.

    LISFLOOD outputs discharge in m³/s in NetCDF format (dis*.nc) or as
    time series text files.
    """

    model_name = "LISFLOOD"
    output_file_pattern = "dis*"
    streamflow_variable = "dis"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "LISFLOOD"

    def _setup_model_specific_paths(self) -> None:
        self.lisflood_output_dir = self.get_simulation_dir('LISFLOOD')
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        return self.get_simulation_dir('LISFLOOD')

    def extract_streamflow(self) -> Optional[Path]:
        self.logger.info("Extracting streamflow from LISFLOOD outputs")
        output_dir = self._get_output_dir()

        # Try NetCDF output first (dis*.nc)
        nc_file = None
        for pattern in ["dis*.nc", "dis.nc", "*discharge*.nc"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                nc_file = matches[0]
                break

        if nc_file is not None:
            try:
                import xarray as xr

                ds = xr.open_dataset(nc_file)
                q_var = None
                for var in ["dis", "discharge", "Qsim", "chanq"]:
                    if var in ds.data_vars:
                        q_var = ds[var]
                        break
                if q_var is None and len(ds.data_vars) == 1:
                    q_var = ds[list(ds.data_vars)[0]]
                if q_var is None:
                    self.logger.error(f"No discharge variable found in {nc_file}")
                    ds.close()
                    return None
                spatial_dims = [d for d in q_var.dims if d not in ["time"]]
                if spatial_dims:
                    streamflow = q_var.max(dim=spatial_dims)
                else:
                    streamflow = q_var
                streamflow = streamflow.to_series()
                ds.close()
                if self.resample_frequency:
                    streamflow = streamflow.resample(self.resample_frequency).mean()
                return self.save_streamflow_to_results(streamflow, model_column_name="LISFLOOD_discharge_cms")
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Failed to read NetCDF output: {e}, trying TSS")

        # Fallback: LISFLOOD TSS (time series) text file
        tss_file = None
        for pattern in ["dis*.tss", "*discharge*.tss"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                tss_file = matches[0]
                break

        if tss_file is not None:
            try:
                streamflow = self._read_tss_file(tss_file)
                if self.resample_frequency:
                    streamflow = streamflow.resample(self.resample_frequency).mean()
                return self.save_streamflow_to_results(streamflow, model_column_name="LISFLOOD_discharge_cms")
            except Exception as e:  # noqa: BLE001
                import traceback

                self.logger.error(f"Error reading TSS file: {e}")
                self.logger.debug(traceback.format_exc())

        self.logger.error(f"LISFLOOD output not found in {output_dir}")
        return None

    def _read_tss_file(self, tss_path: Path) -> pd.Series:
        """Read LISFLOOD TSS (time series) output as a dated series.

        Uses the shared reader (``extractor.read_tss_series``) with the
        model step the preprocessor writes as DtSec
        (LISFLOOD_FORCING_TIMESTEP, default 86400 s) -- the same rule the
        calibration worker uses.
        """
        from .extractor import DEFAULT_LISFLOOD_TIMESTEP_SECONDS, read_tss_series

        start_date, _ = self._get_simulation_dates()
        timestep_secs = int(self._get_config_value(
            lambda: self.config.model.lisflood.forcing_timestep,
            default=DEFAULT_LISFLOOD_TIMESTEP_SECONDS, dict_key="LISFLOOD_FORCING_TIMESTEP",
        ))
        return read_tss_series(tss_path, start_date, timestep_secs).rename("LISFLOOD_discharge_cms")

    def _get_simulation_dates(self):
        start = self._get_config_value(lambda: self.config.domain.time_start)
        end = self._get_config_value(lambda: self.config.domain.time_end)
        return to_pydatetime(start), to_pydatetime(end)
