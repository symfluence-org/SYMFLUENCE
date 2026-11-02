# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WRF-Hydro model postprocessor.

Handles extraction and processing of WRF-Hydro model simulation results.
Uses StandardModelPostProcessor for reduced boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('WRFHYDRO')
class WRFHydroPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for the WRF-Hydro model.

    WRF-Hydro outputs streamflow in CHRTOUT NetCDF files with variable
    'streamflow' in cubic meters per second (cms). When routing is
    disabled (standalone HRLDAS mode), streamflow is derived from
    LDASOUT accumulated runoff fields (SFCRNOFF + UGDRNOFF).

    Attributes:
        model_name: "WRFHYDRO"
        output_file_pattern: "*CHRTOUT*.nc"
        streamflow_variable: "streamflow"
        streamflow_unit: "cms"
    """

    model_name = "WRFHYDRO"

    output_file_pattern = "*CHRTOUT*.nc"

    streamflow_variable = "streamflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "WRFHYDRO"

    def _setup_model_specific_paths(self) -> None:
        """Set up WRF-Hydro-specific paths."""
        self.wrfhydro_output_dir = self.get_simulation_dir('WRFHYDRO')
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """WRF-Hydro outputs to standard simulation directory."""
        return self.get_simulation_dir('WRFHYDRO')

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from WRF-Hydro outputs.

        Tries CHRTOUT files first (routing enabled). Falls back to
        LDASOUT-derived streamflow when routing is disabled.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        output_dir = self._get_output_dir()

        # Try CHRTOUT files first (routing enabled)
        chrtout_files = sorted(output_dir.glob('*CHRTOUT*.nc'))
        if chrtout_files:
            return self._extract_from_chrtout(chrtout_files)

        # Fall back to LDASOUT files (standalone HRLDAS / no routing)
        ldasout_files = sorted(output_dir.glob('*LDASOUT*'))
        if ldasout_files:
            self.logger.info(
                f"No CHRTOUT files found; deriving streamflow from "
                f"{len(ldasout_files)} LDASOUT files"
            )
            return self._extract_from_ldasout(ldasout_files)

        self.logger.error(f"No WRF-Hydro output files found in {output_dir}")
        return None

    def _extract_from_chrtout(self, output_files: list) -> Optional[Path]:
        """Extract streamflow from CHRTOUT NetCDF files."""
        self.logger.info("Extracting streamflow from WRF-Hydro CHRTOUT outputs")

        try:
            import xarray as xr

            streamflow_series = []

            for fpath in output_files:
                ds = xr.open_dataset(fpath)

                flow = None
                for var in ['streamflow', 'q_lateral', 'qSfcLatRunoff']:
                    if var in ds.data_vars:
                        flow = ds[var]
                        break

                if flow is not None:
                    if 'feature_id' in flow.dims:
                        flow = flow.isel(feature_id=-1)

                    if 'time' in flow.dims:
                        streamflow_series.append(flow.to_series())
                    else:
                        time_val = ds.attrs.get('model_output_valid_time',
                                                fpath.stem[:10])
                        try:
                            t = pd.Timestamp(time_val)
                        except Exception:  # noqa: BLE001 — model execution resilience
                            t = pd.Timestamp.now()
                        streamflow_series.append(
                            pd.Series([float(flow.values)], index=[t])
                        )
                ds.close()

            if not streamflow_series:
                self.logger.error("No streamflow data found in CHRTOUT files")
                return None

            streamflow = pd.concat(streamflow_series).sort_index()

            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='WRFHYDRO_discharge_cms'
            )

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Error extracting WRF-Hydro streamflow: {str(e)}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return None

    def _extract_from_ldasout(self, ldasout_files: list) -> Optional[Path]:
        """
        Derive streamflow from LDASOUT surface and subsurface runoff.

        SFCRNOFF and UGDRNOFF are accumulated mm from simulation start.
        Daily values are derived by differencing consecutive timesteps
        and converting to m3/s using catchment area.
        """
        import netCDF4

        catchment_area_km2 = self._get_config_value(
            lambda: None,
            default=2210.0,
            dict_key='CATCHMENT_AREA_KM2'
        )
        area_m2 = float(catchment_area_km2) * 1e6

        timestamps = []
        accum_values = []

        for fpath in sorted(ldasout_files):
            fname = fpath.name
            ts_str = fname.split('.')[0]
            try:
                if len(ts_str) >= 10:
                    ts = pd.Timestamp(
                        year=int(ts_str[:4]), month=int(ts_str[4:6]),
                        day=int(ts_str[6:8]), hour=int(ts_str[8:10])
                    )
                else:
                    continue
            except (ValueError, IndexError):
                continue

            try:
                nc = netCDF4.Dataset(str(fpath), 'r')
                sfcrnoff = float(nc['SFCRNOFF'][:].mean()) if 'SFCRNOFF' in nc.variables else 0.0
                ugdrnoff = float(nc['UGDRNOFF'][:].mean()) if 'UGDRNOFF' in nc.variables else 0.0
                nc.close()
            except Exception:  # noqa: BLE001 — model execution resilience
                continue

            if np.isnan(sfcrnoff) or sfcrnoff < -9000:
                sfcrnoff = 0.0
            if np.isnan(ugdrnoff) or ugdrnoff < -9000:
                ugdrnoff = 0.0

            timestamps.append(ts)
            accum_values.append(sfcrnoff + ugdrnoff)

        if not timestamps:
            self.logger.error("No valid LDASOUT runoff data found")
            return None

        accum = pd.Series(accum_values, index=pd.DatetimeIndex(timestamps)).sort_index()
        delta_mm = accum.diff()
        delta_mm.iloc[0] = accum.iloc[0]

        # Infer timestep (handles hourly or daily output)
        if len(accum) >= 2:
            dt_seconds = (accum.index[1] - accum.index[0]).total_seconds()
        else:
            dt_seconds = 86400.0

        # mm/timestep -> m3/s
        q_cms = delta_mm * area_m2 / (dt_seconds * 1000.0)
        q_cms.name = 'WRFHYDRO_discharge_cms'

        # Resample to daily
        streamflow = q_cms.resample('D').mean()

        self.logger.info(
            f"Derived streamflow from LDASOUT: {len(streamflow)} daily values, "
            f"mean={streamflow.mean():.3f} m3/s"
        )

        return self.save_streamflow_to_results(
            streamflow,
            model_column_name='WRFHYDRO_discharge_cms'
        )
