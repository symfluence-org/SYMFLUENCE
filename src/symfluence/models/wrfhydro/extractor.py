# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WRF-Hydro Result Extractor.

Handles extraction of simulation results from WRF-Hydro model outputs.
WRF-Hydro outputs are in NetCDF format with CHRTOUT (channel) and
LDASOUT (land surface) file types.

When routing is disabled (standalone Noah-MP / HRLDAS mode), streamflow
is derived from LDASOUT runoff fields by differencing accumulated values:
  delta(t) = (SFCRNOFF(t) + UGDRNOFF(t)) - (SFCRNOFF(t-1) + UGDRNOFF(t-1))
  Q(t) = delta(t) * basin_area_m2 / (dt * 1000)
where SFCRNOFF/UGDRNOFF are accumulated mm from simulation start.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.modeling.base import ModelResultExtractor


class WRFHydroResultExtractor(ModelResultExtractor):
    """WRF-Hydro-specific result extraction.

    Handles WRF-Hydro model's output characteristics:
    - File formats: NetCDF (.nc)
    - CHRTOUT files: Channel discharge (streamflow)
    - LDASOUT files: Land surface variables (ET, soil moisture, SWE)
    - Variable naming: streamflow, ACSNOM, SOIL_M, SNEQV, etc.
    - Standalone mode: Derives streamflow from LDASOUT runoff fields
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for WRF-Hydro outputs."""
        return {
            'streamflow': [
                '*CHRTOUT*.nc',
                '*CHANOBS*.nc',
                '*LDASOUT*',
            ],
            'et': [
                '*LDASOUT*',
            ],
            'soil_moisture': [
                '*LDASOUT*',
            ],
            'snow': [
                '*LDASOUT*',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get WRF-Hydro variable names for different types."""
        variable_mapping = {
            'streamflow': [
                'streamflow',
                'q_lateral',
                'qSfcLatRunoff',
                'qBucket',
            ],
            'runoff': [
                'SFCRNOFF',
                'UGDRNOFF',
            ],
            'et': [
                'ACCET',
                'ECAN',
                'ETRAN',
                'EDIR',
            ],
            'soil_moisture': [
                'SOIL_M',
                'SH2O',
                'SMC',
                'SMCWTD',
            ],
            'snow': [
                'SNEQV',
                'SNOWH',
                'ACSNOM',
                'FSNO',
            ],
            'precipitation': [
                'RAINRATE',
                'RAINNC',
                'RAINC',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from WRF-Hydro output."""
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        aggregate = kwargs.get('aggregate', True)

        try:
            ds = xr.open_dataset(output_file)

            found_var = None
            for var_name in var_names:
                if var_name in ds.data_vars:
                    found_var = var_name
                    break

            if found_var is None:
                available = list(ds.data_vars)
                ds.close()
                raise ValueError(
                    f"No suitable variable found in {output_file}. "
                    f"Tried: {var_names}. Available: {available}"
                )

            data = ds[found_var]

            if aggregate and len(data.dims) > 1:
                spatial_dims = [d for d in data.dims if d not in ['time']]
                method = self.get_spatial_aggregation_method(variable_type)
                if method == 'sum':
                    data = data.sum(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)

            if 'time' in data.dims:
                series = data.to_series()
            else:
                series = pd.Series([float(data.values)], index=[pd.Timestamp.now()])

            ds.close()
            return series

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(
                f"Error reading WRF-Hydro output file {output_file}: {str(e)}"
            ) from e

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from WRF-Hydro output.

        First tries CHRTOUT files (routed streamflow in m3/s).
        If no CHRTOUT files exist (standalone/no-routing mode), derives
        streamflow from LDASOUT runoff fields:
          Q = (SFCRNOFF + UGDRNOFF) * area_m2 / (dt * 1000)

        SFCRNOFF/UGDRNOFF are accumulated mm over each output timestep.

        Args:
            output_dir: Directory containing WRF-Hydro outputs
            catchment_area: Basin area in km2 (required for LDASOUT mode)
            **kwargs: Additional options

        Returns:
            Time series of streamflow in m3/s
        """

        # Try CHRTOUT files first (routing enabled)
        chrtout_files = sorted(output_dir.glob("*CHRTOUT*.nc"))
        if chrtout_files:
            return self._extract_chrtout_streamflow(chrtout_files)

        # Fall back to LDASOUT-derived streamflow (no routing)
        ldasout_files = sorted(output_dir.glob("*LDASOUT*"))
        if not ldasout_files:
            raise FileNotFoundError(
                f"No WRF-Hydro output files found in {output_dir}"
            )

        if not catchment_area:
            raise ConfigurationError(
                "WRF-Hydro wrote no CHRTOUT files; deriving streamflow from LDASOUT "
                "runoff needs catchment_area (km2)"
            )
        return self._extract_ldasout_streamflow(ldasout_files, catchment_area)

    def _extract_chrtout_streamflow(self, files: list) -> pd.Series:
        """Extract streamflow from CHRTOUT files."""
        import xarray as xr

        all_series = []
        for fpath in files:
            ds = xr.open_dataset(fpath)
            for var in self.get_variable_names('streamflow'):
                if var in ds.data_vars:
                    flow = ds[var]
                    if 'feature_id' in flow.dims:
                        flow = flow.isel(feature_id=-1)
                    if 'time' in flow.dims:
                        all_series.append(flow.to_series())
                    break
            ds.close()

        if not all_series:
            raise ValueError("No streamflow data found in CHRTOUT files")
        return pd.concat(all_series).sort_index()

    def _extract_ldasout_streamflow(
        self, files: list, catchment_area_km2: float
    ) -> pd.Series:
        """
        Derive streamflow from LDASOUT surface and subsurface runoff.

        SFCRNOFF and UGDRNOFF are accumulated mm from simulation start.
        Streamflow is derived by differencing consecutive timesteps:
          delta_mm = total_runoff(t) - total_runoff(t-1)
          Q = delta_mm * area_m2 / (dt_s * 1000)
        """
        import xarray as xr

        area_m2 = catchment_area_km2 * 1e6  # km2 → m2
        timestamps = []
        accum_values = []

        for fpath in sorted(files):
            try:
                ds = xr.open_dataset(fpath)
            except Exception:  # noqa: BLE001 — model execution resilience
                continue

            sfcrnoff = 0.0
            ugdrnoff = 0.0

            if 'SFCRNOFF' in ds.data_vars:
                val = float(ds['SFCRNOFF'].values.mean())
                if not np.isnan(val) and val > -9000:
                    sfcrnoff = val
            if 'UGDRNOFF' in ds.data_vars:
                val = float(ds['UGDRNOFF'].values.mean())
                if not np.isnan(val) and val > -9000:
                    ugdrnoff = val

            total_accum_mm = sfcrnoff + ugdrnoff

            # Parse timestamp from filename: YYYYMMDDHHMM.LDASOUT_DOMAIN1
            fname = fpath.name
            try:
                ts_str = fname.split('.')[0]
                if len(ts_str) == 12:
                    ts = pd.Timestamp(
                        year=int(ts_str[:4]), month=int(ts_str[4:6]),
                        day=int(ts_str[6:8]), hour=int(ts_str[8:10]),
                        minute=int(ts_str[10:12])
                    )
                elif len(ts_str) == 10:
                    ts = pd.Timestamp(
                        year=int(ts_str[:4]), month=int(ts_str[4:6]),
                        day=int(ts_str[6:8]), hour=int(ts_str[8:10])
                    )
                else:
                    ds.close()
                    continue
            except (ValueError, IndexError):
                ds.close()
                continue

            timestamps.append(ts)
            accum_values.append(total_accum_mm)
            ds.close()

        if not timestamps:
            raise ValueError("No valid LDASOUT runoff data found")

        # Build accumulated series and difference to get per-timestep runoff
        accum = pd.Series(accum_values, index=pd.DatetimeIndex(timestamps)).sort_index()
        delta_mm = accum.diff()
        delta_mm.iloc[0] = accum.iloc[0]  # first timestep: use raw value

        # Infer timestep from consecutive outputs (handles hourly or daily)
        if len(accum) >= 2:
            dt_seconds = (accum.index[1] - accum.index[0]).total_seconds()
        else:
            dt_seconds = 86400.0  # default to daily

        # Convert mm/timestep → m3/s
        q_cms = delta_mm * area_m2 / (dt_seconds * 1000.0)

        return q_cms

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion."""
        return variable_type not in ['streamflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['et', 'precipitation', 'streamflow']:
            return 'sum'
        else:
            return 'mean'
