# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP Result Extractor.

noah-owp-modular writes a single ``output.nc`` with all water/energy
variables on the model time axis.  Variable naming follows the NOAH-MP
convention (uppercase Fortran-style identifiers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor


def read_route_k(settings_dir) -> Optional[float]:
    """Calibrated Nash routing time constant (days) from ``route_k.txt``, if any."""
    if not settings_dir:
        return None
    f = Path(settings_dir) / 'route_k.txt'
    if not f.exists():
        return None
    try:
        return float(f.read_text().strip())
    except (OSError, ValueError):
        return None


def route_nash(series: pd.Series, k_days: Optional[float], n: int = 2) -> pd.Series:
    """Route a flow series through an n-reservoir Nash cascade.

    Each linear reservoir is y[t] = a*y[t-1] + (1-a)*x[t] with
    a = exp(-1/k). Volume-preserving; attenuates and delays the response.
    """
    import numpy as np

    if k_days is None or k_days <= 0 or len(series) < 2:
        return series
    a = float(np.exp(-1.0 / k_days))
    y = series.to_numpy(dtype=float)
    for _ in range(max(1, int(n))):
        out = np.empty_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = a * out[i - 1] + (1.0 - a) * y[i]
        y = out
    return pd.Series(y, index=series.index)


def noahmp_daily_discharge_cms(
    output_file: Path, area_m2: float, route_k: Optional[float] = None,
) -> pd.Series:
    """Daily basin discharge (m3/s) from a Noah-MP ``output.nc``.

    The one conversion shared by the calibration worker and the
    postprocessor: SFCRNOFF + UGDRNOFF (mm per output step, mean over any
    spatial dims) -> m3/s using the catchment area and the output step,
    daily mean, then the calibrated Nash cascade when ``route_k`` is set.
    """
    import numpy as np
    import xarray as xr

    with xr.open_dataset(output_file) as ds:
        components = [ds[v] for v in ('SFCRNOFF', 'UGDRNOFF') if v in ds.data_vars]
        if not components:
            raise ModelExecutionError(f"No SFCRNOFF/UGDRNOFF in {output_file}")
        runoff = sum(components)
        spatial_dims = [d for d in runoff.dims if d != 'time']
        if spatial_dims:
            runoff = runoff.mean(dim=spatial_dims)
        runoff_mm = np.asarray(runoff.values, dtype=float)
        times = pd.to_datetime(ds['time'].values)

    dt_seconds = 3600.0
    if len(times) > 1:
        dt_seconds = (times[1] - times[0]).total_seconds()
    q = pd.Series(runoff_mm * area_m2 / (1000.0 * dt_seconds), index=times)
    q = q.resample('D').mean()
    if route_k and route_k > 0:
        q = route_nash(q, route_k, n=2)
    return q.rename('NOAHMP_discharge_cms')


class NoahMPResultExtractor(ModelResultExtractor):
    """Result extraction for standalone Noah-MP (noah-owp-modular).

    Output file: single ``output.nc`` containing time series of water
    balance, energy balance, snow, and soil state variables.
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': ['output.nc'],
            'runoff': ['output.nc'],
            'et': ['output.nc'],
            'snow': ['output.nc'],
            'soil_moisture': ['output.nc'],
            'energy': ['output.nc'],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            'streamflow': ['SFCRNOFF', 'UGDRNOFF'],
            'runoff': ['SFCRNOFF', 'UGDRNOFF', 'QINSUR'],
            'et': ['ECAN', 'ETRAN', 'EDIR', 'LH'],
            'snow': ['SNEQV', 'SNOWH', 'QSNBOT', 'QSNSUB', 'QSNFRO'],
            'soil_moisture': ['SMC', 'SH2O', 'ZWT'],
            'energy': ['LH', 'FSH', 'GH', 'FSA', 'FIRA', 'TV', 'TG', 'T2M'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(self, output_file: Path, variable_type: str, **kwargs) -> pd.Series:
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        try:
            ds = xr.open_dataset(output_file)

            if variable_type in ('streamflow', 'runoff'):
                components = []
                for var_name in ['SFCRNOFF', 'UGDRNOFF']:
                    if var_name in ds.data_vars:
                        components.append(ds[var_name])
                if components:
                    data = sum(components)
                    series = data.to_series()
                    ds.close()
                    return series

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
            spatial_dims = [d for d in data.dims if d not in ['time']]
            if spatial_dims:
                agg = self.get_spatial_aggregation_method(variable_type)
                if agg == 'sum':
                    data = data.sum(dim=spatial_dims)
                elif agg == 'max':
                    data = data.max(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)
            series = data.to_series()
            ds.close()
            return series
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Error reading Noah-MP output file {output_file}: {e}"
            ) from e

    def extract_streamflow(
        self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs
    ) -> pd.Series:
        patterns = self.get_output_file_patterns()['streamflow']
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                return self.extract_variable(matches[0], 'streamflow', **kwargs)

        raise FileNotFoundError(
            f"No Noah-MP output found in {output_dir}"
        )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type in ('et', 'runoff', 'streamflow'):
            return 'sum'
        return 'mean'
