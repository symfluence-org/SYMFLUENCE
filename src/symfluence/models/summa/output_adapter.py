# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SUMMA standard output-variable adapter (see core.modeling.output_variables)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R

SECONDS_PER_DAY = 86400.0


def _find(sim_dir: Path, names: Iterable[str]) -> Optional[Path]:
    """First SUMMA NetCDF in ``sim_dir`` (daily files first) holding any of ``names``."""
    names = list(names)
    files = sorted(sim_dir.glob('*_day.nc')) + sorted(
        p for p in sim_dir.glob('*.nc') if not p.name.endswith('_day.nc'))
    for path in files:
        with xr.open_dataset(path) as ds:
            if any(n in ds.variables for n in names):
                return path
    return None


def _spatial_mean(da: xr.DataArray, ds: xr.Dataset) -> pd.Series:
    """Mean over hru/gru, area-weighted when ``HRUarea`` / ``GRUarea`` is in the file."""
    da = da.where(da > -9998)
    for dim, area_name in (('hru', 'HRUarea'), ('gru', 'GRUarea')):
        if dim in da.dims:
            if area_name in ds.variables and ds[area_name].dims == (dim,):
                w = ds[area_name].astype(float)
                da = (da * w).sum(dim) / w.sum()
            else:
                da = da.mean(dim)
    extra = [d for d in da.dims if d != 'time']
    if extra:
        da = da.isel({d: 0 for d in extra})
    return da.to_series()


@R.output_adapters.add('SUMMA')
class SUMMAOutputAdapter(OutputVariableAdapter):
    """SUMMA NetCDF outputs (``*_day.nc`` preferred, then other ``*.nc``)."""

    MODEL_NAME = 'SUMMA'
    SUPPORTED = {
        'glacier_storage': 'basin__GlacierStorage (Gt, ice + firn + snow on glacier), summed over gru',
        'glacier_area': 'basin__GlacierArea (m2), summed over gru',
        'swe': 'scalarSWE (kg m-2), HRU-area weighted when HRUarea is written, else mean',
        'snow_covered_fraction': 'scalarGroundSnowFraction (may be binary per HRU in some builds)',
        'et': 'scalarTotalET (kg m-2 s-1, negative = loss), sign flipped, x 86400',
    }

    def _open(self, sim_dir: Path, var: str) -> xr.Dataset:
        path = _find(sim_dir, [var])
        if path is None:
            raise EvaluationError(f'{var} not found in SUMMA output under {sim_dir}')
        return xr.open_dataset(path)

    def _glacier_storage(self, sim_dir: Path) -> pd.Series:
        with self._open(sim_dir, 'basin__GlacierStorage') as ds:
            da = ds['basin__GlacierStorage'].astype(float)
            da = da.where(da > -9998)
            if 'gru' in da.dims:
                da = da.sum('gru', min_count=1)
            return da.to_series() * 1e9  # Gt water -> m3 w.e.

    def _glacier_area(self, sim_dir: Path) -> pd.Series:
        with self._open(sim_dir, 'basin__GlacierArea') as ds:
            da = ds['basin__GlacierArea'].astype(float)
            da = da.where(da > -9998)
            if 'gru' in da.dims:
                da = da.sum('gru', min_count=1)
            return da.to_series() / 1e6  # m2 -> km2

    def _swe(self, sim_dir: Path) -> pd.Series:
        with self._open(sim_dir, 'scalarSWE') as ds:
            return _spatial_mean(ds['scalarSWE'].astype(float), ds)

    def _snow_covered_fraction(self, sim_dir: Path) -> pd.Series:
        with self._open(sim_dir, 'scalarGroundSnowFraction') as ds:
            return _spatial_mean(ds['scalarGroundSnowFraction'].astype(float), ds).clip(0, 1)

    def _et(self, sim_dir: Path) -> pd.Series:
        with self._open(sim_dir, 'scalarTotalET') as ds:
            series = _spatial_mean(ds['scalarTotalET'].astype(float), ds)
        return pd.Series(-np.asarray(series, dtype=float) * SECONDS_PER_DAY, index=series.index)
