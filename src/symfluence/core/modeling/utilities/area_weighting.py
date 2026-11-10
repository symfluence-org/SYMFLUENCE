# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Area weights for SUMMA-style HRU/GRU runoff.

One rule, used by the streamflow evaluator, the SUMMA calibration worker and
the SUMMA postprocessor, for turning per-unit runoff depth rates (m/s) into
basin discharge (m3/s): ``sum_i runoff_i * area_i``.

SUMMA's ``averageRoutedRunoff`` is per GRU while its attributes file usually
carries only ``HRUarea`` (per HRU) plus the ``hru2gruId`` mapping. When
there are several HRUs per GRU the GRU areas are the HRU areas summed by
GRU -- *not* "GRU 0 times the whole basin area".
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def gru_areas_m2(attrs: Any) -> Optional[np.ndarray]:
    """Per-GRU areas (m2), in ``gruId`` order, from a SUMMA attributes dataset.

    Uses ``GRUarea`` when present; otherwise sums ``HRUarea`` by
    ``hru2gruId``; otherwise, when nHRU == nGRU (or there is no GRU
    dimension), treats HRUs as GRUs. Returns None when it cannot tell.
    """
    if 'GRUarea' in attrs:
        return np.asarray(attrs['GRUarea'].values, dtype=float).ravel()
    if 'HRUarea' not in attrs:
        return None
    hru_area = np.asarray(attrs['HRUarea'].values, dtype=float).ravel()
    if 'hru2gruId' in attrs and 'gruId' in attrs:
        hru2gru = np.asarray(attrs['hru2gruId'].values).ravel()
        gru_ids = np.asarray(attrs['gruId'].values).ravel()
        if hru2gru.size == hru_area.size:
            return np.array([hru_area[hru2gru == g].sum() for g in gru_ids], dtype=float)
    sizes = getattr(attrs, 'sizes', {})
    n_gru = sizes.get('gru')
    if n_gru is None or n_gru == hru_area.size:
        return hru_area
    return None


def area_weighted_discharge(var: Any, attrs: Any) -> Optional[Any]:
    """Basin discharge (m3/s): runoff rate (m/s) x unit area, summed over units.

    ``var`` is an xarray DataArray with a ``time`` dim and an ``hru`` or
    ``gru`` dim; ``attrs`` is a SUMMA attributes (or output) dataset holding
    the areas. Returns a 1-D DataArray over time, or None when matching areas
    are unavailable (callers decide the fallback).
    """
    import xarray as xr

    if 'hru' in var.dims and 'HRUarea' in attrs:
        areas = np.asarray(attrs['HRUarea'].values, dtype=float).ravel()
        if areas.size == var.sizes['hru']:
            return (var * xr.DataArray(areas, dims='hru')).sum(dim='hru')
    if 'gru' in var.dims:
        areas = gru_areas_m2(attrs)
        if areas is not None and areas.size == var.sizes['gru']:
            return (var * xr.DataArray(areas, dims='gru')).sum(dim='gru')
    return None
