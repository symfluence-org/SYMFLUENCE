# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Repair SUMMA restart files that SUMMA itself refuses to start from.

SUMMA can end a run with a thin, nearly melted top snow layer whose water
content is above the limit that ``check_icond`` enforces when the same file is
read back as initial conditions::

    cannot initialize the model with total water fraction [liquid + ice] > 0.80

so a restart written by SUMMA cannot be used to continue the run.  For snow
layers ``check_icond`` computes the total water fraction as

    theta = mLayerVolFracIce * (iden_ice / iden_water) + mLayerVolFracLiq

with ``iden_ice = 917`` and ``iden_water = 1000`` kg m-3, and also refuses a
layer whose ice fraction alone is above 0.80.

:func:`repair_snow_restart` thickens each offending snow layer until both
quantities are at :data:`SNOW_TARGET_FRAC` (0.79).  All volumetric quantities
of the layer (liquid, ice, total water, enthalpy) are divided by the same
factor the depth is multiplied by, so the layer's water and ice mass and its
energy are unchanged.  The snow interface heights above the soil surface, the
layer mid-point heights and ``scalarSnowDepth`` are recomputed.  Layers that
SUMMA accepts are left untouched, so the repair is a no-op on valid files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import xarray as xr

from symfluence.core.exceptions import ValidationError

#: SUMMA's ``check_icond`` limit on snow total water and ice fractions.
SNOW_MAX_FRAC = 0.80
#: Fraction a repaired layer is brought down to (just inside the limit).
SNOW_TARGET_FRAC = 0.79
#: SUMMA's lower limit on snow ice fraction (``check_icond``).
SNOW_MIN_ICE_FRAC = 0.05
#: Intrinsic densities used by SUMMA (``multiconst``), kg m-3.
IDEN_ICE = 917.0
IDEN_WATER = 1000.0

_REQUIRED = ('mLayerVolFracLiq', 'mLayerVolFracIce', 'mLayerDepth',
             'iLayerHeight', 'mLayerHeight', 'nSnow')
_LAYER_VARS = ('mLayerVolFracLiq', 'mLayerVolFracIce', 'mLayerVolFracWat',
               'mLayerEnthalpy', 'mLayerDepth', 'iLayerHeight', 'mLayerHeight')


def snow_total_water_fraction(vol_frac_liq: np.ndarray, vol_frac_ice: np.ndarray) -> np.ndarray:
    """Total water fraction of a snow layer as SUMMA's ``check_icond`` computes it."""
    return vol_frac_ice * (IDEN_ICE / IDEN_WATER) + vol_frac_liq


def _layer_hru(da: xr.DataArray) -> np.ndarray:
    """Values of a (layer, hru) variable as a writable (layer, hru) array."""
    if 'hru' not in da.dims or da.ndim != 2:
        raise ValidationError(f"expected a (layer, hru) variable, got {da.name} with dims {da.dims}")
    layer_dim = next(d for d in da.dims if d != 'hru')
    return np.array(da.transpose(layer_dim, 'hru').values, dtype=float)


def repair_snow_restart(path: Union[str, Path]) -> List[str]:
    """Make a SUMMA restart readable by SUMMA again, in place.

    Args:
        path: SUMMA restart / initial-condition NetCDF file.

    Returns:
        One note per repaired snow layer (empty if nothing needed repair, in
        which case the file is not rewritten).  A file that is not a SUMMA
        layered state (missing variables) is left alone and returns ``[]``.
    """
    path = Path(path)
    with xr.open_dataset(path) as src:
        if any(v not in src for v in _REQUIRED) or 'hru' not in src.dims:
            return []
        ds = src.load()

    arrays = {v: _layer_hru(ds[v]) for v in _LAYER_VARS if v in ds}
    liq, ice, depth = arrays['mLayerVolFracLiq'], arrays['mLayerVolFracIce'], arrays['mLayerDepth']
    ifc, mid = arrays['iLayerHeight'], arrays['mLayerHeight']
    scaled = [v for v in ('mLayerVolFracLiq', 'mLayerVolFracIce', 'mLayerVolFracWat', 'mLayerEnthalpy')
              if v in arrays]
    n_snow = np.asarray(ds['nSnow'].values).ravel().astype(int)
    # mLayerHeight may carry one more slot than mLayerDepth (a leading fill value)
    mid_offset = mid.shape[0] - depth.shape[0]

    notes: List[str] = []
    for h in range(depth.shape[1]):
        changed = False
        for k in range(n_snow[h]):
            theta = float(snow_total_water_fraction(liq[k, h], ice[k, h]))
            worst = max(theta, float(ice[k, h]))
            if worst <= SNOW_MAX_FRAC:
                continue
            f = SNOW_TARGET_FRAC / worst
            note = (f"hru index {h}, snow layer {k}: total water fraction {theta:.3f}, "
                    f"ice fraction {ice[k, h]:.3f}; depth {depth[k, h] * 1e3:.2f} -> "
                    f"{depth[k, h] / f * 1e3:.2f} mm")
            for v in scaled:
                arrays[v][k, h] *= f
            depth[k, h] /= f
            if ice[k, h] < SNOW_MIN_ICE_FRAC:
                note += f" (ice fraction now {ice[k, h]:.3f}, below SUMMA's {SNOW_MIN_ICE_FRAC})"
            notes.append(note)
            changed = True
        if changed:
            # Snow interfaces are measured downward from the snow surface to the
            # soil surface at iLayerHeight[nSnow]; rebuild them bottom-up.
            for k in range(n_snow[h] - 1, -1, -1):
                ifc[k, h] = ifc[k + 1, h] - depth[k, h]
                mid[k + mid_offset, h] = 0.5 * (ifc[k, h] + ifc[k + 1, h])
            if 'scalarSnowDepth' in ds:
                snow_depth = np.array(ds['scalarSnowDepth'].values, dtype=float)
                snow_depth.reshape(-1)[h] = depth[:n_snow[h], h].sum()
                ds['scalarSnowDepth'].values = snow_depth

    if not notes:
        return notes
    for name, values in arrays.items():
        da = ds[name]
        layer_dim = next(d for d in da.dims if d != 'hru')
        ds[name].values = xr.DataArray(values, dims=(layer_dim, 'hru')).transpose(*da.dims).values.astype(da.dtype)
    tmp = path.with_name(path.name + '.repair.tmp')
    ds.to_netcdf(tmp)
    tmp.replace(path)
    return notes
