"""Explicit gauge support for rasterized distributed catchments."""
from __future__ import annotations

import numpy as np
import pyflwdir
from rasterio.transform import rowcol
from shapely.geometry import Point

from symfluence.core.exceptions import ValidationError


def retain_gauge_cell(labels, transform, geometries, gauge_xy):
    """Retain a gauge-containing cell only when the vector basin covers the gauge."""
    point = Point(*gauge_xy)
    hits = [i for i, g in enumerate(geometries) if g.covers(point)]
    if not hits:
        raise ValidationError('Configured gauge is outside the elevation-HRU catchment')
    row, col = rowcol(transform, *gauge_xy)
    if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
        raise ValidationError('Configured gauge lies outside the native grid extent')
    result = labels.copy()
    added = result[row, col] == 0
    if added:
        result[row, col] = hits[0] + 1
    return result, int(row * labels.shape[1] + col), bool(added)


def flow_to_gauge(dem, transform, outlet_index, nodata=-9999.):
    """Condition one connected native catchment to its explicit gauge cell."""
    if not 0 <= outlet_index < dem.size or dem.ravel()[outlet_index] == nodata:
        raise ValidationError('Gauge outlet is not an active DEM cell')
    filled, d8 = pyflwdir.dem.fill_depressions(
        dem, nodata=nodata, idxs_pit=np.array([outlet_index], dtype=np.int64))
    flow = pyflwdir.from_array(d8, ftype='d8', transform=transform)
    active = np.isfinite(dem) & (dem != nodata)
    if not np.array_equal(flow.idxs_pit, [outlet_index]):
        raise ValidationError('Explicit gauge routing did not produce the requested single outlet')
    routed = flow.accuflux(active.astype(float)).ravel()[outlet_index]
    if not np.isclose(routed, active.sum()):
        raise ValidationError('Some active cells do not drain to the explicit gauge')
    uplift = filled[active] - dem[active]
    return flow, {'conditioning_max_fill_m': float(uplift.max()),
                  'conditioning_mean_fill_m': float(uplift.mean()),
                  'all_active_cells_reach_gauge': True}
