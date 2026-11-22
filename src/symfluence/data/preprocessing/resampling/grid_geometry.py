# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared cell edges reconstructed from projected structured-grid centres."""
from __future__ import annotations

import numpy as np
import shapely


def projected_cell_polygons(x, y):
    """Return row-major polygons; extrapolate outer edges by half a grid step.

    Adjacent cells share corners, including grids rotated in the native CRS.
    These are inferred footprints, not a substitute for authoritative cell bounds.
    """
    def corners(a):
        a = np.asarray(a, dtype=float)
        if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
            raise ValueError('Cell reconstruction needs finite 2D coordinates with at least two rows and columns')
        padded = np.pad(a, 1, mode='edge')
        padded[0, 1:-1] = 2*a[0] - a[1]
        padded[-1, 1:-1] = 2*a[-1] - a[-2]
        padded[:, 0] = 2*padded[:, 1] - padded[:, 2]
        padded[:, -1] = 2*padded[:, -2] - padded[:, -3]
        return (padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:])/4
    if np.shape(x) != np.shape(y):
        raise ValueError('Coordinate shapes differ')
    vx, vy = corners(x), corners(y)
    xy = np.stack([np.stack([vx[:-1, :-1], vy[:-1, :-1]], -1),
                   np.stack([vx[:-1, 1:], vy[:-1, 1:]], -1),
                   np.stack([vx[1:, 1:], vy[1:, 1:]], -1),
                   np.stack([vx[1:, :-1], vy[1:, :-1]], -1)], -2)
    cells = shapely.polygons(xy.reshape(-1, 4, 2))
    if not np.all(shapely.is_valid(cells)) or np.any(shapely.area(cells) <= 0):
        raise ValueError('Invalid reconstructed forcing cells')
    return cells


def curvilinear_cell_polygons(lon2d, lat2d, rows=None, cols=None, nominal_step=None):
    """Footprints centred on each cell of a 2D lon/lat grid (EPSG:4326 degrees).

    Corners are the means of the four surrounding cell centres, so every
    polygon is centred on (and tagged by) its own cell rather than spanning
    to its neighbours' centres. Only the window ``lon2d[r0:r1, c0:c1]`` is
    returned (row-major, shape ``(r1 - r0, c1 - c0)``); one ring of real
    neighbours around the window is used where the grid has one, and the
    outer grid edge is extrapolated by half a step.

    Args:
        lon2d, lat2d: 2D centre coordinates in degrees.
        rows, cols: ``(start, stop)`` index ranges of the window (default all).
        nominal_step: Grid spacing in degrees, used only when the grid has a
            single row or column and the other axis gives no spacing (1x1 grid).
    """
    lon = np.asarray(lon2d, dtype=float)
    lat = np.asarray(lat2d, dtype=float)
    if lon.ndim != 2 or lon.shape != lat.shape:
        raise ValueError('Curvilinear cells need matching 2D lon/lat arrays')
    ny, nx = lon.shape
    r0, r1 = rows if rows is not None else (0, ny)
    c0, c1 = cols if cols is not None else (0, nx)
    if not (0 <= r0 < r1 <= ny and 0 <= c0 < c1 <= nx):
        raise ValueError(f'Invalid cell window rows={rows} cols={cols} for grid {lon.shape}')
    p0, p1 = max(r0 - 1, 0), min(r1 + 1, ny)
    q0, q1 = max(c0 - 1, 0), min(c1 + 1, nx)
    sub_lon = lon[p0:p1, q0:q1]
    sub_lat = lat[p0:p1, q0:q1]
    # Keep footprints continuous where a window straddles the antimeridian.
    sub_lon = np.unwrap(np.unwrap(sub_lon, period=360.0, axis=1), period=360.0, axis=0)
    r_off, c_off = r0 - p0, c0 - q0

    if min(sub_lon.shape) < 2:
        if max(sub_lon.shape) >= 2:
            dlon = np.diff(sub_lon, axis=int(sub_lon.shape[1] >= 2))
            dlat = np.diff(sub_lat, axis=int(sub_lon.shape[1] >= 2))
            other = float(np.nanmean(np.hypot(dlon * np.cos(np.radians(sub_lat.mean())), dlat)))
        else:
            other = None
        step = other if other is not None and np.isfinite(other) and other > 0 else nominal_step
        if not step:
            raise ValueError('Cannot infer cell size of a single-cell grid without nominal_step')
        coslat = max(np.cos(np.radians(float(np.nanmean(sub_lat)))), 1e-6)
        if sub_lon.shape[0] == 1:  # single row: add synthetic rows south/north
            sub_lon = np.vstack([sub_lon, sub_lon, sub_lon])
            sub_lat = np.vstack([sub_lat - step, sub_lat, sub_lat + step])
            r_off += 1
        if sub_lon.shape[1] == 1:  # single column: add synthetic columns west/east
            dl = step / coslat
            sub_lon = np.hstack([sub_lon - dl, sub_lon, sub_lon + dl])
            sub_lat = np.hstack([sub_lat, sub_lat, sub_lat])
            c_off += 1

    cells = projected_cell_polygons(sub_lon, sub_lat).reshape(sub_lon.shape)
    return cells[r_off:r_off + (r1 - r0), c_off:c_off + (c1 - c0)]
