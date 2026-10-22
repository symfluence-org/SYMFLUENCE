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
