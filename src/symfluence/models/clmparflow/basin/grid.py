# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regular ParFlow-CLM grid over a real basin.

- Mask: cells with at least half their area inside the basin outline, restricted to the 4-connected component
  holding the outlet.
- Elevation: the DEM averaged to the grid.
- Units: each cell takes the forcing unit covering most of its area; a cell is glacier when at least half of it
  is glacier-unit area, and then takes the majority glacier unit (land cells the majority land unit).
- Slopes: D4 priority flood from the outlet cell (the active cell nearest the pour point). Each cell slopes only
  towards its D4 downstream neighbour (ParFlow OverlandFlow, cell-centred slopes: Sx > 0 sends water to -x);
  magnitude = conditioned elevation drop / dx, at least ``min_slope``. The outlet slopes out of the domain.
- Optional (all off by default):
  - ``rotation``: the grid axes are rotated ``rotation`` degrees counter-clockwise from east about ``pivot`` (the
    basin centroid), so ParFlow's grid-aligned permeability tensor can follow a structural strike. Every input is
    sampled with the rotated transform; ParFlow itself only sees a regular grid.
  - ``kzone``: cells at least half inside a polygon (a permeability zone; its factor is a run parameter).
  - ``lake``: cells at least half inside a lake polygon. Their elevation is set to the median of the lake cells
    (the water surface) before the priority flood, so the lake slopes at ``min_slope`` towards its outlet; the
    run gives them their own Manning's n (``case``).

Arrays are stored with row j = 0 at the south edge (ParFlow y increases northwards).
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from symfluence.core.exceptions import GeospatialError

FINE = 10  # sub-cells per cell side for area fractions


@dataclass
class BasinGrid:
    """Grid fields, all [ny, nx] with j = 0 at the south edge."""

    mask: np.ndarray
    z: np.ndarray
    zc: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    unit: np.ndarray
    ice: np.ndarray
    icefrac: np.ndarray
    x0: float
    y0: float
    dx: float
    crs: str
    units: List[str]
    info: Dict
    rotation: float = 0.0  # degrees counter-clockwise from east; x0, y0 are then in the rotated frame
    pivot: Tuple[float, float] = (0.0, 0.0)  # rotation centre (CRS coordinates)
    kzone: Optional[np.ndarray] = None  # [ny, nx] bool: permeability-zone cells
    lake: Optional[np.ndarray] = None  # [ny, nx] bool: lake cells

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mask.shape

    def transform(self, fine: int = 1):
        """Affine from (column, row) of a north-up array (row 0 = north edge) to CRS coordinates; ``fine``
        subdivides each cell side."""
        return grid_transform(self.x0, self.y0 + self.shape[0] * self.dx, self.dx, self.rotation, self.pivot, fine)

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """CRS x, y of the cell centres, [ny, nx] with j = 0 at the south edge."""
        ny, nx = self.shape
        i, j = np.meshgrid(np.arange(nx) + .5, np.arange(ny) + .5)
        if not self.rotation:
            return self.x0 + i * self.dx, self.y0 + j * self.dx
        x, y = self.transform() * (i, ny - j)
        return np.asarray(x), np.asarray(y)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        extra: Dict[str, Any] = {}  # optional fields only when set, so default grids are written as before
        if self.rotation:
            extra.update(rotation=self.rotation, pivot=np.asarray(self.pivot))
        if self.kzone is not None:
            extra['kzone'] = self.kzone
        if self.lake is not None:
            extra['lake'] = self.lake
        np.savez_compressed(directory / 'grid.npz', mask=self.mask, z=self.z, zc=self.zc, sx=self.sx, sy=self.sy,
                            unit=self.unit, ice=self.ice, icefrac=self.icefrac, x0=self.x0, y0=self.y0, dx=self.dx,
                            **extra)
        (directory / 'grid.json').write_text(json.dumps(dict(self.info, crs=self.crs, units=self.units), indent=1))

    @classmethod
    def load(cls, directory: Path) -> 'BasinGrid':
        g = np.load(directory / 'grid.npz')
        info = json.loads((directory / 'grid.json').read_text())
        return cls(mask=g['mask'], z=g['z'], zc=g['zc'], sx=g['sx'], sy=g['sy'], unit=g['unit'], ice=g['ice'],
                   icefrac=g['icefrac'], x0=float(g['x0']), y0=float(g['y0']), dx=float(g['dx']),
                   crs=info['crs'], units=info['units'], info=info,
                   rotation=float(g['rotation']) if 'rotation' in g else 0.0,
                   pivot=(float(g['pivot'][0]), float(g['pivot'][1])) if 'pivot' in g else (0.0, 0.0),
                   kzone=g['kzone'] if 'kzone' in g else None, lake=g['lake'] if 'lake' in g else None)


def grid_transform(x0: float, y1: float, dx: float, rotation: float = 0.0, pivot: Tuple[float, float] = (0., 0.),
                   fine: int = 1):
    """Affine (column, row) -> CRS of a north-up grid of cell size ``dx / fine`` whose north-west corner is
    (x0, y1) in the frame rotated ``rotation`` degrees counter-clockwise about ``pivot``. Without rotation it is
    ``from_origin(x0, y1, dx / fine, dx / fine)``, as basin mode has always used."""
    from affine import Affine
    from rasterio.transform import from_origin

    d = dx / fine
    if not rotation:
        return from_origin(x0, y1, d, d)
    return (Affine.translation(*pivot) * Affine.rotation(rotation) * Affine.translation(x0 - pivot[0], y1 - pivot[1])
            * Affine.scale(d, -d))


def polygon_cells(polygons: Path, crs: str, fine_transform, shape: Tuple[int, int]) -> np.ndarray:
    """Fraction of each cell (north-up [ny, nx]) inside the polygons of ``polygons``; ``fine_transform`` is the
    grid's transform at ``FINE`` sub-cells per side."""
    import geopandas as gpd
    from rasterio.features import rasterize

    ny, nx = shape
    geoms = [g for g in gpd.read_file(polygons).to_crs(crs).geometry if g is not None and not g.is_empty]
    if not geoms:
        raise GeospatialError(f'no polygon in {polygons}')
    fine = rasterize([(g, 1) for g in geoms], out_shape=(ny * FINE, nx * FINE), transform=fine_transform,
                     fill=0).astype(float)
    return fine.reshape(ny, FINE, nx, FINE).mean(axis=(1, 3))


def _outlet_component(mask: np.ndarray, outlet: Tuple[int, int]) -> np.ndarray:
    keep = np.zeros_like(mask)
    stack = [outlet]
    keep[outlet] = True
    ny, nx = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if 0 <= q[0] < ny and 0 <= q[1] < nx and mask[q] and not keep[q]:
                keep[q] = True
                stack.append(q)
    return keep


def d4_slopes(z: np.ndarray, mask: np.ndarray, outlet: Tuple[int, int], dx: float,
              min_slope: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Priority-flood D4 slopes on a north-up grid (row 0 = north).

    Returns (sx, sy, conditioned elevation), still north-up. ``z`` outside the mask may be NaN; the outlet drains
    across a face to an outside neighbour, using that neighbour's raw elevation when finite.
    """
    ny, nx = mask.shape
    outside = [(dr, dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
               if not (0 <= outlet[0] + dr < ny and 0 <= outlet[1] + dc < nx) or not mask[outlet[0] + dr, outlet[1] + dc]]
    if not outside:
        raise GeospatialError('outlet cell is not on the domain edge')
    zc = np.where(mask, z, np.nan)
    down = np.zeros((ny, nx, 2), int)
    seen = np.zeros_like(mask)
    eps = 1e-3
    queue = [(zc[outlet], outlet)]
    seen[outlet] = True
    while queue:
        zz, (r, c) = heapq.heappop(queue)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if 0 <= q[0] < ny and 0 <= q[1] < nx and mask[q] and not seen[q]:
                seen[q] = True
                zc[q] = max(zc[q], zz + eps)
                down[q] = (-dr, -dc)
                heapq.heappush(queue, (zc[q], q))
    if not seen[mask].all():
        raise GeospatialError('mask is not 4-connected to the outlet')
    down[outlet] = outside[0]
    drop = np.zeros((ny, nx))
    for r, c in zip(*np.nonzero(mask)):
        dr, dc = down[r, c]
        if (r, c) != outlet:
            drop[r, c] = (zc[r, c] - zc[r + dr, c + dc]) / dx
        else:
            rr, cc = r + dr, c + dc
            inside = 0 <= rr < ny and 0 <= cc < nx
            drop[r, c] = (zc[r, c] - z[rr, cc]) / dx if inside and np.isfinite(z[rr, cc]) else 0.0
    s = np.maximum(drop, min_slope)
    # ParFlow: x = column (east), y = north = -row. Sx > 0 => flow to -x (west); Sy > 0 => flow to -y (south)
    sx = np.where(mask & (down[..., 1] == -1), s, 0.) - np.where(mask & (down[..., 1] == 1), s, 0.)
    sy = np.where(mask & (down[..., 0] == 1), s, 0.) - np.where(mask & (down[..., 0] == -1), s, 0.)
    return sx, sy, zc


def build_basin_grid(outline: Path, units_file: Path, unit_field: str, dem: Path, pour_point: Path,
                     crs: str, dx: float = 1000.0, ice_field: Optional[str] = None, ice_value: str = 'ice',
                     min_slope: float = 1e-4, rotation: float = 0.0, k_zone: Optional[Path] = None,
                     lake: Optional[Path] = None) -> BasinGrid:
    """Rasterise the basin, its forcing units and the DEM onto a regular grid with D4 overland slopes.

    ``rotation`` (degrees counter-clockwise from east) rotates the grid axes about the basin centroid; ``k_zone``
    and ``lake`` are polygon files whose cells (at least half inside, active only) are flagged on the grid.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, reproject

    basin = gpd.read_file(outline).to_crs(crs)
    units = gpd.read_file(units_file).to_crs(crs)
    pivot = (0.0, 0.0)
    if rotation:
        c = basin.union_all().centroid if hasattr(basin, 'union_all') else basin.unary_union.centroid
        pivot = (float(c.x), float(c.y))
        # bounds in the rotated frame: rotate the outline by -rotation about the pivot
        from shapely import affinity

        x0, y0, x1, y1 = gpd.GeoSeries([affinity.rotate(g, -rotation, origin=pivot) for g in basin.geometry]).total_bounds
    else:
        x0, y0, x1, y1 = basin.total_bounds
    x0 = np.floor(x0 / dx) * dx - dx
    y1 = np.ceil(y1 / dx) * dx + dx
    nx = int(np.ceil((x1 - x0) / dx)) + 1
    ny = int(np.ceil((y1 - y0) / dx)) + 1
    transform = grid_transform(x0, y1, dx, rotation, pivot)  # rows north -> south (of the rotated frame)
    fine_transform = grid_transform(x0, y1, dx, rotation, pivot, FINE)
    fine_shape = (ny * FINE, nx * FINE)

    def coarse(a: np.ndarray) -> np.ndarray:
        return a.reshape(ny, FINE, nx, FINE).mean(axis=(1, 3))

    inside = coarse(rasterize([(g, 1) for g in basin.geometry], out_shape=fine_shape, transform=fine_transform,
                              fill=0).astype(float))
    names = [str(u) for u in units[unit_field]]
    ids = {u: i + 1 for i, u in enumerate(names)}
    fine = rasterize([(g, ids[u]) for g, u in zip(units.geometry, names)], out_shape=fine_shape,
                     transform=fine_transform, fill=0)
    frac = np.stack([coarse((fine == k).astype(float)) for k in range(1, len(ids) + 1)])
    is_ice = (units[ice_field].astype(str) == ice_value).to_numpy() if ice_field else np.zeros(len(names), bool)
    ice_k = [k for k in range(len(names)) if is_ice[k]]
    land_k = [k for k in range(len(names)) if not is_ice[k]]
    icefrac = frac[ice_k].sum(0) / np.maximum(frac.sum(0), 1e-9) if ice_k else np.zeros((ny, nx))
    ice = icefrac >= .5
    land_unit = np.array(land_k)[frac[land_k].argmax(0)] if land_k else np.zeros((ny, nx), int)
    unit = np.where(ice, np.array(ice_k)[frac[ice_k].argmax(0)], land_unit) if ice_k else land_unit

    z = np.full((ny, nx), np.nan)
    with rasterio.open(dem) as src:
        reproject(rasterio.band(src, 1), z, dst_transform=transform, dst_crs=crs, resampling=Resampling.average,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    mask = (inside >= .5) & np.isfinite(z)
    gauge = gpd.read_file(pour_point).to_crs(crs).geometry.iloc[0]
    rr, cc = np.nonzero(mask)
    if rotation:
        gx, gy = ~transform * (gauge.x, gauge.y)  # gauge in (column, row) units
        dist = np.hypot(cc + .5 - gx, rr + .5 - gy)
    else:
        dist = np.hypot(x0 + (cc + .5) * dx - gauge.x, y1 - (rr + .5) * dx - gauge.y)
    outlet = (int(rr[dist.argmin()]), int(cc[dist.argmin()]))
    keep = _outlet_component(mask, outlet)
    dropped = int(mask.sum() - keep.sum())
    mask = keep
    flags: Dict[str, np.ndarray] = {}
    for key, polygons in (('kzone', k_zone), ('lake', lake)):
        if polygons is not None:
            cells = mask & (polygon_cells(Path(polygons), crs, fine_transform, (ny, nx)) >= .5)
            if not cells.any():
                raise GeospatialError(f'{key} polygon {polygons} covers no active cell')
            flags[key] = cells
    if 'lake' in flags:
        # level the lake at its median cell elevation (the water surface; shore cells averaged with land would
        # otherwise give the lake steep slopes), so lake cells slope at min_slope towards the lake outlet
        lake_level = float(np.median(z[flags['lake']]))
        z = np.where(flags['lake'], lake_level, z)
    sx, sy, zc = d4_slopes(z, mask, outlet, dx, min_slope)

    def flip(a: np.ndarray) -> np.ndarray:
        return a[::-1].copy()  # -> j = 0 south

    s = np.abs(sx) + np.abs(sy)
    info = dict(nx=nx, ny=ny, active=int(mask.sum()), area_km2=float(mask.sum() * dx * dx / 1e6),
                basin_km2=float(basin.area.sum() / 1e6), dropped_disconnected=dropped,
                outlet_rc_north_up=[outlet[0], outlet[1]], ice_cells=int((mask & ice).sum()),
                ice_km2_fractional=float((icefrac * mask).sum() * dx * dx / 1e6),
                z_range=[float(np.nanmin(z[mask])), float(np.nanmax(z[mask]))],
                unit_cells={names[k]: int(((unit == k) & mask).sum()) for k in range(len(names)) if ((unit == k) & mask).any()},
                min_slope=min_slope, slope_median=float(np.median(s[mask])))
    for key, polygons in (('kzone', k_zone), ('lake', lake)):
        if key in flags:
            info[f'{key}_cells'] = int(flags[key].sum())
            info[f'{key}_source'] = str(polygons)
            flags[key] = flip(flags[key])
    if 'lake' in flags:
        info['lake_level'] = lake_level
    if rotation:
        info.update(rotation_deg=float(rotation), pivot=list(pivot))
    return BasinGrid(mask=flip(mask), z=flip(np.where(mask, z, 0)), zc=flip(np.where(mask, zc, 0)), sx=flip(sx),
                     sy=flip(sy), unit=flip(np.where(mask, unit, -1)), ice=flip(mask & ice), icefrac=flip(icefrac),
                     x0=float(x0), y0=float(y1 - ny * dx), dx=float(dx), crs=str(crs), units=names, info=info,
                     rotation=float(rotation), pivot=pivot, kzone=flags.get('kzone'), lake=flags.get('lake'))
