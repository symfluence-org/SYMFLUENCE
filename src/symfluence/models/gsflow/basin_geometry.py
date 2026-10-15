# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Real-basin GSFLOW geometry: HRUs, MODFLOW grid, PRMS-MODFLOW mapping and SFR.

Built from the delineated subbasins, river network and DEM (plus optional
glacier outlines), replacing the schematic square grid:

* HRUs: subbasin x elevation band x {land, glacier}; slivers below a minimum
  area are merged into a neighbour of the same subbasin.
* MODFLOW grid: square cells over the basin bounding box; a cell is active
  when it overlaps the basin. Land surface (TOP) is the DEM mean over the
  cell's basin part.
* Gravity-reservoir mapping from true HRU x cell overlaps: ``cell_pct`` is a
  fraction of cell area, ``hru_pct`` a fraction of HRU area (sums to 1).
* SFR: one segment per river-network link, numbered upstream-first; reaches
  are the link pieces in each cell, ordered downstream, with streambed tops
  below the DEM and strictly decreasing downstream.
* Glacier HRUs of a subbasin drain down a ``tohru`` chain to the lowest one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from symfluence.core.exceptions import ConfigurationError, ValidationError


@dataclass
class BasinGeometry:
    hrus: gpd.GeoDataFrame
    nrow: int
    ncol: int
    cell_size: float
    origin: tuple            # (x of left edge, y of top edge) in ``crs``
    active: np.ndarray       # (nrow, ncol) bool
    top: np.ndarray          # (nrow, ncol) land surface, NaN where inactive
    gvr: pd.DataFrame        # hru_id, cell_id, cell_pct, hru_pct
    segments: pd.DataFrame   # seg, linkno, outseg, gru_id
    reaches: pd.DataFrame    # seg, ireach, row, col, length, strtop, slope
    cell_seg: np.ndarray     # (nrow, ncol) segment receiving cell runoff (0 inactive)
    crs: object = None
    notes: List[str] = field(default_factory=list)

    @property
    def nhru(self) -> int:
        return len(self.hrus)

    def cell_id(self, row: int, col: int) -> int:
        """MODFLOW cell number (1-based, row-major) of 1-based row/col."""
        return (row - 1) * self.ncol + col


def _dem_read(dem_path: Path, geom, crs):
    """DEM values and transform inside ``geom`` (given in ``crs``)."""
    import rasterio
    from rasterio.mask import mask
    with rasterio.open(dem_path) as src:
        g = gpd.GeoSeries([geom], crs=crs).to_crs(src.crs).iloc[0]
        arr, transform = mask(src, [g], crop=True, filled=True, nodata=np.nan, all_touched=False)
        dem_crs = src.crs
    z = arr[0].astype('float64')
    z[(z < -100.0) | ~np.isfinite(z)] = np.nan
    return z, transform, dem_crs


def _mean_elevation(dem_path: Path, geom, crs) -> float:
    try:
        z, _, _ = _dem_read(dem_path, geom, crs)
    except ValueError:      # geometry smaller than a DEM pixel
        z = np.array([np.nan])
    if np.isfinite(z).any():
        return float(np.nanmean(z))
    # Fall back to the pixel under the representative point.
    import rasterio
    with rasterio.open(dem_path) as src:
        p = gpd.GeoSeries([geom.representative_point()], crs=crs).to_crs(src.crs).iloc[0]
        return float(next(src.sample([(p.x, p.y)]))[0])


def _band_polygons(dem_path: Path, geom, crs, band_size: float) -> gpd.GeoDataFrame:
    """Polygons of DEM elevation bands (floor(z / band_size)) inside ``geom``."""
    from rasterio.features import shapes
    from shapely.geometry import shape
    z, transform, dem_crs = _dem_read(dem_path, geom, crs)
    band = np.where(np.isfinite(z), np.floor(z / band_size), -9999).astype('int32')
    polys = [(shape(s), int(v)) for s, v in shapes(band, mask=band != -9999, transform=transform)]
    if not polys:
        raise ValidationError('No DEM cells inside subbasin geometry')
    g = gpd.GeoDataFrame({'band': [v for _, v in polys]}, geometry=[p for p, _ in polys], crs=dem_crs)
    g = g.to_crs(crs).dissolve(by='band', as_index=False)
    g['geometry'] = g.geometry.buffer(0).intersection(geom)
    return g[~g.geometry.is_empty]


def _merge_slivers(pieces: gpd.GeoDataFrame, min_area: float, min_glacier_area: float) -> gpd.GeoDataFrame:
    """Merge pieces smaller than the class minimum into the nearest-band piece of
    the same subbasin, preferring the same land/glacier class. Glacier pieces use
    a smaller minimum so ice area is preserved for the mass-balance constraint."""
    pieces = pieces.copy()
    while True:
        pieces['area'] = pieces.geometry.area
        limit = np.where(pieces.glacier == 1, min_glacier_area, min_area)
        small = pieces[pieces.area < limit].sort_values('area')
        if small.empty:
            return pieces.drop(columns='area')
        i = small.index[0]
        p = pieces.loc[i]
        cand = pieces[(pieces.gru_id == p.gru_id) & (pieces.index != i)]
        if cand.empty:
            return pieces.drop(columns='area')   # a whole tiny subbasin stays one HRU
        same = cand[cand.glacier == p.glacier]
        pool = same if not same.empty else cand
        j = (pool.band - p.band).abs().sort_values(kind='stable').index[0]
        pieces.loc[j, 'geometry'] = pieces.loc[j, 'geometry'].union(p.geometry)
        pieces = pieces.drop(index=i)


def _orient_downstream(line, row):
    """TauDEM links: DOUTSTART is the start's distance to the outlet."""
    if {'DOUTSTART', 'DOUTEND'} <= set(row.index) and row['DOUTSTART'] < row['DOUTEND']:
        from shapely.geometry import LineString
        return LineString(list(line.coords)[::-1])
    return line


def _as_line(geom):
    from shapely.ops import linemerge
    if geom.geom_type == 'MultiLineString':
        merged = linemerge(geom)
        if merged.geom_type == 'LineString':
            return merged
        return max(merged.geoms, key=lambda g: g.length)
    return geom


def build_basin_geometry(subbasins: gpd.GeoDataFrame, network: gpd.GeoDataFrame, dem_path: Path,
                         crs, cell_size: float = 2000.0, band_size: float = 200.0,
                         min_hru_area_km2: float = 2.0, min_glacier_hru_area_km2: float = 0.25,
                         glacier_outlines: Optional[gpd.GeoDataFrame] = None,
                         channel_depth: float = 1.0, min_reach_length: float = 10.0) -> BasinGeometry:
    """Build the full real-basin geometry (see module docstring)."""
    if cell_size <= 0 or band_size <= 0:
        raise ConfigurationError('GSFLOW cell size and elevation band size must be positive')
    for col in ('GRU_ID', 'gru_to_seg'):
        if col not in subbasins:
            raise ConfigurationError(f'Subbasin shapefile lacks {col}')
    for col in ('LINKNO', 'DSLINKNO'):
        if col not in network:
            raise ConfigurationError(f'River network lacks {col}')
    sub = subbasins.to_crs(crs)
    net = network.to_crs(crs)
    basin = sub.geometry.union_all() if hasattr(sub.geometry, 'union_all') else sub.unary_union
    ice = None
    if glacier_outlines is not None and len(glacier_outlines):
        ice_all = glacier_outlines.to_crs(crs).geometry
        ice = (ice_all.union_all() if hasattr(ice_all, 'union_all') else ice_all.unary_union).intersection(basin)
    notes: List[str] = []

    # --- segments: one per link, numbered upstream first -------------------
    links = net.set_index('LINKNO')
    order: List[int] = []
    pending = set(links.index)
    while pending:
        ready = [k for k in pending if not any(links.loc[u, 'DSLINKNO'] == k for u in pending if u != k)]
        if not ready:
            raise ValidationError('River network contains a cycle')
        for k in sorted(ready):
            order.append(k)
            pending.discard(k)
    seg_of = {k: i + 1 for i, k in enumerate(order)}
    outlets = [k for k in order if links.loc[k, 'DSLINKNO'] not in seg_of]
    if len(outlets) != 1:
        raise ValidationError(f'River network must have exactly one outlet, found {len(outlets)}')
    segments = pd.DataFrame({'seg': [seg_of[k] for k in order], 'linkno': order,
                             'outseg': [seg_of.get(links.loc[k, 'DSLINKNO'], 0) for k in order]})
    gru_seg = {}
    for _, s in sub.iterrows():
        if s['gru_to_seg'] not in seg_of:
            raise ValidationError(f"Subbasin {s['GRU_ID']} drains to unknown link {s['gru_to_seg']}")
        gru_seg[s['GRU_ID']] = seg_of[s['gru_to_seg']]
    segments['gru_id'] = segments.seg.map({v: k for k, v in gru_seg.items()})

    # --- HRUs: subbasin x band x land/glacier --------------------------------
    pieces = []
    for _, s in sub.iterrows():
        bands = _band_polygons(dem_path, s.geometry, crs, band_size)
        for _, b in bands.iterrows():
            parts = [(0, b.geometry)]
            if ice is not None and not ice.is_empty:
                parts = [(0, b.geometry.difference(ice)), (1, b.geometry.intersection(ice))]
            for glacier, geom in parts:
                if not geom.is_empty and geom.area > 0:
                    pieces.append({'gru_id': s['GRU_ID'], 'band': int(b.band), 'glacier': glacier, 'geometry': geom})
    hrus = gpd.GeoDataFrame(pieces, crs=crs)
    hrus = _merge_slivers(hrus, min_hru_area_km2 * 1e6, min_glacier_hru_area_km2 * 1e6).sort_values(['gru_id', 'glacier', 'band'])
    hrus = hrus.reset_index(drop=True)
    hrus['hru_id'] = np.arange(1, len(hrus) + 1)
    hrus['seg'] = hrus.gru_id.map(gru_seg)
    hrus['area_m2'] = hrus.geometry.area
    hrus['elev'] = [_mean_elevation(dem_path, g, crs) for g in hrus.geometry]
    ll = hrus.geometry.representative_point().to_crs(4326)
    hrus['lat'], hrus['lon'] = ll.y.to_numpy(), ll.x.to_numpy()
    lost = abs(hrus.area_m2.sum() - basin.area) / basin.area
    if lost > 1e-3:
        raise ValidationError(f'HRUs cover {100 * (1 - lost):.2f}% of the basin; geometry is inconsistent')

    # --- glacier routing: chain down each subbasin's glacier HRUs ------------
    hrus['tohru'] = 0
    hrus['hru_length_km'] = 0.0
    hrus['hru_width_km'] = 0.0
    for gru, grp in hrus[hrus.glacier == 1].groupby('gru_id'):
        chain = grp.sort_values('elev', ascending=False).index.tolist()
        for up, down in zip(chain[:-1], chain[1:]):
            hrus.loc[up, 'tohru'] = int(hrus.loc[down, 'hru_id'])
        for i in chain:
            length = float(np.sqrt(hrus.loc[i, 'area_m2'])) / 1000.0
            hrus.loc[i, 'hru_length_km'] = length
            hrus.loc[i, 'hru_width_km'] = hrus.loc[i, 'area_m2'] / 1e6 / length

    # --- MODFLOW grid ---------------------------------------------------------
    minx, miny, maxx, maxy = basin.bounds
    ncol = int(np.ceil((maxx - minx) / cell_size))
    nrow = int(np.ceil((maxy - miny) / cell_size))
    x0, y0 = minx, maxy
    cells = []
    for r in range(nrow):
        for c in range(ncol):
            cells.append({'row': r + 1, 'col': c + 1, 'cell_id': r * ncol + c + 1,
                          'geometry': box(x0 + c * cell_size, y0 - (r + 1) * cell_size,
                                          x0 + (c + 1) * cell_size, y0 - r * cell_size)})
    grid = gpd.GeoDataFrame(cells, crs=crs)
    inside = grid.geometry.intersection(basin)
    grid['basin_area'] = inside.area
    active = np.zeros((nrow, ncol), bool)
    top = np.full((nrow, ncol), np.nan)
    for i, g in grid[grid.basin_area > 0].iterrows():
        active[g.row - 1, g.col - 1] = True
        top[g.row - 1, g.col - 1] = _mean_elevation(dem_path, inside[i], crs)

    # --- gravity reservoirs: true HRU x cell overlaps ------------------------
    ov = gpd.overlay(hrus[['hru_id', 'geometry']], grid[grid.basin_area > 0][['cell_id', 'geometry']],
                     how='intersection', keep_geom_type=True)
    ov['a'] = ov.geometry.area
    ov = ov[ov.a > 1.0]
    gvr = ov.groupby(['hru_id', 'cell_id'], as_index=False).a.sum()
    gvr['cell_pct'] = gvr.a / (cell_size * cell_size)
    gvr['hru_pct'] = gvr.a / gvr.hru_id.map(gvr.groupby('hru_id').a.sum())
    gvr = gvr.drop(columns='a').sort_values(['hru_id', 'cell_id']).reset_index(drop=True)
    if set(gvr.hru_id) != set(hrus.hru_id):
        raise ValidationError('Some HRUs do not overlap any active MODFLOW cell')

    # --- cell runoff: segment of the subbasin covering most of the cell -----
    cs = gpd.overlay(sub[['GRU_ID', 'geometry']], grid[grid.basin_area > 0][['row', 'col', 'geometry']],
                     how='intersection', keep_geom_type=True)
    cs['a'] = cs.geometry.area
    cell_seg = np.zeros((nrow, ncol), int)
    for (r, c), grp in cs.groupby(['row', 'col']):
        cell_seg[r - 1, c - 1] = gru_seg[grp.sort_values('a').GRU_ID.iloc[-1]]

    # --- SFR reaches ----------------------------------------------------------
    import rasterio
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        rows: List[Dict] = []
        for k in order:
            row = links.loc[k]
            line = _orient_downstream(_as_line(row.geometry), row)
            if line.length < min_reach_length:
                raise ValidationError(f'River link {k} is shorter than {min_reach_length} m')
            # A piece belongs to the cell holding its midpoint (half-open cells),
            # so a river running along a cell edge is counted once, not twice.
            pieces_on_line: List[tuple] = []
            for _, g in grid[grid.geometry.intersects(line)].iterrows():
                piece = line.intersection(g.geometry)
                for p in getattr(piece, 'geoms', [piece]):
                    if p.geom_type != 'LineString' or p.length <= 0:
                        continue
                    mid = p.interpolate(0.5, normalized=True)
                    c = min(max(int((mid.x - x0) // cell_size) + 1, 1), ncol)
                    r = min(max(int((y0 - mid.y) // cell_size) + 1, 1), nrow)
                    if (r, c) == (g.row, g.col):
                        pieces_on_line.append((line.project(mid), r, c, p))
            pieces_on_line.sort(key=lambda t: t[0])
            merged: List[list] = []
            for pos, r, c, p in pieces_on_line:
                if not active[r - 1, c - 1]:
                    notes.append(f'link {k}: {p.length:.0f} m outside the active grid skipped')
                    continue
                if merged and (merged[-1][1:3] == [r, c] or p.length < min_reach_length):
                    merged[-1][3] += p.length
                    continue
                merged.append([pos, r, c, p.length, p])
            slope = max(float(row.get('Slope', 1e-4) or 1e-4), 1e-4)
            for i, (pos, r, c, length, p) in enumerate(merged, 1):
                mid = gpd.GeoSeries([p.interpolate(0.5, normalized=True)], crs=crs).to_crs(dem_crs).iloc[0]
                z = float(next(src.sample([(mid.x, mid.y)]))[0])
                if not np.isfinite(z) or z < -100:
                    z = top[r - 1, c - 1]
                rows.append({'seg': seg_of[k], 'ireach': i, 'row': r, 'col': c, 'length': length,
                             'z': min(z, top[r - 1, c - 1]) - channel_depth, 'slope': slope})
    reaches = pd.DataFrame(rows)
    if reaches.empty:
        raise ValidationError('No SFR reaches inside the active MODFLOW grid')
    # Streambed tops follow the DEM (minus channel depth) and are made strictly
    # decreasing downstream by a small step only; link slopes are not used to
    # extrapolate, which overshoots where TauDEM headwater slopes are steep.
    reaches['strtop'] = np.nan
    seg_end: Dict[int, float] = {}
    for k in order:
        s = seg_of[k]
        ups = [seg_end[seg_of[u]] for u in order if links.loc[u, 'DSLINKNO'] == k and seg_of[u] in seg_end]
        prev = min(ups) if ups else np.inf
        for i in reaches.index[reaches.seg == s]:
            z = min(float(reaches.at[i, 'z']), prev - 0.01)
            reaches.at[i, 'strtop'] = z
            prev = z
        seg_end[s] = prev
    reaches = reaches.drop(columns='z')
    return BasinGeometry(hrus=hrus, nrow=nrow, ncol=ncol, cell_size=float(cell_size), origin=(x0, y0),
                         active=active, top=top, gvr=gvr, segments=segments, reaches=reaches,
                         cell_seg=cell_seg, crs=crs, notes=notes)
