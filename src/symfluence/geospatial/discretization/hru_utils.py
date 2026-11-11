# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared building blocks for raster-based HRU discretization.

- reading a raster window for one GRU with separate "inside the GRU" and
  "has data" masks, and filling nodata holes so the HRUs tile the GRU;
- merging HRUs smaller than ``MIN_HRU_SIZE``;
- checking that HRU areas add up to the GRU area.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import geopandas as gpd
import numpy as np
from rasterio.mask import raster_geometry_mask
from scipy import ndimage
from shapely.ops import unary_union

from symfluence.geospatial.raster_utils import valid_pixel_mask

#: Relative HRU-vs-GRU area difference above which a warning is raised
AREA_CONSERVATION_TOLERANCE = 1e-3


def read_zone_window(
    src: Any, geometry: Any, *, all_touched: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Read the raster window covering *geometry*.

    Unlike ``rasterio.mask.mask(filled=False)`` this keeps "outside the
    geometry" and "nodata" apart.

    Returns:
        ``(data, inside, valid, transform)`` — the window values, the mask of
        pixels belonging to the geometry, the mask of those that hold data,
        and the window transform.
    """
    shape_mask, transform, window = raster_geometry_mask(
        src, [geometry], all_touched=all_touched, crop=True
    )
    data = src.read(1, window=window)
    inside = ~shape_mask
    valid = inside & valid_pixel_mask(data, src.nodata)
    return data, inside, valid, transform


def fill_nodata_from_nearest(
    data: np.ndarray, valid: np.ndarray, inside: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Give nodata pixels inside a zone the value of the nearest valid pixel.

    Returns:
        ``(filled, n_filled)``. *data* is returned unchanged (``n_filled`` 0)
        when there is nothing to fill or no valid pixel to fill from.
    """
    holes = inside & ~valid
    n_holes = int(holes.sum())
    if n_holes == 0 or not valid.any():
        return data, 0
    _, (rows, cols) = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = np.array(data, copy=True)
    filled[holes] = data[rows[holes], cols[holes]]
    return filled, n_holes


def _projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """*gdf* in a metric CRS (its own when projected, else the estimated UTM zone)."""
    if gdf.crs is not None and gdf.crs.is_geographic:
        return gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def merge_small_hrus(
    hru_gdf: gpd.GeoDataFrame,
    min_area_km2: float,
    logger: Optional[logging.Logger] = None,
    *,
    gru_col: str = 'GRU_ID',
) -> gpd.GeoDataFrame:
    """Dissolve HRUs smaller than *min_area_km2* into a neighbour of the same GRU.

    Smallest first, each undersized HRU is merged into the HRU of the same GRU
    with which it shares the longest boundary (the nearest one when it touches
    none).  The receiving HRU keeps its own attributes; its geometry becomes
    the union, so the GRU's area is conserved.  A GRU's last remaining HRU is
    never removed, even when it is below the threshold.

    Areas are measured in a metric CRS; the result is returned in the input
    CRS with a fresh index.  HRU ids are NOT renumbered here.
    """
    log = logger or logging.getLogger(__name__)
    if min_area_km2 is None or min_area_km2 <= 0 or len(hru_gdf) < 2 or gru_col not in hru_gdf.columns:
        return hru_gdf

    min_area_m2 = float(min_area_km2) * 1e6
    work = _projected(hru_gdf).reset_index(drop=True)
    geoms = dict(enumerate(work.geometry))
    areas = {i: g.area for i, g in geoms.items()}
    alive = set(geoms)
    n_merged = 0
    n_kept_single = 0

    for _, positions in work.groupby(gru_col, sort=False).indices.items():
        members = {int(p) for p in positions}
        while len(members) > 1:
            small = [i for i in members if areas[i] < min_area_m2]
            if not small:
                break
            i = min(small, key=lambda k: (areas[k], k))
            others = sorted(members - {i})
            shared = {j: geoms[i].intersection(geoms[j]).length for j in others}
            best = max(others, key=lambda j: (shared[j], -j))
            if shared[best] <= 0:
                best = min(others, key=lambda j: (geoms[i].distance(geoms[j]), j))
            geoms[best] = unary_union([geoms[best], geoms[i]])
            areas[best] = geoms[best].area
            members.discard(i)
            alive.discard(i)
            n_merged += 1
        if len(members) == 1 and areas[next(iter(members))] < min_area_m2:
            n_kept_single += 1

    if n_merged == 0:
        if n_kept_single:
            log.info(
                f"MIN_HRU_SIZE={min_area_km2} km2: {n_kept_single} GRU(s) consist of a single "
                "HRU below the threshold; kept."
            )
        return hru_gdf

    keep = sorted(alive)
    merged = work.loc[keep].copy()
    merged['geometry'] = [geoms[i] for i in keep]
    merged = merged.set_geometry('geometry', crs=work.crs)
    if merged.crs != hru_gdf.crs:
        merged = merged.to_crs(hru_gdf.crs)
    log.info(
        f"MIN_HRU_SIZE={min_area_km2} km2: merged {n_merged} undersized HRU(s) into "
        f"neighbouring HRUs of the same GRU ({len(hru_gdf)} -> {len(merged)} HRUs)"
        + (f"; {n_kept_single} single-HRU GRU(s) remain below the threshold." if n_kept_single else ".")
    )
    return merged.reset_index(drop=True)


def check_area_conservation(
    hru_gdf: gpd.GeoDataFrame,
    gru_gdf: gpd.GeoDataFrame,
    logger: Optional[logging.Logger] = None,
    *,
    gru_col: str = 'GRU_ID',
    tolerance: float = AREA_CONSERVATION_TOLERANCE,
) -> float:
    """Warn when the HRUs of a GRU do not add up to the GRU's area.

    Both layers are measured in the same metric CRS, so the comparison is not
    affected by how a stored ``GRU_area`` attribute was computed.

    Returns:
        The largest relative difference found (0.0 when nothing to compare).
    """
    log = logger or logging.getLogger(__name__)
    if gru_col not in hru_gdf.columns or gru_col not in gru_gdf.columns or len(hru_gdf) == 0:
        return 0.0

    gru_m = _projected(gru_gdf)
    hru_m = hru_gdf.to_crs(gru_m.crs) if hru_gdf.crs != gru_m.crs else hru_gdf
    gru_area = gru_m.geometry.area.groupby(gru_m[gru_col].values).sum()
    hru_area = hru_m.geometry.area.groupby(hru_m[gru_col].values).sum()
    hru_area = hru_area.reindex(gru_area.index).fillna(0.0)

    rel = ((hru_area - gru_area).abs() / gru_area.where(gru_area > 0)).dropna()
    if rel.empty:
        return 0.0
    bad = rel[rel > tolerance].sort_values(ascending=False)
    if len(bad):
        worst = ', '.join(
            f"{gid}: {100 * (hru_area[gid] - gru_area[gid]) / gru_area[gid]:+.2f}%"
            for gid in bad.index[:10]
        )
        total = 100 * (hru_area.sum() - gru_area.sum()) / gru_area.sum()
        log.warning(
            f"HRU areas do not add up to the GRU area in {len(bad)} of {len(gru_area)} GRU(s) "
            f"(tolerance {100 * tolerance:.1f}%; domain total {total:+.2f}%). Largest "
            f"differences (GRU: HRU-GRU): {worst}. Area outside the HRUs is not modelled."
        )
    return float(rel.max())
