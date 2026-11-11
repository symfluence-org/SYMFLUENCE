# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Zonal statistics that do not leave small zones without data.

``rasterstats.zonal_stats`` samples pixel centres by default, so a polygon
smaller than (or lying between the centres of) the raster pixels gets no
pixels at all.  Written out as ``elev_mean = -9999`` or as all-zero class
fractions, that silently corrupts model inputs.  The helpers here

1. retry every zone that received no pixel with ``all_touched=True`` (any
   pixel the polygon touches), and
2. fill zones that are still empty from the nearest zone that has data —
   preferring a zone of the same parent group (GRU) — and say which.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rasterstats import zonal_stats


def _is_empty(stat: Optional[Dict[str, Any]]) -> bool:
    return stat is None or not stat.get('count')


def zonal_stats_covered(
    geometries: Any,
    raster: Any,
    *,
    affine: Any = None,
    nodata: Any = None,
    stats: Sequence[str] = ('mean',),
    categorical: bool = False,
    logger: Optional[logging.Logger] = None,
    label: str = 'raster',
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Zonal statistics with an ``all_touched`` retry for zones without pixels.

    Args:
        geometries: Zone geometries (GeoSeries / sequence), in the raster CRS.
        raster: Raster path or 2-D array (with *affine*).
        affine: Affine transform when *raster* is an array.
        nodata: Raster nodata value.
        stats: Statistics to compute; ``count`` is always added.
        categorical: Also return per-class pixel counts.
        logger: Logger for the summary line.
        label: Name of the raster layer, for messages.

    Returns:
        ``(results, empty)`` — one stats dict per zone, and the positions of
        zones that have no valid pixel even with ``all_touched=True``.
    """
    geoms = list(geometries)
    wanted = list(dict.fromkeys([*stats, 'count']))
    kwargs: Dict[str, Any] = {'stats': wanted, 'categorical': categorical, 'nodata': nodata}
    if affine is not None:
        kwargs['affine'] = affine

    results = list(zonal_stats(geoms, raster, all_touched=False, **kwargs))
    retry = [i for i, stat in enumerate(results) if _is_empty(stat)]
    if not retry:
        return results, []

    retried = zonal_stats([geoms[i] for i in retry], raster, all_touched=True, **kwargs)
    empty = []
    for i, stat in zip(retry, retried):
        results[i] = stat
        if _is_empty(stat):
            empty.append(i)

    if logger is not None:
        logger.info(
            f"{label}: {len(retry)} zone(s) contain no pixel centre; "
            f"{len(retry) - len(empty)} resolved from touched pixels, {len(empty)} have no data."
        )
    return results, empty


def nearest_donor_zones(
    geometries: Any,
    empty: Sequence[int],
    group_ids: Optional[Sequence[Any]] = None,
) -> Dict[int, int]:
    """For each empty zone, the position of the nearest zone that has data.

    A donor in the same group (same GRU) is preferred; otherwise the nearest
    donor overall is used.  Distances are between centroids.  Zones for which
    no donor exists are absent from the result.
    """
    import shapely
    from scipy.spatial import cKDTree

    geoms = np.asarray(list(geometries), dtype=object)
    empty_set = set(int(i) for i in empty)
    donors = np.array([i for i in range(len(geoms)) if i not in empty_set], dtype=int)
    if donors.size == 0 or not empty_set:
        return {}

    centroids = shapely.centroid(geoms)
    xy = np.column_stack([shapely.get_x(centroids), shapely.get_y(centroids)])
    global_tree = cKDTree(xy[donors])

    groups = None if group_ids is None else np.array([str(g) for g in group_ids])
    group_trees: Dict[str, Any] = {}

    out: Dict[int, int] = {}
    for i in sorted(empty_set):
        if groups is not None:
            key = groups[i]
            if key not in group_trees:
                members = donors[groups[donors] == key]
                group_trees[key] = (cKDTree(xy[members]), members) if members.size else None
            if group_trees[key] is not None:
                tree, members = group_trees[key]
                out[i] = int(members[tree.query(xy[i])[1]])
                continue
        out[i] = int(donors[global_tree.query(xy[i])[1]])
    return out


def fill_empty_zones(
    results: List[Dict[str, Any]],
    empty: Sequence[int],
    geometries: Any,
    *,
    group_ids: Optional[Sequence[Any]] = None,
    zone_ids: Optional[Sequence[Any]] = None,
    logger: Optional[logging.Logger] = None,
    label: str = 'raster',
) -> List[int]:
    """Copy the statistics of the nearest zone with data into each empty zone.

    Modifies *results* in place and warns with the affected zone ids.

    Returns:
        Positions of zones that could not be filled (no zone has data).
    """
    if not empty:
        return []
    log = logger or logging.getLogger(__name__)

    def _name(idx: int) -> str:
        return str(zone_ids[idx]) if zone_ids is not None else str(idx)

    donors = nearest_donor_zones(geometries, empty, group_ids)
    unfilled = [i for i in empty if i not in donors]
    for i, donor in donors.items():
        results[i] = dict(results[donor])
    if donors:
        pairs = ', '.join(f"{_name(i)}<-{_name(d)}" for i, d in list(donors.items())[:20])
        log.warning(
            f"{label}: {len(donors)} zone(s) have no valid pixel even when every touched "
            f"pixel is counted; filled from the nearest zone with data "
            f"(zone<-donor: {pairs}{', ...' if len(donors) > 20 else ''})."
        )
    if unfilled:
        log.error(
            f"{label}: {len(unfilled)} zone(s) have no valid pixel and no zone with data "
            f"exists to fill them from (e.g. {[_name(i) for i in unfilled[:20]]}). "
            "Check that the raster covers the domain."
        )
    return unfilled
