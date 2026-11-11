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
    donor overall is used.  Distances are between representative points.
    Zones for which no donor exists are absent from the result.
    """
    geoms = list(geometries)
    empty_set = set(empty)
    donors = [i for i in range(len(geoms)) if i not in empty_set]
    if not donors or not empty:
        return {}

    def _xy(idx: int) -> Tuple[float, float]:
        point = geoms[idx].representative_point()
        return point.x, point.y

    donor_xy = np.array([_xy(i) for i in donors])
    donor_groups = None if group_ids is None else np.array([str(group_ids[i]) for i in donors])

    out: Dict[int, int] = {}
    for i in empty:
        dist2 = np.sum((donor_xy - np.array(_xy(i))) ** 2, axis=1)
        if donor_groups is not None:
            same = donor_groups == str(group_ids[i])  # type: ignore[index]
            if same.any():
                dist2 = np.where(same, dist2, np.inf)
        out[i] = donors[int(np.argmin(dist2))]
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
