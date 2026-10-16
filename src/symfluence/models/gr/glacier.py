# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Degree-day glacier ice melt for lumped GR4J + CemaNeige.

airGR has no glacier process. In glacier mode the runner calls CemaNeige and
GR4J separately and adds ice melt between them: each CemaNeige elevation layer
carries the glacierised fraction of its band (from an outline), and ice melts
at ``ddf * max(T_layer, 0)`` on days when the layer's snowpack is gone. The
basin-mean melt joins CemaNeige's liquid output as GR4J's precipitation input.

This is a forcing-side workaround, not glacier physics: ice storage is
unlimited and the glacier area is prescribed (fixed outline, optionally scaled
per water year to an observed area series).

When an observed mean net mass balance is given, the ice degree-day factor is
not calibrated. It is solved in each evaluation so that the mean glacier-wide
net balance (change in snow on the ice minus ice melt) over the target water
years equals the observation. The balance is linear in the factor because ice
melt does not feed back into CemaNeige.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class GlacierGeometry:
    """Hypsometry and glacierised fraction of each equal-area elevation layer."""

    hypso: np.ndarray            # 101 elevation percentiles (airGR HypsoData)
    z_mean: float                # mean catchment elevation (airGR ZInputs)
    layer_fraction: np.ndarray   # glacierised fraction of each layer, 0..1
    glacier_area_km2: float      # outline area inside the catchment


@dataclass
class IceMeltResult:
    """Basin ice melt series and the glacier-wide balance diagnostics."""

    ice_melt: np.ndarray                         # mm/day over the basin
    ddf: float                                   # mm/K/day actually applied
    ddf_unclipped: float                         # solved value before bounds (nan if not solved)
    mass_balance: Dict[int, float] = field(default_factory=dict)  # water year -> mm w.e.


def layer_glacier_fractions(elevation: np.ndarray, on_glacier: np.ndarray, n_layers: int) -> np.ndarray:
    """Glacierised fraction of each equal-area elevation band.

    The bands follow airGR's layer definition: ``n_layers`` equal-area slices
    of the catchment hypsometry, lowest first.
    """
    elevation = np.asarray(elevation, dtype=float)
    on_glacier = np.asarray(on_glacier, dtype=bool)
    if elevation.shape != on_glacier.shape or elevation.size == 0:
        raise ConfigurationError("Glacier mask and DEM sample must be non-empty and the same shape")
    edges = np.percentile(elevation, np.linspace(0, 100, n_layers + 1))
    layer = np.clip(np.searchsorted(edges, elevation, side='right') - 1, 0, n_layers - 1)
    return np.array([on_glacier[layer == i].mean() if np.any(layer == i) else 0.0
                     for i in range(n_layers)])


@lru_cache(maxsize=8)
def load_glacier_geometry(dem_path: str, catchment_path: str, outline_path: str, n_layers: int) -> GlacierGeometry:
    """Catchment hypsometry and per-layer glacier fractions from a DEM and an outline.

    Uses the same catchment masking as the lumped runner, so ``hypso`` and
    ``z_mean`` match what the non-glacier path passes to airGR. Cached because
    calibration builds a new runner for every evaluation.
    """
    import geopandas as gpd
    import rasterio
    import rasterio.mask

    catchment = gpd.read_file(catchment_path)
    outline = gpd.read_file(outline_path)
    with rasterio.open(dem_path) as src:
        catchment = catchment.to_crs(src.crs)
        outline = outline.to_crs(src.crs)
        fill = src.nodata if src.nodata is not None else 0
        dem, _ = rasterio.mask.mask(src, catchment.geometry, crop=False, nodata=fill)
        ice, _ = rasterio.mask.mask(src, outline.geometry, crop=False, nodata=fill)
        pixel_km2 = abs(src.transform.a * src.transform.e) / 1e6 if src.crs.is_projected else None
    inside = dem[0] != fill
    if not inside.any():
        raise ConfigurationError(f"No valid DEM pixels inside the catchment ({dem_path})")
    elevation = dem[0][inside]
    on_glacier = (ice[0] != fill)[inside]
    if pixel_km2 is None:
        pixel_km2 = float(catchment.to_crs(catchment.estimate_utm_crs()).area.sum()) / 1e6 / elevation.size
    return GlacierGeometry(
        hypso=np.percentile(elevation, np.arange(0, 101, 1)),
        z_mean=float(elevation.mean()),
        layer_fraction=layer_glacier_fractions(elevation, on_glacier, n_layers),
        glacier_area_km2=float(on_glacier.sum() * pixel_km2),
    )


def water_years(dates: pd.DatetimeIndex) -> np.ndarray:
    """October-September water year of each date (WY2013 starts 1 Oct 2012)."""
    return np.asarray(dates.year + (dates.month >= 10).astype(int))


def complete_water_years(dates: pd.DatetimeIndex) -> Dict[int, tuple]:
    """Water years fully covered by ``dates``: year -> (first index, last index)."""
    out = {}
    for wy in np.unique(water_years(dates)):
        first = np.flatnonzero(dates == pd.Timestamp(f'{wy - 1}-10-01'))
        last = np.flatnonzero(dates == pd.Timestamp(f'{wy}-09-30'))
        if first.size and last.size:
            out[int(wy)] = (int(first[0]), int(last[0]))
    return out


def compute_ice_melt(
    dates: pd.DatetimeIndex,
    layer_swe: np.ndarray,
    layer_temp: np.ndarray,
    layer_fraction: Sequence[float],
    area_scale: np.ndarray,
    *,
    ddf: Optional[float] = None,
    mb_target_mm: Optional[float] = None,
    mb_years: Sequence[int] = (),
    ddf_bounds: Sequence[float] = (1.0, 20.0),
    snowfree_swe: float = 1.0,
) -> IceMeltResult:
    """Ice melt from CemaNeige layer states.

    Args:
        dates: daily dates of the run.
        layer_swe, layer_temp: CemaNeige ``SnowPack`` (mm) and ``Temp`` (degC), shape (time, layer).
        layer_fraction: glacierised fraction of each layer.
        area_scale: per-day multiplier on ``layer_fraction`` (observed / outline area), shape (time,).
        ddf: fixed ice degree-day factor (mm/K/day); used when ``mb_target_mm`` is None.
        mb_target_mm: observed mean net balance (mm w.e./yr) over ``mb_years``; solves the factor.
        ddf_bounds: limits applied to the solved factor.
        snowfree_swe: layer snowpack (mm) below which ice is exposed.
    """
    frac = np.asarray(layer_fraction, dtype=float)
    if layer_swe.shape != layer_temp.shape or layer_swe.shape[1] != frac.size:
        raise ConfigurationError("CemaNeige layer arrays do not match the glacier layer fractions")
    if frac.sum() <= 0:
        raise ConfigurationError("Glacier outline does not overlap the catchment")
    weights = frac / frac.sum()
    pdd_exposed = np.maximum(layer_temp, 0.0) * (layer_swe < snowfree_swe)

    wys = complete_water_years(dates)
    dswe = {y: float(weights @ (layer_swe[b] - layer_swe[a])) for y, (a, b) in wys.items()}
    pdd = {y: float(weights @ pdd_exposed[a:b + 1].sum(axis=0)) for y, (a, b) in wys.items()}

    ddf_unclipped = float('nan')
    if mb_target_mm is not None:
        years = [y for y in mb_years if y in wys]
        if not years:
            raise ConfigurationError(f"No complete simulated water year among glacier MB years {list(mb_years)}")
        mean_pdd = np.mean([pdd[y] for y in years])
        ddf_unclipped = ((np.mean([dswe[y] for y in years]) - mb_target_mm) / mean_pdd
                         if mean_pdd > 0 else float('inf'))
        ddf = float(np.clip(ddf_unclipped, ddf_bounds[0], ddf_bounds[1]))
    elif ddf is None:
        raise ConfigurationError("Glacier mode needs either an ice degree-day factor or a mass-balance target")

    ice = ddf * (pdd_exposed @ frac) / frac.size * np.asarray(area_scale, dtype=float)
    balance = {y: dswe[y] - ddf * pdd[y] for y in wys}
    return IceMeltResult(ice_melt=ice, ddf=float(ddf), ddf_unclipped=ddf_unclipped, mass_balance=balance)


def area_scale_series(dates: pd.DatetimeIndex, outline_area_km2: float,
                      area_file: Optional[Path]) -> np.ndarray:
    """Per-day multiplier taking the outline area to an observed glacier area.

    ``area_file`` is a LamaH-Ice style ``glacier_timeseries.csv`` (``;``
    separated, year index, ``g_area_dyn`` in km2). Years outside the series
    take the nearest value. Without a file the outline area is used as is.
    """
    if area_file is None:
        return np.ones(len(dates))
    series = pd.read_csv(area_file, sep=';', index_col=0)['g_area_dyn'].dropna()
    if series.empty:
        raise ConfigurationError(f"No glacier areas (g_area_dyn) in {area_file}")
    wy = water_years(dates)
    area = np.interp(wy, series.index.to_numpy(dtype=float), series.to_numpy(dtype=float))
    return area / outline_area_km2
