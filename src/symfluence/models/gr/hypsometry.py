# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Hypsometry inputs for airGR's CemaNeige (``HypsoData`` and ``ZInputs``).

One implementation for the lumped, lumped-glacier and distributed GR paths:
the elevations of the DEM pixels inside a set of geometries, and the 101
elevation percentiles plus mean elevation airGR derives its snow layers from.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from symfluence.core.exceptions import ModelExecutionError


def masked_dem_elevations(src: Any, geometries: Any) -> np.ndarray:
    """Elevations of the valid DEM pixels inside *geometries* (1-D, may be empty).

    Args:
        src: open rasterio dataset of the DEM.
        geometries: geometries in the DEM's CRS.
    """
    import rasterio.mask

    # Use a known fill value so masked pixels can be filtered reliably
    fill_value = src.nodata if src.nodata is not None else 0
    try:
        out_image, _ = rasterio.mask.mask(src, geometries, crop=True, nodata=fill_value)
    except ValueError:
        # rasterio: "Input shapes do not overlap raster."
        return np.array([], dtype=float)
    masked_dem = out_image[0]
    return masked_dem[masked_dem != fill_value]


def hypsometry(elevations: np.ndarray) -> Tuple[np.ndarray, Any]:
    """airGR ``HypsoData`` (percentiles 0..100) and ``ZInputs`` (mean) of a pixel sample."""
    return np.percentile(elevations, np.arange(0, 101, 1)), np.mean(elevations)


def flat_hypsometry(elevation: float) -> Tuple[np.ndarray, float]:
    """Hypsometry of a unit known only by its mean elevation: no elevation spread.

    Every CemaNeige layer then sits at the input elevation, so airGR applies
    no altitudinal extrapolation to the unit's precipitation and temperature.
    """
    return np.full(101, float(elevation)), float(elevation)


def catchment_hypsometry(dem_path: Any, catchment: Any) -> Tuple[np.ndarray, Any, np.ndarray]:
    """``(HypsoData, ZInputs, elevations)`` of the DEM inside *catchment* (a GeoDataFrame)."""
    import rasterio

    with rasterio.open(dem_path) as src:
        if catchment.crs != src.crs:
            catchment = catchment.to_crs(src.crs)
        elevations = masked_dem_elevations(src, catchment.geometry)
    if elevations.size == 0:
        raise ModelExecutionError(
            f"No valid DEM pixels after masking with catchment. Check DEM at {dem_path}"
        )
    hypso, z_mean = hypsometry(elevations)
    return hypso, z_mean, elevations
