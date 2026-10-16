"""Conservative aggregation of mapped glacier area and water-equivalent volume."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject


def mapped_glaciers(mask, transform, outlines_file, thickness_dir, basin_geometry, crs=3057):
    resolution = abs(transform.a)
    factor = max(1, int(np.ceil(resolution / 100)))
    fine_transform = transform * transform.scale(1 / factor, 1 / factor)
    ny, nx = mask.shape
    shape = (ny * factor, nx * factor)
    outlines = gpd.read_file(outlines_file).to_crs(crs)
    outline = outlines.geometry.make_valid().union_all()
    # Use the same native basin footprint as forcing and routing. A second
    # vector clipping would remove edge ice while retaining full hydrological
    # cells. Check native glacier area against the independent vector basin.
    clipped = outline.intersection(basin_geometry)
    ice = rasterize([(outline, 1)], out_shape=shape, transform=fine_transform).astype(bool)
    ice &= np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    thickness = np.zeros(shape)
    tiles = sorted(Path(thickness_dir).glob('*thickness.tif'))
    if not tiles:
        raise ValueError('Mapped glaciers require thickness rasters')
    for tile in tiles:
        values = np.full(shape, np.nan)
        with rasterio.open(tile) as src:
            reproject(rasterio.band(src, 1), values, src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=fine_transform,
                      dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        thickness = np.maximum(thickness, np.nan_to_num(values, nan=0.0))
    covered = ice & (thickness > 0)
    coverage = covered.sum() / max(ice.sum(), 1)
    if coverage < .95 or not covered.any():
        raise ValueError(f'Insufficient glacier thickness coverage: {coverage:.2%}')
    def aggregate(values):
        return values.reshape(ny, factor, nx, factor).mean(axis=(1, 3))
    fraction = aggregate(covered.astype(float))
    water_depth = aggregate(np.where(covered, thickness * 917., 0.))
    store = np.divide(water_depth, fraction, out=np.zeros_like(fraction), where=fraction > 0)
    vector_area = clipped.area
    raster_area = fraction.sum() * resolution**2
    if abs(raster_area / vector_area - 1) > .02:
        raise ValueError('Glacier boundary area differs from vector outlines by >2%; refine grid')
    return fraction, store, dict(
        glacier_area_km2=float(raster_area / 1e6), vector_glacier_area_km2=float(vector_area / 1e6),
        initial_ice_water_km3=float(water_depth.sum() * resolution**2 / 1e12),
        boundary_method='native basin footprint; no inter-cell glacier relocation',
        thickness_coverage=float(coverage), ice_density_kg_m3=917.,
        outline_file=str(outlines_file), thickness_files=[str(p) for p in tiles])
