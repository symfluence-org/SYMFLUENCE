"""Model-independent, polygon-conservative glacier geometry manifests.

Thickness is integrated over raster-cell/polygon intersections in a metre CRS.
Missing thickness never reduces outline-defined glacier area or becomes zero ice.
The manifest describes geometry, not a model's ability to deplete that ice.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import Polygon
from shapely.ops import unary_union


def build_glacier_manifest(units, outlines, thickness_files, *, id_column,
                           ice_density=917.0):
    """Integrate non-overlapping thickness tiles over non-overlapping units.

    All inputs must use the same projected metre CRS: callers must explicitly
    reproject first, retaining their chosen resampling in dataset provenance.
    IDs identify spatial units, not generic land-cover classes. Partial raster
    boundary cells are integrated exactly. Positive ice requires full positive
    thickness coverage (apart from geometric roundoff).
    """
    unit_source = str(Path(units).resolve()) if isinstance(units, (str, Path)) else 'in_memory'
    outline_source = str(Path(outlines).resolve()) if isinstance(outlines, (str, Path)) else 'in_memory'
    units = gpd.read_file(units) if isinstance(units, (str, Path)) else units.copy()
    outlines = gpd.read_file(outlines) if isinstance(outlines, (str, Path)) else outlines.copy()
    crs = units.crs
    if crs is None or not crs.is_projected or any(
        axis.unit_conversion_factor != 1 for axis in crs.axis_info[:2]
    ):
        raise ValueError('Glacier units require a projected metre CRS')
    if outlines.crs != crs:
        raise ValueError('Outlines and units must have the same CRS')
    if not np.isfinite(ice_density) or not 0 < ice_density <= 1000:
        raise ValueError('Invalid ice density')
    ids = units[id_column].astype(str)
    if ids.duplicated().any() or units[id_column].isna().any():
        raise ValueError('Glacier unit IDs must be unique and non-null')
    for frame in (units, outlines):
        if frame.empty or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
            raise ValueError('Empty or invalid glacier geometry')
    union = unary_union(units.geometry)
    if abs(units.area.sum() - union.area) > max(1e-6, union.area * 1e-10):
        raise ValueError('Glacier units overlap')
    ice = unary_union(outlines.geometry)
    footprint = ice.intersection(union)
    files = [Path(p) for p in thickness_files]
    cells = []
    for path in files:
        with rasterio.open(path) as src:
            if src.crs != crs:
                raise ValueError('Thickness raster and units must have the same CRS')
            if footprint.is_empty:
                continue
            try:
                window = geometry_window(src, [footprint])
            except WindowError:
                continue
            values = src.read(1, window=window, masked=True)
            transform = src.window_transform(window)
            inside = geometry_mask([footprint], out_shape=values.shape, transform=transform,
                                   invert=True, all_touched=True)
            for row, col in zip(*np.where(~np.ma.getmaskarray(values) & inside)):
                value = float(values[row, col])
                if not np.isfinite(value) or value <= 0:
                    continue
                polygon = Polygon([transform * xy for xy in
                                   ((col, row), (col+1, row), (col+1, row+1), (col, row+1))])
                clipped = polygon.intersection(ice).intersection(union)
                if clipped.area > 0:
                    cells.append((clipped, value))
    rows = []
    for uid, polygon in zip(ids, units.geometry):
        area = polygon.area
        if area <= 0:
            raise ValueError('Unit area must be positive')
        glacier = polygon.intersection(ice)
        pieces = [(cell.intersection(glacier), value) for cell, value in cells
                  if cell.intersects(glacier)]
        covered = unary_union([piece for piece, _ in pieces]).area
        summed = sum(piece.area for piece, _ in pieces)
        tolerance = max(1e-6, glacier.area * 1e-9)
        if summed - covered > tolerance:
            raise ValueError(f'Overlapping thickness coverage for unit {uid}')
        if glacier.area - covered > tolerance:
            raise ValueError(f'Missing positive thickness for unit {uid}: '
                             f'{glacier.area-covered:.3f} m2')
        volume = sum(piece.area * value for piece, value in pieces)
        rows.append(dict(unit_id=uid, area_m2=area, glacier_area_m2=glacier.area,
                         land_area_m2=area-glacier.area, glacier_fraction=glacier.area/area,
                         thickness_coverage=covered/glacier.area if glacier.area else 1.,
                         ice_volume_m3=volume, ice_water_volume_m3=volume*ice_density/1000))
    return dict(schema_version=1, crs=crs.to_string(), ice_density_kg_m3=ice_density,
                unit_source=unit_source, outline_source=outline_source,
                thickness_files=[str(p.resolve()) for p in files],
                area_method='exact polygon/raster-cell intersections', units=rows)
