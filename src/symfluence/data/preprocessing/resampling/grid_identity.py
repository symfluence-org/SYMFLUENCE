# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Coordinate identity and provenance for remapping caches.

Array shape alone is insufficient: a shifted or reordered grid needs new weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

VERSION = 'coordinate-remapping-v1'
ATTRIBUTE = 'symfluence_remapping_signature'


def grid_signature(path, coordinate_names):
    digest = hashlib.sha256(VERSION.encode())
    with xr.open_dataset(path) as ds:
        for name in coordinate_names:
            a = ds[name]
            values = np.asarray(a.values, dtype='<f8')
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite forcing coordinates in {path}: {name}')
            digest.update(json.dumps([name, a.dims, a.shape]).encode())
            digest.update(values.tobytes())
            digest.update(json.dumps(dict(a.attrs), sort_keys=True, default=str).encode())
        projection = {k: v for k, v in ds.attrs.items()
                      if k in ('MAP_PROJ4_STR', 'DX', 'DY', 'crs', 'spatial_ref')}
        for name, var in ds.variables.items():
            if 'grid_mapping_name' in var.attrs or 'spatial_ref' in var.attrs:
                projection[name] = dict(var.attrs)
        digest.update(json.dumps(projection, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def target_signature(path):
    """Include target IDs, attributes, CRS and geometry, not just the basename."""
    path = Path(path)
    parts = [path.with_suffix(s) for s in ('.shp', '.shx', '.dbf', '.prj', '.cpg')] if path.suffix == '.shp' else [path]
    digest = hashlib.sha256()
    for part in parts:
        if part.exists():
            digest.update(part.suffix.encode())
            digest.update(part.read_bytes())
    return digest.hexdigest()


# Configuration keys that change how forcing is remapped: field names used to
# read the source/target shapefiles, how the source grid polygons are built
# (bounding box, DEM-based elevation filtering), and which target/HRU set the
# weights point at. Everything else (experiment id, model choice, process
# count, calibration settings, the experiment time window, ...) cannot change
# a remapped file and must not invalidate the shared cache. The target
# shapefile's CONTENT is hashed separately (target_signature) and the forcing
# grid by grid_signature.
REMAP_CONFIG_KEYS = (
    'FORCING_DATASET',
    'FORCING_SHAPE_LAT_NAME', 'FORCING_SHAPE_LON_NAME', 'FORCING_SHAPE_ID_NAME',
    'CATCHMENT_SHP_LAT', 'CATCHMENT_SHP_LON', 'CATCHMENT_SHP_HRUID',
    'CATCHMENT_SHP_NAME', 'CATCHMENT_PATH',
    'DOMAIN_NAME', 'DOMAIN_DEFINITION_METHOD', 'SUB_GRID_DISCRETIZATION',
    'BOUNDING_BOX_COORDS', 'POUR_POINT_COORDS',
    'DEM_NAME', 'DEM_PATH',
    'REMOVE_INVALID_ELEVATION_CELLS', 'CATCHMENT_MEAN_ELEVATION',
)
REMAP_CONFIG_VERSION = 'remap-config-v2'


def remap_config_signature(config_dict, forcing_dataset=None):
    """Hash only the remap-relevant part of a flat configuration dict.

    Includes the keys in REMAP_CONFIG_KEYS plus any ``{forcing_dataset}_*``
    dataset-specific option (e.g. ``HRRR_BUFFER_CELLS``), which can change
    the source grid construction.
    """
    config_dict = dict(config_dict or {})
    dataset = str(forcing_dataset or '').upper()
    prefix = f'{dataset}_' if dataset else None
    relevant = {k: config_dict.get(k) for k in REMAP_CONFIG_KEYS}
    if prefix:
        relevant.update({k: v for k, v in config_dict.items()
                         if str(k).upper().startswith(prefix)})
    payload = json.dumps([REMAP_CONFIG_VERSION, relevant], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def manifest_path(weights):
    return Path(weights).with_suffix('.grid.json')


def read_manifest(weights):
    path = manifest_path(weights)
    return json.loads(path.read_text()) if path.exists() else None


def output_matches(path, signature):
    try:
        with xr.open_dataset(path) as ds:
            return ds.attrs.get(ATTRIBUTE) == signature
    except (OSError, ValueError):
        return False
