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
