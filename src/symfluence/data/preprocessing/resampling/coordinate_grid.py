# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate coordinates before reusing positional forcing-remapping weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _coordinate_names(ds, names=None):
    if names is not None and all(name in ds for name in names):
        return names
    for pair in [("latitude", "longitude"), ("lat", "lon")]:
        if all(name in ds for name in pair):
            return pair
    raise ValueError("Cannot verify source latitude/longitude coordinates")


def forcing_grid_signature(path: Path, names: Optional[Tuple[str, str]] = None) -> str:
    """Fingerprint coordinate locations, ordering and declared cell geometry."""
    signature = hashlib.sha256()
    with xr.open_dataset(path) as ds:
        lat_name, lon_name = _coordinate_names(ds, names)
        for name in [lat_name, lon_name]:
            values = np.asarray(ds[name].values, dtype=np.float64)
            if not values.size or not np.isfinite(values).all():
                raise ValueError(f"Invalid forcing coordinates in {path}: {name}")
            if name == lon_name:
                values = (values + 180.0) % 360.0 - 180.0
            # Ignore representation noise below approximately one millimetre.
            values = np.round(values, decimals=8).astype("<f8")
            signature.update(json.dumps([ds[name].dims, values.shape]).encode())
            signature.update(values.tobytes(order="C"))
        metadata: dict[str, object] = {"projection": str(ds.attrs.get("MAP_PROJ4_STR", ""))}
        for key in ["DX", "DY"]:
            if key in ds.attrs:
                value = np.asarray(ds.attrs[key], dtype=float)
                if value.size != 1 or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in {path}")
                metadata[key] = float(value.ravel()[0])
        signature.update(json.dumps(metadata, sort_keys=True).encode())
    return signature.hexdigest()


def require_same_forcing_grid(files: Iterable[Path], names=None) -> None:
    """Reject mixed grids before weights are generated or outputs are skipped."""
    paths = iter(files)
    first = next(paths, None)
    if first is None:
        return
    expected = forcing_grid_signature(first, names)
    for path in paths:
        if forcing_grid_signature(path, names) != expected:
            raise ValueError(
                f"Forcing grid changes between {first} and {path}. "
                "Generate and apply separate weights for each coordinate grid."
            )


def weights_match_forcing_coordinates(remap_file: Path, forcing_file: Path, names=None) -> bool:
    """Check the actual zero-based source indices stored by EASYMORE."""
    weights = pd.read_csv(remap_file)
    required = ["rows", "cols", "lat_s", "lon_s"]
    if weights.empty or any(column not in weights for column in required):
        return False
    values = weights[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return False
    indices = values[:, :2]
    if (indices < 0).any() or not np.equal(indices, np.floor(indices)).all():
        return False
    rows, cols = indices.astype(np.int64).T
    with xr.open_dataset(forcing_file) as ds:
        lat_name, lon_name = _coordinate_names(ds, names)
        lat, lon = ds[lat_name], ds[lon_name]
        if lat.ndim == lon.ndim == 2 and lat.shape == lon.shape and lat.dims == lon.dims:
            if rows.max() >= lat.shape[0] or cols.max() >= lat.shape[1]:
                return False
            actual_lat, actual_lon = lat.values[rows, cols], lon.values[rows, cols]
        elif lat.ndim == lon.ndim == 1:
            if lat.dims == lon.dims:
                if cols.any() or rows.max() >= len(lat):
                    return False
                actual_lat, actual_lon = lat.values[rows], lon.values[rows]
            else:
                if rows.max() >= len(lat) or cols.max() >= len(lon):
                    return False
                actual_lat, actual_lon = lat.values[rows], lon.values[cols]
        else:
            return False
    longitude_error = (actual_lon - values[:, 3] + 180.0) % 360.0 - 180.0
    return bool(
        np.isfinite(actual_lat).all()
        and np.isfinite(actual_lon).all()
        and np.allclose(actual_lat, values[:, 2], atol=1e-5, rtol=0)
        and np.allclose(longitude_error, 0.0, atol=1e-5, rtol=0)
    )
