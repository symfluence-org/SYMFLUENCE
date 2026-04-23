# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for Copernicus DEM bbox clipping.

Co-authors running the 08_large_sample verification hit very slow
TauDEM runs (d8flowdir taking 10+ minutes per basin) for basins as
small as 4 km². Root cause: the Copernicus DEM acquirer fetched
whole 1° tiles and merged them without clipping to the requested
bbox, so a basin whose bbox straddled a tile boundary ended up with
a 2°×1° (~26M-pixel) raster instead of a ~0.25°×0.25° clip. Every
downstream TauDEM step then spent most of its wall-clock on
mostly-wasted pixels.

Pin the fix: _merge_tiles now accepts a ``bounds`` argument and
the Copernicus acquirers pass the request bbox through, so the
merged output is tight to the bbox.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _write_tile(path, x_origin, y_origin, size=120, res=0.01, fill=1):
    """Write a small synthetic 1° × 1° GeoTIFF for the merge test."""
    transform = from_origin(x_origin, y_origin, res, res)
    data = np.full((size, size), fill, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size,
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as ds:
        ds.write(data, 1)


def _make_dem_mixin_instance(tmp_path):
    """Build a minimal object that exposes _merge_tiles — we don't
    need the rest of the acquirer for this test."""
    from symfluence.data.acquisition.handlers.dem import _TileDownloadMixin

    class _Probe(_TileDownloadMixin):
        def __init__(self):
            self.logger = MagicMock()

    return _Probe()


def test_merge_tiles_honours_bounds(tmp_path):
    """Two adjacent 1° tiles merged with bounds clipped to a small
    region must produce a raster sized to the bbox, not the full
    2°-tile envelope."""
    t1 = tmp_path / "tile_n63_w22.tif"
    t2 = tmp_path / "tile_n64_w22.tif"
    # Tile 1: covers lon -22..-21, lat 63..64
    _write_tile(t1, x_origin=-22.0, y_origin=64.0, size=100, res=0.01)
    # Tile 2: covers lon -22..-21, lat 64..65
    _write_tile(t2, x_origin=-22.0, y_origin=65.0, size=100, res=0.01, fill=2)

    out = tmp_path / "merged.tif"
    probe = _make_dem_mixin_instance(tmp_path)
    # Small bbox straddling the tile boundary — this is the basin-82
    # case (~0.25° x 0.25° region over two tiles).
    bounds = (-21.8, 63.94, -21.58, 64.19)
    probe._merge_tiles([t1, t2], out, bounds=bounds)

    with rasterio.open(out) as ds:
        # Without clipping the merge would have been 200 rows × 100 cols
        # covering the full 2° × 1° envelope. With clipping, we should
        # see ~25 rows × ~22 cols covering only the requested bbox.
        assert ds.width <= 40, f"merged width {ds.width} > 40 — bbox clip did not take effect"
        assert ds.height <= 40, f"merged height {ds.height} > 40 — bbox clip did not take effect"
        # Clip bounds must enclose, not exceed, the requested bbox.
        assert ds.bounds.left >= bounds[0] - 0.02
        assert ds.bounds.right <= bounds[2] + 0.02
        assert ds.bounds.bottom >= bounds[1] - 0.02
        assert ds.bounds.top <= bounds[3] + 0.02


def test_merge_tiles_without_bounds_keeps_old_behaviour(tmp_path):
    """bounds is opt-in. Callers that don't pass it (SRTM, Mapzen,
    ALOS) must still merge the full union of tiles — we don't want
    this change to quietly shrink rasters for other acquirers."""
    t1 = tmp_path / "tile_n63_w22.tif"
    t2 = tmp_path / "tile_n64_w22.tif"
    _write_tile(t1, x_origin=-22.0, y_origin=64.0, size=100, res=0.01)
    _write_tile(t2, x_origin=-22.0, y_origin=65.0, size=100, res=0.01, fill=2)

    out = tmp_path / "merged.tif"
    probe = _make_dem_mixin_instance(tmp_path)
    probe._merge_tiles([t1, t2], out)

    with rasterio.open(out) as ds:
        # Full 2° × 1° union — width 100, height 200.
        assert ds.width == 100
        assert ds.height == 200
