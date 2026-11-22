# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for forcing-cell footprints, EM-Earth precip units and
NEX-GDDP shortwave disaggregation.

* Curvilinear forcing cells must be centred on the cell whose lat/lon tags
  them (previously each polygon spanned to the neighbours' centres, shifting
  it half a cell and collapsing the last row/column to zero area).
* EM-Earth mm/h into a 'mm s-1' forcing must divide by 3600, not 3.6e6
  (the 'm s-1' substring test used to capture 'mm s-1').
* NEX-GDDP daily shortwave must peak near local solar noon, not 12 UTC.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

gpd = pytest.importorskip("geopandas")

from symfluence.data.preprocessing.resampling.grid_geometry import (  # noqa: E402
    curvilinear_cell_polygons,
)


def _grid(nrow=4, ncol=5, lat0=64.0, lon0=-20.0, step=0.1):
    lon2, lat2 = np.meshgrid(lon0 + step * np.arange(ncol), lat0 + step * np.arange(nrow))
    return lon2, lat2


# --------------------------------------------------------------------------
# Item 1: curvilinear footprints
# --------------------------------------------------------------------------

def test_curvilinear_cells_centred_and_non_degenerate():
    lon2, lat2 = _grid()
    cells = curvilinear_cell_polygons(lon2, lat2)
    assert cells.shape == lon2.shape
    for i in range(lon2.shape[0]):
        for j in range(lon2.shape[1]):
            c = cells[i, j].centroid
            assert c.x == pytest.approx(lon2[i, j], abs=1e-9)
            assert c.y == pytest.approx(lat2[i, j], abs=1e-9)
            assert cells[i, j].area == pytest.approx(0.01, rel=1e-9)


def test_curvilinear_window_matches_full_grid():
    lon2, lat2 = _grid(8, 9)
    full = curvilinear_cell_polygons(lon2, lat2)
    win = curvilinear_cell_polygons(lon2, lat2, rows=(2, 5), cols=(3, 9))
    assert win.shape == (3, 6)
    for i in range(3):
        for j in range(6):
            assert win[i, j].equals(full[2 + i, 3 + j])


def test_curvilinear_single_cell_uses_nominal_step():
    lon2, lat2 = np.array([[-110.0]]), np.array([[50.0]])
    cell = curvilinear_cell_polygons(lon2, lat2, nominal_step=0.1)[0, 0]
    assert cell.area > 0
    assert cell.centroid.x == pytest.approx(-110.0)
    assert cell.centroid.y == pytest.approx(50.0)
    with pytest.raises(ValueError):
        curvilinear_cell_polygons(lon2, lat2)


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
def test_rdrs_shapefile_cells_centred(tmp_path):
    from symfluence.data.preprocessing.dataset_handlers.rdrs_utils import RDRSHandler

    merged, out = tmp_path / "m", tmp_path / "s"
    merged.mkdir()
    lon2, lat2 = _grid()
    rlat, rlon = np.arange(4.0), np.arange(5.0)
    xr.Dataset(
        {"t": (("rlat", "rlon"), np.zeros((4, 5)))},
        coords={"rlat": rlat, "rlon": rlon,
                "lat": (("rlat", "rlon"), lat2), "lon": (("rlat", "rlon"), lon2)},
    ).to_netcdf(merged / "RDRS_monthly_201501.nc")

    h = RDRSHandler.__new__(RDRSHandler)
    h.logger = logging.getLogger("test")
    h._config = MagicMock()
    names = {"FORCING_SHAPE_LAT_NAME": "lat", "FORCING_SHAPE_LON_NAME": "lon"}
    h._get_config_value = lambda f, default=None, dict_key=None: names.get(dict_key, "RDRS")
    h.open_dataset = xr.open_dataset

    path = h.create_shapefile(out, merged, "dem.tif", lambda g, d, batch_size=0: np.zeros(len(g)))
    g = gpd.read_file(path)
    assert list(g["ID"]) == list(range(20))
    assert (g.geometry.area > 0).all()
    np.testing.assert_allclose(g.geometry.centroid.x, g["lon"], atol=1e-9)
    np.testing.assert_allclose(g.geometry.centroid.y, g["lat"], atol=1e-9)


# --------------------------------------------------------------------------
# Item 2: EM-Earth precipitation units
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "units, expected",
    [("mm s-1", 1e-3), ("mm/s", 1e-3), ("kg m-2 s-1", 1e-3), ("kg m**-2 s**-1", 1e-3),
     ("m s-1", 1e-6), ("mm/h", 3.6)],
)
def test_em_earth_precip_unit_conversion(tmp_path, units, expected):
    from symfluence.data.preprocessing.em_earth_integrator import EMEarthIntegrator

    t = pd.date_range("2015-01-01", periods=24, freq="h")
    forc = xr.Dataset({"precipitation_flux": (("time", "hru"), np.full((24, 2), 1e-4))},
                      coords={"time": t})
    forc["precipitation_flux"].attrs["units"] = units
    forc.to_netcdf(tmp_path / "forc.nc")
    xr.Dataset({"prcp": (("time", "hru"), np.full((24, 2), 3.6))},
               coords={"time": t}).to_netcdf(tmp_path / "em.nc")

    obj = EMEarthIntegrator.__new__(EMEarthIntegrator)
    obj.logger = logging.getLogger("test")
    obj._update_single_forcing_file(tmp_path / "forc.nc", {"201501": tmp_path / "em.nc"})
    with xr.open_dataset(tmp_path / "forc.nc") as ds:
        np.testing.assert_allclose(ds["precipitation_flux"].values, expected, rtol=1e-9)


@pytest.mark.parametrize("units, offset", [("K", 273.15), ("degC", 0.0), ("Celsius", 0.0)])
def test_em_earth_temperature_units(tmp_path, units, offset):
    from symfluence.data.preprocessing.em_earth_integrator import EMEarthIntegrator

    t = pd.date_range("2015-01-01", periods=24, freq="h")
    forc = xr.Dataset({"air_temperature": (("time", "hru"), np.full((24, 2), 1.0))},
                      coords={"time": t})
    forc["air_temperature"].attrs["units"] = units
    forc.to_netcdf(tmp_path / "forc.nc")
    xr.Dataset({"tmean": (("time", "hru"), np.full((24, 2), 5.0))},
               coords={"time": t}).to_netcdf(tmp_path / "em.nc")

    obj = EMEarthIntegrator.__new__(EMEarthIntegrator)
    obj.logger = logging.getLogger("test")
    obj._update_single_forcing_file(tmp_path / "forc.nc", {"201501": tmp_path / "em.nc"})
    with xr.open_dataset(tmp_path / "forc.nc") as ds:
        np.testing.assert_allclose(ds["air_temperature"].values, 5.0 + offset, rtol=1e-9)


# --------------------------------------------------------------------------
# Item 3: NEX-GDDP shortwave diurnal cycle
# --------------------------------------------------------------------------

def test_nex_gddp_shortwave_solar_noon_and_daily_mean():
    from symfluence.data.preprocessing.dataset_handlers.nex_gddp_utils import NEXGDDPCMIP6Handler

    days = pd.date_range("2015-06-01", periods=5, freq="D")
    lons = np.array([-110.0, 0.0])
    ds = xr.Dataset(
        {"surface_downwelling_shortwave_flux": (("time", "lat", "lon"), np.full((5, 1, 2), 250.0))},
        coords={"time": days, "lat": [45.0], "lon": lons},
    )
    h = NEXGDDPCMIP6Handler.__new__(NEXGDDPCMIP6Handler)
    h.logger = logging.getLogger("test")
    sw = h._interpolate_daily_to_hourly(ds)["surface_downwelling_shortwave_flux"]
    assert sw.dims == ("time", "lat", "lon")
    assert float(sw.min()) >= 0.0

    daily = sw.resample(time="1D").mean()
    np.testing.assert_allclose(daily.values, 250.0, rtol=1e-9)

    day = sw.sel(time="2015-06-03")
    hours = pd.DatetimeIndex(day.time.values).hour
    # Local solar noon: 12 UTC - lon/15 -> ~19.3 UTC at -110E, 12 UTC at 0E.
    assert hours[int(np.argmax(day.isel(lat=0, lon=0).values))] == 19
    assert hours[int(np.argmax(day.isel(lat=0, lon=1).values))] == 12
    # Night at -110E (local midnight ~07 UTC) is dark; it is daytime at 0E then.
    assert float(day.isel(lat=0, lon=0).sel(time="2015-06-03T07")) == 0.0
    assert float(day.isel(lat=0, lon=1).sel(time="2015-06-03T07")) > 0.0
