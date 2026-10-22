# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

from __future__ import annotations

import numpy as np
import pytest

ec = pytest.importorskip("eccodes")
from symfluence.data.acquisition.handlers.carra2 import cycle_rates, read_grib


def fixture(path, hole=False):
    g=ec.codes_grib_new_from_samples('polar_stereographic_sfc_grib2')
    try:
        for k,v in [('Nx',3),('Ny',3),('DxInMetres',2500),('DyInMetres',2500),('latitudeOfFirstGridPointInDegrees',64.),('longitudeOfFirstGridPointInDegrees',342.),('paramId',167),('stepType','instant'),('endStep',9)]:
            ec.codes_set(g,k,v)
        ec.codes_set_values(g,np.full(9,280.))
        lat=ec.codes_get_array(g,'latitudes');lon=(ec.codes_get_array(g,'longitudes')+180)%360-180
        # A very small geographic area isolates one cell; other cells are outside.
        area=[lat[4]+.0001,lon[4]-.0001,lat[4]-.0001,lon[4]+.0001]
        ec.codes_set(g,'bitmapPresent',1);ec.codes_set(g,'missingValue',9999.)
        v=np.full(9,9999.);v[4]=9999. if hole else 280.;ec.codes_set_values(g,v)
        with path.open('wb') as f:ec.codes_write(g,f)
        return area
    finally:ec.codes_release(g)


def test_outside_bitmap_is_not_missing_requested_data(tmp_path):
    p=tmp_path/'sample.grib';area=fixture(p)
    records,(lat,lon,attrs)=read_grib(p,area=area)
    assert lat.shape==(1,1)
    assert next(iter(records.values())).item()==280
    assert attrs['_native_indices'].tolist()==[4]
    with pytest.raises(ValueError,match='Missing'):read_grib(p)


def test_missing_requested_cell_rejected(tmp_path):
    p=tmp_path/'sample.grib';area=fixture(p,hole=True)
    with pytest.raises(ValueError,match='inside requested support'):read_grib(p,area=area)


def test_small_negatives_conserve_cycle_total():
    totals=np.array([0.,1.,.996,2.,3.])[:,None,None]
    rates,audit=cycle_rates(totals,.01)
    assert rates.min()>=0
    assert rates.sum()*10800==pytest.approx(3.)
    assert audit['negative_increments']==1
    assert audit['max_cycle_volume_residual']<1e-12
    with pytest.raises(ValueError,match='budget'):cycle_rates(totals,.001)


def test_net_negative_trace_is_explicitly_accounted():
    rates,audit=cycle_rates(np.array([.004,.003,.002,.001,0.])[:,None],.01)
    assert (rates==0).all()
    assert audit['max_cycle_volume_residual']==pytest.approx(.004)
