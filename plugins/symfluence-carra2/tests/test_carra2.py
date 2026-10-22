import numpy as np
import pandas as pd
import pytest
from symfluence_carra2.acquisition import interval_rate, request_for_days, convert, FIELDS

def test_cycle_accumulation_volume():
    totals = np.array([7.,10.,16.,16.,20.])
    rates = [interval_rate(a,b) for a,b in zip(totals[:-1],totals[1:])]
    assert sum(rates)*10800 == pytest.approx(13)
    with pytest.raises(ValueError): interval_rate(10.,1.)
    with pytest.raises(ValueError): interval_rate(1.,np.nan)

def test_request_not_carra1():
    request = request_for_days(pd.date_range('2013-07-01',periods=2),[65,-20,63,-17])
    assert request['leadtime_hour'] == ['6','9','12','15','18']
    assert request['level_type'] == 'single_levels'
    assert 'domain' not in request
    with pytest.raises(ValueError):request_for_days(pd.date_range('2013-06-30',periods=2),[65,-20,63,-17])

def test_cycle_boundary_and_missing_coverage(monkeypatch):
    records = {}
    for c in pd.to_datetime(['2013-06-30 12:00','2013-07-01 00:00','2013-07-01 12:00']):
        for p,(_,_,accum) in FIELDS.items():
            for step in [6,9,12,15,18]:
                records[p,c,step]=np.ones((2,2))*(step if accum else 280.)
    grid=(np.ones((2,2))*64,np.ones((2,2))*-18,{'MAP_PROJ4_STR':'test'})
    monkeypatch.setattr('symfluence_carra2.acquisition.read_grib',lambda path:(records,grid))
    ds=convert('unused','2013-07-01 00:00','2013-07-01 21:00')
    assert ds.sizes['time']==8
    np.testing.assert_allclose(ds.precipitation_flux,1/3600)
    assert ds.time_bounds.values[0,0]==np.datetime64('2013-06-30T21:00')
    del records[167,pd.Timestamp('2013-07-01 00:00'),9]
    with pytest.raises(ValueError,match='coverage'):convert('unused','2013-07-01','2013-07-01 21:00')

def test_native_grib_decode_and_workflow_download(tmp_path, monkeypatch):
    import eccodes as ec
    import logging
    from symfluence_carra2.acquisition import CARRA2Acquirer
    fixture=tmp_path/'fixture.grib'
    with fixture.open('wb') as f:
        for cycle in [0,1200]:
            for param,(_,_,accum) in FIELDS.items():
                for step in [6,9,12,15,18]:
                    g=ec.codes_grib_new_from_samples('polar_stereographic_sfc_grib2')
                    try:
                        for k,v in [('Nx',2),('Ny',2),('latitudeOfFirstGridPointInDegrees',64.0),('longitudeOfFirstGridPointInDegrees',342.0),('dataDate',20130702),('dataTime',cycle),('paramId',param),('stepType','accum' if accum else 'instant'),('endStep',step)]:ec.codes_set(g,k,v)
                        if accum:ec.codes_set(g,'startStep',0)
                        ec.codes_set_values(g,np.full(4,step if accum else 280.))
                        ec.codes_write(g,f)
                    finally:ec.codes_release(g)
    class FakeCDS:
        def retrieve(self,dataset,request,target):
            assert dataset=='reanalysis-pan-carra'
            assert request['area']==[64.6,-19.4,63.7,-17.5]
            from shutil import copyfile
            copyfile(fixture,target)
    monkeypatch.setattr('cdsapi.Client',FakeCDS)
    cfg={'DOMAIN_NAME':'test_carra2', 'BOUNDING_BOX_COORDS':'64.6/-19.4/63.7/-17.5',
         'EXPERIMENT_TIME_START':'2013-07-02 09:00', 'EXPERIMENT_TIME_END':'2013-07-03 06:00'}
    out=CARRA2Acquirer(cfg,logging.getLogger('test')).download(tmp_path/'output')
    import xarray as xr
    with xr.open_dataset(out) as ds:
        assert ds.sizes['time']==8
        assert ds.air_temperature.dims == ('time', 'point')
        assert '+proj=stere' in ds.attrs['MAP_PROJ4_STR']
        np.testing.assert_allclose(ds.precipitation_flux,1/3600)


def test_unused_forecast_cycle_cannot_block_requested_window(monkeypatch):
    records = {}
    used = pd.Timestamp('2013-06-07 12:00')
    unused = pd.Timestamp('2013-06-08 12:00')
    for c in (used, unused):
        for p, (_, _, accum) in FIELDS.items():
            for step in (6, 9, 12, 15, 18):
                records[p, c, step] = np.full((1, 1), step * 2000. if accum else 280.)
    # Downloaded next-day forecast has a real rejection, but supplies no output.
    records[169, unused, 15] = records[169, unused, 12] - 1086.8125
    grid = (np.array([[64.]]), np.array([[-18.]]), {})
    monkeypatch.setattr('symfluence_carra2.acquisition.read_grib', lambda path: (records, grid))
    ds = convert('unused', '2013-06-07 21:00', '2013-06-08 00:00')
    np.testing.assert_allclose(ds.surface_downwelling_shortwave_flux, 6000. / 10800.)
    assert '2013-06-08T12:00:00' not in ds.attrs['numerical_correction_audit']
    # The same decrease in a consumed cycle must still fail, even if the
    # offending interval is outside the requested subset of that cycle:
    # full-cycle correction depends on all five accumulated values.
    records[169, used, 15] = records[169, used, 12] - 1086.8125
    with pytest.raises(ValueError, match='shortwave_flux, forecast cycle.*budget'):
        convert('unused', '2013-06-07 21:00', '2013-06-08 00:00')
