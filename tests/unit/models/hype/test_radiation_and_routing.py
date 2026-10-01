from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.forcing_processor import HYPEForcingProcessor
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager


def test_shortwave_energy_conversion_and_missing_day_rejection(tmp_path):
    dates = pd.date_range('2010-01-01', periods=16, freq='3h')
    ds = xr.Dataset({'surface_downwelling_shortwave_flux': (('time','hru'), np.full((16,2),100.)),
                     'hruId': ('hru',[4,9])}, coords={'time':dates,'hru':[0,1]})
    ds.surface_downwelling_shortwave_flux.attrs['units']='W m-2'
    p=tmp_path/'forcing.nc';ds.to_netcdf(p)
    proc=HYPEForcingProcessor({},logging.getLogger('test'),tmp_path,tmp_path,tmp_path)
    out=pd.read_csv(proc.write_shortwave_obs([p]),sep='\t',index_col=0)
    assert list(out.columns)==['4','9']
    np.testing.assert_allclose(out.to_numpy(),8.64)
    ds.isel(time=slice(1,None)).to_netcdf(p)
    with pytest.raises(ValidationError,match='Incomplete'):
        proc.write_shortwave_obs([p])


def test_radiation_info_and_glacier_parameter_substitution(tmp_path):
    cfg={'HYPE_SNOW_MELT_MODEL':2,'HYPE_GLACIER_ALBEDO':.31}
    mgr=HYPEConfigManager(cfg,logging.getLogger('test'),tmp_path)
    mgr.write_par_file(land_uses=np.array([15,16]),params={'glaccmrad':.7})
    par=(tmp_path/'par.txt').read_text()
    assert 'glaccmrad\t0.7' in par
    assert 'glacalb\t0.31' in par
    pd.DataFrame({'time':pd.date_range('2010-01-01',periods=2),'1':[1,1]}).to_csv(tmp_path/'Pobs.txt',sep='\t',index=False)
    mgr.write_info_filedir(0,str(tmp_path/'results'),experiment_start='2010-01-01',experiment_end='2010-01-02')
    info=(tmp_path/'info.txt').read_text()
    assert 'readswobs\ty' in info
    assert 'modeloption snowmeltmodel\t2' in info


def test_river_override_targets_only_specified_id(tmp_path):
    mgr=HYPEGeoDataManager({'HYPE_RIVER_LENGTH_OVERRIDES':{1:11557.4}},logging.getLogger('test'),tmp_path,{})
    geo=pd.DataFrame({'subid':[5,1],'rivlen':[100.,100.]})
    assert mgr.apply_river_length_overrides(geo).rivlen.tolist()==[100.,11557.4]
    with pytest.raises(ValidationError,match='does not identify'):
        mgr.apply_river_length_overrides(geo.iloc[:1])


def test_percolation_parameters_cover_every_soil_class(tmp_path):
    HYPEGeoDataManager({},logging.getLogger('test'),tmp_path,{})._write_geoclass(
        pd.DataFrame({'SLC':[1,2],'landcover':[15,16],'soil':[8,8]}))
    mgr=HYPEConfigManager({'HYPE_SOIL_PERCOLATION':[20.,5.]},logging.getLogger('test'),tmp_path)
    mgr.write_par_file(land_uses=np.array([15,16]),params={'mperc2':3.})
    rows={s.split()[0]:s.split()[1:] for s in (tmp_path/'par.txt').read_text().splitlines() if s and not s.startswith('!')}
    assert [float(v) for v in rows['mperc1']]==[20.]*8
    assert [float(v) for v in rows['mperc2']]==[3.]*8
    mgr=HYPEConfigManager({'HYPE_SOIL_PERCOLATION':[-1,5]},logging.getLogger('test'),tmp_path)
    with pytest.raises(ValidationError,match='nonnegative'):
        mgr.write_par_file(land_uses=np.array([15,16]))
