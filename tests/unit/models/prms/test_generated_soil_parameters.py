"""Generated PRMS5 soil capacities must be active and internally consistent."""
from types import SimpleNamespace as NS
import pytest
from symfluence.models.prms.preprocessor import PRMSPreProcessor

PROPS=dict(lat=64.,lon=-19.,elev=700.,area_km2=1145.,slope=5.,aspect=0.)
def values(path):
    body=path.read_text().split('** Parameters **')[1]
    out={}
    for b in body.split('####'):
        x=b.strip().splitlines()
        if x:
            n=int(x[1]);out[x[0]]=[float(v) for v in x[4+n:]]
    return out

@pytest.mark.parametrize('mode',['PRMS5','DAILY'])
def test_prms_soil_defaults_preserve_depths(tmp_path,mode):
    p=object.__new__(PRMSPreProcessor);p.settings_dir=tmp_path;p.use_obs_solar=True
    p.config=NS(model=NS(prms=NS(nhru=5,model_mode=mode,parameter_overrides={},routing_segments=1,
                                  separate_glacier_depletion_curve=False,forcing_mode='station')))
    p._get_config_value=lambda getter,default=None,**kw:default
    p._get_catchment_properties=lambda:PROPS
    p._get_dominant_classes=lambda:dict(cov_type=0,soil_type=2)
    p._generate_parameter_file();v=values(tmp_path/'params.dat')
    if mode=='PRMS5':
        assert 'soil_rechr_max' not in v
        for cap,rechr,mi,ri in zip(v['soil_moist_max'],v['soil_rechr_max_frac'],v['soil_moist_init_frac'],v['soil_rechr_init_frac']):
            assert cap*rechr==pytest.approx(2.)
            assert cap*mi==pytest.approx(3.)
            assert cap*rechr*ri==pytest.approx(1.)
    else:assert v['soil_rechr_max']==[2.]*5
    p.config.model.prms.parameter_overrides={'soil_rechr_max':{}}
    if mode=='PRMS5':
        with pytest.raises(ValueError,match='ignores depth'):
            p._generate_parameter_file()

@pytest.mark.parametrize('glacier,requested,expected',[
    (False,'auto','strmflow'),(True,'auto','muskingum'),(False,'muskingum','muskingum')])
def test_glacier_discharge_uses_supported_routing(tmp_path,glacier,requested,expected):
    from datetime import datetime
    from symfluence.models.prms.config_schema import PRMSConfig
    p=object.__new__(PRMSPreProcessor);p.settings_dir=tmp_path;p.forcing_dir=tmp_path
    p.config=NS(model=NS(prms=PRMSConfig(glacier_flag=glacier,streamflow_module=requested)))
    p._get_config_value=lambda getter,default=None,**kw:getter()
    p._generate_control_file(datetime(2010,1,1),datetime(2010,12,31))
    s=(tmp_path/'control.dat').read_text()
    assert f'strmflow_module\n1\n4\n{expected}\n' in s
    assert 'basin_storage\n' in s and 'basin_stflow_out\n' in s

def test_glacier_rejects_routing_that_discards_melt(tmp_path):
    from datetime import datetime
    from symfluence.models.prms.config_schema import PRMSConfig
    p=object.__new__(PRMSPreProcessor);p.settings_dir=tmp_path;p.forcing_dir=tmp_path
    p.config=NS(model=NS(prms=PRMSConfig(glacier_flag=True,streamflow_module='strmflow')))
    p._get_config_value=lambda getter,default=None,**kw:getter()
    with pytest.raises(ValueError,match='omits glacier discharge'):
        p._generate_control_file(datetime(2010,1,1),datetime(2010,12,31))

def test_fixed_glacier_geometry_requires_capable_binary_and_pure_hrus(tmp_path):
    from datetime import datetime
    from symfluence.models.prms.config_schema import PRMSConfig
    p=object.__new__(PRMSPreProcessor);p.settings_dir=tmp_path;p.forcing_dir=tmp_path
    p._get_config_value=lambda getter,default=None,**kw:getter()
    cfg=dict(glacier_flag=True,glacier_dynamics=False,nhru=2,install_path=str(tmp_path),
             parameter_overrides={'glacier_frac_init':{'dimensions':['nhru'],'dtype':2,'values':[0,1]}})
    p.config=NS(model=NS(prms=PRMSConfig(**cfg)))
    with pytest.raises(ValueError,match='native PRMS build'):
        p._generate_control_file(datetime(2010,1,1),datetime(2010,12,31))
    (tmp_path/'prms').write_bytes(b'glacier_dynamics')
    p._generate_control_file(datetime(2010,1,1),datetime(2010,12,31))
    assert 'glacier_dynamics\n1\n1\n0' in (tmp_path/'control.dat').read_text()
    cfg['parameter_overrides']['glacier_frac_init']['values']=[0,.4]
    p.config=NS(model=NS(prms=PRMSConfig(**cfg)))
    with pytest.raises(ValueError,match='pure glacier'):
        p._generate_control_file(datetime(2010,1,1),datetime(2010,12,31))
