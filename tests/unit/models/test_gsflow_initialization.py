"""Physical constraints and complete solver controls for generated GSFLOW."""
from datetime import datetime
import logging
from types import SimpleNamespace as NS
import pytest
from symfluence.models.gsflow.config_schema import GSFLOWConfig
from symfluence.models.gsflow.preprocessor import GSFLOWPreProcessor
from symfluence.models.gsflow.calibration.worker import GSFLOWWorker
from symfluence.models.gsflow.calibration.parameter_manager import GSFLOWParameterManager


def test_generation_and_specific_yield_constraint(tmp_path):
    cfg=GSFLOWConfig(initial_recharge=.003,uzf_saturated_water_content=.30)
    p=object.__new__(GSFLOWPreProcessor)
    p.config=NS(model=NS(gsflow=cfg));p._get_config_value=lambda getter,default=None,**kw:getter()
    p._props={'area_km2':10,'elev':100.,'relief':100.}
    p.distributed_gw=False;p.gw_grid_n=3;p.modflow_dir=tmp_path;p._output_relpath='out'
    p._generate_modflow_packages(datetime(2010,1,1),datetime(2010,12,31))
    assert 'COMPLEX' in (tmp_path/'bow.nwt').read_text()
    assert 'SPECIFIED' not in (tmp_path/'bow.nwt').read_text()
    assert 'CONSTANT  0.003' in (tmp_path/'bow.uzf').read_text()
    worker=object.__new__(GSFLOWWorker);worker.logger=logging.getLogger('test')
    assert worker._update_upw_parameters(tmp_path,{'SY':.29,'K':1.})
    before=(tmp_path/'bow.upw').read_text()
    assert not worker._update_upw_parameters(tmp_path,{'SY':.31})
    assert (tmp_path/'bow.upw').read_text()==before


def test_calibration_bounds_respect_configured_porosity():
    p=object.__new__(GSFLOWParameterManager);p.gsflow_params=['SY','K']
    p._get_config_value=lambda *a,**kw: {'K':{'transform':'log'}} if kw['dict_key']=='GSFLOW_PARAM_BOUNDS' else .25
    bounds=p._load_parameter_bounds()
    assert bounds['SY']['max']==pytest.approx(.24)
    assert bounds['K']['transform']=='log'
