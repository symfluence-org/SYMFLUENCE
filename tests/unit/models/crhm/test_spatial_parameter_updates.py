"""Calibration must retain one parameter value per CRHM HRU."""
import logging
from symfluence.models.crhm.calibration.worker import CRHMWorker

def test_worker_retains_hru_vector(tmp_path):
    p=tmp_path/'model.prj'
    p.write_text('Parameters:\n######\nShared soil_rechr_max <0 to 350>\n60 60 60 60 60\n######\n')
    worker=object.__new__(CRHMWorker);worker.logger=logging.getLogger('test')
    assert worker._update_prj_file(p,{'soil_rechr_max':40.0})
    assert p.read_text().splitlines()[3].split()==['40.000000']*5

def test_generated_forcing_rejects_missing_net_radiation_calibration():
    from types import SimpleNamespace as NS
    import pytest
    from symfluence.models.crhm.preprocessor import CRHMPreProcessor
    from symfluence.models.crhm.config_schema import CRHMConfig
    assert 'nfactor' not in CRHMConfig().params_to_calibrate.split(',')
    p=object.__new__(CRHMPreProcessor)
    p.config=NS(model=NS(crhm=CRHMConfig(params_to_calibrate='nfactor')))
    with pytest.raises(ValueError,match='QnD'):
        p._generate_project_file()

def test_integer_sublimation_switch_is_not_continuously_calibrated():
    from types import SimpleNamespace as NS
    import pytest
    from symfluence.models.crhm.preprocessor import CRHMPreProcessor
    from symfluence.models.crhm.config_schema import CRHMConfig
    p=object.__new__(CRHMPreProcessor)
    p.config=NS(model=NS(crhm=CRHMConfig(params_to_calibrate='Qe_subl_from_SWE')))
    with pytest.raises(ValueError,match='integer switch'):
        p._generate_project_file()
