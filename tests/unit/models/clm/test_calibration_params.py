from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.clm.calibration.parameter_manager import (
 CLM_DEFAULT_BOUNDS,
 CLM_PARAM_DEFS,
 CLM_REMOVED_PARAMS,
 CLM_SURFDATA_FIELDS_READ,
 CLMParameterManager,
 apply_texture_offsets,
 check_removed_params,
)
from symfluence.models.clm.calibration.worker import CLMWorker


def test_every_surfdata_parameter_targets_a_field_ctsm_reads():
 # CTSM 5.3 derives watsat/hksat/sucsat/bsw from texture; editing them is a no-op.
 for name,(target,var,_) in CLM_PARAM_DEFS.items():
  if target=='surfdata':assert var in CLM_SURFDATA_FIELDS_READ,name
 assert {t for t,_,_ in CLM_PARAM_DEFS.values()}<={'namelist','params','surfdata'}
 assert set(CLM_DEFAULT_BOUNDS)==set(CLM_PARAM_DEFS)

@pytest.mark.parametrize('name',['hksat_mult','watsat_mult','bsw_mult','sucsat_mult','route_k'])
def test_removed_parameters_are_rejected_with_reason(name):
 assert name not in CLM_PARAM_DEFS and name in CLM_REMOVED_PARAMS
 with pytest.raises(ConfigurationError,match=name):check_removed_params(['e_ice',name])

def test_parameter_manager_rejects_removed_names(tmp_path):
 cfg={'CLM_PARAMS_TO_CALIBRATE':'e_ice,route_k','SYMFLUENCE_DATA_DIR':str(tmp_path),'DOMAIN_NAME':'d','EXPERIMENT_ID':'e'}
 with pytest.raises(ConfigurationError,match='route_k'):CLMParameterManager(cfg,logging.getLogger('t'),tmp_path)

def test_texture_offsets_stay_valid():
 sand,clay=apply_texture_offsets(np.array([65.,95.,5.]),np.array([7.,4.,20.]),sand_offset=10.,clay_offset=5.)
 np.testing.assert_allclose(sand,[75.,100.,15.])
 np.testing.assert_allclose(clay,[12.,0.,25.])  # clay yields so sand+clay <= 100
 sand,clay=apply_texture_offsets(np.array([3.]),np.array([2.]),sand_offset=-10.,clay_offset=-10.)
 assert sand[0]==0 and clay[0]==0

def _surfdata(path):
 with nc.Dataset(path,'w') as d:
  for n,s in [('nlevsoi',2),('lsmlat',1),('lsmlon',1)]:d.createDimension(n,s)
  for v,val in [('PCT_SAND',[65.,70.]),('PCT_CLAY',[7.,6.])]:
   x=d.createVariable(v,'f8',('nlevsoi','lsmlat','lsmlon'));x[:]=np.array(val).reshape(2,1,1)
  d.createVariable('FMAX','f8',('lsmlat','lsmlon'))[:]=.5

def test_worker_applies_texture_and_fmax(tmp_path):
 _surfdata(tmp_path/'surfdata_clm.nc')
 w=object.__new__(CLMWorker);w.logger=logging.getLogger('t')
 assert w._update_surfdata(tmp_path,{'sand_offset':-5.,'clay_offset':2.,'fmax':.3})
 with nc.Dataset(tmp_path/'surfdata_clm.nc') as d:
  np.testing.assert_allclose(d['PCT_SAND'][:].ravel(),[60.,65.])
  np.testing.assert_allclose(d['PCT_CLAY'][:].ravel(),[9.,8.])
  assert float(d['FMAX'][:])==pytest.approx(.3)

def test_worker_has_no_post_hoc_routing():
 assert not hasattr(CLMWorker,'_apply_routing')

def test_runner_creates_init_interp_directory(tmp_path):
 from symfluence.models.clm.runner import CLMRunner
 settings=tmp_path/'settings';settings.mkdir();(settings/'lnd_in').write_text('&clm_inparm\n/\n')
 runner=object.__new__(CLMRunner);runner.settings_dir=settings;runner.logger=logging.getLogger('t')
 run=tmp_path/'run';run.mkdir();runner._setup_run_directory(run)
 assert (run/'init_generated_files').is_dir() and (run/'lnd_in').exists()
