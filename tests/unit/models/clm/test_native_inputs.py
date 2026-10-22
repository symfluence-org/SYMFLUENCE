from pathlib import Path
from types import SimpleNamespace
import netCDF4 as nc
import numpy as np
import pytest
from symfluence.models.clm.surface_layout import canonicalize
from symfluence.models.clm.nuopc_generator import CLMNuopcGenerator
def test_surface_vertical_axes_precede_horizontal_axes(tmp_path):
 p=tmp_path/'surface.nc'
 with nc.Dataset(p,'w') as d:
  for n,size in [('lsmlat',1),('lsmlon',2),('nglcecp1',11),('unused',10)]:d.createDimension(n,size)
  v=d.createVariable('PCT_GLC_MEC','f8',('lsmlat','lsmlon','nglcecp1'));v[:]=np.arange(22).reshape(1,2,11)
 canonicalize(p)
 with nc.Dataset(p) as d:
  assert d['PCT_GLC_MEC'].dimensions==('nglcecp1','lsmlat','lsmlon')
  np.testing.assert_array_equal(d['PCT_GLC_MEC'][:,0,1],np.arange(11,22))
  assert len(d.dimensions['unused'])==10
def test_field_dictionary_matches_binary_case(tmp_path):
 case=tmp_path/'cases/actual';(case/'bld').mkdir(parents=True);(case/'run').mkdir()
 (case/'bld/cesm.exe').write_text('binary');(case/'run/fd.yaml').write_text('fields')
 (tmp_path/'bin').mkdir();(tmp_path/'bin/cesm.exe').symlink_to(case/'bld/cesm.exe')
 g=object.__new__(CLMNuopcGenerator);g.pp=SimpleNamespace(_get_install_path=lambda:tmp_path)
 assert g._build_case_path()==case
def test_clm_native_failure_is_not_success(monkeypatch):
 from symfluence.models.clm.runner import CLMRunner
 from symfluence.core.modeling.base import BaseModelRunner
 from symfluence.core.exceptions import ModelExecutionError
 monkeypatch.setattr(BaseModelRunner,'run',lambda *a,**k:None)
 runner=object.__new__(CLMRunner)
 with pytest.raises(ModelExecutionError,match='successful native run'):runner.run()
