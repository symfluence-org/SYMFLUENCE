import numpy as np
import pytest
from symfluence.models.prms.balance import check_water_balance

@pytest.mark.parametrize('leak',[0.,.1])
def test_glacier_snow_is_not_double_counted(tmp_path,leak):
    names=['basin_storage','basin_ppt','basin_gwstor_minarea_wb','basin_actet','basin_stflow_out','basin_gwsink','basin_glacrevap','pkwater_equiv','glacier_frac']
    # One fully glaciated HRU, one inch snowfall. Both native storage terms
    # contain this snow; physical storage rises one inch, not two.
    data=np.array([[1,2013,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
                   [2,2013,1,2,0,0,0,2,1+leak,0,0,0,0,0,1,1]],dtype=float)
    p=tmp_path/'statvar.dat'
    p.write_text(str(len(names))+'\n'+''.join(n+' 1\n' for n in names))
    with p.open('a') as f:np.savetxt(f,data)
    params=tmp_path/'params.dat';params.write_text('** Parameters **\n####\nhru_area\n1\nnhru\n1\n2\n10\n')
    if leak:
        with pytest.raises(ValueError,match='water balance failed'):check_water_balance(p,params,.05,True)
    else:assert check_water_balance(p,params,.05,True)['cumulative_residual_mm']==0
