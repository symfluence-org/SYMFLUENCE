import logging
import pytest
from symfluence.models.gsflow.calibration.worker import GSFLOWWorker

@pytest.mark.parametrize('bad',['Grav_S','Sat2Grav_Q','StreamOut_Q',None])
def test_finite_outlet_does_not_hide_invalid_coupled_storage(tmp_path,bad):
    cols=['StreamOut_Q','Grav_S','Sat2Grav_Q'];v={c:86400. for c in cols}
    if bad:v[bad]=float('nan')
    p=tmp_path/'gsflow.csv';p.write_text('Date,'+','.join(cols)+'\n2013-01-01,'+','.join(str(v[c]) for c in cols)+'\n')
    w=object.__new__(GSFLOWWorker);w.logger=logging.getLogger('test')
    result=w._extract_streamflow_from_csv(p)
    if bad:assert result is None
    else:assert result.iloc[0]==1.
