from __future__ import annotations

import json

import pandas as pd
import pytest

from symfluence.models.raven.cryosphere import glacier_metrics


def test_only_complete_calibration_years_and_liquid_storage(tmp_path):
    geometry=tmp_path/'geometry.json'
    geometry.write_text(json.dumps({'hrus':[{'id':1,'area_km2':1,'ice':True}]}))
    for name, values in {'GLACIER_ICE':[10000,9000,8000], 'SNOW':[0,0,0],
                         'SNOW_LIQ':[0,100,100], 'GLACIER':[0,100,100]}.items():
        (tmp_path/f'{name}_Daily_Average_ByHRU.csv').write_text(
            ',HRU:,1,\ntime,day,mean,\n'+''.join(f'{i},{y}-09-30,{v},\n' for i,(y,v) in enumerate(zip([2013,2014,2015],values))))
    obs=tmp_path/'obs.csv'
    pd.DataFrame({'g_area_dyn':[1,1,1], 'annual_net_MB':[-999,-.8,999]},index=[2013,2014,2015]).to_csv(obs,sep=';')
    result=glacier_metrics(tmp_path,geometry,obs,'2012-10-01, 2014-09-30')
    assert result['glacier_balance_year_count']==1
    assert result['glacier_mb_rmse_mwe']==pytest.approx(0)
    (tmp_path/'GLACIER_Daily_Average_ByHRU.csv').unlink()
    with pytest.raises(FileNotFoundError):
        glacier_metrics(tmp_path,geometry,obs,'2012-10-01, 2014-09-30')
