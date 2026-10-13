import json
import numpy as np
import pandas as pd
import pytest
from symfluence.models.prms.cryosphere import calculate_constraints


def test_constraints_exclude_future_and_require_outputs(tmp_path):
    obs = tmp_path / 'data/observations/cryosphere'
    obs.mkdir(parents=True)
    dates = pd.date_range('2012-10-01', '2017-09-30')
    snow = pd.DataFrame(dict(YYYY=dates.year, MM=dates.month, DD=dates.day,
                             fsca_outside_glaciers=np.where(dates.year < 2017, 50., 0.)))
    snow.to_csv(obs/'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';', index=False)
    pd.DataFrame({'annual_net_MB':[0.,0.,0.,999.], 'g_area_dyn':[1.]*4},index=[2014,2015,2016,2017]).to_csv(obs/'glacier_timeseries.csv',sep=';')
    p=tmp_path/'params.dat'
    p.write_text('** Parameters **\n####\nhru_area\n1\nnhru\n2\n2\n247.105381467\n247.105381467\n####\nglacier_frac_init\n1\nnhru\n2\n2\n0\n1\n')
    names=['snowcov_area 1','snowcov_area 2','glacier_frac 1','glacier_frac 2','basin_gl_storage 1']
    s=tmp_path/'statvar.dat'
    text='5\n'+'\n'.join(names)+'\n'
    text+='\n'.join(f'{i+1} {d.year} {d.month} {d.day} 0 0 0 .5 .5 0 1 0' for i,d in enumerate(dates))
    s.write_text(text)
    r=calculate_constraints(s,p,tmp_path,'2012-10-01, 2016-09-30')
    assert r['snow_score']==1 and r['glacier_score']==1
    assert r['snow_observation_count']==1461
    saved=json.loads((tmp_path/'cryosphere_metrics.json').read_text())
    assert [x['year'] for x in saved['annual_glacier_balance']]==[2014,2015,2016]
    s.write_text(text.replace('snowcov_area 1','wrong_variable 1'))
    with pytest.raises(KeyError):calculate_constraints(s,p,tmp_path,'2012-10-01, 2016-09-30')
