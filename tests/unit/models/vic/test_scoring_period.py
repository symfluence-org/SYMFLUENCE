"""Protect calibration against evaluation-period leakage in the VIC worker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.vic.calibration.worker import VICWorker


@pytest.mark.parametrize('nested', [False, True])
def test_vic_scores_only_configured_period(tmp_path, monkeypatch, nested):
    times = pd.date_range('2012-09-01', periods=60)
    obs = np.arange(1., 61.)
    simulated = obs.copy()
    simulated[30:] = obs[30:][::-1] * 3
    xr.Dataset({'OUT_RUNOFF': (('time','lat','lon'), simulated[:,None,None]),
                'OUT_BASEFLOW': (('time','lat','lon'), np.zeros((60,1,1)))},
               coords={'time':times}).to_netcdf(tmp_path/'vic_output.nc')
    worker = VICWorker()
    monkeypatch.setattr(worker._streamflow_metrics,'get_catchment_area',lambda *a,**kw:86.4)
    monkeypatch.setattr(worker._streamflow_metrics,'load_observations',lambda *a,**kw:(obs,times))
    cfg={'DOMAIN_NAME':'test','SYMFLUENCE_DATA_DIR':str(tmp_path),'VIC_WARMUP_DAYS':0}
    period='2012-09-01, 2012-09-30'
    cfg.update({'domain':{'calibration_period':period}} if nested else {'CALIBRATION_PERIOD':period})
    assert worker.calculate_metrics(tmp_path,cfg)['kge']==pytest.approx(1.)
    assert worker.calculate_metrics(tmp_path,cfg,period='2012-10-01, 2012-10-30')['kge']<0
    assert 'error' in worker.calculate_metrics(tmp_path,cfg,period='not-a-period')
