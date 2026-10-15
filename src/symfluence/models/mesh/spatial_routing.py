"""Conservative fixed travel-time routing for explicit MESH response units.

Kernels must be supplied from documented drainage geometry. This is linear
translation, not dynamic channel hydraulics or lake routing. No runoff scale
factor is fitted. Water arriving after the run end is retained in a tail file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .errors import MESHConfigurationError, MESHInputError, MESHOutputError


def route_spatial_runoff(directory, kernel_path):
    directory = Path(directory)
    units = json.loads((directory/'spatial_units.json').read_text())['units']
    ids = [str(u['unit_id']) for u in units]
    kernel = pd.read_csv(kernel_path, dtype={'unit_id': str})
    if set(kernel.unit_id) != set(ids):
        raise MESHInputError('Routing kernel and spatial unit identities differ')
    values = kernel[['lag_days', 'weight']].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise MESHInputError('Routing kernel must have finite nonnegative lags and weights')
    if (kernel.lag_days > 365).any() or not np.all(kernel.lag_days == kernel.lag_days.astype(int)):
        raise MESHConfigurationError('Routing kernels require integer daily lags up to 365 days')
    if not np.allclose(kernel.groupby('unit_id').weight.sum(), 1., rtol=0., atol=1e-10):
        raise MESHConfigurationError('Routing kernel must preserve each unit volume')
    frame = pd.read_csv(directory/'results/RFF_D.csv', header=None, index_col=0).dropna(axis=1, how='all')
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq='D')):
        raise MESHConfigurationError('Routing requires unique complete daily runoff')
    with xr.open_dataset(directory/'MESH_drainage_database.nc') as db:
        area = db.GridArea.values
    if len(area) != len(ids)+1 or area[-1] != 0 or frame.shape[1] != len(area):
        raise MESHConfigurationError('Routing requires one explicit grid per unit plus a zero-area outlet')
    runoff = frame.iloc[:, :len(ids)].to_numpy()*area[:-1]/86400000.
    if not np.isfinite(runoff).all() or (runoff < -1e-8).any():
        raise MESHInputError('Invalid unit runoff for routing')
    routed = np.zeros(len(runoff)+int(kernel.lag_days.max()))
    for j, uid in enumerate(ids):
        rows = kernel[kernel.unit_id == uid]
        for row in rows.itertuples():
            lag = int(row.lag_days)
            routed[lag:lag+len(runoff)] += runoff[:, j]*row.weight
    residual_m3 = float((routed.sum()-runoff.sum())*86400)
    if not np.isclose(routed.sum(), runoff.sum(), rtol=1e-12, atol=1e-8):
        raise MESHOutputError('Routing volume conservation failed')
    pd.DataFrame({'YEAR': dates.year, 'JDAY': dates.dayofyear, 'QOSIM1': routed[:len(dates)]}).to_csv(
        directory/'MESH_spatial_streamflow.csv', index=False)
    tail = pd.Series(routed[len(dates):], index=pd.date_range(dates[-1]+pd.Timedelta(days=1), periods=len(routed)-len(dates)), name='discharge_cms')
    tail.to_csv(directory/'MESH_routing_tail.csv', index_label='date')
    report = {'method': 'fixed geometry-derived daily travel-time kernel',
              'kernel': str(Path(kernel_path).resolve()), 'volume_residual_m3': residual_m3,
              'end_routing_storage_m3': float(tail.sum()*86400), 'initial_routing_storage_m3': 0.}
    (directory/'MESH_routing_balance.json').write_text(json.dumps(report, indent=2))
    return report
