"""merge_restarts must use only the newest restart time (SUMMA writes one per water year)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'src' / 'symfluence' / 'models' / 'summa'))
national_run = pytest.importorskip('national_run')


def block(path, ids, val):
    xr.Dataset({'hruId': ('hru', np.array(ids)), 'scalarSWE': (('scalarv', 'hru'), np.full((1, len(ids)), val)),
                'gruId': ('gru', np.array(ids))}).to_netcdf(path)


def test_latest_time_only(tmp_path):
    out, settings = tmp_path / 'out', tmp_path / 'settings'
    out.mkdir(), settings.mkdir()
    xr.Dataset({'hruId': ('hru', np.arange(1, 5))}).to_netcdf(settings / 'attributes.nc')
    for year, val in ((2012, 1.0), (2013, 2.0), (2016, 5.0)):
        block(out / f'ice_WY2012-2016_restart_{year}100100_G00001-00002.nc', [1, 2], val)
        block(out / f'ice_WY2012-2016_restart_{year}100100_G00003-00004.nc', [3, 4], val)
    assert [f.name[24:34] for f in national_run.latest_restarts(out, 'ice_WY2012-2016')] == ['2016100100'] * 2
    name = national_run.merge_restarts(out, settings, 'ice_WY2012-2016')
    with xr.open_dataset(settings / name) as m:
        assert m.sizes['hru'] == 4
        assert list(m.hruId.values) == [1, 2, 3, 4]
        assert float(m.scalarSWE.min()) == 5.0


def test_count_guard(tmp_path):
    out, settings = tmp_path / 'out', tmp_path / 'settings'
    out.mkdir(), settings.mkdir()
    xr.Dataset({'hruId': ('hru', np.arange(1, 6))}).to_netcdf(settings / 'attributes.nc')
    block(out / 'land_WY2012-2016_restart_2016100100_G00001-00002.nc', [1, 2], 1.0)
    with pytest.raises(ValueError):
        national_run.merge_restarts(out, settings, 'land_WY2012-2016')
