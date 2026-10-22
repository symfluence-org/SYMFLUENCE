# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mHM parameter-namelist updates and forcing-side pseudo-parameters.

The worker used to match parameter names case-insensitively anywhere on a
line, so writing ``degreeDayFactor_pervious`` also rewrote
``maxDegreeDayFactor_pervious``. Every calibration then capped snowmelt at the
(near-zero) base degree-day factor.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from symfluence.models.mhm.calibration.worker import MHMWorker

NML = """&snow1
  degreeDayFactor_pervious                      = 0.0000, 2.0, 0.5000, 1, 1
  maxDegreeDayFactor_pervious                   = 0.0000, 8.0, 4.0, 1, 1
  degreeDayFactor_impervious                    = 0.0000, 1.0, 0.5000, 1, 1
  maxDegreeDayFactor_impervious                 = 0.0000, 8.0, 3.5, 1, 1
/
"""


def _worker():
    return MHMWorker({}, logging.getLogger("test"))


def test_base_factor_does_not_rewrite_max_factor(tmp_path):
    nml = tmp_path / "mhm_parameter.nml"
    nml.write_text(NML)
    assert _worker()._update_namelist_file(
        nml, {"degreeDayFactor_pervious": 0.25, "degreeDayFactor_impervious": 0.75})
    lines = {ln.split("=")[0].strip(): ln.split("=")[1] for ln in nml.read_text().splitlines() if "=" in ln}
    assert lines["degreeDayFactor_pervious"].split(",")[2].strip() == "0.250000"
    assert lines["maxDegreeDayFactor_pervious"].split(",")[2].strip() == "4.0"
    assert lines["degreeDayFactor_impervious"].split(",")[2].strip() == "0.750000"
    assert lines["maxDegreeDayFactor_impervious"].split(",")[2].strip() == "3.5"


def test_localise_outputs_rewrites_only_output_paths(tmp_path):
    nml = tmp_path / "mhm.nml"
    nml.write_text(
        '  dirConfigOut = "/shared/out/"\n'
        '  dir_Out(1) = "/shared/out/"\n'
        '  mhm_file_RestartOut(1) = "/shared/out/mHM_restart_001.nc"\n'
        '  dir_Precipitation(1) = "/shared/forcing/"\n')
    _worker()._localise_outputs(nml, tmp_path / "proc_1")
    text = nml.read_text()
    assert f'dir_Out(1) = "{tmp_path}/proc_1/"' in text
    assert f'mhm_file_RestartOut(1) = "{tmp_path}/proc_1/mHM_restart_001.nc"' in text
    assert 'dir_Precipitation(1) = "/shared/forcing/"' in text


def test_glacier_ice_melt_waits_for_snow_cover():
    # Day 0: 10 mm snowfall at -5 C. Days 1-3: +4 C, no precipitation.
    pre = np.array([[10.0], [0.0], [0.0], [0.0]])
    tavg = np.array([[-5.0], [4.0], [4.0], [4.0]])
    melt = MHMWorker._glacier_ice_melt(pre, tavg, ddf_ice=6.0, t_snow=0.0)
    # Snow melts at 3 mm/C/d -> 12 mm potential on day 1 clears the 10 mm and
    # leaves 1/6 of the energy for ice; days 2-3 are bare ice.
    np.testing.assert_allclose(melt[:, 0], [0.0, 24.0 / 6.0, 24.0, 24.0])


def _synthetic_glacier(years=3):
    import pandas as pd
    index = pd.date_range('2012-10-01', periods=365 * years + 1, freq='D')
    doy = index.dayofyear.to_numpy()
    tavg = (6.0 * np.sin(2 * np.pi * (doy - 110) / 365.25) - 1.0)[:, None].repeat(2, axis=1)
    pre = np.full_like(tavg, 4.0)
    return index, pre, tavg


def test_net_balance_counts_only_complete_water_years():
    index, pre, tavg = _synthetic_glacier()
    melt, snow = MHMWorker._glacier_ice_melt(pre, tavg, 6.0, 0.0, return_snow=True)
    balance = MHMWorker._glacier_net_balance(melt, snow, np.ones(2), index)
    assert sorted(balance) == [2013, 2014, 2015]
    first = index.get_indexer(['2013-09-30'])[0]
    expected = (snow[first].mean() - melt[:first + 1].mean(axis=1).sum()) / 1000.0
    assert balance[2013] == pytest.approx(expected)


def test_solved_ice_factor_reproduces_the_target_balance():
    index, pre, tavg = _synthetic_glacier()
    worker = _worker()
    cfg = {'CALIBRATION_PERIOD': '2012-10-01, 2014-09-30'}
    years = worker._calibration_water_years(cfg, index)
    assert years == [2013, 2014]
    ddf = worker._solve_ice_ddf(pre, tavg, 0.0, np.ones(2), index, years, -1.5)
    melt, snow = MHMWorker._glacier_ice_melt(pre, tavg, ddf, 0.0, return_snow=True)
    balance = MHMWorker._glacier_net_balance(melt, snow, np.ones(2), index)
    assert np.mean([balance[y] for y in years]) == pytest.approx(-1.5, abs=1e-3)
