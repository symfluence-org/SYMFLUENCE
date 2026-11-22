"""Regression tests: HYPE spin-up length, partial forcing days, timeCOUT outlet."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.extractor import HYPEResultExtractor
from symfluence.models.hype.forcing_processor import HYPEForcingProcessor, complete_day_window
from symfluence.models.hype.preprocessor import HYPEPreProcessor

LOG = logging.getLogger("test_round8_hype")


# --- 1. spin-up -------------------------------------------------------------

def test_spinup_period_counts_both_ends_so_cdate_is_evaluation_start(tmp_path):
    days = HYPEPreProcessor._spinup_days_from_period("2000-01-01, 2000-12-31")
    assert days == 366  # leap year, inclusive of 2000-12-31
    # Sub-daily period bounds count the same calendar days.
    assert HYPEPreProcessor._spinup_days_from_period("2000-01-01 00:00, 2000-12-31 23:00") == 366

    HYPEConfigManager({}, LOG, tmp_path).write_info_filedir(
        days, str(tmp_path / "results"),
        experiment_start="2000-01-01", experiment_end="2002-12-31")
    info = (tmp_path / "info.txt").read_text()
    assert "bdate\t2000-01-01" in info
    assert "cdate\t2001-01-01" in info


def _cfg(**extra):
    base = dict(
        SYMFLUENCE_DATA_DIR="/tmp/x", SYMFLUENCE_CODE_DIR="/tmp/y", DOMAIN_NAME="d",
        EXPERIMENT_ID="e", EXPERIMENT_TIME_START="2000-01-01 00:00",
        EXPERIMENT_TIME_END="2002-12-31 23:00", DOMAIN_DEFINITION_METHOD="lumped",
        SUB_GRID_DISCRETIZATION="GRUs", HYDROLOGICAL_MODEL="HYPE", FORCING_DATASET="ERA5",
    )
    base.update(extra)
    return SymfluenceConfig(**base)


def test_spinup_period_used_unless_hype_spinup_days_set_explicitly():
    # HYPE_SPINUP_DAYS defaults to None so SPINUP_PERIOD is used unless it is set.
    implicit = SimpleNamespace(config=_cfg(SPINUP_PERIOD="2000-01-01, 2000-12-31"))
    explicit = SimpleNamespace(config=_cfg(SPINUP_PERIOD="2000-01-01, 2000-12-31", HYPE_SPINUP_DAYS=10))
    assert HYPEPreProcessor._hype_spinup_days_explicit(implicit) is False
    assert HYPEPreProcessor._hype_spinup_days_explicit(explicit) is True


# --- 2. partial first/last forcing days ---------------------------------------

def test_complete_day_window_drops_partial_edges_only():
    # ERA5-style period-ending stamps: 01:00 on day 1 .. 00:00 on day 4.
    times = pd.date_range("2000-01-01 01:00", "2000-01-04 00:00", freq="h")
    first, last, dropped, interior = complete_day_window(times)
    assert first == pd.Timestamp("2000-01-02")
    assert last == pd.Timestamp("2000-01-03")
    assert dropped == [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-04")]
    assert interior == []
    # Daily input is passed through untouched.
    daily = pd.date_range("2000-01-01", periods=5, freq="D")
    assert complete_day_window(daily)[:2] == (daily[0], daily[-1])


def _write_merged(path, times, precip, temp):
    ds = xr.Dataset(
        {"pptrate": (("time", "hru"), precip.reshape(-1, 1)),
         "airtemp": (("time", "hru"), temp.reshape(-1, 1))},
        coords={"time": times, "hru": [0]},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_partial_first_and_last_days_are_dropped_from_pobs_tobs(tmp_path):
    out = tmp_path / "settings"
    out.mkdir()
    proc = HYPEForcingProcessor(config={}, logger=LOG, forcing_input_dir=tmp_path,
                                output_path=out, cache_path=tmp_path)
    # Starts 06:00 on Jan 1, ends 11:00 on Jan 4: Jan 1 and Jan 4 are partial.
    times = pd.date_range("2000-01-01 06:00", "2000-01-04 11:00", freq="h")
    precip = np.full(len(times), 1.0 / 3600.0)  # 1 mm/h
    temp = np.full(len(times), 273.15 + 5.0)
    temp[0] = 273.15 - 20.0  # a cold partial-day hour that would bias TMIN
    nc = tmp_path / "merged.nc"
    _write_merged(nc, times, precip, temp)

    proc._convert_hourly_to_daily(nc, "pptrate", "Pobs", stat="sum",
                                  output_file_name_txt=out / "Pobs.txt", unit_conversion="mm/s")
    proc._convert_hourly_to_daily(nc, "airtemp", "TMINobs", stat="min",
                                  output_file_name_txt=out / "TMINobs.txt", unit_conversion="K")

    pobs = pd.read_csv(out / "Pobs.txt", sep="\t")
    tmin = pd.read_csv(out / "TMINobs.txt", sep="\t")
    assert pobs["time"].tolist() == ["2000-01-02", "2000-01-03"]
    np.testing.assert_allclose(pobs["1"], 24.0)  # full-day sums only
    assert tmin["time"].tolist() == ["2000-01-02", "2000-01-03"]
    np.testing.assert_allclose(tmin["1"], 5.0)

    # info.txt run dates stay within the written forcing.
    HYPEConfigManager({}, LOG, out).write_info_filedir(
        0, str(tmp_path / "results"),
        experiment_start="2000-01-01 06:00", experiment_end="2000-01-04 11:00")
    info = (out / "info.txt").read_text()
    assert "bdate\t2000-01-02" in info
    assert "edate\t2000-01-03" in info


# --- 3. timeCOUT outlet column ------------------------------------------------

TIMECOUT = (
    "!! computed outflow (COUT) [m3/s] - generated by HYPE 5.x\n"
    "DATE\t1\t2\t3\n"
    "2001-01-01\t1.000\t2.000\t3.500\n"
    "2001-01-02\t1.500\t2.500\t4.500\n"
    "2001-01-03\t1.200\t2.100\t3.900\n"
)


@pytest.fixture
def project(tmp_path):
    sim = tmp_path / "simulations" / "exp" / "HYPE"
    sim.mkdir(parents=True)
    (sim / "timeCOUT.txt").write_text(TIMECOUT)
    return tmp_path, sim / "timeCOUT.txt"


def test_timecout_returns_outlet_not_sum(project):
    _, cout = project
    series = HYPEResultExtractor("HYPE").extract_variable(cout, "streamflow")
    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index[0] == pd.Timestamp("2001-01-01")
    # Highest-discharge (outlet) column, not 1 + 2 + 3.5.
    np.testing.assert_allclose(series.to_numpy(), [3.5, 4.5, 3.9])


def test_timecout_uses_sim_reach_id_through_subid_map(project):
    root, cout = project
    settings = root / "settings" / "HYPE"
    settings.mkdir(parents=True)
    pd.DataFrame({"original_id": [0, 1, 2], "hype_subid": [1, 2, 3]}).to_csv(
        settings / "subid_map.csv", index=False)
    series = HYPEResultExtractor("HYPE").extract_variable(cout, "streamflow", sim_reach_id=1)
    np.testing.assert_allclose(series.to_numpy(), [2.0, 2.5, 2.1])  # subid 2
    # Without the map the raw id is used.
    (settings / "subid_map.csv").unlink()
    series = HYPEResultExtractor("HYPE").extract_variable(cout, "streamflow", sim_reach_id=1)
    np.testing.assert_allclose(series.to_numpy(), [1.0, 1.5, 1.2])
