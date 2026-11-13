"""Regression tests: calibration worker and postprocessor share one extraction rule.

Each test pins a case where the results CSV (final evaluation) used to score
different numbers from calibration. Synthetic fixtures only; no model binaries.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ConfigurationError, EvaluationError
from symfluence.core.mixins.config import SCHEMA_DEFAULT, fallback_default

pytestmark = [pytest.mark.unit, pytest.mark.quick]
LOG = logging.getLogger("test")


def _cfg_value(accessor=None, default=SCHEMA_DEFAULT, dict_key=None):
    try:
        value = accessor()
    except (AttributeError, KeyError, TypeError):
        value = None
    return fallback_default(default, dict_key) if value is None else value


# 1. ParFlow / CLM-ParFlow dump interval -------------------------------------

@pytest.mark.parametrize("model", ["parflow", "clmparflow"])
def test_parflow_postprocessor_passes_dump_interval(tmp_path, model):
    from symfluence.models.clmparflow.postprocessor import CLMParFlowPostProcessor
    from symfluence.models.parflow.extractor import resolve_dump_interval_hours
    from symfluence.models.parflow.postprocessor import ParFlowPostProcessor

    assert resolve_dump_interval_hours(None, model) == 24.0
    cls, ext = (
        (ParFlowPostProcessor, "symfluence.models.parflow.extractor.ParFlowResultExtractor")
        if model == "parflow" else
        (CLMParFlowPostProcessor, "symfluence.models.clmparflow.extractor.CLMParFlowResultExtractor")
    )
    cfg = SimpleNamespace(model=SimpleNamespace(**{model: SimpleNamespace()}),
                          domain=SimpleNamespace(experiment_id="e"))
    fake = SimpleNamespace(config=cfg, _get_config_value=_cfg_value, logger=LOG,
                           save_streamflow_to_results=MagicMock())
    seen = []

    def _record(self, output_dir, variable, **kwargs):
        seen.append(kwargs)
        return pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1))

    with patch(f"{ext}.extract_variable", _record), contextlib.suppress(Exception):
        cls.extract_streamflow(fake, output_dir=tmp_path)
    assert seen and all(k["dump_interval_hours"] == 24.0 for k in seen)


# 2. VIC: area-weighted basin mean, not a sum of depths ----------------------

def test_vic_basin_mean_is_area_weighted():
    from symfluence.models.vic.extractor import basin_mean_vic_runoff

    times = pd.date_range("2020-01-01", periods=2)
    ds = xr.Dataset({
        "OUT_RUNOFF": (("time", "lat", "lon"), np.array([[[1.0, 3.0]], [[1.0, 3.0]]])),
        "OUT_BASEFLOW": (("time", "lat", "lon"), np.zeros((2, 1, 2))),
    }, coords={"time": times, "lat": [0.0], "lon": [0.0, 1.0]})
    assert basin_mean_vic_runoff(ds).values.tolist() == [2.0, 2.0]          # was 4 (sum)
    weighted = basin_mean_vic_runoff(ds, cell_area=np.array([[3.0, 1.0]]))
    assert weighted.values.tolist() == [1.5, 1.5]


# 4. Wflow distributed: end-of-day labels shifted like the worker -------------

def test_wflow_postprocessor_shifts_distributed_labels():
    from symfluence.models.wflow.postprocessor import WflowPostProcessor

    q = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-02", periods=2))
    fake = SimpleNamespace(config={"WFLOW_SPATIAL_MODE": "distributed"})
    shifted = WflowPostProcessor._align_time_labels(fake, q)
    assert shifted.index[0] == pd.Timestamp("2020-01-01")
    lumped = WflowPostProcessor._align_time_labels(SimpleNamespace(config={}), q)
    assert lumped.index[0] == pd.Timestamp("2020-01-02")


# 5. One outlet-reach rule ---------------------------------------------------

def _routed(tmp_path, means=(1.0, 9.0, 5.0), ids=(10, 20, 30), down=(20, 0, 20)):
    times = pd.date_range("2020-01-01", periods=3)
    ds = xr.Dataset(
        {"IRFroutedRunoff": (("time", "seg"), np.tile(np.array(means), (3, 1))),
         "reachID": ("seg", np.array(ids))},
        coords={"time": times},
    )
    topo = tmp_path / "settings" / "mizuRoute" / "topology.nc"
    topo.parent.mkdir(parents=True)
    xr.Dataset({"segId": ("seg", np.array(ids)), "downSegId": ("seg", np.array(down))}).to_netcdf(topo)
    return ds, topo


def test_reach_selection_rule(tmp_path):
    from symfluence.core.modeling.utilities.reach_selection import select_outlet_index

    # Topology outlet (seg 30 drains nowhere) beats both highest-mean and last-seg
    ds, topo = _routed(tmp_path, means=(1.0, 9.0, 5.0), ids=(10, 20, 30), down=(30, 30, 0))
    var = ds["IRFroutedRunoff"]
    assert select_outlet_index(ds, var, None, [topo]) == 2
    # SIM_REACH_ID wins (string config values accepted)
    assert select_outlet_index(ds, var, "10", [topo]) == 0
    with pytest.raises(EvaluationError):
        select_outlet_index(ds, var, 99, [topo])
    # No topology: highest mean flow
    assert select_outlet_index(ds, var, None, []) == 1


def test_standard_postprocessor_uses_shared_reach_rule(tmp_path):
    from symfluence.core.modeling.base.standard_postprocessor import RoutedModelPostProcessor

    ds, topo = _routed(tmp_path, means=(1.0, 9.0, 5.0), ids=(10, 20, 30), down=(20, 0, 20))
    f = tmp_path / "routed.nc"
    ds.to_netcdf(f)
    fake = SimpleNamespace(
        config={"DOMAIN_NAME": "d", "SYMFLUENCE_DATA_DIR": str(tmp_path / "nowhere")},
        routing_variable="IRFroutedRunoff", logger=LOG,
    )
    # near=file finds tmp_path/settings/mizuRoute/topology.nc -> outlet is reach 20,
    # not the last segment (reach 30, the old seg=-1 rule)
    q = RoutedModelPostProcessor._extract_from_netcdf(fake, f)
    assert q.iloc[0] == 9.0


# 6. Noah-MP writes discharge (m3/s) like the worker -------------------------

def test_noahmp_discharge_matches_worker_conversion(tmp_path):
    from symfluence.models.noahmp.extractor import noahmp_daily_discharge_cms

    times = pd.date_range("2020-01-01", periods=48, freq="h")
    f = tmp_path / "output.nc"
    xr.Dataset({"SFCRNOFF": ("time", np.full(48, 0.5)),
                "UGDRNOFF": ("time", np.full(48, 0.5))},
               coords={"time": times}).to_netcdf(f)
    q = noahmp_daily_discharge_cms(f, area_m2=3.6e6)
    # 1 mm/h over 3.6 km2 = 1 m3/s
    assert q.tolist() == pytest.approx([1.0, 1.0])
    assert q.name == "NOAHMP_discharge_cms"


# 7. LISFLOOD .tss dated with the model DtSec --------------------------------

def test_lisflood_tss_dates_use_model_timestep(tmp_path):
    from symfluence.models.lisflood.calibration.worker import LisfloodWorker

    tss = tmp_path / "dis.tss"
    tss.write_text("timeseries scalar\n2\ntimestep\n1\n 1 0.5\n 2 1.5\n 3 2.5\n")
    q = LisfloodWorker._read_tss_file(tss, {"EXPERIMENT_TIME_START": "2005-01-01 06:00"})
    assert list(q.index) == list(pd.date_range("2005-01-01", periods=3, freq="D"))
    q = LisfloodWorker._read_tss_file(
        tss, {"EXPERIMENT_TIME_START": "2005-01-01 00:00", "LISFLOOD_FORCING_TIMESTEP": 3600})
    assert q.index[-1] == pd.Timestamp("2005-01-01 02:00")


# 8. No invented catchment areas ---------------------------------------------

def test_catchment_area_required(tmp_path):
    from symfluence.core.metrics.streamflow_metrics import require_catchment_area_km2

    assert require_catchment_area_km2({}, tmp_path, "d", configured_km2=12.5) == 12.5
    with pytest.raises(ValueError, match="CATCHMENT_AREA"):
        require_catchment_area_km2({}, tmp_path, "d")


# 9. SUMMA multi-GRU aggregation and experiment-id filter --------------------

def test_summa_gru_runoff_area_weighted_by_hru2gru(tmp_path):
    from symfluence.core.modeling.utilities.area_weighting import area_weighted_discharge
    from symfluence.models.summa.postprocessor import SUMMAPostProcessor

    attrs = xr.Dataset({"HRUarea": ("hru", [1.0, 2.0, 4.0]),
                        "hru2gruId": ("hru", [1, 1, 2]), "gruId": ("gru", [1, 2])})
    var = xr.DataArray(np.array([[1.0, 10.0]]), dims=("time", "gru"))
    assert area_weighted_discharge(var, attrs).values.tolist() == [1.0 * 3 + 10.0 * 4]

    exp = "run_daymet"
    out = tmp_path / "simulations" / exp / "SUMMA"
    out.mkdir(parents=True)
    (tmp_path / "settings" / "SUMMA").mkdir(parents=True)
    attrs.to_netcdf(tmp_path / "settings" / "SUMMA" / "attributes.nc")
    times = pd.date_range("2020-01-01", periods=24, freq="h")
    xr.Dataset({"averageRoutedRunoff": (("time", "gru"), np.tile([1e-3, 2e-3], (24, 1)))},
               coords={"time": times}).to_netcdf(out / f"{exp}_timestep.nc")
    fake = SimpleNamespace(project_dir=tmp_path, experiment_id=exp, logger=MagicMock(),
                           resample_frequency="D", save_streamflow_to_results=MagicMock())
    SUMMAPostProcessor._extract_native_summa_streamflow(fake)
    series = fake.save_streamflow_to_results.call_args.args[0]
    assert isinstance(series, pd.Series)
    assert series.iloc[0] == pytest.approx(1e-3 * 3 + 2e-3 * 4)


# 10. CRHM end-of-interval labels --------------------------------------------

def test_crhm_basinflow_shifted_to_interval_start():
    from symfluence.models.crhm.extractor import basinflow_to_cms

    vol = pd.Series([3600.0, 7200.0], index=pd.date_range("2020-01-01 01:00", periods=2, freq="h"))
    q = basinflow_to_cms(vol)
    assert q.tolist() == [1.0, 2.0]
    assert q.index[0] == pd.Timestamp("2020-01-01 00:00")


# 11. PRMS statvar: element counts and units ---------------------------------

def test_prms_statvar_respects_counts_and_units(tmp_path):
    from symfluence.models.prms.calibration.worker import PRMSWorker
    from symfluence.models.prms.statvar import CFS_TO_CMS, read_statvar_streamflow

    sv = tmp_path / "statvar.dat"
    sv.write_text("3\nseg_outflow 3\nbasin_actet 1\nbasin_cfs 1\n"
                  "1 2000 1 1 0 0 0 1 2 3 0.1 100\n"
                  "2 2000 1 2 0 0 0 1 2 3 0.1 200\n")
    q = read_statvar_streamflow(sv)
    assert q.tolist() == pytest.approx([100 * CFS_TO_CMS, 200 * CFS_TO_CMS])
    assert PRMSWorker().read_simulated_streamflow(tmp_path, {}).tolist() == q.tolist()
    assert read_statvar_streamflow(sv, preferred=("seg_outflow",), element=2).iloc[0] == \
        pytest.approx(2 * CFS_TO_CMS)


# 12. HYPE removes the previous trial's output --------------------------------

def test_hype_clears_stale_output(tmp_path):
    from symfluence.models.hype.calibration.worker import HYPEWorker

    out, settings = tmp_path / "out", tmp_path / "settings"
    out.mkdir()
    settings.mkdir()
    (out / "timeCOUT.txt").write_text("stale")
    (settings / "info.txt").write_text("keep")
    HYPEWorker._clear_stale_outputs(SimpleNamespace(logger=LOG, _OUTPUT_PATTERNS=HYPEWorker._OUTPUT_PATTERNS),
                                    out, settings)
    assert not (out / "timeCOUT.txt").exists()
    assert (settings / "info.txt").exists()


# 13. MESH stale gauge output, exit code, strict start date -------------------

def test_mesh_run_options_and_stale_gauge(tmp_path):
    from symfluence.models.mesh.runner import MESHRunner, parse_run_options_dates

    ro = tmp_path / "MESH_input_run_options.ini"
    ro.write_text("header\n1 2 3\n2004 032 0 0\n2005 001 0 0\n")
    start, end = parse_run_options_dates(ro)
    assert start == pd.Timestamp("2004-02-01") and end == pd.Timestamp("2005-01-01")
    ro.write_text("garbage\n")
    with pytest.raises(ConfigurationError):
        parse_run_options_dates(ro)

    (tmp_path / "MESH_streamflow_Gauge_1.txt").write_text("old")
    MESHRunner._clear_stale_outputs(SimpleNamespace(), tmp_path)
    assert not (tmp_path / "MESH_streamflow_Gauge_1.txt").exists()

    log = tmp_path / "run.log"
    log.write_text("...\nAbnormal exit.\n")
    assert not MESHRunner._log_reports_normal_termination(log)
    log.write_text("...\nProgram has terminated normally.\n")
    assert MESHRunner._log_reports_normal_termination(log)


# 14. SUMMA point mode fails when any site fails ------------------------------

def test_summa_point_mode_reports_site_failure(tmp_path):
    from symfluence.models.summa.runner import SummaRunner

    settings = tmp_path / "settings"
    settings.mkdir()
    (settings / "list_fileManager_IC.txt").write_text("fm_a_IC.txt\nfm_b_IC.txt\n")
    (settings / "list_fileManager.txt").write_text("fm_a.txt\nfm_b.txt\n")
    results = iter([True, True, False, True])  # site b IC fails
    runner = SimpleNamespace(
        settings_path=settings, logger=LOG, domain_name="d", model_exe=Path("summa.exe"),
        verify_required_files=lambda *a, **k: None,
        get_simulation_dir=lambda *_: tmp_path / "sim",
        execute_subprocess=lambda **k: SimpleNamespace(success=next(results), error_message=None),
    )
    assert SummaRunner.run_summa_point(runner) is None
