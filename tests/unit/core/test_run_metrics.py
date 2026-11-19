# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""results/<exp>/metrics.json: schema, missing-obs warnings, calibration stage, figure."""
from __future__ import annotations

import json
import logging
import math
import os
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symfluence.core.metrics import run_metrics as rm

pytestmark = pytest.mark.unit

DOMAIN = "testbasin"
EXP = "run_1"


def _config(**domain_extra):
    domain = dict(
        name=DOMAIN, experiment_id=EXP,
        calibration_period="2001-01-01, 2002-12-31",
        evaluation_period="2003-01-01, 2003-12-31",
        spinup_period="2000-01-01, 2000-12-31",
    )
    domain.update(domain_extra)
    return types.SimpleNamespace(
        domain=types.SimpleNamespace(**domain),
        paths=types.SimpleNamespace(observations_path="default"),
        evaluation=types.SimpleNamespace(
            spinup_years=0, summary_report=True,
            streamflow=types.SimpleNamespace(station_id="05BB001"),
        ),
        optimization=types.SimpleNamespace(target="streamflow"),
    )


def _flows(index, seed=0):
    rng = np.random.default_rng(seed)
    doy = index.dayofyear.values
    return 10 + 8 * np.sin(2 * np.pi * doy / 365.0) ** 2 + rng.gamma(1.0, 1.0, len(index))


def _write_project(tmp_path: Path, *, with_obs: bool = True, sim_offset: float = 0.0) -> Path:
    project = tmp_path / f"domain_{DOMAIN}"
    idx = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    q = _flows(idx)
    results = project / "results"
    results.mkdir(parents=True)
    pd.DataFrame({"SUMMA_discharge_cms": q * 1.1 + sim_offset}, index=idx).rename_axis(
        "datetime").to_csv(results / f"{EXP}_results.csv")
    if with_obs:
        obs_dir = project / "data" / "observations" / "streamflow" / "preprocessed"
        obs_dir.mkdir(parents=True)
        pd.DataFrame({"datetime": idx, "discharge_cms": q}).to_csv(
            obs_dir / f"{DOMAIN}_streamflow_processed.csv", index=False)
    return project


def test_postprocess_metrics_schema_and_values(tmp_path):
    project = _write_project(tmp_path)
    doc = rm.build_postprocess_metrics(_config(), project, project / "results" / f"{EXP}_results.csv")

    assert doc["schema_version"] == rm.SCHEMA_VERSION
    assert doc["stage"] == "postprocess"
    assert doc["experiment_id"] == EXP and doc["domain"] == DOMAIN
    assert doc["target"]["observations_found"] is True
    assert doc["target"]["station_id"] == "05BB001"
    assert doc["periods"]["calibration"][0].startswith("2001-01-01")
    assert doc["periods"]["test"] is None
    assert doc["resolution"] == "daily"
    assert doc["warnings"] == []

    metrics = doc["models"]["SUMMA"]["metrics"]
    assert set(metrics) == {"calibration", "evaluation", "full"}
    cal = metrics["calibration"]
    assert set(rm.METRIC_KEYS) <= set(cal)
    assert cal["n_obs"] == 730
    # sim = 1.1 * obs: perfectly correlated, 10 % high
    assert cal["r"] == pytest.approx(1.0)
    assert cal["beta"] == pytest.approx(1.1)
    assert cal["alpha"] == pytest.approx(1.1)
    assert cal["PBIAS"] == pytest.approx(10.0)
    assert cal["KGE"] == pytest.approx(1 - math.sqrt(0.01 + 0.01))
    # spin-up (2000) is excluded from the full record
    assert metrics["full"]["start"].startswith("2001-01-01")


def test_test_period_is_scored_when_configured(tmp_path):
    project = _write_project(tmp_path)
    cfg = _config(test_period="2004-01-01, 2004-12-31")
    doc = rm.build_postprocess_metrics(cfg, project, project / "results" / f"{EXP}_results.csv")
    assert doc["periods"]["test"][0].startswith("2004-01-01")
    assert doc["models"]["SUMMA"]["metrics"]["test"]["n_obs"] == 366


def test_benchmarks_feature_detected(tmp_path, monkeypatch):
    project = _write_project(tmp_path)
    csv = project / "results" / f"{EXP}_results.csv"

    monkeypatch.setattr(rm, "_benchmark_module", lambda: None)
    assert rm.build_postprocess_metrics(_config(), project, csv)["models"]["SUMMA"]["benchmarks"] == {}

    def benchmark_series(obs, calibration_period):
        return {"mean_flow": pd.Series(float(obs.mean()), index=obs.index),
                "climatology": obs.shift(1).bfill()}

    fake = types.SimpleNamespace(
        benchmark_series=benchmark_series,
        constant_benchmark_kge=lambda beta: 1 - math.sqrt(2 + (beta - 1) ** 2),
        skill_score=lambda m, b: (m - b) / (1 - b),
    )
    monkeypatch.setattr(rm, "_benchmark_module", lambda: fake)
    bms = rm.build_postprocess_metrics(_config(), project, csv)["models"]["SUMMA"]["benchmarks"]
    assert set(bms) == {"calibration", "evaluation"}
    assert set(bms["calibration"]) == {"mean_flow", "climatology"}
    assert bms["calibration"]["mean_flow"]["KGE"] < 0
    assert bms["calibration"]["mean_flow"]["KGE_skill"] > 0


def test_missing_observations_warning_names_expected_path(tmp_path):
    project = _write_project(tmp_path, with_obs=False)
    doc = rm.build_postprocess_metrics(_config(), project, project / "results" / f"{EXP}_results.csv")
    expected = project / "data" / "observations" / "streamflow" / "preprocessed" / \
        f"{DOMAIN}_streamflow_processed.csv"
    assert doc["target"]["observations_found"] is False
    assert len(doc["warnings"]) == 1
    assert str(expected) in doc["warnings"][0]
    assert doc["models"]["SUMMA"]["metrics"] == {}


def test_non_overlapping_observations_are_warned(tmp_path):
    project = _write_project(tmp_path)
    obs = project / "data" / "observations" / "streamflow" / "preprocessed" / \
        f"{DOMAIN}_streamflow_processed.csv"
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    pd.DataFrame({"datetime": idx, "discharge_cms": 1.0}).to_csv(obs, index=False)
    doc = rm.build_postprocess_metrics(_config(), project, project / "results" / f"{EXP}_results.csv")
    assert any("do not overlap" in w for w in doc["warnings"])


def test_write_metrics_json_and_csv(tmp_path):
    project = _write_project(tmp_path)
    doc = rm.build_postprocess_metrics(_config(), project, project / "results" / f"{EXP}_results.csv")
    out = rm.write_metrics(doc, rm.run_results_dir(project, EXP))
    assert out == project / "results" / EXP / "metrics.json"
    loaded = json.loads(out.read_text())  # strict JSON: no NaN
    assert loaded == rm.load_metrics(out)
    table = pd.read_csv(out.parent / "metrics.csv")
    assert list(table.columns) == ["model", "period", *rm.METRIC_KEYS]
    assert set(table["period"]) == {"calibration", "evaluation", "full"}


def test_final_evaluation_mapping_with_test_and_benchmarks(tmp_path):
    payload = {
        "algorithm": "DDS",
        "calibration_metrics": {"KGE": 0.8, "NSE": 0.7, "RMSE": 1.2, "PBIAS": -3.0,
                                "r": 0.9, "alpha": 1.05, "beta": 0.97,
                                "KGE_benchmark_mean_flow": -0.41,
                                "KGE_skill_mean_flow": 0.86},
        "evaluation_metrics": {"KGE": 0.6, "correlation": 0.8},
        "test_metrics": {"KGE": 0.5},
        "period_warning": "no calibration period configured",
    }
    path = tmp_path / "optimization" / "SUMMA" / "dds_run_1" / f"{EXP}_dds_final_evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    doc = rm.build_calibration_metrics(_config(), tmp_path, [("SUMMA", path)])
    entry = doc["models"]["SUMMA"]
    assert doc["stage"] == "calibration"
    assert entry["metrics"]["calibration"]["KGE"] == 0.8
    assert entry["metrics"]["evaluation"]["r"] == 0.8  # 'correlation' fallback
    assert entry["metrics"]["test"]["KGE"] == 0.5
    assert entry["metrics"]["calibration"]["logNSE"] is None
    assert entry["benchmarks"]["calibration"]["mean_flow"] == {"KGE": -0.41, "KGE_skill": 0.86}
    assert any("no calibration period" in w for w in doc["warnings"])
    assert doc["provenance"]["final_evaluation"] == str(path)


# ---------------------------------------------------------------------------
# Integration with the managers (unbound calls on light stand-ins)
# ---------------------------------------------------------------------------

def _manager_stub(project: Path, config):
    from symfluence.project.model_manager import ModelManager

    stub = types.SimpleNamespace(
        project_dir=project, experiment_id=EXP, config=config,
        logger=logging.getLogger("test_run_metrics"),
        _get_config_value=lambda accessor, default=None, dict_key=None: accessor(),
    )
    stub._write_summary_figure = types.MethodType(ModelManager._write_summary_figure, stub)
    return stub


def test_baseline_writes_metrics_and_summary_figure(tmp_path):
    from symfluence.project.model_manager import ModelManager

    project = _write_project(tmp_path)
    path = ModelManager.log_baseline_performance(_manager_stub(project, _config()))
    assert path == project / "results" / EXP / "metrics.json"
    assert (path.parent / "metrics.csv").exists()
    png = path.parent / "summary.png"
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"


def test_summary_figure_can_be_disabled(tmp_path):
    from symfluence.project.model_manager import ModelManager

    project = _write_project(tmp_path)
    cfg = _config()
    cfg.evaluation.summary_report = False
    path = ModelManager.log_baseline_performance(_manager_stub(project, cfg))
    assert path.exists()
    assert not (path.parent / "summary.png").exists()


def test_baseline_missing_obs_logs_warning(tmp_path, caplog):
    from symfluence.project.model_manager import ModelManager

    project = _write_project(tmp_path, with_obs=False)
    with caplog.at_level(logging.WARNING, logger="test_run_metrics"):
        path = ModelManager.log_baseline_performance(_manager_stub(project, _config()))
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(f"{DOMAIN}_streamflow_processed.csv" in r.getMessage() for r in warned)
    doc = rm.load_metrics(path)
    assert doc["warnings"] and doc["target"]["observations_found"] is False


def test_calibration_stage_overwrites_metrics(tmp_path):
    from symfluence.optimization.optimization_manager import OptimizationManager

    project = _write_project(tmp_path)
    started = time.time()
    fe = project / "optimization" / "SUMMA" / "dds_run_1" / f"{EXP}_dds_final_evaluation.json"
    fe.parent.mkdir(parents=True)
    fe.write_text(json.dumps({"algorithm": "DDS", "calibration_metrics": {"KGE": 0.9}}))
    stale = project / "optimization" / "FUSE" / "dds_run_1" / f"{EXP}_dds_final_evaluation.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"calibration_metrics": {"KGE": 0.1}}))
    os.utime(stale, (started - 3600, started - 3600))

    stub = types.SimpleNamespace(project_dir=project, experiment_id=EXP, config=_config(),
                                 logger=logging.getLogger("test_run_metrics"))
    path = OptimizationManager._write_calibrated_run_metrics(stub, started)
    doc = rm.load_metrics(path)
    assert doc["stage"] == "calibration"
    assert list(doc["models"]) == ["SUMMA"]
    assert doc["models"]["SUMMA"]["metrics"]["calibration"]["KGE"] == 0.9
