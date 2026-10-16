"""Portable input/spec/output contract and optional corrected-engine integration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symfluence.models.sacsma_cryosphere.replay import (
    MODULES,
    ReplaySpec,
    load_audited_candidate,
    replay_bundle,
    runtime_fingerprints,
    sha,
)


def document(tmp_path):
    times = pd.date_range("2012-09-30", periods=5)
    frame = pd.DataFrame(
        {"datetime": times, "precip_1": [10, 0, 0, 1, 0], "temp_1": [-5, 10, 10, 5, 5], "pet_1": [0] * 5}
    )
    frame.to_csv(tmp_path / "forcing.csv", index=False)
    return {
        "forcing": {"path": "forcing.csv", "sha256": sha(tmp_path / "forcing.csv")},
        "hru_ids": ["1"],
        "geometry_reference": "Synthetic uniform glacier tile",
        "config": {
            "area_km2": 1.0,
            "weights": [1.0],
            "glacier": [True],
            "latitudes": [45.0],
            "elevations_m": [0.0],
            "initial_ice_mm": 10.0,
        },
        "parameters": {"DAYGM": 0.0, "NMF": 0.0, "PLWHC": 0.0, "MFMAX": 1.0, "MFMIN": 1.0},
        "implementation_hashes": {name: "0" * 64 for name in MODULES},
        "package_versions": {"jsacsma": "0.3.0", "jsnow17": "0.3.0"},
        "periods": {"Cal": {"start": "2012-10-01", "end": "2012-10-04", "role": "calibration"}},
        "qualifications": ["Synthetic test; no field calibration or ensemble promotion."],
    }


def require_engine():
    pytest.importorskip("jsacsma.cryosphere", reason="Requires jSACSMA/jSnow17 correction PRs (0.3.0)")
    if not hasattr(__import__("jsnow17"), "__version__"):
        pytest.skip("Requires corrected jSnow17")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(ice_initial_m=100.0),
        lambda d: d["config"].update(initial_ice_mm=-1.0),
        lambda d: d["config"].update(weights=[0.9]),
        lambda d: d["forcing"].update(path="/absolute/forcing.csv"),
        lambda d: d["forcing"].update(path="../forcing.csv"),
        lambda d: d["periods"]["Cal"].update(role="untouched_holdout"),
        lambda d: d["config"].update(glacier=["yes"]),
    ],
)
def test_spec_rejects_unknown_or_invalid_options(tmp_path, mutation):
    doc = document(tmp_path)
    mutation(doc)
    with pytest.raises(ValueError):
        ReplaySpec.model_validate(doc)


def prepare(tmp_path):
    require_engine()
    doc = document(tmp_path)
    hashes, versions = runtime_fingerprints()
    doc.update(implementation_hashes=hashes, package_versions=versions)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(doc))
    return doc, path


def test_input_and_implementation_hash_checks(tmp_path):
    doc, path = prepare(tmp_path)
    doc["forcing"]["sha256"] = "1" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="checksum"):
        replay_bundle(path, tmp_path / "out")
    doc["forcing"]["sha256"] = sha(tmp_path / "forcing.csv")
    doc["implementation_hashes"][MODULES[0]] = "1" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="implementation"):
        replay_bundle(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_replay_honors_ice_and_bundle_survives_move(tmp_path):
    doc, path = prepare(tmp_path)
    first = replay_bundle(path, tmp_path / "out")
    assert first["balance"]["max_cumulative_residual_mm"] < 1e-8
    doc["config"]["initial_ice_mm"] = 1000.0
    path.write_text(json.dumps(doc))
    replay_bundle(path, tmp_path / "more_ice")
    low, _ = load_audited_candidate(tmp_path / "out")
    high, _ = load_audited_candidate(tmp_path / "more_ice")
    assert high.sum() > low.sum()
    moved = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "out", moved)
    series, meta = load_audited_candidate(moved)
    np.testing.assert_allclose(series, low)
    replay_bundle(moved / "replay_spec.json", tmp_path / "rerun")
    again, _ = load_audited_candidate(tmp_path / "rerun")
    np.testing.assert_allclose(again, low)
    with pytest.raises(FileExistsError):
        replay_bundle(path, tmp_path / "out")
    (moved / "streamflow.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_audited_candidate(moved)


def test_time_axis_must_match_declared_timestep(tmp_path):
    doc, path = prepare(tmp_path)
    frame = pd.read_csv(tmp_path / "forcing.csv")
    frame.loc[2, "datetime"] = "2012-10-01"
    frame.to_csv(tmp_path / "forcing.csv", index=False)
    doc["forcing"]["sha256"] = sha(tmp_path / "forcing.csv")
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Time axis"):
        replay_bundle(path, tmp_path / "out")


def test_glacier_area_convention_and_roles_are_explicit(tmp_path):
    doc, path = prepare(tmp_path)
    pd.DataFrame({"start_label": ["2012-09-30"], "end_label": ["2012-10-04"], "observed_mb_mwe": [-0.01]}).to_csv(
        tmp_path / "glacier.csv", index=False
    )
    doc["glacier_constraint"] = {"targets": {"path": "glacier.csv", "sha256": sha(tmp_path / "glacier.csv")}}
    path.write_text(json.dumps(doc))
    replay_bundle(path, tmp_path / "out")
    metrics = json.loads((tmp_path / "out/metrics.json").read_text())["Cal"]
    assert metrics["role"] == "calibration" and metrics["glacier_area_basis"] == "model_fixed_area_specific"
    assert metrics["glacier_balance_samples"] == 1
    doc["glacier_constraint"]["area_basis"] = "observed_dynamic_area"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        ReplaySpec.model_validate_json(path.read_text())


def test_ensemble_adapter_rejects_changed_observations_and_periods(tmp_path):
    from symfluence.models.sacsma_cryosphere.replay import audit_candidate

    doc, path = prepare(tmp_path)
    times = pd.date_range("2012-09-30", periods=5)
    observations = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=times)
    pd.DataFrame({"datetime": times, "discharge_cms": observations.values}).to_csv(tmp_path / "obs.csv", index=False)
    doc["observations"] = {"path": "obs.csv", "sha256": sha(tmp_path / "obs.csv")}
    path.write_text(json.dumps(doc))
    replay_bundle(path, tmp_path / "out")
    periods = {"Cal": ("2012-10-01", "2012-10-04", "calibration")}
    flow, meta = audit_candidate(tmp_path / "out", observations, periods)
    assert meta["family"] == "SACSMA" and len(flow) == 5
    with pytest.raises(ValueError, match="observations differ"):
        audit_candidate(tmp_path / "out", observations * 1.01, periods)
    with pytest.raises(ValueError, match="periods differ"):
        audit_candidate(tmp_path / "out", observations, {})


def test_native_replay_scores_complete_daily_means(tmp_path):
    from symfluence.models.sacsma_cryosphere.replay import _metrics

    doc, path = prepare(tmp_path)
    times = pd.date_range("2012-10-01", periods=24, freq="3h")
    pd.DataFrame(
        {"datetime": times, "precip_1": [0.0] * 24, "temp_1": np.repeat([1.0, 5.0, 10.0], 8), "pet_1": [0.0] * 24}
    ).to_csv(tmp_path / "forcing.csv", index=False)
    daily_obs = np.array([1.0, 2.0, 4.0])
    pd.DataFrame({"datetime": pd.date_range("2012-10-01", periods=3), "discharge_cms": daily_obs}).to_csv(
        tmp_path / "obs.csv", index=False
    )
    doc["forcing"]["sha256"] = sha(tmp_path / "forcing.csv")
    doc["observations"] = {"path": "obs.csv", "sha256": sha(tmp_path / "obs.csv")}
    doc["config"].update(timestep_days=0.125, initial_ice_mm=1000.0)
    pd.DataFrame(
        {"start_label": ["2012-10-01 21:00:00"], "end_label": ["2012-10-03 21:00:00"], "observed_mb_mwe": [-0.01]}
    ).to_csv(tmp_path / "glacier.csv", index=False)
    doc["glacier_constraint"] = {"targets": {"path": "glacier.csv", "sha256": sha(tmp_path / "glacier.csv")}}
    doc["observation_aggregation"] = "daily_mean"
    doc["periods"] = {"Cal": {"start": "2012-10-01", "end": "2012-10-03", "role": "calibration"}}
    path.write_text(json.dumps(doc))
    replay_bundle(path, tmp_path / "out")
    q = pd.read_csv(tmp_path / "out/streamflow.csv").streamflow_cms.to_numpy()
    actual = json.loads((tmp_path / "out/metrics.json").read_text())["Cal"]
    assert actual["samples"] == 3
    assert actual["glacier_balance_samples"] == 1
    expected = _metrics(q.reshape(3, 8).mean(axis=1), daily_obs)
    assert actual["KGE"] == pytest.approx(expected["KGE"])
    assert actual["observation_aggregation"] == "daily_mean"


def test_v1_identity_cannot_be_replayed_with_v2_engine(tmp_path):
    doc, path = prepare(tmp_path)
    doc["model"] = "SACSMA_CRYOSPHERE_UNIFORM_HRU_V1"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="identity"):
        replay_bundle(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
