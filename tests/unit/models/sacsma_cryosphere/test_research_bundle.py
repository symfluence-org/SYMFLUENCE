"""Research bundle contracts; scientific physics are checked by the real replay audit."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from symfluence.models.sacsma_cryosphere.replay import ReplayValidationError
from symfluence.models.sacsma_cryosphere.research_bundle import (
    MODEL_ID,
    REQUIRED_FILES,
    load_research_candidate,
    load_selected_research_candidate,
    manifest_fingerprint,
)


def bundle(tmp_path, frozen=False, days=3):
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (tmp_path / "config.json").write_text(json.dumps({"frozen_soil": frozen}))
    (tmp_path / "selected.json").write_text(json.dumps({"frozen_soil": frozen}))
    dates = pd.date_range("2012-10-01", periods=days)
    obs = pd.Series([float(i + 1) for i in range(days)], index=dates)
    (obs * 1.1).to_csv(tmp_path / "daily.csv")
    obs.to_csv(tmp_path / "observations.csv")
    manifest = {"model_id": MODEL_ID, "artifacts_sha256": {
        n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in REQUIRED_FILES
    }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = {"manifest_fingerprint": manifest_fingerprint(manifest), "model_id": MODEL_ID,
              "discharge_replay_match": True, "all_finite": True,
              "minimum_terminal_glacier_ice_mm": 0., "max_cumulative_water_mm": 1e-10,
              "max_layer_HRU_substep_heat_J_m2": 0.}
    (tmp_path / "replay_verification.json").write_text(json.dumps(report))
    return obs


def test_explicit_historical_contract_and_single_family_vote(tmp_path):
    obs = bundle(tmp_path)
    q, metadata = load_research_candidate(tmp_path, obs, "2012-10-01", "2012-10-03")
    assert q.tolist() == pytest.approx([1.1, 2.2, 3.3])
    assert metadata["family_votes"] == 1
    assert metadata["scientific_promotion_certified"] is False


def test_source_tampering_rejected(tmp_path):
    obs = bundle(tmp_path)
    (tmp_path / "soil_kernel.py").write_text("changed")
    with pytest.raises(ReplayValidationError, match="fingerprint differs"):
        load_research_candidate(tmp_path, obs)


def test_stale_replay_rejected(tmp_path):
    obs = bundle(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["new_selection"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReplayValidationError, match="stale"):
        load_research_candidate(tmp_path, obs)


def test_frozen_research_stays_gated(tmp_path):
    obs = bundle(tmp_path, frozen=True)
    with pytest.raises(ReplayValidationError, match="explicitly unfrozen"):
        load_research_candidate(tmp_path, obs)


def test_observation_mismatch_rejected(tmp_path):
    obs = bundle(tmp_path)
    obs.iloc[1] += .01
    with pytest.raises(ReplayValidationError, match="observations"):
        load_research_candidate(tmp_path, obs, "2012-10-01", "2012-10-03")


@pytest.mark.parametrize("key,value", [
    ("minimum_terminal_glacier_ice_mm", float("nan")),
    ("max_cumulative_water_mm", 1.),
    ("all_finite", False),
])
def test_failed_numerical_evidence_rejected(tmp_path, key, value):
    obs = bundle(tmp_path)
    path = tmp_path / "replay_verification.json"
    report = json.loads(path.read_text())
    report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ReplayValidationError):
        load_research_candidate(tmp_path, obs)


@pytest.mark.parametrize("field,value", [("adapter", "native_v2"), ("model", "HBV"),
                                         ("manifest_fingerprint", "wrong")])
def test_selection_must_pin_adapter_family_and_manifest(tmp_path, field, value):
    obs = bundle(tmp_path, days=4017)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    selection = {"bundle": str(tmp_path), "adapter": "research_bundle_unfrozen_v4",
                 "model": "SACSMA", "manifest_fingerprint": manifest_fingerprint(manifest)}
    q, metadata = load_selected_research_candidate(selection, obs)
    assert len(q) == 4017 and metadata["family_votes"] == 1
    selection[field] = value
    with pytest.raises(ReplayValidationError):
        load_selected_research_candidate(selection, obs)
