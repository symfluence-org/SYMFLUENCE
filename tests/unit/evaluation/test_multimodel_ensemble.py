# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Multi-model ensemble combination: weights, no leakage, scoring, outputs, CLI."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from symfluence.core.metrics.metrics_core import kge
from symfluence.evaluation.multimodel_ensemble import (
    fit_weights,
    member_conditions,
    parse_period,
    run_ensemble,
)

CAL = "2000-01-01, 2001-12-31"
EVAL = "2002-01-01, 2002-12-31"
TEST = "2003-01-01, 2003-12-31"


@pytest.fixture
def synthetic():
    idx = pd.date_range("2000-01-01", "2003-12-31", freq="D")
    rng = np.random.default_rng(0)
    t = np.arange(len(idx))
    obs = pd.Series(10 + 5 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 0.5, len(idx)), idx)
    members = {
        "good": obs + rng.normal(0, 0.5, len(idx)),
        "biased": obs * 1.3 + rng.normal(0, 1.0, len(idx)),
        "poor": pd.Series(10 + rng.normal(0, 3, len(idx)), idx),
    }
    return members, obs


def test_parse_period():
    start, end = parse_period(CAL)
    assert start == pd.Timestamp("2000-01-01")
    assert end.date() == pd.Timestamp("2001-12-31").date() and end.hour == 23
    assert parse_period(None) is None and parse_period("None") is None
    with pytest.raises(ValueError):
        parse_period("2001-01-01, 2000-01-01")


@pytest.mark.parametrize("method", ["kge", "inverse_mse"])
def test_weights_use_calibration_period_only(synthetic, method):
    members, obs = synthetic
    sims = pd.DataFrame(members)
    cal = parse_period(CAL)
    w = fit_weights(sims, obs, cal, method)
    assert w.sum() == pytest.approx(1.0)
    assert w["good"] > w["biased"] > 0

    # Corrupt everything outside the calibration period: weights must not move.
    outside = sims.index > cal[1]
    sims2 = sims.copy()
    sims2.loc[outside, "good"] = 1e6
    obs2 = obs.copy()
    obs2[outside] = -5.0
    w2 = fit_weights(sims2, obs2, cal, method)
    pd.testing.assert_series_equal(w, w2)


def test_kge_weights_formula(synthetic):
    members, obs = synthetic
    sims = pd.DataFrame(members)
    cal = parse_period(CAL)
    mask = (sims.index >= cal[0]) & (sims.index <= cal[1])
    raw = {n: max(kge(obs[mask].to_numpy(), sims.loc[mask, n].to_numpy()), 0) for n in sims}
    expected = pd.Series(raw) / sum(raw.values())
    pd.testing.assert_series_equal(fit_weights(sims, obs, cal, "kge"), expected, check_names=False)


def test_negative_skill_member_gets_zero_kge_weight(synthetic):
    members, obs = synthetic
    members = dict(members)
    members["anti"] = 20 - obs
    w = fit_weights(pd.DataFrame(members), obs, parse_period(CAL), "kge")
    assert w["anti"] == 0.0


def test_run_ensemble_scores_and_outputs(synthetic, tmp_path):
    members, obs = synthetic
    res = run_ensemble(members=members, observed=obs,
                       periods={"calibration": CAL, "evaluation": EVAL, "test": TEST},
                       output_dir=tmp_path, median=True, votes={"good": 2, "biased": 1})
    for combo in ("equal_weight", "skill_weighted", "median", "vote_weighted"):
        assert combo in res.scores.index
    assert res.weights["vote_weighted"] == pytest.approx({"good": 2 / 3, "biased": 1 / 3, "poor": 0})
    for p in ("calibration", "evaluation", "test"):
        assert f"{p}_KGE" in res.scores.columns
    # Combination scores are the scores of the combined series.
    eq = pd.DataFrame(members).mean(axis=1)
    ev = parse_period(EVAL)
    m = (eq.index >= ev[0]) & (eq.index <= ev[1])
    assert res.scores.loc["equal_weight", "evaluation_KGE"] == pytest.approx(
        kge(obs[m].to_numpy(), eq[m].to_numpy()))
    # Skill weighting beats the poor member out of sample.
    assert res.scores.loc["skill_weighted", "evaluation_KGE"] > res.scores.loc["poor", "evaluation_KGE"]
    for key in ("scores", "conditions", "series", "json", "figure"):
        assert (tmp_path / res.files[key].split("/")[-1]).exists()
    payload = json.loads((tmp_path / "ensemble_comparison.json").read_text())
    assert payload["weights_fitted_on"] == "calibration"
    assert set(payload["conditions"]) == {"good", "biased", "poor"}
    assert payload["conditions"]["good"]["seed"] == "unknown"


def test_requires_calibration_period(synthetic):
    members, obs = synthetic
    with pytest.raises(ValueError, match="calibration period"):
        run_ensemble(members=members, observed=obs, periods={"evaluation": EVAL})


def _write_member(tmp_path, name, series, with_provenance=False):
    d = tmp_path / name
    d.mkdir()
    series.rename("q_sim").to_csv(d / "q.csv", index_label="time")
    if with_provenance:
        res = d / "results"
        res.mkdir()
        (res / "exp_dds_calibration_conditions.json").write_text(json.dumps({
            "model": name.upper(), "algorithm": "DDS", "random_seed": 99,
            "random_seed_source": "drawn", "n_params": 4, "evaluations_used": 400,
            "budget": {"mode": "per_parameter"}, "resolved_config_sha256": "abc"}))
        (res / "resolved_config.yaml").write_text(yaml.safe_dump({
            "HYDROLOGICAL_MODEL": name.upper(), "FORCING_DATASET": "CARRA",
            "DOMAIN_DEFINITION_METHOD": "lumped", "SUB_GRID_DISCRETIZATION": "GRUs",
            "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS"}))
        (res / "forcing_correction.json").write_text(json.dumps({
            "mode": "fixed", "applied": {"precip_multiplier": 1.12}}))
        (res / "metrics.json").write_text(json.dumps({"n_hrus": 1, "KGE": 0.8}))
    return d


def test_manifest_and_conditions(synthetic, tmp_path):
    members, obs = synthetic
    _write_member(tmp_path, "gr", members["good"], with_provenance=True)
    _write_member(tmp_path, "hype", members["biased"])
    obs.rename("discharge_cms").to_csv(tmp_path / "obs.csv", index_label="time")
    manifest = {
        "observations": "obs.csv",
        "periods": {"calibration": CAL, "evaluation": EVAL},
        "options": {"weighting": "inverse_mse"},
        "members": [
            {"name": "GR", "simulated": "gr/q.csv", "results_dir": "gr/results"},
            {"name": "HYPE", "simulated": "hype/q.csv", "conditions": {"forcing_dataset": "ERA5"}},
        ],
    }
    (tmp_path / "ensemble.yaml").write_text(yaml.safe_dump(manifest))
    res = run_ensemble(manifest=tmp_path / "ensemble.yaml", output_dir=tmp_path / "out")
    gr = res.conditions.loc["GR"]
    assert gr["forcing_dataset"] == "CARRA"
    assert gr["forcing_correction"] == "fixed (precip_multiplier=1.12)"
    assert gr["discretisation"] == "lumped / GRUs"
    assert gr["calibration_algorithm"] == "DDS"
    assert gr["seed"] == "99 (drawn)"
    assert gr["n_params"] == 4 and gr["evaluations_used"] == 400 and gr["n_hrus"] == 1
    hype = res.conditions.loc["HYPE"]
    assert hype["forcing_dataset"] == "ERA5" and hype["seed"] == "unknown"
    assert "test_KGE" not in res.scores.columns


def test_member_conditions_unknown_without_files(tmp_path):
    cond = member_conditions("X", tmp_path / "nope")
    assert all(v == "unknown" for k, v in cond.items() if k != "member")


def test_cli_ensemble(synthetic, tmp_path):
    from symfluence.cli.argument_parser import CLIParser

    members, obs = synthetic
    for name in ("a", "b"):
        members["good" if name == "a" else "biased"].rename("q").to_csv(tmp_path / f"{name}.csv")
    obs.rename("q").to_csv(tmp_path / "obs.csv")
    args = CLIParser().parse_args([
        "ensemble", "--member", f"A={tmp_path / 'a.csv'}", "--member", f"B={tmp_path / 'b.csv'}",
        "--obs", str(tmp_path / "obs.csv"), "--calibration-period", CAL,
        "--evaluation-period", EVAL, "--output", str(tmp_path / "out"), "--median",
    ])
    assert args.category == "ensemble"
    assert args.func(args) == 0
    scores = pd.read_csv(tmp_path / "out" / "ensemble_scores.csv", index_col=0)
    assert {"A", "B", "equal_weight", "skill_weighted", "median"} <= set(scores.index)
