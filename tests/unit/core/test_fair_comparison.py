# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Seeds, resolved-config hash, input fingerprints and per-parameter budget."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from symfluence.core import provenance as prov_mod
from symfluence.core.calibration.budget import resolve_budget
from symfluence.core.calibration.optimizers.algorithms.dds import DDSAlgorithm
from symfluence.core.calibration.optimizers.algorithms.pso import PSOAlgorithm
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.provenance import (
    capture_provenance,
    config_sha256,
    fingerprint_files,
    resolve_seed,
    write_resolved_config,
)

LOG = logging.getLogger("test_fair_comparison")


def _callbacks(points):
    def evaluate_solution(x, it):
        points.append(np.array(x, dtype=float))
        return float(-np.sum((np.asarray(x) - 0.3) ** 2))

    def evaluate_population(pop, it):
        points.extend(np.array(pop, dtype=float))
        return -np.sum((np.asarray(pop) - 0.3) ** 2, axis=1)

    return dict(
        evaluate_solution=evaluate_solution,
        evaluate_population=evaluate_population,
        denormalize_params=lambda x: {f"p{i}": float(v) for i, v in enumerate(np.ravel(x))},
        record_iteration=lambda *a, **k: None,
        update_best=lambda *a, **k: None,
        log_progress=lambda *a, **k: None,
    )


# --- seeds -----------------------------------------------------------------

def test_configured_seed_is_unchanged():
    assert resolve_seed(42) == (42, "config")
    assert resolve_seed("7") == (7, "config")


def test_unset_seed_is_drawn_and_logged(caplog):
    with caplog.at_level(logging.INFO):
        seed, source = resolve_seed(None, LOG, context="calibration")
    assert source == "drawn"
    assert 0 <= seed < prov_mod.SEED_UPPER
    assert f"RANDOM_SEED: {seed}" in caplog.text
    for unset in ("None", "", "null"):
        assert resolve_seed(unset)[1] == "drawn"


@pytest.mark.parametrize("cls,cfg", [
    (DDSAlgorithm, {"NUMBER_OF_ITERATIONS": 15}),
    (PSOAlgorithm, {"NUMBER_OF_ITERATIONS": 3, "POPULATION_SIZE": 5}),
])
def test_unseeded_run_is_regenerable_from_recorded_seed(cls, cfg):
    first: list = []
    algo = cls(dict(cfg), LOG)
    algo.optimize(n_params=3, **_callbacks(first))
    drawn = algo.random_seed
    assert isinstance(drawn, int)

    again: list = []
    cls({**cfg, "RANDOM_SEED": drawn}, LOG).optimize(n_params=3, **_callbacks(again))
    np.testing.assert_array_equal(np.array(first), np.array(again))


def test_seeded_run_behaviour_unchanged():
    a: list = []
    b: list = []
    DDSAlgorithm({"NUMBER_OF_ITERATIONS": 10, "RANDOM_SEED": 3}, LOG).optimize(n_params=2, **_callbacks(a))
    # The seeded stream is exactly numpy's default_rng(3), as before.
    algo = DDSAlgorithm({"NUMBER_OF_ITERATIONS": 10, "RANDOM_SEED": 3}, LOG)
    assert algo._new_rng().random() == np.random.default_rng(3).random()
    DDSAlgorithm({"NUMBER_OF_ITERATIONS": 10, "RANDOM_SEED": 3}, LOG).optimize(n_params=2, **_callbacks(b))
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_seed_and_budget_recorded_in_calibration_outputs(tmp_path):
    prov = capture_provenance("exp", "dom")
    prov_mod.set_active_provenance(prov)
    try:
        fake = SimpleNamespace(
            get_crash_stats=lambda: {"total_evaluations": 37},
            _n_population_evaluations=0,
            _get_model_name=lambda: "GR",
            experiment_id="exp",
            random_seed=1234,
            random_seed_source="drawn",
            param_manager=SimpleNamespace(all_param_names=["x1", "x2"]),
            target_metric="KGE",
            config={"A": 1},
            logger=LOG,
            results_dir=tmp_path,
            _iteration_history=[],
        )
        budget = resolve_budget("per_parameter", 2, 1000, "DDS", 50,
                                evaluations_per_parameter=20, min_evaluations=10)
        path = BaseModelOptimizer._record_calibration_conditions(
            fake, "dds", SimpleNamespace(name="DDS"), budget)
    finally:
        prov_mod.set_active_provenance(None)
    data = json.loads(path.read_text())
    assert path.name == "exp_dds_calibration_conditions.json"
    assert data["random_seed"] == 1234 and data["random_seed_source"] == "drawn"
    assert data["evaluations_used"] == 37
    assert data["n_params"] == 2 and data["budget"]["target_evaluations"] == 40
    assert prov.calibration[-1]["random_seed"] == 1234
    assert "calibration" in prov.to_dict()


# --- resolved config -------------------------------------------------------

def test_config_hash_is_stable_and_order_independent():
    a = {"DOMAIN_NAME": "x", "NUMBER_OF_ITERATIONS": 10, "PATH": "/a"}
    b = {"PATH": "/a", "NUMBER_OF_ITERATIONS": 10, "DOMAIN_NAME": "x"}
    assert config_sha256(a) == config_sha256(b)
    assert config_sha256(a) != config_sha256({**a, "NUMBER_OF_ITERATIONS": 11})


def test_resolved_config_secrets_masked_and_hash_reproducible(tmp_path):
    cfg = {"DOMAIN_NAME": "x", "CDS_API_KEY": "s3cret", "N": 1.5}
    prov = capture_provenance("exp", "dom")
    path, sha = write_resolved_config(cfg, tmp_path, prov=prov)
    text = path.read_text()
    assert "s3cret" not in text
    assert sha in text
    assert config_sha256(yaml.safe_load(text)) == sha
    assert prov.resolved_config["sha256"] == sha
    assert json.loads(json.dumps(prov.to_dict()))["resolved_config"]["sha256"] == sha


def test_typed_config_hash_stable_across_loads(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig

    base = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path), "SYMFLUENCE_CODE_DIR": str(tmp_path),
        "DOMAIN_NAME": "t", "EXPERIMENT_ID": "e", "EXPERIMENT_TIME_START": "2000-01-01 00:00",
        "EXPERIMENT_TIME_END": "2001-01-01 00:00", "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs", "HYDROLOGICAL_MODEL": "GR",
        "FORCING_DATASET": "ERA5", "POUR_POINT_COORDS": "51/-115",
    }
    one = SymfluenceConfig(**base)
    two = SymfluenceConfig(**dict(reversed(list(base.items()))))
    assert config_sha256(one) == config_sha256(two)
    # Spelling out a default does not change the resolved config.
    three = SymfluenceConfig(**base, NUMBER_OF_ITERATIONS=1000)
    assert config_sha256(one) == config_sha256(three)
    four = SymfluenceConfig(**base, NUMBER_OF_ITERATIONS=999)
    assert config_sha256(one) != config_sha256(four)


# --- input fingerprints ----------------------------------------------------

def test_fingerprint_listing_and_content(tmp_path):
    d = tmp_path / "forcing"
    d.mkdir()
    (d / "a.nc").write_bytes(b"abc")
    (d / "b.nc").write_bytes(b"defg")
    listing = fingerprint_files(d, "listing", patterns=("*.nc",))
    content = fingerprint_files(d, "content", patterns=("*.nc",))
    assert listing["n_files"] == 2 and listing["total_bytes"] == 7
    assert listing["sha256"] != content["sha256"]
    assert fingerprint_files(d, "content", patterns=("*.nc",))["sha256"] == content["sha256"]
    (d / "b.nc").write_bytes(b"defh")
    assert fingerprint_files(d, "content", patterns=("*.nc",))["sha256"] != content["sha256"]
    assert fingerprint_files(d, "none") is None
    assert fingerprint_files(tmp_path / "missing") is None


# --- budget ----------------------------------------------------------------

def test_fixed_budget_is_todays_behaviour():
    b = resolve_budget("fixed", 30, 1000, "DDS", 50)
    assert b.iterations == 1000
    assert resolve_budget(None, 4, 1000, "PSO", 50).iterations == 1000


def test_per_parameter_budget_scales_with_params_and_caps():
    gr = resolve_budget("per_parameter", 4, 1000, "DDS", 50, evaluations_per_parameter=100,
                        min_evaluations=100)
    summa = resolve_budget("per_parameter", 30, 1000, "DDS", 50, evaluations_per_parameter=100,
                           min_evaluations=100)
    assert gr.iterations == 400 and summa.iterations == 3000
    assert gr.target_evaluations / 4 == summa.target_evaluations / 30
    capped = resolve_budget("per_parameter", 30, 1000, "DDS", 50, evaluations_per_parameter=100,
                            min_evaluations=100, max_evaluations=2000)
    assert capped.iterations == 2000
    floor = resolve_budget("per_parameter", 1, 1000, "DDS", 50, evaluations_per_parameter=10,
                           min_evaluations=250)
    assert floor.iterations == 250


def test_per_parameter_budget_converts_to_generations():
    b = resolve_budget("per_parameter", 10, 1000, "PSO", 20, evaluations_per_parameter=100,
                       min_evaluations=1)
    assert b.target_evaluations == 1000 and b.evaluations_per_iteration == 20
    assert b.iterations == 50


def test_unknown_budget_mode_raises():
    with pytest.raises(ValueError):
        resolve_budget("bogus", 1, 1, "DDS", 1)


def test_optimizer_applies_budget_to_algorithm():
    from symfluence.core.config.models import SymfluenceConfig  # noqa: F401  (schema import)

    algo = DDSAlgorithm({"NUMBER_OF_ITERATIONS": 1000, "RANDOM_SEED": 1}, LOG)
    cfg = {"CALIBRATION_BUDGET_MODE": "per_parameter", "EVALUATIONS_PER_PARAMETER": 50,
           "CALIBRATION_BUDGET_MIN_EVALUATIONS": 10}
    tracker = SimpleNamespace(max_iterations=1000)

    def get(_fn, default=None, dict_key=None):
        return cfg.get(dict_key, default)

    fake = SimpleNamespace(_get_config_value=get, logger=LOG, max_iterations=1000,
                           _metrics_tracker=tracker, config=None)
    budget = BaseModelOptimizer._apply_calibration_budget(fake, algo, 6)
    assert budget.iterations == 300
    assert algo.max_iterations == 300 and fake.max_iterations == 300 and tracker.max_iterations == 300

    cfg["CALIBRATION_BUDGET_MODE"] = "fixed"
    algo2 = DDSAlgorithm({"NUMBER_OF_ITERATIONS": 1000, "RANDOM_SEED": 1}, LOG)
    assert BaseModelOptimizer._apply_calibration_budget(fake, algo2, 6).iterations == 1000
    assert algo2.max_iterations == 1000


def test_run_manifest_carries_config_hash_and_input_fingerprints(tmp_path):
    from symfluence.project.system import SYMFLUENCE

    domain = tmp_path / "domain_d"
    forcing = domain / "data" / "forcing" / "basin_averaged_data"
    obs = domain / "data" / "observations" / "streamflow" / "preprocessed"
    for d in (forcing, obs):
        d.mkdir(parents=True)
    (forcing / "f_2000.nc").write_bytes(b"x" * 10)
    (obs / "d_streamflow_processed.csv").write_text("time,q\n")
    typed = SimpleNamespace(
        paths=SimpleNamespace(root_path=str(tmp_path)),
        domain=SimpleNamespace(name="d"),
        system=SimpleNamespace(provenance_input_hash="content"),
        to_dict=lambda flatten=True: {"DOMAIN_NAME": "d", "N": 1},
    )
    fake = SimpleNamespace(
        provenance=capture_provenance("e", "d"), typed_config=typed, logger=LOG,
        logging_manager=SimpleNamespace(log_dir=tmp_path / "_workLog_d"),
    )
    SYMFLUENCE._record_input_provenance(fake)
    SYMFLUENCE._write_run_manifest(fake)
    manifest = json.loads((tmp_path / "_workLog_d" / "run_manifest.json").read_text())
    assert manifest["resolved_config"]["sha256"] == config_sha256({"DOMAIN_NAME": "d", "N": 1})
    assert (tmp_path / "_workLog_d" / "resolved_config.yaml").exists()
    assert manifest["inputs"]["forcing"]["n_files"] == 1
    assert manifest["inputs"]["forcing"]["mode"] == "content"
    assert manifest["inputs"]["observations"]["n_files"] == 1
