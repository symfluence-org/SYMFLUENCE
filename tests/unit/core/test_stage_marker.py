"""
Unit tests for stage_marker module.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.core.stage_marker import (
    DOMAIN_SHARED_STAGES,
    STAGE_CONFIG_SECTIONS,
    StageMarker,
    clear_markers,
    compute_config_hash,
    compute_stage_hash,
    is_stage_current,
    read_marker,
    write_marker,
)

# ---------------------------------------------------------------------------
# Lightweight config stub
# ---------------------------------------------------------------------------


class _Section:
    """Minimal stand-in for a Pydantic config section."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self, *, by_alias: bool = False) -> dict:
        return dict(self._data)


class _FakeConfig:
    """Fake SymfluenceConfig with a few sections for hashing tests."""

    def __init__(self, **sections):
        for name, data in sections.items():
            setattr(self, name, _Section(data))


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_deterministic(self):
        cfg = _FakeConfig(domain={"name": "bow"}, data={"source": "era5"})
        h1 = compute_config_hash(cfg, ["domain", "data"])
        h2 = compute_config_hash(cfg, ["domain", "data"])
        assert h1 == h2

    def test_changes_when_config_changes(self):
        cfg_a = _FakeConfig(domain={"name": "bow"})
        cfg_b = _FakeConfig(domain={"name": "columbia"})
        assert compute_config_hash(cfg_a, ["domain"]) != compute_config_hash(
            cfg_b, ["domain"]
        )

    def test_section_order_irrelevant(self):
        cfg = _FakeConfig(domain={"name": "bow"}, data={"source": "era5"})
        h1 = compute_config_hash(cfg, ["domain", "data"])
        h2 = compute_config_hash(cfg, ["data", "domain"])
        assert h1 == h2

    def test_missing_section_ignored(self):
        cfg = _FakeConfig(domain={"name": "bow"})
        h1 = compute_config_hash(cfg, ["domain"])
        h2 = compute_config_hash(cfg, ["domain", "nonexistent"])
        assert h1 == h2

    def test_empty_sections(self):
        cfg = _FakeConfig()
        h = compute_config_hash(cfg, [])
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# write_marker / read_marker
# ---------------------------------------------------------------------------


class TestMarkerIO:
    def test_write_read_roundtrip(self, tmp_path):
        write_marker(tmp_path, "setup_project", "abc123", git_commit="deadbeef")
        marker = read_marker(tmp_path, "setup_project")

        assert marker is not None
        assert marker.stage == "setup_project"
        assert marker.config_hash == "abc123"
        assert marker.git_commit == "deadbeef"
        assert marker.completed_utc  # non-empty

    def test_missing_marker_returns_none(self, tmp_path):
        assert read_marker(tmp_path, "no_such_stage") is None

    def test_corrupt_json_returns_none(self, tmp_path):
        marker_dir = tmp_path / ".symfluence" / "stage_markers"
        marker_dir.mkdir(parents=True)
        (marker_dir / "broken.json").write_text("NOT JSON", encoding="utf-8")
        assert read_marker(tmp_path, "broken") is None

    def test_missing_key_returns_none(self, tmp_path):
        marker_dir = tmp_path / ".symfluence" / "stage_markers"
        marker_dir.mkdir(parents=True)
        (marker_dir / "partial.json").write_text(
            json.dumps({"stage": "partial"}), encoding="utf-8"
        )
        assert read_marker(tmp_path, "partial") is None

    def test_write_without_git_commit(self, tmp_path):
        with patch(
            "symfluence.core.stage_marker._current_git_commit", return_value=None
        ):
            write_marker(tmp_path, "run_models", "hash456")
        marker = read_marker(tmp_path, "run_models")
        assert marker is not None
        assert marker.git_commit is None


# ---------------------------------------------------------------------------
# is_stage_current
# ---------------------------------------------------------------------------


class TestIsStageCurrent:
    def test_true_when_hash_matches(self, tmp_path):
        write_marker(tmp_path, "define_domain", "myhash")
        assert is_stage_current(tmp_path, "define_domain", "myhash") is True

    def test_false_when_hash_differs(self, tmp_path):
        write_marker(tmp_path, "define_domain", "oldhash")
        assert is_stage_current(tmp_path, "define_domain", "newhash") is False

    def test_false_when_no_marker(self, tmp_path):
        assert is_stage_current(tmp_path, "define_domain", "any") is False


# ---------------------------------------------------------------------------
# clear_markers
# ---------------------------------------------------------------------------


class TestClearMarkers:
    def test_clear_all(self, tmp_path):
        write_marker(tmp_path, "a", "h1")
        write_marker(tmp_path, "b", "h2")
        clear_markers(tmp_path)
        assert read_marker(tmp_path, "a") is None
        assert read_marker(tmp_path, "b") is None

    def test_clear_selective(self, tmp_path):
        write_marker(tmp_path, "a", "h1")
        write_marker(tmp_path, "b", "h2")
        clear_markers(tmp_path, stage_names=["a"])
        assert read_marker(tmp_path, "a") is None
        assert read_marker(tmp_path, "b") is not None

    def test_clear_nonexistent_is_noop(self, tmp_path):
        # Should not raise
        clear_markers(tmp_path)
        clear_markers(tmp_path, stage_names=["ghost"])


# ---------------------------------------------------------------------------
# Coverage of orchestrator step names
# ---------------------------------------------------------------------------


EXPECTED_STAGES = [
    "setup_project",
    "create_pour_point",
    "acquire_attributes",
    "define_domain",
    "discretize_domain",
    "process_observed_data",
    "acquire_forcings",
    "run_model_agnostic_preprocessing",
    "build_model_ready_store",
    "preprocess_models",
    "run_models",
    "postprocess_results",
    "calibrate_model",
    "run_benchmarking",
    "run_decision_analysis",
    "run_sensitivity_analysis",
]


class TestStageConfigSectionsCoverage:
    @pytest.mark.parametrize("stage", EXPECTED_STAGES)
    def test_stage_has_sections(self, stage):
        assert stage in STAGE_CONFIG_SECTIONS
        assert len(STAGE_CONFIG_SECTIONS[stage]) > 0

    def test_no_system_or_paths_sections(self):
        for sections in STAGE_CONFIG_SECTIONS.values():
            assert "system" not in sections
            assert "paths" not in sections


# ---------------------------------------------------------------------------
# Shared-domain concurrency: markers for domain-shared stages must agree
# across models/experiments running on the same domain.
# ---------------------------------------------------------------------------


def _shared_domain_config(*, experiment_id: str, model: str) -> _FakeConfig:
    """Two workflows on one domain differing only in experiment_id + model."""
    return _FakeConfig(
        domain={"name": "bow", "discretization": "lumped", "experiment_id": experiment_id},
        data={"data_access": "MAF"},
        forcing={"dataset": "RDRS", "time_step_size": 3600},
        model={"hydrological_model": model, "params_to_calibrate": f"{model}_params"},
        evaluation={"metric": "KGE"},
        optimization={"algorithm": "DDS"},
    )


class TestDomainSharedStageHashes:
    """Regression: SYMFLUENCE#P3-16 — concurrent workflows in one domain.

    The model-agnostic products (basin-averaged forcing, model-ready store)
    are shared by every model/experiment in a domain. If a per-model or
    per-experiment config field feeds their stage hash, each concurrent
    workflow judges the shared product stale and rebuilds it underneath the
    others — deleting/rewriting files the others are reading.
    """

    def test_build_model_ready_store_does_not_hash_model_section(self):
        assert "model" not in STAGE_CONFIG_SECTIONS["build_model_ready_store"]

    @pytest.mark.parametrize("stage", sorted(DOMAIN_SHARED_STAGES))
    def test_shared_stage_hash_is_model_and_experiment_invariant(self, stage):
        hbv = _shared_domain_config(experiment_id="cal_ensemble_hbv_abc", model="HBV")
        fuse = _shared_domain_config(experiment_id="cal_ensemble_fuse_dds", model="FUSE")
        assert compute_stage_hash(hbv, stage) == compute_stage_hash(fuse, stage)

    def test_model_only_change_does_not_invalidate_store_marker(self, tmp_path):
        hbv = _shared_domain_config(experiment_id="exp", model="HBV")
        fuse = _shared_domain_config(experiment_id="exp", model="FUSE")

        write_marker(
            tmp_path,
            "build_model_ready_store",
            compute_stage_hash(hbv, "build_model_ready_store"),
        )

        assert is_stage_current(
            tmp_path,
            "build_model_ready_store",
            compute_stage_hash(fuse, "build_model_ready_store"),
        )

    def test_experiment_only_change_does_not_invalidate_forcing_marker(self, tmp_path):
        a = _shared_domain_config(experiment_id="exp_a", model="HBV")
        b = _shared_domain_config(experiment_id="exp_b", model="HBV")

        stage = "run_model_agnostic_preprocessing"
        write_marker(tmp_path, stage, compute_stage_hash(a, stage))

        assert is_stage_current(tmp_path, stage, compute_stage_hash(b, stage))

    def test_experiment_scoped_stages_still_invalidate(self):
        """Experiment-scoped stages must keep re-running per experiment."""
        a = _shared_domain_config(experiment_id="exp_a", model="HBV")
        b = _shared_domain_config(experiment_id="exp_b", model="HBV")
        assert compute_stage_hash(a, "run_models") != compute_stage_hash(b, "run_models")
        assert compute_stage_hash(a, "calibrate_model") != compute_stage_hash(b, "calibrate_model")

    def test_forcing_change_still_invalidates_shared_stage(self):
        """The exclusion is narrow: real model-agnostic changes still rebuild."""
        a = _shared_domain_config(experiment_id="exp", model="HBV")
        b = _shared_domain_config(experiment_id="exp", model="HBV")
        b.forcing = _Section({"dataset": "ERA5", "time_step_size": 3600})
        stage = "run_model_agnostic_preprocessing"
        assert compute_stage_hash(a, stage) != compute_stage_hash(b, stage)

    def test_unknown_stage_hashes_to_empty(self):
        cfg = _shared_domain_config(experiment_id="exp", model="HBV")
        assert compute_stage_hash(cfg, "not_a_stage") == ""


# ---------------------------------------------------------------------------
# Stage-level key scope + per-experiment markers (audit regression)
# ---------------------------------------------------------------------------


def _real_config(tmp_path, **extra):
    from symfluence.core.config.models import SymfluenceConfig

    base = dict(
        domain_name="scope",
        model="SUMMA",
        time_start="2010-01-01 00:00",
        time_end="2012-12-31 23:00",
        SYMFLUENCE_DATA_DIR=tmp_path,
        EXPERIMENT_ID="exp_a",
    )
    base.update(extra)
    return SymfluenceConfig.from_minimal(**base)


def _stale(a, b):
    return {s for s in STAGE_CONFIG_SECTIONS if compute_stage_hash(a, s) != compute_stage_hash(b, s)}


class TestStageKeyScope:
    def test_calibration_period_only_invalidates_experiment_stages(self, tmp_path):
        a = _real_config(tmp_path, CALIBRATION_PERIOD="2010-01-01, 2010-12-31")
        b = _real_config(tmp_path, CALIBRATION_PERIOD="2011-01-01, 2011-12-31")
        stale = _stale(a, b)
        assert not (stale & DOMAIN_SHARED_STAGES), stale
        assert {"run_models", "calibrate_model"} <= stale

    def test_spinup_period_does_not_invalidate_domain_stages(self, tmp_path):
        a = _real_config(tmp_path)
        b = _real_config(tmp_path, SPINUP_PERIOD="2010-01-01, 2010-06-30")
        assert not (_stale(a, b) & DOMAIN_SHARED_STAGES)

    def test_dem_source_spares_setup_and_pour_point(self, tmp_path):
        a = _real_config(tmp_path)
        b = _real_config(tmp_path, DEM_SOURCE="merit")
        stale = _stale(a, b)
        assert "setup_project" not in stale
        assert "create_pour_point" not in stale
        # conservative: everything built from the DEM still invalidates
        assert {"acquire_attributes", "define_domain", "discretize_domain"} <= stale

    def test_time_window_still_invalidates_forcing(self, tmp_path):
        a = _real_config(tmp_path)
        b = _real_config(tmp_path, EXPERIMENT_TIME_END="2013-12-31 23:00")
        stale = _stale(a, b)
        assert "acquire_forcings" in stale
        assert "define_domain" not in stale

    def test_analyses_change_does_not_invalidate_calibration(self, tmp_path):
        a = _real_config(tmp_path, ANALYSES=["benchmarking"])
        b = _real_config(tmp_path, ANALYSES=["benchmarking", "decision"])
        stale = _stale(a, b)
        assert "calibrate_model" not in stale
        assert "process_observed_data" not in stale


class TestPerExperimentMarkers:
    def test_experiments_do_not_overwrite_each_other(self, tmp_path):
        from symfluence.core.stage_marker import mark_stage_complete, stage_is_current

        a = _real_config(tmp_path, EXPERIMENT_ID="exp_a")
        b = _real_config(tmp_path, EXPERIMENT_ID="exp_b")
        proj = tmp_path / "proj"
        mark_stage_complete(a, proj, "run_models")
        mark_stage_complete(b, proj, "run_models")
        assert stage_is_current(a, proj, "run_models")
        assert stage_is_current(b, proj, "run_models")
        # domain-shared stages stay one marker for the whole domain
        mark_stage_complete(a, proj, "define_domain")
        assert stage_is_current(b, proj, "define_domain")

    def test_legacy_marker_still_honoured(self, tmp_path):
        """A marker written before hash_version existed keeps its stage current."""
        from symfluence.core.stage_marker import compute_legacy_stage_hash, stage_is_current

        cfg = _real_config(tmp_path, CALIBRATION_PERIOD="2010-01-01, 2010-12-31")
        proj = tmp_path / "proj"
        for stage in ("define_domain", "run_models"):
            path = proj / ".symfluence" / "stage_markers" / f"{stage}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "stage": stage, "completed_utc": "x", "symfluence_version": "0",
                "config_hash": compute_legacy_stage_hash(cfg, stage),
            }))
            assert stage_is_current(cfg, proj, stage), stage

    def test_selective_clear_keeps_other_experiments(self, tmp_path):
        from symfluence.core.stage_marker import (
            clear_stage_markers,
            mark_stage_complete,
            stage_is_current,
        )

        a = _real_config(tmp_path, EXPERIMENT_ID="exp_a")
        b = _real_config(tmp_path, EXPERIMENT_ID="exp_b")
        proj = tmp_path / "proj"
        for cfg in (a, b):
            mark_stage_complete(cfg, proj, "run_models")
        clear_stage_markers(a, proj, ["run_models"])
        assert not stage_is_current(a, proj, "run_models")
        assert stage_is_current(b, proj, "run_models")
