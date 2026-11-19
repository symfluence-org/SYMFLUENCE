"""Tests for symfluence.core.provenance."""
from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path

import pytest

from symfluence.core.provenance import (
    RunProvenance,
    _dependency_versions,
    _git_info,
    _platform_info,
    capture_provenance,
    finalize,
    make_run_id,
    record_step,
)

# ---------------------------------------------------------------------------
# _git_info
# ---------------------------------------------------------------------------

def test_git_info_returns_dict():
    """_git_info should return a dict with the expected keys."""
    info = _git_info()
    assert set(info.keys()) == {"commit", "commit_short", "branch", "dirty"}
    # If run inside a git repo the short hash is 8 chars
    if info["commit"] is not None:
        assert len(info["commit_short"]) == 8


def test_git_info_graceful_outside_repo(tmp_path):
    """_git_info should return Nones when pointed at a non-repo dir."""
    info = _git_info(tmp_path)
    assert info["commit"] is None
    assert info["commit_short"] is None
    assert info["branch"] is None
    assert info["dirty"] is None


# ---------------------------------------------------------------------------
# _dependency_versions / _platform_info
# ---------------------------------------------------------------------------

def test_dependency_versions():
    """numpy should be present; a made-up package should not."""
    deps = _dependency_versions()
    assert "numpy" in deps
    assert "nonexistent_pkg_xyz" not in deps


def test_platform_info():
    """Keys exist and python version matches the running interpreter."""
    info = _platform_info()
    assert set(info.keys()) == {"os", "os_version", "arch", "hostname", "python"}
    assert info["python"] == platform.python_version()


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------

def test_make_run_id_format():
    """Run ID matches expected pattern: <exp>_<YYYYMMDD_HHMMSS>_<hash>."""
    rid = make_run_id("bow_banff", git_short="abcd1234")
    assert re.match(r"bow_banff_\d{8}_\d{6}_abcd1234$", rid)


def test_make_run_id_no_git():
    """Without git info the run ID ends with '_nogit'."""
    rid = make_run_id("test_exp")
    assert rid.endswith("_nogit")


# ---------------------------------------------------------------------------
# capture_provenance
# ---------------------------------------------------------------------------

def test_capture_provenance(tmp_path):
    """capture_provenance returns a populated RunProvenance."""
    config_path = str(tmp_path / "cfg.yaml")
    prov = capture_provenance("exp1", "domain1", config_path=config_path)
    assert prov.experiment_id == "exp1"
    assert prov.domain_name == "domain1"
    assert prov.config_path == config_path
    assert prov.symfluence_version  # non-empty
    assert prov.start_utc  # non-empty
    assert prov.steps == []
    assert prov.status == "running"


# ---------------------------------------------------------------------------
# record_step
# ---------------------------------------------------------------------------

def test_record_step():
    """record_step appends entries and captures errors."""
    prov = capture_provenance("e", "d")
    record_step(prov, "step_a", 1.23)
    record_step(prov, "step_b", 0.0, status="failed", error="boom")

    assert len(prov.steps) == 2
    assert prov.steps[0]["name"] == "step_a"
    assert prov.steps[0]["status"] == "completed"
    assert prov.steps[1]["error"] == "boom"

    # No-op on None
    record_step(None, "x", 0.0)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def test_finalize():
    """finalize stamps end_utc, elapsed, and status."""
    prov = capture_provenance("e", "d")
    time.sleep(0.01)  # ensure measurable elapsed
    finalize(prov, "completed")

    assert prov.end_utc is not None
    assert prov.elapsed_seconds >= 0
    assert prov.status == "completed"

    # None provenance is a no-op
    assert finalize(None, "completed") is None


# ---------------------------------------------------------------------------
# write / to_dict
# ---------------------------------------------------------------------------

def test_write_manifest(tmp_path):
    """write() produces a valid JSON file that round-trips."""
    prov = capture_provenance("e", "d")
    record_step(prov, "a", 0.5)
    finalize(prov, "completed")

    path = prov.write(tmp_path)
    assert path.exists()
    assert path.name == "run_manifest.json"

    data = json.loads(path.read_text())
    assert data["experiment_id"] == "e"
    assert len(data["steps"]) == 1


def test_to_dict_serializable():
    """to_dict() output must be JSON-serializable."""
    prov = capture_provenance("e", "d")
    record_step(prov, "s", 0.1)
    finalize(prov, "completed", errors=["oops"])

    # This will raise if any value isn't serializable
    text = json.dumps(prov.to_dict())
    assert "oops" in text


# ---------------------------------------------------------------------------
# record_executable: upstream source of a binary
# ---------------------------------------------------------------------------

def _fake_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF not really")
    return path


def test_record_executable_records_source_from_install_record(tmp_path):
    from symfluence.core.provenance import record_executable

    tool_dir = tmp_path / "installs" / "summa"
    exe = _fake_exe(tool_dir / "bin" / "summa_sundials.exe")
    (tool_dir / ".symfluence_source.json").write_text(json.dumps({
        "tool": "summa", "repository": "https://github.com/CH-Earth/summa.git",
        "ref": "develop_sundials", "pinned_commit": "a" * 40, "commit": "a" * 40,
        "pinned_by": "binary_pins.json", "tracked_changes": False, "patched": False,
        "installed_utc": "2026-11-19T00:00:00Z",
    }))
    prov = capture_provenance("exp", "dom")
    record_executable(prov, "SUMMA", exe)

    info = prov.executables["SUMMA"]
    assert info["sha256"]  # existing behaviour kept
    assert info["source"]["repository"] == "https://github.com/CH-Earth/summa.git"
    assert info["source"]["commit"] == info["source"]["pinned_commit"] == "a" * 40
    assert info["source"]["ref"] == "develop_sundials"
    assert "installed_utc" not in info["source"]
    assert json.loads(json.dumps(prov.to_dict()))["executables"]["SUMMA"]["source"]


def test_record_executable_falls_back_to_bundle_toolchain_manifest(tmp_path):
    """npm/release bundles ship bin/<tool> next to a toolchain.json."""
    from symfluence.core.provenance import record_executable

    exe = _fake_exe(tmp_path / "dist" / "bin" / "mizuroute")
    (tmp_path / "dist" / "toolchain.json").write_text(json.dumps({"tools": {
        "mizuroute": {"commit": "b" * 40, "branch": "HEAD",
                      "repository": "https://github.com/ESCOMP/mizuRoute.git",
                      "source": {"pinned_commit": "b" * 40, "ref": "serial"}},
    }}))
    prov = capture_provenance("exp", "dom")
    record_executable(prov, "MIZUROUTE", exe)

    source = prov.executables["MIZUROUTE"]["source"]
    assert source["commit"] == source["pinned_commit"] == "b" * 40
    assert source["ref"] == "serial"


def test_record_executable_without_source_record_has_no_source(tmp_path):
    from symfluence.core.provenance import record_executable

    exe = _fake_exe(tmp_path / "somewhere" / "bin" / "model")
    prov = capture_provenance("exp", "dom")
    record_executable(prov, "MODEL", exe)
    assert "source" not in prov.executables["MODEL"]
