"""Tests for CI dependency/cache ownership policy."""
from __future__ import annotations

from pathlib import Path

from scripts.check_ci_dependency_policy import check_workflow_policy


def _write_required_hdf_workflows(root: Path) -> None:
    (root / "install-validate.yml").write_text(
        "jobs:\n  install:\n    steps:\n"
        "      - run: pip install 'h5py>=3.16' netCDF4\n",
        encoding="utf-8",
    )
    (root / "install-validate-arm.yml").write_text(
        "jobs:\n  install:\n    steps:\n"
        "      - run: pip install 'h5py<3.16' 'netCDF4<1.7.5'\n",
        encoding="utf-8",
    )


def test_uv_job_cannot_claim_setup_python_pip_cache(tmp_path):
    _write_required_hdf_workflows(tmp_path)
    (tmp_path / "ci.yml").write_text(
        """
jobs:
  unit:
    steps:
      - uses: actions/setup-python@sha
        with:
          cache: pip
      - run: uv sync --locked
      - run: uv pip install 'h5py>=3.16' netCDF4
""",
        encoding="utf-8",
    )

    issues = check_workflow_policy(tmp_path)

    assert any("uv-managed jobs" in issue for issue in issues)


def test_pip_cache_is_allowed_for_pip_managed_job(tmp_path):
    _write_required_hdf_workflows(tmp_path)
    (tmp_path / "ci.yml").write_text(
        """
jobs:
  lint:
    steps:
      - uses: actions/setup-python@sha
        with:
          cache: pip
      - run: pip install ruff
  unit:
    steps:
      - uses: actions/setup-python@sha
      - run: uv sync --locked
      - run: uv pip install 'h5py>=3.16' netCDF4
""",
        encoding="utf-8",
    )

    assert check_workflow_policy(tmp_path) == []


def test_arm_workflow_rejects_x86_hdf5_policy(tmp_path):
    _write_required_hdf_workflows(tmp_path)
    (tmp_path / "ci.yml").write_text(
        "jobs:\n  unit:\n    steps:\n"
        "      - run: uv pip install 'h5py>=3.16' netCDF4\n",
        encoding="utf-8",
    )
    with (tmp_path / "install-validate-arm.yml").open("a", encoding="utf-8") as stream:
        stream.write("      - run: pip install 'h5py>=3.16' netCDF4\n")

    issues = check_workflow_policy(tmp_path)

    assert any("contains the x86 HDF5 policy" in issue for issue in issues)
