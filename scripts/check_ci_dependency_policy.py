#!/usr/bin/env python3
"""Guard package-cache ownership and architecture-specific HDF5 policy."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

X86_HDF5_POLICY = "'h5py>=3.16' netCDF4"
ARM_HDF5_POLICY = "'h5py<3.16' 'netCDF4<1.7.5'"


def _uses_uv_sync(job: dict) -> bool:
    return any(
        "uv sync" in str(step.get("run", ""))
        for step in job.get("steps", [])
        if isinstance(step, dict)
    )


def _uses_setup_python_pip_cache(job: dict) -> bool:
    for step in job.get("steps", []):
        if not isinstance(step, dict) or "actions/setup-python@" not in str(
            step.get("uses", "")
        ):
            continue
        settings = step.get("with", {})
        if not isinstance(settings, dict):
            continue
        cache = settings.get("cache")
        if str(cache).strip().lower() == "pip":
            return True
    return False


def check_workflow_policy(workflows: Path = WORKFLOWS) -> list[str]:
    """Return policy violations across workflow YAML files."""
    issues: list[str] = []
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (document.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            if _uses_uv_sync(job) and _uses_setup_python_pip_cache(job):
                issues.append(
                    f"{path.name}:{job_name}: uv-managed jobs must not enable "
                    "setup-python's pip cache"
                )

    expected = {
        "ci.yml": X86_HDF5_POLICY,
        "install-validate.yml": X86_HDF5_POLICY,
        "install-validate-arm.yml": ARM_HDF5_POLICY,
    }
    for filename, policy in expected.items():
        path = workflows / filename
        text = path.read_text(encoding="utf-8")
        if policy not in text:
            issues.append(f"{filename}: missing architecture HDF5 policy {policy}")

    arm_text = (workflows / "install-validate-arm.yml").read_text(encoding="utf-8")
    if X86_HDF5_POLICY in arm_text:
        issues.append("install-validate-arm.yml: contains the x86 HDF5 policy")
    for filename in ("ci.yml", "install-validate.yml"):
        if ARM_HDF5_POLICY in (workflows / filename).read_text(encoding="utf-8"):
            issues.append(f"{filename}: contains the ARM HDF5 policy")
    return issues


def main() -> int:
    issues = check_workflow_policy()
    if issues:
        print("CI dependency policy check FAILED:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("CI dependency policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
