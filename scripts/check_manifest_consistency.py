#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Fail CI if the two authoritative install manifests disagree.

Review item 16 (Tier 2): the seven install paths must stay at release quality.
``pyproject.toml`` owns Python package metadata and ``pixi.toml`` owns the
Conda/system environment. Their shared packages must agree without flattening
optional groups into the base environment.

This guard checks, for a curated set of CORE packages that must stay consistent:

  1. **Presence** — each core package is declared in both manifests
     (accounting for PyPI vs conda naming: netCDF4/netcdf4, torch/pytorch,
     pvlib/pvlib-python and pint_xarray/pint-xarray).
  2. **Version bounds** — where both pyproject.toml and pixi.toml pin bounds,
     the lower (>=) and upper (<) bounds agree once normalised (2.0 == 2.0.0).
  3. **Groups** — curated optional packages remain in their matching Pixi
     feature rather than leaking into Pixi's default environment.

Packages that are deliberately ecosystem-specific (gdal as an optional/system
extra, conda-only build tools, dev-only Jupyter) are simply absent from the
core list. Run with ``--list`` to print what is checked.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI = REPO_ROOT / "pixi.toml"

# Core packages that MUST stay consistent across both authoritative manifests.
# `names` lists every spelling the package goes by across PyPI and conda so the
# presence check matches regardless of ecosystem naming. `check_version` is
# False for packages pixi/pyproject intentionally leave unpinned ("*").
CORE_PACKAGES: List[Dict] = [
    {"canonical": "numpy", "names": {"numpy"}},
    {"canonical": "pandas", "names": {"pandas"}},
    {"canonical": "scipy", "names": {"scipy"}},
    {"canonical": "xarray", "names": {"xarray"}},
    {"canonical": "cftime", "names": {"cftime"}},
    {"canonical": "numexpr", "names": {"numexpr"}},
    {"canonical": "bottleneck", "names": {"bottleneck"}},
    {"canonical": "networkx", "names": {"networkx"}},
    {"canonical": "netcdf4", "names": {"netcdf4"}},
    {"canonical": "h5netcdf", "names": {"h5netcdf"}},
    {"canonical": "geopandas", "names": {"geopandas"}},
    {"canonical": "rasterio", "names": {"rasterio"}},
    {"canonical": "pyproj", "names": {"pyproj"}},
    {"canonical": "shapely", "names": {"shapely"}},
    {"canonical": "fiona", "names": {"fiona"}},
    {"canonical": "matplotlib", "names": {"matplotlib"}},
    {"canonical": "seaborn", "names": {"seaborn"}},
    {"canonical": "pyyaml", "names": {"pyyaml"}},
    {"canonical": "requests", "names": {"requests"}},
    {"canonical": "psutil", "names": {"psutil"}},
    {"canonical": "tqdm", "names": {"tqdm"}},
    {"canonical": "rich", "names": {"rich"}},
    {"canonical": "pydantic", "names": {"pydantic"}},
    {"canonical": "pvlib", "names": {"pvlib", "pvlib-python"}},
    {"canonical": "cdsapi", "names": {"cdsapi"}},
    {"canonical": "distributed", "names": {"distributed"}},
    {"canonical": "pint-xarray", "names": {"pint-xarray"}},
    {"canonical": "rasterstats", "names": {"rasterstats"}},
]

OPTIONAL_FEATURE_PACKAGES = {
    "ml": [
        {"canonical": "scikit-learn", "pypi": {"scikit-learn"}, "pixi": {"scikit-learn"}},
        {"canonical": "torch", "pypi": {"torch"}, "pixi": {"pytorch"}},
    ],
    "sensitivity": [
        {"canonical": "salib", "pypi": {"salib"}, "pixi": {"salib"}},
    ],
    "conus404": [
        {"canonical": "intake-xarray", "pypi": {"intake-xarray"}, "pixi": {"intake-xarray"}},
    ],
}


def normalize(name: str) -> str:
    """Normalise a package name for cross-ecosystem comparison."""
    # Drop extras/markers and channel prefixes (e.g. ``pytorch::pytorch``).
    name = name.strip().split("::")[-1]
    name = re.split(r"[\s;\[<>=!~]", name, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def parse_bounds(spec: str) -> Dict[str, Tuple[int, ...]]:
    """Parse ``>=X,<Y`` style constraints into ``{op: version-tuple}``."""
    bounds: Dict[str, Tuple[int, ...]] = {}
    for op, ver in re.findall(r"(>=|<=|==|>|<)\s*([0-9][0-9.]*)", spec):
        parts = ver.rstrip(".").split(".")
        bounds[op] = tuple(int(p) for p in parts if p.isdigit())
    return bounds


def _norm_version(v: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip trailing zeros so 2.0.0 == 2.0 == 2."""
    out = list(v)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def bounds_disagree(a: Dict[str, Tuple[int, ...]], b: Dict[str, Tuple[int, ...]]) -> List[str]:
    """Return the operators on which two bound sets disagree (both must pin it)."""
    diffs = []
    for op in (">=", ">", "<", "<="):
        if op in a and op in b and _norm_version(a[op]) != _norm_version(b[op]):
            diffs.append(f"{op}{'.'.join(map(str, a[op]))} vs {op}{'.'.join(map(str, b[op]))}")
    return diffs


def load_pyproject(path: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Return the base and grouped Python dependency specifications."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: Dict[str, str] = {}
    for entry in project.get("dependencies", []):
        name = normalize(entry)
        specs[name] = entry
    groups = {}
    for group, entries in project.get("optional-dependencies", {}).items():
        groups[group] = {normalize(entry): entry for entry in entries}
    return specs, groups


def load_pixi(path: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Return Pixi's default and feature dependency specifications."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    specs: Dict[str, str] = {}
    for section in ("dependencies", "pypi-dependencies"):
        for name, val in data.get(section, {}).items():
            if isinstance(val, dict):  # {version = "...", channel = "..."}
                val = val.get("version", "*")
            specs[normalize(name)] = str(val)
    features = {}
    for feature, config in data.get("feature", {}).items():
        feature_specs = {}
        for section in ("dependencies", "pypi-dependencies"):
            for name, val in config.get(section, {}).items():
                if isinstance(val, dict):
                    val = val.get("version", "*")
                feature_specs[normalize(name)] = str(val)
        features[feature] = feature_specs
    return specs, features


def _present(names: Set[str], declared: Set[str]) -> bool:
    return bool(names & declared)


def _spec_for(names: Set[str], specs: Dict[str, str]) -> Optional[str]:
    for n in names:
        if n in specs:
            return specs[n]
    return None


def check_consistency() -> List[str]:
    """Return a list of human-readable consistency issues (empty == consistent)."""
    pyproject, pyproject_groups = load_pyproject(PYPROJECT)
    pixi, pixi_features = load_pixi(PIXI)
    pyproject_names, pixi_names = set(pyproject), set(pixi)

    issues: List[str] = []
    for pkg in CORE_PACKAGES:
        canon, names = pkg["canonical"], pkg["names"]

        missing = [
            label
            for label, declared in (
                ("pyproject.toml", pyproject_names),
                ("pixi.toml", pixi_names),
            )
            if not _present(names, declared)
        ]
        if missing:
            issues.append(f"{canon}: missing from {', '.join(missing)}")
            continue  # can't compare bounds if a manifest lacks it

        py_spec = _spec_for(names, pyproject)
        px_spec = _spec_for(names, pixi)
        diffs = bounds_disagree(parse_bounds(py_spec or ""), parse_bounds(px_spec or ""))
        if diffs:
            issues.append(
                f"{canon}: version bounds disagree (pyproject '{py_spec}' vs pixi '{px_spec}'): "
                + "; ".join(diffs)
            )

    for feature, packages in OPTIONAL_FEATURE_PACKAGES.items():
        py_group = pyproject_groups.get(feature, {})
        pixi_group = pixi_features.get(feature, {})
        for pkg in packages:
            canonical = pkg["canonical"]
            if not _present(pkg["pypi"], set(py_group)):
                issues.append(f"{canonical}: missing from pyproject extra '{feature}'")
            if not _present(pkg["pixi"], set(pixi_group)):
                issues.append(f"{canonical}: missing from pixi feature '{feature}'")
            if _present(pkg["pixi"], pixi_names):
                issues.append(f"{canonical}: optional package leaked into pixi default dependencies")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the core package list and exit")
    args = parser.parse_args()

    if args.list:
        for pkg in CORE_PACKAGES:
            print(pkg["canonical"], "->", "/".join(sorted(pkg["names"])))
        return 0

    issues = check_consistency()
    if issues:
        print("Manifest consistency check FAILED — core dependencies disagree across install paths:\n")
        for issue in issues:
            print(f"  ❌ {issue}")
        print(
            "\nKeep pyproject.toml and pixi.toml in sync for the listed packages.\n"
            "If a divergence is intentional, drop the package from CORE_PACKAGES in this script."
        )
        return 1

    print(f"Manifest consistency check passed ({len(CORE_PACKAGES)} core packages consistent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
