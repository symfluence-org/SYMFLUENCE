# SPDX-License-Identifier: GPL-3.0-or-later
"""Guards for runtime dependencies that are needed but never imported by name.

A dependency audit that only greps for ``import <name>`` flags these as unused.
They are not: removing either breaks forcing preprocessing at run time.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import xarray as xr

from scripts.check_manifest_consistency import normalize

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _declared_runtime_dependencies() -> set[str]:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {normalize(entry) for entry in data["project"]["dependencies"]}


def test_xarray_fill_methods_work_in_this_environment() -> None:
    """xarray's ffill/bfill need bottleneck; the gap-filling paths call both."""
    da = xr.DataArray([np.nan, 1.0, np.nan, 3.0, np.nan], dims="time")

    filled = da.ffill(dim="time").bfill(dim="time")

    np.testing.assert_array_equal(filled.values, [1.0, 1.0, 1.0, 3.0, 3.0])


def test_bottleneck_and_dask_stay_declared() -> None:
    """bottleneck (xarray ffill/bfill) and dask (lazy reads, scheduler choice)."""
    declared = _declared_runtime_dependencies()

    assert "bottleneck" in declared
    assert "dask" in declared
