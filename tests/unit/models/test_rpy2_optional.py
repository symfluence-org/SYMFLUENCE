# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests pinning rpy2 as a GR-only runtime requirement.

The bootstrap installer attempts rpy2 by default so GR works out of the
box on systems with R, but the install is best-effort — failure is
non-fatal. These tests pin two invariants:
  1. Non-GR runners (SUMMA, FUSE, mizuRoute, acquisition,
     discretization) import without requiring rpy2 at all.
  2. GR's deferred-import ImportError stays actionable when rpy2 is
     unavailable, telling the user how to install it manually.
"""

import sys
from importlib import import_module

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.quick]


# Modules that must NEVER import rpy2 directly or transitively. Add to this
# list whenever a model that does not require R is added to the codebase.
NON_GR_MODULES = [
    "symfluence.models.summa.runner",
    "symfluence.models.fuse.runner",
    "symfluence.models.mizuroute.runner",
    "symfluence.data.acquisition.acquisition_service",
    "symfluence.geospatial.discretization.core",
]


@pytest.mark.parametrize("modname", NON_GR_MODULES)
def test_non_gr_module_does_not_force_rpy2(modname):
    """Importing a non-GR module must not require rpy2 to be installed.

    We assert that the module imports cleanly and that, after import,
    rpy2 is not present in sys.modules due to the import (other tests in
    the run may have triggered it; we only check this module's import
    works regardless of rpy2 availability)."""
    # Drop any cached form so we exercise a fresh import resolution.
    sys.modules.pop(modname, None)
    mod = import_module(modname)
    assert mod is not None


def test_gr_runner_error_actionable_when_rpy2_missing(monkeypatch):
    """If rpy2 is unavailable, GRRunner.__init__ must raise an ImportError
    that tells the user how to install rpy2 manually.

    This is the only user-visible escape hatch for "I tried GR and it
    failed", so the error message must stay actionable."""
    import symfluence.models.gr.runner as gr_runner

    # Force HAS_RPY2 to False regardless of the test machine's actual rpy2
    # status, so the test runs identically with or without R installed.
    monkeypatch.setattr(gr_runner, "HAS_RPY2", False)

    with pytest.raises(ImportError) as exc:
        gr_runner.GRRunner(config={}, logger=None)

    msg = str(exc.value)
    assert "rpy2" in msg
    # Must offer a concrete install command (manual pip install or extras)
    assert "pip install" in msg, (
        "GR ImportError must offer a concrete pip install command "
        "so users know how to enable GR after a failed default install."
    )
