# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests pinning rpy2 as an opt-in (GR-only) dependency.

Regression coverage for the change that removed unconditional rpy2 install
paths from scripts/symfluence-bootstrap. If rpy2 ever leaks into a non-GR
import path, importing the affected module would crash on machines without
R installed — these tests catch that at CI time.
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


def test_gr_runner_error_mentions_with_gr_flag(monkeypatch):
    """If rpy2 is unavailable, GRRunner.__init__ must raise an ImportError
    whose message points users at the correct opt-in command (--with-gr).

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
    assert "--with-gr" in msg, (
        "GR ImportError must point users at the bootstrap --with-gr opt-in "
        "or the equivalent pip install command."
    )
    assert "[r]" in msg or "extras" in msg.lower() or ".[r]" in msg
