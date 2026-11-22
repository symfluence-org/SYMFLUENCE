# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional-dependency behavior for SALib-backed sensitivity methods."""
from __future__ import annotations

import numpy as np
import pytest

from symfluence.evaluation import sensitivity_analysis


@pytest.mark.parametrize("fn_name", ["sobol_indices", "morris_indices"])
def test_salib_methods_explain_optional_extra(monkeypatch, fn_name):
    monkeypatch.setattr(sensitivity_analysis, "sobol", None)
    monkeypatch.setattr(sensitivity_analysis, "sobol_sample", None)

    def never_called(x):
        raise AssertionError("no model run may start without SALib")

    with pytest.raises(ImportError, match=r"symfluence\[sensitivity\]"):
        getattr(sensitivity_analysis, fn_name)(["a", "b"], never_called)


def test_surrogate_methods_are_gone():
    """The trajectory-surrogate Sobol / RBD-FAST / VISCOUS paths were removed."""
    cls = sensitivity_analysis.SensitivityAnalyzer
    for name in ("perform_sobol_analysis", "perform_rbd_fast_analysis",
                 "perform_sensitivity_analysis"):
        assert not hasattr(cls, name)
    assert np  # keep numpy import used
