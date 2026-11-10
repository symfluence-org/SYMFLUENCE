# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Output parsing for the WATFLOOD and GSFLOW standalone postprocessors.

These two models ran and calibrated but had no registered postprocessor
(documented in the INCOMPLETE_MODELS baseline) until the implementations
that already lived in their packages were wired into model_manifest().
The parsers only need a logger, so they are exercised unbound.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_self():
    return SimpleNamespace(logger=__import__("logging").getLogger("test"))


class TestRegistration:
    def test_both_postprocessors_registered(self):
        from symfluence.core.registries import R

        assert R.postprocessors.get("WATFLOOD") is not None
        assert R.postprocessors.get("GSFLOW") is not None

    def test_validate_model_reports_complete(self):
        from symfluence.core.registries import R

        for model in ("WATFLOOD", "GSFLOW"):
            report = R.validate_model(model)
            assert report.get("postprocessor") or report.get("valid"), report


class TestWATFLOODTb0Parsing:
    def test_parses_tb0_timeseries(self, tmp_path, fake_self):
        from symfluence.models.watflood.postprocessor import WATFLOODPostProcessor

        tb0 = tmp_path / "spl_test.tb0"
        tb0.write_text(
            ":FileType tb0  ASCII\n"
            ":Name streamflow\n"
            "#\n"
            "2020 01 01 00 1.25\n"
            "2020 01 02 00 2.50\n"
            "2020 01 03 00 3.75\n"
        )
        series = WATFLOODPostProcessor._parse_tb0(fake_self, tb0)
        assert series is not None
        assert len(series) == 3
        assert series.iloc[1] == 2.50
        assert str(series.index[0].date()) == "2020-01-01"

    def test_returns_none_on_garbage(self, tmp_path, fake_self):
        from symfluence.models.watflood.postprocessor import WATFLOODPostProcessor

        bad = tmp_path / "spl_bad.tb0"
        bad.write_text(":Header only\n# no data lines\n")
        assert WATFLOODPostProcessor._parse_tb0(fake_self, bad) is None


class TestGSFLOWStatvarParsing:
    """The postprocessor shares the calibration worker's extraction rule."""

    def test_statvar_fallback_uses_header_and_cfs_conversion(self, tmp_path, fake_self):
        from symfluence.models.gsflow.postprocessor import GSFLOWPostProcessor

        statvar = tmp_path / "statvar.dat"
        statvar.write_text(
            "2\nbasin_actet 1\nbasin_cfs 1\n"
            "1 2020 1 1 0 0 0 5.0 100.0\n"
            "2 2020 1 2 0 0 0 6.0 200.0\n"
        )
        series = GSFLOWPostProcessor.extract_streamflow_from_dir(fake_self, tmp_path, {})
        assert series is not None
        assert len(series) == 2
        # basin_cfs (second column) 100 cfs -> 2.8317 cms
        assert series.iloc[0] == pytest.approx(2.8316847, rel=1e-6)

    def test_coupled_csv_uses_streamout_q(self, tmp_path, fake_self):
        import pandas as pd

        from symfluence.models.gsflow.postprocessor import GSFLOWPostProcessor

        pd.DataFrame(
            {"Date": ["2020-01-01", "2020-01-02"], "StreamOut_Q": [86400.0, 172800.0],
             "basin_cfs": [0.0, 0.0]}
        ).to_csv(tmp_path / "gsflow.csv", index=False)
        # A statvar whose basin_cfs is ~0 (coupled mode) must be ignored
        (tmp_path / "statvar.dat").write_text("1\nbasin_cfs 1\n1 2020 1 1 0 0 0 0.0\n")
        series = GSFLOWPostProcessor.extract_streamflow_from_dir(fake_self, tmp_path, {})
        assert list(series) == pytest.approx([1.0, 2.0])
