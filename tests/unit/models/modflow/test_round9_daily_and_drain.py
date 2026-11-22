"""MODFLOW: daily standalone/dCoupler outputs and area-scaled drain conductance."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from symfluence.models.modflow.calibration.parameter_manager import (
    MODFLOW_DEFAULT_BOUNDS,
    read_drn_conductance,
)
from symfluence.models.modflow.config_schema import MODFLOWConfig
from symfluence.models.modflow.extractor import MODFLOWResultExtractor
from symfluence.models.modflow.postprocessor import MODFLOWPostProcessor
from symfluence.models.modflow.preprocessor import default_drain_conductance

TEMPLATE = (Path(__file__).resolve().parents[4] / "src/symfluence/resources/config_templates"
            / "config_template_comprehensive.yaml")


def _five_day_periods():
    idx = pd.DatetimeIndex(["2000-01-01", "2000-01-06"])
    return pd.Series([86400.0 * 2, 86400.0 * 4], index=idx)  # m3/d → 2 and 4 m3/s


def test_standalone_baseflow_is_daily(monkeypatch):
    pp = MODFLOWPostProcessor.__new__(MODFLOWPostProcessor)
    saved = {}
    monkeypatch.setattr(pp, "save_streamflow_to_results", lambda s, *a, **k: saved.setdefault("s", s), raising=False)
    monkeypatch.setattr(pp, "_try_generate_plot", lambda: None, raising=False)
    out, meta = pp._extract_standalone_baseflow(_five_day_periods(), stress_period_length=5.0)
    assert len(out) == 10
    assert out.index[0] == pd.Timestamp("2000-01-01") and out.index[-1] == pd.Timestamp("2000-01-10")
    np.testing.assert_allclose(out.to_numpy(), [2.0] * 5 + [4.0] * 5)
    assert out.name == "streamflow_m3s"
    assert saved["s"] is out


def test_extractor_daily_option(monkeypatch, tmp_path):
    ext = MODFLOWResultExtractor()
    monkeypatch.setattr(ext, "_extract_drain_discharge", lambda path, **kw: _five_day_periods())
    per_period = ext.extract_variable(tmp_path, "drain_discharge", stress_period_length=5.0)
    daily = ext.extract_variable(tmp_path, "drain_discharge", stress_period_length=5.0, daily=True)
    assert len(per_period) == 2
    assert len(daily) == 10
    # Rates are preserved per day, so the volume over the run is unchanged.
    assert daily.sum() == per_period.mul(5).sum()


def test_drain_conductance_defaults_to_area_scaling():
    assert MODFLOWConfig().drain_conductance is None
    assert default_drain_conductance(100e6) == 50_000.0  # 100 km2
    assert default_drain_conductance(1e3) == 50.0  # floor
    assert yaml.safe_load(TEMPLATE.read_text())["MODFLOW_DRAIN_CONDUCTANCE"] is None
    # The area-scaled default for a modest basin lies inside the calibration range.
    b = MODFLOW_DEFAULT_BOUNDS["DRAIN_CONDUCTANCE"]
    assert b["min"] <= default_drain_conductance(100e6) <= b["max"]


def test_read_drn_conductance(tmp_path):
    drn = tmp_path / "gwf.drn"
    drn.write_text(
        "BEGIN OPTIONS\n  PRINT_INPUT\nEND OPTIONS\n\nBEGIN DIMENSIONS\n  MAXBOUND 1\n"
        "END DIMENSIONS\n\nBEGIN PERIOD 1\n  1 1 1 1450.0 12345.6\nEND PERIOD 1\n"
    )
    assert read_drn_conductance(drn) == 12345.6
    assert read_drn_conductance(tmp_path / "missing.drn") is None


def test_initial_drain_conductance_prefers_config_then_file(tmp_path):
    from symfluence.models.modflow.calibration.parameter_manager import (
        CoupledGWParameterManager,
    )
    (tmp_path / "gwf.drn").write_text("BEGIN PERIOD 1\n  1 1 1 1450.0 777.0\nEND PERIOD 1\n")

    def mk(cond):
        pm = CoupledGWParameterManager.__new__(CoupledGWParameterManager)
        pm.modflow_settings_dir = tmp_path
        pm.config = SimpleNamespace(model=SimpleNamespace(modflow=SimpleNamespace(drain_conductance=cond)))
        pm._get_config_value = lambda getter, default=None, **kw: (getter() if getter() is not None else default)
        return pm

    assert mk(4.2e4)._initial_drain_conductance() == 4.2e4
    assert mk(None)._initial_drain_conductance() == 777.0
    (tmp_path / "gwf.drn").unlink()
    assert mk(None)._initial_drain_conductance() == MODFLOW_DEFAULT_BOUNDS["DRAIN_CONDUCTANCE"]["min"]


def test_coupled_fallback_to_standalone_is_daily(monkeypatch, tmp_path):
    pp = MODFLOWPostProcessor.__new__(MODFLOWPostProcessor)
    pp.config = SimpleNamespace(domain=SimpleNamespace(experiment_id="e"))
    pp.config_dict = {}
    pp.logger = None
    monkeypatch.setattr(pp, "get_simulation_dir", lambda *a: tmp_path / "no_summa", raising=False)
    monkeypatch.setattr(pp, "save_streamflow_to_results", lambda s, *a, **k: None, raising=False)
    monkeypatch.setattr(pp, "_try_generate_plot", lambda: None, raising=False)
    out, meta = pp._extract_coupled_streamflow(_five_day_periods(), tmp_path, stress_period_length=5.0)
    assert meta["mode"] == "standalone"
    assert len(out) == 10
