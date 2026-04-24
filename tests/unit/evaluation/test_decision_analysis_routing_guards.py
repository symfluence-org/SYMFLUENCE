# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Decision analysis: pluggable streamflow reader + loud-fail on all-NaN.

SH (iter-3, 04.SH.4) saw SUMMA+DDS decision analysis silently complete
with zero valid rows: mizuRoute output missing → FileNotFoundError per
combination → NaN row → aggregate "No valid results" warning → ✓ Complete.
Two fixes pinned here:

1. ``_load_streamflow_simulations`` prefers mizuRoute if configured and
   present, otherwise falls back to SUMMA's native averageRoutedRunoff —
   so no-routing configs still produce metrics.
2. ``BaseStructureEnsembleAnalyzer.analyze_results`` raises when every
   combination produced NaN, so the outer workflow step isn't marked
   complete on zero output.
"""

import csv
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = [pytest.mark.unit, pytest.mark.quick]


@pytest.fixture
def summa_analyzer(tmp_path, monkeypatch):
    """Fixture: a SummaStructureAnalyzer wired for isolated testing.

    Several attributes (project_dir, experiment_id, time_start, etc.)
    are read-only properties derived from config in the real class;
    patch them at class level for the test duration so the fixture
    can seed them directly.
    """
    from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

    experiment_id = "exp"
    for attr, val in (
        ('project_dir', tmp_path),
        ('experiment_id', experiment_id),
        ('domain_name', 'test_domain'),
        ('time_start', '2015-01-01'),
        ('project_observations_dir', tmp_path / 'obs'),
    ):
        monkeypatch.setattr(SummaStructureAnalyzer, attr, val, raising=False)

    analyzer = SummaStructureAnalyzer.__new__(SummaStructureAnalyzer)
    analyzer.config = MagicMock()
    analyzer.logger = logging.getLogger("test_routing_reader")
    analyzer.reporting_manager = MagicMock()
    analyzer._mizuroute_runner = None
    analyzer._resolved = {}
    analyzer._resolve = lambda typed, key=None, default=None: analyzer._resolved.get(key, default)
    return analyzer


def _write_summa_native_nc(path: Path, area_m2: float = 1_000_000.0):
    times = pd.date_range("2015-01-01", periods=6, freq="h")
    ds = xr.Dataset(
        data_vars=dict(
            averageRoutedRunoff=(("time", "hru"), np.full((len(times), 1), 1e-6, dtype="float32")),
            HRUarea=(("hru",), np.array([area_m2], dtype="float32")),
        ),
        coords=dict(time=times, hru=np.array([1])),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_native_summa_fallback_used_when_mizuroute_missing(summa_analyzer, tmp_path):
    """ROUTING_MODEL=none should flow to the native-SUMMA reader —
    not raise FileNotFoundError."""
    analyzer = summa_analyzer
    analyzer._resolved['ROUTING_MODEL'] = 'none'
    _write_summa_native_nc(
        tmp_path / 'simulations' / analyzer.experiment_id / 'SUMMA' / f"{analyzer.experiment_id}_timestep.nc"
    )

    series = analyzer._load_streamflow_simulations()
    assert len(series) == 6
    # 1e-6 m/s × 1e6 m² = 1.0 m³/s
    assert float(series.iloc[0]) == pytest.approx(1.0, rel=1e-3)


def test_mizuroute_preferred_when_configured_and_present(summa_analyzer, tmp_path):
    """With ROUTING_MODEL=mizuRoute and the file on disk we should read
    mizuRoute, not the SUMMA native fallback."""
    analyzer = summa_analyzer
    analyzer._resolved['ROUTING_MODEL'] = 'mizuRoute'
    analyzer._resolved['SIM_REACH_ID'] = 1

    mizu_path = analyzer._resolve_mizuroute_output_path()
    mizu_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2015-01-01", periods=4, freq="h")
    xr.Dataset(
        data_vars=dict(
            reachID=(("seg",), np.array([1])),
            IRFroutedRunoff=(("time", "seg"), np.full((len(times), 1), 2.5, dtype="float32")),
        ),
        coords=dict(time=times),
    ).to_netcdf(mizu_path)

    series = analyzer._load_streamflow_simulations()
    assert len(series) == 4
    assert float(series.iloc[0]) == pytest.approx(2.5)


def test_mizuroute_configured_but_missing_falls_back_to_native(summa_analyzer, tmp_path, caplog):
    """If the user sets ROUTING_MODEL=mizuRoute but the run didn't
    actually produce a mizuRoute file, we log a warning and fall back
    to native SUMMA — don't crash."""
    analyzer = summa_analyzer
    analyzer._resolved['ROUTING_MODEL'] = 'mizuRoute'
    _write_summa_native_nc(
        tmp_path / 'simulations' / analyzer.experiment_id / 'SUMMA' / f"{analyzer.experiment_id}_timestep.nc"
    )

    caplog.set_level(logging.WARNING)
    series = analyzer._load_streamflow_simulations()
    assert len(series) == 6
    assert any("falling back to SUMMA native" in r.getMessage() for r in caplog.records)


def test_neither_output_present_raises_named_error(summa_analyzer, tmp_path):
    """When both mizuRoute and SUMMA outputs are missing, raise a named
    FileNotFoundError that tells the user what to check — not a silent
    NaN row."""
    analyzer = summa_analyzer
    analyzer._resolved['ROUTING_MODEL'] = 'none'
    with pytest.raises(FileNotFoundError) as excinfo:
        analyzer._load_streamflow_simulations()
    msg = str(excinfo.value)
    assert 'mizuRoute' in msg
    assert 'averageRoutedRunoff' in msg or 'SUMMA' in msg


def test_analyze_results_raises_when_every_combination_failed(summa_analyzer, tmp_path):
    """If every combination has NaN metrics, analyze_results raises so
    the workflow step is marked FAILED instead of ✓ Complete with no
    output (SH's silent-success report). Reuse the SUMMA fixture since
    its concrete class is fully instantiable."""
    analyzer = summa_analyzer
    analyzer.decision_options = {'opt1': ['A', 'B']}
    analyzer.master_file = tmp_path / 'master.csv'

    master = analyzer.master_file
    with open(master, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Iteration', 'opt1', 'kge', 'kgep', 'nse', 'mae', 'rmse'])
        w.writerow([1, 'A', '', '', '', '', ''])
        w.writerow([2, 'B', '', '', '', '', ''])

    with pytest.raises(RuntimeError, match="all 2 decision combinations"):
        analyzer.analyze_results(master)
