# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FUSE's calibration observation loader reads the model-ready store.

``metrics_calculation.load_observations`` now delegates to the shared
``StreamflowMetrics`` loader, so FUSE (and the other models that went through
their own CSV reads) pick up the model-ready observations store when present and
fall back to the legacy preprocessed CSV otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("xarray")

from symfluence.data.model_ready.observations_builder import ObservationsNetCDFBuilder
from symfluence.models.fuse.calibration.metrics_calculation import load_observations

DOMAIN = "test"


def _write_csv(project_dir, values):
    path = (
        project_dir / "observations" / "streamflow" / "preprocessed"
        / f"{DOMAIN}_streamflow_processed.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
    pd.DataFrame({"datetime": dates, "discharge_cms": values}).to_csv(path, index=False)


def test_load_observations_prefers_store(tmp_path):
    original = np.arange(1.0, 31.0)
    _write_csv(tmp_path, original)
    built = ObservationsNetCDFBuilder(project_dir=tmp_path, domain_name=DOMAIN).build()
    assert built is not None

    # Diverge the CSV after building the store; the store value must win.
    _write_csv(tmp_path, original + 1000.0)

    series = load_observations({"DOMAIN_NAME": DOMAIN}, tmp_path)

    assert isinstance(series, pd.Series)
    assert series.max() < 100  # store values, not the +1000 CSV
    np.testing.assert_allclose(np.sort(series.dropna().values), np.sort(original), rtol=1e-5)


def test_load_observations_falls_back_to_csv(tmp_path):
    values = np.arange(1.0, 21.0)
    _write_csv(tmp_path, values)  # no store built

    series = load_observations({"DOMAIN_NAME": DOMAIN}, tmp_path)

    assert isinstance(series, pd.Series)
    np.testing.assert_allclose(np.sort(series.dropna().values), np.sort(values), rtol=1e-5)
