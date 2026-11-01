# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for StreamflowMetrics reading the model-ready observations store.

Validates the additive "strategy-0" added to ``StreamflowMetrics.load_observations``:
the model-ready store is preferred when present, the legacy preprocessed CSV remains
the fallback, and a corrupt/absent store never raises. The key guarantee is that the
store path yields the *same* series as the CSV path, so calibration objectives do not
silently drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip('netCDF4')  # noqa: N816
pytest.importorskip('xarray')

from symfluence.core.metrics import streamflow_metrics as sm_module
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics
from symfluence.data.model_ready.observations_builder import ObservationsNetCDFBuilder

DOMAIN = 'test'


def _csv_path(project_dir: Path) -> Path:
    return (
        project_dir / 'observations' / 'streamflow' / 'preprocessed'
        / f'{DOMAIN}_streamflow_processed.csv'
    )


def _write_streamflow_csv(project_dir: Path, values: np.ndarray) -> Path:
    path = _csv_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range('2020-01-01', periods=len(values), freq='D')
    pd.DataFrame({'datetime': dates, 'discharge_cms': values}).to_csv(path, index=False)
    return path


def _build_store(project_dir: Path) -> Path:
    result = ObservationsNetCDFBuilder(project_dir=project_dir, domain_name=DOMAIN).build()
    assert result is not None and result.exists()
    # Sanity: builder wrote to the path load_observations looks for.
    assert result == project_dir / 'data' / 'model_ready' / 'observations' / f'{DOMAIN}_observations.nc'
    return result


def test_reads_store_when_present_and_ignores_diverged_csv(tmp_path):
    """Store is preferred over the legacy CSV when both exist."""
    original = np.arange(1, 41, dtype=float)
    _write_streamflow_csv(tmp_path, original)
    _build_store(tmp_path)

    # Diverge the CSV after the store was built; the store value must win.
    _write_streamflow_csv(tmp_path, original + 1000.0)

    values, index = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    assert values is not None and index is not None
    np.testing.assert_allclose(np.sort(values), np.sort(original), rtol=1e-5)
    assert values.max() < 100  # not the +1000 CSV


def test_store_equals_csv_for_same_data(tmp_path):
    """No objective drift: store result matches the CSV-derived series."""
    values = np.linspace(2.0, 50.0, 40)
    _write_streamflow_csv(tmp_path, values)

    # CSV-only result (no store yet)
    csv_values, csv_index = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    # Build store and read again
    _build_store(tmp_path)
    store_values, store_index = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    assert csv_values is not None and store_values is not None
    np.testing.assert_allclose(
        np.sort(store_values), np.sort(csv_values), rtol=1e-5
    )
    assert len(store_index) == len(csv_index)


def test_store_absent_falls_back_to_csv(tmp_path):
    values = np.arange(1, 31, dtype=float)
    _write_streamflow_csv(tmp_path, values)
    # No store built.

    out_values, _ = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    assert out_values is not None
    np.testing.assert_allclose(np.sort(out_values), np.sort(values), rtol=1e-5)


def test_corrupt_store_falls_back_to_csv_without_raising(tmp_path):
    values = np.arange(1, 31, dtype=float)
    _write_streamflow_csv(tmp_path, values)

    # An empty NetCDF with no 'streamflow' group — the store read must miss
    # and fall through to the CSV rather than raising.
    store = tmp_path / 'data' / 'model_ready' / 'observations' / f'{DOMAIN}_observations.nc'
    store.parent.mkdir(parents=True, exist_ok=True)
    netCDF4.Dataset(str(store), 'w', format='NETCDF4').close()

    out_values, _ = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    assert out_values is not None
    np.testing.assert_allclose(np.sort(out_values), np.sort(values), rtol=1e-5)


def test_gapped_data_matches_csv_and_no_sentinel(tmp_path):
    """Gap days (written as the -9999.0 fill) decode to NaN exactly like the CSV
    path — the sentinel never leaks, and store/CSV outputs are identical."""
    values = np.arange(1, 41, dtype=float)
    values[5] = np.nan
    values[12] = np.nan
    _write_streamflow_csv(tmp_path, values)

    # CSV-only baseline (legacy behaviour: NaN at the gap days after resample).
    csv_values, _ = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    _build_store(tmp_path)
    store_values, _ = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)

    assert store_values is not None and csv_values is not None
    # The -9999.0 fill must never surface as a real value.
    assert not np.any(store_values == -9999.0)
    # Store reproduces the CSV series exactly, NaN positions included.
    np.testing.assert_allclose(store_values, csv_values, rtol=1e-5, equal_nan=True)


def test_module_and_instance_loaders_agree(tmp_path):
    values = np.linspace(1.0, 20.0, 25)
    _write_streamflow_csv(tmp_path, values)
    _build_store(tmp_path)

    inst_values, _ = StreamflowMetrics().load_observations({}, tmp_path, DOMAIN)
    mod_values, _ = sm_module.load_observations({}, tmp_path, DOMAIN)

    assert inst_values is not None and mod_values is not None
    np.testing.assert_array_equal(inst_values, mod_values)
