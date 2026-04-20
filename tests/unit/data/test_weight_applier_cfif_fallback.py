# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Defensive standardisation in the EASYMORE weight applier.

NB reported that 03_forcing_ensemble produced a remap output containing
raw RDRS / NEX-GDDP variable names ('orog', 'tas', 'pr', 'huss', ...)
that SUMMA's forcing processor could not consume. Root cause: somewhere
upstream the per-handler ``process_dataset`` standardisation was
bypassed, so EASYMORE remapped raw-named variables and propagated them.

The fix in weight_applier._maybe_rename_with_handler is a defensive
guard: when EASYMORE's input has no CFIF or legacy SUMMA names, fall
back to the active dataset handler's rename map and materialise a
CFIF-renamed copy in-place before remapping. This test pins that
behaviour on a synthetic raw-RDRS file.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from symfluence.data.preprocessing.dataset_handlers.rdrs_utils import RDRSHandler
from symfluence.data.preprocessing.resampling.weight_applier import (
    RemappingWeightApplier,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _write_raw_rdrs_file(path: Path) -> None:
    """Create a tiny NetCDF with native RDRS v2.1 variable names."""
    times = pd.date_range("2015-01-01", periods=3, freq="h")
    ds = xr.Dataset(
        data_vars={
            "RDRS_v2.1_P_TT_1.5m": (("time",), [10.0, 11.0, 12.0]),    # Celsius
            "RDRS_v2.1_P_P0_SFC":  (("time",), [1013.0, 1012.0, 1011.0]),  # mb
            "RDRS_v2.1_A_PR0_SFC": (("time",), [1.0, 0.0, 2.0]),       # mm/hr
            "RDRS_v2.1_P_HU_09975": (("time",), [0.005, 0.006, 0.005]),  # spec hum
            "RDRS_v2.1_P_UVC_10m": (("time",), [3.0, 4.0, 5.0]),       # knots
            "RDRS_v2.1_P_FB_SFC":  (("time",), [200.0, 250.0, 300.0]),
            "RDRS_v2.1_P_FI_SFC":  (("time",), [300.0, 320.0, 340.0]),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)


def test_weight_applier_falls_back_to_handler_rename(tmp_path):
    """File with raw RDRS names must be standardised in-place to CFIF
    so EASYMORE produces a CFIF-named output. Without this guard the
    file would be rejected with 'No forcing variables found'."""
    raw_file = tmp_path / "test_RDRS_raw.nc"
    _write_raw_rdrs_file(raw_file)

    # Sanity check: the file truly has no CFIF / SUMMA-legacy names yet
    with xr.open_dataset(raw_file, engine="h5netcdf") as ds_pre:
        assert "air_temperature" not in ds_pre
        assert "airtemp" not in ds_pre
        assert "RDRS_v2.1_P_TT_1.5m" in ds_pre

    handler = RDRSHandler(
        {"DOMAIN_NAME": "test", "FORCING_DATASET": "RDRS"},
        logging.getLogger("test_handler"),
        tmp_path,
    )
    applier = RemappingWeightApplier(
        config={"DOMAIN_NAME": "test"},
        project_dir=tmp_path,
        output_dir=tmp_path,
        dataset_handler=handler,
        logger=logging.getLogger("test_applier"),
    )

    detected = applier._detect_file_variables(raw_file, worker_str="")
    # Must return a non-empty CFIF-named list — fallback succeeded
    assert detected, (
        "Expected the defensive fallback to materialise a CFIF-renamed copy "
        "and return CFIF variable names; got an empty list."
    )
    expected_subset = {
        "air_temperature",
        "surface_air_pressure",
        "precipitation_flux",
    }
    assert expected_subset.issubset(set(detected)), (
        f"Detected variables {detected} missing expected CFIF names {expected_subset}"
    )

    # Confirm the file on disk was rewritten with CFIF names
    with xr.open_dataset(raw_file, engine="h5netcdf") as ds_post:
        assert "air_temperature" in ds_post
        assert "surface_air_pressure" in ds_post
        assert "precipitation_flux" in ds_post
        # Original raw names should be gone
        assert "RDRS_v2.1_P_TT_1.5m" not in ds_post


def test_weight_applier_passes_through_already_cfif_files(tmp_path):
    """When the file already has CFIF names, the fallback must NOT fire
    (no spurious rewrite, no warning)."""
    times = pd.date_range("2015-01-01", periods=3, freq="h")
    pre = xr.Dataset(
        data_vars={
            "air_temperature":      (("time",), [283.0, 284.0, 285.0]),
            "surface_air_pressure": (("time",), [101300.0, 101200.0, 101100.0]),
            "precipitation_flux":   (("time",), [0.0003, 0.0, 0.0006]),
        },
        coords={"time": times},
    )
    cfif_file = tmp_path / "test_cfif.nc"
    pre.to_netcdf(cfif_file)
    pre_mtime = cfif_file.stat().st_mtime

    handler = RDRSHandler(
        {"DOMAIN_NAME": "test", "FORCING_DATASET": "RDRS"},
        logging.getLogger("test_handler"),
        tmp_path,
    )
    applier = RemappingWeightApplier(
        config={"DOMAIN_NAME": "test"},
        project_dir=tmp_path,
        output_dir=tmp_path,
        dataset_handler=handler,
        logger=logging.getLogger("test_applier"),
    )

    detected = applier._detect_file_variables(cfif_file, worker_str="")
    assert "air_temperature" in detected
    assert "surface_air_pressure" in detected
    assert "precipitation_flux" in detected
    # File mtime unchanged — no rewrite happened
    assert cfif_file.stat().st_mtime == pre_mtime, (
        "Fallback wrongly rewrote a file that already had CFIF names"
    )
