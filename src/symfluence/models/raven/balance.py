"""Reject incomplete runs, broken balances, or unbounded glacier ice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError


def check(output, settings):
    out = Path(output)
    meta = json.loads((Path(settings) / "geometry.json").read_text())
    f = pd.read_csv(out / "WatershedStorage.csv")
    f.columns = f.columns.str.strip()
    if not np.isfinite(f["MB Error [mm]"]).all() or f["MB Error [mm]"].abs().max() > 0.05:
        raise ValidationError("Raven mass-balance error exceeds 0.05 mm")
    expected = sum(h["area_km2"] * meta["ice_mm"] for h in meta["hrus"] if h["ice"]) / sum(
        h["area_km2"] for h in meta["hrus"]
    )
    initial = float(f["Glacier Ice [mm]"].iloc[0])
    if not np.isclose(initial, expected, atol=0.1):
        raise ValidationError("Finite glacier initialization was not honored; patched Raven is required")
    ice = (
        pd.read_csv(out / "GLACIER_ICE_Daily_Average_ByHRU.csv", skiprows=2, header=None)
        .iloc[:, 2 : 2 + len(meta["hrus"])]
        .to_numpy(float)
    )
    if not np.isfinite(ice).all() or np.min(ice) < -1e-6:
        raise ValidationError("Invalid glacier storage")
    return dict(
        days=len(f) - 1,
        max_cumulative_balance_error_mm=float(f["MB Error [mm]"].abs().max()),
        initial_basin_ice_mm=initial,
        minimum_hru_ice_mm=float(ice.min()),
    )
