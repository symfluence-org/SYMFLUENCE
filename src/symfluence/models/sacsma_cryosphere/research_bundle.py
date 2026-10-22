# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit adapter for source-pinned, unfrozen V4 historical research bundles.

This interface does not widen the native V1/V2 replay schema or certify a
scientific promotion. Frozen research variants remain unsupported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .replay import ReplayValidationError

MODEL_ID = "SACSMA_CRYOSPHERE_GRID_MOMENT_PATCH_RESEARCH_V4"
REQUIRED_FILES = {
    "config.json", "selected.json", "forcing.npz", "revised_geometry.json",
    "revised_model.py", "soil_kernel.py", "snow_kernel.py", "frozen_soil.py",
    "glacier_temperature_correction.json", "land_temperature_correction.json",
    "daily.csv", "observations.csv", "replay_v4.py",
    "vendor/jsacsma/parameters.py", "vendor/jsnow17/model.py",
    "vendor/jsnow17/parameters.py", "vendor/jsnow17/depletion.py",
}


def manifest_fingerprint(manifest: dict) -> str:
    """Bind a successful replay report to its complete input/source manifest."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_research_candidate(
    bundle: str | Path,
    observations: pd.Series,
    start: str = "2012-10-01",
    end: str = "2023-09-30",
) -> tuple[pd.Series, dict]:
    """Verify provenance, numerical replay evidence and exact daily alignment.

    Replay the bundle separately before calling this reader. Acceptance here is
    an output-contract check, not independent hydrological or physical validation.
    """
    root = Path(bundle).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model_id") != MODEL_ID:
        raise ReplayValidationError("Unsupported research model; frozen V3/V4/V5 variants are not accepted")
    hashes = manifest.get("artifacts_sha256", {})
    if not REQUIRED_FILES.issubset(hashes):
        raise ReplayValidationError("Research bundle is missing required source or input fingerprints")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ReplayValidationError("Research artifacts must remain inside their portable bundle")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ReplayValidationError(f"Research artifact fingerprint differs: {name}")
    config = json.loads((root / "config.json").read_text())
    selected = json.loads((root / "selected.json").read_text())
    if config.get("frozen_soil") is not False or selected.get("frozen_soil") is not False:
        raise ReplayValidationError("This adapter accepts only the explicitly unfrozen V4 variant")
    report = json.loads((root / "replay_verification.json").read_text())
    if report.get("manifest_fingerprint") != manifest_fingerprint(manifest):
        raise ReplayValidationError("Replay verification is stale for this manifest")
    if report.get("model_id") != MODEL_ID or report.get("discharge_replay_match") is not True:
        raise ReplayValidationError("Successful source-complete replay evidence is required")
    for name, limit in [("max_cumulative_water_mm", 1e-6), ("max_layer_HRU_substep_heat_J_m2", 1.)]:
        value = report.get(name, float("nan"))
        if not np.isfinite(value) or value < 0 or value > limit:
            raise ReplayValidationError(f"Numerical replay gate failed: {name}")
    minimum_ice = report.get("minimum_terminal_glacier_ice_mm", float("nan"))
    if report.get("all_finite") is not True or not np.isfinite(minimum_ice) or minimum_ice < 0:
        raise ReplayValidationError("Finite replay and nonnegative terminal ice evidence are required")
    frame = pd.read_csv(root / "daily.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    if frame.shape[1] != 1:
        raise ReplayValidationError("Daily research discharge must contain exactly one series")
    q = frame.iloc[:, 0]
    if not q.index.is_unique or not q.index.is_monotonic_increasing:
        raise ReplayValidationError("Daily research timestamps are duplicated or unordered")
    dates = pd.date_range(start, end)
    q = q.reindex(dates)
    saved = pd.read_csv(root / "observations.csv", index_col=0, parse_dates=True,
                        float_precision="round_trip").iloc[:, 0].reindex(dates)
    target = observations.reindex(dates)
    if not np.isfinite(q).all() or (q < 0).any():
        raise ReplayValidationError("Research discharge is incomplete, nonfinite or negative")
    if not np.isfinite(target).all() or not np.array_equal(saved.values, target.values):
        raise ReplayValidationError("Research observations do not exactly match ensemble observations")
    q.name = "SACSMA"
    return q, {
        "model_id": MODEL_ID, "adapter": "research_bundle_unfrozen_v4",
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "scored_days": len(q), "family_votes": 1,
        "assessment_role": "historical_development_validation",
        "scientific_promotion_certified": False,
    }


def load_selected_research_candidate(selection: dict, observations: pd.Series) -> tuple[pd.Series, dict]:
    """Read an explicit, fingerprint-pinned ensemble selection for this family."""
    if selection.get("adapter") != "research_bundle_unfrozen_v4" or selection.get("model") != "SACSMA":
        raise ReplayValidationError("An explicit unfrozen research adapter and SACSMA family are required")
    q, metadata = load_research_candidate(selection["bundle"], observations)
    if selection.get("manifest_fingerprint") != metadata["manifest_fingerprint"]:
        raise ReplayValidationError("Selected research manifest fingerprint differs")
    return q, metadata
