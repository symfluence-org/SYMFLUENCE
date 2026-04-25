# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""End-to-end test: large sample config → Lamah-ICE streamflow acquisition.

Verifies that the 08_large_sample config files (117 FUSE + GR4J configs
for 59 Icelandic catchments) correctly resolve ``domain_id`` through
the typed config system and produce processed streamflow output.

Regression: ``domain_id`` was placed under ``data.lamah_ice`` (a Pydantic
extra dict) while ``LAMAHICEConfig`` lived under ``evaluation.lamah_ice``
and had no ``domain_id`` field, so every large sample config silently
failed with "LAMAH_ICE acquisition requires a basin identifier".
"""

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from symfluence.core.config.coercion import coerce_config
from symfluence.data.observation.handlers.lamah_ice import (
    LamahIceStreamflowHandler,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]

SAMPLE_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "paper_case_studies"
    / "configs"
    / "configs_nested"
    / "08_large_sample"
    / "gr4j_v2"
    / "config_lamahice_105_GR_v2.yaml"
)


def _write_fake_lamah_tree(root: Path, station_id: str) -> Path:
    daily = root / "D_gauges" / "2_timeseries" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    fake = daily / f"ID_{station_id}.csv"
    pd.DataFrame({
        "YYYY": [2005, 2005, 2005, 2005, 2005],
        "MM":   [6,    6,    6,    6,    6],
        "DD":   [1,    2,    3,    4,    5],
        "qobs": [23.1, 18.4, 15.9, 20.7, 22.3],
        "qc_flag": [40, 40, 40, 40, 40],
    }).to_csv(fake, sep=";", index=False)
    return fake


@pytest.mark.skipif(not SAMPLE_CONFIG.exists(),
                    reason="08_large_sample configs not present")
def test_large_sample_config_domain_id_resolves(tmp_path):
    """domain_id in evaluation.lamah_ice must reach the handler."""
    with open(SAMPLE_CONFIG) as f:
        raw = yaml.safe_load(f)

    cfg = coerce_config(raw)
    assert cfg.evaluation.lamah_ice.domain_id is not None, (
        "domain_id not populated in evaluation.lamah_ice — "
        "check that the config YAML places it under evaluation.lamah_ice"
    )
    assert str(cfg.evaluation.lamah_ice.domain_id) == "105"


@pytest.mark.skipif(not SAMPLE_CONFIG.exists(),
                    reason="08_large_sample configs not present")
def test_large_sample_acquire_and_process(tmp_path):
    """Full acquire→process pipeline with a real large sample config."""
    with open(SAMPLE_CONFIG) as f:
        raw = yaml.safe_load(f)

    lamah_root = tmp_path / "lamah_ice"
    _write_fake_lamah_tree(lamah_root, "105")

    raw["evaluation"]["lamah_ice"]["path"] = str(lamah_root)
    raw["system"]["data_dir"] = str(tmp_path)

    logger = logging.getLogger("test_lamah_ice_e2e")
    handler = LamahIceStreamflowHandler(raw, logger)

    proj_obs = handler.project_observations_dir
    proj_obs.mkdir(parents=True, exist_ok=True)

    raw_path = handler.acquire()
    assert raw_path.exists(), f"acquire() output missing: {raw_path}"
    assert "105" in raw_path.name

    processed_path = handler.process(raw_path)
    assert processed_path.exists(), f"process() output missing: {processed_path}"

    df = pd.read_csv(processed_path, parse_dates=["datetime"])
    assert "discharge_cms" in df.columns
    assert len(df) > 0
    assert df["discharge_cms"].notna().all()


def test_backward_compat_data_lamah_ice_domain_id(tmp_path):
    """Old configs with domain_id under data.lamah_ice (extra dict) still work.

    Legacy flat configs place LAMAH_ICE_DOMAIN_ID at the top level, but
    some hand-crafted nested configs put it under data.lamah_ice — both
    paths must resolve.
    """
    lamah_root = tmp_path / "lamah_ice"
    _write_fake_lamah_tree(lamah_root, "42")

    raw = {
        "system": {"data_dir": str(tmp_path)},
        "domain": {
            "name": "42",
            "time_start": "2005-01-01",
            "time_end": "2005-12-31",
        },
        "data": {
            "streamflow_data_provider": "LAMAH_ICE",
            "lamah_ice": {"domain_id": 42},
        },
        "evaluation": {
            "lamah_ice": {
                "path": str(lamah_root),
                "download": False,
            },
        },
        "forcing": {"time_step_size": 86400},
        "LAMAH_ICE_PATH": str(lamah_root),
    }

    logger = logging.getLogger("test_lamah_ice_compat")
    handler = LamahIceStreamflowHandler(raw, logger)

    proj_obs = handler.project_observations_dir
    proj_obs.mkdir(parents=True, exist_ok=True)

    raw_path = handler.acquire()
    assert raw_path.exists()
    assert "42" in raw_path.name

    processed_path = handler.process(raw_path)
    assert processed_path.exists()

    df = pd.read_csv(processed_path, parse_dates=["datetime"])
    assert len(df) > 0
