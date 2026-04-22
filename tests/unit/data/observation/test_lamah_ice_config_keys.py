# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for the LAMAH-ICE handler's basin-id config key.

Co-authors reported silent failures across all 08_large_sample
configs (51 FUSE + 66 GR4J = 117 files). Root cause: every config
specifies ``LAMAH_ICE_DOMAIN_ID: <n>`` (matching LaMAH-ICE's own
``D_gauges/.../ID_<n>.csv`` naming), but the handler only looked
up ``STATION_ID`` and raised a misleading ValueError when the
lookup missed.

Pin the fix: the handler accepts ``LAMAH_ICE_DOMAIN_ID`` as the
primary key, falls back to ``STATION_ID`` for backwards compat,
and raises a message that names both keys when neither is set.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from symfluence.data.observation.handlers.lamah_ice import (
    LamahIceStreamflowHandler,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_handler(config_dict, tmp_path):
    """Build a handler backed by a dict config (mimicking the flat
    YAML paper configs) and a tmp-path project dir."""
    config = MagicMock()
    # Force typed lookups to fail so the dict_key path is exercised —
    # that's the path the paper configs go through.
    config.evaluation = None
    config.data = None
    config.get = config_dict.get
    config.__getitem__ = lambda s, k: config_dict[k]
    config.__contains__ = lambda s, k: k in config_dict

    handler = LamahIceStreamflowHandler.__new__(LamahIceStreamflowHandler)
    handler.config = config
    handler.config_dict = config_dict
    handler.logger = MagicMock()
    # project_observations_dir is a read-only property derived from
    # project_dir — assigning to it directly raises AttributeError.
    # Setting project_dir lets the property resolve naturally to
    # project_dir/data/observations.
    proj = tmp_path / "project"
    (proj / "data" / "observations").mkdir(parents=True, exist_ok=True)
    handler.project_dir = proj
    handler.domain_name = "test_domain"

    def _get(getter, dict_key=None, default=None):
        try:
            val = getter()
            if val is not None:
                return val
        except AttributeError:
            pass
        if dict_key and dict_key in config_dict:
            return config_dict[dict_key]
        return default

    handler._get_config_value = _get
    return handler


def _write_fake_lamah_tree(root: Path, station_id: str):
    daily = root / "D_gauges" / "2_timeseries" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    fake = daily / f"ID_{station_id}.csv"
    pd.DataFrame({
        "YYYY": [2015, 2015],
        "MM": [1, 1],
        "DD": [1, 2],
        "qobs": [10.0, 12.0],
    }).to_csv(fake, sep=";", index=False)
    return fake


def test_lamah_ice_domain_id_is_accepted(tmp_path):
    """The 08_large_sample paper configs use LAMAH_ICE_DOMAIN_ID.
    That must be the primary accepted key."""
    lamah_root = tmp_path / "lamah_ice"
    _write_fake_lamah_tree(lamah_root, "105")

    cfg = {
        "LAMAH_ICE_DOMAIN_ID": 105,
        "LAMAH_ICE_PATH": str(lamah_root),
    }
    handler = _make_handler(cfg, tmp_path)
    out = handler.acquire()
    assert out.exists()
    assert "105" in out.name


def test_station_id_still_works_as_alias(tmp_path):
    """STATION_ID was the only accepted key before this fix. Keep it
    working for generic cross-dataset configs."""
    lamah_root = tmp_path / "lamah_ice"
    _write_fake_lamah_tree(lamah_root, "13")

    cfg = {
        "STATION_ID": 13,
        "LAMAH_ICE_PATH": str(lamah_root),
    }
    handler = _make_handler(cfg, tmp_path)
    out = handler.acquire()
    assert out.exists()
    assert "13" in out.name


def test_domain_id_takes_precedence_over_station_id(tmp_path):
    """If both keys are set (e.g., after a partial config migration),
    LAMAH_ICE_DOMAIN_ID wins — it's the LaMAH-ICE-native identifier."""
    lamah_root = tmp_path / "lamah_ice"
    _write_fake_lamah_tree(lamah_root, "105")
    _write_fake_lamah_tree(lamah_root, "13")

    cfg = {
        "LAMAH_ICE_DOMAIN_ID": 105,
        "STATION_ID": 13,
        "LAMAH_ICE_PATH": str(lamah_root),
    }
    handler = _make_handler(cfg, tmp_path)
    out = handler.acquire()
    assert "105" in out.name
    assert "13" not in out.name.replace("105", "")


def test_neither_key_raises_with_both_names(tmp_path):
    """When no basin id is set, the error must name BOTH accepted
    config keys — the previous message only mentioned STATION_ID,
    which sent co-authors on a wild goose chase."""
    cfg = {"LAMAH_ICE_PATH": str(tmp_path)}
    handler = _make_handler(cfg, tmp_path)
    with pytest.raises(ValueError) as excinfo:
        handler.acquire()
    msg = str(excinfo.value)
    assert "LAMAH_ICE_DOMAIN_ID" in msg
    assert "STATION_ID" in msg
