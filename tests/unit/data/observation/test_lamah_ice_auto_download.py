# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for the LaMAH-Ice HydroShare auto-downloader.

Nico (09_large_domain) needed the LaMAH-Ice daily streamflow dataset
to run his Iceland calibration. Rather than distribute the 2 GB
bundle out of band, the LAMAH_ICE_STREAMFLOW handler now fetches
``lamah_ice.zip`` from HydroShare on first use and extracts only
``D_gauges/`` (~57 MB of the 2 GB decompressed).

These tests use a locally-generated fake zip to avoid hitting
HydroShare from CI — the same code path that would be taken in
production.
"""

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.data.observation.handlers.lamah_ice import (
    _LAMAH_ICE_DAILY_ZIP_NAME,
    ensure_lamah_ice_streamflow,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _fake_hydroshare_zip() -> bytes:
    """Build an in-memory zip matching the HydroShare layout with a
    couple of synthetic gauge CSVs and a Gauge_attributes.csv."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(
            "lamah_ice/D_gauges/1_attributes/Gauge_attributes.csv",
            "id;lat;lon\n1;64.0;-22.0\n3;65.3;-18.9\n"
        )
        for station in (1, 3, 11):
            zf.writestr(
                f"lamah_ice/D_gauges/2_timeseries/daily/ID_{station}.csv",
                "YYYY;MM;DD;qobs;qc_flag\n2015;1;1;5.0;40.0\n2015;1;2;4.8;40.0\n"
            )
        # Adjacent subtree we should NOT extract.
        zf.writestr(
            "lamah_ice/A_basins_total_upstrm/3_shapefiles/Basins_A.shp",
            b"\x00" * 128,
        )
    return buf.getvalue()


class _FakeSession:
    def __init__(self, body: bytes):
        self._body = body
    def get(self, url, **_):
        return _FakeResponse(self._body)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"content-length": str(len(body))}
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size=1):
        yield self._body


def test_auto_download_populates_d_gauges(tmp_path, caplog):
    """Given an empty LAMAH_ICE_PATH, ensure_lamah_ice_streamflow
    downloads the HydroShare archive, extracts only D_gauges/, and
    leaves the expected ID_*.csv files on disk."""
    lamah = tmp_path / "lamah_ice"
    caplog.set_level(logging.INFO)
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.create_robust_session',
        return_value=_FakeSession(_fake_hydroshare_zip()),
    ):
        result = ensure_lamah_ice_streamflow(lamah, logging.getLogger("t"))
    assert result == lamah.resolve()
    daily = lamah / "D_gauges" / "2_timeseries" / "daily"
    assert daily.exists()
    assert (daily / "ID_1.csv").exists()
    assert (daily / "ID_3.csv").exists()
    assert (lamah / "D_gauges" / "1_attributes" / "Gauge_attributes.csv").exists()


def test_auto_download_skips_unneeded_subtrees(tmp_path):
    """Only D_gauges/ should land on disk. The ~2 GB of basin
    polygons / stream network data in the archive is irrelevant to
    streamflow calibration and shouldn't bloat the user's data dir."""
    lamah = tmp_path / "lamah_ice"
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.create_robust_session',
        return_value=_FakeSession(_fake_hydroshare_zip()),
    ):
        ensure_lamah_ice_streamflow(lamah, logging.getLogger("t"))
    assert not (lamah / "A_basins_total_upstrm").exists(), \
        "adjacent subtrees must not be extracted"


def test_auto_download_idempotent_when_already_present(tmp_path):
    """If D_gauges is already populated, the downloader must not
    fetch again (important for re-runs + offline use)."""
    lamah = tmp_path / "lamah_ice"
    daily = lamah / "D_gauges" / "2_timeseries" / "daily"
    daily.mkdir(parents=True)
    (daily / "ID_1.csv").write_text("pre-existing\n")

    with patch(
        'symfluence.data.observation.handlers.lamah_ice.create_robust_session',
        side_effect=AssertionError("downloader should not run when data is present"),
    ):
        ensure_lamah_ice_streamflow(lamah, logging.getLogger("t"))
    assert (daily / "ID_1.csv").read_text() == "pre-existing\n"


def test_cache_cleanup_removes_zip_after_extraction(tmp_path):
    """Post-extraction the 636 MB zip gets deleted so the cache dir
    doesn't balloon for every run."""
    lamah = tmp_path / "lamah_ice"
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.create_robust_session',
        return_value=_FakeSession(_fake_hydroshare_zip()),
    ):
        ensure_lamah_ice_streamflow(lamah, logging.getLogger("t"))
    cache_zip = lamah / ".cache" / _LAMAH_ICE_DAILY_ZIP_NAME
    assert not cache_zip.exists()


def test_handler_auto_downloads_when_file_missing(tmp_path, caplog):
    """The LamahIceStreamflowHandler.acquire path must call the
    downloader and then find the requested ID_<n>.csv."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from symfluence.data.observation.handlers.lamah_ice import (
        LamahIceStreamflowHandler,
    )

    handler = LamahIceStreamflowHandler.__new__(LamahIceStreamflowHandler)
    # Plain namespaces so typed-config lookups raise AttributeError
    # cleanly and the _get fallback hits the dict_key branch.
    handler.config = SimpleNamespace()
    handler.logger = logging.getLogger("t_handler")
    proj = tmp_path / "project"
    (proj / "data" / "observations").mkdir(parents=True, exist_ok=True)
    handler.project_dir = proj
    handler.domain_name = "icelandic_domain"
    handler.config_dict = {
        'LAMAH_ICE_DOMAIN_ID': 1,
        'LAMAH_ICE_PATH': str(tmp_path / "lamah_cache"),
    }

    def _get(getter, dict_key=None, default=None):
        try:
            v = getter()
            if v is not None:
                return v
        except AttributeError:
            pass
        return handler.config_dict.get(dict_key, default)
    handler._get_config_value = _get

    with patch(
        'symfluence.data.observation.handlers.lamah_ice.create_robust_session',
        return_value=_FakeSession(_fake_hydroshare_zip()),
    ):
        out = handler.acquire()
    assert out.exists()
    assert "1" in out.name
