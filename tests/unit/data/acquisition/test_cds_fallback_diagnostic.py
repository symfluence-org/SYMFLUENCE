# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Regression test for the CDS → ARCO fallback diagnostic message.

SH reported that ERA5 downloads silently fell back to ARCO when CDS was
misconfigured (old ``~/.cdsapirc`` pointing at ``/api/v2`` and/or
``cdsapi<0.7.0``), and that the ARCO fallback's own unrelated error
became the only user-visible symptom — making the real root cause
(CDS setup) invisible.

The ERA5Acquirer now catches CDS exceptions, keeps the design-intent
fallback to ARCO, and emits a structured warning naming the three most
common post-Sept-2024 CDS setup problems so a user can self-diagnose
without reading source. This test pins the content of that warning.
"""

import logging
from unittest.mock import patch

import pytest

from symfluence.data.acquisition.handlers.era5 import (
    ERA5Acquirer,
    diagnose_cds_credentials,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


class _CDSFail(RuntimeError):
    pass


def test_cds_fallback_warning_names_common_setup_failures(tmp_path, caplog):
    """When the CDS pathway raises, the subsequent warning must name
    the three most common misconfigurations so the user can fix them
    without re-tracing SYMFLUENCE internals.
    """
    from symfluence.core.config.models import SymfluenceConfig

    cfg = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        SYMFLUENCE_CODE_DIR=str(tmp_path / "code"),
        DOMAIN_NAME="cds_diag_test",
        EXPERIMENT_ID="test",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-01-02 00:00",
        FORCING_DATASET="ERA5",
        HYDROLOGICAL_MODEL="SUMMA",
        DOMAIN_DEFINITION_METHOD="lumped",
        SUB_GRID_DISCRETIZATION="GRUs",
        BOUNDING_BOX_COORDS="44.5/-87.9/44.2/-87.5",
        ERA5_USE_CDS=True,
    )
    acquirer = ERA5Acquirer(cfg, logging.getLogger("test_cds_diag"))

    # Make the CDS path fail with a generic RuntimeError and the ARCO
    # path immediately return a sentinel so the exception from CDS is
    # the only thing exercising our warning code.
    with patch(
        "symfluence.data.acquisition.handlers.era5.ERA5CDSAcquirer"
    ) as cds_cls, patch(
        "symfluence.data.acquisition.handlers.era5.ERA5ARCOAcquirer"
    ) as arco_cls:
        cds_cls.return_value.download.side_effect = _CDSFail(
            "Fake CDS error to exercise fallback"
        )
        arco_cls.return_value.download.return_value = tmp_path / "fake_arco_out"

        caplog.set_level(logging.WARNING)
        result = acquirer.download(tmp_path / "out")

    assert result == tmp_path / "fake_arco_out"

    warning_msgs = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    ]
    joined = "\n".join(warning_msgs)

    # The three concrete failure modes must each be named so the user
    # can check them in order without reading source.
    assert "/api" in joined, "warning must name the /api endpoint"
    assert "/v2" in joined, "warning must explicitly call out the /v2 issue"
    assert "cdsapi" in joined.lower() and "0.7.0" in joined, \
        "warning must state the minimum cdsapi version"
    assert "regenerate" in joined.lower() or "profile" in joined.lower(), \
        "warning must point users at key regeneration"


def _isolate_cds_env(monkeypatch, tmp_path, *, home_has_rc: bool = False):
    """Point ``HOME`` at ``tmp_path`` and clear CDSAPI_* env vars so
    each diagnostic case is independent of the developer's real CDS
    credentials."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    if not home_has_rc:
        rc = tmp_path / ".cdsapirc"
        if rc.exists():
            rc.unlink()


def test_diagnose_no_credentials_at_all(tmp_path, monkeypatch):
    """With no ``~/.cdsapirc`` and no env vars, the diagnostic should
    explain how to create the file from scratch and mention the
    gcsfs escape hatch so the user isn't stuck."""
    _isolate_cds_env(monkeypatch, tmp_path)
    msg = diagnose_cds_credentials()
    assert msg is not None
    assert "No CDS API credentials found" in msg
    assert ".cdsapirc" in msg
    assert "cds.climate.copernicus.eu/profile" in msg
    assert "gcsfs" in msg.lower()


def test_diagnose_old_api_v2_url_in_rc(tmp_path, monkeypatch):
    """A ``~/.cdsapirc`` with the pre-Sept-2024 ``/api/v2`` URL must
    be called out explicitly — this was the single most common cause
    of silent CDS failure during the 2024 migration."""
    _isolate_cds_env(monkeypatch, tmp_path, home_has_rc=True)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://cds.climate.copernicus.eu/api/v2\n"
        "key: abcd-efgh-ijkl-mnop\n"
    )
    msg = diagnose_cds_credentials()
    assert msg is not None
    assert "/api/v2" in msg
    assert "pre-September-2024" in msg


def test_diagnose_old_uid_colon_key_format(tmp_path, monkeypatch):
    """A key in the pre-Sept-2024 ``<UID>:<API_KEY>`` colon format
    must be named as the problem, with a pointer to key regeneration."""
    _isolate_cds_env(monkeypatch, tmp_path, home_has_rc=True)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://cds.climate.copernicus.eu/api\n"
        "key: 12345:abcd-efgh-ijkl-mnop\n"
    )
    msg = diagnose_cds_credentials()
    assert msg is not None
    assert "pre-September-2024" in msg
    assert "profile" in msg.lower()


def test_diagnose_missing_url_line(tmp_path, monkeypatch):
    """An rc file with a key but no url line should be named out —
    cdsapi's own error for this case is cryptic."""
    _isolate_cds_env(monkeypatch, tmp_path, home_has_rc=True)
    (tmp_path / ".cdsapirc").write_text("key: abcd-efgh-ijkl-mnop\n")
    msg = diagnose_cds_credentials()
    assert msg is not None
    assert "missing a 'url:' line" in msg


def test_diagnose_missing_key_line(tmp_path, monkeypatch):
    """An rc file with a url but no key line should be named out."""
    _isolate_cds_env(monkeypatch, tmp_path, home_has_rc=True)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://cds.climate.copernicus.eu/api\n"
    )
    msg = diagnose_cds_credentials()
    assert msg is not None
    assert "missing a 'key:' line" in msg


def test_diagnose_valid_setup_returns_none(tmp_path, monkeypatch):
    """A correctly-configured ``~/.cdsapirc`` against the
    post-migration endpoint must return ``None`` so callers proceed
    normally."""
    _isolate_cds_env(monkeypatch, tmp_path, home_has_rc=True)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://cds.climate.copernicus.eu/api\n"
        "key: abcd-efgh-ijkl-mnop\n"
    )
    assert diagnose_cds_credentials() is None


def test_diagnose_old_env_url(tmp_path, monkeypatch):
    """An env-var-only setup with the old ``/api/v2`` URL must be
    named — some CI environments inject ``CDSAPI_URL`` without
    touching ``~``."""
    _isolate_cds_env(monkeypatch, tmp_path)
    monkeypatch.setenv("CDSAPI_KEY", "abcd-efgh-ijkl-mnop")
    monkeypatch.setenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api/v2")
    msg = diagnose_cds_credentials()
    assert msg is not None
    assert "/api/v2" in msg
    assert "CDSAPI_URL" in msg


def test_make_cds_client_preflight_raises_with_diagnostic(tmp_path, monkeypatch):
    """The ``_make_cds_client`` helper must refuse to instantiate
    when :func:`diagnose_cds_credentials` returns a problem, and the
    raised error must carry the diagnostic so the user sees it
    without a stack-trace hunt."""
    from symfluence.data.acquisition.handlers import cds_datasets

    _isolate_cds_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        cds_datasets._make_cds_client()
    assert "CDS API client" in str(excinfo.value)
    assert "No CDS API credentials" in str(excinfo.value)
