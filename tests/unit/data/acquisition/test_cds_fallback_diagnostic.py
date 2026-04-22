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

from symfluence.data.acquisition.handlers.era5 import ERA5Acquirer

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
