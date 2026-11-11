# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCRGLOBWB Hamon PET goes through the shared helper (regression).

The private copy fed kPa into the hPa form of Lu et al. (2005) and returned a
tenth of the reference PET (0.21 instead of 2.12 mm/day at 15 degC with a
12 h day).
"""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.pcrglobwb.preprocessor import PCRGLOBWBPreProcessor


def _preprocessor(tmp_path, **extra):
    config = SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "PCRGLOBWB",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 86400,
        **extra,
    })
    return PCRGLOBWBPreProcessor(config, Mock())


def test_default_hamon_reference_value_in_m_per_day(tmp_path):
    """15 degC, equator (12 h day): 2.12 mm/day = 2.12e-3 m/day."""
    pp = _preprocessor(tmp_path)
    times = pd.DatetimeIndex(["2005-03-21", "2005-09-23"])
    pet = pp._estimate_pet(np.array([15.0, 15.0]), times, 0.0)
    assert pet == pytest.approx(2.116e-3, rel=2e-3)


def test_oudin_still_selectable_and_in_m_per_day(tmp_path):
    pp = _preprocessor(tmp_path, PCRGLOBWB_PET_METHOD="oudin")
    times = pd.DatetimeIndex(["2005-03-21"])
    pet = pp._estimate_pet(np.array([15.0]), times, 0.0)
    # Ra at the equator near the equinox is ~37.9 MJ/m2/day -> ~3.1 mm/day
    assert pet == pytest.approx(3.1e-3, rel=0.03)
