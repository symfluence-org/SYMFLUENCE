# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Unit tests for paper case study configuration files.

Pins reproducibility-critical properties in
examples/paper_case_studies/configs/configs_nested/* so that accidental
removal of required fields (e.g. random_seed on stochastic experiments)
is caught in CI rather than in downstream reproducibility runs.
"""

from pathlib import Path

import pytest
import yaml

from symfluence.core.config.models import SymfluenceConfig

pytestmark = [pytest.mark.unit, pytest.mark.quick]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_NESTED = REPO_ROOT / "examples" / "paper_case_studies" / "configs" / "configs_nested"


def _load(path: Path) -> SymfluenceConfig:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return SymfluenceConfig.model_validate(data)


def test_decision_ensemble_sets_random_seed():
    """Decision ensemble must pin a random_seed for paper reproducibility.

    PW reported best-run performance diverging from the paper (0.89 vs 0.86)
    with three different decision choices. Root cause: DDS initialized with
    unseeded RNG. This test prevents the seed from being silently dropped.
    """
    cfg = _load(CONFIGS_NESTED / "06_decision_ensemble" / "config_fuse_decisions.yaml")
    assert cfg.system.random_seed is not None, (
        "06_decision_ensemble config must set system.random_seed for "
        "reproducibility of DDS-driven decision enumeration."
    )
    assert isinstance(cfg.system.random_seed, int)
