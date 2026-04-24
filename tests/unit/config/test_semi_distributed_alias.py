# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""DOMAIN_DEFINITION_METHOD accepts hyphen/underscore spelling variants.

SH (iter-3, 11.SH.3) reported that the paper configs write
``DOMAIN_DEFINITION_METHOD: semi_distributed`` (underscore) but the
validator only accepted ``semidistributed`` (no separator). Accept
both underscore and hyphen variants so the paper configs run as-is.
"""

import pytest
from pydantic import ValidationError

from symfluence.core.config.models.root import SymfluenceConfig

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _base_config(definition_method):
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/data',
        'SYMFLUENCE_CODE_DIR': '/tmp/code',
        'DOMAIN_NAME': 'test',
        'EXPERIMENT_ID': 'exp_001',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': definition_method,
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_DATASET': 'ERA5',
    }


@pytest.mark.parametrize('v', ['semidistributed', 'semi_distributed', 'semi-distributed'])
def test_semidistributed_spelling_variants_accepted(v):
    cfg = SymfluenceConfig(**_base_config(v))
    assert cfg.domain.definition_method == 'semidistributed'


def test_other_legacy_aliases_still_work():
    for v in ('delineate', 'discretized', 'subset'):
        cfg = SymfluenceConfig(**_base_config(v))
        assert cfg.domain.definition_method == 'semidistributed'


def test_typo_still_rejected():
    """A close typo (missing letter) must still raise."""
    with pytest.raises(ValidationError):
        SymfluenceConfig(**_base_config('semi_disributed'))
