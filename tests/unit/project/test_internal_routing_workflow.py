# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Internally routed models get the router preprocessed, but not run twice."""
from __future__ import annotations

import logging

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.project.model_manager import ModelManager

pytestmark = pytest.mark.unit


def _manager(tmp_path, **extra):
    cfg = SymfluenceConfig.from_minimal(
        domain_name='wf', model=extra.pop('model'), time_start='2004-01-01 00:00',
        time_end='2004-01-03 23:00', SYMFLUENCE_DATA_DIR=str(tmp_path), **extra)
    return ModelManager(cfg, logging.getLogger('t'))


def test_semidistributed_fuse_preprocesses_mizuroute_but_runs_it_internally(tmp_path):
    mm = _manager(tmp_path, model='FUSE', DOMAIN_DEFINITION_METHOD='semidistributed',
                  FUSE_SPATIAL_MODE='semi_distributed', ROUTING_MODEL='mizuRoute')
    assert mm._resolve_model_workflow(for_preprocessing=True) == ['FUSE', 'MIZUROUTE']
    assert mm._resolve_model_workflow() == ['FUSE']


def test_lumped_fuse_needs_no_router(tmp_path):
    mm = _manager(tmp_path, model='FUSE', DOMAIN_DEFINITION_METHOD='lumped',
                  FUSE_SPATIAL_MODE='lumped', ROUTING_MODEL='mizuRoute')
    assert mm._resolve_model_workflow(for_preprocessing=True) == ['FUSE']


def test_summa_still_runs_mizuroute_as_a_step(tmp_path):
    mm = _manager(tmp_path, model='SUMMA', DOMAIN_DEFINITION_METHOD='semidistributed',
                  ROUTING_MODEL='mizuRoute')
    assert mm._resolve_model_workflow() == ['SUMMA', 'MIZUROUTE']
