# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Two raw forcing files for one period must not share a remapped output."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.core.exceptions import FileOperationError
from symfluence.data.preprocessing.resampling.file_processor import FileProcessor

pytestmark = pytest.mark.unit


def _fp(tmp_path):
    cfg = {'DOMAIN_NAME': 'bow', 'FORCING_DATASET': 'ERA5', 'SUB_GRID_DISCRETIZATION': 'GRUs'}
    return FileProcessor(cfg, tmp_path, logging.getLogger('test'))


def test_same_month_files_are_rejected_by_name(tmp_path):
    files = [Path('bow_ERA5_200401_1month.nc'), Path('bow_ERA5_200401_3hr.nc')]
    with pytest.raises(FileOperationError, match='bow_ERA5_200401_1month.nc, bow_ERA5_200401_3hr.nc'):
        _fp(tmp_path).assert_unique_outputs(files)


def test_distinct_months_pass(tmp_path):
    files = [Path('domain_bow_ERA5_merged_200401.nc'), Path('domain_bow_ERA5_merged_200402.nc')]
    _fp(tmp_path).assert_unique_outputs(files)
