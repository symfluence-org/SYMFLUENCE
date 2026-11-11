# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SUMMA forcing paths longer than its Fortran limit fail early and clearly."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from symfluence.core.exceptions import FileOperationError
from symfluence.models.summa.forcing_processor import SUMMA_PATH_MAX, SummaForcingProcessor

pytestmark = pytest.mark.unit


class _Proc(SummaForcingProcessor):
    """Just enough state for create_forcing_file_list."""

    def __init__(self, forcing_dir, setup_dir):
        self._forcing_dir = forcing_dir
        self.setup_dir = setup_dir
        self.logger = logging.getLogger('t')
        self.forcing_selector = None
        self._cfg = SimpleNamespace(
            forcing=SimpleNamespace(dataset='ERA5'), domain=SimpleNamespace(name='dom', discretization='GRUs'),
            model=SimpleNamespace(summa=SimpleNamespace(forcing_list='list.txt')))

    @property
    def project_forcing_dir(self):
        return self._forcing_dir

    @property
    def config(self):
        return self._cfg

    def _get_config_value(self, getter, default=None, dict_key=None):
        return getter()


def _processor(forcing_dir, setup_dir):
    return _Proc(forcing_dir, setup_dir)


def _forcing(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({'pptrate': (('time', 'hru'), np.zeros((2, 1)))},
               coords={'time': np.array(['2004-01-01T00', '2004-01-01T01'], dtype='datetime64[ns]')}
               ).to_netcdf(path)


def test_overlong_forcing_path_raises(tmp_path):
    deep = tmp_path / ('d' * max(1, SUMMA_PATH_MAX - len(str(tmp_path))))
    _forcing(deep / 'SUMMA_input' / 'dom_ERA5_remapped_grus-h1-0123abcd_2004-01-01-00-00-00.nc')
    (tmp_path / 'setup').mkdir()
    with pytest.raises(FileOperationError, match='character limit'):
        _processor(deep, tmp_path / 'setup').create_forcing_file_list()


def test_short_forcing_path_is_listed(tmp_path):
    _forcing(tmp_path / 'f' / 'SUMMA_input' / 'dom_ERA5_remapped_grus-h1-0123abcd_2004-01-01-00-00-00.nc')
    (tmp_path / 'setup').mkdir()
    _processor(tmp_path / 'f', tmp_path / 'setup').create_forcing_file_list()
    assert (tmp_path / 'setup' / 'list.txt').read_text().strip().endswith('00-00-00.nc')
