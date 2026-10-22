"""Routing conversion must retain process diagnostics in native SUMMA output."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from symfluence.core.modeling.execution.spatial_orchestrator import SpatialOrchestrator


@pytest.mark.parametrize('ngr', [1, 3])
@pytest.mark.parametrize('separate_output', [False, True])
def test_native_routing_retains_hru_diagnostics(tmp_path, ngr, separate_output):
    src = tmp_path / 'summa.nc'
    dst = tmp_path / 'routing.nc' if separate_output else None
    original = xr.Dataset({
        'averageRoutedRunoff': (('time', 'gru'), np.ones((4, ngr))),
        'gruId': ('gru', np.arange(ngr) * 10 + 65),
        'hruId': ('hru', np.arange(5)),
        'scalarSWE': (('time', 'hru'), np.arange(20).reshape(4, 5)),
        'scalarTotalSoilWat': (('time', 'hru'), np.full((4, 5), 100.)),
    }, coords={'time': np.arange(4)}, attrs={'origin': 'SUMMA'})
    original.to_netcdf(src)
    owner = SimpleNamespace(logger=logging.getLogger('routing-test'))
    result = SpatialOrchestrator.convert_to_routing_format(owner, src, dst)
    assert result == (dst or src)
    with xr.open_dataset(result) as actual:
        xr.testing.assert_identical(actual, original)
    with xr.open_dataset(src) as actual:
        xr.testing.assert_identical(actual, original)
