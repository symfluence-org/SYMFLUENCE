# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the SUMMA restart repair (snow layers refused by check_icond)."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.modeling.state import ModelState, StateFormat, StateMetadata
from symfluence.models.summa.restart_repair import (
    IDEN_ICE,
    IDEN_WATER,
    SNOW_MAX_FRAC,
    repair_snow_restart,
    snow_total_water_fraction,
)
from symfluence.models.summa.runner import SummaRunner

pytestmark = pytest.mark.unit


def _restart(liq0: float, ice0: float) -> xr.Dataset:
    """Two snow layers over one soil layer, one HRU; top layer 1 mm thick."""
    mid, ifc = ('midToto', 'hru'), ('ifcToto', 'hru')
    liq = np.array([[liq0], [0.1], [0.3]])
    ice = np.array([[ice0], [0.3], [0.0]])
    return xr.Dataset({
        'mLayerVolFracLiq': (mid, liq),
        'mLayerVolFracIce': (mid, ice),
        'mLayerVolFracWat': (mid, liq + ice * IDEN_ICE / IDEN_WATER),
        'mLayerEnthalpy': (mid, np.array([[-2.0e7], [-1.0e8], [1.0e6]])),
        'mLayerTemp': (mid, np.array([[273.15], [270.0], [275.0]])),
        'mLayerDepth': (mid, np.array([[0.001], [0.02], [0.5]])),
        'iLayerHeight': (ifc, np.array([[-0.021], [-0.02], [0.0], [0.5]])),
        # SUMMA restarts can carry one leading slot in mLayerHeight
        'mLayerHeight': (('midPlus', 'hru'), np.array([[-9999.], [-0.0205], [-0.01], [0.25]])),
        'scalarSnowDepth': (('scalarv', 'hru'), np.array([[0.021]])),
        'scalarSWE': (('scalarv', 'hru'), np.array([[7.0]])),
        'nSnow': (('scalarv', 'hru'), np.array([[2]], dtype='int32')),
    })


def _snow_mass(d: xr.Dataset) -> float:
    m = (d.mLayerVolFracLiq * IDEN_WATER + d.mLayerVolFracIce * IDEN_ICE) * d.mLayerDepth
    return float(m[:2].sum())


def _energy(d: xr.Dataset) -> float:
    return float((d.mLayerEnthalpy * d.mLayerDepth)[:2].sum())


def test_total_water_fraction_matches_check_icond():
    assert snow_total_water_fraction(np.float64(0.1), np.float64(0.5)) == pytest.approx(0.1 + 0.5 * 0.917)


def test_repair_keeps_snow_mass_and_energy(tmp_path):
    ds = _restart(0.82, 0.18)          # theta = 0.82 + 0.18 * 0.917 = 0.985
    p = tmp_path / 'restart.nc'
    ds.to_netcdf(p)

    notes = repair_snow_restart(p)

    assert len(notes) == 1 and 'snow layer 0' in notes[0]
    with xr.open_dataset(p) as r:
        assert _snow_mass(r) == pytest.approx(_snow_mass(ds), rel=1e-12)
        assert _energy(r) == pytest.approx(_energy(ds), rel=1e-12)
        theta = snow_total_water_fraction(r.mLayerVolFracLiq.values[0, 0], r.mLayerVolFracIce.values[0, 0])
        assert theta == pytest.approx(0.79)
        assert float(r.mLayerVolFracWat[0, 0]) == pytest.approx(theta)
        # interfaces and mid-points follow the new depth; soil surface stays at 0
        ifc = r.iLayerHeight.values[:, 0]
        assert ifc[2] == 0.0
        assert ifc[1] - ifc[0] == pytest.approx(float(r.mLayerDepth[0, 0]))
        assert r.mLayerHeight.values[1, 0] == pytest.approx(0.5 * (ifc[0] + ifc[1]))
        assert float(r.scalarSnowDepth[0, 0]) == pytest.approx(float(r.mLayerDepth[:2, 0].sum()))
        # untouched: lower snow layer, soil layer, temperature, SWE
        np.testing.assert_array_equal(r.mLayerDepth.values[1:], ds.mLayerDepth.values[1:])
        np.testing.assert_array_equal(r.mLayerTemp.values, ds.mLayerTemp.values)
        assert float(r.scalarSWE[0, 0]) == 7.0
    # idempotent: a repaired file needs nothing more
    assert repair_snow_restart(p) == []


def test_layer_valid_by_summa_formula_is_left_alone(tmp_path):
    # liquid + ice = 0.81, but SUMMA's theta = 0.20 + 0.61 * 0.917 = 0.759 < 0.80
    p = tmp_path / 'restart.nc'
    _restart(0.20, 0.61).to_netcdf(p)
    before = p.read_bytes()
    assert repair_snow_restart(p) == []
    assert p.read_bytes() == before


def test_ice_fraction_above_limit_is_repaired(tmp_path):
    # theta = 0.0 + 0.85 * 0.917 = 0.779 passes, but ice alone > 0.80 does not
    p = tmp_path / 'restart.nc'
    _restart(0.0, 0.85).to_netcdf(p)
    assert len(repair_snow_restart(p)) == 1
    with xr.open_dataset(p) as r:
        assert float(r.mLayerVolFracIce[0, 0]) == pytest.approx(0.79)
        assert float(r.mLayerVolFracIce[0, 0]) <= SNOW_MAX_FRAC


def test_hru_first_dimension_order(tmp_path):
    ds = _restart(0.82, 0.18)
    ds = ds.transpose('hru', ...)
    p = tmp_path / 'restart.nc'
    ds.to_netcdf(p)
    assert len(repair_snow_restart(p)) == 1
    with xr.open_dataset(p) as r:
        assert r.mLayerDepth.dims == ('hru', 'midToto')
        assert _snow_mass(r.transpose('midToto', 'midPlus', 'ifcToto', ...)) == pytest.approx(
            _snow_mass(_restart(0.82, 0.18)), rel=1e-12)


def test_non_state_file_is_ignored(tmp_path):
    p = tmp_path / 'other.nc'
    xr.Dataset({'x': ('hru', np.arange(3.0))}).to_netcdf(p)
    assert repair_snow_restart(p) == []


# ---------------------------------------------------------------------------
# Runner hooks
# ---------------------------------------------------------------------------

def _runner(tmp_path: Path, repair_flag: bool) -> SummaRunner:
    data_dir = tmp_path / 'data'
    settings_dir = data_dir / 'domain_test_domain' / 'settings' / 'SUMMA'
    install_dir = tmp_path / 'summa_bin'
    for d in (settings_dir, install_dir, tmp_path / 'code'):
        d.mkdir(parents=True)
    (install_dir / 'summa_sundials.exe').write_text('', encoding='utf-8')
    (settings_dir / 'fileManager.txt').write_text(
        "initConditionFile    'warmState.nc'    ! initial conditions\n",
        encoding='utf-8',
    )
    config = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(data_dir),
        SYMFLUENCE_CODE_DIR=str(tmp_path / 'code'),
        DOMAIN_NAME='test_domain',
        EXPERIMENT_ID='test_run',
        EXPERIMENT_TIME_START='2020-01-01 00:00',
        EXPERIMENT_TIME_END='2020-01-02 00:00',
        DOMAIN_DEFINITION_METHOD='lumped',
        SUB_GRID_DISCRETIZATION='GRUs',
        HYDROLOGICAL_MODEL='SUMMA',
        ROUTING_MODEL='none',
        FORCING_DATASET='ERA5',
        SUMMA_INSTALL_PATH=str(install_dir),
        SETTINGS_SUMMA_PATH=str(settings_dir),
        SUMMA_REPAIR_INITIAL_STATE=repair_flag,
    )
    return SummaRunner(config, logging.getLogger(__name__))


def test_load_state_repairs_the_copied_restart(tmp_path):
    runner = _runner(tmp_path, repair_flag=False)
    src = tmp_path / 'saved_restart.nc'
    _restart(0.82, 0.18).to_netcdf(src)
    state = ModelState(
        metadata=StateMetadata(model_name='SUMMA', timestamp='2020-01-01', format=StateFormat.FILE_NETCDF),
        files=[src],
    )

    runner.load_state(state)

    with xr.open_dataset(runner.settings_path / 'coldState.nc') as r:
        assert snow_total_water_fraction(r.mLayerVolFracLiq.values[0, 0],
                                         r.mLayerVolFracIce.values[0, 0]) <= SNOW_MAX_FRAC
    # the saved state itself is not modified
    with xr.open_dataset(src) as s:
        assert float(s.mLayerVolFracLiq[0, 0]) == 0.82


def test_initial_state_path_follows_file_manager(tmp_path):
    runner = _runner(tmp_path, repair_flag=True)
    assert runner._initial_state_path() == runner.settings_path / 'warmState.nc'


@pytest.mark.parametrize('flag', [True, False])
def test_run_summa_repairs_initial_state_only_when_enabled(tmp_path, flag):
    runner = _runner(tmp_path, repair_flag=flag)
    init = runner.settings_path / 'warmState.nc'
    _restart(0.82, 0.18).to_netcdf(init)
    runner.run = MagicMock(return_value=tmp_path)

    assert runner.run_summa() == tmp_path

    with xr.open_dataset(init) as r:
        repaired = float(r.mLayerVolFracLiq[0, 0]) < 0.82
    assert repaired is flag
