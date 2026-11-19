"""Standard output-variable adapters, one synthetic output set per model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import symfluence  # noqa: F401  (registers adapters)
from symfluence.core.modeling.output_variables import (
    ADAPTER_VARIABLES,
    UnsupportedVariableError,
    cumulative_specific_balance,
    get_output_adapter,
    support_matrix,
)

pytestmark = pytest.mark.unit

DATES = pd.date_range('2001-01-01', periods=10, freq='D')


def _project(tmp_path, model):
    settings = tmp_path / 'settings' / model
    settings.mkdir(parents=True)
    out = tmp_path / 'sim'
    out.mkdir()
    return tmp_path, settings, out


# ----------------------------------------------------------------------
def test_cumulative_specific_balance_uses_step_area():
    idx = pd.date_range('2000-01-01', periods=3, freq='YS')
    storage = pd.Series([10e9, 9e9, 8e9], index=idx)          # m3 w.e.
    area = pd.Series([10.0, 10.0, 10.0], index=idx)            # km2
    mb = cumulative_specific_balance(storage, area)
    np.testing.assert_allclose(mb.values, [0.0, -100000.0, -200000.0])  # 1e9 m3 / 1e7 m2 = 100 m


def test_support_matrix_lists_core_adapters():
    matrix = support_matrix()
    for model in ('SUMMA', 'HYPE', 'GR', 'PRMS', 'GSFLOW', 'RAVEN', 'WFLOW', 'MHM'):
        assert model in matrix
        assert set(matrix[model]) == set(ADAPTER_VARIABLES)
    assert matrix['GR']['glacier_mass_balance'] and not matrix['GR']['swe']
    assert all(matrix['SUMMA'].values())


def test_unknown_model_has_no_adapter():
    assert get_output_adapter('NOT_A_MODEL') is None


def test_unsupported_variable_raises_cleanly(tmp_path):
    adapter = get_output_adapter('GR', tmp_path)
    with pytest.raises(UnsupportedVariableError):
        adapter.extract('swe', tmp_path)
    with pytest.raises(UnsupportedVariableError):
        adapter.extract('not_a_variable', tmp_path)


# ----------------------------------------------------------------------
def test_summa_adapter(tmp_path):
    project, _, out = _project(tmp_path, 'SUMMA')
    n = len(DATES)
    ds = xr.Dataset(
        {
            'basin__GlacierStorage': (('time', 'gru'), np.column_stack([np.linspace(2, 1, n), np.ones(n)])),
            'basin__GlacierArea': (('time', 'gru'), np.full((n, 2), 5e6)),
            'scalarSWE': (('time', 'hru'), np.column_stack([np.full(n, 100.0), np.full(n, 300.0)])),
            'HRUarea': (('hru',), [3.0, 1.0]),
            'scalarGroundSnowFraction': (('time', 'hru'), np.full((n, 2), 0.5)),
            'scalarTotalET': (('time', 'hru'), np.full((n, 2), -1.0 / 86400)),
        },
        coords={'time': DATES},
    )
    ds.to_netcdf(out / 'run_day.nc')
    a = get_output_adapter('SUMMA', project)
    storage = a.extract('glacier_storage', out)
    assert storage.iloc[0] == pytest.approx(3e9)          # (2 + 1) Gt
    assert a.extract('glacier_area', out).iloc[0] == pytest.approx(10.0)
    assert a.extract('swe', out).iloc[0] == pytest.approx(150.0)   # area-weighted 3:1
    assert a.extract('snow_covered_fraction', out).iloc[0] == pytest.approx(0.5)
    assert a.extract('et', out).iloc[0] == pytest.approx(1.0)
    mb = a.extract('glacier_mass_balance', out)
    # 1 Gt lost over 10 km2 -> -100 m w.e. = -1e5 mm
    assert mb.iloc[-1] == pytest.approx(-1e5)


def _hype_file(path, values: dict):
    frame = pd.DataFrame(values, index=DATES)
    with open(path, 'w') as f:
        f.write('!! HYPE output\n')
        f.write('DATE\t' + '\t'.join(frame.columns) + '\n')
        for d, row in frame.iterrows():
            f.write(d.strftime('%Y-%m-%d') + '\t' + '\t'.join(f'{v:.6f}' for v in row) + '\n')


def test_hype_adapter(tmp_path):
    project, settings, out = _project(tmp_path, 'HYPE')
    pd.DataFrame({'subid': [1, 2], 'area': [3e6, 1e6], 'glacier_fraction': [0.0, 0.5]}).to_csv(
        settings / 'GeoData.txt', sep='\t', index=False)
    n = len(DATES)
    _hype_file(out / 'timeSNOW.txt', {'1': [100.0] * n, '2': [300.0] * n})
    _hype_file(out / 'timeEVAP.txt', {'1': [1.0] * n, '2': [3.0] * n})
    _hype_file(out / 'timeGLCV.txt', {'1': [0.0] * n, '2': list(np.linspace(1.0, 0.9, n))})
    _hype_file(out / 'timeGLCA.txt', {'1': [0.0] * n, '2': [0.5] * n})
    _hype_file(out / 'timeCFSC.txt', {'1': [0.5] * n, '2': [0.5] * n})
    a = get_output_adapter('HYPE', project)
    assert a.extract('swe', out).iloc[0] == pytest.approx(150.0)
    assert a.extract('et', out).iloc[0] == pytest.approx(1.5)
    assert a.extract('glacier_storage', out).iloc[0] == pytest.approx(0.85e9)
    assert a.extract('glacier_area', out).iloc[0] == pytest.approx(0.5)
    # covered = 0.5 * 4 km2 = 2 km2 over ice-free 3.5 km2
    assert a.extract('snow_covered_fraction', out).iloc[0] == pytest.approx(2 / 3.5)


def test_gr_adapter(tmp_path):
    out = tmp_path / 'sim'
    out.mkdir()
    (out / 'GR_glacier_diagnostics.json').write_text(json.dumps({
        'glacier_area_km2_outline': 12.0,
        'mass_balance_mwe': {'2002': -1.0, '2003': -0.5},
    }))
    a = get_output_adapter('GR', tmp_path)
    mb = a.extract('glacier_mass_balance', out)
    assert list(mb.index) == [pd.Timestamp(2001, 9, 30), pd.Timestamp(2002, 9, 30), pd.Timestamp(2003, 9, 30)]
    np.testing.assert_allclose(mb.values, [0.0, -1000.0, -1500.0])
    assert a.extract('glacier_area', out).iloc[0] == pytest.approx(12.0)


def _prms_params(path, hru_area_acres, ice):
    def block(name, values):
        return f'####\n{name}\n1\nnhru\n{len(values)}\n2\n' + '\n'.join(str(v) for v in values) + '\n'
    path.write_text('header\n** Dimensions **\n** Parameters **\n'
                    + block('hru_area', hru_area_acres) + block('glacier_frac_init', ice))


@pytest.mark.parametrize('model', ['PRMS', 'GSFLOW'])
def test_prms_family_adapter(tmp_path, model):
    project, settings, out = _project(tmp_path, model)
    acres = 247.105381467
    _prms_params(settings / 'params.dat', [3 * acres, acres], [0, 1])
    with open(out / 'statvar.dat', 'w') as f:
        f.write('5\nbasin_pweqv 1\nbasin_actet 1\nbasin_gl_storage 1\nsnowcov_area 1\nsnowcov_area 2\n')
        for i, d in enumerate(DATES):
            f.write(f'{i + 1} {d.year} {d.month} {d.day} 0 0 0 2.0 0.1 {10 - i * 0.1} 0.4 1.0\n')
    a = get_output_adapter(model, project)
    assert a.extract('swe', out).iloc[0] == pytest.approx(50.8)
    assert a.extract('et', out).iloc[0] == pytest.approx(2.54)
    assert a.extract('glacier_area', out).iloc[0] == pytest.approx(1.0)
    # 10 in basin mean over 4 km2 -> 0.254 m * 4e6 m2
    assert a.extract('glacier_storage', out).iloc[0] == pytest.approx(0.254 * 4e6)
    assert a.extract('snow_covered_fraction', out).iloc[0] == pytest.approx(0.4)


def _raven_by_hru(path, ids, values):
    lines = ['time,date,' + ','.join(str(i) for i in ids), 'units,,' + ','.join('mm' for _ in ids)]
    for k, d in enumerate(DATES):
        lines.append(f'{k},{d:%Y-%m-%d},' + ','.join(str(v) for v in values))
    path.write_text('\n'.join(lines) + '\n')


def test_raven_adapter(tmp_path):
    project, settings, out = _project(tmp_path, 'RAVEN')
    (settings / 'geometry.json').write_text(json.dumps({'hrus': [
        {'id': 1, 'area_km2': 3.0, 'ice': False}, {'id': 2, 'area_km2': 1.0, 'ice': True}]}))
    _raven_by_hru(out / 'GLACIER_ICE_Daily_Average_ByHRU.csv', [1, 2], [0.0, 1000.0])
    _raven_by_hru(out / 'SNOW_Daily_Average_ByHRU.csv', [1, 2], [100.0, 500.0])
    a = get_output_adapter('RAVEN', project)
    # ice HRU: (1000 + 500) mm * 1 km2 = 1.5e6 m3
    assert a.extract('glacier_storage', out).iloc[0] == pytest.approx(1.5e6)
    assert a.extract('glacier_area', out).iloc[0] == pytest.approx(1.0)
    assert a.extract('swe', out).iloc[0] == pytest.approx(200.0)


def test_wflow_adapter(tmp_path):
    project, settings, out = _project(tmp_path, 'WFLOW')
    xr.Dataset({
        'wflow_subcatch': (('y', 'x'), [[1.0, 1.0], [1.0, np.nan]]),
        'wflow_cellarea': (('y', 'x'), np.full((2, 2), 1e6)),
        'wflow_glacierfrac': (('y', 'x'), [[0.5, 0.0], [0.0, 0.0]]),
    }, coords={'y': [0, 1], 'x': [0, 1]}).to_netcdf(settings / 'wflow_staticmaps.nc')
    pd.DataFrame({'time': DATES + pd.Timedelta(days=1), 'SWE': 40.0, 'Ice': 100.0}).to_csv(
        out / 'output.csv', index=False)
    a = get_output_adapter('WFLOW', project)
    swe = a.extract('swe', out)
    assert swe.index[0] == DATES[0] and swe.iloc[0] == pytest.approx(40.0)
    assert a.extract('glacier_storage', out).iloc[0] == pytest.approx(100.0 * 3 * 1e3)
    assert a.extract('glacier_area', out).iloc[0] == pytest.approx(0.5)


def test_mhm_adapter(tmp_path):
    out = tmp_path / 'sim'
    out.mkdir()
    xr.Dataset({
        'snowpack': (('time', 'y', 'x'), np.full((len(DATES), 2, 2), 20.0)),
        'aET': (('time', 'y', 'x'), np.full((len(DATES), 2, 2), 1.2)),
    }, coords={'time': DATES}).to_netcdf(out / 'mHM_Fluxes_States.nc')
    a = get_output_adapter('MHM', tmp_path)
    assert a.extract('swe', out).iloc[0] == pytest.approx(20.0)
    assert a.extract('et', out).iloc[0] == pytest.approx(1.2)
    with pytest.raises(UnsupportedVariableError):
        a.extract('glacier_storage', out)
