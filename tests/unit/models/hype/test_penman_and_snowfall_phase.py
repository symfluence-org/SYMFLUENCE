from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.models.hype.calibration.parameter_manager import HYPEParameterManager
from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.forcing_processor import (
    HYPEForcingProcessor,
    required_extra_obs,
    wet_bulb_temperature,
)

LOG = logging.getLogger('test')


def _forcing(path, t_c, rh, p_mm_step, wind=4.0, ids=(4, 9)):
    """Two subbasins, eight three-hour steps per day; arrays are (time, hru) or scalars."""
    n = max(np.size(t_c), np.size(rh), np.size(p_mm_step), 8)
    shape = (n, len(ids))
    dates = pd.date_range('2010-01-01', periods=n, freq='3h')
    full = {k: np.broadcast_to(np.asarray(v, float).reshape(-1, 1) if np.ndim(v) == 1 else v, shape).copy()
            for k, v in dict(t=t_c, rh=rh, p=p_mm_step, u=wind).items()}
    ds = xr.Dataset({'air_temperature': (('time', 'hru'), full['t'] + 273.15),
                     'relative_humidity': (('time', 'hru'), full['rh']),
                     'precipitation_flux': (('time', 'hru'), full['p'] / 10800),
                     'wind_speed': (('time', 'hru'), full['u']),
                     'hruId': ('hru', list(ids))}, coords={'time': dates, 'hru': range(len(ids))})
    for name, unit in [('air_temperature', 'K'), ('relative_humidity', '%'),
                       ('precipitation_flux', 'kg m-2 s-1'), ('wind_speed', 'm s-1')]:
        ds[name].attrs['units'] = unit
    ds.to_netcdf(path)
    return path


def _par_rows(path):
    return {s.split()[0]: s.split('!!')[0].split()[1:] for s in path.read_text().splitlines()
            if s.strip() and not s.startswith('!')}


def test_required_extra_obs_follow_model_options():
    assert required_extra_obs({}) == []
    assert required_extra_obs({'HYPE_SNOW_MELT_MODEL': 2}) == ['SWobs']
    assert required_extra_obs({'HYPE_PET_MODEL': 5, 'HYPE_SNOW_MELT_MODEL': 2,
                               'HYPE_SNOWFALL_FRACTION': 'wet_bulb'}) == ['SFobs', 'SWobs', 'RHobs', 'Uobs']


def test_wet_bulb_matches_stull_reference():
    # Stull (2011): T = 20 C, RH = 50 % gives Tw = 13.7 C.
    assert wet_bulb_temperature(20.0, 50.0) == pytest.approx(13.7, abs=0.05)
    assert wet_bulb_temperature(0.0, 100.0) == pytest.approx(0.0, abs=0.3)


def test_humidity_and_wind_daily_files(tmp_path):
    p = _forcing(tmp_path / 'f.nc', 1.0, np.r_[[60.0] * 8, [100.0] * 8], 0.0, wind=5.0)
    proc = HYPEForcingProcessor({'HYPE_WIND_MEASUREMENT_HEIGHT': 10.0}, LOG, tmp_path, tmp_path, tmp_path)
    rh = pd.read_csv(proc.write_humidity_obs([p]), sep='\t', index_col=0)
    assert list(rh.columns) == ['4', '9']
    np.testing.assert_allclose(rh.to_numpy(), [[0.6, 0.6], [1.0, 1.0]])
    u = pd.read_csv(proc.write_wind_obs([p]), sep='\t', index_col=0)
    np.testing.assert_allclose(u.to_numpy(), 5.0 * 4.87 / np.log(67.8 * 10 - 5.42), rtol=1e-5)


def test_humidity_falls_back_to_specific_humidity(tmp_path):
    dates = pd.date_range('2010-01-01', periods=8, freq='3h')
    t, p = 10.0, 100000.0
    es = 610.94 * np.exp(17.625 * t / (t + 243.04))
    q = 0.622 * 0.5 * es / (p - 0.378 * 0.5 * es)  # RH 50 %
    ds = xr.Dataset({'air_temperature': (('time', 'hru'), np.full((8, 1), t + 273.15)),
                     'specific_humidity': (('time', 'hru'), np.full((8, 1), q)),
                     'surface_air_pressure': (('time', 'hru'), np.full((8, 1), p))},
                    coords={'time': dates, 'hru': [0]})
    for name, unit in [('air_temperature', 'K'), ('specific_humidity', 'kg kg-1'), ('surface_air_pressure', 'Pa')]:
        ds[name].attrs['units'] = unit
    ds.to_netcdf(tmp_path / 'f.nc')
    proc = HYPEForcingProcessor({}, LOG, tmp_path, tmp_path, tmp_path)
    rh = pd.read_csv(proc.write_humidity_obs([tmp_path / 'f.nc']), sep='\t', index_col=0)
    np.testing.assert_allclose(rh.to_numpy(), 0.5, atol=1e-4)


def test_snowfall_fraction_is_precipitation_weighted_per_step(tmp_path):
    # Day 1: all-snow steps carry 3 mm, all-rain steps 1 mm -> 0.75. Day 2: dry, half the steps snow -> 0.5.
    t = np.r_[[-5.0] * 4, [8.0] * 4, [-5.0] * 4, [8.0] * 4]
    precip = np.r_[[0.75] * 4, [0.25] * 4, [0.0] * 8]
    p = _forcing(tmp_path / 'f.nc', t, 90.0, precip)
    proc = HYPEForcingProcessor({'HYPE_SNOWFALL_FRACTION': 'wet_bulb'}, LOG, tmp_path, tmp_path, tmp_path)
    sf = pd.read_csv(proc.write_snowfall_fraction_obs([p]), sep='\t', index_col=0)
    np.testing.assert_allclose(sf.to_numpy(), [[0.75, 0.75], [0.5, 0.5]])
    proc = HYPEForcingProcessor({'HYPE_WET_BULB_SNOW_RANGE': [2.0, 0.0]}, LOG, tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValidationError, match='increasing'):
        proc.write_snowfall_fraction_obs([p])


def test_snowfall_fraction_uses_lapsed_band_temperature(tmp_path):
    # One lumped column at 1 C: the high band is lapsed below freezing, the low band above.
    p = _forcing(tmp_path / 'f.nc', 1.0, 100.0, 1.0, ids=(1,))
    proc = HYPEForcingProcessor({}, LOG, tmp_path, tmp_path, tmp_path)
    proc.set_elevation_bands([{'hru_id': 1, 'elev_mean': 400.0, 'area': 1.0},
                              {'hru_id': 2, 'elev_mean': 1600.0, 'area': 1.0}], lapse_rate=0.0065)
    sf = pd.read_csv(proc.write_snowfall_fraction_obs([p]), sep='\t', index_col=0)
    assert sf['1'].iloc[0] < 0.5 < sf['2'].iloc[0]


def test_penman_monteith_and_snowfall_info_flags(tmp_path):
    cfg = {'HYPE_PET_MODEL': 5, 'HYPE_SNOWFALL_FRACTION': 'wet_bulb'}
    mgr = HYPEConfigManager(cfg, LOG, tmp_path)
    pd.DataFrame({'time': pd.date_range('2010-01-01', periods=2), '1': [1, 1]}).to_csv(tmp_path / 'Pobs.txt', sep='\t', index=False)
    mgr.write_info_filedir(0, str(tmp_path / 'results'), experiment_start='2010-01-01', experiment_end='2010-01-02')
    info = (tmp_path / 'info.txt').read_text()
    for line in ['readsfobs\ty', 'readswobs\ty', 'readuobs\ty', 'readrhobs\ty',
                 'modeloption snowfallmodel\t1', 'modeloption petmodel\t5']:
        assert line in info
    mgr.write_par_file(land_uses=np.array([15, 16]))
    assert _par_rows(tmp_path / 'par.txt')['kc'] == ['1.0'] * 16


def test_extra_parameters_add_replace_and_validate(tmp_path):
    cfg = {'HYPE_EXTRA_PARAMETERS': {'pcusnow': 0.25, 'snalbmin': 0.5, 'cmlt': 4.0, 'mperc1': 1.0}}
    mgr = HYPEConfigManager(cfg, LOG, tmp_path)
    mgr.write_par_file(land_uses=np.array([15, 16]), params={'pcusnow': 0.3})
    rows = _par_rows(tmp_path / 'par.txt')
    assert rows['pcusnow'] == ['0.3']  # calibration value wins over the fixed entry
    assert rows['snalbmin'] == ['0.5'] * 16  # new land-use parameter: one column per class
    assert set(rows['cmlt']) == {'4.0'} and len(rows['cmlt']) == 16  # existing entry keeps its width
    assert rows['mperc1'] == ['1.0'] * 2  # no GeoClass.txt: default two soil classes
    with pytest.raises(ValidationError, match='needs 1 or 16'):
        HYPEConfigManager({'HYPE_EXTRA_PARAMETERS': {'snalbmax': [0.8, 0.9]}}, LOG, tmp_path).write_par_file(
            land_uses=np.array([15, 16]))


def test_calibrated_parameter_missing_from_par_fails(tmp_path):
    mgr = HYPEConfigManager({}, LOG, tmp_path)
    with pytest.raises(ValidationError, match='pcusnow'):
        mgr.write_par_file(land_uses=np.array([15, 16]), params={'pcusnow': 0.3})


def test_par_updater_sets_every_class_column(tmp_path):
    (tmp_path / 'par.txt').write_text('ttmp\t0.0\t0.0\t0.0\t!! threshold\nlp\t0.5\n')
    mgr = HYPEParameterManager({'HYPE_PARAMS_TO_CALIBRATE': 'ttmp,lp', 'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'e',
                                'SYMFLUENCE_DATA_DIR': str(tmp_path)}, LOG, tmp_path)
    mgr.par_file_path = tmp_path / 'par.txt'
    assert mgr.update_par_file({'ttmp': -1.5, 'lp': 0.9})
    rows = _par_rows(tmp_path / 'par.txt')
    assert [float(v) for v in rows['ttmp']] == [-1.5] * 3
    assert [float(v) for v in rows['lp']] == [0.9]
    assert not mgr.update_par_file({'pcusnow': 0.2})
