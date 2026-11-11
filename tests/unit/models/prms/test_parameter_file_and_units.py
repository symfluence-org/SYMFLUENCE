"""PRMS parameter-file updates, mode-dependent parameter names, units and HRU areas."""
from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.gsflow.parameter_bounds import PARAMS as GSFLOW_PARAMS
from symfluence.models.prms.calibration.parameter_manager import PRMSParameterManager
from symfluence.models.prms.calibration.worker import PRMSWorker
from symfluence.models.prms.config_schema import PRMSConfig
from symfluence.models.prms.forcing_window import daily_forcing_window
from symfluence.models.prms.parameter_file import update_parameter_file
from symfluence.models.prms.parameters import (
    DEFAULT_CALIBRATION_PARAMS,
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    default_calibration_parameters,
)
from symfluence.models.prms.preprocessor import PRMSPreProcessor

PROPS = dict(lat=64., lon=-19., elev=700., area_km2=1145., slope=5., aspect=0.)
LOG = logging.getLogger('test_prms')


def preprocessor(tmp_path, **cfg):
    p = object.__new__(PRMSPreProcessor)
    p.settings_dir = p.forcing_dir = tmp_path
    p.use_obs_solar = False
    p.config = NS(model=NS(prms=PRMSConfig(**cfg)))
    p._get_config_value = lambda getter, default=None, **kw: getter() if default is None else default
    p._get_catchment_properties = lambda: PROPS
    p._get_dominant_classes = lambda: dict(cov_type=0, soil_type=2)
    return p


def values(path):
    out = {}
    for b in path.read_text().split('** Parameters **')[1].split('####'):
        x = b.strip().splitlines()
        if x:
            out[x[0]] = [float(v) for v in x[4 + int(x[1]):]]
    return out


def manager(tmp_path, **config):
    return PRMSParameterManager(
        {'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'e', 'SYMFLUENCE_DATA_DIR': str(tmp_path), **config}, LOG, tmp_path)


# --- missing parameters -----------------------------------------------------

PARAMS_DAT = ("hdr\n** Parameters **\n####\nsoil_moist_max\n1\nnhru\n1\n2\n6.0\n"
              "####\nsoil_rechr_max_frac\n1\nnhru\n1\n2\n0.333\n")


def test_missing_calibration_parameter_raises_in_worker(tmp_path):
    """Regression: the update returned True and left the file unchanged."""
    path = tmp_path / 'params.dat'
    path.write_text(PARAMS_DAT)
    worker = object.__new__(PRMSWorker)
    worker.logger = LOG
    with pytest.raises(ConfigurationError, match=r"not in params.dat: \['soil_rechr_max', 'tmax_allrain'\]"):
        worker._update_parameter_file(path, {'soil_moist_max': 7.0, 'soil_rechr_max': 3.21, 'tmax_allrain': 4.0})
    assert path.read_text() == PARAMS_DAT  # nothing half-written
    assert worker._update_parameter_file(path, {'soil_moist_max': 7.0, 'soil_rechr_max_frac': 0.5})
    assert values(path) == {'soil_moist_max': [7.0], 'soil_rechr_max_frac': [0.5]}


def test_missing_calibration_parameter_raises_in_manager(tmp_path):
    (tmp_path / 'params.dat').write_text(PARAMS_DAT)
    m = manager(tmp_path, PRMS_PARAMS_TO_CALIBRATE='soil_moist_max,rain_cbh_adj')
    with pytest.raises(ConfigurationError, match='rain_cbh_adj'):
        m.update_model_files({'soil_moist_max': 7.0, 'rain_cbh_adj': 1.1})
    assert m.update_model_files({'soil_moist_max': 7.0})


def test_gsflow_error_names_gsflow(tmp_path):
    path = tmp_path / 'params.dat'
    path.write_text(PARAMS_DAT)
    with pytest.raises(ConfigurationError, match='GSFLOW parameters not in params.dat'):
        update_parameter_file(path, {'nope': 1.0}, model='GSFLOW')


# --- default calibration list per mode ---------------------------------------

def test_schema_default_matches_shared_constant():
    assert PRMSConfig().params_to_calibrate == DEFAULT_CALIBRATION_PARAMS


@pytest.mark.parametrize('mode', ['DAILY', 'PRMS5'])
def test_default_calibration_list_exists_in_generated_file(tmp_path, mode):
    """Every default calibration parameter has a block in the file the
    preprocessor writes for that mode (soil_rechr_max / tmax_allrain do not
    under PRMS5)."""
    p = preprocessor(tmp_path, model_mode=mode)
    p._generate_parameter_file()
    m = manager(tmp_path, PRMS_MODEL_MODE=mode, PRMS_PARAMS_TO_CALIBRATE=DEFAULT_CALIBRATION_PARAMS)
    assert m.prms_params == default_calibration_parameters(mode)
    assert set(m.prms_params) <= set(values(tmp_path / 'params.dat'))
    assert set(m.prms_params) <= set(m.param_bounds)
    initial = m.get_initial_parameters()
    assert m.update_model_files(initial)


def test_prms5_default_list_uses_prms5_names(tmp_path):
    m = manager(tmp_path, PRMS_MODEL_MODE='PRMS5', PRMS_PARAMS_TO_CALIBRATE=DEFAULT_CALIBRATION_PARAMS)
    assert 'soil_rechr_max_frac' in m.prms_params and 'tmax_allrain_offset' in m.prms_params
    assert 'soil_rechr_max' not in m.prms_params and 'tmax_allrain' not in m.prms_params


def test_prms5_rejects_explicit_parameters_it_does_not_read(tmp_path):
    with pytest.raises(ConfigurationError, match='tmax_allrain_offset'):
        manager(tmp_path, PRMS_MODEL_MODE='PRMS5', PRMS_PARAMS_TO_CALIBRATE='soil_moist_max,tmax_allrain')


def test_prms5_writes_rain_threshold_offset(tmp_path):
    """PRMS5 ignores tmax_allrain; without the offset block PRMS falls back to
    its 1.0 default instead of the intended 3.3 degC threshold."""
    preprocessor(tmp_path, model_mode='PRMS5')._generate_parameter_file()
    v = values(tmp_path / 'params.dat')
    assert 'tmax_allrain' not in v
    assert v['tmax_allsnow'] == [0.0] * 12
    assert v['tmax_allrain_offset'] == [3.3] * 12
    with pytest.raises(ConfigurationError, match='tmax_allrain_offset'):
        preprocessor(tmp_path, model_mode='PRMS5', parameter_overrides={
            'tmax_allrain': {'dimensions': ['nmonths'], 'values': 2.0}})._generate_parameter_file()


def test_daily_mode_still_writes_tmax_allrain(tmp_path):
    preprocessor(tmp_path)._generate_parameter_file()
    v = values(tmp_path / 'params.dat')
    assert v['tmax_allrain'] == [3.3] * 12 and 'tmax_allrain_offset' not in v


# --- units --------------------------------------------------------------------

def test_temperature_bounds_are_celsius_like_the_parameter_file(tmp_path):
    """Regression: bounds were degF (30..50, -10..35) for a degC parameter file."""
    preprocessor(tmp_path)._generate_parameter_file()
    v = values(tmp_path / 'params.dat')
    assert v['temp_units'] == [1.0]
    for name in ('tmax_allrain', 'tmax_allsnow'):
        bounds = PARAM_BOUNDS[name]
        assert bounds['min'] <= v[name][0] <= bounds['max']
        assert bounds['min'] <= DEFAULT_PARAMS[name] <= bounds['max']
        gsflow = GSFLOW_PARAMS[f'gsflow_{name}']
        assert (bounds['min'], bounds['max']) == (gsflow.min, gsflow.max)
    assert PARAM_BOUNDS['tmax_allsnow']['max'] < PARAM_BOUNDS['tmax_allrain']['max']


# --- HRU area -----------------------------------------------------------------

def test_hru_areas_sum_to_basin_area(tmp_path):
    """Regression: every HRU got the whole basin area (basin_cfs nhru x too high)."""
    preprocessor(tmp_path, nhru=4)._generate_parameter_file()
    areas = values(tmp_path / 'params.dat')['hru_area']
    assert len(areas) == 4
    assert sum(areas) == pytest.approx(PROPS['area_km2'] * 247.105, rel=1e-4)


def test_single_hru_area_unchanged(tmp_path):
    preprocessor(tmp_path)._generate_parameter_file()
    assert values(tmp_path / 'params.dat')['hru_area'] == [pytest.approx(1145. * 247.105, abs=0.05)]


def test_hru_area_override_wins(tmp_path):
    preprocessor(tmp_path, nhru=2, parameter_overrides={
        'hru_area': {'dimensions': ['nhru'], 'values': [10.0, 30.0]}})._generate_parameter_file()
    assert values(tmp_path / 'params.dat')['hru_area'] == [10.0, 30.0]


# --- last forcing day ---------------------------------------------------------

def test_window_covers_whole_end_day():
    assert daily_forcing_window(datetime(2010, 1, 1), datetime(2010, 1, 3)) == (
        '2010-01-01 00:00:00', '2010-01-03 23:59:59')
    assert daily_forcing_window(datetime(2010, 1, 1), datetime(2010, 1, 3, 21))[1] == '2010-01-03 23:59:59'


def test_last_day_of_data_file_is_a_full_day(tmp_path, monkeypatch):
    """Regression: with an end date at midnight the last day held one
    timestep (precip ~1/24 of the day, tmax == tmin)."""
    times = pd.date_range('2010-01-01', '2010-01-04 23:00', freq='h')
    ds = xr.Dataset({
        'precipitation_flux': (('time',), np.full(len(times), 1e-4)),
        'air_temperature': (('time',), 273.15 + (times.hour.values % 24).astype(float)),
    }, coords={'time': times}, attrs={'timestep_seconds': 3600.})
    (tmp_path / 'basin').mkdir()
    ds.to_netcdf(tmp_path / 'basin' / 'forcing.nc')
    import symfluence.core.modeling.model_ready.forcing_reader as reader
    monkeypatch.setattr(reader, 'open_canonical_forcing', lambda files, **kw: xr.open_mfdataset(files))
    p = preprocessor(tmp_path)
    p.forcing_basin_path = tmp_path / 'basin'
    p.config.domain = NS(discretization=None)
    daily = p._load_forcing_data(datetime(2010, 1, 1), datetime(2010, 1, 3))
    assert list(daily.index.strftime('%Y-%m-%d')) == ['2010-01-01', '2010-01-02', '2010-01-03']
    assert daily['precip'].tolist() == pytest.approx([24 * 0.36] * 3)
    assert daily['tmax'].iloc[-1] == pytest.approx(23.0) and daily['tmin'].iloc[-1] == pytest.approx(0.0)
