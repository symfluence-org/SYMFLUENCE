"""Elevation corrections of remapped forcing (the ``lapse`` config section)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.config.models.forcing import LapseRateConfig
from symfluence.core.exceptions import ConfigurationError
from symfluence.data.preprocessing.resampling import elevation_correction as ec
from symfluence.data.preprocessing.resampling.elevation_correction import (
    ElevationCorrectionSettings,
    apply_elevation_corrections,
    hru_elevation_differences,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _config(lapse=None, **overrides):
    cfg = SymfluenceConfig.from_minimal(
        domain_name="test_basin", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
        **overrides,
    )
    if lapse is not None:
        forcing = cfg.forcing.model_copy(update={"lapse": LapseRateConfig(**lapse)})
        cfg = cfg.model_copy(update={"forcing": forcing})
    return cfg


def _forcing_file(path, n_time=4):
    times = pd.date_range('2020-01-01', periods=n_time, freq='h')
    shape = (n_time, 2)
    ds = xr.Dataset(
        {
            'hruId': ('hru', np.array([1.0, 2.0])),
            'air_temperature': (('time', 'hru'), np.full(shape, 278.15, dtype='float32')),
            'surface_air_pressure': (('time', 'hru'), np.full(shape, 90000.0, dtype='float32')),
            'specific_humidity': (('time', 'hru'), np.full(shape, 0.004, dtype='float32')),
            'surface_downwelling_longwave_flux': (('time', 'hru'), np.full(shape, 300.0, dtype='float32')),
            'surface_downwelling_shortwave_flux': (('time', 'hru'), np.full(shape, 200.0, dtype='float32')),
            'precipitation_flux': (('time', 'hru'), np.full(shape, 1e-4, dtype='float32')),
            'wind_speed': (('time', 'hru'), np.full(shape, 3.0, dtype='float32')),
        },
        coords={'time': times},
    )
    ds.to_netcdf(path)
    return path


DZ = pd.Series({1: 0.0, 2: 500.0})


# --- settings -------------------------------------------------------------

def test_inactive_settings_have_an_empty_signature():
    """Existing caches must keep their remapping signature."""
    assert ElevationCorrectionSettings.from_config(_config()).signature() == ''
    assert ElevationCorrectionSettings.from_config(_config({'lw_gradient': -0.03})).signature() == ''


def test_signature_follows_the_active_settings():
    a = ElevationCorrectionSettings.from_config(_config({'precipitation': True}))
    b = ElevationCorrectionSettings.from_config(_config({'precipitation': True, 'precip_gradient': 0.0002}))
    assert a.active and a.signature() and a.signature() != b.signature()
    assert ElevationCorrectionSettings.from_dict(a.to_dict()) == a


@pytest.mark.parametrize('raw, apply, expected', [
    (0.0065, True, 0.0065), (6.5, True, 0.0065), (0.0065, False, 0.0)])
def test_temperature_lapse_rate_comes_from_the_forcing_settings(raw, apply, expected):
    cfg = _config({'pressure': True}, LAPSE_RATE=raw, APPLY_LAPSE_RATE=apply)
    assert ElevationCorrectionSettings.from_config(cfg).temperature_lapse_rate == pytest.approx(expected)


# --- elevation differences --------------------------------------------------

def test_dz_is_hru_minus_weighted_forcing_elevation(tmp_path):
    table = tmp_path / 'd_ERA5_intersected_shapefile.csv'
    pd.DataFrame({
        'S_1_HRU_ID': [1, 1, 2],
        'S_1_elev_m': [1000.0, 1000.0, 400.0],
        'S_2_elev_m': [800.0, 600.0, 400.0],
        'weight': [0.75, 0.25, 1.0],
        'geometry': ['x', 'y', 'z'],
    }).to_csv(table, index=False)
    dz = hru_elevation_differences(ec.find_intersection_table(tmp_path / 'w_remapping.csv'), 'HRU_ID')
    assert dz.loc[1] == pytest.approx(1000.0 - (0.75 * 800.0 + 0.25 * 600.0))
    assert dz.loc[2] == pytest.approx(0.0)


def test_invalid_forcing_elevation_gives_no_correction(tmp_path, caplog):
    table = tmp_path / 'd_ERA5_intersected_shapefile.csv'
    pd.DataFrame({
        'S_1_HRU_ID': [1, 2], 'S_1_elev_m': [1400.0, 900.0],
        'S_2_elev_m': [0.0, -9999.0], 'weight': [1.0, 1.0],
    }).to_csv(table, index=False)
    with caplog.at_level(logging.WARNING):
        dz = hru_elevation_differences(table, 'HRU_ID')
    assert dz.tolist() == [0.0, 0.0]
    assert 'invalid elevation' in caplog.text


def test_missing_elevation_columns_raise(tmp_path):
    table = tmp_path / 'd_ERA5_intersected_shapefile.csv'
    pd.DataFrame({'S_1_HRU_ID': [1], 'S_2_elev_m': [5.0], 'weight': [1.0]}).to_csv(table, index=False)
    with pytest.raises(ConfigurationError, match='S_1_elev_m'):
        hru_elevation_differences(table, 'HRU_ID')
    with pytest.raises(ConfigurationError, match='intersected shapefile'):
        ec.find_intersection_table(tmp_path / 'elsewhere' / 'w_remapping.csv')


# --- physics ---------------------------------------------------------------

def test_pressure_drops_about_twelve_hpa_per_hundred_metres():
    p = ec.correct_pressure(101325.0, 288.15, 100.0, 0.0065)
    assert 101325.0 - p == pytest.approx(1194.0, abs=15.0)
    assert ec.correct_pressure(90000.0, 280.0, 0.0, 0.0065) == pytest.approx(90000.0)


def test_humidity_conserves_relative_humidity():
    q, p0, t0 = 0.004, 90000.0, 278.15
    p1, t1 = ec.correct_pressure(p0, t0, 500.0, 0.0065), t0 - 0.0065 * 500.0
    q1 = ec.correct_specific_humidity(q, p0, t0, p1, t1)

    def rel_hum(q_, p_, t_):
        return (q_ * p_ / (0.622 + 0.378 * q_)) / ec.saturation_vapour_pressure(t_)

    assert rel_hum(q1, p1, t1) == pytest.approx(rel_hum(q, p0, t0))
    assert q1 < q  # colder air at the same relative humidity holds less water


# --- application -------------------------------------------------------------

def test_only_requested_variables_change_and_only_where_dz_is_nonzero(tmp_path):
    path = _forcing_file(tmp_path / 'f.nc')
    settings = ElevationCorrectionSettings(longwave=True, precipitation=True)
    assert apply_elevation_corrections(path, DZ, settings) == ['longwave', 'precipitation']
    with xr.open_dataset(path) as ds:
        np.testing.assert_allclose(ds.surface_downwelling_longwave_flux.values[:, 0], 300.0)
        np.testing.assert_allclose(ds.surface_downwelling_longwave_flux.values[:, 1], 300.0 - 0.029 * 500.0)
        np.testing.assert_allclose(ds.precipitation_flux.values[:, 0], 1e-4, rtol=1e-6)
        np.testing.assert_allclose(ds.precipitation_flux.values[:, 1], 1e-4 * 1.2, rtol=1e-6)
        for untouched, value in (('air_temperature', 278.15), ('surface_air_pressure', 90000.0),
                                 ('specific_humidity', 0.004), ('wind_speed', 3.0),
                                 ('surface_downwelling_shortwave_flux', 200.0)):
            np.testing.assert_allclose(ds[untouched].values, value, rtol=1e-6)
        record = json.loads(ds.attrs[ec.ATTRIBUTE])
    assert record['applied'] == ['longwave', 'precipitation']
    assert record['settings']['precip_gradient'] == 0.0004


def test_all_corrections_and_temperature_is_never_written(tmp_path):
    path = _forcing_file(tmp_path / 'f.nc')
    settings = ElevationCorrectionSettings(
        pressure=True, humidity=True, longwave=True, shortwave=True, precipitation=True,
        wind=True, wind_gradient=0.0002, temperature_lapse_rate=0.0065)
    assert apply_elevation_corrections(path, DZ, settings) == list(ec.TOGGLES)
    with xr.open_dataset(path) as ds:
        np.testing.assert_allclose(ds.air_temperature.values, 278.15, rtol=1e-6)
        expected_p = ec.correct_pressure(90000.0, 278.15, 500.0, 0.0065)
        np.testing.assert_allclose(ds.surface_air_pressure.values[:, 1], expected_p, rtol=1e-6)
        np.testing.assert_allclose(ds.surface_air_pressure.values[:, 0], 90000.0)
        np.testing.assert_allclose(
            ds.specific_humidity.values[:, 1],
            ec.correct_specific_humidity(0.004, 90000.0, 278.15, expected_p, 278.15 - 3.25), rtol=1e-5)
        np.testing.assert_allclose(ds.specific_humidity.values[:, 0], 0.004, rtol=1e-6)
        np.testing.assert_allclose(ds.surface_downwelling_shortwave_flux.values[:, 1], 200.0 * 1.01, rtol=1e-6)
        np.testing.assert_allclose(ds.wind_speed.values[:, 1], 3.0 * 1.1, rtol=1e-6)


def test_scaling_is_floored_at_zero(tmp_path):
    path = _forcing_file(tmp_path / 'f.nc')
    settings = ElevationCorrectionSettings(precipitation=True, longwave=True, lw_gradient=-1.0)
    apply_elevation_corrections(path, pd.Series({1: -5000.0, 2: 500.0}), settings)
    with xr.open_dataset(path) as ds:
        assert (ds.precipitation_flux.values[:, 0] == 0.0).all()
        assert (ds.surface_downwelling_longwave_flux.values[:, 1] == 0.0).all()


def test_second_application_is_refused(tmp_path):
    path = _forcing_file(tmp_path / 'f.nc')
    settings = ElevationCorrectionSettings(longwave=True)
    apply_elevation_corrections(path, DZ, settings)
    with pytest.raises(ValueError, match='already carries'):
        apply_elevation_corrections(path, DZ, settings)


def test_inactive_settings_leave_the_file_alone(tmp_path):
    path = _forcing_file(tmp_path / 'f.nc')
    before = path.read_bytes()
    assert apply_elevation_corrections(path, DZ, ElevationCorrectionSettings()) == []
    assert path.read_bytes() == before


def test_missing_variable_is_reported_not_invented(tmp_path, caplog):
    path = tmp_path / 'f.nc'
    with xr.open_dataset(_forcing_file(tmp_path / 'full.nc')) as ds:
        ds.drop_vars('wind_speed').to_netcdf(path)
    with caplog.at_level(logging.WARNING):
        applied = apply_elevation_corrections(
            path, DZ, ElevationCorrectionSettings(wind=True, shortwave=True))
    assert applied == ['shortwave'] and 'wind' in caplog.text


def test_unknown_hru_raises(tmp_path):
    path = _forcing_file(tmp_path / 'f.nc')
    with pytest.raises(ValueError, match='no elevation'):
        apply_elevation_corrections(path, pd.Series({1: 0.0}), ElevationCorrectionSettings(longwave=True))


# --- applier wiring ------------------------------------------------------------

def test_weight_applier_corrects_staged_files(tmp_path):
    from symfluence.data.preprocessing.resampling.weight_applier import RemappingWeightApplier

    applier = RemappingWeightApplier(_config(), tmp_path, tmp_path, None, logging.getLogger('t'))
    applier.cached_hru_field = 'HRU_ID'
    cache = tmp_path / 'cache'
    staged = tmp_path / 'staged'
    cache.mkdir()
    staged.mkdir()
    pd.DataFrame({'S_1_HRU_ID': [1, 2], 'S_1_elev_m': [500.0, 1000.0],
                  'S_2_elev_m': [500.0, 500.0], 'weight': [1.0, 1.0]}).to_csv(
        cache / 'd_ERA5_intersected_shapefile.csv', index=False)
    path = _forcing_file(staged / 'f.nc')

    applier._apply_elevation_corrections(staged, cache / 'w_remapping.csv')  # inactive: no-op
    with xr.open_dataset(path) as ds:
        assert ec.ATTRIBUTE not in ds.attrs

    applier.elevation_corrections = ElevationCorrectionSettings(precipitation=True)
    applier._apply_elevation_corrections(staged, cache / 'w_remapping.csv')
    with xr.open_dataset(path) as ds:
        np.testing.assert_allclose(ds.precipitation_flux.values[:, 1], 1.2e-4, rtol=1e-6)
        np.testing.assert_allclose(ds.precipitation_flux.values[:, 0], 1e-4, rtol=1e-6)


# --- config loading -------------------------------------------------------------

_FLAT = dict(
    DOMAIN_NAME='x', EXPERIMENT_ID='e', EXPERIMENT_TIME_START='2012-01-01 00:00',
    EXPERIMENT_TIME_END='2012-02-01 00:00', DOMAIN_DEFINITION_METHOD='lumped',
    SUB_GRID_DISCRETIZATION='lumped', HYDROLOGICAL_MODEL='SUMMA', FORCING_DATASET='ERA5',
)
_NESTED = dict(
    domain=dict(name='x', experiment_id='e', time_start='2012-01-01 00:00',
                time_end='2012-02-01 00:00', definition_method='lumped', discretization='lumped'),
    model=dict(hydrological_model='SUMMA'),
)


def _load(tmp_path, content, caplog):
    import yaml
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(content))
    with caplog.at_level(logging.WARNING):
        cfg = SymfluenceConfig.from_file(path, use_env=False)
    assert 'Unrecognized configuration key' not in caplog.text
    return cfg


@pytest.mark.parametrize('layout', ['flat', 'nested_top_level', 'nested_under_forcing'])
def test_lapse_section_is_read_from_a_config_file(tmp_path, caplog, layout):
    """A top-level ``lapse`` section used to be dropped with only a warning."""
    lapse = {'pressure': True, 'precip_gradient': 0.0002}
    system = dict(SYMFLUENCE_DATA_DIR=str(tmp_path), SYMFLUENCE_CODE_DIR=str(tmp_path))
    if layout == 'flat':
        content = {**_FLAT, **system, 'lapse': lapse}
    else:
        content = {**_NESTED, 'system': dict(data_dir=str(tmp_path), code_dir=str(tmp_path)),
                   'forcing': {'dataset': 'ERA5'}}
        if layout == 'nested_top_level':
            content['lapse'] = lapse
        else:
            content['forcing']['lapse'] = lapse
    cfg = _load(tmp_path, content, caplog)
    assert cfg.forcing.lapse.pressure is True
    assert cfg.forcing.lapse.precip_gradient == 0.0002
    assert ElevationCorrectionSettings.from_config(cfg).requested == ['pressure']
