"""Lumped Wflow: the scored discharge and the saved discharge are one series.

Two defects, one cause (worker and postprocessor each had their own copy):

- the worker took the model step from ``WFLOW_TIMESTEP`` (a key nothing sets,
  default 3600 s), so with 3-hourly forcing calibration scored Q three times
  too high (24x for daily) while the postprocessor read ``timestepsecs``;
- the ``ROUTE_*`` post-hoc routing ran only inside the worker, so the saved
  ``WFLOW_discharge_cms`` was not the series that had been scored.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.mixins.config import SCHEMA_DEFAULT, fallback_default
from symfluence.models.wflow import lumped_discharge as ld
from symfluence.models.wflow.calibration.parameter_manager import WflowParameterManager
from symfluence.models.wflow.calibration.worker import WflowWorker
from symfluence.models.wflow.postprocessor import WflowPostProcessor
from symfluence.models.wflow.preprocessor import WflowPreProcessor

AREA = 250e6   # m2
ROUTING = {'ROUTE_ALPHA': 0.6, 'ROUTE_BETA': 0.97, 'ROUTE_SPLIT': 0.4, 'ROUTE_BASEFLOW': 1.5}


def _toml(step, parameter='soil_surface_water__net_runoff_volume_flux'):
    return (f'[time]\ntimestepsecs = {step}\n'
            f'[[output.csv.column]]\nheader = "Q"\nparameter = "{parameter}"\nreducer = "mean"\n')


def _domain(tmp_path, step=10800, n=48):
    """A lumped run: settings (TOML + staticmaps) and an output.csv in mm/step."""
    settings = tmp_path / 'domain_test' / 'settings' / 'WFLOW'
    out = tmp_path / 'domain_test' / 'simulations' / 'run_1' / 'WFLOW'
    settings.mkdir(parents=True)
    out.mkdir(parents=True)
    (settings / 'wflow_sbm.toml').write_text(_toml(step))
    cell = np.full((3, 3), np.nan)
    cell[1, 1] = AREA
    xr.Dataset({'wflow_cellarea': (('y', 'x'), cell),
                'KsatVer': (('y', 'x'), np.where(np.isnan(cell), np.nan, 250.0))}
               ).to_netcdf(settings / 'wflow_staticmaps.nc')
    times = pd.date_range('2020-01-01', periods=n, freq=f'{step}s')
    mm = 0.5 + np.abs(np.sin(np.arange(n) / 5.0))
    pd.DataFrame({'time': times, 'Q': mm}).to_csv(out / 'output.csv', index=False)
    config = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'DOMAIN_NAME': 'test'}
    return settings, out, config, pd.Series(mm, index=times)


def _postprocessor(tmp_path, saved):
    post = WflowPostProcessor.__new__(WflowPostProcessor)
    post.logger = logging.getLogger('wflow-post')
    post.project_dir = tmp_path / 'domain_test'
    post._get_config_value = lambda accessor, default=SCHEMA_DEFAULT, dict_key=None: fallback_default(default, dict_key)
    post._get_output_dir = lambda: tmp_path / 'domain_test' / 'simulations' / 'run_1' / 'WFLOW'
    post._align_time_labels = lambda q: q
    post.resample_frequency = None
    post.save_streamflow_to_results = lambda q, model_column_name: saved.setdefault(model_column_name, q)
    return post


@pytest.mark.parametrize('step', [3600, 10800, 86400])
def test_worker_takes_the_step_from_the_toml(tmp_path, step):
    _, out, config, mm = _domain(tmp_path, step=step)
    q = WflowWorker(config, logging.getLogger('w'))._load_simulated_streamflow(out, config)
    np.testing.assert_allclose(q.values, mm.values * AREA / (step * 1000.0))


def test_worker_ignores_the_phantom_wflow_timestep_key(tmp_path):
    _, out, config, mm = _domain(tmp_path, step=10800)
    q = WflowWorker(config, logging.getLogger('w'))._load_simulated_streamflow(
        out, dict(config, WFLOW_TIMESTEP=3600))
    np.testing.assert_allclose(q.values, mm.values * AREA / (10800 * 1000.0))


def test_missing_step_or_area_is_an_error_not_an_hourly_guess(tmp_path):
    toml = tmp_path / 'wflow_sbm.toml'
    toml.write_text('[time]\nstarttime = 2020-01-01T00:00:00\n')
    with pytest.raises(ConfigurationError, match='timestepsecs'):
        ld.model_timestep_seconds(toml)
    with pytest.raises(ConfigurationError, match='staticmaps'):
        ld.basin_area_m2(tmp_path / 'absent.nc')


def test_saved_series_is_the_scored_series(tmp_path):
    """Postprocessor output == what the worker scored, routing included."""
    settings, out, config, _ = _domain(tmp_path)
    worker = WflowWorker(config, logging.getLogger('w'))
    params = dict(ROUTING, KsatVer=300.0)
    assert worker.apply_parameters(params, settings, config=config)
    scored = worker._load_simulated_streamflow(out, config, worker._routing_params)

    saved = {}
    _postprocessor(tmp_path, saved).extract_streamflow()
    np.testing.assert_allclose(saved['WFLOW_discharge_cms'].values, scored.values, rtol=1e-12)

    # ...and it really is routed: the unrouted conversion differs.
    unrouted = worker._load_simulated_streamflow(out, config, None)
    assert not np.allclose(scored.values, unrouted.values)
    assert scored.iloc[-1] != pytest.approx(unrouted.iloc[-1], rel=0.01)


def test_parameter_set_without_routing_clears_the_sidecar(tmp_path):
    settings, out, config, mm = _domain(tmp_path)
    pm = WflowParameterManager(config, logging.getLogger('pm'), settings)
    assert pm.update_model_files(dict(ROUTING, KsatVer=300.0), settings)
    assert ld.load_routing_params(settings) == ROUTING
    assert pm.update_model_files({'KsatVer': 300.0}, settings)
    assert ld.load_routing_params(settings) is None

    saved = {}
    _postprocessor(tmp_path, saved).extract_streamflow()
    np.testing.assert_allclose(saved['WFLOW_discharge_cms'].values, mm.values * AREA / (10800 * 1000.0))


def test_distributed_discharge_is_left_alone(tmp_path):
    settings, out, config, mm = _domain(tmp_path)
    (settings / 'wflow_sbm.toml').write_text(_toml(86400, 'river_water__volume_flow_rate'))
    ld.save_routing_params(settings, ROUTING)   # must not be applied to river discharge
    q = WflowWorker(config, logging.getLogger('w'))._load_simulated_streamflow(out, config, ROUTING)
    np.testing.assert_allclose(q.values, mm.values, rtol=1e-12)
    saved = {}
    _postprocessor(tmp_path, saved).extract_streamflow()
    np.testing.assert_allclose(saved['WFLOW_discharge_cms'].values, mm.values, rtol=1e-12)


def test_routing_conserves_volume_minus_storage():
    q = pd.Series(np.r_[np.ones(10), np.zeros(4000)])
    out = ld.apply_routing(q, {'ROUTE_ALPHA': 0.5, 'ROUTE_BETA': 0.99, 'ROUTE_SPLIT': 0.5, 'ROUTE_BASEFLOW': 0.0})
    assert out.sum() == pytest.approx(q.sum(), rel=1e-6)


# --- Hamon PET through the shared implementation ------------------------------

def _pre(method, step_seconds):
    pre = WflowPreProcessor.__new__(WflowPreProcessor)
    pre.logger = logging.getLogger('wflow-pre')
    values = {'WFLOW_PET_METHOD': method, 'FORCING_TIME_STEP_SIZE': step_seconds}
    pre._get_config_value = lambda accessor, default=SCHEMA_DEFAULT, dict_key=None: (
        values[dict_key] if dict_key in values else fallback_default(default, dict_key))
    return pre


def test_hamon_is_the_lu_reference_not_a_tenth_of_it():
    times = pd.DatetimeIndex(['2021-03-21'])
    pet = _pre('hamon', 86400)._estimate_pet(np.array([15.0]), times, 45.0)
    assert pet[0] == pytest.approx(2.12, rel=0.01)       # was 0.21 (kPa fed to an hPa formula)
    per_step = _pre('hamon', 10800)._estimate_pet(np.array([15.0]), times, 45.0)
    assert per_step[0] == pytest.approx(pet[0] / 8.0)


def test_oudin_is_unchanged():
    from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
    times = pd.date_range('2021-06-01', periods=5)
    temp = np.array([-8.0, 0.0, 5.0, 12.0, 20.0])
    pet = _pre('oudin', 86400)._estimate_pet(temp, times, 64.0)
    np.testing.assert_array_equal(
        pet, PETCalculatorMixin.oudin_pet_numpy(temp, np.asarray(times.dayofyear), 64.0))
    assert not hasattr(WflowPreProcessor, '_estimate_pet_hamon')


# --- Lumped Q is the whole outflow of the cell --------------------------------
#
# One-cell Wflow.jl 1.0.4 run (daily and 3-hourly, 4 years): P - ET - dS equals
# net_runoff + river lateral inflow + lateral subsurface flow to 1e-12 mm per
# step; net runoff alone was 93 % of it.

LAT = 64.0
#: Wflow's own area of a 0.01 degree cell at 64 degN (utils.jl lattometres).
WFLOW_CELL_AREA = 545477.588


def _full_toml(step, extra=''):
    text = _toml(step)
    for header, (parameter, _) in ld.LUMPED_OUTFLOW_COLUMNS.items():
        text += f'[[output.csv.column]]\nheader = "{header}"\nparameter = "{parameter}"\nreducer = "mean"\n'
    return text + extra


def _full_domain(tmp_path, step=10800, n=48):
    """A lumped run whose TOML and CSV carry the three outflow columns."""
    settings, out, config, mm = _domain(tmp_path, step=step, n=n)
    (settings / 'wflow_sbm.toml').write_text(_full_toml(step))
    cell = np.full((3, 3), np.nan)
    cell[1, 1] = AREA
    xr.Dataset(
        {'wflow_cellarea': (('y', 'x'), cell),
         'wflow_subcatch': (('y', 'x'), np.where(np.isnan(cell), np.nan, 1.0)),
         'KsatVer': (('y', 'x'), np.where(np.isnan(cell), np.nan, 250.0))},
        coords={'y': [LAT - 0.01, LAT, LAT + 0.01], 'x': [-19.01, -19.0, -18.99]},
    ).to_netcdf(settings / 'wflow_staticmaps.nc')
    river = 0.002 + 0.001 * np.cos(np.arange(n) / 7.0)        # m3/s on the Wflow cell
    ssf = 30.0 + 5.0 * np.sin(np.arange(n) / 11.0)            # m3/d on the Wflow cell
    table = pd.DataFrame({'Q': mm.values, 'Q_river_inflow': river, 'Q_subsurface': ssf}, index=mm.index)
    table.rename_axis('time').to_csv(out / 'output.csv')
    return settings, out, config, table


def test_wflow_cell_area_follows_wflow_not_the_basin_area(tmp_path):
    settings, _, _, _ = _full_domain(tmp_path)
    toml = settings / 'wflow_sbm.toml'
    area = ld.wflow_cell_area_m2(settings / 'wflow_staticmaps.nc', toml)
    assert area == pytest.approx(WFLOW_CELL_AREA, rel=1e-6)
    toml.write_text(_full_toml(10800, '[model]\ncell_length_in_meter__flag = true\n'))
    assert ld.wflow_cell_area_m2(settings / 'wflow_staticmaps.nc', toml) == pytest.approx(1e-4)


@pytest.mark.parametrize('step', [10800, 86400])
def test_lumped_outflow_adds_river_fraction_and_subsurface_flow(tmp_path, step):
    settings, _, _, table = _full_domain(tmp_path, step=step)
    total = ld.lumped_outflow_mm_per_step(table, settings / 'wflow_sbm.toml', settings / 'wflow_staticmaps.nc')
    river_mm = table['Q_river_inflow'] * step / WFLOW_CELL_AREA * 1000.0             # m3/s -> mm/step
    ssf_mm = table['Q_subsurface'] * (step / 86400.0) / WFLOW_CELL_AREA * 1000.0     # m3/d -> mm/step
    np.testing.assert_allclose(total.values, (table['Q'] + river_mm + ssf_mm).values, rtol=1e-6)
    assert (total > table['Q']).all()

    q = ld.lumped_discharge_cms(table, settings / 'wflow_sbm.toml', settings / 'wflow_staticmaps.nc')
    np.testing.assert_allclose(q.values, total.values * AREA / (step * 1000.0), rtol=1e-12)


def test_worker_and_postprocessor_both_use_the_full_outflow(tmp_path):
    settings, out, config, table = _full_domain(tmp_path)
    expected = ld.lumped_discharge_cms(table, settings / 'wflow_sbm.toml', settings / 'wflow_staticmaps.nc')
    net_only = table['Q'] * AREA / (10800 * 1000.0)
    assert (expected > net_only).all()

    scored = WflowWorker(config, logging.getLogger('w'))._load_simulated_streamflow(out, config)
    np.testing.assert_allclose(scored.values, expected.values, rtol=1e-9)
    saved = {}
    _postprocessor(tmp_path, saved).extract_streamflow()
    np.testing.assert_allclose(saved['WFLOW_discharge_cms'].values, scored.values, rtol=1e-12)


def test_settings_from_before_the_columns_keep_net_runoff_and_say_so(tmp_path, caplog):
    settings, _, _, mm = _domain(tmp_path)
    ld._warned_legacy_tomls.clear()
    with caplog.at_level(logging.WARNING, logger=ld.logger.name):
        total = ld.lumped_outflow_mm_per_step(mm.rename('Q'), settings / 'wflow_sbm.toml',
                                              settings / 'wflow_staticmaps.nc')
        ld.lumped_outflow_mm_per_step(mm.rename('Q'), settings / 'wflow_sbm.toml',
                                      settings / 'wflow_staticmaps.nc')
    np.testing.assert_array_equal(total.values, mm.values)
    warnings = [r for r in caplog.records if 'Regenerate the Wflow settings' in r.getMessage()]
    assert len(warnings) == 1          # once per TOML, not once per calibration iteration


def test_declared_columns_missing_from_the_output_are_an_error(tmp_path):
    settings, _, _, table = _full_domain(tmp_path)
    toml, static = settings / 'wflow_sbm.toml', settings / 'wflow_staticmaps.nc'
    with pytest.raises(ConfigurationError, match='does not contain'):
        ld.lumped_outflow_mm_per_step(table[['Q', 'Q_river_inflow']], toml, static)
    with pytest.raises(ConfigurationError, match='does not contain'):
        ld.lumped_outflow_mm_per_step(table['Q'], toml, static)
    toml.write_text(_toml(10800) + '[[output.csv.column]]\nheader = "Q_subsurface"\n'
                    'parameter = "subsurface_water__volume_flow_rate"\nreducer = "mean"\n')
    with pytest.raises(ConfigurationError, match='needs both'):
        ld.lumped_outflow_mm_per_step(table, toml, static)


def test_more_than_one_active_cell_is_not_a_lumped_run(tmp_path):
    settings, _, _, _ = _full_domain(tmp_path)
    static = settings / 'wflow_staticmaps.nc'
    with xr.open_dataset(static) as ds:
        ds = ds.load()
    ds['wflow_subcatch'][0, 0] = 1.0
    ds.to_netcdf(static)
    with pytest.raises(ConfigurationError, match='exactly one active cell'):
        ld.wflow_cell_area_m2(static, settings / 'wflow_sbm.toml')


@pytest.mark.parametrize('mode, expected', [('lumped', True), ('distributed', False)])
def test_generated_toml_writes_the_outflow_columns_in_lumped_mode_only(tmp_path, mode, expected):
    import tomllib
    from datetime import datetime

    pre = WflowPreProcessor.__new__(WflowPreProcessor)
    pre.logger = logging.getLogger('wflow-pre')
    pre.spatial_mode = mode
    pre.settings_dir = tmp_path / 'settings'
    pre.forcing_out_dir = tmp_path / 'forcing'
    pre.settings_dir.mkdir()
    values = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'DOMAIN_NAME': 'test', 'FORCING_TIME_STEP_SIZE': 86400}
    pre._get_config_value = lambda accessor, default=SCHEMA_DEFAULT, dict_key=None: (
        values[dict_key] if dict_key in values else fallback_default(default, dict_key))
    pre._get_simulation_dates = lambda: (datetime(2020, 1, 1), datetime(2020, 12, 31))
    pre._soil_layer_thickness = lambda: [100, 300, 800]
    pre._generate_toml_config()

    toml = pre.settings_dir / 'wflow_sbm.toml'
    columns = {c['header']: c['parameter'] for c in tomllib.loads(toml.read_text())['output']['csv']['column']}
    assert ld.is_runoff_flux_output(toml) is expected
    for header, (parameter, _) in ld.LUMPED_OUTFLOW_COLUMNS.items():
        assert (columns.get(header) == parameter) is expected
