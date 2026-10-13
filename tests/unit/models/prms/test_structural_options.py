"""PRMS routing chain, land/glacier depletion curves and climate_hru forcing."""
from datetime import datetime
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.prms.calibration.parameter_manager import PRMSParameterManager
from symfluence.models.prms.calibration.worker import PRMSWorker
from symfluence.models.prms.config_schema import PRMSConfig
from symfluence.models.prms.derived_parameters import apply_derived_parameters, read_derived_parameter
from symfluence.models.prms.preprocessor import PRMSPreProcessor

PROPS = dict(lat=64., lon=-19., elev=700., area_km2=1145., slope=5., aspect=0.)
ICE = {'glacier_frac_init': {'dimensions': ['nhru'], 'dtype': 2, 'values': [0, 0, 1]}}


def values(path):
    body = path.read_text().split('** Parameters **')[1]
    out = {}
    for b in body.split('####'):
        x = b.strip().splitlines()
        if x:
            n = int(x[1])
            out[x[0]] = [float(v) for v in x[4 + n:]]
    return out


def dims(path):
    head = path.read_text().split('** Parameters **')[0].split('** Dimensions **')[1]
    return {b.split()[0]: int(b.split()[1]) for b in head.split('####') if b.strip()}


def preprocessor(tmp_path, **cfg):
    p = object.__new__(PRMSPreProcessor)
    p.settings_dir = p.forcing_dir = tmp_path
    p.use_obs_solar = False
    p.config = NS(model=NS(prms=PRMSConfig(**cfg)))
    p._get_config_value = lambda getter, default=None, **kw: getter() if default is None else default
    p._get_catchment_properties = lambda: PROPS
    p._get_dominant_classes = lambda: dict(cov_type=0, soil_type=2)
    return p


def test_routing_chain_and_separate_curves(tmp_path):
    p = preprocessor(tmp_path, nhru=3, model_mode='PRMS5', glacier_flag=True, routing_segments=3,
                     separate_glacier_depletion_curve=True, parameter_overrides=ICE)
    p._generate_parameter_file()
    v, d = values(tmp_path / 'params.dat'), dims(tmp_path / 'params.dat')
    assert d['nsegment'] == 3 and d['ndepl'] == 2 and d['ndeplval'] == 22
    assert v['tosegment'] == [2, 3, 0] and v['hru_segment'] == [1, 1, 1]
    assert v['K_coef'] == [12.] * 3
    assert v['hru_deplcrv'] == [1, 1, 2]
    assert v['snarea_curve'][:11] == v['snarea_curve'][11:]


def test_separate_curves_need_land_and_glacier_hrus(tmp_path):
    p = preprocessor(tmp_path, nhru=3, glacier_flag=True, separate_glacier_depletion_curve=True)
    with pytest.raises(ValueError, match='pure land/glacier'):
        p._generate_parameter_file()


def test_multiple_reaches_require_muskingum(tmp_path):
    p = preprocessor(tmp_path, routing_segments=2, streamflow_module='strmflow')
    with pytest.raises(ValueError, match='muskingum'):
        p._generate_control_file(datetime(2010, 1, 1), datetime(2010, 12, 31))


def test_land_sca_max_is_idempotent_and_readable(tmp_path):
    p = preprocessor(tmp_path, nhru=3, glacier_flag=True, separate_glacier_depletion_curve=True,
                     parameter_overrides=ICE)
    p._generate_parameter_file()
    path = tmp_path / 'params.dat'
    text, native = apply_derived_parameters(path.read_text(), {'land_sca_max': 0.7, 'K_coef': 5.})
    assert native == {'K_coef': 5.}
    again, _ = apply_derived_parameters(text, {'land_sca_max': 0.7})
    assert again == text
    curve = values_from_text(text)['snarea_curve']
    assert curve[10] == pytest.approx(0.7) and curve[21] == pytest.approx(1.0)
    assert read_derived_parameter(text, 'land_sca_max') == pytest.approx(0.7)
    # worker and manager both route the derived parameter
    path.write_text(text)
    worker = object.__new__(PRMSWorker)
    worker.logger = NS(debug=lambda *a, **k: None, error=lambda *a, **k: None)
    assert worker._update_parameter_file(path, {'land_sca_max': 0.5, 'K_coef': 6.})
    v = values(path)
    assert v['snarea_curve'][10] == pytest.approx(0.5) and v['K_coef'] == [6.]
    manager = object.__new__(PRMSParameterManager)
    manager.settings_dir, manager.param_file = tmp_path, 'params.dat'
    manager.prms_params, manager.logger = ['land_sca_max', 'K_coef'], worker.logger
    assert manager.get_initial_parameters() == pytest.approx({'land_sca_max': 0.5, 'K_coef': 6.})


def test_land_sca_max_rejects_single_curve(tmp_path):
    p = preprocessor(tmp_path, nhru=3)
    p._generate_parameter_file()
    with pytest.raises(ValueError, match='separate land and glacier'):
        apply_derived_parameters((tmp_path / 'params.dat').read_text(), {'land_sca_max': 0.7})


def values_from_text(text):
    out = {}
    for b in text.split('** Parameters **')[1].split('####'):
        x = b.strip().splitlines()
        if x:
            out[x[0]] = [float(v) for v in x[4 + int(x[1]):]]
    return out


def write_hru_forcing(folder):
    folder.mkdir()
    times = pd.date_range('2010-01-01', '2010-01-02 21:00', freq='3h')
    for month, chunk in (('a', times[:8]), ('b', times[8:])):
        n = len(chunk)
        ds = xr.Dataset({
            'precipitation_flux': (('time', 'hru'), np.tile([1e-4, 2e-4], (n, 1))),
            'air_temperature': (('time', 'hru'), np.tile([273.15, 263.15], (n, 1)) + np.arange(n)[:, None]),
            'surface_downwelling_shortwave_flux': (('time', 'hru'), np.full((n, 2), 100.)),
            'hruId': (('hru',), [1, 2]),
        }, coords={'time': chunk}, attrs={'timestep_seconds': 10800.})
        ds.to_netcdf(folder / f'forcing_{month}.nc')


def test_climate_hru_forcing_maps_split_hrus(tmp_path):
    write_hru_forcing(tmp_path / 'hru')
    p = preprocessor(tmp_path, nhru=3, forcing_mode='climate_hru', hru_forcing_path=str(tmp_path / 'hru'),
                     hru_forcing_map=[1, 2, 2])
    p._write_cbh_files(datetime(2010, 1, 1), datetime(2010, 1, 2, 23))
    lines = (tmp_path / 'precip.cbh').read_text().splitlines()
    assert lines[1] == 'precip 3' and lines[2].startswith('####')
    day1 = [float(x) for x in lines[3].split()[6:]]
    assert day1 == pytest.approx([8 * 1e-4 * 10800, 8 * 2e-4 * 10800, 8 * 2e-4 * 10800])
    tmax = [float(x) for x in (tmp_path / 'tmax.cbh').read_text().splitlines()[3].split()[6:]]
    assert tmax == pytest.approx([7., -3., -3.])
    swrad = [float(x) for x in (tmp_path / 'swrad.cbh').read_text().splitlines()[3].split()[6:]]
    assert swrad == pytest.approx([206.51] * 3)
    p._generate_control_file(datetime(2010, 1, 1), datetime(2010, 1, 2))
    control = (tmp_path / 'control.dat').read_text()
    assert 'precip_module\n1\n4\nclimate_hru\n' in control
    assert f'precip_day\n1\n4\n{tmp_path / "precip.cbh"}\n' in control
    p._generate_parameter_file()
    v = values(tmp_path / 'params.dat')
    assert v['rain_cbh_adj'] == [1.] * 36 and v['tmax_cbh_adj'] == [0.] * 36


def test_climate_hru_rejects_missing_source_hru(tmp_path):
    write_hru_forcing(tmp_path / 'hru')
    p = preprocessor(tmp_path, nhru=2, forcing_mode='climate_hru', hru_forcing_path=str(tmp_path / 'hru'),
                     hru_forcing_map=[1, 5])
    with pytest.raises(ValueError, match='lacks source hruIds'):
        p._write_cbh_files(datetime(2010, 1, 1), datetime(2010, 1, 2, 23))
