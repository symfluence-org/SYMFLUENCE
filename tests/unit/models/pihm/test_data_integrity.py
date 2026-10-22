"""Regression checks for production forcing and interval-mean outlet flow."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ValidationError
from symfluence.models.pihm.extractor import PIHMResultExtractor
from symfluence.models.pihm.preprocessor import PIHMPreProcessor

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def test_missing_forcing_cannot_produce_synthetic_run(tmp_path):
    pre = PIHMPreProcessor({'PROJECT_DIR': str(tmp_path)}, logging.getLogger('test'))
    with pytest.raises(ValidationError, match='synthetic fallback is disabled'):
        pre.run_preprocessing()
    assert not list(tmp_path.rglob('*.meteo'))


def test_outlet_selection_sign_and_interval_end(tmp_path):
    path = tmp_path / 'test.river.flx1.txt'
    path.write_text('"2013-01-01 01:00"\t2\t-3\n"2013-01-01 02:00"\t4\t-5\n')
    q = PIHMResultExtractor().extract_variable(path, 'river_flux', outlet_segment=2)
    assert q.tolist() == [-3, -5]
    assert q.index.tolist() == list(pd.date_range('2013-01-01', periods=2, freq='h'))


def test_output_gap_rejected(tmp_path):
    path = tmp_path / 'test.river.flx1.txt'
    path.write_text('"2013-01-01 01:00"\t1\n"2013-01-01 02:00"\t2\n"2013-01-01 04:00"\t3\n')
    with pytest.raises(ValidationError, match='time gaps'):
        PIHMResultExtractor().extract_variable(path, 'river_flux')


@pytest.mark.parametrize('bad', ['gap', 'nan', 'units'])
def test_forcing_integrity(bad):
    times = pd.date_range('2013-01-01', periods=4, freq='3h')
    frame = pd.DataFrame(dict(time=times, prcp=0., temp=273., rh=80., wind=2.,
                              solar=100., longwave=250., pres=90000.))
    if bad == 'gap': frame = frame.drop(index=1)
    elif bad == 'nan': frame.loc[1, 'temp'] = np.nan
    else: frame.loc[1, 'temp'] = 1.  # Celsius mistakenly supplied as Kelvin
    with pytest.raises(ValidationError):
        PIHMPreProcessor._validate_forcing(frame, times[0], times[-1])


def test_multihru_requires_weights_and_preserves_volume(tmp_path, monkeypatch):
    import xarray as xr

    import symfluence.core.modeling.model_ready.forcing_reader as reader
    times = pd.date_range('2013-01-01', periods=4, freq='3h')
    vals = dict(precipitation_flux=[1e-5, 3e-5], air_temperature=[270., 274.],
                specific_humidity=[.003, .004], wind_speed=[2., 4.],
                surface_downwelling_shortwave_flux=[100., 200.],
                surface_downwelling_longwave_flux=[250., 270.],
                surface_air_pressure=[90000., 91000.])
    ds = xr.Dataset({k: (('time','hru'), np.tile(v, (4,1))) for k,v in vals.items()},
                    coords={'time':times})
    monkeypatch.setattr(reader, 'open_canonical_forcing', lambda _: ds.copy())
    pre = PIHMPreProcessor({'PROJECT_DIR':str(tmp_path)}, logging.getLogger('test'))
    with pytest.raises(ValidationError, match='explicit area weights'):
        pre._load_forcing_data([tmp_path/'dummy.nc'], times[0], times[-1])
    forcing = pre._load_forcing_data([tmp_path/'dummy.nc'], times[0], times[-1], hru_weights=[1,3])
    assert np.allclose(forcing.prcp, 2.5e-5)
    assert np.allclose(forcing.temp, 273.)


def test_snow17_parameters_cannot_be_silently_ignored(tmp_path):
    from symfluence.models.pihm.calibration.parameter_manager import PIHMParameterManager

    manager = PIHMParameterManager({}, logging.getLogger('test'), tmp_path)
    assert not manager.update_model_files({'SNOW17_MFMAX': 2.0})
    assert not list(tmp_path.iterdir())


def test_aquifer_depth_does_not_change_surface_infiltration_layer(tmp_path):
    from symfluence.models.pihm.calibration.parameter_manager import PIHMParameterManager

    pre = PIHMPreProcessor({'PROJECT_DIR': str(tmp_path)}, logging.getLogger('test'))
    pre._write_soil(tmp_path, 'basin', 1e-5, .45, 1., 2., 1e-4, .5, 4.)
    manager = PIHMParameterManager({}, logging.getLogger('test'), tmp_path)
    manager._update_soil_file({'SOIL_DEPTH': 8., 'K_SAT': 2e-5})
    rows = [line.split() for line in (tmp_path / 'basin.soil').read_text().splitlines()]
    assert float(next(row[1] for row in rows if row[0] == 'DINF')) == .1
    assert float(rows[2][rows[1].index('KSATV')]) == 2e-5
