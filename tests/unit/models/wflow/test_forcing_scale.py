"""Precipitation correction is applied through Wflow's native forcing scale."""
from __future__ import annotations

import tomllib

import pytest

from symfluence.models.wflow.calibration.parameter_manager import WflowParameterManager

TOML = '''[input.forcing]
atmosphere_water__precipitation_volume_flux = "precip"
atmosphere_air__temperature = "temp"
'''
NAME = 'atmosphere_water__precipitation_volume_flux'


def test_scale_written_and_idempotent(tmp_path):
    path = tmp_path / 'wflow_sbm.toml'
    path.write_text(TOML)
    WflowParameterManager.set_forcing_scales(path, {'PCORR': 1.25})
    WflowParameterManager.set_forcing_scales(path, {'PCORR': 1.1})
    forcing = tomllib.loads(path.read_text())['input']['forcing']
    assert forcing[NAME] == {'netcdf_variable_name': 'precip', 'scale': 1.1}
    assert forcing['atmosphere_air__temperature'] == 'temp'


def test_missing_entry_raises(tmp_path):
    path = tmp_path / 'wflow_sbm.toml'
    path.write_text('[input.forcing]\natmosphere_air__temperature = "temp"\n')
    with pytest.raises(ValueError):
        WflowParameterManager.set_forcing_scales(path, {'PCORR': 1.2})
