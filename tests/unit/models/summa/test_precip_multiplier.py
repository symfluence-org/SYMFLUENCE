from __future__ import annotations

import json
import logging

import netCDF4 as nc
import numpy as np

from symfluence.models.summa.calibration.forcing_scaling import (
    SCALED_DIR_NAME,
    SOURCE_RECORD,
    apply_precip_multiplier,
    split_forcing_params,
)
from symfluence.models.summa.calibration.worker_impl import _apply_parameters_worker


def _domain(tmp_path):
    settings = tmp_path / 'settings/SUMMA'
    forcing = tmp_path / 'forcing'
    settings.mkdir(parents=True)
    forcing.mkdir()
    for i, name in enumerate(['a.nc', 'b.nc']):
        with nc.Dataset(forcing / name, 'w') as ds:
            ds.createDimension('time', 4)
            ds.createDimension('hru', 3)
            ds.createVariable('pptrate', 'f8', ('time', 'hru'))[:] = np.arange(12.).reshape(4, 3) + i
            ds.createVariable('airtemp', 'f8', ('time', 'hru'))[:] = 270.
    (settings / 'forcingFileList.txt').write_text("a.nc\nb.nc\n")
    (settings / 'fileManager.txt').write_text(
        "controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
        f"forcingPath          '{forcing}/'\n"
        "forcingListFile      'forcingFileList.txt'\n")
    return settings, forcing


def _ppt(path):
    with nc.Dataset(path) as ds:
        return ds['pptrate'][:].copy(), ds['airtemp'][:].copy()


def test_scales_total_precipitation_from_unmodified_source(tmp_path):
    settings, forcing = _domain(tmp_path)
    original, _ = _ppt(forcing / 'b.nc')
    assert apply_precip_multiplier(settings, 1.3)
    assert apply_precip_multiplier(settings, 0.8)  # second trial must not compound
    scaled_dir = tmp_path / SCALED_DIR_NAME
    scaled, temp = _ppt(scaled_dir / 'b.nc')
    np.testing.assert_allclose(scaled, original * 0.8)
    np.testing.assert_allclose(temp, 270.)
    np.testing.assert_allclose(_ppt(forcing / 'b.nc')[0], original)  # source untouched
    assert f"'{scaled_dir}/'" in (settings / 'fileManager.txt').read_text()
    assert json.loads((settings / SOURCE_RECORD).read_text())['forcing_path'] == str(forcing)


def test_rejects_nonpositive_multiplier(tmp_path):
    settings, _ = _domain(tmp_path)
    assert not apply_precip_multiplier(settings, 0.0)


def test_worker_excludes_precip_mult_from_trial_params(tmp_path):
    settings, forcing = _domain(tmp_path)
    original, _ = _ppt(forcing / 'a.nc')
    debug = {'errors': [], 'files_checked': []}
    assert _apply_parameters_worker({'precip_mult': np.array([1.2])}, {'config': {}}, settings,
                                    logging.getLogger('test'), debug)
    np.testing.assert_allclose(_ppt(tmp_path / SCALED_DIR_NAME / 'a.nc')[0], original * 1.2)
    assert not (settings / 'trialParams.nc').exists()


def test_split_forcing_params():
    assert split_forcing_params(['k_soil', 'precip_mult']) == (['k_soil'], ['precip_mult'])
