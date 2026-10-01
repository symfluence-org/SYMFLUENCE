"""Regression checks for active HYPE glacier classes and melt parameters."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager


def test_glacier_class_and_melt_parameters(tmp_path):
    config = {'HYPE_GLACIER_MELT_FACTOR': 6.0, 'HYPE_GLACIER_MELT_THRESHOLD': -0.5}
    manager = HYPEGeoDataManager(config, logging.getLogger('test'), tmp_path, {})
    manager._write_geoclass(pd.DataFrame({'SLC': [1, 2], 'landcover': [15, 16], 'soil': [1, 1]}))
    classes = pd.read_csv(tmp_path / 'GeoClass.txt', sep='\t', skiprows=1, header=None)
    assert classes.iloc[:, 7].tolist() == [3, 0]  # glacier, ordinary land; never lake code 2
    HYPEConfigManager(config, logging.getLogger('test'), tmp_path).write_par_file(land_uses=np.array([15, 16]))
    params = (tmp_path / 'par.txt').read_text()
    assert 'glaccmlt\t6.0' in params
    assert 'glacttmp\t-0.5' in params


def test_glacier_diagnostics_and_inclusive_last_day(tmp_path):
    pd.DataFrame({'time': pd.date_range('2009-07-01', '2009-07-31'), '86': 1.0}).to_csv(tmp_path / 'Pobs.txt', sep='\t', index=False)
    config = {'HYPE_GLACIER_FRACTION': 0.102}
    HYPEConfigManager(config, logging.getLogger('test'), tmp_path).write_info_filedir(
        0, str(tmp_path / 'results'), experiment_start='2009-07-01 00:00', experiment_end='2009-07-31 21:00')
    info = (tmp_path / 'info.txt').read_text()
    assert 'edate\t2009-07-31' in info
    assert 'GLCV\tGLCA\tGMLT' in info


def test_bands_redistribute_actual_glacier_classes_and_conserve_areas(tmp_path):
    manager = HYPEGeoDataManager({}, logging.getLogger('test'), tmp_path, {})
    parent = pd.DataFrame([dict(subid=1, maindown=0, grwdown=0, area=1000.,
                               glacier_fraction=0., SLC_1=.3, SLC_2=.42, SLC_3=.28)])
    bands = {1: [dict(hru_id=1, elev_mean=500., area_frac=.8),
                 dict(hru_id=2, elev_mean=1000., area_frac=.2)]}
    result = manager._build_banded_geodata(parent, bands, ['SLC_1'])
    # Ice exceeds the top band's capacity: the remainder must spill downward.
    np.testing.assert_allclose(result.SLC_1, [.125, 1.])
    np.testing.assert_allclose(result.glacier_fraction, result.SLC_1)
    cols = ['SLC_1', 'SLC_2', 'SLC_3']
    np.testing.assert_allclose(result[cols].sum(axis=1), 1.)
    np.testing.assert_allclose(result[cols].mul(result.area, axis=0).sum(), [300., 420., 280.])


def test_band_statistics_aggregate_before_parent_glacier_override():
    bands = {1: [dict(hru_id=1, area_frac=3.), dict(hru_id=2, area_frac=1.)]}
    frame = pd.DataFrame({'IGBP_15': [0., 1.], 'IGBP_16': [1., 0.],
                          'elev_mean': [500., 1000.], 'majority': [8, 2]}, index=[1, 2])
    result = HYPEGeoDataManager._aggregate_band_stats(frame, bands)
    assert list(result.index) == [1]
    assert result.loc[1, 'IGBP_15'] == .25
    assert result.loc[1, 'IGBP_16'] == .75
    assert result.loc[1, 'elev_mean'] == 625.
    assert result.loc[1, 'majority'] == 8
    # Parent-level statistics must not be mistaken for the lowest band.
    pd.testing.assert_frame_equal(HYPEGeoDataManager._aggregate_band_stats(result, bands), result)


def test_banding_preserves_parent_ice_volume_and_inverse_area(tmp_path):
    manager = HYPEGeoDataManager({'HYPE_PRESERVE_ICECAP_VOLUME': True, 'HYPE_GLACIER_TYPE': 1},
                                 logging.getLogger('test'), tmp_path, {})
    parent = pd.DataFrame([dict(subid=1, maindown=0, grwdown=0, area=1e9,
                               glacier_fraction=.3, SLC_1=.3, SLC_2=.7)])
    bands = {1: [dict(hru_id=1, elev_mean=500., area_frac=.8),
                 dict(hru_id=2, elev_mean=1000., area_frac=.2)]}
    result = manager._build_banded_geodata(parent, bands, ['SLC_1'])
    ice_area = result.area * result.SLC_1
    coefficient = np.exp(result._glacier_logvolcor) * 1.701
    volume = coefficient * ice_area**1.25
    np.testing.assert_allclose(volume.sum(), 1.701 * (3e8)**1.25)
    np.testing.assert_allclose((volume/coefficient)**(1/1.25), ice_area)
