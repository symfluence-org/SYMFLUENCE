"""HYPE outlet lakes (HYPE_OUTLET_LAKES): lake class, GeoData fractions and depth, rating parameters."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ValidationError
from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager

LOG = logging.getLogger('test')


def _frames():
    slc = pd.DataFrame({'SLC': [1, 2], 'landcover': [15, 16], 'soil': [1, 1]})
    base = pd.DataFrame([dict(subid=5, maindown=7, area=1e7, glacier_fraction=.2, SLC_1=.2, SLC_2=.8),
                         dict(subid=7, maindown=0, area=8e7, glacier_fraction=0., SLC_1=0., SLC_2=1.)])
    return base, slc


def test_outlet_lake_class_fractions_and_depth(tmp_path):
    cfg = {'HYPE_OUTLET_LAKES': {'7': {'depth': 34.0}, 5: {'depth': 2.0, 'fraction': .25}}}
    m = HYPEGeoDataManager(cfg, LOG, tmp_path, {})
    base, slc = _frames()
    out, slc2 = m._apply_outlet_lakes(base, slc, ['SLC_1'])
    assert slc2.iloc[-1][['SLC', 'landcover', 'special']].tolist() == [3, 17, 1]
    np.testing.assert_allclose(out.SLC_3, [.25, 1.])
    np.testing.assert_allclose(out[['SLC_1', 'SLC_2', 'SLC_3']].sum(axis=1), 1.)
    np.testing.assert_allclose(out.SLC_1, [.15, 0.])            # other classes scaled by 1 - fraction
    np.testing.assert_allclose(out.glacier_fraction, [.15, 0.])
    np.testing.assert_allclose(out.lake_depth, [2., 34.])
    m._write_geoclass(slc2)
    classes = pd.read_csv(tmp_path / 'GeoClass.txt', sep='\t', skiprows=1, header=None)
    assert classes.iloc[:, 7].tolist() == [3, 0, 1]            # glacier, land, outlet lake


def test_no_lakes_leaves_frames_and_geoclass_unchanged(tmp_path):
    m = HYPEGeoDataManager({}, LOG, tmp_path, {})
    base, slc = _frames()
    out, slc2 = m._apply_outlet_lakes(base, slc, ['SLC_1'])
    assert out is base and slc2 is slc
    m._write_geoclass(slc2)
    classes = pd.read_csv(tmp_path / 'GeoClass.txt', sep='\t', skiprows=1, header=None)
    assert classes.iloc[:, 7].tolist() == [3, 0]


@pytest.mark.parametrize('lakes', [{'9': {'depth': 10.0}}, {'7': {'depth': 0.0}}, {'7': {'depth': 5.0, 'fraction': 1.5}}])
def test_invalid_outlet_lakes(tmp_path, lakes):
    base, slc = _frames()
    with pytest.raises(ValidationError):
        HYPEGeoDataManager({'HYPE_OUTLET_LAKES': lakes}, LOG, tmp_path, {})._apply_outlet_lakes(base, slc, ['SLC_1'])


def test_rating_parameters_written_and_overridable(tmp_path):
    cfg = {'HYPE_OUTLET_LAKES': {'7': {'depth': 34.0}}}
    HYPEConfigManager(cfg, LOG, tmp_path).write_par_file(land_uses=np.array([16, 17]))
    par = (tmp_path / 'par.txt').read_text()
    assert 'gratk\t100.0' in par and 'gratp\t1.5' in par and 'gldepo\t34.0' in par
    cfg['HYPE_EXTRA_PARAMETERS'] = {'gratk': 250.0}
    HYPEConfigManager(cfg, LOG, tmp_path).write_par_file(land_uses=np.array([16, 17]), params={'gratp': 2.0})
    par = (tmp_path / 'par.txt').read_text()
    assert par.count('\ngratk') == 1 and 'gratk\t250' in par.replace(' ', '\t')
    assert 'gratp\t2.0' in par.replace(' ', '\t') or 'gratp\t2' in par.replace(' ', '\t')
    HYPEConfigManager({}, LOG, tmp_path).write_par_file(land_uses=np.array([16]))
    assert 'gratk' not in (tmp_path / 'par.txt').read_text()
