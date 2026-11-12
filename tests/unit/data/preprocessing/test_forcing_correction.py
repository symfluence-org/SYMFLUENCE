# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Fitting and applying source-to-reference forcing corrections (synthetic pairs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.data.preprocessing.forcing_correction import (
    CorrectionSpec,
    ForcingCorrection,
    fit_correction,
    split_sample_test,
)

pytestmark = pytest.mark.unit


def _pairs(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range('2024-01-01', '2025-12-31 21:00', freq='3h')
    month = t.month.to_numpy()
    ref_t = 5 + 10 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 2, len(t))
    ref_p = rng.gamma(0.5, 2e-4, len(t))
    return pd.DataFrame({
        'time': t,
        'ref_airtemp': ref_t,
        'airtemp': ref_t - np.where(np.isin(month, [12, 1, 2]), 2.0, 0.5),   # source too cold, more in winter
        'ref_pptrate': ref_p,
        'pptrate': ref_p / 1.25,                                             # source 20 % dry
    })


SPECS = [CorrectionSpec('airtemp', 'add', 'season'), CorrectionSpec('pptrate', 'mult', 'annual', clip_min=0.0)]


def test_fit_recovers_the_bias():
    c = fit_correction(_pairs(), SPECS, source='nwp-v1')
    assert c.factors['airtemp']['1'] == pytest.approx(2.0)
    assert c.factors['airtemp']['7'] == pytest.approx(0.5)
    assert list(c.factors['pptrate'].values()) == pytest.approx([1.25] * 12)
    assert c.meta['n_pairs'] == len(_pairs())


def test_apply_dataset_and_source_guard():
    c = fit_correction(_pairs(), SPECS, source='nwp-v1')
    t = pd.date_range('2026-01-01', periods=2, freq='181D')     # January and July
    ds = xr.Dataset({'airtemp': (('time', 'hru'), np.zeros((2, 3))),
                     'pptrate': (('time', 'hru'), np.full((2, 3), -1.0)),
                     'windspd': (('time', 'hru'), np.ones((2, 3)))},
                    coords={'time': t, 'hru': [1, 2, 3]})
    ds['airtemp'].attrs['units'] = 'degC'
    out = c.apply(ds, 'nwp-v1')
    np.testing.assert_allclose(out['airtemp'].values[:, 0], [2.0, 0.5])
    assert (out['pptrate'].values == 0).all()                  # clipped at 0
    np.testing.assert_array_equal(out['windspd'].values, ds['windspd'].values)
    assert out['airtemp'].attrs['units'] == 'degC' and 'nwp-v1' in out.attrs['forcing_correction']
    with pytest.raises(ValidationError):
        c.apply(ds, 'nwp-v2')


def test_none_level_and_missing_groups_stay_neutral():
    p = _pairs()
    p = p[p['time'].dt.month != 5]
    c = fit_correction(p, [CorrectionSpec('airtemp', 'add', 'none'), CorrectionSpec('pptrate', 'mult', 'month')],
                       source='s')
    assert set(c.factors['airtemp'].values()) == {0.0}
    assert c.factors['pptrate']['5'] == 1.0 and c.factors['pptrate']['6'] == pytest.approx(1.25)


def test_save_load_roundtrip(tmp_path):
    c = fit_correction(_pairs(), SPECS, source='nwp-v1')
    back = ForcingCorrection.load(c.save(tmp_path / 'c' / 'corr.json'))
    assert back.factors == c.factors and back.kinds == c.kinds and back.source == 'nwp-v1'
    with pytest.raises(ValidationError):
        ForcingCorrection(source='s', factors={'x': {'1': 1.0}}, kinds={'x': 'add'})


def test_split_sample_test_reports_out_of_sample_gain():
    rep = split_sample_test(_pairs(), SPECS)
    assert set(rep['test']) == {'forward', 'reverse'}
    t = rep[rep['variable'] == 'airtemp']
    assert (t['corrected_bias'].abs() < t['raw_bias'].abs()).all()
    assert (t['corrected_rmse'] < t['raw_rmse']).all()


def test_missing_columns():
    with pytest.raises(ValidationError):
        fit_correction(_pairs().drop(columns='ref_pptrate'), SPECS, source='s')
