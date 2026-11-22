# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Fitting and applying source-to-reference forcing corrections (synthetic pairs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.data.preprocessing.forcing_source_correction import (
    CorrectionSpec,
    ForcingSourceCorrection,
    fit_source_correction,
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
    c = fit_source_correction(_pairs(), SPECS, source='nwp-v1')
    assert c.factors['airtemp']['1'] == pytest.approx(2.0)
    assert c.factors['airtemp']['7'] == pytest.approx(0.5)
    assert list(c.factors['pptrate'].values()) == pytest.approx([1.25] * 12)
    assert c.meta['n_pairs'] == len(_pairs())


def test_apply_dataset_and_source_guard():
    c = fit_source_correction(_pairs(), SPECS, source='nwp-v1')
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
    assert out['airtemp'].attrs['units'] == 'degC' and 'nwp-v1' in out.attrs['forcing_source_correction']
    with pytest.raises(ValidationError):
        c.apply(ds, 'nwp-v2')


def test_none_level_and_missing_groups_stay_neutral():
    p = _pairs()
    p = p[p['time'].dt.month != 5]
    c = fit_source_correction(p, [CorrectionSpec('airtemp', 'add', 'none'), CorrectionSpec('pptrate', 'mult', 'month')],
                       source='s')
    assert set(c.factors['airtemp'].values()) == {0.0}
    assert c.factors['pptrate']['5'] == 1.0 and c.factors['pptrate']['6'] == pytest.approx(1.25)


def test_save_load_roundtrip(tmp_path):
    c = fit_source_correction(_pairs(), SPECS, source='nwp-v1')
    back = ForcingSourceCorrection.load(c.save(tmp_path / 'c' / 'corr.json'))
    assert back.factors == c.factors and back.kinds == c.kinds and back.source == 'nwp-v1'
    with pytest.raises(ValidationError):
        ForcingSourceCorrection(source='s', factors={'x': {'1': 1.0}}, kinds={'x': 'add'})


def test_split_sample_test_reports_out_of_sample_gain():
    rep = split_sample_test(_pairs(), SPECS)
    assert set(rep['test']) == {'forward', 'reverse'}
    t = rep[rep['variable'] == 'airtemp']
    assert (t['corrected_bias'].abs() < t['raw_bias'].abs()).all()
    assert (t['corrected_rmse'] < t['raw_rmse']).all()


def test_missing_columns():
    with pytest.raises(ValidationError):
        fit_source_correction(_pairs().drop(columns='ref_pptrate'), SPECS, source='s')


def _dry_pairs() -> pd.DataFrame:
    """July is almost dry in the source, so ref/src would be a huge ratio."""
    t = pd.date_range('2024-01-01', '2024-12-31 21:00', freq='3h')
    src = np.where(t.month == 7, 1e-9, 1e-4)
    ref = np.where(t.month == 7, 5e-5, 1.2e-4)
    return pd.DataFrame({'time': t, 'pptrate': src, 'ref_pptrate': ref})


def test_dry_month_below_min_source_mean_stays_neutral():
    spec = CorrectionSpec('pptrate', 'mult', 'month', clip_min=0.0, min_source_mean=1e-6)
    corr = fit_source_correction(_dry_pairs(), [spec], source='nwp')
    assert corr.factors['pptrate']['7'] == 1.0
    assert corr.factors['pptrate']['1'] == pytest.approx(1.2)
    assert any('pptrate[7]' in note and 'left at 1' in note for note in corr.meta['limited'])


def test_multiplicative_factor_is_clamped_by_default():
    corr = fit_source_correction(_dry_pairs(), [CorrectionSpec('pptrate', 'mult', 'month')], source='nwp')
    assert corr.factors['pptrate']['7'] == 4.0
    assert any('clamped' in note for note in corr.meta['limited'])
    unbounded = fit_source_correction(_dry_pairs(), [CorrectionSpec('pptrate', 'mult', 'month', factor_bounds=None)],
                                      source='nwp')
    assert unbounded.factors['pptrate']['7'] == pytest.approx(5e4)


def test_factor_bounds_must_bracket_one():
    with pytest.raises(ValidationError):
        CorrectionSpec('pptrate', 'mult', factor_bounds=(1.5, 3.0))


def test_temperature_correction_keeps_relative_humidity():
    from symfluence.data.preprocessing.forcing_source_correction import HumidityVariables
    from symfluence.data.preprocessing.resampling.elevation_correction import saturation_vapour_pressure

    t = pd.date_range('2024-01-01', periods=24, freq='MS')
    ds = xr.Dataset({'airtemp': ('time', np.full(24, 275.0)), 'spechum': ('time', np.full(24, 3e-3)),
                     'airpres': ('time', np.full(24, 9e4))}, coords={'time': t})
    corr = ForcingSourceCorrection(source='nwp', factors={'airtemp': {m: 2.0 for m in [str(i) for i in range(1, 13)]}},
                                   kinds={'airtemp': 'add'})

    def rh(d):
        q, p = d['spechum'].values, d['airpres'].values
        e = q * p / (0.622 + 0.378 * q)
        return e / saturation_vapour_pressure(d['airtemp'].values)

    kept = corr.apply(ds, 'nwp', humidity=HumidityVariables())
    np.testing.assert_allclose(kept['airtemp'], 277.0)
    np.testing.assert_allclose(rh(kept), rh(ds), rtol=1e-3)
    assert (kept['spechum'] > ds['spechum']).all()
    # without the option specific humidity is untouched
    np.testing.assert_array_equal(corr.apply(ds, 'nwp')['spechum'], ds['spechum'])
    with pytest.raises(ValidationError):
        corr.apply(ds.drop_vars('airpres'), 'nwp', humidity=HumidityVariables())
