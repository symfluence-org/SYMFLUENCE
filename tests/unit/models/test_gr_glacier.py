# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GR lumped glacier mode: layer fractions, mass-balance-solved ice melt, area scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.gr.glacier import (
    area_scale_series,
    complete_water_years,
    compute_ice_melt,
    layer_glacier_fractions,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _two_years():
    dates = pd.date_range('2012-10-01', '2014-09-30', freq='D')
    n = len(dates)
    temp = np.column_stack([np.full(n, 5.0), np.full(n, 2.0)])
    swe = np.zeros((n, 2))
    swe[:200, 1] = 50.0          # upper layer snow-covered for the first 200 days
    return dates, swe, temp


def test_layer_fractions_follow_equal_area_bands():
    z = np.arange(100, dtype=float)
    ice = z >= 80                 # top fifth glacierised
    frac = layer_glacier_fractions(z, ice, 5)
    np.testing.assert_allclose(frac, [0, 0, 0, 0, 1])
    np.testing.assert_allclose(layer_glacier_fractions(z, ice, 10)[-2:], [1, 1])


def test_complete_water_years_skip_partial_years():
    dates = pd.date_range('2012-07-01', '2014-12-31', freq='D')
    assert list(complete_water_years(dates)) == [2013, 2014]


def test_fixed_ddf_melts_only_exposed_ice():
    dates, swe, temp = _two_years()
    res = compute_ice_melt(dates, swe, temp, [0.0, 0.5], np.ones(len(dates)), ddf=4.0)
    # basin melt = ddf * T * fraction / n_layers on snow-free days, zero under snow
    assert res.ice_melt[0] == 0.0
    assert res.ice_melt[-1] == pytest.approx(4.0 * 2.0 * 0.5 / 2)


def test_solved_ddf_reproduces_mass_balance_target():
    dates, swe, temp = _two_years()
    target = -2000.0
    res = compute_ice_melt(dates, swe, temp, [0.0, 0.5], np.ones(len(dates)),
                           mb_target_mm=target, mb_years=[2013, 2014], ddf_bounds=(0.1, 50))
    assert np.mean([res.mass_balance[y] for y in (2013, 2014)]) == pytest.approx(target)
    assert res.ddf == pytest.approx(res.ddf_unclipped)


def test_solved_ddf_is_clipped_to_bounds():
    dates, swe, temp = _two_years()
    res = compute_ice_melt(dates, swe, temp, [0.0, 0.5], np.ones(len(dates)),
                           mb_target_mm=-1e6, mb_years=[2013], ddf_bounds=(1, 20))
    assert res.ddf == 20 and res.ddf_unclipped > 20


def test_mass_balance_target_needs_simulated_years():
    dates, swe, temp = _two_years()
    with pytest.raises(ConfigurationError):
        compute_ice_melt(dates, swe, temp, [0.0, 0.5], np.ones(len(dates)),
                         mb_target_mm=-1000.0, mb_years=[2020])


def test_area_scale_follows_water_year_and_clamps(tmp_path):
    f = tmp_path / 'glacier_timeseries.csv'
    f.write_text(';annual_net_MB;g_area_dyn\n2013;-1;120\n2014;-1;110\n')
    dates = pd.DatetimeIndex(['2012-09-30', '2012-10-01', '2014-06-01', '2020-01-01'])
    np.testing.assert_allclose(area_scale_series(dates, 100.0, f), [1.2, 1.2, 1.1, 1.1])
    np.testing.assert_allclose(area_scale_series(dates, 100.0, None), 1.0)
