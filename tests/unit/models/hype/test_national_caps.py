"""Per-ice-cap mass balance term of the national calibration (ADR 0014)."""
import numpy as np
import pytest

from symfluence.models.hype import national_calibration as nc


def test_cap_penalty_uses_common_years_and_sigma():
    sim = {'Vatnajökull': {2000: -1.0, 2001: -0.5, 2002: -9.0}, 'Langjökull': {2000: -1.2}}
    obs = {'Vatnajökull': {'2000': -0.7, '2001': -0.2}, 'Langjökull': {'2000': -1.2}, 'Hofsjökull': {'2000': -1.0}}
    pen, diag = nc.cap_penalty(sim, obs)
    # Vatnajökull: years 2000-2001 only, |(-0.75) - (-0.45)| / 0.3 = 1.0; Langjökull 0; Hofsjökull no sim -> skipped
    assert diag['Vatnajökull'] == {'sim': -0.75, 'obs': pytest.approx(-0.45), 'n': 2}
    assert set(diag) == {'Vatnajökull', 'Langjökull'}
    assert pen == pytest.approx(0.5)


def test_cap_penalty_nan_without_overlap():
    pen, diag = nc.cap_penalty({'Vatnajökull': {1990: 0.0}}, {'Vatnajökull': {'2000': 0.0}})
    assert np.isnan(pen) and diag == {}
