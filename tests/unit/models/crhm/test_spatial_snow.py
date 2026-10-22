"""Physical invariants for snowfall quadrature used by native snow tiles."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm
from symfluence.models.crhm.spatial_snow import snow_tile_quadrature, split_solid_precipitation


@pytest.mark.parametrize('cv', [0, .5, 1, 2, 4])
@pytest.mark.parametrize('count', [1, 3, 5, 9])
def test_precipitation_conserved_for_dry_and_wet_intervals(cv, count):
    weights, factors = snow_tile_quadrature(cv, count)
    snow = np.array([0, .01, 1, 25.])
    rain = np.array([0, 10, 2, 0.])
    tiles = split_solid_precipitation(snow, weights, factors)
    np.testing.assert_allclose((tiles + rain[:, None]) @ weights, snow + rain, atol=1e-12)
    assert (factors >= 0).all()
    assert np.all(np.diff(factors) >= -1e-12)


def test_conditional_means_include_mass_in_the_upper_tail():
    weights, factors = snow_tile_quadrature(2, 5)
    sigma = np.sqrt(np.log(5))
    distribution = lognorm(s=sigma, scale=np.exp(-sigma*sigma/2))
    cutoff = distribution.ppf(.8)
    tail_mass = quad(lambda x: x * distribution.pdf(x), cutoff, np.inf)[0]
    assert weights[-1] * factors[-1] == pytest.approx(tail_mass, abs=1e-9)


@pytest.mark.parametrize('cv', [-1, np.nan, np.inf])
def test_invalid_deposition_assumptions_rejected(cv):
    with pytest.raises(ValueError):
        snow_tile_quadrature(cv)


def test_nonconservative_weights_cannot_be_used():
    with pytest.raises(ValueError):
        split_solid_precipitation([1], [.5, .5], [1, 2])
