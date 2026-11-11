"""Tests for perturbation strategies."""
from __future__ import annotations

import numpy as np
import pytest

from symfluence.data_assimilation.enkf.perturbation import GaussianPerturbation


class TestGaussianPerturbation:
    """Tests for GaussianPerturbation."""

    def test_parameter_perturbation_within_bounds(self):
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(param_std=0.1, rng=rng)

        base = {'k1': 0.1, 'fc': 250.0}
        bounds = {'k1': (0.01, 0.5), 'fc': (50.0, 500.0)}

        for i in range(100):
            p = pert.perturb_parameters(base, bounds, i)
            assert bounds['k1'][0] <= p['k1'] <= bounds['k1'][1]
            assert bounds['fc'][0] <= p['fc'] <= bounds['fc'][1]

    def test_parameter_perturbation_statistics(self):
        """Mean of many perturbations should be close to base."""
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(param_std=0.05, rng=rng)

        base = {'k1': 0.1}
        bounds = {'k1': (0.0, 0.5)}

        values = [pert.perturb_parameters(base, bounds, i)['k1'] for i in range(1000)]
        mean_val = np.mean(values)
        assert abs(mean_val - 0.1) < 0.02, f"Mean {mean_val} too far from base 0.1"

    def test_precip_perturbation_nonnegative(self):
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(precip_std=0.3, rng=rng)

        precip = np.array([0.0, 5.0, 10.0, 0.5, 20.0])
        perturbed = pert.perturb_forcing(precip, member_id=0, variable='precip')

        assert np.all(perturbed >= 0.0), "Precipitation should remain non-negative"

    def test_temp_perturbation_additive(self):
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(temp_std=1.0, rng=rng)

        temp = np.ones(100) * 5.0
        perturbed = pert.perturb_forcing(temp, member_id=0, variable='temp')

        # Should be approximately centered at 5.0 with std ~1.0
        assert abs(np.mean(perturbed) - 5.0) < 0.5
        assert 0.5 < np.std(perturbed) < 2.0

    def test_state_perturbation_nonnegative(self):
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(state_std=0.1, rng=rng)

        state = {'snow': np.array([10.0]), 'sm': np.array([150.0])}
        perturbed = pert.perturb_state(state, member_id=0)

        assert perturbed['snow'][0] >= 0.0
        assert perturbed['sm'][0] >= 0.0

    def test_different_members_differ(self):
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(param_std=0.1, rng=rng)

        base = {'k1': 0.1}
        bounds = {'k1': (0.0, 0.5)}

        p1 = pert.perturb_parameters(base, bounds, 0)
        p2 = pert.perturb_parameters(base, bounds, 1)

        assert p1['k1'] != p2['k1'], "Different members should get different perturbations"

    @pytest.mark.parametrize("variable,sigma", [("precip", 0.3), ("pet", 0.15)])
    def test_multiplicative_perturbation_is_unbiased(self, variable, sigma):
        """lognormal(mean=0, sigma) has mean exp(sigma^2/2) -- a 4.6 % wet bias
        at sigma=0.3. The multiplier must average 1."""
        rng = np.random.default_rng(42)
        pert = GaussianPerturbation(precip_std=0.3, rng=rng)

        data = np.full(400_000, 10.0)
        perturbed = pert.perturb_forcing(data, member_id=0, variable=variable)

        assert perturbed.mean() == pytest.approx(10.0, rel=2e-3)
        # the spread is still the requested lognormal sigma
        assert np.log(perturbed / 10.0).std() == pytest.approx(sigma, rel=1e-2)
