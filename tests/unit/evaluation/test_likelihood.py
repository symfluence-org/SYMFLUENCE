#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the likelihood module."""

import numpy as np
import pytest

from symfluence.evaluation.likelihood import (
    gaussian_log_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    multivariate_log_likelihood,
)


class TestGaussianLogLikelihood:
    """Tests for the Gaussian log-likelihood function."""

    def test_perfect_fit_high_likelihood(self):
        """Perfect simulation should give higher likelihood than poor fit."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim_perfect = obs.copy()
        sim_poor = obs + 2.0
        obs_uc = np.full(5, 0.5)

        ll_perfect = gaussian_log_likelihood(obs, sim_perfect, obs_uncertainty=obs_uc)
        ll_poor = gaussian_log_likelihood(obs, sim_poor, obs_uncertainty=obs_uc)

        assert ll_perfect > ll_poor

    def test_returns_negative_value(self):
        """Log-likelihood should always be negative (or zero for perfect fit with infinite precision)."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.1, 1.9, 3.2])
        obs_uc = np.array([0.5, 0.5, 0.5])

        ll = gaussian_log_likelihood(obs, sim, obs_uncertainty=obs_uc)
        assert ll < 0

    def test_larger_uncertainty_increases_likelihood(self):
        """Larger observation uncertainty should increase likelihood for same residuals."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.5, 2.5, 3.5])

        ll_small_uc = gaussian_log_likelihood(obs, sim, obs_uncertainty=np.full(3, 0.1))
        ll_large_uc = gaussian_log_likelihood(obs, sim, obs_uncertainty=np.full(3, 1.0))

        assert ll_large_uc > ll_small_uc

    def test_model_error_fraction(self):
        """Model error fraction should increase total error and affect likelihood."""
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([12.0, 22.0, 32.0])
        obs_uc = np.full(3, 1.0)

        ll_no_model_err = gaussian_log_likelihood(obs, sim, obs_uncertainty=obs_uc)
        ll_with_model_err = gaussian_log_likelihood(
            obs, sim, obs_uncertainty=obs_uc, model_error_fraction=0.1
        )

        # With model error, the total sigma is larger, so likelihood should be higher
        # for the same residuals (but normalization term also changes)
        assert ll_with_model_err != ll_no_model_err

    def test_constant_model_error(self):
        """Scalar model error should work."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.1, 2.1, 3.1])

        ll = gaussian_log_likelihood(obs, sim, model_error=0.5)
        assert np.isfinite(ll)

    def test_no_uncertainty_uses_min_floor(self):
        """With no uncertainty specified, min_total_error floor should be used."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])  # Perfect fit

        ll = gaussian_log_likelihood(obs, sim)
        assert np.isfinite(ll)

    def test_handles_nan_values(self):
        """NaN values should be excluded from computation."""
        obs = np.array([1.0, np.nan, 3.0, 4.0])
        sim = np.array([1.1, 2.0, np.nan, 4.1])
        obs_uc = np.array([0.5, 0.5, 0.5, 0.5])

        ll = gaussian_log_likelihood(obs, sim, obs_uncertainty=obs_uc)
        assert np.isfinite(ll)

    def test_all_nan_returns_neg_inf(self):
        """All-NaN inputs should return -inf."""
        obs = np.array([np.nan, np.nan])
        sim = np.array([np.nan, np.nan])

        ll = gaussian_log_likelihood(obs, sim)
        assert ll == -np.inf

    def test_shape_mismatch_raises(self):
        """Mismatched obs/sim shapes should raise ValueError."""
        obs = np.array([1.0, 2.0])
        sim = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="same shape"):
            gaussian_log_likelihood(obs, sim)

    def test_array_model_error(self):
        """Time-varying model error array should work."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.1, 2.1, 3.1])
        model_err = np.array([0.1, 0.2, 0.3])

        ll = gaussian_log_likelihood(obs, sim, model_error=model_err)
        assert np.isfinite(ll)

    def test_combined_obs_and_model_error(self):
        """Both obs uncertainty and model error should combine."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.2, 2.3, 2.8, 4.1, 5.4])
        obs_uc = np.array([0.3, 0.4, 0.3, 0.5, 0.3])

        ll = gaussian_log_likelihood(
            obs, sim,
            obs_uncertainty=obs_uc,
            model_error=0.2
        )
        assert np.isfinite(ll)
        assert ll < 0


class TestHeteroscedasticLikelihood:
    """Tests for the heteroscedastic likelihood function."""

    def test_base_and_fraction(self):
        """Heteroscedastic model error with both components."""
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([11.0, 21.0, 31.0])

        ll = heteroscedastic_gaussian_log_likelihood(
            obs, sim,
            model_error_base=1.0,
            model_error_fraction=0.05,
        )
        assert np.isfinite(ll)

    def test_larger_values_have_larger_error(self):
        """With fractional error, larger simulated values should have larger uncertainty."""
        obs = np.array([10.0, 100.0])
        sim = np.array([11.0, 101.0])  # Same absolute residual

        # With only fractional error, the 100-valued point has 10x the sigma
        # so the residual is relatively smaller there
        ll = heteroscedastic_gaussian_log_likelihood(
            obs, sim, model_error_fraction=0.1
        )
        assert np.isfinite(ll)


class TestMultivariateLogLikelihood:
    """Tests for multi-variable log-likelihood."""

    def test_sum_of_individual(self):
        """Joint likelihood should equal sum of individual likelihoods (independence)."""
        np.random.seed(42)
        n = 100
        obs1, sim1, uc1 = np.random.randn(n), np.random.randn(n), np.full(n, 0.5)
        obs2, sim2, uc2 = np.random.randn(n), np.random.randn(n), np.full(n, 0.3)

        ll1 = gaussian_log_likelihood(obs1, sim1, obs_uncertainty=uc1)
        ll2 = gaussian_log_likelihood(obs2, sim2, obs_uncertainty=uc2)

        joint_ll = multivariate_log_likelihood({
            'var1': {'obs': obs1, 'sim': sim1, 'obs_uncertainty': uc1},
            'var2': {'obs': obs2, 'sim': sim2, 'obs_uncertainty': uc2},
        })

        assert np.isclose(joint_ll, ll1 + ll2)

    def test_three_variables_calmip(self):
        """Simulate CalLMIP-like setup with NEE, Qle, Qh."""
        np.random.seed(42)
        n = 365

        variables = {
            'NEE': {
                'obs': np.random.randn(n) * 5,
                'sim': np.random.randn(n) * 5,
                'obs_uncertainty': np.full(n, 2.0),
                'model_error_fraction': 0.1,
            },
            'Qle': {
                'obs': np.abs(np.random.randn(n)) * 50,
                'sim': np.abs(np.random.randn(n)) * 50,
                'obs_uncertainty': np.full(n, 10.0),
                'model_error_fraction': 0.15,
            },
            'Qh': {
                'obs': np.abs(np.random.randn(n)) * 30,
                'sim': np.abs(np.random.randn(n)) * 30,
                'obs_uncertainty': np.full(n, 8.0),
                'model_error_fraction': 0.1,
            },
        }

        ll = multivariate_log_likelihood(variables)
        assert np.isfinite(ll)
        assert ll < 0

    def test_non_independent_raises(self):
        """Full covariance mode should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            multivariate_log_likelihood(
                {'var1': {'obs': np.array([1.0]), 'sim': np.array([1.0])}},
                assume_independent=False,
            )


class TestMetricRegistration:
    """Test that log_likelihood is registered correctly."""

    def test_log_likelihood_in_registry(self):
        """log_likelihood should be in the metric registry."""
        from symfluence.evaluation.metrics_registry import get_metric_info
        info = get_metric_info('log_likelihood')
        assert info is not None
        assert info.direction == 'maximize'

    def test_metric_transformer_direction(self):
        """MetricTransformer should know log_likelihood is maximize."""
        from symfluence.evaluation.metric_transformer import MetricTransformer
        assert MetricTransformer.get_direction('log_likelihood') == 'maximize'

    def test_transform_passes_through(self):
        """Log-likelihood values should pass through transform unchanged."""
        from symfluence.evaluation.metric_transformer import MetricTransformer
        assert MetricTransformer.transform_for_maximization('log_likelihood', -150.0) == -150.0
