"""Twin experiments and verification for the operational-rigour DA features.

A two-store conceptual model (fast and slow linear reservoirs) stands in for a
hydrological model behind the DAModel state interface. The truth run uses the
true parameters, the true initial state and precipitation that differs from
the forcing the ensemble sees; observations are the truth plus heteroscedastic
noise. The tests check that the EnKF beats the open loop on CRPS, that the
precipitation-bias state learns the bias, that lead-time scores are computed
against the right benchmarks, and the statistics of the AR(1) perturbations
and of the observation-error model.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from symfluence.core.config.models.state_config import EnKFConfig
from symfluence.data_assimilation import diagnostics as dg
from symfluence.data_assimilation.cycling import CycleSettings, run_cycle
from symfluence.data_assimilation.da_manager import DataAssimilationManager
from symfluence.data_assimilation.enkf.enkf_algorithm import EnKFAlgorithm
from symfluence.data_assimilation.enkf.ensemble_manager import DAModel, EnsembleManager
from symfluence.data_assimilation.enkf.observation_error import StreamflowObservationError
from symfluence.data_assimilation.enkf.perturbation import (
    GaussianPerturbation,
    ar1_coefficient,
)
from symfluence.data_assimilation.enkf.state_vector import StateVariableSpec

K_FAST, K_SLOW, SPLIT = 0.35, 0.03, 0.4


class TwoStore(EnsembleManager):
    """Fast/slow linear reservoirs; optional second gauge = fast flow only."""

    perturbable_forcing = {'precip': 'multiplicative'}

    def __init__(self, n_members, precip, init_fast, init_slow, two_gauges=False):
        super().__init__(n_members)
        n = len(precip)
        self.forcing = {'precip': np.asarray(precip, float),
                        'time': pd.date_range("2000-01-01", periods=n, freq="D")}
        self.state_specs = [StateVariableSpec('fast', 1, lower_bound=0.0),
                            StateVariableSpec('slow', 1, lower_bound=0.0)]
        self.timestep_hours = 24
        self._init = (np.asarray(init_fast, float), np.asarray(init_slow, float))
        self.two_gauges = two_gauges
        self.X = np.zeros((n_members, 2))

    def initialize_members(self, **kwargs):
        self.X = np.column_stack([np.broadcast_to(self._init[0], self.n_members),
                                  np.broadcast_to(self._init[1], self.n_members)]).astype(float)

    def get_state(self):
        return self.X.copy()

    def set_state(self, X):
        self.X = np.asarray(X, float).copy()

    def step(self, t, forcing):
        p = forcing['precip']
        fast = self.X[:, 0] + (1 - SPLIT) * p
        slow = self.X[:, 1] + SPLIT * p
        qf, qs = K_FAST * fast, K_SLOW * slow
        self.X = np.column_stack([fast - qf, slow - qs])
        q = qf + qs
        return np.column_stack([q, qf]) if self.two_gauges else q


def _truth(precip, fast0=5.0, slow0=150.0, two_gauges=False):
    m = TwoStore(1, precip, fast0, slow0, two_gauges)
    m.initialize_members()
    return np.array([m.step(t, {'precip': precip[t:t + 1]})[0] for t in range(len(precip))])


def _experiment(seed=0, n=400, n_members=40, bias=1.0, **cfg):
    """Identical twin: the ensemble's forcing error model is the true one.

    The truth is driven by the true precipitation; the ensemble only sees a
    measured precipitation carrying a lognormal AR(1) error (and, with
    ``bias``, a systematic undercatch), and starts from wrong states.
    Observations get noise from the filter's own error model.
    """
    rng = np.random.default_rng(seed)
    p_true = rng.gamma(0.6, 8.0, n)
    q_true = _truth(p_true)
    forcing_error = GaussianPerturbation(precip_std=0.3, rng=np.random.default_rng(seed + 10))
    precip = p_true * forcing_error.forcing_noise('precip', 1, n, 2.0)[0] / bias
    obs_err = 0.01 * q_true.mean() + 0.1 * q_true
    obs = np.maximum(q_true + rng.normal(0.0, obs_err), 0.0)

    ens = TwoStore(n_members, precip,
                   init_fast=rng.uniform(0, 20, n_members),
                   init_slow=rng.uniform(20, 80, n_members))  # slow store far too low
    ens.perturbation = GaussianPerturbation(precip_std=0.3, rng=np.random.default_rng(seed + 1))
    ens.initialize_members()
    cfg.setdefault('ENKF_FORCING_CORRELATION_HOURS', 48.0)
    config = EnKFConfig(ENKF_ENSEMBLE_SIZE=n_members, **cfg)
    mgr = DataAssimilationManager.__new__(DataAssimilationManager)
    mgr.logger = logging.getLogger("test.da_added_value")
    result = mgr.run_filter(
        enkf=EnKFAlgorithm(rng=np.random.default_rng(seed + 2)),
        enkf_config=config, ensemble_mgr=ens, state_specs=None,
        obs_model_units=obs, warmup_steps=10, rng=np.random.default_rng(seed + 3),
    )
    return result, obs, q_true


class TestStateInterface:
    def test_two_store_implements_protocol(self):
        assert isinstance(TwoStore(3, np.ones(5), 1.0, 1.0), DAModel)

    def test_get_set_state_roundtrip_reproduces_run(self):
        m = TwoStore(4, np.full(10, 3.0), 1.0, 10.0)
        m.initialize_members()
        m.step(0, {'precip': np.full(4, 3.0)})
        X = m.get_state()
        a = m.step(1, {'precip': np.full(4, 3.0)})
        m.set_state(X)
        b = m.step(1, {'precip': np.full(4, 3.0)})
        np.testing.assert_array_equal(a, b)


class TestTwinExperiment:
    def test_enkf_beats_open_loop_and_persistence_on_crps(self):
        result, obs, _ = _experiment()
        summary, _ = DataAssimilationManager.verify(result, warmup_steps=10, timestep_hours=24)
        assert summary['enkf']['crps'] < summary['open_loop']['crps']
        assert summary['crpss_vs_open_loop'] > 0.15
        # persistence of a slowly varying, well-observed flow is a hard
        # benchmark; the filter priors must still beat it
        assert summary['crpss_vs_persistence'] > 0.0
        assert summary['enkf']['nse'] > summary['open_loop']['nse']
        assert summary['enkf']['kge'] > summary['open_loop']['kge']
        assert len(summary['enkf']['rank_histogram']) == result.ensemble_predictions.shape[1] + 1
        assert 0.2 < summary['enkf']['spread_error_ratio'] < 5.0
        assert np.isfinite(summary['open_loop']['spread_error_ratio'])

    def test_open_loop_shares_the_perturbations(self):
        """Before the first analysis (warm-up), filter and open loop are identical."""
        result, _, _ = _experiment()
        np.testing.assert_allclose(result.ensemble_predictions[:11],
                                   result.open_loop_predictions[:11])
        assert result.analysis_times[0] == 10

    def test_precip_bias_state_learns_the_bias(self):
        result, _, _ = _experiment(bias=1.4, n=500, ENKF_PRECIP_BIAS_STATE=True,
                                   ENKF_PRECIP_BIAS_STD=0.3)
        start = result.precip_bias[0].mean()
        end = result.precip_bias[-100:].mean()
        assert abs(start - 1.0) < 0.15
        assert end > 1.15

    def test_lead_time_skill_decays_with_lead(self):
        result, obs, _ = _experiment(ENKF_FORECAST_LEAD_DAYS=10, ENKF_FORECAST_ISSUE_INTERVAL=5)
        assert result.forecasts.shape[1:] == (10, 40)
        assert result.forecast_issue_steps[0] == 10
        assert np.diff(result.forecast_issue_steps).tolist() == [5] * (len(result.forecast_issue_steps) - 1)
        _, table = DataAssimilationManager.verify(result, 10, 24)
        assert list(table['lead_steps']) == list(range(1, 11))
        assert (table['crpss_vs_open_loop'] > 0).all()
        assert table['crpss_vs_open_loop'].iloc[0] > table['crpss_vs_open_loop'].iloc[-1]
        # persistence degrades faster than the forecast
        assert table['crpss_vs_persistence'].iloc[-1] > table['crpss_vs_persistence'].iloc[0]
        for col in ('crps', 'crps_open_loop', 'mae_persistence', 'spread_error_ratio',
                    'nse', 'kge', 'bias', 'rmse', 'skill_vs_persistence', 'skill_vs_open_loop'):
            assert col in table

    def test_forecast_from_analysis_matches_filter_without_obs(self):
        """A 1-step forecast issued at t equals the filter prior at t+1."""
        result, _, _ = _experiment(ENKF_FORECAST_LEAD_DAYS=1)
        for i, t in enumerate(result.forecast_issue_steps[:20]):
            np.testing.assert_allclose(result.forecasts[i, 0], result.ensemble_predictions[t + 1])

    def test_white_noise_still_beats_open_loop(self):
        result, _, _ = _experiment(ENKF_FORCING_CORRELATION_HOURS=0.0)
        summary, _ = DataAssimilationManager.verify(result, 10, 24)
        assert summary['crpss_vs_open_loop'] > 0.15


class TestMultipleGauges:
    def test_vector_observation_with_diagonal_r(self):
        rng = np.random.default_rng(4)
        n, m = 300, 30
        p_true = rng.gamma(0.6, 8.0, n)
        truth = TwoStore(1, p_true, 5.0, 150.0, two_gauges=True)
        truth.initialize_members()
        q = np.vstack([truth.step(t, {'precip': p_true[t:t + 1]})[0] for t in range(n)])
        obs = q * (1 + 0.05 * rng.standard_normal(q.shape))
        obs[::3, 1] = np.nan  # second gauge reports every third day only
        pert = GaussianPerturbation(precip_std=0.3, rng=np.random.default_rng(5))
        precip = p_true * pert.forcing_noise('precip', 1, n, 2.0)[0]
        ens = TwoStore(m, precip, rng.uniform(0, 20, m), rng.uniform(20, 80, m), two_gauges=True)
        ens.initialize_members()
        err = StreamflowObservationError.from_config(EnKFConfig(), obs)
        assert np.shape(err.a) == (2,)
        noise = {'precip': pert.forcing_noise('precip', m, n, 2.0)}
        res = run_cycle(ens, EnKFAlgorithm(rng=np.random.default_rng(1)), obs, err,
                        CycleSettings(warmup_steps=0), forcing_noise=noise,
                        rng=np.random.default_rng(2))
        assert res.ensemble_predictions.shape == (n, m, 2)
        assert res.open_loop_predictions.shape == (n, m, 2)
        for g in range(2):
            assert dg.crps(res.ensemble_predictions[:, :, g], obs[:, g]) < \
                dg.crps(res.open_loop_predictions[:, :, g], obs[:, g])


class TestAR1Perturbations:
    @pytest.mark.parametrize("tau", [0.0, 1.0, 3.0, 10.0])
    def test_autocorrelation_and_debiasing(self, tau):
        pert = GaussianPerturbation(precip_std=0.4, rng=np.random.default_rng(0))
        mult = pert.forcing_noise('precip', 400, 400, tau)
        assert mult.shape == (400, 400)
        assert mult.min() > 0
        # expectation 1 (the debiasing survives the correlation)
        assert mult.mean() == pytest.approx(1.0, abs=0.02)
        logm = np.log(mult)
        assert logm.std() == pytest.approx(0.4, rel=0.03)
        z = logm - logm.mean(axis=1, keepdims=True)
        lag1 = np.mean(z[:, 1:] * z[:, :-1]) / np.mean(z * z)
        assert lag1 == pytest.approx(ar1_coefficient(tau), abs=0.03)

    def test_stationary_from_the_first_step(self):
        pert = GaussianPerturbation(temp_std=2.0, rng=np.random.default_rng(1))
        off = pert.forcing_noise('temp', 20000, 5, 20.0)
        np.testing.assert_allclose(off.std(axis=0), 2.0, rtol=0.03)
        np.testing.assert_allclose(off.mean(axis=0), 0.0, atol=0.05)

    def test_correlation_hours_convert_with_timestep(self):
        from symfluence.data_assimilation.cycling import draw_forcing_noise

        class Hourly:
            n_members = 300
            timestep_hours = 6
            perturbable_forcing = {'precip': 'multiplicative', 'temp': 'additive'}

        noise = draw_forcing_noise(GaussianPerturbation(rng=np.random.default_rng(2)),
                                   Hourly(), 2000, correlation_hours=24.0)
        z = np.log(noise['precip'])
        z = z - z.mean()
        lag1 = np.mean(z[:, 1:] * z[:, :-1]) / np.mean(z * z)
        assert lag1 == pytest.approx(np.exp(-6 / 24), abs=0.02)
        assert set(noise) == {'precip', 'temp'}
        assert draw_forcing_noise(GaussianPerturbation(), Hourly(), 10, 24.0, enabled=False) == {}

    def test_unknown_variable(self):
        with pytest.raises(ValueError):
            GaussianPerturbation().forcing_noise('wind', 2, 2)


class TestObservationErrorModel:
    def test_heteroscedastic_is_default_and_affine(self):
        cfg = EnKFConfig()
        assert cfg.obs_error_type == 'heteroscedastic'
        obs = np.array([0.0, 10.0, np.nan, 30.0])  # mean 40/3
        err = StreamflowObservationError.from_config(cfg, obs)
        assert err.a == pytest.approx(0.01 * 40.0 / 3.0)
        np.testing.assert_allclose(err.std(np.array([0.0, 10.0, 50.0])),
                                   np.array([0.0, 1.0, 5.0]) + 0.4 / 3.0)
        R = err.covariance(np.array([10.0, 50.0]))
        assert R.shape == (2, 2)
        assert R[0, 1] == 0.0
        assert R[1, 1] == pytest.approx((5.0 + 0.4 / 3.0) ** 2)

    def test_absolute_floor_given_in_m3s(self):
        cfg = EnKFConfig(ENKF_OBS_ERROR_ABS=2.0, ENKF_OBS_ERROR_STD=0.15)
        factor = DataAssimilationManager.cms_to_model_units(24, 100e6)  # 0.864 mm per m3/s
        err = StreamflowObservationError.from_config(cfg, np.ones(5), cms_to_model=factor)
        assert err.a == pytest.approx(2.0 * 0.864)
        assert err.std(np.array([10.0]))[0] == pytest.approx(1.728 + 1.5)
        with pytest.raises(ValueError, match="m3/s"):
            StreamflowObservationError.from_config(cfg, np.ones(5))

    def test_legacy_relative_and_absolute(self):
        rel = StreamflowObservationError.from_config(
            EnKFConfig(ENKF_OBS_ERROR_TYPE='relative'), np.array([10.0, 30.0]))
        assert rel.std(np.array([50.0]))[0] == pytest.approx(5.0)
        assert rel.std(np.array([0.0]))[0] == pytest.approx(0.1 * 0.1 * 20.0)
        ab = StreamflowObservationError.from_config(
            EnKFConfig(ENKF_OBS_ERROR_TYPE='absolute', ENKF_OBS_ERROR_STD=0.7), np.ones(3))
        assert ab.covariance(np.array([123.0]))[0, 0] == pytest.approx(0.49)

    def test_per_gauge_floor(self):
        obs = np.column_stack([np.full(4, 10.0), np.full(4, 1000.0)])
        err = StreamflowObservationError.from_config(EnKFConfig(), obs)
        np.testing.assert_allclose(err.std(np.array([0.0]), np.array([1])), [10.0])


class TestScores:
    def test_crps_matches_gaussian_closed_form(self):
        rng = np.random.default_rng(0)
        ens = rng.standard_normal((200, 400))
        # CRPS(N(0,1), 0) = 2 phi(0) - 1/sqrt(pi)
        expected = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
        assert dg.crps(ens, np.zeros(200)) == pytest.approx(expected, rel=0.02)

    def test_crps_of_one_member_is_absolute_error(self):
        obs = np.array([1.0, 2.0, np.nan])
        assert dg.crps(np.array([2.0, 0.0, 5.0]), obs) == pytest.approx(1.5)

    def test_reliable_ensemble_has_unit_spread_error_ratio(self):
        rng = np.random.default_rng(1)
        truth_and_members = rng.standard_normal((4000, 11))
        ratio = dg.spread_error_ratio(truth_and_members[:, 1:], truth_and_members[:, 0])
        assert ratio == pytest.approx(1.0, abs=0.05)

    def test_persistence_lag(self):
        np.testing.assert_array_equal(dg.persistence_forecast(np.array([1.0, 2.0, 3.0])),
                                      [np.nan, 1.0, 2.0])

    def test_lead_time_benchmarks(self):
        obs = np.arange(10, dtype=float)
        ens = np.repeat(obs[:, None], 3, axis=1)
        # perfect forecasts issued at steps 2 and 5 for 3 leads
        fc = np.stack([ens[t + 1:t + 4] for t in (2, 5)])
        table = dg.lead_time_scores(fc, [2, 5], obs, open_loop_predictions=ens + 1.0)
        assert list(table['n']) == [2, 2, 2]
        np.testing.assert_allclose(table['crps'], 0.0)
        np.testing.assert_allclose(table['crps_open_loop'], 1.0)
        # persistence holds obs[issue]: error = lead
        np.testing.assert_allclose(table['mae_persistence'], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(table['crpss_vs_persistence'], 1.0)
        np.testing.assert_allclose(table['skill_vs_open_loop'], 1.0)

    def test_lead_time_drops_targets_past_the_end(self):
        obs = np.arange(5, dtype=float)
        fc = np.ones((1, 3, 2))
        table = dg.lead_time_scores(fc, [3], obs)
        assert list(table['n']) == [1, 0, 0]

    def test_point_scores_matches_pr512_contract(self):
        o = pd.Series([1.0, 2.0, 3.0, 4.0])
        out = dg.point_scores(o + 1, o, {'persistence': o.shift(1)})
        assert set(out) == {'n', 'bias', 'mae', 'rmse', 'nse', 'skill_vs_persistence'}
        assert out['n'] == 3 and out['bias'] == pytest.approx(1.0)
        assert out['skill_vs_persistence'] == pytest.approx(0.0)

    def test_nse_kge_perfect(self):
        x = np.array([1.0, 3.0, 2.0, 5.0])
        assert dg.nse(x, x) == pytest.approx(1.0)
        assert dg.kge(x, x) == pytest.approx(1.0)


class TestBiasStateBounds:
    def test_log_bias_is_not_clipped_at_zero(self):
        """The log-bias may go negative; non-negativity applies to storages only."""
        alg = EnKFAlgorithm(rng=np.random.default_rng(0))
        X = np.column_stack([np.linspace(1, 2, 20), np.linspace(-0.5, -0.1, 20),
                             np.linspace(1, 3, 20)])
        H = np.array([[0.0, 0.0, 1.0]])
        out = alg.analyze(X, np.array([0.5]), H, np.array([[0.01]]),
                          nonnegative=np.array([True, False, True]))
        assert (out[:, 1] < 0).any()
        assert (out[:, [0, 2]] >= 0).all()


def _conceptual_twin(manager_factory, forcing, n_members=20, seed=0):
    """Run the filter on a JAX conceptual model: truth = one more member."""
    truth = manager_factory(1, forcing)
    truth.initialize_members()
    n = len(forcing['precip'])
    q_true = np.array([truth.step(t, {k: np.atleast_1d(v[t]) for k, v in forcing.items()
                                      if k != 'time'})[0] for t in range(n)])
    rng = np.random.default_rng(seed)
    obs = np.maximum(q_true * (1 + 0.1 * rng.standard_normal(n)), 0.0)
    measured = dict(forcing)
    pert = GaussianPerturbation(precip_std=0.3, rng=np.random.default_rng(seed + 1))
    measured['precip'] = forcing['precip'] * pert.forcing_noise('precip', 1, n, 2.0)[0]
    ens = manager_factory(n_members, measured)
    ens.initialize_members()
    X = ens.get_state()
    ens.set_state(X)
    np.testing.assert_array_equal(ens.get_state(), X)
    mgr = DataAssimilationManager.__new__(DataAssimilationManager)
    mgr.logger = logging.getLogger("test.da_added_value")
    result = mgr.run_filter(
        enkf=EnKFAlgorithm(rng=np.random.default_rng(seed + 2)),
        enkf_config=EnKFConfig(ENKF_ENSEMBLE_SIZE=n_members), ensemble_mgr=ens,
        state_specs=None, obs_model_units=obs, warmup_steps=30,
        rng=np.random.default_rng(seed + 3))
    return DataAssimilationManager.verify(result, 30, 24)[0]


def _forcing(n=240, seed=0):
    rng = np.random.default_rng(seed)
    return {'precip': rng.gamma(0.6, 8.0, n),
            'temp': 8 + 6 * np.sin(np.arange(n) / 30.0),
            'pet': np.full(n, 1.5),
            'time': pd.date_range("2000-01-01", periods=n, freq="D")}


class TestConceptualModelsBehindTheInterface:
    def test_hbv(self):
        pytest.importorskip("jhbv")
        pytest.importorskip("jax")
        from jhbv.model import PARAM_BOUNDS
        from jhbv.parameters import DEFAULT_PARAMS

        from symfluence.data_assimilation.enkf.ensemble_manager import HBVEnsembleManager

        def factory(n, forcing):
            params = {**DEFAULT_PARAMS, 'fc': 200.0, 'k1': 0.1, 'k2': 0.02}
            bounds = {k: v for k, v in PARAM_BOUNDS.items() if k in params}
            return HBVEnsembleManager(n, params, bounds, forcing,
                                      GaussianPerturbation(param_std=0.0))

        summary = _conceptual_twin(factory, _forcing())
        assert summary['crpss_vs_open_loop'] > 0.0

    def test_hbv_jax_and_numpy_agree(self):
        pytest.importorskip("jhbv")
        pytest.importorskip("jax")
        from jhbv.model import PARAM_BOUNDS
        from jhbv.parameters import DEFAULT_PARAMS

        from symfluence.data_assimilation.enkf.ensemble_manager import HBVEnsembleManager

        forcing = _forcing(30)
        out = []
        for use_jax in (True, False):
            m = HBVEnsembleManager(3, dict(DEFAULT_PARAMS), dict(PARAM_BOUNDS), forcing,
                                   GaussianPerturbation(param_std=0.05,
                                                        rng=np.random.default_rng(0)),
                                   use_jax=use_jax)
            m.initialize_members()
            out.append(np.array([m.step(t, {k: np.full(3, forcing[k][t])
                                            for k in ('precip', 'temp', 'pet')})
                                 for t in range(30)]))
        np.testing.assert_allclose(out[0], out[1], rtol=1e-5, atol=1e-8)

    def test_xinanjiang(self):
        pytest.importorskip("jxaj")
        pytest.importorskip("jax")
        from jxaj.parameters import DEFAULT_PARAMS, PARAM_BOUNDS

        from symfluence.data_assimilation.enkf.ensemble_manager import XinanjiangEnsembleManager

        def factory(n, forcing):
            return XinanjiangEnsembleManager(n, dict(DEFAULT_PARAMS), dict(PARAM_BOUNDS),
                                             forcing, GaussianPerturbation(param_std=0.0))

        summary = _conceptual_twin(factory, _forcing())
        assert summary['crpss_vs_open_loop'] > 0.0
