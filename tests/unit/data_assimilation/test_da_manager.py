"""Tests for the EnKF driver (DataAssimilationManager) with a toy model.

The toy ensemble is a linear store ``S' = S + P - k*S`` whose "streamflow" is
``k*S`` in mm/timestep, so the driver's pairing of observations with model
steps, unit conversion, warm-up handling and observation error can be checked
without any hydrological model installed.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from symfluence.core.config.models.state_config import EnKFConfig
from symfluence.data_assimilation.da_manager import DataAssimilationManager
from symfluence.data_assimilation.enkf.enkf_algorithm import EnKFAlgorithm
from symfluence.data_assimilation.enkf.ensemble_manager import EnsembleManager
from symfluence.data_assimilation.enkf.state_vector import StateVariableSpec

AREA_M2 = 100.0e6  # 100 km2
K = 0.5


class ToyEnsemble(EnsembleManager):
    """N linear stores with member-specific initial storage."""

    def __init__(self, n_members, precip, times, timestep_hours=24, warmup_days=0,
                 spread=5.0, seed=0):
        super().__init__(n_members)
        self.forcing = {'precip': np.asarray(precip, dtype=float), 'time': np.asarray(times)}
        self.timestep_hours = timestep_hours
        self.warmup_days = warmup_days
        self.catchment_area_m2 = AREA_M2
        self._rng = np.random.default_rng(seed)
        self._spread = spread
        self.storage = np.zeros(n_members)
        self.q = np.zeros(n_members)
        self.injections = []

    def initialize_members(self, **kwargs):
        self.storage = 20.0 + self._spread * self._rng.standard_normal(self.n_members)

    def forecast_step(self, t_start, t_end):
        self.storage = self.storage + self.forcing['precip'][int(t_start)]
        self.q = K * self.storage
        self.storage = self.storage - self.q

    def extract_states(self):
        return self.storage.reshape(-1, 1).copy()

    def inject_states(self, updated_states):
        self.injections.append(self._last_step)
        self.storage = np.asarray(updated_states)[:, 0].copy()

    def extract_predictions(self, variable='streamflow'):
        return self.q.copy()

    # record which step an injection happened at
    _last_step = -1


class _Recording(ToyEnsemble):
    def forecast_step(self, t_start, t_end):
        self._last_step = int(t_start)
        super().forecast_step(t_start, t_end)


def _manager():
    """A DataAssimilationManager without the config/filesystem plumbing."""
    mgr = DataAssimilationManager.__new__(DataAssimilationManager)
    mgr.logger = logging.getLogger("test.da_manager")
    return mgr


def _cms(mm_per_step, timestep_hours=24):
    """mm/timestep -> m3/s for the toy catchment (the HBV runner's formula)."""
    return np.asarray(mm_per_step) * AREA_M2 / (1000.0 * timestep_hours * 3600.0)


SPECS = [StateVariableSpec('storage', 1, lower_bound=0.0)]


def _run(ensemble, obs_model_units, warmup_steps=0, **cfg):
    config = EnKFConfig(ENKF_ENSEMBLE_SIZE=ensemble.n_members, **cfg)
    enkf = EnKFAlgorithm(rng=np.random.default_rng(1))
    ensemble.initialize_members()
    return _manager().run_filter(
        enkf=enkf, enkf_config=config, ensemble_mgr=ensemble, state_specs=SPECS,
        obs_model_units=obs_model_units, warmup_steps=warmup_steps,
    )


class TestAlignObservations:
    def test_paired_by_timestamp_not_position(self):
        """Observations that start 3 days into the model period must land on
        steps 3.. (the old driver paired obs[0] with model step 0)."""
        times = pd.date_range("2020-01-01", periods=10, freq="D")
        obs = pd.Series(_cms([1.0, 2.0, 3.0]),
                        index=pd.date_range("2020-01-04", periods=3, freq="D"))
        aligned = DataAssimilationManager.align_observations(obs, times, 24, AREA_M2)
        assert np.isnan(aligned[:3]).all()
        np.testing.assert_allclose(aligned[3:6], [1.0, 2.0, 3.0])
        assert np.isnan(aligned[6:]).all()

    def test_units_converted_with_catchment_area(self):
        times = pd.date_range("2020-01-01", periods=2, freq="D")
        # 100 km2 at 1 m3/s for a day = 86400 m3 / 1e8 m2 = 0.864 mm
        obs = pd.Series([1.0, 2.0], index=times)
        aligned = DataAssimilationManager.align_observations(obs, times, 24, AREA_M2)
        np.testing.assert_allclose(aligned, [0.864, 1.728])

    def test_gaps_and_out_of_period_observations_give_no_update(self):
        times = pd.date_range("2020-01-01", periods=4, freq="D")
        obs = pd.Series(
            _cms([9.0, 1.0, np.nan, 4.0, 9.0]),
            index=pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02",
                                  "2020-01-04", "2020-01-05"]),
        )
        aligned = DataAssimilationManager.align_observations(obs, times, 24, AREA_M2)
        np.testing.assert_allclose(aligned, [1.0, np.nan, np.nan, 4.0])

    def test_subdaily_observations_averaged_into_model_step(self):
        times = pd.date_range("2020-01-01", periods=2, freq="D")
        obs = pd.Series(_cms([1.0] * 24 + [3.0] * 24),
                        index=pd.date_range("2020-01-01", periods=48, freq="h"))
        aligned = DataAssimilationManager.align_observations(obs, times, 24, AREA_M2)
        np.testing.assert_allclose(aligned, [1.0, 3.0])

    @pytest.mark.parametrize("area", [None, 0.0, -1.0, float("nan")])
    def test_missing_area_is_an_error(self, area):
        times = pd.date_range("2020-01-01", periods=2, freq="D")
        obs = pd.Series([1.0, 2.0], index=times)
        with pytest.raises(ValueError, match="catchment area"):
            DataAssimilationManager.align_observations(obs, times, 24, area)


class TestRunFilter:
    def test_updates_only_where_an_observation_exists(self):
        n = 12
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        obs = np.full(n, np.nan)
        obs[[4, 7]] = [12.0, 11.0]
        ens = _Recording(30, np.full(n, 10.0), times)
        result = _run(ens, obs)
        assert result.analysis_times == [4, 7]
        assert ens.injections == [4, 7]
        assert result.ensemble_predictions.shape == (n, 30)
        np.testing.assert_array_equal(result.observed_values, obs)

    def test_no_assimilation_during_warmup(self):
        n = 10
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        ens = _Recording(30, np.full(n, 10.0), times)
        result = _run(ens, np.full(n, 12.0), warmup_steps=4)
        assert result.analysis_times == list(range(4, n))

    def test_assimilation_interval(self):
        n = 9
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        ens = _Recording(30, np.full(n, 10.0), times)
        result = _run(ens, np.full(n, 12.0), ENKF_ASSIMILATION_INTERVAL=3)
        assert result.analysis_times == [0, 3, 6]

    def test_analysis_pulls_ensemble_towards_truth(self):
        """Twin experiment: assimilating the truth must beat the open loop."""
        n = 40
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.default_rng(3)
        precip = rng.gamma(2.0, 4.0, size=n)

        truth = _Recording(1, precip, times, spread=0.0)
        truth.initialize_members()
        truth.storage[:] = 60.0  # far from the ensemble's ~20
        q_truth = []
        for t in range(n):
            truth.forecast_step(t, t + 1)
            q_truth.append(truth.q[0])
        q_truth = np.array(q_truth)

        open_loop = _run(_Recording(40, precip, times, seed=5), np.full(n, np.nan))
        assimilated = _run(_Recording(40, precip, times, seed=5), q_truth,
                           ENKF_OBS_ERROR_STD=0.05)

        err_open = np.abs(open_loop.ensemble_means[1:6] - q_truth[1:6]).mean()
        err_da = np.abs(assimilated.ensemble_means[1:6] - q_truth[1:6]).mean()
        assert open_loop.analysis_times == []
        assert err_da < 0.5 * err_open

    def test_misaligned_observations_are_rejected(self):
        times = pd.date_range("2020-01-01", periods=5, freq="D")
        ens = _Recording(10, np.full(5, 1.0), times)
        with pytest.raises(ValueError, match="aligned to the model time axis"):
            _run(ens, np.ones(3))


class TestObservationError:
    def test_relative_error_follows_the_observation_at_that_step(self):
        cfg = EnKFConfig(ENKF_OBS_ERROR_STD=0.1, ENKF_OBS_ERROR_TYPE='relative')
        low = DataAssimilationManager._obs_error_covariance(cfg, 1.0, floor=1e-9)
        high = DataAssimilationManager._obs_error_covariance(cfg, 50.0, floor=1e-9)
        assert low[0, 0] == pytest.approx(0.1 ** 2)
        assert high[0, 0] == pytest.approx(5.0 ** 2)

    def test_relative_error_has_a_floor(self):
        cfg = EnKFConfig(ENKF_OBS_ERROR_STD=0.1, ENKF_OBS_ERROR_TYPE='relative')
        obs = np.array([0.0, 10.0, np.nan, 30.0])
        floor = DataAssimilationManager._relative_error_floor(cfg, obs)
        assert floor > 0.0
        zero_flow = DataAssimilationManager._obs_error_covariance(cfg, 0.0, floor)
        assert zero_flow[0, 0] == pytest.approx(floor ** 2)

    def test_absolute_error(self):
        cfg = EnKFConfig(ENKF_OBS_ERROR_STD=0.7, ENKF_OBS_ERROR_TYPE='absolute')
        R = DataAssimilationManager._obs_error_covariance(cfg, 123.0, floor=5.0)
        assert R[0, 0] == pytest.approx(0.49)


class TestDriver:
    def _wired(self, tmp_path, ensemble, obs, **cfg):
        mgr = _manager()
        enkf_cfg = EnKFConfig(ENKF_ENSEMBLE_SIZE=ensemble.n_members, **cfg)
        mgr._config = SimpleNamespace(
            data_assimilation=SimpleNamespace(enkf=enkf_cfg),
            system=SimpleNamespace(random_seed=7),
        )
        mgr._create_ensemble_manager = lambda enkf_config, rng=None: (ensemble, SPECS)
        mgr._load_observations = lambda variable: obs
        mgr.__dict__['project_dir'] = tmp_path
        mgr.__dict__['experiment_id'] = 'exp'
        return mgr

    def test_end_to_end_writes_aligned_output(self, tmp_path):
        import xarray as xr

        n = 20
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        ens = _Recording(20, np.full(n, 10.0), times, warmup_days=5)
        # observations cover only days 10..14, in m3/s
        obs = pd.Series(_cms(np.full(5, 12.0)),
                        index=pd.date_range("2020-01-11", periods=5, freq="D"))
        mgr = self._wired(tmp_path, ens, obs)

        out = mgr.run_data_assimilation()

        with xr.open_dataset(out) as ds:
            performed = np.flatnonzero(ds['analysis_performed'].values)
            np.testing.assert_array_equal(performed, [10, 11, 12, 13, 14])
            # observations are stored in the model's units, on its time axis
            np.testing.assert_allclose(ds['observed_streamflow'].values[10:15], 12.0)
            assert np.isnan(ds['observed_streamflow'].values[:10]).all()
            assert pd.Timestamp(ds['time'].values[0]) == times[0]
            # the placeholder "effective ensemble size" is gone
            assert 'effective_ensemble_size' not in ds

    def test_warmup_days_honoured(self, tmp_path):
        n = 12
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        ens = _Recording(20, np.full(n, 10.0), times, warmup_days=8)
        obs = pd.Series(_cms(np.full(n, 12.0)), index=times)
        mgr = self._wired(tmp_path, ens, obs)
        mgr.run_data_assimilation()
        assert ens.injections == [8, 9, 10, 11]

    def test_failure_raises_instead_of_returning_none(self, tmp_path):
        n = 5
        times = pd.date_range("2020-01-01", periods=n, freq="D")
        ens = _Recording(10, np.full(n, 10.0), times)
        # observations entirely outside the model period
        obs = pd.Series(_cms(np.ones(3)),
                        index=pd.date_range("1999-01-01", periods=3, freq="D"))
        mgr = self._wired(tmp_path, ens, obs)
        with pytest.raises(ValueError, match="No streamflow observations overlap"):
            mgr.run_data_assimilation()

    def test_unimplemented_options_warn(self, caplog):
        mgr = _manager()
        cfg = EnKFConfig(ENKF_LOCALIZATION_RADIUS=50.0, ENKF_FORCING_PERTURBATION=False,
                         ENKF_ESTIMATE_PARAMS=True)
        with caplog.at_level(logging.WARNING, logger="test.da_manager"):
            mgr._warn_unimplemented_options(cfg)
        text = caplog.text
        for alias in ("ENKF_LOCALIZATION_RADIUS", "ENKF_FORCING_PERTURBATION",
                      "ENKF_ESTIMATE_PARAMS"):
            assert alias in text

    def test_default_options_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="test.da_manager"):
            _manager()._warn_unimplemented_options(EnKFConfig())
        assert caplog.text == ""
