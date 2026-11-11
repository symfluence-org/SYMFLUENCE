# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data Assimilation Manager.

Top-level orchestrator for the EnKF data assimilation workflow.
Coordinates ensemble initialization, forecast–analysis cycling,
and output writing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.base_manager import BaseManager

from .config import DataAssimilationConfig, EnKFConfig
from .enkf.enkf_algorithm import EnKFAlgorithm
from .enkf.ensemble_manager import HBVEnsembleManager
from .enkf.observation_operator import StreamflowObservationOperator
from .enkf.perturbation import GaussianPerturbation
from .enkf.state_vector import StateVariableSpec, StateVector

logger = logging.getLogger(__name__)


# Lower bound on the "relative" observation error, as a fraction of the mean
# observed flow times ENKF_OBS_ERROR_STD. Without it a zero or near-zero
# observation would get R ~ 0 and be trusted absolutely.
_RELATIVE_ERROR_FLOOR_FRACTION = 0.1

# EnKF options that are accepted by the configuration but not implemented.
# (field, alias, default)
_UNIMPLEMENTED_OPTIONS = (
    ('localization_radius', 'ENKF_LOCALIZATION_RADIUS', None),
    ('forcing_perturbation', 'ENKF_FORCING_PERTURBATION', True),
    ('estimate_parameters', 'ENKF_ESTIMATE_PARAMS', False),
)


@dataclass
class FilterResult:
    """Arrays produced by one pass of the forecast-analysis loop."""

    ensemble_predictions: np.ndarray  # (n_timesteps, n_members), model units
    ensemble_means: np.ndarray        # (n_timesteps,)
    ensemble_stds: np.ndarray         # (n_timesteps,)
    observed_values: np.ndarray       # (n_timesteps,), model units, NaN = none
    analysis_times: List[int] = field(default_factory=list)


class DataAssimilationManager(BaseManager):
    """Orchestrates EnKF data assimilation for hydrological models.

    Workflow:
        1. Load config (model, DA settings, observation paths)
        2. Initialize ensemble (via EnsembleManager)
        3. Load observations and align them to the model time axis by
           timestamp, converted to the model's prediction units
        4. For each model timestep:
           a. Forecast: advance all members from t to t+dt
           b. Extract: read states and predictions from all members
           c. Analyze: apply the EnKF update -- only after warm-up, on an
              assimilation step, and when an observation exists for that
              timestamp
           d. Inject: write updated states back to members
           e. Record: store ensemble mean/spread
        5. Write output NetCDF
        6. Return output path
    """

    def _initialize_services(self) -> None:
        """Initialize DA-specific services."""
        pass

    def run_data_assimilation(self) -> Path:
        """Execute the full data assimilation workflow.

        Returns:
            Path to the output NetCDF file.

        Raises:
            Exception: Any failure propagates. This method used to log the
                error and return ``None``, which let the workflow carry on
                and report success with no assimilation results.
        """
        self.logger.info("Starting data assimilation workflow")

        # 1. Load DA configuration
        da_config = self._get_da_config()
        enkf_config = da_config.enkf
        self._warn_unimplemented_options(enkf_config)

        # One generator for the whole run, seeded from RANDOM_SEED when set
        seed = self._get_config_value(
            lambda: self.config.system.random_seed, None, dict_key='RANDOM_SEED'
        )
        rng = np.random.default_rng(seed)

        # 2. Build EnKF algorithm
        enkf = EnKFAlgorithm(
            inflation_factor=enkf_config.inflation_factor,
            enforce_nonnegative=enkf_config.enforce_nonnegative_states,
            rng=rng,
        )

        # 3. Initialize ensemble manager (model-specific)
        ensemble_mgr, state_specs = self._create_ensemble_manager(enkf_config, rng)
        ensemble_mgr.initialize_members()

        # 4. Observations on the model time axis, in model units
        time_index = self._model_time_index(ensemble_mgr)
        timestep_hours = float(getattr(ensemble_mgr, 'timestep_hours', 24))
        obs_series = self._load_observations(enkf_config.assimilation_variable)
        area_m2 = getattr(ensemble_mgr, 'catchment_area_m2', None)
        obs_model_units = self.align_observations(
            obs_series, time_index, timestep_hours, area_m2
        )
        n_valid = int(np.count_nonzero(~np.isnan(obs_model_units)))
        if n_valid == 0:
            raise ValueError(
                "No streamflow observations overlap the model period "
                f"({time_index[0]} to {time_index[-1]}); nothing to assimilate."
            )
        self.logger.info(
            "Aligned %d observation(s) to %d model timestep(s)", n_valid, len(time_index)
        )

        warmup_steps = self._warmup_steps(
            getattr(ensemble_mgr, 'warmup_days', 0), timestep_hours
        )

        # 5. Forecast-analysis loop
        result = self.run_filter(
            enkf=enkf,
            enkf_config=enkf_config,
            ensemble_mgr=ensemble_mgr,
            state_specs=state_specs,
            obs_model_units=obs_model_units,
            warmup_steps=warmup_steps,
        )

        # 6. Write output
        output_path = self._write_output(
            result,
            time_index=time_index,
            global_attrs={
                'catchment_area_m2': float(area_m2) if area_m2 else 'not used',
                'timestep_hours': timestep_hours,
                'warmup_timesteps': warmup_steps,
            },
        )

        self.logger.info(
            "Data assimilation completed: %d timesteps, %d analyses",
            len(result.ensemble_means), len(result.analysis_times)
        )
        return output_path

    # ------------------------------------------------------------------
    # Filter loop
    # ------------------------------------------------------------------

    def run_filter(
        self,
        enkf: EnKFAlgorithm,
        enkf_config: EnKFConfig,
        ensemble_mgr,
        state_specs: List[StateVariableSpec],
        obs_model_units: np.ndarray,
        warmup_steps: int = 0,
    ) -> FilterResult:
        """Run the forecast-analysis cycle over the whole model period.

        Args:
            enkf: Analysis algorithm.
            enkf_config: EnKF settings.
            ensemble_mgr: Ensemble manager; ``forcing['precip']`` gives the
                number of timesteps.
            state_specs: State layout.
            obs_model_units: Observation per model timestep (same length as
                the model time axis), already in the units of
                ``ensemble_mgr.extract_predictions``; NaN where there is no
                observation.
            warmup_steps: Leading timesteps during which nothing is
                assimilated (the model spins up; errors there are dominated
                by the arbitrary initial state).

        Returns:
            FilterResult with per-timestep ensemble statistics.
        """
        n_timesteps = len(ensemble_mgr.forcing['precip'])
        obs_model_units = np.asarray(obs_model_units, dtype=float)
        if obs_model_units.shape != (n_timesteps,):
            raise ValueError(
                f"Observations must be aligned to the model time axis: expected "
                f"{n_timesteps} values, got shape {obs_model_units.shape}."
            )
        interval = enkf_config.assimilation_interval

        state_vector = StateVector(state_specs)
        n_state = state_vector.n_state
        augment = enkf_config.augment_state_with_predictions
        obs_operator = StreamflowObservationOperator(n_obs=1)
        H = obs_operator.get_matrix(n_state + 1 if augment else n_state)

        obs_error_floor = self._relative_error_floor(enkf_config, obs_model_units)

        ensemble_predictions = []
        ensemble_means = []
        ensemble_stds = []
        analysis_times: List[int] = []

        for t in range(n_timesteps):
            # a. Forecast step
            ensemble_mgr.forecast_step(t, t + 1)

            # b. Extract
            X_states = ensemble_mgr.extract_states()
            predictions = ensemble_mgr.extract_predictions(
                enkf_config.assimilation_variable
            )

            ensemble_predictions.append(predictions.copy())
            ensemble_means.append(float(predictions.mean()))
            ensemble_stds.append(float(predictions.std()))

            # c. Analyze only after warm-up, on an assimilation step, and when
            #    this timestamp actually has an observation.
            y = obs_model_units[t]
            if t < warmup_steps or t % interval != 0 or np.isnan(y):
                continue

            y_obs = np.atleast_1d(y)
            R = self._obs_error_covariance(enkf_config, float(y), obs_error_floor)

            if augment:
                X_aug = state_vector.augment_with_predictions(X_states, predictions)
                X_aug_a = enkf.analyze(X_aug, y_obs, H, R, enkf_config.filter_variant)
                X_a, _ = state_vector.split_augmented(X_aug_a, 1)
            else:
                X_a = enkf.analyze(X_states, y_obs, H, R, enkf_config.filter_variant)

            X_a = state_vector.enforce_bounds(X_a)

            # d. Inject
            ensemble_mgr.inject_states(X_a)
            analysis_times.append(t)

        return FilterResult(
            ensemble_predictions=np.array(ensemble_predictions),
            ensemble_means=np.array(ensemble_means),
            ensemble_stds=np.array(ensemble_stds),
            observed_values=obs_model_units,
            analysis_times=analysis_times,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    @staticmethod
    def _model_time_index(ensemble_mgr) -> pd.DatetimeIndex:
        """The model time axis (one stamp per timestep) from the forcing."""
        times = ensemble_mgr.forcing.get('time') if hasattr(ensemble_mgr.forcing, 'get') else None
        if times is None:
            raise ValueError(
                "The ensemble forcing carries no 'time' axis; observations cannot "
                "be paired with model timesteps without timestamps."
            )
        index = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
        if len(index) != len(ensemble_mgr.forcing['precip']):
            raise ValueError("Forcing 'time' and 'precip' have different lengths.")
        return index

    @staticmethod
    def _warmup_steps(warmup_days: Any, timestep_hours: float) -> int:
        """Warm-up length in model timesteps."""
        days = float(warmup_days or 0)
        if days <= 0:
            return 0
        return int(round(days * 24.0 / float(timestep_hours)))

    @staticmethod
    def align_observations(
        obs_cms: pd.Series,
        time_index: pd.DatetimeIndex,
        timestep_hours: float,
        catchment_area_m2: Optional[float],
    ) -> np.ndarray:
        """Put discharge observations on the model time axis in mm/timestep.

        Observations are matched to model timesteps by timestamp, never by
        position: each model step ``[t, t + dt)`` gets the mean of the
        observations stamped inside it, and NaN (no update) when there are
        none. The conversion from discharge to runoff depth is the inverse of
        the one the HBV runner applies to its output::

            runoff [mm/timestep] = Q [m3/s] * dt [s] / area [m2] * 1000

        Args:
            obs_cms: Observed discharge in m3/s with a datetime index.
            time_index: Model timestep start times.
            timestep_hours: Model timestep length.
            catchment_area_m2: Catchment area. Required: discharge cannot be
                compared with a runoff depth without it.

        Returns:
            Array aligned with ``time_index``.
        """
        if catchment_area_m2 is None or not np.isfinite(catchment_area_m2) \
                or catchment_area_m2 <= 0:
            raise ValueError(
                "A positive catchment area is required to convert observed "
                "discharge (m3/s) to the model's runoff depth (mm/timestep); "
                f"got {catchment_area_m2!r}. Provide a catchment shapefile or "
                "CATCHMENT_AREA."
            )
        if not isinstance(obs_cms.index, pd.DatetimeIndex):
            raise ValueError("Observations must be indexed by datetime.")

        obs = obs_cms.astype(float)
        obs = obs[~obs.index.isna()].sort_index()
        if time_index.tz is None and obs.index.tz is not None:
            obs.index = obs.index.tz_convert('UTC').tz_localize(None)

        if not time_index.is_monotonic_increasing:
            raise ValueError("The model time axis must be increasing.")

        # Bin each observation into the model step [start, start + dt) that
        # contains its timestamp; observations outside every step are dropped.
        step = pd.Timedelta(hours=float(timestep_hours))
        values = obs.to_numpy()
        positions = time_index.searchsorted(obs.index, side='right') - 1
        starts = time_index[np.clip(positions, 0, None)]
        keep = (
            (positions >= 0)
            & np.asarray((obs.index - starts) < step)
            & ~np.isnan(values)
        )
        aligned_cms = np.full(len(time_index), np.nan)
        if keep.any():
            binned = pd.Series(values[keep]).groupby(positions[keep]).mean()
            aligned_cms[binned.index.to_numpy()] = binned.to_numpy()

        seconds_per_timestep = float(timestep_hours) * 3600.0
        return aligned_cms * seconds_per_timestep / float(catchment_area_m2) * 1000.0

    def _load_observations(self, variable: str) -> pd.Series:
        """Load the observed discharge series (m3/s, datetime-indexed).

        Raises:
            FileNotFoundError: If the preprocessed observation file is absent.
        """
        obs_path = self.project_observations_dir / 'streamflow' / 'preprocessed'
        domain_name = self._get_config_value(lambda: self.config.domain.name, 'domain')

        obs_file = obs_path / f"{domain_name}_streamflow_processed.csv"
        if not obs_file.exists():
            raise FileNotFoundError(
                f"Observation file not found: {obs_file}. Run 'process_observed_data' "
                "before data assimilation."
            )

        df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
        if df.shape[1] == 0 or df.empty:
            raise ValueError(f"Observation file has no data: {obs_file}")
        return df.iloc[:, 0]

    # ------------------------------------------------------------------
    # Observation error
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_error_floor(enkf_config: EnKFConfig, obs_model_units: np.ndarray) -> float:
        """Smallest observation error std used with ``obs_error_type: relative``."""
        valid = obs_model_units[~np.isnan(obs_model_units)]
        mean_abs = float(np.mean(np.abs(valid))) if valid.size else 0.0
        return max(
            enkf_config.obs_error_std * _RELATIVE_ERROR_FLOOR_FRACTION * mean_abs,
            1e-9,
        )

    @staticmethod
    def _obs_error_covariance(
        enkf_config: EnKFConfig, y_obs: float, floor: float
    ) -> np.ndarray:
        """Observation error covariance R for one observation.

        ``relative``: the std is ``obs_error_std`` times *this* observation
        (not the mean of the whole series, which made low flows far too
        uncertain and floods far too certain), bounded below by ``floor``.
        ``absolute``: ``obs_error_std`` is the std itself, in model units.
        """
        if enkf_config.obs_error_type == 'relative':
            obs_std = max(enkf_config.obs_error_std * abs(y_obs), floor)
        else:
            obs_std = enkf_config.obs_error_std
        return np.array([[obs_std ** 2]])

    def _warn_unimplemented_options(self, enkf_config: EnKFConfig) -> None:
        """Tell the user when a non-default EnKF option will have no effect."""
        for attr, alias, default in _UNIMPLEMENTED_OPTIONS:
            value = getattr(enkf_config, attr, default)
            if value != default:
                self.logger.warning(
                    "%s=%r is set but not implemented by the EnKF driver; it has "
                    "no effect on this run.", alias, value,
                )

    def _get_da_config(self) -> DataAssimilationConfig:
        """Get data assimilation configuration."""
        da = self._get_config_value(
            lambda: self.config.data_assimilation,
            None
        )
        if da is None:
            da = DataAssimilationConfig()
        return da

    def _create_ensemble_manager(
        self, enkf_config: EnKFConfig, rng: Optional[np.random.Generator] = None
    ):
        """Create the appropriate ensemble manager for the configured model.

        Returns:
            Tuple of (EnsembleManager, List[StateVariableSpec]).
        """
        model_name = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            'HBV'
        ).split(',')[0].strip().upper()

        if model_name == 'HBV':
            return self._create_hbv_ensemble(enkf_config, rng)
        else:
            raise NotImplementedError(
                f"EnKF ensemble manager not implemented for {model_name}. "
                "Currently supported: HBV"
            )

    def _create_hbv_ensemble(
        self, enkf_config: EnKFConfig, rng: Optional[np.random.Generator] = None
    ):
        """Create HBV-specific ensemble manager."""
        from jhbv.model import PARAM_BOUNDS

        # Load forcing
        from jhbv.runner import HBVRunner
        runner = HBVRunner(self.config, self.logger)
        forcing, _ = runner._load_forcing()

        # Get params and bounds
        base_params = runner._get_default_params()
        bounds = {k: v for k, v in PARAM_BOUNDS.items() if k in base_params}

        # Perturbation
        perturbation = GaussianPerturbation(
            param_std=enkf_config.param_perturbation_std,
            precip_std=enkf_config.precip_perturbation_std,
            temp_std=enkf_config.temp_perturbation_std,
            rng=rng,
        )

        # Routing buffer size
        from jhbv.model import get_routing_buffer_length
        routing_len = get_routing_buffer_length(10, runner.timestep_hours)

        # State variable specs
        state_specs = [
            StateVariableSpec('snow', 1, lower_bound=0.0),
            StateVariableSpec('snow_water', 1, lower_bound=0.0),
            StateVariableSpec('sm', 1, lower_bound=0.0),
            StateVariableSpec('suz', 1, lower_bound=0.0),
            StateVariableSpec('slz', 1, lower_bound=0.0),
            StateVariableSpec('routing_buffer', routing_len, lower_bound=0.0),
        ]

        ensemble = HBVEnsembleManager(
            n_members=enkf_config.ensemble_size,
            base_params=base_params,
            param_bounds=bounds,
            forcing=forcing,
            perturbation=perturbation,
            use_jax=(runner.backend == 'jax'),
            timestep_hours=runner.timestep_hours,
            warmup_days=runner.warmup_days,
        )
        # HBV predicts runoff depth (mm/timestep); observations are discharge
        # (m3/s). The runner's own area drives the conversion, exactly as it
        # does when the runner writes streamflow.
        ensemble.catchment_area_m2 = float(runner._get_catchment_area())

        return ensemble, state_specs

    def _write_output(
        self,
        result: FilterResult,
        time_index: Optional[pd.DatetimeIndex] = None,
        global_attrs: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write DA results to NetCDF.

        Args:
            result: Output of :meth:`run_filter`.
            time_index: Model time axis for the ``time`` coordinate.
            global_attrs: Extra global attributes.

        Returns:
            Path to the output file.
        """
        from .output import DAOutputManager

        output_dir = self.project_dir / 'simulations' / self.experiment_id / 'data_assimilation'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{self.experiment_id}_enkf_results.nc"

        writer = DAOutputManager()
        writer.write(
            output_path=output_path,
            ensemble_predictions=result.ensemble_predictions,
            ensemble_means=result.ensemble_means,
            ensemble_stds=result.ensemble_stds,
            observed_values=result.observed_values,
            analysis_times=result.analysis_times,
            time_index=None if time_index is None else time_index.to_numpy(),
            global_attrs=global_attrs,
        )

        self.logger.info("DA results written to %s", output_path)
        return output_path
