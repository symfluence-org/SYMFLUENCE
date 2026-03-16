# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data Assimilation Manager.

Top-level orchestrator for the EnKF data assimilation workflow.
Coordinates ensemble initialization, forecast–analysis cycling,
and output writing.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from symfluence.core.base_manager import BaseManager

from .config import DataAssimilationConfig, EnKFConfig
from .enkf.enkf_algorithm import EnKFAlgorithm
from .enkf.ensemble_manager import HBVEnsembleManager
from .enkf.observation_operator import StreamflowObservationOperator
from .enkf.perturbation import GaussianPerturbation
from .enkf.state_vector import StateVariableSpec, StateVector

logger = logging.getLogger(__name__)


class DataAssimilationManager(BaseManager):
    """Orchestrates EnKF data assimilation for hydrological models.

    Workflow:
        1. Load config (model, DA settings, observation paths)
        2. Build observation schedule from observation timestamps
        3. Initialize ensemble (via EnsembleManager)
        4. For each assimilation window:
           a. Forecast: advance all members from t to t+dt
           b. Extract: read states and predictions from all members
           c. Observe: load observations at t+dt
           d. Analyze: apply EnKF update (if observation exists)
           e. Inject: write updated states back to members
           f. Record: store ensemble mean/spread
        5. Write output NetCDF
        6. Return output path
    """

    def _initialize_services(self) -> None:
        """Initialize DA-specific services."""
        pass

    def run_data_assimilation(self) -> Optional[Path]:
        """Execute the full data assimilation workflow.

        Returns:
            Path to the output NetCDF file, or None on failure.
        """
        self.logger.info("Starting data assimilation workflow")

        try:
            # 1. Load DA configuration
            da_config = self._get_da_config()
            enkf_config = da_config.enkf

            # 2. Build EnKF algorithm
            enkf = EnKFAlgorithm(
                inflation_factor=enkf_config.inflation_factor,
                enforce_nonnegative=enkf_config.enforce_nonnegative_states,
            )

            # 3. Initialize ensemble manager (model-specific)
            ensemble_mgr, state_specs = self._create_ensemble_manager(enkf_config)
            ensemble_mgr.initialize_members()

            # 4. Load observations
            obs_times, obs_values = self._load_observations(
                enkf_config.assimilation_variable
            )

            if obs_times is None or len(obs_values) == 0:
                self.logger.error("No observations available for assimilation")
                return None

            # 5. Set up state vector and observation operator
            state_vector = StateVector(state_specs)
            n_state = state_vector.n_state

            if enkf_config.augment_state_with_predictions:
                obs_operator = StreamflowObservationOperator(n_obs=1)
                H = obs_operator.get_matrix(n_state + 1)
            else:
                obs_operator = StreamflowObservationOperator(n_obs=1)
                H = obs_operator.get_matrix(n_state)

            # 6. Set up observation error covariance
            R = self._build_obs_error_covariance(enkf_config, obs_values)

            # 7. Assimilation loop
            n_timesteps = len(ensemble_mgr.forcing['precip'])
            interval = enkf_config.assimilation_interval

            # Storage for results
            ensemble_predictions = []
            ensemble_means = []
            ensemble_stds = []
            observed_values = []
            analysis_times = []

            obs_idx = 0
            for t in range(n_timesteps):
                # a. Forecast step
                ensemble_mgr.forecast_step(t, t + 1)

                # b. Extract
                X_states = ensemble_mgr.extract_states()
                predictions = ensemble_mgr.extract_predictions(
                    enkf_config.assimilation_variable
                )

                # Store ensemble predictions
                ensemble_predictions.append(predictions.copy())
                ensemble_means.append(float(predictions.mean()))
                ensemble_stds.append(float(predictions.std()))

                # c. Check if observation is available at this timestep
                if obs_idx < len(obs_values) and t % interval == 0:
                    y_obs = np.atleast_1d(obs_values[obs_idx])

                    if not np.isnan(y_obs).any():
                        # d. Analyze
                        if enkf_config.augment_state_with_predictions:
                            X_aug = state_vector.augment_with_predictions(X_states, predictions)
                            X_aug_a = enkf.analyze(X_aug, y_obs, H, R, enkf_config.filter_variant)
                            X_a, _ = state_vector.split_augmented(X_aug_a, 1)
                        else:
                            X_a = enkf.analyze(X_states, y_obs, H, R, enkf_config.filter_variant)

                        # Enforce bounds
                        X_a = state_vector.enforce_bounds(X_a)

                        # e. Inject
                        ensemble_mgr.inject_states(X_a)

                        analysis_times.append(t)
                        observed_values.append(float(y_obs[0]))
                    else:
                        observed_values.append(np.nan)

                    obs_idx += 1
                else:
                    observed_values.append(np.nan)

            # 8. Write output
            output_path = self._write_output(
                ensemble_predictions=np.array(ensemble_predictions),
                ensemble_means=np.array(ensemble_means),
                ensemble_stds=np.array(ensemble_stds),
                observed_values=np.array(observed_values[:n_timesteps]),
                analysis_times=analysis_times,
            )

            self.logger.info(
                "Data assimilation completed: %d timesteps, %d analyses",
                n_timesteps, len(analysis_times)
            )
            return output_path

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error("Data assimilation failed: %s", e)
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_da_config(self) -> DataAssimilationConfig:
        """Get data assimilation configuration."""
        da = self._get_config_value(
            lambda: self.config.data_assimilation,
            None
        )
        if da is None:
            da = DataAssimilationConfig()
        return da

    def _create_ensemble_manager(self, enkf_config: EnKFConfig):
        """Create the appropriate ensemble manager for the configured model.

        Returns:
            Tuple of (EnsembleManager, List[StateVariableSpec]).
        """
        model_name = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            'HBV'
        ).split(',')[0].strip().upper()

        if model_name == 'HBV':
            return self._create_hbv_ensemble(enkf_config)
        else:
            raise NotImplementedError(
                f"EnKF ensemble manager not implemented for {model_name}. "
                "Currently supported: HBV"
            )

    def _create_hbv_ensemble(self, enkf_config: EnKFConfig):
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

        return ensemble, state_specs

    def _load_observations(self, variable: str):
        """Load observation time series for assimilation.

        Returns:
            Tuple of (times, values) arrays.
        """
        obs_path = self.project_observations_dir / 'streamflow' / 'preprocessed'
        domain_name = self._get_config_value(lambda: self.config.domain.name, 'domain')

        obs_file = obs_path / f"{domain_name}_streamflow_processed.csv"
        if not obs_file.exists():
            self.logger.warning("Observation file not found: %s", obs_file)
            return None, np.array([])

        import pandas as pd
        df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
        return df.index.values, df.iloc[:, 0].values

    def _build_obs_error_covariance(
        self, enkf_config: EnKFConfig, obs_values: np.ndarray
    ) -> np.ndarray:
        """Build observation error covariance matrix R.

        Args:
            enkf_config: EnKF configuration.
            obs_values: Observation values (for relative error).

        Returns:
            R matrix of shape (n_obs, n_obs).
        """
        if enkf_config.obs_error_type == 'relative':
            obs_std = enkf_config.obs_error_std * np.nanmean(np.abs(obs_values))
        else:
            obs_std = enkf_config.obs_error_std

        return np.array([[obs_std ** 2]])

    def _write_output(
        self,
        ensemble_predictions: np.ndarray,
        ensemble_means: np.ndarray,
        ensemble_stds: np.ndarray,
        observed_values: np.ndarray,
        analysis_times: List[int],
    ) -> Path:
        """Write DA results to NetCDF.

        Args:
            ensemble_predictions: Per-member predictions (n_timesteps, n_members).
            ensemble_means: Ensemble mean streamflow (n_timesteps,).
            ensemble_stds: Ensemble std streamflow (n_timesteps,).
            observed_values: Observed streamflow (n_timesteps,).
            analysis_times: Timestep indices where analysis was performed.

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
            ensemble_predictions=ensemble_predictions,
            ensemble_means=ensemble_means,
            ensemble_stds=ensemble_stds,
            observed_values=observed_values,
            analysis_times=analysis_times,
        )

        self.logger.info("DA results written to %s", output_path)
        return output_path
