# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data Assimilation Manager.

Top-level orchestrator for the EnKF data assimilation workflow.
Coordinates ensemble initialization, forecast–analysis cycling,
and output writing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from symfluence.core.base_manager import BaseManager

from . import diagnostics
from .config import DataAssimilationConfig, EnKFConfig
from .cycling import CycleResult, CycleSettings, draw_forcing_noise, run_cycle
from .enkf.enkf_algorithm import EnKFAlgorithm
from .enkf.ensemble_manager import HBVEnsembleManager, XinanjiangEnsembleManager
from .enkf.observation_error import StreamflowObservationError
from .enkf.perturbation import GaussianPerturbation
from .enkf.state_vector import StateVariableSpec

logger = logging.getLogger(__name__)


# EnKF options that are accepted by the configuration but not implemented.
# (field, alias, default)
_UNIMPLEMENTED_OPTIONS = (
    ('localization_radius', 'ENKF_LOCALIZATION_RADIUS', None),
    ('estimate_parameters', 'ENKF_ESTIMATE_PARAMS', False),
)

#: Models with an in-process ensemble manager.
SUPPORTED_MODELS = ('HBV', 'XINANJIANG')

# Kept for callers of the first EnKF API.
FilterResult = CycleResult


class DataAssimilationManager(BaseManager):
    """Orchestrates EnKF data assimilation for hydrological models.

    Workflow:
        1. Load config (model, DA settings, observation paths)
        2. Initialize the ensemble through the model-agnostic state
           interface (:class:`~.enkf.ensemble_manager.DAModel`)
        3. Load observations and align them to the model time axis by
           timestamp, converted to the model's prediction units
        4. Draw the period's forcing perturbations (AR(1) in time)
        5. Cycle (:func:`~.cycling.run_cycle`): filter ensemble with EnKF
           analyses, the open-loop ensemble (same members and perturbations,
           no updates) and optional lead-time forecasts
        6. Verify against the open loop and persistence
           (:mod:`~.diagnostics`), write NetCDF + JSON/CSV
        7. Return the NetCDF path
    """

    def _initialize_services(self) -> None:
        """Initialize DA-specific services."""
        pass

    def run_data_assimilation(self) -> Path:
        """Execute the full data assimilation workflow.

        Returns:
            Path to the output NetCDF file. A ``*_verification.json`` with
            the added-value summary (and ``*_lead_scores.csv`` when forecasts
            are enabled) is written next to it.

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
        obs_error = StreamflowObservationError.from_config(
            enkf_config, obs_model_units,
            cms_to_model=self.cms_to_model_units(timestep_hours, area_m2),
        )

        # 5. Forecast-analysis loop (filter, open loop, forecasts)
        result = self.run_filter(
            enkf=enkf,
            enkf_config=enkf_config,
            ensemble_mgr=ensemble_mgr,
            state_specs=state_specs,
            obs_model_units=obs_model_units,
            warmup_steps=warmup_steps,
            rng=rng,
            obs_error=obs_error,
        )

        # 6. Verification and output
        summary, lead_scores = self.verify(result, warmup_steps, timestep_hours)
        self._log_summary(summary)
        output_path = self._write_output(
            result,
            time_index=time_index,
            global_attrs={
                'catchment_area_m2': float(area_m2) if area_m2 else 'not used',
                'timestep_hours': timestep_hours,
                'warmup_timesteps': warmup_steps,
                'obs_error_model': obs_error.kind,
                'forcing_correlation_hours': float(enkf_config.forcing_correlation_hours),
            },
        )
        self._write_verification(output_path, summary, lead_scores)

        self.logger.info(
            "Data assimilation completed: %d timesteps, %d analyses",
            len(result.ensemble_means), len(result.analysis_times)
        )
        return output_path

    # ------------------------------------------------------------------
    # Filter loop
    # ------------------------------------------------------------------

    @staticmethod
    def cycle_settings(enkf_config: EnKFConfig, warmup_steps: int,
                       timestep_hours: float) -> CycleSettings:
        """Translate EnKF configuration into :class:`CycleSettings`."""
        lead_steps = int(round(float(enkf_config.forecast_lead_days) * 24.0
                               / float(timestep_hours)))
        return CycleSettings(
            assimilation_interval=enkf_config.assimilation_interval,
            warmup_steps=warmup_steps,
            filter_variant=enkf_config.filter_variant,
            augment_state=enkf_config.augment_state_with_predictions,
            enforce_nonnegative=enkf_config.enforce_nonnegative_states,
            open_loop=enkf_config.open_loop,
            forecast_steps=lead_steps,
            forecast_issue_interval=enkf_config.forecast_issue_interval,
            precip_bias=enkf_config.precip_bias_state,
            precip_bias_std=enkf_config.precip_bias_std,
            precip_bias_walk_std=enkf_config.precip_bias_walk_std,
        )

    def run_filter(
        self,
        enkf: EnKFAlgorithm,
        enkf_config: EnKFConfig,
        ensemble_mgr,
        state_specs: Optional[List[StateVariableSpec]],
        obs_model_units: np.ndarray,
        warmup_steps: int = 0,
        rng: Optional[np.random.Generator] = None,
        obs_error: Optional[StreamflowObservationError] = None,
        forcing_noise: Optional[Mapping[str, np.ndarray]] = None,
    ) -> CycleResult:
        """Run the forecast-analysis cycle over the whole model period.

        Args:
            enkf: Analysis algorithm.
            enkf_config: EnKF settings.
            ensemble_mgr: An initialised :class:`DAModel`.
            state_specs: State layout (default: ``ensemble_mgr.state_specs``).
            obs_model_units: Observations per model timestep, (n_t,) or
                (n_t, n_gauges), in the units of the model's predictions;
                NaN where there is no observation.
            warmup_steps: Leading timesteps during which nothing is
                assimilated (the model spins up; errors there are dominated
                by the arbitrary initial state).
            rng: Generator for forcing noise and bias draws.
            obs_error: Observation-error model (default: built from
                ``enkf_config`` and the observations, with the default
                absolute floor).
            forcing_noise: Pre-drawn forcing perturbations; drawn from the
                manager's perturbation strategy when omitted.

        Returns:
            :class:`CycleResult` with filter priors, open loop and forecasts.
        """
        rng = rng if rng is not None else np.random.default_rng()
        obs = np.asarray(obs_model_units, dtype=float)
        n_t = len(np.asarray(ensemble_mgr.forcing['precip']))
        if obs.shape[0] != n_t:
            raise ValueError(
                f"Observations must be aligned to the model time axis: expected "
                f"{n_t} values, got shape {obs.shape}."
            )
        if obs_error is None:
            obs_error = StreamflowObservationError.from_config(enkf_config, obs)
        if forcing_noise is None:
            perturbation = getattr(ensemble_mgr, 'perturbation', None)
            if perturbation is not None and getattr(perturbation, 'rng', None) is None:
                perturbation.rng = rng
            forcing_noise = draw_forcing_noise(
                perturbation, ensemble_mgr, n_t,
                enkf_config.forcing_correlation_hours,
                enabled=enkf_config.forcing_perturbation,
            )
        settings = self.cycle_settings(
            enkf_config, warmup_steps, float(getattr(ensemble_mgr, 'timestep_hours', 24)))
        return run_cycle(
            ensemble_mgr, enkf, obs, obs_error, settings,
            forcing_noise=forcing_noise, rng=rng, state_specs=state_specs,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify(result: CycleResult, warmup_steps: int, timestep_hours: float):
        """Added-value summary of a single-gauge run.

        Returns:
            (summary dict, lead-time DataFrame or None). The summary scores
            the filter priors (one-step forecasts), the open loop and
            persistence after warm-up on a common sample.
        """
        obs = np.asarray(result.observed_values, dtype=float)
        if obs.ndim != 1:
            raise ValueError("verify() handles one gauge; call diagnostics per gauge.")
        mask = np.arange(len(obs)) >= warmup_steps
        summary = diagnostics.verification_summary(
            result.ensemble_predictions, obs,
            open_loop_predictions=result.open_loop_predictions,
            persistence=diagnostics.persistence_forecast(obs, lag=1),
            mask=mask,
        )
        lead_scores = None
        if result.forecasts is not None and len(result.forecast_issue_steps):
            lead_scores = diagnostics.lead_time_scores(
                result.forecasts, result.forecast_issue_steps, obs,
                open_loop_predictions=result.open_loop_predictions,
                timestep_hours=timestep_hours,
            )
        return summary, lead_scores

    def _log_summary(self, summary: Dict[str, Any]) -> None:
        enkf = summary['enkf']
        self.logger.info(
            "EnKF one-step priors (n=%d): CRPS %.4g, spread/error %.2f, NSE %.3f, KGE %.3f",
            summary['n'], enkf['crps'], enkf['spread_error_ratio'], enkf['nse'], enkf['kge'])
        if 'open_loop' in summary:
            self.logger.info("Open loop: CRPS %.4g; CRPSS of EnKF vs open loop %.3f",
                             summary['open_loop']['crps'], summary['crpss_vs_open_loop'])
        if 'persistence' in summary:
            self.logger.info("Persistence: MAE %.4g; CRPSS of EnKF vs persistence %.3f",
                             summary['persistence']['crps'], summary['crpss_vs_persistence'])

    @staticmethod
    def cms_to_model_units(timestep_hours: float, area_m2: Optional[float]) -> Optional[float]:
        """Factor converting m3/s into mm/timestep (None without an area)."""
        if area_m2 is None or not np.isfinite(area_m2) or area_m2 <= 0:
            return None
        return float(timestep_hours) * 3600.0 / float(area_m2) * 1000.0

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
        if model_name in ('XINANJIANG', 'XAJ'):
            return self._create_xinanjiang_ensemble(enkf_config, rng)
        raise NotImplementedError(
            f"EnKF ensemble manager not implemented for {model_name}. "
            f"Currently supported: {', '.join(SUPPORTED_MODELS)}. Other models "
            "can take part by implementing the DAModel state interface "
            "(symfluence.data_assimilation.enkf.ensemble_manager)."
        )

    @staticmethod
    def _perturbation(enkf_config: EnKFConfig,
                      rng: Optional[np.random.Generator]) -> GaussianPerturbation:
        return GaussianPerturbation(
            param_std=enkf_config.param_perturbation_std,
            precip_std=enkf_config.precip_perturbation_std,
            temp_std=enkf_config.temp_perturbation_std,
            rng=rng,
        )

    def _create_hbv_ensemble(
        self, enkf_config: EnKFConfig, rng: Optional[np.random.Generator] = None
    ):
        """Create HBV-specific ensemble manager."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.runner import HBVRunner

        runner = HBVRunner(self.config, self.logger)
        forcing, _ = runner._load_forcing()

        base_params = runner._get_default_params()
        bounds = {k: v for k, v in PARAM_BOUNDS.items() if k in base_params}

        ensemble = HBVEnsembleManager(
            n_members=enkf_config.ensemble_size,
            base_params=base_params,
            param_bounds=bounds,
            forcing=forcing,
            perturbation=self._perturbation(enkf_config, rng),
            use_jax=(runner.backend == 'jax'),
            timestep_hours=runner.timestep_hours,
            warmup_days=runner.warmup_days,
        )
        # HBV predicts runoff depth (mm/timestep); observations are discharge
        # (m3/s). The runner's own area drives the conversion, exactly as it
        # does when the runner writes streamflow.
        ensemble.catchment_area_m2 = float(runner._get_catchment_area())

        return ensemble, ensemble.state_specs

    def _create_xinanjiang_ensemble(
        self, enkf_config: EnKFConfig, rng: Optional[np.random.Generator] = None
    ):
        """Create the stand-alone Xinanjiang ensemble manager (daily, JAX)."""
        from jxaj.parameters import DEFAULT_PARAMS, PARAM_BOUNDS
        from jxaj.runner import XinanjiangRunner

        runner = XinanjiangRunner(self.config, self.logger)
        if str(getattr(runner, 'snow_module', 'none')).lower() not in ('none', ''):
            raise NotImplementedError(
                "EnKF for Xinanjiang supports the stand-alone model only "
                f"(XINANJIANG_SNOW_MODULE={runner.snow_module!r}).")
        forcing = runner._load_forcing()
        base_params = dict(getattr(runner, '_external_params', None) or DEFAULT_PARAMS)
        bounds = {k: v for k, v in PARAM_BOUNDS.items() if k in base_params}

        ensemble = XinanjiangEnsembleManager(
            n_members=enkf_config.ensemble_size,
            base_params=base_params,
            param_bounds=bounds,
            forcing=forcing,
            perturbation=self._perturbation(enkf_config, rng),
            warmup_days=runner.warmup_days,
        )
        ensemble.catchment_area_m2 = float(runner._get_catchment_area())
        return ensemble, ensemble.state_specs

    def _write_output(
        self,
        result: CycleResult,
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
            open_loop_predictions=result.open_loop_predictions,
            forecasts=result.forecasts,
            forecast_issue_steps=result.forecast_issue_steps,
            precip_bias=result.precip_bias,
        )

        self.logger.info("DA results written to %s", output_path)
        return output_path

    def _write_verification(self, output_path: Path, summary: Dict[str, Any],
                            lead_scores: Optional[pd.DataFrame]) -> None:
        """Write the added-value summary (JSON) and lead-time scores (CSV)."""
        stem = output_path.with_suffix('')
        json_path = Path(f"{stem}_verification.json")

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            if isinstance(obj, (float, np.floating)):
                return float(obj) if np.isfinite(obj) else None
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        json_path.write_text(json.dumps(_clean(summary), indent=2))
        self.logger.info("DA verification written to %s", json_path)
        if lead_scores is not None:
            csv_path = Path(f"{stem}_lead_scores.csv")
            lead_scores.to_csv(csv_path, index=False)
            self.logger.info("Lead-time scores written to %s", csv_path)
