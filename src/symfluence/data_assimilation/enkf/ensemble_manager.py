# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Ensemble manager.

Manages the lifecycle and execution of ensemble members for the EnKF.
Provides base class and model-specific implementations for HBV
(in-memory) and subprocess-based models (SUMMA, etc.).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .perturbation import GaussianPerturbation, PerturbationStrategy

logger = logging.getLogger(__name__)


class EnsembleManager(ABC):
    """Abstract base class for ensemble management.

    Handles member initialization, forecast execution, and state
    extraction/injection for the EnKF assimilation loop.

    Args:
        n_members: Number of ensemble members.
        perturbation: Perturbation strategy for member generation.
    """

    def __init__(
        self,
        n_members: int,
        perturbation: Optional[PerturbationStrategy] = None,
    ):
        self.n_members = n_members
        self.perturbation = perturbation or GaussianPerturbation()

    @abstractmethod
    def initialize_members(self, **kwargs) -> None:
        """Create N members with perturbed parameters/states."""
        ...

    @abstractmethod
    def forecast_step(self, t_start: Any, t_end: Any) -> None:
        """Advance all members from t_start to t_end."""
        ...

    @abstractmethod
    def extract_states(self) -> np.ndarray:
        """Extract state matrix (n_members, n_state) from all members."""
        ...

    @abstractmethod
    def inject_states(self, updated_states: np.ndarray) -> None:
        """Write updated state matrix back into members."""
        ...

    @abstractmethod
    def extract_predictions(self, variable: str = 'streamflow') -> np.ndarray:
        """Extract predicted observations (n_members,) from all members."""
        ...


class HBVEnsembleManager(EnsembleManager):
    """Ensemble manager for in-memory HBV model.

    Uses direct function calls (optionally vectorized with JAX vmap)
    for per-timestep advancement with no subprocess overhead.

    Args:
        n_members: Number of ensemble members.
        base_params: Base parameter dictionary.
        param_bounds: Parameter bounds {name: (lower, upper)}.
        forcing: Dict with 'precip', 'temp', 'pet' arrays.
        perturbation: Perturbation strategy.
        use_jax: Whether to use JAX backend.
        timestep_hours: Model timestep in hours.
        warmup_days: Number of warmup days.
    """

    def __init__(
        self,
        n_members: int,
        base_params: Dict[str, float],
        param_bounds: Dict[str, tuple],
        forcing: Dict[str, np.ndarray],
        perturbation: Optional[PerturbationStrategy] = None,
        use_jax: bool = True,
        timestep_hours: int = 24,
        warmup_days: int = 0,
    ):
        super().__init__(n_members, perturbation)
        self.base_params = base_params
        self.param_bounds = param_bounds
        self.forcing = forcing
        self.use_jax = use_jax
        self.timestep_hours = timestep_hours
        self.warmup_days = warmup_days

        # Per-member state
        self.member_params: List[Dict[str, float]] = []
        self.member_states: List[Any] = []
        self.member_predictions: List[float] = []
        self._current_step: int = 0

    def initialize_members(self, initial_state=None, **kwargs) -> None:
        """Create N members with perturbed parameters.

        Args:
            initial_state: Optional base HBVState to perturb from.
        """
        from jhbv.model import create_initial_state

        self.member_params = []
        self.member_states = []

        for i in range(self.n_members):
            # Perturb parameters
            params = self.perturbation.perturb_parameters(
                self.base_params, self.param_bounds, i
            )
            self.member_params.append(params)

            # Initial state (optionally perturbed)
            if initial_state is not None:
                self.member_states.append(initial_state)
            else:
                state = create_initial_state(
                    use_jax=self.use_jax,
                    timestep_hours=self.timestep_hours,
                )
                self.member_states.append(state)

        self._current_step = 0
        logger.info("Initialized %d HBV ensemble members", self.n_members)

    def forecast_step(self, t_start: Any, t_end: Any) -> None:
        """Advance all members by one timestep.

        t_start and t_end are interpreted as integer timestep indices.
        """
        from jhbv.model import (
            HAS_JAX,
            create_params_from_dict,
            scale_params_for_timestep,
            step_jax,
        )

        t = int(t_start)
        self.member_predictions = []

        for i in range(self.n_members):
            # Get forcing for this timestep
            precip = self.forcing['precip'][t]
            temp = self.forcing['temp'][t]
            pet = self.forcing['pet'][t]

            # Optionally perturb forcing
            if hasattr(self.perturbation, 'perturb_forcing'):
                precip_arr = self.perturbation.perturb_forcing(
                    np.atleast_1d(precip), i, 'precip'
                )
                temp_arr = self.perturbation.perturb_forcing(
                    np.atleast_1d(temp), i, 'temp'
                )
                precip = float(precip_arr[0])
                temp = float(temp_arr[0])

            # Scale params
            scaled = scale_params_for_timestep(self.member_params[i], self.timestep_hours)
            hbv_params = create_params_from_dict(scaled, use_jax=(self.use_jax and HAS_JAX))

            # Convert to JAX if needed
            if self.use_jax and HAS_JAX:
                import jax.numpy as jnp
                p, t_val, e = jnp.array(precip), jnp.array(temp), jnp.array(pet)
            else:
                p, t_val, e = precip, temp, pet

            # Single timestep
            new_state, runoff = step_jax(p, t_val, e, self.member_states[i], hbv_params, self.timestep_hours)
            self.member_states[i] = new_state
            self.member_predictions.append(float(np.asarray(runoff)))

        self._current_step = t + 1

    def extract_states(self) -> np.ndarray:
        """Extract state as (n_members, n_state) matrix."""
        states = []
        for state in self.member_states:
            sv = np.concatenate([
                np.atleast_1d(np.asarray(state.snow, dtype=np.float64)),
                np.atleast_1d(np.asarray(state.snow_water, dtype=np.float64)),
                np.atleast_1d(np.asarray(state.sm, dtype=np.float64)),
                np.atleast_1d(np.asarray(state.suz, dtype=np.float64)),
                np.atleast_1d(np.asarray(state.slz, dtype=np.float64)),
                np.asarray(state.routing_buffer, dtype=np.float64),
            ])
            states.append(sv)
        return np.array(states)

    def inject_states(self, updated_states: np.ndarray) -> None:
        """Write updated state matrix back into member HBVState objects."""
        from jhbv.model import HAS_JAX, HBVState

        for i in range(self.n_members):
            sv = updated_states[i]
            routing_len = len(self.member_states[i].routing_buffer)

            if self.use_jax and HAS_JAX:
                import jax.numpy as jnp
                self.member_states[i] = HBVState(
                    snow=jnp.array(sv[0]),
                    snow_water=jnp.array(sv[1]),
                    sm=jnp.array(sv[2]),
                    suz=jnp.array(sv[3]),
                    slz=jnp.array(sv[4]),
                    routing_buffer=jnp.array(sv[5:5 + routing_len]),
                )
            else:
                self.member_states[i] = HBVState(
                    snow=np.float64(sv[0]),
                    snow_water=np.float64(sv[1]),
                    sm=np.float64(sv[2]),
                    suz=np.float64(sv[3]),
                    slz=np.float64(sv[4]),
                    routing_buffer=sv[5:5 + routing_len].copy(),
                )

    def extract_predictions(self, variable: str = 'streamflow') -> np.ndarray:
        return np.array(self.member_predictions)


class SubprocessEnsembleManager(EnsembleManager):
    """Ensemble manager for subprocess-based models (SUMMA, etc.).

    Creates per-member directories and runs the model as subprocesses,
    using the StateCapableMixin interface for state I/O.

    Args:
        n_members: Number of ensemble members.
        model_runner_factory: Callable that creates a model runner for a member.
        work_dir: Root working directory for ensemble members.
        perturbation: Perturbation strategy.
    """

    def __init__(
        self,
        n_members: int,
        model_runner_factory: Any,
        work_dir: Path,
        perturbation: Optional[PerturbationStrategy] = None,
    ):
        super().__init__(n_members, perturbation)
        self.model_runner_factory = model_runner_factory
        self.work_dir = Path(work_dir)
        self.member_dirs: List[Path] = []
        self.member_runners: List[Any] = []

    def initialize_members(self, **kwargs) -> None:
        """Create per-member directory layout and runners."""
        self.member_dirs = []
        self.member_runners = []

        for i in range(self.n_members):
            member_dir = self.work_dir / f"member_{i:03d}"
            member_dir.mkdir(parents=True, exist_ok=True)
            (member_dir / 'settings').mkdir(exist_ok=True)
            (member_dir / 'output').mkdir(exist_ok=True)
            (member_dir / 'state').mkdir(exist_ok=True)

            runner = self.model_runner_factory(member_dir, i)
            self.member_dirs.append(member_dir)
            self.member_runners.append(runner)

        logger.info("Initialized %d subprocess ensemble members in %s", self.n_members, self.work_dir)

    def forecast_step(self, t_start: Any, t_end: Any) -> None:
        """Run all members for the given time window.

        Each runner must implement a method that accepts time bounds.
        """
        for i, runner in enumerate(self.member_runners):
            try:
                runner.run_forecast_window(t_start, t_end)
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                logger.error("Member %d forecast failed: %s", i, e)

    def extract_states(self) -> np.ndarray:
        """Extract states from all member runners via StateCapableMixin."""
        from symfluence.models.state import StateCapableMixin

        all_states = []
        for runner in self.member_runners:
            if isinstance(runner, StateCapableMixin):
                state = runner.save_state(
                    runner.get_state_directory() or self.work_dir / 'state',
                    timestamp='',
                )
                # Flatten arrays to 1-D vector
                arrays = list(state.arrays.values())
                if arrays:
                    all_states.append(np.concatenate([np.atleast_1d(a) for a in arrays]))

        return np.array(all_states) if all_states else np.array([])

    def inject_states(self, updated_states: np.ndarray) -> None:
        """Inject updated states back into member runners."""
        from symfluence.models.state import ModelState, StateCapableMixin, StateMetadata

        for i, runner in enumerate(self.member_runners):
            if not isinstance(runner, StateCapableMixin):
                continue

            # Reconstruct state dict from flat vector
            variables = runner.get_state_variables()
            # Simple: assume each variable is a scalar except the last which gets remaining
            sv = updated_states[i]
            arrays = {}
            offset = 0
            for j, var in enumerate(variables):
                if j < len(variables) - 1:
                    arrays[var] = sv[offset:offset + 1]
                    offset += 1
                else:
                    arrays[var] = sv[offset:]

            model_name = runner._get_model_name() if hasattr(runner, '_get_model_name') else 'unknown'
            metadata = StateMetadata(
                model_name=model_name,
                timestamp='',
                format=runner.get_state_format(),
                variables=variables,
                ensemble_member=i,
            )
            state = ModelState(metadata=metadata, arrays=arrays)
            runner.load_state(state)

    def extract_predictions(self, variable: str = 'streamflow') -> np.ndarray:
        """Extract predictions from member outputs (model-specific)."""
        predictions = []
        for runner in self.member_runners:
            # Default: try to read from output file
            try:
                import xarray as xr
                output_dir = getattr(runner, 'output_dir', None)
                if output_dir:
                    nc_files = sorted(Path(output_dir).glob('*output*.nc'))
                    if nc_files:
                        ds = xr.open_dataset(nc_files[-1])
                        if variable in ds:
                            predictions.append(float(ds[variable].values[-1]))
                            ds.close()
                            continue
                        ds.close()
                predictions.append(np.nan)
            except Exception:  # noqa: BLE001 — must-not-raise contract
                predictions.append(np.nan)

        return np.array(predictions)
