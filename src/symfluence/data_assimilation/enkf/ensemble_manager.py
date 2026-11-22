# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Ensemble managers: the model-agnostic state interface used by the EnKF.

The filter driver never touches model internals. A model takes part in data
assimilation by implementing :class:`DAModel` (or subclassing
:class:`EnsembleManager`, which declares the same methods abstractly):

``initialize_members()``
    Create the N members (parameters perturbed, initial states set).
``get_state() -> X``
    The complete prognostic state of every member, shape (n_members, n_state),
    laid out as ``state_specs`` says. Complete means: ``set_state(get_state())``
    followed by ``step`` reproduces the run exactly. The driver relies on this
    to run the open loop and lead-time forecasts on the same model object by
    swapping state matrices in and out.
``set_state(X)``
    Overwrite every member's state.
``step(t, forcing) -> predictions``
    Advance every member over timestep ``t`` with the given per-member forcing
    (``{name: array (n_members,)}``) and return the predicted observable(s),
    shape (n_members,) or (n_members, n_obs), in model units.
``perturbable_forcing``
    ``{name: 'multiplicative' | 'additive'}``: forcing variables the driver may
    perturb, and how. Other entries of ``forcing`` are passed through unchanged.

Forcing perturbation, the precipitation bias state, and the open-loop and
forecast bookkeeping all live in the driver, so a new model only has to
expose its state.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from .perturbation import GaussianPerturbation, PerturbationStrategy
from .state_vector import StateVariableSpec

logger = logging.getLogger(__name__)


@runtime_checkable
class DAModel(Protocol):
    """What the EnKF driver needs from a model (see the module docstring)."""

    n_members: int
    state_specs: List[StateVariableSpec]
    forcing: Mapping[str, Any]
    perturbable_forcing: Mapping[str, str]
    timestep_hours: float

    def initialize_members(self) -> None: ...

    def get_state(self) -> np.ndarray: ...

    def set_state(self, X: np.ndarray) -> None: ...

    def step(self, t: int, forcing: Mapping[str, np.ndarray]) -> np.ndarray: ...


class EnsembleManager(ABC):
    """Base class for ensemble managers implementing :class:`DAModel`.

    Args:
        n_members: Number of ensemble members.
        perturbation: Perturbation strategy for the parameters.
    """

    #: Forcing variables the driver may perturb and how.
    perturbable_forcing: Mapping[str, str] = {}
    #: Model timestep length.
    timestep_hours: float = 24.0
    #: Spin-up length during which nothing is assimilated.
    warmup_days: float = 0.0
    #: Catchment area for converting discharge to the model's runoff depth.
    catchment_area_m2: Optional[float] = None

    def __init__(
        self,
        n_members: int,
        perturbation: Optional[PerturbationStrategy] = None,
    ):
        self.n_members = n_members
        self.perturbation = perturbation or GaussianPerturbation()
        self.state_specs: List[StateVariableSpec] = []
        self.forcing: Dict[str, Any] = {}

    @abstractmethod
    def initialize_members(self, **kwargs) -> None:
        """Create N members with perturbed parameters/states."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """State matrix (n_members, n_state) of all members."""

    @abstractmethod
    def set_state(self, X: np.ndarray) -> None:
        """Write a state matrix (n_members, n_state) into the members."""

    @abstractmethod
    def step(self, t: int, forcing: Mapping[str, np.ndarray]) -> np.ndarray:
        """Advance all members over timestep ``t``; return predictions."""

    # Older names, kept for callers written against the first EnKF API.
    def extract_states(self) -> np.ndarray:
        return self.get_state()

    def inject_states(self, updated_states: np.ndarray) -> None:
        self.set_state(updated_states)


# ----------------------------------------------------------------------
# Models whose state and parameters are NamedTuples of arrays (the JAX
# conceptual models): one batched step over all members.
# ----------------------------------------------------------------------

def _stack_namedtuples(items: List[Any], xp) -> Any:
    """Stack a list of NamedTuples leaf-wise along a new leading axis."""
    cls = type(items[0])
    return cls(*[xp.stack([xp.asarray(getattr(it, f)) for it in items])
                 for f in cls._fields])


def _batched_step(fn: Callable, use_jax: bool) -> Callable:
    """``fn`` maps one member; the result maps arrays with a leading member axis."""
    if use_jax:
        import jax
        return jax.jit(jax.vmap(fn))

    def loop(*args):
        n = len(np.atleast_1d(args[0]))
        results = []
        for i in range(n):
            member_args = [
                type(a)(*[np.asarray(leaf)[i] for leaf in a]) if hasattr(a, '_fields')
                else np.asarray(a)[i]
                for a in args
            ]
            results.append(fn(*member_args))
        states = _stack_namedtuples([r[0] for r in results], np)
        return states, np.array([float(np.asarray(r[1])) for r in results])

    return loop


class NamedTupleEnsembleManager(EnsembleManager):
    """Shared machinery for models with NamedTuple state and parameters.

    Subclasses set ``self._params`` (batched parameters), ``self._state``
    (batched state) and ``self._state_cls``, and call ``self._compile(fn)``
    with the one-member step ``fn(*forcing, state, params) -> (state, q)``
    in ``initialize_members``. ``_step_forcing`` names the forcing arguments
    in order.
    """

    _step_forcing: tuple = ()

    def __init__(self, n_members: int, perturbation: Optional[PerturbationStrategy],
                 use_jax: bool):
        super().__init__(n_members, perturbation)
        self.use_jax = bool(use_jax)
        self._state: Any = None
        self._params: Any = None
        self._state_cls: Any = None
        self._step: Optional[Callable] = None

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp
            return jnp
        return np

    def _compile(self, fn: Callable) -> None:
        self._step = _batched_step(fn, self.use_jax)

    def get_state(self) -> np.ndarray:
        cols = [np.asarray(getattr(self._state, spec.name), dtype=np.float64)
                .reshape(self.n_members, -1) for spec in self.state_specs]
        return np.hstack(cols)

    def set_state(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        xp = self._xp
        values, offset = {}, 0
        for spec in self.state_specs:
            block = X[:, offset:offset + spec.size]
            values[spec.name] = xp.asarray(block[:, 0] if spec.size == 1 else block)
            offset += spec.size
        self._state = self._state_cls(**values)

    def step(self, t: int, forcing: Mapping[str, np.ndarray]) -> np.ndarray:
        if self._step is None:
            raise RuntimeError("initialize_members() must be called before step()")
        xp = self._xp
        args = [xp.asarray(np.asarray(forcing[name], dtype=np.float64))
                for name in self._step_forcing]
        self._state, q = self._step(*args, self._state, self._params)
        return np.asarray(q, dtype=np.float64).reshape(self.n_members)


class HBVEnsembleManager(NamedTupleEnsembleManager):
    """Ensemble of HBV-96 members (``jhbv``), advanced together.

    With JAX the members are advanced by one jit-compiled ``vmap`` of
    ``jhbv.model.step_jax``; without it, member by member.

    Args:
        n_members: Number of ensemble members.
        base_params: Base parameter dictionary.
        param_bounds: Parameter bounds {name: (lower, upper)}.
        forcing: Dict with 'precip', 'temp', 'pet' (and 'time') arrays.
        perturbation: Perturbation strategy for the parameters.
        use_jax: Whether to use the JAX backend.
        timestep_hours: Model timestep in hours.
        warmup_days: Number of warm-up days.
    """

    perturbable_forcing = {'precip': 'multiplicative', 'temp': 'additive'}
    _step_forcing = ('precip', 'temp', 'pet')

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
        from jhbv.model import HAS_JAX
        super().__init__(n_members, perturbation, use_jax and HAS_JAX)
        self.base_params = base_params
        self.param_bounds = param_bounds
        self.forcing = forcing
        self.timestep_hours = timestep_hours
        self.warmup_days = warmup_days
        self.member_params: List[Dict[str, float]] = []
        self.state_specs = self.hbv_state_specs(timestep_hours)

    @staticmethod
    def hbv_state_specs(timestep_hours: float) -> List[StateVariableSpec]:
        from jhbv.model import get_routing_buffer_length
        routing_len = get_routing_buffer_length(10, timestep_hours)
        return [
            StateVariableSpec('snow', 1, lower_bound=0.0),
            StateVariableSpec('snow_water', 1, lower_bound=0.0),
            StateVariableSpec('sm', 1, lower_bound=0.0),
            StateVariableSpec('suz', 1, lower_bound=0.0),
            StateVariableSpec('slz', 1, lower_bound=0.0),
            StateVariableSpec('routing_buffer', routing_len, lower_bound=0.0),
        ]

    def initialize_members(self, initial_state=None, **kwargs) -> None:
        """Perturb the parameters and set every member to the initial state."""
        import functools

        from jhbv.model import (
            HBVState,
            create_initial_state,
            create_params_from_dict,
            scale_params_for_timestep,
            step_jax,
        )

        self.member_params = [
            self.perturbation.perturb_parameters(self.base_params, self.param_bounds, i)
            for i in range(self.n_members)
        ]
        xp = self._xp
        self._params = _stack_namedtuples([
            create_params_from_dict(
                scale_params_for_timestep(p, self.timestep_hours), use_jax=self.use_jax)
            for p in self.member_params
        ], xp)
        base = initial_state if initial_state is not None else create_initial_state(
            use_jax=self.use_jax, timestep_hours=self.timestep_hours)
        self._state = _stack_namedtuples([base] * self.n_members, xp)
        self._state_cls = HBVState
        self._compile(functools.partial(step_jax, timestep_hours=self.timestep_hours))
        logger.info("Initialized %d HBV ensemble members", self.n_members)


class XinanjiangEnsembleManager(NamedTupleEnsembleManager):
    """Ensemble of stand-alone Xinanjiang members (``jxaj``, daily, JAX).

    The second model behind the state interface. Xinanjiang's step is written
    for JAX only, so JAX is required. Snow-17 coupling is not supported here.

    Args:
        n_members: Number of ensemble members.
        base_params: Base parameter dictionary.
        param_bounds: Parameter bounds {name: (lower, upper)}.
        forcing: Dict with 'precip', 'pet' (and 'time') arrays, mm/day.
        perturbation: Perturbation strategy for the parameters.
        warmup_days: Number of warm-up days.
    """

    perturbable_forcing = {'precip': 'multiplicative'}
    _step_forcing = ('precip', 'pet')

    def __init__(
        self,
        n_members: int,
        base_params: Dict[str, float],
        param_bounds: Dict[str, tuple],
        forcing: Dict[str, Any],
        perturbation: Optional[PerturbationStrategy] = None,
        warmup_days: int = 0,
    ):
        from jxaj.model import HAS_JAX
        if not HAS_JAX:
            raise ImportError("Xinanjiang data assimilation requires JAX (pip install jax)")
        super().__init__(n_members, perturbation, use_jax=True)
        self.base_params = base_params
        self.param_bounds = param_bounds
        self.forcing = {k: v for k, v in forcing.items() if k in ('precip', 'pet', 'time')}
        self.timestep_hours = 24
        self.warmup_days = warmup_days
        self.member_params: List[Dict[str, float]] = []
        self.state_specs = [
            StateVariableSpec('wu', 1, lower_bound=0.0),
            StateVariableSpec('wl', 1, lower_bound=0.0),
            StateVariableSpec('wd', 1, lower_bound=0.0),
            StateVariableSpec('s', 1, lower_bound=0.0),
            StateVariableSpec('fr', 1, lower_bound=1e-6, upper_bound=1.0),
            StateVariableSpec('qi', 1, lower_bound=0.0),
            StateVariableSpec('qg', 1, lower_bound=0.0),
        ]

    def initialize_members(self, **kwargs) -> None:
        from jxaj.model import create_initial_state, step_jax
        from jxaj.parameters import XinanjiangState, params_dict_to_namedtuple

        self.member_params = [
            self.perturbation.perturb_parameters(self.base_params, self.param_bounds, i)
            for i in range(self.n_members)
        ]
        xp = self._xp
        self._params = _stack_namedtuples(
            [params_dict_to_namedtuple(p, use_jax=True) for p in self.member_params], xp)
        self._state = _stack_namedtuples(
            [create_initial_state(use_jax=True)] * self.n_members, xp)
        self._state_cls = XinanjiangState
        self._compile(step_jax)
        logger.info("Initialized %d Xinanjiang ensemble members", self.n_members)
