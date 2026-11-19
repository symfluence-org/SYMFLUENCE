# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration checkpoint/resume.

A calibration that dies part-way (node pre-emption, wall-clock limit, a
crashed host) used to restart from scratch. This module lets it continue.

Design: deterministic replay
----------------------------
Every built-in algorithm is a pure function of (a) its per-run numpy
Generator (:meth:`OptimizationAlgorithm._new_rng`) and (b) the objective
values the evaluation callbacks return to it. So the checkpoint stores

* every evaluation request and its result, in order (the *evaluation log*);
* the RNG seed (the configured ``RANDOM_SEED``, or the entropy drawn for an
  unseeded run) and the Generator state at the checkpoint;
* the iteration counter, evaluation count, best solution, crash counters and
  the iteration history written so far;
* a fingerprint of the parameter space and objective.

On resume the algorithm is started again from the beginning, but the wrapped
evaluation callbacks answer the logged requests from the log instead of
running the model. Replaying a 2,000-evaluation log takes a fraction of a
second, after which the algorithm holds *exactly* the state it had at the
checkpoint -- DDS's current best and position in its perturbation schedule,
DE/SCE-UA/PSO/NSGA-II populations and fitnesses, DREAM chains, GLUE samples,
simulated-annealing temperature, CMA-ES covariance, L-BFGS curvature pairs,
and the RNG -- and from there it continues with real model runs. A resumed
seeded run therefore ends with the same result as an uninterrupted one.

This avoids a hand-written ``get_state``/``set_state`` pair per algorithm
(whose main loops keep that state in local variables), and it cannot drift
out of sync when an algorithm changes. Replay is verified rather than
trusted: every replayed request must match the logged parameters, and the
Generator state at the checkpoint must match the saved one; any mismatch
raises :class:`CalibrationResumeError` instead of continuing on a different
trajectory.

An algorithm class that cannot be replayed sets ``supports_resume = False``;
resuming it raises :class:`CalibrationResumeError`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from symfluence.core.exceptions import OptimizationError

#: Version of the on-disk resume-state layout.
RESUME_STATE_VERSION = 1

# Replayed parameter vectors must equal the logged ones. They are bitwise
# identical on the same build; the tolerance only absorbs last-bit differences
# from a different BLAS/numpy build, far below anything that changes a model run.
_REPLAY_RTOL = 1e-9
_REPLAY_ATOL = 1e-12


class CalibrationResumeError(OptimizationError):
    """A calibration cannot be resumed from its checkpoint."""


def resume_state_path(results_dir: Path, experiment_id: str, slug: str) -> Path:
    """Path of the resume-state file for one experiment and algorithm."""
    return Path(results_dir) / f"{experiment_id}_{slug}_resume_state.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    return repr(obj)


def to_jsonable(obj: Any) -> Any:
    """Round-trip ``obj`` through JSON so it holds only plain Python types."""
    return json.loads(json.dumps(obj, default=_json_default))


def make_fingerprint(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical fingerprint of what a checkpoint is only valid for."""
    canonical = to_jsonable(fields)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode('utf-8')
    ).hexdigest()
    return {'sha256': digest, 'fields': canonical}


def check_fingerprint(saved: Dict[str, Any], current: Dict[str, Any], path: Path) -> None:
    """Raise :class:`CalibrationResumeError` if the fingerprints differ."""
    if saved.get('sha256') == current['sha256']:
        return
    saved_fields = saved.get('fields', {}) or {}
    current_fields = current['fields']
    changed = sorted(
        key for key in set(saved_fields) | set(current_fields)
        if saved_fields.get(key) != current_fields.get(key)
    )
    details = "; ".join(
        f"{key}: checkpoint={saved_fields.get(key)!r}, config={current_fields.get(key)!r}"
        for key in changed
    ) or "fingerprint hash differs"
    raise CalibrationResumeError(
        f"Cannot resume calibration from {path}: the configuration changed since the "
        f"checkpoint was written ({details}). Restore the original settings to resume, "
        "or set CALIBRATION_RESUME: false (drop --resume) to start a new run; the "
        "previous results are then kept aside as *.prev-<timestamp> files."
    )


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to ``path`` atomically (temporary file in the same dir + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_resume_state(path: Path) -> Dict[str, Any]:
    """Load and sanity-check a resume-state file."""
    path = Path(path)
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            state = json.load(handle)
    except (OSError, ValueError) as exc:
        raise CalibrationResumeError(f"Cannot read calibration resume state {path}: {exc}") from exc
    version = state.get('version')
    if version != RESUME_STATE_VERSION:
        raise CalibrationResumeError(
            f"Calibration resume state {path} has layout version {version!r}; this "
            f"SYMFLUENCE reads version {RESUME_STATE_VERSION}. Start a new run "
            "(drop --resume / CALIBRATION_RESUME)."
        )
    return state


class EvaluationLog:
    """Ordered log of evaluation requests that can replay them.

    Wraps the evaluation callbacks handed to ``algorithm.optimize``. While
    entries from a loaded log remain, a request is answered from the log after
    checking it matches; afterwards it runs the real callback and appends.
    """

    def __init__(self, entries: Optional[List[Dict[str, Any]]] = None):
        self.entries: List[Dict[str, Any]] = list(entries or [])
        self.n_replay = len(self.entries)
        self.cursor = 0

    @property
    def replaying(self) -> bool:
        """True while logged requests remain to be answered from the log."""
        return self.cursor < self.n_replay

    @property
    def n_model_evaluations(self) -> int:
        """Number of parameter sets evaluated so far (population rows count)."""
        return sum(
            len(entry['x']) if entry['kind'] in ('population', 'objectives') else 1
            for entry in self.entries[:self.cursor]
        )

    def _replay(self, kind: str, x: np.ndarray, extra: Any = None) -> Any:
        entry = self.entries[self.cursor]
        logged_x = np.asarray(entry['x'], dtype=float)
        if (
            entry['kind'] != kind
            or logged_x.shape != x.shape
            or entry.get('extra') != extra
            or not np.allclose(logged_x, x, rtol=_REPLAY_RTOL, atol=_REPLAY_ATOL, equal_nan=True)
        ):
            raise CalibrationResumeError(
                f"Calibration resume diverged at logged evaluation request "
                f"{self.cursor + 1} of {self.n_replay}: the algorithm asked for a "
                f"different {kind} evaluation than the checkpoint recorded "
                f"(logged {entry['kind']} with shape {logged_x.shape}, now {kind} with "
                f"shape {x.shape}). The algorithm settings (e.g. NUMBER_OF_ITERATIONS, "
                "algorithm-specific options, NUM_PROCESSES for batch sizes, INITIAL_GUESS) "
                "or the code changed since the checkpoint. Restore them, or start a new "
                "run without --resume."
            )
        self.cursor += 1
        return entry['result']

    def _append(self, kind: str, x: np.ndarray, result: Any, extra: Any = None) -> None:
        entry = {'kind': kind, 'x': x.tolist(), 'result': result}
        if extra is not None:
            entry['extra'] = extra
        self.entries.append(entry)
        self.cursor = len(self.entries)

    def wrap_solution(self, fn: Callable) -> Callable:
        def evaluate_solution(x, *args, **kwargs):
            arr = np.array(x, dtype=float)
            if self.replaying:
                return float(self._replay('solution', arr))
            result = float(fn(x, *args, **kwargs))
            self._append('solution', arr, result)
            return result
        return evaluate_solution

    def wrap_population(self, fn: Callable) -> Callable:
        def evaluate_population(population, *args, **kwargs):
            arr = np.array(population, dtype=float)
            if self.replaying:
                return np.array(self._replay('population', arr), dtype=float)
            result = np.asarray(fn(population, *args, **kwargs), dtype=float)
            self._append('population', arr, result.tolist())
            return result.copy()
        return evaluate_population

    def wrap_objectives(self, fn: Callable) -> Callable:
        def evaluate_population_objectives(population, objective_names, *args, **kwargs):
            arr = np.array(population, dtype=float)
            names = list(objective_names) if objective_names is not None else None
            if self.replaying:
                return np.array(self._replay('objectives', arr, names), dtype=float)
            result = np.asarray(fn(population, objective_names, *args, **kwargs), dtype=float)
            self._append('objectives', arr, result.tolist(), names)
            return result.copy()
        return evaluate_population_objectives

    def wrap_gradient(self, fn: Callable) -> Callable:
        def compute_gradient(x, *args, **kwargs):
            arr = np.array(x, dtype=float)
            if self.replaying:
                loss, grad = self._replay('gradient', arr)
                return float(loss), np.array(grad, dtype=float)
            loss, grad = fn(x, *args, **kwargs)
            grad_arr = np.asarray(grad, dtype=float)
            self._append('gradient', arr, [float(loss), grad_arr.tolist()])
            return float(loss), grad_arr.copy()
        return compute_gradient


def _rng_state(algorithm: Any) -> Optional[Dict[str, Any]]:
    rng = algorithm.__dict__.get('_rng_instance')
    if rng is None:
        return None
    return to_jsonable(rng.bit_generator.state)


class CalibrationCheckpointer:
    """Writes resume state at each checkpoint and drives replay on resume.

    Args:
        path: Resume-state file.
        fingerprint: Output of :func:`make_fingerprint` for this run.
        algorithm: The algorithm instance about to run.
        logger: Logger.
        snapshot_extra: Callable returning the optimizer-side state to store
            (history, best, crash counters); called at each checkpoint.
        resumed_state: A loaded resume state to replay, or None for a new run.
    """

    def __init__(
        self,
        path: Path,
        fingerprint: Dict[str, Any],
        algorithm: Any,
        logger: logging.Logger,
        snapshot_extra: Callable[[], Dict[str, Any]],
        resumed_state: Optional[Dict[str, Any]] = None,
    ):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.algorithm = algorithm
        self.logger = logger
        self._snapshot_extra = snapshot_extra
        self.resumed_state = resumed_state
        self._records_to_skip = 0
        self._records_seen = 0
        self._verified = resumed_state is None

        if resumed_state is not None:
            check_fingerprint(resumed_state.get('fingerprint', {}), fingerprint, self.path)
            self.log = EvaluationLog(resumed_state.get('evaluations', []))
            self._records_to_skip = len(resumed_state.get('history', []))
            # An unseeded run is replayed with the entropy it drew originally.
            entropy = (resumed_state.get('rng') or {}).get('entropy')
            if entropy is not None:
                algorithm._resume_entropy = int(entropy)
        else:
            self.log = EvaluationLog()

    @property
    def replaying(self) -> bool:
        return self.log.replaying

    def wrap(self, callbacks: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Route the evaluation callbacks through the evaluation log (in place)."""
        callbacks['evaluate_solution'] = self.log.wrap_solution(callbacks['evaluate_solution'])
        callbacks['evaluate_population'] = self.log.wrap_population(callbacks['evaluate_population'])
        if kwargs.get('evaluate_population_objectives') is not None:
            kwargs['evaluate_population_objectives'] = self.log.wrap_objectives(
                kwargs['evaluate_population_objectives'])
        if kwargs.get('compute_gradient') is not None:
            kwargs['compute_gradient'] = self.log.wrap_gradient(kwargs['compute_gradient'])

    def skip_record(self) -> bool:
        """Whether this ``record_iteration`` call re-records a restored history row."""
        self._records_seen += 1
        return self._records_seen <= self._records_to_skip

    def on_checkpoint(self, iteration: int) -> None:
        """Called at each regular checkpoint: write state, or verify replay."""
        if self.replaying:
            return
        if not self._verified:
            self._verify_replay(final=False)
        self.save(iteration, complete=False)

    def on_complete(self, iteration: Optional[int]) -> None:
        """Called after the algorithm returned."""
        if self.replaying:
            self.logger.warning(
                f"Resumed calibration finished after replaying only {self.log.cursor} of "
                f"{self.log.n_replay} logged evaluation requests (fewer iterations than "
                "the checkpointed run?); the final result reflects the shorter run."
            )
            return
        if not self._verified:
            self._verify_replay(final=True)
        self.save(iteration, complete=True)

    def _verify_replay(self, final: bool) -> None:
        """Check the Generator reached the checkpointed state after replay."""
        self._verified = True
        state = self.resumed_state or {}
        saved = (state.get('rng') or {}).get('state')
        at_same_point = (
            state.get('n_requests') == self.log.n_replay
            and state.get('n_history') == self._records_seen
        )
        if saved is None or not at_same_point or (final and not state.get('complete')):
            return
        current = _rng_state(self.algorithm)
        if current is not None and current != saved:
            raise CalibrationResumeError(
                f"Calibration resume from {self.path} diverged: after replaying "
                f"{self.log.n_replay} logged evaluation requests the random-number "
                "generator is not in the checkpointed state, so the run would continue "
                "on a different trajectory. Was RANDOM_SEED or the SYMFLUENCE/numpy "
                "version changed? Start a new run without --resume."
            )
        self.logger.info("Calibration resume: replay verified against the checkpoint")

    def save(self, iteration: Optional[int], complete: bool) -> None:
        """Write the resume state atomically."""
        extra = self._snapshot_extra()
        history = extra.pop('history', [])
        payload = {
            'version': RESUME_STATE_VERSION,
            'written': datetime.now().isoformat(),
            'complete': bool(complete),
            'algorithm': self.algorithm.name,
            'fingerprint': self.fingerprint,
            'iteration': iteration,
            'n_requests': len(self.log.entries),
            'n_evaluations': self.log.n_model_evaluations,
            'n_history': len(history),
            'rng': {
                'entropy': getattr(self.algorithm, '_run_entropy', None),
                'state': _rng_state(self.algorithm),
            },
            **extra,
            'history': history,
            'evaluations': self.log.entries,
        }
        write_json_atomic(self.path, payload)
        self.logger.debug(
            f"Resume state written to {self.path} ({payload['n_evaluations']} evaluations)"
        )
