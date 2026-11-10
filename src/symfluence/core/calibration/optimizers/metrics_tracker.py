# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Evaluation Metrics Tracker

Tracks crash rates and logs optimization progress in a consistent format.

Progress-line schema (fixed — every field always present)::

    [P##] {ALG} {i}/{max} {unit} ({pct}%) | Best: {score} | \
Improved: {a}/{b} | Crashes: {c}/{d} | Elapsed: {t}

where ``unit`` is one of :data:`EvaluationMetricsTracker.VALID_UNITS`
(``evals``, ``gens``, ``epochs``, ``loops``) and the leading ``[P##]``
worker/context tag is rendered only when one is provided.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from symfluence.core.constants import ModelDefaults
from symfluence.core.metrics.penalty import is_penalty


class EvaluationMetricsTracker:
    """Tracks evaluation crash rates and logs iteration progress.

    Separated from BaseModelOptimizer to isolate metrics bookkeeping
    from algorithm execution logic.

    Args:
        max_iterations: Total iterations for progress reporting (default
            denominator; individual calls may override via ``total``)
        logger: Logger instance
        elapsed_time_fn: Callable returning formatted elapsed time string
        log_interval: Emit a progress line every ``log_interval`` iterations
            (the final iteration is always emitted)
        worker_tag: Optional worker/context tag (e.g. ``'P03'``) rendered as
            a ``[P03]`` prefix on every progress line
    """

    PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    #: Allowed values for the ``unit`` field of the progress line.
    VALID_UNITS = ('evals', 'gens', 'epochs', 'loops')

    def __init__(
        self,
        max_iterations: int,
        logger: logging.Logger,
        elapsed_time_fn: Callable[[], str],
        log_interval: int = 10,
        worker_tag: Optional[str] = None,
    ):
        self.max_iterations = max_iterations
        self.logger = logger
        self._elapsed_time_fn = elapsed_time_fn
        self.log_interval = max(1, int(log_interval))
        self.worker_tag = worker_tag

        # Crash-rate counters
        self._total_evaluations: int = 0
        self._crash_count: int = 0
        self._last_crash_warning: int = 0

    # ------------------------------------------------------------------
    # Crash tracking
    # ------------------------------------------------------------------

    def track_evaluation(self, score: float) -> None:
        """Record an evaluation result, incrementing crash counter if penalty.

        Args:
            score: Fitness score from evaluation
        """
        self._total_evaluations += 1
        if is_penalty(score, self.PENALTY_SCORE):
            self._crash_count += 1

        # Warn every 50 evaluations when crash rate exceeds 10%
        if (self._total_evaluations % 50 == 0
                and self._total_evaluations > self._last_crash_warning):
            crash_rate = self._crash_count / self._total_evaluations
            if crash_rate > 0.10:
                self.logger.warning(
                    f"High crash rate: {self._crash_count}/{self._total_evaluations} "
                    f"({crash_rate:.1%}) evaluations returned penalty score"
                )
                self._last_crash_warning = self._total_evaluations

    def get_crash_stats(self) -> Dict[str, Any]:
        """Return crash rate statistics.

        Returns:
            Dictionary with 'crash_count', 'total_evaluations', 'crash_rate'.
        """
        rate = (self._crash_count / self._total_evaluations
                if self._total_evaluations > 0 else 0.0)
        return {
            'crash_count': self._crash_count,
            'total_evaluations': self._total_evaluations,
            'crash_rate': rate,
        }

    # ------------------------------------------------------------------
    # Progress logging
    # ------------------------------------------------------------------

    def log_iteration_progress(
        self,
        algorithm_name: str,
        iteration: int,
        best_score: float,
        secondary_score: Optional[float] = None,
        secondary_label: Optional[str] = None,
        n_improved: Optional[int] = None,
        population_size: Optional[int] = None,
        crash_stats: Optional[Dict[str, Any]] = None,
        unit: str = 'evals',
        total: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """Log optimization progress in a fixed, comparable schema.

        Format (all fields always present)::

            [P##] {ALG} {i}/{max} {unit} ({pct}%) | Best: {score} | \
Improved: {a}/{b} | Crashes: {c}/{d} | Elapsed: {t}

        Emission is throttled to every ``log_interval`` iterations; the
        final iteration (``iteration >= total``) is always emitted, and
        ``force=True`` bypasses the throttle.

        Args:
            algorithm_name: Algorithm display name (e.g. 'DDS', 'CMA-ES')
            iteration: Current progress count, in ``unit`` units
            best_score: Best fitness score so far
            secondary_score: Optional extra objective value (multi-objective)
            secondary_label: Label for ``secondary_score``
            n_improved: Numerator of the Improved field (algorithm-defined,
                e.g. individuals that improved this generation); '-' if None
            population_size: Denominator of the Improved field; '-' if None
            crash_stats: Crash statistics dict; defaults to this tracker's
                own counters when omitted
            unit: One of :data:`VALID_UNITS` ('evals', 'gens', 'epochs',
                'loops') — what ``iteration``/``total`` count
            total: Denominator for progress; defaults to ``max_iterations``
            force: Emit even when the throttle would suppress this iteration
        """
        total_ref = total if total is not None else self.max_iterations
        if not force and iteration % self.log_interval != 0 and iteration < total_ref:
            return

        if unit not in self.VALID_UNITS:
            unit = 'evals'

        progress_pct = (iteration / total_ref) * 100 if total_ref else 0.0
        elapsed = self._elapsed_time_fn()

        msg_parts = [
            f"{algorithm_name} {iteration}/{total_ref} {unit} ({progress_pct:.0f}%)",
            f"Best: {best_score:.4f}",
        ]

        if secondary_score is not None:
            label = secondary_label or "Secondary"
            msg_parts.append(f"{label}: {secondary_score:.4f}")

        improved_num = '-' if n_improved is None else str(int(n_improved))
        improved_den = '-' if population_size is None else str(int(population_size))
        msg_parts.append(f"Improved: {improved_num}/{improved_den}")

        stats = crash_stats if crash_stats is not None else self.get_crash_stats()
        msg_parts.append(
            f"Crashes: {stats.get('crash_count', 0)}/{stats.get('total_evaluations', 0)}"
        )

        msg_parts.append(f"Elapsed: {elapsed}")

        message = " | ".join(msg_parts)
        if self.worker_tag:
            message = f"[{self.worker_tag}] {message}"

        self.logger.info(message)

    def log_initial_population(
        self,
        algorithm_name: str,
        population_size: int,
        best_score: float
    ) -> None:
        """Log initial population evaluation completion.

        Args:
            algorithm_name: Algorithm name
            population_size: Population size
            best_score: Best score from initial evaluation
        """
        message = (
            f"{algorithm_name} initial population ({population_size} individuals) "
            f"complete | Best score: {best_score:.4f}"
        )
        if self.worker_tag:
            message = f"[{self.worker_tag}] {message}"
        self.logger.info(message)
