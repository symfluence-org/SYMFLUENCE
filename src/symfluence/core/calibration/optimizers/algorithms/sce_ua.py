#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
SCE-UA (Shuffled Complex Evolution - University of Arizona) Algorithm

A global optimization algorithm that combines the strengths of the simplex
procedure with competitive evolution. Widely used in hydrological modeling.

Reference:
    Duan, Q., Sorooshian, S., and Gupta, V.K. (1992). Effective and Efficient
    Global Optimization for Conceptual Rainfall-Runoff Models.
    Water Resources Research, 28(4), 1015-1031.

Parallel Execution Design
---------------------------
The original serial SCE-UA processes each complex sequentially, issuing one
``evaluate_solution`` call at a time.  This implementation replaces that serial
pattern with batched evaluation via ``evaluate_population``, which dispatches
all candidates in a batch to the underlying parallel executor (ProcessPool or
MPI) in a single call.

**Why the parallelisation is algorithmically valid**

The CCE procedure evolves each complex independently.  After the population is
partitioned into complexes, no information flows *between* complexes until the
shuffle step that reunites them at the end of the iteration.  This inter-complex
independence is an explicit, load-bearing property of the SCE algorithm: it is
what gives the S (Shuffled) in the name its meaning.  Exploiting it for
parallelism therefore does not alter the algorithm's search behaviour -- it only
changes the order in which the model is called.

**What is preserved**

The *intra*-complex dependency chain is fully respected.  Within a single
complex, step k+1 uses the complex state that was updated by step k -- this
serial chain cannot be parallelised without changing the algorithm, so it is
not.

**Batching strategy per evolution step**

At each CCE evolution step every complex independently selects a random
simplex, generates a reflection point, and may fall back to a contraction or a
random replacement.  Since the candidates across complexes are independent, we
batch-evaluate them in up to three rounds per step:

  Round 1  -- Reflection batch (always n_complexes candidates)
             Complexes whose reflection improves the worst simplex point accept
             it immediately.  The rest proceed to round 2.

  Round 2  -- Contraction batch (only complexes that failed reflection)
             A contraction point is formed by pulling the worst point halfway
             towards the centroid.  Complexes that improve accept it.  The rest
             proceed to round 3.

  Round 3  -- Random replacement batch (only complexes that failed both)
             A uniformly random point replaces the worst point unconditionally
             (matching the original MATLAB cceua fallback behaviour).

**Worst-case model calls per shuffle iteration**

  Serial original : n_complexes x n_evolution_steps x 3  (all fallback paths)
  Parallel new    : n_evolution_steps x 3 batches          (same calls, fewer wall-clock rounds)

With default settings (5 complexes, 33 evolution steps) this reduces the
number of sequential evaluation rounds from up to 495 to at most 99, giving an
ideal 5x wall-clock speedup when using 5 or more parallel workers.

**Evolution tracking CSVs**

Detailed candidate tracking:
    sce_<run_id>_evolution_tracking.csv

Every evaluated candidate is written to this per-run CSV in the optimization
results directory.  Each row records:

    timestamp, run_id, metric_name, shuffle_iteration, evolution_id,
    complex_id, stage, accepted, score, <param_1>, ..., <param_N>

Key columns:

  stage     -- which CCE sub-step produced this candidate:
               'reflection' | 'contraction' | 'random'

  accepted  -- True  : candidate beat the worst simplex member and was written
                       into the complex.
               False : candidate did NOT improve; the complex is UNCHANGED for
                       this step (the next stage is tried instead).

The 'accepted' flag is essential for interpreting the CSV.  Because all
evaluated candidates are logged (not just accepted ones), a reflection row with
accepted=False simply means that attempt failed and contraction was tried next
-- the complex did not regress.

Per-evolution best-score tracking:
    sce_<run_id>_evolution_best_tracking.csv

This companion CSV writes one row per evolution step with:

        timestamp, run_id, metric_name, shuffle_iteration, evolution_id,
        evolution_best_score, best_score_so_far

This file is designed for quick monotonic-progress inspection without filtering
by stage or accepted flag.
"""
from __future__ import annotations

import csv
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_algorithm import OptimizationAlgorithm


class SCEUAAlgorithm(OptimizationAlgorithm):
    """Shuffled Complex Evolution algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "SCE-UA"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_reflection_candidates(
        self,
        sub_complexes: List[np.ndarray],
        sub_fitnesses: List[np.ndarray],
        complex_sizes: List[int],
        n_params: int,
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        """Generate one reflection candidate per complex and collect simplex state.

        For each complex a random simplex of size n_params+1 is chosen from the
        complex members.  The worst point in the simplex is identified, and the
        centroid of the remaining n_params points is used to compute the classic
        Nelder-Mead reflection:

            reflection = 2 * centroid - worst_point

        The result is clipped to [0, 1] to stay inside the normalised parameter
        space.

        Simplex size is n_params+1 -- the minimum number of vertices needed to
        span an n-dimensional simplex.  The complex holds 2*n_params+1 members,
        so the simplex is always a strict sub-set of the complex.

        Args:
            sub_complexes: List of parameter arrays, one per complex
                           (shape: [n_members_in_complex, n_params]).
            sub_fitnesses: Corresponding fitness vectors.
            complex_sizes: Number of members in each complex.
            n_params: Number of optimisation parameters.

        Returns:
            reflection_batch:
                Array of shape (n_complexes, n_params) ready for batch
                evaluation.
            per_step_state:
                List of (worst_idx_in_sub_complex, centroid) tuples, one per
                complex.  Reused in the contraction stage without recomputation.
        """
        n_complexes = len(sub_complexes)
        # Simplex size per SCE-UA paper: n_params + 1 vertices.
        simplex_size = n_params + 1
        reflection_batch = np.empty((n_complexes, n_params))
        per_step_state: List[Tuple[int, np.ndarray]] = []

        for c in range(n_complexes):
            sub_complex = sub_complexes[c]
            sub_fitness = sub_fitnesses[c]
            n_members = complex_sizes[c]

            # Draw n_params+1 unique indices from the complex (without replacement).
            simplex_idx = self._rng.choice(n_members, simplex_size, replace=False)

            # worst_in_simplex is an index into sub_complex (range 0..n_members-1)
            # pointing to the simplex member with the lowest fitness score.  This is
            # the point that will be replaced if a better candidate is found.
            worst_in_simplex = simplex_idx[np.argmin(sub_fitness[simplex_idx])]
            others = [i for i in simplex_idx if i != worst_in_simplex]
            centroid = np.mean(sub_complex[others], axis=0)

            # Reflection: reflect the worst point through the centroid of the others.
            reflection = 2.0 * centroid - sub_complex[worst_in_simplex]
            reflection = np.clip(reflection, 0.0, 1.0)

            reflection_batch[c] = reflection
            # Store worst index and centroid for reuse in the contraction step.
            per_step_state.append((worst_in_simplex, centroid))

        return reflection_batch, per_step_state

    def _apply_cce_step(
        self,
        sub_complexes: List[np.ndarray],
        sub_fitnesses: List[np.ndarray],
        per_step_state: List[Tuple[int, np.ndarray]],
        reflection_batch: np.ndarray,
        reflection_fitness: np.ndarray,
        n_params: int,
        iteration: int,
        evolution_id: int,
        evaluate_population: Callable,
        log_evolution_batch: Optional[
            Callable[[List[int], np.ndarray, np.ndarray, str, List[bool]], None]
        ] = None,
        log_complex_batch: Optional[
            Callable[[List[int], List[float], List[float], List[float], str, List[np.ndarray]], None]
        ] = None,
    ) -> None:
        """Apply one CCE step to all complexes using batched fallback evaluation.

        Accepts reflection candidates where they improve the worst simplex member.
        For complexes where reflection fails, batches contraction candidates and
        evaluates them in parallel.  For complexes where contraction also fails,
        batches random replacements and evaluates those too.

        All updates are performed in-place on sub_complexes and sub_fitnesses.

        Acceptance threshold
        --------------------
        Reflection and contraction use the comparison:

            new_score > sub_fitnesses[c][worst_idx]

        where worst_idx is the index of the worst member in the SIMPLEX (not the
        worst member of the entire complex).  This is the point being replaced.

        Random fallback differs: when both reflection and contraction fail, the
        random point replaces the worst simplex member unconditionally.

        Logging order
        -------------
        log_evolution_batch is called AFTER the accept/reject loop for each
        stage.  This ensures the 'accepted' flag in the CSV accurately reflects
        whether the candidate was written into the complex.  If logging were done
        before acceptance, rejected reflection candidates would appear in the CSV
        with low scores under stage='reflection', making it misleadingly look as
        though the complex regressed.

        Args:
            sub_complexes: List of parameter arrays per complex (modified in-place).
            sub_fitnesses: Corresponding fitness vectors (modified in-place).
            per_step_state: (worst_idx, centroid) tuples from
                            _generate_reflection_candidates.
            reflection_batch: Candidate array (n_complexes, n_params).
            reflection_fitness: Fitness scores for each reflection candidate.
            n_params: Number of optimisation parameters.
            iteration: Current shuffle iteration (forwarded to evaluate_population).
            evaluate_population: Batch evaluation callback.
            log_evolution_batch: Optional callback with signature
                (complex_ids, batch, fitness, stage, accepted).
                accepted[i] is True when the candidate was written back into the
                complex. For random fallback this is always True.
            log_complex_batch: Optional callback with signature
                (complex_ids, best_scores, worst_scores, accepted_scores, stage,
                 sub_fitnesses). Called when a candidate is accepted.
        """
        n_complexes = len(sub_complexes)
        self.logger.debug(
            "DEBUG: SCE: Evolution %d, Trying %d number of simplexes with reflection",
            evolution_id,
            n_complexes,
        )

        # ---- Stage 1: Accept reflections; collect accepted flags ----
        # Run all accept/reject decisions first.  The results are stored in
        # reflection_accepted so that the log callback receives accurate status.
        contraction_complex_ids: List[int] = []
        contraction_candidates: List[np.ndarray] = []
        reflection_accepted: List[bool] = []

        for c in range(n_complexes):
            worst_idx, centroid = per_step_state[c]
            best_score = float(np.max(sub_fitnesses[c]))
            worst_score = float(sub_fitnesses[c][worst_idx])

            if reflection_fitness[c] > sub_fitnesses[c][worst_idx]:
                # Reflection improved the worst simplex point -- accept it.
                sub_complexes[c][worst_idx] = reflection_batch[c]
                sub_fitnesses[c][worst_idx] = reflection_fitness[c]
                reflection_accepted.append(True)

                # Log this acceptance
                if log_complex_batch:
                    log_complex_batch(
                        [c],
                        [best_score],
                        [worst_score],
                        [float(reflection_fitness[c])],
                        'reflection',
                        sub_fitnesses,
                    )
            else:
                # Reflection failed -- compute contraction: move worst point
                # halfway towards the simplex centroid.
                contraction = (
                    sub_complexes[c][worst_idx]
                    + 0.5 * (centroid - sub_complexes[c][worst_idx])
                )
                contraction_complex_ids.append(c)
                contraction_candidates.append(contraction)
                reflection_accepted.append(False)

        # Log all reflection candidates now that accepted status is known.
        if log_evolution_batch:
            log_evolution_batch(
                list(range(n_complexes)),
                reflection_batch,
                reflection_fitness,
                'reflection',
                reflection_accepted,
            )

        if not contraction_complex_ids:
            # All complexes accepted their reflections; nothing left to do.
            return

        # ---- Stage 2: Batch-evaluate all contraction candidates ----
        self.logger.debug(
            "DEBUG: SCE: Evolution %d, Trying %d number of simplexes with contraction",
            evolution_id,
            len(contraction_complex_ids),
        )
        contraction_batch = np.array(contraction_candidates)
        contraction_fitness = evaluate_population(contraction_batch, iteration)

        random_complex_ids: List[int] = []
        random_candidates: List[np.ndarray] = []
        contraction_accepted: List[bool] = []

        for batch_i, c in enumerate(contraction_complex_ids):
            worst_idx, _ = per_step_state[c]
            best_score = float(np.max(sub_fitnesses[c]))
            worst_score = float(sub_fitnesses[c][worst_idx])

            if contraction_fitness[batch_i] > sub_fitnesses[c][worst_idx]:
                # Contraction improved the worst simplex point -- accept it.
                sub_complexes[c][worst_idx] = contraction_batch[batch_i]
                sub_fitnesses[c][worst_idx] = contraction_fitness[batch_i]
                contraction_accepted.append(True)

                # Log this acceptance
                if log_complex_batch:
                    log_complex_batch(
                        [c],
                        [best_score],
                        [worst_score],
                        [float(contraction_fitness[batch_i])],
                        'contraction',
                        sub_fitnesses,
                    )
            else:
                # Contraction also failed -- fall back to a uniformly random
                # point.  This is the original SCE-UA final fallback.
                random_point = self._rng.uniform(0.0, 1.0, n_params)
                random_complex_ids.append(c)
                random_candidates.append(random_point)
                contraction_accepted.append(False)

        if log_evolution_batch:
            log_evolution_batch(
                contraction_complex_ids,
                contraction_batch,
                contraction_fitness,
                'contraction',
                contraction_accepted,
            )

        if not random_complex_ids:
            # All remaining complexes accepted their contractions.
            return

        # ---- Stage 3: Batch-evaluate all random replacement candidates ----
        self.logger.debug(
            "DEBUG: SCE: Evolution %d, Trying %d number of simplexes with random",
            evolution_id,
            len(random_complex_ids),
        )
        random_batch = np.array(random_candidates)
        random_fitness = evaluate_population(random_batch, iteration)

        random_accepted: List[bool] = []

        for batch_i, c in enumerate(random_complex_ids):
            worst_idx, _ = per_step_state[c]
            best_score = float(np.max(sub_fitnesses[c]))
            worst_score = float(sub_fitnesses[c][worst_idx])

            # MATLAB cceua fallback behavior: random replacement is unconditional
            # once reflection and contraction have both failed.
            sub_complexes[c][worst_idx] = random_batch[batch_i]
            sub_fitnesses[c][worst_idx] = random_fitness[batch_i]
            random_accepted.append(True)

            # Log this replacement
            if log_complex_batch:
                log_complex_batch(
                    [c],
                    [best_score],
                    [worst_score],
                    [float(random_fitness[batch_i])],
                    'random',
                    sub_fitnesses,
                )

        if log_evolution_batch:
            log_evolution_batch(
                random_complex_ids,
                random_batch,
                random_fitness,
                'random',
                random_accepted,
            )

    # ------------------------------------------------------------------
    # Main optimisation entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        evaluate_population_objectives: Optional[Callable] = None,
        compute_gradient: Optional[Callable] = None,
        gradient_mode: str = 'auto',
        log_initial_population: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the SCE-UA optimisation algorithm with parallel CCE evaluation.

        The algorithm proceeds in three phases:

        1. Initialisation -- A random population of pop_size points is drawn
           and evaluated in one parallel batch.  Points are ranked by fitness
           (descending), establishing the initial complex partition.

        2. Shuffle iterations -- Each iteration:

           a. Partition the ranked population into n_complexes complexes
              using the standard interleaved assignment
              (complex c owns ranks c, c+n_complexes, c+2*n_complexes, ...).

           b. Run n_evolution_steps CCE steps.  At each step:
              - Generate one reflection candidate per complex (CPU-only).
              - Batch-evaluate all reflections in parallel.
              - Accept successful reflections; batch-evaluate contractions for
                failures; accept successful contractions; batch-evaluate random
                replacements for remaining failures.
              All three rounds involve only the complexes that reached that
              stage, minimising total model calls.

           c. Merge the evolved complexes back and re-sort by fitness (shuffle).

        3. Early stopping -- If the best fitness changes by less than
           percent_change_threshold for evolution_stagnation consecutive
           shuffle iterations, optimisation terminates.

        Args:
            n_params: Number of parameters to optimise.
            evaluate_solution: Single-solution callback (not used internally;
                               present for interface compatibility).
            evaluate_population: Batch evaluation callback.
                                  Signature: (population, iteration) -> scores.
            denormalize_params: Convert normalised params to a human-readable dict.
            record_iteration: Persist result of each iteration.
            update_best: Update tracked global best.
            log_progress: Emit a progress log line.
            evaluate_population_objectives: Unused (single-objective algorithm).
            compute_gradient: Unused (gradient-free algorithm).
            gradient_mode: Unused.
            log_initial_population: Optional callback after initial population.
            **kwargs: Accepts 'results_dir' (Path) and 'experiment_id' (str)
                      injected by BaseModelOptimizer for CSV tracking.
            evaluate_solution: Single-solution callback (not used internally;
                               present for interface compatibility).
            evaluate_population: Batch evaluation callback.
                                  Signature: (population, iteration) -> scores.
            denormalize_params: Convert normalised params to a human-readable dict.

        Returns:
            dict with keys:
                best_solution  -- Best normalised parameter vector found.
                best_score     -- Best fitness score.
                best_params    -- Best parameters as a denormalised dict.
        """
        # Per-run RNG (OptimizationAlgorithm._new_rng): never the process-global stream.
        self._rng = self._new_rng()
        self.logger.info(f"Starting SCE-UA optimization with {n_params} parameters")

        # ---- Evolution tracking CSV setup ----
        run_id = kwargs.get('experiment_id')
        if run_id is None:
            run_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='optimization',
                dict_key='EXPERIMENT_ID'
            )

        results_dir_arg = kwargs.get('results_dir')
        tracking_csv_path: Optional[Path] = None
        best_tracking_csv_path: Optional[Path] = None
        if results_dir_arg is not None:
            try:
                results_dir = Path(results_dir_arg)
                results_dir.mkdir(parents=True, exist_ok=True)
                tracking_csv_path = results_dir / f"sce_{run_id}_evolution_tracking.csv"
                best_tracking_csv_path = (
                    results_dir / f"sce_{run_id}_evolution_best_tracking.csv"
                )
            except Exception as e:  # noqa: BLE001 - best-effort path setup
                self.logger.warning(f"Failed to initialize SCE evolution tracking path: {e}")
        else:
            self.logger.warning(
                "SCE evolution tracking disabled: results_dir was not provided to optimizer callbacks"
            )

        # ---- Read SCE-UA parameters from config ----
        n_complexes = self._get_config_value(
            lambda: self.config.optimization.sce_ua.number_of_complexes,
            default=max(2, self.population_size // 10),
            dict_key='NUMBER_OF_COMPLEXES'
        )
        n_evolution_steps = self._get_config_value(
            lambda: self.config.optimization.sce_ua.number_of_evolution_steps,
            default=2 * n_params + 1,
            dict_key='NUMBER_OF_EVOLUTION_STEPS'
        )

        # Each complex holds 2*n_params+1 members so that a simplex of
        # n_params+1 points can always be drawn from it with room to spare,
        # giving the simplex operations enough geometric coverage.
        n_per_complex = 2 * n_params + 1
        pop_size = n_complexes * n_per_complex

        # Early stopping parameters from config
        stagnation_limit = self._get_config_value(
            lambda: self.config.optimization.sce_ua.evolution_stagnation,
            default=5,
            dict_key='EVOLUTION_STAGNATION'
        )
        pct_change_threshold = self._get_config_value(
            lambda: self.config.optimization.sce_ua.percent_change_threshold,
            default=0.01,
            dict_key='PERCENT_CHANGE_THRESHOLD'
        )

        self.logger.info(
            f"SCE-UA structure: {n_complexes} complexes x {n_per_complex} points/complex "
            f"= {pop_size} total, {n_evolution_steps} evolution steps per shuffle"
        )

        # ---- Phase 1: Initialise and evaluate the full population ----
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
        population = self._seed_population(
            self._rng.uniform(0.0, 1.0, (pop_size, n_params)), kwargs)
        fitness = evaluate_population(population, 0)

        # Rank population: index 0 is the best individual (descending fitness)
        sorted_idx = np.argsort(-fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        best_pos = population[0].copy()
        best_fit = fitness[0]

        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, pop_size, best_fit)

        # ---- Initialize CSV header once parameter names are known ----
        # The 'accepted' column is critical for interpreting the CSV correctly.
        # All evaluated candidates are logged (accepted=True and accepted=False)
        # to preserve the full search trajectory.  Without 'accepted', a rejected
        # reflection row with a low score would look like the complex regressed,
        # when in fact the complex was unchanged and contraction was tried next.
        # For stage='random', accepted is always True because random fallback
        # replacement is unconditional.
        param_names = list(params_dict.keys())
        tracking_headers = [
            'timestamp',
            'run_id',
            'metric_name',
            'shuffle_iteration',
            'evolution_id',
            'complex_id',
            'stage',
            'accepted',   # True = candidate improved worst simplex member
            'score',
            *param_names,
        ]
        best_tracking_headers = [
            'timestamp',
            'run_id',
            'metric_name',
            'shuffle_iteration',
            'evolution_id',
            'evolution_best_score',
            'best_score_so_far',
        ]

        if tracking_csv_path is not None:
            with tracking_csv_path.open('w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=tracking_headers)
                writer.writeheader()
            self.logger.info(f"SCE evolution tracking CSV: {tracking_csv_path}")
        if best_tracking_csv_path is not None:
            with best_tracking_csv_path.open('w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=best_tracking_headers)
                writer.writeheader()
            self.logger.info(f"SCE evolution best CSV: {best_tracking_csv_path}")

        # ---- Per-complex CSV setup ----
        per_complex_csv_paths: Dict[int, Optional[Path]] = {}
        per_complex_headers = [
            'timestamp',
            'run_id',
            'metric_name',
            'shuffle_iteration',
            'evolution_id',
            'stage',
            'best_score',
            'worst_score',
            'accepted_score',
            'score_variance',
        ]

        if results_dir_arg is not None:
            for c in range(n_complexes):
                try:
                    complex_csv_path = (
                        Path(results_dir_arg)
                        / f"sce_{run_id}_complex{c:02d}.csv"
                    )
                    with complex_csv_path.open('w', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=per_complex_headers)
                        writer.writeheader()
                    per_complex_csv_paths[c] = complex_csv_path
                    self.logger.info(f"SCE per-complex CSV for complex {c}: {complex_csv_path}")
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(f"Failed to initialize per-complex CSV for complex {c}: {e}")
                    per_complex_csv_paths[c] = None
        else:
            for c in range(n_complexes):
                per_complex_csv_paths[c] = None

        tracking_rows_buffer: List[Dict[str, Any]] = []
        best_tracking_rows_buffer: List[Dict[str, Any]] = []

        metric_name = self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE',
            dict_key='OPTIMIZATION_METRIC'
        )

        best_score_so_far = float(best_fit)

        def _to_csv_scalar(value: Any) -> Any:
            """Coerce numpy scalars/arrays to plain Python types for CSV output."""
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.size == 1:
                    return float(value.ravel()[0])
                return ';'.join(str(float(v)) for v in value.ravel())
            if isinstance(value, (np.floating, np.integer)):
                return float(value)
            return value

        def log_evolution_batch(
            complex_ids: List[int],
            candidate_batch: np.ndarray,
            candidate_fitness: np.ndarray,
            stage: str,
            accepted: List[bool],
            shuffle_iteration: int,
            evolution_step: int,
        ) -> None:
            """Buffer one batch of per-complex evolution diagnostics.

            Called once per stage (reflection/contraction/random) per evolution
            step, AFTER accept/reject decisions have been made.  accepted[i] is
            True when the candidate for complex_ids[i] was written into the complex
            because it was accepted at that stage. For stage='random', this is
            always True.
            """
            if tracking_csv_path is None:
                return

            rows: List[Dict[str, Any]] = []
            for row_idx, complex_id in enumerate(complex_ids):
                denorm = denormalize_params(candidate_batch[row_idx])
                row: Dict[str, Any] = {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'metric_name': metric_name,
                    'shuffle_iteration': shuffle_iteration,
                    'evolution_id': evolution_step,
                    'complex_id': complex_id,
                    'stage': stage,
                    'accepted': accepted[row_idx],
                    'score': float(candidate_fitness[row_idx]),
                }
                for name in param_names:
                    row[name] = _to_csv_scalar(denorm.get(name))
                rows.append(row)

            tracking_rows_buffer.extend(rows)

        def flush_tracking_rows() -> None:
            """Write buffered evolution rows to CSV and clear the buffer.

            Rows are buffered in memory during the CCE steps and flushed once
            per shuffle iteration to reduce per-evaluation file I/O overhead.
            """
            if tracking_csv_path is None or not tracking_rows_buffer:
                return
            with tracking_csv_path.open('a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=tracking_headers)
                writer.writerows(tracking_rows_buffer)
            tracking_rows_buffer.clear()

        def flush_best_tracking_rows() -> None:
            """Write buffered per-evolution best-score rows to CSV and clear buffer."""
            if best_tracking_csv_path is None or not best_tracking_rows_buffer:
                return
            with best_tracking_csv_path.open('a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=best_tracking_headers)
                writer.writerows(best_tracking_rows_buffer)
            best_tracking_rows_buffer.clear()

        def log_complex_batch(
            complex_ids: List[int],
            best_scores: List[float],
            worst_scores: List[float],
            accepted_scores: List[float],
            stage: str,
            sub_fitnesses: List[np.ndarray],
            shuffle_iteration: int,
            evolution_step: int,
        ) -> None:
            """Log per-complex evolution data for a batch of complexes.

            Each complex gets its own CSV file tracking the evolution of its scores,
            best/worst values, and variance across stages (reflection, contraction, random).

            Args:
                complex_ids: IDs of the complexes being logged.
                best_scores: Best fitness score in each complex before evolution.
                worst_scores: Worst fitness score in each complex before evolution.
                accepted_scores: Score of the accepted candidate in each complex.
                stage: Current evolution stage ('reflection', 'contraction', 'random').
                sub_fitnesses: Fitness arrays for all complexes (for variance calculation).
                shuffle_iteration: Current shuffle iteration.
                evolution_step: Current evolution step within the shuffle.
            """
            timestamp = datetime.now().isoformat()

            for idx, complex_id in enumerate(complex_ids):
                csv_path = per_complex_csv_paths.get(complex_id)
                if csv_path is None:
                    continue

                # Calculate variance of all scores in this complex
                if complex_id < len(sub_fitnesses):
                    complex_fitness = sub_fitnesses[complex_id]
                    score_variance = float(np.var(complex_fitness))
                else:
                    score_variance = 0.0

                row: Dict[str, Any] = {
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'metric_name': metric_name,
                    'shuffle_iteration': shuffle_iteration,
                    'evolution_id': evolution_step,
                    'stage': stage,
                    'best_score': float(best_scores[idx]),
                    'worst_score': float(worst_scores[idx]),
                    'accepted_score': float(accepted_scores[idx]),
                    'score_variance': score_variance,
                }

                try:
                    with csv_path.open('a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=per_complex_headers)
                        writer.writerow(row)
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(
                        f"Failed to write per-complex CSV for complex {complex_id}: {e}"
                    )

        # ---- Precompute the static complex partitioning pattern ----
        # Complex c owns population rows c, c+n_complexes, c+2*n_complexes, ...
        # The population is re-sorted after every shuffle iteration so the
        # specific parameter vectors at these positions change each iteration,
        # but the index pattern itself is constant and can be computed once.
        complex_members_list: List[List[int]] = [
            list(range(c, pop_size, n_complexes)) for c in range(n_complexes)
        ]
        complex_sizes: List[int] = [len(m) for m in complex_members_list]

        # ---- Phase 2: Shuffle iterations ----
        stagnation_count = 0
        prev_best_fit = best_fit

        # max_iterations comes from base class (optimization.iterations in config)
        for iteration in range(1, self.max_iterations + 1):
            # Evolution index is per-shuffle and resets each shuffle iteration.
            evolution_id = 0

            self.logger.debug(f"DEBUG: SCE: Starting shuffle iteration {iteration}")

            # -- 2a. Extract independent sub-complexes from the ranked population --
            # Each sub-complex is a working copy; we evolve it without touching the
            # main population array so that complexes remain fully independent.
            sub_complexes: List[np.ndarray] = [
                population[members].copy() for members in complex_members_list
            ]
            sub_fitnesses: List[np.ndarray] = [
                fitness[members].copy() for members in complex_members_list
            ]

            # -- 2b. CCE: n_evolution_steps parallel steps across all complexes --
            # At each step we generate one candidate per complex (reflection) and
            # batch-evaluate all of them simultaneously.  Complexes that fail
            # reflection proceed to contraction, then random replacement -- these
            # fallback stages are also batched across the complexes that need them.
            for _ in range(n_evolution_steps):
                evolution_id += 1

                # Generate reflection candidates for all complexes (CPU-only, cheap)
                reflection_batch, per_step_state = self._generate_reflection_candidates(
                    sub_complexes, sub_fitnesses, complex_sizes, n_params
                )

                # Batch-evaluate all reflections in parallel
                reflection_fitness = evaluate_population(reflection_batch, iteration)

                # Accept reflections / fall back to contraction / random (all batched).
                self._apply_cce_step(
                    sub_complexes, sub_fitnesses,
                    per_step_state,
                    reflection_batch, reflection_fitness,
                    n_params, iteration, evolution_id, evaluate_population,
                    log_evolution_batch=partial(
                        log_evolution_batch,
                        shuffle_iteration=iteration,
                        evolution_step=evolution_id,
                    ),
                    log_complex_batch=partial(
                        log_complex_batch,
                        shuffle_iteration=iteration,
                        evolution_step=evolution_id,
                    ),
                )

            # -- 2c. Merge evolved sub-complexes back into the population array --
            for c, members in enumerate(complex_members_list):
                population[members] = sub_complexes[c]
                fitness[members] = sub_fitnesses[c]

            # -- 2d. Shuffle: re-sort the merged population by fitness --
            # This is the "shuffle" that gives the algorithm its name.  It breaks
            # up any complex that has converged prematurely by redistributing the
            # best individuals across all complexes in the next iteration.
            self.logger.debug(
                "DEBUG: SCE: Iteration %d complete, shuffling parameters",
                iteration,
            )
            sorted_idx = np.argsort(-fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]

            # Update global best from the newly shuffled population
            if fitness[0] > best_fit:
                best_pos = population[0].copy()
                best_fit = fitness[0]

            # ---- Record and log iteration results ----
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict)
            update_best(best_fit, params_dict, iteration)
            # Progress line (tracker throttles emission); Improved counts
            # population members beating the previous shuffle's best.
            n_improved = int(np.sum(fitness > prev_best_fit))
            log_progress(
                self.name, iteration, best_fit,
                n_improved=n_improved, pop_size=pop_size,
                unit='loops'
            )

            best_score_so_far = max(best_score_so_far, float(best_fit))
            best_tracking_rows_buffer.append({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'metric_name': metric_name,
                'shuffle_iteration': iteration,
                'evolution_id': evolution_id,
                'evolution_best_score': float(best_fit),
                'best_score_so_far': best_score_so_far,
            })

            # Persist evolution tracking rows once per shuffle iteration to keep
            # per-batch overhead low while still providing near-real-time output.
            flush_tracking_rows()
            flush_best_tracking_rows()

            # ---- Early stopping: check for stagnation ----
            # Use relative change from the previous best; fall back to absolute
            # change when prev_best_fit is zero to avoid division-by-zero.
            if prev_best_fit != 0:
                pct_change = abs(best_fit - prev_best_fit) / abs(prev_best_fit)
            else:
                pct_change = abs(best_fit - prev_best_fit)

            if pct_change < pct_change_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= stagnation_limit:
                self.logger.info(
                    f"SCE-UA early stopping at iteration {iteration}: "
                    f"no improvement > {pct_change_threshold:.4f} for "
                    f"{stagnation_limit} consecutive shuffles"
                )
                break

            prev_best_fit = best_fit

        # Flush any remaining buffered rows (e.g., if loop exits via early stopping).
        flush_tracking_rows()
        flush_best_tracking_rows()

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos)
        }
