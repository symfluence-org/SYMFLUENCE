# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Global sensitivity analysis of model parameters on a dedicated sampling design.

Variance-based (Sobol) and elementary-effects (Morris) indices are only
meaningful for responses evaluated on the design they were derived for: a
Saltelli design for Sobol (Saltelli et al., 2010) and Morris trajectories for
elementary effects (Morris, 1991; Campolongo et al., 2007). Each design point
here is a real model run, evaluated through the calibration worker / parallel
infrastructure (:meth:`BaseModelOptimizer.evaluate_design`), and the response
is the calibration objective (OPTIMIZATION_METRIC, in maximisation
convention) on the calibration period.

Earlier versions estimated "Sobol" and RBD-FAST indices from the optimiser's
trajectory through an additive mean-of-1D-interpolations surrogate. That
surrogate has no interactions by construction (S_T == S_1) and the
trajectory is not a uniform design, so those indices were artefacts; the
path was removed. The only trajectory-based output left is
:meth:`SensitivityAnalyzer.trajectory_correlation`, a Spearman correlation of
the calibration samples that describes the optimiser path, not sensitivity.

References:
    Saltelli, A., et al. (2010). Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index. Computer
    Physics Communications, 181(2), 259-270.

    Campolongo, F., Cariboni, J. and Saltelli, A. (2007). An effective
    screening design for sensitivity analysis of large models. Environmental
    Modelling & Software, 22(10), 1509-1518.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from symfluence.core.exceptions import OptionalDependencyError
from symfluence.core.mixins import ConfigMixin

try:
    from SALib.analyze import morris as morris_analyze
    from SALib.analyze import sobol
    from SALib.sample import morris as morris_sample
    from SALib.sample import sobol as sobol_sample
except ImportError:  # SALib is provided by the optional ``sensitivity`` extra.
    sobol = None
    sobol_sample = None
    morris_analyze = None
    morris_sample = None

# Columns in the calibration-results CSV that are NOT calibration
# parameters and must be excluded from sensitivity analysis.
#
# Gradient-based optimisers (ADAM, L-BFGS) write extra diagnostic
# columns — grad_norm, lr, current_params — that co-author SH's
# iter-3 feedback called out (04.SH.3): "Sensitivity analysis step
# fails entirely for gradient-based algorithms — all parameters
# throw a type error caused by ADAM-specific columns that the
# sensitivity analyzer cannot handle." current_params in particular
# is a Python list that pandas writes as a string literal.
#
# We also filter any non-numeric column up front in
# ``_parameter_columns_for_sa`` as a belt-and-suspenders guard so
# future optimisers that add their own diagnostic columns don't
# regress this path.
_NON_PARAM_COLS = frozenset({
    # Bookkeeping
    'Iteration', 'iteration', 'step', 'timestamp', 'crash_count', 'crash_rate',
    # Objective scores (various conventions)
    'score', 'objective', 'Objective', 'fitness',
    'RMSE', 'KGE', 'KGEp', 'KGEnp', 'NSE', 'MAE',
    'Calib_RMSE', 'Calib_KGE', 'Calib_KGEp', 'Calib_KGEnp', 'Calib_NSE', 'Calib_MAE',
    'Eval_RMSE', 'Eval_KGE', 'Eval_KGEp', 'Eval_KGEnp', 'Eval_NSE', 'Eval_MAE',
    # Gradient-optimiser internals (ADAM / L-BFGS)
    'grad_norm', 'lr', 'current_params', 'step_size',
    'f_calls', 'n_iter', 'fun', 'gtol', 'ftol', 'xtol',
    # Sampler / population diagnostics (DREAM, ABC, PSO, GA, ...): these are
    # optimiser state that some algorithms append to the results CSV, not model
    # parameters. Constant ones are also caught by the zero-variance filter in
    # _parameter_columns_for_sa, but the varying ones (e.g. acceptance_rate)
    # must be named here so they don't pollute the sensitivity output.
    'acceptance_rate', 'n_chains',
})


def salib_available() -> bool:
    """True when SALib (the ``sensitivity`` extra) is installed."""
    return None not in (sobol, sobol_sample, morris_analyze, morris_sample)


def _require_salib():
    """Return SALib modules or explain how to enable global methods."""
    if None in (sobol, sobol_sample, morris_analyze, morris_sample):
        raise OptionalDependencyError(
            "Sobol and Morris sensitivity analysis", "sensitivity", dependency="SALib"
        )
    return sobol, sobol_sample, morris_analyze, morris_sample


def design_size(method: str, n_params: int, sobol_n: int = 256, morris_trajectories: int = 10) -> int:
    """Number of model runs a design needs (first- and total-order Sobol)."""
    if method == 'sobol':
        return int(sobol_n) * (n_params + 2)
    if method == 'morris':
        return int(morris_trajectories) * (n_params + 1)
    raise ValueError(f"Unknown sensitivity method {method!r}; expected 'sobol' or 'morris'")


def _unit_problem(names: Sequence[str]) -> Dict:
    """SALib problem on the normalized [0, 1] parameter space."""
    return {'num_vars': len(names), 'names': list(names), 'bounds': [[0.0, 1.0]] * len(names)}


def _fill_failed(y: np.ndarray, logger) -> tuple:
    """Replace failed runs (NaN) by the worst successful response.

    Saltelli / Morris estimators need a response at every design point. A run
    that crashed is treated as the worst behaviour seen; the count is
    reported so the user can judge the effect.
    """
    y = np.asarray(y, dtype=float).copy()
    failed = ~np.isfinite(y)
    n_failed = int(failed.sum())
    if n_failed == y.size:
        raise ValueError("Every sensitivity design run failed; no response to analyse")
    if n_failed:
        worst = float(np.nanmin(y))
        y[failed] = worst
        logger.warning(
            f"{n_failed}/{y.size} sensitivity design runs failed; they were assigned "
            f"the worst successful response ({worst:.4g}). Indices are unreliable if "
            "this fraction is large."
        )
    return y, n_failed


def sobol_indices(
    names: Sequence[str],
    evaluate: Callable[[np.ndarray], np.ndarray],
    n: int = 256,
    seed: Optional[int] = None,
    logger=None,
) -> pd.DataFrame:
    """First- and total-order Sobol indices from a Saltelli design.

    Args:
        names: Parameter names (the design lives on [0, 1] per parameter).
        evaluate: Maps a design ``(n_runs, d)`` in [0, 1] to responses
            ``(n_runs,)``; NaN marks a failed run.
        n: Base sample size N (a power of 2); N * (d + 2) runs.
        seed: Seed of the scrambled Sobol sequence and the bootstrap.

    Returns:
        DataFrame indexed by parameter with S1, S1_conf, ST, ST_conf.
    """
    salib_sobol, salib_sobol_sample, _, _ = _require_salib()
    logger = logger or logging.getLogger(__name__)
    problem = _unit_problem(names)
    X = salib_sobol_sample.sample(problem, int(n), calc_second_order=False, seed=seed)
    y, n_failed = _fill_failed(evaluate(X), logger)
    Si = salib_sobol.analyze(problem, y, calc_second_order=False, seed=seed)
    df = pd.DataFrame(
        {'S1': Si['S1'], 'S1_conf': Si['S1_conf'], 'ST': Si['ST'], 'ST_conf': Si['ST_conf']},
        index=list(names),
    )
    df.attrs.update({'n_runs': int(X.shape[0]), 'n_failed': n_failed})
    return df


def morris_indices(
    names: Sequence[str],
    evaluate: Callable[[np.ndarray], np.ndarray],
    trajectories: int = 10,
    levels: int = 4,
    seed: Optional[int] = None,
    logger=None,
) -> pd.DataFrame:
    """Morris elementary-effect statistics (mu*, mu, sigma) on [0, 1] inputs."""
    _, _, salib_morris_analyze, salib_morris_sample = _require_salib()
    logger = logger or logging.getLogger(__name__)
    problem = _unit_problem(names)
    X = salib_morris_sample.sample(problem, int(trajectories), num_levels=int(levels), seed=seed)
    y, n_failed = _fill_failed(evaluate(X), logger)
    Si = salib_morris_analyze.analyze(problem, X, y, num_levels=int(levels), seed=seed)
    df = pd.DataFrame(
        {'mu_star': Si['mu_star'], 'mu_star_conf': Si['mu_star_conf'],
         'mu': Si['mu'], 'sigma': Si['sigma']},
        index=list(names),
    )
    df.attrs.update({'n_runs': int(X.shape[0]), 'n_failed': n_failed})
    return df


def _parameter_columns_for_sa(samples) -> list:
    """Return the subset of columns that should be treated as
    calibration parameters.

    Drops:
      1. Anything in ``_NON_PARAM_COLS`` (known bookkeeping / score /
         optimiser-internal columns).
      2. Non-numeric columns (string literals like ``"[0.1, 0.5]"``
         from list-valued optimiser diagnostics, or bracketed
         stringified arrays from older calibration runs that PR
         #54 fixes at source but that still exist in on-disk
         historical CSVs).
    """
    numeric_cols = [
        c for c in samples.columns
        if c not in _NON_PARAM_COLS
        and pd.api.types.is_numeric_dtype(samples[c])
        and samples[c].nunique(dropna=True) > 1
    ]
    return numeric_cols


def _constant_parameter_columns(samples) -> list:
    """Numeric, non-bookkeeping columns that are constant across all samples.

    These are excluded from sensitivity analysis by
    ``_parameter_columns_for_sa`` (a constant column has no variance to
    attribute, and it makes SALib's Sobol sampler raise "Bounds are not
    legal"). Returned separately so the caller can report *which* parameters
    were dropped — e.g. an optimiser-internal like DREAM's ``n_chains`` or a
    genuine model parameter that a run pinned at a bound.
    """
    return [
        c for c in samples.columns
        if c not in _NON_PARAM_COLS
        and pd.api.types.is_numeric_dtype(samples[c])
        and samples[c].nunique(dropna=True) <= 1
    ]


class SensitivityAnalyzer(ConfigMixin):
    """
    Global sensitivity analysis of model parameters.

    :meth:`run_design_analysis` runs Sobol and/or Morris designs through a
    caller-supplied evaluator (each point a real model run).
    :meth:`run_trajectory_correlation` writes the Spearman correlation of a
    calibration trajectory, a diagnostic of the optimiser path only.

    Attributes:
        config: Configuration dictionary with domain settings.
        logger: Logger instance for status messages.
        reporting_manager: Optional manager for generating reports.
        output_folder: Directory for sensitivity analysis outputs.
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the sensitivity analyzer.

        Args:
            config: Configuration dictionary containing SYMFLUENCE_DATA_DIR
                and DOMAIN_NAME settings.
            logger: Logger instance for status and debug messages.
            reporting_manager: Optional reporting manager for visualizations.
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.reporting_manager = reporting_manager
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_folder = self.project_dir / "reporting" / "sensitivity_analysis"
        # Scope the output by experiment so runs that share a domain (e.g. the
        # calibration-algorithm ensemble: one experiment per algorithm) each
        # keep their own sensitivity results instead of overwriting a single
        # domain-level folder. Falls back to the flat folder when no experiment
        # id is configured, preserving the single-run layout.
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID'
        )
        if experiment_id:
            self.output_folder = self.output_folder / str(experiment_id)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @property
    def methods(self) -> List[str]:
        """Configured design-based methods (SENSITIVITY_METHODS)."""
        value = self._get_config_value(
            lambda: self.config.optimization.sensitivity.methods, dict_key='SENSITIVITY_METHODS')
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(',')
        return [str(m).strip().lower() for m in value if str(m).strip()]

    @property
    def trajectory_correlation_enabled(self) -> bool:
        return bool(self._get_config_value(
            lambda: self.config.optimization.sensitivity.trajectory_correlation,
            dict_key='SENSITIVITY_TRAJECTORY_CORRELATION'))

    # ------------------------------------------------------------------
    # Design-based analysis
    # ------------------------------------------------------------------
    def planned_runs(self, n_params: int, methods: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Model runs each configured design needs for ``n_params`` parameters."""
        return {
            m: design_size(
                m, n_params,
                sobol_n=int(self._get_config_value(
                    lambda: self.config.optimization.sensitivity.sobol_n,
                    dict_key='SENSITIVITY_SOBOL_N')),
                morris_trajectories=int(self._get_config_value(
                    lambda: self.config.optimization.sensitivity.morris_trajectories,
                    dict_key='SENSITIVITY_MORRIS_TRAJECTORIES')),
            )
            for m in (self.methods if methods is None else methods)
        }

    def run_design_analysis(
        self,
        names: Sequence[str],
        evaluate: Callable[[np.ndarray], np.ndarray],
        methods: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Run the configured Sobol / Morris designs and save the indices.

        Args:
            names: Calibration parameter names (design space [0, 1]^d, mapped
                to the parameter bounds by the evaluator).
            evaluate: Design ``(n_runs, d)`` -> responses ``(n_runs,)``;
                NaN marks a failed run.
            methods: Override of SENSITIVITY_METHODS.
            seed: Design / bootstrap seed (RANDOM_SEED).

        Returns:
            Comparison DataFrame (one column per index) or None when no method
            is configured.

        Raises:
            ValueError: when a design exceeds SENSITIVITY_MAX_EVALUATIONS.
        """
        methods = list(self.methods if methods is None else methods)
        if not methods:
            self.logger.info("No design-based sensitivity method configured (SENSITIVITY_METHODS)")
            return None
        names = list(names)
        planned = self.planned_runs(len(names), methods)
        budget = int(self._get_config_value(
            lambda: self.config.optimization.sensitivity.max_evaluations,
            dict_key='SENSITIVITY_MAX_EVALUATIONS'))
        total = sum(planned.values())
        if total > budget:
            raise ValueError(
                f"Sensitivity designs need {total} model runs ({planned}) for {len(names)} "
                f"parameters, above SENSITIVITY_MAX_EVALUATIONS={budget}. Lower "
                "SENSITIVITY_SOBOL_N / SENSITIVITY_MORRIS_TRAJECTORIES, use 'morris' only, "
                "or raise the budget."
            )
        self.logger.info(f"Sensitivity designs: {planned} model runs for {len(names)} parameters")

        columns: Dict[str, pd.Series] = {}
        for method in methods:
            if method == 'sobol':
                df = sobol_indices(
                    names, evaluate,
                    n=int(self._get_config_value(
                        lambda: self.config.optimization.sensitivity.sobol_n,
                        dict_key='SENSITIVITY_SOBOL_N')),
                    seed=seed, logger=self.logger)
                columns['Sobol_S1'] = df['S1']
                columns['Sobol_ST'] = df['ST']
            elif method == 'morris':
                df = morris_indices(
                    names, evaluate,
                    trajectories=int(self._get_config_value(
                        lambda: self.config.optimization.sensitivity.morris_trajectories,
                        dict_key='SENSITIVITY_MORRIS_TRAJECTORIES')),
                    levels=int(self._get_config_value(
                        lambda: self.config.optimization.sensitivity.morris_levels,
                        dict_key='SENSITIVITY_MORRIS_LEVELS')),
                    seed=seed, logger=self.logger)
                columns['Morris_mu_star'] = df['mu_star']
                columns['Morris_sigma'] = df['sigma']
            else:
                raise ValueError(f"Unknown sensitivity method {method!r}")
            out = self.output_folder / f'{method}_sensitivity.csv'
            df.to_csv(out)
            self.logger.info(
                f"Saved {method} indices ({df.attrs.get('n_runs')} runs, "
                f"{df.attrs.get('n_failed')} failed) to {out}")
            if self.reporting_manager:
                key = 'ST' if method == 'sobol' else 'mu_star'
                self.reporting_manager.visualize_sensitivity_analysis(
                    df[key], self.output_folder / f'{method}_sensitivity.png', plot_type='single')

        comparison = pd.DataFrame(columns)
        comparison.to_csv(self.output_folder / 'all_sensitivity_results.csv')
        if self.reporting_manager and len(comparison.columns) > 1:
            self.reporting_manager.visualize_sensitivity_analysis(
                comparison, self.output_folder / 'sensitivity_comparison.png',
                plot_type='comparison')
        return comparison

    # ------------------------------------------------------------------
    # Calibration-trajectory diagnostic
    # ------------------------------------------------------------------
    def read_calibration_results(self, file_path):
        """
        Read calibration results from CSV file.

        Args:
            file_path: Path to calibration results CSV file.

        Returns:
            pd.DataFrame: Calibration results with NaN rows removed.
        """
        df = pd.read_csv(file_path)
        return df.dropna()

    def preprocess_data(self, samples, metric='RMSE'):
        """
        Preprocess calibration samples for the trajectory diagnostic.

        Steps:
          1. Drop rows whose metric value is non-finite (NaN from crashed
             iterations, inf from degenerate metric maths).
          2. Drop rows whose metric value is a known SYMFLUENCE failure
             sentinel (``<= -900``) -- crashed-run markers, not responses.
          3. De-duplicate on parameter columns -- DDS produces repeated
             rows when it restarts from the best-so-far.

        Args:
            samples: DataFrame of calibration samples with parameter values.
            metric: Metric column name used for failure-sentinel filtering.

        Returns:
            pd.DataFrame: Cleaned, deduplicated samples.
        """
        n_in = len(samples)

        if metric in samples.columns:
            metric_values = pd.to_numeric(samples[metric], errors='coerce')
            finite_mask = np.isfinite(metric_values)
            sentinel_mask = metric_values > -900
            clean = samples[finite_mask & sentinel_mask].copy()
            n_dropped = n_in - len(clean)
            if n_dropped > 0:
                self.logger.info(
                    f"Sensitivity preprocessing dropped {n_dropped}/{n_in} rows "
                    f"with non-finite or failure-sentinel {metric} values "
                    f"(NaN / <=-900). Keeping {len(clean)} usable rows."
                )
        else:
            clean = samples

        samples_unique = clean.drop_duplicates(
            subset=[col for col in clean.columns if col != 'Iteration']
        )
        if len(samples_unique) < len(clean):
            self.logger.info(
                f"Sensitivity preprocessing dropped "
                f"{len(clean) - len(samples_unique)} duplicate rows."
            )
        return samples_unique

    def trajectory_correlation(self, samples, metric='RMSE'):
        """
        Absolute Spearman correlation between each parameter and the metric
        along the calibration trajectory.

        This describes where the optimiser went, not how sensitive the model
        is: the samples are concentrated near the optimum and are not a
        designed experiment. Use :meth:`run_design_analysis` for indices.

        Args:
            samples: DataFrame with parameter values and metric columns.
            metric: Name of the objective metric column.

        Returns:
            pd.Series: |Spearman rho| for each parameter.
        """
        self.logger.info(f"Computing trajectory correlation with {metric}")
        parameter_columns = _parameter_columns_for_sa(samples)
        correlations = []
        for param in parameter_columns:
            corr, _ = spearmanr(samples[param], samples[metric])
            correlations.append(abs(corr))
        return pd.Series(correlations, index=parameter_columns, name='trajectory_correlation')

    def run_trajectory_correlation(self, results_file):
        """
        Write ``trajectory_correlation.csv`` from a calibration results CSV.

        Args:
            results_file: Path to calibration results CSV file.

        Returns:
            pd.Series of correlations, or None when too few usable rows exist.
        """
        results = self.read_calibration_results(results_file)
        self.logger.info(f"Read {len(results)} calibration results")

        if len(results) < 10:
            self.logger.warning("Not enough calibration rows for a trajectory correlation.")
            return None

        # Auto-detect metric column: prefer Calib_RMSE (MESH), fall back to score (generic DDS)
        metric_col = 'Calib_RMSE'
        if metric_col not in results.columns:
            for candidate in ['score', 'Calib_KGE', 'Calib_KGEnp', 'Calib_NSE']:
                if candidate in results.columns:
                    metric_col = candidate
                    break
        self.logger.info(f"Using metric column: {metric_col}")

        results_preprocessed = self.preprocess_data(results, metric=metric_col)

        # The raw-row guard above counts rows before preprocessing drops
        # non-finite and failure-sentinel (<= -900) scores; when a model crashed
        # on most/all trials nothing usable remains.
        if len(results_preprocessed) < 10:
            self.logger.warning(
                f"Only {len(results_preprocessed)} usable calibration samples remain "
                f"after dropping non-finite / failure-sentinel '{metric_col}' values "
                f"(from {len(results)} raw rows). This usually means the model crashed "
                f"on most or all calibration trials. Skipping the trajectory correlation."
            )
            return None

        dropped = _constant_parameter_columns(results_preprocessed)
        if dropped:
            self.logger.warning(
                f"Excluding {len(dropped)} constant column(s) from the trajectory "
                f"correlation (no variance): {', '.join(dropped)}"
            )
        if not _parameter_columns_for_sa(results_preprocessed):
            self.logger.warning(
                "No varying calibration parameters remain after excluding "
                "constant columns; skipping the trajectory correlation."
            )
            return None

        corr = self.trajectory_correlation(results_preprocessed, metric=metric_col)
        corr.to_csv(self.output_folder / 'trajectory_correlation.csv')
        return corr

    def run_sensitivity_analysis(self, results_file):
        """Backward-compatible entry point: the trajectory-correlation diagnostic.

        Kept for model-specific analyzers registered against the old
        interface. Design-based indices come from :meth:`run_design_analysis`.
        """
        return self.run_trajectory_correlation(results_file)
