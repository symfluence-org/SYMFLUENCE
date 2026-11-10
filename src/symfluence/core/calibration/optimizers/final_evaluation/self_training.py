# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Split-sample final evaluation for self-training (ML) models.

LSTM/GNN learn their weights during the run step and have no external
parameter search, so nothing would otherwise write the standard
``optimization/<MODEL>/<algo>_<experiment>/<eid>_<algo>_final_evaluation.json``
(+ ``best_params.json``) that every calibrated model emits. This scores the
run's saved streamflow against observations for the calibration and evaluation
windows and writes those artifacts through :class:`FinalResultsSaver`.

Called from both the model runners (right after results are saved, so a run
without a calibrate step still emits metrics) and
``OptimizationManager.calibrate_model`` — the output is identical, so running
both just rewrites the same files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.modeling.utilities.periods import split_period

from .results_saver import FinalResultsSaver


def _load_simulated_streamflow(
    config: Any,
    project_dir: Path,
    experiment_id: str,
    model_name: str,
    logger: logging.Logger,
) -> Optional[pd.Series]:
    """Load the run's discharge (m³/s) via the streamflow evaluator.

    The evaluator consults the model's registered result extractor first
    (lumped ``predicted_streamflow``), honours volumetric units on distributed
    per-GRU output, and falls back to ``<MODEL>_discharge_cms`` in
    ``results/<exp>_results.csv``.
    """
    from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator

    sim_dir = project_dir / 'simulations' / experiment_id / model_name
    if not sim_dir.is_dir():
        logger.warning(
            f"{model_name} evaluation skipped: run output directory {sim_dir} not found "
            "(run the model step first)"
        )
        return None
    evaluator = StreamflowEvaluator(config, project_dir, logger, model_name=model_name)
    sim_files = evaluator.get_simulation_files(sim_dir)
    if not sim_files:
        logger.warning(f"{model_name} evaluation skipped: no streamflow output in {sim_dir}")
        return None
    try:
        series = evaluator.extract_simulated_data(sim_files)
    except (ValueError, KeyError, OSError) as e:
        logger.warning(f"{model_name} evaluation skipped: could not read simulated streamflow: {e}")
        return None
    series = pd.Series(series).astype(float)
    series.index = pd.DatetimeIndex(series.index)
    return series


def evaluate_self_training_model(
    config: Any,
    config_dict: Dict[str, Any],
    project_dir: Path,
    experiment_id: str,
    domain_name: str,
    model_name: str,
    algorithm: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Score a self-training model's run and persist the standard artifacts.

    Returns:
        Path to the saved ``*_final_evaluation.json``, or None when the run
        output, observations or periods do not allow any metric.
    """
    from symfluence.core.metrics.metrics_core import calculate_all_metrics
    from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

    logger = logger or logging.getLogger(__name__)
    project_dir = Path(project_dir)
    model_name = model_name.upper()

    sim_series = _load_simulated_streamflow(config, project_dir, experiment_id, model_name, logger)
    if sim_series is None or sim_series.dropna().empty:
        return None
    sim_series = sim_series.resample('D').mean()

    sm = StreamflowMetrics()
    obs_values, obs_index = sm.load_observations(config_dict, project_dir, domain_name, resample_freq='D')
    if obs_values is None:
        logger.warning(f"{model_name} evaluation skipped: no observations found")
        return None
    obs_series = pd.Series(obs_values, index=obs_index)

    periods = {
        'calibration_metrics': resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD'),
        'evaluation_metrics': resolve_config_value(config, lambda: config.domain.evaluation_period, '', 'EVALUATION_PERIOD'),
    }
    final_result: Dict[str, Any] = {'best_params': {}}
    for key, period_str in periods.items():
        final_result[key] = {}
        period = split_period(period_str, allow_extra=True)
        if period is None:
            continue
        try:
            obs_aligned, sim_aligned = sm.align_timeseries(sim_series, obs_series, calibration_period=period)
        except ValueError as e:
            logger.warning(f"{model_name} {key} not computable: {e}")
            continue
        final_result[key] = calculate_all_metrics(pd.Series(obs_aligned), pd.Series(sim_aligned))

    if not final_result['calibration_metrics'] and not final_result['evaluation_metrics']:
        logger.warning(f"{model_name} evaluation produced no metrics")
        return None

    safe_algorithm = algorithm.lower().replace('/', '_')
    results_dir = project_dir / 'optimization' / model_name / f"{algorithm.lower()}_{experiment_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    saver = FinalResultsSaver(results_dir, experiment_id, domain_name, logger)
    saver.log_results(final_result['calibration_metrics'], final_result['evaluation_metrics'])
    saved = saver.save_results(final_result, algorithm)

    # Repro tooling (repro_status/collect_metrics.py) discovers runs via
    # <eid>_<algo>_best_params.json gated on a final_evaluation/ dir —
    # emit both so self-training members are collected like everyone else.
    calib_kge = final_result['calibration_metrics'].get('KGE')
    best_params_file = results_dir / f"{experiment_id}_{safe_algorithm}_best_params.json"
    best_params_file.write_text(json.dumps({
        'experiment_id': experiment_id,
        'model': model_name,
        'metric': 'KGE',
        'best_score': FinalResultsSaver._convert_to_serializable(calib_kge),
        'best_iteration': '',
        'best_params': {},
        'note': 'self-training model: score is the trained simulation, no parameter search',
    }, indent=2), encoding='utf-8')
    (results_dir / 'final_evaluation').mkdir(exist_ok=True)
    return saved


def persist_run_evaluation(runner: Any, model_name: str) -> Optional[Path]:
    """Runner hook: evaluate a finished self-training run, never raising.

    ``runner`` is a model runner exposing ``config``, ``config_dict``,
    ``project_dir``, ``experiment_id``, ``domain_name`` and ``logger``. The
    algorithm label matches what ``calibrate_model`` would use
    (``ITERATIVE_OPTIMIZATION_ALGORITHM``, default PSO) so both writers target the same
    ``optimization/<MODEL>/<algo>_<experiment>/`` directory.
    """
    logger = getattr(runner, 'logger', None) or logging.getLogger(__name__)
    try:
        config = runner.config
        algorithm = resolve_config_value(config, lambda: config.optimization.algorithm, 'PSO', 'ITERATIVE_OPTIMIZATION_ALGORITHM')
        return evaluate_self_training_model(
            config, runner.config_dict, Path(runner.project_dir), runner.experiment_id,
            runner.domain_name, model_name, str(algorithm or 'PSO'), logger,
        )
    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.warning(f"{model_name} final evaluation could not be written: {e}", exc_info=True)
        return None
