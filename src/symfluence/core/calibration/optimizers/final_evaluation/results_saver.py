# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Final Results Saver

Handles saving and formatting final evaluation results.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class FinalResultsSaver:
    """
    Saves and formats final evaluation results.

    Handles:
    - Serialization of numpy types
    - JSON result file creation
    - Metric extraction and formatting
    """

    def __init__(
        self,
        results_dir: Path,
        experiment_id: str,
        domain_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize results saver.

        Args:
            results_dir: Results directory path
            experiment_id: Experiment identifier
            domain_name: Domain name
            logger: Optional logger instance
        """
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.domain_name = domain_name
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """
        Recursively convert numpy types to Python native types.

        Args:
            obj: Object to convert

        Returns:
            Serializable version of object
        """
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: FinalResultsSaver._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FinalResultsSaver._convert_to_serializable(i) for i in obj]
        return obj

    def save_results(
        self,
        final_result: Dict[str, Any],
        algorithm: str
    ) -> Optional[Path]:
        """
        Save final evaluation results to JSON file.

        Args:
            final_result: Final evaluation results dictionary
            algorithm: Algorithm name (e.g., 'PSO', 'DDS')

        Returns:
            Path to saved file, or None if failed
        """
        try:
            safe_algorithm = algorithm.lower().replace('/', '_')
            output_file = self.results_dir / f'{self.experiment_id}_{safe_algorithm}_final_evaluation.json'

            serializable_result = {
                'algorithm': algorithm,
                'experiment_id': self.experiment_id,
                'domain_name': self.domain_name,
                'calibration_metrics': self._convert_to_serializable(
                    final_result.get('calibration_metrics', {})
                ),
                'evaluation_metrics': self._convert_to_serializable(
                    final_result.get('evaluation_metrics', {})
                ),
                'best_params': self._convert_to_serializable(
                    final_result.get('best_params', {})
                ),
                **({'test_metrics': self._convert_to_serializable(final_result['test_metrics'])}
                   if final_result.get('test_metrics') else {}),
                **({'period_warning': final_result['period_warning']}
                   if final_result.get('period_warning') else {}),
                'timestamp': datetime.now().isoformat()
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2)

            self.logger.info(f"Saved final evaluation results to {output_file}")
            return output_file

        except (IOError, OSError, ValueError) as e:
            self.logger.error(f"Failed to save final evaluation results: {e}")
            return None

    @staticmethod
    def extract_period_metrics(
        all_metrics: Dict[str, Any],
        period_prefix: str
    ) -> Dict[str, Any]:
        """
        Extract metrics for a specific period (Calib, Eval or Test).

        Args:
            all_metrics: All metrics dictionary
            period_prefix: Period prefix ('Calib' or 'Eval')

        Returns:
            Dictionary of period-specific metrics
        """
        period_metrics = {}
        for key, value in all_metrics.items():
            if key.startswith(f"{period_prefix}_"):
                # Remove prefix for cleaner reporting
                period_metrics[key.replace(f"{period_prefix}_", "")] = value
            elif period_prefix == 'Calib' and not any(
                key.startswith(p) for p in ['Calib_', 'Eval_', 'Test_']
            ) and key != 'period_warning':
                # Include unprefixed metrics in calibration (backwards compatibility)
                period_metrics[key] = value
        return period_metrics

    def log_results(
        self,
        calib_metrics: Dict[str, Any],
        eval_metrics: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        period_warning: Optional[str] = None,
    ) -> None:
        """
        Log detailed final evaluation results.

        Args:
            calib_metrics: Calibration period metrics
            eval_metrics: Evaluation period metrics
            test_metrics: Independent test period metrics (optional)
            period_warning: Split-sample warning from the evaluator (optional)
        """
        lines = ["=" * 60, "FINAL EVALUATION RESULTS", "=" * 60]
        if period_warning:
            lines.append(f"WARNING: {period_warning}")

        def _add(metrics: Dict[str, Any]) -> None:
            for metric, value in sorted(metrics.items()):
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not np.isnan(value):
                        lines.append(f"   {metric}: {value:.6f}")
                elif value is not None:
                    lines.append(f"   {metric}: {value}")

        # Calibration period
        if calib_metrics:
            lines.append("CALIBRATION PERIOD PERFORMANCE:")
            _add(calib_metrics)

        # Evaluation period
        if eval_metrics:
            lines.append("EVALUATION PERIOD PERFORMANCE:")
            _add(eval_metrics)
        else:
            lines.append("EVALUATION PERIOD: No evaluation period configured")

        # Independent test period (never used for optimisation or selection)
        if test_metrics:
            lines.append("TEST PERIOD PERFORMANCE (independent hold-out):")
            _add(test_metrics)

        lines.append("=" * 60)

        # One multi-line record keeps the banner intact while emitting a
        # single log record instead of one per metric.
        self.logger.info("\n".join(lines))
