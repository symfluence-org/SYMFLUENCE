#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Error Logging for SUMMA Workers

This module provides centralized error artifact capture for debugging
failed model runs during calibration. When enabled, it preserves:
- Parameter files (trialParams.nc) that caused failures
- SUMMA/mizuRoute log files from failed runs
- Debug information dictionaries with error context

Configuration Options:
    PARAMS_KEEP_TRIALS: bool - Convenience flag that enables error logging
        with sensible defaults (ERROR_LOGGING_MODE='failures')
    ERROR_LOGGING_MODE: str - 'none', 'failures', or 'all'
    STOP_ON_MODEL_FAILURE: bool - Halt optimization on first failure
    ERROR_LOG_DIR: str - Subdirectory name for error artifacts
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.mixins.config import resolve_config_value

from ..parameter_files import resolve_settings_file_names


class ErrorLogger:
    """Centralized error logging for calibration runs.

    Captures and preserves artifacts from failed model runs to help
    binary teams debug issues in SUMMA/mizuRoute code.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        base_output_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the error logger.

        Args:
            config: Configuration dictionary
            base_output_dir: Base output directory for optimization
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Determine error logging mode
        # PARAMS_KEEP_TRIALS is a convenience flag that enables 'failures' mode
        params_keep_trials = resolve_config_value(config, lambda: config.optimization.params_keep_trials, False, 'PARAMS_KEEP_TRIALS')

        if params_keep_trials:
            # PARAMS_KEEP_TRIALS enables failure logging by default
            self.mode = resolve_config_value(config, lambda: config.optimization.error_logging_mode, 'failures', 'ERROR_LOGGING_MODE')
        else:
            self.mode = resolve_config_value(config, lambda: config.optimization.error_logging_mode, 'none', 'ERROR_LOGGING_MODE')

        self.stop_on_failure = resolve_config_value(config, lambda: config.optimization.stop_on_model_failure, False, 'STOP_ON_MODEL_FAILURE')

        # Set up error log directory
        error_log_subdir = resolve_config_value(config, lambda: config.optimization.error_log_dir, 'error_logs', 'ERROR_LOG_DIR')
        self.error_log_dir = base_output_dir / error_log_subdir

        # Track if we've had a failure (for STOP_ON_MODEL_FAILURE)
        self.has_failure = False
        self._failure_count = 0

        # Create directory if logging is enabled
        if self.mode != 'none':
            self.error_log_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Error logging enabled (mode={self.mode}), artifacts will be saved to: {self.error_log_dir}")

    @property
    def should_stop(self) -> bool:
        """Check if optimization should stop due to failure."""
        return self.stop_on_failure and self.has_failure

    @property
    def failure_count(self) -> int:
        """Return the number of failures logged."""
        return self._failure_count

    def log_failure(
        self,
        iteration: int,
        params: Dict[str, Any],
        debug_info: Dict[str, Any],
        settings_dir: Path,
        summa_dir: Path,
        error_message: str,
        proc_id: int = 0,
        individual_id: int = 0
    ) -> Optional[Path]:
        """Log a failed model run with all relevant artifacts.

        Args:
            iteration: Current optimization iteration
            params: Parameter dictionary that caused the failure
            debug_info: Debug information from the worker
            settings_dir: SUMMA settings directory (contains trialParams.nc)
            summa_dir: SUMMA simulation directory (contains logs)
            error_message: Error message describing the failure
            proc_id: Process ID (for parallel runs)
            individual_id: Individual ID within the population

        Returns:
            Path to the error log directory for this failure, or None if logging disabled
        """
        if self.mode == 'none':
            return None

        self.has_failure = True
        self._failure_count += 1

        # Create unique identifier for this failure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        failure_id = f"iter{iteration:05d}_proc{proc_id:02d}_ind{individual_id:03d}_{timestamp}"
        failure_dir = self.error_log_dir / failure_id
        failure_dir.mkdir(parents=True, exist_ok=True)

        self.logger.warning(f"Logging failure artifacts to: {failure_dir}")

        # 1. Copy trialParams.nc if it exists
        trial_params_src = settings_dir / resolve_settings_file_names(self.config).trialparams
        if trial_params_src.exists():
            try:
                trial_params_dst = failure_dir / f"trialParams_{failure_id}.nc"
                shutil.copy2(trial_params_src, trial_params_dst)
                self.logger.debug(f"Copied trialParams.nc to {trial_params_dst}")
            except (IOError, OSError, KeyError) as e:
                self.logger.warning(f"Failed to copy trialParams.nc: {e}")

        # 2. Copy SUMMA log file if it exists
        summa_log_dir = summa_dir / "logs"
        if summa_log_dir.exists():
            for log_file in summa_log_dir.glob("summa_worker_*.log"):
                try:
                    log_dst = failure_dir / f"summa_{failure_id}.log"
                    shutil.copy2(log_file, log_dst)
                    self.logger.debug(f"Copied SUMMA log to {log_dst}")
                    break  # Only copy the most recent one
                except (IOError, OSError, KeyError) as e:
                    self.logger.warning(f"Failed to copy SUMMA log: {e}")

        # 3. Copy mizuRoute log if it exists
        mizuroute_log_dir = summa_dir.parent / "mizuRoute" / "logs"
        if mizuroute_log_dir.exists():
            for log_file in mizuroute_log_dir.glob("mizuroute_worker_*.log"):
                try:
                    log_dst = failure_dir / f"mizuroute_{failure_id}.log"
                    shutil.copy2(log_file, log_dst)
                    self.logger.debug(f"Copied mizuRoute log to {log_dst}")
                    break
                except (IOError, OSError, KeyError) as e:
                    self.logger.warning(f"Failed to copy mizuRoute log: {e}")

        # 4. Save debug info as JSON
        debug_output = {
            'failure_id': failure_id,
            'iteration': iteration,
            'proc_id': proc_id,
            'individual_id': individual_id,
            'timestamp': timestamp,
            'error_message': error_message,
            'parameters': _serialize_params(params),
            'debug_info': _serialize_debug_info(debug_info),
            'settings_dir': str(settings_dir),
            'summa_dir': str(summa_dir)
        }

        debug_file = failure_dir / f"debug_info_{failure_id}.json"
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_output, f, indent=2, default=str)
            self.logger.debug(f"Saved debug info to {debug_file}")
        except (IOError, OSError, KeyError) as e:
            self.logger.warning(f"Failed to save debug info: {e}")

        # 5. Copy coldState.nc if it exists (for soil depth debugging)
        coldstate_src = settings_dir / resolve_settings_file_names(self.config).coldstate
        if coldstate_src.exists():
            try:
                coldstate_dst = failure_dir / f"coldState_{failure_id}.nc"
                shutil.copy2(coldstate_src, coldstate_dst)
                self.logger.debug(f"Copied coldState.nc to {coldstate_dst}")
            except (IOError, OSError, KeyError) as e:
                self.logger.warning(f"Failed to copy coldState.nc: {e}")

        return failure_dir

    def log_success(
        self,
        iteration: int,
        params: Dict[str, Any],
        settings_dir: Path,
        score: float
    ) -> Optional[Path]:
        """Log a successful model run (only if mode='all').

        Args:
            iteration: Current optimization iteration
            params: Parameter dictionary
            settings_dir: SUMMA settings directory
            score: Resulting score from the run

        Returns:
            Path to the logged artifacts, or None if not logging successes
        """
        if self.mode != 'all':
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        success_id = f"iter{iteration:05d}_success_{timestamp}"
        success_dir = self.error_log_dir / "successes" / success_id
        success_dir.mkdir(parents=True, exist_ok=True)

        # Copy trialParams.nc
        trial_params_src = settings_dir / resolve_settings_file_names(self.config).trialparams
        if trial_params_src.exists():
            try:
                trial_params_dst = success_dir / f"trialParams_{success_id}.nc"
                shutil.copy2(trial_params_src, trial_params_dst)
            except (IOError, OSError, KeyError) as e:
                self.logger.warning(f"Failed to copy trialParams.nc: {e}")

        # Save parameters and score
        success_info = {
            'iteration': iteration,
            'timestamp': timestamp,
            'score': score,
            'parameters': _serialize_params(params)
        }

        info_file = success_dir / f"run_info_{success_id}.json"
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(success_info, f, indent=2, default=str)
        except (IOError, OSError, KeyError) as e:
            self.logger.warning(f"Failed to save success info: {e}")

        return success_dir

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged errors.

        Returns:
            Dictionary with error logging statistics
        """
        summary = {
            'mode': self.mode,
            'error_log_dir': str(self.error_log_dir),
            'failure_count': self._failure_count,
            'stop_on_failure': self.stop_on_failure,
            'has_failure': self.has_failure
        }

        # Count actual failure directories
        if self.error_log_dir.exists():
            failure_dirs = [d for d in self.error_log_dir.iterdir()
                          if d.is_dir() and d.name.startswith('iter')]
            summary['logged_failures'] = len(failure_dirs)

        return summary


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize parameters for JSON output."""
    import numpy as np

    serialized = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized


def _serialize_debug_info(debug_info: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize debug info for JSON output."""
    import numpy as np

    serialized = {}
    for key, value in debug_info.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, list):
            serialized[key] = [str(v) if isinstance(v, Path) else v for v in value]
        elif isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized


# Module-level error logger instance for worker processes
_worker_error_logger: Optional[ErrorLogger] = None


def init_worker_error_logger(config: Dict[str, Any], output_dir: Path) -> ErrorLogger:
    """Initialize the worker error logger.

    This should be called once per worker process.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for the optimization

    Returns:
        Initialized ErrorLogger instance
    """
    global _worker_error_logger
    _worker_error_logger = ErrorLogger(config, output_dir)
    return _worker_error_logger


def get_worker_error_logger() -> Optional[ErrorLogger]:
    """Get the current worker error logger instance."""
    return _worker_error_logger


def log_worker_failure(
    iteration: int,
    params: Dict[str, Any],
    debug_info: Dict[str, Any],
    settings_dir: Path,
    summa_dir: Path,
    error_message: str,
    proc_id: int = 0,
    individual_id: int = 0,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """Convenience function to log a failure from worker code.

    If no error logger is initialized, this will attempt to create one
    using the provided config.

    Args:
        iteration: Current optimization iteration
        params: Parameter dictionary
        debug_info: Debug information
        settings_dir: SUMMA settings directory
        summa_dir: SUMMA simulation directory
        error_message: Error description
        proc_id: Process ID
        individual_id: Individual ID
        config: Optional config dict (used to initialize logger if needed)

    Returns:
        Path to failure artifacts, or None
    """
    global _worker_error_logger

    # Check if error logging is disabled in config
    if config:
        params_keep_trials = resolve_config_value(config, lambda: config.optimization.params_keep_trials, False, 'PARAMS_KEEP_TRIALS')
        error_mode = resolve_config_value(config, lambda: config.optimization.error_logging_mode, 'none', 'ERROR_LOGGING_MODE')
        if not params_keep_trials and error_mode == 'none':
            return None

    # Try to use existing logger or create one
    if _worker_error_logger is None and config is not None:
        # Determine output directory from config or settings_dir
        output_dir = settings_dir.parent.parent  # Go up from settings/SUMMA
        _worker_error_logger = ErrorLogger(config, output_dir)

    if _worker_error_logger is not None:
        return _worker_error_logger.log_failure(
            iteration=iteration,
            params=params,
            debug_info=debug_info,
            settings_dir=settings_dir,
            summa_dir=summa_dir,
            error_message=error_message,
            proc_id=proc_id,
            individual_id=individual_id
        )

    return None
