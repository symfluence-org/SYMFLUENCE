#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the error logging system in SUMMA optimization workers.

Tests cover:
- ErrorLogger initialization with different config options
- PARAMS_KEEP_TRIALS convenience flag behavior
- Failure artifact capture
- Success logging in 'all' mode
- Summary generation
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from symfluence.models.summa.calibration.worker_impl.error_logging import (
    ErrorLogger,
    _serialize_debug_info,
    _serialize_params,
    get_worker_error_logger,
    init_worker_error_logger,
    log_worker_failure,
)


class TestErrorLoggerInitialization:
    """Test ErrorLogger initialization with various config options."""

    def test_default_mode_is_none(self, tmp_path):
        """Default mode should be 'none' when no options set."""
        config = {}
        logger = ErrorLogger(config, tmp_path)

        assert logger.mode == 'none'
        assert not logger.stop_on_failure
        assert not logger.has_failure

    def test_params_keep_trials_enables_failures_mode(self, tmp_path):
        """PARAMS_KEEP_TRIALS=True should enable 'failures' mode."""
        config = {'PARAMS_KEEP_TRIALS': True}
        logger = ErrorLogger(config, tmp_path)

        assert logger.mode == 'failures'
        assert not logger.stop_on_failure

    def test_params_keep_trials_with_a_validated_config(self, tmp_path):
        """A validated config carries the schema's ERROR_LOGGING_MODE='none'."""
        config = {'PARAMS_KEEP_TRIALS': True, 'ERROR_LOGGING_MODE': 'none'}
        assert ErrorLogger(config, tmp_path).mode == 'failures'
        config = {'PARAMS_KEEP_TRIALS': True, 'ERROR_LOGGING_MODE': 'all'}
        assert ErrorLogger(config, tmp_path).mode == 'all'

    def test_explicit_error_logging_mode(self, tmp_path):
        """Explicit ERROR_LOGGING_MODE should override default."""
        config = {'ERROR_LOGGING_MODE': 'all'}
        logger = ErrorLogger(config, tmp_path)

        assert logger.mode == 'all'

    def test_params_keep_trials_with_explicit_mode(self, tmp_path):
        """PARAMS_KEEP_TRIALS + explicit mode should use explicit mode."""
        config = {
            'PARAMS_KEEP_TRIALS': True,
            'ERROR_LOGGING_MODE': 'all'
        }
        logger = ErrorLogger(config, tmp_path)

        assert logger.mode == 'all'

    def test_stop_on_failure_option(self, tmp_path):
        """STOP_ON_MODEL_FAILURE should be tracked."""
        config = {
            'PARAMS_KEEP_TRIALS': True,
            'STOP_ON_MODEL_FAILURE': True
        }
        logger = ErrorLogger(config, tmp_path)

        assert logger.stop_on_failure
        assert not logger.should_stop  # No failure yet

    def test_custom_error_log_dir(self, tmp_path):
        """ERROR_LOG_DIR should customize the directory name."""
        config = {
            'PARAMS_KEEP_TRIALS': True,
            'ERROR_LOG_DIR': 'my_custom_errors'
        }
        logger = ErrorLogger(config, tmp_path)

        assert logger.error_log_dir == tmp_path / 'my_custom_errors'

    def test_error_dir_created_when_enabled(self, tmp_path):
        """Error log directory should be created when logging enabled."""
        config = {'PARAMS_KEEP_TRIALS': True}
        logger = ErrorLogger(config, tmp_path)

        assert logger.error_log_dir.exists()

    def test_error_dir_not_created_when_disabled(self, tmp_path):
        """Error log directory should NOT be created when logging disabled."""
        config = {'PARAMS_KEEP_TRIALS': False}
        logger = ErrorLogger(config, tmp_path)

        assert not logger.error_log_dir.exists()


class TestErrorLoggerFailureLogging:
    """Test failure logging functionality."""

    @pytest.fixture
    def enabled_logger(self, tmp_path):
        """Create an enabled error logger."""
        config = {'PARAMS_KEEP_TRIALS': True}
        return ErrorLogger(config, tmp_path)

    @pytest.fixture
    def disabled_logger(self, tmp_path):
        """Create a disabled error logger."""
        config = {'PARAMS_KEEP_TRIALS': False}
        return ErrorLogger(config, tmp_path)

    @pytest.fixture
    def mock_settings_dir(self, tmp_path):
        """Create a mock settings directory with test files."""
        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir()

        # Create a mock trialParams.nc (just a text file for testing)
        (settings_dir / 'trialParams.nc').write_text('mock netcdf content')
        (settings_dir / 'coldState.nc').write_text('mock coldstate content')

        return settings_dir

    @pytest.fixture
    def mock_summa_dir(self, tmp_path):
        """Create a mock SUMMA directory with log files."""
        summa_dir = tmp_path / 'summa'
        summa_dir.mkdir()

        log_dir = summa_dir / 'logs'
        log_dir.mkdir()
        (log_dir / 'summa_worker_12345.log').write_text('SUMMA log content')

        return summa_dir

    def test_log_failure_when_disabled(self, disabled_logger, mock_settings_dir, mock_summa_dir):
        """log_failure should return None when logging disabled."""
        result = disabled_logger.log_failure(
            iteration=1,
            params={'k_soil': np.array([0.5])},
            debug_info={'stage': 'summa_execution', 'errors': ['test error']},
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='SUMMA failed'
        )

        assert result is None
        assert disabled_logger.failure_count == 0

    def test_log_failure_creates_directory(self, enabled_logger, mock_settings_dir, mock_summa_dir):
        """log_failure should create a failure directory."""
        result = enabled_logger.log_failure(
            iteration=42,
            params={'k_soil': np.array([0.5])},
            debug_info={'stage': 'summa_execution', 'errors': ['test error']},
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='SUMMA failed',
            proc_id=1,
            individual_id=3
        )

        assert result is not None
        assert result.exists()
        assert 'iter00042' in result.name
        assert 'proc01' in result.name
        assert 'ind003' in result.name

    def test_log_failure_copies_trial_params(self, enabled_logger, mock_settings_dir, mock_summa_dir):
        """log_failure should copy trialParams.nc."""
        result = enabled_logger.log_failure(
            iteration=1,
            params={'k_soil': np.array([0.5])},
            debug_info={'stage': 'test'},
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='test'
        )

        # Check that a trialParams file was copied
        trial_params_files = list(result.glob('trialParams_*.nc'))
        assert len(trial_params_files) == 1

    def test_log_failure_copies_summa_log(self, enabled_logger, mock_settings_dir, mock_summa_dir):
        """log_failure should copy SUMMA log file."""
        result = enabled_logger.log_failure(
            iteration=1,
            params={'k_soil': np.array([0.5])},
            debug_info={'stage': 'test'},
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='test'
        )

        # Check that a summa log was copied
        summa_log_files = list(result.glob('summa_*.log'))
        assert len(summa_log_files) == 1

    def test_log_failure_saves_debug_info(self, enabled_logger, mock_settings_dir, mock_summa_dir):
        """log_failure should save debug info as JSON."""
        params = {'k_soil': np.array([0.5]), 'theta_sat': 0.4}
        debug_info = {
            'stage': 'summa_execution',
            'errors': ['error1', 'error2'],
            'files_checked': ['/path/to/file']
        }

        result = enabled_logger.log_failure(
            iteration=1,
            params=params,
            debug_info=debug_info,
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='SUMMA crashed'
        )

        # Check that debug info JSON was created
        debug_files = list(result.glob('debug_info_*.json'))
        assert len(debug_files) == 1

        # Verify content
        with open(debug_files[0]) as f:
            saved_debug = json.load(f)

        assert saved_debug['error_message'] == 'SUMMA crashed'
        assert saved_debug['iteration'] == 1
        assert 'k_soil' in saved_debug['parameters']

    def test_log_failure_updates_counters(self, enabled_logger, mock_settings_dir, mock_summa_dir):
        """log_failure should update failure counters."""
        enabled_logger.log_failure(
            iteration=1,
            params={},
            debug_info={},
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='test'
        )

        assert enabled_logger.has_failure
        assert enabled_logger.failure_count == 1

    def test_should_stop_after_failure(self, tmp_path, mock_settings_dir, mock_summa_dir):
        """should_stop should be True after failure when STOP_ON_MODEL_FAILURE enabled."""
        config = {
            'PARAMS_KEEP_TRIALS': True,
            'STOP_ON_MODEL_FAILURE': True
        }
        logger = ErrorLogger(config, tmp_path)

        assert not logger.should_stop

        logger.log_failure(
            iteration=1,
            params={},
            debug_info={},
            settings_dir=mock_settings_dir,
            summa_dir=mock_summa_dir,
            error_message='test'
        )

        assert logger.should_stop


class TestErrorLoggerSuccessLogging:
    """Test success logging in 'all' mode."""

    def test_log_success_in_all_mode(self, tmp_path):
        """log_success should work in 'all' mode."""
        config = {'ERROR_LOGGING_MODE': 'all'}
        logger = ErrorLogger(config, tmp_path)

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir()
        (settings_dir / 'trialParams.nc').write_text('mock content')

        result = logger.log_success(
            iteration=5,
            params={'k_soil': 0.5},
            settings_dir=settings_dir,
            score=0.85
        )

        assert result is not None
        assert result.exists()
        assert 'successes' in str(result)

    def test_log_success_skipped_in_failures_mode(self, tmp_path):
        """log_success should return None in 'failures' mode."""
        config = {'PARAMS_KEEP_TRIALS': True}  # mode='failures'
        logger = ErrorLogger(config, tmp_path)

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir()

        result = logger.log_success(
            iteration=5,
            params={'k_soil': 0.5},
            settings_dir=settings_dir,
            score=0.85
        )

        assert result is None


class TestErrorLoggerSummary:
    """Test summary generation."""

    def test_get_summary_no_failures(self, tmp_path):
        """Summary should show zero failures when none logged."""
        config = {'PARAMS_KEEP_TRIALS': True}
        logger = ErrorLogger(config, tmp_path)

        summary = logger.get_summary()

        assert summary['mode'] == 'failures'
        assert summary['failure_count'] == 0
        assert not summary['has_failure']

    def test_get_summary_with_failures(self, tmp_path):
        """Summary should count logged failures."""
        config = {'PARAMS_KEEP_TRIALS': True}
        logger = ErrorLogger(config, tmp_path)

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir()
        summa_dir = tmp_path / 'summa'
        summa_dir.mkdir()

        # Log multiple failures
        for i in range(3):
            logger.log_failure(
                iteration=i,
                params={},
                debug_info={},
                settings_dir=settings_dir,
                summa_dir=summa_dir,
                error_message=f'error {i}'
            )

        summary = logger.get_summary()

        assert summary['failure_count'] == 3
        assert summary['has_failure']
        assert summary['logged_failures'] == 3


class TestSerializationHelpers:
    """Test parameter and debug info serialization."""

    def test_serialize_numpy_arrays(self):
        """Numpy arrays should be converted to lists."""
        params = {
            'scalar': 1.5,
            'array': np.array([1.0, 2.0, 3.0]),
            'int_array': np.array([1, 2, 3])
        }

        result = _serialize_params(params)

        assert result['scalar'] == 1.5
        assert result['array'] == [1.0, 2.0, 3.0]
        assert result['int_array'] == [1, 2, 3]

    def test_serialize_numpy_scalars(self):
        """Numpy scalars should be converted to Python types."""
        params = {
            'float64': np.float64(1.5),
            'int32': np.int32(42)
        }

        result = _serialize_params(params)

        assert isinstance(result['float64'], float)
        assert isinstance(result['int32'], float)  # All converted to float

    def test_serialize_debug_info_with_paths(self):
        """Paths should be converted to strings."""
        debug_info = {
            'stage': 'test',
            'files_checked': [Path('/path/to/file1'), Path('/path/to/file2')],
            'single_path': Path('/single/path')
        }

        result = _serialize_debug_info(debug_info)

        assert Path(result['single_path']) == Path('/single/path')
        assert all(isinstance(f, str) for f in result['files_checked'])


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_log_worker_failure_with_disabled_config(self, tmp_path):
        """log_worker_failure should return None when logging disabled."""
        config = {'PARAMS_KEEP_TRIALS': False, 'ERROR_LOGGING_MODE': 'none'}

        result = log_worker_failure(
            iteration=1,
            params={},
            debug_info={},
            settings_dir=tmp_path,
            summa_dir=tmp_path,
            error_message='test',
            config=config
        )

        assert result is None

    def test_init_and_get_worker_error_logger(self, tmp_path):
        """init_worker_error_logger should create retrievable logger."""
        config = {'PARAMS_KEEP_TRIALS': True}

        logger = init_worker_error_logger(config, tmp_path)
        retrieved = get_worker_error_logger()

        assert retrieved is logger
        assert retrieved.mode == 'failures'


class TestConfigModelIntegration:
    """Test that config model properly supports new options."""

    def test_optimization_config_has_error_logging_fields(self):
        """OptimizationConfig should have error logging fields."""
        from symfluence.core.config.models.optimization import OptimizationConfig

        # Create with defaults
        config = OptimizationConfig()

        assert hasattr(config, 'params_keep_trials')
        assert hasattr(config, 'error_logging_mode')
        assert hasattr(config, 'stop_on_model_failure')
        assert hasattr(config, 'error_log_dir')

        # Check defaults
        assert config.params_keep_trials == False
        assert config.error_logging_mode == 'none'
        assert config.stop_on_model_failure == False
        assert config.error_log_dir == 'error_logs'

    def test_optimization_config_with_params_keep_trials(self):
        """OptimizationConfig should accept PARAMS_KEEP_TRIALS."""
        from symfluence.core.config.models.optimization import OptimizationConfig

        config = OptimizationConfig(PARAMS_KEEP_TRIALS=True)

        assert config.params_keep_trials == True
