"""
Unit tests for DataProcessor.

Tests data loading and preparation functionality for visualization.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from symfluence.reporting.processors.data_processor import DataProcessor


@pytest.fixture
def data_processor(mock_config, mock_logger):
    """Create a DataProcessor instance."""
    return DataProcessor(mock_config, mock_logger)


class TestDataProcessor:
    """Test suite for DataProcessor."""

    def test_initialization(self, data_processor, mock_config):
        """Test that DataProcessor initializes correctly."""
        assert data_processor.logger is not None
        assert data_processor.project_dir is not None

    def test_project_dir_construction(self, mock_config, mock_logger):
        """Test that project_dir is constructed correctly."""
        processor = DataProcessor(mock_config, mock_logger)
        expected_path = Path(mock_config['SYMFLUENCE_DATA_DIR']) / f"domain_{mock_config['DOMAIN_NAME']}"
        assert processor.project_dir == expected_path


class TestResultsFileLoading:
    """Test results file loading methods."""

    def test_read_results_file_success(self, data_processor):
        """Test successful reading of results file."""
        if hasattr(data_processor, 'read_results_file'):
            with tempfile.TemporaryDirectory() as tmpdir:
                data_processor.project_dir = Path(tmpdir)
                results_dir = Path(tmpdir) / "results"
                results_dir.mkdir()

                # Create results CSV with proper experiment ID from config
                exp_id = data_processor.config.get('EXPERIMENT_ID', 'test_exp')
                dates = pd.date_range('2020-01-01', periods=100, freq='D')
                df = pd.DataFrame({
                    'obs_discharge': np.random.random(100) * 10,
                    'SUMMA_discharge': np.random.random(100) * 10,
                }, index=dates)
                df.to_csv(results_dir / f"{exp_id}_results.csv")

                result = data_processor.read_results_file()
                assert isinstance(result, pd.DataFrame)

    def test_read_results_file_not_found(self, data_processor, mock_logger):
        """Test reading nonexistent results file."""
        if hasattr(data_processor, 'read_results_file'):
            with tempfile.TemporaryDirectory() as tmpdir:
                data_processor.project_dir = Path(tmpdir)
                # Don't create the results file - test that FileNotFoundError is raised

                with pytest.raises(FileNotFoundError):
                    data_processor.read_results_file()
