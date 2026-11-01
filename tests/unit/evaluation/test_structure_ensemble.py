#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from symfluence.core.modeling.structure_ensemble import BaseStructureEnsembleAnalyzer


class MockStructureAnalyzer(BaseStructureEnsembleAnalyzer):
    """Concrete implementation for testing the base class."""

    def _initialize_decision_options(self):
        return {'OPTION1': ['val1', 'val2'], 'OPTION2': ['a', 'b']}

    def _initialize_output_folder(self):
        return self.project_dir / "output"

    def _initialize_master_file(self):
        return self.project_dir / "master.csv"

    def update_model_decisions(self, combination):
        pass

    def run_model(self):
        pass

    def calculate_performance_metrics(self):
        return {'kge': 0.8, 'nse': 0.7, 'mae': 1.0, 'rmse': 1.2, 'kgep': 0.75}

def test_generate_combinations(mock_config, mock_logger):
    analyzer = MockStructureAnalyzer(mock_config, mock_logger)
    combinations = analyzer.generate_combinations()

    assert len(combinations) == 4
    assert ('val1', 'a') in combinations
    assert ('val1', 'b') in combinations
    assert ('val2', 'a') in combinations
    assert ('val2', 'b') in combinations


@pytest.fixture
def mock_config():
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/data',
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp'
    }

@patch('pathlib.Path.mkdir')
@patch('builtins.open', new_callable=mock_open)
def test_run_analysis(mock_file, mock_mkdir, mock_config, mock_logger):
    analyzer = MockStructureAnalyzer(mock_config, mock_logger)

    # Mocking combinations and other methods
    analyzer.generate_combinations = MagicMock(return_value=[('val1', 'a'), ('val2', 'b')])
    analyzer.run_model = MagicMock()
    analyzer.update_model_decisions = MagicMock()

    results_file = analyzer.run_analysis()

    assert results_file == Path('/tmp/data/domain_test_domain/master.csv')
    assert analyzer.run_model.call_count == 2
    assert analyzer.update_model_decisions.call_count == 2

    # Verify header and rows written
    handle = mock_file()
    # Initial write (header) + 2 iteration writes
    assert handle.write.call_count >= 3

def test_analyze_results(mock_config, mock_logger, tmp_path):
    # Create a dummy results file
    results_file = tmp_path / "results.csv"
    df = pd.DataFrame({
        'Iteration': [1, 2],
        'OPTION1': ['val1', 'val2'],
        'OPTION2': ['a', 'b'],
        'kge': [0.5, 0.8],
        'nse': [0.4, 0.7],
        'mae': [2.0, 1.0],
        'rmse': [3.0, 1.5],
        'kgep': [0.45, 0.75]
    })
    df.to_csv(results_file, index=False)

    analyzer = MockStructureAnalyzer(mock_config, mock_logger)
    analyzer.master_file = results_file

    best = analyzer.analyze_results(results_file)

    assert best['kge']['score'] == 0.8
    assert best['kge']['combination']['OPTION1'] == 'val2'
    assert best['mae']['score'] == 1.0
    assert best['mae']['combination']['OPTION1'] == 'val2'
