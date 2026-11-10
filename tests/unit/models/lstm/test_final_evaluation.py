"""LSTM run writes the standard final_evaluation.json / best_params.json (#342)."""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from symfluence.core.config.models import SymfluenceConfig


def test_run_lstm_writes_final_evaluation(tmp_path):
    from symfluence.models.lstm.runner import LSTMRunner

    config = SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'lstm_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'LSTM',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'CALIBRATION_PERIOD': '2020-01-01, 2020-06-30',
        'EVALUATION_PERIOD': '2020-07-01, 2020-12-31',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
        'LSTM_LOAD': True,  # skip training; the checkpoint load is mocked
    })
    domain_dir = tmp_path / 'data' / 'domain_test_domain'
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    rng = np.random.default_rng(0)
    obs = 20 + 10 * np.sin(np.arange(len(dates)) / 30) + rng.normal(0, 1, len(dates))
    obs_dir = domain_dir / 'data' / 'observations' / 'streamflow' / 'preprocessed'
    obs_dir.mkdir(parents=True)
    pd.DataFrame({'discharge_cms': obs}, index=pd.Index(dates, name='datetime')).to_csv(
        obs_dir / 'test_domain_streamflow_processed.csv')

    runner = LSTMRunner(config, logging.getLogger('lstm-final-eval-test'))
    sim = pd.DataFrame({'predicted_streamflow': obs * 0.95 + 1.0}, index=dates)

    def _create_model(*args, **kwargs):
        runner.model = MagicMock()

    with patch.object(runner.preprocessor, 'load_data', return_value=(MagicMock(), MagicMock(), pd.DataFrame())), \
         patch.object(runner.preprocessor, 'set_scalers'), \
         patch.object(runner.preprocessor, 'process_data',
                      return_value=(MagicMock(shape=(1, 1, 3)), None, dates, None, [1])), \
         patch.object(runner, '_load_model_checkpoint', return_value={
             'feature_scaler': None, 'target_scaler': None, 'output_size': 1,
             'target_names': ['streamflow'], 'model_state_dict': {}}), \
         patch.object(runner, '_create_model_instance', side_effect=_create_model), \
         patch.object(runner, '_simulate', return_value=sim), \
         patch.object(runner, 'requires_routing', return_value=False):
        runner.run_lstm()

    results_dir = domain_dir / 'optimization' / 'LSTM' / 'dds_lstm_test'
    final = json.loads((results_dir / 'lstm_test_dds_final_evaluation.json').read_text())
    assert final['experiment_id'] == 'lstm_test'
    assert final['calibration_metrics']['KGE'] > 0.8
    assert final['evaluation_metrics']['KGE'] > 0.8
    best = json.loads((results_dir / 'lstm_test_dds_best_params.json').read_text())
    assert best['model'] == 'LSTM'
    assert best['best_score'] == final['calibration_metrics']['KGE']
    assert (results_dir / 'final_evaluation').is_dir()
