"""A preprocessor that returns False must fail the preprocessing step.

Most model preprocessors catch their own exceptions, log them and return
False. ModelManager used to ignore that return value, so the workflow reported
success with half-written model inputs.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.exceptions import ModelExecutionError
from symfluence.project.model_manager import ModelManager


def _manager(tmp_path):
    return SimpleNamespace(
        config={}, logger=MagicMock(), reporting_manager=None,
        project_forcing_dir=tmp_path,
        _resolve_model_workflow=lambda: ['MHM'],
    )


def _preprocessor(result):
    class Pre:
        def __init__(self, config, logger):
            pass

        def run_preprocessing(self):
            return result
    return Pre


@pytest.mark.parametrize('result', [True, None])
def test_successful_or_silent_preprocessor_passes(tmp_path, result):
    with patch('symfluence.project.model_manager.R.preprocessors.get',
               return_value=_preprocessor(result)):
        ModelManager.preprocess_models(_manager(tmp_path))


def test_preprocessor_returning_false_raises(tmp_path):
    with patch('symfluence.project.model_manager.R.preprocessors.get',
               return_value=_preprocessor(False)):
        with pytest.raises(ModelExecutionError, match='MHM preprocessing failed'):
            ModelManager.preprocess_models(_manager(tmp_path))
