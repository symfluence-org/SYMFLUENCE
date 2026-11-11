"""ModelManager propagates run/postprocess failures (no silent 'success')."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.exceptions import ModelExecutionError
from symfluence.project.model_manager import ModelManager

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _manager():
    return SimpleNamespace(config={}, logger=MagicMock(), reporting_manager=None)


def test_unknown_model_raises():
    with patch('symfluence.project.model_manager.R.runners.get', return_value=None):
        with pytest.raises(ModelExecutionError, match='no runner registered: NOPE'):
            ModelManager._run_sequential(_manager(), ['NOPE'])


def test_postprocess_failure_raises_after_all_models():
    manager = _manager()
    manager._resolve_model_workflow = lambda **_: ['SUMMA', 'MIZUROUTE']
    calls = []

    def _pp(model):
        calls.append(model)
        if model == 'SUMMA':
            raise ModelExecutionError('bad output')

    manager._postprocess_model = _pp
    manager.log_baseline_performance = MagicMock()
    with pytest.raises(ModelExecutionError, match='SUMMA: bad output'):
        ModelManager.postprocess_results(manager)
    assert calls == ['SUMMA', 'MIZUROUTE']  # every model still attempted
    manager.log_baseline_performance.assert_not_called()


def test_postprocessor_exception_converted():
    manager = _manager()
    pp = MagicMock()
    pp.return_value.extract_streamflow.side_effect = KeyError('q_routed')
    with patch('symfluence.project.model_manager.R.postprocessors.get', return_value=pp):
        with pytest.raises(ModelExecutionError, match='q_routed'):
            ModelManager._postprocess_model(manager, 'SUMMA')
