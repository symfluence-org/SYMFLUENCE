from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.summa.runner import SummaRunner
from symfluence.project.model_manager import ModelManager


@pytest.mark.parametrize('mode', ['serial', 'parallel', 'point'])
@pytest.mark.parametrize('success', [False, True])
def test_summa_failure_raises_in_every_mode(mode, success):
    output = Path('/tmp/summa-output') if success else None
    runner = SimpleNamespace(
        domain_definition_method='point' if mode == 'point' else 'semidistributed',
        _get_config_value=lambda *args, **kwargs: mode == 'parallel',
        run=MagicMock(return_value=output),
        run_summa_point=MagicMock(return_value=output),
        run_parallel_summa=MagicMock(return_value=output),
    )
    if success:
        assert SummaRunner.run_summa(runner) == output
    else:
        with pytest.raises(ModelExecutionError, match='routing is blocked'):
            SummaRunner.run_summa(runner)


def test_failed_summa_prevents_downstream_runner():
    summa = MagicMock()
    summa.run_summa.side_effect = ModelExecutionError('SUMMA timeout')
    routing_class = MagicMock()
    manager = SimpleNamespace(config={}, logger=MagicMock(), reporting_manager=None)
    with patch('symfluence.project.model_manager.R.runners.get',
               side_effect=[MagicMock(return_value=summa), routing_class]), \
         patch('symfluence.project.model_manager.R.runners.meta',
               return_value={'runner_method': 'run_summa'}):
        with pytest.raises(ModelExecutionError, match='SUMMA timeout'):
            ModelManager._run_sequential(manager, ['SUMMA', 'MIZUROUTE'])
    routing_class.assert_not_called()
