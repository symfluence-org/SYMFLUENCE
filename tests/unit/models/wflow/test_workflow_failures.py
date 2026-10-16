from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from symfluence.models.wflow.preprocessor import WflowPreProcessor
from symfluence.models.wflow.runner import WflowRunner


def test_preprocessing_error_is_not_swallowed():
    pre = object.__new__(WflowPreProcessor)
    pre.logger = logging.getLogger('test')
    pre.spatial_mode = 'distributed'
    with patch.object(pre, '_create_directory_structure'), patch(
        'symfluence.models.wflow.distributed.prepare', side_effect=ValueError('bad geometry')
    ), pytest.raises(ValueError, match='bad geometry'):
        pre.run_preprocessing()


def test_native_failure_is_not_reported_as_completed():
    from symfluence.core.exceptions import ModelExecutionError
    runner = object.__new__(WflowRunner)
    with patch('symfluence.core.modeling.base.BaseModelRunner.run', return_value=None), pytest.raises(ModelExecutionError):
        runner.run()
