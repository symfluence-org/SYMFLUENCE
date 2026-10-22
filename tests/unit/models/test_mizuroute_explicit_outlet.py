"""A configured gauge reach must override flow-magnitude heuristics."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator


def test_explicit_reach_is_selected_and_missing_reach_rejected(tmp_path):
    path = tmp_path / 'flow.nc'
    xr.Dataset({'reachID': ('seg', [65, 7]), 'IRFroutedRunoff': (('time', 'seg'), [[2., 100.], [3., 200.]])}, coords={'time': pd.date_range('2013-01-01', periods=2)}).to_netcdf(path)
    evaluator = StreamflowEvaluator.__new__(StreamflowEvaluator)
    evaluator.config = SimpleNamespace(evaluation=SimpleNamespace(sim_reach_id=65))
    assert evaluator._extract_mizuroute_streamflow(path).tolist() == [2., 3.]
    evaluator.config.evaluation.sim_reach_id = 999
    with pytest.raises(EvaluationError, match='missing'):
        evaluator._extract_mizuroute_streamflow(path)
