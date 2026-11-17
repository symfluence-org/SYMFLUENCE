"""MODFLOWProcessComponent: the graph's recharge must reach MODFLOW's RCH input."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip('dcoupler')

from symfluence.coupling.adapters.process_adapters import MODFLOWProcessComponent  # noqa: E402
from symfluence.models.modflow.runner import MODFLOWRunner  # noqa: E402


def _component(settings_dir, config=None):
    runner = MODFLOWRunner.__new__(MODFLOWRunner)
    runner.settings_dir = settings_dir
    runner.config_dict = {}
    comp = MODFLOWProcessComponent('modflow', config=config or {})
    comp._runner = runner
    return comp, runner


def test_recharge_written_to_rch_and_used_by_execute(tmp_path):
    (tmp_path / 'gwf.tdis').write_text('BEGIN DIMENSIONS\n  NPER 2\nEND DIMENSIONS\n'
                                       'BEGIN PERIODDATA\n  2.0 1 1.0\n  2.0 1 1.0\nEND PERIODDATA\n')
    comp, runner = _component(tmp_path)
    comp.write_inputs({'recharge': torch.tensor([[1e-3], [3e-3], [2e-3], [4e-3]])}, tmp_path / 'work')

    assert not (tmp_path / 'work' / 'recharge_tas6.txt').exists()
    text = (tmp_path / 'gwf.rch').read_text()
    vals = [float(l.split()[1]) for l in text.splitlines() if l.strip().startswith('CONSTANT')]
    np.testing.assert_allclose(vals, [2e-3, 3e-3])

    with patch.object(MODFLOWRunner, 'run_modflow', return_value=tmp_path) as run:
        assert comp.execute(tmp_path) == 0
    assert run.call_args.kwargs['prepare_recharge'] is False


def test_execute_without_graph_recharge_prepares_from_land_output(tmp_path):
    comp, _ = _component(tmp_path)
    comp.write_inputs({}, tmp_path)
    with patch.object(MODFLOWRunner, 'run_modflow', return_value=tmp_path) as run:
        assert comp.execute(tmp_path) == 0
    assert run.call_args.kwargs['prepare_recharge'] is True


def test_basin_mode_rejects_graph_recharge(tmp_path):
    comp, _ = _component(tmp_path, {'MODFLOW_GRID_MODE': 'basin'})
    with pytest.raises(NotImplementedError):
        comp.write_inputs({'recharge': torch.tensor([1e-3])}, tmp_path)
