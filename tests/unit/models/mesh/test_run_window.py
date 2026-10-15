from __future__ import annotations

import logging

import pandas as pd

from symfluence.models.mesh.preprocessing.run_options_builder import RunOptionsConfigBuilder


def test_inclusive_forcing_records_become_exclusive_mesh_stop(tmp_path):
    p=tmp_path/'run.ini'
    p.write_text('BASINFORCINGFLAG fname=MESH_forcing\n2009 001 1 0\n2023 273 23 0\n')
    b=object.__new__(RunOptionsConfigBuilder)
    b._path=p;b.logger=logging.getLogger('test')
    b._get_config_value=lambda *args,**kwargs:3600
    b.update_for_safe_forcing(pd.Timestamp('2009-01-01'),pd.Timestamp('2023-09-30 23:00'),0)
    lines=p.read_text().splitlines()
    assert lines[-2].split()==['2009','001','0','0']
    assert lines[-1].split()==['2023','274','0','0']


def test_runner_counts_exclusive_midnight_stop(tmp_path):
    from symfluence.models.mesh.runner import MESHRunner
    p=tmp_path/'MESH_input_run_options.ini'
    p.write_text('2009 001 0 0\n2023 274 0 0\n')
    runner=object.__new__(MESHRunner)
    runner.forcing_mesh_path=tmp_path
    assert runner._get_expected_days()==5386
