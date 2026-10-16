from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.models.wflow.calibration.worker import WflowWorker


@pytest.mark.parametrize('csv_path', ['output.csv', '/old/run/output.csv'])
def test_final_run_redirects_paths_without_apply_parameters(tmp_path, csv_path):
    settings=tmp_path/'settings';settings.mkdir()
    exe=tmp_path/'wflow_cli';exe.touch()
    toml=settings/'wflow_sbm.toml'
    toml.write_text(f'dir_output = "/old/run"\n[input]\npath_static = "/old/static.nc"\n[output.csv]\npath = "{csv_path}"\n')
    output=tmp_path/'final_evaluation'
    worker=WflowWorker({},logging.getLogger('test'))
    def native(command,**kwargs):
        import tomllib
        data=tomllib.loads(toml.read_text())
        assert Path(data['dir_output'])==output
        assert Path(data['output']['csv']['path'])==output/'output.csv'
        assert Path(command[1])==toml
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0,stderr='')
    with patch('symfluence.models.wflow.calibration.worker.run_subprocess',side_effect=native):
        assert worker.run_model({'WFLOW_INSTALL_PATH':str(tmp_path)},settings,output)


@pytest.mark.parametrize('final,end', [(False, '2016-10-01'), (True, '2023-10-01')])
def test_trial_end_and_full_final_restore(tmp_path, final, end):
    import tomllib
    from types import SimpleNamespace

    import pandas as pd
    settings = tmp_path / 'settings'
    settings.mkdir()
    (tmp_path / 'wflow_cli').touch()
    toml = settings / 'wflow_sbm.toml'
    toml.write_text('[time]\nendtime = 2016-10-01T00:00:00\n')
    config = dict(WFLOW_INSTALL_PATH=str(tmp_path), WFLOW_CALIBRATION_ONLY_PERIOD=True,
                  CALIBRATION_PERIOD='2012-10-01,2016-09-30', EXPERIMENT_TIME_END='2023-09-30 21:00')
    def native(*args, **kwargs):
        assert pd.Timestamp(tomllib.loads(toml.read_text())['time']['endtime']) == pd.Timestamp(end)
        return SimpleNamespace(returncode=0, stderr='')
    worker = WflowWorker({}, logging.getLogger('test'))
    with patch('symfluence.models.wflow.calibration.worker.run_subprocess', side_effect=native):
        assert worker.run_model(config, settings, tmp_path / 'out', final_evaluation=final)
