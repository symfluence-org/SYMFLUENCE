# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow-CLM: bounded rerun after a segmentation fault, stdin closed for mpirun."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

import pytest

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.clmparflow import runner as runner_mod
from symfluence.models.clmparflow.basin import case as case_mod
from symfluence.models.clmparflow.segfault_retry import is_segfault, run_with_segfault_retry

pytestmark = pytest.mark.unit

SEGV_MPI = ('mpirun noticed that process rank 0 with PID 0 on node n exited on signal 11 '
            '(Segmentation fault).\n')


def _done(rc: int, out: str = '') -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=['parflow'], returncode=rc, stdout=out, stderr='')


class _Launcher:
    """Returns the queued results in order and leaves partial output like a crashed ParFlow."""

    def __init__(self, run_dir: Path, results: List[subprocess.CompletedProcess]):
        self.run_dir, self.results, self.calls = run_dir, list(results), 0
        self.seen_partial: List[bool] = []

    def __call__(self) -> subprocess.CompletedProcess:
        self.seen_partial.append((self.run_dir / 'case.out.press.00000.pfb').exists())
        self.calls += 1
        (self.run_dir / 'case.out.press.00000.pfb').write_bytes(b'partial')
        return self.results.pop(0)


@pytest.mark.parametrize('rc,out,expected', [
    (139, '', True),
    (-11, '', True),
    (1, SEGV_MPI, True),
    (0, SEGV_MPI, False),       # a clean exit is never a segfault
    (1, 'Time step failed', False),
    (134, 'signal 6 (Aborted)', False),
])
def test_is_segfault(rc, out, expected):
    assert is_segfault(rc, out) is expected


def test_segfault_then_success_reruns_once(tmp_path):
    launch = _Launcher(tmp_path, [_done(139, SEGV_MPI), _done(0, 'Problem solved')])
    warnings: List[str] = []

    r = run_with_segfault_retry(launch, tmp_path, 'case', retries=2, log=warnings.append)

    assert r.returncode == 0 and launch.calls == 2
    assert launch.seen_partial == [False, False]        # partial output removed before the rerun
    assert (tmp_path / 'case.segv1.log').read_text(encoding='utf-8') == SEGV_MPI
    assert len(warnings) == 1 and 'attempt 1 of 3' in warnings[0]


def test_retries_are_bounded(tmp_path):
    launch = _Launcher(tmp_path, [_done(139)] * 5)
    r = run_with_segfault_retry(launch, tmp_path, 'case', retries=2, log=lambda m: None)
    assert r.returncode == 139 and launch.calls == 3
    assert sorted(p.name for p in tmp_path.glob('*.segv*.log')) == ['case.segv1.log', 'case.segv2.log']


def test_other_failures_and_zero_retries_are_not_rerun(tmp_path):
    launch = _Launcher(tmp_path, [_done(1, 'Time step failed')])
    assert run_with_segfault_retry(launch, tmp_path, 'case', retries=2).returncode == 1
    assert launch.calls == 1
    launch = _Launcher(tmp_path, [_done(139)])
    assert run_with_segfault_retry(launch, tmp_path, 'case', retries=0).returncode == 139
    assert launch.calls == 1


def test_basin_run_parflow_retries_and_closes_stdin(tmp_path, monkeypatch):
    calls = []
    results = [_done(139, SEGV_MPI), _done(0, 'Problem solved')]

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)
        return results.pop(0)

    monkeypatch.setattr(case_mod.subprocess, 'run', fake_run)
    rc, out = case_mod.run_parflow(tmp_path, 'seg', tmp_path / 'pf', ranks=2, mpirun='mpirun', segv_retries=1)

    assert rc == 0 and 'Problem solved' in out and len(calls) == 2
    assert all(k['stdin'] is subprocess.DEVNULL for k in calls)


def test_chain_settings_carry_segv_retries(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.models.clmparflow.basin.settings import chain_settings

    c = SymfluenceConfig(**_config(tmp_path), CLMPARFLOW_SEGV_RETRIES=4)
    assert chain_settings(c.model.clmparflow, '2012-10-01 00:00', '2023-09-30 23:00', 1).segv_retries == 4


def _config(tmp_path: Path) -> dict:
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'), 'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'e', 'EXPERIMENT_TIME_START': '2012-10-01 00:00',
        'EXPERIMENT_TIME_END': '2023-09-30 23:00', 'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs', 'HYDROLOGICAL_MODEL': 'CLMPARFLOW', 'FORCING_DATASET': 'ERA5',
    }


@pytest.mark.parametrize('retries,results,ok', [
    (2, [139, 0], True),
    (1, [139, 139], False),
])
def test_runner_retries_segfaults(tmp_path, monkeypatch, retries, results, ok):
    from symfluence.core.config.models import SymfluenceConfig

    config = SymfluenceConfig(**_config(tmp_path), CLMPARFLOW_SEGV_RETRIES=retries)
    runner = runner_mod.CLMParFlowRunner(config, logging.getLogger(__name__))
    exe = tmp_path / 'pf' / 'bin' / 'parflow'
    monkeypatch.setattr(runner, '_get_parflow_executable', lambda: exe)
    monkeypatch.setattr(runner, '_get_runname', lambda: 'case')
    monkeypatch.setattr(runner, '_setup_sim_directory', lambda sim_dir: None)
    monkeypatch.setattr(runner, '_verify_output', lambda runname: None)
    queue = [_done(rc, SEGV_MPI if rc else 'Problem solved') for rc in results]
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)
        return queue.pop(0)

    monkeypatch.setattr(runner_mod, 'run_subprocess', fake_run)
    sim_dir = tmp_path / 'sim'
    if ok:
        assert runner.run_clmparflow(sim_dir=sim_dir) == sim_dir
    else:
        with pytest.raises(ModelExecutionError):
            runner.run_clmparflow(sim_dir=sim_dir)
    assert len(calls) == len(results)
    assert all(k['stdin'] is subprocess.DEVNULL for k in calls)
