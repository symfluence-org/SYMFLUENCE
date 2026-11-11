"""SLURM job status: only COMPLETED on every task is success."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from symfluence.core.modeling.execution.model_executor import ExecutionMode, SlurmJobConfig
from symfluence.core.modeling.mixins import slurm_execution as se


class _Runner(se.SlurmExecutionMixin):
    logger = logging.getLogger('test_slurm')

    def is_slurm_available(self) -> bool:
        return True

    def execute_subprocess(self, *args, **kwargs):  # pragma: no cover - not reached
        raise AssertionError('local execution must not be used')


def _fake_slurm(monkeypatch, sacct_outputs, scontrol='', squeue=''):
    """Route sbatch/squeue/sacct/scontrol/scancel through canned outputs."""
    calls = []
    outputs = list(sacct_outputs)

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        tool = cmd[0]
        if tool == 'sbatch':
            out = 'Submitted batch job 4242\n'
        elif tool == 'squeue':
            out = squeue
        elif tool == 'sacct':
            out = outputs.pop(0) if len(outputs) > 1 else outputs[0]
        elif tool == 'scontrol':
            out = scontrol
        else:
            out = ''
        return SimpleNamespace(stdout=out, stderr='', returncode=0)

    monkeypatch.setattr(se, 'run_subprocess', _run)
    monkeypatch.setattr(se.time, 'sleep', lambda _s: None)
    return calls


@pytest.mark.parametrize('sacct,expected', [
    ('CANCELLED by 1000\n', 'CANCELLED'),
    ('OUT_OF_MEMORY\n', 'OUT_OF_MEMORY'),
    ('PREEMPTED\n', 'PREEMPTED'),
    ('COMPLETED\nCOMPLETED\nFAILED\nCOMPLETED\n', 'FAILED'),
    ('COMPLETED\nTIMEOUT\n', 'TIMEOUT'),
    ('BOOT_FAIL\n', 'BOOT_FAIL'),
])
def test_any_non_completed_task_fails_the_job(monkeypatch, sacct, expected):
    _fake_slurm(monkeypatch, [sacct])
    result = _Runner().monitor_slurm_job('4242', poll_interval=0, max_wait_time=60)
    assert result.success is False
    assert result.metadata['slurm_status'] == expected


def test_all_tasks_completed_is_success_and_uses_allocation_rows_only(monkeypatch):
    calls = _fake_slurm(monkeypatch, ['COMPLETED\nCOMPLETED\nCOMPLETED\n'])
    result = _Runner().monitor_slurm_job('4242', poll_interval=0, max_wait_time=60)
    assert result.success is True
    sacct_call = next(c for c in calls if c[0] == 'sacct')
    assert '-X' in sacct_call


def test_empty_accounting_is_retried_then_succeeds(monkeypatch):
    calls = _fake_slurm(monkeypatch, ['', '', 'RUNNING\n', 'COMPLETED\n'])
    result = _Runner().monitor_slurm_job('4242', poll_interval=0, max_wait_time=60)
    assert result.success is True
    assert sum(1 for c in calls if c[0] == 'sacct') == 4


def test_accounting_that_never_settles_is_a_failure_not_success(monkeypatch):
    calls = _fake_slurm(monkeypatch, [''])
    result = _Runner().monitor_slurm_job('4242', poll_interval=0, max_wait_time=60)
    assert result.success is False
    assert 'no final state' in result.error_message
    assert sum(1 for c in calls if c[0] == 'sacct') == se.SLURM_STATUS_RETRIES + 1


def test_scontrol_is_used_when_sacct_has_no_record(monkeypatch):
    _fake_slurm(
        monkeypatch, [''],
        scontrol='JobId=4242 ArrayTaskId=0 JobState=COMPLETED Reason=None\n'
                 'JobId=4243 ArrayTaskId=1 JobState=OUT_OF_MEMORY Reason=OOM\n',
    )
    result = _Runner().monitor_slurm_job('4242', poll_interval=0, max_wait_time=60)
    assert result.success is False
    assert result.metadata['slurm_status'] == 'OUT_OF_MEMORY'


def _clock(monkeypatch, step=600):
    """Each poll (time.sleep) advances the clock by *step* seconds."""
    now = {'t': 0.0}
    monkeypatch.setattr(se.time, 'time', lambda: now['t'])
    monkeypatch.setattr(se.time, 'sleep', lambda _s: now.__setitem__('t', now['t'] + step))
    return now


def _queue_script(monkeypatch, states):
    """squeue answers with successive entries of *states* ('' = left the queue)."""
    calls = []
    remaining = list(states)

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        out = ''
        if cmd[0] == 'squeue':
            out = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        elif cmd[0] == 'sacct':
            out = 'COMPLETED\n'
        return SimpleNamespace(stdout=out, stderr='', returncode=0)

    monkeypatch.setattr(se, 'run_subprocess', _run)
    return calls


def test_time_in_the_queue_never_triggers_a_cancel(monkeypatch):
    """10 h PENDING, then a normal 2 h run under a 3 h limit: no scancel."""
    _clock(monkeypatch, step=600)
    states = ['PENDING|3:00:00\n'] * 60 + ['RUNNING|3:00:00\n'] * 12 + ['']
    calls = _queue_script(monkeypatch, states)
    result = _Runner().monitor_slurm_job('4242', poll_interval=60, max_wait_time=3600)
    assert result.success is True
    assert not any(c[0] == 'scancel' for c in calls)


def test_running_job_is_not_cancelled_before_its_own_time_limit(monkeypatch):
    """max_wait_time=3600 must not cut short a job allowed 3 h."""
    now = _clock(monkeypatch, step=600)
    calls = _queue_script(monkeypatch, ['RUNNING|3:00:00\n'])
    result = _Runner().monitor_slurm_job('4242', poll_interval=60, max_wait_time=3600)
    assert result.success is False
    assert ['scancel', '4242'] in calls
    # cancelled only after limit + margin of run time, not after the fixed hour
    assert now['t'] >= 3 * 3600 + se.SLURM_TIME_LIMIT_MARGIN_S


def test_array_with_tasks_still_pending_is_not_cancelled(monkeypatch):
    """While any task waits, the backstop clock does not run."""
    _clock(monkeypatch, step=3600)
    mixed = 'RUNNING|1:00:00\nPENDING|1:00:00\n'
    calls = _queue_script(monkeypatch, [mixed] * 30 + [''])
    result = _Runner().monitor_slurm_job('4242', poll_interval=60, max_wait_time=60)
    assert result.success is True
    assert not any(c[0] == 'scancel' for c in calls)


def test_script_time_limit_is_used_until_squeue_reports_one(monkeypatch, tmp_path):
    now = _clock(monkeypatch, step=600)
    script = tmp_path / 'x.sh'
    script.write_text('#!/bin/bash\n#SBATCH --job-name=j\n#SBATCH --time=02:00:00\n', encoding='utf-8')
    assert _Runner._script_time_limit(script) == '02:00:00'

    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        out = {'sbatch': 'Submitted batch job 4242\n', 'squeue': 'RUNNING\n'}.get(cmd[0], '')
        return SimpleNamespace(stdout=out, stderr='', returncode=0)

    monkeypatch.setattr(se, 'run_subprocess', _run)
    result = _Runner().submit_slurm_job(script, wait=True, poll_interval=60, max_wait_time=60)
    assert result.success is False
    assert ['scancel', '4242'] in calls
    assert now['t'] >= 2 * 3600 + se.SLURM_TIME_LIMIT_MARGIN_S


@pytest.mark.parametrize('spec,seconds', [
    ('03:00:00', 10800), ('3:00:00', 10800), ('30', 1800), ('30:15', 1815),
    ('1-00:00:00', 86400), ('2-12', 216000), ('1-01:30', 91800),
    ('UNLIMITED', None), ('NOT_SET', None), ('Partition_Limit', None), ('', None), (None, None),
])
def test_parse_slurm_time_limit(spec, seconds):
    assert se.parse_slurm_time_limit(spec) == seconds


def test_summa_does_not_pass_a_fixed_monitor_timeout():
    root = Path(se.__file__).resolve().parents[3]
    text = (root / 'models/summa/runner.py').read_text(encoding='utf-8')
    assert 'max_wait_time=3600' not in text


def test_unmonitored_submit_is_not_a_finished_job(monkeypatch, tmp_path):
    _fake_slurm(monkeypatch, ['COMPLETED\n'])
    runner = _Runner()
    result = runner.submit_slurm_job(tmp_path / 'x.sh', wait=False)
    assert result.success is True
    assert runner.slurm_job_finished(result) is False

    finished = runner.submit_slurm_job(tmp_path / 'x.sh', wait=True, poll_interval=0)
    assert runner.slurm_job_finished(finished) is True


def test_template_runner_does_not_merge_after_bare_sbatch():
    """Both SLURM callers gate merging on slurm_job_finished()."""
    root = Path(se.__file__).resolve().parents[3]
    for rel in ('core/modeling/templates/model_template.py', 'models/summa/runner.py'):
        text = (root / rel).read_text(encoding='utf-8')
        before_merge = text.split('return self._merge_parallel_outputs()')[0]
        assert 'slurm_job_finished(result)' in before_merge, rel


def test_execute_in_mode_writes_one_command_line_with_cwd_and_env(monkeypatch, tmp_path):
    _fake_slurm(monkeypatch, ['COMPLETED\n'])
    runner = _Runner()
    work = tmp_path / 'work dir'
    work.mkdir()
    result = runner.execute_in_mode(
        ExecutionMode.SLURM,
        ['/opt/summa.exe', '-m', '/p/file manager.txt'],
        tmp_path / 'run.log',
        slurm_config=SlurmJobConfig(job_name='j'),
        wait_for_slurm=False,
        cwd=work,
        env={'OMP_NUM_THREADS': '4', 'NOTE': 'a b'},
    )
    assert result.success
    lines = (tmp_path / 'j.sh').read_text(encoding='utf-8').splitlines()
    assert "/opt/summa.exe -m '/p/file manager.txt'" in lines
    assert '-m' not in lines  # argv elements are not separate script lines
    assert 'export OMP_NUM_THREADS=4' in lines
    assert "export NOTE='a b'" in lines
    cd_line = next(line for line in lines if line.startswith('cd '))
    assert str(work) in cd_line
    assert lines.index(cd_line) < lines.index("/opt/summa.exe -m '/p/file manager.txt'")


def test_sbatch_failure_is_reported(monkeypatch, tmp_path):
    def _run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr='bad partition')

    monkeypatch.setattr(se, 'run_subprocess', _run)
    result = _Runner().submit_slurm_job(tmp_path / 'x.sh')
    assert result.success is False
