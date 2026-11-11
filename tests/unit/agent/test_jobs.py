# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the background job store behind the agent's MCP tools."""
from __future__ import annotations

import sys
import time

import pytest

from symfluence.agent import jobs


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))


def _wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_job_lifecycle_success():
    record = jobs.start_job(
        [sys.executable, '-c', "print('hello from job')"], description='greet')
    assert record['job_id'] and record['pid']

    assert _wait_for(
        lambda: jobs.job_state(record['job_id'])['state'] == jobs.SUCCEEDED)
    state = jobs.job_state(record['job_id'])
    assert state['exit_code'] == 0
    assert state['description'] == 'greet'
    assert 'hello from job' in '\n'.join(jobs.tail_log(record['job_id']))


def test_job_failure_reports_exit_code():
    record = jobs.start_job([sys.executable, '-c', 'raise SystemExit(3)'])
    assert _wait_for(
        lambda: jobs.job_state(record['job_id'])['state'] == jobs.FAILED)
    assert jobs.job_state(record['job_id'])['exit_code'] == 3


def test_job_running_then_cancelled():
    record = jobs.start_job([sys.executable, '-c', 'import time; time.sleep(60)'])
    assert _wait_for(
        lambda: jobs.job_state(record['job_id'])['state'] == jobs.RUNNING)

    state = jobs.cancel_job(record['job_id'], grace_s=2.0)
    assert state['state'] == jobs.CANCELLED
    # The process group is really gone.
    assert _wait_for(lambda: not jobs._pid_alive(record['pid']))


def test_unknown_job_raises():
    with pytest.raises(ValueError, match='Unknown job'):
        jobs.job_state('nope')


def test_list_jobs_newest_first():
    first = jobs.start_job([sys.executable, '-c', 'pass'])
    time.sleep(1.1)  # job ids embed second-resolution timestamps
    second = jobs.start_job([sys.executable, '-c', 'pass'])
    listed = [j['job_id'] for j in jobs.list_jobs()]
    assert listed.index(second['job_id']) < listed.index(first['job_id'])


def test_cancel_after_finish_signals_nothing_and_keeps_the_outcome(monkeypatch):
    """A finished job's pid may have been reused: cancel must not signal it,
    and must not relabel a SUCCEEDED/FAILED job as CANCELLED."""
    record = jobs.start_job([sys.executable, '-c', 'raise SystemExit(3)'])
    job_id = record['job_id']
    assert _wait_for(lambda: jobs.job_state(job_id)['state'] == jobs.FAILED)

    def _no_signals(*_args, **_kwargs):
        raise AssertionError("a finished job must not be signalled")

    # even if some process now answers to the recorded pid
    monkeypatch.setattr(jobs, '_pid_alive', lambda pid: True)
    if sys.platform != 'win32':
        monkeypatch.setattr(jobs.os, 'killpg', _no_signals)
    monkeypatch.setattr(jobs.subprocess, 'run', _no_signals)

    state = jobs.cancel_job(job_id)

    assert state['state'] == jobs.FAILED
    assert state['exit_code'] == 3
    assert jobs.job_state(job_id)['state'] == jobs.FAILED

