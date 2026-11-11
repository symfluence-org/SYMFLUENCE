"""execute_subprocess: a timeout is a failure the caller cannot ignore."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import pytest

from symfluence.core import process_exec
from symfluence.core.modeling.mixins.subprocess_execution import SubprocessExecutionMixin


class _Runner(SubprocessExecutionMixin):
    logger = logging.getLogger('test_subprocess_timeout')


_SLEEP = [sys.executable, '-c', 'import time; time.sleep(30)']


def test_timeout_raises_when_check_is_true(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        _Runner().execute_subprocess(_SLEEP, tmp_path / 'l.log', timeout=1)


def test_timeout_returns_failed_result_when_check_is_false(tmp_path):
    result = _Runner().execute_subprocess(_SLEEP, tmp_path / 'l.log', timeout=1, check=False)
    assert result.success is False
    assert result.return_code == -1
    assert 'Timeout' in result.error_message


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(sys.platform == 'win32', reason='process groups are POSIX-only')
def test_timeout_kills_grandchildren(tmp_path):
    """A wrapper (mpirun, sh -c) must not leave its children running."""
    pid_file = tmp_path / 'grandchild.pid'
    child = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pid_file}', 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _Runner().execute_subprocess([sys.executable, '-c', child], tmp_path / 'l.log', timeout=3)

    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert not _alive(grandchild), 'grandchild survived the timeout'
    finally:
        if _alive(grandchild):
            os.kill(grandchild, 9)


@pytest.mark.skipif(sys.platform == 'win32', reason='process groups are POSIX-only')
def test_kill_group_run_reports_exit_code_and_output():
    done = process_exec.run(
        [sys.executable, '-c', 'import sys; print("hi"); sys.exit(3)'],
        capture_output=True, text=True, kill_group=True,
    )
    assert done.returncode == 3
    assert done.stdout.strip() == 'hi'
    with pytest.raises(subprocess.CalledProcessError):
        process_exec.run([sys.executable, '-c', 'raise SystemExit(2)'], check=True, kill_group=True)
