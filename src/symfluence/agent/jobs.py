# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Background workflow jobs for the agent's MCP tools.

``run_workflow_step`` blocks the MCP connection for the whole step — fine for a
quick preprocessing step, hopeless for an hours-long calibration a chat session
wants to watch. This module gives the server a start/poll/cancel job pattern:

- Jobs run detached (``start_new_session``; a new process group on Windows)
  under a tiny Python wrapper that records the exit code to a file, so
  completion is knowable even after the MCP server (or the whole TUI) restarts.
- Each job persists a JSON record under ``agent_cache_root()/jobs/`` next to
  its log file; ``get_job_status`` needs nothing in memory.
- Cancellation signals the job's process group: TERM first, KILL if ignored.
  Windows has no graceful console signal for a detached tree, so cancellation
  goes straight to ``taskkill /T /F`` there.

Everything here is stdlib-only, consistent with the hand-rolled MCP server.
"""
from __future__ import annotations

import json
import os
import signal

# Security rationale: argv-built commands, no shell
import subprocess  # nosec B404
import sys
import time
import uuid
from pathlib import Path

# The wrapper runs the real command and durably records its exit code.
# argv[1] = exit-code file, argv[2:] = command.
_WRAPPER = (
    "import subprocess, sys\n"
    "code = subprocess.call(sys.argv[2:])\n"
    "open(sys.argv[1], 'w').write(str(code))\n"
    "sys.exit(code)\n"
)

RUNNING = 'running'
SUCCEEDED = 'succeeded'
FAILED = 'failed'
CANCELLED = 'cancelled'
UNKNOWN = 'unknown'


def jobs_root() -> Path:
    """Directory holding job records, logs, and exit-code files."""
    from symfluence.resources import agent_cache_root
    return agent_cache_root() / 'jobs'


def _paths(job_id: str) -> tuple[Path, Path, Path]:
    root = jobs_root()
    return root / f'{job_id}.json', root / f'{job_id}.log', root / f'{job_id}.exit'


def _read_record(job_id: str) -> dict:
    record_path, _, _ = _paths(job_id)
    if not record_path.is_file():
        raise ValueError(f"Unknown job: {job_id!r}")
    return json.loads(record_path.read_text(encoding='utf-8'))


def _write_record(record: dict) -> None:
    record_path, _, _ = _paths(record['job_id'])
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')


def _pid_alive(pid: int) -> bool:
    if sys.platform == 'win32':
        # No kill-0 probe on Windows (os.kill would TerminateProcess!); ask
        # the kernel for the process's exit status instead.
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    # Reap the wrapper if it is a finished child of this process — a zombie
    # would otherwise answer the kill-0 probe as "alive" forever.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass  # not our child (e.g. the server restarted); the probe decides
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_job(argv: list[str], description: str = '') -> dict:
    """Start ``argv`` as a detached background job and persist its record.

    Returns the job record: ``{job_id, pid, log_path, ...}``.
    """
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    record_path, log_path, exit_path = _paths(job_id)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    detach: dict = (
        {'creationflags': (subprocess.CREATE_NEW_PROCESS_GROUP
                           | subprocess.CREATE_NO_WINDOW)}
        if sys.platform == 'win32' else {'start_new_session': True}
    )
    with open(log_path, 'wb') as log:
        # Security rationale: argv assembled by the caller, no shell
        proc = subprocess.Popen(  # nosec B603
            [sys.executable, '-c', _WRAPPER, str(exit_path), *argv],
            stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **detach,
        )

    record = {
        'job_id': job_id,
        'pid': proc.pid,
        'argv': argv,
        'description': description,
        'log_path': str(log_path),
        'started_at': time.time(),
        'cancelled': False,
    }
    _write_record(record)
    return record


def job_state(job_id: str) -> dict:
    """Resolve a job's current state from its record, exit file, and pid."""
    record = _read_record(job_id)
    _, log_path, exit_path = _paths(job_id)

    exit_code: int | None = None
    if exit_path.is_file():
        try:
            exit_code = int(exit_path.read_text(encoding='utf-8').strip())
        except ValueError:
            exit_code = None

    if record.get('cancelled'):
        state = CANCELLED
    elif exit_code is not None:
        state = SUCCEEDED if exit_code == 0 else FAILED
    elif _pid_alive(record['pid']):
        state = RUNNING
    else:
        # Died without writing an exit code (e.g. killed externally).
        state = UNKNOWN

    return {
        'job_id': job_id,
        'state': state,
        'exit_code': exit_code,
        'pid': record['pid'],
        'description': record.get('description', ''),
        'runtime_s': round(time.time() - record['started_at'], 1),
        'log_path': str(log_path),
    }


def tail_log(job_id: str, lines: int = 50) -> list[str]:
    """The last ``lines`` lines of a job's log (empty if none yet)."""
    _, log_path, _ = _paths(job_id)
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding='utf-8', errors='replace')
    return text.splitlines()[-max(0, lines):]


def cancel_job(job_id: str, grace_s: float = 5.0) -> dict:
    """Cancel a running job: TERM its process group, then KILL if it lingers."""
    record = _read_record(job_id)
    pid = record['pid']

    # A job that already wrote its exit code has finished: its pid may since
    # have been reused by an unrelated process (group), so nothing is
    # signalled, and the recorded outcome is kept rather than overwritten
    # with CANCELLED.
    _, _, exit_path = _paths(job_id)
    if exit_path.is_file():
        return job_state(job_id)

    if _pid_alive(pid):
        if sys.platform == 'win32':
            # No graceful console signal reaches a detached tree on Windows;
            # kill the wrapper and everything under it in one shot.
            # Security rationale: fixed argv, no shell.
            subprocess.run(  # nosec B603 B607
                ['taskkill', '/PID', str(pid), '/T', '/F'],
                capture_output=True, check=False,
            )
            deadline = time.time() + grace_s
            while time.time() < deadline and _pid_alive(pid):
                time.sleep(0.1)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)  # wrapper is the session/group leader
            except (ProcessLookupError, PermissionError):
                pass
            deadline = time.time() + grace_s
            while time.time() < deadline and _pid_alive(pid):
                time.sleep(0.1)
            if _pid_alive(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    record['cancelled'] = True
    _write_record(record)
    return job_state(job_id)


def list_jobs() -> list[dict]:
    """States of every recorded job, newest first."""
    root = jobs_root()
    if not root.is_dir():
        return []
    states = []
    for record_path in sorted(root.glob('*.json'), reverse=True):
        try:
            states.append(job_state(record_path.stem))
        except (ValueError, OSError):
            continue
    return states
