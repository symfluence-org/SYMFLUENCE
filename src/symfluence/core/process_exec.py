# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Subprocess execution that survives Windows exits which never signal.

On Windows a process object is signaled only once its *last thread* exits.
Native model binaries here regularly terminate — handing back a real exit code
— while leaving one thread parked in a kernel wait, so the object never
signals. Measured on a wedged SUMMA worker::

    GetExitCodeProcess(38420)        -> 1     # the Fortran STOP 1
    WaitForSingleObject(38420, 5000) -> 258   # WAIT_TIMEOUT, never signals

Seven stuck processes were checked (SUMMA and FUSE, four domains): every one
had a valid exit code and none was signaled. ``Popen.wait()`` blocks on
exactly that signal, and ``Popen.poll()`` is no escape — CPython's Windows
implementation gates it on ``WaitForSingleObject(handle, 0)`` too. So the
result is already known to the OS and unreachable through the stdlib.

The cost was not subtle: a SUMMA evaluation that failed instantly with
``STOP 1`` still billed its full 7200s timeout, and a paper-reproduction group
spent 21h without finishing one NSGA-II generation.

``wait`` and ``run`` below prefer the process's exit code over its signal, so
a finished process is reported finished. The stuck thread is left behind; it
consumes no CPU and cannot be killed, and a leaked zombie is a far better
outcome than blocking the workflow for hours.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from typing import Any, List, Optional

IS_WINDOWS = sys.platform == 'win32'

#: GetExitCodeProcess returns this while a process is still running. A process
#: may also legitimately exit with 259, which is why a positive signal check
#: still decides that one case.
STILL_ACTIVE = 259

#: How often to re-check a process that has not signaled.
POLL_INTERVAL = 0.25

#: Grace period for output readers to finish once the process is known to have
#: exited. A wedged process may never release its pipe write handles, so this
#: is bounded: truncated output beats hanging forever.
_READER_GRACE = 5.0


def _terminated_exit_code(proc: subprocess.Popen) -> Optional[int]:
    """Exit code if the process has terminated, else None.

    Windows-only. Trusts ``GetExitCodeProcess`` over the process object's
    signal, which is the whole point of this module.
    """
    handle = getattr(proc, '_handle', None)
    if handle is None:
        return None
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32
    code = ctypes.wintypes.DWORD()
    if not kernel32.GetExitCodeProcess(ctypes.wintypes.HANDLE(int(handle)),
                                       ctypes.byref(code)):
        return None
    if code.value != STILL_ACTIVE:
        return code.value
    # Ambiguous: either still running, or genuinely exited with 259. Only a
    # signaled object distinguishes them, and here the signal is meaningful
    # because it can only be set by real termination.
    if kernel32.WaitForSingleObject(ctypes.wintypes.HANDLE(int(handle)), 0) == 0:
        return STILL_ACTIVE
    return None


def wait(proc: subprocess.Popen, timeout: Optional[float] = None) -> int:
    """Wait for ``proc``, treating a set exit code as termination.

    Raises:
        subprocess.TimeoutExpired: if the process is still running at timeout.
    """
    if not IS_WINDOWS:
        return proc.wait(timeout=timeout)

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if deadline is None:
            step = POLL_INTERVAL
        else:
            step = min(POLL_INTERVAL, max(0.0, deadline - time.monotonic()))
        try:
            return proc.wait(timeout=step)
        except subprocess.TimeoutExpired:
            code = _terminated_exit_code(proc)
            if code is not None:
                # Exited without signaling. Publish the result ourselves;
                # nothing else will.
                proc.returncode = code
                return code
            if deadline is not None and time.monotonic() >= deadline:
                raise


def _reader(pipe: Any, sink: List[Any]) -> None:
    try:
        sink.append(pipe.read())
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def run(*popenargs: Any,
        timeout: Optional[float] = None,
        check: bool = False,
        capture_output: bool = False,
        input: Optional[Any] = None,  # mirrors subprocess.run
        **kwargs: Any) -> subprocess.CompletedProcess:
    """Drop-in for ``subprocess.run`` that tolerates unsignaled exits.

    Delegates verbatim to ``subprocess.run`` off Windows, and when ``input`` is
    supplied (no model runner uses it, and stdin handling is not worth
    duplicating).
    """
    if not IS_WINDOWS or input is not None:
        return subprocess.run(*popenargs, timeout=timeout, check=check,  # check forwarded
                              capture_output=capture_output, input=input, **kwargs)

    if capture_output:
        # A run()-only kwarg; Popen does not accept it. Reject the same
        # conflict subprocess.run rejects rather than silently dropping one.
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used '
                             'with capture_output.')
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.PIPE

    # Deliberately not `with Popen(...)`: its __exit__ calls an unbounded
    # wait(), which is the very hang this module exists to avoid. Every exit
    # path below therefore leaves proc.returncode set.
    proc = subprocess.Popen(*popenargs, **kwargs)
    readers = []
    chunks: dict = {}
    for name in ('stdout', 'stderr'):
        pipe = getattr(proc, name, None)
        if pipe is not None:
            chunks[name] = []
            thread = threading.Thread(target=_reader, args=(pipe, chunks[name]),
                                      daemon=True)
            thread.start()
            readers.append(thread)

    def _close_pipes() -> None:
        for name in ('stdout', 'stderr', 'stdin'):
            pipe = getattr(proc, name, None)
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    try:
        returncode = wait(proc, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            wait(proc, timeout=1)
        except subprocess.TimeoutExpired:
            # Kernel-stuck and unkillable. Publish a result so nothing else
            # (a later wait, __del__, the GC) blocks on it again.
            proc.returncode = -1
        _close_pipes()
        raise

    for thread in readers:
        thread.join(_READER_GRACE)
    _close_pipes()

    def _text(name: str) -> Optional[Any]:
        collected = chunks.get(name)
        # Empty when a reader outlived its grace period: report no output
        # rather than raising, since the exit code is the meaningful result.
        return collected[0] if collected else None

    stdout, stderr = _text('stdout'), _text('stderr')

    if check and returncode:
        raise subprocess.CalledProcessError(returncode, proc.args,
                                            output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(proc.args, returncode, stdout, stderr)
