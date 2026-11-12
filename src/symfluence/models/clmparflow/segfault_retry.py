# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Bounded rerun of ParFlow-CLM runs that die with a segmentation fault.

CLM (``clm_lsm``) inside ParFlow segfaults now and then, typically at the
first time step, on inputs that run cleanly when started again (a few percent
of runs).  Clean reruns of identical inputs are bit-identical, so a run that
ended with signal 11 is simply started again, a bounded number of times.
Any other failure is returned to the caller unchanged on the first attempt.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

DEFAULT_SEGV_RETRIES = 2

# 139 = 128 + SIGSEGV from a shell/mpirun wrapper; -11 = Python's report of a
# child killed by SIGSEGV when ParFlow runs without a launcher.
_SEGV_RETURN_CODES = (139, -11)
_SEGV_MARKERS = ('signal 11', 'Segmentation fault')

_logger = logging.getLogger(__name__)


def is_segfault(returncode: int, output: str) -> bool:
    """True if a ParFlow run ended in a segmentation fault (exit code or launcher message)."""
    if returncode == 0:
        return False
    return returncode in _SEGV_RETURN_CODES or any(m in output for m in _SEGV_MARKERS)


def run_with_segfault_retry(
    launch: Callable[[], subprocess.CompletedProcess],
    run_dir: Path,
    runname: str,
    retries: int = DEFAULT_SEGV_RETRIES,
    log: Optional[Callable[[str], None]] = None,
) -> subprocess.CompletedProcess:
    """Call ``launch`` until it does not segfault, at most ``1 + retries`` times.

    Before each rerun the output of the failed attempt is kept as
    ``<runname>.segv<n>.log`` in ``run_dir`` and ParFlow's partial output
    (``<runname>.out.*``) is removed.  The result of the last attempt is
    returned whatever it is; the caller decides whether it failed.

    Args:
        launch: Starts ParFlow once and returns the completed process
            (``stdout``/``stderr`` captured as text).
        run_dir: ParFlow working directory.
        runname: ParFlow run name (prefix of its output files).
        retries: Reruns allowed after a segfault (0 disables the retry).
        log: Warning sink; defaults to this module's logger.
    """
    warn = log or _logger.warning
    attempts = 1 + max(0, int(retries))
    result = launch()
    for attempt in range(1, attempts):
        output = (result.stdout or '') + (result.stderr or '')
        if not is_segfault(result.returncode, output):
            break
        (run_dir / f'{runname}.segv{attempt}.log').write_text(output, encoding='utf-8')
        warn(f"ParFlow-CLM segmentation fault in {run_dir} (exit {result.returncode}, "
             f"attempt {attempt} of {attempts}); removing partial output and running again")
        for partial in run_dir.glob(f'{runname}.out.*'):
            if partial.is_file():
                partial.unlink()
        result = launch()
    return result
