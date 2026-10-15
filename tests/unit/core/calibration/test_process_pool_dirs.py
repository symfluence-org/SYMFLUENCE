from __future__ import annotations

import os
import time
from pathlib import Path

from symfluence.core.calibration.mixins.parallel.execution_strategies.process_pool import (
    ProcessPoolExecutionStrategy,
)


def _occupy(task):
    """Fail if another task holds the same per-process directory concurrently."""
    lock = Path(task['root']) / f"proc_{task['proc_id']}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return {'index': task['index'], 'clash': True}
    try:
        if task['index'] == 0:  # hold proc 0 until task 1 is done and its worker is free
            deadline = time.time() + 60
            while not (Path(task['root']) / 'done_1').exists() and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(0.5)
        time.sleep(task['sleep'])
    finally:
        os.close(fd)
        lock.unlink()
        (Path(task['root']) / f"done_{task['index']}").touch()
    return {'index': task['index'], 'clash': False}


def test_tasks_sharing_proc_id_never_overlap(tmp_path):
    # Task 0 runs until task 1 has finished; with plain executor.map the freed
    # worker would then start task 2 (also proc 0) while task 0 still runs.
    sleeps = [0.05] * 6
    tasks = [{'index': i, 'proc_id': i % 2, 'root': str(tmp_path), 'sleep': s} for i, s in enumerate(sleeps)]
    results = ProcessPoolExecutionStrategy().execute(tasks, _occupy, max_workers=2)
    assert [r['index'] for r in results] == list(range(6))
    assert not any(r['clash'] for r in results)


def test_waves_split_on_repeated_proc_id():
    tasks = [{'proc_id': p} for p in (0, 1, 0, 1, 2)]
    waves = ProcessPoolExecutionStrategy._waves_without_shared_dirs(tasks)
    assert [[t['proc_id'] for t in w] for w in waves] == [[0, 1], [0, 1, 2]]


def _quick(task):
    return {'index': task['index']}


def test_completed_tasks_beat_the_eval_watchdog(monkeypatch):
    # Without beats the watchdog kills a healthy parallel run once it has been
    # armed for longer than the timeout, however many trials have finished.
    from symfluence.core.calibration import eval_watchdog

    class Recorder:
        beats = 0

        def beat(self):
            Recorder.beats += 1

    monkeypatch.setattr(eval_watchdog, 'get_watchdog', lambda *a, **k: Recorder())
    tasks = [{'index': i, 'proc_id': i % 2} for i in range(4)]
    ProcessPoolExecutionStrategy().execute(tasks, _quick, max_workers=2)
    assert Recorder.beats >= len(tasks)
