"""Eval watchdog: model-timeout floor, MPI beats, child process groups."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from symfluence.core.calibration import eval_watchdog as ew
from symfluence.core.calibration.mixins.parallel.execution_strategies import mpi as mpi_mod
from symfluence.core.calibration.mixins.parallel.execution_strategies import (
    mpi_persistent as mpip_mod,
)
from symfluence.core.calibration.mixins.parallel.execution_strategies.mpi import (
    MPIExecutionStrategy,
    batch_deadline_seconds,
    symfluence_import_root,
)
from symfluence.core.calibration.mixins.parallel.execution_strategies.mpi_persistent import (
    PersistentMPIExecutionStrategy,
)


@pytest.fixture
def fresh_watchdog(monkeypatch):
    """A private singleton so tests never touch (or leave) the process-wide one."""
    monkeypatch.delenv(ew._ENV_VAR, raising=False)
    monkeypatch.setattr(ew, '_singleton', None)
    yield
    monkeypatch.setattr(ew, '_singleton', None)


def test_model_timeout_from_flat_and_typed_config():
    flat = {'HYDROLOGICAL_MODEL': 'SUMMA', 'ROUTING_MODEL': 'mizuRoute',
            'SUMMA_TIMEOUT': 7200, 'MIZUROUTE_TIMEOUT': 3600, 'FUSE_TIMEOUT': 99999}
    assert ew.model_timeout_from_config(flat) == 10800.0
    assert ew.model_timeout_from_tasks([{'config': flat}]) == 10800.0
    assert ew.model_timeout_from_config({'HYDROLOGICAL_MODEL': 'SUMMA'}) is None
    assert ew.model_timeout_from_tasks([]) is None

    typed = SimpleNamespace(model=SimpleNamespace(
        hydrological_model='FUSE', routing_model='none', fuse=SimpleNamespace(timeout=4000)))
    assert ew.model_timeout_from_config(typed) == 4000.0


def test_default_limit_is_raised_to_cover_the_model_timeout(fresh_watchdog):
    watchdog = ew.get_watchdog()
    assert watchdog.timeout == 1800.0
    watchdog.ensure_covers(7200)
    assert watchdog.timeout >= 7200
    watchdog.ensure_covers(600)  # never lowered
    assert watchdog.timeout >= 7200


def test_env_override_stays_authoritative(fresh_watchdog, monkeypatch):
    monkeypatch.setenv(ew._ENV_VAR, '900')
    watchdog = ew.get_watchdog()
    watchdog.ensure_covers(7200)
    assert watchdog.timeout == 900.0

    monkeypatch.setattr(ew, '_singleton', None)
    monkeypatch.setenv(ew._ENV_VAR, '0')
    disabled = ew.get_watchdog()
    disabled.ensure_covers(7200)
    assert not disabled.enabled


def test_batch_deadline_covers_queued_tasks_at_full_model_timeout():
    assert batch_deadline_seconds(4, 4) == 7200
    assert batch_deadline_seconds(100, 4) == 30000
    # 8 tasks on 4 workers = 2 per worker, each allowed 7200 s
    assert batch_deadline_seconds(8, 4, 7200) == 2 * 7200 + 300


class _Beats:
    def __init__(self):
        self.beats = 0
        self.registered = []
        self.unregistered = []
        self.covers = []

    def beat(self):
        self.beats += 1

    def ensure_covers(self, value):
        self.covers.append(value)

    def register_child_group(self, pid):
        self.registered.append(pid)

    def unregister_child_group(self, pid):
        self.unregistered.append(pid)


def test_mpi_batch_wait_beats_and_registers_the_launcher_group(tmp_path, monkeypatch):
    fake = _Beats()
    monkeypatch.setattr(ew, 'get_watchdog', lambda logger=None: fake)
    monkeypatch.setattr(mpi_mod, '_WAIT_POLL_SECONDS', 0.2)
    strategy = MPIExecutionStrategy(tmp_path, 2, logging.getLogger('test_mpi'))
    tasks = [{'config': {'HYDROLOGICAL_MODEL': 'SUMMA', 'SUMMA_TIMEOUT': 7200}}]

    result = strategy._run_mpi_batch(
        [sys.executable, '-c', 'import time; time.sleep(1); print("done")'],
        dict(os.environ), tasks, 2,
    )

    assert result.returncode == 0 and 'done' in result.stdout
    assert fake.beats >= 3  # start, at least one poll, end
    assert fake.covers == [7200.0]
    assert len(fake.registered) == 1 and fake.unregistered == fake.registered


def test_mpi_batch_past_its_deadline_is_killed(tmp_path, monkeypatch):
    fake = _Beats()
    monkeypatch.setattr(ew, 'get_watchdog', lambda logger=None: fake)
    monkeypatch.setattr(mpi_mod, '_WAIT_POLL_SECONDS', 0.2)
    monkeypatch.setattr(mpi_mod, 'batch_deadline_seconds', lambda *a, **k: 0.5)
    strategy = MPIExecutionStrategy(tmp_path, 2, logging.getLogger('test_mpi'))

    started = time.monotonic()
    result = strategy._run_mpi_batch(
        [sys.executable, '-c', 'import time; time.sleep(60)'], dict(os.environ), [{}], 2)

    assert result.returncode != 0
    assert 'deadline' in result.stderr
    assert time.monotonic() - started < 30


def test_persistent_wait_loop_beats_the_watchdog(tmp_path, monkeypatch):
    fake = _Beats()
    monkeypatch.setattr(ew, 'get_watchdog', lambda logger=None: fake)
    monkeypatch.setattr(mpip_mod, '_BEAT_INTERVAL_SECONDS', 0.0)
    strategy = PersistentMPIExecutionStrategy(tmp_path, 4, logging.getLogger('test_mpi'))
    comm = tmp_path / 'comm'
    comm.mkdir()
    strategy._comm_dir = comm
    polls = {'n': 0}

    def poll():
        # the broker "answers" on the third poll
        polls['n'] += 1
        if polls['n'] == 3:
            import pickle
            (comm / 'results.pkl').write_bytes(pickle.dumps([{'score': 1.0}]))
            (comm / 'results.ready').write_text('1')
        return None

    strategy._process = SimpleNamespace(poll=poll, pid=None)
    monkeypatch.setattr(mpip_mod.time, 'sleep', lambda _s: None)

    results = strategy.execute([{'config': {}}], lambda t: t, 4)

    assert results == [{'score': 1.0}]
    assert fake.beats >= 2


def test_mpi_workers_get_the_package_root_not_the_core_directory(tmp_path):
    import symfluence

    root = symfluence_import_root()
    assert (root / 'symfluence' / '__init__.py').exists()
    assert root == Path(symfluence.__file__).resolve().parent.parent

    for strategy in (
        MPIExecutionStrategy(tmp_path, 2, logging.getLogger('test_mpi')),
        PersistentMPIExecutionStrategy(tmp_path, 2, logging.getLogger('test_mpi')),
    ):
        first = strategy._build_mpi_environment()['PYTHONPATH'].split(os.pathsep)[0]
        assert Path(first) == root
        # 'config', 'metrics', ... must not become importable top-level modules
        assert not (Path(first) / 'config').exists()

    script = tmp_path / 'worker.py'
    MPIExecutionStrategy(tmp_path, 2)._create_worker_script(
        script, tmp_path / 't.pkl', tmp_path / 'r.pkl', 'some.module', 'fn')
    assert f'sys.path.insert(0, r"{root}")' in script.read_text(encoding='utf-8')


def test_mpi_python_is_this_interpreter_unless_overridden(tmp_path, monkeypatch):
    """A stray ~/venv/bin/python must not be preferred over sys.executable."""
    home = tmp_path / 'home'
    venv_bin = home / 'venv' / ('Scripts' if os.name == 'nt' else 'bin')
    venv_bin.mkdir(parents=True)
    for name in ('python', 'python3'):
        (venv_bin / name).write_text('#!/bin/sh\n')
        (venv_bin / name).chmod(0o755)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.delenv('SYMFLUENCE_PYTHON', raising=False)
    monkeypatch.delenv('SLURM_TMPDIR', raising=False)
    monkeypatch.setenv('VIRTUAL_ENV', str(home / 'venv'))

    strategies = (
        MPIExecutionStrategy(tmp_path, 2, logging.getLogger('test_mpi')),
        PersistentMPIExecutionStrategy(tmp_path, 2, logging.getLogger('test_mpi')),
    )
    for strategy in strategies:
        assert strategy._find_python_executable() == sys.executable

    monkeypatch.setenv('SYMFLUENCE_PYTHON', sys.executable + '')
    for strategy in strategies:
        assert strategy._find_python_executable() == sys.executable


@pytest.mark.skipif(sys.platform == 'win32', reason='process groups are POSIX-only')
def test_watchdog_kill_takes_registered_child_groups_with_it(tmp_path):
    """os._exit skips atexit reapers; the watchdog must kill the groups itself."""
    src = Path(ew.__file__).resolve().parents[3]
    child = textwrap.dedent('''
        import os, subprocess, sys, time
        os.environ["SYMFLUENCE_EVAL_WATCHDOG_SECONDS"] = "1"
        from symfluence.core.calibration.eval_watchdog import get_watchdog
        D = subprocess.DEVNULL
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             start_new_session=True, stdin=D, stdout=D, stderr=D)
        print("CHILD", p.pid, flush=True)
        w = get_watchdog()
        w.register_child_group(p.pid)
        w.arm()
        time.sleep(60)
    ''')
    env = dict(os.environ, PYTHONPATH=str(src))
    done = subprocess.run([sys.executable, '-c', child], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, env=env, timeout=120)
    pid = int(done.stdout.split('CHILD')[1].split()[0])
    try:
        assert done.returncode == 75
        deadline = time.monotonic() + 5
        alive = True
        while alive and time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                alive = False
        assert not alive, 'registered child group survived the watchdog exit'
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_sequential_and_pool_strategies_raise_the_limit(fresh_watchdog):
    from symfluence.core.calibration.mixins.parallel.execution_strategies.sequential import (
        SequentialExecutionStrategy,
    )

    tasks = [{'config': {'HYDROLOGICAL_MODEL': 'SUMMA', 'SUMMA_TIMEOUT': 7200}}]
    SequentialExecutionStrategy().execute(tasks, lambda task: {'ok': True}, 1)
    assert ew.get_watchdog().timeout >= 7200
