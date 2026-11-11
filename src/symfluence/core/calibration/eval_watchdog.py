# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wall-clock watchdog for calibration evaluations.

A single model evaluation is seconds-to-minutes of work, but on Windows we have
observed individual evaluations wedge *indefinitely* on things no Python thread
or signal can interrupt: a JAX/XLA compile deadlock, and a ``gc.collect()``
finalizer deadlock (an xarray/netCDF handle closing into the netCDF-S3 shutdown
hang). One such eval freezes the whole calibration run for days, so an
orchestrating group runner never advances.

This watchdog converts that indefinite freeze into a fast, recoverable process
death: the optimizer ``arm()``s it, each evaluation ``beat()``s it, and if no
beat arrives for longer than the timeout a background thread dumps every
thread's stack (so the log shows *where* it wedged) and calls ``os._exit`` --
NOT ``sys.exit``, because the hang may itself be in a finalizer/atexit path that
a clean shutdown would re-enter and deadlock on. The orchestrator then re-spawns
the config on a fresh process.

Configure with ``SYMFLUENCE_EVAL_WATCHDOG_SECONDS`` (``0`` or negative
disables). When the variable is set it is authoritative. When it is not, the
limit starts at 1800 s and is raised to cover the configured per-evaluation
model timeout (``<MODEL>_TIMEOUT`` of the active models) as soon as an
execution strategy sees its first batch — otherwise a model that is allowed
7200 s would be killed by the watchdog after 1800.

Strategies that wait on external worker processes (the MPI strategies)
``beat()`` from their own wait loops, which carry their own batch deadline, and
``register_child_group()`` the process groups they launched: ``os._exit``
skips every ``atexit`` reaper, so the watchdog kills those groups itself before
exiting rather than leaving ranks running into per-process directories the
re-spawned run will reuse.
"""
from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Set

_DEFAULT_TIMEOUT_S = 1800.0          # 30 min
_EXIT_CODE = 75                      # distinct, so a harness can tag watchdog kills
_ENV_VAR = "SYMFLUENCE_EVAL_WATCHDOG_SECONDS"

#: Slack added on top of the model timeout, so the model's own timeout (which
#: produces a clean, scored failure) always fires before the watchdog does.
_MODEL_TIMEOUT_MARGIN_S = 300.0


def model_timeout_from_config(config: Any) -> Optional[float]:
    """Configured wall-clock budget (s) of one evaluation, or None if unknown.

    One evaluation runs the hydrological model(s) and then the routing model,
    so their ``<MODEL>_TIMEOUT`` values add up. Reads a flat-dict config (what
    calibration tasks carry) or a typed config.
    """
    if config is None:
        return None
    flat = isinstance(config, Mapping)

    def _value(dict_key: str, getter: Any) -> Any:
        if flat:
            return config.get(dict_key)
        try:
            return getter()
        except AttributeError:
            return None

    names: list = []
    for raw in (
        _value('HYDROLOGICAL_MODEL', lambda: config.model.hydrological_model),
        _value('ROUTING_MODEL', lambda: config.model.routing_model),
    ):
        for name in str(raw or '').split(','):
            name = name.strip().upper()
            if name and name not in ('NONE', 'DEFAULT') and name not in names:
                names.append(name)

    total = 0.0
    for name in names:
        raw_timeout = _value(
            f'{name}_TIMEOUT',
            lambda name=name: getattr(config.model, name.lower()).timeout,
        )
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            continue
        if timeout > 0:
            total += timeout
    return total or None


def model_timeout_from_tasks(tasks: Iterable[Any]) -> Optional[float]:
    """:func:`model_timeout_from_config` for the config carried by a task batch."""
    for task in tasks:
        if isinstance(task, Mapping) and task.get('config') is not None:
            return model_timeout_from_config(task['config'])
        return None
    return None


class EvalWatchdog:
    """No-progress detector that self-terminates a wedged calibration process."""

    def __init__(self, timeout_s: float, logger=None, explicit: bool = False):
        self._timeout = float(timeout_s)
        # True when the limit came from the environment override: it is then
        # authoritative and never adjusted from the model configuration.
        self._explicit = bool(explicit)
        self._logger = logger
        self._last_beat: Optional[float] = None
        self._armed = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._child_groups: Set[int] = set()

    @property
    def timeout(self) -> float:
        return self._timeout

    def ensure_covers(self, model_timeout_s: Optional[float]) -> None:
        """Raise the limit so it is not below one evaluation's model timeout.

        Never lowers the limit, and does nothing when the limit was set
        explicitly through the environment or the watchdog is disabled.
        """
        if not model_timeout_s or self._explicit or not self.enabled:
            return
        needed = float(model_timeout_s) + _MODEL_TIMEOUT_MARGIN_S
        with self._lock:
            if needed <= self._timeout:
                return
            previous, self._timeout = self._timeout, needed
        if self._logger:
            self._logger.info(
                f"Eval watchdog limit raised from {previous:.0f}s to {needed:.0f}s to "
                f"cover the configured model timeout ({float(model_timeout_s):.0f}s); "
                f"set {_ENV_VAR} to override."
            )

    def register_child_group(self, pid: int) -> None:
        """Remember a child process-group leader to kill if the watchdog fires."""
        with self._lock:
            self._child_groups.add(int(pid))

    def unregister_child_group(self, pid: int) -> None:
        with self._lock:
            self._child_groups.discard(int(pid))

    @property
    def enabled(self) -> bool:
        return self._timeout > 0

    def arm(self) -> None:
        """Begin (or resume) monitoring; starts the daemon thread on first use."""
        if not self.enabled:
            return
        with self._lock:
            self._armed = True
            self._last_beat = time.monotonic()
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._monitor, name="eval-watchdog", daemon=True
                )
                self._thread.start()
        if self._logger:
            self._logger.info(
                f"Eval watchdog armed: a single evaluation exceeding "
                f"{self._timeout:.0f}s terminates the run for auto-recovery "
                f"(set {_ENV_VAR}=0 to disable)."
            )

    def beat(self) -> None:
        """Record evaluation progress. Cheap; safe to call on every evaluation."""
        if not self.enabled:
            return
        with self._lock:
            if self._armed:
                self._last_beat = time.monotonic()

    def disarm(self) -> None:
        """Stop monitoring (e.g. once optimization returns)."""
        with self._lock:
            self._armed = False

    def _monitor(self) -> None:
        poll = max(5.0, min(30.0, self._timeout / 6.0))
        while True:
            time.sleep(poll)
            with self._lock:
                armed, last = self._armed, self._last_beat
            if not armed or last is None:
                continue
            stuck = time.monotonic() - last
            if stuck > self._timeout:
                self._fire(stuck)

    def _fire(self, stuck: float) -> None:
        msg = (
            f"EVAL WATCHDOG: no evaluation progress for {stuck:.0f}s "
            f"(limit {self._timeout:.0f}s). The run is wedged on a single "
            f"evaluation (e.g. a JAX/XLA compile or gc/finalizer deadlock). "
            f"Dumping thread stacks and terminating so the orchestrator can "
            f"recover."
        )
        try:
            if self._logger:
                self._logger.error(msg)
        except Exception:  # noqa: BLE001 — never mask the kill
            pass
        try:
            sys.stderr.write("\n" + msg + "\n")
            sys.stderr.flush()
            faulthandler.dump_traceback(all_threads=True)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        # os._exit skips atexit, so no strategy's reaper will run: kill the
        # worker process groups here, or they keep running (and writing into
        # per-process directories the re-spawned run reuses).
        self._kill_child_groups()
        # os._exit, not sys.exit: skip atexit/finalizers -- the hang may BE in a
        # finalizer/gc, which a clean shutdown would re-enter and deadlock on.
        os._exit(_EXIT_CODE)

    def _kill_child_groups(self) -> None:
        """Hard-kill every registered child process group. Never raises."""
        # Deliberately lock-free: the thread that holds the lock may be the
        # one that is wedged.
        for pid in list(self._child_groups):
            try:
                if os.name == 'nt':
                    subprocess.run(  # nosec B603,B607 — fixed argv, trusted pid
                        ['taskkill', '/F', '/T', '/PID', str(pid)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=10, check=False,
                    )
                else:
                    from symfluence.core.process_exec import kill_process_group
                    kill_process_group(pid)
            except (OSError, ValueError, ImportError, RuntimeError, subprocess.SubprocessError):
                pass


_singleton: Optional[EvalWatchdog] = None
_singleton_lock = threading.Lock()


def get_watchdog(logger=None) -> EvalWatchdog:
    """Return the process-wide watchdog, constructing it from the env on first use."""
    return _process_watchdog(logger)


def _process_watchdog(logger=None) -> EvalWatchdog:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            raw = os.environ.get(_ENV_VAR)
            explicit = raw not in (None, "")
            try:
                timeout = float(raw) if raw not in (None, "") else _DEFAULT_TIMEOUT_S
            except (TypeError, ValueError):
                timeout, explicit = _DEFAULT_TIMEOUT_S, False
            _singleton = EvalWatchdog(timeout, logger, explicit=explicit)
        elif logger is not None and _singleton._logger is None:
            _singleton._logger = logger
        return _singleton


def cover_model_timeout(tasks: Iterable[Any]) -> None:
    """Raise the process-wide watchdog limit to cover this batch's model timeout.

    For strategies whose evaluations run under the default limit, which is
    below most models' own ``<MODEL>_TIMEOUT``. No-op when the limit was set
    through the environment.
    """
    _process_watchdog().ensure_covers(model_timeout_from_tasks(tasks))
