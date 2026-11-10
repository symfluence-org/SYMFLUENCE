# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Background execution and log capture for the SYMFLUENCE GUI.

WorkflowThread runs SYMFLUENCE steps in a daemon thread so the Panel
Tornado event loop remains responsive.  GUILogHandler captures log
records and routes them to WorkflowState.log_text via pn.state.execute.

Steps are executed individually so the progress indicator can update
between each step.
"""
from __future__ import annotations

import logging
import threading

import panel as pn

# Human-readable labels for step names
STEP_LABELS = {
    'setup_project': 'Project Setup',
    'create_pour_point': 'Pour Point',
    'acquire_attributes': 'Acquire Attributes',
    'define_domain': 'Delineation',
    'discretize_domain': 'Discretization',
    'acquire_forcings': 'Acquire Forcings',
    'process_observed_data': 'Observed Data',
    'model_agnostic_preprocessing': 'Preprocessing',
    'build_model_ready_store': 'Data Store',
    'model_specific_preprocessing': 'Model Preprocessing',
    'run_model': 'Run Model',
    'postprocess_results': 'Post-processing',
    'calibrate_model': 'Calibration',
    'run_benchmarking': 'Benchmarking',
    'run_decision_analysis': 'Decision Analysis',
    'run_sensitivity_analysis': 'Sensitivity Analysis',
}


def run_on_ui_thread(callback):
    """Execute callback in the Panel UI context when available.

    Falls back to direct execution if Panel session is unavailable
    (e.g. during tests or from daemon threads without a session).
    """
    try:
        pn.state.execute(callback)
    except Exception:  # noqa: BLE001 — UI resilience
        try:
            callback()
        except Exception:  # noqa: BLE001 — UI resilience
            pass


class GUILogHandler(logging.Handler):
    """
    Logging handler that routes records into WorkflowState.log_text.

    Attaches to the 'symfluence' logger and uses pn.state.execute()
    for thread-safe UI updates.
    """

    def __init__(self, state, level=logging.INFO):
        super().__init__(level)
        self.state = state
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            run_on_ui_thread(lambda m=msg: self.state.append_log(m))
        except Exception:  # noqa: BLE001 — UI resilience
            pass  # never crash the logger


class WorkflowThread:
    """
    Manages a daemon thread for running SYMFLUENCE workflow steps.

    Steps are run individually so step_statuses and progress can be
    updated between each step.

    Usage:
        wt = WorkflowThread(state)
        wt.run_steps(['setup_project', 'create_pour_point'])
    """

    def __init__(self, state):
        self.state = state
        self._thread = None
        self._log_handler = None

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def run_steps(self, step_names, force_rerun=False):
        """Launch step execution in a background thread."""
        step_label = step_names[0] if step_names else None
        if self.alive or not self.state.try_begin_run(step_label):
            self.state.append_log("A workflow is already running.\n")
            return

        # Initialize progress tracking
        statuses = [(name, 'pending') for name in step_names]
        self.state.steps_total = len(step_names)
        self.state.steps_done = 0
        self.state.step_statuses = statuses

        self._thread = threading.Thread(
            target=self._execute,
            args=(step_names, force_rerun),
            daemon=False,
        )
        try:
            self._thread.start()
        except Exception:  # noqa: BLE001 — shutdown cleanup
            self.state.end_run()
            raise

    def run_workflow(self, force_rerun=False):
        """Launch full workflow in a background thread."""
        if self.alive or not self.state.try_begin_run('full_workflow'):
            self.state.append_log("A workflow is already running.\n")
            return

        self._thread = threading.Thread(
            target=self._execute_full,
            args=(force_rerun,),
            daemon=False,
        )
        try:
            self._thread.start()
        except Exception:  # noqa: BLE001 — shutdown cleanup
            self.state.end_run()
            raise

    def _attach_logger(self):
        """Attach GUILogHandler to the symfluence root logger."""
        self._log_handler = GUILogHandler(self.state)
        logging.getLogger('symfluence').addHandler(self._log_handler)

    def _detach_logger(self):
        """Remove GUILogHandler from the logger."""
        if self._log_handler:
            logging.getLogger('symfluence').removeHandler(self._log_handler)
            self._log_handler = None

    def _update_step_status(self, step_names, index, status):
        """Update step_statuses list with new status for step at index."""
        statuses = []
        for i, name in enumerate(step_names):
            if i < index:
                statuses.append((name, 'done'))
            elif i == index:
                statuses.append((name, status))
            else:
                statuses.append((name, 'pending'))
        run_on_ui_thread(lambda s=statuses: setattr(self.state, 'step_statuses', s))

    def _execute(self, step_names, force_rerun):
        """Thread target: run steps one at a time with progress updates."""
        # zarr v3 codecs use asyncio.to_thread() which submits to the global
        # ThreadPoolExecutor.  If _global_shutdown was set (e.g. Panel dev-
        # server reload), reset it so zarr reads don't raise RuntimeError.
        # zarr v3 codecs use asyncio.to_thread() which submits to the global
        # ThreadPoolExecutor.  If _python_exit() ran (e.g. Panel dev-server
        # reload or main-thread exit), reset the shutdown flag so zarr reads
        # don't raise RuntimeError.  The flag is called _shutdown in CPython.
        import concurrent.futures.thread as _cft
        if hasattr(_cft, '_shutdown'):
            _cft._shutdown = False
        if hasattr(_cft, '_global_shutdown'):
            _cft._global_shutdown = False

        sf = None
        try:
            sf = self.state.initialize_symfluence()
            # Attach AFTER initialize: LoggingManager.setup_logging() clears
            # all handlers on the 'symfluence' logger during construction.
            self._attach_logger()

            # Run each step individually for per-step progress
            for i, step_name in enumerate(step_names):
                run_on_ui_thread(
                    lambda name=step_name: setattr(self.state, 'running_step', name)
                )
                self._update_step_status(step_names, i, 'running')

                sf.run_individual_steps([step_name])

                self._update_step_status(step_names, i, 'done')
                run_on_ui_thread(
                    lambda done=i + 1: setattr(self.state, 'steps_done', done)
                )

            # Mark all complete
            run_on_ui_thread(lambda: self.state.append_log("All steps completed successfully.\n"))

        except Exception as exc:  # noqa: BLE001 — UI resilience
            # Mark current step as failed
            run_on_ui_thread(lambda e=exc: self.state.append_log(f"ERROR: {e}\n"))
            done = self.state.steps_done
            if done < len(step_names):
                self._update_step_status(step_names, done, 'error')
        finally:
            self._detach_logger()
            run_on_ui_thread(self.state.end_run)
            # Refresh workflow status using the sf we already have
            if sf is not None:
                try:
                    status = sf.get_workflow_status()
                    run_on_ui_thread(
                        lambda s=status: setattr(self.state, 'workflow_status', s)
                    )
                except Exception:  # noqa: BLE001 — UI resilience
                    pass
            # Signal results-producing steps so the Results tab auto-refreshes
            _RESULTS_STEPS = {'calibrate_model', 'run_benchmarking', 'postprocess_results'}
            if any(s in _RESULTS_STEPS for s in step_names):
                exp_id = (self.state.typed_config.domain.experiment_id
                          if self.state.typed_config else None)
                run_on_ui_thread(
                    lambda eid=exp_id: setattr(self.state, 'last_completed_run', eid)
                )

    def _execute_full(self, force_rerun):
        """Thread target for full workflow."""
        import concurrent.futures.thread as _cft
        if hasattr(_cft, '_shutdown'):
            _cft._shutdown = False
        if hasattr(_cft, '_global_shutdown'):
            _cft._global_shutdown = False

        sf = None
        try:
            sf = self.state.initialize_symfluence()
            self._attach_logger()
            sf.run_workflow(force_run=True if force_rerun else None)
            run_on_ui_thread(lambda: self.state.append_log("Full workflow completed successfully.\n"))
        except Exception as exc:  # noqa: BLE001 — UI resilience
            run_on_ui_thread(lambda e=exc: self.state.append_log(f"ERROR: {e}\n"))
        finally:
            self._detach_logger()
            run_on_ui_thread(self.state.end_run)
            if sf is not None:
                try:
                    status = sf.get_workflow_status()
                    run_on_ui_thread(
                        lambda s=status: setattr(self.state, 'workflow_status', s)
                    )
                except Exception:  # noqa: BLE001 — UI resilience
                    pass
            exp_id = (self.state.typed_config.domain.experiment_id
                      if self.state.typed_config else None)
            run_on_ui_thread(
                lambda eid=exp_id: setattr(self.state, 'last_completed_run', eid)
            )
