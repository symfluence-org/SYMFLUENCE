from __future__ import annotations

import threading
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

pn = pytest.importorskip("panel")

from symfluence.cli.commands.gui_commands import GUICommands
from symfluence.cli.exit_codes import ExitCode
from symfluence.gui.components.results_viewer import ResultsViewer
from symfluence.gui.models.workflow_state import WorkflowState
from symfluence.gui.utils.threading_utils import WorkflowThread


class _Domain:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id


class _Config:
    def __init__(self, experiment_id: str):
        self.domain = _Domain(experiment_id)


def test_gui_launch_command_smoke(monkeypatch):
    calls = {}

    def _fake_serve_app(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("symfluence.gui.serve_app", _fake_serve_app)

    rc = GUICommands.launch(
        Namespace(config="example.yaml", port=5099, no_browser=True, demo="bow")
    )

    assert rc == ExitCode.SUCCESS
    assert calls == {
        "config_path": "example.yaml",
        "port": 5099,
        "show": False,
        "demo": "bow",
        "address": "127.0.0.1",
        "allow_origin": [],
    }


def test_workflow_thread_uses_shared_state_run_lock(monkeypatch):
    state = WorkflowState()
    state.refresh_status = lambda: None

    started = threading.Event()
    release = threading.Event()

    class _FakeSYMFLUENCE:
        def run_individual_steps(self, step_names):
            started.set()
            release.wait(timeout=2)

    monkeypatch.setattr(state, "initialize_symfluence", lambda: _FakeSYMFLUENCE())

    t1 = WorkflowThread(state)
    t2 = WorkflowThread(state)

    t1.run_steps(["setup_project"])
    assert started.wait(timeout=1), "First workflow thread never started"
    assert state.is_running is True

    t2.run_steps(["run_model"])
    assert "A workflow is already running." in state.log_text

    release.set()
    t1._thread.join(timeout=2)
    assert state.is_running is False
    assert state.running_step is None


def test_results_viewer_recreates_loader_when_config_changes(monkeypatch, tmp_path):
    created_loaders = []

    class _FakeLoader:
        def __init__(self, project_dir, config=None):
            self.project_dir = Path(project_dir) if project_dir else None
            self.config = config
            self.sim_calls = []
            created_loaders.append(self)

        def clear_cache(self):
            return None

        def load_observed_streamflow(self):
            return pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))

        def load_simulated_streamflow(self, experiment_id=None):
            self.sim_calls.append(experiment_id)
            return pd.Series([1.1], index=pd.to_datetime(["2020-01-01"]))

        def load_optimization_history(self, experiment_id=None):
            return pd.DataFrame({"iteration": [0], "score": [0.5]})

    monkeypatch.setattr("symfluence.gui.components.results_viewer.ResultsLoader", _FakeLoader)

    state = WorkflowState()
    state.project_dir = str(tmp_path)
    cfg1 = _Config("exp_1")
    cfg2 = _Config("exp_2")
    state.typed_config = cfg1

    viewer = ResultsViewer(state)
    loader1 = viewer._get_loader()
    assert loader1.config is cfg1

    state.typed_config = cfg2
    loader2 = viewer._get_loader()
    assert loader2 is not loader1
    assert loader2.config is cfg2

    viewer._build_hydrograph_tab()
    assert loader2.sim_calls[-1] == "exp_2"


def test_saved_plot_factory_supports_jpg_and_png():
    jpg_pane = ResultsViewer._plot_pane_for_file(Path("plot.jpg"))
    png_pane = ResultsViewer._plot_pane_for_file(Path("plot.png"))
    svg_pane = ResultsViewer._plot_pane_for_file(Path("plot.svg"))
    pdf_pane = ResultsViewer._plot_pane_for_file(Path("plot.pdf"))

    assert isinstance(jpg_pane, pn.pane.Image)
    assert isinstance(png_pane, pn.pane.Image)
    assert isinstance(svg_pane, pn.pane.SVG)
    assert isinstance(pdf_pane, pn.pane.Str)
