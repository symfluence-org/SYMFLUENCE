# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Console colour handling: no ANSI when piped, NO_COLOR honoured; run summary table."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from symfluence.cli.console import Console, ConsoleConfig

pytestmark = pytest.mark.unit

ESC = "\x1b["


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _render(stream, monkeypatch, **env):
    for key in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    console = Console(ConsoleConfig(output_stream=stream, error_stream=io.StringIO()))
    console.success("done")
    console.table(["a", "b"], [[1, 2]], title="t")
    return stream.getvalue()


def test_piped_output_has_no_ansi(monkeypatch):
    out = _render(io.StringIO(), monkeypatch)
    assert "[OK] done" in out
    assert ESC not in out


def test_tty_output_is_coloured(monkeypatch):
    out = _render(_TTY(), monkeypatch, TERM="xterm-256color")
    assert ESC in out


def test_no_color_env_disables_colour_on_tty(monkeypatch):
    out = _render(_TTY(), monkeypatch, TERM="xterm-256color", NO_COLOR="1")
    assert "[OK] done" in out
    # Rich may still emit bold/underline on a TTY, but never colours
    assert "\x1b[32m" not in out and "\x1b[31m" not in out


def test_use_colors_false_is_plain_even_on_tty(monkeypatch):
    stream = _TTY()
    console = Console(ConsoleConfig(output_stream=stream, error_stream=io.StringIO(),
                                    use_colors=False))
    console.success("done")
    assert ESC not in stream.getvalue()


def test_workflow_run_summary_table_is_plain_when_piped(monkeypatch):
    from symfluence.cli.commands import workflow_commands as wc
    from symfluence.cli.commands.base import BaseCommand
    from symfluence.project.run_result import RunResult

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    stream = io.StringIO()
    monkeypatch.setattr(BaseCommand, "_console",
                        Console(ConsoleConfig(output_stream=stream, error_stream=io.StringIO())))
    metrics = {
        "stage": "postprocess", "domain": "d", "experiment_id": "e",
        "models": {"SUMMA": {"metrics": {
            "calibration": {"KGE": 0.81234, "r": 0.9, "alpha": 1.0, "beta": 1.02,
                            "NSE": 0.7, "logNSE": None, "PBIAS": 2.0, "RMSE": 1.0, "n_obs": 730},
        }}},
        "warnings": ["Observed streamflow not found (expected /x.csv)"],
    }
    result = RunResult(status="completed", metrics=metrics,
                       outputs={"metrics_json": Path("/p/results/e/metrics.json")})
    wc._print_run_summary(result)
    out = stream.getvalue()
    assert ESC not in out
    assert "SUMMA" in out and "0.812" in out and "730" in out
    assert "expected /x.csv" in out
    assert str(Path("/p/results/e/metrics.json")) in out


def test_run_summary_silent_without_metrics(monkeypatch):
    from symfluence.cli.commands import workflow_commands as wc
    from symfluence.cli.commands.base import BaseCommand
    from symfluence.project.run_result import RunResult

    stream = io.StringIO()
    monkeypatch.setattr(BaseCommand, "_console",
                        Console(ConsoleConfig(output_stream=stream, error_stream=io.StringIO())))
    wc._print_run_summary(RunResult(status="completed"))
    wc._print_run_summary(object())
    assert stream.getvalue() == ""
