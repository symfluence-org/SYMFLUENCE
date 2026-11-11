# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Console message bodies are printed literally (no Rich markup parsing)."""
from __future__ import annotations

import io

import pytest

from symfluence.cli.console import Console, ConsoleConfig
from symfluence.core.exceptions import OptionalDependencyError

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


@pytest.fixture
def streams():
    out, err = io.StringIO(), io.StringIO()
    console = Console(ConsoleConfig(output_stream=out, error_stream=err, use_colors=False))
    return console, out, err


def _flat(stream):
    return " ".join(stream.getvalue().split())


@pytest.mark.parametrize("feature,extra", [("The TUI", "tui"), ("The GUI", "gui")])
def test_install_hint_keeps_its_extra(streams, feature, extra):
    """`pip install "symfluence[tui]"` was printed as `pip install "symfluence"`:
    Rich took `[tui]` for a style tag."""
    console, _, err = streams
    console.error(str(OptionalDependencyError(feature, extra)))
    assert f'pip install "symfluence[{extra}]"' in _flat(err)
    assert "[ERROR]" in _flat(err)


@pytest.mark.parametrize("method,prefix", [
    ("info", ""), ("success", "[OK]"), ("warning", "[WARN]"), ("debug", "[DEBUG]"),
])
def test_brackets_survive_in_every_level(streams, method, prefix):
    console, out, _ = streams
    getattr(console, method)("use symfluence[gui] or [bold]x[/bold] in [red]")
    text = _flat(out)
    assert "use symfluence[gui] or [bold]x[/bold] in [red]" in text
    assert prefix in text


def test_non_string_messages_still_print(streams):
    console, out, _ = streams
    console.info({"total_steps": 3})
    assert "total_steps" in out.getvalue()
