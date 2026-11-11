# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The Panel GUI binds loopback by default; non-loopback is opt-in (review item 12)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.gui import is_loopback_address, websocket_serve_kwargs

# --- panel-free: the bind decision (always runs) -------------------------


@pytest.mark.parametrize("addr", ["127.0.0.1", "localhost", "::1"])
def test_loopback_addresses(addr):
    assert is_loopback_address(addr) is True


@pytest.mark.parametrize("addr", ["0.0.0.0", "192.168.1.5", "10.0.0.1", ""])  # nosec B104
def test_non_loopback_addresses(addr):
    assert is_loopback_address(addr) is False


# --- integration: kwargs passed to pn.serve (needs Panel) ----------------


def _call_serve(**kwargs):
    """Run _serve_app with Panel fully mocked; return the pn.serve kwargs.

    Skips when Panel (the optional [gui] extra) is not installed.
    """
    pytest.importorskip("panel")
    from symfluence.gui import server

    with patch.object(server, "pn", MagicMock()) as mock_pn:
        server._serve_app(**kwargs)
    _, serve_kwargs = mock_pn.serve.call_args
    return serve_kwargs


def test_default_binds_loopback():
    serve_kwargs = _call_serve()
    assert serve_kwargs["address"] == "127.0.0.1"
    assert "websocket_origin" not in serve_kwargs


def test_nonloopback_uses_allowed_origins_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="symfluence.gui"):
        serve_kwargs = _call_serve(
            address="0.0.0.0", port=9999,  # nosec B104 - test input
            allow_origin=["gui.example.org:9999", "10.0.0.5:9999"],
        )
    assert serve_kwargs["address"] == "0.0.0.0"  # nosec B104
    assert serve_kwargs["websocket_origin"] == ["gui.example.org:9999", "10.0.0.5:9999"]
    assert any("reachable from other machines" in r.getMessage() for r in caplog.records)


# --- panel-free: origin validation (always runs) --------------------------


def test_loopback_needs_no_origin():
    assert websocket_serve_kwargs("127.0.0.1", 5006) == {}
    assert websocket_serve_kwargs("localhost", 5006, ["ignored:1"]) == {}


@pytest.mark.parametrize("origins", [None, [], ["", "  "]])
def test_nonloopback_without_origin_is_refused(origins):
    """The bind address is not a browser origin (no browser sends 0.0.0.0:5006),
    so the old ``websocket_origin = f"{address}:{port}"`` could never match."""
    with pytest.raises(ValueError, match="--allow-origin"):
        websocket_serve_kwargs("0.0.0.0", 5006, origins)  # nosec B104 - test input


def test_nonloopback_warns_about_missing_authentication(caplog):
    with caplog.at_level(logging.WARNING, logger="symfluence.gui"):
        kwargs = websocket_serve_kwargs(
            "0.0.0.0", 5006, ["gui.example.org:5006"])  # nosec B104 - test input
    assert kwargs == {"websocket_origin": ["gui.example.org:5006"]}
    assert any("NO AUTHENTICATION" in r.getMessage() for r in caplog.records)


def test_cli_refuses_nonloopback_without_allow_origin():
    from argparse import Namespace

    from symfluence.cli.commands.gui_commands import GUICommands
    from symfluence.cli.exit_codes import ExitCode

    args = Namespace(address="0.0.0.0", port=5006, allow_origin=None)  # nosec B104
    with patch("symfluence.gui.serve_app") as serve:
        assert GUICommands.launch(args) == ExitCode.USAGE_ERROR
    serve.assert_not_called()


def test_cli_passes_allow_origin_through():
    from symfluence.cli.argument_parser import CLIParser
    from symfluence.cli.commands.gui_commands import GUICommands

    args = CLIParser().parse_args([
        "gui", "launch", "--address", "0.0.0.0", "--no-browser",  # nosec B104
        "--allow-origin", "a.example:5006", "--allow-origin", "b.example:5006",
    ])
    with patch("symfluence.gui.serve_app") as serve:
        assert GUICommands.launch(args) == 0
    assert serve.call_args.kwargs["allow_origin"] == ["a.example:5006", "b.example:5006"]
    assert serve.call_args.kwargs["address"] == "0.0.0.0"  # nosec B104
