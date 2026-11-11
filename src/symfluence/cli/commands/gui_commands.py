# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GUI command handlers for SYMFLUENCE CLI.

This module implements the handler for launching the Panel-based GUI.
"""
from __future__ import annotations

from argparse import Namespace

from symfluence.core.exceptions import OptionalDependencyError

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


class GUICommands(BaseCommand):
    """Handlers for the gui command category."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence gui launch

        Starts the Panel web application server and opens a browser.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.gui import serve_app
        except ImportError as exc:
            raise OptionalDependencyError("The GUI", "gui", dependency=str(exc)) from exc

        config_path = BaseCommand.get_arg(args, 'config', None)
        port = BaseCommand.get_arg(args, 'port', 5006)
        no_browser = BaseCommand.get_arg(args, 'no_browser', False)
        demo = BaseCommand.get_arg(args, 'demo', None)
        address = BaseCommand.get_arg(args, 'address', '127.0.0.1')

        allow_origin = BaseCommand.get_arg(args, 'allow_origin', None) or []

        from symfluence.gui import is_loopback_address

        if not is_loopback_address(address):
            if not allow_origin:
                BaseCommand._console.error(
                    f"Binding the GUI to {address} requires --allow-origin HOST:PORT "
                    "(repeatable): the address browsers will use to reach it, e.g. "
                    f"--allow-origin myserver.example.org:{port}. The bind address "
                    "itself is not a valid browser origin."
                )
                return ExitCode.USAGE_ERROR
            BaseCommand._console.warning(
                f"Binding to {address} exposes the GUI on the network, and the GUI "
                "has NO AUTHENTICATION: anyone who can reach it can edit "
                "configurations and start runs as this user."
            )

        BaseCommand._console.info(f"Launching SYMFLUENCE GUI on {address}:{port}...")
        if demo:
            BaseCommand._console.indent(f"Demo mode: {demo}")
        if config_path:
            BaseCommand._console.indent(f"Preloading config: {config_path}")

        serve_app(
            config_path=config_path,
            port=port,
            show=not no_browser,
            demo=demo,
            address=address,
            allow_origin=allow_origin,
        )

        return ExitCode.SUCCESS
