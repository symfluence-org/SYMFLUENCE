# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Panel server launch logic.

Provides serve_app() which initializes Panel extensions, builds the
SymfluenceApp, and starts the Bokeh/Tornado server.

The app is served as a factory function so each browser session gets
its own instance with the Panel event loop running for periodic
log flushing.
"""
from __future__ import annotations

import logging

import panel as pn

from . import websocket_serve_kwargs
from .app import GUI_THEME_CSS, SymfluenceApp

logger = logging.getLogger(__name__)


def _serve_app(config_path=None, port=5006, show=True, demo=None, address="127.0.0.1",
               allow_origin=None):
    """
    Build and serve the SYMFLUENCE GUI as a Panel web application.

    Args:
        config_path: Optional path to a YAML config file to preload
        port: Port number for the web server (default 5006)
        show: Whether to auto-open a browser tab (default True)
        demo: Optional demo name (e.g. 'bow') to load on startup
        address: Network interface to bind. Defaults to 127.0.0.1 (loopback);
            the GUI is reachable only from the local machine. A non-loopback
            value (e.g. 0.0.0.0) exposes it on the network and is opt-in.
        allow_origin: ``host:port`` origins browsers will use to reach the GUI
            (the value of their ``Origin`` header, without the scheme). Required
            for a non-loopback bind: the bind address itself is not a usable
            origin (no browser sends ``0.0.0.0:5006``).

    Raises:
        ValueError: On a non-loopback bind without ``allow_origin``.
    """
    serve_kwargs = websocket_serve_kwargs(address, port, allow_origin)

    # Pre-import model modules on the main thread so config adapters
    # are registered before the Tornado daemon thread starts.
    # (Importing on a daemon thread can fail due to atexit restrictions.)
    try:
        import symfluence.models  # noqa: F401
    except Exception:  # noqa: BLE001 — UI resilience
        pass

    pn.extension(
        'terminal',
        sizing_mode='stretch_width',
        raw_css=[GUI_THEME_CSS],
    )

    def create_app():
        """Factory called per-session with Panel event loop running."""
        app = SymfluenceApp(config_path=config_path, demo=demo)
        template = app.build()

        # Start periodic UI updates (event loop is active in factory)
        app.log_viewer.start_polling()
        app.progress_panel.start_polling()

        return template

    pn.serve(
        {'/': create_app},
        port=port,
        address=address,
        show=show,
        title='SYMFLUENCE',
        **serve_kwargs,
    )
