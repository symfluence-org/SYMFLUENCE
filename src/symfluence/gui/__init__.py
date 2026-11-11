# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE GUI Package.

Panel-based web application for interactive hydrological modeling workflows.
Provides map interaction, configuration editing, workflow execution, and results viewing.

Install dependencies: pip install "symfluence[gui]"
Launch: symfluence gui launch
"""

#: Bind addresses that keep the GUI reachable only from the local machine.
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback_address(address: str) -> bool:
    """Return True if *address* keeps the server local-only (loopback)."""
    return address in _LOOPBACK_ADDRESSES


def websocket_serve_kwargs(address, port, allow_origin=None) -> dict:
    """Extra ``pn.serve`` kwargs for the requested bind, validated.

    Loopback binds need nothing. A non-loopback bind must name the origins
    browsers will connect from: Bokeh rejects websocket connections whose
    ``Origin`` is not allow-listed, and the bind address is not an origin a
    browser ever sends for ``0.0.0.0``. The GUI has no authentication, so
    every non-loopback bind is also warned about.
    """
    if is_loopback_address(address):
        return {}
    origins = [o.strip() for o in (allow_origin or []) if o and o.strip()]
    if not origins:
        raise ValueError(
            f"Binding the GUI to {address} needs at least one --allow-origin "
            f"HOST:PORT (the address browsers will use, e.g. "
            f"--allow-origin myserver.example.org:{port}); the bind address "
            "itself is not a valid browser origin."
        )
    logger.warning(
        "SYMFLUENCE GUI is binding to %s:%s, which is reachable from other "
        "machines on the network, and the GUI has NO AUTHENTICATION: anyone "
        "who can reach it can edit configurations and start runs as this "
        "user. Allowed websocket origins: %s. Use the default 127.0.0.1 "
        "(e.g. behind an SSH tunnel) unless you intend to expose it.",
        address,
        port,
        ", ".join(origins),
    )
    return {'websocket_origin': origins}


def serve_app(config_path=None, port=5006, show=True, demo=None, address="127.0.0.1",
              allow_origin=None):
    """
    Build and serve the SYMFLUENCE GUI as a Panel web application.

    Thin wrapper that imports the actual server module lazily so that
    ``from symfluence.gui import serve_app`` succeeds even when Panel
    is not installed (the ImportError is raised only when called).

    Args:
        config_path: Optional path to a YAML config file to preload.
        port: Server port (default 5006).
        show: Open a browser tab automatically.
        demo: Optional demo name (e.g. 'bow') to load a built-in config.
        address: Interface to bind (default 127.0.0.1, loopback only).
        allow_origin: ``host:port`` websocket origins to accept; required for
            a non-loopback bind (see :func:`websocket_serve_kwargs`).
    """
    try:
        import panel  # noqa: F401
    except ImportError:
        raise ImportError(
            "Panel is required for the SYMFLUENCE GUI.\n"
            'Install with:  pip install "symfluence[gui]"'
        ) from None

    from .server import _serve_app
    _serve_app(config_path=config_path, port=port, show=show, demo=demo, address=address,
               allow_origin=allow_origin)


__all__ = ['serve_app', 'is_loopback_address', 'websocket_serve_kwargs']
