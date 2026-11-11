# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Interactive permission approvals for headless modelling sessions.

Headless Claude Code has no human behind a permission prompt, so the modelling
chat wires ``--permission-prompt-tool`` at the MCP server's ``approve_action``
tool: when the agent wants a tool outside its allowlist, Claude Code calls
``approve_action`` and waits for its verdict.

The MCP server runs as a subprocess of ``claude``, not of the TUI, so the two
sides meet in the filesystem (the agent cache): the server writes a
``*.request.json`` and blocks polling for the matching ``*.reply.json``; the
chat screen watches the directory, pops a modal, and writes the reply. No
reply within the timeout is a denial — permission is never granted by
silence. Stdlib-only, like everything else in this package.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path

# How long the MCP server waits for a human verdict before denying.
DEFAULT_TIMEOUT_S = 300.0
_POLL_INTERVAL_S = 0.2


def approvals_root() -> Path:
    """Directory where approval requests and replies are exchanged.

    ``agent_cache_root()`` guarantees (or refuses with ``PermissionError``) a
    directory owned by the current user with mode 0700, so no other local
    user can read a request or plant an ``approved: true`` reply.
    """
    from symfluence.resources import agent_cache_root
    return agent_cache_root() / 'approvals'


def _written_by_current_user(path: Path) -> bool:
    """True when ``path`` is a regular file owned by the current user.

    Defence in depth behind the private directory: a verdict is only honoured
    if this user wrote it. (Windows has no POSIX owner to compare.)
    """
    if sys.platform == 'win32':
        return path.is_file()
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()


def _request_path(request_id: str) -> Path:
    return approvals_root() / f'{request_id}.request.json'


def _reply_path(request_id: str) -> Path:
    return approvals_root() / f'{request_id}.reply.json'


def request_approval(
    tool_name: str,
    tool_input: dict,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Ask the human (via the watching UI) to approve one tool use. Blocks.

    Returns the Claude Code permission-prompt verdict:
    ``{'behavior': 'allow', 'updatedInput': ...}`` or
    ``{'behavior': 'deny', 'message': ...}``.
    """
    request_id = uuid.uuid4().hex[:12]
    root = approvals_root()
    root.mkdir(parents=True, exist_ok=True)
    _request_path(request_id).write_text(json.dumps({
        'id': request_id,
        'tool_name': tool_name,
        'input': tool_input,
        'created_at': time.time(),
    }, indent=2), encoding='utf-8')

    deadline = time.time() + timeout_s
    reply_path = _reply_path(request_id)
    try:
        while time.time() < deadline:
            if reply_path.is_file():
                if not _written_by_current_user(reply_path):
                    break  # not ours: never grant on someone else's say-so
                try:
                    reply = json.loads(reply_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    break  # unreadable reply: fall through to deny
                if reply.get('approved'):
                    return {'behavior': 'allow', 'updatedInput': tool_input}
                return {
                    'behavior': 'deny',
                    'message': reply.get('message')
                               or 'Denied by the user in the SYMFLUENCE TUI.',
                }
            time.sleep(_POLL_INTERVAL_S)
        return {
            'behavior': 'deny',
            'message': f'No approval within {timeout_s:.0f}s '
                       f'(is the SYMFLUENCE chat still open?).',
        }
    finally:
        _request_path(request_id).unlink(missing_ok=True)
        reply_path.unlink(missing_ok=True)


def list_pending(max_age_s: float = DEFAULT_TIMEOUT_S) -> list[dict]:
    """Unanswered approval requests, oldest first (for the watching UI)."""
    root = approvals_root()
    if not root.is_dir():
        return []
    pending = []
    now = time.time()
    for path in sorted(root.glob('*.request.json')):
        try:
            request = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        request_id = request.get('id') or path.stem.removesuffix('.request')
        if _reply_path(request_id).is_file():
            continue
        if now - float(request.get('created_at') or now) > max_age_s:
            continue  # the requesting server has already given up
        request['id'] = request_id
        pending.append(request)
    return pending


def reply(request_id: str, approved: bool, message: str = '') -> None:
    """Record the human verdict for one pending request."""
    root = approvals_root()
    root.mkdir(parents=True, exist_ok=True)
    _reply_path(request_id).write_text(json.dumps({
        'approved': bool(approved),
        'message': message,
    }), encoding='utf-8')
