# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The agent cache (approvals, jobs, skills) is private to the current user.

It used to be ``<tempdir>/symfluence-agent-skills``: a predictable name in a
directory every local user can write to, where another user could read
approval requests, drop an ``approved: true`` reply, or replace skill files.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from symfluence.agent import approvals
from symfluence.resources import agent_cache_root
from symfluence.resources import manager as resources_manager

posix_only = pytest.mark.skipif(sys.platform == 'win32', reason='POSIX permissions')


def test_not_in_the_shared_temp_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv('SYMFLUENCE_AGENT_CACHE_DIR', raising=False)
    monkeypatch.setattr(resources_manager, '_user_cache_base', lambda: tmp_path / 'user-cache')
    root = agent_cache_root()
    assert root == tmp_path / 'user-cache' / 'symfluence' / 'agent'
    assert Path(tempfile.gettempdir()) / 'symfluence-agent-skills' != root


def test_user_cache_base_is_under_the_home_or_xdg_dir(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    base = resources_manager._user_cache_base()
    assert base != Path(tempfile.gettempdir())
    assert base in (tmp_path / 'xdg', tmp_path / 'local') or Path.home() in base.parents


@posix_only
def test_created_with_mode_0700():
    root = agent_cache_root()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert root.stat().st_uid == os.getuid()


@posix_only
@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777, 0o701])
def test_refuses_a_directory_open_to_others(mode):
    root = agent_cache_root()
    os.chmod(root, mode)
    try:
        with pytest.raises(PermissionError, match='accessible to other users'):
            agent_cache_root()
        # ... and so does everything built on it
        with pytest.raises(PermissionError):
            approvals.request_approval('Bash', {'command': 'ls'}, timeout_s=0.1)
    finally:
        os.chmod(root, 0o700)


@posix_only
def test_refuses_a_symlink(monkeypatch, tmp_path):
    target = tmp_path / 'elsewhere'
    target.mkdir(mode=0o700)
    link = tmp_path / 'link'
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv('SYMFLUENCE_AGENT_CACHE_DIR', str(link))
    with pytest.raises(PermissionError, match='not a real directory'):
        agent_cache_root()


@posix_only
def test_refuses_a_directory_owned_by_someone_else(monkeypatch):
    root = agent_cache_root()
    monkeypatch.setattr(resources_manager.os, 'getuid', lambda: os.stat(root).st_uid + 1)
    with pytest.raises(PermissionError, match='owned by another user'):
        agent_cache_root()


@posix_only
def test_reply_from_another_user_is_not_honoured(monkeypatch):
    """Defence in depth: an `approved: true` file this user did not write
    denies the request."""
    import threading
    import time

    def _plant():
        deadline = time.time() + 30
        while time.time() < deadline:
            pending = approvals.list_pending()
            if pending:
                approvals.reply(pending[0]['id'], True)
                return
            time.sleep(0.02)

    monkeypatch.setattr(approvals, '_written_by_current_user', lambda path: False)
    thread = threading.Thread(target=_plant, daemon=True)
    thread.start()
    verdict = approvals.request_approval('Bash', {'command': 'rm -rf /'}, timeout_s=5.0)
    thread.join(timeout=10)
    assert verdict['behavior'] == 'deny'


def test_own_reply_is_honoured():
    import threading
    import time

    def _answer():
        deadline = time.time() + 30
        while time.time() < deadline:
            pending = approvals.list_pending()
            if pending:
                approvals.reply(pending[0]['id'], True)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=_answer, daemon=True)
    thread.start()
    verdict = approvals.request_approval('Bash', {'command': 'ls'}, timeout_s=20.0)
    thread.join(timeout=10)
    assert verdict == {'behavior': 'allow', 'updatedInput': {'command': 'ls'}}
    assert json.dumps(verdict)
