# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the Agent home screen and its service."""
from __future__ import annotations

import asyncio

import pytest

textual = pytest.importorskip("textual", reason="textual not installed")

from symfluence.agent.handoff import AgentHandoff  # noqa: E402
from symfluence.agent.modes import AgentMode  # noqa: E402
from symfluence.tui.app import SymfluenceTUI  # noqa: E402
from symfluence.tui.services.agent_service import AgentService  # noqa: E402

# ---------------------------------------------------------------- service

def test_snapshot_gathers_everything(tmp_path):
    snapshot = AgentService().snapshot(tmp_path)
    assert snapshot.workdir == tmp_path
    assert [r.name for r in snapshot.runtimes[:3]] == ['claude', 'codex', 'gemini']
    assert len(snapshot.skills) >= 6
    assert len(snapshot.subagents) >= 2
    assert len(snapshot.mcp_tools) >= 13
    assert snapshot.checks


def test_snapshot_detects_project_context(tmp_path):
    (tmp_path / 'config.yaml').write_text(
        'DOMAIN_NAME: bow\nHYDROLOGICAL_MODEL: SUMMA\n', encoding='utf-8'
    )
    (tmp_path / 'domain_bow').mkdir()
    snapshot = AgentService().snapshot(tmp_path)
    assert snapshot.configs and snapshot.configs[0][0] == 'config.yaml'
    assert snapshot.domains == ['domain_bow']


# ----------------------------------------------------------------- screen
#
# run_test uses the headless driver, which cannot suspend — so starting a
# session always falls back to the AgentHandoff exec contract, which is what
# these tests assert. The suspend round-trip itself is covered by the driver
# probe in _suspend_session and exercised manually in real terminals.

@pytest.fixture(autouse=True)
def _fake_claude(monkeypatch):
    """Make runtime detection deterministic: only 'claude' is installed."""
    import symfluence.agent.launcher as launcher
    import symfluence.tui.services.agent_service as service

    fake = lambda binary: '/usr/bin/claude' if binary == 'claude' else None
    monkeypatch.setattr(launcher.shutil, 'which', fake)
    monkeypatch.setattr(service.shutil, 'which', fake)
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')  # no cache writes from tests
    monkeypatch.delenv('SYMFLUENCE_AGENT_CLI', raising=False)


def _run(coro):
    return asyncio.run(coro)


async def _press_and_return(app, *keys):
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for key in keys:
            await pilot.press(key)
    return app.return_value


def test_code_key_hands_off_with_coding_mode():
    result = _run(_press_and_return(SymfluenceTUI(initial_mode='agent'), 'c'))
    assert isinstance(result, AgentHandoff)
    assert result.mode is AgentMode.CODING
    assert result.prompt is None


def test_model_key_opens_native_chat():
    # claude (fake-installed) supports headless driving → modelling opens the
    # native chat screen instead of leaving the app.
    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('m')
            await pilot.pause()
            return type(app.screen).__name__

    assert _run(_test()) == 'AgentChatScreen'


def test_enter_selects_highlighted_mode_card():
    # First card is Model → native chat opens.
    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
            return type(app.screen).__name__

    assert _run(_test()) == 'AgentChatScreen'


def test_agent_defaults_flow_into_handoff():
    app = SymfluenceTUI(
        initial_mode='agent',
        agent_defaults={'cli': 'claude', 'no_skills': True,
                        'extra_args': ['--resume']},
    )
    result = _run(_press_and_return(app, 'c'))
    assert result.cli == 'claude'
    assert result.no_skills is True
    assert result.extra_args == ['--resume']


def test_details_modal_opens_and_closes():
    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('d')
            await pilot.pause()
            opened = type(app.screen).__name__
            await pilot.press('escape')
            await pilot.pause()
            closed = type(app.screen).__name__
        return opened, closed

    opened, closed = _run(_test())
    assert opened == 'AgentDetailsScreen'
    assert closed == 'AgentHomeScreen'


def test_agent_mode_registered_in_app():
    async def _test():
        app = SymfluenceTUI()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('7')
            await pilot.pause()
            return app.current_mode

    assert _run(_test()) == 'agent'


def test_no_runtime_blocks_session(monkeypatch):
    import symfluence.agent.launcher as launcher
    import symfluence.tui.services.agent_service as service
    monkeypatch.setattr(launcher.shutil, 'which', lambda binary: None)
    monkeypatch.setattr(service.shutil, 'which', lambda binary: None)

    async def _test():
        app = SymfluenceTUI(initial_mode='agent')
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press('c')
            await pilot.pause()
            still_running = not app._exit
        return still_running, app.return_value

    still_running, result = _run(_test())
    assert still_running  # no handoff was attempted
    assert result is None
