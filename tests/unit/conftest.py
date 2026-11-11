"""
Unit test fixtures and configuration.

Fixtures specific to unit tests (fast, isolated tests).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Auto-apply the "unit" marker to every test collected under tests/unit/
# so that ``pytest -m unit`` picks up the full suite without requiring
# each test file to declare the marker explicitly.
#
# The hook receives *every* collected item for the session (not just those
# under this directory), so it must filter by path; otherwise ``pytest -m unit
# tests/`` would also select integration and e2e tests.
_UNIT_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    unit_marker = pytest.mark.unit
    for item in items:
        if not Path(str(item.path)).resolve().is_relative_to(_UNIT_ROOT):
            continue
        if "unit" not in {m.name for m in item.iter_markers()}:
            item.add_marker(unit_marker)


@pytest.fixture(autouse=True)
def _isolate_npm_bundle(monkeypatch, tmp_path):
    """Keep executable resolution off any real ``npm install -g symfluence``.

    ``get_model_executable`` falls back to the npm bundle when the source
    install is absent (#156 G6), and ``npm_bundle_bin()`` finds it via
    ``npm root -g`` whenever ``SYMFLUENCE_NPM_DIST_BIN`` is unset. On a
    developer machine that has the npm package installed, that fallback reaches
    a real ``dist/bin`` full of real binaries, so a test asserting "this
    executable is missing" resolves one instead — passing in CI and failing
    locally. Pointing the override at a path that does not exist makes
    ``npm_bundle_bin()`` return None (its documented contract for a
    non-directory override) and also spares every such test the ``npm root -g``
    subprocess. Tests that exercise the fallback set the variable themselves in
    the test body, which runs after this fixture and wins.
    """
    monkeypatch.setenv("SYMFLUENCE_NPM_DIST_BIN", str(tmp_path / "no-npm-bundle"))


# ============================================================================
# Common Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a basic mock configuration for unit tests."""
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/test',
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp'
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for unit tests."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _isolate_agent_cache(monkeypatch, tmp_path):
    """Keep agent approvals, jobs and skills out of the real user cache.

    ``agent_cache_root()`` resolves to a per-user directory under the user's
    cache dir; ``SYMFLUENCE_AGENT_CACHE_DIR`` redirects it so no test reads or
    writes the developer's actual agent state.
    """
    monkeypatch.setenv("SYMFLUENCE_AGENT_CACHE_DIR", str(tmp_path / "agent-cache"))
