"""
Unit test fixtures and configuration.

Fixtures specific to unit tests (fast, isolated tests).
"""

from unittest.mock import MagicMock

import pytest


# Auto-apply the "unit" marker to every test collected under tests/unit/
# so that ``pytest -m unit`` picks up the full suite without requiring
# each test file to declare the marker explicitly.
def pytest_collection_modifyitems(items):
    unit_marker = pytest.mark.unit
    for item in items:
        if "unit" not in {m.name for m in item.iter_markers()}:
            item.add_marker(unit_marker)

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
