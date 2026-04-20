# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Regression tests for external build-tool registrations.

Pins reproducibility-critical properties (e.g. TauDEM pinned to a tagged
release) so that accidental reversions to floating HEAD are caught in CI.
"""

import pytest

from symfluence.cli.external_tools_config import get_external_tools_definitions

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def test_taudem_pinned_to_release_tag():
    """TauDEM must be pinned to a tagged release, not HEAD.

    AI/IA/AP reported TauDEM build failing intermittently with 'pitremove'
    missing. Root cause: upstream HEAD drift. Pinning to a tag (git clone -b
    works for both branches and tags) freezes the install against a known-good
    version. Dropping the pin would regress to floating HEAD.
    """
    tools = get_external_tools_definitions()
    assert 'taudem' in tools, "TauDEM build instructions must be registered"
    spec = tools['taudem']
    assert spec.get('branch'), (
        "TauDEM must pin 'branch' to a tagged release (e.g. 'v5.4.0'); "
        "a None branch lets upstream HEAD drift break installs."
    )
    # Must look like a release tag, not a moving branch name
    branch = str(spec['branch'])
    assert branch.startswith('v') and any(c.isdigit() for c in branch), (
        f"TauDEM branch '{branch}' does not look like a release tag. "
        "Pin to something like 'v5.4.0'."
    )
