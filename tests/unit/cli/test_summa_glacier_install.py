"""Exercise glacier target discovery, isolation and installer revision selection."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from symfluence.cli.external_tools_config import get_external_tools_definitions
from symfluence.cli.services.tool_installer import ToolInstaller

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def test_glacier_target_is_discoverable_and_isolated():
    tools = get_external_tools_definitions()
    glacier, standard = tools['summa-glacier'], tools['summa']
    assert glacier['optional'] is True
    assert glacier['install_dir'] != standard['install_dir']
    assert glacier['default_path_suffix'] != standard['default_path_suffix']
    assert glacier['repository'] == 'https://github.com/ashleymedin/summa.git'
    assert len(glacier['git_hash']) == 40
    assert not glacier['requires']


@pytest.mark.parametrize('branch,override,expected', [
    (None, None, 'pinned'),
    ('other-branch', None, None),
    (None, 'explicit', 'explicit'),
    ('other-branch', 'explicit', 'explicit'),
])
def test_installer_passes_selected_revision(tmp_path, monkeypatch, branch, override, expected):
    installer = ToolInstaller(external_tools={'test': {
        'repository': 'https://example.invalid/test.git',
        'branch': 'main', 'git_hash': 'pinned',
    }})
    monkeypatch.setattr(installer, '_load_config', lambda _: {})
    monkeypatch.setattr(installer, '_resolve_data_dir', lambda _: (tmp_path, 'test'))
    clone = Mock(return_value=False)  # Stop before any build or network access.
    monkeypatch.setattr(installer, '_clone_repository', clone)
    installer.install(specific_tools=['test'], branch_override=branch, git_hash=override)
    assert clone.call_args.kwargs['git_hash'] == expected
    assert clone.call_args.args[1] == (branch or 'main')
