# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The global ``--dry-run`` never executes a command for real.

``--dry-run`` is a global flag ("Preview supported operations without making
changes") that only ``workflow clean`` used to read: ``--dry-run workflow run``
ran the workflow (creating the domain directory) and ``--dry-run binary install
summa --force`` deleted and rebuilt SUMMA. Every leaf subcommand is now either
refused under ``--dry-run`` or has declared (``supports_dry_run``) that it
previews / is read-only -- and the declared ones are pinned here, so a new
command cannot gain the mark without a reviewer seeing this file change.
"""
from __future__ import annotations

import argparse
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.argument_parser import CLIParser
from symfluence.cli.commands.base import BaseCommand, dispatch_command
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli]

#: Commands that implement a preview under --dry-run (tested individually below).
PREVIEW = {
    'workflow run', 'workflow step', 'workflow steps', 'workflow resume',
    'workflow status', 'workflow clean',
    'binary install', 'binary install-sysdeps',
    'data download', 'job submit',
}

#: Commands that only read and report, whatever the flag.
READ_ONLY = {
    'workflow validate', 'workflow list-steps',
    'project list-presets', 'project show-preset',
    'binary validate', 'binary doctor', 'binary info',
    'config list-templates', 'config validate', 'config validate-env', 'config resolve',
    'example list', 'list', 'data list', 'data info', 'doctor',
}


def _leaves():
    """(command path, handler) for every leaf of the CLI, incl. bare `agent`."""
    found = {}

    def walk(parser, path):
        subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        func = parser._defaults.get('func')
        if func is not None:
            found[' '.join(path)] = func
        for sub in subs:
            for name, child in sub.choices.items():
                walk(child, path + [name])

    walk(CLIParser().parser, [])
    return found


LEAVES = _leaves()


def test_every_leaf_is_classified():
    declared = {path for path, func in LEAVES.items()
                if getattr(func, 'supports_dry_run', False) is True}
    assert declared == PREVIEW | READ_ONLY, (
        "A command changed its --dry-run declaration. Preview commands need a "
        "test in this file; read-only commands must really be read-only."
    )


@pytest.mark.parametrize("path", sorted(set(LEAVES) - PREVIEW - READ_ONLY))
def test_unsupported_commands_are_refused_not_run(path):
    handler = MagicMock(spec=[])  # no supports_dry_run attribute
    category, _, action = path.partition(' ')
    args = argparse.Namespace(func=handler, dry_run=True, category=category,
                              action=action or None)
    # the real handler object carries no mark either
    assert getattr(LEAVES[path], 'supports_dry_run', False) is not True

    with patch.object(BaseCommand, '_console') as console:
        rc = dispatch_command(args)

    assert rc == ExitCode.USAGE_ERROR
    handler.assert_not_called()
    message = console.error.call_args.args[0]
    assert f"dry-run is not supported for '{path}'" in message


def test_without_dry_run_the_handler_runs():
    handler = MagicMock(spec=[], return_value=0)
    args = argparse.Namespace(func=handler, dry_run=False, category='gui', action='launch')
    assert dispatch_command(args) == 0
    handler.assert_called_once_with(args)


# ---------------------------------------------------------------------------
# Preview commands: run the real handler with the service layer booby-trapped
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path, monkeypatch):
    """A config whose data dir does not exist yet; cwd is an empty directory."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path / "env_data"))
    config = tmp_path / "config.yaml"
    config.write_text(textwrap.dedent(f"""\
        SYMFLUENCE_DATA_DIR: {tmp_path / 'data'}
        SYMFLUENCE_CODE_DIR: {tmp_path / 'code'}
        DOMAIN_NAME: dry
        EXPERIMENT_ID: run_1
        EXPERIMENT_TIME_START: "2020-01-01 00:00"
        EXPERIMENT_TIME_END: "2020-12-31 23:00"
        DOMAIN_DEFINITION_METHOD: lumped
        SUB_GRID_DISCRETIZATION: lumped
        HYDROLOGICAL_MODEL: SUMMA
        FORCING_DATASET: ERA5
    """))
    return tmp_path, config


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def _run(argv):
    args = CLIParser().parse_args(argv)
    assert args.dry_run is True
    return dispatch_command(args)


def _boom(*_args, **_kwargs):
    raise AssertionError("--dry-run must not reach the service layer")


class _NoSymfluence:
    """Stands in for the SYMFLUENCE class: constructing it is the side effect."""

    def __init__(self, *_args, **_kwargs):
        _boom()


@pytest.mark.parametrize("argv", [
    ['workflow', 'run'],
    ['workflow', 'run', '--force-rerun'],
    ['workflow', 'step', 'define_domain'],
    ['workflow', 'steps', 'setup_project', 'run_model'],
    ['workflow', 'resume', 'run_model'],
    ['workflow', 'status'],
    ['workflow', 'clean', '--level', 'all'],
])
def test_workflow_commands_preview_without_side_effects(project, argv):
    """No SYMFLUENCE instance (it creates the domain's log directory), no
    step executed, nothing written anywhere."""
    root, config = project
    before = _tree(root)
    with patch('symfluence.SYMFLUENCE', new=_NoSymfluence), \
            patch('symfluence.project.system.SYMFLUENCE', new=_NoSymfluence), \
            patch.object(BaseCommand, 'confirm_action', side_effect=_boom):
        rc = _run(['--dry-run', *argv, '--config', str(config)])
    assert rc == ExitCode.SUCCESS
    assert _tree(root) == before
    assert not (root / 'data').exists()


def test_workflow_run_preview_lists_steps_and_completion(project):
    root, config = project
    # make the first step look complete to the real completion checks
    (root / 'data' / 'domain_dry' / 'shapefiles').mkdir(parents=True)
    with patch.object(BaseCommand, '_console') as console:
        rc = _run(['--dry-run', 'workflow', 'run', '--config', str(config)])
    assert rc == ExitCode.SUCCESS
    lines = [c.args[0] for c in console.indent.call_args_list]
    step_lines = [ln for ln in lines if 'would run' in ln or 'would skip' in ln]
    from symfluence.workflow_steps import WORKFLOW_STEP_NAMES
    assert len(step_lines) == len(WORKFLOW_STEP_NAMES)
    assert 'setup_project' in step_lines[0]
    assert 'not done' in ' '.join(step_lines)


def test_workflow_preview_rejects_missing_config(project):
    root, _ = project
    rc = _run(['--dry-run', 'workflow', 'run', '--config', str(root / 'nope.yaml')])
    assert rc == ExitCode.CONFIG_ERROR


def test_binary_install_preview_installs_nothing(project):
    """`--dry-run binary install <tool> --force` used to delete and rebuild."""
    root, _ = project
    from symfluence.cli.services.tool_installer import ToolInstaller

    existing = root / 'env_data' / 'installs' / 'TauDEM'
    existing.mkdir(parents=True)
    (existing / 'keep.txt').write_text('x')
    before = _tree(root)
    with patch.object(ToolInstaller, '_clone_repository', side_effect=_boom), \
            patch.object(ToolInstaller, '_run_build_commands', side_effect=_boom), \
            patch.object(ToolInstaller, '_generate_toolchain_metadata', side_effect=_boom), \
            patch('shutil.rmtree', side_effect=_boom), \
            patch.object(ToolInstaller, '_load_config',
                         return_value={"SYMFLUENCE_DATA_DIR": str(root / 'env_data')}):
        rc = _run(['--dry-run', 'binary', 'install', 'taudem', '--force'])
    assert rc == ExitCode.SUCCESS
    assert _tree(root) == before


def test_binary_install_sysdeps_preview_runs_no_installer(project):
    with patch('subprocess.run', side_effect=_boom), \
            patch('subprocess.Popen', side_effect=_boom), \
            patch('symfluence.cli.services.system_deps.SystemDepsRegistry') as registry:
        registry.return_value.platform.value = 'test'
        registry.return_value.check_all_deps.return_value = [
            MagicMock(found=False, display_name='netcdf')]
        registry.return_value.generate_install_script.return_value = 'brew install netcdf'
        rc = _run(['--dry-run', 'binary', 'install-sysdeps'])
    assert rc == ExitCode.SUCCESS


def test_data_download_preview_downloads_nothing(project):
    root, _ = project
    import symfluence.data.acquisition  # noqa: F401 — registers handlers
    from symfluence.data.acquisition.registry import AcquisitionRegistry

    dataset = AcquisitionRegistry.list_datasets()[0]
    before = _tree(root)
    with patch.object(AcquisitionRegistry, 'get_handler', side_effect=_boom):
        rc = _run(['--dry-run', 'data', 'download', dataset, '--bbox', '51/-116/50/-115',
                   '--start', '2020-01-01', '--end', '2020-01-02'])
    assert rc == ExitCode.SUCCESS
    assert _tree(root) == before


def test_job_submit_preview_submits_nothing(project):
    root, _ = project
    before = _tree(root)
    with patch('subprocess.run', side_effect=_boom):
        rc = _run(['--dry-run', 'job', 'submit', '--name', 'j1', 'workflow', 'run'])
    assert rc == ExitCode.SUCCESS
    assert _tree(root) == before


def test_global_dry_run_reaches_install_sysdeps():
    """The subcommand's own --dry-run default used to reset a global one."""
    parser = CLIParser()
    assert parser.parse_args(['--dry-run', 'binary', 'install-sysdeps']).dry_run is True
    assert parser.parse_args(['binary', 'install-sysdeps', '--dry-run']).dry_run is True
    assert getattr(parser.parse_args(['binary', 'install-sysdeps']), 'dry_run', False) is False

