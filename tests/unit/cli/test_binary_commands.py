"""Unit tests for binary command handlers."""
from __future__ import annotations

import subprocess
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.commands.binary_commands import BinaryCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestBinaryInstall:
    """Test binary install command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_all_success(self, mock_binary_manager_class):
        """Test successful installation of all tools."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=None,  # Install all
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.SUCCESS
        mock_manager.get_executables.assert_called_once_with(
            specific_tools=None, force=False, dry_run=False, patched=False,
            branch_override=None, git_hash=None,
        )

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_specific_tools(self, mock_binary_manager_class):
        """Test installation of specific tools."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa', 'mizuroute'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.SUCCESS
        mock_manager.get_executables.assert_called_once_with(
            specific_tools=['summa', 'mizuroute'],
            force=False,
            dry_run=False,
            patched=False,
            branch_override=None,
            git_hash=None,
        )

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_force_reinstall(self, mock_binary_manager_class):
        """Test force reinstall of tools."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['taudem'],
            force=True,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.SUCCESS
        mock_manager.get_executables.assert_called_once_with(
            specific_tools=['taudem'], force=True, dry_run=False, patched=False,
            branch_override=None, git_hash=None,
        )

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_dry_run_is_passed_to_the_installer(self, mock_binary_manager_class):
        """--dry-run used to be dropped here: `--dry-run binary install summa
        --force` deleted and rebuilt the tool."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = {
            "successful": ["summa (dry run)"], "failed": [], "skipped": [], "errors": [],
        }
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(tools=['summa'], force=True, debug=False, dry_run=True)

        assert BinaryCommands.install(args) == ExitCode.SUCCESS
        assert mock_manager.get_executables.call_args.kwargs["dry_run"] is True

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_unknown_tool_exits_non_zero(self, mock_binary_manager_class):
        """An unknown tool only added to `errors`; the command exited 0."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = {
            "successful": [], "failed": ["zzz"], "skipped": [],
            "errors": ["Unknown tool: zzz"],
        }
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(tools=['summa', 'zzz'], force=False, debug=False)

        assert BinaryCommands.install(args) == ExitCode.BINARY_ERROR

    @pytest.mark.parametrize("extra", [{"branch": "develop"}, {"git_hash": "deadbeef"}])
    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_branch_or_hash_without_named_tools_is_refused(
        self, mock_binary_manager_class, extra
    ):
        mock_manager = MagicMock()
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(tools=[], force=False, debug=False, **extra)

        assert BinaryCommands.install(args) == ExitCode.USAGE_ERROR
        mock_manager.get_executables.assert_not_called()

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_failure(self, mock_binary_manager_class):
        """Test installation failure."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.BINARY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_build_error(self, mock_binary_manager_class):
        """Test installation with build error."""
        mock_manager = MagicMock()
        mock_manager.get_executables.side_effect = subprocess.CalledProcessError(1, 'make')
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.BINARY_BUILD_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_permission_error(self, mock_binary_manager_class):
        """Test installation with permission error."""
        mock_manager = MagicMock()
        mock_manager.get_executables.side_effect = PermissionError("Cannot write to install dir")
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.PERMISSION_ERROR


class TestBinaryValidate:
    """Test binary validate command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_success(self, mock_binary_manager_class):
        """Test successful binary validation."""
        mock_manager = MagicMock()
        mock_manager.validate_binaries.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            verbose=False,
            debug=False
        )

        result = BinaryCommands.validate(args)

        assert result == ExitCode.SUCCESS
        mock_manager.validate_binaries.assert_called_once_with(
            verbose=False, required_tools=None
        )

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_failure_returns_error_for_results_dict(
        self, mock_binary_manager_class
    ):
        """A results dict means failure, even though a non-empty dict is truthy.

        Regression test: validate_binaries returns True on success or a results
        dict on failure, and the command tested that value for truthiness. The
        dict is never empty, so failures were reported as successes and
        `binary validate` exited 0 over a broken install. `False` — what the
        sibling failure test mocks — is falsy, so it never caught this.
        """
        mock_manager = MagicMock()
        mock_manager.validate_binaries.return_value = {
            "valid_tools": ["summa"],
            "missing_tools": ["fuse"],
            "failed_tools": [],
            "skipped_tools": [],
            "warnings": [],
            "summary": {},
        }
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(verbose=False, debug=False)

        result = BinaryCommands.validate(args)

        assert result == ExitCode.BINARY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_paper_repro_requires_the_paper_tools(
        self, mock_binary_manager_class
    ):
        """--paper-repro forwards the paper tool set as required."""
        from symfluence.cli.argument_parser import PAPER_REPRO_TOOLS

        mock_manager = MagicMock()
        mock_manager.validate_binaries.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(verbose=False, debug=False, paper_repro=True)

        result = BinaryCommands.validate(args)

        assert result == ExitCode.SUCCESS
        mock_manager.validate_binaries.assert_called_once_with(
            verbose=False, required_tools=list(PAPER_REPRO_TOOLS)
        )

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_failure(self, mock_binary_manager_class):
        """Test binary validation failure."""
        mock_manager = MagicMock()
        mock_manager.validate_binaries.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            verbose=True,
            debug=False
        )

        result = BinaryCommands.validate(args)

        assert result == ExitCode.BINARY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_binary_not_found(self, mock_binary_manager_class):
        """Test validation when binary not found."""
        mock_manager = MagicMock()
        mock_manager.validate_binaries.side_effect = FileNotFoundError("summa.exe not found")
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            verbose=False,
            debug=False
        )

        result = BinaryCommands.validate(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestBinaryDoctor:
    """Test binary doctor command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_doctor_success(self, mock_binary_manager_class):
        """Test successful diagnostics."""
        mock_manager = MagicMock()
        mock_manager.doctor.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.doctor(args)

        assert result == ExitCode.SUCCESS
        mock_manager.doctor.assert_called_once()

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_doctor_issues_found(self, mock_binary_manager_class):
        """Test diagnostics with issues found."""
        mock_manager = MagicMock()
        mock_manager.doctor.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.doctor(args)

        assert result == ExitCode.DEPENDENCY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_doctor_import_error(self, mock_binary_manager_class):
        """Test diagnostics with import error."""
        mock_binary_manager_class.side_effect = ImportError("Cannot import BinaryManager")

        args = Namespace(debug=False)

        result = BinaryCommands.doctor(args)

        assert result == ExitCode.DEPENDENCY_ERROR


class TestBinaryInfo:
    """Test binary info command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_info_success(self, mock_binary_manager_class):
        """Test successful tools info display."""
        mock_manager = MagicMock()
        mock_manager.tools_info.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.info(args)

        assert result == ExitCode.SUCCESS
        mock_manager.tools_info.assert_called_once()

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_info_failure(self, mock_binary_manager_class):
        """Test tools info failure."""
        mock_manager = MagicMock()
        mock_manager.tools_info.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.info(args)

        assert result == ExitCode.GENERAL_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_info_directory_not_found(self, mock_binary_manager_class):
        """Test info when tools directory not found."""
        mock_manager = MagicMock()
        mock_manager.tools_info.side_effect = FileNotFoundError("Tools dir not found")
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.info(args)

        assert result == ExitCode.FILE_NOT_FOUND
