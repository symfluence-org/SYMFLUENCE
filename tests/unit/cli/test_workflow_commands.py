"""Unit tests for workflow command handlers."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from symfluence.cli.commands.workflow_commands import WorkflowCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestWorkflowRun:
    """Test workflow run command."""

    @patch('symfluence.SYMFLUENCE')
    def test_run_success(self, mock_symfluence_class, temp_config_dir):
        """Test successful workflow run."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run(args)

        assert result == ExitCode.SUCCESS
        mock_instance.run_workflow.assert_called_once()

    @patch('symfluence.SYMFLUENCE')
    def test_run_passes_quiet_mode(self, mock_symfluence_class, temp_config_dir):
        """--quiet is plumbed through to the SYMFLUENCE system."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_symfluence_class.return_value = MagicMock()

        args = Namespace(
            config=str(config_file),
            debug=False,
            visualise=False,
            profile=False,
            quiet=True,
        )

        result = WorkflowCommands.run(args)

        assert result == ExitCode.SUCCESS
        assert mock_symfluence_class.call_args.kwargs['quiet_mode'] is True

    @patch('symfluence.SYMFLUENCE')
    def test_run_quiet_defaults_false(self, mock_symfluence_class, temp_config_dir):
        """Without --quiet the system is constructed with quiet_mode=False."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_symfluence_class.return_value = MagicMock()

        args = Namespace(
            config=str(config_file),
            debug=False,
            visualise=False,
            profile=False,
        )

        result = WorkflowCommands.run(args)

        assert result == ExitCode.SUCCESS
        assert mock_symfluence_class.call_args.kwargs['quiet_mode'] is False

    @patch('symfluence.SYMFLUENCE')
    def test_run_missing_config(self, mock_symfluence_class):
        """Test workflow run with missing config file."""
        args = Namespace(
            config='/nonexistent/config.yaml',
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run(args)

        assert result == ExitCode.CONFIG_ERROR
        mock_symfluence_class.assert_not_called()

    @patch('symfluence.SYMFLUENCE')
    def test_run_workflow_error(self, mock_symfluence_class, temp_config_dir):
        """Test workflow run with workflow error."""
        from symfluence.core.exceptions import ModelExecutionError

        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_instance.run_workflow.side_effect = ModelExecutionError("Test workflow error")
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run(args)

        assert result == ExitCode.WORKFLOW_ERROR

    @patch('symfluence.SYMFLUENCE')
    def test_run_config_error(self, mock_symfluence_class, temp_config_dir):
        """Test workflow run with configuration error."""
        from symfluence.core.exceptions import ConfigurationError

        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_symfluence_class.side_effect = ConfigurationError("Invalid config")

        args = Namespace(
            config=str(config_file),
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run(args)

        assert result == ExitCode.CONFIG_ERROR


class TestWorkflowRunStep:
    """Test workflow run-step command."""

    @patch('symfluence.SYMFLUENCE')
    def test_run_step_success(self, mock_symfluence_class, temp_config_dir):
        """Test successful single step execution."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            step_name='setup_project',
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run_step(args)

        assert result == ExitCode.SUCCESS
        mock_instance.run_individual_steps.assert_called_once_with(['setup_project'])

    @patch('symfluence.SYMFLUENCE')
    def test_run_step_file_not_found(self, mock_symfluence_class, temp_config_dir):
        """Test step execution with file not found error."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_instance.run_individual_steps.side_effect = FileNotFoundError("Data file missing")
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            step_name='acquire_forcings',
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run_step(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestWorkflowRunSteps:
    """Test workflow run-steps command."""

    @patch('symfluence.SYMFLUENCE')
    def test_run_steps_success(self, mock_symfluence_class, temp_config_dir):
        """Test successful multiple steps execution."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            step_names=['setup_project', 'define_domain'],
            debug=False,
            visualise=False,
            profile=False
        )

        result = WorkflowCommands.run_steps(args)

        assert result == ExitCode.SUCCESS
        mock_instance.run_individual_steps.assert_called_once_with(['setup_project', 'define_domain'])


class TestWorkflowStatus:
    """Test workflow status command."""

    @patch('symfluence.SYMFLUENCE')
    def test_status_success(self, mock_symfluence_class, temp_config_dir):
        """Test workflow status retrieval."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_instance.get_workflow_status.return_value = "Step 3 of 10 complete"
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            debug=False,
            visualise=False
        )

        result = WorkflowCommands.status(args)

        assert result == ExitCode.SUCCESS
        mock_instance.get_workflow_status.assert_called_once()


class TestWorkflowValidate:
    """Test workflow validate command."""

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_validate_success(self, mock_load_config, temp_config_dir):
        """Test successful config validation."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_load_config.return_value = MagicMock()

        args = Namespace(
            config=str(config_file),
            debug=False
        )

        result = WorkflowCommands.validate(args)

        assert result == ExitCode.SUCCESS

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_validate_invalid_config(self, mock_load_config):
        """Test validation with invalid config."""
        mock_load_config.return_value = None

        args = Namespace(
            config='/some/config.yaml',
            debug=False
        )

        result = WorkflowCommands.validate(args)

        assert result == ExitCode.CONFIG_ERROR


class TestWorkflowListSteps:
    """Test workflow list-steps command."""

    def test_list_steps_success(self):
        """Test listing available workflow steps."""
        args = Namespace()

        result = WorkflowCommands.list_steps(args)

        assert result == ExitCode.SUCCESS

    def test_workflow_steps_defined(self):
        """Test that workflow steps are properly defined."""
        assert len(WorkflowCommands.WORKFLOW_STEPS) > 0
        assert 'setup_project' in WorkflowCommands.WORKFLOW_STEPS
        assert 'run_model' in WorkflowCommands.WORKFLOW_STEPS


class TestWorkflowResume:
    """Test workflow resume command."""

    @patch('symfluence.SYMFLUENCE')
    def test_resume_success(self, mock_symfluence_class, temp_config_dir):
        """Test successful workflow resume."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            step_name='run_model',
            debug=False,
            visualise=False
        )

        result = WorkflowCommands.resume(args)

        assert result == ExitCode.SUCCESS
        # Should have called run_individual_steps with steps from run_model onwards
        mock_instance.run_individual_steps.assert_called_once()
        called_steps = mock_instance.run_individual_steps.call_args[0][0]
        assert called_steps[0] == 'run_model'

    @patch('symfluence.SYMFLUENCE')
    def test_resume_unknown_step(self, mock_symfluence_class, temp_config_dir):
        """Test resume with unknown step name."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        args = Namespace(
            config=str(config_file),
            step_name='nonexistent_step',
            debug=False,
            visualise=False
        )

        result = WorkflowCommands.resume(args)

        assert result == ExitCode.USAGE_ERROR


class TestWorkflowClean:
    """Test workflow clean command."""

    @patch('symfluence.SYMFLUENCE')
    def test_clean_with_method(self, mock_symfluence_class, temp_config_dir):
        """Test clean when clean_workflow_files exists."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            level='intermediate',
            dry_run=False,
            debug=False
        )

        result = WorkflowCommands.clean(args)

        assert result == ExitCode.SUCCESS
        mock_instance.clean_workflow_files.assert_called_once_with(level='intermediate', dry_run=False)

    @patch('symfluence.cli.commands.base.BaseCommand.confirm_action', return_value=False)
    @patch('symfluence.SYMFLUENCE')
    def test_clean_outputs_requires_confirmation(
        self, mock_symfluence_class, mock_confirm, temp_config_dir
    ):
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        args = Namespace(
            config=str(config_file), level='outputs', dry_run=False, debug=False
        )

        result = WorkflowCommands.clean(args)

        assert result == ExitCode.SUCCESS
        mock_confirm.assert_called_once()
        mock_symfluence_class.assert_not_called()

    @patch('symfluence.SYMFLUENCE')
    def test_clean_without_method(self, mock_symfluence_class, temp_config_dir):
        """Test clean when clean_workflow_files doesn't exist (beta message)."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock(spec=[])  # No clean_workflow_files
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            level='all',
            dry_run=True,
            debug=False
        )

        result = WorkflowCommands.clean(args)

        assert result == ExitCode.SUCCESS

    @patch('symfluence.cli.commands.base.BaseCommand.confirm_action', return_value=True)
    @patch('symfluence.SYMFLUENCE')
    def test_clean_permission_error(self, mock_symfluence_class, mock_confirm, temp_config_dir):
        """Test clean with permission error."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_instance = MagicMock()
        mock_instance.clean_workflow_files.side_effect = PermissionError("Cannot delete")
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(
            config=str(config_file),
            level='outputs',
            dry_run=False,
            debug=False
        )

        result = WorkflowCommands.clean(args)

        assert result == ExitCode.PERMISSION_ERROR


class TestWorkflowFailureExitCodes:
    """--continue-on-error / STOP_ON_ERROR=false runs that had failures exit non-zero."""

    @patch('symfluence.SYMFLUENCE')
    def test_run_with_failed_steps_exits_workflow_error(self, mock_symfluence_class, temp_config_dir):
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_instance = MagicMock()
        mock_instance.get_failed_steps.return_value = ["run_model", "calibrate_model"]
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(config=str(config_file), debug=False, visualise=False,
                         profile=False, continue_on_error=True)
        result = WorkflowCommands.run(args)

        assert result == ExitCode.WORKFLOW_ERROR
        overrides = mock_symfluence_class.call_args.kwargs['config_overrides']
        assert overrides['STOP_ON_ERROR'] is False

    @patch('symfluence.SYMFLUENCE')
    def test_steps_with_failed_step_exits_workflow_error(self, mock_symfluence_class, temp_config_dir):
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_instance = MagicMock()
        mock_instance.get_failed_steps.return_value = ["define_domain"]
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(config=str(config_file), debug=False, visualise=False,
                         profile=False, step_names=["define_domain"])
        assert WorkflowCommands.run_steps(args) == ExitCode.WORKFLOW_ERROR


class TestForceRerunPlumbing:
    @patch('symfluence.SYMFLUENCE')
    def test_run_force_rerun_sets_config_override(self, mock_symfluence_class, temp_config_dir):
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_instance = MagicMock()
        mock_instance.get_failed_steps.return_value = []
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(config=str(config_file), debug=False, visualise=False,
                         profile=False, force_rerun=True)
        assert WorkflowCommands.run(args) == ExitCode.SUCCESS
        assert mock_symfluence_class.call_args.kwargs['config_overrides'] == {'FORCE_RUN_ALL_STEPS': True}
        mock_instance.run_workflow.assert_called_once_with(force_run=True)

    @patch('symfluence.SYMFLUENCE')
    def test_run_without_flag_defers_to_config(self, mock_symfluence_class, temp_config_dir):
        """No --force-rerun -> force_run=None so FORCE_RUN_ALL_STEPS in the config applies."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_instance = MagicMock()
        mock_instance.get_failed_steps.return_value = []
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(config=str(config_file), debug=False, visualise=False, profile=False)
        WorkflowCommands.run(args)
        mock_instance.run_workflow.assert_called_once_with(force_run=None)

    @patch('symfluence.SYMFLUENCE')
    def test_resume_force_rerun_sets_config_override(self, mock_symfluence_class, temp_config_dir):
        config_file = temp_config_dir / "config_files" / "config_template.yaml"
        mock_instance = MagicMock()
        mock_instance.get_failed_steps.return_value = []
        mock_symfluence_class.return_value = mock_instance

        args = Namespace(config=str(config_file), debug=False, visualise=False,
                         step_name='calibrate_model', force_rerun=True)
        assert WorkflowCommands.resume(args) == ExitCode.SUCCESS
        assert mock_symfluence_class.call_args.kwargs['config_overrides'] == {'FORCE_RUN_ALL_STEPS': True}
