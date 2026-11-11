"""Unit tests for job command handlers."""
from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.commands.job_commands import JobCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestJobSubmit:
    """Test job submit command."""

    @patch('symfluence.cli.services.JobScheduler')
    def test_submit_success(self, mock_job_scheduler_class, temp_config_dir):
        """Test successful job submission."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_scheduler = MagicMock()
        mock_scheduler.handle_slurm_job_submission.return_value = True
        mock_job_scheduler_class.return_value = mock_scheduler

        args = Namespace(
            config=str(config_file),
            job_name='test_job',
            job_time='24:00:00',
            job_nodes=1,
            job_ntasks=4,
            job_memory='32G',
            job_account='def-account',
            job_partition=None,
            job_modules='symfluence_modules',
            conda_env='symfluence',
            submit_and_wait=False,
            slurm_template=None,
            workflow_args=['workflow', 'run'],
            debug=False
        )

        result = JobCommands.submit(args)

        assert result == ExitCode.SUCCESS
        mock_scheduler.handle_slurm_job_submission.assert_called_once()

    @patch('symfluence.cli.services.JobScheduler')
    def test_submit_and_wait(self, mock_job_scheduler_class, temp_config_dir):
        """Test job submission with wait option."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_scheduler = MagicMock()
        mock_scheduler.handle_slurm_job_submission.return_value = True
        mock_job_scheduler_class.return_value = mock_scheduler

        args = Namespace(
            config=str(config_file),
            job_name='test_job',
            job_time='48:00:00',
            job_nodes=2,
            job_ntasks=8,
            job_memory='64G',
            job_account=None,
            job_partition='compute',
            job_modules='symfluence_modules',
            conda_env='symfluence',
            submit_and_wait=True,
            slurm_template=None,
            workflow_args=['workflow', 'run'],
            debug=False
        )

        result = JobCommands.submit(args)

        assert result == ExitCode.SUCCESS

    @patch('symfluence.cli.services.JobScheduler')
    def test_submit_failure(self, mock_job_scheduler_class, temp_config_dir):
        """Test job submission failure."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_scheduler = MagicMock()
        mock_scheduler.handle_slurm_job_submission.return_value = False
        mock_job_scheduler_class.return_value = mock_scheduler

        args = Namespace(
            config=str(config_file),
            job_name='test_job',
            job_time='24:00:00',
            job_nodes=1,
            job_ntasks=1,
            job_memory='50G',
            job_account=None,
            job_partition=None,
            job_modules='symfluence_modules',
            conda_env='symfluence',
            submit_and_wait=False,
            slurm_template=None,
            workflow_args=[],
            debug=False
        )

        result = JobCommands.submit(args)

        assert result == ExitCode.JOB_SUBMIT_ERROR

    @patch('symfluence.cli.services.JobScheduler')
    def test_submit_import_error(self, mock_job_scheduler_class):
        """Test job submission with import error."""
        mock_job_scheduler_class.side_effect = ImportError("Cannot import JobScheduler")

        args = Namespace(
            config='/some/config.yaml',
            job_name='test_job',
            job_time='24:00:00',
            job_nodes=1,
            job_ntasks=1,
            job_memory='50G',
            job_account=None,
            job_partition=None,
            job_modules='symfluence_modules',
            conda_env='symfluence',
            submit_and_wait=False,
            slurm_template=None,
            workflow_args=[],
            debug=False
        )

        result = JobCommands.submit(args)

        assert result == ExitCode.DEPENDENCY_ERROR

    @patch('symfluence.cli.services.JobScheduler')
    def test_submit_permission_error(self, mock_job_scheduler_class, temp_config_dir):
        """Test job submission with permission error."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_scheduler = MagicMock()
        mock_scheduler.handle_slurm_job_submission.side_effect = PermissionError(
            "Cannot write to job directory"
        )
        mock_job_scheduler_class.return_value = mock_scheduler

        args = Namespace(
            config=str(config_file),
            job_name='test_job',
            job_time='24:00:00',
            job_nodes=1,
            job_ntasks=1,
            job_memory='50G',
            job_account=None,
            job_partition=None,
            job_modules='symfluence_modules',
            conda_env='symfluence',
            submit_and_wait=False,
            slurm_template=None,
            workflow_args=[],
            debug=False
        )

        result = JobCommands.submit(args)

        assert result == ExitCode.PERMISSION_ERROR

    @patch('symfluence.cli.services.JobScheduler')
    def test_submit_file_not_found(self, mock_job_scheduler_class, temp_config_dir):
        """Test job submission with file not found error."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_scheduler = MagicMock()
        mock_scheduler.handle_slurm_job_submission.side_effect = FileNotFoundError(
            "SLURM template not found"
        )
        mock_job_scheduler_class.return_value = mock_scheduler

        args = Namespace(
            config=str(config_file),
            job_name='test_job',
            job_time='24:00:00',
            job_nodes=1,
            job_ntasks=1,
            job_memory='50G',
            job_account=None,
            job_partition=None,
            job_modules='symfluence_modules',
            conda_env='symfluence',
            submit_and_wait=False,
            slurm_template='/nonexistent/template.sh',
            workflow_args=[],
            debug=False
        )

        result = JobCommands.submit(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestJobSubmitWaitOutcome:
    """`job submit --wait` exits non-zero when the job did not complete."""

    def _args(self):
        return Namespace(
            config='c.yaml', job_name='j', job_time='01:00:00', job_nodes=1,
            job_ntasks=1, job_memory='4G', job_account=None, job_partition=None,
            job_modules=None, conda_env=None, submit_and_wait=True,
            slurm_template=None, workflow_args=['workflow', 'run'], debug=False,
        )

    @patch('symfluence.cli.services.JobScheduler')
    def test_failed_job_maps_to_job_execution_error(self, mock_job_scheduler_class):
        scheduler = MagicMock()
        scheduler.handle_slurm_job_submission.return_value = False
        scheduler.last_result = {'job_id': '9', 'final_state': 'TIMEOUT', 'success': False}
        mock_job_scheduler_class.return_value = scheduler

        assert JobCommands.submit(self._args()) == ExitCode.JOB_EXECUTION_ERROR

    @patch('symfluence.cli.services.JobScheduler')
    def test_no_environment_activation_by_default(self, mock_job_scheduler_class):
        from symfluence.cli.argument_parser import CLIParser

        scheduler = MagicMock()
        scheduler.handle_slurm_job_submission.return_value = True
        mock_job_scheduler_class.return_value = scheduler
        args = CLIParser().parse_args(['job', 'submit', 'workflow', 'run'])

        assert JobCommands.submit(args) == ExitCode.SUCCESS
        options = scheduler.handle_slurm_job_submission.call_args.args[0]['slurm_options']
        assert options['job_modules'] is None
        assert options['conda_env'] is None

