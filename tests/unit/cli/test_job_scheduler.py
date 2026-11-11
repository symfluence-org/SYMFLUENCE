"""Unit tests for JobScheduler."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from symfluence.cli.services import JobScheduler

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestInitialization:
    """Test JobScheduler initialization."""

    def test_initialization(self):
        """Test JobScheduler creates successfully."""
        scheduler = JobScheduler()
        assert scheduler is not None


class TestSlurmAvailability:
    """Test SLURM availability checking."""

    @patch('shutil.which')
    def test_slurm_available(self, mock_which, job_scheduler):
        """Test SLURM detected when sbatch exists."""
        mock_which.return_value = '/usr/bin/sbatch'

        result = job_scheduler._check_slurm_available()

        assert result is True
        mock_which.assert_called_with('sbatch')

    @patch('shutil.which')
    def test_slurm_not_available(self, mock_which, job_scheduler):
        """Test SLURM not detected when sbatch missing."""
        mock_which.return_value = None

        result = job_scheduler._check_slurm_available()

        assert result is False


class TestEnvironmentDetection:
    """Test HPC vs laptop environment detection."""

    @patch.dict('os.environ', {'SLURM_CLUSTER_NAME': 'cluster1'}, clear=False)
    @patch('shutil.which')
    def test_detect_hpc_via_slurm_env(self, mock_which, job_scheduler):
        """Test HPC detection via SLURM env var."""
        mock_which.return_value = '/usr/bin/sbatch'

        env = job_scheduler.detect_environment()

        assert env == 'hpc'

    @patch('pathlib.Path.exists')
    @patch('shutil.which')
    @patch.dict('os.environ', {}, clear=True)
    def test_detect_hpc_via_scratch_dir(self, mock_which, mock_exists, job_scheduler):
        """Test HPC detection via /scratch directory."""
        mock_which.return_value = '/usr/bin/sbatch'

        def exists_side_effect(path):
            # Return True for /scratch, False for others
            return str(path) == '/scratch'

        mock_exists.side_effect = exists_side_effect

        env = job_scheduler.detect_environment()

        assert env == 'hpc'

    @patch('shutil.which')
    @patch.dict('os.environ', {}, clear=True)
    def test_detect_laptop(self, mock_which, job_scheduler):
        """Test laptop detection when no HPC indicators."""
        mock_which.return_value = None

        env = job_scheduler.detect_environment()

        assert env == 'laptop'


class TestSlurmScriptGeneration:
    """Test SLURM script generation."""

    def test_workflow_mode_script(self, job_scheduler, sample_config):
        """Test script generation for workflow mode."""
        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow'
        }
        slurm_options = {
            'job_name': 'test_job',
            'job_account': 'test_account',
            'job_time': '24:00:00',
            'job_ntasks': 8,
            'job_memory': '32GB'
        }
        config_file = '/path/to/config.yaml'

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        # Assertions
        assert '#SBATCH --job-name=test_job' in script
        assert '#SBATCH --account=test_account' in script
        assert '#SBATCH --time=24:00:00' in script
        assert config_file in script

    def test_individual_steps_script(self, job_scheduler, sample_config):
        """Test script generation for individual steps mode."""
        execution_plan = {
            'mode': 'individual_steps',
            'config': sample_config,
            'job_mode': 'individual_steps',
            'job_steps': ['define_domain', 'discretize_domain']
        }
        slurm_options = {
            'job_name': 'steps_job',
            'job_account': 'test_account',
            'job_time': '12:00:00',
            'job_ntasks': 4,
            'job_memory': '16GB'
        }
        config_file = '/path/to/config.yaml'

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        assert '--define_domain' in script or 'define_domain' in script
        assert '--discretize_domain' in script or 'discretize_domain' in script

    def test_pour_point_setup_script(self, job_scheduler):
        """Test script generation for pour point setup."""
        execution_plan = {
            'mode': 'pour_point_setup',
            'job_mode': 'pour_point_setup',
            'pour_point': {
                'coordinates': '51.0/115.0',
                'domain_name': 'test_domain',
                'domain_definition_method': 'delineate'
            }
        }
        slurm_options = {
            'job_name': 'pour_point_job',
            'job_account': 'test_account',
            'job_time': '06:00:00',
            'job_ntasks': 4,
            'job_memory': '8GB'
        }
        config_file = '/path/to/config.yaml'

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        assert '--pour_point 51.0/115.0' in script or '51.0/115.0' in script
        assert 'test_domain' in script

    def test_script_includes_sbatch_directives(self, job_scheduler, sample_config):
        """Test that script includes all necessary SBATCH directives."""
        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow'
        }
        slurm_options = {
            'job_name': 'test',
            'job_account': 'test_account',
            'job_time': '10:00:00',
            'job_ntasks': 2,
            'job_memory': '4GB',
            'job_partition': 'compute'
        }

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, 'config.yaml'
        )

        assert '#!/bin/bash' in script
        assert '#SBATCH' in script
        assert '#SBATCH --job-name=test' in script
        assert '#SBATCH --partition=compute' in script or script.count('#SBATCH') > 0


class TestJobSubmission:
    """Test SLURM job submission."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_successful_submission(self, mock_which, mock_subprocess, mock_file, job_scheduler, sample_config):
        """Test successful job submission."""
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.return_value = MagicMock(
            stdout='Submitted batch job 12345\n',
            stderr='',
            returncode=0
        )

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow',
            'slurm_options': {
                'job_name': 'test',
                'job_account': 'test',
                'job_time': '10:00:00',
                'job_ntasks': 2,
                'job_memory': '4GB'
            }
        }

        result = job_scheduler.submit_slurm_job(execution_plan)

        assert result['success'] is True
        assert result['job_id'] == '12345'
        assert mock_subprocess.called

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_submission_failure(self, mock_which, mock_subprocess, mock_file, job_scheduler, sample_config):
        """Test handling of submission failure."""
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=['sbatch', 'script.sh'],
            stderr='sbatch: error: Invalid account'
        )

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow',
            'slurm_options': {
                'job_name': 'test',
                'job_account': 'invalid',
                'job_time': '10:00:00',
                'job_ntasks': 2,
                'job_memory': '4GB'
            }
        }

        with pytest.raises(RuntimeError, match="Failed to submit SLURM job"):
            job_scheduler.submit_slurm_job(execution_plan)

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_auto_job_name_generation(self, mock_which, mock_subprocess, mock_file, job_scheduler, sample_config):
        """Test automatic job name generation."""
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.return_value = MagicMock(
            stdout='Submitted batch job 12345\n',
            stderr='',
            returncode=0
        )

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow',
            'slurm_options': {
                'job_account': 'test',
                'job_time': '10:00:00',
                'job_ntasks': 2,
                'job_memory': '4GB'
                # No job_name provided
            }
        }

        result = job_scheduler.submit_slurm_job(execution_plan)

        # Job name should be auto-generated
        assert 'job_name' in result
        assert result['success'] is True

    @patch('shutil.which')
    def test_slurm_not_available_raises_error(self, mock_which, job_scheduler, sample_config):
        """Test that missing SLURM raises error."""
        mock_which.return_value = None

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'slurm_options': {}
        }

        with pytest.raises(RuntimeError, match="SLURM commands .* not available"):
            job_scheduler.submit_slurm_job(execution_plan)


def _proc(stdout=''):
    return MagicMock(stdout=stdout, stderr='', returncode=0)


class TestJobMonitoring:
    """Test SLURM job monitoring."""

    @patch('subprocess.run')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_monitor_until_completion(self, mock_sleep, mock_subprocess, job_scheduler):
        """The queue empties, then accounting says COMPLETED."""
        mock_subprocess.side_effect = [
            _proc('12345 RUNNING\n'),
            _proc('12345 RUNNING\n'),
            _proc(''),               # squeue: no longer queued
            _proc('COMPLETED|0:0\n'),  # sacct
        ]

        assert job_scheduler._monitor_slurm_job('12345') == ('COMPLETED', '0:0')
        assert mock_subprocess.call_args_list[-1].args[0] == [
            'sacct', '-j', '12345', '-X', '-o', 'State,ExitCode', '--parsable2', '-n']

    @pytest.mark.parametrize("sacct_line,state", [
        ('FAILED|1:0', 'FAILED'),
        ('TIMEOUT|0:0', 'TIMEOUT'),
        ('CANCELLED by 1234|0:15', 'CANCELLED'),
        ('OUT_OF_MEMORY|0:125', 'OUT_OF_MEMORY'),
    ])
    @patch('subprocess.run')
    @patch('time.sleep')
    def test_monitor_failure_detection(self, mock_sleep, mock_subprocess, job_scheduler,
                                       sacct_line, state):
        """An empty squeue used to be reported as "completed" whatever happened."""
        mock_subprocess.side_effect = [_proc(''), _proc(sacct_line + '\n')]

        final_state, _ = job_scheduler._monitor_slurm_job('12345')

        assert final_state == state

    @patch('subprocess.run')
    @patch('time.sleep')
    def test_sacct_retries_through_accounting_lag(self, mock_sleep, mock_subprocess,
                                                  job_scheduler):
        mock_subprocess.side_effect = [
            _proc(''), _proc(''), _proc('RUNNING|0:0\n'), _proc('COMPLETED|0:0\n')]

        assert job_scheduler._final_job_state('12345') == ('COMPLETED', '0:0')

    @patch('subprocess.run')
    @patch('time.sleep')
    def test_no_accounting_record_is_not_success(self, mock_sleep, mock_subprocess,
                                                 job_scheduler):
        mock_subprocess.return_value = _proc('')

        assert job_scheduler._final_job_state('12345') == ('UNKNOWN', '')

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_wait_on_failed_job_is_reported_as_failure(self, mock_which, mock_subprocess,
                                                       mock_file, job_scheduler):
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.return_value = _proc('Submitted batch job 777\n')
        plan = {
            'config_file': 'c.yaml', 'job_mode': 'workflow', 'job_steps': [],
            'slurm_options': {
                'job_name': 'j', 'job_time': '01:00:00', 'job_ntasks': 1,
                'job_memory': '4G', 'submit_and_wait': True,
            },
        }
        with patch.object(job_scheduler, '_monitor_slurm_job',
                          return_value=('FAILED', '1:0')):
            assert job_scheduler.handle_slurm_job_submission(plan) is False
        assert job_scheduler.last_result['final_state'] == 'FAILED'
        assert job_scheduler.last_result['job_id'] == '777'


class TestScriptHardening:
    """Generated script: optional activation, quoting, strict mode, templates."""

    OPTS = {'job_name': 'run1', 'job_time': '01:00:00', 'job_ntasks': 1, 'job_memory': '4G'}

    def _script(self, job_scheduler, steps=('workflow', 'run'), **options):
        return job_scheduler._create_symfluence_slurm_script(
            {'job_mode': 'workflow', 'job_steps': list(steps)},
            {**self.OPTS, **options}, 'c.yaml')

    def test_no_activation_lines_unless_configured(self, job_scheduler):
        script = self._script(job_scheduler)
        assert 'module restore' not in script
        assert 'conda activate' not in script

    def test_activation_is_quoted_and_precedes_strict_mode(self, job_scheduler):
        script = self._script(job_scheduler, job_modules='my mods; rm -rf ~',
                              conda_env='env name')
        assert "module restore 'my mods; rm -rf ~'" in script
        assert 'eval "$(conda shell.bash hook)"' in script
        assert "conda activate 'env name'" in script
        assert script.index('conda activate') < script.index('set -euo pipefail')
        assert script.index('set -euo pipefail') < script.index('symfluence workflow run')

    def test_command_failure_reaches_the_exit_code(self, job_scheduler):
        script = self._script(job_scheduler)
        assert 'symfluence workflow run --config c.yaml || exit_code=$?' in script
        assert script.rstrip().endswith('exit $exit_code')

    def test_script_is_valid_bash(self, job_scheduler):
        import shutil
        import subprocess
        bash = shutil.which('bash')
        if bash is None:
            pytest.skip('bash not available')
        script = self._script(job_scheduler, steps=('workflow', 'run', '--config', "my 'cfg'.yaml"),
                              job_modules='m', conda_env='e', job_account='def-x')
        assert subprocess.run([bash, '-n'], input=script, text=True).returncode == 0

    @pytest.mark.parametrize("option,value", [
        ('job_name', 'a b'), ('job_name', 'x\n#SBATCH --wrap=evil'),
        ('job_account', 'acct; touch x'), ('job_partition', '$(id)'),
        ('job_time', '1:00:00 --uid=0'),
    ])
    def test_unsafe_sbatch_values_are_refused(self, job_scheduler, option, value):
        with pytest.raises(ValueError, match='Invalid value for SLURM option'):
            self._script(job_scheduler, **{option: value})

    def test_template_with_placeholder(self, job_scheduler, tmp_path):
        template = tmp_path / 'job.sh'
        template.write_text('#!/bin/bash\n#SBATCH --qos=high\nsource env.sh\n'
                            '{{SYMFLUENCE_COMMAND}}\necho after\n')
        script = self._script(job_scheduler, slurm_template=str(template))
        assert script.startswith('#!/bin/bash\n#SBATCH --qos=high\nsource env.sh\n')
        assert '#SBATCH --job-name' not in script
        assert 'symfluence workflow run --config c.yaml || exit_code=$?' in script
        assert script.rstrip().endswith('echo after')

    def test_template_without_placeholder_gets_command_appended(self, job_scheduler, tmp_path):
        template = tmp_path / 'job.sh'
        template.write_text('#!/bin/bash\n#SBATCH --qos=high\n')
        script = self._script(job_scheduler, slurm_template=str(template))
        assert script.startswith('#!/bin/bash\n#SBATCH --qos=high\n')
        assert script.rstrip().endswith('exit $exit_code')

    def test_missing_template_is_an_error(self, job_scheduler, tmp_path):
        with pytest.raises(FileNotFoundError, match='SLURM template not found'):
            self._script(job_scheduler, slurm_template=str(tmp_path / 'nope.sh'))

    def test_dry_run_writes_and_submits_nothing(self, job_scheduler, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plan = {'config_file': 'c.yaml', 'job_mode': 'workflow', 'job_steps': [],
                'dry_run': True, 'slurm_options': dict(self.OPTS)}
        with patch('subprocess.run') as run:
            assert job_scheduler.handle_slurm_job_submission(plan) is True
        run.assert_not_called()
        assert list(tmp_path.iterdir()) == []


class TestHandleSlurmJobSubmission:
    """Test high-level SLURM job submission handler."""

    @patch.object(JobScheduler, 'submit_slurm_job')
    def test_successful_submission_workflow(self, mock_submit, job_scheduler):
        """Test successful submission workflow."""
        mock_submit.return_value = {
            'success': True,
            'job_id': '12345',
            'job_name': 'test_job'
        }

        execution_plan = {
            'slurm_options': {
                'job_account': 'test'
            }
        }

        result = job_scheduler.handle_slurm_job_submission(execution_plan)

        assert result is True
        mock_submit.assert_called_once()

    @patch.object(JobScheduler, 'submit_slurm_job')
    def test_failed_submission_workflow(self, mock_submit, job_scheduler):
        """Test failed submission workflow."""
        mock_submit.return_value = {
            'success': False,
            'error': 'Submission failed'
        }

        execution_plan = {
            'slurm_options': {}
        }

        result = job_scheduler.handle_slurm_job_submission(execution_plan)

        assert result is False

    @patch.object(JobScheduler, 'submit_slurm_job')
    def test_exception_handling(self, mock_submit, job_scheduler):
        """Test exception handling in submission workflow."""
        mock_submit.side_effect = RuntimeError("SLURM error")

        execution_plan = {
            'slurm_options': {}
        }

        result = job_scheduler.handle_slurm_job_submission(execution_plan)

        assert result is False
