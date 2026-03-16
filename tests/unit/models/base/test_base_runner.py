"""
Unit tests for BaseModelRunner.

Tests for the shared model runner infrastructure including:
- Installation path resolution
- Subprocess execution (execute_subprocess and legacy execute_model_subprocess)
- SLURM method accessibility
- File verification
- Configuration path resolution
- Output verification
- Experiment directory management
- Legacy path aliases
"""

import logging
import os
import subprocess
import warnings
from unittest.mock import Mock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.execution.model_executor import (
    ExecutionMode,
    ExecutionResult,
    SlurmJobConfig,
    augment_conda_library_paths,
)


class ConcreteModelRunner(BaseModelRunner):
    """Concrete implementation of BaseModelRunner for testing."""

    MODEL_NAME = "TEST_MODEL"


def _create_config(tmp_path, overrides=None):
    """Create a valid SymfluenceConfig with required fields."""
    base = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'exp_001',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_DATASET': 'ERA5',
    }
    if overrides:
        base.update(overrides)
    return SymfluenceConfig(**base)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path




@pytest.fixture
def base_config(temp_dir):
    """Create a base configuration for testing."""
    return _create_config(temp_dir)


@pytest.fixture
def runner(base_config, mock_logger, temp_dir):
    """Create a ConcreteModelRunner instance."""
    # Create required directories
    data_dir = base_config.system.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    return ConcreteModelRunner(base_config, mock_logger)


class TestGetInstallPath:
    """Tests for get_install_path method."""

    def test_default_path_data_dir(self, runner, temp_dir):
        """Test default installation path relative to data_dir."""
        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin'
        )

        expected = temp_dir / 'data' / 'installs' / 'test_model' / 'bin'
        assert result.resolve() == expected.resolve()

    def test_default_path_project_dir(self, runner, temp_dir):
        """Test default installation path relative to project_dir."""
        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'custom/path',
            relative_to='project_dir'
        )

        expected = temp_dir / 'data' / 'domain_test_domain' / 'custom' / 'path'
        assert result.resolve() == expected.resolve()

    def test_custom_path(self, runner, temp_dir):
        """Test custom installation path from config."""
        custom_path = (temp_dir / 'custom_install').resolve()

        # Re-init runner with custom config
        config = _create_config(temp_dir, {'TEST_INSTALL_PATH': str(custom_path)})
        runner = ConcreteModelRunner(config, runner.logger)

        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin'
        )

        assert result.resolve() == custom_path

    def test_none_uses_default(self, runner, temp_dir):
        """Test that None config value uses default path."""
        # By default the key is missing in base_config, which is effectively None/default behavior
        # But let's be explicit with None if possible, or just rely on absence

        # In SymfluenceConfig, if key is not defined, it won't be in config_dict unless it's a model field
        # For TEST_INSTALL_PATH, it's an extra field.

        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin'
        )

        expected = temp_dir / 'data' / 'installs' / 'test_model' / 'bin'
        assert result.resolve() == expected.resolve()

    def test_must_exist_valid(self, runner, temp_dir):
        """Test must_exist parameter with existing path."""
        install_path = temp_dir / 'data' / 'installs' / 'test_model' / 'bin'
        install_path.mkdir(parents=True, exist_ok=True)

        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin',
            must_exist=True
        )

        assert result.resolve() == install_path.resolve()

    def test_must_exist_raises_error(self, runner):
        """Test must_exist parameter raises error for non-existent path."""
        with pytest.raises(FileNotFoundError) as exc_info:
            runner.get_install_path(
                'TEST_INSTALL_PATH',
                'nonexistent/path',
                must_exist=True
            )

        assert 'Installation path not found' in str(exc_info.value)
        assert 'TEST_INSTALL_PATH' in str(exc_info.value)


class TestExecuteModelSubprocess:
    """Tests for execute_model_subprocess method."""

    def test_success(self, runner, temp_dir, mock_logger):
        """Test successful subprocess execution."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file
            )

            assert result.returncode == 0
            mock_logger.log.assert_any_call(logging.INFO, "Model execution completed successfully")

    def test_custom_success_message(self, runner, temp_dir, mock_logger):
        """Test custom success message."""
        log_file = temp_dir / 'test.log'
        custom_message = "Custom success!"

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file,
                success_message=custom_message
            )

            mock_logger.log.assert_any_call(logging.INFO, custom_message)

    def test_nonzero_return_code_with_check_false(self, runner, temp_dir, mock_logger):
        """Test non-zero return code when check=False."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            result = runner.execute_model_subprocess(
                ['false'],
                log_file,
                check=False
            )

            assert result.returncode == 1
            mock_logger.debug.assert_called_with("Process exited with code 1")

    def test_failure_with_check_true(self, runner, temp_dir, mock_logger):
        """Test subprocess failure when check=True."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'test_cmd')

            with pytest.raises(subprocess.CalledProcessError):
                runner.execute_model_subprocess(
                    ['false'],
                    log_file,
                    check=True
                )

            mock_logger.error.assert_called()

    def test_error_context_logged(self, runner, temp_dir, mock_logger):
        """Test that error context is logged on failure."""
        log_file = temp_dir / 'test.log'
        error_context = {
            'binary_path': '/usr/bin/test',
            'ld_library_path': '/usr/lib'
        }

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'test_cmd')

            with pytest.raises(subprocess.CalledProcessError):
                runner.execute_model_subprocess(
                    ['false'],
                    log_file,
                    error_context=error_context
                )

            # Check that error context was logged
            assert mock_logger.error.call_count >= 2

    def test_timeout_expired(self, runner, temp_dir, mock_logger):
        """Test subprocess timeout."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('test_cmd', 10)

            with pytest.raises(subprocess.TimeoutExpired):
                runner.execute_model_subprocess(
                    ['sleep', '100'],
                    log_file,
                    timeout=10
                )

            # Verify timeout error was logged
            timeout_logged = any(
                'timeout' in str(call).lower()
                for call in mock_logger.error.call_args_list
            )
            assert timeout_logged

    def test_environment_variables_merged(self, runner, temp_dir):
        """Test that environment variables are properly merged."""
        log_file = temp_dir / 'test.log'
        custom_env = {'CUSTOM_VAR': 'custom_value'}

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file,
                env=custom_env
            )

            # Verify subprocess.run was called with merged environment
            call_kwargs = mock_run.call_args[1]
            assert 'CUSTOM_VAR' in call_kwargs['env']
            assert call_kwargs['env']['CUSTOM_VAR'] == 'custom_value'

    def test_log_directory_created(self, runner, temp_dir):
        """Test that log directory is created if it doesn't exist."""
        log_file = temp_dir / 'subdir' / 'nested' / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file
            )

            assert log_file.parent.exists()


class TestVerifyRequiredFiles:
    """Tests for verify_required_files method."""

    def test_all_files_exist(self, runner, temp_dir, mock_logger):
        """Test verification when all files exist."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'
        file1.write_text('test')
        file2.write_text('test')

        # Should not raise
        runner.verify_required_files([file1, file2], context="testing")

        # Should log debug message
        assert mock_logger.debug.called

    def test_single_file_exists(self, runner, temp_dir):
        """Test verification with single file path."""
        file1 = temp_dir / 'file1.txt'
        file1.write_text('test')

        # Should not raise
        runner.verify_required_files(file1, context="testing")

    def test_missing_files_raises_error(self, runner, temp_dir, mock_logger):
        """Test that missing files raise FileNotFoundError."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'  # Does not exist

        with pytest.raises(FileNotFoundError) as exc_info:
            runner.verify_required_files([file1, file2], context="testing")

        error_msg = str(exc_info.value)
        assert 'testing' in error_msg
        assert str(file1) in error_msg or str(file2) in error_msg
        mock_logger.error.assert_called()

    def test_all_missing_files_in_error(self, runner, temp_dir):
        """Test that all missing files are listed in error."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'
        file3 = temp_dir / 'file3.txt'

        with pytest.raises(FileNotFoundError) as exc_info:
            runner.verify_required_files([file1, file2, file3], context="testing")

        error_msg = str(exc_info.value)
        assert str(file1) in error_msg
        assert str(file2) in error_msg
        assert str(file3) in error_msg


class TestGetConfigPath:
    """Tests for get_config_path method."""

    def test_default_path(self, runner, temp_dir):
        """Test config path resolution with default."""
        result = runner.get_config_path(
            'TEST_CONFIG_PATH',
            'settings/test_model'
        )

        expected = temp_dir / 'data' / 'domain_test_domain' / 'settings' / 'test_model'
        assert result.resolve() == expected.resolve()

    def test_custom_path(self, runner, temp_dir):
        """Test config path resolution with custom path."""
        custom_path = (temp_dir / 'custom_settings').resolve()

        # Re-init
        config = _create_config(temp_dir, {'TEST_CONFIG_PATH': str(custom_path)})
        runner = ConcreteModelRunner(config, runner.logger)

        result = runner.get_config_path(
            'TEST_CONFIG_PATH',
            'settings/test_model'
        )

        assert result.resolve() == custom_path


class TestVerifyModelOutputs:
    """Tests for verify_model_outputs method."""

    def test_all_outputs_exist(self, runner, temp_dir, mock_logger):
        """Test verification when all output files exist."""
        runner.output_dir = temp_dir / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)

        (runner.output_dir / 'output1.nc').write_text('test')
        (runner.output_dir / 'output2.nc').write_text('test')

        result = runner.verify_model_outputs(['output1.nc', 'output2.nc'])

        assert result is True
        mock_logger.debug.assert_called()

    def test_single_output_exists(self, runner, temp_dir):
        """Test verification with single output file."""
        runner.output_dir = temp_dir / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)
        (runner.output_dir / 'output.nc').write_text('test')

        result = runner.verify_model_outputs('output.nc')

        assert result is True

    def test_missing_outputs_returns_false(self, runner, temp_dir, mock_logger):
        """Test that missing outputs return False."""
        runner.output_dir = temp_dir / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)

        (runner.output_dir / 'output1.nc').write_text('test')
        # output2.nc does not exist

        result = runner.verify_model_outputs(['output1.nc', 'output2.nc'])

        assert result is False
        mock_logger.error.assert_called()

    def test_custom_output_dir(self, runner, temp_dir):
        """Test verification with custom output directory."""
        custom_dir = temp_dir / 'custom_output'
        custom_dir.mkdir(parents=True, exist_ok=True)
        (custom_dir / 'output.nc').write_text('test')

        result = runner.verify_model_outputs('output.nc', output_dir=custom_dir)

        assert result is True


class TestGetExperimentOutputDir:
    """Tests for get_experiment_output_dir method."""

    def test_default_experiment_id(self, runner, temp_dir):
        """Test experiment output directory with default experiment ID."""
        result = runner.get_experiment_output_dir()

        expected = temp_dir / 'data' / 'domain_test_domain' / 'simulations' / 'exp_001' / 'TEST_MODEL'
        assert result.resolve() == expected.resolve()

    def test_custom_experiment_id(self, runner, temp_dir):
        """Test experiment output directory with custom experiment ID."""
        result = runner.get_experiment_output_dir(experiment_id='exp_custom')

        expected = temp_dir / 'data' / 'domain_test_domain' / 'simulations' / 'exp_custom' / 'TEST_MODEL'
        assert result.resolve() == expected.resolve()


class TestSetupPathAliases:
    """Tests for setup_path_aliases method."""

    def test_valid_aliases(self, runner, mock_logger):
        """Test setting up valid path aliases."""
        runner.setup_path_aliases({
            'root_path': 'data_dir',
            'result_dir': 'output_dir'
        })

        assert hasattr(runner, 'root_path')
        assert runner.root_path == runner.data_dir
        assert hasattr(runner, 'result_dir')
        assert runner.result_dir == runner.output_dir

    def test_invalid_source_attribute(self, runner, mock_logger):
        """Test handling of invalid source attribute."""
        runner.setup_path_aliases({
            'test_alias': 'nonexistent_attr'
        })

        assert not hasattr(runner, 'test_alias')
        mock_logger.warning.assert_called()

    def test_multiple_aliases(self, runner):
        """Test setting up multiple aliases at once."""
        runner.setup_path_aliases({
            'alias1': 'data_dir',
            'alias2': 'project_dir',
            'alias3': 'model_name'
        })

        assert runner.alias1 == runner.data_dir
        assert runner.alias2 == runner.project_dir
        assert runner.alias3 == runner.model_name


class TestBaseRunnerIntegration:
    """Integration tests for BaseModelRunner."""

    def test_initialization_sequence(self, base_config, mock_logger, temp_dir):
        """Test complete initialization sequence."""
        data_dir = base_config.system.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        runner = ConcreteModelRunner(base_config, mock_logger)

        # Verify all base attributes are set
        assert runner.data_dir == data_dir.resolve()
        assert runner.domain_name == 'test_domain'
        assert runner.project_dir == data_dir.resolve() / 'domain_test_domain'
        assert runner.model_name == 'TEST_MODEL'
        assert hasattr(runner, 'output_dir')

    def test_backup_settings_integration(self, runner, temp_dir):
        """Test backup_settings method (existing method)."""
        # Create source directory with files
        source_dir = temp_dir / 'settings'
        source_dir.mkdir()
        (source_dir / 'config.txt').write_text('test config')
        (source_dir / 'params.txt').write_text('test params')

        runner.backup_settings(source_dir)

        # Verify backup was created
        backup_path = runner.output_dir / 'run_settings'
        assert backup_path.exists()
        assert (backup_path / 'config.txt').exists()
        assert (backup_path / 'params.txt').exists()

    def test_get_log_path_integration(self, runner):
        """Test get_log_path method (existing method)."""
        log_path = runner.get_log_path()

        assert log_path.exists()
        assert log_path.parent == runner.output_dir
        assert log_path.name == 'logs'


class TestAugmentCondaLibraryPaths:
    """Tests for augment_conda_library_paths helper."""

    def test_linux_ld_library_path(self):
        """Test that CONDA_PREFIX/lib is prepended to LD_LIBRARY_PATH on Linux."""
        env = {'CONDA_PREFIX': '/opt/conda', 'LD_LIBRARY_PATH': '/usr/lib'}
        with patch('symfluence.models.execution.model_executor.sys') as mock_sys:
            mock_sys.platform = 'linux'
            augment_conda_library_paths(env)
        conda_lib = os.path.join('/opt/conda', 'lib')
        assert env['LD_LIBRARY_PATH'].startswith(conda_lib)
        assert '/usr/lib' in env['LD_LIBRARY_PATH']

    def test_macos_dyld_library_path(self):
        """Test that CONDA_PREFIX/lib is prepended to DYLD_LIBRARY_PATH on macOS."""
        env = {'CONDA_PREFIX': '/opt/conda', 'DYLD_LIBRARY_PATH': '/usr/lib'}
        with patch('symfluence.models.execution.model_executor.sys') as mock_sys:
            mock_sys.platform = 'darwin'
            augment_conda_library_paths(env)
        conda_lib = os.path.join('/opt/conda', 'lib')
        assert env['DYLD_LIBRARY_PATH'].startswith(conda_lib)
        assert '/usr/lib' in env['DYLD_LIBRARY_PATH']

    def test_windows_path(self):
        """Test that CONDA_PREFIX/Library/bin is prepended to PATH on Windows."""
        conda_lib = os.path.join('/opt/conda', 'Library', 'bin')
        env = {'CONDA_PREFIX': '/opt/conda', 'PATH': '/usr/bin'}
        with patch('symfluence.models.execution.model_executor.sys') as mock_sys:
            mock_sys.platform = 'win32'
            augment_conda_library_paths(env)
        assert conda_lib in env['PATH']
        assert env['PATH'].index(conda_lib) < env['PATH'].index('/usr/bin')

    def test_no_conda_prefix_is_noop(self):
        """Test that missing CONDA_PREFIX is a no-op."""
        env = {'PATH': '/usr/bin'}
        original = env.copy()
        augment_conda_library_paths(env)
        assert env == original

    def test_empty_conda_prefix_is_noop(self):
        """Test that empty CONDA_PREFIX is a no-op."""
        env = {'CONDA_PREFIX': '', 'PATH': '/usr/bin'}
        original = env.copy()
        augment_conda_library_paths(env)
        assert env == original

    def test_idempotent(self):
        """Test that calling twice doesn't duplicate the path."""
        env = {'CONDA_PREFIX': '/opt/conda', 'LD_LIBRARY_PATH': '/usr/lib'}
        with patch('symfluence.models.execution.model_executor.sys') as mock_sys:
            mock_sys.platform = 'linux'
            augment_conda_library_paths(env)
            first_value = env['LD_LIBRARY_PATH']
            augment_conda_library_paths(env)
            assert env['LD_LIBRARY_PATH'] == first_value

    def test_empty_existing_path(self):
        """Test augmentation when the library path variable is empty."""
        env = {'CONDA_PREFIX': '/opt/conda'}
        with patch('symfluence.models.execution.model_executor.sys') as mock_sys:
            mock_sys.platform = 'linux'
            augment_conda_library_paths(env)
        assert env['LD_LIBRARY_PATH'] == os.path.join('/opt/conda', 'lib')

    def test_execute_model_subprocess_augments_env(self, runner, temp_dir):
        """Test that execute_model_subprocess passes conda-augmented env to subprocess."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run, \
             patch.dict(os.environ, {'CONDA_PREFIX': '/opt/conda'}, clear=False), \
             patch('symfluence.models.mixins.subprocess_execution.augment_conda_library_paths') as mock_augment:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(['echo', 'test'], log_file)

            # Verify augment_conda_library_paths was called with the env dict
            mock_augment.assert_called_once()
            call_arg = mock_augment.call_args[0][0]
            assert isinstance(call_arg, dict)


class TestExecuteSubprocess:
    """Tests for the canonical execute_subprocess method on BaseModelRunner."""

    def test_returns_execution_result(self, runner, temp_dir):
        """Test that execute_subprocess returns an ExecutionResult."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            result = runner.execute_subprocess(
                ['echo', 'test'],
                log_file,
                check=False
            )

            assert isinstance(result, ExecutionResult)
            assert result.success is True
            assert result.return_code == 0
            assert result.log_file == log_file

    def test_failure_returns_execution_result(self, runner, temp_dir):
        """Test non-zero exit code returns ExecutionResult with success=False."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 42
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            result = runner.execute_subprocess(
                ['false'],
                log_file,
                check=False
            )

            assert isinstance(result, ExecutionResult)
            assert result.success is False
            assert result.return_code == 42
            assert result.error_message is not None

    def test_check_raises_on_failure(self, runner, temp_dir):
        """Test that check=True raises CalledProcessError on non-zero exit."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            with pytest.raises(subprocess.CalledProcessError):
                runner.execute_subprocess(
                    ['false'],
                    log_file,
                    check=True
                )

    def test_timeout_returns_execution_result(self, runner, temp_dir):
        """Test that timeout returns ExecutionResult rather than raising."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('test_cmd', 5)

            result = runner.execute_subprocess(
                ['sleep', '100'],
                log_file,
                timeout=5,
                check=False
            )

            assert isinstance(result, ExecutionResult)
            assert result.success is False
            assert result.return_code == -1
            assert 'Timeout' in result.error_message

    def test_duration_tracked(self, runner, temp_dir):
        """Test that execution duration is tracked."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            result = runner.execute_subprocess(
                ['echo', 'test'],
                log_file,
                check=False
            )

            assert result.duration_seconds >= 0.0

    def test_env_merged(self, runner, temp_dir):
        """Test that custom env vars are merged."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            runner.execute_subprocess(
                ['echo', 'test'],
                log_file,
                env={'MY_VAR': 'my_value'},
                check=False
            )

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['env']['MY_VAR'] == 'my_value'

    def test_log_dir_created(self, runner, temp_dir):
        """Test that log directory is created if missing."""
        log_file = temp_dir / 'deep' / 'nested' / 'dir' / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            runner.execute_subprocess(
                ['echo', 'test'],
                log_file,
                check=False
            )

            assert log_file.parent.exists()

    def test_custom_success_message(self, runner, temp_dir, mock_logger):
        """Test custom success message is logged."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            runner.execute_subprocess(
                ['echo', 'test'],
                log_file,
                success_message="Custom done!",
                check=False
            )

            mock_logger.log.assert_any_call(logging.INFO, "Custom done!")


class TestExecuteModelSubprocessDeprecation:
    """Tests for backward compatibility of the deprecated execute_model_subprocess."""

    def test_emits_deprecation_warning(self, runner, temp_dir):
        """Test that execute_model_subprocess emits a DeprecationWarning."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                runner.execute_model_subprocess(['echo', 'test'], log_file)

                # Find deprecation warnings
                dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(dep_warnings) >= 1
                assert 'execute_subprocess' in str(dep_warnings[0].message)

    def test_still_returns_completed_process(self, runner, temp_dir):
        """Test that deprecated method still returns CompletedProcess."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock(spec=subprocess.CompletedProcess)
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                result = runner.execute_model_subprocess(['echo', 'test'], log_file)

            assert result.returncode == 0


class TestSlurmMethodsOnBaseRunner:
    """Tests that SLURM methods are accessible on plain BaseModelRunner."""

    def test_is_slurm_available(self, runner):
        """Test is_slurm_available is callable on base runner."""
        # Should return bool (False on dev machines without SLURM)
        result = runner.is_slurm_available()
        assert isinstance(result, bool)

    def test_create_slurm_script(self, runner):
        """Test create_slurm_script produces valid script content."""
        config = SlurmJobConfig(
            job_name='test-job',
            time_limit='01:00:00',
            memory='2G',
        )
        script = runner.create_slurm_script(
            config=config,
            commands=['echo "hello"'],
        )

        assert '#!/bin/bash' in script
        assert '#SBATCH --job-name=test-job' in script
        assert '#SBATCH --time=01:00:00' in script
        assert '#SBATCH --mem=2G' in script
        assert 'echo "hello"' in script

    def test_create_slurm_script_with_array(self, runner):
        """Test SLURM script includes array directive."""
        config = SlurmJobConfig(
            job_name='array-job',
            array_size=9,
        )
        script = runner.create_slurm_script(
            config=config,
            commands=['echo $SLURM_ARRAY_TASK_ID'],
        )

        assert '#SBATCH --array=0-9' in script
        assert 'SLURM Array Task ID' in script

    def test_estimate_optimal_grus_per_job(self, runner):
        """Test GRU estimation logic."""
        # Small domain
        assert runner.estimate_optimal_grus_per_job(5) == 1

        # Medium domain
        result = runner.estimate_optimal_grus_per_job(200)
        assert result > 0

        # Large domain
        result = runner.estimate_optimal_grus_per_job(50000)
        assert result > 0
        assert result <= 50000

    def test_execute_in_mode_local(self, runner, temp_dir):
        """Test execute_in_mode with LOCAL mode delegates to execute_subprocess."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            result = runner.execute_in_mode(
                mode=ExecutionMode.LOCAL,
                command=['echo', 'test'],
                log_file=log_file,
                check=False,
            )

            assert isinstance(result, ExecutionResult)
            assert result.success is True

    def test_execute_in_mode_slurm_requires_config(self, runner, temp_dir):
        """Test execute_in_mode raises ValueError when SLURM mode lacks config."""
        with pytest.raises(ValueError, match="slurm_config required"):
            runner.execute_in_mode(
                mode=ExecutionMode.SLURM,
                command=['echo', 'test'],
                log_file=temp_dir / 'test.log',
            )

    def test_run_with_retry_success_first_attempt(self, runner, temp_dir):
        """Test run_with_retry succeeds on first attempt."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = None
            mock_result.stderr = None
            mock_run.return_value = mock_result

            result = runner.run_with_retry(
                command=['echo', 'test'],
                log_file=log_file,
                max_attempts=3,
                retry_delay=0,
            )

            assert result.success is True
            # Should only be called once (first attempt succeeds)
            assert mock_run.call_count == 1


class TestRunnerHierarchyBackwardCompat:
    """Tests that existing inheritance patterns still work."""

    def test_pattern_c_with_model_executor(self, base_config, mock_logger):
        """Test Pattern C: BaseModelRunner + ModelExecutor works."""
        from symfluence.models.execution.model_executor import ModelExecutor

        class PatternCRunner(BaseModelRunner, ModelExecutor):
            def _get_model_name(self):
                return "PATTERN_C"

        data_dir = base_config.system.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        runner = PatternCRunner(base_config, mock_logger)
        assert runner.model_name == "PATTERN_C"
        # execute_subprocess comes from BaseModelRunner
        assert hasattr(runner, 'execute_subprocess')
        assert hasattr(runner, 'is_slurm_available')

    def test_pattern_b_with_unified_executor(self, base_config, mock_logger):
        """Test Pattern B: BaseModelRunner + UnifiedModelExecutor works."""
        from symfluence.models.execution.unified_executor import UnifiedModelExecutor

        class PatternBRunner(BaseModelRunner, UnifiedModelExecutor):
            def _get_model_name(self):
                return "PATTERN_B"

        data_dir = base_config.system.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        runner = PatternBRunner(base_config, mock_logger)
        assert runner.model_name == "PATTERN_B"
        assert hasattr(runner, 'execute_subprocess')
        # SpatialOrchestrator methods come via UnifiedModelExecutor
        assert hasattr(runner, 'get_spatial_config')

    def test_pattern_d_base_only(self, base_config, mock_logger):
        """Test Pattern D: BaseModelRunner only now has execution methods."""
        class PatternDRunner(BaseModelRunner):
            def _get_model_name(self):
                return "PATTERN_D"

        data_dir = base_config.system.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        runner = PatternDRunner(base_config, mock_logger)
        assert runner.model_name == "PATTERN_D"
        # These used to require ModelExecutor mixin, now built-in
        assert hasattr(runner, 'execute_subprocess')
        assert hasattr(runner, 'is_slurm_available')
        assert hasattr(runner, 'create_slurm_script')
        assert hasattr(runner, 'run_with_retry')
        assert hasattr(runner, 'execute_in_mode')
