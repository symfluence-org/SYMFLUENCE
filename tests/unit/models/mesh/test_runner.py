"""
Tests for MESH model runner.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from symfluence.core.registries import R


class TestMESHRunnerInitialization:
    """Tests for MESH runner initialization."""

    def test_runner_can_be_imported(self):
        """Test that MESHRunner can be imported."""
        from symfluence.models.mesh.runner import MESHRunner
        assert MESHRunner is not None

    def test_runner_initialization(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test runner initializes with config."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        assert runner is not None
        assert runner.domain_name == 'test_domain'

    def test_runner_model_name(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test runner returns correct model name."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        assert runner._get_model_name() == 'MESH'


class TestMESHRunnerPaths:
    """Tests for MESH runner path setup."""

    def test_mesh_exe_path_configured(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test MESH executable path is configured."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        assert runner.mesh_exe is not None
        assert runner.mesh_exe.name == 'mesh.exe'

    def test_forcing_dir_configured(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test forcing directory is configured."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        assert runner.forcing_dir is not None
        assert 'MESH_input' in str(runner.forcing_dir)

    def test_setup_dir_configured(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test MESH setup directory is configured."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        assert runner.mesh_setup_dir is not None
        assert 'MESH' in str(runner.mesh_setup_dir)


class TestMESHRunnerProcessDirectories:
    """Tests for MESH runner process directory management."""

    def test_set_process_directories(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test setting process-specific directories."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)

        forcing_dir = setup_mesh_directories['forcing_dir'] / 'process_1'
        output_dir = setup_mesh_directories['simulations_dir'] / 'process_1'
        forcing_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        runner.set_process_directories(forcing_dir, output_dir)

        assert runner.forcing_mesh_path == forcing_dir
        assert runner.output_dir == output_dir


class TestMESHRunnerExecution:
    """Tests for MESH runner execution."""

    def test_native_stop_without_outputs_raises(self, mesh_config, mock_logger, setup_mesh_directories):
        from symfluence.core.exceptions import ModelExecutionError
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        runner.forcing_mesh_path.mkdir(parents=True, exist_ok=True)
        stale = runner.forcing_mesh_path/'Basin_average_water_balance.csv'
        stale.write_text('stale results')
        (runner.forcing_mesh_path/'MESH_input_run_options.ini').write_text('FROZENSOILINFILFLAG 0\n')
        with patch.object(runner, '_create_run_command', return_value=['mesh']), \
                patch.object(runner, 'execute_subprocess', return_value=Mock(success=True, return_code=0)):
            with pytest.raises(ModelExecutionError):
                runner.run_mesh()
        assert not stale.exists()

    def test_run_mesh_creates_command(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test MESH execution creates proper command."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)

        # Create the executable in the source location
        runner.mesh_exe.parent.mkdir(parents=True, exist_ok=True)
        runner.mesh_exe.touch()

        # Also create the forcing_mesh_path
        runner.forcing_mesh_path.mkdir(parents=True, exist_ok=True)

        cmd = runner._create_run_command()

        assert cmd is not None
        assert './mesh.exe' in cmd

    def test_verify_outputs_returns_false_when_missing(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test output verification returns False when files missing."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        runner.output_dir = setup_mesh_directories['simulations_dir']
        runner.forcing_mesh_path = setup_mesh_directories['forcing_dir']

        result = runner._verify_outputs()

        assert result is False

    def test_verify_outputs_returns_true_when_present(self, mesh_config, mock_logger, setup_mesh_directories, sample_mesh_output_csv):
        """Test output verification returns True when files present."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        runner.output_dir = setup_mesh_directories['simulations_dir']
        runner.forcing_mesh_path = setup_mesh_directories['forcing_dir']

        result = runner._verify_outputs()

        assert result is True


class TestMESHRunnerOutputCopy:
    """Tests for MESH runner output copying."""

    def test_copy_outputs_creates_output_dir(self, mesh_config, mock_logger, setup_mesh_directories, sample_mesh_output_csv):
        """Test copy outputs creates output directory if needed."""
        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        runner.forcing_mesh_path = setup_mesh_directories['forcing_dir']
        runner.output_dir = setup_mesh_directories['simulations_dir'] / 'new_output'

        runner._copy_outputs()

        assert runner.output_dir.exists()

    def test_copy_outputs_copies_streamflow(self, mesh_config, mock_logger, setup_mesh_directories, sample_mesh_output_csv):
        """Test copy outputs copies streamflow file."""
        import shutil

        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        runner.forcing_mesh_path = setup_mesh_directories['forcing_dir']
        runner.output_dir = setup_mesh_directories['simulations_dir'] / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)

        runner._copy_outputs()

        copied_file = runner.output_dir / 'MESH_output_streamflow.csv'
        assert copied_file.exists()


class TestMESHModelRegistry:
    """Tests for MESH model registry integration."""

    def test_runner_registered_with_registry(self):
        """Test MESH runner is registered with model registry."""

        # Check if MESH is registered
        runners = R.runners
        assert 'MESH' in runners

    def test_registry_method_name(self):
        """Test MESH runner method name is correct."""

        runner_info = R.runners.get('MESH')
        assert runner_info is not None
        # The method_name should be stored in the tuple
        assert hasattr(runner_info, '__name__') or 'run_mesh' in str(runner_info)


def test_custom_native_executable_is_respected(mesh_config, mock_logger):
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.models.mesh.runner import MESHRunner

    values = mesh_config.to_dict(flatten=True)
    values['MESH_EXE'] = 'sa_mesh'
    config = SymfluenceConfig(**values)
    runner = MESHRunner(config, mock_logger)
    assert runner.mesh_exe.name == 'sa_mesh'


def test_prune_run_logs_keeps_newest(tmp_path):
    from symfluence.models.mesh.runner import MESHRunner
    names = [f'mesh_run_20261014_2100{i:02d}.log' for i in range(5)]
    for name in names:
        (tmp_path / name).write_text('x')
    (tmp_path / 'other.log').write_text('x')
    MESHRunner._prune_run_logs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names[-2:] + ['other.log'])
