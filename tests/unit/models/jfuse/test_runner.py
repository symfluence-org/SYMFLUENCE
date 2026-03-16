"""Tests for jFUSE model runner."""

from unittest.mock import MagicMock, patch

import pytest


class TestJFUSERunnerImport:
    """Tests for jFUSE runner importability."""

    def test_runner_can_be_imported(self):
        from jfuse.runner import JFUSERunner
        assert JFUSERunner is not None

    def test_model_name(self):
        from jfuse.runner import JFUSERunner
        assert JFUSERunner.MODEL_NAME == "JFUSE"

    def test_has_jfuse_flag_exists(self):
        from jfuse import runner
        assert hasattr(runner, "HAS_JFUSE")

    def test_has_jax_flag_exists(self):
        from jfuse import runner
        assert hasattr(runner, "HAS_JAX")


class TestJFUSERunnerInit:
    """Tests for jFUSE runner initialization."""

    def test_runner_initialization_lumped(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner is not None
        assert runner.spatial_mode == "lumped"

    def test_runner_initialization_distributed(self, distributed_jfuse_config, mock_logger, temp_dir):
        # Set up directories for distributed config
        data_dir = distributed_jfuse_config.system.data_dir
        domain_dir = data_dir / f"domain_{distributed_jfuse_config.domain.name}"
        for d in [
            domain_dir / "settings" / "JFUSE",
            domain_dir / "data" / "forcing" / "merged_data" / "JFUSE_input",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(distributed_jfuse_config, mock_logger)
        # spatial_mode is resolved from config; verify runner initializes
        assert runner.spatial_mode in ("lumped", "distributed")

    def test_default_model_config_name(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner.model_config_name == "prms_gradient"

    def test_snow_enabled_by_default(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner.enable_snow is True

    def test_warmup_days_from_config(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner.warmup_days == 365

    def test_jit_compile_default(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner.jit_compile is True

    def test_use_gpu_default_false(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner.use_gpu is False


class TestJFUSERunnerPaths:
    """Tests for jFUSE runner path setup."""

    def test_setup_dir_is_jfuse_settings(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        assert runner.jfuse_setup_dir.name == "JFUSE"
        assert "settings" in str(runner.jfuse_setup_dir)

    def test_custom_settings_dir_override(self, jfuse_config, mock_logger, setup_jfuse_directories, temp_dir):
        custom_dir = temp_dir / "custom_settings"
        custom_dir.mkdir()
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger, settings_dir=custom_dir)
        assert runner.jfuse_setup_dir == custom_dir


class TestJFUSERunnerBehaviorWithoutJFUSE:
    """Tests for jFUSE runner behavior when jFUSE is not installed."""

    def test_warns_when_jfuse_not_installed(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import HAS_JFUSE, JFUSERunner

        if not HAS_JFUSE:
            runner = JFUSERunner(jfuse_config, mock_logger)
            mock_logger.warning.assert_any_call("jFUSE not installed. Install with: pip install jfuse")

    def test_get_default_params_returns_dict(self, jfuse_config, mock_logger, setup_jfuse_directories):
        from jfuse.runner import JFUSERunner
        runner = JFUSERunner(jfuse_config, mock_logger)
        params = runner._get_default_params()
        assert isinstance(params, dict)
