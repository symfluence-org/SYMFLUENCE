"""Unit tests for base command class."""
from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.commands.base import BaseCommand
from symfluence.cli.defaults import DEFAULT_CONFIG_PATH
from symfluence.cli.exit_codes import ExitCode
from symfluence.core.exceptions import (
    ConfigurationError,
    ModelExecutionError,
    SYMFLUENCEError,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestDefaultConfigPath:
    """Test default config path handling."""

    def test_default_config_path_exists(self):
        """Test that DEFAULT_CONFIG_PATH is defined."""
        assert DEFAULT_CONFIG_PATH is not None
        assert isinstance(DEFAULT_CONFIG_PATH, str)

    def test_default_config_path_value(self):
        """Test default config path value when env var not set."""
        # If SYMFLUENCE_DEFAULT_CONFIG is not set, should use default
        if 'SYMFLUENCE_DEFAULT_CONFIG' not in os.environ:
            assert DEFAULT_CONFIG_PATH == './config.yaml'


class TestGetConfigPath:
    """Test get_config_path method."""

    def test_get_config_path_from_args(self):
        """Test getting config path from args."""
        args = Namespace(config='/custom/path/config.yaml')

        result = BaseCommand.get_config_path(args)

        assert result == '/custom/path/config.yaml'

    def test_get_config_path_fallback_to_default(self):
        """Test fallback to default config path."""
        args = Namespace(config=None)

        result = BaseCommand.get_config_path(args)

        assert result == DEFAULT_CONFIG_PATH

    def test_get_config_path_no_config_attr(self):
        """Test fallback when config attribute missing."""
        args = Namespace()  # No config attribute

        result = BaseCommand.get_config_path(args)

        assert result == DEFAULT_CONFIG_PATH


class TestValidateConfig:
    """Test validate_config method."""

    def test_validate_config_exists(self, tmp_path):
        """Test validation of existing config file."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        result = BaseCommand.validate_config(str(config_file), required=True)

        assert result is True

    def test_validate_config_not_exists_required(self):
        """Test validation of missing required config."""
        result = BaseCommand.validate_config('/nonexistent/config.yaml', required=True)

        assert result is False

    def test_validate_config_not_exists_not_required(self):
        """Test validation of missing non-required config."""
        result = BaseCommand.validate_config('/nonexistent/config.yaml', required=False)

        assert result is True


class TestLoadTypedConfig:
    """Test load_typed_config method."""

    @patch('symfluence.core.config.models.SymfluenceConfig')
    def test_load_typed_config_success(self, mock_config_class, tmp_path):
        """Test successful config loading."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        mock_config = MagicMock()
        mock_config_class.from_file.return_value = mock_config

        result = BaseCommand.load_typed_config(str(config_file), required=True)

        assert result == mock_config
        mock_config_class.from_file.assert_called_once()

    def test_load_typed_config_file_not_found(self):
        """Test loading config from nonexistent file."""
        result = BaseCommand.load_typed_config('/nonexistent/config.yaml', required=True)

        assert result is None

    def test_load_typed_config_not_required_and_not_found(self):
        """Test loading non-required config from nonexistent file."""
        result = BaseCommand.load_typed_config('/nonexistent/config.yaml', required=False)

        assert result is None

    @patch('symfluence.core.config.models.SymfluenceConfig')
    def test_load_typed_config_configuration_error(self, mock_config_class, tmp_path):
        """Test loading config with configuration error."""
        from symfluence.core.exceptions import ConfigurationError

        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        mock_config_class.from_file.side_effect = ConfigurationError("Invalid config")

        result = BaseCommand.load_typed_config(str(config_file), required=True)

        assert result is None


class TestSetConsole:
    """Test set_console method."""

    def test_set_console(self):
        """Test setting a custom console."""
        mock_console = MagicMock()
        original_console = BaseCommand._console

        try:
            BaseCommand.set_console(mock_console)
            assert BaseCommand._console == mock_console
        finally:
            # Restore original console
            BaseCommand._console = original_console


class TestDeprecatedMethods:
    """Test deprecated backward compatibility methods."""

    def test_print_error(self):
        """Test deprecated print_error method."""
        with patch.object(BaseCommand._console, 'error') as mock_error:
            BaseCommand.print_error("Test error")
            mock_error.assert_called_once_with("Test error")

    def test_print_success(self):
        """Test deprecated print_success method."""
        with patch.object(BaseCommand._console, 'success') as mock_success:
            BaseCommand.print_success("Test success")
            mock_success.assert_called_once_with("Test success")

    def test_print_info(self):
        """Test deprecated print_info method."""
        with patch.object(BaseCommand._console, 'info') as mock_info:
            BaseCommand.print_info("Test info")
            mock_info.assert_called_once_with("Test info")


class TestConsoleQuietMode:
    """Test the Console quiet toggle wired to the global --quiet flag."""

    @staticmethod
    def _make_console():
        import io

        from symfluence.cli.console import Console, ConsoleConfig

        stream = io.StringIO()
        err_stream = io.StringIO()
        console = Console(ConsoleConfig(
            use_colors=False,
            output_stream=stream,
            error_stream=err_stream,
        ))
        return console, stream, err_stream

    def test_set_quiet_suppresses_info_but_not_errors(self):
        console, stream, err_stream = self._make_console()
        console.set_quiet(True)

        assert console.is_quiet is True
        console.info("hidden info")
        console.success("hidden success")
        console.error("visible error")

        assert stream.getvalue() == ""
        assert "visible error" in err_stream.getvalue()

    def test_set_quiet_can_be_disabled_again(self):
        console, stream, _ = self._make_console()
        console.set_quiet(True)
        console.set_quiet(False)

        assert console.is_quiet is False
        console.info("shown again")
        assert "shown again" in stream.getvalue()

    def test_main_quiet_flag_sets_global_console_quiet(self):
        """cli.main() flips the global console into quiet mode for --quiet."""
        from symfluence.cli import main
        from symfluence.cli.console import get_console

        console = get_console()
        original = console.is_quiet
        try:
            with patch(
                'symfluence.cli.commands.workflow_commands.WorkflowCommands.list_steps',
                return_value=0,
            ), patch('sys.argv', ['symfluence', '--quiet', 'workflow', 'list-steps']):
                main()
            assert console.is_quiet is True
        finally:
            console.set_quiet(original)

    def test_entry_point_quiet_flag_sets_global_console_quiet(self):
        """main_cli.main() — the installed `symfluence` entry point — must
        apply --quiet too, not just symfluence.cli.main()."""
        from symfluence.cli.console import get_console
        from symfluence.main_cli import main as entry_main

        console = get_console()
        original = console.is_quiet
        try:
            with patch(
                'symfluence.cli.commands.workflow_commands.WorkflowCommands.list_steps',
                return_value=0,
            ), patch('sys.argv', ['symfluence', '--quiet', 'workflow', 'list-steps']):
                entry_main()
            assert console.is_quiet is True
        finally:
            console.set_quiet(original)


class TestExceptionHandlerFactory:
    """cli_exception_handler and the profiling variant share one exception table."""

    @pytest.mark.parametrize("exc,code", [
        (ConfigurationError("c"), 3),
        (ModelExecutionError("m"), 14),
        (FileNotFoundError("f"), 5),
        (PermissionError("p"), 10),
        (ImportError("i"), 12),
        (ValueError("v"), 4),
        (OSError("o"), 1),
        (SYMFLUENCEError("s"), 1),
    ])
    @pytest.mark.parametrize("profiling", [False, True])
    def test_same_exit_codes(self, exc, code, profiling):
        from symfluence.cli.commands.base import (
            cli_exception_handler,
            cli_exception_handler_with_profiling,
        )

        deco = cli_exception_handler_with_profiling if profiling else cli_exception_handler

        @deco
        def cmd(args):
            raise exc

        assert cmd(Namespace(debug=False, profile=False)) == code

    def test_profiling_finalized_on_error(self):
        from symfluence.cli.commands import base

        @base.cli_exception_handler_with_profiling
        def cmd(args):
            raise ValueError("x")

        with patch.object(base, '_setup_profiling', return_value=("p", "s")) as setup, \
             patch.object(base, '_finalize_profiling') as finalize:
            assert cmd(Namespace(debug=False, profile=True)) == 4
        setup.assert_called_once()
        finalize.assert_called_once()

    def test_unhandled_exception_propagates(self):
        from symfluence.cli.commands.base import cli_exception_handler

        @cli_exception_handler
        def cmd(args):
            raise KeyError("k")

        with pytest.raises(KeyError):
            cmd(Namespace(debug=False))
