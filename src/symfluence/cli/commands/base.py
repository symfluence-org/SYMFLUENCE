# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base command class for SYMFLUENCE CLI commands.

This module provides the base class that all command handlers inherit from,
providing common utilities and interfaces.
"""
from __future__ import annotations

import functools
from abc import ABC
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional

# Import exceptions for the decorator
from symfluence.core.exceptions import (
    ConfigurationError,
    ModelExecutionError,
    OptionalDependencyError,
    SYMFLUENCEError,
)

from ..console import Console
from ..console import console as global_console
from ..defaults import DEFAULT_CONFIG_PATH
from ..exit_codes import ExitCode
from ..validators import validate_config_exists

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


def _print_traceback_if_debug(args: Namespace) -> None:
    if getattr(args, 'debug', False):
        import traceback
        traceback.print_exc()


def _exit_code_for_exception(e: BaseException, args: Namespace) -> Optional[int]:
    """Report *e* on the console and map it to an exit code.

    Returns None for exceptions the CLI does not handle (they propagate).
    Order matters: the first matching type wins.
    """
    console = BaseCommand._console
    if isinstance(e, OptionalDependencyError):
        from ..optional_dependencies import handle_optional_dependency
        return handle_optional_dependency(e, console)
    if isinstance(e, ConfigurationError):
        console.error(f"Configuration error: {e}")
        _print_traceback_if_debug(args)
        return ExitCode.CONFIG_ERROR
    if isinstance(e, ModelExecutionError):
        console.error(f"Execution failed: {e}")
        _print_traceback_if_debug(args)
        return ExitCode.WORKFLOW_ERROR
    if isinstance(e, FileNotFoundError):
        console.error(f"File not found: {e}")
        return ExitCode.FILE_NOT_FOUND
    if isinstance(e, PermissionError):
        console.error(f"Permission denied: {e}")
        return ExitCode.PERMISSION_ERROR
    if isinstance(e, ImportError):
        console.error(f"Failed to import required module: {e}")
        return ExitCode.DEPENDENCY_ERROR
    if isinstance(e, ValueError):
        console.error(f"Invalid value: {e}")
        return ExitCode.VALIDATION_ERROR
    if isinstance(e, OSError):
        console.error(f"I/O error: {e}")
        _print_traceback_if_debug(args)
        return ExitCode.GENERAL_ERROR
    if isinstance(e, SYMFLUENCEError):
        console.error(f"Operation failed: {e}")
        _print_traceback_if_debug(args)
        return ExitCode.GENERAL_ERROR
    return None


def _make_cli_exception_handler(
    *, profiling: bool
) -> Callable[[Callable[..., int]], Callable[..., int]]:
    """Build the CLI exception-handling decorator.

    ``profiling=True`` additionally sets up profiling when ``--profile`` is
    given and always finalizes it, even when the command raised.
    """
    def decorator(func: Callable[..., int]) -> Callable[..., int]:
        @functools.wraps(func)
        def wrapper(args: Namespace) -> int:
            profilers = None
            try:
                if profiling and getattr(args, 'profile', False):
                    profilers = _setup_profiling(args)
                return func(args)
            except (SYMFLUENCEError, ImportError, OSError, ValueError) as e:
                code = _exit_code_for_exception(e, args)
                if code is None:  # pragma: no cover - table covers every caught type
                    raise
                return code
            finally:
                if profilers is not None:
                    _finalize_profiling(args, profilers)
        return wrapper
    return decorator


#: Consistent exception handling for CLI commands (exception -> ExitCode).
cli_exception_handler = _make_cli_exception_handler(profiling=False)

#: Same as ``cli_exception_handler``, plus ``--profile`` setup/finalization
#: (finalized even when the command raised). Use for workflow commands.
cli_exception_handler_with_profiling = _make_cli_exception_handler(profiling=True)


def _setup_profiling(args: Namespace):
    """
    Setup profiling if enabled via CLI flag.

    Args:
        args: Parsed arguments namespace

    Returns:
        Tuple of (python_profiler, system_profiler) or None if profiling disabled
    """
    from symfluence.core.profiling import (
        enable_profiling,
        enable_system_profiling,
        setup_profiling_environment,
    )
    capture_stacks = getattr(args, 'profile_stacks', False)

    # Enable Python-level profiling
    python_profiler = enable_profiling(capture_stack_traces=capture_stacks)

    # Enable system-level profiling (external tools)
    system_profiler = enable_system_profiling()

    # Determine profile directory for cross-process profiling
    output_path = getattr(args, 'profile_output', None)
    if output_path is None:
        config_path = BaseCommand.get_config_path(args)
        if config_path:
            output_dir = Path(config_path).parent
        else:
            output_dir = Path.cwd()
    else:
        output_dir = Path(output_path).parent

    profile_dir = output_dir / '.symfluence_profiling'

    # Set up environment variables so worker processes can profile
    setup_profiling_environment(str(profile_dir), capture_stacks=capture_stacks)

    BaseCommand._console.info("I/O profiling enabled (Python + System levels)")
    BaseCommand._console.indent(f"Worker profile data: {profile_dir}")
    if capture_stacks:
        BaseCommand._console.indent("Stack trace capture enabled (this adds overhead)")

    return (python_profiler, system_profiler)


def _finalize_profiling(args: Namespace, profilers):
    """
    Generate profiling reports if profiling was enabled.

    Args:
        args: Parsed arguments namespace
        profilers: Tuple of (python_profiler, system_profiler)
    """
    if profilers is None:
        return

    # Unpack profilers (tuple of python_profiler, system_profiler)
    if isinstance(profilers, tuple):
        python_profiler, system_profiler = profilers
    else:
        # Backward compatibility - single profiler
        python_profiler = profilers
        system_profiler = None

    try:
        from symfluence.core.profiling import get_profile_directory

        # Determine output path
        output_path = getattr(args, 'profile_output', None)
        if output_path is None:
            config_path = BaseCommand.get_config_path(args)
            if config_path:
                output_dir = Path(config_path).parent
            else:
                output_dir = Path.cwd()
            output_path = output_dir / 'profile_report.json'

        # Aggregate worker profile data
        profile_dir = get_profile_directory()
        worker_files_count = 0
        if profile_dir and python_profiler:
            worker_files_count = python_profiler.aggregate_from_directory(profile_dir)
            if worker_files_count > 0:
                BaseCommand._console.info(f"Aggregated profiling data from {worker_files_count} worker process(es)")

            # Clean up the profile directory
            try:
                import shutil
                profile_dir_path = Path(profile_dir)
                if profile_dir_path.exists():
                    shutil.rmtree(profile_dir_path)
            except (OSError, IOError):
                pass

        # Generate Python-level I/O reports
        if python_profiler:
            json_path = Path(output_path)
            text_path = json_path.with_suffix('.txt')

            python_profiler.generate_report(str(json_path), format='json')
            python_profiler.generate_report(str(text_path), format='text')

        # Generate System-level I/O reports
        if system_profiler:
            system_json_path = Path(output_path).parent / 'system_io_report.json'
            system_text_path = system_json_path.with_suffix('.txt')

            system_profiler.generate_report(str(system_json_path), format='json')
            system_profiler.generate_report(str(system_text_path), format='text')

        # Print combined summary using console
        BaseCommand._console.newline()
        BaseCommand._console.rule("COMBINED I/O PROFILING SUMMARY")

        if python_profiler:
            python_stats = python_profiler.get_statistics()
            BaseCommand._console.info("Python-Level I/O (NetCDF, Pickle, etc.):")
            BaseCommand._console.indent(f"Report: {Path(output_path)}")
            BaseCommand._console.indent(f"Text:   {Path(output_path).with_suffix('.txt')}")
            BaseCommand._console.indent(f"Worker processes: {worker_files_count}")
            BaseCommand._console.indent(f"Total operations: {python_stats['summary']['total_operations']:,}")
            BaseCommand._console.indent(f"Bytes written: {python_profiler._format_bytes(python_stats['summary']['total_bytes_written'])}")
            BaseCommand._console.indent(f"Average IOPS: {python_stats['summary']['average_iops']:.1f}")

        if system_profiler:
            system_stats = system_profiler.get_statistics()
            BaseCommand._console.newline()
            BaseCommand._console.info("System-Level I/O (SUMMA, mizuRoute, etc.):")
            BaseCommand._console.indent(f"Report: {Path(output_path).parent / 'system_io_report.json'}")
            BaseCommand._console.indent(f"Text:   {Path(output_path).parent / 'system_io_report.txt'}")
            BaseCommand._console.indent(f"Total subprocesses: {system_stats['summary']['total_operations']:,}")
            BaseCommand._console.indent(f"Read bytes: {system_profiler._format_bytes(system_stats['summary']['total_read_bytes'])}")
            BaseCommand._console.indent(f"Write bytes: {system_profiler._format_bytes(system_stats['summary']['total_write_bytes'])}")
            BaseCommand._console.indent(f"Read IOPS: {system_stats['summary']['average_read_iops']:.1f}")
            BaseCommand._console.indent(f"Write IOPS: {system_stats['summary']['average_write_iops']:.1f}")
            BaseCommand._console.indent(f"Total IOPS: {system_stats['summary']['average_total_iops']:.1f}")
            BaseCommand._console.indent(f"Peak IOPS: {system_stats['summary']['peak_iops']:.1f}")

        BaseCommand._console.rule()

    except (OSError, IOError, ImportError) as e:
        BaseCommand._console.error(f"Error generating profiling report: {e}")
        if getattr(args, 'debug', False):
            import traceback
            traceback.print_exc()


@contextmanager
def profiling_context(args: Namespace):
    """
    Context manager for workflow profiling.

    Use this context manager to wrap workflow execution code that should be profiled.
    Profiling will be set up at entry and finalized at exit, even if exceptions occur.

    Args:
        args: Parsed arguments namespace containing profile-related flags

    Yields:
        Tuple of profilers if profiling is enabled, None otherwise

    Example:
        with profiling_context(args) as profilers:
            # Workflow code here
            pass
    """
    profilers = _setup_profiling(args) if getattr(args, 'profile', False) else None
    try:
        yield profilers
    finally:
        if profilers:
            _finalize_profiling(args, profilers)


class BaseCommand(ABC):
    """
    Base class for all CLI command handlers.

    Provides common functionality for loading configuration, handling errors,
    and executing commands.

    Attributes:
        _console: Shared console instance for all commands
    """

    _console: ClassVar[Console] = global_console

    @classmethod
    def set_console(cls, console: Console) -> None:
        """
        Set the console instance for all commands.

        Useful for testing or configuring output behavior.

        Args:
            console: Console instance to use
        """
        cls._console = console

    @staticmethod
    def load_typed_config(
        config_path: str,
        required: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional["SymfluenceConfig"]:
        """
        Load configuration using the typed SymfluenceConfig system.

        This is the preferred method for loading configuration as it provides
        type-safe access and validation.

        Args:
            config_path: Path to configuration file
            required: Whether config file is required
            overrides: Optional overrides to apply

        Returns:
            SymfluenceConfig instance, or None if not required and not found
        """
        from symfluence.core.config.models import SymfluenceConfig
        from symfluence.core.exceptions import ConfigurationError

        path = Path(config_path)
        if not path.exists():
            if required:
                BaseCommand._console.error(f"Config file not found: {config_path}")
                return None
            return None

        try:
            return SymfluenceConfig.from_file(path, overrides=overrides)
        except ConfigurationError as e:
            BaseCommand._console.error(f"Configuration error: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — top-level fallback
            BaseCommand._console.error(f"Failed to load config: {e}")
            return None

    @staticmethod
    def validate_config(config_path: str, required: bool = True) -> bool:
        """
        Validate that config file exists and is readable.

        Args:
            config_path: Path to configuration file
            required: Whether config is required

        Returns:
            True if valid (or not required and doesn't exist), False otherwise
        """
        result = validate_config_exists(config_path)

        if result.is_err:
            if required:
                error = result.first_error()
                BaseCommand._console.error(error.message if error else "Config validation failed")
                return False
            else:
                # Not required and doesn't exist is OK
                return True

        return True

    @staticmethod
    def get_config_path(args: Namespace) -> str:
        """
        Get configuration file path from args, with fallback to default.

        The default path can be customized via the SYMFLUENCE_DEFAULT_CONFIG
        environment variable.

        Args:
            args: Parsed arguments namespace

        Returns:
            Path to configuration file
        """
        if hasattr(args, 'config') and args.config:
            return args.config
        else:
            # Use configurable default path
            return DEFAULT_CONFIG_PATH

    @staticmethod
    def get_arg(args: Namespace, name: str, default: Any = None) -> Any:
        """
        Safely get argument value with default fallback.

        Args:
            args: Parsed arguments namespace
            name: Name of the argument to retrieve
            default: Default value if argument not found

        Returns:
            Argument value or default
        """
        return getattr(args, name, default)

    @staticmethod
    def get_args_dict(args: Namespace, keys: List[str]) -> Dict[str, Any]:
        """
        Extract multiple args as dict, filtering None values.

        Args:
            args: Parsed arguments namespace
            keys: List of argument names to extract

        Returns:
            Dictionary with non-None argument values
        """
        return {k: v for k, v in ((k, getattr(args, k, None)) for k in keys) if v is not None}

    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """
        Prompt user to confirm a destructive action.

        Args:
            message: Message to display to user
            default: Default response if user just presses Enter

        Returns:
            True if user confirms, False otherwise
        """
        suffix = " [y/N]: " if not default else " [Y/n]: "
        response = input(f"{message}{suffix}").strip().lower()
        if not response:
            return default
        return response in ('y', 'yes')

    # Backward compatibility aliases for deprecated methods
    @classmethod
    def print_error(cls, message: str) -> None:
        """Deprecated: Use _console.error() instead."""
        cls._console.error(message)

    @classmethod
    def print_success(cls, message: str) -> None:
        """Deprecated: Use _console.success() instead."""
        cls._console.success(message)

    @classmethod
    def print_info(cls, message: str) -> None:
        """Deprecated: Use _console.info() instead."""
        cls._console.info(message)
