# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Logging management for SYMFLUENCE workflow execution.

Provides comprehensive logging infrastructure including console and file handlers,
workflow step formatting, run summaries, and log archival capabilities.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from symfluence.core.logging_utils import (
    FILE_DATEFMT,
    FILE_FORMAT,
    FILE_FORMAT_DEBUG,
    ShortNameFilter,
    silence_third_party,
)
from symfluence.core.mixins import ConfigMixin

try:
    from symfluence.symfluence_version import __version__ as _symfluence_version
except ImportError:
    _symfluence_version = "0+unknown"

# Suppress pyogrio field width warnings (non-fatal - data is still written)
warnings.filterwarnings('ignore',
                       message='.*not successfully written.*field width.*',
                       category=RuntimeWarning,
                       module='pyogrio.raw')

# Suppress xarray FutureWarning about timedelta64 decoding
# These are informational and clutter logs without affecting functionality
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')


# Idempotence state for setup_logging(): the (domain, experiment_id) the
# process-wide 'symfluence' logger is currently configured for, the log
# file that configuration writes to, and the counting handler attached to
# it. A re-setup with the same key reuses the existing handlers instead of
# clearing them and opening a new file.
_active_setup_key: Optional[Tuple[str, str]] = None
_active_log_file: Optional[Path] = None
_active_counting_handler: Optional["CountingHandler"] = None


class CountingHandler(logging.Handler):
    """
    Count WARNING/ERROR/CRITICAL records emitted on the ``symfluence`` tree.

    Attached to the ``symfluence`` root logger by
    :meth:`LoggingManager.setup_logging` so run summaries can report *real*
    warning/error totals instead of only workflow-level exception lists.
    Scope: records emitted in *this process only* — spawned worker processes
    (parallel calibration) log to their own handlers, so their counts do not
    reach the parent's summary.
    Besides the counts, the handler keeps the most recent unique ERROR (and
    CRITICAL) messages so a run summary can show what went wrong without
    re-reading the log file.

    Attributes:
        warning_count: Number of WARNING records seen.
        error_count: Number of ERROR and CRITICAL records seen.
    """

    def __init__(self, max_recent_errors: int = 20):
        super().__init__(level=logging.WARNING)
        self.max_recent_errors = max_recent_errors
        self.warning_count = 0
        self.error_count = 0
        # Ordered set of unique error messages (insertion order == recency)
        self._recent_errors: "OrderedDict[str, None]" = OrderedDict()

    def emit(self, record: logging.LogRecord) -> None:
        """Count the record; retain recent unique ERROR+ messages."""
        try:
            if record.levelno >= logging.ERROR:
                self.error_count += 1
                message = record.getMessage()
                # Re-inserting moves an already-seen message to most-recent
                self._recent_errors.pop(message, None)
                self._recent_errors[message] = None
                while len(self._recent_errors) > self.max_recent_errors:
                    self._recent_errors.popitem(last=False)
            elif record.levelno >= logging.WARNING:
                self.warning_count += 1
        except Exception:  # noqa: BLE001 — logging must never raise
            self.handleError(record)

    @property
    def recent_errors(self) -> List[str]:
        """The last (up to ``max_recent_errors``) unique error messages."""
        return list(self._recent_errors)

    def reset(self) -> None:
        """Zero the counts and clear the retained error messages."""
        self.warning_count = 0
        self.error_count = 0
        self._recent_errors.clear()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger in the ``symfluence`` hierarchy without configuring
    any handlers (no ``logging.basicConfig`` side effects).

    When the ``symfluence`` root is unconfigured, a NullHandler is attached
    so library-style use never triggers the "No handlers could be found"
    warning. Real handlers are installed by ``LoggingManager.setup_logging``.

    Args:
        name: Sub-logger name; joined under ``symfluence.``. None (or
            'symfluence') returns the root ``symfluence`` logger.
    """
    root_logger = logging.getLogger("symfluence")
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())

    if name is None or name == "symfluence":
        return root_logger
    # Tolerate callers that already pass a fully-qualified name.
    short = name[len("symfluence."):] if name.startswith("symfluence.") else name
    return logging.getLogger(f"symfluence.{short}")

_SENSITIVE_KEY_RE = re.compile(
    r'PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY|CREDENTIAL', re.IGNORECASE
)
MASKED_VALUE = '***MASKED***'


def mask_sensitive_config(value: Any) -> Any:
    """Return a copy of *value* with credential-like keys masked, recursively.

    A key is sensitive when it contains (case-insensitive) PASSWORD, PASSWD,
    SECRET, TOKEN, API_KEY/APIKEY or CREDENTIAL. ``None``/empty values are
    left as-is (nothing to hide, and it keeps "unset" visible).
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE_KEY_RE.search(k) and v not in (None, ''):
                out[k] = MASKED_VALUE
            else:
                out[k] = mask_sensitive_config(v)
        return out
    if isinstance(value, list):
        return [mask_sensitive_config(v) for v in value]
    return value


class LoggingManager(ConfigMixin):
    """
    Manages logging configuration and setup for the SYMFLUENCE framework.

    The LoggingManager is responsible for establishing a structured and comprehensive
    logging system for SYMFLUENCE operations. It creates and configures loggers that
    provide detailed tracking of workflow execution, error conditions, and system
    status. The logging system supports both console output for interactive feedback
    and file-based logging for detailed analysis and troubleshooting.

    Key responsibilities:
    - Setting up the main logger with appropriate handlers and formatters
    - Creating and managing module-specific loggers
    - Capturing and preserving the configuration used for each run
    - Creating run summaries with execution statistics
    - Providing archiving capabilities for log persistence

    The logging system uses a hierarchical approach with different levels of detail
    for console (less verbose) and file (more verbose) outputs. It also supports
    module-specific logging configurations to allow fine-grained control over
    log verbosity.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        debug_mode (bool): Whether debug mode is enabled
        quiet_mode (bool): Whether console output is limited to WARNING+
        data_dir (Path): Path to the SYMFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        log_dir (Path): Path to the logging directory
        logger (logging.Logger): Main logger instance
        config_log_file (Path): Path to the logged configuration file
    """

    def __init__(self, config: Dict[str, Any], debug_mode: bool = False,
                 quiet_mode: bool = False):
        """
        Initialize the logging manager.

        This method sets up the logging directory structure and initializes the main
        logger for SYMFLUENCE. It also captures and preserves the configuration
        used for the current run.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            debug_mode (bool): Whether to enable debug mode for detailed console output
            quiet_mode (bool): Suppress console INFO output (console handler at
                WARNING). The file log is unaffected. Ignored when debug_mode
                is set (debug wins).

        Raises:
            KeyError: If essential configuration values are missing
            PermissionError: If log directories cannot be created due to permissions
            OSError: If other file system operations fail
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.debug_mode = debug_mode
        self.quiet_mode = quiet_mode
        self._counting_handler: Optional[CountingHandler] = None
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        # experiment_id comes from the ConfigMixin property (config-backed,
        # platform default 'run_1'; may be empty for minimal dict configs)
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.log_dir = self.project_dir / f"_workLog_{self.domain_name}"

        # Create log directory with fallback for restricted locations
        self.log_dir = self._ensure_log_dir(self.log_dir)

        # Set up main logger with debug mode consideration
        self.logger = self.setup_logging()

        # Log debug mode status
        if self.debug_mode:
            self.logger.debug("DEBUG MODE ENABLED - Detailed console output active")

        # Log configuration
        self.config_log_file = self.log_configuration()

    def _ensure_log_dir(self, log_dir: Path) -> Path:
        """
        Ensure the log directory is writable, falling back to a temp directory if needed.
        """
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            test_file = log_dir / ".symfluence_log_test"
            test_file.touch()
            test_file.unlink()
            return log_dir
        except (PermissionError, OSError):
            fallback_dir = Path(tempfile.mkdtemp(prefix="symfluence_logs_"))
            warnings.warn(
                f"Log directory '{log_dir}' is not writable; using '{fallback_dir}' instead.",
                RuntimeWarning
            )
            return fallback_dir

    def _sanitized_experiment_id(self) -> str:
        """Experiment id reduced to filename-safe characters ('' if unset)."""
        if not self.experiment_id:
            return ''
        sanitized = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(self.experiment_id)).strip('-_.')
        return sanitized

    def setup_logging(self, log_level: Optional[str] = None) -> logging.Logger:
        """
        Set up logging configuration with console and file handlers.

        This method configures the process-wide 'symfluence' logger with:
        1. A file handler that captures all levels (DEBUG and above) using the
           canonical FILE_FORMAT (FILE_FORMAT_DEBUG in debug mode)
        2. A console handler at INFO (DEBUG in debug mode, WARNING in quiet mode)
        3. A CountingHandler that tallies WARNING/ERROR records for run summaries

        Setup is idempotent per (domain, experiment_id): calling it again for
        the same domain and experiment reuses the existing handlers and log
        file instead of clearing them and opening a new file. A different
        domain/experiment reconfigures the logger.

        The log file is named
        ``symfluence_{domain}_{experiment_id}_{YYYYMMDD_HHMMSS}.log``
        (the experiment segment is omitted when no experiment id is set).

        Args:
            log_level (Optional[str]): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                                      If None, uses the level from configuration

        Returns:
            logging.Logger: Configured logger instance ready for use

        Raises:
            ValueError: If an invalid log level is specified
            PermissionError: If log file cannot be created due to permissions
        """
        global _active_setup_key, _active_log_file, _active_counting_handler

        # Get log level from config or parameter
        if log_level is None:
            log_level = self._get_config_value(lambda: self.config.system.log_level, default='INFO', dict_key='LOG_LEVEL')

        # Force DEBUG level if debug_mode is enabled
        if self.debug_mode:
            log_level = 'DEBUG'

        logger = logging.getLogger('symfluence')
        setup_key = (str(self.domain_name), self._sanitized_experiment_id())

        # Idempotence: same domain + experiment id -> reuse existing handlers.
        has_real_handlers = any(
            not isinstance(h, logging.NullHandler) for h in logger.handlers
        )
        # An empty experiment id cannot distinguish two runs, so it never
        # reuses handlers (each construction gets its own log file, as before).
        if has_real_handlers and setup_key[1] and _active_setup_key == setup_key:
            self.log_file = _active_log_file
            self._counting_handler = _active_counting_handler
            # Re-apply level and formats so a quiet/debug toggle between
            # constructions takes effect on the reused handlers.
            logger.setLevel(getattr(logging, log_level.upper()))
            self._apply_console_level(logger)
            file_formatter = logging.Formatter(
                FILE_FORMAT_DEBUG if self.debug_mode else FILE_FORMAT,
                datefmt=FILE_DATEFMT
            )
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setFormatter(file_formatter)
            return logger

        # Create timestamp for log file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_segment = f'_{setup_key[1]}' if setup_key[1] else ''
        log_file = self.log_dir / f'symfluence_{self.domain_name}{exp_segment}_{current_time}.log'

        # Disable the root logger to prevent interference
        logging.getLogger().handlers = []
        logging.getLogger().setLevel(logging.CRITICAL)

        # Cap noisy third-party loggers (single table in core.logging_utils)
        silence_third_party()

        logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers (reconfiguration or first real setup)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # File formatter: canonical protocol format with short logger names
        file_formatter = logging.Formatter(
            FILE_FORMAT_DEBUG if self.debug_mode else FILE_FORMAT,
            datefmt=FILE_DATEFMT
        )

        # Console formatter - different based on debug mode
        if self.debug_mode:
            # Detailed console formatter for debug mode
            console_formatter = self.EnhancedFormatter(
                '%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Clean console formatter for normal mode
            console_formatter = self.EnhancedFormatter(
                '%(asctime)s ● %(message)s',
                datefmt='%H:%M:%S'
            )

        # File handler with protocol formatting (always logs everything)
        file_handler = None
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        except (PermissionError, OSError):
            fallback_dir = self._ensure_log_dir(Path(tempfile.mkdtemp(prefix="symfluence_logs_")))
            log_file = fallback_dir / log_file.name
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            except (PermissionError, OSError):
                file_handler = None
        if file_handler:
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(ShortNameFilter())

        # Console handler - level depends on debug/quiet mode
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._console_level())
        console_handler.setFormatter(console_formatter)

        # Counting handler: real warning/error totals for run summaries
        counting_handler = CountingHandler()

        # Add handlers to logger
        if file_handler:
            logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(counting_handler)

        # Record active configuration for idempotent re-setup
        self.log_file = log_file if file_handler else None
        self._counting_handler = counting_handler
        _active_setup_key = setup_key
        _active_log_file = self.log_file
        _active_counting_handler = counting_handler

        # Log startup information
        startup_lines = [
            "=" * 60,
            "SYMFLUENCE Logging Initialized",
        ]
        if self.debug_mode:
            startup_lines.append("DEBUG MODE: Enabled")
        startup_lines.extend([
            f"Domain: {self.domain_name}",
            f"Experiment ID: {self.experiment_id or 'N/A'}",
            f"Log Level: {log_level}",
            f"Log File: {log_file}" if file_handler else "Log File: disabled (no write access)",
            "=" * 60,
        ])
        logger.info("\n".join(startup_lines))

        return logger

    def _console_level(self) -> int:
        """Console handler level for the current debug/quiet mode.

        Debug wins over quiet; otherwise quiet raises the console handler to
        WARNING while the file log keeps capturing everything.
        """
        if self.debug_mode:
            return logging.DEBUG
        if self.quiet_mode:
            return logging.WARNING
        return logging.INFO

    def _apply_console_level(self, logger: logging.Logger) -> None:
        """Set the console (stream) handler level to match the current mode."""
        level = self._console_level()
        for handler in logger.handlers:
            if (isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)):
                handler.setLevel(level)

    @property
    def log_counts(self) -> Dict[str, int]:
        """Warning/error record totals observed on the 'symfluence' logger.

        Returns:
            Dict with ``warnings`` (WARNING records) and ``errors``
            (ERROR + CRITICAL records). Zeros when no counting handler is
            attached (e.g. file-only fallback situations).
        """
        handler = self._counting_handler
        return {
            'warnings': handler.warning_count if handler else 0,
            'errors': handler.error_count if handler else 0,
        }

    @property
    def recent_errors(self) -> List[str]:
        """Most recent unique ERROR/CRITICAL messages (up to ~20)."""
        handler = self._counting_handler
        return handler.recent_errors if handler else []

    def log_configuration(self) -> Path:
        """
        Log the current configuration to a YAML file for reproducibility.

        This method captures the complete configuration used for the current run
        and saves it to a timestamped YAML file. This is essential for
        reproducibility and debugging, as it preserves the exact settings used
        for each model run.

        The configuration is saved with masked sensitive information (such as
        passwords or API keys, though none are currently expected in SYMFLUENCE).

        Returns:
            Path: Path to the saved configuration file

        Raises:
            PermissionError: If configuration file cannot be written
            IOError: If file writing operations fail
        """
        # Create timestamp for config file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.log_dir / f'config_{self.domain_name}_{current_time}.yaml'

        # Create a copy and mask sensitive information
        # Use config_dict to get a mutable dict (config.copy() returns Pydantic model)
        config_to_log = self.config_dict.copy()

        # Mask credentials (any key naming a password/secret/token/API key,
        # at any nesting depth) before the copy lands in _workLog.
        config_to_log = mask_sensitive_config(config_to_log)

        # Add metadata
        metadata = {
            'logged_at': current_time,
            'symfluence_version': _symfluence_version,
            'debug_mode': self.debug_mode,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        config_to_log['_metadata'] = metadata

        # Save configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_log, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Configuration logged to: {config_file}")
        return config_file

    def log_step_header(self, step_number: int, total_steps: int,
                       step_name: str, description: str = "") -> None:
        """
        Log a formatted header for a workflow step.

        This method creates visually distinct headers for workflow steps to improve
        log readability. It uses box drawing characters to create a clear visual
        separation between different workflow stages.

        Args:
            step_number (int): Current step number
            total_steps (int): Total number of steps in the workflow
            step_name (str): Name of the current step
            description (str): Optional detailed description of the step

        Raises:
            ValueError: If step_number > total_steps or if values are negative
        """
        # Box width follows the longest content line (min 58) so multi-digit
        # step numbers and long names stay aligned.
        content_lines = [f"Step {step_number}/{total_steps}: {step_name}"]
        if description:
            content_lines.append(description)
        inner_width = max(58, max(len(line) for line in content_lines) + 2)

        box_lines = [f"┌{'─' * inner_width}┐"]
        for line in content_lines:
            box_lines.append(f"│ {line:<{inner_width - 2}} │")
        box_lines.append(f"└{'─' * inner_width}┘")

        # Emit the whole box as a single log record (embedded newlines)
        self.logger.info("\n" + "\n".join(box_lines))

    def log_substep(self, message: str, level: str = 'INFO') -> None:
        """
        Log a substep or progress message with consistent formatting.

        This method provides consistent formatting for substep messages within
        a larger workflow step. It uses arrow indicators to show progression
        and maintains visual consistency throughout the logs.

        Args:
            message (str): The message to log
            level (str): Logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)

        Raises:
            ValueError: If an invalid logging level is specified
        """
        formatted_message = f"  → {message}"
        log_method = getattr(self.logger, level.lower())
        log_method(formatted_message)

    def log_completion(self, success: bool = True, message: str = "",
                      duration: Optional[float] = None) -> None:
        """
        Log the completion of a step or operation.

        This method logs the completion status of an operation with visual
        indicators for success or failure. It can also include execution time
        if provided.

        Args:
            success (bool): Whether the operation completed successfully
            message (str): Optional completion message
            duration (Optional[float]): Execution time in seconds

        Raises:
            ValueError: If duration is negative
        """
        status_symbol = "✓" if success else "✗"
        status_text = "Completed" if success else "Failed"

        completion_msg = f"  {status_symbol} {status_text}"
        if message:
            completion_msg += f": {message}"
        if duration is not None:
            completion_msg += f" (Duration: {duration:.2f}s)"

        if success:
            self.logger.info(completion_msg)
        else:
            self.logger.error(completion_msg)

    # Allowed per-step status vocabulary (aligned with core.provenance)
    _STEP_STATUSES = frozenset({'completed', 'skipped', 'failed'})

    @classmethod
    def _normalize_step_entry(cls, entry: Any) -> Dict[str, Any]:
        """Coerce a step record into the fixed run-summary step schema.

        The fixed schema is ``{name, cli_name, description, status,
        duration_s}`` with ``status`` in {completed, skipped, failed} —
        the same vocabulary the provenance manifest uses.
        """
        if not isinstance(entry, dict):
            return {
                'name': str(entry),
                'cli_name': str(entry),
                'description': '',
                'status': 'completed',
                'duration_s': 0.0,
            }

        name = str(entry.get('name') or entry.get('fn') or entry.get('cli') or '')
        cli_name = str(entry.get('cli_name') or entry.get('cli') or name)
        status = str(entry.get('status') or
                     ('completed' if entry.get('success', True) else 'failed'))
        if status not in cls._STEP_STATUSES:
            status = 'failed' if status in ('error', 'errored') else 'completed'
        try:
            duration_s = round(float(entry.get('duration_s',
                                               entry.get('duration', 0.0)) or 0.0), 3)
        except (TypeError, ValueError):
            duration_s = 0.0

        normalized: Dict[str, Any] = {
            'name': name or cli_name,
            'cli_name': cli_name or name,
            'description': str(entry.get('description') or ''),
            'status': status,
            'duration_s': duration_s,
        }
        if entry.get('error'):
            normalized['error'] = str(entry['error'])
        return normalized

    def create_run_summary(self, steps: List[Any],
                          execution_time: float,
                          status: str = 'completed',
                          errors: Optional[List[Dict[str, Any]]] = None) -> Path:
        """
        Create a summary JSON file for the entire run.

        Writes an accurate account of the run: one fixed step-entry schema
        ``{name, cli_name, description, status, duration_s}`` (status in
        {completed, skipped, failed}), warning/error totals counted from the
        actual log records via the CountingHandler, the most recent unique
        error messages, and pointers to the log file and provenance
        ``run_manifest.json`` instead of duplicating manifest content.

        Args:
            steps: Per-step records from the workflow orchestrator (dicts are
                normalized into the fixed step schema)
            execution_time (float): Total execution time in seconds
            status (str): Overall run status ('completed', 'partial', 'failed')
            errors: Optional workflow-level error dicts (exceptions surfaced
                by the orchestrator), recorded as ``workflow_errors``

        Returns:
            Path: Path to the created summary file

        Raises:
            IOError: If summary file cannot be written
        """
        summary_file = self.log_dir / f'run_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        step_entries = [self._normalize_step_entry(s) for s in (steps or [])]
        counts = self.log_counts

        summary = {
            'schema_version': 2,
            'timestamp': datetime.now().isoformat(),
            'domain': self.domain_name,
            'experiment_id': self._get_config_value(lambda: self.config.domain.experiment_id, default='N/A', dict_key='EXPERIMENT_ID'),
            'status': status,
            'execution_time_seconds': round(float(execution_time), 3),
            'steps': step_entries,
            'total_steps': len(step_entries),
            'steps_completed': sum(1 for s in step_entries if s['status'] == 'completed'),
            'steps_skipped': sum(1 for s in step_entries if s['status'] == 'skipped'),
            'steps_failed': sum(1 for s in step_entries if s['status'] == 'failed'),
            # Real totals counted from log records, not workflow exception lists
            'total_errors': counts['errors'],
            'total_warnings': counts['warnings'],
            'recent_errors': self.recent_errors,
            'workflow_errors': list(errors or []),
            'debug_mode': self.debug_mode,
            # Pointers instead of duplicated content
            'log_file': str(self.log_file) if getattr(self, 'log_file', None) else None,
            'run_manifest': str(self.log_dir / 'run_manifest.json'),
            'configuration': {
                'hydrological_model': self._get_config_value(lambda: self.config.model.hydrological_model, dict_key='HYDROLOGICAL_MODEL'),
                'domain_definition_method': self._get_config_value(lambda: self.config.domain.definition_method, dict_key='DOMAIN_DEFINITION_METHOD'),
                'optimization_algorithm': self._get_config_value(lambda: self.config.optimization.algorithm, dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM'),
                'force_run_all': self._get_config_value(lambda: self.config.system.force_run_all_steps, default=False, dict_key='FORCE_RUN_ALL_STEPS')
            }
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"Run summary created: {summary_file}")
        return summary_file

    def archive_logs(self, archive_name: Optional[str] = None) -> Path:
        """
        Archive all logs for the current run into a compressed file.

        This method collects all log files for the current run and compresses
        them into a ZIP archive. This is useful for preserving logs long-term,
        sharing logs for troubleshooting, or cleaning up log directories while
        retaining the information.

        If no archive name is provided, one is generated using the domain name
        and current timestamp.

        Args:
            archive_name (Optional[str]): Name for the archive file.
                                         If None, a name is generated automatically

        Returns:
            Path: Path to the created archive file

        Raises:
            PermissionError: If the archive cannot be created
            ImportError: If the shutil module is not available
            IOError: For other errors during archiving
        """
        import shutil

        if archive_name is None:
            archive_name = f'logs_{self.domain_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        archive_path = self.log_dir / 'archives'
        archive_path.mkdir(exist_ok=True)

        # Create archive
        archive_file = shutil.make_archive(
            str(archive_path / archive_name),
            'zip',
            self.log_dir,
            '.'
        )

        self.logger.info(f"Logs archived to: {archive_file}")
        return Path(archive_file)

    def get_log_file_path(self, log_type: str = 'general') -> Optional[Path]:
        """
        Get the path to a specific log file type.

        This method locates and returns the path to a specific type of log file,
        such as the general log, configuration log, or module-specific logs.
        For types other than 'config', it returns the most recent log file
        of that type.

        Args:
            log_type (str): Type of log file to retrieve:
                          - 'general': Main SYMFLUENCE log
                          - 'config': Configuration log
                          - Any module name: Module-specific log

        Returns:
            Optional[Path]: Path to the requested log file if it exists, None otherwise

        Raises:
            ValueError: If an invalid log_type is specified
        """
        if log_type == 'general':
            # Find the most recent general log. New protocol names the file
            # symfluence_{domain}[_{experiment_id}]_{ts}.log; also accept the
            # legacy symfluence_general_{domain}_*.log for old runs.
            # Sort by modification time: the new and legacy filename patterns
            # do not sort chronologically against each other by name.
            log_files = sorted(
                set(self.log_dir.glob(f'symfluence_{self.domain_name}_*.log'))
                | set(self.log_dir.glob(f'symfluence_general_{self.domain_name}_*.log')),
                key=lambda p: p.stat().st_mtime,
            )
            return log_files[-1] if log_files else None
        elif log_type == 'config':
            return self.config_log_file
        else:
            # Find module-specific log
            log_files = sorted(self.log_dir.glob(f'{log_type}_{self.domain_name}_*.log'))
            return log_files[-1] if log_files else None

    class EnhancedFormatter(logging.Formatter):
        """
        Console formatter with plain, level-based coloring.

        Formats every record with the configured format string and, when the
        terminal supports it, colors the level indicator (the ``●`` bullet in
        normal mode, the level name in debug mode) according to the record's
        level. No per-record message sniffing is performed.
        """

        # ANSI color codes for different log levels (with cross-platform check)
        use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.name != 'nt'

        if use_colors:
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            RESET = '\033[0m'
        else:
            COLORS = {}
            RESET = ''

        def format(self, record: logging.LogRecord) -> str:
            """
            Format log record with level-based coloring.

            Args:
                record (logging.LogRecord): The log record to format

            Returns:
                str: Formatted log message
            """
            formatted = super().format(record)
            if not self.use_colors:
                return formatted

            color = self.COLORS.get(record.levelname, '')
            if not color:
                return formatted

            if ' ● ' in formatted:
                return formatted.replace(' ● ', f' {color}●{self.RESET} ', 1)
            return formatted.replace(
                record.levelname, f'{color}{record.levelname}{self.RESET}', 1
            )
