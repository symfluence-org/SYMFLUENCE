# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base class for model runners.

Provides shared infrastructure for all model execution modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation for outputs and logs
- Common experiment structure
- Settings file backup utilities
- Subprocess and SLURM execution
- Spatial mode validation (Phase 3)
"""

import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from symfluence.core.mixins import ShapefileAccessMixin
from symfluence.core.path_resolver import PathResolverMixin
from symfluence.models.mixins import (
    ModelComponentMixin,
    SlurmExecutionMixin,
    SubprocessExecutionMixin,
)
from symfluence.models.spatial_modes import get_spatial_mode_from_config, validate_spatial_mode

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelRunner(ABC, ModelComponentMixin, PathResolverMixin, ShapefileAccessMixin,  # type: ignore[misc]
                      SubprocessExecutionMixin, SlurmExecutionMixin):
    """Abstract base class for all model runners.

    Provides initialization, path management, subprocess/SLURM execution,
    and utility methods shared across hydrological model runners.

    Subclasses should set ``MODEL_NAME`` as a class variable. The legacy
    ``_get_model_name()`` method is still supported as a fallback.

    Optional hooks: ``_setup_model_specific_paths()``,
    ``_should_create_output_dir()``, ``_get_output_dir()``,
    ``_validate_required_config()``.

    Execution methods (``execute_subprocess``, ``run_with_retry``,
    ``execute_in_mode``, SLURM helpers) are provided via
    ``SubprocessExecutionMixin`` and ``SlurmExecutionMixin``.
    """

    MODEL_NAME: str = ""

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize base model runner.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
            reporting_manager: ReportingManager instance

        Raises:
            ConfigurationError: If required configuration keys are missing
        """
        # Common initialization via mixin
        self._init_model_component(config, logger, reporting_manager)

        # Runner-specific: code_dir handling
        self.code_dir = self._get_config_value(
            lambda: self.config.system.code_dir,
            default=None
        )
        if self.code_dir:
            self.code_dir = Path(self.code_dir)

        # Workers set quiet=True to suppress per-iteration INFO messages
        self.quiet = False

        # Track resolved executables for provenance capture
        self._resolved_executables: List[tuple] = []

        # Allow subclasses to perform custom setup before output dir creation
        self._setup_model_specific_paths()

        # Validate spatial mode compatibility (Phase 3)
        self._validate_spatial_mode()

        # Create output directory if configured to do so
        if self._should_create_output_dir():
            self.output_dir = self._get_output_dir()
            self.ensure_dir(self.output_dir)

    def _validate_required_config(self) -> None:
        """
        Validate that all required configuration keys are present.

        Subclasses can override to add model-specific required keys.

        Raises:
            ConfigurationError: If required keys are missing
        """
        required_keys = [
            'SYMFLUENCE_DATA_DIR',
            'DOMAIN_NAME',
        ]
        self.validate_config(
            required_keys,
            f"{self.MODEL_NAME or self.__class__.__name__} runner initialization"
        )

    def _validate_spatial_mode(self) -> None:
        """
        Validate spatial mode compatibility for this model.

        Called during initialization to ensure the model supports the configured
        spatial mode and that routing is properly configured if required.

        Phase 3 Addition: Centralized spatial mode validation across all models.

        Logs warnings for suboptimal configurations but does not raise exceptions
        by default to maintain backward compatibility.
        """
        try:
            # Get current spatial mode from config
            spatial_mode = get_spatial_mode_from_config(self.config_dict)

            # Check if routing is configured
            routing_model = self._get_config_value(lambda: self.config.model.routing_model, default='none')
            has_routing = routing_model and routing_model.lower() not in ('none', 'default', '')

            # Validate against model capabilities
            is_valid, message = validate_spatial_mode(
                self.model_name,
                spatial_mode,
                has_routing_configured=has_routing
            )

            if message:
                if is_valid:
                    # It's a warning, not an error
                    self.logger.warning(f"Spatial mode validation: {message}")
                else:
                    # Configuration is invalid
                    self.logger.error(f"Spatial mode validation error: {message}")
                    # Don't raise to maintain backward compatibility
                    # Subclasses can override to raise if needed

        except Exception as e:  # noqa: BLE001 — model execution resilience
            # Don't let validation failures prevent model initialization
            self.logger.debug(f"Spatial mode validation skipped: {e}")

    def _has_routing_configured(self) -> bool:
        """
        Check if a routing model is configured.

        Returns:
            True if routing model is configured, False otherwise
        """
        routing_model = self._get_config_value(lambda: self.config.model.routing_model, default='none')
        return routing_model and routing_model.lower() not in ('none', 'default', '')

    def _get_model_name(self) -> str:
        """
        Return the name of the model.

        Prefers the ``MODEL_NAME`` class variable. Subclasses that still
        override this method will continue to work.

        Returns:
            Model name (e.g., 'SUMMA', 'FUSE', 'GR')
        """
        if self.MODEL_NAME:
            return self.MODEL_NAME
        raise NotImplementedError(
            f"{self.__class__.__name__} must set MODEL_NAME class variable "
            "or implement _get_model_name()"
        )

    def _setup_model_specific_paths(self) -> None:
        """
        Hook for subclasses to set up model-specific paths.

        Called after base paths are initialized but before output_dir creation.
        Override this method to add model-specific path attributes.

        Example:
            def _setup_model_specific_paths(self):
                self.setup_dir = self.project_dir / "settings" / self.model_name
                self.forcing_path = self.project_forcing_dir / f'{self.model_name}_input'
        """
        pass

    def _should_create_output_dir(self) -> bool:
        """
        Determine if output directory should be created in __init__.

        Default behavior is to create it. Subclasses can override.

        Returns:
            True if output_dir should be created, False otherwise
        """
        return True

    def _get_output_dir(self) -> Path:
        """
        Get the output directory path for this model run.

        Default implementation uses EXPERIMENT_ID from config.
        Subclasses can override for custom behavior.

        Returns:
            Path to output directory
        """
        experiment_id = self.config.domain.experiment_id
        return self.project_dir / 'simulations' / experiment_id / self.model_name

    # =========================================================================
    # Template Method Hooks for Default run()
    # =========================================================================

    def _build_run_command(self) -> Optional[List[str]]:
        """Build subprocess command. Return None for in-process models.

        Override this to opt into the default ``run()`` template. The
        template handles logging, subprocess execution, and output
        verification automatically.

        Returns:
            List of command arguments, or None to skip template execution.
        """
        return None

    def _get_expected_outputs(self) -> List[str]:
        """Return expected output filenames for verification.

        Empty list skips verification.

        Returns:
            List of filenames expected in ``output_dir``.
        """
        return []

    def _get_run_cwd(self) -> Optional[Path]:
        """Working directory for subprocess. Defaults to ``setup_dir``.

        Returns:
            Path to working directory, or None for default.
        """
        return getattr(self, 'setup_dir', None)

    def _get_run_environment(self) -> Optional[Dict[str, str]]:
        """Extra environment variables merged into ``os.environ``.

        Return None to use the default environment.

        Returns:
            Dict of extra env vars, or None.
        """
        return None

    def _get_run_timeout(self) -> int:
        """Subprocess timeout in seconds.

        Returns:
            Timeout value (default 3600).
        """
        return 3600

    def _prepare_run(self) -> None:
        """Pre-run preparation hook.

        Called after ``output_dir`` is created but before subprocess
        execution.  Use this to copy files to ``output_dir``, create
        required subdirectories, etc.
        """
        pass

    # =========================================================================
    # run() — Template Method + Legacy Dispatch
    # =========================================================================

    def run(self, **kwargs) -> Optional[Path]:
        """Execute the model.

        Dispatch order:

        1. **Legacy dispatch**: If the registry has a ``method_name`` other
           than ``'run'`` for this model (e.g. ``'run_fuse'``), delegate to
           that method.  This preserves backward compatibility for complex
           runners.
        2. **Template execution**: If ``_build_run_command()`` returns a
           command list, execute it via ``execute_subprocess``, verify
           outputs, and return ``output_dir``.
        3. Raise ``NotImplementedError`` if neither path applies.

        Args:
            **kwargs: Passed to the legacy run method if used.

        Returns:
            Path to output directory on success, None on failure.
        """
        from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
        from symfluence.models.registry import ModelRegistry

        # --- Legacy dispatch ---
        method_name = ModelRegistry.get_runner_method(self.model_name)
        if method_name and method_name != 'run':
            run_method = getattr(self, method_name, None)
            if run_method is not None:
                return run_method(**kwargs)

        # --- Default template ---
        cmd = self._build_run_command()
        if cmd is not None:
            _log = self.logger.debug if self.quiet else self.logger.info
            _log(f"Running {self.model_name} for domain: {self.domain_name}")

            with symfluence_error_handler(
                f"{self.model_name} model execution",
                self.logger,
                error_type=ModelExecutionError,
            ):
                # Ensure output directory exists
                self.output_dir = self._get_output_dir()
                self.ensure_dir(self.output_dir)

                # Pre-run preparation (file staging, etc.)
                self._prepare_run()

                # Set up logging
                log_dir = self.get_log_path()
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f'{self.model_name.lower()}_run_{current_time}.log'

                # Build environment
                env = self._get_run_environment()

                # Execute — use DEBUG log level during calibration (quiet mode)
                import logging as _logging
                _log_level = _logging.DEBUG if self.quiet else _logging.INFO
                result = self.execute_subprocess(
                    cmd,
                    log_file,
                    cwd=self._get_run_cwd(),
                    env=env,
                    timeout=self._get_run_timeout(),
                    check=False,
                    success_message=f"{self.model_name} simulation completed successfully",
                    success_log_level=_log_level,
                )

                if not result.success:
                    self.logger.error(
                        f"{self.model_name} simulation failed "
                        f"(rc={result.return_code})"
                    )
                    return None

                # Verify expected outputs
                expected = self._get_expected_outputs()
                if expected and not self.verify_model_outputs(expected):
                    return None

                return self.output_dir

        # --- Neither path matched ---
        raise NotImplementedError(
            f"Model {self.model_name} must either override run(), implement "
            f"_build_run_command(), or register a method_name in the registry."
        )

    def backup_settings(self, source_dir: Path, backup_subdir: str = "run_settings") -> None:
        """
        Backup settings files to the output directory for reproducibility.

        Args:
            source_dir: Source directory containing settings to backup
            backup_subdir: Subdirectory name within output_dir for backups

        Raises:
            FileOperationError: If backup fails
        """
        if not hasattr(self, 'output_dir'):
            self.logger.warning("Cannot backup settings: output_dir not initialized")
            return

        backup_path = self.output_dir / backup_subdir
        self.ensure_dir(backup_path)

        # Copy all files from source to backup using copy_file and copy_tree
        for item in source_dir.iterdir():
            if item.is_file():
                self.copy_file(item, backup_path / item.name)
            elif item.is_dir() and not item.name.startswith('.'):
                self.copy_tree(item, backup_path / item.name)

        self.logger.info(f"Settings backed up to {backup_path}")

    def get_log_path(self, log_subdir: str = "logs") -> Path:
        """
        Get or create log directory path for this model run.

        Args:
            log_subdir: Subdirectory name for logs

        Returns:
            Path to log directory (created if it doesn't exist)
        """
        if hasattr(self, 'output_dir'):
            log_path = self.output_dir / log_subdir
        else:
            # Fallback if output_dir not set
            experiment_id = self.experiment_id
            log_path = self.project_dir / 'simulations' / experiment_id / self.model_name / log_subdir

        return self.ensure_dir(log_path)

    def get_install_path(
        self,
        config_key: str,
        default_subpath: str,
        relative_to: str = 'data_dir',
        must_exist: bool = False,
        typed_accessor: Optional[Any] = None
    ) -> Path:
        """
        Resolve model installation path from config or use default.

        Args:
            config_key: Configuration key (e.g., 'SUMMA_INSTALL_PATH')
            default_subpath: Default path relative to base (e.g., 'installs/summa/bin')
            relative_to: Base directory ('data_dir' or 'project_dir')
            must_exist: If True, raise FileNotFoundError if path doesn't exist
            typed_accessor: Optional lambda to access typed config directly

        Returns:
            Path to installation directory

        Raises:
            FileNotFoundError: If must_exist=True and path doesn't exist

        Example:
            self.summa_exe = self.get_install_path(
                'SUMMA_INSTALL_PATH',
                'installs/summa/bin',
                must_exist=True,
                typed_accessor=lambda: self.config.model.summa.install_path
            ) / 'summa.exe'
        """
        self.logger.debug(f"Resolving install path for key: {config_key}, default: {default_subpath}, relative_to: {relative_to}")

        # Get install path from typed config or config_dict
        if typed_accessor:
            install_path = self._get_config_value(typed_accessor, default='default')
        else:
            # Fallback to config_dict for legacy keys
            install_path = self._get_config_value(lambda: None, default='default', dict_key=config_key)

        if install_path == 'default' or install_path is None:
            if relative_to == 'data_dir':
                path = self.data_dir / default_subpath
                # Fallback search if not found in current data_dir
                if not path.exists():
                    # 1. Try code_dir
                    if self.code_dir:
                        fallback_path = self.code_dir / default_subpath
                        if fallback_path.exists():
                            self.logger.debug(f"Default path not found in data_dir, using fallback from code_dir: {fallback_path}")
                            path = fallback_path
                        else:
                            # 2. Try default sibling data directory (SYMFLUENCE_data)
                            sibling_data = self.code_dir.parent / 'SYMFLUENCE_data'
                            fallback_path = sibling_data / default_subpath
                            if fallback_path.exists():
                                self.logger.debug(f"Default path not found in data_dir or code_dir, using fallback from sibling data dir: {fallback_path}")
                                path = fallback_path
            elif relative_to == 'code_dir':
                path = self.code_dir / default_subpath if self.code_dir else self.data_dir / default_subpath
                # Fallback search if not found in code_dir
                if self.code_dir and not path.exists():
                    # Try default sibling data directory (SYMFLUENCE_data)
                    sibling_data = self.code_dir.parent / 'SYMFLUENCE_data'
                    fallback_path = sibling_data / default_subpath
                    if fallback_path.exists():
                        self.logger.debug(f"Default path not found in code_dir, using fallback from sibling data dir: {fallback_path}")
                        path = fallback_path
            else:
                path = self.project_dir / default_subpath
            self.logger.debug(f"Resolved default install path: {path}")
        else:
            path = Path(install_path)
            self.logger.debug(f"Using custom install path: {path}")

        # Optional validation
        if must_exist and not path.exists():
            raise FileNotFoundError(
                f"Installation path not found: {path}\n"
                f"Config key: {config_key}"
            )

        return path

    def get_model_executable(
        self,
        install_path_key: str,
        default_install_subpath: str,
        exe_name_key: Optional[str] = None,
        default_exe_name: Optional[str] = None,
        typed_exe_accessor: Optional[Any] = None,
        relative_to: str = 'data_dir',
        must_exist: bool = False,
        candidates: Optional[List[str]] = None
    ) -> Path:
        """
        Resolve complete model executable path (install dir + exe name).

        Standardizes the common pattern of:
        1. Resolving installation directory from config
        2. Resolving executable name from config
        3. Combining them into full executable path

        Args:
            install_path_key: Config key for install directory (e.g., 'FUSE_INSTALL_PATH')
            default_install_subpath: Default install dir (e.g., 'installs/fuse/bin')
            exe_name_key: Config key for exe name (e.g., 'FUSE_EXE')
            default_exe_name: Default exe name (e.g., 'fuse.exe')
            typed_exe_accessor: Optional lambda for typed config exe name
            relative_to: Base directory ('data_dir' or 'project_dir')
            must_exist: If True, raise FileNotFoundError if executable doesn't exist
            candidates: Optional list of subdirectory candidates to try.
                        The method tries each candidate in order and returns
                        the first existing path. Use '' for the root install dir.
                        e.g., ['', 'cmake_build', 'bin'] for NGEN

        Returns:
            Complete path to model executable

        Raises:
            FileNotFoundError: If must_exist=True and executable doesn't exist

        Example:
            >>> # Simple case
            >>> self.fuse_exe = self.get_model_executable(
            ...     'FUSE_INSTALL_PATH',
            ...     'installs/fuse/bin',
            ...     'FUSE_EXE',
            ...     'fuse.exe'
            ... )

            >>> # With typed config accessor
            >>> self.mesh_exe = self.get_model_executable(
            ...     'MESH_INSTALL_PATH',
            ...     'installs/MESH-DEV',
            ...     'MESH_EXE',
            ...     'sa_mesh',
            ...     typed_exe_accessor=lambda: self.config.model.mesh.exe if self.config.model.mesh else None
            ... )

            >>> # With candidates (search multiple subdirectories)
            >>> self.ngen_exe = self.get_model_executable(
            ...     'NGEN_INSTALL_PATH',
            ...     'installs/ngen',
            ...     default_exe_name='ngen',
            ...     candidates=['', 'cmake_build', 'bin'],
            ...     must_exist=True
            ... )
        """
        # Get installation directory
        install_dir = self.get_install_path(
            install_path_key,
            default_install_subpath,
            relative_to=relative_to,
            must_exist=False  # We'll check exe existence instead
        )

        # Get executable name
        if typed_exe_accessor:
            exe_name = self._get_config_value(typed_exe_accessor, default=default_exe_name)
        elif exe_name_key:
            exe_name = self._get_config_value(lambda: None, default=default_exe_name, dict_key=exe_name_key)
        else:
            exe_name = default_exe_name

        # Handle candidates: try each subdirectory in order
        if candidates:
            for candidate in candidates:
                if candidate:
                    candidate_path = install_dir / candidate / exe_name
                else:
                    candidate_path = install_dir / exe_name

                if candidate_path.exists():
                    self.logger.debug(f"Found executable at: {candidate_path}")
                    return candidate_path

            # No candidate found - use first candidate (or root) as default path for error message
            exe_path = install_dir / (candidates[0] if candidates[0] else '') / exe_name
        else:
            # Standard behavior - combine into full path
            exe_path = install_dir / exe_name

        # Optional validation
        if must_exist and not exe_path.exists():
            if candidates:
                searched_paths = [
                    str(install_dir / (c if c else '') / exe_name)
                    for c in candidates
                ]
                raise FileNotFoundError(
                    "Model executable not found in any candidate location.\n"
                    "Searched paths:\n  " + "\n  ".join(searched_paths) + "\n"
                    f"Install path key: {install_path_key}"
                )
            else:
                raise FileNotFoundError(
                    f"Model executable not found: {exe_path}\n"
                    f"Install path key: {install_path_key}\n"
                    f"Exe name key: {exe_name_key}"
                )

        # Track for provenance capture
        label = install_path_key.replace('_INSTALL_PATH', '').replace('_PATH', '')
        self._resolved_executables.append((label, exe_path))

        return exe_path

    # =========================================================================
    # File Verification
    # =========================================================================

    def verify_required_files(
        self,
        files: Union[Path, List[Path]],
        context: str = "model execution"
    ) -> None:
        """
        Verify that required files exist, raise FileNotFoundError if missing.

        Args:
            files: Single path or list of paths to verify
            context: Description of what these files are for (used in error message)

        Raises:
            FileNotFoundError: If any required file is missing
        """
        # Normalize to list
        if isinstance(files, Path):
            files = [files]

        # Check existence
        missing_files = [f for f in files if not f.exists()]

        if missing_files:
            error_msg = f"Required files for {context} not found:\n"
            error_msg += "\n".join(f"  - {f}" for f in missing_files)
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.debug(f"Verified {len(files)} required file(s) for {context}")

    def get_config_path(
        self,
        config_key: str,
        default_subpath: str,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve configuration path with default fallback.

        This is a convenience wrapper around PathResolverMixin._get_default_path
        with consistent naming for model runners.

        Args:
            config_key: Configuration key to look up
            default_subpath: Default path relative to project_dir
            must_exist: Whether to raise error if path doesn't exist

        Returns:
            Resolved Path object
        """
        return self._get_default_path(config_key, default_subpath, must_exist)

    def verify_model_outputs(
        self,
        expected_files: Union[str, List[str]],
        output_dir: Optional[Path] = None
    ) -> bool:
        """
        Verify that expected model output files exist.

        Args:
            expected_files: Single filename or list of expected output filenames
            output_dir: Directory to check (defaults to self.output_dir)

        Returns:
            True if all files exist, False otherwise
        """
        if isinstance(expected_files, str):
            expected_files = [expected_files]

        check_dir = output_dir or self.output_dir

        missing_files = []
        for filename in expected_files:
            if not (check_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            self.logger.error(
                f"Missing {len(missing_files)} expected output file(s) in {check_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )
            return False

        self.logger.debug(f"Verified {len(expected_files)} output file(s) in {check_dir}")
        return True

    def get_experiment_output_dir(
        self,
        experiment_id: Optional[str] = None
    ) -> Path:
        """
        Get the experiment-specific output directory for this model.

        Standard pattern: {project_dir}/simulations/{experiment_id}/{model_name}

        Args:
            experiment_id: Experiment identifier (defaults to config.domain.experiment_id)

        Returns:
            Path to experiment output directory
        """
        exp_id = experiment_id or self.config.domain.experiment_id
        return self.project_dir / 'simulations' / exp_id / self.model_name

    def setup_path_aliases(self, aliases: Dict[str, str]) -> None:
        """
        Set up legacy path aliases for backward compatibility.

        Args:
            aliases: Dictionary mapping alias name to source attribute
                     Example: {'root_path': 'data_dir', 'result_dir': 'output_dir'}
        """
        for alias, source_attr in aliases.items():
            if hasattr(self, source_attr):
                setattr(self, alias, getattr(self, source_attr))
                self.logger.debug(f"Set legacy alias: {alias} -> {source_attr}")
            else:
                self.logger.warning(
                    f"Cannot create alias '{alias}': source attribute '{source_attr}' not found"
                )
