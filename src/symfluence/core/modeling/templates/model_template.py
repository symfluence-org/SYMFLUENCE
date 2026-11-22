# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
UnifiedModelRunner - Template for implementing new hydrological models.

This module provides a base class that combines:
- BaseModelRunner (paths, config, logging, subprocess/SLURM execution)
- SpatialOrchestrator (spatial modes, routing)

New models can inherit from UnifiedModelRunner and implement only the
model-specific methods, reducing boilerplate by ~50%.

Example Implementation:
    @R.runners.add('MYMODEL', runner_method='run_mymodel')
    class MyModelRunner(UnifiedModelRunner):
        def _get_model_name(self) -> str:
            return "MYMODEL"

        def _build_command(self) -> List[str]:
            return [str(self.model_exe), '-c', str(self.config_file)]

        def _post_execution(self, result: ExecutionResult) -> ModelRunResult:
            if result.success:
                return ModelRunResult(success=True, output_path=self.output_dir)
            return ModelRunResult(success=False, error=result.error_message)
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from symfluence.core.exceptions import ConfigurationError, ModelExecutionError, symfluence_error_handler
from symfluence.core.registries import R

from ..base import BaseModelRunner
from ..config_schema import ModelConfigSchema, get_model_schema
from ..execution import (
    ExecutionMode,
    ExecutionResult,
    SlurmJobConfig,
    SpatialOrchestrator,
)


@dataclass
class ModelRunResult:
    """Result of a complete model run.

    Attributes:
        success: Whether the run completed successfully
        output_path: Path to primary output (file or directory)
        routed_output: Path to routed output (if routing was performed)
        metrics: Performance metrics from the run
        error: Error message if run failed
        warnings: Non-fatal warnings from the run
        metadata: Additional run information
    """
    success: bool
    output_path: Optional[Path] = None
    routed_output: Optional[Path] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedModelRunner(BaseModelRunner, SpatialOrchestrator):  # type: ignore[misc]
    """
    Base class for hydrological model runners using the unified framework.

    This class combines all framework components into a single, easy-to-use
    base class. New models inherit from this and implement a minimal set
    of abstract methods.

    The run workflow is:
        1. validate_config() - Validate configuration
        2. prepare_execution() - Set up paths, files, environment
        3. execute() - Run the model (subprocess or SLURM)
        4. post_execution() - Process outputs, run routing
        5. finalize() - Cleanup, metrics collection

    Subclasses must implement:
        - _get_model_name() -> str
        - _build_command() -> List[str]

    Subclasses may override:
        - _setup_model_specific_paths()
        - _get_environment() -> Dict[str, str]
        - _validate_model_specific() -> List[str]
        - _pre_execution() -> bool
        - _post_execution(result) -> ModelRunResult

    Attributes:
        schema: ModelConfigSchema for this model
        spatial_config: SpatialConfig for this run
        model_exe: Path to model executable
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: Any,
        reporting_manager: Optional[Any] = None
    ):
        # Initialize base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Load model schema
        self.schema = self._get_model_schema()

        # Validate configuration
        self._validate_and_apply_defaults()

        # Set up spatial configuration
        self.spatial_config = self.get_spatial_config(self.model_name)

        # Set up model executable path
        self.model_exe = self._setup_model_executable()

    def _get_model_schema(self) -> Optional[ModelConfigSchema]:
        """Get configuration schema for this model."""
        try:
            return get_model_schema(self._get_model_name())
        except KeyError:
            self.logger.debug(f"No schema registered for {self._get_model_name()}")
            return None

    def _validate_and_apply_defaults(self) -> None:
        """Validate configuration and apply defaults from schema."""
        if self.schema is None:
            return

        # Get config dict for validation
        # For typed config (SymfluenceConfig), defaults are already applied via pydantic
        # For legacy dict config, we need to apply defaults
        config_for_validation = self.config_dict

        # Only apply defaults for legacy dict config (not typed SymfluenceConfig)
        # Typed config handles defaults internally via pydantic
        if self.config is None or isinstance(self.config, dict):
            config_for_validation = self.schema.apply_defaults(config_for_validation)

        # Validate
        errors = self.schema.validate(config_for_validation)
        model_errors = self._validate_model_specific()
        errors.extend(model_errors)

        if errors:
            error_msg = f"Configuration errors for {self.model_name}:\n"
            error_msg += "\n".join(f"  - {e}" for e in errors)
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)

    def _validate_model_specific(self) -> List[str]:
        """
        Perform model-specific validation.

        Override to add custom validation logic.

        Returns:
            List of error messages (empty if valid)
        """
        return []

    def _setup_model_executable(self) -> Optional[Path]:
        """
        Set up path to model executable.

        Uses schema if available, otherwise falls back to standard pattern.
        """
        if self.schema and self.schema.installation:
            return self.get_model_executable(
                install_path_key=self.schema.installation.install_path_key,
                default_install_subpath=self.schema.installation.default_install_subpath,
                exe_name_key=self.schema.installation.exe_name_key,
                default_exe_name=self.schema.installation.default_exe_name,
                must_exist=True
            )
        return None

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    def _get_model_name(self) -> str:
        """Return the model name (e.g., 'SUMMA', 'FUSE').

        Prefers the ``MODEL_NAME`` class variable. Subclasses that still
        override this method will continue to work.
        """
        if self.MODEL_NAME:
            return self.MODEL_NAME
        raise NotImplementedError(
            f"{self.__class__.__name__} must set MODEL_NAME class variable "
            "or implement _get_model_name()"
        )

    @abstractmethod
    def _build_command(self) -> List[str]:
        """
        Build the command to execute the model.

        Returns:
            Command as list of strings for subprocess

        Example:
            return [str(self.model_exe), '-m', str(self.file_manager)]
        """
        pass

    # =========================================================================
    # Optional Hooks (may be overridden by subclasses)
    # =========================================================================

    def _get_environment(self) -> Dict[str, str]:
        """
        Get environment variables for model execution.

        Override to add model-specific environment variables.

        Returns:
            Dictionary of environment variable names and values
        """
        return {}

    def _pre_execution(self) -> bool:
        """
        Perform pre-execution setup.

        Override to add model-specific preparation (file generation, etc.)

        Returns:
            True if setup successful, False to abort execution
        """
        return True

    def _post_execution(self, result: ExecutionResult) -> ModelRunResult:
        """
        Process execution results and generate ModelRunResult.

        Override to add model-specific output processing.

        Args:
            result: ExecutionResult from model run

        Returns:
            ModelRunResult with processed outputs
        """
        return ModelRunResult(
            success=result.success,
            output_path=self.output_dir if hasattr(self, 'output_dir') else None,
            error=result.error_message,
            metadata={
                'duration_seconds': result.duration_seconds,
                'return_code': result.return_code,
            }
        )

    def _get_slurm_config(self) -> Optional[SlurmJobConfig]:
        """
        Get SLURM configuration for parallel execution.

        Override to customize SLURM job parameters.

        Returns:
            SlurmJobConfig for SLURM execution, or None for local
        """
        if not self.schema or not self.schema.execution.supports_parallel:
            return None

        parallel_key = self.schema.execution.parallel_key
        if parallel_key and not self._get_config_value(lambda: None, default=False, dict_key=parallel_key):
            return None

        return SlurmJobConfig(
            job_name=f"{self.model_name}-{self.domain_name}",
            time_limit="03:00:00",
            memory=self.schema.execution.default_memory,
            cpus_per_task=self.schema.execution.default_cpus,
        )

    # =========================================================================
    # Main Execution Interface
    # =========================================================================

    def run(self, **kwargs: Any) -> Optional[Path]:  # type: ignore[override]
        """
        Execute the model using the unified framework.

        This is the main entry point that orchestrates:
        1. Pre-execution setup
        2. Model execution (local or SLURM)
        3. Post-execution processing
        4. Routing (if required)

        Returns:
            Path to output directory on success, None on failure

        Raises:
            ModelExecutionError: If execution fails critically
        """
        with symfluence_error_handler(
            f"{self.model_name} model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            self.logger.info(f"Starting {self.model_name} model run")

            # Pre-execution setup
            if not self._pre_execution():
                self.logger.error("Pre-execution setup failed")
                return None

            # Build command
            command = self._build_command()
            self.logger.debug(f"Command: {' '.join(command)}")

            # Determine execution mode
            slurm_config = self._get_slurm_config()
            mode = ExecutionMode.SLURM if slurm_config else ExecutionMode.LOCAL

            # Execute
            log_file = self.get_log_path() / f"{self.model_name.lower()}_run.log"

            # Determine timeout
            timeout = self.schema.execution.default_timeout if self.schema else 3600

            # Check for override in config
            timeout_key = f"{self.model_name}_TIMEOUT"
            if timeout_key in self.config_dict:
                try:
                    timeout = int(self.config_dict[timeout_key])
                    self.logger.debug(f"Using overridden timeout: {timeout}s")
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid timeout value for {timeout_key}, using default: {timeout}s")

            result = self.execute_in_mode(
                mode=mode,
                command=command,
                log_file=log_file,
                slurm_config=slurm_config,
                env=self._get_environment(),
                cwd=getattr(self, 'setup_dir', None),
                timeout=timeout,
            )

            # Post-execution processing
            run_result = self._post_execution(result)

            # Handle routing if needed
            if run_result.success and self.spatial_config.requires_routing():
                self.logger.info("Executing routing")
                routed = self.route_model_output(
                    run_result.output_path,
                    self.spatial_config
                )
                if routed:
                    run_result.routed_output = routed
                else:
                    run_result.warnings.append("Routing failed")

            # Log results
            if run_result.success:
                self.logger.info(f"{self.model_name} run completed successfully")
                return run_result.output_path
            else:
                self.logger.error(f"{self.model_name} run failed: {run_result.error}")
                return None

    def run_parallel(
        self,
        n_units: int,
        units_per_job: Optional[int] = None,
        unit_type: str = "gru"
    ) -> Optional[Path]:
        """
        Execute the model in parallel using SLURM arrays.

        Args:
            n_units: Total number of units (GRUs, subcatchments, etc.)
            units_per_job: Units per array task (auto-calculated if None)
            unit_type: Type of unit for logging

        Returns:
            Path to merged output on success, None on failure
        """
        if not self.is_slurm_available():
            self.logger.warning("SLURM not available, falling back to serial execution")
            return self.run()

        self.logger.info(f"Starting parallel {self.model_name} run with {n_units} {unit_type}s")

        # Calculate optimal parallelization
        if units_per_job is None:
            units_per_job = self.estimate_optimal_grus_per_job(n_units)

        self.logger.info(f"Using {units_per_job} {unit_type}s per job")

        # Create parallel script
        script_content = self.create_gru_parallel_script(
            model_exe=self.model_exe,
            file_manager=getattr(self, 'file_manager', self.setup_dir / 'fileManager.txt'),
            log_dir=self.get_log_path(),
            total_grus=n_units,
            grus_per_job=units_per_job,
            job_name=f"{self.model_name}-{self.domain_name}",
        )

        # Write and submit
        script_path = self.project_dir / f"run_{self.model_name.lower()}_parallel.sh"
        script_path.write_text(script_content, encoding='utf-8')
        script_path.chmod(0o755)

        result = self.submit_slurm_job(
            script_path=script_path,
            wait=self._get_config_value(
                lambda: self.config.model.summa.monitor_slurm_job,
                dict_key='MONITOR_SLURM_JOB',
            )
        )

        if result.success and not self.slurm_job_finished(result):
            self.logger.warning(
                f"SLURM job {result.job_id} was submitted without monitoring "
                f"(MONITOR_SLURM_JOB is off); it has not finished, so outputs "
                f"are not merged. Merge after the job completes."
            )
            return None

        if result.success:
            # Merge outputs
            return self._merge_parallel_outputs()
        else:
            self.logger.error(f"Parallel execution failed: {result.error_message}")
            return None

    def _merge_parallel_outputs(self) -> Optional[Path]:
        """
        Merge outputs from parallel execution.

        Override to implement model-specific merging logic.

        Returns:
            Path to merged output
        """
        self.logger.warning("Parallel output merging not implemented for this model")
        return self.output_dir if hasattr(self, 'output_dir') else None


# =============================================================================
# Factory Function
# =============================================================================

def create_model_runner(
    model_name: str,
    config: Union[Dict[str, Any], Any],
    logger: Any,
    reporting_manager: Optional[Any] = None
) -> UnifiedModelRunner:
    """
    Create a model runner instance using the registry.

    This is a convenience function that looks up the registered runner
    for a model and creates an instance.

    Args:
        model_name: Name of the model (case-insensitive)
        config: Configuration dict or SymfluenceConfig
        logger: Logger instance
        reporting_manager: Optional ReportingManager

    Returns:
        Configured model runner instance

    Raises:
        KeyError: If model is not registered
    """
    runner_class = R.runners.get(model_name)
    if runner_class is None:
        raise KeyError(f"No runner registered for model: {model_name}")
    return runner_class(config, logger, reporting_manager=reporting_manager)
