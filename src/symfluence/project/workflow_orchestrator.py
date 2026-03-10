# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Workflow orchestration for SYMFLUENCE hydrological modeling pipeline.

Coordinates the execution sequence of modeling steps including domain definition,
data preprocessing, model execution, optimization, and analysis phases.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from symfluence.core.config.coercion import ensure_config
from symfluence.core.exceptions import SYMFLUENCEError
from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.provenance import record_executable, record_step
from symfluence.core.stage_marker import (
    STAGE_CONFIG_SECTIONS,
    clear_markers,
    compute_config_hash,
    is_stage_current,
    write_marker,
)
from symfluence.data.observation.paths import observation_output_candidates_by_family

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@dataclass
class WorkflowStep(ConfigMixin):
    """
    Represents a single step in the SYMFLUENCE workflow.
    """
    name: str
    cli_name: str
    func: Callable
    check_func: Callable
    description: str


class WorkflowOrchestrator(ConfigMixin):
    """
    Orchestrates the SYMFLUENCE workflow execution and manages the step sequence.

    The WorkflowOrchestrator is responsible for defining, coordinating, and executing
    the complete SYMFLUENCE modeling workflow. It integrates the various manager
    components into a coherent sequence of operations, handling dependencies between
    steps, tracking progress, and providing status information.

    Key responsibilities:
    - Defining the sequence of workflow steps and their validation checks
    - Coordinating execution across different manager components
    - Handling execution flow (skipping completed steps, stopping on errors)
    - Providing status information and execution reports
    - Validating prerequisites before workflow execution

    This class represents the "conductor" of the SYMFLUENCE system, ensuring that
    each component performs its tasks in the correct order and with the necessary
    inputs from previous steps.

    Attributes:
        managers (Dict[str, Any]): Dictionary of manager instances
        config (SymfluenceConfig): Typed configuration object
        logger (logging.Logger): Logger instance
        domain_name (str): Name of the hydrological domain
        experiment_id (str): ID of the current experiment
        project_dir (Path): Path to the project directory
        logging_manager: Reference to logging manager for enhanced formatting
    """

    def __init__(
        self,
        managers: Dict[str, Any],
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        logging_manager=None,
        provenance=None,
    ):
        """
        Initialize the workflow orchestrator.

        Args:
            managers: Dictionary of manager instances for each functional area
            config: SymfluenceConfig instance (dicts are auto-converted)
            logger: Logger instance for recording operations
            logging_manager: Reference to LoggingManager for enhanced formatting
            provenance: Optional RunProvenance instance for step-level tracking

        Raises:
            KeyError: If essential configuration values are missing
        """
        self.managers = managers
        self._config = ensure_config(config)
        self.logger = logger
        self.logging_manager = logging_manager
        self.provenance = provenance
        self.domain_name = self.config.domain.name
        self.experiment_id = self.config.domain.experiment_id

        data_dir = self.config.system.data_dir
        if not data_dir:
            raise KeyError("system.data_dir not configured")

        self.project_dir = Path(data_dir) / f"domain_{self.domain_name}"

    @staticmethod
    def _normalize_config_list(value: Any) -> List[str]:
        """Normalize scalar/string/list config values to uppercase string tokens."""
        if value is None:
            return []

        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
            return [item.upper() for item in items]

        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value]
            return [item.upper() for item in items if item]

        normalized = str(value).strip()
        return [normalized.upper()] if normalized else []

    @staticmethod
    def _tokens_include(tokens: List[str], *needles: str) -> bool:
        """Return True when any needle is present in any token."""
        return any(any(needle in token for needle in needles) for token in tokens)

    def _observation_output_paths(self) -> Dict[str, List[Path]]:
        """Canonical + legacy candidate output paths by observation family."""
        return observation_output_candidates_by_family(self.project_dir, self.domain_name)

    def _has_observation_output(self, family: str) -> bool:
        """Return True when any candidate output exists for an observation family."""
        return any(path.exists() for path in self._observation_output_paths().get(family, []))

    def _check_observed_data_exists(self) -> bool:
        """
        Check if required observed data files exist based on configuration.

        Checks for required observation families based on config:
        - Streamflow data (if EVALUATION_DATA or ADDITIONAL_OBSERVATIONS includes streamflow-like sources)
        - Snow data (SWE, SCA if EVALUATION_DATA includes SWE/SCA or DOWNLOAD_MODIS_SNOW/DOWNLOAD_SNOTEL)
        - Soil moisture data (if EVALUATION_DATA includes SM_ISMN, SM_SMAP, etc.)
        - ET data (if EVALUATION_DATA includes ET)

        Returns:
            bool: True only if all required observation families have been processed
        """
        evaluation_data = self._normalize_config_list(
            self.config.evaluation.evaluation_data
        )
        additional_observations = self._normalize_config_list(
            self.config.data.additional_observations
        )
        requested_tokens = evaluation_data + additional_observations
        required_families: List[str] = []

        check_snow = (
            self._tokens_include(requested_tokens, "SWE", "SCA", "SNOW")
            or bool(self.config.evaluation.snotel.download)
            or bool(self.config.evaluation.modis_snow.download)
        )
        if check_snow:
            required_families.append("snow")

        check_soil_moisture = self._tokens_include(
            requested_tokens,
            "SM_",
            "SMAP",
            "ISMN",
            "SOIL_MOISTURE",
        )
        if check_soil_moisture:
            required_families.append("soil_moisture")

        check_streamflow = (
            self._tokens_include(
                requested_tokens,
                "STREAMFLOW",
                "DISCHARGE",
                "USGS_STREAMFLOW",
                "WSC_STREAMFLOW",
                "SMHI_STREAMFLOW",
                "LAMAH_ICE_STREAMFLOW",
                "GRDC_STREAMFLOW",
            )
            or bool(self.config.data.download_usgs_data)
            or bool(self.config.evaluation.streamflow.download_wsc)
        )
        if check_streamflow:
            required_families.append("streamflow")

        check_et = self._tokens_include(requested_tokens, "ET", "FLUXNET", "MODIS_ET", "OPENET")
        if check_et:
            required_families.append("et")

        # Default for generic configs: streamflow is the minimum expected observation.
        if not required_families:
            return self._has_observation_output("streamflow")

        return all(self._has_observation_output(family) for family in required_families)

    def define_workflow_steps(self) -> List[WorkflowStep]:
        """
        Define the workflow steps with their output validation checks and descriptions.

        Returns:
            List[WorkflowStep]: List of WorkflowStep objects
        """

        # Get configured analyses
        analyses = self.config.evaluation.analyses or []
        optimizations = self.config.optimization.methods or []

        return [
            # --- Project Initialization ---
            WorkflowStep(
                name="setup_project",
                cli_name="setup_project",
                func=self.managers['project'].setup_project,
                check_func=lambda: (self.project_dir / 'shapefiles').exists(),
                description="Setting up project structure and directories"
            ),

            # --- Geospatial Domain Definition and Analysis ---
            WorkflowStep(
                name="create_pour_point",
                cli_name="create_pour_point",
                func=self.managers['project'].create_pour_point,
                check_func=lambda: (self.project_dir / "shapefiles" / "pour_point" /
                        f"{self.domain_name}_pourPoint.shp").exists(),
                description="Creating watershed pour point"
            ),
            WorkflowStep(
                name="acquire_attributes",
                cli_name="acquire_attributes",
                func=self.managers['data'].acquire_attributes,
                check_func=lambda: (resolve_data_subdir(self.project_dir, 'attributes') / "soilclass" /
                        f"domain_{self.domain_name}_soil_classes.tif").exists(),
                description="Acquiring geospatial attributes and data"
            ),
            WorkflowStep(
                name="define_domain",
                cli_name="define_domain",
                func=self.managers['domain'].define_domain,
                check_func=lambda: (self.project_dir / "shapefiles" / "river_basins" /
                        f"{self.domain_name}_riverBasins_{self.config.domain.definition_method}.shp").exists(),
                description="Defining hydrological domain boundaries"
            ),
            WorkflowStep(
                name="discretize_domain",
                cli_name="discretize_domain",
                func=self.managers['domain'].discretize_domain,
                check_func=lambda: (self.project_dir / "shapefiles" / "catchment" /
                        f"{self.domain_name}_HRUs_{str(self.config.domain.discretization).replace(',','_')}.shp").exists(),
                description="Discretizing domain into hydrological response units"
            ),

            # --- Model-Agnostic Data Preprocessing ---
            WorkflowStep(
                name="process_observed_data",
                cli_name="process_observed_data",
                func=self.managers['data'].process_observed_data,
                check_func=self._check_observed_data_exists,
                description="Processing observed data"
            ),
            WorkflowStep(
                name="acquire_forcings",
                cli_name="acquire_forcings",
                func=self.managers['data'].acquire_forcings,
                check_func=lambda: (resolve_data_subdir(self.project_dir, 'forcing') / "raw_data").exists(),
                description="Acquiring meteorological forcing data"
            ),
            WorkflowStep(
                name="run_model_agnostic_preprocessing",
                cli_name="model_agnostic_preprocessing",
                func=self.managers['data'].run_model_agnostic_preprocessing,
                check_func=lambda: (resolve_data_subdir(self.project_dir, 'forcing') / "basin_averaged_data").exists(),
                description="Running model-agnostic data preprocessing"
            ),
            WorkflowStep(
                name="build_model_ready_store",
                cli_name="build_model_ready_store",
                func=self.managers['data'].build_model_ready_store,
                check_func=lambda: (self.project_dir / "data" / "model_ready").exists(),
                description="Building model-ready data store"
            ),

            # --- Model-Specific Preprocessing and Execution ---
            WorkflowStep(
                name="preprocess_models",
                cli_name="model_specific_preprocessing",
                func=self.managers['model'].preprocess_models,
                check_func=lambda: any((self.project_dir / "settings").glob(f"*_{self.config.model.hydrological_model or 'SUMMA'}*")),
                description="Preprocessing model-specific input files"
            ),
            WorkflowStep(
                name="run_models",
                cli_name="run_model",
                func=self.managers['model'].run_models,
                check_func=lambda: (self.project_dir / "simulations" /
                        f"{self.experiment_id}_{self.config.model.hydrological_model or 'SUMMA'}_output.nc").exists(),
                description="Running hydrological model simulation"
            ),
            WorkflowStep(
                name="postprocess_results",
                cli_name="postprocess_results",
                func=self.managers['model'].postprocess_results,
                check_func=lambda: (self.project_dir / "simulations" /
                        f"{self.experiment_id}_postprocessed.nc").exists(),
                description="Post-processing simulation results"
            ),

            # --- Optimization and Emulation Steps ---
            WorkflowStep(
                name="calibrate_model",
                cli_name="calibrate_model",
                func=self.managers['optimization'].calibrate_model,
                check_func=lambda: ('optimization' in optimizations and
                        (self.project_dir / "optimization" /
                        f"{self.experiment_id}_parallel_iteration_results.csv").exists()),
                description="Calibrating model parameters"
            ),

            # --- Analysis Steps ---
            WorkflowStep(
                name="run_benchmarking",
                cli_name="run_benchmarking",
                func=self.managers['analysis'].run_benchmarking,
                check_func=lambda: ('benchmarking' in analyses and
                        (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()),
                description="Running model benchmarking analysis"
            ),

            WorkflowStep(
                name="run_decision_analysis",
                cli_name="run_decision_analysis",
                func=self.managers['analysis'].run_decision_analysis,
                check_func=lambda: ('decision' in analyses and
                        (self.project_dir / "optimization" /
                        f"{self.experiment_id}_model_decisions_comparison.csv").exists()),
                description="Analyzing modeling decisions impact"
            ),

            WorkflowStep(
                name="run_sensitivity_analysis",
                cli_name="run_sensitivity_analysis",
                func=self.managers['analysis'].run_sensitivity_analysis,
                check_func=lambda: ('sensitivity' in analyses and
                        (self.project_dir / "reporting" / "sensitivity_analysis" /
                        "all_sensitivity_results.csv").exists()),
                description="Running parameter sensitivity analysis"
            ),

        ]

    def run_workflow(self, force_run: bool = False):
        """
        Run the complete workflow according to the defined steps.

        This method executes each step in the workflow sequence, handling:
        - Conditional execution based on existing outputs
        - Error handling with configurable stop-on-error behavior
        - Progress tracking and timing information
        - Comprehensive logging of each operation

        The workflow can be configured to:
        - Skip steps that have already been completed (default)
        - Force re-execution of all steps (force_run=True)
        - Continue or stop on errors (based on STOP_ON_ERROR config)

        Args:
            force_run (bool): If True, forces execution of all steps even if outputs exist.
                            If False (default), skips steps with existing outputs.

        Raises:
            Exception: If a step fails and STOP_ON_ERROR is True in configuration

        Note:
            The method provides detailed logging throughout execution, including:
            - Step headers with progress indicators
            - Execution timing for each step
            - Clear success/skip/failure indicators
            - Final summary statistics
        """
        # Check prerequisites
        if not self.validate_workflow_prerequisites():
            raise ValueError("Workflow prerequisites not met")

        # Log workflow start
        start_time = datetime.now()

        # FIXED: Use direct logging instead of non-existent format_section_header()
        self.logger.info("=" * 60)
        self.logger.info("SYMFLUENCE WORKFLOW EXECUTION")
        self.logger.info(f"Domain: {self.domain_name}")
        self.logger.info(f"Experiment: {self.experiment_id}")
        self.logger.info("=" * 60)

        # Get workflow steps
        workflow_steps = self.define_workflow_steps()
        total_steps = len(workflow_steps)
        completed_steps = 0
        skipped_steps = 0
        failed_steps = 0

        # Clear all markers when force-running the entire workflow
        if force_run:
            clear_markers(self.project_dir)

        # Execute each step
        for idx, step in enumerate(workflow_steps, 1):
            step_name = step.name

            # FIXED: Use log_step_header() instead of non-existent format_step_header()
            if self.logging_manager:
                self.logging_manager.log_step_header(idx, total_steps, step_name, step.description)
            else:
                self.logger.info(f"\nStep {idx}/{total_steps}: {step_name}")
                self.logger.info(f"{step.description}")
                self.logger.info("=" * 40)

            try:
                # Determine whether the step needs to run
                output_exists = step.check_func()
                sections = STAGE_CONFIG_SECTIONS.get(step_name, [])
                if sections:
                    current_hash = compute_config_hash(self._config, sections)
                    marker_current = is_stage_current(
                        self.project_dir, step_name, current_hash
                    )
                else:
                    current_hash = ""
                    marker_current = True  # unknown stages fall back to output-only

                needs_run = force_run or not output_exists or not marker_current

                if needs_run:
                    if output_exists and not marker_current and not force_run:
                        self.logger.info(
                            f"Configuration changed — re-executing: {step_name}"
                        )

                    step_start_time = datetime.now()
                    self.logger.info(f"Executing: {step.description}")

                    step.func()

                    # Record model executable versions in provenance
                    if step_name == "run_models" and self.provenance is not None:
                        model_mgr = self.managers.get('model')
                        for label, exe_path in getattr(model_mgr, 'resolved_executables', []):
                            record_executable(self.provenance, label, exe_path)

                    step_end_time = datetime.now()
                    duration = (step_end_time - step_start_time).total_seconds()

                    # Write marker after successful execution
                    if sections:
                        write_marker(self.project_dir, step_name, current_hash)

                    # FIXED: Use log_completion() instead of non-existent format_step_completion()
                    if self.logging_manager:
                        self.logging_manager.log_completion(
                            success=True,
                            message=step.description,
                            duration=duration
                        )
                    else:
                        self.logger.info(f"✓ Completed: {step_name} (Duration: {duration:.2f}s)")

                    completed_steps += 1
                    record_step(self.provenance, step_name, duration)
                else:
                    # Log skip
                    if self.logging_manager:
                        self.logging_manager.log_substep(f"Skipping: {step.description} (Output already exists)")
                    else:
                        self.logger.info(f"→ Skipping: {step_name} (Output already exists)")

                    skipped_steps += 1
                    record_step(self.provenance, step_name, 0.0, status="skipped")

            except (SYMFLUENCEError, FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
                # Log failure
                if self.logging_manager:
                    self.logging_manager.log_completion(
                        success=False,
                        message=f"{step.description}: {str(e)}"
                    )
                else:
                    self.logger.error(f"✗ Failed: {step_name}")
                    self.logger.error(f"Error: {str(e)}")

                failed_steps += 1
                record_step(self.provenance, step_name, 0.0, status="failed", error=str(e))

                # Decide whether to continue or stop
                if self.config.system.stop_on_error:
                    self.logger.error("Workflow stopped due to error (STOP_ON_ERROR=True)")
                    raise
                else:
                    self.logger.warning("Continuing despite error (STOP_ON_ERROR=False)")
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                if self.logging_manager:
                    self.logging_manager.log_completion(
                        success=False,
                        message=f"{step.description}: Unexpected error: {str(e)}"
                    )
                self.logger.exception(f"Unexpected failure in workflow step '{step_name}'")

                failed_steps += 1
                record_step(self.provenance, step_name, 0.0, status="failed", error=str(e))

                if self.config.system.stop_on_error:
                    self.logger.error("Workflow stopped due to unexpected error (STOP_ON_ERROR=True)")
                    raise
                self.logger.warning("Continuing despite unexpected error (STOP_ON_ERROR=False)")

        # Summary report
        end_time = datetime.now()
        total_duration = end_time - start_time

        # FIXED: Use direct logging instead of non-existent format_section_header()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 60)

        self.logger.info(f"Total execution time: {total_duration}")
        self.logger.info(f"Steps completed: {completed_steps}/{total_steps}")
        self.logger.info(f"Steps skipped: {skipped_steps}")

        if failed_steps > 0:
            self.logger.warning(f"Steps failed: {failed_steps}")
            self.logger.warning("Workflow completed with errors")
        else:
            self.logger.info("✓ Workflow completed successfully")

        self.logger.info("═" * 60)

    def validate_workflow_prerequisites(self) -> bool:
        """
        Validate that all prerequisites are met before running the workflow.

        Config-level validation (required keys, types, ranges) is handled by
        Pydantic at SymfluenceConfig construction time. This method focuses on
        runtime prerequisites: manager initialization and manager readiness.

        Returns:
            bool: True if all prerequisites are met, False otherwise
        """
        valid = True

        # Check manager initialization
        required_managers = ['project', 'domain', 'data', 'model', 'analysis', 'optimization']
        for manager_name in required_managers:
            if manager_name not in self.managers:
                self.logger.error(f"Required manager not initialized: {manager_name}")
                valid = False

        # Check manager readiness
        for name, manager in self.managers.items():
            if hasattr(manager, 'validate_readiness'):
                readiness = manager.validate_readiness()
                for check, passed in readiness.items():
                    if not passed:
                        self.logger.warning(f"Manager '{name}' readiness check failed: {check}")

        return valid

    def run_individual_steps(self, step_names: List[str], continue_on_error: bool = False) -> List[Dict[str, Any]]:
        """
        Execute a specific list of workflow steps by their CLI names.

        Args:
            step_names: List of step CLI names to execute
            continue_on_error: Whether to continue to next step if one fails

        Returns:
            List of dictionaries containing execution results for each step
        """
        # Resolve workflow steps from orchestrator
        workflow_steps = self.define_workflow_steps()
        cli_to_step = {step.cli_name: step for step in workflow_steps}

        results: List[Dict[str, Any]] = []

        self.logger.info(f"Starting individual step execution: {', '.join(step_names)}")

        for idx, cli_name in enumerate(step_names, 1):
            step = cli_to_step.get(cli_name)
            if not step:
                valid = ", ".join(sorted(cli_to_step.keys()))
                message = (
                    f"Step '{cli_name}' not recognized. "
                    f"Valid steps: {valid}"
                )
                self.logger.error(message)
                if self.logging_manager:
                    self.logging_manager.log_completion(False, message)
                results.append({"cli": cli_name, "fn": None, "success": False, "error": message})
                if not continue_on_error:
                    raise ValueError(message)
                continue

            # Log step header
            if self.logging_manager:
                self.logging_manager.log_step_header(idx, len(step_names), step.name, step.description)
            else:
                self.logger.info(f"\nExecuting step: {cli_name} -> {step.name}")

            step_start_time = datetime.now()

            try:
                # Force execution; skip completion checks for individual steps
                step.func()

                duration = (datetime.now() - step_start_time).total_seconds()

                # Write marker after successful execution
                sections = STAGE_CONFIG_SECTIONS.get(step.name, [])
                if sections:
                    current_hash = compute_config_hash(self._config, sections)
                    write_marker(self.project_dir, step.name, current_hash)

                if self.logging_manager:
                    self.logging_manager.log_completion(True, step.description, duration)
                else:
                    self.logger.info(f"✓ Completed step: {cli_name}")

                results.append({"cli": cli_name, "fn": step.name, "success": True, "duration": duration})
                record_step(self.provenance, step.name, duration)

            except (SYMFLUENCEError, FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
                self.logger.error(f"Step '{cli_name}' failed: {e}")

                if self.logging_manager:
                    self.logging_manager.log_completion(False, f"{step.description}: {str(e)}")

                results.append({"cli": cli_name, "fn": step.name, "success": False, "error": str(e)})
                record_step(self.provenance, step.name, 0.0, status="failed", error=str(e))

                if not continue_on_error:
                    raise
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.exception(f"Unexpected failure in step '{cli_name}'")

                if self.logging_manager:
                    self.logging_manager.log_completion(False, f"{step.description}: Unexpected error: {str(e)}")

                results.append({"cli": cli_name, "fn": step.name, "success": False, "error": str(e)})
                record_step(self.provenance, step.name, 0.0, status="failed", error=str(e))

                if not continue_on_error:
                    raise

        return results

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get the current status of the workflow execution.

        This method examines each step in the workflow to determine whether it has
        been completed, using the same output validation checks used during execution.
        It provides a comprehensive view of workflow progress, including which steps
        are complete and which are pending.

        The status information is useful for:
        - Monitoring long-running workflows
        - Generating progress reports
        - Diagnosing execution issues
        - Providing feedback to users

        Returns:
            Dict[str, Any]: Dictionary containing workflow status information, including:
                - total_steps: Total number of workflow steps
                - completed_steps: Number of completed steps
                - pending_steps: Number of pending steps
                - step_details: List of dictionaries with details for each step
                  (name and completion status)
        """
        workflow_steps = self.define_workflow_steps()

        status = {
            'total_steps': len(workflow_steps),
            'completed_steps': 0,
            'pending_steps': 0,
            'step_details': []
        }

        for step in workflow_steps:
            step_name = step.name
            output_exists = step.check_func()

            sections = STAGE_CONFIG_SECTIONS.get(step_name, [])
            if sections:
                current_hash = compute_config_hash(self._config, sections)
                marker_valid = is_stage_current(
                    self.project_dir, step_name, current_hash
                )
            else:
                marker_valid = True
                current_hash = ""

            config_stale = output_exists and not marker_valid
            is_complete = output_exists and marker_valid

            if is_complete:
                status['completed_steps'] += 1
            else:
                status['pending_steps'] += 1

            status['step_details'].append({
                'name': step_name,
                'cli_name': step.cli_name,
                'description': step.description,
                'complete': is_complete,
                'marker_valid': marker_valid,
                'config_stale': config_stale,
            })

        return status
