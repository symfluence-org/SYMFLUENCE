# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE Core System Module.

Provides the main SYMFLUENCE class that serves as the primary entry point
for hydrological modeling workflows. This module coordinates all manager
components and orchestrates the complete modeling pipeline from domain
definition through model calibration and analysis.

Example:
    >>> from symfluence import SYMFLUENCE
    >>> s = SYMFLUENCE("config.yaml")
    >>> s.run_workflow()
"""
from __future__ import annotations

try:
    from symfluence.symfluence_version import __version__
except ImportError:
    __version__ = "0+unknown"


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import SYMFLUENCEError
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.path_resolver import find_basin_shapefile, find_river_basins_shapefile
from symfluence.core.provenance import capture_provenance, record_input
from symfluence.core.provenance import finalize as finalize_provenance
from symfluence.project.logging_manager import LoggingManager
from symfluence.project.manager_factory import LazyManagerDict

# Import core components
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator

# ---------------------------------------------------------------------------
# Diagnostic spec — declarative descriptor for workflow step diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DiagnosticSpec:
    """Declarative descriptor for a single workflow diagnostic.

    Each spec describes *what* to load and *which* plotter method to call,
    replacing the procedural closures that previously lived inside
    ``_get_step_diagnostic_mapping``.
    """

    plotter_method: str
    """Name of the method on ``WorkflowDiagnosticPlotter``."""

    loader: Callable[['SymfluenceConfig', Path], Optional[Dict[str, Any]]]
    """Callable(config, project_dir) → kwargs dict for the plotter, or None to skip."""


def _domain_ctx(config: SymfluenceConfig) -> tuple:
    """Extract (domain_name, definition_method, experiment_id) for shapefile lookup."""
    domain = getattr(config, 'domain', None)
    return (
        getattr(domain, 'name', None),
        getattr(domain, 'definition_method', 'lumped'),
        getattr(domain, 'experiment_id', 'run_1'),
    )


def _load_domain(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    import geopandas as gpd
    name, _method, _exp = _domain_ctx(config)
    basin_path = find_river_basins_shapefile(project_dir / "shapefiles", name)
    if basin_path is None:
        return None
    dem_path = resolve_data_subdir(project_dir, 'attributes') / "dem" / "dem.tif"
    return dict(
        basin_gdf=gpd.read_file(basin_path),
        dem_path=dem_path if dem_path.exists() else None,
    )


def _load_discretization(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    import geopandas as gpd
    name, method, exp = _domain_ctx(config)
    hru_path = find_basin_shapefile(project_dir / "shapefiles", name, method, exp)
    if hru_path is None:
        return None
    return dict(
        hru_gdf=gpd.read_file(hru_path),
        method=getattr(config.discretization, 'method', 'unknown'),
    )


def _load_observations(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    import pandas as pd
    obs_path = (resolve_data_subdir(project_dir, 'observations')
                / "streamflow" / "preprocessed" / "streamflow_obs.csv")
    if not obs_path.exists():
        return None
    return dict(
        obs_df=pd.read_csv(obs_path, parse_dates=['datetime'], index_col='datetime'),
        obs_type='streamflow',
    )


def _load_forcing_raw(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    forcing_dir = resolve_data_subdir(project_dir, 'forcing') / "raw_data"
    if not forcing_dir.exists():
        return None
    nc_files = list(forcing_dir.glob("*.nc"))
    if not nc_files:
        return None
    name, _method, _exp = _domain_ctx(config)
    domain_shp = find_river_basins_shapefile(project_dir / "shapefiles", name)
    return dict(
        forcing_nc=nc_files[0],
        domain_shp=domain_shp,
    )


def _load_forcing_remapped(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    raw_dir = resolve_data_subdir(project_dir, 'forcing') / "raw_data"
    remapped_dir = resolve_data_subdir(project_dir, 'forcing') / "basin_averaged_data"
    if not raw_dir.exists() or not remapped_dir.exists():
        return None
    raw_files = list(raw_dir.glob("*.nc"))
    remapped_files = list(remapped_dir.glob("*.nc"))
    if not raw_files or not remapped_files:
        return None
    name, method, exp = _domain_ctx(config)
    hru_shp = find_basin_shapefile(project_dir / "shapefiles", name, method, exp)
    return dict(
        raw_nc=raw_files[0],
        remapped_nc=remapped_files[0],
        hru_shp=hru_shp,
    )


def _load_model_preprocessing(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    model_name = getattr(config.model, 'name', 'SUMMA')
    input_dir = project_dir / "simulations" / model_name.lower() / "run_settings"
    if not input_dir.exists():
        return None
    return dict(input_dir=input_dir, model_name=model_name)


def _load_model_output(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    model_name = getattr(config.model, 'name', 'SUMMA')
    output_dir = project_dir / "simulations" / model_name.lower() / "output"
    if not output_dir.exists():
        return None
    nc_files = list(output_dir.glob("*.nc"))
    if not nc_files:
        return None
    return dict(output_nc=nc_files[0], model_name=model_name)


def _load_attributes(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    attr_dir = resolve_data_subdir(project_dir, 'attributes')
    dem_path = attr_dir / "dem" / "dem.tif"
    soil_path = attr_dir / "soilclass" / "soilclass.tif"
    land_path = attr_dir / "landclass" / "landclass.tif"
    if not any(p.exists() for p in [dem_path, soil_path, land_path]):
        return None
    return dict(
        dem_path=dem_path if dem_path.exists() else None,
        soil_path=soil_path if soil_path.exists() else None,
        land_path=land_path if land_path.exists() else None,
    )


def _load_calibration(config: SymfluenceConfig, project_dir: Path) -> Optional[Dict[str, Any]]:
    import pandas as pd
    model_name = getattr(config.model, 'name', 'SUMMA')
    history_file = (project_dir / "simulations" / model_name.lower()
                    / "calibration" / "calibration_history.csv")
    if not history_file.exists():
        return None
    return dict(
        history=pd.read_csv(history_file).to_dict('records'),
        model_name=model_name,
    )


# Step name → diagnostic spec (declarative table)
_DIAGNOSTIC_SPECS: Dict[str, _DiagnosticSpec] = {
    'define_domain':                _DiagnosticSpec('plot_domain_definition_diagnostic',   _load_domain),
    'discretize_domain':            _DiagnosticSpec('plot_discretization_diagnostic',      _load_discretization),
    'process_observed_data':        _DiagnosticSpec('plot_observations_diagnostic',        _load_observations),
    'acquire_forcings':             _DiagnosticSpec('plot_forcing_raw_diagnostic',         _load_forcing_raw),
    'model_agnostic_preprocessing': _DiagnosticSpec('plot_forcing_remapped_diagnostic',    _load_forcing_remapped),
    'model_specific_preprocessing': _DiagnosticSpec('plot_model_preprocessing_diagnostic', _load_model_preprocessing),
    'run_model':                    _DiagnosticSpec('plot_model_output_diagnostic',        _load_model_output),
    'acquire_attributes':           _DiagnosticSpec('plot_attributes_diagnostic',          _load_attributes),
    'calibrate_model':              _DiagnosticSpec('plot_calibration_diagnostic',         _load_calibration),
}


class SYMFLUENCE:
    """
    Enhanced SYMFLUENCE main class with comprehensive CLI support.

    This class serves as the central coordinator for all SYMFLUENCE operations,
    with enhanced CLI capabilities including individual step execution,
    pour point setup, SLURM job submission, and comprehensive workflow management.
    """

    def __init__(self, config_input: Union[Path, str, SymfluenceConfig], config_overrides: Dict[str, Any] = None, debug_mode: bool = False, visualize: bool = False, diagnostic: bool = False, quiet_mode: bool = False):
        """
        Initialize the SYMFLUENCE system with configuration and CLI options.

        Args:
            config_input: Path to the configuration file or a SymfluenceConfig instance
            config_overrides: Dictionary of configuration overrides from CLI
            debug_mode: Whether to enable debug mode
            visualize: Whether to enable visualization
            diagnostic: Whether to enable diagnostic plots for workflow validation
            quiet_mode: Suppress console INFO logging (console at WARNING;
                the file log is unaffected). Ignored when debug_mode is set.
        """
        self.debug_mode = debug_mode
        self.quiet_mode = quiet_mode
        self.visualize = visualize
        self.diagnostic = diagnostic
        self.config_overrides = config_overrides or {}

        # Handle different config input types
        if isinstance(config_input, SymfluenceConfig):
            self.typed_config = config_input
            # If overrides provided, we merge them into a flat dict and re-create the model
            if self.config_overrides:
                flat_config = self.typed_config.to_dict(flatten=True)
                flat_config.update(self.config_overrides)
                self.typed_config = SymfluenceConfig(**flat_config)
            self.config_path = getattr(config_input, '_source_file', None)
        else:
            self.config_path = Path(config_input)
            self.typed_config = self._load_typed_config()

        self.config = self.typed_config.to_dict(flatten=True)  # Backward compatibility

        # Ensure log level consistency with debug mode
        if self.debug_mode:
            self.config['LOG_LEVEL'] = 'DEBUG'

        # Initialize logging
        self.logging_manager = LoggingManager(self.config, debug_mode=debug_mode,
                                              quiet_mode=quiet_mode)
        self.logger = self.logging_manager.logger

        self.logger.info("SYMFLUENCE initialized")
        if self.config_path:
            self.logger.info(f"Config path: {self.config_path}")
        if self.config_overrides:
            self.logger.info(f"Configuration overrides applied: {list(self.config_overrides.keys())}")


        # Capture provenance metadata (can be disabled via record_provenance: false)
        if getattr(self.typed_config.system, 'record_provenance', True):
            self.provenance = capture_provenance(
                experiment_id=getattr(self.typed_config.domain, 'experiment_id', 'unknown') or 'unknown',
                domain_name=getattr(self.typed_config.domain, 'name', 'unknown') or 'unknown',
                config_path=str(self.config_path) if self.config_path else None,
            )
        else:
            self.logger.info("Provenance capture disabled via configuration")
            self.provenance = None

        # Initialize managers (lazy loaded)
        self.managers = LazyManagerDict(self.typed_config, self.logger, self.visualize, self.diagnostic)

        # Initialize workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            self.managers, self.typed_config, self.logger, self.logging_manager,
            provenance=self.provenance,
        )


    def _load_typed_config(self) -> SymfluenceConfig:
        """
        Load configuration using new hierarchical SymfluenceConfig.

        Returns:
            SymfluenceConfig: Fully validated hierarchical configuration
        """
        try:
            return SymfluenceConfig.from_file(
                self.config_path,
                overrides=self.config_overrides,
                use_env=True,
                validate=True
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}") from None

    def run_workflow(self, force_run: Optional[bool] = None) -> None:
        """Execute the complete SYMFLUENCE workflow (CLI wrapper)."""
        start = datetime.now()
        errors: List[Any] = []

        try:
            self.logger.info("Starting complete SYMFLUENCE workflow execution")

            # Determine force-run behavior (CLI override beats config)
            if force_run is None:
                force_run = getattr(self.typed_config.system, "force_run_all_steps", False)

            # Run the workflow
            self.workflow_orchestrator.run_workflow(force_run=bool(force_run))

            # Outcome of *this* run (STOP_ON_ERROR=False lets a run finish
            # with failed steps; those must not be reported as success).
            failed = self.get_failed_steps()
            ran = [s for s in getattr(self.workflow_orchestrator, 'last_step_results', [])
                   if s.get('status') == 'completed']
            if not failed:
                status = "completed"
            elif ran:
                status = "partial"
            else:
                status = "failed"
            errors.extend(
                {"where": s.get('cli_name'), "error": s.get('error', '')}
                for s in getattr(self.workflow_orchestrator, 'last_step_results', [])
                if s.get('status') == 'failed'
            )

            if failed:
                self.logger.error(
                    "SYMFLUENCE workflow finished with failed step(s): %s", ", ".join(failed)
                )
            else:
                self.logger.info("Complete SYMFLUENCE workflow execution completed")

        except (SYMFLUENCEError, FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
            status = "failed"
            errors.append({"where": "run_workflow", "error": str(e)})
            self.logger.error(f"Workflow execution failed: {e}")
            # re-raise after summary so the CI can fail meaningfully if needed
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            status = "failed"
            errors.append({"where": "run_workflow", "error": str(e)})
            self.logger.exception(f"Unexpected workflow execution failure: {e}")
            raise
        finally:
            end = datetime.now()
            elapsed_s = (end - start).total_seconds()
            # Per-step results (name/cli_name/description/status/duration_s)
            # tracked by the orchestrator during execution — accurate even
            # when a failing step aborted the run.
            step_results = list(
                getattr(self.workflow_orchestrator, 'last_step_results', [])
            )
            self.logging_manager.create_run_summary(
                steps=step_results,
                execution_time=elapsed_s,
                status=status,
                errors=errors,
            )
            # Write provenance manifest
            self._record_input_provenance()
            finalize_provenance(self.provenance, status,
                                errors=[e.get("error", str(e)) for e in errors] if errors else None)
            if self.provenance is not None:
                manifest = self.provenance.write(self.logging_manager.log_dir)
                self.logger.info(f"Run manifest written to: {manifest}")


    def _record_input_provenance(self) -> None:
        """Hash the source DEM into the run manifest.

        The DEM is the input that decides delineation geometry, so a change in
        it shifts discretization counts in ways that otherwise look identical
        to a code or platform difference. Best-effort: a missing DEM (a run
        that never delineated) is simply not recorded.
        """
        if self.provenance is None:
            return
        try:
            project_dir = Path(self.typed_config.paths.root_path) / f"domain_{self.typed_config.domain.name}"
            dem_dir = project_dir / "data" / "attributes" / "elevation" / "dem"
            for dem in sorted(dem_dir.glob("*.tif")):
                record_input(self.provenance, "dem", dem)
                break
        except (AttributeError, OSError, TypeError):
            pass

    def run_individual_steps(self, step_names: List[str]) -> None:
        """
        Execute specific workflow steps by name.

        Allows selective execution of individual workflow steps rather than
        running the complete pipeline. Useful for debugging, testing, or
        re-running specific portions of the workflow.

        Args:
            step_names: List of step names to execute (e.g., ['setup_project', 'calibrate_model'])
        """
        start = datetime.now()
        errors: List[Any] = []

        try:
            continue_on_error = self.config_overrides.get("continue_on_error", False)

            # Execute individual steps via orchestrator
            self.workflow_orchestrator.run_individual_steps(step_names, continue_on_error)

        finally:
            end = datetime.now()
            elapsed_s = (end - start).total_seconds()
            # Per-step results tracked by the orchestrator (accurate even
            # when a failing step raised out of the loop).
            step_results = list(
                getattr(self.workflow_orchestrator, 'last_step_results', [])
            )
            failed = [s for s in step_results if s.get('status') == 'failed']
            succeeded = [s for s in step_results if s.get('status') == 'completed']
            errors = [{"step": s.get('cli_name'), "error": s.get('error', '')}
                      for s in failed]
            if not failed:
                status = "completed"
            elif succeeded:
                status = "partial"
            else:
                status = "failed"
            self.logging_manager.create_run_summary(
                steps=step_results,
                execution_time=elapsed_s,
                status=status,
                errors=errors,
            )
            # Write provenance manifest
            self._record_input_provenance()
            finalize_provenance(self.provenance, status,
                                errors=[e.get("error", str(e)) for e in errors] if errors else None)
            if self.provenance is not None:
                manifest = self.provenance.write(self.logging_manager.log_dir)
                self.logger.info(f"Run manifest written to: {manifest}")

    def get_failed_steps(self) -> List[str]:
        """CLI names of the steps that failed in the most recent workflow/steps run."""
        orchestrator = getattr(self, 'workflow_orchestrator', None)
        if orchestrator is None or not hasattr(orchestrator, 'failed_step_names'):
            return []
        return list(orchestrator.failed_step_names())

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Return workflow completion status from the orchestrator.

        Returns:
            Dict[str, Any]: Workflow status payload with step_details and counts.
        """
        return self.workflow_orchestrator.get_workflow_status()

    def _run_diagnostic(self, step_name: str, spec: _DiagnosticSpec) -> Optional[str]:
        """Execute a single diagnostic spec against the current project."""
        project_dir = Path(self.typed_config.paths.root_path) / f"domain_{self.typed_config.domain.name}"
        kwargs = spec.loader(self.typed_config, project_dir)
        if kwargs is None:
            return None
        reporting_manager = self.managers.get('reporting')
        if not reporting_manager:
            return None
        plotter_fn = getattr(reporting_manager.workflow_diagnostic_plotter, spec.plotter_method)
        return plotter_fn(**kwargs)

    def run_diagnostics_for_step(self, step_name: str) -> List[str]:
        """Run diagnostic plots for a specific workflow step on existing outputs.

        Args:
            step_name: Name of the workflow step to diagnose

        Returns:
            List of paths to generated diagnostic plots
        """
        self.logger.info(f"Running diagnostics for step: {step_name}")
        spec = _DIAGNOSTIC_SPECS.get(step_name)
        if spec is None:
            self.logger.warning(f"No diagnostic available for step: {step_name}")
            return []
        try:
            result = self._run_diagnostic(step_name, spec)
            return [result] if result else []
        except (SYMFLUENCEError, FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
            self.logger.error(f"Failed to generate diagnostic for {step_name}: {e}")
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.exception(f"Unexpected diagnostic failure for {step_name}: {e}")
        return []

    def run_all_diagnostics(self) -> List[str]:
        """Run all available diagnostic plots on existing workflow outputs.

        Returns:
            List of paths to generated diagnostic plots
        """
        self.logger.info("Running all available diagnostics...")
        results: List[str] = []
        for step_name, spec in _DIAGNOSTIC_SPECS.items():
            self.logger.debug(f"Checking diagnostics for: {step_name}")
            try:
                result = self._run_diagnostic(step_name, spec)
                if result:
                    results.append(result)
                    self.logger.info(f"Generated diagnostic for {step_name}: {result}")
            except (SYMFLUENCEError, FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
                self.logger.debug(f"Skipping {step_name} diagnostic: {e}")
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.exception(f"Unexpected diagnostic failure for {step_name}: {e}")
        self.logger.info(f"Generated {len(results)} diagnostic plot(s)")
        return results
