# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Runner Module

Refactored to use the model execution framework:
- SpatialOrchestrator: Combined execution and spatial orchestration
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from symfluence.core.exceptions import FileOperationError, ModelExecutionError, symfluence_error_handler
from symfluence.core.logging_utils import log_once
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.execution import SpatialOrchestrator
from symfluence.core.modeling.mixins import OutputConverterMixin, SpatialModeDetectionMixin
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

from ..mizuroute.mixins import MizuRouteConfigMixin
from .elevation_band_manager import read_band_parameters
from .subcatchment_processor import SubcatchmentProcessor


@R.runners.add('FUSE', runner_method='run_fuse')
class FUSERunner(BaseModelRunner, SpatialOrchestrator, OutputConverterMixin, MizuRouteConfigMixin, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Runner class for the FUSE (Framework for Understanding Structural Errors) model.
    Handles model execution, output processing, and file management.

    Now uses the model execution framework for:

    - Subprocess execution (via BaseModelRunner execution mixins)
    - Spatial mode handling and routing (via SpatialOrchestrator)
    - Output format conversion (via OutputConverterMixin)

    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    MODEL_NAME = "FUSE"
    _fuse_file_id: Optional[str] = None

    def __init__(self, config, logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the FUSE runner.

        Sets up FUSE execution environment including spatial mode detection,
        routing requirements check, and lazy initialization of subcatchment
        processor for distributed runs.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                FUSE model settings, paths, and execution parameters.
            logger: Logger instance for status messages and debugging output.
            reporting_manager: Optional reporting manager for experiment tracking.

        Note:
            Uses model execution framework mixins for subprocess execution,
            spatial orchestration, output conversion, and mizuRoute integration.
        """
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # FUSE-specific initialization - determine spatial mode using mixin
        self.spatial_mode = self.detect_spatial_mode('FUSE')

        self.needs_routing = self._check_routing_requirements()
        self._subcatchment_processor = None

    @property
    def subcatchment_processor(self) -> SubcatchmentProcessor:
        """Lazy-loaded subcatchment processor for distributed runs."""
        if self._subcatchment_processor is None:
            self._subcatchment_processor = SubcatchmentProcessor(
                project_dir=self.project_dir,
                domain_name=self.domain_name,
                experiment_id=self.experiment_id,
                config_dict=self.config_dict,
                setup_dir=self.setup_dir,
                output_path=self.output_path,
                fuse_exe=self.fuse_exe,
                logger=self.logger,
                config=self.config  # Pass typed config
            )
        assert self._subcatchment_processor is not None
        return self._subcatchment_processor

    def _get_fuse_file_id(self) -> str:
        """Return a short file ID for FUSE outputs/settings.

        FUSE Fortran uses a CHARACTER(LEN=6) buffer for FMODEL_ID,
        so the ID must be kept to 6 chars max to avoid truncation.
        """
        if hasattr(self, '_fuse_file_id') and self._fuse_file_id is not None:
            return self._fuse_file_id
        fuse_id: str = self._get_config_value(
            lambda: self.config.model.fuse.file_id,
            default=self.experiment_id
        )
        if len(fuse_id) > 6:
            import hashlib
            fuse_id = hashlib.md5(fuse_id.encode(), usedforsecurity=False).hexdigest()[:6]
        self._fuse_file_id = fuse_id
        return fuse_id

    def _setup_model_specific_paths(self) -> None:
        """Set up FUSE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "FUSE"
        self.forcing_fuse_path = self.project_forcing_dir / 'FUSE_input'

        # FUSE executable path (installation dir + exe name)
        self.fuse_exe = self.get_model_executable(
            install_path_key='FUSE_INSTALL_PATH',
            default_install_subpath='installs/fuse/bin',
            exe_name_key='FUSE_EXE',
            default_exe_name='fuse.exe',
            must_exist=True
        )
        self.output_path = self.get_config_path('EXPERIMENT_OUTPUT_FUSE', f"simulations/{self.experiment_id}/FUSE")

        # FUSE-specific: result_dir is an alias for output_dir (backward compatibility)
        self.output_dir = self.get_experiment_output_dir()
        self.setup_path_aliases({'result_dir': 'output_dir'})

    def _build_fuse_env(self) -> Dict[str, str]:
        """Build subprocess environment with HDF5/NetCDF library paths for FUSE."""
        env = os.environ.copy()

        lib_paths = []

        exe_lib = self.fuse_exe.parent.parent / 'lib'
        if exe_lib.is_dir():
            lib_paths.append(str(exe_lib))

        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_lib = Path(conda_prefix) / 'lib'
            if conda_lib.is_dir():
                lib_paths.append(str(conda_lib))

        for prefix in ['/opt/homebrew', '/usr/local']:
            for pkg in ['hdf5', 'netcdf', 'netcdf-fortran']:
                pkg_lib = Path(prefix) / 'opt' / pkg / 'lib'
                if pkg_lib.is_dir():
                    lib_paths.append(str(pkg_lib))
            general_lib = Path(prefix) / 'lib'
            if general_lib.is_dir():
                lib_paths.append(str(general_lib))

        if lib_paths:
            lib_path_str = os.pathsep.join(lib_paths)
            for var in ['DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
                existing = env.get(var, '')
                env[var] = f"{lib_path_str}{os.pathsep}{existing}" if existing else lib_path_str
            self.logger.debug(f"FUSE library paths: {lib_path_str}")

        return env

    def _get_output_dir(self) -> Path:
        """FUSE uses custom result_dir naming."""
        return self.get_experiment_output_dir()

    def _fuse_output_candidates(self, output_dir: Path, fuse_id: str) -> List[Path]:
        """Return FUSE output candidates ordered by preferred usage."""
        return [
            output_dir / f"{self.domain_name}_{fuse_id}_runs_def.nc",
            output_dir / f"{self.domain_name}_{fuse_id}_runs_best.nc",
            output_dir / f"{self.domain_name}_{fuse_id}_runs_pre.nc",
        ]

    @property
    def mizu_routing_var(self) -> str:
        """FUSE-specific override: default to q_routed instead of SUMMA's averageRoutedRunoff."""
        var = self._get_config_value(
            lambda: self.config.model.mizuroute.routing_var,
            dict_key='SETTINGS_MIZU_ROUTING_VAR',
        )
        if var in ('default', None, '', 'averageRoutedRunoff'):
            return 'q_routed'
        return var

    def _ensure_routing_variable(self, file_path: Path) -> None:
        """Rename FUSE runoff variable to match what the mizuRoute control file expects.

        FUSE run modes produce different variable names (q_routed vs q_instnt).
        The control writer uses the FUSE model default (q_routed) unless overridden,
        so the NetCDF must match.
        """
        target_var = self.mizu_routing_var
        fuse_runoff_candidates = ['q_routed', 'q_instnt', 'total_discharge', 'runoff']

        with xr.open_dataset(file_path) as ds:
            actual_var = next((v for v in fuse_runoff_candidates if v in ds.data_vars), None)
            if actual_var is None or actual_var == target_var:
                return
            ds_loaded = ds.load()

        ds_renamed = ds_loaded.rename({actual_var: target_var})
        ds_renamed.to_netcdf(file_path)
        self.logger.debug(f"Renamed FUSE variable {actual_var} → {target_var} for mizuRoute")

    def _convert_routing_units_for_mizuroute(self, dataset: xr.Dataset) -> bool:
        """Convert routed runoff units from mm/day to m/s when required."""
        routing_var = self.mizu_routing_var
        if self.mizu_routing_units != 'm/s' or routing_var not in dataset:
            return False

        fuse_timestep = int(self._get_config_value(
            lambda: self.config.model.fuse.output_timestep_seconds,
            dict_key='FUSE_OUTPUT_TIMESTEP_SECONDS',
        ))
        conversion_factor = 1.0 / (1000.0 * fuse_timestep)
        dataset[routing_var] = dataset[routing_var] * conversion_factor
        dataset[routing_var].attrs['units'] = 'm/s'
        self.logger.debug(f"Converted {routing_var}: mm/day → m/s (factor={conversion_factor:.2e})")
        return True

    def _convert_fuse_distributed_to_mizuroute_format(self):
        """
        Convert FUSE spatial dimensions to mizuRoute format.

        Uses OutputConverterMixin for the core conversion:
        - Squeezes singleton latitude dimension
        - Renames longitude → gru
        - Adds gruId variable
        - Filters out coastal GRUs and sets gruId to match topology hruId
        """
        import pandas as pd

        experiment_id = self.experiment_id
        fuse_id = self._get_fuse_file_id()
        domain = self.domain_name

        fuse_out_dir = self.get_simulation_dir('FUSE', experiment_id)

        # Find FUSE output file
        target = next((path for path in self._fuse_output_candidates(fuse_out_dir, fuse_id) if path.exists()), None)

        if target is None:
            raise FileOperationError(
                f"FUSE output not found. Tried: {[str(path) for path in self._fuse_output_candidates(fuse_out_dir, fuse_id)]}"
            )

        self.logger.debug(f"Converting FUSE spatial dimensions: {target}")

        # Use generic mixin method with FUSE-specific parameters.
        # param_set is a singleton dimension from FUSE run_pre mode that
        # mizuRoute cannot handle (expects 2D: time × gru).
        self.convert_to_mizuroute_format(
            input_path=target,
            squeeze_dims=['latitude', 'param_set'],
            rename_dims={'longitude': 'gru'},
            add_id_var='gruId',
            id_source_dim='gru',
            create_backup=True
        )

        # FUSE run modes produce different variable names (q_instnt in run_pre,
        # q_routed in run_def). Rename to match what the control file expects.
        self._ensure_routing_variable(target)

        # Filter coastal GRUs using mapping file if available
        mapping_file = self.project_dir / 'settings' / 'mizuRoute' / 'fuse_to_routing_mapping.csv'
        if mapping_file.exists():
            ds = xr.open_dataset(target)
            ds = ds.load()
            ds.close()

            mapping = pd.read_csv(mapping_file)
            non_coastal = mapping[~mapping['is_coastal']]
            fuse_indices = non_coastal['fuse_gru_idx'].values
            gru_ids = non_coastal['gru_to_seg'].values.astype(np.int32)

            n_total = ds.sizes.get('gru', 0)
            n_keep = len(non_coastal)
            self.logger.debug(f"Filtering coastal GRUs: {n_total} total → {n_keep} non-coastal")

            ds = ds.isel(gru=fuse_indices)
            ds = ds.assign_coords(gru=np.arange(n_keep))
            ds['gruId'] = xr.DataArray(gru_ids, dims=['gru'],
                                       attrs={'long_name': 'gru identifier'})

            self._convert_routing_units_for_mizuroute(ds)

            ds.to_netcdf(target)
            self.logger.debug(f"Saved filtered output with {n_keep} GRUs to {target}")
        else:
            self.logger.warning(f"Mapping file not found at {mapping_file}, skipping coastal GRU filtering")
            # Still need to convert units from mm/day to m/s for mizuRoute
            if self.mizu_routing_units == 'm/s':
                ds = xr.open_dataset(target)
                ds = ds.load()
                ds.close()
                if self._convert_routing_units_for_mizuroute(ds):
                    ds.to_netcdf(target)

        # Ensure _runs_def.nc exists if we processed a different file
        def_file = fuse_out_dir / f"{domain}_{fuse_id}_runs_def.nc"
        if target != def_file and not def_file.exists():
            shutil.copy2(target, def_file)
            self.logger.debug(f"Created runs_def file: {def_file}")

    def run_fuse(self) -> Optional[Path]:
        """
        Run FUSE model with distributed support.

        Returns:
            Path to output directory on success, None on failure

        Raises:
            ModelExecutionError: If model execution fails
        """
        self.logger.debug(f"Starting FUSE model run in {self.spatial_mode} mode")

        with symfluence_error_handler(
            "FUSE model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Run FUSE simulations
            success = self._execute_fuse_workflow()

            if not success:
                raise ModelExecutionError(
                    "FUSE simulation failed. Check log for convergence errors "
                    "(e.g. 'STOP failed to converge in implicit_solve')."
                )

            # Handle routing if needed
            if self.needs_routing:
                self._convert_fuse_distributed_to_mizuroute_format()
                success = self._run_distributed_routing()

            if success:
                self._process_outputs()
                self.logger.debug("FUSE run completed successfully")
                return self.output_path
            else:
                raise ModelExecutionError("FUSE routing (mizuRoute) failed after FUSE simulation.")

    def _check_routing_requirements(self) -> bool:
        """
        Check if distributed routing is needed for the current configuration.

        Determines whether mizuRoute should be executed after FUSE based on
        the spatial mode and routing integration settings. Routing is needed
        for distributed/semi-distributed modes or when routing delineation
        uses river_network in lumped mode.

        Returns:
            bool: True if mizuRoute routing should be executed, False otherwise.
        """
        routing_integration = self._get_config_value(
            lambda: self.config.model.fuse.routing_integration,
            dict_key='FUSE_ROUTING_INTEGRATION',
        )

        # When FUSE_ROUTING_INTEGRATION is 'default', infer from ROUTING_MODEL
        if routing_integration == 'default':
            routing_model = self._get_config_value(
                lambda: self.config.model.routing_model,
                default='none'
            )
            if routing_model and routing_model.lower() in ('mizuroute', 'mizu_route', 'mizu'):
                routing_integration = 'mizuRoute'

        if routing_integration == 'mizuRoute':
            if self.spatial_mode in (SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED):
                return True
            routing_delineation = self._get_config_value(
                lambda: self.config.domain.delineation.routing if self.config.domain and self.config.domain.delineation else None,
                default=None
            )
            if self.spatial_mode == SpatialMode.LUMPED and routing_delineation == 'river_network':
                return True

        return False

    def _execute_fuse_workflow(self) -> bool:
        """
        Execute the main FUSE workflow based on spatial mode.

        Routes to either lumped or distributed execution workflow based on
        the configured spatial mode. Lumped mode runs a single catchment
        simulation while distributed mode processes the full multi-HRU dataset.

        Returns:
            bool: True if FUSE execution completed successfully, False otherwise.
        """
        # Pre-flight: verify forcing data exists before running FUSE
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        if not forcing_file.exists():
            self.logger.error(
                f"FUSE forcing file not found: {forcing_file}. "
                f"The FUSE preprocessing step may not have completed. "
                f"Re-run preprocessing to generate forcing data."
            )
            return False

        if self.spatial_mode == SpatialMode.LUMPED:
            # Original lumped workflow
            return self._run_lumped_fuse()
        else:
            # Distributed workflow
            return self._run_distributed_fuse()

    def _run_distributed_fuse(self) -> bool:
        """Run FUSE in distributed mode - always process the full dataset at once"""
        self.logger.debug("Running distributed FUSE workflow with full dataset")

        try:
            # Run FUSE once with the complete distributed forcing file
            return self._run_multidimensional_fuse()

        except (subprocess.CalledProcessError, OSError, FileNotFoundError) as e:
            self.logger.error(f"Error in distributed FUSE execution: {str(e)}")
            return False

    def _run_multidimensional_fuse(self) -> bool:
        """Run FUSE once with the full distributed forcing file"""

        try:
            self.logger.debug("Running FUSE with complete distributed forcing dataset")

            # Clear stale output files to prevent validation from matching
            # previous run data if the current run fails silently
            fuse_id = self._get_fuse_file_id()
            for suffix in ['_runs_def.nc', '_runs_best.nc']:
                stale = self.output_path / f"{self.domain_name}_{fuse_id}{suffix}"
                if stale.exists():
                    stale.unlink()
                    self.logger.debug(f"Cleared stale output: {stale.name}")

            # Run FUSE with the distributed forcing file (all HRUs at once)
            success = self._execute_fuse_distributed()

            if success:
                # Validate output before proceeding (catches silent convergence failures)
                if not self._validate_fuse_output():
                    self.logger.error("FUSE execution appeared to succeed but output validation failed")
                    return False
                self.logger.debug("Distributed FUSE run completed successfully")
                return True
            else:
                self.logger.error("Distributed FUSE run failed")
                return False

        except (subprocess.CalledProcessError, OSError) as e:
            self.logger.error(f"Error in multidimensional FUSE execution: {str(e)}")
            return False

    def _execute_fuse_distributed(self) -> bool:
        """Execute FUSE with the complete distributed forcing file.

        Two-phase flow: ``run_def`` runs only as the initial default run
        (when no para_def.nc exists yet, generating it); ``run_pre`` is the
        production mode for every subsequent simulation.
        """

        try:
            fuse_id = self._get_fuse_file_id()
            para_def_path = self.output_path / f"{self.domain_name}_{fuse_id}_para_def.nc"

            # Use the main file manager (points to distributed forcing file)
            control_file = self.setup_dir / 'fm_catch.txt'

            # Determine run mode: prefer run_pre to avoid NC_UNLIMITED conflict
            if para_def_path.exists():
                mode = 'run_pre'
                command = [
                    str(self.fuse_exe),
                    str(control_file),
                    self.domain_name,
                    mode,
                    str(para_def_path.name),
                    '1'
                ]
            else:
                # Fall back to run_def (creates para_def.nc as side effect)
                mode = 'run_def'
                command = [
                    str(self.fuse_exe),
                    str(control_file),
                    self.domain_name,
                    mode
                ]

            # Create log file
            log_file = self.output_path / 'fuse_distributed_run.log'

            self.logger.debug(f"Executing distributed FUSE ({mode}): {' '.join(command)}")

            env = self._build_fuse_env()

            timeout = self._get_config_value(lambda: self.config.model.fuse.timeout, dict_key='FUSE_TIMEOUT')

            with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
                result = run_subprocess(
                    command,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=str(self.setup_dir),
                    env=env,
                    timeout=timeout,
                )

            if result.returncode == 0:
                # Fortran STOP returns exit code 0, so scan log for failures
                if log_file.exists():
                    try:
                        log_content = log_file.read_text(encoding='utf-8', errors='replace')
                        if 'NetCDF:' in log_content:
                            nc_first = next(
                                (l.strip() for l in log_content.splitlines() if 'NetCDF:' in l), ''
                            )
                            log_once(
                                self.logger, logging.ERROR,
                                f"fuse-distributed-netcdf:{nc_first}",
                                f"FUSE NetCDF error (exit code 0): {nc_first}; "
                                f"full output: {log_file}",
                            )
                            # If run_def failed with NC error, retry with run_pre
                            if mode == 'run_def' and para_def_path.exists():
                                self.logger.info("Retrying distributed FUSE with run_pre mode")
                                return self._execute_fuse_distributed()
                            return False
                        if 'STOP' in log_content:
                            stop_first = next(
                                (line.strip() for line in log_content.splitlines() if 'STOP' in line), ''
                            )
                            log_once(
                                self.logger, logging.ERROR,
                                f"fuse-distributed-stop:{stop_first}",
                                f"FUSE Fortran STOP (exit code 0): {stop_first}; "
                                f"full output: {log_file}",
                            )
                            return False
                    except OSError:
                        pass  # If we can't read the log, proceed with output validation
                self.logger.debug(f"Distributed FUSE execution completed successfully ({mode})")
                return True
            else:
                self.logger.error(f"FUSE failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired as e:
            # subprocess.run has already killed the child. Without this the
            # workflow blocks forever on a wedged binary: observed on Windows
            # as fuse.exe sitting at 0% CPU for 37h after writing only the
            # output NetCDF header, holding its run slot the whole time.
            self.logger.error(
                f"FUSE exceeded its {e.timeout:.0f}s timeout and was killed "
                f"(log: {log_file}). Raise FUSE_TIMEOUT if this domain is "
                f"legitimately slower than that."
            )
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed: {str(e)}")
            return False
        except (OSError, FileNotFoundError) as e:
            self.logger.error(f"Error executing distributed FUSE: {str(e)}")
            return False


    def _validate_fuse_output(self) -> bool:
        """
        Validate that FUSE produced non-empty output.

        FUSE's Fortran STOP statement returns exit code 0, so subprocess
        won't detect convergence failures. This method checks the output
        NetCDF to ensure timesteps were actually written.

        Returns:
            bool: True if output is valid, False if empty or missing.
        """
        fuse_id = self._get_fuse_file_id()
        target = next(
            (path for path in self._fuse_output_candidates(self.output_path, fuse_id) if path.exists()),
            None
        )

        if target is None:
            self.logger.error("FUSE output file not found - model may have failed silently")
            return False

        try:
            with xr.open_dataset(target) as ds:
                time_size = ds.sizes.get('time', 0)
                if time_size == 0:
                    self.logger.error(
                        f"FUSE output has empty time dimension in {target.name}. "
                        "This typically indicates the implicit solver failed to converge. "
                        "Check fuse_zNumerix.txt: SOLUTION should be 1 (explicit Heun) "
                        "and convergence tolerances should be relaxed (e.g. 1e-4)."
                    )
                    return False
                self.logger.debug(f"FUSE output validated: {time_size} timesteps in {target.name}")
                return True
        except (OSError, ValueError) as e:
            self.logger.error(f"Failed to validate FUSE output {target}: {e}")
            return False

    def _create_subcatchment_settings(self, subcat_id: int, index: int) -> Path:
        """Create subcatchment-specific settings files. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.create_subcatchment_settings(subcat_id, index)

    def _execute_fuse_subcatchment(self, subcat_id: int, forcing_file: Path, settings_dir: Path) -> Optional[Path]:
        """Execute FUSE for a specific subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.execute_fuse_subcatchment(subcat_id, forcing_file, settings_dir)

    def _ensure_best_output_file(self):
        """Ensure the expected 'best' output file exists by copying from run output if needed.

        Checks for _runs_pre.nc (preferred, from run_pre mode) and _runs_def.nc
        (fallback, from run_def mode) as source files.
        """
        fuse_id = self._get_fuse_file_id()
        pre_file = self.output_path / f"{self.domain_name}_{fuse_id}_runs_pre.nc"
        def_file = self.output_path / f"{self.domain_name}_{fuse_id}_runs_def.nc"
        best_file = self.output_path / f"{self.domain_name}_{fuse_id}_runs_best.nc"

        # Prefer run_pre output (valid when run_def is broken)
        source_file = None
        if pre_file.exists() and pre_file.stat().st_size > 1024:
            source_file = pre_file
        elif def_file.exists() and def_file.stat().st_size > 1024:
            source_file = def_file

        # Copy if best_file is missing or too small (FUSE creates an empty shell
        # before crashing, so the file may exist but contain no time steps)
        best_needs_copy = not best_file.exists() or best_file.stat().st_size < 1024
        if source_file and best_needs_copy:
            self.logger.debug(f"Copying {source_file.name} to {best_file.name} for compatibility")
            shutil.copy2(source_file, best_file)

        return best_file if best_file.exists() else (source_file or def_file)

    def _extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """Extract forcing data for a specific subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.extract_subcatchment_forcing(subcat_id, index)

    def _combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """Combine outputs from all subcatchments. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.combine_subcatchment_outputs(outputs)

    def _load_subcatchment_info(self):
        """Load subcatchment information for distributed mode. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.load_subcatchment_info(self.catchment_name_col)

    def _run_individual_subcatchments(self, subcatchments) -> bool:
        """Run FUSE separately for each subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.run_individual_subcatchments(subcatchments)

    def _create_subcatchment_elevation_bands(self, subcat_id: int) -> Path:
        """Create elevation bands file for a specific subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.create_subcatchment_elevation_bands(subcat_id)

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed FUSE output.

        Uses SpatialOrchestrator._run_mizuroute() for unified routing integration.
        """
        self.logger.debug("Starting mizuRoute routing for distributed FUSE")

        # Update config for FUSE-mizuRoute integration
        self._setup_fuse_mizuroute_config()

        # Use orchestrator method (creates control file and runs mizuRoute)
        spatial_config = self.get_spatial_config('FUSE')
        result = self._run_mizuroute(spatial_config, model_name='fuse')

        return result is not None

    def _convert_fuse_to_mizuroute_format(self) -> bool:
        """
        Convert FUSE distributed output to the mizuRoute input format *in place*
        so it matches what the FUSE-specific mizu control file expects:
        - dims: (time, gru)
        - var:  <routing_var> = config['SETTINGS_MIZU_ROUTING_VAR']
        - id:   gruId (int)
        """
        try:
            # 1) Locate the FUSE output that the control file points to
            #    Control uses: <fname_qsim> DOMAIN_EXPERIMENT_runs_def.nc
            #    Prefer runs_def; fall back to runs_best if needed.
            out_dir = self.get_simulation_dir('FUSE')
            fuse_id = self._get_fuse_file_id()
            base = f"{self.domain_name}_{fuse_id}"
            candidates = [
                out_dir / f"{base}_runs_def.nc",
                out_dir / f"{base}_runs_best.nc",
            ]
            fuse_output_file = next((p for p in candidates if p.exists()), None)
            if fuse_output_file is None:
                self.logger.error(f"FUSE output file not found. Tried: {candidates}")
                return False

            # 2) Open and convert
            with xr.open_dataset(fuse_output_file) as ds:
                mizu_ds = self._create_mizuroute_forcing_dataset(ds)

            # 3) Overwrite in place so mizuRoute reads exactly what control declares
            #    If the in-use file was runs_best, still write the converted data
            #    back to _runs_def.nc since that's what the control file names.
            write_target = out_dir / f"{base}_runs_def.nc"
            mizu_ds.to_netcdf(write_target, format="NETCDF4")
            self.logger.debug(f"Converted FUSE output → mizuRoute format: {write_target}")
            return True

        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            self.logger.error(f"Error converting FUSE output: {e}")
            return False


    def _create_mizuroute_forcing_dataset(self, fuse_ds: xr.Dataset) -> xr.Dataset:
        """
        Build a mizuRoute-compatible dataset from distributed FUSE output.

        Transforms FUSE spatial output (latitude/longitude dimensions) to
        mizuRoute format (time, gru dimensions). Automatically detects which
        spatial coordinate holds multiple subcatchments and reshapes accordingly.

        Args:
            fuse_ds: FUSE output dataset with dimensions (time, latitude, longitude)
                where one spatial dimension contains subcatchment data.

        Returns:
            xr.Dataset: mizuRoute-compatible dataset with:
                - dims: (time, gru)
                - vars: routing variable (from SETTINGS_MIZU_ROUTING_VAR)
                - gruId: Integer GRU identifiers from spatial coordinates

        Raises:
            ModelExecutionError: If no suitable runoff variable found in FUSE output.
            ValueError: If spatial dimensions cannot be mapped to subcatchments.
        """
        # --- Choose runoff variable (prefer q_routed, else sensible fallbacks)
        routing_var_name = self.mizu_routing_var
        candidates = [
            'q_routed', 'q_instnt', 'qsim', 'runoff',
            # fallbacks by substring
            *[v for v in fuse_ds.data_vars if v.lower().startswith("q_")],
            *[v for v in fuse_ds.data_vars if "runoff" in v.lower()],
        ]
        runoff_src = next((v for v in candidates if v in fuse_ds.data_vars), None)
        if runoff_src is None:
            raise ModelExecutionError(f"No suitable runoff variable found in FUSE output. "
                            f"Available: {list(fuse_ds.data_vars)}")

        # --- Identify spatial axis (one of latitude/longitude must have length > 1)
        lat_len = fuse_ds.sizes.get('latitude', 0)
        lon_len = fuse_ds.sizes.get('longitude', 0)

        # Squeeze singleton param_set from run_pre mode before spatial reshape
        if 'param_set' in fuse_ds.sizes and fuse_ds.sizes['param_set'] == 1:
            fuse_ds = fuse_ds.squeeze('param_set', drop=True)

        if lat_len > 1 and (lon_len in (0, 1)):
            # (time, latitude, 1)
            data = fuse_ds[runoff_src].squeeze('longitude', drop=True).transpose('time', 'latitude')
            spatial_name = 'latitude'
            ids = fuse_ds[spatial_name].values
        elif lon_len > 1 and (lat_len in (0, 1)):
            # (time, 1, longitude)
            data = fuse_ds[runoff_src].squeeze('latitude', drop=True).transpose('time', 'longitude')
            spatial_name = 'longitude'
            ids = fuse_ds[spatial_name].values
        else:
            # If both >1 (unlikely for your setup) or neither, fail loudly
            raise ModelExecutionError(f"Could not infer subcatchment axis from dims: {fuse_ds.dims}")

        # --- Rename spatial dimension to 'gru'
        data = data.rename({data.dims[1]: 'gru'})

        # --- Build output dataset
        mizu = xr.Dataset()
        # copy/forward the time coordinate as-is
        mizu['time'] = fuse_ds['time']
        mizu['time'].attrs.update(fuse_ds['time'].attrs)

        # Add gruId from the spatial coordinate; cast to int32 if possible
        try:
            gid = ids.astype('int32')
        except (ValueError, TypeError):
            gid = ids
        mizu['gru'] = xr.DataArray(range(data.sizes['gru']), dims=('gru',))
        mizu['gruId'] = xr.DataArray(gid, dims=('gru',), attrs={
            'long_name': 'ID of grouped response unit', 'units': '-'
        })

        # Ensure variable is named exactly as control expects
        if runoff_src != routing_var_name:
            data = data.rename(routing_var_name)
        mizu[routing_var_name] = data
        # Add/normalize attrs (units default to m/s unless overridden)
        units = self.mizu_routing_units
        mizu[routing_var_name].attrs.update({'long_name': 'FUSE runoff for mizuRoute routing',
                                            'units': units})

        # Preserve some useful globals if present
        mizu.attrs.update({k: v for k, v in fuse_ds.attrs.items()})

        return mizu


    def _setup_fuse_mizuroute_config(self):
        """Update configuration for FUSE-mizuRoute integration"""

        # Backup experiment_id as instance attribute (was config_dict write-back)
        self._experiment_id_backup = self.experiment_id

        # Set mizuRoute to look for FUSE output instead of SUMMA

    def _is_snow_optimization(self) -> bool:
        """Check if this is a snow optimization run by examining the forcing data."""
        try:
            # Check if q_obs contains only dummy values
            forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"

            if forcing_file.exists():
                with xr.open_dataset(forcing_file) as ds:
                    if 'q_obs' in ds.variables:
                        q_obs_values = ds['q_obs'].values
                        # If all values are -9999 or very close to it, it's dummy data
                        if np.all(np.abs(q_obs_values + 9999) < 0.1):
                            return True

            # Also check optimization target from config
            optimization_target = self._get_config_value(
                lambda: self.config.optimization.target,
                dict_key='OPTIMIZATION_TARGET',
            )
            if optimization_target in ['swe', 'sca', 'snow_depth', 'snow']:
                return True

            return False

        except (FileNotFoundError, KeyError, ValueError) as e:
            self.logger.warning(f"Could not determine if snow optimization: {str(e)}")
            # Fall back to checking config
            optimization_target = self._get_config_value(
                lambda: self.config.optimization.target,
                dict_key='OPTIMIZATION_TARGET',
            )
            return optimization_target in ['swe', 'sca', 'snow_depth', 'snow']

    def _copy_default_to_best_params(self):
        """Copy default parameter file to best parameter file for snow optimization."""
        try:
            fuse_id = self._get_fuse_file_id()
            default_params = self.output_path / f"{self.domain_name}_{fuse_id}_para_def.nc"
            best_params = self.output_path / f"{self.domain_name}_{fuse_id}_para_sce.nc"

            if default_params.exists():
                # Use module-level shutil import (already imported at top of file)
                shutil.copy2(default_params, best_params)
                self.logger.debug("Copied default parameters to best parameters file for snow optimization")
            else:
                self.logger.warning("Default parameter file not found - snow optimization may fail")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error copying default to best parameters: {str(e)}", exc_info=True)

    def _forcing_elevation(self, bands: Dict[str, Any]) -> float:
        """Z_FORCING: ``FUSE_FORCING_ELEVATION`` when set, else the bands' mean elevation."""
        configured = self._get_config_value(
            lambda: self.config.model.fuse.forcing_elevation if self.config.model and self.config.model.fuse else None,
            default=None,
        )
        return float(configured) if configured is not None else float(bands['z_forcing'])

    def _add_elevation_params_to_constraints(self) -> bool:
        """
        Add elevation band parameters to the FUSE constraints file as FIXED parameters.

        FUSE reads its parameters from the constraints file. If elevation band
        parameters (N_BANDS, Z_FORCING, Z_MID01, AF01) are missing, FUSE defaults
        them to zeros which breaks snow aggregation - snowmelt never becomes
        effective precipitation.

        This method reads the elevation bands file and adds the parameters to
        the constraints file BEFORE running FUSE, so they are used correctly.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Read elevation bands file
            elev_bands_path = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            if not elev_bands_path.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_path}")
                return False

            # Per-band values of the (first) spatial unit; Z_FORCING is the
            # area-weighted mean elevation
            bands = read_band_parameters(elev_bands_path)
            n_bands = bands['n_bands']
            mean_elevs = bands['mean_elev']
            area_fracs = bands['area_frac']
            z_forcing = self._forcing_elevation(bands)

            # Find constraints file
            constraints_file = self.setup_dir / 'fuse_zConstraints_snow.txt'
            if not constraints_file.exists():
                self.logger.warning(f"Constraints file not found: {constraints_file}")
                return False

            # Read existing constraints
            with open(constraints_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Check if elevation params already exist
            existing_params = set()
            for line in lines:
                for param in ['N_BANDS', 'Z_FORCING', 'Z_MID01', 'AF01']:
                    if param in line and not line.strip().startswith('!'):
                        existing_params.add(param)

            if existing_params:
                self.logger.debug(f"Elevation params already in constraints: {existing_params}")
                # Update existing values instead of adding duplicates
                updated_lines = []
                for line in lines:
                    if 'N_BANDS' in line and not line.strip().startswith('!'):
                        line = f"F 0 {float(n_bands):9.3f} {float(n_bands):9.3f} {float(n_bands):9.3f} .10   1.0 0 0 0  0 0 0 N_BANDS   NO_CHILD1 NO_CHILD2 ! number of elevation bands\n"
                    elif 'Z_FORCING' in line and not line.strip().startswith('!'):
                        line = f"F 0 {z_forcing:9.3f} {z_forcing:9.3f} {z_forcing:9.3f} .10   1.0 0 0 0  0 0 0 Z_FORCING NO_CHILD1 NO_CHILD2 ! forcing elevation (m)\n"
                    elif 'Z_MID01' in line and not line.strip().startswith('!'):
                        line = f"F 0 {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} .10   1.0 0 0 0  0 0 0 Z_MID01   NO_CHILD1 NO_CHILD2 ! band 1 elevation (m)\n"
                    elif 'AF01' in line and not line.strip().startswith('!'):
                        line = f"F 0 {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} .10   1.0 0 0 0  0 0 0 AF01      NO_CHILD1 NO_CHILD2 ! band 1 area fraction\n"
                    updated_lines.append(line)
                lines = updated_lines

            # Add elevation params if not present
            params_to_add = []
            if 'N_BANDS' not in existing_params:
                params_to_add.append(f"F 0 {float(n_bands):9.3f} {float(n_bands):9.3f} {float(n_bands):9.3f} .10   1.0 0 0 0  0 0 0 N_BANDS   NO_CHILD1 NO_CHILD2 ! number of elevation bands\n")
            if 'Z_FORCING' not in existing_params:
                params_to_add.append(f"F 0 {z_forcing:9.3f} {z_forcing:9.3f} {z_forcing:9.3f} .10   1.0 0 0 0  0 0 0 Z_FORCING NO_CHILD1 NO_CHILD2 ! forcing elevation (m)\n")
            if 'Z_MID01' not in existing_params:
                params_to_add.append(f"F 0 {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} .10   1.0 0 0 0  0 0 0 Z_MID01   NO_CHILD1 NO_CHILD2 ! band 1 elevation (m)\n")
            if 'AF01' not in existing_params:
                params_to_add.append(f"F 0 {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} .10   1.0 0 0 0  0 0 0 AF01      NO_CHILD1 NO_CHILD2 ! band 1 area fraction\n")

            if params_to_add:
                # Find position to insert (before the description section)
                insert_pos = len(lines)
                for i, line in enumerate(lines):
                    if '*****' in line or 'description' in line.lower():
                        insert_pos = i
                        break

                # Insert the new parameters
                for param_line in reversed(params_to_add):
                    lines.insert(insert_pos, param_line)

            # Write back
            with open(constraints_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            self.logger.debug(f"Added elevation params to constraints: N_BANDS={n_bands}, "
                           f"Z_FORCING={z_forcing:.1f}, Z_MID01={mean_elevs[0]:.1f}, AF01={area_fracs[0]:.3f}")
            return True

        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to add elevation params to constraints: {e}")
            return False

    def _fix_elevation_band_params_in_para_def(self) -> bool:
        """
        Fix elevation band parameters in para_def.nc after FUSE run_def creates it.

        FUSE's run_def mode generates para_def.nc with zeros for elevation band
        parameters (N_BANDS, Z_FORCING, Z_MID01, AF01), which causes snow
        aggregation to fail - snowmelt never becomes effective precipitation.

        Also handles the case where FUSE creates para_def.nc with par=UNLIMITED
        and 0 records (silent failure), by initializing default parameter values.

        This method reads the elevation bands file and updates para_def.nc with
        the correct values so that calib_sce and run_best will work properly.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            fuse_id = self._get_fuse_file_id()
            para_def_path = self.output_path / f"{self.domain_name}_{fuse_id}_para_def.nc"

            if not para_def_path.exists():
                self.logger.warning(f"para_def.nc not found: {para_def_path}")
                return False

            # Read elevation bands file to get correct values
            elev_bands_path = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            if not elev_bands_path.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_path}")
                return False

            # Per-band values of the (first) spatial unit; Z_FORCING is the
            # area-weighted mean elevation
            bands = read_band_parameters(elev_bands_path)
            n_bands = bands['n_bands']
            mean_elevs = bands['mean_elev']
            area_fracs = bands['area_frac']
            z_forcing = self._forcing_elevation(bands)

            # Update para_def.nc with correct elevation band parameters
            with xr.open_dataset(para_def_path) as ds:
                ds_attrs = dict(ds.attrs)
                par_size = ds.sizes.get('par', 0)
                ds_dict = {}

                if par_size == 0:
                    # FUSE created the file structure but wrote 0 records
                    # (silent failure). Initialize all variables with defaults
                    # from the constraints file, falling back to bounds midpoints.
                    self.logger.warning(
                        f"para_def.nc has empty par dimension (0 records). "
                        f"Initializing with default values for {len(ds.data_vars)} variables."
                    )
                    from symfluence.core.calibration.parameters.parameter_bounds_registry import get_fuse_bounds
                    from symfluence.models.fuse.calibration.parameter_application import parse_fuse_constraints_defaults
                    bounds = get_fuse_bounds()

                    # Read defaults from FUSE constraints file (authoritative source)
                    constraints_path = self.forcing_fuse_path.parent.parent / 'settings' / 'FUSE' / 'fuse_zConstraints_snow.txt'
                    constraint_defaults = parse_fuse_constraints_defaults(constraints_path)
                    if constraint_defaults:
                        self.logger.debug(f"Loaded {len(constraint_defaults)} defaults from constraints file")

                    for var in ds.data_vars:
                        var_name = str(var)
                        if var_name in constraint_defaults:
                            default_val = constraint_defaults[var_name]
                        elif var_name in bounds:
                            default_val = (bounds[var_name]['min'] + bounds[var_name]['max']) / 2.0
                        else:
                            default_val = 0.0
                        ds_dict[var_name] = np.array([default_val])
                else:
                    # Normal case: keep only first parameter set
                    for var in ds.data_vars:
                        vals = ds[var].values
                        var_name = str(var)
                        if vals.ndim > 0 and len(vals) > 0:
                            ds_dict[var_name] = np.array([vals[0]])
                        else:
                            ds_dict[var_name] = np.array([0.0])

            # Update the elevation band parameters
            ds_dict['N_BANDS'] = np.array([float(n_bands)])
            ds_dict['Z_FORCING'] = np.array([z_forcing])

            for i in range(n_bands):
                z_mid_name = f'Z_MID{i+1:02d}'
                af_name = f'AF{i+1:02d}'
                ds_dict[z_mid_name] = np.array([float(mean_elevs[i])])
                ds_dict[af_name] = np.array([float(area_fracs[i])])

            # Compute derived storage parameters from base calibration params.
            # FUSE's run_pre mode reads ALL values — if derived params are 0,
            # the model has zero storage capacity and produces garbage output.
            maxwatr_1 = float(ds_dict.get('MAXWATR_1', np.array([0.0]))[0])
            fracten = float(ds_dict.get('FRACTEN', np.array([0.5]))[0])
            maxwatr_2 = float(ds_dict.get('MAXWATR_2', np.array([0.0]))[0])
            rtfrac1 = float(ds_dict.get('RTFRAC1', np.array([0.5]))[0])

            ds_dict['MAXTENS_1'] = np.array([maxwatr_1 * fracten])
            ds_dict['MAXFREE_1'] = np.array([maxwatr_1 * (1.0 - fracten)])
            ds_dict['MAXTENS_1A'] = np.array([maxwatr_1 * fracten * 0.5])
            ds_dict['MAXTENS_1B'] = np.array([maxwatr_1 * fracten * 0.5])
            ds_dict['MAXTENS_2'] = np.array([maxwatr_2 * fracten])
            ds_dict['MAXFREE_2'] = np.array([maxwatr_2 * (1.0 - fracten)])
            ds_dict['MAXFREE_2A'] = np.array([maxwatr_2 * (1.0 - fracten) * 0.5])
            ds_dict['MAXFREE_2B'] = np.array([maxwatr_2 * (1.0 - fracten) * 0.5])
            ds_dict['RTFRAC2'] = np.array([1.0 - rtfrac1])

            # Write back to NetCDF with single parameter set
            new_ds = xr.Dataset(
                {var: (['par'], vals) for var, vals in ds_dict.items()},
                coords={'par': [0]},
                attrs=ds_attrs
            )
            new_ds.to_netcdf(para_def_path, mode='w')

            # Apply numerix defaults via the calibration module
            from symfluence.models.fuse.calibration.parameter_application import compute_derived_parameters
            compute_derived_parameters(para_def_path, self.logger)

            self.logger.debug(
                f"Fixed para_def.nc: N_BANDS={n_bands}, Z_FORCING={z_forcing:.1f}, "
                f"MAXTENS_1={maxwatr_1 * fracten:.1f}, MAXFREE_1={maxwatr_1 * (1.0 - fracten):.1f}"
            )
            return True

        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to fix elevation band params: {e}")
            return False

    def _fix_elevation_band_params_in_para_sce(self) -> bool:
        """
        Fix elevation band parameters in para_sce.nc after FUSE calibration.

        Similar to _fix_elevation_band_params_in_para_def, but for the calibration
        output file. FUSE's calib_sce creates para_sce.nc with zeros for elevation
        band parameters, causing run_best to fail in aggregating snowmelt.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            fuse_id = self._get_fuse_file_id()
            para_sce_path = self.output_path / f"{self.domain_name}_{fuse_id}_para_sce.nc"

            if not para_sce_path.exists():
                self.logger.warning(f"para_sce.nc not found: {para_sce_path}")
                return False

            # Read elevation bands file to get correct values
            elev_bands_path = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            if not elev_bands_path.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_path}")
                return False

            # Per-band values of the (first) spatial unit; Z_FORCING is the
            # area-weighted mean elevation
            bands = read_band_parameters(elev_bands_path)
            n_bands = bands['n_bands']
            mean_elevs = bands['mean_elev']
            area_fracs = bands['area_frac']
            z_forcing = self._forcing_elevation(bands)

            # Update para_sce.nc with correct elevation band parameters
            # Need to handle the 'par' dimension which may have multiple parameter sets
            with xr.open_dataset(para_sce_path) as ds:
                par_size = ds.sizes.get('par', 1)
                ds_dict = {}
                for var in ds.data_vars:
                    ds_dict[var] = ds[var].values.copy()
                ds_attrs = dict(ds.attrs)
                ds_coords = {c: ds.coords[c].values for c in ds.coords}

            # Update elevation band parameters - broadcast to all parameter sets
            ds_dict['N_BANDS'] = np.full(par_size, float(n_bands))
            ds_dict['Z_FORCING'] = np.full(par_size, z_forcing)

            for i in range(n_bands):
                z_mid_name = f'Z_MID{i+1:02d}'
                af_name = f'AF{i+1:02d}'
                ds_dict[z_mid_name] = np.full(par_size, float(mean_elevs[i]))
                ds_dict[af_name] = np.full(par_size, float(area_fracs[i]))

            # Write back to NetCDF
            new_ds = xr.Dataset(
                {var: (['par'], vals) for var, vals in ds_dict.items()},
                coords={'par': ds_coords.get('par', np.arange(par_size))},
                attrs=ds_attrs
            )
            new_ds.to_netcdf(para_sce_path, mode='w')

            self.logger.debug(f"Fixed elevation band params in para_sce.nc: N_BANDS={n_bands}, "
                           f"Z_FORCING={z_forcing:.1f}, Z_MID01={mean_elevs[0]:.1f}, AF01={area_fracs[0]:.3f}")
            return True

        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to fix elevation band params in para_sce.nc: {e}")
            return False

    def _update_filemanager_for_run(self) -> bool:
        """
        Update file manager with current experiment settings before running FUSE.

        Ensures OUTPUT_PATH and FMODEL_ID match the current experiment configuration,
        allowing the same preprocessed setup to be used for different experiment runs.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            fuse_fm = self._get_config_value(
                lambda: self.config.model.fuse.filemanager,
                dict_key='SETTINGS_FUSE_FILEMANAGER',
            )
            if fuse_fm == 'default':
                fuse_fm = 'fm_catch.txt'
            fm_path = self.project_dir / 'settings' / 'FUSE' / fuse_fm

            if not fm_path.exists():
                self.logger.warning(f"File manager not found: {fm_path}")
                return False

            # Read current file manager with encoding fallback
            try:
                with open(fm_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(f"UTF-8 decode error reading {fm_path}, falling back to latin-1")
                with open(fm_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Get current settings
            fuse_id = self._get_fuse_file_id()
            output_path = str(self.output_path) + '/'

            # Find actual decisions file
            settings_dir = self.project_dir / 'settings' / 'FUSE'
            decisions_file = f"fuse_zDecisions_{fuse_id}.txt"
            if not (settings_dir / decisions_file).exists():
                # Find any decisions file
                decisions = list(settings_dir.glob("fuse_zDecisions_*.txt"))
                if decisions:
                    decisions_file = decisions[0].name
                    self.logger.debug(f"Using available decisions file: {decisions_file}")

            # Update relevant lines
            updated_lines = []
            input_path_str = str(self.forcing_fuse_path) + '/'

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("'") and 'SETNGS_PATH' in line:
                    settings_path = str(self.project_dir / 'settings' / 'FUSE') + '/'
                    updated_lines.append(f"'{settings_path}'     ! SETNGS_PATH\n")
                elif stripped.startswith("'") and 'OUTPUT_PATH' in line:
                    updated_lines.append(f"'{output_path}'       ! OUTPUT_PATH\n")
                elif stripped.startswith("'") and 'INPUT_PATH' in line:
                     updated_lines.append(f"'{input_path_str}'       ! INPUT_PATH\n")
                elif stripped.startswith("'") and 'FORCING INFO' in line:
                    updated_lines.append("'input_info.txt'                 ! FORCING INFO       = definition of the forcing file\n")
                elif stripped.startswith("'") and 'FMODEL_ID' in line:
                    updated_lines.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
                elif stripped.startswith("'") and 'M_DECISIONS' in line:
                    updated_lines.append(f"'{decisions_file}'        ! M_DECISIONS        = definition of model decisions\n")
                else:
                    updated_lines.append(line)

            # Write updated file manager
            with open(fm_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.debug(
                f"Updated file manager ({fm_path.name}): "
                f"OUTPUT_PATH={output_path}, INPUT_PATH={input_path_str}, "
                f"FMODEL_ID={fuse_id}, M_DECISIONS={decisions_file}"
            )

            # Verify paths were written correctly by reading back
            with open(fm_path, 'r', encoding='utf-8') as f:
                verify_lines = f.readlines()
            has_relative = any(
                "'./" in ln and any(k in ln for k in ('SETNGS_PATH', 'OUTPUT_PATH', 'INPUT_PATH'))
                for ln in verify_lines
            )
            if has_relative:
                self.logger.warning(
                    "File manager still contains relative paths after line-matching "
                    "update — applying brute-force replacement."
                )
                settings_path = str(self.project_dir / 'settings' / 'FUSE') + '/'
                patched = []
                for ln in verify_lines:
                    if 'SETNGS_PATH' in ln and ln.strip().startswith("'"):
                        patched.append(f"'{settings_path}'     ! SETNGS_PATH\n")
                    elif 'OUTPUT_PATH' in ln and ln.strip().startswith("'"):
                        patched.append(f"'{output_path}'       ! OUTPUT_PATH\n")
                    elif 'INPUT_PATH' in ln and ln.strip().startswith("'"):
                        patched.append(f"'{input_path_str}'        ! INPUT_PATH\n")
                    elif 'FMODEL_ID' in ln and ln.strip().startswith("'"):
                        patched.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
                    else:
                        patched.append(ln)
                with open(fm_path, 'w', encoding='utf-8') as f:
                    f.writelines(patched)
                self.logger.debug("Brute-force path replacement applied to file manager")

            return True

        except (FileNotFoundError, OSError, PermissionError) as e:
            self.logger.error(f"Failed to update file manager: {e}")
            return False

    def _execute_fuse(self, mode: str, para_file: Optional[Path] = None,
                      param_index: int = 1) -> bool:
        """
        Execute the FUSE model with specified run mode.

        Constructs and executes the FUSE command with the given mode,
        capturing output to a log file. Uses BaseModelRunner execution mixins for
        subprocess management.

        Args:
            mode: FUSE run mode, one of:
                - 'run_def': Run with default parameters
                - 'calib_sce': Run SCE-UA calibration
                - 'run_best': Run with calibrated parameters
                - 'run_pre': Run with provided parameter file
            para_file: Path to parameter file for 'run_pre' mode (optional).
            param_index: 1-based parameter set index for 'run_pre' mode.

        Returns:
            bool: True if execution was successful, False otherwise.
        """
        # Update file manager with current experiment settings
        self._update_filemanager_for_run()

        self.logger.debug("Executing FUSE model")

        # Construct command
        fuse_fm = self._get_config_value(
            lambda: self.config.model.fuse.filemanager,
            dict_key='SETTINGS_FUSE_FILEMANAGER',
        )
        if fuse_fm == 'default':
            fuse_fm = 'fm_catch.txt'

        control_file = self.project_dir / 'settings' / 'FUSE' / fuse_fm

        command = [
            str(self.fuse_exe),
            str(control_file),
            self.domain_name,
            mode
        ]
        # For run_pre mode, append parameter file name and index
        # FUSE expects: fuse.exe fm_catch.txt basin_id run_pre para_file.nc <index>
        if mode == 'run_pre' and para_file:
            command.append(str(para_file.name))
            command.append(str(param_index))

        # Create log file path - use mode-specific log files to avoid overwriting
        log_file = self.get_log_path() / f'fuse_{mode}.log'

        try:
            result = self.execute_subprocess(
                command,
                log_file,
                cwd=self.setup_dir,  # Run from settings directory where input_info.txt lives
                check=False,  # Don't raise, we'll return boolean
                success_message="FUSE execution completed",
                timeout=self._get_config_value(lambda: self.config.model.fuse.timeout, dict_key='FUSE_TIMEOUT'),
            )
            self.logger.debug(f"FUSE return code: {result.return_code}")

            # Check log for silent failures (FUSE returns exit 0 on Fortran STOP
            # and NetCDF errors on macOS)
            if result.success and log_file.exists():
                try:
                    log_content = log_file.read_text(encoding='utf-8', errors='replace')
                    if 'NetCDF:' in log_content:
                        nc_first = next(
                            (l.strip() for l in log_content.splitlines() if 'NetCDF:' in l), ''
                        )
                        log_once(
                            self.logger, logging.ERROR,
                            f"fuse-netcdf:{nc_first}",
                            f"FUSE NetCDF error (exit code 0): {nc_first}; "
                            f"full output: {log_file}",
                        )
                        return False
                    # Check for Fortran STOP statements. Exclude SCE's normal
                    # completion messages ("SEARCH WAS STOPPED AT", "KSTOP")
                    # which contain STOP as a substring but are not errors.
                    import re
                    stop_first = next(
                        (
                            l.strip() for l in log_content.splitlines()
                            if re.search(r'\bSTOP\b', l) and 'STOPPED' not in l and 'KSTOP' not in l
                        ),
                        None,
                    )
                    if stop_first is not None:
                        log_once(
                            self.logger, logging.ERROR,
                            f"fuse-stop:{stop_first}",
                            f"FUSE Fortran STOP (exit code 0): {stop_first}; "
                            f"full output: {log_file}",
                        )
                        return False
                except OSError:
                    pass

            return result.success

        except subprocess.TimeoutExpired as e:
            self.logger.error(
                f"FUSE exceeded its {e.timeout:.0f}s timeout and was killed "
                f"(log: {log_file}). Raise FUSE_TIMEOUT if this domain is "
                f"legitimately slower than that."
            )
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed with error: {str(e)}")
            return False

    def _process_outputs(self):
        """Process and organize FUSE output files."""
        self.logger.debug("Processing FUSE outputs")

        output_dir = self.output_path / 'output'

        # Read and process streamflow output
        q_file = output_dir / 'streamflow.nc'
        if q_file.exists():
            with xr.open_dataset(q_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.experiment_id
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save processed output
                processed_file = self.output_path / f"{self.experiment_id}_streamflow.nc"
                ds.to_netcdf(processed_file)
                self.logger.debug(f"Processed streamflow output saved to: {processed_file}")

        # Process state variables if they exist
        state_file = output_dir / 'states.nc'
        if state_file.exists():
            with xr.open_dataset(state_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.experiment_id
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save processed output
                processed_file = self.output_path / f"{self.experiment_id}_states.nc"
                ds.to_netcdf(processed_file)
                self.logger.debug(f"Processed state variables saved to: {processed_file}")


    def _recover_para_def_from_cwd(self, fuse_id: str, expected_path: Path) -> Path:
        """Search for para_def.nc in FUSE's working directory and copy to output.

        When FUSE's file manager has relative paths (e.g. './'), output files
        land in the working directory (settings/FUSE/) instead of the intended
        output directory.  This method finds the file there, copies it to the
        expected location, and returns the path.

        Args:
            fuse_id: FUSE model file ID (≤6 chars).
            expected_path: Where the caller expects para_def.nc to be.

        Returns:
            *expected_path* if recovery succeeded, otherwise *expected_path*
            unchanged (caller should check ``exists()``).
        """
        cwd_candidates = list(self.setup_dir.glob(f"*{fuse_id}*para_def.nc"))
        if not cwd_candidates:
            # Also try any para_def.nc in cwd (different FMODEL_ID)
            cwd_candidates = list(self.setup_dir.glob("*para_def.nc"))

        if cwd_candidates:
            # Pick the newest, non-trivially-sized file
            cwd_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            source = next(
                (c for c in cwd_candidates if c.stat().st_size > 1024), None
            )
            if source:
                self.logger.warning(
                    f"para_def.nc not in output dir but found in FUSE working "
                    f"directory: {source.name} — FUSE may have used relative "
                    f"paths.  Copying to {expected_path.name}."
                )
                self.ensure_dir(expected_path.parent)
                shutil.copy2(source, expected_path)

                # Also rescue any runs_def.nc (FUSE creates it alongside para_def)
                for suffix in ('runs_def', 'runs_pre'):
                    run_file = source.parent / source.name.replace('para_def', suffix)
                    if run_file.exists() and run_file.stat().st_size > 1024:
                        dest = expected_path.parent / expected_path.name.replace('para_def', suffix)
                        if not dest.exists():
                            shutil.copy2(run_file, dest)
        return expected_path

    def _run_lumped_fuse(self) -> bool:
        """Run FUSE in lumped mode using the two-phase workflow.

        FUSE's two run modes have distinct roles:

        - ``run_def`` is the *initial default run only*: it derives
          para_def.nc from the constraints file and produces the
          default-parameter baseline. It runs exactly once, when no
          para_def.nc exists yet (FUSE creates para_def.nc even when the
          run itself crashes on the NC_UNLIMITED NETCDF3 conflict present
          in some builds).
        - ``run_pre`` is the production mode for every subsequent
          simulation — and the only mode calibration uses — reading
          parameters directly from para_def.nc.
        """
        self.logger.info("Running lumped FUSE workflow")

        try:
            # Check if this is a snow optimization case
            if self._is_snow_optimization():
                self.logger.debug("Snow optimization detected - copying default to best parameters")
                self._copy_default_to_best_params()

            fuse_id = self._get_fuse_file_id()
            para_def_path = self.output_path / f"{self.domain_name}_{fuse_id}_para_def.nc"

            # Use run_pre when possible (run_def is broken in many FUSE builds
            # due to NC_UNLIMITED conflict in NETCDF3_CLASSIC format)
            if para_def_path.exists():
                self.logger.debug(f"Using run_pre mode with existing para_def: {para_def_path.name}")
                success = self._execute_fuse('run_pre', para_file=para_def_path)
            else:
                # No para_def.nc yet — run_def creates it as a side effect
                # (even if run_def itself crashes on runs_def.nc)
                self.logger.debug("No para_def.nc found, running run_def to generate it")
                self._execute_fuse('run_def')

                # If not found in output_path, check the FUSE working directory
                # (settings/FUSE/) — FUSE writes there when fm_catch.txt has
                # relative paths (e.g. './') instead of absolute OUTPUT_PATH
                if not para_def_path.exists():
                    para_def_path = self._recover_para_def_from_cwd(
                        fuse_id, para_def_path
                    )

                if para_def_path.exists():
                    self._fix_elevation_band_params_in_para_def()
                    self.logger.debug("Retrying with run_pre using generated para_def.nc")
                    success = self._execute_fuse('run_pre', para_file=para_def_path)
                else:
                    self.logger.error(
                        "FUSE run_def did not create para_def.nc. "
                        "Cannot proceed with run_pre mode."
                    )
                    success = False

            # Fix elevation params for downstream uses
            if para_def_path.exists():
                self._fix_elevation_band_params_in_para_def()

            # Check if FUSE internal calibration should run (independent of external optimization)
            run_internal_calibration = self._get_config_value(
                lambda: self.config.model.fuse.run_internal_calibration,
                dict_key='FUSE_RUN_INTERNAL_CALIBRATION',
            )

            if run_internal_calibration:
                try:
                    # Run FUSE internal SCE-UA calibration as benchmark
                    # Internal calibration failure is non-fatal for the overall workflow
                    self.logger.info("Running FUSE internal calibration (calib_sce) as benchmark")
                    calib_success = self._execute_fuse('calib_sce')

                    if calib_success:
                        # CRITICAL: Fix elevation band parameters in para_sce.nc too
                        # FUSE calib_sce creates para_sce.nc with zeros for Z_FORCING, Z_MID, AF
                        self._fix_elevation_band_params_in_para_sce()

                        # Run FUSE with best parameters from internal calibration
                        if not self._execute_fuse('run_best'):
                            self.logger.warning(
                                "FUSE run_best failed (non-fatal). "
                                "External calibration will proceed using run_pre output."
                            )
                    else:
                        self.logger.warning("FUSE calib_sce failed (non-fatal for workflow)")
                except (subprocess.CalledProcessError, OSError, RuntimeError) as e:
                    self.logger.warning(f'FUSE internal calibration failed: {e}')
            else:
                self.logger.info("FUSE internal calibration disabled (FUSE_RUN_INTERNAL_CALIBRATION=false)")

            if success:
                # Ensure the expected output file exists
                self._ensure_best_output_file()
                self.logger.debug("Lumped FUSE run completed successfully")
                return True
            else:
                self.logger.error("Lumped FUSE run failed")
                return False

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error in lumped FUSE execution: {str(e)}", exc_info=True)
            return False

    def backup_run_files(self):
        """Backup important run files for reproducibility."""
        self.logger.debug("Backing up run files")

        backup_dir = self.output_path / 'run_settings'
        backup_dir.mkdir(exist_ok=True)

        files_to_backup = [
            self.output_path / 'settings' / 'control.txt',
            self.output_path / 'settings' / 'structure.txt',
            self.output_path / 'settings' / 'params.txt'
        ]

        for file in files_to_backup:
            if file.exists():
                shutil.copy2(file, backup_dir / file.name)
                self.logger.debug(f"Backed up {file.name}")
