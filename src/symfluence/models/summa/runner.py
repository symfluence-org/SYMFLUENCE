# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Runner Module

This module contains the SummaRunner class for executing the SUMMA
(Structure for Unifying Multiple Modeling Alternatives) model.

The SummaRunner handles model execution in various modes:
- Serial execution for single-threaded runs
- Parallel execution using SLURM job arrays or local GRU-split execution
- Point simulation mode for multiple point-based simulations

Refactored to use the Unified Model Execution Framework:
- ModelExecutor: For subprocess and SLURM execution
- SpatialOrchestrator: For routing integration

Author: SYMFLUENCE Development Team
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr

from symfluence.core.modeling.execution import ExecutionResult, SlurmJobConfig
from symfluence.core.modeling.state import ModelState, StateCapableMixin, StateFormat, StateMetadata
from symfluence.core.modeling.templates import ModelRunResult, UnifiedModelRunner
from symfluence.core.registries import R

from .parallel_gru_execution import run_summa_gru_parallel


@R.runners.add('SUMMA', runner_method='run_summa')
class SummaRunner(UnifiedModelRunner, StateCapableMixin):  # type: ignore[misc]
    """
    A class to run the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    This class handles the execution of the SUMMA model, including setting up paths,
    running the model, and managing log files.

    Now uses the Unified Model Execution Framework for:
    - SLURM job submission and monitoring (via ModelExecutor)
    - Routing integration (via SpatialOrchestrator)

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """

    MODEL_NAME = "SUMMA"

    def _setup_model_specific_paths(self) -> None:
        """Set up SUMMA-specific paths."""
        self.settings_path = self.get_config_path(
            'SETTINGS_SUMMA_PATH',
            'settings/SUMMA/'
        )
        self.file_manager = self.settings_path / self._get_config_value(
            lambda: self.config.model.summa.filemanager,
            default='fileManager.txt'
        )

        # Legacy alias for backward compatibility
        self.setup_path_aliases({'root_path': 'data_dir'})

    def _should_create_output_dir(self) -> bool:
        """SUMMA creates output dirs on-demand."""
        return False

    def _build_command(self) -> List[str]:
        """Build SUMMA execution command."""
        return [
            str(self.model_exe),
            '-m', str(self.file_manager)
        ]

    def _get_environment(self) -> Dict[str, str]:
        """Get environment variables for SUMMA."""
        import os
        import platform
        env: Dict[str, str] = {}
        # On x86-64, preload libftz.so to flush denormals to zero.
        # Without this, gfortran's Jacobian produces denormals that
        # propagate to NaN (ARM and Intel Fortran do this by default).
        if platform.machine() in ('x86_64', 'AMD64'):
            summa_bin = self.model_exe.parent if hasattr(self, 'model_exe') else None
            if summa_bin:
                libftz = summa_bin / 'libftz.so'
                if libftz.exists():
                    existing = os.environ.get('LD_PRELOAD', '')
                    env['LD_PRELOAD'] = f"{libftz}:{existing}" if existing else str(libftz)
        return env

    def _validate_model_specific(self) -> List[str]:
        """Validate SUMMA-specific configuration."""
        errors = []

        # Check file manager exists
        if hasattr(self, 'file_manager') and not self.file_manager.exists():
            errors.append(f"File manager not found: {self.file_manager}")

        return errors

    def _get_slurm_config(self) -> Optional[SlurmJobConfig]:
        """Get SLURM configuration for parallel SUMMA."""
        use_parallel = self._get_config_value(
            lambda: self.config.model.summa.use_parallel, default=False
        )

        if not use_parallel:
            return None

        log_path = self.get_config_path(
            'EXPERIMENT_LOG_SUMMA',
            f"simulations/{self.experiment_id}/SUMMA/SUMMA_logs/"
        )

        return SlurmJobConfig(
            job_name=f"SUMMA-{self.domain_name}",
            time_limit="03:00:00",
            memory="4G",
            cpus_per_task=1,
            output_pattern=str(log_path / "summa_%A_%a.out"),
            error_pattern=str(log_path / "summa_%A_%a.err"),
        )

    def _pre_execution(self) -> bool:
        """Set up for SUMMA execution."""
        # Create output directories
        self.output_dir = self.get_config_path(
            'EXPERIMENT_OUTPUT_SUMMA',
            f"simulations/{self.experiment_id}/SUMMA/"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.get_config_path(
            'EXPERIMENT_LOG_SUMMA',
            f"simulations/{self.experiment_id}/SUMMA/SUMMA_logs/"
        )
        log_path.mkdir(parents=True, exist_ok=True)

        # Backup settings if requested
        backup = self._get_config_value(
            lambda: self.config.model.summa.backup_settings, default='no'
        )
        if backup == 'yes':
            self.backup_settings(self.settings_path)

        return True

    def _post_execution(self, result: ExecutionResult) -> ModelRunResult:
        """Process SUMMA outputs."""
        run_result = ModelRunResult(
            success=result.success,
            output_path=self.output_dir,
            error=result.error_message,
            metadata={
                'duration_seconds': result.duration_seconds,
                'job_id': result.job_id,
            }
        )

        if not result.success:
            return run_result

        from symfluence.core.exceptions import ValidationError

        from .output_validation import validate_output_coverage
        try:
            validate_output_coverage(self.file_manager, self.output_dir,
                                     self._get_config_value(
                                         lambda: self.config.forcing.time_step_size, default=None))
        except (ValidationError, ValueError) as exc:
            run_result.success = False
            run_result.error = str(exc)
            self.logger.error(str(exc))
            return run_result

        # Check if we need to convert lumped output for distributed routing
        domain_method = self.domain_definition_method
        routing_delineation = self._get_config_value(
            lambda: self.config.domain.delineation.routing, default='lumped'
        )

        if domain_method == 'lumped' and routing_delineation == 'river_network':
            self.logger.info("Converting lumped output for distributed routing")
            self._convert_lumped_for_routing()

        return run_result

    # =========================================================================
    # State Save/Restore (StateCapableMixin)
    # =========================================================================

    def get_state_format(self) -> StateFormat:
        return StateFormat.FILE_NETCDF

    def get_state_variables(self) -> List[str]:
        return [
            'scalarSWE', 'scalarSnowDepth', 'mLayerTemp', 'mLayerVolFracIce',
            'mLayerVolFracLiq', 'mLayerMatricHead', 'scalarAquiferStorage',
            'scalarSurfaceTemp', 'mLayerDepth',
        ]

    def get_state_directory(self) -> Optional[Path]:
        return getattr(self, 'output_dir', None)

    def get_state_file_pattern(self) -> str:
        return "*_restart_*.nc"

    def save_state(
        self,
        target_dir: Path,
        timestamp: str,
        ensemble_member: Optional[int] = None,
    ) -> ModelState:
        """Locate the newest SUMMA restart file and copy to target_dir."""
        import shutil

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Find restart files in output directory
        state_dir = self.get_state_directory() or self.output_dir
        restart_files = sorted(state_dir.glob(self.get_state_file_pattern()))

        if not restart_files:
            from symfluence.core.modeling.state.exceptions import StateError
            raise StateError(f"No SUMMA restart files found in {state_dir}")

        newest = max(restart_files, key=lambda p: p.stat().st_mtime)
        dst = target_dir / newest.name
        shutil.copy2(str(newest), str(dst))
        self.logger.info("Saved SUMMA state: %s -> %s", newest, dst)

        metadata = StateMetadata(
            model_name='SUMMA',
            timestamp=timestamp,
            format=StateFormat.FILE_NETCDF,
            variables=self.get_state_variables(),
            ensemble_member=ensemble_member,
        )
        return ModelState(metadata=metadata, files=[dst])

    def load_state(self, state: ModelState) -> None:
        """Copy state file into settings directory as coldState.nc."""
        import shutil

        if not state.files:
            self.logger.warning("No state files to load for SUMMA")
            return

        coldstate_name = self._get_config_value(
            lambda: self.config.model.summa.coldstate, default='coldState.nc'
        )
        dst = self.settings_path / coldstate_name

        src = state.files[0]
        shutil.copy2(str(src), str(dst))
        self.logger.info("Loaded SUMMA state: %s -> %s", src, dst)

    # =========================================================================
    # SUMMA-Specific Methods
    # =========================================================================

    def run_summa(self) -> Optional[Path]:
        """
        Run the SUMMA model.

        This method selects the appropriate run mode (parallel, serial, or point)
        based on configuration settings and executes the SUMMA model accordingly.

        Delegates to appropriate execution method based on configuration.
        """
        # Check for point mode
        if self.domain_definition_method == 'point':
            output = self.run_summa_point()
        else:
            use_parallel = self._get_config_value(
                lambda: self.config.model.summa.use_parallel, default=False
            )
            if use_parallel:
                output = self.run_parallel_summa()
            else:
                # Serial execution handled by the base class.
                output = self.run()

        # The sequential manager ignores return values. A failed upstream run
        # must raise so downstream routing never consumes unfinished output.
        if output is None:
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError(
                'SUMMA did not complete successfully; downstream routing is blocked. '
                'See the SUMMA execution log for the underlying failure.'
            )
        return output

    def run_parallel_summa(self) -> Optional[Path]:
        """
        Run SUMMA in parallel across GRUs using the configured backend.

        The default 'slurm' backend uses SLURM job arrays, while the 'local' backend uses
        Python's ThreadPoolExecutor (also used as a fallback if SLURM is unavailable).
        """
        backend = self._get_config_value(
            lambda: self.config.model.summa.parallel_backend, default='slurm'
        )
        backend = str(backend or 'slurm').lower()

        if backend == 'local':
            return self._run_parallel_summa_local()
        if backend == 'slurm':
            # Use the local GRU splitter when SLURM is unavailable
            if not self.is_slurm_available():
                self.logger.warning(
                    "SLURM not available, falling back to local SUMMA "
                    "GRU splitting"
                )
                return self._run_parallel_summa_local()
            return self._run_parallel_summa_slurm()

        self.logger.error(f"Unknown SUMMA parallel backend: {backend}")
        return None

    def _run_parallel_summa_slurm(self) -> Optional[Path]:
        """Run SUMMA in parallel using SLURM arrays."""
        self.logger.info("Starting parallel SUMMA run with SLURM")

        # Get GRU count from shapefile
        total_grus = self._count_grus()
        self.logger.info(f"Total GRUs: {total_grus}")

        # Calculate optimal parallelization
        grus_per_job = self.estimate_optimal_grus_per_job(total_grus)
        self.logger.info(f"GRUs per job: {grus_per_job}")

        # Pre-execution setup
        if not self._pre_execution():
            return None

        # Create and submit SLURM script
        script_content = self.create_gru_parallel_script(
            model_exe=self.model_exe,
            file_manager=self.file_manager,
            log_dir=self.get_log_path(),
            total_grus=total_grus,
            grus_per_job=grus_per_job,
            job_name=f"SUMMA-{self.domain_name}",
        )

        script_path = self.project_dir / 'run_summa_parallel.sh'
        script_path.write_text(script_content, encoding='utf-8')
        script_path.chmod(0o755)

        # Backup settings if requested
        backup = self._get_config_value(
            lambda: self.config.model.summa.backup_settings, default='no'
        )
        if backup == 'yes':
            self.backup_settings(self.settings_path)

        # Submit and optionally wait
        result = self.submit_slurm_job(
            script_path=script_path,
            wait=self._get_config_value(
                lambda: self.config.model.summa.monitor_slurm_job, default=True
            ),
            max_wait_time=3600
        )

        if result.success:
            return self._merge_parallel_outputs()
        else:
            self.logger.error(f"Parallel SUMMA failed: {result.error_message}")
            return None

    def _run_parallel_summa_local(self) -> Optional[Path]:
        """Run local SUMMA subprocess workers over GRU splits."""
        self.logger.info("Starting local parallel SUMMA run")

        if not self._pre_execution():
            return None

        timeout = self._get_config_value(
            lambda: self.config.model.summa.timeout, default=7200
        )
        cpus_per_task = self._get_config_value(
            lambda: self.config.model.summa.cpus_per_task, default=32
        )
        debug_info: Dict[str, List[str]] = {'errors': []}

        success = run_summa_gru_parallel(
            summa_exe=self.model_exe,
            file_manager=self.file_manager,
            summa_dir=self.output_dir,
            settings_dir=self.settings_path,
            num_parallel=int(cpus_per_task),
            logger=self.logger,
            debug_info=debug_info,
            timeout=int(timeout),
            env=self._get_environment(),
        )

        if not success:
            self.logger.error("Local parallel SUMMA failed")
            return None

        from symfluence.core.exceptions import ValidationError

        from .output_validation import validate_output_coverage
        try:
            validate_output_coverage(self.file_manager, self.output_dir)
        except (ValidationError, ValueError) as exc:
            self.logger.error(str(exc))
            return None

        self.logger.info("Local parallel SUMMA completed")
        return self.output_dir

    def _count_grus(self) -> int:
        """Count total GRUs from catchment shapefile."""
        # Resolve legacy and organized catchment layouts
        shapefile = self._get_catchment_file_path()

        try:
            gdf = gpd.read_file(shapefile)
            gru_col = self._get_config_value(
                lambda: self.config.paths.catchment_gruid, default='GRU_ID'
            )
            return len(gdf[gru_col].unique())
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error counting GRUs: {e}")
            raise

    def _merge_parallel_outputs(self) -> Optional[Path]:
        """
        Merge parallel SUMMA outputs into unified files.

        Creates:
            - {experiment_id}_timestep.nc
            - {experiment_id}_day.nc
        """
        self.logger.info("Merging parallel SUMMA outputs")

        experiment_id = self.experiment_id

        try:
            # Process timestep and daily files
            for pattern, suffix in [
                (f"{experiment_id}_*_timestep.nc", "timestep"),
                (f"{experiment_id}_*_day.nc", "day")
            ]:
                output_file = self.output_dir / f"{experiment_id}_{suffix}.nc"
                self._merge_files(pattern, output_file)

            self.logger.info("SUMMA output merging completed")
            from .output_validation import validate_output_coverage
            validate_output_coverage(self.file_manager, self.output_dir)
            return self.output_dir

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error merging outputs: {e}", exc_info=True)
            return None

    def _merge_files(self, pattern: str, output_file: Path) -> None:
        """Merge files matching pattern into output file."""
        input_files = sorted(self.output_dir.glob(pattern))

        if not input_files:
            self.logger.warning(f"No files matching: {pattern}")
            return

        merged_ds = None
        reference_date = pd.Timestamp('1990-01-01')

        for src_file in input_files:
            try:
                ds = xr.open_dataset(src_file)

                # Convert time to seconds since reference
                time_values = pd.to_datetime(ds.time.values)
                seconds_since_ref = (time_values - reference_date).total_seconds()
                ds = ds.assign_coords(time=seconds_since_ref)
                ds.time.attrs = {
                    'units': 'seconds since 1990-1-1 0:0:0.0 -0:00',
                    'calendar': 'standard',
                }

                if merged_ds is None:
                    merged_ds = ds
                else:
                    merged_ds = xr.merge([merged_ds, ds])

                ds.close()

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Error processing {src_file}: {e}", exc_info=True)

        if merged_ds is not None:
            encoding = {'time': {'dtype': 'double', '_FillValue': None}}
            for var in merged_ds.data_vars:
                encoding[str(var)] = {'_FillValue': None}

            merged_ds.to_netcdf(
                output_file,
                encoding=encoding,
                unlimited_dims=['time'],
                format='NETCDF4'
            )
            merged_ds.close()
            self.logger.info(f"Created: {output_file}")

    def _convert_lumped_for_routing(self) -> None:
        """Convert lumped SUMMA output for distributed routing."""
        timestep_file = self.output_dir / f"{self.experiment_id}_timestep.nc"

        if not timestep_file.exists():
            self.logger.warning(f"Timestep file not found: {timestep_file}")
            return

        # Use SpatialOrchestrator's conversion
        routing_config = self.spatial_config.routing
        self.convert_to_routing_format(
            timestep_file,
            routing_config=routing_config
        )

    def run_summa_point(self) -> Optional[Path]:
        """
        Run SUMMA in point simulation mode.

        Executes SUMMA for multiple point simulations based on file manager lists.
        """
        self.logger.info("Starting SUMMA point simulations")

        fm_ic_list_path = self.settings_path / 'list_fileManager_IC.txt'
        fm_list_path = self.settings_path / 'list_fileManager.txt'

        # Verify files exist
        self.verify_required_files(
            [fm_ic_list_path, fm_list_path],
            "SUMMA point simulations"
        )

        # Read file manager lists
        with open(fm_ic_list_path, encoding='utf-8') as f:
            fm_ic_list = [line.strip() for line in f if line.strip()]
        with open(fm_list_path, encoding='utf-8') as f:
            fm_list = [line.strip() for line in f if line.strip()]

        # Create output directory
        output_path = self.project_dir / 'simulations' / self.experiment_id / 'SUMMA'
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each site
        for i, (ic_fm, main_fm) in enumerate(zip(fm_ic_list, fm_list)):
            # Extract site name from file manager filename
            # For simple point simulations, use domain name if naming convention not met
            fm_stem = Path(ic_fm).stem
            fm_parts = fm_stem.split('_')
            if len(fm_parts) > 1:
                site_name = fm_parts[1]
            else:
                # Simple point simulation - use domain name
                site_name = self.domain_name
            self.logger.info(f"Processing site {i+1}/{len(fm_list)}: {site_name}")

            site_output = output_path / site_name
            site_output.mkdir(parents=True, exist_ok=True)
            log_path = site_output / "logs"
            log_path.mkdir(parents=True, exist_ok=True)

            # Run IC simulation
            ic_log_file = log_path / f"{site_name}_IC.log"
            self.logger.debug(f"Writing IC logs to: {ic_log_file}")

            ic_result = self.execute_subprocess(
                command=[str(self.model_exe), '-m', ic_fm, '-r', 'e'],
                log_file=ic_log_file,
                check=False,
                timeout=300  # 5 minute timeout per site
            )

            if not ic_result.success:
                self.logger.error(f"IC simulation failed for {site_name}. See {ic_log_file}")
                if ic_result.error_message:
                    self.logger.error(f"Error: {ic_result.error_message}")
                continue

            # Copy restart file
            restart_files = list(site_output.glob("*restart*"))
            if restart_files:
                import shutil
                newest = max(restart_files, key=lambda p: p.stat().st_mtime)
                shutil.copy(newest, Path(ic_fm).parent / "warm_state.nc")

            # Run main simulation
            main_log_file = log_path / f"{site_name}_main.log"
            self.logger.debug(f"Writing main logs to: {main_log_file}")

            main_result = self.execute_subprocess(
                command=[str(self.model_exe), '-m', main_fm],
                log_file=main_log_file,
                check=False,
                timeout=300  # 5 minute timeout per site
            )

            if main_result.success:
                self.logger.info(f"Completed site: {site_name}")
            else:
                self.logger.error(f"Main simulation failed for {site_name}. See {main_log_file}")
                if main_result.error_message:
                    self.logger.error(f"Error: {main_result.error_message}")

        self.logger.info(f"Completed {len(fm_list)} point simulations")
        return output_path
