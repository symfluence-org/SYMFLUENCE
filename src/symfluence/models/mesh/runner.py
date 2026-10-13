# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH model runner.

Handles MESH model execution, state management, and output processing.
Refactored to use the Unified Model Execution Framework.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R
from symfluence.models.mesh.glacier_runtime import check_finite_output, check_finite_runtime


@R.runners.add('MESH', runner_method='run_mesh')
class MESHRunner(BaseModelRunner):  # type: ignore[misc]
    """
    Runner class for the MESH model.
    Handles model execution, state management, and output processing.

    Uses the Unified Model Execution Framework for subprocess execution.

    Attributes:

        config (Dict[str, Any]): Configuration settings for MESH model
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    MODEL_NAME = "MESH"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the MESH model runner.

        Sets up MESH-specific paths including executable location, forcing
        directory, and catchment shapefile paths.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                MESH installation path, domain settings, and execution parameters.
            logger: Logger instance for status messages and debugging.
            reporting_manager: Optional reporting manager for experiment tracking.
        """
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        self.mesh_exe = self.get_model_executable(
            install_path_key='MESH_INSTALL_PATH',
            default_install_subpath='installs/mesh/bin',
            exe_name_key='MESH_EXE',
            default_exe_name='mesh.exe',
            typed_exe_accessor=lambda: self.config.model.mesh.exe if self.config.model.mesh else None
        )

        # Catchment paths (use backward-compatible path resolution)
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

        # MESH-specific paths
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_dir = self.project_forcing_dir / 'MESH_input'

        # Initialize forcing_mesh_path to forcing_dir (can be overridden for parallel execution)
        self.forcing_mesh_path = self.forcing_dir

    def _get_output_dir(self) -> Path:
        """MESH output directory."""
        return self.get_experiment_output_dir()

    def set_process_directories(self, forcing_dir: Path, output_dir: Path) -> None:
        """
        Set process-specific directories for parallel execution.

        Args:
            forcing_dir: Process-specific forcing directory
            output_dir: Process-specific output directory
        """
        self.forcing_mesh_path = forcing_dir
        self.output_dir = output_dir
        self.logger.debug(f"Set MESH paths: forcing={forcing_dir}, output={output_dir}")

    def run_mesh(self) -> Optional[Path]:
        """
        Run the MESH model simulation.

        Executes MESH in the forcing directory, verifies outputs, and cleans
        up temporary files on success. MESH requires execution from its input
        directory due to relative path assumptions in the model.

        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise.

        Note:
            MESH executable is temporarily copied to the forcing directory for
            execution and removed after successful completion.
        """
        self.logger.debug("Starting MESH model run")

        with symfluence_error_handler(
            "MESH model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            from symfluence.models.mesh.preprocessing.frozen_soil import validate_frozen_soil
            # Clear stale output files from output_dir to prevent a crashed
            # iteration from finding valid files left by the previous iteration
            # on the same process (output_dir is reused across iterations).
            self._clear_stale_outputs(self.output_dir)
            self._clear_stale_outputs(self.forcing_mesh_path)
            self._clear_stale_outputs(self.forcing_mesh_path / 'results')
            validate_frozen_soil(self.forcing_mesh_path)

            from symfluence.models.mesh.preprocessing.spatial_units import prepare_spatial_baseflow
            prepare_spatial_baseflow(self.forcing_mesh_path)
            if self._get_config_value(
                lambda: self.config.model.mesh.spatial_diagnostics,
                default=False, dict_key='MESH_SPATIAL_DIAGNOSTICS',
            ):
                from symfluence.models.mesh.preprocessing.spatial_units import prepare_spatial_diagnostics
                prepare_spatial_diagnostics(self.forcing_mesh_path, cryosphere_endpoints=self._get_config_value(
                    lambda: self.config.model.mesh.cryosphere_constraints,
                    default=False, dict_key='MESH_CRYOSPHERE_CONSTRAINTS'))
            # Create run command
            finite_ice = check_finite_runtime(
                Path(self.mesh_exe), self.forcing_mesh_path,
                self._get_config_value(
                    lambda: self.config.model.mesh.glacier_mode,
                    default='none', dict_key='MESH_GLACIER_MODE'),
            )
            cmd = self._create_run_command()

            # Set up logging
            log_dir = self.get_log_path()
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'mesh_run_{current_time}.log'

            # Execute MESH (it must run in the forcing directory)
            self.logger.debug(f"Executing command: {' '.join(map(str, cmd))}")

            # Prepare environment delta (base handles os.environ copy + conda augmentation)
            mesh_env: Dict[str, str] = {}
            if sys.platform == 'darwin':
                brew_lib = "/opt/homebrew/lib"
                current = os.environ.get("DYLD_LIBRARY_PATH", "")
                if brew_lib not in current:
                    mesh_env['DYLD_LIBRARY_PATH'] = f"{brew_lib}:{current}" if current else brew_lib

            result = self.execute_subprocess(
                cmd,
                log_file,
                cwd=self.forcing_mesh_path,
                env=mesh_env if mesh_env else None,
                check=False,  # Don't raise on non-zero exit, we'll handle it
                success_message="MESH simulation completed successfully",
                success_log_level=logging.DEBUG,
            )

            if finite_ice:
                if not result.success:
                    raise ModelExecutionError('Finite glacier run failed; partial output cannot be scored')
                check_finite_output(self.forcing_mesh_path)
            from symfluence.models.mesh.preprocessing.spatial_units import check_spatial_output
            check_spatial_output(self.forcing_mesh_path)
            routing_kernel = self._get_config_value(
                lambda: self.config.model.mesh.spatial_routing_kernel,
                default=None, dict_key='MESH_SPATIAL_ROUTING_KERNEL',
            )
            if routing_kernel:
                from symfluence.models.mesh.spatial_routing import route_spatial_runoff
                route_spatial_runoff(self.forcing_mesh_path, routing_kernel)
            outputs_ok = self._verify_outputs()
            # Check execution success (accept non-zero if outputs are valid)
            if outputs_ok:
                if not result.success:
                    self.logger.warning(
                        f"MESH exited with code {result.return_code} but required outputs were found; treating as success."
                    )
                # Copy outputs from forcing directory to output directory
                self._copy_outputs()

                # Clean up copied executable only on success
                mesh_exe_in_forcing = self.forcing_mesh_path / self.mesh_exe.name
                if mesh_exe_in_forcing.exists() and mesh_exe_in_forcing.is_file():
                    mesh_exe_in_forcing.unlink()
                return self.output_dir
            else:
                reason = getattr(self, '_last_verify_reason', None)
                if reason:
                    self.logger.error(f"MESH produced no usable output — {reason}")
                self.logger.debug(f"MESH simulation failed with code {result.return_code}")
                # Log the end of the log file for easier debugging
                if log_file.exists():
                     with open(log_file, 'r', encoding='utf-8', errors='replace') as f:  # Handle non-UTF-8 characters
                         lines = f.readlines()
                         last_lines = lines[-20:]
                         self.logger.debug("Last 20 lines of model log:")
                         for line in last_lines:
                             self.logger.debug(f"  {line.strip()}")
                raise ModelExecutionError(reason or 'MESH produced no complete usable output')

    def _create_run_command(self) -> List[str]:
        """
        Create MESH execution command.

        Copies the MESH executable to the forcing directory (required by MESH),
        ensures it has execute permissions, and creates the results subdirectory
        that MESH expects for output.

        Returns:
            List[str]: Command arguments for subprocess execution.
        """
        # Copy mesh executable to forcing path (only if missing or outdated)
        mesh_exe_dest = self.forcing_mesh_path / self.mesh_exe.name
        if (self.forcing_mesh_path / 'MESH_glacier_storage.ini').exists() or not mesh_exe_dest.exists() or (
            self.mesh_exe.stat().st_mtime > mesh_exe_dest.stat().st_mtime
        ):
            shutil.copy2(self.mesh_exe, mesh_exe_dest)
            mesh_exe_dest.chmod(0o755)

        # Create results directory that MESH expects
        results_dir = self.forcing_mesh_path / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale output files to prevent _verify_outputs from matching
        # previous iteration's data if the current run fails silently
        for stale_pattern in ['MESH_output_*.csv', 'MESH_output_*.txt',
                              'Basin_average_water_balance.csv',
                              'GRU_water_balance.csv']:
            for stale_file in self.forcing_mesh_path.glob(stale_pattern):
                stale_file.unlink()
            for stale_file in results_dir.glob(stale_pattern):
                stale_file.unlink()

        self.logger.debug(f"Created MESH results directory: {results_dir}")

        cmd = [
            f'./{self.mesh_exe.name}'
        ]
        return cmd

    def _is_lumped_mode(self) -> bool:
        """Check if running in lumped (single-cell) mode.

        Detects single-cell domains regardless of RUNMODE (noroute or run_def).
        In lumped mode, Basin_average_water_balance.csv is the primary output
        for metric calculation (avoids self-referential routing artifacts in
        MESH_output_streamflow.csv for single-cell run_def domains).
        """
        # Check drainage database for single cell
        ddb_path = self.forcing_mesh_path / 'MESH_drainage_database.nc'
        if ddb_path.exists():
            try:
                import xarray as xr
                with xr.open_dataset(ddb_path) as ds:
                    for dim in ['subbasin', 'n', 'N']:
                        if dim in ds.sizes and ds.sizes[dim] == 1:
                            return True
            except Exception:  # noqa: BLE001 — model execution resilience
                pass

        # Fallback: check RUNMODE for noroute (legacy)
        run_options = self.forcing_mesh_path / 'MESH_input_run_options.ini'
        if run_options.exists():
            try:
                with open(run_options, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import re
                    if re.search(r'RUNMODE\s*[:=]?\s*noroute', content, re.IGNORECASE):
                        return True
            except Exception:  # noqa: BLE001 — model execution resilience
                pass
        return False

    def _verify_outputs(self) -> bool:
        """
        Verify MESH output files exist.

        Checks for required output files in both the output directory and
        forcing directory (MESH writes outputs to its working directory).
        In lumped (single-cell) mode, accepts water balance output
        (primary) or streamflow CSV (secondary).

        Returns:
            bool: True if all required outputs found, False otherwise.
        """
        is_lumped = self._is_lumped_mode()
        expected_days = self._get_expected_days()
        # Records why verification failed so callers (run_mesh / final evaluation)
        # can report a specific cause instead of a bare "MESH run failed".
        self._last_verify_reason: Optional[str] = None
        if is_lumped:
            # Lumped mode (noroute or run_def): accept water balance or streamflow.
            # Basin_average is preferred (avoids self-referential routing artifacts).
            output_candidates = [
                'Basin_average_water_balance.csv',
                'GRU_water_balance.csv',
                'MESH_output_streamflow.csv',
            ]
        else:
            # Routed (multi-cell) mode. Prefer a per-gauge routed streamflow
            # file when MESH writes one, but also accept the basin-average water
            # balance: it is always produced (BASINAVGWBFILEFLAG) and, with the
            # routing-aware extractor, its RFF is the total routed runoff
            # (OVRFLW + LATFLW + LKG) at the basin outlet — a valid daily
            # streamflow. MESH's per-gauge CSV requires the gauge to resolve to
            # a grid cell, which is fragile on subbasin ('nc_subbasin') domains,
            # so it must not be the sole accepted output.
            output_candidates = [
                'MESH_output_streamflow.csv',
                'MESH_output_streamflow_ts.csv',
                'MESH_streamflow_Gauge_*.txt',
                'Basin_average_water_balance.csv',
                'GRU_water_balance.csv',
            ]

        # Check in output directory (may be process-specific during parallel calibration)
        # or fall back to forcing directory (default MESH behavior)
        check_dirs = [self.output_dir, self.forcing_mesh_path, self.forcing_mesh_path / 'results']

        def _candidate_paths(name: str):
            if any(ch in name for ch in ['*', '?', '[']):
                for d in check_dirs:
                    yield from d.glob(name)
            else:
                for d in check_dirs:
                    yield d / name

        def _has_data(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                if path.suffix.lower() == '.csv':
                    data_rows = self._count_csv_rows(path)
                    if data_rows <= 0:
                        return False
                    if expected_days is not None:
                        if not self._has_full_coverage(path, data_rows, expected_days):
                            self._last_verify_reason = (
                                f"output '{path.name}' is truncated ({data_rows} data rows; "
                                f"the run needs ~{expected_days} days of coverage) — the forcing "
                                f"likely ends before the configured simulation end date"
                            )
                            self.logger.warning(
                                f"MESH output coverage check failed for {path}: "
                                f"{self._last_verify_reason}"
                            )
                            return False
                    return True

                # For non-CSV outputs (e.g., gauge text files), just ensure there
                # are non-empty, non-comment lines to avoid false negatives.
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        if line.strip() and not line.lstrip().startswith('#'):
                            return True
                return False
            except OSError:
                return False

        found_any = False
        for output_file in output_candidates:
            for output_path in _candidate_paths(output_file):
                if _has_data(output_path):
                    found_any = True
                    break
            if found_any:
                break

        if not found_any:
            if not self._last_verify_reason:
                checked = ', '.join(str(d) for d in check_dirs)
                self._last_verify_reason = (
                    f"no MESH output file found (looked for {output_candidates} in {checked})"
                )
            self.logger.warning(f"MESH output verification failed: {self._last_verify_reason}")
            return False

        return True

    def _get_expected_days(self) -> Optional[int]:
        """Infer expected simulation length from run options."""
        run_options = self.forcing_mesh_path / 'MESH_input_run_options.ini'
        if not run_options.exists():
            return None

        try:
            with open(run_options, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except OSError:
            return None

        date_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 4 and all(p.isdigit() for p in parts):
                year = int(parts[0])
                jday = int(parts[1])
                hour = int(parts[2])
                minute = int(parts[3])
                if 1 <= jday <= 366 and 0 <= hour <= 23 and minute in (0, 30):
                    date_lines.append((year, jday, hour, minute))

        if len(date_lines) < 2:
            return None

        start = date_lines[-2]
        end = date_lines[-1]
        try:
            from datetime import timedelta
            start_dt = datetime(start[0], 1, 1) + timedelta(days=start[1] - 1, hours=start[2], minutes=start[3])
            end_dt = datetime(end[0], 1, 1) + timedelta(days=end[1] - 1, hours=end[2], minutes=end[3])
        except ValueError:
            return None

        if end_dt < start_dt:
            return None

        # MESH's native stop is exclusive; midnight is not another output day.
        import math
        return math.ceil((end_dt - start_dt).total_seconds() / 86400)

    def _count_csv_rows(self, path: Path) -> int:
        """Count data rows in a CSV file (excluding header)."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line for line in f.readlines() if line.strip()]
            return max(0, len(lines) - 1)
        except OSError:
            return 0

    def _has_full_coverage(self, path: Path, rows: int, expected_days: int) -> bool:
        """Check whether output rows meet expected simulation coverage."""
        name = path.name.lower()
        if 'gru_water_balance' in name:
            expected_rows = expected_days * 24
            return rows >= expected_rows
        # Allow a small tolerance for daily outputs (end-date handling edge case)
        tolerance_days = 1
        try:
            tolerance_days = int(self._get_config_value(lambda: self.config.model.mesh.daily_tolerance_days, default=tolerance_days))
        except (TypeError, ValueError):
            tolerance_days = 1
        return rows >= max(0, expected_days - tolerance_days)

    def _copy_outputs(self) -> None:
        """
        Copy MESH outputs from forcing directory to simulation directory.

        MESH writes outputs to its working directory (forcing_mesh_path).
        This method copies key output files to the standard simulation
        output directory for consistency with other models.

        Copied files:
            - MESH_output_streamflow.csv: Simulated streamflow timeseries
            - MESH_output_echo_print.txt: Model run summary
            - MESH_output_echo_results.txt: Detailed results log
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        outputs_to_copy = [
            'MESH_output_streamflow.csv',
            'MESH_output_echo_print.txt',
            'MESH_output_echo_results.txt',
            'MESH_input_run_options.ini',
            # Lumped mode water balance outputs
            'Basin_average_water_balance.csv',
            'GRU_water_balance.csv',  # Hourly GRU water balance with ROF
            'MESH_glacier_ledger.csv',
            'MESH_glacier_storage.ini',
            'glacier_manifest.json',
            'MESH_spatial_streamflow.csv', 'MESH_routing_tail.csv', 'MESH_routing_balance.json',
        ]
        if (self.forcing_mesh_path / 'spatial_diagnostics.json').exists():
            from symfluence.models.mesh.preprocessing.spatial_units import SPATIAL_DIAGNOSTIC_FIELDS
            outputs_to_copy += ['spatial_units.json', 'spatial_diagnostics.json', 'SNO_H.csv', 'LQWSSNO_H.csv']
            outputs_to_copy += [f'{variable}_D.csv' for variable in SPATIAL_DIAGNOSTIC_FIELDS]

        for out_file in outputs_to_copy:
            # Check both forcing directory and results subdirectory
            src = self.forcing_mesh_path / out_file
            if not src.exists():
                src = self.forcing_mesh_path / 'results' / out_file
            if src.exists():
                shutil.copy2(src, self.output_dir / out_file)

    def _clear_stale_outputs(self, directory: Path) -> None:
        """Clear stale MESH output files from a directory.

        During parallel calibration, output_dir is reused across iterations.
        If a previous iteration succeeded and copied files there, a subsequent
        crashed iteration would find those stale files and be incorrectly
        scored as successful.

        Args:
            directory: Directory to clear stale output files from.
        """
        if not directory or not directory.exists():
            return

        stale_patterns = [
            'MESH_glacier_ledger.csv', 'MESH_glacier_depletion.txt',
            'SNO_H.csv', 'LQWSSNO_H.csv', 'cryosphere_metrics.json',
            'MESH_spatial_streamflow.csv', 'MESH_routing_tail.csv', 'MESH_routing_balance.json',
            'MESH_output_*.csv', 'MESH_output_*.txt',
            'Basin_average_water_balance.csv',
            'GRU_water_balance.csv',
        ]
        from symfluence.models.mesh.preprocessing.spatial_units import SPATIAL_DIAGNOSTIC_FIELDS
        stale_patterns += [f'{variable}_D.csv' for variable in SPATIAL_DIAGNOSTIC_FIELDS]
        for pattern in stale_patterns:
            for stale_file in directory.glob(pattern):
                stale_file.unlink()
