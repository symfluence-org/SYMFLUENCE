# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Worker

Worker implementation for SWAT model optimization.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.calibration.workers.base_worker import run_worker_task
from symfluence.core.calibration.workers.streamflow_worker import StreamflowWorker
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_executable
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

from ..parameters import PARAM_CHANGE_METHOD, PARAM_FILE_MAP


def _fsync_directory(target_dir: Path) -> None:
    """Best-effort flush of a directory's entries to disk.

    Opening a directory with ``os.open(..., O_RDONLY)`` and calling
    ``os.fsync`` is a POSIX-only technique used to guarantee that files
    freshly copied into ``target_dir`` are durable before the Fortran
    executable reads them (this prevents a SIGBUS on arm64).

    On Windows a directory cannot be opened this way — ``os.open`` raises
    ``PermissionError`` ([Errno 13]) on a directory path — and the durability
    concern does not apply, so this is a no-op there.  Any other platform
    that cannot fsync a directory (e.g. some network filesystems) is also
    tolerated silently.
    """
    if os.name == 'nt':
        return
    try:
        fd = os.open(str(target_dir), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@R.workers.add('SWAT')
class SWATWorker(StreamflowWorker):
    """
    Worker for SWAT model calibration.

    Handles parameter application to SWAT text files, SWAT execution,
    and metric calculation from output.rch.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SWAT worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to SWAT text files in a worker TxtInOut directory.

        Copies fresh input files from the original TxtInOut location,
        then applies parameter changes using the r__, v__, or a__ methods.

        Args:
            params: Parameter values to apply
            settings_dir: Worker-specific TxtInOut directory
            **kwargs: Additional arguments (config)

        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Applying SWAT parameters to {settings_dir}")

            # Copy fresh files from settings + forcing into worker dir
            config = kwargs.get('config', self.config) or {}
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f'domain_{domain_name}'
            swat_settings = project_dir / 'settings' / 'SWAT'
            swat_forcing = project_dir / 'data' / 'forcing' / 'SWAT_input'
            # Legacy fallback
            if not swat_settings.exists():
                swat_forcing_legacy = project_dir / 'forcing' / 'SWAT_input'
                if swat_forcing_legacy.exists():
                    swat_forcing = swat_forcing_legacy

            has_source = swat_settings.exists() or swat_forcing.exists()
            if has_source and (not settings_dir.exists() or settings_dir.resolve() != swat_settings.resolve()):
                settings_dir.mkdir(parents=True, exist_ok=True)
                for src_dir in (swat_settings, swat_forcing):
                    if src_dir.exists():
                        for f in src_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(f"Assembled SWAT TxtInOut in {settings_dir}")
            elif not settings_dir.exists():
                self.logger.error(
                    f"SWAT settings not found: {swat_settings} "
                    f"and forcing not found: {swat_forcing}"
                )
                return False

            # Apply parameter changes
            return self._update_swat_files(settings_dir, params)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying SWAT parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_swat_files(
        self,
        txtinout_dir: Path,
        params: Dict[str, float]
    ) -> bool:
        """
        Update SWAT text files with parameter values.

        Groups parameters by file extension and applies changes.

        Args:
            txtinout_dir: Path to TxtInOut directory
            params: Parameters to apply

        Returns:
            True if successful
        """
        try:
            # Group parameters by file extension
            params_by_ext: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                if param_name not in PARAM_FILE_MAP:
                    continue
                ext = PARAM_FILE_MAP[param_name]
                if ext not in params_by_ext:
                    params_by_ext[ext] = {}
                params_by_ext[ext][param_name] = value

            # Apply to each file type
            for ext, ext_params in params_by_ext.items():
                target_files = list(txtinout_dir.glob(f'*{ext}'))
                for target_file in target_files:
                    if ext == '.sol':
                        self._update_sol_file(target_file, ext_params)
                    else:
                        self._update_single_file(target_file, ext_params)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating SWAT files: {e}", exc_info=True)
            return False

    def _update_single_file(
        self,
        file_path: Path,
        params: Dict[str, float]
    ) -> None:
        """
        Update a single SWAT text file with parameter changes.

        Args:
            file_path: Path to the SWAT text file
            params: Parameters to update in this file
        """
        ext = file_path.suffix
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            modified = False
            for param_name, value in params.items():
                if PARAM_FILE_MAP.get(param_name) != ext:
                    continue

                # Use word-boundary matching to avoid partial matches
                # (e.g. '| ALPHA_BF' must not match '| ALPHA_BF_D')
                if self._line_matches_param(line, param_name):
                    method = PARAM_CHANGE_METHOD.get(param_name, 'v__')
                    new_line = self._apply_change_to_line(line, param_name, value, method)
                    new_lines.append(new_line)
                    modified = True
                    break

            if not modified:
                new_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    def _update_sol_file(
        self,
        file_path: Path,
        params: Dict[str, float]
    ) -> None:
        """
        Update a SWAT .sol file with parameter changes.

        .sol files use a colon-delimited format with multiple values per line
        (one per soil layer), e.g.:
            SOL_AWC(mm/mm)           :        0.20        0.14
        This differs from the pipe-delimited format of other SWAT input files.
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            modified = False
            for param_name, value in params.items():
                if PARAM_FILE_MAP.get(param_name) != '.sol':
                    continue
                # Match lines like "SOL_AWC(mm/mm)  :  0.20  0.14"
                if param_name in line and ':' in line:
                    method = PARAM_CHANGE_METHOD.get(param_name, 'r__')
                    new_line = self._apply_sol_change(line, value, method)
                    new_lines.append(new_line)
                    modified = True
                    break

            if not modified:
                new_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    def _apply_sol_change(
        self,
        line: str,
        value: float,
        method: str
    ) -> str:
        """
        Apply a parameter change to all numeric values in a .sol file line.

        Handles multi-value lines (one value per soil layer) by applying
        the change method to every number after the colon.
        """
        colon_idx = line.index(':')
        prefix = line[:colon_idx + 1]
        data_part = line[colon_idx + 1:]

        def _replace_value(match):
            whitespace = match.group(1)
            original_str = match.group(2)
            try:
                original_value = float(original_str)
            except ValueError:
                return match.group(0)

            if method == 'r__':
                new_value = original_value * (1.0 + value)
            elif method == 'v__':
                new_value = value
            elif method == 'a__':
                new_value = original_value + value
            else:
                new_value = value

            orig_width = len(original_str)
            if '.' in original_str:
                decimal_places = len(original_str.split('.')[-1])
                new_value_str = f"{new_value:.{decimal_places}f}"
            else:
                new_value_str = str(int(round(new_value)))

            # Preserve total field width: steal from / pad whitespace
            overflow = len(new_value_str) - orig_width
            if overflow > 0 and len(whitespace) > overflow:
                whitespace = whitespace[overflow:]
            elif overflow < 0:
                whitespace = ' ' * (-overflow) + whitespace

            return f"{whitespace}{new_value_str}"

        new_data = re.sub(
            r'(\s+)([-+]?\d+\.?\d*)',
            _replace_value,
            data_part
        )

        result = prefix + new_data
        if not result.endswith('\n'):
            result += '\n'
        return result

    @staticmethod
    def _line_matches_param(line: str, param_name: str) -> bool:
        """Check if a SWAT file line contains the given parameter marker.

        Uses word-boundary logic so that e.g. 'ALPHA_BF' does not match
        a line containing 'ALPHA_BF_D'.
        """
        for sep in ('| ', '|'):
            marker = f'{sep}{param_name}'
            idx = line.find(marker)
            if idx < 0:
                continue
            # Check that the character after the param name is NOT
            # alphanumeric or underscore (i.e. it's a word boundary).
            end_pos = idx + len(marker)
            if end_pos >= len(line) or not (line[end_pos].isalnum() or line[end_pos] == '_'):
                return True
        return False

    def _apply_change_to_line(
        self,
        line: str,
        param_name: str,
        value: float,
        method: str
    ) -> str:
        """
        Apply a parameter change to a SWAT text file line.

        Args:
            line: Original file line
            param_name: Parameter name
            value: New value or change amount
            method: Change method ('r__', 'v__', or 'a__')

        Returns:
            Modified line string
        """
        # Preserve trailing newline so SWAT file format is not corrupted
        line_stripped = line.rstrip('\n\r')
        trailing = line[len(line_stripped):]

        match = re.match(
            r'^(\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(.*)', line_stripped
        )
        if not match:
            match = re.match(r'^(\s*)([-+]?\d+)(.*)', line_stripped)

        if not match:
            return line

        prefix = match.group(1)
        original_str = match.group(2)
        suffix = match.group(3)

        try:
            original_value = float(original_str)
        except ValueError:
            return line

        # Apply change method
        if method == 'r__':
            new_value = original_value * (1.0 + value)
        elif method == 'v__':
            new_value = value
        elif method == 'a__':
            new_value = original_value + value
        else:
            new_value = value

        # Format the new value, preserving total line width.
        # If the formatted value is wider than the original, steal
        # characters from the leading whitespace (prefix) so that the
        # pipe-delimited suffix stays in the same column.
        orig_width = len(original_str)
        if '.' in original_str:
            decimal_places = len(original_str.split('.')[-1])
            new_value_str = f"{new_value:.{decimal_places}f}"
        else:
            new_value_str = str(int(round(new_value)))

        overflow = len(new_value_str) - orig_width
        if overflow > 0 and len(prefix) >= overflow:
            prefix = prefix[overflow:]
        elif overflow < 0:
            prefix = ' ' * (-overflow) + prefix

        result = f"{prefix}{new_value_str}{suffix}{trailing}"
        # Safety: ensure line always ends with newline to prevent line merging
        if not result.endswith('\n'):
            result += '\n'
        return result

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run SWAT model.

        SWAT is executed from within the TxtInOut directory. The executable
        reads file.cio and produces output.rch.

        The SWAT2022 arm64 binary suffers from non-deterministic behaviour
        caused by uninitialised Fortran variables.  A single invocation may
        produce all-zero output even with valid inputs.  To mitigate this
        we retry up to ``max_retries`` times, checking that output.rch
        contains at least one non-zero FLOW_OUT value before accepting the
        run.

        Args:
            config: Configuration dictionary
            settings_dir: TxtInOut directory (SWAT runs from here)
            output_dir: Output directory (unused, SWAT writes to TxtInOut)
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        max_retries = int(config.get('SWAT_MAX_RETRIES', 3))

        try:
            # Use sim_dir as working directory if provided
            work_dir = Path(kwargs.get('sim_dir', settings_dir))

            # Copy files from settings_dir to work_dir if they differ.
            # apply_parameters() modifies files in settings_dir each
            # iteration, so we must ALWAYS overwrite to propagate parameter changes.
            if work_dir.resolve() != settings_dir.resolve() and settings_dir.exists():
                work_dir.mkdir(parents=True, exist_ok=True)
                for f in settings_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, work_dir / f.name)
                # Also copy forcing files if not already present in settings_dir
                domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
                data_dir_path = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
                project_dir = data_dir_path / f'domain_{domain_name}'
                swat_forcing = project_dir / 'data' / 'forcing' / 'SWAT_input'
                if not swat_forcing.exists():
                    swat_forcing = project_dir / 'forcing' / 'SWAT_input'
                if swat_forcing.exists():
                    for f in swat_forcing.iterdir():
                        if f.is_file():
                            dest = work_dir / f.name
                            if not dest.exists():
                                shutil.copy2(f, dest)

            # Get executable (check once before retry loop)
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            swat_exe = self._get_swat_executable(config, data_dir)

            if not swat_exe.exists():
                self.logger.error(f"SWAT executable not found: {swat_exe}")
                return False

            # Verify file.cio exists
            file_cio = work_dir / 'file.cio'
            if not file_cio.exists():
                self.logger.error(f"file.cio not found in {work_dir}")
                return False

            # Build command - SWAT runs from TxtInOut
            cmd = [str(swat_exe)]

            # Set single-threaded environment for Fortran runtime
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
            })

            timeout = resolve_config_value(config, lambda: config.model.swat.timeout, dict_key='SWAT_TIMEOUT')
            stdout_file = work_dir / 'swat_stdout.log'
            stderr_file = work_dir / 'swat_stderr.log'

            for attempt in range(1, max_retries + 1):
                # Clean up stale output
                self._cleanup_stale_output(work_dir)

                # Fsync the working directory so all copied files are flushed
                # before the Fortran executable reads them (prevents SIGBUS on arm64)
                _fsync_directory(work_dir)

                try:
                    with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                         open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                        result = run_subprocess(
                            cmd,
                            cwd=str(work_dir),
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_f,
                            stderr=stderr_f,
                            timeout=timeout
                        )
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"SWAT timed out after {timeout}s (attempt {attempt}/{max_retries})")
                    continue

                if result.returncode != 0:
                    self.logger.debug(f"SWAT returned {result.returncode} (attempt {attempt}/{max_retries})")
                    continue

                # Verify output.rch exists and has non-zero flow
                output_rch = work_dir / 'output.rch'
                if not output_rch.exists() or output_rch.stat().st_size == 0:
                    self.logger.debug(f"output.rch missing or empty (attempt {attempt}/{max_retries})")
                    continue

                if self._has_nonzero_flow(output_rch):
                    if attempt > 1:
                        self.logger.debug(f"SWAT produced valid output on attempt {attempt}")
                    return True

                self.logger.debug(
                    f"SWAT produced all-zero flow (attempt {attempt}/{max_retries})"
                )

            # All retries exhausted
            self._last_error = f"SWAT produced no valid flow after {max_retries} attempts"
            self.logger.warning(self._last_error)
            return False

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running SWAT: {e}", exc_info=True)
            return False

    def _has_nonzero_flow(self, output_rch: Path, threshold: float = 1e-6) -> bool:
        """Quick check whether output.rch contains any non-zero FLOW_OUT."""
        try:
            with open(output_rch, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.split()
                    if not parts or parts[0].upper() != 'REACH':
                        continue
                    if len(parts) > 6:
                        flow_out = float(parts[6])
                        if abs(flow_out) > threshold:
                            return True
            return False
        except Exception:  # noqa: BLE001 — calibration resilience
            return False

    def _get_swat_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get SWAT executable path."""
        install_path = resolve_config_value(
            config,
            lambda: config.model.swat.install_path,
            dict_key='SWAT_INSTALL_PATH',
        )
        exe_name = resolve_config_value(config, lambda: config.model.swat.exe, dict_key='SWAT_EXE')

        return resolve_install_executable(install_path, data_dir, 'installs/swat/bin', exe_name)

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale SWAT output files."""
        for pattern in ['output.*', '*.log']:
            for file_path in output_dir.glob(pattern):
                if file_path.name.startswith('output.'):
                    try:
                        file_path.unlink()
                    except (OSError, IOError):
                        pass

    def read_simulated_streamflow(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Optional[pd.Series]:
        """Daily outlet discharge (m3/s) from this trial's SWAT output.rch."""
        sim_dir = Path(kwargs.get('sim_dir', output_dir))
        output_rch = sim_dir / 'output.rch'

        if not output_rch.exists():
            self.logger.error(f"output.rch not found in {sim_dir}")
            raise ModelExecutionError('No output.rch')

        # Parse output.rch for outlet reach streamflow
        flow_data = self._parse_output_rch(output_rch, reach_id=1)

        if flow_data is None or len(flow_data) == 0:
            raise ModelExecutionError('No flow data parsed')

        # Get simulation start date
        try:
            start_str = (
                resolve_config_value(config, lambda: config.domain.time_start, None, 'EXPERIMENT_TIME_START')
                or config.get('TIME_START')
            )
            if start_str is None and hasattr(config, 'domain'):
                start_str = config.domain.time_start
            start_date = pd.to_datetime(start_str).normalize() if start_str else pd.to_datetime('2000-01-01')
        except (AttributeError, TypeError):
            start_date = pd.to_datetime('2000-01-01')

        # Build time series
        # SWAT output.rch already starts AFTER the NYSKIP warmup period
        # (file.cio NYSKIP is set to warmup_years by the preprocessor).
        # We must NOT skip warmup again; just offset the start date.
        warmup_years = int(resolve_config_value(config, lambda: config.model.swat.warmup_years, dict_key='SWAT_WARMUP_YEARS'))
        output_start_date = start_date + pd.DateOffset(years=warmup_years)
        dates = pd.date_range(start=output_start_date, periods=len(flow_data), freq='D')
        sim_series = pd.Series(flow_data, index=dates)
        self.logger.debug(
            f"SWAT output dates: {sim_series.index[0]} to {sim_series.index[-1]} "
            f"({len(sim_series)} days, warmup={warmup_years}yr already skipped by NYSKIP)"
        )
        return sim_series

    def _parse_output_rch(
        self,
        output_rch: Path,
        reach_id: int = 1
    ) -> Optional[np.ndarray]:
        """
        Parse SWAT output.rch and extract FLOW_OUTcms for a given reach.

        Args:
            output_rch: Path to output.rch
            reach_id: Reach number to extract

        Returns:
            numpy array of flow values, or None
        """
        try:
            with open(output_rch, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Find header end
            header_end = 0
            for i, line in enumerate(lines):
                if 'RCH' in line.upper() and ('FLOW' in line.upper() or 'MON' in line.upper()):
                    header_end = i + 1
                    break
            if header_end == 0:
                header_end = 9

            # Parse data - handle both formats:
            #   "1  0  1  0.00  0.00  0.00 ..."   (reach number at col 0, flow at col 5)
            #   "REACH  1  0  1  0.00  0.00  0.00 ..."  (REACH prefix, reach at col 1, flow at col 6)
            flow_values = []
            for line in lines[header_end:]:
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    # Detect REACH prefix
                    if parts[0].upper() == 'REACH':
                        rch = int(parts[1])
                        flow_col = 6  # FLOW_OUTcms after REACH prefix
                    else:
                        rch = int(parts[0])
                        flow_col = 5  # FLOW_OUTcms without prefix
                    if rch == reach_id and len(parts) > flow_col:
                        flow_out = float(parts[flow_col])
                        flow_values.append(max(0.0, flow_out))
                except (ValueError, IndexError):
                    continue

            if not flow_values:
                return None

            return np.array(flow_values)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error parsing output.rch: {e}", exc_info=True)
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_swat_parameters_worker(task_data)


def _evaluate_swat_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        SWATWorker,
        task_data,
        model_name='SWAT',
    )
