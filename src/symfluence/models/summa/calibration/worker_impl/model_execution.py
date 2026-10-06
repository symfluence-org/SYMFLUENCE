#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Model Execution for SUMMA Workers

This module contains functions for executing SUMMA and mizuRoute
models in worker processes.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

from symfluence.core.logging_utils import log_once
from symfluence.core.profiling import get_system_profiler

from .netcdf_utilities import fix_summa_time_precision


def _cleanup_stale_output_files(output_dir: Path, logger) -> list:
    """Remove stale output files from previous iterations.

    This is critical for calibration: if routing fails, we don't want metrics to be
    calculated from old output files from previous iterations.

    Returns:
        The files that could NOT be removed. A non-empty list means this
        directory is unsafe to reuse: the caller must not run into it, or the
        very substitution this function exists to prevent happens anyway.

    Args:
        output_dir: Directory containing model output files
        logger: Logger instance
    """
    stuck: list = []
    if not output_dir.exists():
        return stuck

    # Patterns to clean up (SUMMA and mizuRoute output files)
    cleanup_patterns = [
        "*.nc",           # All NetCDF files (SUMMA timestep, day, mizuRoute output)
        "*_restart_*.nc", # Restart files
        "runinfo.txt",    # SUMMA run info
    ]

    files_removed = 0
    for pattern in cleanup_patterns:
        for file_path in output_dir.glob(pattern):
            if file_path.is_dir():
                continue
            try:
                file_path.unlink()
                files_removed += 1
            except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
                stuck.append(file_path)
                log_once(logger, logging.WARNING, key=f'summa-stale-file-{file_path}',
                         message=f"Could not remove stale file {file_path}: {e}")

    # Clean up stale GRU-parallel split directories
    for split_dir in output_dir.glob('gru_split_*'):
        if split_dir.is_dir():
            try:
                import shutil
                shutil.rmtree(split_dir, ignore_errors=True)
                files_removed += 1
            except OSError as e:
                log_once(logger, logging.WARNING, key=f'summa-stale-split-{split_dir}',
                         message=f"Could not remove stale split dir {split_dir}: {e}")

    if files_removed > 0:
        logger.debug(f"Cleaned up {files_removed} stale output files/dirs from {output_dir}")

    return stuck

def _deduplicate_output_control(output_control_path: Path, logger):
    """Ensure outputControl.txt doesn't have duplicate variables for the same frequency"""
    if not output_control_path.exists():
        return

    try:
        with open(output_control_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        seen_vars = {} # (var_name, frequency) -> True
        new_lines = []
        changed = False

        for line in lines:
            trimmed = line.strip()
            # Skip comments, empty lines, and lines without frequency separator
            if not trimmed or trimmed.startswith('!') or '|' not in trimmed:
                new_lines.append(line)
                continue

            parts = [p.strip() for p in trimmed.split('|')]
            var_name = parts[0]
            freq = parts[1] if len(parts) > 1 else '1'

            key = (var_name, freq)
            if key in seen_vars:
                log_once(logger, logging.WARNING, key=f'summa-dup-output-{var_name}-{freq}',
                               message=f"Removing duplicate output request: {var_name} | {freq}")
                changed = True
                continue

            seen_vars[key] = True
            new_lines.append(line)

        if changed:
            with open(output_control_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            logger.debug(f"Deduplicated {output_control_path.name}")

    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
        log_once(logger, logging.WARNING, key='summa-dedup-output-control-failed',
                       message=f"Failed to deduplicate output control: {e}")


def _ensure_coupling_output_vars(output_control_path: Path, config, logger) -> None:
    """Guarantee the groundwater-coupling flux is in SUMMA's output control.

    When SUMMA is coupled to an external groundwater model (GROUNDWATER_MODEL,
    e.g. MODFLOW), the coupler reads a recharge flux — scalarSoilDrainage by
    default — from SUMMA's output. A calibration-target rewrite can leave the
    per-run outputControl with only the streamflow variable, so the coupler then
    fails with "Variable 'scalarSoilDrainage' not found in SUMMA output". This
    runs right before each SUMMA execution (every trial and the final
    evaluation) and re-adds the coupling variables if missing — idempotent.
    """
    if not config:
        return
    gw_model = str(config.get('GROUNDWATER_MODEL', '') or '').upper()
    if gw_model in ('', 'NONE'):
        return
    if not output_control_path.exists():
        return
    recharge_var = (config.get('MODFLOW_RECHARGE_VARIABLE')
                    or 'scalarSoilDrainage')
    needed = [v for v in (recharge_var, 'scalarSurfaceRunoff') if v]
    try:
        content = output_control_path.read_text(encoding='utf-8')
        added = []
        for var in needed:
            if var not in content:
                if content and not content.endswith('\n'):
                    content += '\n'
                content += f"{var} | 1\n"
                added.append(var)
        if added:
            output_control_path.write_text(content, encoding='utf-8')
            log_once(
                logger, logging.INFO,
                key='summa-coupling-vars-added',
                message=(
                    f"Ensured groundwater-coupling output vars in "
                    f"{output_control_path.name}: {added}"),
            )
    except OSError as e:
        log_once(logger, logging.WARNING, key='summa-coupling-vars-failed',
                 message=f"Could not ensure coupling output vars: {e}")


def _rewrite_mizuroute_control_for_run(
    control_file: Path,
    summa_dir: Path,
    mizuroute_dir: Path,
    qsim_filename: str
) -> None:
    """Point mizuRoute at the SUMMA output and route output for this run."""
    def normalize_path(path: Path) -> str:
        return str(path).replace("\\", "/").rstrip("/") + "/"

    def update_control_line(line: str, key: str, value: str) -> str:
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        comment = ""
        if "!" in content:
            comment = "!" + "!".join(content.split("!")[1:]).rstrip()
        if comment:
            return f"{key:<24}{value}    {comment}{newline}"
        return f"{key:<24}{value}{newline}"

    replacements = {
        "<input_dir>": normalize_path(summa_dir),
        "<output_dir>": normalize_path(mizuroute_dir),
        "<fname_qsim>": qsim_filename,
    }

    with open(control_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        stripped = line.strip()
        for key, value in replacements.items():
            if stripped.startswith(key):
                updated_lines.append(update_control_line(line, key, value))
                break
        else:
            updated_lines.append(line)

    with open(control_file, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(updated_lines)



def _usable_output_dir(summa_dir: Path, logger) -> Path:
    """Return a directory this evaluation can safely own.

    A wedged predecessor never releases its output NetCDF handle (see
    symfluence.core.process_exec), so the stale file cannot be deleted. Running
    into that directory anyway has two bad outcomes: SUMMA cannot create its
    output and dies with a bare ``STOP 1``, or — worse — metrics get computed
    from the previous iteration's leftovers, which is exactly what
    _cleanup_stale_output_files exists to prevent.

    So when the directory cannot be cleared, this evaluation gets a fresh
    sibling instead. Both the fileManager's outputPath and the metric reader
    derive from this path, so redirecting here moves writer and reader
    together and no glob can straddle two iterations.
    """
    stuck = _cleanup_stale_output_files(summa_dir, logger)
    if not stuck:
        return summa_dir

    for attempt in range(1, 1000):
        candidate = summa_dir.parent / f"{summa_dir.name}__r{attempt}"
        if candidate.exists() and _cleanup_stale_output_files(candidate, logger):
            continue  # that one is poisoned too
        candidate.mkdir(parents=True, exist_ok=True)
        log_once(
            logger, logging.WARNING, key=f'summa-output-redirect-{summa_dir}',
            message=(
                f"{len(stuck)} stale output file(s) in {summa_dir} could not be "
                f"removed (a previous model process exited without releasing "
                f"them). Redirecting this evaluation to {candidate.name} so it "
                f"neither collides with them nor reads them."
            ),
        )
        return candidate

    return summa_dir

def _run_summa_worker(summa_exe: Path, file_manager: Path, summa_dir: Path, logger, debug_info: Dict, summa_settings_dir: Path = None, timeout: int = 7200, config: Dict = None) -> bool:
    """SUMMA execution with iteration-aware logging.

    Log files include iteration number to prevent overwriting during calibration.
    This enables post-hoc debugging of failed runs when PARAMS_KEEP_TRIALS is enabled.

    Args:
        timeout: Maximum execution time in seconds (default: 7200 = 2 hours)
    """
    try:
        # Create log directory
        log_dir = summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Clean up stale output files from previous iterations
        # This prevents metrics being calculated from old files if current run fails
        _cleanup_stale_output_files(summa_dir, logger)

        # Include iteration in log filename to prevent overwriting
        iteration = debug_info.get('iteration', 0)
        individual_id = debug_info.get('individual_id', 0)
        log_file = log_dir / f"summa_worker_{os.getpid()}_iter{iteration:05d}_ind{individual_id:03d}.log"

        # Set environment for single-threaded execution
        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'
        })

        # Convert paths to strings for subprocess
        summa_exe_str = str(summa_exe)
        file_manager_str = str(file_manager)

        # Verify executable permissions
        if not os.access(summa_exe, os.X_OK):
            error_msg = f"SUMMA executable is not executable: {summa_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        # Update file manager with correct output path and settings path
        try:
            with open(file_manager, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Remove stale runinfo to avoid SUMMA Fortran write errors on reruns
            runinfo_path = summa_dir / "runinfo.txt"
            if runinfo_path.exists():
                try:
                    if os.name != 'nt':
                        runinfo_path.chmod(0o644)
                    runinfo_path.unlink()
                except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
                    log_once(logger, logging.WARNING, key='summa-runinfo-remove-failed',
                                message=f"Could not remove existing runinfo.txt: {e}")

            updated_lines = []
            output_path_updated = False
            settings_path_updated = False

            # Ensure paths end with slash for SUMMA
            output_path_str = str(summa_dir)
            if not output_path_str.endswith(os.sep):
                output_path_str += os.sep

            # Prepare settings path if provided
            settings_path_str = None
            if summa_settings_dir is not None:
                settings_path_str = str(summa_settings_dir)
                if not settings_path_str.endswith(os.sep):
                    settings_path_str += os.sep

            for line in lines:
                if 'outputPath' in line and not line.strip().startswith('!'):
                    # Preserve comment if present
                    parts = line.split('!')
                    comment = '!' + parts[1] if len(parts) > 1 else ''

                    # Update path - SUMMA requires quoted strings
                    updated_lines.append(f"outputPath '{output_path_str}' {comment}\n")
                    output_path_updated = True
                elif 'settingsPath' in line and not line.strip().startswith('!') and settings_path_str is not None:
                    # Preserve comment if present
                    parts = line.split('!')
                    comment = '!' + parts[1] if len(parts) > 1 else ''

                    # Update settings path - SUMMA requires quoted strings
                    updated_lines.append(f"settingsPath '{settings_path_str}' {comment}\n")
                    settings_path_updated = True
                else:
                    updated_lines.append(line)

            if not output_path_updated:
                # If outputPath not found, append it (though uncommon for valid file manager)
                updated_lines.append(f"outputPath '{output_path_str}'\n")

            if settings_path_str is not None and not settings_path_updated:
                # If settingsPath not found, append it
                updated_lines.append(f"settingsPath '{settings_path_str}'\n")

            with open(file_manager, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            # Deduplicate outputControl.txt if it exists
            output_control_name = None
            for line in updated_lines:
                if 'outputControlFile' in line and not line.strip().startswith('!'):
                    output_control_name = line.split("'")[1] if "'" in line else line.split()[1]
                    break

            if output_control_name and summa_settings_dir:
                oc_path = summa_settings_dir / output_control_name
                _deduplicate_output_control(oc_path, logger)
                _ensure_coupling_output_vars(oc_path, config, logger)

            logger.debug(f"Updated file manager output path to: {output_path_str}")
            if settings_path_str is not None:
                logger.debug(f"Updated file manager settings path to: {settings_path_str}")

        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            log_once(logger, logging.WARNING, key='summa-filemanager-update-failed',
                       message=f"Failed to update file manager paths: {e}")
            # Continue anyway, hoping it works or fails later

        # Check for intra-iteration parallel GRU execution
        use_parallel = config.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False) if config else False
        logger.debug(f"Parallel SUMMA check: config={config is not None}, use_parallel={use_parallel}, settings_dir={summa_settings_dir}")
        if use_parallel and summa_settings_dir is not None:
            from symfluence.models.summa.parallel_gru_execution import run_summa_gru_parallel

            num_parallel = int(config.get('SETTINGS_SUMMA_CPUS_PER_TASK', 8))
            logger.debug(f"Launching parallel SUMMA with {num_parallel} processes")
            result = run_summa_gru_parallel(
                summa_exe=Path(summa_exe_str),
                file_manager=Path(file_manager_str),
                summa_dir=summa_dir,
                settings_dir=Path(summa_settings_dir) if not isinstance(summa_settings_dir, Path) else summa_settings_dir,
                num_parallel=num_parallel,
                logger=logger,
                debug_info=debug_info,
                timeout=timeout,
                env=env,
            )
            if result:
                from symfluence.core.exceptions import ValidationError
                from symfluence.models.summa.output_validation import validate_output_coverage
                try:
                    validate_output_coverage(file_manager, summa_dir,
                                             (config or {}).get('FORCING_TIME_STEP_SIZE'))
                except (ValidationError, ValueError) as exc:
                    logger.error(str(exc))
                    debug_info['errors'].append(str(exc))
                    return False
                return True
            log_once(logger, logging.WARNING, key='summa-parallel-gru-fallback',
                     message="Parallel GRU execution did not succeed, falling back to sequential")

        # Build command as list to avoid shell=True security concerns
        cmd = [summa_exe_str, "-m", file_manager_str]
        cmd_str = " ".join(cmd)

        logger.debug(f"Executing SUMMA command: {cmd_str}")
        logger.debug(f"Working directory: {summa_dir}")

        debug_info['commands_run'].append(f"SUMMA: {cmd_str}")

        # Run SUMMA with system-level I/O profiling
        system_profiler = get_system_profiler()

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("SUMMA Execution Log\n")
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Working Directory: {summa_dir}\n")
            f.write(f"Environment: OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', 'unset')}\n")
            f.write("=" * 50 + "\n")
            f.flush()

            # Profile SUMMA execution
            with system_profiler.profile_subprocess(
                command=cmd,
                component='summa',
                iteration=debug_info.get('iteration'),
                cwd=summa_dir,
                env=env,
                track_files=True,
                output_dir=summa_dir
            ) as proc:
                proc.run(
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout
                )

        # Check if output files were created
        from symfluence.core.exceptions import ValidationError
        from symfluence.models.summa.output_validation import validate_output_coverage
        try:
            validate_output_coverage(file_manager, summa_dir,
                                     (config or {}).get('FORCING_TIME_STEP_SIZE'))
        except (ValidationError, ValueError) as exc:
            logger.error(str(exc))
            debug_info['errors'].append(str(exc))
            return False
        timestep_files = list(summa_dir.glob("*timestep.nc"))
        if not timestep_files:
            # Look for any .nc files
            nc_files = list(summa_dir.glob("*.nc"))
            if not nc_files:
                dir_contents = list(summa_dir.glob("*"))
                error_msg = f"No SUMMA output files found in {summa_dir}. Contents: {[f.name for f in dir_contents]}"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return False

        logger.debug(f"SUMMA execution completed successfully. Output files: {len(timestep_files)} timestep files")
        debug_info['summa_output_files'] = [str(f) for f in timestep_files[:3]]  # First 3 files

        return True

    except subprocess.CalledProcessError as e:
        # Without the reason, a rejected parameter set is indistinguishable
        # from a poorly performing one once it has been folded into a penalty
        # score, so surface SUMMA's own diagnosis here.
        reason = _extract_summa_fatal_error(log_file)
        error_msg = f"SUMMA simulation failed with exit code {e.returncode}"
        if reason:
            error_msg = f"{error_msg}: {reason}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

    except subprocess.TimeoutExpired:
        error_msg = f"SUMMA simulation timed out ({timeout}s)"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

    except (FileNotFoundError, OSError) as e:
        error_msg = f"Error running SUMMA: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _extract_summa_fatal_error(log_file, max_tail_bytes: int = 8192) -> str:
    """Pull SUMMA's ``FATAL ERROR`` diagnosis out of a worker log.

    SUMMA writes the reason it aborted to stdout and exits 1. The calibration
    layer only sees the exit code, so a parameter set rejected by
    ``paramCheck`` scores the same generic penalty as one that merely fits
    badly. Recovering the message keeps rejections attributable.

    Args:
        log_file: Path to the SUMMA worker log.
        max_tail_bytes: How much of the end of the log to inspect.

    Returns:
        The fatal error message, or an empty string when none is found.
    """
    try:
        path = Path(log_file)
        size = path.stat().st_size
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            if size > max_tail_bytes:
                f.seek(size - max_tail_bytes)
            tail = f.read()
    except (OSError, ValueError):
        return ""

    for line in reversed(tail.splitlines()):
        stripped = line.strip()
        if stripped.startswith('FATAL ERROR'):
            return stripped
    return ""


def _run_mizuroute_worker(task_data: Dict, mizuroute_dir: Path, logger, debug_info: Dict, summa_dir: Path = None) -> bool:
    """Updated mizuRoute worker with fixed time precision handling"""
    try:
        # Clean up stale mizuRoute output files from previous iterations
        _cleanup_stale_output_files(mizuroute_dir, logger)

        # Verify SUMMA output exists first
        if summa_dir is None:
            summa_dir = Path(task_data['summa_dir'])

        # Prefer *_for_routing.nc (from lumped→distributed conversion) over *_timestep.nc
        routing_files = list(summa_dir.glob("*_for_routing.nc"))
        expected_files = list(summa_dir.glob("*timestep.nc"))

        if not routing_files and not expected_files:
            error_msg = f"No SUMMA output files found for mizuRoute input: {summa_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        # Fix time precision on the file mizuRoute will actually read
        time_fix_file = routing_files[0] if routing_files else expected_files[0]

        try:
            logger.debug(f"Fixing time precision for mizuRoute compatibility: {time_fix_file.name}")
            fix_summa_time_precision(time_fix_file)
            logger.debug("Time precision fixed successfully")
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            error_msg = f"Failed to fix SUMMA time precision: {str(e)}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        logger.debug(f"Found {len(expected_files)} SUMMA output files for mizuRoute")
        config = task_data['config']
        mizu_timeout = int(config.get('MIZUROUTE_TIMEOUT', 3600))

        # Get mizuRoute executable (canonical keys first, then deprecated aliases)
        mizu_path = config.get('MIZUROUTE_INSTALL_PATH',
                               config.get('INSTALL_PATH_MIZUROUTE', 'default'))
        if mizu_path == 'default':
            mizu_path = Path(config.get('SYMFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)

        mizu_exe = mizu_path / config.get('MIZUROUTE_EXE',
                                          config.get('EXE_NAME_MIZUROUTE', 'mizuRoute.exe'))
        control_file = Path(task_data['mizuroute_settings_dir']) / 'mizuroute.control'

        # Verify files exist
        if not mizu_exe.exists():
            error_msg = f"mizuRoute executable not found: {mizu_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        if not control_file.exists():
            error_msg = f"mizuRoute control file not found: {control_file}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        _rewrite_mizuroute_control_for_run(
            control_file, summa_dir, mizuroute_dir, time_fix_file.name
        )

        debug_info['files_checked'].extend([
            f"mizuRoute exe: {mizu_exe}",
            f"mizuRoute control: {control_file}"
        ])

        # Create log directory with iteration-aware naming
        log_dir = mizuroute_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        iteration = debug_info.get('iteration', 0)
        individual_id = debug_info.get('individual_id', 0)
        log_file = log_dir / f"mizuroute_worker_{os.getpid()}_iter{iteration:05d}_ind{individual_id:03d}.log"

        # Build command as list to avoid shell=True security concerns
        cmd = [str(mizu_exe), str(control_file)]
        cmd_str = " ".join(cmd)

        # Set OpenMP thread count for mizuRoute (if compiled with OpenMP)
        mizu_env = os.environ.copy()
        mizu_num_threads = str(config.get('MIZUROUTE_NUM_THREADS', 1))
        mizu_env['OMP_NUM_THREADS'] = mizu_num_threads

        logger.debug(f"Executing mizuRoute command: {cmd_str} (OMP_NUM_THREADS={mizu_num_threads})")
        debug_info['commands_run'].append(f"mizuRoute: {cmd_str}")
        debug_info['mizuroute_log'] = str(log_file)

        # Run mizuRoute with system-level I/O profiling
        system_profiler = get_system_profiler()

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("mizuRoute Execution Log\n")
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Working Directory: {control_file.parent}\n")
            f.write(f"OMP_NUM_THREADS: {mizu_num_threads}\n")
            f.write("=" * 50 + "\n")
            f.flush()

            # Profile mizuRoute execution
            with system_profiler.profile_subprocess(
                command=cmd,
                component='mizuroute',
                iteration=debug_info.get('iteration'),
                cwd=control_file.parent,
                env=mizu_env,
                track_files=True,
                output_dir=mizuroute_dir
            ) as proc:
                proc.run(
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=mizu_timeout
                )

        # Check for output files
        nc_files = list(mizuroute_dir.glob("*.nc"))
        if not nc_files:
            error_msg = f"No mizuRoute output files found in {mizuroute_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        logger.debug(f"mizuRoute execution completed successfully. Output files: {len(nc_files)}")
        debug_info['mizuroute_output_files'] = [str(f) for f in nc_files[:3]]

        return True

    except subprocess.TimeoutExpired:
        error_msg = f"mizuRoute simulation timed out ({mizu_timeout}s)"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
        error_msg = f"mizuRoute execution failed: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _needs_mizuroute_routing_worker(config: Dict) -> bool:
    """Check if mizuRoute routing is needed"""
    domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
    routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

    if domain_method not in ['point', 'lumped']:
        return True

    if domain_method == 'lumped' and routing_delineation == 'river_network':
        return True

    return False
