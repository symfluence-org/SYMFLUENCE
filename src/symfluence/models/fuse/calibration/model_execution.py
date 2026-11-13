# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Model Execution

Standalone functions for executing the FUSE model subprocess,
managing input file symlinks, and validating output.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from symfluence.core.logging_utils import log_once
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.process_exec import run as run_subprocess
from symfluence.models.fuse.calibration.file_manager import resolve_fuse_id

logger = logging.getLogger(__name__)

# Run modes already warned about — the rejection fires on every calibration
# trial, so without dedupe a 1000-iteration run logs it 1000 times.
_REJECTED_MODES_WARNED: set = set()


def detect_fuse_run_mode(config: Dict[str, Any], kwargs: Dict[str, Any],
                         log: Optional[logging.Logger] = None) -> str:
    """
    Determine the FUSE run mode for a calibration trial.

    Calibration always runs FUSE in ``run_pre`` mode: trial parameters are
    written into para_def.nc and read directly by run_pre. ``run_def`` has
    exactly one role in the workflow — the runner's *initial default run*,
    which generates para_def.nc (and the default-parameter baseline) before
    any calibration starts. It is never a valid calibration mode: it ignores
    the trial parameters and re-derives para_def.nc from the constraints
    file, so every trial would score the default parameter set.

    An explicit ``FUSE_RUN_MODE: run_def`` (or a run_def kwarg) is therefore
    rejected with a warning rather than honored.

    Args:
        config: Configuration dictionary
        kwargs: Additional keyword arguments (may contain 'mode')
        log: Logger instance

    Returns:
        The calibration run mode (always 'run_pre').
    """
    log = log or logger

    requested = resolve_config_value(config, lambda: config.model.fuse.run_mode, None, 'FUSE_RUN_MODE') or kwargs.get('mode')
    if requested and requested != 'run_pre' and requested not in _REJECTED_MODES_WARNED:
        _REJECTED_MODES_WARNED.add(requested)
        log.warning(
            f"FUSE run mode '{requested}' requested for calibration — ignored "
            "(warning shown once per run). Calibration always uses run_pre "
            "(trial parameters are read from para_def.nc); run_def is only "
            "used by the runner's initial default run to generate para_def.nc."
        )

    return 'run_pre'


def resolve_fuse_paths(
    config: Dict[str, Any],
    settings_dir: Path,
    log: Optional[logging.Logger] = None
) -> Tuple[Path, Path, Path]:
    """
    Resolve FUSE executable, file manager, and execution directory paths.

    Args:
        config: Configuration dictionary
        settings_dir: FUSE settings directory
        log: Logger instance

    Returns:
        Tuple of (fuse_exe, filemanager_path, execution_cwd)
    """
    log = log or logger

    # Resolve executable
    fuse_install = resolve_config_value(config, lambda: config.model.fuse.install_path, dict_key='FUSE_INSTALL_PATH')
    data_dir = resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR')
    fuse_exe = resolve_install_dir(fuse_install, data_dir, 'installs/fuse/bin') / 'fuse.exe'

    # Resolve file manager and execution directory
    if settings_dir.name == 'FUSE':
        filemanager_path = settings_dir / 'fm_catch.txt'
        execution_cwd = settings_dir
    elif (settings_dir / 'FUSE').exists():
        filemanager_path = settings_dir / 'FUSE' / 'fm_catch.txt'
        execution_cwd = settings_dir / 'FUSE'
    else:
        filemanager_path = settings_dir / 'fm_catch.txt'
        execution_cwd = settings_dir

    return fuse_exe, filemanager_path, execution_cwd


def prepare_input_files(
    config: Dict[str, Any],
    execution_cwd: Path,
    log: Optional[logging.Logger] = None
) -> Optional[Tuple[str, str]]:
    """
    Create symlinks and copies for FUSE input files.

    Args:
        config: Configuration dictionary
        execution_cwd: Directory where FUSE will execute
        log: Logger instance

    Returns:
        Tuple of (fuse_run_id, actual_decisions_file), or None if critical
        input files are missing.
    """
    log = log or logger

    fuse_run_id = 'sim'
    data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
    domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
    project_dir = data_dir / f"domain_{domain_name}"
    fuse_input_dir = resolve_data_subdir(project_dir, 'forcing') / 'FUSE_input'
    experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, 'run_1', 'EXPERIMENT_ID')
    fuse_id = resolve_fuse_id(config, experiment_id)

    # Input files to symlink
    input_files = [
        (fuse_input_dir / f"{domain_name}_input.nc", f"{fuse_run_id}_input.nc"),
        (fuse_input_dir / f"{domain_name}_elev_bands.nc", f"{fuse_run_id}_elev_bands.nc")
    ]

    # COPY (not symlink) the parameter file to match the short alias
    _copy_parameter_file(execution_cwd, domain_name, fuse_id, fuse_run_id, log)

    # Ensure configuration files are present
    project_settings_dir = project_dir / 'settings' / 'FUSE'
    actual_decisions_file = _ensure_config_files(
        execution_cwd, project_settings_dir, fuse_id, input_files, log
    )

    # Create symlinks (but NOT para_def.nc which was copied above)
    missing = _create_symlinks(input_files, execution_cwd, log)

    # Check for missing critical input files (forcing, elev_bands)
    critical_missing = [
        (name, src) for name, src in missing
        if '_input.nc' in name or '_elev_bands.nc' in name
    ]
    if critical_missing:
        names = ', '.join(f"{name} (expected at {src})" for name, src in critical_missing)
        log.error(
            f"FUSE cannot run: critical input files are missing: {names}. "
            f"This usually means the FUSE preprocessing step did not complete. "
            f"Re-run preprocessing or check the forcing data directory."
        )
        return None

    # Verify para_def copy exists
    param_file_dst = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
    if not param_file_dst.exists():
        log.error(f"FUSE para_def copy was not created. Expected: {param_file_dst}")

    return fuse_run_id, actual_decisions_file


def _copy_parameter_file(
    execution_cwd: Path,
    domain_name: str,
    fuse_id: str,
    fuse_run_id: str,
    log: logging.Logger
) -> None:
    """Copy para_def.nc to match the short alias."""
    param_file_src = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
    param_file_dst = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"

    if param_file_src.exists():
        if param_file_dst.exists():
            param_file_dst.unlink()
        shutil.copy2(param_file_src, param_file_dst)
        log.debug(f"FUSE para_def copied (not symlinked): {param_file_dst.name}")
    else:
        log_once(log, logging.WARNING, key=f'fuse-paradef-missing-{param_file_src}',
                 message=f"FUSE para_def source not found: {param_file_src}")


def _ensure_config_files(
    execution_cwd: Path,
    project_settings_dir: Path,
    experiment_id: str,
    input_files: list,
    log: logging.Logger
) -> str:
    """Ensure FUSE configuration files are present, return actual decisions filename."""
    log.debug(f"Checking for config files in: {project_settings_dir}")

    config_files = ['input_info.txt', 'fuse_zNumerix.txt']

    # Add decisions file
    actual_decisions_file = f"fuse_zDecisions_{experiment_id}.txt"
    if (project_settings_dir / actual_decisions_file).exists():
        config_files.append(actual_decisions_file)
    else:
        log_once(log, logging.WARNING, key=f'fuse-decisions-missing-{actual_decisions_file}',
                    message=f"Decisions file {actual_decisions_file} not found in {project_settings_dir}")
        try:
            decisions = list(project_settings_dir.glob("fuse_zDecisions_*.txt"))
            if decisions:
                actual_decisions_file = decisions[0].name
                config_files.append(actual_decisions_file)
                log_once(log, logging.WARNING, key=f'fuse-decisions-fallback-{actual_decisions_file}',
                         message=f"Using fallback decisions file: {actual_decisions_file}")
        except Exception as e:  # noqa: BLE001 — calibration resilience
            log.warning(f"Error searching for decisions files: {e}", exc_info=True)

    for cfg_file in config_files:
        target_path = execution_cwd / cfg_file
        if not target_path.exists():
            src_path = project_settings_dir / cfg_file
            if src_path.exists():
                input_files.append((src_path, cfg_file))
                log_once(log, logging.WARNING, key=f'fuse-restore-config-{cfg_file}',
                         message=f"Restoring missing config file: {cfg_file}")
            else:
                log.error(f"Source config file not found: {src_path}")

    return actual_decisions_file


def _create_symlinks(
    input_files: list,
    execution_cwd: Path,
    log: logging.Logger
) -> list:
    """Create symlinks for input files.

    Returns:
        List of (link_name, source_path) tuples for sources that were missing.
    """
    missing = []
    for src, link_name in input_files:
        if src.exists():
            link_path = execution_cwd / link_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                link_path.symlink_to(src)
                log.debug(f"Created symlink: {link_path} -> {src}")
            except OSError:
                shutil.copy2(src, link_path)
                log.debug(f"Copied (symlink unavailable): {src} -> {link_path}")
        else:
            missing.append((link_name, src))
            log.error(f"FUSE input file not found: {src}")
    return missing


def validate_fuse_inputs(
    execution_cwd: Path,
    fuse_run_id: str,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> bool:
    """
    Pre-flight validation of all FUSE input files before execution.

    Checks that forcing data, elevation bands, and configuration files
    are present and readable. This catches missing or broken files BEFORE
    running FUSE, providing clear diagnostics instead of silent Fortran
    crashes.

    Returns:
        True if all inputs are valid, False otherwise.
    """
    log = log or logger
    errors = []

    # Check forcing input symlink resolves to a real file
    forcing_link = execution_cwd / f"{fuse_run_id}_input.nc"
    if forcing_link.is_symlink():
        target = forcing_link.resolve()
        if not target.exists():
            errors.append(
                f"Forcing file symlink is broken: {forcing_link.name} -> {target}. "
                f"The FUSE_input directory may not exist."
            )
    elif not forcing_link.exists():
        errors.append(f"Forcing input file not found: {forcing_link}")

    # Check elevation bands
    elev_link = execution_cwd / f"{fuse_run_id}_elev_bands.nc"
    if elev_link.is_symlink():
        target = elev_link.resolve()
        if not target.exists():
            errors.append(f"Elevation bands symlink is broken: {elev_link.name} -> {target}")
    elif not elev_link.exists():
        errors.append(f"Elevation bands file not found: {elev_link}")

    # Check file manager
    fm_path = execution_cwd / 'fm_catch.txt'
    if not fm_path.exists():
        errors.append(f"File manager not found: {fm_path}")

    if errors:
        log.error(
            f"FUSE pre-flight validation FAILED ({len(errors)} issues):"
        )
        for i, err in enumerate(errors, 1):
            log.error(f"  [{i}] {err}")
        log.error(
            "FUSE will not be executed. Fix the above issues and re-run. "
            "Most commonly, this means the preprocessing step (prepare_forcing) "
            "did not complete successfully."
        )
        return False

    log.debug("FUSE pre-flight validation passed — all input files present")
    return True


def _build_fuse_library_env(fuse_exe: Path) -> Dict[str, str]:
    """Build subprocess environment with HDF5/NetCDF library paths for FUSE."""
    env = os.environ.copy()

    lib_paths = []

    exe_lib = Path(fuse_exe).parent.parent / 'lib'
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

    return env


def execute_fuse(
    fuse_exe: Path,
    filemanager_path: Path,
    execution_cwd: Path,
    fuse_run_id: str,
    mode: str,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Optional[subprocess.CompletedProcess]:
    """
    Execute the FUSE subprocess.

    Args:
        fuse_exe: Path to FUSE executable
        filemanager_path: Path to file manager
        execution_cwd: Working directory for execution
        fuse_run_id: Short run alias (e.g., 'sim')
        mode: Run mode ('run_def' or 'run_pre')
        config: Configuration dictionary
        log: Logger instance

    Returns:
        CompletedProcess result, or None if execution failed
    """
    log = log or logger
    fuse_id = resolve_fuse_id(config)

    cmd = [str(fuse_exe), str(filemanager_path.name), fuse_run_id, mode]

    # For run_pre mode, append parameter file
    if mode == 'run_pre':
        param_file = _find_run_pre_param_file(execution_cwd, fuse_run_id, fuse_id,
                                               resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME'), log)
        if param_file:
            cmd.append(str(param_file.name))
            cmd.append('1')
        else:
            return None

    log.debug(f"Executing FUSE: {' '.join(cmd)} in {execution_cwd}")

    # Verify key files
    expected_para_def = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
    if not expected_para_def.exists() and not expected_para_def.is_symlink():
        log.error(f"FUSE parameter file not found: {expected_para_def}")

    # Clean stale output files before execution.
    # FUSE in run_def mode creates para_def.nc and runs_def.nc as output files.
    # If a stale runs_def.nc exists from a previous iteration, FUSE's NetCDF
    # library fails with "NC_UNLIMITED size already in use" because NETCDF3
    # doesn't allow redefining unlimited dimensions on existing files.
    run_suffix = 'runs_def' if mode == 'run_def' else 'runs_pre'
    stale_runs = execution_cwd / f"{fuse_run_id}_{fuse_id}_{run_suffix}.nc"
    if stale_runs.exists():
        try:
            stale_runs.unlink()
            log.debug(f"Removed stale output: {stale_runs.name}")
        except OSError as e:
            log_once(log, logging.WARNING, key=f'fuse-stale-runs-{stale_runs.name}',
                     message=f"Could not remove stale output {stale_runs.name}: {e}")

    # In run_def mode, FUSE also recreates para_def.nc from the constraints
    # file. Remove the stale one so FUSE creates it fresh in its native
    # NETCDF3 format (avoids format conflicts with our NETCDF4 copy).
    if mode == 'run_def' and expected_para_def.exists():
        try:
            expected_para_def.unlink()
            log.debug(f"Removed stale para_def for run_def: {expected_para_def.name}")
        except OSError as e:
            log_once(log, logging.WARNING, key=f'fuse-stale-paradef-{expected_para_def.name}',
                     message=f"Could not remove stale para_def {expected_para_def.name}: {e}")

    env = _build_fuse_library_env(fuse_exe)

    result = run_subprocess(
        cmd,
        cwd=str(execution_cwd),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=resolve_config_value(config, lambda: config.model.fuse.timeout, dict_key='FUSE_TIMEOUT'),
        env=env
    )

    if result.returncode != 0:
        log.error(f"FUSE failed with return code {result.returncode}")
        log.error(f"STDOUT: {result.stdout}")
        log.error(f"STDERR: {result.stderr}")
        return None

    # Detect Fortran STOP messages and NetCDF errors
    # (FUSE returns exit code 0 on macOS even on fatal errors)
    combined_output = (result.stdout or '') + (result.stderr or '')

    # Check for NetCDF library errors (e.g. "NC_UNLIMITED size already in use")
    if 'NetCDF:' in combined_output:
        nc_lines = [
            line.strip() for line in combined_output.splitlines()
            if 'NetCDF:' in line
        ]
        nc_context = '; '.join(nc_lines[:3])
        log.error(f"FUSE NetCDF error (exit code 0): {nc_context}")
        return None

    if 'STOP' in combined_output:
        # Extract STOP lines for context
        stop_lines = [
            line.strip() for line in combined_output.splitlines()
            if 'STOP' in line
        ]
        stop_context = '; '.join(stop_lines[:3]) if stop_lines else combined_output[-300:]
        log.error(
            f"FUSE hit Fortran STOP (exit code 0 on macOS): {stop_context}"
        )
        return None

    if result.stdout:
        log.debug(f"FUSE stdout (last 500 chars): {result.stdout[-500:]}")
    if result.stderr:
        log.debug(f"FUSE stderr: {result.stderr}")

    return result


def _find_run_pre_param_file(
    execution_cwd: Path,
    fuse_run_id: str,
    fuse_id: str,
    domain_name: str,
    log: logging.Logger
) -> Optional[Path]:
    """Find the parameter file for run_pre mode."""
    param_file = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
    if param_file.exists():
        return param_file

    param_file_alt = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
    if param_file_alt.exists():
        return param_file_alt

    log.error(f"Parameter file not found for run_pre: tried {param_file} and {param_file_alt}")
    return None


def handle_fuse_output(
    execution_cwd: Path,
    fuse_output_dir: Path,
    fuse_run_id: str,
    mode: str,
    config: Dict[str, Any],
    result: subprocess.CompletedProcess,
    log: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Move FUSE output to final destination and validate it.

    Args:
        execution_cwd: FUSE execution directory
        fuse_output_dir: Final output directory
        fuse_run_id: Short run alias
        mode: Run mode
        config: Configuration dictionary
        result: Subprocess result
        log: Logger instance

    Returns:
        Path to final output file, or None if validation failed
    """
    log = log or logger

    domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
    fuse_id = resolve_fuse_id(config)

    run_suffix = 'runs_def' if mode == 'run_def' else 'runs_pre'
    local_output_filename = f"{fuse_run_id}_{fuse_id}_{run_suffix}.nc"
    local_output_path = execution_cwd / local_output_filename
    final_output_path = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

    if local_output_path.exists():
        try:
            if final_output_path.exists():
                final_output_path.unlink()
            shutil.move(str(local_output_path), str(final_output_path))
            log.debug(f"Moved output from {local_output_path} to {final_output_path}")
        except Exception as e:  # noqa: BLE001 — calibration resilience
            log.error(f"Failed to move output file: {e}", exc_info=True)
            return None
    else:
        log.error(f"FUSE returned success but local output file not created: {local_output_path}")
        if result.stdout:
            stdout_lines = result.stdout.split('\n')
            if len(stdout_lines) > 20:
                log.error(f"FUSE stdout (first 10 lines): {chr(10).join(stdout_lines[:10])}")
                log.error(f"FUSE stdout (last 10 lines): {chr(10).join(stdout_lines[-10:])}")
            else:
                log.error(f"FUSE stdout: {result.stdout}")
        return None

    # Validate output has actual data
    if not _validate_fuse_output(final_output_path, result, log):
        # Log comprehensive diagnostics on first failure
        if not getattr(handle_fuse_output, '_diagnostics_logged', False):
            filemanager_path = execution_cwd / 'fm_catch.txt'
            cmd_repr = f"fuse.exe fm_catch.txt {fuse_run_id} {mode}"
            _log_fuse_diagnostics(execution_cwd, filemanager_path, cmd_repr.split(), log)
            handle_fuse_output._diagnostics_logged = True
        return None

    log.debug(f"FUSE completed successfully, output: {final_output_path}")
    return final_output_path


def _validate_fuse_output(
    output_path: Path,
    result: subprocess.CompletedProcess,
    log: logging.Logger
) -> bool:
    """Validate that FUSE output file exists, is readable, and has actual data."""
    # Check file size first — a valid FUSE NetCDF should be at least a few KB
    try:
        file_size = output_path.stat().st_size
        if file_size < 1024:
            log.error(
                f"FUSE output file is too small ({file_size} bytes) — "
                f"model likely crashed silently (Fortran STOP returns exit code 0)."
            )
            _log_fuse_subprocess_output(result, log)
            return False
    except OSError as e:
        log.error(f"Cannot stat FUSE output file {output_path}: {e}")
        return False

    try:
        import xarray as xr
        with xr.open_dataset(output_path, decode_times=False) as ds_check:
            time_dim = 'time' if 'time' in ds_check.dims else None
            if time_dim and ds_check.sizes[time_dim] == 0:
                log.error(
                    "FUSE output has 0 time steps — model likely crashed silently "
                    "(Fortran STOP returns exit code 0)."
                )
                _log_fuse_subprocess_output(result, log)
                return False
            n_time = ds_check.sizes.get(time_dim, 0) if time_dim else -1
            log.debug(f"FUSE output validated: {n_time} time steps in {output_path.name}")
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(
            f"FUSE output file is not a readable NetCDF: {output_path.name} — {e}."
        , exc_info=True)
        _log_fuse_subprocess_output(result, log)
        return False

    return True


def _log_fuse_subprocess_output(
    result: subprocess.CompletedProcess,
    log: logging.Logger
) -> None:
    """Log FUSE subprocess stdout/stderr for diagnostics."""
    stdout = (result.stdout or '').strip()
    stderr = (result.stderr or '').strip()
    if stdout:
        log.error(f"FUSE stdout: {stdout[-1000:]}")
    else:
        log.error("FUSE stdout: (empty — Fortran may have written to a log file)")
    if stderr:
        log.error(f"FUSE stderr: {stderr[-500:]}")


def _log_fuse_diagnostics(
    execution_cwd: Path,
    filemanager_path: Path,
    cmd: list,
    log: logging.Logger
) -> None:
    """Log comprehensive diagnostics when FUSE fails."""
    log.error(f"FUSE command: {' '.join(cmd)}")
    log.error(f"FUSE working directory: {execution_cwd}")

    # Log key files in the working directory
    try:
        files = sorted(execution_cwd.iterdir())
        file_info = []
        for f in files:
            if f.is_symlink():
                target = f.resolve()
                exists = target.exists()
                file_info.append(f"  {f.name} -> {target} ({'OK' if exists else 'BROKEN'})")
            elif f.is_file():
                file_info.append(f"  {f.name} ({f.stat().st_size} bytes)")
        log.error("Files in execution dir:\n" + "\n".join(file_info[:20]))
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(f"Could not list execution directory: {e}", exc_info=True)

    # Log file manager content
    try:
        content = filemanager_path.read_text(encoding='utf-8', errors='replace')
        log.error(f"File manager content ({filemanager_path.name}):\n{content}")
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(f"Could not read file manager: {e}", exc_info=True)


def log_execution_directory(execution_cwd: Path, log: logging.Logger) -> None:
    """Log files in execution directory at DEBUG level."""
    log.debug(f"Files in execution CWD ({execution_cwd}):")
    try:
        for f in execution_cwd.iterdir():
            if f.is_symlink():
                log.debug(f"  {f.name} -> {f.resolve()}")
            else:
                log.debug(f"  {f.name}")
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.debug(f"Could not list directory: {e}", exc_info=True)
