# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE File Manager Utilities

Handles updates to FUSE file manager (fm_catch.txt) for parallel execution
and path isolation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.mixins.project import resolve_data_subdir

logger = logging.getLogger(__name__)


def resolve_fuse_id(config: Dict[str, Any], experiment_id: Optional[str] = None) -> str:
    """Resolve the FUSE file ID, hashing long IDs to fit FUSE's 6-char buffer.

    Mirrors FUSERunner._get_fuse_file_id so worker-side filenames match
    runner-side filenames (FUSE Fortran uses CHARACTER(LEN=6) for FMODEL_ID).
    """
    if experiment_id is None:
        experiment_id = (resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID') if config else None) or 'run_1'
    fuse_id = (resolve_config_value(config, lambda: config.model.fuse.file_id, None, 'FUSE_FILE_ID') if config else None) or experiment_id
    if fuse_id and len(fuse_id) > 6:
        import hashlib
        fuse_id = hashlib.md5(fuse_id.encode(), usedforsecurity=False).hexdigest()[:6]
    return fuse_id


def update_fuse_file_manager(
    filemanager_path: Path,
    settings_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None,
    experiment_id: Optional[str] = None,
    use_local_input: bool = False,
    decisions_file: Optional[str] = None
) -> bool:
    """
    Update FUSE file manager with isolated paths for parallel execution.

    Args:
        filemanager_path: Path to fm_catch.txt
        settings_dir: Isolated settings directory (where input files are)
        output_dir: Isolated output directory
        config: Configuration dictionary
        log: Logger instance
        experiment_id: Experiment ID to use for FMODEL_ID and decisions file
        use_local_input: If True, set INPUT_PATH to ./ and expect files to be symlinked
        decisions_file: Actual decisions filename to use (if known from pre-check)

    Returns:
        True if successful
    """
    log = log or logger

    try:
        # Read file with encoding fallback
        lines = _read_file_with_fallback(filemanager_path, log)

        # Resolve experiment/fuse IDs (hash long IDs to fit FUSE's 6-char buffer)
        if experiment_id is None:
            experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, 'run_1', 'EXPERIMENT_ID') if config else 'run_1'
        fuse_id = resolve_fuse_id(config, experiment_id)

        # Resolve paths
        settings_path_str = "./"
        output_path_str = "./"
        input_path_str = _resolve_input_path(config, use_local_input)

        log.debug(f"Using paths - Settings: {settings_path_str}, Output: {output_path_str}")

        # Resolve simulation dates
        sim_start, sim_end, eval_start, eval_end = _resolve_simulation_dates(config, log)

        # Resolve decisions file
        execution_cwd = filemanager_path.parent
        actual_decisions = _resolve_decisions_file(
            decisions_file, experiment_id, execution_cwd, log
        )

        # Update lines
        updated_lines = _update_file_manager_lines(
            lines,
            settings_path_str=settings_path_str,
            output_path_str=output_path_str,
            input_path_str=input_path_str,
            actual_decisions=actual_decisions,
            fuse_id=fuse_id,
            sim_start=sim_start,
            sim_end=sim_end,
            eval_start=eval_start,
            eval_end=eval_end,
        )

        with open(filemanager_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

        log.debug(f"Updated file manager: decisions={experiment_id}, fmodel_id={fuse_id}")
        return True

    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(f"Failed to update FUSE file manager: {e}", exc_info=True)
        return False


def _read_file_with_fallback(path: Path, log: logging.Logger) -> list:
    """Read file with UTF-8 encoding, falling back to latin-1."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError as ue:
        log.warning(
            f"UTF-8 decode error reading {path} at position {ue.start}: "
            f"falling back to latin-1 encoding"
        )
        with open(path, 'r', encoding='latin-1') as f:
            return f.readlines()


def _resolve_input_path(
    config: Optional[Dict[str, Any]],
    use_local_input: bool
) -> Optional[str]:
    """Resolve the INPUT_PATH for the file manager."""
    if use_local_input:
        return "./"

    if not config:
        return None

    data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
    domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
    project_dir = data_dir / f"domain_{domain_name}"
    fuse_input_dir = resolve_data_subdir(project_dir, 'forcing') / 'FUSE_input'

    if fuse_input_dir.exists():
        input_path_str = str(fuse_input_dir)
        if not input_path_str.endswith('/'):
            input_path_str += '/'
        return input_path_str

    return None


def _resolve_simulation_dates(
    config: Optional[Dict[str, Any]],
    log: logging.Logger
) -> tuple:
    """Resolve simulation and evaluation dates from forcing file or config.

    Returns:
        (sim_start, sim_end, eval_start, eval_end) — any may be None
    """
    sim_start = None
    sim_end = None
    eval_start = None
    eval_end = None

    if not config:
        return sim_start, sim_end, eval_start, eval_end

    # Try to read actual dates from forcing file
    data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
    domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
    project_dir = data_dir / f"domain_{domain_name}"
    forcing_file = resolve_data_subdir(project_dir, 'forcing') / 'FUSE_input' / f"{domain_name}_input.nc"

    if forcing_file.exists():
        try:
            import numpy as np
            import pandas as pd
            import xarray as xr
            with xr.open_dataset(forcing_file) as ds:
                time_vals = ds['time'].values
                if len(time_vals) > 0:
                    first_val = time_vals[0]
                    last_val = time_vals[-1]

                    if np.issubdtype(type(first_val), np.datetime64):
                        forcing_start = pd.Timestamp(first_val)
                        forcing_end = pd.Timestamp(last_val)
                    elif isinstance(first_val, (int, float, np.integer, np.floating)):
                        forcing_start = pd.Timestamp('1970-01-01') + pd.Timedelta(days=float(first_val))
                        forcing_end = pd.Timestamp('1970-01-01') + pd.Timedelta(days=float(last_val))
                    else:
                        forcing_start = pd.Timestamp(first_val)
                        forcing_end = pd.Timestamp(last_val)

                    sim_start = forcing_start.strftime('%Y-%m-%d')
                    sim_end = forcing_end.strftime('%Y-%m-%d')
                    log.debug(f"Using forcing file dates: {sim_start} to {sim_end}")
        except Exception as e:  # noqa: BLE001 — calibration resilience
            log.warning(f"Could not read forcing file dates: {e}", exc_info=True)

    # Fallback to config dates
    if sim_start is None:
        exp_start = resolve_config_value(config, lambda: config.domain.time_start, '', 'EXPERIMENT_TIME_START')
        if exp_start:
            sim_start = str(exp_start).split()[0]
    if sim_end is None:
        exp_end = resolve_config_value(config, lambda: config.domain.time_end, '', 'EXPERIMENT_TIME_END')
        if exp_end:
            sim_end = str(exp_end).split()[0]

    # Calibration period
    calib_period = resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD')
    if calib_period and ',' in str(calib_period):
        parts = str(calib_period).split(',')
        eval_start = parts[0].strip()
        eval_end = parts[1].strip()

    return sim_start, sim_end, eval_start, eval_end


def _resolve_decisions_file(
    decisions_file: Optional[str],
    experiment_id: str,
    execution_cwd: Path,
    log: logging.Logger
) -> str:
    """Resolve the actual decisions filename to use."""
    if decisions_file:
        return decisions_file

    actual_decisions = f"fuse_zDecisions_{experiment_id}.txt"
    if not (execution_cwd / actual_decisions).exists():
        found_files = list(execution_cwd.glob('fuse_zDecisions_*.txt'))
        if found_files:
            actual_decisions = found_files[0].name
            log.debug(f"Using available decisions file: {actual_decisions}")

    return actual_decisions


def _update_file_manager_lines(
    lines: list,
    settings_path_str: str,
    output_path_str: str,
    input_path_str: Optional[str],
    actual_decisions: str,
    fuse_id: str,
    sim_start: Optional[str],
    sim_end: Optional[str],
    eval_start: Optional[str],
    eval_end: Optional[str],
) -> list:
    """Apply updates to file manager lines."""
    updated_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("'") and 'SETNGS_PATH' in line:
            updated_lines.append(f"'{settings_path_str}'     ! SETNGS_PATH\n")
        elif stripped.startswith("'") and 'INPUT_PATH' in line:
            if input_path_str:
                updated_lines.append(f"'{input_path_str}'        ! INPUT_PATH\n")
            else:
                updated_lines.append(line)
        elif stripped.startswith("'") and 'OUTPUT_PATH' in line:
            updated_lines.append(f"'{output_path_str}'       ! OUTPUT_PATH\n")
        elif stripped.startswith("'") and 'M_DECISIONS' in line:
            updated_lines.append(f"'{actual_decisions}'        ! M_DECISIONS        = definition of model decisions\n")
        elif stripped.startswith("'") and 'FMODEL_ID' in line:
            updated_lines.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
        elif stripped.startswith("'") and 'FORCING INFO' in line:
            updated_lines.append("'input_info.txt'                 ! FORCING INFO       = definition of the forcing file\n")
        elif stripped.startswith("'") and 'date_start_sim' in line and sim_start:
            updated_lines.append(f"'{sim_start}'                     ! date_start_sim     = date start simulation\n")
        elif stripped.startswith("'") and 'date_end_sim' in line and sim_end:
            updated_lines.append(f"'{sim_end}'                     ! date_end_sim       = date end simulation\n")
        elif stripped.startswith("'") and 'date_start_eval' in line and eval_start:
            updated_lines.append(f"'{eval_start}'                     ! date_start_eval    = date start evaluation period\n")
        elif stripped.startswith("'") and 'date_end_eval' in line and eval_end:
            updated_lines.append(f"'{eval_end}'                     ! date_end_eval      = date end evaluation period\n")
        else:
            updated_lines.append(line)

    return updated_lines
