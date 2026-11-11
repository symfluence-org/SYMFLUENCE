# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared writers for the files SUMMA calibration modifies.

``SUMMAParameterManager.update_model_files`` and the process-pool worker
(``worker_impl.parameter_application``) both apply a trial parameter set by
writing ``trialParams.nc``, rescaling the soil layers in ``coldState.nc`` and
editing mizuRoute's ``param.nml.default``. They used to carry their own copies
of that logic and the copies drifted (different pseudo-parameter filtering,
different handling of ``total_soil_depth_multiplier``, hard-coded file names in
one of them). Everything that touches those files lives here, once.
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np
import xarray as xr

from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.profiling import get_profiler

from .forcing_scaling import FORCING_PARAMS

DEFAULT_ATTRIBUTES_FILE = 'attributes.nc'
DEFAULT_COLDSTATE_FILE = 'coldState.nc'
DEFAULT_TRIALPARAMS_FILE = 'trialParams.nc'

# Soil-depth pseudo-parameters: they rescale coldState layers, SUMMA has no
# parameter of that name.
DEPTH_MULTIPLIER_PARAM = 'total_soil_depth_multiplier'
LEGACY_DEPTH_MULTIPLIER_PARAM = 'total_mult'
DEPTH_SHAPE_PARAM = 'shape_factor'
DEPTH_PSEUDO_PARAMS: Tuple[str, ...] = (
    DEPTH_MULTIPLIER_PARAM, LEGACY_DEPTH_MULTIPLIER_PARAM, DEPTH_SHAPE_PARAM,
)

# Parameters SUMMA reads on the gru dimension (everything in basinParamInfo.txt).
GRU_LEVEL_PARAMS: Tuple[str, ...] = (
    'routingGammaShape', 'routingGammaScale',
    'basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond',
)

# Every parameter mizuRoute calibration can address in param.nml.default.
MIZUROUTE_PARAMS: Tuple[str, ...] = ('velo', 'diff', 'mann_n', 'wscale', 'fshape', 'tscale')
_INTEGER_MIZUROUTE_PARAMS = ('tscale',)


class SummaSettingsFiles(NamedTuple):
    """Names of the settings files calibration reads and rewrites."""

    attributes: str
    coldstate: str
    trialparams: str


def _file_name(value: Any, default: str) -> str:
    return default if value in (None, '', 'default') else str(value)


# fileManager keys naming the three files calibration touches.
_FILE_MANAGER_FIELDS = {
    'attributeFile': 'attributes',
    'initConditionFile': 'coldstate',
    'trialParamFile': 'trialparams',
}
_FILE_MANAGER_ENTRY = re.compile(r"""^\s*(\w+)\s+(['"])(.*?)\2""")
DEFAULT_FILE_MANAGER = 'fileManager.txt'


def read_file_manager_names(file_manager_path: Path) -> Dict[str, str]:
    """Return the attributes / coldstate / trialparams names a file manager sets.

    Keys are :class:`SummaSettingsFiles` field names; entries the file manager
    does not set (or sets to an empty string) are absent. A missing or
    unreadable file manager gives an empty dict.
    """
    names: Dict[str, str] = {}
    try:
        with open(file_manager_path, 'r', encoding='utf-8') as fm:
            for line in fm:
                match = _FILE_MANAGER_ENTRY.match(line)
                if match and match.group(1) in _FILE_MANAGER_FIELDS and match.group(3).strip():
                    names[_FILE_MANAGER_FIELDS[match.group(1)]] = match.group(3).strip()
    except OSError:
        return {}
    return names


def file_manager_name(config: Any) -> str:
    """The configured SUMMA file manager name (``SETTINGS_SUMMA_FILEMANAGER``)."""
    return _file_name(resolve_config_value(
        config, lambda: config.model.summa.filemanager, None, 'SETTINGS_SUMMA_FILEMANAGER'),
        DEFAULT_FILE_MANAGER)


def resolve_settings_file_names(config: Any, settings_dir: Optional[Path] = None) -> SummaSettingsFiles:
    """Resolve the attributes / coldState / trialParams file names.

    Works for typed configs and for the flat dicts calibration workers get.

    SUMMA reads whatever its file manager names, so when ``settings_dir`` is
    given and holds the configured file manager, the names come from that file
    manager. Config (``SETTINGS_SUMMA_ATTRIBUTES`` / ``_COLDSTATE`` /
    ``_TRIALPARAMS``) fills in anything the file manager does not set, and is
    the only source without ``settings_dir``. For a file manager SYMFLUENCE
    generated the two agree; they differ for a hand-written one, typically a
    glacier setup (``fileManager_glac.txt`` naming ``attributes_glac.nc`` /
    ``coldState_glac.nc``) whose config still carries the default names --
    calibration then wrote soil depths and trial parameters to files SUMMA
    never opened.
    """
    configured = SummaSettingsFiles(
        attributes=_file_name(resolve_config_value(
            config, lambda: config.model.summa.attributes, None, 'SETTINGS_SUMMA_ATTRIBUTES'),
            DEFAULT_ATTRIBUTES_FILE),
        coldstate=_file_name(resolve_config_value(
            config, lambda: config.model.summa.coldstate, None, 'SETTINGS_SUMMA_COLDSTATE'),
            DEFAULT_COLDSTATE_FILE),
        trialparams=_file_name(resolve_config_value(
            config, lambda: config.model.summa.trialparams, None, 'SETTINGS_SUMMA_TRIALPARAMS'),
            DEFAULT_TRIALPARAMS_FILE),
    )
    if settings_dir is None:
        return configured
    from_file_manager = read_file_manager_names(Path(settings_dir) / file_manager_name(config))
    return configured._replace(**from_file_manager)


def _record_error(message: str, logger, debug_info: Optional[Dict]) -> None:
    logger.error(message)
    if debug_info is not None:
        debug_info.setdefault('errors', []).append(message)


def _record_file(entry: str, debug_info: Optional[Dict]) -> None:
    if debug_info is not None:
        debug_info.setdefault('files_checked', []).append(entry)


# ----------------------------------------------------------------------------
# Splitting a trial parameter set by destination
# ----------------------------------------------------------------------------

def select_trial_params(params: Dict[str, Any], mizuroute_params: Iterable[str] = ()) -> Dict[str, Any]:
    """Return the entries of ``params`` that belong in trialParams.nc.

    Soil-depth multipliers, mizuRoute parameters (the configured ones and
    every name mizuRoute calibration knows) and forcing pseudo-parameters are
    applied elsewhere and are never SUMMA parameters.
    """
    excluded = set(DEPTH_PSEUDO_PARAMS) | set(FORCING_PARAMS) | set(mizuroute_params) | set(MIZUROUTE_PARAMS)
    return {name: value for name, value in params.items() if name not in excluded}


def has_depth_params(params: Dict[str, Any]) -> bool:
    """Whether ``params`` carries any soil-depth pseudo-parameter."""
    return any(name in params for name in DEPTH_PSEUDO_PARAMS)


# ----------------------------------------------------------------------------
# Soil depths (coldState)
# ----------------------------------------------------------------------------

def _first(value: Any) -> Any:
    return value[0] if isinstance(value, np.ndarray) else value


def depth_multipliers(params: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(total_mult, shape_factor)`` from a trial parameter set.

    ``total_soil_depth_multiplier`` is the current name of the total-depth
    multiplier and ``total_mult`` the legacy one; either is accepted, and a
    missing multiplier or shape factor is neutral (1.0).
    """
    total_mult = params.get(DEPTH_MULTIPLIER_PARAM)
    if total_mult is None:
        total_mult = params.get(LEGACY_DEPTH_MULTIPLIER_PARAM, 1.0)
    shape_factor = params.get(DEPTH_SHAPE_PARAM, 1.0)
    return _first(total_mult), _first(shape_factor)


def calculate_soil_depths(original_depths: np.ndarray, total_mult: float, shape_factor: float) -> np.ndarray:
    """Rescale soil layer depths.

    ``shape_factor`` redistributes thickness exponentially over the layers
    (>1 thickens the deep layers, <1 the shallow ones) with weights that
    average 1; ``total_mult`` then scales every layer.
    """
    arr = np.asarray(original_depths).copy()
    n = len(arr)
    idx = np.arange(n)

    if shape_factor > 1:
        w = np.exp(idx / (n - 1) * np.log(shape_factor))
    elif shape_factor < 1:
        w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
    else:
        w = np.ones(n)

    w /= w.mean()
    return arr * w * total_mult


def update_soil_depths(
    coldstate_path: Path,
    original_depths: Sequence[float],
    params: Dict[str, Any],
    logger,
    debug_info: Optional[Dict] = None,
    component: str = 'summa_worker',
) -> bool:
    """Write rescaled ``mLayerDepth`` / ``iLayerHeight`` into the coldState file."""
    try:
        total_mult, shape_factor = depth_multipliers(params)
        logger.debug(f"Updating soil depths: total_mult={total_mult:.3f}, shape_factor={shape_factor:.3f}")

        new_depths = calculate_soil_depths(np.array(original_depths), total_mult, shape_factor)

        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]

        coldstate_path = Path(coldstate_path)
        if not coldstate_path.exists():
            _record_error(f"{coldstate_path.name} not found: {coldstate_path}", logger, debug_info)
            return False

        _record_file(f"{coldstate_path.name}: {coldstate_path}", debug_info)

        # Track coldState write (secondary IOPS bottleneck)
        with get_profiler().track_netcdf_write(str(coldstate_path), component=component):
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    _record_error(
                        f"Required depth variables not found in {coldstate_path.name}", logger, debug_info)
                    return False

                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights

        logger.debug("Soil depths updated successfully")
        return True

    except (FileNotFoundError, IOError, ValueError) as e:
        _record_error(f"Error updating soil depths: {str(e)}", logger, debug_info)
        return False


# ----------------------------------------------------------------------------
# mizuRoute parameters (param.nml.default)
# ----------------------------------------------------------------------------

def update_mizuroute_params(
    param_file: Path,
    mizuroute_params: Iterable[str],
    params: Dict[str, Any],
    logger,
    debug_info: Optional[Dict] = None,
    component: str = 'summa_worker',
) -> bool:
    """Substitute calibrated values into mizuRoute's parameter namelist."""
    profiler = get_profiler()
    try:
        param_file = Path(param_file)
        if not param_file.exists():
            logger.warning(f"mizuRoute param file not found: {param_file}")
            return True

        _record_file(f"mizuRoute param file: {param_file}", debug_info)

        with profiler.track_file_read(str(param_file), component=component):
            with open(param_file, 'r', encoding='utf-8') as f:
                content = f.read()

        updated_content = content
        for param_name in mizuroute_params:
            if param_name in params:
                param_value = float(np.asarray(params[param_name]).ravel()[0])
                pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'

                if param_name in _INTEGER_MIZUROUTE_PARAMS:
                    replacement = rf'\g<1>{int(param_value)}'
                else:
                    replacement = rf'\g<1>{param_value:.6f}'

                updated_content = re.sub(pattern, replacement, updated_content)
                logger.debug(f"Updated {param_name} = {param_value}")

        with profiler.track_file_write(str(param_file), size_bytes=len(updated_content), component=component):
            with open(param_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)

        logger.debug("mizuRoute parameters updated successfully")
        return True

    except (FileNotFoundError, IOError, ValueError) as e:
        _record_error(f"Error updating mizuRoute params: {str(e)}", logger, debug_info)
        return False


# ----------------------------------------------------------------------------
# trialParams
# ----------------------------------------------------------------------------

def _can_update_inplace(trial_params_path: Path, params: Dict, logger) -> bool:
    """Check if we can update the trialParams file in-place.

    Returns True if the file exists and contains all required parameters.
    """
    if not trial_params_path.exists():
        return False

    try:
        with nc.Dataset(trial_params_path, 'r') as ds:
            for param_name in params.keys():
                if param_name not in ds.variables:
                    logger.debug(f"Parameter {param_name} not in existing file, need full recreation")
                    return False
        return True
    except (FileNotFoundError, IOError, ValueError) as e:
        logger.debug(f"Cannot read existing trialParams for in-place update: {e}")
        return False


def _update_trial_params_inplace(trial_params_path: Path, params: Dict, logger,
                                 debug_info: Optional[Dict], profiler, component: str) -> bool:
    """Update parameter values in-place in an existing trialParams file.

    This is much faster than recreating the file because it only updates
    the data values without recreating the file structure.
    """
    max_retries = 3
    base_delay = 0.1

    for attempt in range(max_retries):
        try:
            with profiler.track_netcdf_write(str(trial_params_path), component=f"{component}_inplace"):
                with nc.Dataset(trial_params_path, 'r+') as ds:
                    for param_name, param_values in params.items():
                        if param_name in ds.variables:
                            param_values_array = np.atleast_1d(np.asarray(param_values))
                            if param_values_array.ndim > 1:
                                param_values_array = param_values_array.flatten()

                            var = ds.variables[param_name]
                            var_size = var.size

                            if len(param_values_array) >= var_size:
                                var[:] = param_values_array[:var_size]
                            else:
                                var[:] = param_values_array[0]

                    ds.sync()

            logger.debug(f"Updated {len(params)} parameters in-place")
            return True

        except (FileNotFoundError, IOError, ValueError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue

            error_msg = f"Failed in-place update after {max_retries} attempts: {e}"
            logger.warning(error_msg)
            if debug_info is not None:
                debug_info.setdefault('errors', []).append(error_msg)
            return False

    return False


def write_trial_params(
    trial_params_path: Path,
    attributes_path: Path,
    params: Dict[str, Any],
    logger,
    debug_info: Optional[Dict] = None,
    gru_level_params: Iterable[str] = (),
    component: str = 'summa_worker',
) -> bool:
    """Write SUMMA trial parameters, updating in place when the layout allows.

    ``params`` must already be restricted to SUMMA parameters (see
    :func:`select_trial_params`). If the file exists and holds every
    parameter, only the values are rewritten; otherwise the file is recreated
    through a temporary file and an atomic rename. ``gru_level_params`` adds
    to :data:`GRU_LEVEL_PARAMS` (e.g. the configured basin parameters).
    """
    profiler = get_profiler()
    trial_params_path = Path(trial_params_path)
    attr_file_path = Path(attributes_path)

    try:
        if not params:
            logger.debug("No hydrological parameters to write")
            return True

        if not attr_file_path.exists():
            _record_error(f"Attributes file not found: {attr_file_path}", logger, debug_info)
            return False

        _record_file(f"{attr_file_path.name}: {attr_file_path}", debug_info)

        if _can_update_inplace(trial_params_path, params, logger):
            return _update_trial_params_inplace(
                trial_params_path, params, logger, debug_info, profiler, component)

        gru_level: List[str] = list(GRU_LEVEL_PARAMS) + [p for p in gru_level_params if p not in GRU_LEVEL_PARAMS]

        # Full recreation with retries: several workers may share a filesystem.
        max_retries = 5
        base_delay = 0.1

        for attempt in range(max_retries):
            temp_path = trial_params_path.with_suffix(f'.tmp_{os.getpid()}_{random.randint(1000,9999)}')
            try:
                logger.debug(f"Attempt {attempt + 1}: Writing trial parameters to {temp_path}")

                with profiler.track_file_read(str(attr_file_path), component=component):
                    with xr.open_dataset(attr_file_path) as ds:
                        num_hrus = ds.sizes.get('hru', 1)
                        num_grus = ds.sizes.get('gru', 1)
                        hru_ids = ds['hruId'].values if 'hruId' in ds else np.arange(1, num_hrus + 1)
                        gru_ids = ds['gruId'].values if 'gruId' in ds else np.array([1])

                logger.debug(f"Writing parameters for {num_hrus} HRUs, {num_grus} GRUs")

                # Track NetCDF write (PRIMARY IOPS BOTTLENECK)
                with profiler.track_netcdf_write(str(trial_params_path), component=component):
                    with nc.Dataset(temp_path, 'w', format='NETCDF4') as output_ds:
                        output_ds.createDimension('hru', num_hrus)
                        output_ds.createDimension('gru', num_grus)

                        # zlib level 4 is a good balance of speed/compression
                        compress_opts = {'zlib': True, 'complevel': 4, 'shuffle': True}

                        hru_var = output_ds.createVariable(  # type: ignore[call-overload]
                            'hruId', 'i4', ('hru',), fill_value=-9999, **compress_opts)
                        hru_var[:] = hru_ids

                        gru_var = output_ds.createVariable(  # type: ignore[call-overload]
                            'gruId', 'i4', ('gru',), fill_value=-9999, **compress_opts)
                        gru_var[:] = gru_ids

                        for param_name, param_values in params.items():
                            param_values_array = np.atleast_1d(np.asarray(param_values))

                            if param_values_array.ndim > 1:
                                param_values_array = param_values_array.flatten()

                            if param_name in gru_level:
                                param_var = output_ds.createVariable(  # type: ignore[call-overload]
                                    param_name, 'f8', ('gru',), fill_value=np.nan, **compress_opts)
                                param_var.long_name = f"Trial value for {param_name}"

                                if len(param_values_array) >= num_grus:
                                    param_var[:] = param_values_array[:num_grus]
                                else:
                                    param_var[:] = param_values_array[0]
                            else:
                                param_var = output_ds.createVariable(  # type: ignore[call-overload]
                                    param_name, 'f8', ('hru',), fill_value=np.nan, **compress_opts)
                                param_var.long_name = f"Trial value for {param_name}"

                                if len(param_values_array) == num_hrus:
                                    param_var[:] = param_values_array
                                elif len(param_values_array) == 1:
                                    param_var[:] = param_values_array[0]
                                else:
                                    param_var[:] = param_values_array[:num_hrus]

                            logger.debug(f"Added parameter {param_name} with shape {param_var.shape}")

                # Atomically move temporary file to final location
                try:
                    if os.name != 'nt':
                        # Security rationale: Group-writable for HPC shared access
                        os.chmod(temp_path, 0o664)  # nosec B103
                    temp_path.replace(trial_params_path)
                    logger.debug(f"Trial parameters file created successfully: {trial_params_path}")
                    _record_file(f"{trial_params_path.name} (created): {trial_params_path}", debug_info)
                    return True
                except (FileNotFoundError, IOError, ValueError) as move_error:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise move_error

            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")

                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except (OSError, PermissionError):
                        pass  # Best-effort cleanup, non-critical

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                else:
                    _record_error(
                        f"Failed to generate trial params after {max_retries} attempts: {str(e)}",
                        logger, debug_info)
                    return False

        return False

    except (FileNotFoundError, IOError, ValueError) as e:
        _record_error(f"Error generating trial params: {str(e)}", logger, debug_info)
        return False
