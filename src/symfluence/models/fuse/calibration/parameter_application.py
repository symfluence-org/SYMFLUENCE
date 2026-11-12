# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Parameter Application

Standalone functions for applying calibration parameters to FUSE files:
- para_def.nc (NetCDF parameter definition file)
- fuse_zConstraints_snow.txt (Fortran fixed-width constraints file)
- Regionalized parameters via transfer functions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

from symfluence.core.mixins.config import resolve_config_value

logger = logging.getLogger(__name__)


def parse_fuse_constraints_defaults(constraints_path: Path) -> Dict[str, float]:
    """
    Parse default parameter values from a FUSE constraints file.

    The constraints file has Fortran fixed-width format:
        T/F  stoch  default  lower  upper  ...  name  child1  child2

    Returns:
        Dictionary mapping parameter name -> default value
    """
    defaults = {}
    try:
        with open(constraints_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header, empty lines, comments, and description lines
                if not line or line.startswith('(') or line.startswith('!') or line.startswith('*'):
                    continue
                parts = line.split()
                if len(parts) < 14 or parts[0] not in ('T', 'F'):
                    continue
                # Format: fit stoch default lower upper offset scale ...  name child1 child2
                try:
                    default_val = float(parts[2])
                    param_name = parts[13]
                    if not param_name.startswith('NO_'):
                        defaults[param_name] = default_val
                except (ValueError, IndexError):
                    continue
    except (OSError, IOError):
        pass
    return defaults


def update_para_def_nc(
    para_def_path: Path,
    params: Dict[str, float],
    log: Optional[logging.Logger] = None
) -> Set[str]:
    """
    Update FUSE para_def.nc file with new parameter values.

    If the 'par' dimension has 0 records (FUSE silent failure), this method
    will initialize a single record with the provided parameter values.

    Args:
        para_def_path: Path to para_def.nc file
        params: Parameter values to apply
        log: Logger instance

    Returns:
        Set of parameter names that were updated
    """
    import netCDF4 as nc

    log = log or logger
    params_updated: set = set()

    try:
        with nc.Dataset(para_def_path, 'r+') as ds:
            if 'par' not in ds.dimensions:
                log.error(f"Missing 'par' dimension in {para_def_path}")
                return params_updated

            par_size = ds.dimensions['par'].size
            if par_size == 0:
                # FUSE created the file with UNLIMITED par but wrote 0 records.
                # Initialize a single record so parameters can be written.
                if not ds.dimensions['par'].isunlimited():
                    log.error(f"Empty fixed 'par' dimension in {para_def_path} - cannot resize")
                    return params_updated

                log.warning(f"Empty 'par' dimension in {para_def_path} - initializing 1 record")

                # Write a default value (0.0) for all variables to create the record
                for vname in ds.variables:
                    if vname == 'par':
                        continue
                    var = ds.variables[vname]
                    if 'par' in var.dimensions:
                        var[0] = 0.0

                # Set the par coordinate
                if 'par' in ds.variables:
                    ds.variables['par'][0] = 0

                ds.sync()
                par_size = 1

            for param_name, value in params.items():
                if param_name in ds.variables:
                    try:
                        before = float(ds.variables[param_name][0])
                        ds.variables[param_name][0] = float(value)
                        after = float(ds.variables[param_name][0])
                        log.debug(f"  NC: {param_name}: {before:.4f} -> {after:.4f}")
                        params_updated.add(param_name)
                    except (IndexError, ValueError, TypeError) as e:
                        log.warning(f"Error updating {param_name} in NetCDF: {e}")
                else:
                    log.debug(f"  NC: {param_name} not in file (may be structure param)")

            ds.sync()

    except (OSError, IOError) as e:
        log.error(f"I/O error updating {para_def_path}: {e}")
    except (KeyError, ValueError) as e:
        log.error(f"Data error updating {para_def_path}: {e}")

    # Recompute derived parameters (MAXTENS, MAXFREE, etc.) from base params.
    # FUSE's run_pre mode reads ALL values from para_def.nc — if derived
    # parameters are stale or zero, the model produces garbage output.
    if params_updated:
        compute_derived_parameters(para_def_path, log)
        _verify_para_def_write(para_def_path, params, params_updated, log)

    return params_updated


def compute_derived_parameters(
    para_def_path: Path,
    log: Optional[logging.Logger] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Recompute FUSE derived parameters from base calibration parameters.

    FUSE's run_pre mode reads ALL parameters from para_def.nc including
    derived quantities like MAXTENS_1, MAXFREE_1, etc. These must be
    consistent with base parameters (MAXWATR_1, FRACTEN, etc.) or the
    model produces garbage output.

    This function also sets reasonable numerix defaults for parameters
    that control the numerical solver (SOLUTION, ERRITERFUNC, etc.).

    Must be called after updating base calibration parameters.

    Args:
        para_def_path: Path to para_def.nc file
        log: Logger instance
        config: Optional config dict (passes FUSE_SOLUTION_METHOD etc. through)
    """
    import netCDF4 as nc

    log = log or logger

    try:
        with nc.Dataset(para_def_path, 'r+') as ds:
            if 'par' not in ds.dimensions or ds.dimensions['par'].size == 0:
                return

            def _get(name: str, default: float = 0.0) -> float:
                if name in ds.variables:
                    return float(ds.variables[name][0])
                return default

            def _set(name: str, value: float) -> None:
                if name in ds.variables:
                    ds.variables[name][0] = value

            # --- Derived storage parameters ---
            # Upper layer: tension + free = total
            maxwatr_1 = _get('MAXWATR_1')
            fracten = _get('FRACTEN', 0.5)
            maxtens_1 = maxwatr_1 * fracten
            maxfree_1 = maxwatr_1 * (1.0 - fracten)
            _set('MAXTENS_1', maxtens_1)
            _set('MAXFREE_1', maxfree_1)
            # Sub-zone splits (equal split for general case)
            _set('MAXTENS_1A', maxtens_1 * 0.5)
            _set('MAXTENS_1B', maxtens_1 * 0.5)

            # Lower layer
            maxwatr_2 = _get('MAXWATR_2')
            maxtens_2 = maxwatr_2 * fracten
            maxfree_2 = maxwatr_2 * (1.0 - fracten)
            _set('MAXTENS_2', maxtens_2)
            _set('MAXFREE_2', maxfree_2)
            _set('MAXFREE_2A', maxfree_2 * 0.5)
            _set('MAXFREE_2B', maxfree_2 * 0.5)

            # Routing fraction consistency
            rtfrac1 = _get('RTFRAC1', 0.5)
            _set('RTFRAC2', 1.0 - rtfrac1)

            # --- Rainfall error safety ---
            # RFERR_MLT is a multiplicative rainfall factor. If 0, all precip
            # is zeroed out producing no streamflow. Must be ~1.0.
            rferr_mlt = _get('RFERR_MLT', 1.0)
            if rferr_mlt == 0.0:
                _set('RFERR_MLT', 1.0)
                log.warning("RFERR_MLT was 0.0 (would zero all precipitation), reset to 1.0")

            # --- Numerix defaults (critical for stable runs) ---
            _enforce_numerix_defaults(ds, log, config=config)

            ds.sync()

            log.debug(
                f"Derived params: MAXTENS_1={maxtens_1:.1f}, MAXFREE_1={maxfree_1:.1f}, "
                f"MAXTENS_2={maxtens_2:.1f}, MAXFREE_2={maxfree_2:.1f}"
            )

    except (OSError, IOError) as e:
        log.error(f"Error computing derived parameters: {e}")
    except (KeyError, ValueError) as e:
        log.error(f"Data error computing derived parameters: {e}")


def _verify_para_def_write(
    para_def_path: Path,
    params: Dict[str, float],
    params_updated: Set[str],
    log: logging.Logger
) -> None:
    """Verify that para_def.nc write succeeded by reading back a value."""
    import netCDF4 as nc

    try:
        with nc.Dataset(para_def_path, 'r') as ds:
            first_param = next(iter(params_updated))
            if first_param in ds.variables:
                actual_value = float(ds.variables[first_param][0])
                expected_value = params[first_param]
                # Tolerance of 1e-3 to match FUSE's Fortran F9.3 format
                if abs(actual_value - expected_value) > 1e-3:
                    log.warning(
                        f"Parameter write verification: {first_param} expected {expected_value:.6f} "
                        f"but file contains {actual_value:.6f} (diff={abs(actual_value - expected_value):.2e})"
                    )
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.debug(f"Could not verify para_def.nc write: {e}", exc_info=True)


def apply_regionalization(
    para_def_path: Path,
    calibration_params: Dict[str, float],
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Set[str]:
    """
    Apply parameter regionalization to generate spatially distributed parameters.

    Supports multiple regionalization methods:
    - lumped: Uniform parameters across all subcatchments
    - transfer_function: Power-law functions based on catchment attributes
    - zones: Shared parameters within predefined zones
    - distributed: Independent parameters per subcatchment

    Args:
        para_def_path: Path to para_def.nc file
        calibration_params: Calibration parameter/coefficient values
        config: Configuration dictionary
        log: Logger instance

    Returns:
        Set of parameter names that were updated
    """
    import netCDF4 as nc
    import pandas as pd

    from symfluence.core.calibration.regionalization.strategies import RegionalizationFactory
    from symfluence.models.fuse.calibration.parameter_regionalization import (
        FUSE_DEFAULT_PARAM_CONFIG,
    )

    log = log or logger
    params_updated: set = set()

    try:
        # Get regionalization method
        method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
        if resolve_config_value(
            config,
            lambda: config.model.fuse.use_transfer_functions,
            dict_key='USE_TRANSFER_FUNCTIONS',
        ) and method == 'lumped':
            method = 'transfer_function'

        # Get original parameter bounds
        param_bounds = resolve_config_value(config, lambda: config.model.fuse.param_bounds, {}, 'FUSE_PARAM_BOUNDS')
        if not param_bounds:
            log.error("FUSE_PARAM_BOUNDS not configured")
            return params_updated

        param_bounds_tuples = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in param_bounds.items()
        }

        # Load attributes if needed
        attributes = None
        if method == 'transfer_function':
            attributes_path = resolve_config_value(config, lambda: config.model.summa.transfer_function_attributes_path, None, 'TRANSFER_FUNCTION_ATTRIBUTES')
            if attributes_path:
                attributes = pd.read_csv(attributes_path)
            else:
                log.error("TRANSFER_FUNCTION_ATTRIBUTES not configured")
                return params_updated

        # Determine number of subcatchments
        if attributes is not None:
            n_subcatchments = len(attributes)
        else:
            with nc.Dataset(para_def_path, 'r') as ds:
                n_subcatchments = ds.dimensions['par'].size

        # Inject FUSE default param config if not already in config
        if 'TRANSFER_FUNCTION_PARAM_CONFIG' not in config:
            config = dict(config, TRANSFER_FUNCTION_PARAM_CONFIG=FUSE_DEFAULT_PARAM_CONFIG)

        # Create regionalization strategy
        regionalization = RegionalizationFactory.create(
            method=method,
            param_bounds=param_bounds_tuples,
            n_units=n_subcatchments,
            config=config,
            attributes=attributes,
            logger=log
        )

        log.debug(f"Using '{regionalization.name}' parameter regionalization")

        # Convert calibration parameters to distributed values
        param_array, param_names = regionalization.to_distributed(calibration_params)

        log.debug(
            f"Regionalization: {len(param_names)} params x {n_subcatchments} subcatchments"
        )

        # Read existing file structure for potential resize
        _resize_para_def_if_needed(para_def_path, n_subcatchments, log)

        # Write the distributed values
        with nc.Dataset(para_def_path, 'r+') as ds:
            for i, param_name in enumerate(param_names):
                if param_name not in ds.variables:
                    continue

                values = param_array[:, i]
                ds.variables[param_name][:] = values
                log.debug(
                    f"  {param_name}: distributed [{values.min():.3f}, {values.max():.3f}]"
                )
                params_updated.add(param_name)

            # Ensure numerical solver settings are reasonable
            _enforce_numerix_defaults(ds, log, config=config)
            ds.sync()

        if params_updated:
            log.debug(
                f"Applied regionalization to {len(params_updated)} parameters: "
                f"{', '.join(sorted(params_updated))}"
            )

    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(f"Error applying transfer functions: {e}", exc_info=True)
        import traceback
        log.error(traceback.format_exc())

    return params_updated


def _resize_para_def_if_needed(
    para_def_path: Path,
    n_subcatchments: int,
    log: logging.Logger
) -> None:
    """Resize para_def.nc if par dimension doesn't match n_subcatchments."""
    import shutil

    import netCDF4 as nc

    with nc.Dataset(para_def_path, 'r') as ds:
        par_size = ds.dimensions['par'].size
        if par_size == n_subcatchments:
            return

        # Read existing data
        existing_vars = {}
        for vname in ds.variables:
            if vname == 'par':
                continue
            existing_vars[vname] = {
                'values': ds.variables[vname][:].copy(),
                'attrs': {a: ds.variables[vname].getncattr(a)
                          for a in ds.variables[vname].ncattrs()},
            }
        global_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}

    log.debug(f"Resizing para_def.nc: par={par_size} -> {n_subcatchments}")

    tmp_path = para_def_path.with_suffix('.tmp.nc')
    with nc.Dataset(tmp_path, 'w', format='NETCDF4') as ds_new:
        ds_new.createDimension('par', n_subcatchments)
        ds_new.createVariable('par', 'i4', ('par',))
        ds_new.variables['par'][:] = np.arange(n_subcatchments)
        for attr_name, attr_val in global_attrs.items():
            ds_new.setncattr(attr_name, attr_val)

        for vname, vinfo in existing_vars.items():
            fill_value = vinfo['attrs'].get('_FillValue', None)
            ds_new.createVariable(vname, 'f8', ('par',), fill_value=fill_value)
            ds_new.variables[vname][:] = np.full(
                n_subcatchments, float(vinfo['values'][0])
            )
            for attr_name, attr_val in vinfo['attrs'].items():
                if attr_name == '_FillValue':
                    continue
                ds_new.variables[vname].setncattr(attr_name, attr_val)

    shutil.move(str(tmp_path), str(para_def_path))


def _enforce_numerix_defaults(
    ds: Any,
    log: logging.Logger,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Ensure numerical solver settings are reasonable in para_def.nc.

    Args:
        ds: Open netCDF4 Dataset (read-write)
        log: Logger instance
        config: Optional config dict. If FUSE_SOLUTION_METHOD or
                FUSE_TIMESTEP_TYPE are set, those values are used instead
                of the hardcoded defaults (0=explicit Euler, 0=fixed).
    """
    # Determine solver settings from config or use defaults
    solution_method = 0.0  # Explicit Euler (fastest)
    timestep_type = 0.0    # Fixed time steps

    if config:
        cfg_solution = resolve_config_value(config, lambda: config.model.fuse.solution_method, None, 'FUSE_SOLUTION_METHOD')
        if cfg_solution is not None:
            solution_method = float(cfg_solution)
            log.debug(f"  Using configured FUSE_SOLUTION_METHOD={int(solution_method)}")

        cfg_timestep = resolve_config_value(config, lambda: config.model.fuse.timestep_type, None, 'FUSE_TIMESTEP_TYPE')
        if cfg_timestep is not None:
            timestep_type = float(cfg_timestep)
            log.debug(f"  Using configured FUSE_TIMESTEP_TYPE={int(timestep_type)}")

    numerix_defaults = {
        'SOLUTION': solution_method,
        'TIMSTEP_TYP': timestep_type,
        'ERRITERFUNC': 1e-4,
        'ERR_ITER_DX': 1e-4,
        'NITER_TOTAL': 5000.0,
        'MIN_TSTEP': 0.001 / 1440.0,
    }
    for nvar, nval in numerix_defaults.items():
        if nvar in ds.variables:
            cur = float(ds.variables[nvar][0])
            if nvar in ('SOLUTION', 'TIMSTEP_TYP'):
                if cur != nval:
                    ds.variables[nvar][:] = nval
                    log.debug(f"  Set {nvar}: {cur:.0f} -> {nval:.0f}")
            elif cur < nval * 0.01:
                ds.variables[nvar][:] = nval
                log.debug(f"  Relaxed {nvar}: {cur:.2e} -> {nval:.2e}")


def update_constraints_file(
    constraints_file: Path,
    params: Dict[str, float],
    log: Optional[logging.Logger] = None
) -> Set[str]:
    """
    Update FUSE constraints file with new parameter default values.

    FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
    The default value column starts at position 4 and is exactly 9 characters.

    Args:
        constraints_file: Path to constraints file
        params: Parameter values to apply
        log: Logger instance

    Returns:
        Set of parameter names that were updated
    """
    log = log or logger
    params_updated: set = set()

    try:
        lines = _read_constraints_file(constraints_file, log)

        # Fortran format: default value column at position 4-12 (9 chars, F9.3)
        DEFAULT_VALUE_START = 4
        DEFAULT_VALUE_WIDTH = 9

        updated_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('(') or stripped.startswith('*') or stripped.startswith('!'):
                updated_lines.append(line)
                continue

            updated = False
            for param_name, value in params.items():
                parts = line.split()
                if len(parts) >= 14 and param_name in parts:
                    if parts[13] == param_name:
                        new_value = f"{value:9.3f}"
                        if len(line) > DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:
                            new_line = (
                                line[:DEFAULT_VALUE_START] +
                                new_value +
                                line[DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:]
                            )
                            updated_lines.append(new_line)
                            params_updated.add(param_name)
                            updated = True
                            break

            if not updated:
                updated_lines.append(line)

        with open(constraints_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

    except (OSError, IOError) as e:
        log.warning(f"I/O error updating constraints file: {e}")
    except (IndexError, ValueError) as e:
        log.warning(f"Format error updating constraints file: {e}")

    return params_updated


def _read_constraints_file(constraints_file: Path, log: logging.Logger) -> list:
    """Read constraints file with encoding fallback."""
    try:
        with open(constraints_file, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        log.warning(
            f"UTF-8 decode error reading {constraints_file}, falling back to latin-1"
        )
        with open(constraints_file, 'r', encoding='latin-1') as f:
            return f.readlines()
