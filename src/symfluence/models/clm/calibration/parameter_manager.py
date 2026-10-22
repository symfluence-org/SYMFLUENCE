# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Parameter Manager

Handles CLM5 parameter bounds, normalization, and file updates.
Parameters act on 3 target files:
- namelist (lnd_in, mirrored in user_nl_clm): hydrology scalar knobs
- params.nc (clm5_params.nc): snow + PFT parameters
- surfdata (surfdata_clm.nc): FMAX, organic matter and soil texture

CTSM 5.3 derives watsat, hksat, sucsat and bsw from PCT_SAND, PCT_CLAY and
ORGANIC (pedotransfer functions); it never reads those four fields from
surfdata, so soil hydraulics are calibrated through texture, not multipliers.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.exceptions import ConfigurationError
from symfluence.core.registries import R

# CLM5 parameter definitions:
# (target_file, nc_variable_or_None, transform)
# target_file: 'namelist' | 'params' | 'surfdata'
# For 'namelist' params, nc_variable stores the lnd_in section name.
# Every surfdata target must be a field CTSM actually reads (see
# CLM_SURFDATA_FIELDS_READ); *_offset params add percentage points.
CLM_PARAM_DEFS = {
    # --- Hydrology (params.nc scalars) ---
    'baseflow_scalar':  ('namelist', 'soilhydrology_inparm', 'log'),
    'fff':              ('params', 'fff', 'linear'),
    'wimp':             ('params', 'wimp', 'linear'),
    'ksatdecay':        ('params', 'pc', 'log'),
    'n_baseflow':       ('params', 'n_baseflow', 'linear'),
    'e_ice':            ('params', 'e_ice', 'linear'),
    'perched_baseflow_scalar': ('params', 'perched_baseflow_scalar', 'log'),
    'interception_fraction':   ('params', 'interception_fraction', 'linear'),
    'max_leaf_wetted_frac':    ('params', 'maximum_leaf_wetted_fraction', 'linear'),
    # --- Surfdata (fields CTSM reads; texture drives soil hydraulics) ---
    'fmax':             ('surfdata', 'FMAX', 'linear'),
    'organic_max':      ('surfdata', 'ORGANIC', 'linear'),
    'sand_offset':      ('surfdata', 'PCT_SAND', 'linear'),
    'clay_offset':      ('surfdata', 'PCT_CLAY', 'linear'),
    # --- Snow (params.nc) ---
    'fresh_snw_rds_max': ('params', 'fresh_snw_rds_max', 'linear'),
    'snw_aging_bst':     ('params', 'snw_aging_bst', 'linear'),
    'SNO_Z0MV':          ('params', 'SNO_Z0MV', 'log'),
    'accum_factor':      ('params', 'accum_factor', 'linear'),
    'SNOW_DENSITY_MAX':  ('params', 'SNOW_DENSITY_MAX', 'linear'),
    'SNOW_DENSITY_MIN':  ('params', 'SNOW_DENSITY_MIN', 'linear'),
    'n_melt_coef':       ('params', 'n_melt_coef', 'linear'),
    # --- Snow (namelist in lnd_in) ---
    'int_snow_max':     ('namelist', 'scf_swenson_lawrence_2012_inparm', 'linear'),
    # --- Vegetation/PFT (params.nc, PFT-indexed) ---
    'medlynslope':      ('params', 'medlynslope', 'linear'),
    'slatop':           ('params', 'slatop', 'linear'),
    'flnr':             ('params', 'flnr', 'linear'),
    'froot_leaf':       ('params', 'froot_leaf', 'linear'),
    'stem_leaf':        ('params', 'stem_leaf', 'linear'),
}

# Surface-dataset fields CTSM 5.3 reads that calibration may edit.
CLM_SURFDATA_FIELDS_READ = frozenset({'FMAX', 'ORGANIC', 'PCT_SAND', 'PCT_CLAY'})

# Former parameters that are rejected with an explanation instead of being
# silently accepted.
CLM_REMOVED_PARAMS: Dict[str, str] = {
    **{name: (
        f"'{name}' was removed: CTSM 5.3 computes {name[:-5]} from PCT_SAND/PCT_CLAY/"
        "ORGANIC and never reads it from surfdata, so the multiplier had no effect. "
        "Calibrate soil hydraulics with 'sand_offset'/'clay_offset' (and 'organic_max')."
    ) for name in ('hksat_mult', 'watsat_mult', 'bsw_mult', 'sucsat_mult')},
    'route_k': (
        "'route_k' was removed: it applied a linear reservoir to CLM runoff after the "
        "model ran, which is not CLM physics. Represent slow flow inside CLM "
        "(baseflow_scalar, n_baseflow, e_ice, soil texture, bedrock depth) or route "
        "with a routing model."
    ),
}

# Default bounds for all parameters
CLM_DEFAULT_BOUNDS: Dict[str, Dict[str, Any]] = {
    # Hydrology
    'baseflow_scalar':  {'min': 0.001, 'max': 0.1, 'transform': 'log'},
    'fff':              {'min': 0.02, 'max': 1.0},
    'wimp':             {'min': 0.01, 'max': 0.1},
    'ksatdecay':        {'min': 0.1, 'max': 10.0, 'transform': 'log'},
    'n_baseflow':       {'min': 0.5, 'max': 5.0},
    'e_ice':            {'min': 1.0, 'max': 6.0},
    'perched_baseflow_scalar': {'min': 1e-7, 'max': 1e-3, 'transform': 'log'},
    'interception_fraction':   {'min': 0.2, 'max': 1.0},
    'max_leaf_wetted_frac':    {'min': 0.01, 'max': 0.2},
    # Surfdata
    'fmax':             {'min': 0.0, 'max': 1.0},
    'organic_max':      {'min': 0.0, 'max': 130.0},
    'sand_offset':      {'min': -20.0, 'max': 20.0},
    'clay_offset':      {'min': -10.0, 'max': 10.0},
    # Snow
    'fresh_snw_rds_max': {'min': 50.0, 'max': 200.0},
    'snw_aging_bst':     {'min': 0.0, 'max': 200.0},
    'SNO_Z0MV':          {'min': 0.0001, 'max': 0.01, 'transform': 'log'},
    'accum_factor':      {'min': -0.1, 'max': 0.1},
    'SNOW_DENSITY_MAX':  {'min': 250.0, 'max': 550.0},
    'SNOW_DENSITY_MIN':  {'min': 50.0, 'max': 200.0},
    'n_melt_coef':       {'min': 50.0, 'max': 500.0},
    'int_snow_max':      {'min': 500.0, 'max': 5000.0},
    # Vegetation/PFT
    'medlynslope':      {'min': 2.0, 'max': 12.0},
    'slatop':           {'min': 0.005, 'max': 0.06},
    'flnr':             {'min': 0.05, 'max': 0.25},
    'froot_leaf':       {'min': 0.5, 'max': 3.0},
    'stem_leaf':        {'min': 0.5, 'max': 3.0},
}


def apply_texture_offsets(sand: np.ndarray, clay: np.ndarray,
                          sand_offset: float = 0.0, clay_offset: float = 0.0):
    """Shift PCT_SAND/PCT_CLAY by percentage points, keeping a valid texture.

    Each fraction stays within [0, 100] and sand + clay never exceeds 100
    (clay yields first, since sand is the primary hydraulic control).
    """
    new_sand = np.clip(np.asarray(sand, dtype=float) + sand_offset, 0.0, 100.0)
    new_clay = np.clip(np.asarray(clay, dtype=float) + clay_offset, 0.0, 100.0)
    new_clay = np.minimum(new_clay, 100.0 - new_sand)
    return new_sand, new_clay


def check_removed_params(names: List[str]) -> None:
    """Raise for parameters that no longer act on CLM."""
    removed = [n for n in names if n in CLM_REMOVED_PARAMS]
    if removed:
        raise ConfigurationError(' '.join(CLM_REMOVED_PARAMS[n] for n in removed))


@R.parameter_managers.add('CLM')
class CLMParameterManager(BaseParameterManager):
    """Handles CLM5 parameter bounds, normalization, and file updates."""

    def __init__(
        self,
        config: Dict,
        logger: logging.Logger,
        clm_settings_dir: Path,
    ):
        super().__init__(config, logger, clm_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse parameters to calibrate
        clm_params_str = self._get_config_value(lambda: self.config.model.clm.params_to_calibrate, default=None, dict_key='CLM_PARAMS_TO_CALIBRATE')

        if clm_params_str is None:
            # Default: all parameters
            clm_params_str = ','.join(CLM_DEFAULT_BOUNDS.keys())
            logger.warning(
                f"CLM_PARAMS_TO_CALIBRATE missing; using all {len(CLM_DEFAULT_BOUNDS)} params"
            )

        self.clm_params = [
            p.strip() for p in str(clm_params_str).split(',') if p.strip()
        ]
        check_removed_params(self.clm_params)

        # Setup paths
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.params_dir = self.project_dir / 'settings' / 'CLM' / 'parameters'

        # Get file names
        self.params_nc_name = 'clm5_params.nc'
        self.surfdata_nc_name = 'surfdata_clm.nc'
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'clm'):
                self.params_nc_name = config.model.clm.params_file or self.params_nc_name
                self.surfdata_nc_name = config.model.clm.surfdata_file or self.surfdata_nc_name
        except (AttributeError, TypeError):
            pass

    def _get_parameter_names(self) -> List[str]:
        return self.clm_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return CLM parameter bounds with log transform info."""
        bounds = {}
        for param_name, default in CLM_DEFAULT_BOUNDS.items():
            bounds[param_name] = {
                'min': default['min'],
                'max': default['max'],
            }
            if 'transform' in default:
                bounds[param_name]['transform'] = default['transform']

        # Config overrides
        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='CLM_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)

        # Log bounds
        for param_name in self.clm_params:
            if param_name in bounds:
                b = bounds[param_name]
                transform = b.get('transform', 'linear')
                self.logger.info(
                    f"CLM param {param_name}: "
                    f"bounds=[{b['min']:.6g}, {b['max']:.6g}] "
                    f"transform={transform}"
                )

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update all CLM parameter files."""
        try:
            # Separate params by target file
            nl_params = {}
            params_nc_updates = {}
            surfdata_updates = {}

            for name, value in params.items():
                if name not in CLM_PARAM_DEFS:
                    self.logger.warning(f"Unknown CLM param: {name}")
                    continue

                target, nc_var, _ = CLM_PARAM_DEFS[name]
                if target == 'namelist':
                    nl_params[name] = value
                elif target == 'params':
                    params_nc_updates[name] = value
                elif target == 'surfdata':
                    surfdata_updates[name] = value

            success = True

            if nl_params:
                success &= self._update_namelist(nl_params)
            if params_nc_updates:
                success &= self._update_params_nc(params_nc_updates)
            if surfdata_updates:
                success &= self._update_surfdata(surfdata_updates)

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating CLM files: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_namelist(self, params: Dict[str, float]) -> bool:
        """Regenerate user_nl_clm with hydrology parameters."""
        nl_path = self.settings_dir / 'user_nl_clm'
        if not nl_path.exists():
            self.logger.error(f"user_nl_clm not found: {nl_path}")
            return False

        # Read existing namelist
        content = nl_path.read_text()

        # Add/update parameter lines
        for name, value in params.items():
            # Remove existing line if present
            lines = content.split('\n')
            lines = [l for l in lines if not l.strip().startswith(f'{name} ')]
            lines = [l for l in lines if not l.strip().startswith(f'{name}=')]
            content = '\n'.join(lines)

            # Append new line
            content += f"\n{name} = {value:.8g}"

        nl_path.write_text(content)
        self.logger.debug(f"Updated user_nl_clm with {len(params)} params")
        return True

    def _update_params_nc(self, params: Dict[str, float]) -> bool:
        """Update clm5_params.nc with snow and PFT parameters.

        Uses netCDF4 directly to modify in-place, preserving the original
        NETCDF3_CLASSIC format that CLM/PIO requires.
        """
        import netCDF4

        params_file = self.params_dir / self.params_nc_name
        if not params_file.exists():
            self.logger.error(f"CLM params file not found: {params_file}")
            return False

        active_pfts = self._get_active_pfts()

        ds = netCDF4.Dataset(str(params_file), 'r+')

        for name, value in params.items():
            if name not in CLM_PARAM_DEFS:
                continue

            _, nc_var, _ = CLM_PARAM_DEFS[name]
            if nc_var is None or nc_var not in ds.variables:
                continue

            var = ds.variables[nc_var]
            is_pft_param = name in (
                'medlynslope', 'slatop', 'flnr', 'froot_leaf', 'stem_leaf'
            )

            if is_pft_param and 'pft' in var.dimensions and active_pfts:
                for pft_idx in active_pfts:
                    if pft_idx < var.shape[var.dimensions.index('pft')]:
                        var[pft_idx] = value
            else:
                var[:] = value

            self.logger.debug(f"Updated params.nc: {nc_var} = {value:.6g}")

        # Validate: SNOW_DENSITY_MIN < SNOW_DENSITY_MAX
        if 'SNOW_DENSITY_MIN' in ds.variables and 'SNOW_DENSITY_MAX' in ds.variables:
            dmin = float(ds.variables['SNOW_DENSITY_MIN'][:].flat[0])
            dmax = float(ds.variables['SNOW_DENSITY_MAX'][:].flat[0])
            if dmin >= dmax:
                ds.variables['SNOW_DENSITY_MIN'][:] = dmax * 0.5
                self.logger.warning(
                    f"Clamped SNOW_DENSITY_MIN ({dmin}) < SNOW_DENSITY_MAX ({dmax})"
                )

        ds.close()
        return True

    def _update_surfdata(self, params: Dict[str, float]) -> bool:
        """Update surfdata_clm.nc with soil multipliers and FMAX.

        Uses netCDF4 directly to modify in-place, preserving the exact
        binary format that CLM/PIO requires.
        """
        import netCDF4

        surfdata_file = self.params_dir / self.surfdata_nc_name
        if not surfdata_file.exists():
            self.logger.error(f"Surfdata file not found: {surfdata_file}")
            return False

        ds = netCDF4.Dataset(str(surfdata_file), 'r+')

        for name, value in params.items():
            if name not in CLM_PARAM_DEFS:
                continue

            _, nc_var, _ = CLM_PARAM_DEFS[name]

            if name == 'fmax' and 'FMAX' in ds.variables:
                ds.variables['FMAX'][:] = value
                self.logger.debug(f"Updated FMAX = {value:.4f}")
                continue

            if name == 'organic_max' and 'ORGANIC' in ds.variables:
                org = ds.variables['ORGANIC'][:].copy()
                ds.variables['ORGANIC'][:] = np.minimum(org, value)
                self.logger.debug(f"Capped ORGANIC at {value:.1f}")
                continue

        if {'sand_offset', 'clay_offset'} & set(params) and \
                {'PCT_SAND', 'PCT_CLAY'} <= set(ds.variables):
            sand, clay = apply_texture_offsets(
                ds.variables['PCT_SAND'][:], ds.variables['PCT_CLAY'][:],
                params.get('sand_offset', 0.0), params.get('clay_offset', 0.0))
            ds.variables['PCT_SAND'][:] = sand
            ds.variables['PCT_CLAY'][:] = clay
            self.logger.debug(
                f"Applied texture offsets sand={params.get('sand_offset', 0.0):+.2f} "
                f"clay={params.get('clay_offset', 0.0):+.2f}"
            )

        ds.close()
        return True

    # DDS initial guess for warm-restart calibration.
    # Based on the v10 run that achieved KGE=0.72 after 30 iterations.
    # All params from v3 best (proven to run fast and produce good KGE).
    # The 8yr-spinup restart provides well-equilibrated soil state that
    # accommodates these parameter values without drift.
    CLM_INITIAL_GUESS: Dict[str, float] = {
        # Hydrology — v3 best values (proven fast-running)
        'baseflow_scalar': 0.01,
        'fff': 0.361,
        'wimp': 0.047,
        'ksatdecay': 1.587,
        'n_baseflow': 2.075,
        'perched_baseflow_scalar': 6.87e-6,
        'interception_fraction': 0.508,
        'max_leaf_wetted_frac': 0.096,
        # Surfdata — v3 best values; texture starts at the dataset values
        'fmax': 0.549,
        'organic_max': 44.6,
        'sand_offset': 0.0,
        'clay_offset': 0.0,
        # Snow — v3 best (snow physics independent of soil state)
        'fresh_snw_rds_max': 165.7,
        'snw_aging_bst': 152.0,
        'SNO_Z0MV': 0.000151,
        'accum_factor': 0.000122,
        'SNOW_DENSITY_MAX': 416.8,
        'SNOW_DENSITY_MIN': 165.5,
        'n_melt_coef': 136.4,
        'int_snow_max': 4277.4,
        # Vegetation — v3 best (PFT physics independent of soil state)
        'medlynslope': 8.932,
        'slatop': 0.019,
        'flnr': 0.1106,
        'froot_leaf': 2.547,
        'stem_leaf': 1.426,
        # Frozen soil impedance
        'e_ice': 3.5,
    }

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values.

        Uses CLM5 defaults and previous calibration best values as a warm
        start for DDS, falling back to midpoint/geometric mean for any
        params not in CLM_INITIAL_GUESS.
        """
        initial = {}
        for name in self.clm_params:
            if name not in CLM_DEFAULT_BOUNDS:
                continue
            b = CLM_DEFAULT_BOUNDS[name]
            if name in self.CLM_INITIAL_GUESS:
                val = self.CLM_INITIAL_GUESS[name]
                # Clamp to bounds
                val = max(b['min'], min(b['max'], val))
                initial[name] = val
            else:
                transform = b.get('transform', 'linear')
                if transform == 'log' and b['min'] > 0:
                    initial[name] = math.sqrt(b['min'] * b['max'])
                else:
                    initial[name] = (b['min'] + b['max']) / 2.0
        return initial if initial else None

    def _get_active_pfts(self) -> List[int]:
        """Get active PFT indices from surface data."""
        surfdata_file = self.params_dir / self.surfdata_nc_name
        if not surfdata_file.exists():
            return [1, 12]  # Default: needleleaf evergreen + C3 arctic grass

        try:
            ds = xr.open_dataset(surfdata_file)
            if 'PCT_NAT_PFT' in ds:
                pct = ds['PCT_NAT_PFT'].values.flatten()
                active = [i for i, p in enumerate(pct) if p > 0.0]
                ds.close()
                return active if active else [1, 12]
            ds.close()
        except Exception:  # noqa: BLE001 — calibration resilience
            pass

        return [1, 12]
