# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Parameter Manager

Handles CLMParFlow parameter bounds, normalization, and .pfidb file updates.
Parameters are written directly into the ParFlow database file (.pfidb)
which maps keys to subsurface properties, van Genuchten parameters,
and Manning's roughness.

Snow-17 parameters (SCF, MFMAX, MFMIN, PXTEMP) are also supported.
When Snow-17 parameters change, the forcing is regenerated from the
cached hourly ERA5 data using the new snow model parameters.

Reuses ParFlow's .pfidb read/write utilities and parameter mappings,
since CLMParFlow uses the same .pfidb format.

.pfidb key mapping:
    K_SAT     -> Geom.domain.Perm.Value
    POROSITY  -> Geom.domain.Porosity.Value
    VG_ALPHA  -> Geom.domain.RelPerm.Alpha, Geom.domain.Saturation.Alpha
    VG_N      -> Geom.domain.RelPerm.N, Geom.domain.Saturation.N
    S_RES     -> Geom.domain.Saturation.SRes
    MANNINGS_N-> Mannings.Geom.domain.Value
    SNOW17_*  -> regenerates daily BC values via Snow-17
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.mixins.project import resolve_data_subdir

# Reuse ParFlow's .pfidb utilities and parameter constants
from symfluence.models.parflow.calibration.parameter_manager import (
    PARAM_TO_PFIDB_KEYS,
    ROUTING_PARAM_NAMES,
    SNOW17_PARAM_NAMES,
    _read_pfidb,
    _write_pfidb,
)
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry

# Same physically-based parameter bounds as ParFlow
CLMPARFLOW_DEFAULT_BOUNDS = {
    'K_SAT': {
        'min': 0.001, 'max': 100.0,
        'transform': 'log',
        'description': 'Saturated hydraulic conductivity (m/hr)',
    },
    'POROSITY': {
        'min': 0.05, 'max': 0.6,
        'transform': 'linear',
        'description': 'Total porosity (-)',
    },
    'VG_ALPHA': {
        'min': 0.01, 'max': 10.0,
        'transform': 'log',
        'description': 'van Genuchten alpha (1/m)',
    },
    'VG_N': {
        'min': 1.1, 'max': 5.0,
        'transform': 'linear',
        'description': 'van Genuchten n shape parameter (-)',
    },
    'S_RES': {
        'min': 0.01, 'max': 0.4,
        'transform': 'linear',
        'description': 'Residual saturation (-)',
    },
    'MANNINGS_N': {
        'min': 0.005, 'max': 0.3,
        'transform': 'log',
        'description': "Manning's roughness coefficient (s/m^1/3)",
    },
    'SS': {
        'min': 1e-7, 'max': 1e-3,
        'transform': 'log',
        'description': 'Specific storage (1/m)',
    },
    'S_SAT': {
        'min': 0.8, 'max': 1.0,
        'transform': 'linear',
        'description': 'Maximum saturation (-)',
    },
    # Snow-17 parameters (affect forcing, not subsurface)
    'SNOW17_SCF': {
        'min': 0.7, 'max': 1.4,
        'transform': 'linear',
        'description': 'Snowfall correction factor',
    },
    'SNOW17_MFMAX': {
        'min': 0.5, 'max': 4.0,
        'transform': 'linear',
        'description': 'Max melt factor Jun 21 (mm/C/6hr)',
    },
    'SNOW17_MFMIN': {
        'min': 0.05, 'max': 1.5,
        'transform': 'linear',
        'description': 'Min melt factor Dec 21 (mm/C/6hr)',
    },
    'SNOW17_PXTEMP': {
        'min': -4.0, 'max': 3.0,
        'transform': 'linear',
        'description': 'Rain/snow threshold temperature (C)',
    },
    'SNOW_LAPSE_RATE': {
        'min': 0.003, 'max': 0.010,
        'transform': 'linear',
        'description': 'Temperature lapse rate for elevation bands (C/m)',
    },
    # Linear reservoir routing parameters (post-processing, not in .pfidb)
    'ROUTE_ALPHA': {
        'min': 0.0, 'max': 1.0,
        'transform': 'linear',
        'description': 'Quick flow fraction of overland flow (-)',
    },
    'ROUTE_K_SLOW': {
        'min': 1.0, 'max': 100.0,
        'transform': 'log',
        'description': 'Slow reservoir time constant (days)',
    },
    'ROUTE_BASEFLOW': {
        'min': 0.0, 'max': 30.0,
        'transform': 'linear',
        'description': 'Constant baseflow component (m3/s)',
    },
}


@OptimizerRegistry.register_parameter_manager('CLMPARFLOW')
class CLMParFlowParameterManager(BaseParameterManager):
    """Handles CLMParFlow parameter bounds, normalization, and .pfidb file updates."""

    def __init__(self, config: Dict, logger: logging.Logger, clmparflow_settings_dir: Path):
        super().__init__(config, logger, clmparflow_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Forcing data lives in data/forcing/CLMPARFLOW_input/
        data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        self.forcing_input_dir = resolve_data_subdir(
            project_dir, 'forcing'
        ) / 'CLMPARFLOW_input'

        # Parse parameters to calibrate from config
        pf_params_str = self._get_config_value(
            lambda: self.config.model.clmparflow.params_to_calibrate,
            default='K_SAT,POROSITY,VG_ALPHA,VG_N,S_RES,MANNINGS_N',
            dict_key='CLMPARFLOW_PARAMS_TO_CALIBRATE'
        )
        self.pf_params = [p.strip() for p in str(pf_params_str).split(',') if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        """Return CLMParFlow parameter names from config."""
        return self.pf_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Return CLMParFlow parameter bounds, with optional config overrides."""
        bounds: Dict[str, Dict[str, Any]] = {
            k: {
                'min': float(v['min']),
                'max': float(v['max']),
                'transform': v.get('transform', 'linear'),
            }
            for k, v in CLMPARFLOW_DEFAULT_BOUNDS.items()
        }

        # Allow config overrides (preserves transform metadata from registry)
        config_bounds = self._get_config_value(lambda: None, default=None, dict_key='CLMPARFLOW_PARAM_BOUNDS')

        if config_bounds and isinstance(config_bounds, dict):
            self.logger.info("Using config-specified CLMParFlow parameter bounds")
            self._apply_config_bounds_override(bounds, config_bounds)

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update .pfidb file with new parameter values.

        Reads the existing .pfidb, updates the relevant keys, and writes
        it back. If Snow-17 parameters are included, regenerates the
        daily forcing from cached hourly data.
        """
        pfidb_files = list(self.settings_dir.glob('*.pfidb'))
        if not pfidb_files:
            self.logger.error(f"No .pfidb file found in {self.settings_dir}")
            return False

        pfidb_path = pfidb_files[0]

        try:
            entries = _read_pfidb(pfidb_path)

            # Separate routing, Snow-17, and subsurface params
            routing_params = {k: v for k, v in params.items() if k in ROUTING_PARAM_NAMES}
            snow_params = {k: v for k, v in params.items() if k in SNOW17_PARAM_NAMES}
            subsurface_params = {
                k: v for k, v in params.items()
                if k not in SNOW17_PARAM_NAMES and k not in ROUTING_PARAM_NAMES
            }

            # Write routing params to sidecar JSON for target extraction
            if routing_params:
                sidecar = self.settings_dir / 'routing_params.json'
                sidecar.write_text(json.dumps(routing_params))
                self.logger.debug(f"Wrote routing params to {sidecar}")

            # Update subsurface .pfidb keys
            for param_name, value in subsurface_params.items():
                pfidb_keys = PARAM_TO_PFIDB_KEYS.get(param_name, [])
                for key in pfidb_keys:
                    if key in entries:
                        entries[key] = f'{value:g}'
                    else:
                        self.logger.debug(
                            f"Key {key} not in .pfidb, adding for {param_name}"
                        )
                        entries[key] = f'{value:g}'

            # If Snow-17 params changed, regenerate daily forcing BC values
            if snow_params:
                self._update_snow17_forcing(entries, snow_params)

            _write_pfidb(pfidb_path, entries)
            self.logger.debug(f"Updated .pfidb with {len(params)} parameters")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to update .pfidb: {e}")
            return False

    def _update_snow17_forcing(
        self, entries: Dict[str, str], snow_params: Dict[str, float]
    ) -> None:
        """Regenerate daily forcing BC values using updated Snow-17 params.

        Delegates to ParFlow's implementation since the forcing format is identical.
        """
        import pandas as pd
        from jsnow17.bmi import Snow17BMI

        from symfluence.models.clmparflow.preprocessor import CLMParFlowPreProcessor

        cache_path = self.forcing_input_dir / 'hourly_forcing_cache.npz'
        if not cache_path.exists():
            # Fallback to settings_dir for backward compatibility
            cache_path = self.settings_dir / 'hourly_forcing_cache.npz'
        if not cache_path.exists():
            self.logger.warning("No hourly forcing cache; skipping Snow-17 update")
            return

        cache = np.load(cache_path)
        ppt_mm_hr = cache['ppt_mm_hr']
        pet_mm_hr = cache['pet_mm_hr']
        temp_c = cache['temp_c']
        times = pd.DatetimeIndex(cache['times'].astype('datetime64[ns]'))

        # Build Snow-17 param dict (strip SNOW17_ prefix)
        s17_params = {}
        lapse_rate = CLMParFlowPreProcessor.DEFAULT_LAPSE_RATE
        for k, v in snow_params.items():
            if k == 'SNOW_LAPSE_RATE':
                lapse_rate = v
            else:
                s17_key = k.replace('SNOW17_', '')
                s17_params[s17_key] = v

        # Get latitude from config
        lat = float(self._get_config_value(lambda: None, default=51.36, dict_key='CATCHMENT_LATITUDE'))
        try:
            pp_coords = self._get_config_value(lambda: self.config.domain.pour_point_coords, default='', dict_key='POUR_POINT_COORDS')
            if '/' in str(pp_coords):
                lat = float(str(pp_coords).split('/')[0])
        except (ValueError, AttributeError):
            pass

        # Aggregate hourly to daily for Snow-17
        df_hourly = pd.DataFrame({'ppt': ppt_mm_hr, 'temp': temp_c}, index=times)
        daily = df_hourly.resample('D').agg({'ppt': 'sum', 'temp': 'mean'})
        daily['doy'] = daily.index.dayofyear

        # Run elevation-band Snow-17
        n_days = len(daily)
        rpm_daily = np.zeros(n_days)

        for band_elev, area_frac in CLMParFlowPreProcessor.ELEVATION_BANDS:
            temp_offset = -lapse_rate * (band_elev - CLMParFlowPreProcessor.BASIN_MEAN_ELEV)
            temp_band = daily['temp'].values + temp_offset

            snow = Snow17BMI(params=s17_params, latitude=lat, dt=1.0)
            snow.initialize()
            rpm_band = snow.update_batch(
                daily['ppt'].values, temp_band, daily['doy'].values,
            )
            rpm_daily += rpm_band * area_frac

        # Distribute to hourly and compute daily effective rate
        df_pet = pd.DataFrame({'pet': pet_mm_hr}, index=times)
        daily_pet = df_pet.resample('D').mean()

        # Update BC values in entries
        for i, (dt, pet_row) in enumerate(daily_pet.iterrows()):
            label = f"d{dt.strftime('%Y%m%d')}"
            key = f'Patch.z_upper.BCPressure.{label}.Value'
            if key in entries:
                n_hours = 24
                eff_mm_hr = rpm_daily[i] / n_hours
                pet_mean = pet_row['pet']
                net_m_hr = (eff_mm_hr - pet_mean) * 0.001
                entries[key] = f'{-net_m_hr:g}'

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from .pfidb or defaults."""
        pfidb_files = list(self.settings_dir.glob('*.pfidb'))
        if pfidb_files:
            try:
                entries = _read_pfidb(pfidb_files[0])
                params = {}
                for param_name in self.pf_params:
                    pfidb_keys = PARAM_TO_PFIDB_KEYS.get(param_name, [])
                    if pfidb_keys and pfidb_keys[0] in entries:
                        params[param_name] = float(entries[pfidb_keys[0]])
                if params:
                    self.logger.info(
                        f"Loaded {len(params)} initial parameters from .pfidb"
                    )
                    return params
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.debug(f"Could not read initial params from .pfidb: {e}")

        # Fallback to physically-reasonable defaults
        defaults = {
            'K_SAT': 5.0,
            'POROSITY': 0.4,
            'VG_ALPHA': 1.0,
            'VG_N': 2.0,
            'S_RES': 0.1,
            'S_SAT': 1.0,
            'MANNINGS_N': 0.03,
            'SS': 1e-5,
            'SNOW17_SCF': 1.0,
            'SNOW17_MFMAX': 1.0,
            'SNOW17_MFMIN': 0.3,
            'SNOW17_PXTEMP': 0.0,
            'SNOW_LAPSE_RATE': 0.0065,
            'ROUTE_ALPHA': 0.3,
            'ROUTE_K_SLOW': 20.0,
            'ROUTE_BASEFLOW': 5.0,
        }
        return {k: v for k, v in defaults.items() if k in self.pf_params}
