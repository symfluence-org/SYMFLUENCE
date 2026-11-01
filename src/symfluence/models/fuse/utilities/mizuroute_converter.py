# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE to mizuRoute Format Conversion Utilities

Handles conversion of FUSE model output to mizuRoute-compatible format.
Extracted from FUSEWorker for reusability and testability.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.fuse.calibration.file_manager import resolve_fuse_id

# Suppress xarray FutureWarning about timedelta64 decoding
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')

logger = logging.getLogger(__name__)


class FuseToMizurouteConverter:
    """
    Converts FUSE distributed output to mizuRoute-compatible format.

    Handles:
    - Variable name detection (q_routed, q_instnt, total_discharge, runoff)
    - Dimension reshaping (param_set, latitude, longitude → time, gru)
    - Unit conversion (mm/timestep → m/s, mm/s, mm/day, etc.)
    - Time rounding for mizuRoute precision
    """

    # FUSE variable names to try in order of preference
    FUSE_RUNOFF_VARS = ['q_routed', 'q_instnt', 'total_discharge', 'runoff']

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def convert(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any],
        proc_id: int = 0
    ) -> bool:
        """
        Convert FUSE distributed output to mizuRoute-compatible format.

        Args:
            fuse_output_dir: Directory containing FUSE output
            config: Configuration dictionary
            proc_id: Process ID for parallel calibration (used in filename)

        Returns:
            True if conversion successful
        """
        try:
            import xarray as xr

            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
            fuse_id = resolve_fuse_id(config, experiment_id)

            # Find FUSE output file (runs_def.nc for run_def mode)
            output_file = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

            if not output_file.exists():
                self.logger.error(f"FUSE output file not found: {output_file}")
                return False

            # Open dataset
            ds = xr.open_dataset(output_file, decode_timedelta=True)

            # Find runoff variable
            q_fuse = self._find_runoff_variable(ds)
            if q_fuse is None:
                self.logger.error(f"FUSE output missing runoff variable. Available: {list(ds.variables)}")
                ds.close()
                return False

            # Get routing variable name
            routing_var = self._get_routing_var_name(config)

            # Reshape dimensions
            q_fuse = self._reshape_dimensions(q_fuse, config)

            # Filter to non-coastal GRUs and get correct IDs from mapping
            gru_ids = None
            mapping_file = self._find_mapping_file(fuse_output_dir, config)
            if mapping_file is not None:
                q_fuse, gru_ids = self._filter_coastal_grus(q_fuse, mapping_file)

            # Fall back: extract GRU IDs from the FUSE output's longitude dim
            # (FUSE uses longitude values as GRU identifiers in distributed mode)
            if gru_ids is None and 'longitude' in ds.variables:
                lon_vals = ds['longitude'].values.flatten()
                if len(lon_vals) == q_fuse.shape[-1] and lon_vals.max() > 180:
                    gru_ids = lon_vals.astype(np.int32)
                    self.logger.debug(f"Using GRU IDs from longitude dim: {gru_ids}")

            # Convert units
            q_fuse_values, target_units = self._convert_units(q_fuse.values, config)

            # Prepare time values
            time_values = self._prepare_time_values(ds)

            # Create routing dataset
            ds_routing = self._create_routing_dataset(
                q_fuse_values, time_values, routing_var, target_units,
                gru_ids=gru_ids
            )

            # Close input dataset before writing
            ds.close()

            # Save converted output
            expected_filename = f"proc_{proc_id:02d}_{experiment_id}_timestep.nc"
            expected_file = fuse_output_dir / expected_filename

            ds_routing.to_netcdf(expected_file)
            self.logger.debug(f"Created mizuRoute input file: {expected_file}")

            ds_routing.close()
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error converting FUSE output to mizuRoute format: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _find_runoff_variable(self, ds) -> Optional[Any]:
        """Find the runoff variable in FUSE output."""
        for var_name in self.FUSE_RUNOFF_VARS:
            if var_name in ds.variables:
                self.logger.debug(f"Using FUSE variable '{var_name}' for routing")
                return ds[var_name]
        return None

    def _get_routing_var_name(self, config: Dict[str, Any]) -> str:
        """Get the routing variable name from config."""
        routing_var = resolve_config_value(config, lambda: config.model.mizuroute.routing_var, 'q_routed', 'SETTINGS_MIZU_ROUTING_VAR')
        if routing_var in ('default', None, ''):
            routing_var = 'q_routed'
        return routing_var

    def _reshape_dimensions(self, q_fuse, config: Dict[str, Any]):
        """
        Reshape FUSE output dimensions for mizuRoute.

        FUSE: (time, param_set, latitude, longitude) → mizuRoute: (time, gru)
        """
        # Squeeze out param_set dimension if it exists
        if 'param_set' in q_fuse.dims:
            q_fuse = q_fuse.isel(param_set=0)

        # Handle spatial dimensions
        subcatchment_dim = resolve_config_value(config, lambda: config.model.fuse.subcatchment_dim, 'longitude', 'FUSE_SUBCATCHMENT_DIM')

        if 'latitude' in q_fuse.dims and 'longitude' in q_fuse.dims:
            # Squeeze singleton dimensions
            if q_fuse.sizes.get('longitude', 1) == 1:
                q_fuse = q_fuse.squeeze('longitude', drop=True)
            if q_fuse.sizes.get('latitude', 1) == 1:
                q_fuse = q_fuse.squeeze('latitude', drop=True)

            # Rename the remaining spatial dimension to 'gru'
            if subcatchment_dim in q_fuse.dims:
                q_fuse = q_fuse.rename({subcatchment_dim: 'gru'})

        # If still no 'gru' dimension, create one (lumped case after squeeze)
        if 'gru' not in q_fuse.dims:
            q_fuse = q_fuse.expand_dims('gru')

        # Ensure dimensions are in (time, gru) order
        if 'time' in q_fuse.dims and 'gru' in q_fuse.dims:
            q_fuse = q_fuse.transpose('time', 'gru')

        return q_fuse

    def _convert_units(
        self,
        values: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, str]:
        """
        Convert FUSE output units (mm/timestep) to target units.

        Returns:
            Tuple of (converted values, target units string)
        """
        target_units = resolve_config_value(config, lambda: config.model.mizuroute.routing_units, 'm/s', 'SETTINGS_MIZU_ROUTING_UNITS')
        if target_units in ('default', None, ''):
            target_units = 'm/s'

        # FUSE writes daily output regardless of forcing cadence, so the divisor
        # is FUSE's own output timestep and never the forcing one. This used to
        # fall back to FORCING_TIME_STEP_SIZE, contradicting the sentence above
        # it: with hourly forcing that divides mm/day by 3600 instead of 86400
        # and inflates routed runoff 24-fold. Unreachable through the typed
        # config — FUSE_OUTPUT_TIMESTEP_SECONDS carries a schema default of
        # 86400, so the fallback never fired — but it sat one raw dict away from
        # firing, and the parity of the two `config.get` defaults (86400 vs the
        # forcing step) is exactly the kind of detail a later edit gets wrong.
        timestep_seconds = int(resolve_config_value(config, lambda: config.model.fuse.output_timestep_seconds, None, 'FUSE_OUTPUT_TIMESTEP_SECONDS') or 86400)

        self.logger.debug(f"Converting FUSE runoff to {target_units} (timestep={timestep_seconds}s)")

        if target_units == 'm/s':
            # mm/timestep -> m/s
            conversion_factor = 1.0 / (1000.0 * timestep_seconds)
            values = values * conversion_factor
            self.logger.debug(f"Applied conversion factor: {conversion_factor:.2e}")

        elif target_units == 'mm/s':
            # mm/timestep -> mm/s
            conversion_factor = 1.0 / timestep_seconds
            values = values * conversion_factor

        elif target_units in ['mm/d', 'mm/day']:
            # mm/timestep -> mm/day
            if timestep_seconds != 86400:
                conversion_factor = 86400.0 / timestep_seconds
                values = values * conversion_factor

        elif target_units in ['mm/h', 'mm/hr', 'mm/hour']:
            # mm/timestep -> mm/hour
            if timestep_seconds != 3600:
                conversion_factor = 3600.0 / timestep_seconds
                values = values * conversion_factor

        else:
            self.logger.warning(f"Unknown target units '{target_units}', passing through without conversion")

        return values, target_units

    def _prepare_time_values(self, ds):
        """Round time values to nearest hour for mizuRoute precision."""
        try:
            import pandas as pd
            time_values = pd.to_datetime(ds['time'].values).round('h')
            # Remove timezone if present
            if hasattr(time_values, 'tz') and time_values.tz is not None:
                time_values = time_values.tz_localize(None)
            return time_values
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not round times for mizuRoute: {e}", exc_info=True)
            return ds['time'].values

    def _find_mapping_file(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Locate fuse_to_routing_mapping.csv in the settings directory.

        Searches relative to fuse_output_dir (going up to project root)
        or uses the SYMFLUENCE_DATA_DIR from config.
        """
        # Try deriving from fuse_output_dir:
        # fuse_output_dir is typically <project>/simulations/<exp>/FUSE/
        # mapping is at <project>/settings/mizuRoute/fuse_to_routing_mapping.csv
        project_dir = fuse_output_dir
        for _ in range(4):
            project_dir = project_dir.parent
            candidate = project_dir / 'settings' / 'mizuRoute' / 'fuse_to_routing_mapping.csv'
            if candidate.exists():
                self.logger.debug(f"Found routing mapping file: {candidate}")
                return candidate

        # Try from config
        data_dir = config.get('SYMFLUENCE_DATA_DIR', '')
        domain = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
        if data_dir and domain:
            candidate = Path(data_dir) / f'domain_{domain}' / 'settings' / 'mizuRoute' / 'fuse_to_routing_mapping.csv'
            if candidate.exists():
                self.logger.debug(f"Found routing mapping file: {candidate}")
                return candidate

        self.logger.warning("fuse_to_routing_mapping.csv not found — outputting all GRUs without filtering")
        return None

    def _filter_coastal_grus(self, q_fuse, mapping_file: Path):
        """
        Filter out coastal GRUs and return correct gruId values.

        Args:
            q_fuse: xarray DataArray with dims (time, gru)
            mapping_file: Path to fuse_to_routing_mapping.csv

        Returns:
            Tuple of (filtered DataArray, gru_ids array matching topology hruId)
        """
        mapping = pd.read_csv(mapping_file)

        non_coastal = mapping[~mapping['is_coastal']]
        n_non_coastal = len(non_coastal)
        n_total = q_fuse.sizes.get('gru', 0)

        self.logger.debug(
            f"Filtering coastal GRUs: {n_total} total → {n_non_coastal} non-coastal"
        )

        # Get the FUSE indices of non-coastal GRUs
        fuse_indices = non_coastal['fuse_gru_idx'].values

        # Subset the runoff array to non-coastal GRUs only
        q_fuse = q_fuse.isel(gru=fuse_indices)

        # Re-index gru dimension after subsetting
        q_fuse = q_fuse.assign_coords(gru=np.arange(n_non_coastal))

        # Get the gruId values that match topology hruId
        gru_ids = non_coastal['gru_to_seg'].values.astype(np.int32)

        return q_fuse, gru_ids

    def _create_routing_dataset(
        self,
        q_values: np.ndarray,
        time_values,
        routing_var: str,
        units: str,
        gru_ids: Optional[np.ndarray] = None
    ):
        """Create xarray Dataset for mizuRoute input."""
        import xarray as xr

        n_gru = q_values.shape[1] if q_values.ndim > 1 else 1

        ds_routing = xr.Dataset({
            routing_var: (('time', 'gru'), q_values)
        }, coords={
            'time': time_values,
            'gru': np.arange(n_gru)
        })

        # Set variable attributes
        ds_routing[routing_var].attrs['units'] = units
        ds_routing[routing_var].attrs['long_name'] = 'FUSE runoff for mizuRoute routing'

        # Add gruId variable — use provided IDs (matching topology hruId) or fall back to 1-indexed
        if gru_ids is not None:
            ds_routing['gruId'] = ('gru', gru_ids)
        else:
            ds_routing['gruId'] = ('gru', np.arange(1, n_gru + 1))

        return ds_routing
