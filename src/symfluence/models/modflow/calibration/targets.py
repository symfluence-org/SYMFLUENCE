# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coupled Groundwater Calibration Target

Extracts combined streamflow from any land surface model's surface runoff
+ MODFLOW drain discharge for comparison against observed streamflow.

The land surface model is determined by LAND_SURFACE_MODEL config key.
Uses SUMMAToMODFLOWCoupler for recharge extraction and flow combination,
and MODFLOWResultExtractor for drain discharge from .bud files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.core.registries import R

logger = logging.getLogger(__name__)

# Map land surface models to their surface runoff variable names.
# In coupled mode (noXplict / no internal aquifer), surface runoff is
# the fast-flow component that bypasses the groundwater model.
SURFACE_RUNOFF_VARIABLES = {
    'SUMMA': 'scalarSurfaceRunoff',
    'CLM': 'QOVER',
    'MESH': 'RUNOFF',
}


@R.calibration_targets.add('COUPLED_GW_STREAMFLOW')
class CoupledGWStreamflowTarget(StreamflowEvaluator):
    """Streamflow target for coupled land-surface + MODFLOW calibration.

    Combines any land surface model's surface runoff with MODFLOW
    groundwater baseflow (drain discharge) to produce total simulated
    streamflow.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger,
    ):
        super().__init__(config, project_dir, logger)
        self.model_name = 'COUPLED_GW'
        from symfluence.models.modflow.land_adapters import resolve_land_model
        self.land_model_name = resolve_land_model(self.config_dict or config)

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return output files from both land surface and MODFLOW sub-directories."""
        files = []

        # Land surface model output
        land_dir = sim_dir / self.land_model_name
        if land_dir.exists():
            files.extend(sorted(land_dir.glob('*.nc')))
            files.extend(sorted(land_dir.glob('time*.txt')))   # HYPE

        # MODFLOW output (lumped: binary budget/heads; basin: budget CSV)
        modflow_dir = sim_dir / 'MODFLOW'
        if modflow_dir.exists():
            files.extend(sorted(modflow_dir.glob('*.bud')))
            files.extend(sorted(modflow_dir.glob('*.hds')))
            files.extend(sorted(modflow_dir.glob('*.bud.csv')))

        return files

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract combined streamflow from land-surface + MODFLOW output."""
        if not sim_files:
            return pd.Series(dtype=float)

        # Determine directories from file paths
        land_dir = None
        modflow_dir = None
        for f in sim_files:
            if f.parent.name == self.land_model_name:
                land_dir = f.parent
            elif f.parent.name == 'MODFLOW':
                modflow_dir = f.parent

        if land_dir is None or modflow_dir is None:
            self.logger.error(
                f"Could not find {self.land_model_name} and/or MODFLOW output directories"
            )
            return pd.Series(dtype=float)

        try:
            return self._extract_combined_streamflow(land_dir, modflow_dir)
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to extract combined streamflow: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return pd.Series(dtype=float)

    def _extract_combined_streamflow(
        self,
        land_dir: Path,
        modflow_dir: Path,
    ) -> pd.Series:
        """Extract and combine surface runoff + baseflow.

        Returns:
            Total streamflow in m3/s with daily datetime index.
        """
        from symfluence.models.modflow.basin_model import GRID_FILE
        if (modflow_dir / GRID_FILE).exists():
            return self._extract_basin_streamflow(modflow_dir)

        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler
        from symfluence.models.modflow.extractor import MODFLOWResultExtractor

        coupler = SUMMAToMODFLOWCoupler(self.config_dict, self.logger)
        extractor = MODFLOWResultExtractor()

        # Extract surface runoff directly.  In coupled mode SUMMA runs with
        # groundwatr=noXplict (no internal aquifer), so averageRoutedRunoff ≈
        # scalarSoilDrainage and subtracting them gives ≈ 0.  The correct
        # fast-flow component is scalarSurfaceRunoff (infiltration excess +
        # saturation excess), which goes directly to the stream while soil
        # drainage feeds MODFLOW as recharge.
        surface_var = SURFACE_RUNOFF_VARIABLES.get(
            self.land_model_name, 'scalarSurfaceRunoff',
        )
        surface_runoff = coupler.extract_surface_runoff(
            land_dir, variable=surface_var,
        )

        # Extract MODFLOW drain discharge (m3/d)
        # Use date-only (no time component) so dates align with SUMMA daily output
        start_str = str(self._get_config_value(lambda: self.config.domain.experiment_time_start, default='2000-01-01', dict_key='EXPERIMENT_TIME_START'))
        start_date = pd.Timestamp(start_str).strftime('%Y-%m-%d')
        drain_discharge = extractor.extract_variable(
            modflow_dir,
            'drain_discharge',
            start_date=start_date,
            stress_period_length=1.0,
        )

        # Get catchment area for unit conversion
        catchment_area_m2 = self._get_catchment_area()

        # Combine flows
        total_streamflow = coupler.combine_flows(
            surface_runoff, drain_discharge, catchment_area_m2,
        )

        return total_streamflow

    def _extract_basin_streamflow(self, modflow_dir: Path) -> pd.Series:
        """Basin mode: land-model fast flow + MODFLOW discharge to rivers (m3/s)."""
        from symfluence.models.modflow.basin_model import read_coupling_record, read_groundwater_discharge
        from symfluence.models.modflow.land_adapters import adapter_options, get_land_adapter

        rec = read_coupling_record(modflow_dir)
        adapter = get_land_adapter(rec['land_model'], adapter_options(self.config_dict))
        fast = adapter.read_fast_flow(Path(rec['land_output']), Path(rec['land_settings']))
        gw = read_groundwater_discharge(modflow_dir)
        fast.index = pd.to_datetime(fast.index).normalize()
        total = (fast + gw).dropna()
        if total.empty:
            from symfluence.core.exceptions import ValidationError
            raise ValidationError('Land fast flow and MODFLOW discharge do not overlap in time')
        return total.rename('total_streamflow_m3s')
