# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Model Postprocessor

Extracts overland flow and subsurface drainage from ParFlow-CLM output
and combines them into total streamflow. No external coupling needed —
CLM handles ET/snow internally within the ParFlow simulation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from symfluence.core.modeling.base.standard_postprocessor import StandardModelPostProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.postprocessors.add("CLMPARFLOW")
class CLMParFlowPostProcessor(StandardModelPostProcessor):
    """
    Postprocesses CLMParFlow output.

    Extracts overland flow and subsurface drainage from .pfb files
    and combines into total streamflow. CLM handles ET/snow internally,
    so no external coupling is needed.
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='unknown',
            dict_key='DOMAIN_NAME',
        )

    def _get_model_name(self) -> str:
        return "CLMPARFLOW"

    def extract_streamflow(  # type: ignore[override]
        self,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """
        Extract streamflow from CLMParFlow output.

        Standalone mode: combines overland flow + subsurface drainage.
        CLM handles ET and snow internally — no coupling needed.

        Returns:
            Tuple of (streamflow Series in m3/s, metadata dict)
        """
        if output_dir is None:
            output_dir = (
                self.project_dir / "simulations"
                / self.config.domain.experiment_id / "CLMPARFLOW"
            )

        basin_outlet = output_dir / 'clmparflow_basin_outlet.csv'
        if basin_outlet.exists():
            # Basin mode: ParFlow routes overland flow to the outlet itself (PrintOverlandSum, daily m3/s)
            streamflow = pd.read_csv(basin_outlet, index_col='date', parse_dates=True)['discharge_cms']
            streamflow.name = 'streamflow_m3s'
            self.save_streamflow_to_results(streamflow)
            return streamflow, {'units': 'm3/s', 'source': 'CLMParFlow basin-mode outlet overland flow',
                                'mode': 'basin'}

        try:
            from .extractor import CLMParFlowResultExtractor
            extractor = CLMParFlowResultExtractor()

            timestep_hours = self._get_config_value(
                lambda: self.config.model.clmparflow.timestep_hours,
                default=1.0,
                dict_key='CLMPARFLOW_TIMESTEP_HOURS',
            )

            start_date = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01',
                dict_key='EXPERIMENT_TIME_START',
            )

            k_sat = self._get_config_value(
                lambda: self.config.model.clmparflow.k_sat,
                default=5.0,
                dict_key='CLMPARFLOW_K_SAT',
            )
            dx = self._get_config_value(
                lambda: self.config.model.clmparflow.dx,
                default=1000.0,
                dict_key='CLMPARFLOW_DX',
            )
            dy = self._get_config_value(
                lambda: self.config.model.clmparflow.dy,
                default=1000.0,
                dict_key='CLMPARFLOW_DY',
            )
            dz = self._get_config_value(
                lambda: self.config.model.clmparflow.dz,
                default=2.0,
                dict_key='CLMPARFLOW_DZ',
            )

            common_kwargs = dict(
                timestep_hours=float(timestep_hours),
                start_date=str(start_date),
                k_sat=float(k_sat),
                dx=float(dx),
                dy=float(dy),
                dz=float(dz),
            )

            try:
                overland_m3s = extractor.extract_variable(
                    output_dir, 'overland_flow', **common_kwargs
                )
            except Exception:  # noqa: BLE001 — model execution resilience
                overland_m3s = pd.Series(dtype=float, name='overland_flow_m3s')

            try:
                subsurface_m3hr = extractor.extract_variable(
                    output_dir, 'subsurface_drainage', **common_kwargs
                )
            except Exception:  # noqa: BLE001 — model execution resilience
                subsurface_m3hr = pd.Series(dtype=float, name='subsurface_drainage_m3hr')

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Failed to extract CLMParFlow output: {e}", exc_info=True)
            return None

        return self._extract_standalone_flow(overland_m3s, subsurface_m3hr)

    def _extract_standalone_flow(
        self,
        overland_m3s: pd.Series,
        subsurface_m3hr: pd.Series,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract streamflow from standalone CLMParFlow run."""
        subsurface_m3s = subsurface_m3hr / 3600.0

        if not overland_m3s.empty and not subsurface_m3s.empty:
            common_idx = overland_m3s.index.intersection(subsurface_m3s.index)
            streamflow = (
                overland_m3s.reindex(common_idx).fillna(0.0)
                + subsurface_m3s.reindex(common_idx).fillna(0.0)
            )
        elif not overland_m3s.empty:
            streamflow = overland_m3s
        elif not subsurface_m3s.empty:
            streamflow = subsurface_m3s
        else:
            logger.warning("No CLMParFlow output data extracted")
            return None

        streamflow.name = 'streamflow_m3s'

        metadata = {
            'units': 'm3/s',
            'source': 'CLMParFlow overland flow + subsurface drainage',
            'mode': 'standalone_clm',
        }

        logger.info(
            f"Extracted CLMParFlow streamflow: {len(streamflow)} timesteps, "
            f"mean={streamflow.mean():.4f} m3/s"
        )

        self.save_streamflow_to_results(streamflow)

        return streamflow, metadata
