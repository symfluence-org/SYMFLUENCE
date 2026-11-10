# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Streamflow Calibration Target

Extracts simulated streamflow from CLMParFlow .pfb output (overland flow),
routes it through a two-component linear reservoir model (quick flow +
slow reservoir + baseflow), and compares against observed streamflow for
metric calculation during calibration.

Same approach as ParFlow targets but reads from CLMPARFLOW output directory
and uses CLMPARFLOW_* config keys.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.core.registries import R

# Reuse ParFlow's linear reservoir routing function
from symfluence.models.parflow.calibration.targets import _linear_reservoir_routing

logger = logging.getLogger(__name__)


@R.calibration_targets.add('CLMPARFLOW_STREAMFLOW')
class CLMParFlowStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for CLMParFlow integrated hydrologic model.

    The lumped CLMParFlow domain uses a small grid (e.g. 3x1 km) to represent
    the entire catchment.  Overland flow is scaled by the ratio of the real
    catchment area to the model domain area, then routed through a linear
    reservoir model to add recession behaviour and baseflow before comparison
    with observed discharge.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger,
        settings_dir: Optional[str] = None,
    ):
        super().__init__(config, project_dir, logger)
        self.model_name = 'CLMPARFLOW'
        self.settings_dir = Path(settings_dir) if settings_dir else None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return .pfb pressure files as simulation file list."""
        pfb_files = sorted(sim_dir.glob('*.out.press.*.pfb'))
        if pfb_files:
            return pfb_files
        return sorted(sim_dir.glob('*.pfb'))

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow from CLMParFlow .pfb output."""
        if not sim_files:
            return pd.Series(dtype=float)

        output_dir = sim_files[0].parent
        streamflow = self._extract_clmparflow_streamflow(output_dir)
        if streamflow is None:
            return pd.Series(dtype=float)
        return streamflow

    def _extract_clmparflow_streamflow(
        self,
        output_dir: Path,
    ) -> Optional[pd.Series]:
        """Extract streamflow from CLMParFlow output directory."""
        try:
            from symfluence.models.clmparflow.extractor import CLMParFlowResultExtractor

            extractor = CLMParFlowResultExtractor()

            timestep_hours = float(self._get_config_value(
                lambda: self.config.model.clmparflow.timestep_hours if self.config.model and self.config.model.clmparflow else None,
                default=1.0
            ))
            from symfluence.models.parflow.extractor import resolve_dump_interval_hours
            dump_interval_hours = resolve_dump_interval_hours(self.config, 'clmparflow')
            start_date = str(self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01'
            ))
            k_sat = float(self._get_config_value(
                lambda: self.config.model.clmparflow.k_sat if self.config.model and self.config.model.clmparflow else None,
                default=5.0
            ))
            dx = float(self._get_config_value(
                lambda: self.config.model.clmparflow.dx if self.config.model and self.config.model.clmparflow else None,
                default=1000.0
            ))
            dy = float(self._get_config_value(
                lambda: self.config.model.clmparflow.dy if self.config.model and self.config.model.clmparflow else None,
                default=1000.0
            ))
            dz = float(self._get_config_value(
                lambda: self.config.model.clmparflow.dz if self.config.model and self.config.model.clmparflow else None,
                default=2.0
            ))

            common_kwargs = dict(
                timestep_hours=timestep_hours,
                dump_interval_hours=dump_interval_hours,
                start_date=start_date,
                k_sat=k_sat,
                dx=dx,
                dy=dy,
                dz=dz,
            )

            # Extract overland flow
            try:
                streamflow = extractor.extract_variable(
                    output_dir, 'overland_flow', **common_kwargs
                )
            except Exception:  # noqa: BLE001 — calibration resilience
                streamflow = pd.Series(dtype=float)

            if streamflow.empty:
                self.logger.warning("No CLMParFlow overland flow data extracted")
                return None

            # Scale from model domain area to actual catchment area
            nx = int(self._get_config_value(
                lambda: self.config.model.clmparflow.nx if self.config.model and self.config.model.clmparflow else None,
                default=3
            ))
            ny = int(self._get_config_value(
                lambda: self.config.model.clmparflow.ny if self.config.model and self.config.model.clmparflow else None,
                default=1
            ))
            domain_area_m2 = nx * dx * ny * dy

            try:
                catchment_area_m2 = self._get_catchment_area()
            except Exception:  # noqa: BLE001 — calibration resilience
                catchment_area_m2 = domain_area_m2

            if catchment_area_m2 > domain_area_m2:
                area_scale = catchment_area_m2 / domain_area_m2
                streamflow = streamflow * area_scale
                self.logger.debug(
                    f"CLMParFlow area scaling: {area_scale:.1f}x "
                    f"(catchment={catchment_area_m2/1e6:.1f} km2, "
                    f"domain={domain_area_m2/1e6:.1f} km2)"
                )

            streamflow.name = 'streamflow_m3s'

            # Resample to daily for comparison with observed
            streamflow_daily = streamflow.resample('D').mean()

            # Apply linear reservoir routing if routing params are available
            routing_params = self._load_routing_params()
            if routing_params:
                alpha = routing_params.get('ROUTE_ALPHA', 0.3)
                k_slow = routing_params.get('ROUTE_K_SLOW', 20.0)
                baseflow = routing_params.get('ROUTE_BASEFLOW', 5.0)

                raw_values = streamflow_daily.values.copy()
                routed = _linear_reservoir_routing(raw_values, alpha, k_slow, baseflow)
                streamflow_daily = pd.Series(
                    routed, index=streamflow_daily.index, name='streamflow_m3s'
                )
                self.logger.debug(
                    f"Routing applied: alpha={alpha:.3f}, k_slow={k_slow:.1f}d, "
                    f"baseflow={baseflow:.2f} m3/s"
                )

            self.logger.debug(
                f"CLMParFlow streamflow: {len(streamflow_daily)} daily values, "
                f"mean={streamflow_daily.mean():.4f} m3/s"
            )

            return streamflow_daily

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to extract CLMParFlow streamflow: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _load_routing_params(self) -> Optional[Dict[str, float]]:
        """Load routing parameters from sidecar JSON file."""
        if self.settings_dir:
            sidecar = self.settings_dir / 'routing_params.json'
            if sidecar.exists():
                try:
                    return json.loads(sidecar.read_text())
                except Exception:  # noqa: BLE001 — calibration resilience
                    pass

        settings_path = self.project_dir / 'settings' / 'CLMPARFLOW' / 'routing_params.json'
        if settings_path.exists():
            try:
                return json.loads(settings_path.read_text())
            except Exception:  # noqa: BLE001 — calibration resilience
                pass

        return None
