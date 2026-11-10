# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Streamflow Calibration Target

Extracts simulated streamflow from ParFlow .pfb output (overland flow),
routes it through a two-component linear reservoir model (quick flow +
slow reservoir + baseflow), and compares against observed streamflow for
metric calculation during calibration.

The lumped ParFlow setup uses a sealed bottom boundary (z_lower FluxConst=0),
so all water exits the domain through overland flow at z_upper.  Because
subsurface flow is not explicitly routed to the outlet, the linear reservoir
post-processor provides recession behaviour and baseflow that ParFlow's
lumped cell cannot represent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


def _linear_reservoir_routing(
    overland_flow: np.ndarray,
    alpha: float,
    k_slow: float,
    baseflow: float,
) -> np.ndarray:
    """Route overland flow through a two-component linear reservoir.

    Args:
        overland_flow: Raw overland flow time series (m3/s), daily.
        alpha: Fraction routed as quick (direct) flow [0-1].
        k_slow: Slow reservoir time constant (days). Larger = slower recession.
        baseflow: Constant baseflow component (m3/s).

    Returns:
        Routed streamflow (m3/s), same length as input.

    The model splits overland flow into:
        Q_quick(t) = alpha * Q_raw(t)
        S_slow(t) = c * S_slow(t-1) + (1 - alpha) * Q_raw(t)
        Q_slow(t) = S_slow(t) / k_slow
        Q_total(t) = Q_quick(t) + Q_slow(t) + baseflow

    where c = 1 - 1/k_slow is the slow reservoir retention coefficient.
    """
    n = len(overland_flow)
    q_routed = np.empty(n)
    c = 1.0 - 1.0 / max(k_slow, 1.01)  # retention coefficient
    s_slow = 0.0  # initial slow store (m3/s * days)

    for t in range(n):
        q_raw = max(overland_flow[t], 0.0)
        q_quick = alpha * q_raw
        s_slow = c * s_slow + (1.0 - alpha) * q_raw
        q_slow = s_slow / k_slow
        q_routed[t] = q_quick + q_slow + baseflow

    return q_routed


@R.calibration_targets.add('PARFLOW_STREAMFLOW')
class ParFlowStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for ParFlow integrated hydrologic model.

    The lumped ParFlow domain uses a small grid (e.g. 3x1 km) to represent
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
        self.model_name = 'PARFLOW'
        self.settings_dir = Path(settings_dir) if settings_dir else None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return .pfb pressure files as simulation file list.

        ParFlow outputs .pfb binary files instead of NetCDF. We return
        the pressure files so the base class knows output exists.
        """
        pfb_files = sorted(sim_dir.glob('*.out.press.*.pfb'))
        if pfb_files:
            return pfb_files
        # Also check for any .pfb files
        return sorted(sim_dir.glob('*.pfb'))

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow from ParFlow .pfb output.

        Delegates to _extract_parflow_streamflow using the directory
        containing the simulation files.
        """
        if not sim_files:
            return pd.Series(dtype=float)

        output_dir = sim_files[0].parent
        streamflow = self._extract_parflow_streamflow(output_dir)
        if streamflow is None:
            return pd.Series(dtype=float)
        return streamflow

    def _extract_parflow_streamflow(
        self,
        output_dir: Path,
    ) -> Optional[pd.Series]:
        """Extract streamflow from ParFlow output directory.

        Uses ParFlowResultExtractor directly to avoid circular dependency
        with the postprocessor (which needs a full config object).
        """
        try:
            from symfluence.models.parflow.extractor import ParFlowResultExtractor

            extractor = ParFlowResultExtractor()

            timestep_hours = float(self._get_config_value(
                lambda: self.config.model.parflow.timestep_hours if self.config.model and self.config.model.parflow else None,
                default=1.0
            ))
            from symfluence.models.parflow.extractor import resolve_dump_interval_hours
            dump_interval_hours = resolve_dump_interval_hours(self.config, 'parflow')
            start_date = str(self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01'
            ))
            k_sat = float(self._get_config_value(
                lambda: self.config.model.parflow.k_sat if self.config.model and self.config.model.parflow else None,
                default=5.0
            ))
            dx = float(self._get_config_value(
                lambda: self.config.model.parflow.dx if self.config.model and self.config.model.parflow else None,
                default=1000.0
            ))
            dy = float(self._get_config_value(
                lambda: self.config.model.parflow.dy if self.config.model and self.config.model.parflow else None,
                default=1000.0
            ))
            dz = float(self._get_config_value(
                lambda: self.config.model.parflow.dz if self.config.model and self.config.model.parflow else None,
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

            # Extract overland flow (sealed bottom → all outflow is overland)
            try:
                streamflow = extractor.extract_variable(
                    output_dir, 'overland_flow', **common_kwargs
                )
            except Exception:  # noqa: BLE001 — calibration resilience
                streamflow = pd.Series(dtype=float)

            if streamflow.empty:
                self.logger.warning("No ParFlow overland flow data extracted")
                return None

            # Scale from model domain area to actual catchment area.
            # ParFlow computes flow for the small model grid; the lumped
            # catchment is much larger, so we multiply by the area ratio.
            nx = int(self._get_config_value(
                lambda: self.config.model.parflow.nx if self.config.model and self.config.model.parflow else None,
                default=3
            ))
            ny = int(self._get_config_value(
                lambda: self.config.model.parflow.ny if self.config.model and self.config.model.parflow else None,
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
                    f"ParFlow area scaling: {area_scale:.1f}x "
                    f"(catchment={catchment_area_m2/1e6:.1f} km², "
                    f"domain={domain_area_m2/1e6:.1f} km²)"
                )

            streamflow.name = 'streamflow_m3s'

            # Resample to daily for comparison with observed (typically daily)
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
                f"ParFlow streamflow: {len(streamflow_daily)} daily values, "
                f"mean={streamflow_daily.mean():.4f} m3/s"
            )

            return streamflow_daily

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to extract ParFlow streamflow: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _load_routing_params(self) -> Optional[Dict[str, float]]:
        """Load routing parameters from sidecar JSON file."""
        # Try settings_dir passed from worker
        if self.settings_dir:
            sidecar = self.settings_dir / 'routing_params.json'
            if sidecar.exists():
                try:
                    return json.loads(sidecar.read_text())
                except Exception:  # noqa: BLE001 — calibration resilience
                    pass

        # Fallback: try project settings dir
        settings_path = self.project_dir / 'settings' / 'PARFLOW' / 'routing_params.json'
        if settings_path.exists():
            try:
                return json.loads(settings_path.read_text())
            except Exception:  # noqa: BLE001 — calibration resilience
                pass

        return None
