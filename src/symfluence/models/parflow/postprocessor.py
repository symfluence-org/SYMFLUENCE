# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Model Postprocessor

Extracts overland flow and subsurface drainage from ParFlow output
and optionally combines with SUMMA surface runoff for total streamflow.

When SUMMA+ParFlow coupling is active, the postprocessor:
1. Reads ParFlow overland flow and subsurface drainage from .pfb files
2. Reads SUMMA surface runoff
3. Combines into total streamflow
4. Writes combined output to results CSV
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from symfluence.core.modeling.base.standard_postprocessor import StandardModelPostProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.postprocessors.add("PARFLOW")
class ParFlowPostProcessor(StandardModelPostProcessor):
    """
    Postprocesses ParFlow output.

    Extracts overland flow and subsurface drainage, and optionally combines
    with land surface model (SUMMA) surface runoff for total streamflow.
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='unknown',
            dict_key='DOMAIN_NAME',
        )

    def _get_model_name(self) -> str:
        return "PARFLOW"

    def extract_streamflow(  # type: ignore[override]
        self,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """
        Extract streamflow from ParFlow output, optionally combined with SUMMA.

        For standalone ParFlow: extracts overland flow as streamflow.
        For coupled SUMMA+ParFlow: combines surface runoff + subsurface drainage.

        Returns:
            Tuple of (streamflow Series in m3/s, metadata dict)
        """
        if output_dir is None:
            output_dir = self.get_simulation_dir('PARFLOW', self.config.domain.experiment_id)

        # Get ParFlow output
        try:
            from .extractor import ParFlowResultExtractor
            extractor = ParFlowResultExtractor()

            timestep_hours = self._get_config_value(
                lambda: self.config.model.parflow.timestep_hours,
                default=1.0,
                dict_key='PARFLOW_TIMESTEP_HOURS',
            )

            start_date = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01',
                dict_key='EXPERIMENT_TIME_START',
            )

            k_sat = self._get_config_value(
                lambda: self.config.model.parflow.k_sat,
                default=5.0,
                dict_key='PARFLOW_K_SAT',
            )
            dx = self._get_config_value(
                lambda: self.config.model.parflow.dx,
                default=1000.0,
                dict_key='PARFLOW_DX',
            )
            dy = self._get_config_value(
                lambda: self.config.model.parflow.dy,
                default=1000.0,
                dict_key='PARFLOW_DY',
            )
            dz = self._get_config_value(
                lambda: self.config.model.parflow.dz,
                default=100.0,
                dict_key='PARFLOW_DZ',
            )

            common_kwargs = dict(
                timestep_hours=float(timestep_hours),
                start_date=str(start_date),
                k_sat=float(k_sat),
                dx=float(dx),
                dy=float(dy),
                dz=float(dz),
            )

            # Try overland flow first
            try:
                overland_m3s = extractor.extract_variable(
                    output_dir, 'overland_flow', **common_kwargs
                )
            except Exception:  # noqa: BLE001 — model execution resilience
                overland_m3s = pd.Series(dtype=float, name='overland_flow_m3s')

            # Try subsurface drainage
            try:
                subsurface_m3hr = extractor.extract_variable(
                    output_dir, 'subsurface_drainage', **common_kwargs
                )
            except Exception:  # noqa: BLE001 — model execution resilience
                subsurface_m3hr = pd.Series(dtype=float, name='subsurface_drainage_m3hr')

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Failed to extract ParFlow output: {e}", exc_info=True)
            return None

        # Check for coupled mode
        coupling_source = self._get_config_value(
            lambda: self.config.model.parflow.coupling_source,
            default=None,
            dict_key='PARFLOW_COUPLING_SOURCE',
        )

        if coupling_source and str(coupling_source).upper() == 'SUMMA':
            return self._extract_coupled_streamflow(
                overland_m3s, subsurface_m3hr, output_dir
            )
        else:
            return self._extract_standalone_flow(overland_m3s, subsurface_m3hr)

    def _extract_standalone_flow(
        self,
        overland_m3s: pd.Series,
        subsurface_m3hr: pd.Series,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract streamflow from standalone ParFlow run."""
        # Convert subsurface m3/hr -> m3/s
        subsurface_m3s = subsurface_m3hr / 3600.0

        # Combine overland + subsurface
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
            logger.warning("No ParFlow output data extracted")
            return None

        streamflow.name = 'streamflow_m3s'

        metadata = {
            'units': 'm3/s',
            'source': 'ParFlow overland flow + subsurface drainage',
            'mode': 'standalone',
        }

        logger.info(
            f"Extracted ParFlow streamflow: {len(streamflow)} timesteps, "
            f"mean={streamflow.mean():.4f} m3/s"
        )

        # Save to results
        self.save_streamflow_to_results(streamflow)

        # Generate coupling diagnostics plot
        self._try_generate_plot()

        return streamflow, metadata

    def _extract_coupled_streamflow(
        self,
        overland_m3s: pd.Series,
        subsurface_m3hr: pd.Series,
        parflow_output_dir: Path,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract combined SUMMA surface runoff + ParFlow subsurface flow."""
        try:
            from .coupling import SUMMAToParFlowCoupler

            coupler = SUMMAToParFlowCoupler(self.config_dict, self.logger)

            # Find SUMMA output
            summa_output_dir = self.get_simulation_dir('SUMMA', self.config.domain.experiment_id)

            if not summa_output_dir.exists():
                logger.warning(
                    f"SUMMA output directory not found: {summa_output_dir}. "
                    "Falling back to standalone flow."
                )
                return self._extract_standalone_flow(overland_m3s, subsurface_m3hr)

            # Extract surface runoff from SUMMA
            surface_runoff = coupler.extract_surface_runoff(summa_output_dir)

            # Get catchment area
            area_m2 = self._get_catchment_area() * 1e6

            # Combine flows
            total_streamflow = coupler.combine_flows(
                surface_runoff, subsurface_m3hr, area_m2
            )

            metadata = {
                'units': 'm3/s',
                'source': 'SUMMA surface runoff + ParFlow subsurface drainage',
                'conversion': 'surface(kg/m2/s->m3/s) + subsurface(m3/hr->m3/s)',
                'mode': 'coupled_SUMMA_PARFLOW',
                'area_km2': self._get_catchment_area(),
            }

            logger.info(
                f"Extracted coupled streamflow: {len(total_streamflow)} timesteps, "
                f"mean={total_streamflow.mean():.4f} m3/s"
            )

            # Save to results
            self.save_streamflow_to_results(total_streamflow)

            # Generate coupling diagnostics plot
            self._try_generate_plot()

            return total_streamflow, metadata

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Coupled extraction failed: {e}. Falling back to standalone.", exc_info=True)
            return self._extract_standalone_flow(overland_m3s, subsurface_m3hr)

    def _try_generate_plot(self) -> None:
        """Attempt to generate coupling diagnostics plot if reporting is available."""
        try:
            from .plotter import ParFlowPlotter
            plotter = ParFlowPlotter(self.config_dict, self.logger)
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='run_1',
                dict_key='EXPERIMENT_ID',
            )
            result = plotter.plot_coupling_results(experiment_id)
            if result:
                logger.debug(f"ParFlow coupling plot saved: {result}")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.debug(f"Could not generate ParFlow coupling plot: {e}", exc_info=True)

    def _get_catchment_area(self) -> float:
        """Get catchment area in km2."""
        area = self._get_config_value(
            lambda: self.config.domain.catchment_area,
            default=None,
            dict_key='CATCHMENT_AREA',
        )
        if area:
            return float(area)

        try:
            from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics
            metrics = StreamflowMetrics()
            return metrics.get_catchment_area(
                self.config_dict, self.project_dir, self.domain_name,
                source='shapefile'
            )
        except Exception:  # noqa: BLE001 — model execution resilience
            logger.warning("Could not determine catchment area, using default 2210 km2", exc_info=True)
            return 2210.0
