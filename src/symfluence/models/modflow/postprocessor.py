# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODFLOW 6 Model Postprocessor

Extracts drain discharge (baseflow) from MODFLOW 6 output and optionally
combines with SUMMA surface runoff for total streamflow.

When SUMMA+MODFLOW coupling is active, the postprocessor:
1. Reads MODFLOW drain discharge from budget files
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


@R.postprocessors.add("MODFLOW")
class MODFLOWPostProcessor(StandardModelPostProcessor):
    """
    Postprocesses MODFLOW 6 output.

    Extracts drain discharge (baseflow) and optionally combines with
    land surface model (SUMMA) surface runoff for total streamflow.
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='unknown',
            dict_key='DOMAIN_NAME',
        )

    def _get_model_name(self) -> str:
        return "MODFLOW"

    def extract_streamflow(  # type: ignore[override]
        self,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """
        Extract streamflow from MODFLOW output, optionally combined with SUMMA.

        For standalone MODFLOW: extracts drain discharge as baseflow.
        For coupled SUMMA+MODFLOW: combines surface runoff + baseflow.

        Returns:
            Tuple of (streamflow Series in m3/s, metadata dict)
        """
        if output_dir is None:
            output_dir = self.get_simulation_dir('MODFLOW', self.config.domain.experiment_id)

        # Get MODFLOW drain discharge
        try:
            from .extractor import MODFLOWResultExtractor
            extractor = MODFLOWResultExtractor()

            sp_length = self._get_config_value(
                lambda: self.config.model.modflow.stress_period_length,
                default=1.0,
                dict_key='MODFLOW_STRESS_PERIOD_LENGTH',
            )

            start_date = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01',
                dict_key='EXPERIMENT_TIME_START',
            )

            drain_discharge = extractor.extract_variable(
                output_dir, 'drain_discharge',
                stress_period_length=float(sp_length),
                start_date=str(start_date),
            )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Failed to extract MODFLOW drain discharge: {e}", exc_info=True)
            return None

        if drain_discharge.empty:
            logger.warning("No drain discharge data extracted from MODFLOW")
            return None

        # Check for coupled mode
        coupling_source = self._get_config_value(
            lambda: self.config.model.modflow.coupling_source,
            default=None,
            dict_key='MODFLOW_COUPLING_SOURCE',
        )

        if coupling_source and str(coupling_source).upper() == 'SUMMA':
            return self._extract_coupled_streamflow(
                drain_discharge, output_dir
            )
        else:
            return self._extract_standalone_baseflow(drain_discharge)

    def _extract_standalone_baseflow(
        self, drain_discharge: pd.Series
    ) -> Tuple[pd.Series, Dict]:
        """Extract baseflow-only streamflow from standalone MODFLOW run."""
        # Convert m3/d → m3/s
        baseflow_m3s = drain_discharge / 86400.0
        baseflow_m3s.name = 'streamflow_m3s'

        metadata = {
            'units': 'm3/s',
            'source': 'MODFLOW6 drain discharge',
            'conversion': 'drain(m3/d) / 86400',
            'mode': 'standalone',
        }

        logger.info(
            f"Extracted MODFLOW baseflow: {len(baseflow_m3s)} timesteps, "
            f"mean={baseflow_m3s.mean():.4f} m3/s"
        )

        # Save to results
        self.save_streamflow_to_results(baseflow_m3s)

        # Generate coupling diagnostics plot
        self._try_generate_plot()

        return baseflow_m3s, metadata

    def _extract_coupled_streamflow(
        self,
        drain_discharge: pd.Series,
        modflow_output_dir: Path,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract combined SUMMA surface runoff + MODFLOW baseflow."""
        try:
            from .coupling import SUMMAToMODFLOWCoupler

            coupler = SUMMAToMODFLOWCoupler(self.config_dict, self.logger)

            # Find SUMMA output
            summa_output_dir = self.get_simulation_dir('SUMMA', self.config.domain.experiment_id)

            if not summa_output_dir.exists():
                logger.warning(
                    f"SUMMA output directory not found: {summa_output_dir}. "
                    "Falling back to standalone baseflow."
                )
                return self._extract_standalone_baseflow(drain_discharge)

            # Extract surface runoff from SUMMA
            surface_runoff = coupler.extract_surface_runoff(summa_output_dir)

            # Get catchment area
            area_m2 = self._get_catchment_area() * 1e6

            # Combine flows
            total_streamflow = coupler.combine_flows(
                surface_runoff, drain_discharge, area_m2
            )

            metadata = {
                'units': 'm3/s',
                'source': 'SUMMA surface runoff + MODFLOW6 drain discharge',
                'conversion': 'surface(kg/m2/s→m3/s) + drain(m3/d→m3/s)',
                'mode': 'coupled_SUMMA_MODFLOW',
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
            return self._extract_standalone_baseflow(drain_discharge)

    def _try_generate_plot(self) -> None:
        """Attempt to generate coupling diagnostics plot if reporting is available."""
        try:
            from .plotter import MODFLOWPlotter
            plotter = MODFLOWPlotter(self.config_dict, self.logger)
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='run_1',
                dict_key='EXPERIMENT_ID',
            )
            result = plotter.plot_coupling_results(experiment_id)
            if result:
                logger.debug(f"MODFLOW coupling plot saved: {result}")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.debug(f"Could not generate MODFLOW coupling plot: {e}", exc_info=True)

    def _get_catchment_area(self) -> float:
        """Get catchment area in km2 (config, else domain shapefile; raises if neither)."""
        area = self._get_config_value(
            lambda: self.config.domain.catchment_area,
            default=None,
            dict_key='CATCHMENT_AREA',
        )
        from symfluence.core.metrics.streamflow_metrics import require_catchment_area_km2
        return require_catchment_area_km2(
            self.config_dict, self.project_dir, self.domain_name, configured_km2=area,
        )
