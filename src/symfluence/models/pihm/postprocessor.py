# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Model Postprocessor

Extracts river flux (baseflow) from PIHM output and optionally
combines with SUMMA surface runoff for total streamflow.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from symfluence.core.modeling.base.standard_postprocessor import StandardModelPostProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.postprocessors.add("PIHM")
class PIHMPostProcessor(StandardModelPostProcessor):
    """
    Postprocesses PIHM output.

    Extracts river flux (baseflow) and optionally combines with
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
        return "PIHM"

    def extract_streamflow(  # type: ignore[override]
        self,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """
        Extract streamflow from PIHM output, optionally combined with SUMMA.

        Returns:
            Tuple of (streamflow Series in m3/s, metadata dict)
        """
        if output_dir is None:
            output_dir = self.get_simulation_dir('PIHM', self.config.domain.experiment_id)

        try:
            from .extractor import PIHMResultExtractor
            extractor = PIHMResultExtractor()

            start_date = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01',
                dict_key='EXPERIMENT_TIME_START',
            )

            river_flux = extractor.extract_variable(
                output_dir, 'river_flux',
                start_date=str(start_date),
            )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Failed to extract PIHM river flux: {e}", exc_info=True)
            return None

        if river_flux.empty:
            logger.warning("No river flux data extracted from PIHM")
            return None

        coupling_source = self._get_config_value(
            lambda: self.config.model.pihm.coupling_source,
            default=None,
            dict_key='PIHM_COUPLING_SOURCE',
        )

        if coupling_source and str(coupling_source).upper() == 'SUMMA':
            return self._extract_coupled_streamflow(river_flux, output_dir)
        else:
            return self._extract_standalone_baseflow(river_flux)

    def _extract_standalone_baseflow(
        self, river_flux: pd.Series
    ) -> Tuple[pd.Series, Dict]:
        """Extract baseflow-only streamflow from standalone PIHM run."""
        # PIHM river flux is already in m3/s
        river_flux.name = 'streamflow_m3s'

        metadata = {
            'units': 'm3/s',
            'source': 'PIHM river flux',
            'mode': 'standalone',
        }

        logger.info(
            f"Extracted PIHM river flux: {len(river_flux)} timesteps, "
            f"mean={river_flux.mean():.4f} m3/s"
        )

        self.save_streamflow_to_results(river_flux)
        self._try_generate_plot()

        return river_flux, metadata

    def _extract_coupled_streamflow(
        self,
        river_flux: pd.Series,
        pihm_output_dir: Path,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """Extract combined SUMMA surface runoff + PIHM baseflow."""
        try:
            from .coupling import SUMAToPIHMCoupler

            coupler = SUMAToPIHMCoupler(self.config_dict, self.logger)

            summa_output_dir = self.get_simulation_dir('SUMMA', self.config.domain.experiment_id)

            if not summa_output_dir.exists():
                logger.warning(
                    f"SUMMA output directory not found: {summa_output_dir}. "
                    "Falling back to standalone baseflow."
                )
                return self._extract_standalone_baseflow(river_flux)

            surface_runoff = coupler.extract_surface_runoff(summa_output_dir)
            area_m2 = self._get_catchment_area() * 1e6

            total_streamflow = coupler.combine_flows(
                surface_runoff, river_flux, area_m2
            )

            metadata = {
                'units': 'm3/s',
                'source': 'SUMMA surface runoff + PIHM river flux',
                'mode': 'coupled_SUMMA_PIHM',
                'area_km2': self._get_catchment_area(),
            }

            logger.info(
                f"Extracted coupled streamflow: {len(total_streamflow)} timesteps, "
                f"mean={total_streamflow.mean():.4f} m3/s"
            )

            self.save_streamflow_to_results(total_streamflow)
            self._try_generate_plot()

            return total_streamflow, metadata

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Coupled extraction failed: {e}. Falling back to standalone.", exc_info=True)
            return self._extract_standalone_baseflow(river_flux)

    def _try_generate_plot(self) -> None:
        """Attempt to generate coupling diagnostics plot."""
        try:
            from .plotter import PIHMPlotter
            plotter = PIHMPlotter(self.config_dict, self.logger)
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='run_1',
                dict_key='EXPERIMENT_ID',
            )
            result = plotter.plot_coupling_results(experiment_id)
            if result:
                logger.debug(f"PIHM coupling plot saved: {result}")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.debug(f"Could not generate PIHM coupling plot: {e}", exc_info=True)

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
