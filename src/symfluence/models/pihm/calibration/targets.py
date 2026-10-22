# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Streamflow Calibration Target

Extracts simulated streamflow from PIHM .rivflx output (river flux)
and compares against observed streamflow for metric calculation
during calibration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.calibration_targets.add('PIHM_STREAMFLOW')
class PIHMStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for PIHM integrated hydrologic model."""

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger,
        settings_dir: Optional[str] = None,
    ):
        super().__init__(config, project_dir, logger)
        self.model_name = 'PIHM'
        self.settings_dir = Path(settings_dir) if settings_dir else None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Return PIHM output files as simulation file list.

        MM-PIHM output is in: sim_dir/output/pihm_lumped/*.river.flx1.txt
        Search both the direct directory and the MM-PIHM subdirectory.
        """
        # MM-PIHM directory structure: output/pihm_lumped/
        pihm_output = sim_dir / "output" / "pihm_lumped"
        if pihm_output.exists():
            files = sorted(pihm_output.glob('*.river.flx*.txt'))
            if files:
                return files
            files = sorted(pihm_output.glob('*.gw.txt'))
            if files:
                return files

        # Fallback: search recursively
        files = sorted(sim_dir.rglob('*.river.flx*.txt'))
        if files:
            return files

        # Legacy patterns
        rivflx_files = sorted(sim_dir.glob('*.rivflx*'))
        if rivflx_files:
            return rivflx_files
        return sorted(sim_dir.glob('*.gwhead*'))

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow from PIHM output."""
        if not sim_files:
            return pd.Series(dtype=float)

        output_dir = sim_files[0].parent
        streamflow = self._extract_pihm_streamflow(output_dir)
        if streamflow is None:
            return pd.Series(dtype=float)
        return streamflow

    def _extract_pihm_streamflow(
        self,
        output_dir: Path,
    ) -> Optional[pd.Series]:
        """Extract streamflow from PIHM output directory."""
        try:
            from symfluence.models.pihm.extractor import PIHMResultExtractor

            extractor = PIHMResultExtractor()
            start_date = str(self._get_config_value(
                lambda: self.config.domain.time_start,
                default='2000-01-01',
                dict_key='EXPERIMENT_TIME_START'
            ))

            streamflow = extractor.extract_variable(
                output_dir, 'river_flux', start_date=start_date,
                outlet_segment=int(self._get_config_value(
                    lambda: self.config.model.pihm.outlet_segment,
                    default=1, dict_key='PIHM_OUTLET_SEGMENT')),
            )

            if streamflow.empty:
                self.logger.warning("No PIHM river flux data extracted")
                return None

            streamflow.name = 'streamflow_m3s'

            self.logger.debug(
                f"PIHM streamflow: {len(streamflow)} daily values, "
                f"mean={streamflow.mean():.4f} m3/s"
            )

            return streamflow

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to extract PIHM streamflow: {e}", exc_info=True)
            return None
