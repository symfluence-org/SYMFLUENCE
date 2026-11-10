# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Post-Processor.

Extracts and processes GSFLOW model outputs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.modeling.base import StandardModelPostProcessor

logger = logging.getLogger(__name__)


class GSFLOWPostProcessor(StandardModelPostProcessor):
    """Post-processor for GSFLOW model outputs."""

    model_name = "GSFLOW"
    output_file_pattern = "statvar*"
    streamflow_variable = "seg_outflow"
    streamflow_unit = "cms"

    def extract_streamflow(self) -> Optional[Path]:
        """Extract streamflow from GSFLOW statvar/CSV output.

        Returns:
            Path to saved results CSV, or None if extraction fails.
        """
        try:
            self.logger.info("Extracting GSFLOW streamflow results")
            config = self.config if isinstance(self.config, dict) else {
                'SETTINGS_GSFLOW_PATH': self._get_config_value(
                    lambda: self.config.model.gsflow.settings_path,
                    default='default', dict_key='SETTINGS_GSFLOW_PATH'
                )
            }
            streamflow = self.extract_streamflow_from_dir(self.sim_dir, config)
            if streamflow is None:
                return None
            return self.save_streamflow_to_results(streamflow)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error extracting GSFLOW streamflow: {e}", exc_info=True)
            return None

    def extract_streamflow_from_dir(
        self,
        output_dir: Path,
        config: dict,
    ) -> Optional[pd.Series]:
        """Extract streamflow from GSFLOW output in a specific directory.

        Same rule as the calibration worker (``extract_gsflow_streamflow``):
        gsflow.csv StreamOut_Q when present (routed SFR outflow in COUPLED
        mode), else the PRMS statvar via the shared PRMS parser.
        """
        from .extractor import extract_gsflow_streamflow

        settings_path = resolve_config_value(config, lambda: config.model.gsflow.settings_path, None, 'SETTINGS_GSFLOW_PATH')
        settings_dir = Path(settings_path) if settings_path and settings_path != 'default' else None
        try:
            series = extract_gsflow_streamflow(output_dir, settings_dir)
        except ModelExecutionError as e:
            logger.error(f"{e}")
            return None
        if series is None:
            logger.error(f"No GSFLOW output found in {output_dir}")
        return series
