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
        """Extract streamflow from GSFLOW statvar output in a specific directory."""
        # Search multiple locations
        search_dirs = [output_dir]
        settings_path = resolve_config_value(config, lambda: config.model.gsflow.settings_path, None, 'SETTINGS_GSFLOW_PATH')
        if settings_path and settings_path != 'default':
            search_dirs.append(Path(settings_path))

        for search_dir in search_dirs:
            for pattern in ['statvar*', '*.csv', 'gsflow_*.csv']:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return self._extract_from_file(matches[0])

        logger.error(f"No GSFLOW output found in {output_dir}")
        return None

    def _extract_from_file(self, output_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from output file."""
        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                for col in ['seg_outflow', 'basin_cfs', 'streamflow']:
                    if col in df.columns:
                        series = df[col]
                        if col.endswith('_cfs') or col == 'basin_cfs':
                            series = series * 0.0283168
                        return series
            else:
                # Parse statvar format
                lines = output_file.read_text(encoding='utf-8').strip().split('\n')
                dates, values = [], []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        try:
                            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                            streamflow = float(parts[7])
                            dates.append(pd.Timestamp(year=year, month=month, day=day))
                            values.append(streamflow * 0.0283168)
                        except (ValueError, IndexError):
                            continue
                if dates:
                    return pd.Series(values, index=dates, name='GSFLOW_discharge_cms')
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error extracting streamflow: {e}", exc_info=True)
        return None
