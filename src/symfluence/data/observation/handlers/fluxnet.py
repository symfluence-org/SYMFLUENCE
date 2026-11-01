# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FLUXNET Observation Handler

Provides flux tower observations for ET calibration:
- Latent heat flux (LE) / Evapotranspiration (ET)
- Sensible heat flux (H)
- Net radiation (Rn)
- Quality control flags

Wraps the FLUXNET acquisition handler for the observation pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from ...acquisition.handlers.fluxnet_constants import convert_le_to_et
from ..base import BaseObservationHandler


@R.observation_handlers.add('fluxnet')
@R.observation_handlers.add('fluxnet_et')
@R.observation_handlers.add('ameriflux')
class FLUXNETObservationHandler(BaseObservationHandler):

    obs_type = "et"
    source_name = "FLUXNET"
    """
    FLUXNET flux tower observation handler for ET calibration.

    Acquires and processes FLUXNET/AmeriFlux data for model calibration.
    Supports automatic download via AmeriFlux API or manual data placement.

    Configuration:
        FLUXNET_STATION: Site ID (e.g., 'US-Ne1', 'CA-NS7')
        FLUXNET_PATH: Path to pre-downloaded data (optional)
        AMERIFLUX_API_KEY: API key for automated download
        FLUXNET_QC_FILTER: Apply quality control (default: True)
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)

        self.station_id = resolve_config_value(config, lambda: config.evaluation.fluxnet.station, '', 'FLUXNET_STATION')
        self.domain_name = resolve_config_value(config, lambda: config.domain.name, 'unknown', 'DOMAIN_NAME')

        # Output directories
        self.raw_dir = self.project_observations_dir / 'fluxnet' / 'raw'
        self.processed_dir = self.project_observations_dir / 'et' / 'preprocessed'

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def acquire(self) -> Path:
        """
        Acquire FLUXNET data using the cloud acquisition handler.

        Returns:
            Path to acquired data file
        """
        self.logger.info(f"Acquiring FLUXNET data for station: {self.station_id}")

        try:
            from symfluence.data.acquisition.handlers.fluxnet import FLUXNETETAcquirer

            acquirer = FLUXNETETAcquirer(self.config, self.logger)
            result_path = acquirer.download(self.processed_dir)

            self.logger.info(f"FLUXNET data acquired: {result_path}")
            return result_path

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Failed to acquire FLUXNET data: {e}")

            # Check for pre-existing data
            existing = self._find_existing_data()
            if existing:
                self.logger.info(f"Using existing FLUXNET data: {existing}")
                return existing

            raise

    def _find_existing_data(self) -> Optional[Path]:
        """Find existing FLUXNET data files."""
        search_patterns = [
            f"*{self.station_id}*.csv",
            f"*FLUXNET*{self.station_id}*.csv",
            f"*{self.domain_name}*fluxnet*.csv",
        ]

        search_dirs = [self.processed_dir, self.raw_dir]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in search_patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def process(self, raw_path: Path) -> Path:
        """
        Process FLUXNET data into standardized format.

        The acquisition handler already produces processed data,
        so this mainly handles format conversion if needed.

        Args:
            raw_path: Path to raw/acquired data

        Returns:
            Path to processed data file
        """
        self.logger.info(f"Processing FLUXNET data: {raw_path}")

        output_file = self.processed_dir / f"{self.domain_name}_fluxnet_et_processed.csv"

        if raw_path == output_file:
            # Already processed
            return output_file

        try:
            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

            # Ensure required columns exist
            et_col = None

            # Find ET column
            for col in df.columns:
                if col.lower() in ['et_mm_day', 'et', 'et_from_le_mm_per_day']:
                    et_col = col
                    break

            if et_col is None and 'LE' in df.columns:
                # Convert LE (W/m²) to ET (mm/day) using shared conversion
                df['et_mm_day'] = convert_le_to_et(df['LE'])
                et_col = 'et_mm_day'

            if et_col and et_col != 'et_mm_day':
                df = df.rename(columns={et_col: 'et_mm_day'})

            # Save processed data
            df.to_csv(output_file)
            self.logger.info(f"Processed FLUXNET data saved: {output_file}")

            return output_file

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error processing FLUXNET data: {e}")
            raise

    def get_output_path(self) -> Path:
        """Get path to processed FLUXNET data."""
        return self.processed_dir / f"{self.domain_name}_fluxnet_et_processed.csv"

    def validate(self) -> bool:
        """Validate FLUXNET data availability."""
        output_path = self.get_output_path()

        if not output_path.exists():
            self.logger.warning(f"FLUXNET output not found: {output_path}")
            return False

        try:
            df = pd.read_csv(output_path, index_col=0, parse_dates=True)
            if len(df) == 0:
                self.logger.warning("FLUXNET data file is empty")
                return False

            # Check for ET column
            has_et = any(col.lower() in ['et_mm_day', 'et', 'le'] for col in df.columns)
            if not has_et:
                self.logger.warning("FLUXNET data missing ET/LE column")
                return False

            self.logger.info(f"FLUXNET data validated: {len(df)} records")
            return True

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"FLUXNET validation error: {e}", exc_info=True)
            return False
