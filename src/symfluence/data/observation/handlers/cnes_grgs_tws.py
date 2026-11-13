# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CNES/GRGS GRACE TWS observation handler.

Provides acquisition and preprocessing of CNES/GRGS RL05 regularized
spherical harmonic GRACE solutions for TWS validation. Unlike mascon
solutions, these are already stabilized and require no post-processing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('cnes_grgs')
@R.observation_handlers.add('cnes_grgs_tws')
class CNESGRGSHandler(BaseObservationHandler):
    """
    Handles CNES/GRGS RL05 GRACE Total Water Storage data.
    """

    obs_type = "tws"
    source_name = "CNES_GRGS"

    def acquire(self) -> Path:
        """Locate CNES/GRGS data or download if needed."""
        data_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "tws" / "cnes_grgs", dict_key='CNES_GRGS_DATA_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        has_files = data_dir.exists() and any(data_dir.iterdir())

        if not has_files or force_download:
            self.logger.info("Acquiring CNES/GRGS data...")
            try:
                from symfluence.data.acquisition.handlers.cnes_grgs_tws import CNESGRGSAcquirer
                acquirer = CNESGRGSAcquirer(self.config, self.logger)
                acquirer.download(data_dir)
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"CNES/GRGS acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing CNES/GRGS data in {data_dir}")

        return data_dir

    def process(self, input_path: Path) -> Path:
        """Process CNES/GRGS TWS data."""
        self.logger.info(f"Processing CNES/GRGS TWS for domain: {self.domain_name}")

        # Find raw data
        raw_file = input_path / "cnes_grgs_tws_raw.csv"
        if not raw_file.exists():
            candidates = list(input_path.rglob("*cnes*grgs*.csv"))
            if candidates:
                raw_file = candidates[0]
            else:
                raise FileNotFoundError(f"No CNES/GRGS data found in {input_path}")

        df = pd.read_csv(raw_file, parse_dates=['date'], index_col='date')

        # Re-reference anomaly to baseline period
        baseline_start = self._get_config_value(
            lambda: None,
            default='2004-01-01', dict_key='CNES_GRGS_BASELINE_START')
        baseline_end = self._get_config_value(
            lambda: None,
            default='2009-12-31', dict_key='CNES_GRGS_BASELINE_END')

        if 'tws_anomaly_cm' in df.columns:
            baseline = df.loc[baseline_start:baseline_end, 'tws_anomaly_cm']
            if len(baseline) > 0:
                offset = baseline.mean()
                df['tws_anomaly_cm'] -= offset
                self.logger.info(f"Re-referenced to {baseline_start[:4]}-{baseline_end[:4]} "
                                 f"baseline (offset: {offset:.2f} cm)")

        # Save
        output_dir = self.project_observations_dir / "tws" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_cnes_grgs_tws_processed.csv"
        df.to_csv(output_file)

        self.logger.info(f"CNES/GRGS processing complete: {output_file}")
        return output_file
