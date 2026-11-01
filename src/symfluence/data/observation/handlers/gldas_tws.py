# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GLDAS TWS observation handler.

Provides acquisition and preprocessing of GLDAS-2.1 Noah Land Surface Model
monthly TWS data for model validation. TWS = Soil Moisture + SWE + Canopy Water.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('gldas')
@R.observation_handlers.add('gldas_tws')
class GLDASHandler(BaseObservationHandler):
    """
    Handles GLDAS-2.1 Noah Total Water Storage data.
    """

    obs_type = "tws"
    source_name = "GLDAS_NOAH"

    def acquire(self) -> Path:
        """Locate GLDAS data or download if needed."""
        gldas_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "tws" / "gldas", dict_key='GLDAS_DATA_DIR'))

        force_download = self._get_config_value(
            lambda: self.config.data.force_download, default=False)

        has_files = gldas_dir.exists() and any(gldas_dir.iterdir())

        if not has_files or force_download:
            self.logger.info("Acquiring GLDAS data...")
            try:
                from symfluence.data.acquisition.handlers.gldas_tws import GLDASAcquirer
                acquirer = GLDASAcquirer(self.config, self.logger)
                acquirer.download(gldas_dir)
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"GLDAS acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing GLDAS data in {gldas_dir}")

        return gldas_dir

    def process(self, input_path: Path) -> Path:
        """Process GLDAS TWS data to anomaly time series."""
        self.logger.info(f"Processing GLDAS TWS for domain: {self.domain_name}")

        # Find raw data
        raw_file = input_path / "gldas_noah_tws_raw.csv"
        if not raw_file.exists():
            candidates = list(input_path.rglob("*gldas*tws*.csv"))
            if candidates:
                raw_file = candidates[0]
            else:
                raise FileNotFoundError(f"No GLDAS data found in {input_path}")

        df = pd.read_csv(raw_file, parse_dates=['date'], index_col='date')

        # Compute anomaly
        baseline_start = self._get_config_value(
            lambda: None,
            default='2004-01-01', dict_key='GLDAS_BASELINE_START')
        baseline_end = self._get_config_value(
            lambda: None,
            default='2009-12-31', dict_key='GLDAS_BASELINE_END')

        if 'tws_mm' in df.columns:
            df['tws_cm'] = df['tws_mm'] / 10.0
            baseline = df.loc[baseline_start:baseline_end, 'tws_cm']
            baseline_mean = baseline.mean() if len(baseline) > 0 else df['tws_cm'].mean()
            df['tws_anomaly_cm'] = df['tws_cm'] - baseline_mean
        elif 'tws_anomaly_cm' in df.columns:
            # Already an anomaly, re-reference to baseline
            baseline = df.loc[baseline_start:baseline_end, 'tws_anomaly_cm']
            if len(baseline) > 0:
                df['tws_anomaly_cm'] -= baseline.mean()

        # Save
        output_dir = self.project_observations_dir / "tws" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_gldas_tws_processed.csv"
        df.to_csv(output_file)

        self.logger.info(f"GLDAS processing complete: {output_file}")
        return output_file
