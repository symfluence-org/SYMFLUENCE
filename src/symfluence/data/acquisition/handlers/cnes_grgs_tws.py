# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CNES/GRGS GRACE TWS Data Acquisition Handler

Provides acquisition of CNES/GRGS RL05 regularized spherical harmonic
GRACE solutions. Downloads pre-computed 1° EWH grids from the ForM@Ter
catalogue (no authentication required).

Reference: Lemoine et al. (2007), Bruinsma et al. (2010)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('CNES_GRGS')
@R.acquisition_handlers.add('CNES_GRGS_TWS')
class CNESGRGSAcquirer(BaseAcquisitionHandler):
    """
    Handles CNES/GRGS RL05 GRACE TWS data acquisition.
    Downloads 1° EWH grids from ForM@Ter (publicly available).
    """

    CATALOGUE_UUID = "3f81e4f1-4591-4d0e-bcec-43381b4d2949"
    API_BASE = "https://api.sedoo.fr/formater-catalogue-prod/datasetcontent/v1_0"

    def download(self, output_dir: Path) -> Path:
        """Download CNES/GRGS RL05 monthly EWH grids."""
        self.logger.info("Starting CNES/GRGS RL05 GRACE TWS acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._parse_bool(self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ))

        output_file = output_dir / "cnes_grgs_tws_raw.csv"
        if output_file.exists() and not force_download:
            self.logger.info(f"CNES/GRGS data already exists: {output_file}")
            return output_dir

        start_year = int(self._get_config_value(
            lambda: None,
            default=2002, dict_key='CNES_GRGS_START_YEAR'))
        end_year = int(self._get_config_value(
            lambda: None,
            default=2017, dict_key='CNES_GRGS_END_YEAR'))

        bbox = self._get_bounding_box()

        # Convert lon to 0-360 for CNES grid
        lon_min_360 = bbox[0] + 360.0 if bbox[0] < 0 else bbox[0]
        lon_max_360 = bbox[2] + 360.0 if bbox[2] < 0 else bbox[2]

        # Get file list
        self.logger.info("Fetching file list from ForM@Ter...")
        url = f"{self.API_BASE}/content?collection={self.CATALOGUE_UUID}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        files = resp.json().get('files', [])

        grid_files = []
        for f in files:
            name = f if isinstance(f, str) else f.get('name', '')
            if not name.endswith('.txt'):
                continue
            match = re.search(r'GWH-2_(\d{4})(\d{3})-(\d{4})(\d{3})_', name)
            if match and start_year <= int(match.group(1)) <= end_year:
                grid_files.append(name)

        grid_files.sort()
        self.logger.info(f"Found {len(grid_files)} monthly grid files")

        # Download and extract spatial average
        all_data = []
        for i, fname in enumerate(grid_files):
            try:
                dl_url = (f"{self.API_BASE}/getresource?collection={self.CATALOGUE_UUID}"
                          f"&resource=/{fname}"
                          f"&catalogueName=SEDOOFORMATER&projectName=FORMATER.GRACE")
                text = requests.get(dl_url, timeout=120).text

                # Parse ASCII grid
                lons, lats, vals = [], [], []
                for line in text.strip().split('\n'):
                    if line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        lons.append(float(parts[0]))
                        lats.append(float(parts[1]))
                        vals.append(float(parts[2]))

                lons = np.array(lons)
                lats = np.array(lats)
                vals = np.array(vals)

                mask = ((lats >= bbox[1]) & (lats <= bbox[3]) &
                        (lons >= lon_min_360) & (lons <= lon_max_360))

                if mask.sum() == 0:
                    continue

                # Parse date from filename
                match = re.search(r'GWH-2_(\d{4})(\d{3})-(\d{4})(\d{3})_', fname)
                start = datetime(int(match.group(1)), 1, 1) + timedelta(days=int(match.group(2)) - 1)
                end = datetime(int(match.group(3)), 1, 1) + timedelta(days=int(match.group(4)) - 1)
                mid = start + (end - start) / 2
                date = pd.Timestamp(mid.year, mid.month, 1)

                all_data.append({'date': date, 'tws_anomaly_cm': float(vals[mask].mean())})

                if (i + 1) % 12 == 0:
                    self.logger.info(f"  Processed {i + 1}/{len(grid_files)}")

            except Exception as e:  # noqa: BLE001 — data acquisition resilience
                self.logger.warning(f"Failed {fname}: {e}")

        if all_data:
            df = pd.DataFrame(all_data).set_index('date').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.to_csv(output_file, index_label='date')
            self.logger.info(f"Saved {len(df)} months to {output_file}")

        return output_dir

    def _get_bounding_box(self):
        """Get bounding box as (lon_min, lat_min, lon_max, lat_max)."""
        try:
            bbox = self.config.domain.bounding_box
            return (bbox['lon_min'], bbox['lat_min'], bbox['lon_max'], bbox['lat_max'])
        except (AttributeError, KeyError):
            lat_min = float(self._get_config_value(lambda: None, default=-90, dict_key='LATITUDE_MIN'))
            lat_max = float(self._get_config_value(lambda: None, default=90, dict_key='LATITUDE_MAX'))
            lon_min = float(self._get_config_value(lambda: None, default=-180, dict_key='LONGITUDE_MIN'))
            lon_max = float(self._get_config_value(lambda: None, default=180, dict_key='LONGITUDE_MAX'))
            return (lon_min, lat_min, lon_max, lat_max)

    @staticmethod
    def _parse_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', 'yes', '1')
        return bool(val)
