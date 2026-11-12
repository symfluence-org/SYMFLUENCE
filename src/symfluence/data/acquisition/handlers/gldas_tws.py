# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GLDAS TWS Data Acquisition Handler

Provides cloud acquisition for GLDAS-2.1 Noah Land Surface Model monthly data.
TWS = Soil Moisture (4 layers) + SWE + Canopy Water Storage.
Uses earthaccess for NASA GES DISC authentication.
"""
from __future__ import annotations

from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('GLDAS')
@R.acquisition_handlers.add('GLDAS_TWS')
class GLDASAcquirer(BaseAcquisitionHandler):
    """
    Handles GLDAS-2.1 Noah monthly TWS data acquisition.
    Downloads via earthaccess (requires NASA Earthdata credentials).
    """

    SM_VARS = ['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst',
               'SoilMoi40_100cm_inst', 'SoilMoi100_200cm_inst']
    SWE_VAR = 'SWE_inst'
    CANOPY_VAR = 'CanopInt_inst'

    def download(self, output_dir: Path) -> Path:
        """Download GLDAS-2.1 Noah monthly data and compute TWS."""
        self.logger.info("Starting GLDAS-2.1 Noah TWS acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._parse_bool(self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ))

        output_file = output_dir / "gldas_noah_tws_raw.csv"
        if output_file.exists() and not force_download:
            self.logger.info(f"GLDAS data already exists: {output_file}")
            return output_dir

        # Get config
        start_year = int(self._get_config_value(
            lambda: None,
            default=2002, dict_key='GLDAS_START_YEAR'))
        end_year = int(self._get_config_value(
            lambda: None,
            default=2017, dict_key='GLDAS_END_YEAR'))

        bbox = self._get_bounding_box()

        # Authenticate
        self.logger.info("Authenticating with NASA Earthdata...")
        self._earthdata_auth().earthaccess_login(strategy="netrc", required=True)

        # Search for granules
        self.logger.info(f"Searching GLDAS-2.1 monthly ({start_year}-{end_year})...")
        results = earthaccess.search_data(
            short_name="GLDAS_NOAH025_M",
            version="2.1",
            temporal=(f"{start_year}-01-01", f"{end_year}-12-31"),
            bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),
        )
        self.logger.info(f"Found {len(results)} granules")

        if not results:
            self.logger.error("No GLDAS granules found")
            return output_dir

        # Download to local directory
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        self.logger.info(f"Downloading {len(results)} files...")
        files = earthaccess.download(results, str(raw_dir))
        self.logger.info(f"Downloaded {len(files)} files")

        # Process each file
        all_data = []
        for f in sorted(files):
            try:
                ds = xr.open_dataset(f, engine='h5netcdf')
                ds_sub = ds.sel(
                    lat=slice(bbox[1], bbox[3]),
                    lon=slice(bbox[0], bbox[2]),
                )

                row = {}
                if 'time' in ds_sub.dims:
                    row['date'] = pd.Timestamp(ds_sub.time.values[0])
                else:
                    row['date'] = pd.Timestamp(Path(f).stem.split('.A')[1][:6][:4],
                                               int(Path(f).stem.split('.A')[1][:6][4:6]), 1)

                total_sm = sum(
                    float(ds_sub[v].mean(dim=['lat', 'lon']).values)
                    for v in self.SM_VARS if v in ds_sub
                    and not np.isnan(float(ds_sub[v].mean(dim=['lat', 'lon']).values))
                )
                row['soil_moisture_mm'] = total_sm

                if self.SWE_VAR in ds_sub:
                    row['swe_mm'] = float(ds_sub[self.SWE_VAR].mean(dim=['lat', 'lon']).values)
                if self.CANOPY_VAR in ds_sub:
                    row['canopy_mm'] = float(ds_sub[self.CANOPY_VAR].mean(dim=['lat', 'lon']).values)

                row['tws_mm'] = (row.get('soil_moisture_mm', 0) +
                                 row.get('swe_mm', 0) +
                                 row.get('canopy_mm', 0))
                all_data.append(row)
                ds.close()
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {f}: {e}", exc_info=True)

        if all_data:
            df = pd.DataFrame(all_data).set_index('date').sort_index()
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
