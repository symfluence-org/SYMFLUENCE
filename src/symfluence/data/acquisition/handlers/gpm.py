# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GPM IMERG Precipitation Acquisition Handler

Provides cloud acquisition for NASA GPM IMERG (Integrated Multi-satellitE Retrievals
for GPM) precipitation data.

GPM IMERG Overview:
    Data Type: Satellite-derived precipitation
    Resolution: 0.1° x 0.1° (~10km)
    Coverage: Global (60°N to 60°S)
    Temporal: Half-hourly, daily, monthly
    Source: NASA GES DISC

Products:
    - GPM_3IMERGDF: Final Run (research quality, ~3.5 month latency)
    - GPM_3IMERGDL: Late Run (~14 hour latency)
    - GPM_3IMERGDE: Early Run (~4 hour latency)
    - GPM_3IMERGM: Monthly

Authentication:
    Requires NASA Earthdata Login credentials:
    - ~/.netrc file with machine urs.earthdata.nasa.gov entry
    - Or EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables

Data Access:
    Primary: GES DISC OPeNDAP/HTTPS subsetting
    Fallback: CMR granule-by-granule download
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List

import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('GPM_IMERG')
class GPMIMERGAcquirer(BaseAcquisitionHandler):
    """
    Acquires GPM IMERG precipitation data via GES DISC.
    Requires Earthdata Login credentials (via ~/.netrc or env vars).
    """

    # GES DISC endpoints
    GES_DISC_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov"
    CMR_BASE = "https://cmr.earthdata.nasa.gov/search"

    # Product configurations
    PRODUCTS = {
        'final': {
            'short_name': 'GPM_3IMERGDF',
            'version': '07',
            'collection': 'GPM_3IMERGDF.07',
            'opendap_path': '/opendap/GPM_L3/GPM_3IMERGDF.07',
        },
        'late': {
            'short_name': 'GPM_3IMERGDL',
            'version': '07',
            'collection': 'GPM_3IMERGDL.07',
            'opendap_path': '/opendap/GPM_L3/GPM_3IMERGDL.07',
        },
        'early': {
            'short_name': 'GPM_3IMERGDE',
            'version': '07',
            'collection': 'GPM_3IMERGDE.07',
            'opendap_path': '/opendap/GPM_L3/GPM_3IMERGDE.07',
        },
        'monthly': {
            'short_name': 'GPM_3IMERGM',
            'version': '07',
            'collection': 'GPM_3IMERGM.07',
            'opendap_path': '/opendap/GPM_L3/GPM_3IMERGM.07',
        },
    }

    def download(self, output_dir: Path) -> Path:
        """
        Download GPM IMERG precipitation data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to output directory containing downloaded files
        """
        self.logger.info("Starting GPM IMERG precipitation acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get product type from config
        product_type = self._get_config_value(
            lambda: None,
            default='final',
            dict_key='GPM_PRODUCT'
        )
        if isinstance(product_type, str):
            product_type = product_type.lower()
        if product_type not in self.PRODUCTS:
            self.logger.warning(f"Unknown GPM product '{product_type}', defaulting to 'final'")
            product_type = 'final'

        product = self.PRODUCTS[product_type]

        # Output file
        out_nc = output_dir / f"{self.domain_name}_GPM_IMERG_{product_type}_raw.nc"
        if self._skip_if_exists(out_nc):
            return output_dir

        # Try OPeNDAP subsetting first
        try:
            self._download_via_opendap(product, out_nc)
            return output_dir
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"OPeNDAP download failed: {e}, trying CMR granule download", exc_info=True)

        # Fallback to CMR granule download
        return self._download_via_cmr(product, output_dir)

    def _download_via_opendap(self, product: dict, output_file: Path) -> None:
        """Download via GES DISC OPeNDAP with spatial/temporal subsetting."""

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        # Setup authenticated session
        session = requests.Session()
        # EarthdataAuthError subclasses PermissionError
        session.auth = self._earthdata_auth().require_credentials("GPM IMERG (GES DISC OPeNDAP)")

        # Generate list of dates
        dates = []
        current = self.start_date
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)

        self.logger.info(f"Downloading GPM IMERG for {len(dates)} days via OPeNDAP")

        datasets = []
        for date in dates:
            # Construct OPeNDAP URL for the specific day
            year = date.strftime('%Y')
            doy = date.strftime('%j')
            date_str = date.strftime('%Y%m%d')

            # GPM IMERG daily files naming convention
            filename = f"3B-DAY.MS.MRG.3IMERG.{date_str}-S000000-E235959.V07B.nc4"
            opendap_url = f"{self.GES_DISC_BASE}{product['opendap_path']}/{year}/{doy}/{filename}"

            try:
                # Use xarray with OPeNDAP
                ds = xr.open_dataset(
                    opendap_url,
                    engine='netcdf4',
                    decode_times=True
                )

                # Subset spatially
                # GPM IMERG uses lon: -180 to 180, lat: -90 to 90
                ds_sub = ds.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max)
                )

                # Select precipitation variable
                if 'precipitation' in ds_sub:
                    precip_var = 'precipitation'
                elif 'precipitationCal' in ds_sub:
                    precip_var = 'precipitationCal'
                else:
                    # Find first precipitation-like variable
                    precip_vars = [v for v in ds_sub.data_vars if 'precip' in v.lower()]
                    if precip_vars:
                        precip_var = precip_vars[0]
                    else:
                        ds.close()
                        continue

                # Keep only the precipitation variable
                ds_precip = ds_sub[[precip_var]].copy()
                ds_precip = ds_precip.rename({precip_var: 'precipitation'})

                # Add time coordinate if missing
                if 'time' not in ds_precip.dims:
                    ds_precip = ds_precip.expand_dims(time=[date])

                datasets.append(ds_precip)
                ds.close()

                self.logger.debug(f"Downloaded GPM for {date_str}")

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed to download GPM for {date_str}: {e}", exc_info=True)
                continue

        if not datasets:
            raise RuntimeError("No GPM IMERG data could be downloaded via OPeNDAP")

        # Concatenate along time dimension
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')

        # Save to NetCDF
        merged.to_netcdf(output_file)
        self.logger.info(f"Saved GPM IMERG data to {output_file}")

        # Cleanup
        for ds in datasets:
            ds.close()

    def _download_via_cmr(self, product: dict, output_dir: Path) -> Path:
        """Fallback: download GPM granules via CMR search and HTTPS."""
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        temporal = f"{self.start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}," \
                   f"{self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"

        params = {
            "short_name": product['short_name'],
            "version": product['version'],
            "temporal": temporal,
            "bounding_box": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "page_size": 200,
            "page_num": 1,
        }

        self.logger.info(f"Searching CMR for GPM IMERG granules ({product['short_name']})")

        session = requests.Session()
        user, password = self._get_earthdata_credentials()
        if user and password:
            session.auth = (user, password)

        max_granules = self._get_config_value(
            lambda: None,
            default=None,
            dict_key='GPM_MAX_GRANULES'
        )

        downloaded = 0
        downloaded_files: List[Path] = []

        while True:
            resp = session.get(
                f"{self.CMR_BASE}/granules.json",
                params=params,
                timeout=120
            )
            resp.raise_for_status()
            entries = resp.json().get("feed", {}).get("entry", [])

            if not entries:
                break

            for entry in entries:
                if max_granules and downloaded >= int(max_granules):
                    self.logger.info(f"Reached max granules limit ({max_granules})")
                    break

                # Find data download link
                links = entry.get("links", [])
                data_url = None
                for link in links:
                    href = link.get("href", "")
                    if "data#" in link.get("rel", "") and href.endswith((".nc4", ".nc", ".HDF5")):
                        data_url = href
                        break

                if not data_url:
                    continue

                filename = data_url.split("/")[-1]
                out_file = output_dir / filename

                if out_file.exists() and not self._get_config_value(
                    lambda: self.config.data.force_download,
                    default=False,
                    dict_key='FORCE_DOWNLOAD'
                ):
                    downloaded += 1
                    downloaded_files.append(out_file)
                    continue

                self.logger.info(f"Downloading: {filename}")
                try:
                    with session.get(data_url, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        tmp_file = out_file.with_suffix(out_file.suffix + ".part")
                        with open(tmp_file, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                        tmp_file.replace(out_file)
                    downloaded += 1
                    downloaded_files.append(out_file)
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Failed to download {filename}: {e}", exc_info=True)

            if max_granules and downloaded >= int(max_granules):
                break

            params["page_num"] += 1

        if downloaded == 0:
            raise RuntimeError(
                "No GPM IMERG granules found via CMR for the requested bounds/time range."
            )

        self.logger.info(f"Downloaded {downloaded} GPM IMERG granules to {output_dir}")
        return output_dir
