# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CMC Snow Depth Analysis Acquisition Handler

Provides acquisition for CMC (Canadian Meteorological Centre) Daily Snow Depth
Analysis data (NSIDC-0447).

CMC Snow Overview:
    Data Type: Analyzed snow depth (station + satellite assimilation)
    Resolution: ~24 km (706x706 Northern Hemisphere grid)
    Coverage: Northern Hemisphere
    Variables: Snow depth (cm)
    Temporal: Daily (stored as yearly GeoTIFFs with 365/366 bands)
    Record: 1998-present
    Source: Canadian Meteorological Centre / NSIDC

Data Access:
    Primary: NSIDC via HTTPS (requires Earthdata authentication)
    Format: Yearly GeoTIFFs (one band per day)
    URL: https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0447_CMC_snow_depth_v01/

SWE Conversion:
    SWE (mm) = snow_depth (cm) * (density / 100)
    Default bulk snow density: 200 kg/m^3
    => SWE (mm) = depth_cm * 2.0

References:
    Brown, R. D. and B. Brasnett (2010). Canadian Meteorological Centre (CMC) Daily
    Snow Depth Analysis Data, Version 1. Boulder, CO USA. NASA NSIDC DAAC.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# NSIDC data endpoint
CMC_BASE_URL = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/"
    "nsidc0447_CMC_snow_depth_v01/Snow_Depth/Snow_Depth_Daily_Values/GeoTIFF/"
)

# Default bulk snow density (kg/m^3) for depth-to-SWE conversion
DEFAULT_SNOW_DENSITY = 200.0


@R.acquisition_handlers.add('CMC_SNOW')
@R.acquisition_handlers.add('CMC')
class CMCSnowAcquirer(BaseAcquisitionHandler):
    """
    Acquires CMC Daily Snow Depth Analysis data from NSIDC.

    Downloads yearly GeoTIFFs, extracts the basin subset, converts snow depth
    to SWE, and produces a merged NetCDF file.

    Requires Earthdata authentication (~/.netrc or environment variables).

    Configuration:
        CMC_SNOW_DENSITY: Snow density for depth-to-SWE conversion (default: 200 kg/m^3)
        CMC_TEMPORAL_AGG: 'daily' or 'monthly' (default: 'monthly')
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download and process CMC snow depth data.

        Args:
            output_dir: Directory to save downloaded/processed files

        Returns:
            Path to output directory containing GeoTIFF files
        """
        self.logger.info("Starting CMC Snow Depth acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._get_config_value(lambda: self.config.data.force_download, default=False)
        _snow_density = float(self._get_config_value(lambda: None, default=DEFAULT_SNOW_DENSITY, dict_key='CMC_SNOW_DENSITY'))

        # Get Earthdata credentials
        username, password = self._earthdata_auth().require_credentials(
            "CMC snow depth (NSIDC)"
        )

        # Create authenticated session
        session = requests.Session()
        session.auth = (username, password)
        session.headers.update({'User-Agent': 'symfluence/CMC-snow-download'})

        # Download yearly GeoTIFFs
        start_year = self.start_date.year
        end_year = self.end_date.year

        downloaded_files: Dict[int, Path] = {}
        for year in range(start_year, end_year + 1):
            tif_path = self._download_year(session, year, output_dir, force_download)
            if tif_path:
                downloaded_files[year] = tif_path

        if not downloaded_files:
            raise RuntimeError(
                f"No CMC GeoTIFFs could be downloaded for {start_year}-{end_year}"
            )

        self.logger.info(
            f"CMC acquisition complete: {len(downloaded_files)}/{end_year - start_year + 1} years"
        )
        return output_dir

    def _download_year(
        self,
        session: requests.Session,
        year: int,
        output_dir: Path,
        force: bool
    ) -> Optional[Path]:
        """Download a single year's GeoTIFF from NSIDC."""
        # Try multiple version naming patterns
        filenames = [
            f"cmc_sdepth_dly_{year}_v01.2.tif",
            f"cmc_sdepth_dly_{year}_v01.1.tif",
        ]

        # Check for existing files
        if not force:
            for fn in filenames:
                fp = output_dir / fn
                if fp.exists() and fp.stat().st_size > 1_000_000:
                    self.logger.debug(f"Using existing CMC file: {fn}")
                    return fp

        # Try downloading each version
        for fn in filenames:
            url = CMC_BASE_URL + fn
            fp = output_dir / fn

            for attempt in range(3):
                try:
                    self.logger.debug(f"Downloading {fn} (attempt {attempt + 1})...")
                    resp = session.get(url, timeout=300, allow_redirects=True, stream=True)
                    resp.raise_for_status()

                    # Check we got actual data, not an HTML login page
                    content_type = resp.headers.get('Content-Type', '')
                    if 'html' in content_type.lower():
                        resp = session.get(
                            url, timeout=300, allow_redirects=True,
                            auth=session.auth, stream=True
                        )
                        resp.raise_for_status()

                    total_size = 0
                    with open(fp, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                total_size += len(chunk)

                    if total_size > 1_000_000:
                        self.logger.info(f"Downloaded CMC {year}: {total_size / 1e6:.1f} MB")
                        return fp
                    else:
                        fp.unlink(missing_ok=True)
                        time.sleep(2 ** attempt)

                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and e.response.status_code == 404:
                        break  # Try next filename pattern
                    self.logger.debug(f"HTTP error for {fn}: {e}")
                    time.sleep(2 ** attempt)
                except (OSError, requests.RequestException) as e:
                    self.logger.debug(f"Download error for {fn}: {e}")
                    time.sleep(2 ** attempt)

        self.logger.warning(f"Could not download CMC data for {year}")
        return None
