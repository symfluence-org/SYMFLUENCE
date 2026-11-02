# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FLUXNET Evapotranspiration and Energy Flux Acquisition Handler

Acquires flux tower data from multiple sources:
- AmeriFlux: Automated API access (requires account from ameriflux.lbl.gov)
- FLUXNET2015: Manual download support + processing
- ICOS: European flux data

Provides:
- Latent heat flux (LE) / Evapotranspiration (ET)
- Sensible heat flux (H)
- Net radiation (Rn)
- Ground heat flux (G)
- Quality control flags

Authentication:
    AmeriFlux uses user_id and user_email for API authentication.
    1. Create account at: https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx
    2. Set AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL in config or environment
    3. Or add to .netrc: machine ameriflux.lbl.gov login <user_id> password <email>

References:
- AmeriFlux: https://ameriflux.lbl.gov/
- AmeriFlux API: https://amfcdn.lbl.gov/
- FLUXNET2015: https://fluxnet.org/data/fluxnet2015-dataset/
- ICOS: https://www.icos-cp.eu/
"""
from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from .fluxnet_constants import (
    FLUXNET_VARIABLE_MAPPING,
    convert_le_to_et,
)


@R.acquisition_handlers.add('FLUXNET')
@R.acquisition_handlers.add('FLUXNET2015')
@R.acquisition_handlers.add('AMERIFLUX')
class FLUXNETAcquirer(BaseAcquisitionHandler):
    """
    Acquires FLUXNET/AmeriFlux flux tower data for ET calibration.

    Supports multiple data access methods:
    1. AmeriFlux API (automated, requires user_id and user_email)
    2. Pre-downloaded FLUXNET2015 data (manual download from fluxnet.org)
    3. Direct file path specification

    Configuration:
        FLUXNET_STATION: Site ID (e.g., 'US-Ne1', 'CA-NS7', 'DE-Hai')
        FLUXNET_PATH: Path to pre-downloaded data (optional)
        AMERIFLUX_USER_ID: AmeriFlux account username
        AMERIFLUX_USER_EMAIL: AmeriFlux account email
        AMERIFLUX_DATA_POLICY: 'CCBY4.0' or 'LEGACY' (default: 'CCBY4.0')
        FLUXNET_VARIABLES: Variables to extract (default: LE, H, Rn, G)
        FLUXNET_TEMPORAL_RESOLUTION: 'HH' (half-hourly), 'DD' (daily), 'MM' (monthly)
        FLUXNET_GAP_FILL: Use gap-filled data ('_F_MDS' suffix)
    """

    # AmeriFlux API endpoints (new CDN-based API)
    AMERIFLUX_API_BASE = "https://amfcdn.lbl.gov/api/v1"
    AMERIFLUX_DOWNLOAD_ENDPOINT = "https://amfcdn.lbl.gov/api/v1/data_download"
    AMERIFLUX_SITE_AVAILABILITY = "https://amfcdn.lbl.gov/api/v1/site_availability/AmeriFlux/BASE-BADM"
    AMERIFLUX_DATA_AVAILABILITY = "https://amfcdn.lbl.gov/api/v1/data_availability/AmeriFlux/BASE-BADM"

    # Use FLUXNET_VARIABLE_MAPPING from fluxnet_constants
    VARIABLE_MAPPING = FLUXNET_VARIABLE_MAPPING

    def download(self, output_dir: Path) -> Path:
        """Download FLUXNET data from available sources."""
        self.logger.info("Starting FLUXNET data acquisition")

        output_dir.mkdir(parents=True, exist_ok=True)

        station_id = self._get_config_value(lambda: self.config.evaluation.fluxnet.station, dict_key='FLUXNET_STATION')
        if not station_id:
            raise ValueError("FLUXNET_STATION must be specified in configuration")

        self.logger.info(f"Acquiring data for FLUXNET station: {station_id}")

        # Output file path
        processed_file = output_dir / f"{self.domain_name}_FLUXNET_{station_id}.csv"

        if processed_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing FLUXNET file: {processed_file}")
            return processed_file

        # Try acquisition methods in order of preference
        raw_data = None

        # 1. Check for pre-downloaded data
        local_path = self._get_config_value(lambda: self.config.evaluation.fluxnet.path, dict_key='FLUXNET_PATH')
        if local_path:
            raw_data = self._load_local_data(Path(local_path), station_id)

        # 2. Try AmeriFlux API if we have credentials and it's an AmeriFlux site
        if raw_data is None and station_id.startswith(('US-', 'CA-', 'MX-')):
            user_id, user_email = self._get_ameriflux_credentials()
            if user_id and user_email:
                raw_data = self._download_ameriflux(station_id, user_id, user_email, output_dir)
            else:
                self.logger.info(
                    "AmeriFlux credentials not found for automated download.\n"
                    "To enable automatic download:\n"
                    "1. Create account at: https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx\n"
                    "2. Set AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL in config\n"
                    "   Or add to ~/.netrc: machine ameriflux.lbl.gov login <user_id> password <email>"
                )

        # 3. Fall back to FLUXNET2015 download instructions
        if raw_data is None:
            raw_data = self._try_fluxnet2015_download(station_id, output_dir)

        if raw_data is None:
            raise RuntimeError(
                f"Could not acquire FLUXNET data for {station_id}. "
                f"Options:\n"
                f"1. For AmeriFlux sites (US/CA/MX):\n"
                f"   - Create account at: https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx\n"
                f"   - Set AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL in config\n"
                f"   - Or add to ~/.netrc: machine ameriflux.lbl.gov login <user_id> password <email>\n"
                f"2. Download manually from https://fluxnet.org/ and set FLUXNET_PATH\n"
                f"3. Download from https://ameriflux.lbl.gov/ and place in FLUXNET_PATH"
            )

        # Process and save data
        processed_data = self._process_fluxnet_data(raw_data, station_id)
        processed_data.to_csv(processed_file)

        self.logger.info(f"FLUXNET data saved: {processed_file}")
        return processed_file

    def _get_ameriflux_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get AmeriFlux API credentials (user_id and user_email).

        Returns:
            Tuple of (user_id, user_email) or (None, None) if not found
        """
        user_id = None
        user_email = None

        # 1. Environment variables
        user_id = os.environ.get('AMERIFLUX_USER_ID')
        user_email = os.environ.get('AMERIFLUX_USER_EMAIL')
        if user_id and user_email:
            return user_id, user_email

        # 2. Config file
        user_id = self._get_config_value(lambda: None, default=None, dict_key='AMERIFLUX_USER_ID')
        user_email = self._get_config_value(lambda: None, default=None, dict_key='AMERIFLUX_USER_EMAIL')
        if user_id and user_email:
            return user_id, user_email

        # 3. .netrc file (machine ameriflux.lbl.gov login=user_id password=email)
        from ..utils import read_netrc_credentials

        found = read_netrc_credentials(['ameriflux.lbl.gov'], logger=self.logger)
        if found:
            return found[0], found[1]  # login=user_id, password=email

        return None, None

    def _get_ameriflux_key(self) -> Optional[str]:
        """Legacy method - returns user_id for backward compatibility."""
        user_id, user_email = self._get_ameriflux_credentials()
        return user_id if user_id and user_email else None

    def _load_local_data(self, local_path: Path, station_id: str) -> Optional[pd.DataFrame]:
        """Load pre-downloaded FLUXNET data from local path."""
        self.logger.info(f"Looking for local FLUXNET data in: {local_path}")

        if not local_path.exists():
            self.logger.warning(f"Local path does not exist: {local_path}")
            return None

        # Search for matching files
        patterns = [
            f"*{station_id}*.csv",
            f"*{station_id}*FULLSET*.csv",
            f"FLX_{station_id}_*.csv",
            f"AMF_{station_id}_*.csv",
        ]

        matching_files: List[Path] = []
        for pattern in patterns:
            matching_files.extend(local_path.rglob(pattern))

        if not matching_files:
            self.logger.warning(f"No files found for station {station_id}")
            return None

        # Prefer FULLSET files, then daily, then half-hourly
        resolution_pref = self._get_config_value(lambda: None, default='DD', dict_key='FLUXNET_TEMPORAL_RESOLUTION')

        best_file = None
        for f in matching_files:
            fname = f.name.upper()
            if resolution_pref == 'DD' and '_DD_' in fname:
                best_file = f
                break
            elif resolution_pref == 'HH' and ('_HH_' in fname or '_HR_' in fname):
                best_file = f
                break
            elif resolution_pref == 'MM' and '_MM_' in fname:
                best_file = f
                break

        if best_file is None:
            best_file = matching_files[0]

        self.logger.info(f"Loading FLUXNET data from: {best_file}")

        try:
            # Handle ZIP files
            if best_file.suffix.lower() == '.zip':
                with zipfile.ZipFile(best_file, 'r') as z:
                    csv_files = [n for n in z.namelist() if n.endswith('.csv')]
                    if csv_files:
                        with z.open(csv_files[0]) as f:
                            return pd.read_csv(f)
                    else:
                        self.logger.error(f"No CSV files found in ZIP: {best_file}")
                        return None
            else:
                return pd.read_csv(best_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to load {best_file}: {e}", exc_info=True)
            return None

    def _download_ameriflux(
        self,
        station_id: str,
        user_id: str,
        user_email: str,
        output_dir: Path
    ) -> Optional[pd.DataFrame]:
        """
        Download data from AmeriFlux API using the new CDN-based endpoint.

        The API requires:
        - user_id: AmeriFlux account username
        - user_email: AmeriFlux account email
        - site_ids: Array of site IDs
        - data_product: BASE-BADM, FLUXNET2015, or FLUXNET-CH4
        - data_policy: CCBY4.0, LEGACY, or TIER2
        - intended_use: Description of research purpose
        """
        self.logger.info(f"Downloading from AmeriFlux API: {station_id}")

        # Get data policy for the site
        data_policy = self._get_config_value(lambda: None, default='CCBY4.0', dict_key='AMERIFLUX_DATA_POLICY')

        try:
            # First, check site availability and get correct data policy
            site_policy = self._get_site_data_policy(station_id)
            if site_policy:
                data_policy = site_policy
                self.logger.info(f"Using data policy '{data_policy}' for site {station_id}")

            # Prepare download request payload
            payload = {
                'user_id': user_id,
                'user_email': user_email,
                'site_ids': [station_id],
                'data_product': 'BASE-BADM',
                'data_policy': data_policy,
                'intended_use': self._get_config_value(lambda: None, default='model', dict_key='AMERIFLUX_INTENDED_USE'),
                'description': self._get_config_value(
                    lambda: None,
                    default=f'Hydrological model calibration for {self.domain_name}',
                    dict_key='AMERIFLUX_DESCRIPTION'
                ),
                'is_test': False  # Set True for testing (no email to site PIs)
            }

            headers = {'Content-Type': 'application/json'}

            self.logger.info(f"Submitting AmeriFlux download request for {station_id}")
            response = requests.post(
                self.AMERIFLUX_DOWNLOAD_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"AmeriFlux API response: {result}")

                # Extract download URL from data_urls array
                data_urls = result.get('data_urls', [])
                if data_urls:
                    download_url = data_urls[0].get('url')
                    if download_url:
                        self.logger.info(f"Got download URL: {download_url[:80]}...")
                        return self._download_and_extract(download_url, station_id, output_dir)

                # Fallback: check for direct download_url field
                download_url = result.get('download_url') or result.get('url')
                if download_url:
                    return self._download_and_extract(download_url, station_id, output_dir)

                # If async task, wait for completion
                task_id = result.get('task_id') or result.get('request_id')
                if task_id:
                    return self._wait_for_download(task_id, station_id, output_dir)

                # Direct data return
                if 'data' in result:
                    return pd.DataFrame(result['data'])

                self.logger.warning(f"Unexpected API response format: {result.keys()}")

            elif response.status_code == 401:
                self.logger.error(
                    "AmeriFlux API authentication failed. "
                    "Please verify your AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL"
                )
            elif response.status_code == 403:
                self.logger.error(
                    f"Access denied to site {station_id}. "
                    f"The site may require a different data policy or special access."
                )
            else:
                self.logger.warning(
                    f"AmeriFlux API request failed: {response.status_code} - {response.text}"
                )

        except requests.exceptions.Timeout:
            self.logger.warning("AmeriFlux API request timed out")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"AmeriFlux API request error: {e}")
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"AmeriFlux API download failed: {e}", exc_info=True)

        # Fallback: Try direct data portal approach
        return self._try_ameriflux_direct(station_id, user_id, user_email, output_dir)

    def _get_site_data_policy(self, station_id: str) -> Optional[str]:
        """
        Query AmeriFlux API to determine the correct data policy for a site.

        The site_availability endpoint returns:
        {
            'CCBY4.0': [['US-Ne1', 'Site Name'], ...],
            'LEGACY': [['US-Ne2', 'Site Name'], ...]
        }
        """
        try:
            response = requests.get(self.AMERIFLUX_SITE_AVAILABILITY, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Check each policy for the site
                for policy in ['CCBY4.0', 'LEGACY']:
                    sites = data.get(policy, [])
                    # Sites can be lists of [site_id, site_name] or just site_ids
                    for site in sites:
                        site_id = site[0] if isinstance(site, list) else site
                        if site_id == station_id:
                            self.logger.debug(f"Site {station_id} found under {policy} policy")
                            return policy

                self.logger.debug(f"Site {station_id} not found in availability list")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.debug(f"Could not determine site data policy: {e}", exc_info=True)

        return None

    def _download_and_extract(
        self,
        download_url: str,
        station_id: str,
        output_dir: Path
    ) -> Optional[pd.DataFrame]:
        """Download and extract data from URL."""
        self.logger.info(f"Downloading AmeriFlux data from: {download_url}")

        try:
            response = requests.get(download_url, timeout=300)
            if response.status_code == 200:
                raw_file = output_dir / f"AMF_{station_id}_raw.zip"
                with open(raw_file, 'wb') as f:
                    f.write(response.content)

                # Extract and read
                with zipfile.ZipFile(raw_file, 'r') as z:
                    csv_files = [n for n in z.namelist() if n.endswith('.csv')]
                    if csv_files:
                        # Prefer hourly/daily files if available
                        priority_files = [f for f in csv_files if '_HR_' in f or '_DD_' in f]
                        target_file = priority_files[0] if priority_files else csv_files[0]

                        with z.open(target_file) as f:
                            self.logger.info(f"Reading data from: {target_file}")
                            # AmeriFlux CSVs have comment lines starting with '#'
                            # Use comment='#' to skip metadata headers
                            df = pd.read_csv(
                                f,
                                comment='#',
                                na_values=['-9999', '-9999.0', '-9999.00']
                            )
                            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                            return df

                self.logger.warning("No CSV files found in downloaded archive")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to download/extract AmeriFlux data: {e}", exc_info=True)

        return None

    def _wait_for_download(
        self,
        task_id: str,
        station_id: str,
        output_dir: Path,
        max_wait: int = 600,
        poll_interval: int = 30
    ) -> Optional[pd.DataFrame]:
        """Wait for async download task to complete."""
        self.logger.info(f"Waiting for AmeriFlux download task: {task_id}")

        status_url = f"{self.AMERIFLUX_API_BASE}/task/{task_id}"
        elapsed = 0

        while elapsed < max_wait:
            try:
                response = requests.get(status_url, timeout=30)
                if response.status_code == 200:
                    status = response.json()
                    task_status = status.get('status', '').lower()

                    if task_status in ['done', 'completed', 'ready']:
                        download_url = status.get('download_url') or status.get('url')
                        if download_url:
                            return self._download_and_extract(download_url, station_id, output_dir)

                    elif task_status in ['error', 'failed']:
                        self.logger.error(f"AmeriFlux task failed: {status.get('error')}")
                        return None

                    self.logger.info(f"Task status: {task_status} ({elapsed}s elapsed)")

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Status check failed: {e}", exc_info=True)

            time.sleep(poll_interval)
            elapsed += poll_interval

        self.logger.warning(f"AmeriFlux download task timed out after {max_wait}s")
        return None

    def _try_ameriflux_direct(
        self,
        station_id: str,
        user_id: str,
        user_email: str,
        output_dir: Path
    ) -> Optional[pd.DataFrame]:
        """Try alternative AmeriFlux download method via web portal."""
        self.logger.info("Trying alternative AmeriFlux download...")

        # Provide manual download instructions as fallback
        self.logger.info(
            f"\nAutomated download did not succeed. Manual options:\n"
            f"1. Visit https://ameriflux.lbl.gov/sites/siteinfo/{station_id}\n"
            f"2. Log in with your AmeriFlux account\n"
            f"3. Download the BASE-BADM data product\n"
            f"4. Place the file in: {output_dir}\n"
            f"   Or set FLUXNET_PATH in your config\n"
            f"5. Re-run the pipeline"
        )

        return None

    def _try_fluxnet2015_download(
        self,
        station_id: str,
        output_dir: Path
    ) -> Optional[pd.DataFrame]:
        """
        Attempt FLUXNET2015 data acquisition.

        Note: FLUXNET2015 requires accepting the data policy and manual download.
        This method provides instructions and checks for pre-placed files.
        """
        self.logger.info("Checking for FLUXNET2015 data...")

        # Check common download locations
        possible_paths = [
            output_dir / f"FLX_{station_id}_FLUXNET2015_FULLSET_DD.csv",
            output_dir / f"FLX_{station_id}_FLUXNET2015_FULLSET_HH.csv",
            output_dir.parent / 'raw_data' / f"FLX_{station_id}*.csv",
            Path.home() / 'Downloads' / f"FLX_{station_id}*.csv",
        ]

        for path in possible_paths:
            if '*' in str(path):
                matches = list(path.parent.glob(path.name))
                if matches:
                    self.logger.info(f"Found FLUXNET2015 data: {matches[0]}")
                    return pd.read_csv(matches[0])
            elif path.exists():
                self.logger.info(f"Found FLUXNET2015 data: {path}")
                return pd.read_csv(path)

        # Provide download instructions
        self.logger.warning(
            f"\nFLUXNET2015 data for {station_id} not found.\n"
            f"To download:\n"
            f"1. Visit https://fluxnet.org/data/fluxnet2015-dataset/\n"
            f"2. Register and accept the data use policy\n"
            f"3. Download FULLSET data for site {station_id}\n"
            f"4. Place the CSV/ZIP file in: {output_dir}\n"
            f"5. Re-run the pipeline\n"
        )

        return None

    def _process_fluxnet_data(
        self,
        df: pd.DataFrame,
        station_id: str
    ) -> pd.DataFrame:
        """Process raw FLUXNET data into standardized format."""
        self.logger.info("Processing FLUXNET data")

        # Identify timestamp column
        ts_col = None
        for col in ['TIMESTAMP_START', 'TIMESTAMP', 'datetime', 'Date']:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            raise ValueError("No timestamp column found in FLUXNET data")

        # Parse timestamp
        df['datetime'] = pd.to_datetime(df[ts_col], format='%Y%m%d%H%M', errors='coerce')
        if df['datetime'].isna().all():
            df['datetime'] = pd.to_datetime(df[ts_col], format='%Y%m%d', errors='coerce')
        if df['datetime'].isna().all():
            df['datetime'] = pd.to_datetime(df[ts_col], errors='coerce')

        df = df.dropna(subset=['datetime'])
        df = df.set_index('datetime').sort_index()

        # Filter to requested time period
        df = df.loc[pd.Timestamp(self.start_date):pd.Timestamp(self.end_date)]

        # Extract requested variables
        result = pd.DataFrame(index=df.index)
        result['date'] = df.index

        # Map variables using priority lists
        for target_var, source_vars in self.VARIABLE_MAPPING.items():
            for src in source_vars:
                if src in df.columns:
                    result[target_var] = df[src]
                    self.logger.debug(f"Mapped {target_var} <- {src}")
                    break

        # Convert LE to ET if needed (mm/day)
        if 'LE' in result.columns and 'ET' not in result.columns:
            # Use shared conversion function from fluxnet_constants
            result['ET'] = convert_le_to_et(result['LE'])
            result['ET_from_LE_mm_per_day'] = result['ET']
            self.logger.info("Converted LE (W/m²) to ET (mm/day)")

        # Apply quality filtering if requested
        if self._get_config_value(lambda: None, default=True, dict_key='FLUXNET_QC_FILTER'):
            for var in ['LE', 'H', 'ET']:
                qc_col = f'{var}_QC'
                if qc_col in result.columns and var in result.columns:
                    # QC values: 0=measured, 1=good gap-fill, 2=medium, 3=poor
                    max_qc = self._get_config_value(lambda: None, default=1, dict_key='FLUXNET_MAX_QC')
                    mask = result[qc_col] > max_qc
                    result.loc[mask, var] = np.nan
                    self.logger.info(f"Applied QC filter to {var} (max QC={max_qc})")

        # Resample to daily if half-hourly data
        if len(result) > 0:
            time_diff = (result.index[1] - result.index[0]).total_seconds() / 3600
            if time_diff < 12:  # Sub-daily data
                self.logger.info("Resampling to daily values")
                # For fluxes, sum or mean depending on variable
                result_daily = result.resample('1D').mean()
                result = result_daily

        # Add metadata columns
        result['station_id'] = station_id
        result['data_source'] = 'FLUXNET'

        self.logger.info(
            f"Processed {len(result)} days of FLUXNET data "
            f"({result.index.min()} to {result.index.max()})"
        )

        return result

    def get_site_info(self, station_id: str) -> Dict[str, Any]:
        """Get metadata for a FLUXNET site."""
        # Common site database
        site_db = {
            'CA-NS7': {
                'name': 'Northern Old Black Spruce',
                'lat': 56.6358,
                'lon': -99.9483,
                'country': 'Canada',
                'igbp': 'ENF',  # Evergreen Needleleaf Forest
                'climate': 'Dfc'
            },
            'US-Ne1': {
                'name': 'Mead - irrigated continuous maize site',
                'lat': 41.1651,
                'lon': -96.4766,
                'country': 'USA',
                'igbp': 'CRO',
                'climate': 'Dfa'
            },
            'DE-Hai': {
                'name': 'Hainich',
                'lat': 51.0792,
                'lon': 10.4530,
                'country': 'Germany',
                'igbp': 'DBF',
                'climate': 'Cfb'
            },
            'FR-Pue': {
                'name': 'Puechabon',
                'lat': 43.7413,
                'lon': 3.5957,
                'country': 'France',
                'igbp': 'EBF',
                'climate': 'Csa'
            }
        }

        return site_db.get(station_id, {'name': station_id, 'lat': None, 'lon': None})


@R.acquisition_handlers.add('FLUXNET_ET')
class FLUXNETETAcquirer(FLUXNETAcquirer):
    """Specialized FLUXNET acquirer focused on ET data."""

    def download(self, output_dir: Path) -> Path:
        """Download and process FLUXNET data, focusing on ET output."""
        result_path = super().download(output_dir)

        # Create ET-specific output format compatible with ET evaluator
        df = pd.read_csv(result_path, index_col=0, parse_dates=True)

        et_file = output_dir / f"{self.domain_name}_fluxnet_et_processed.csv"

        # Select ET-relevant columns
        et_cols = ['date', 'ET', 'ET_from_LE_mm_per_day', 'LE', 'LE_QC']
        available_cols = [c for c in et_cols if c in df.columns]

        et_df = df[available_cols].copy()

        # Rename for consistency with ET evaluator
        if 'ET' in et_df.columns:
            et_df = et_df.rename(columns={'ET': 'et_mm_day'})
        elif 'ET_from_LE_mm_per_day' in et_df.columns:
            et_df = et_df.rename(columns={'ET_from_LE_mm_per_day': 'et_mm_day'})

        et_df.to_csv(et_file)
        self.logger.info(f"ET-specific output saved: {et_file}")

        return et_file
