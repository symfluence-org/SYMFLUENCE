# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GLEAM Evapotranspiration Acquisition Handler

Provides acquisition for GLEAM (Global Land Evaporation Amsterdam Model) ET data.

GLEAM Overview:
    Data Type: Model-based evapotranspiration (satellite-driven)
    Resolution: 0.25 deg
    Coverage: Global land
    Variables: E (total evaporation), Ep (potential evaporation), components
    Temporal: Daily or monthly
    Record: 1980-present (v4.x)
    Source: Vrije Universiteit Amsterdam / Ghent University

Data Access:
    Primary: SFTP server at hydras.ugent.be:2225
    Requires free registration at https://www.gleam.eu/
    Format: Yearly NetCDF files

Credentials:
    Looked up in order:
    1. Config keys: GLEAM_USERNAME, GLEAM_PASSWORD
    2. Environment variables: GLEAM_USERNAME, GLEAM_PASSWORD
    3. Credentials file: ~/.gleam (username=... password=...)

References:
    Martens et al. (2017): GLEAM v3. Geoscientific Model Development.
    Miralles et al. (2011): Global land-surface evaporation. Hydrology and
    Earth System Sciences.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

GLEAM_HOST = 'hydras.ugent.be'
GLEAM_PORT = 2225
DEFAULT_VERSION = 'v4.2a'


@R.acquisition_handlers.add('GLEAM_ET')
@R.acquisition_handlers.add('GLEAM')
class GLEAMETAcquirer(BaseAcquisitionHandler):
    """
    Acquires GLEAM evapotranspiration data via SFTP.

    GLEAM data is distributed via SFTP from Ghent University. Free registration
    is required at https://www.gleam.eu/ to obtain credentials.

    Configuration:
        GLEAM_USERNAME: SFTP username
        GLEAM_PASSWORD: SFTP password
        GLEAM_VERSION: Data version (default: 'v4.2a')
        GLEAM_VARIABLE: Variable to download (default: 'E' for total evaporation)
        GLEAM_TEMPORAL: 'monthly' or 'daily' (default: 'monthly')
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download GLEAM ET data via SFTP.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to output directory containing NetCDF files
        """
        self.logger.info("Starting GLEAM ET acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        version = self._get_config_value(lambda: None, default=DEFAULT_VERSION, dict_key='GLEAM_VERSION')
        variable = self._get_config_value(lambda: None, default='E', dict_key='GLEAM_VARIABLE')
        temporal = self._get_config_value(lambda: None, default='monthly', dict_key='GLEAM_TEMPORAL')

        # Get credentials
        username, password = self._get_gleam_credentials()
        if not username or not password:
            self._log_credential_instructions()
            raise RuntimeError(
                "GLEAM credentials required. Register at https://www.gleam.eu/ "
                "and configure credentials (see log for details)."
            )

        # Check for existing files
        start_year = self.start_date.year
        end_year = self.end_date.year

        existing = list(output_dir.glob("*.nc"))
        if existing and not force_download:
            self.logger.info(f"Using existing GLEAM files: {len(existing)} NetCDFs")
            return output_dir

        # Connect via SFTP and download
        downloaded = self._download_via_sftp(
            username, password, version, variable, temporal,
            start_year, end_year, output_dir
        )

        if not downloaded:
            raise RuntimeError(
                f"No GLEAM data could be downloaded for {start_year}-{end_year}. "
                "Check credentials and server availability."
            )

        self.logger.info(f"GLEAM acquisition complete: {len(downloaded)} files")
        return output_dir

    def _get_gleam_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get GLEAM SFTP credentials from config, env vars, or ~/.gleam file.
        """
        # 1. Config
        username = self._get_config_value(lambda: None, default=None, dict_key='GLEAM_USERNAME')
        password = self._get_config_value(lambda: None, default=None, dict_key='GLEAM_PASSWORD')
        if username and password:
            return username, password

        # 2. Environment variables
        username = os.environ.get('GLEAM_USERNAME')
        password = os.environ.get('GLEAM_PASSWORD')
        if username and password:
            return username, password

        # 3. ~/.gleam credentials file
        cred_file = Path.home() / '.gleam'
        if cred_file.exists():
            creds = {}
            for line in cred_file.read_text(encoding='utf-8').strip().splitlines():
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    creds[key.strip()] = val.strip()
            username = creds.get('username')
            password = creds.get('password')
            if username and password:
                return username, password

        return None, None

    def _download_via_sftp(
        self,
        username: str,
        password: str,
        version: str,
        variable: str,
        temporal: str,
        start_year: int,
        end_year: int,
        output_dir: Path
    ) -> List[Path]:
        """Download yearly NetCDF files via SFTP."""
        try:
            import paramiko  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "paramiko is required for GLEAM SFTP downloads. "
                "Install with: pip install paramiko"
            ) from None

        downloaded: List[Path] = []

        ssh = paramiko.SSHClient()
        # Security rationale: legacy GLEAM host has no distributed host-key manifest.
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # nosec

        try:
            self.logger.info(f"Connecting to {GLEAM_HOST}:{GLEAM_PORT}...")
            ssh.connect(
                hostname=GLEAM_HOST,
                port=GLEAM_PORT,
                username=username,
                password=password,
                timeout=60
            )
            sftp = ssh.open_sftp()

            # Try to discover available directory structure
            try:
                root_items = sftp.listdir('/data/')
                self.logger.debug(f"Available versions: {sorted(root_items)}")
            except Exception:  # noqa: BLE001 — preprocessing resilience
                pass

            # Path patterns for different GLEAM versions
            path_patterns = [
                f'/data/{version}/{temporal}/{variable}_{{year}}_GLEAM_{version}_{temporal}.nc',
                f'/data/{version}/{temporal}/{variable}_{{year}}_GLEAM_{version}.nc',
                f'/data/{version}/{{year}}/{variable}_{{year}}_GLEAM_{version}.nc',
                f'/data/{version}/daily/{variable}_{{year}}_GLEAM_{version}.nc',
            ]

            for year in range(start_year, end_year + 1):
                found = False
                for pattern in path_patterns:
                    remote_path = pattern.format(year=year)
                    local_file = output_dir / Path(remote_path).name

                    if local_file.exists() and local_file.stat().st_size > 0:
                        self.logger.debug(f"Existing: {local_file.name}")
                        downloaded.append(local_file)
                        found = True
                        break

                    try:
                        self.logger.info(f"Downloading {Path(remote_path).name}...")
                        sftp.get(remote_path, str(local_file))
                        size_mb = local_file.stat().st_size / 1e6
                        self.logger.info(f"  Downloaded: {size_mb:.1f} MB")
                        downloaded.append(local_file)
                        found = True
                        break
                    except FileNotFoundError:
                        if local_file.exists() and local_file.stat().st_size == 0:
                            local_file.unlink()
                        continue
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.debug(f"Error downloading {remote_path}: {e}", exc_info=True)
                        if local_file.exists() and local_file.stat().st_size == 0:
                            local_file.unlink()
                        continue

                if not found:
                    self.logger.warning(f"No GLEAM {variable} file found for {year}")

            sftp.close()

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"SFTP connection error: {e}", exc_info=True)
        finally:
            ssh.close()

        return downloaded

    def _log_credential_instructions(self):
        """Log instructions for obtaining GLEAM credentials."""
        self.logger.info(
            "\n"
            "================================================================\n"
            "GLEAM Data Access Instructions\n"
            "================================================================\n"
            "\n"
            "GLEAM data requires free registration:\n"
            "  1. Register at: https://www.gleam.eu/\n"
            "  2. Provide credentials via one of:\n"
            "     a. Config: GLEAM_USERNAME / GLEAM_PASSWORD\n"
            "     b. Environment: GLEAM_USERNAME / GLEAM_PASSWORD\n"
            "     c. File ~/.gleam with:\n"
            "        username=your_username\n"
            "        password=your_password\n"
            "\n"
            "================================================================"
        )
