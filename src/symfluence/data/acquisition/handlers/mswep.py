# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MSWEP Precipitation Data Acquisition Handler

Provides acquisition for Multi-Source Weighted-Ensemble Precipitation (MSWEP)
data. MSWEP is a global precipitation product that merges gauge, satellite,
and reanalysis data with optimal weighting.

MSWEP v2.8/v3 features:
- 0.1 deg spatial resolution (approximately 10 km)
- 3-hourly temporal resolution (also daily/monthly available)
- Global land+ocean coverage
- Long record: 1979-present
- Real-time (NRT) updates

Data access requires registration at https://www.gloh2o.org/mswep/ and
Rclone configured with a Google Drive remote pointing to the shared
MSWEP folder.

Products:
- Past: Historical satellite-model merged data with gauge corrections
- Past_nogauge: Historical data without gauge corrections (for validation)
- NRT: Near real-time variant with ~3-hour latency

License: CC BY-NC 4.0

References:
    Beck, H.E., et al. (2019). MSWEP V2 Global 3-Hourly 0.1 deg Precipitation.
    Bull. Am. Meteorol. Soc., 100, 473-500.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Set

import pandas as pd

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('MSWEP')
class MSWEPAcquirer(BaseAcquisitionHandler):
    """
    Handles MSWEP precipitation data acquisition via Rclone + Google Drive.

    MSWEP is distributed via Google Drive shared folders, accessed through
    Rclone. Users must register at https://www.gloh2o.org/mswep/ and
    configure an Rclone remote.

    Setup:
        1. Request access at https://www.gloh2o.org/mswep/
        2. Install rclone (https://rclone.org/downloads/)
        3. Run ``rclone config`` and create a Google Drive remote
        4. Verify: ``rclone lsd --drive-shared-with-me <remote>:``

    Configuration:
        MSWEP_RCLONE_REMOTE: Rclone remote name (default: GoogleDrive)
        MSWEP_VERSION: e.g. 'V280', 'V315', 'V316' (default: V316).
            Note: GloH2O flags a low-precipitation artifact over
            2000-2015 in V3.15/V3.16.
        MSWEP_RESOLUTION: 'daily', '3hourly', 'monthly' (default: daily)
        MSWEP_PRODUCT: 'Past', 'Past_nogauge', 'NRT' (default: Past)
        MSWEP_DATA_DIR: Path to pre-downloaded MSWEP data (skips Rclone)

    Remote layout (GloH2O V3.16 docs): ``{VERSION}/{Past|Past_nogauge|NRT}/
    {Hourly|3hourly|Daily|Monthly}/`` with filenames ``YYYYDOY.HH.nc``
    (3-hourly) / ``YYYYDOY.nc`` (daily) / ``YYYYMM.nc`` (monthly); there
    are NO per-year subfolders. Worked example from the docs:
    ``MSWEP_V315/Past/Hourly/2020116.18.nc``.
    """

    DEFAULT_VERSION = 'V316'

    GDRIVE_FOLDERS = {
        'V280': 'MSWEP_V280',
        'V300': 'MSWEP_V300',
        'V315': 'MSWEP_V315',
        'V316': 'MSWEP_V316',
    }

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Starting MSWEP precipitation data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        resolution = self._get_config_value(
            lambda: None, default='daily', dict_key='MSWEP_RESOLUTION')
        product = self._get_config_value(
            lambda: None, default='Past', dict_key='MSWEP_PRODUCT')
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        file_list = self._generate_file_list(resolution, product)
        self.logger.info(
            f"MSWEP {resolution} data: {len(file_list)} files, "
            f"{self.start_date.date()} to {self.end_date.date()}"
        )

        # Check for pre-downloaded local data
        local_dir = self._get_config_value(
            lambda: None, default=None, dict_key='MSWEP_DATA_DIR')
        if local_dir:
            linked = self._link_local(
                Path(local_dir), output_dir, file_list)
            if linked:
                self.logger.info(
                    f"Linked {len(linked)} files from local MSWEP directory")
                return output_dir

        # Download via Rclone
        downloaded = self._download_rclone(
            output_dir, file_list, resolution, product, force_download)

        if not downloaded:
            raise RuntimeError(
                "Failed to download MSWEP data.\n"
                "\n"
                "MSWEP is distributed via Google Drive (Rclone). Setup:\n"
                "  1. Request access at https://www.gloh2o.org/mswep/\n"
                "  2. Install rclone: https://rclone.org/downloads/\n"
                "  3. Run: rclone config  (create a Google Drive remote)\n"
                "  4. Verify: rclone lsd --drive-shared-with-me GoogleDrive:\n"
                "\n"
                "Or set MSWEP_DATA_DIR to a local directory with MSWEP files."
            )

        self.logger.info(f"MSWEP download complete: {len(downloaded)} files")
        return output_dir

    # =========================================================================
    # File list generation
    # =========================================================================

    def _generate_file_list(self, resolution: str, product: str) -> List[dict]:
        files = []
        # NOTE: remote filenames are YYYYDOY-prefixed and live directly under
        # the resolution folder — there are NO per-year subfolders (GloH2O
        # V3.16 docs, worked example: MSWEP_V315/Past/Hourly/2020116.18.nc).
        if resolution == '3hourly':
            current = self.start_date
            while current <= self.end_date:
                year = current.year
                doy = current.timetuple().tm_yday
                for hour in range(0, 24, 3):
                    files.append({
                        'relative_path': f"{product}/3hourly/{year}{doy:03d}.{hour:02d}.nc",
                        'local_name': f"mswep_{year}{doy:03d}{hour:02d}.nc",
                    })
                current += pd.Timedelta(days=1)

        elif resolution == 'daily':
            current = self.start_date
            while current <= self.end_date:
                year = current.year
                doy = current.timetuple().tm_yday
                files.append({
                    'relative_path': f"{product}/Daily/{year}{doy:03d}.nc",
                    'local_name': f"mswep_{year}{doy:03d}.nc",
                })
                current += pd.Timedelta(days=1)

        elif resolution == 'monthly':
            current = datetime(self.start_date.year, self.start_date.month, 1)
            while current <= self.end_date:
                year = current.year
                month = current.month
                files.append({
                    'relative_path': f"{product}/Monthly/{year}{month:02d}.nc",
                    'local_name': f"mswep_{year}{month:02d}.nc",
                })
                if month == 12:
                    current = datetime(year + 1, 1, 1)
                else:
                    current = datetime(year, month + 1, 1)

        return files

    # =========================================================================
    # Local directory
    # =========================================================================

    def _link_local(
        self, local_dir: Path, output_dir: Path, file_list: List[dict],
    ) -> List[Path]:
        """Symlink files from a pre-downloaded MSWEP directory."""
        if not local_dir.is_dir():
            self.logger.warning(f"MSWEP_DATA_DIR not found: {local_dir}")
            return []

        linked = []
        for file_info in file_list:
            src = local_dir / file_info['relative_path']
            dst = output_dir / file_info['local_name']

            if dst.exists():
                linked.append(dst)
                continue

            if src.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                linked.append(dst)

        return linked

    # =========================================================================
    # Rclone + Google Drive
    # =========================================================================

    def _get_rclone_remote(self) -> str:
        return self._get_config_value(
            lambda: None, default='GoogleDrive', dict_key='MSWEP_RCLONE_REMOTE')

    def _get_gdrive_folder(self) -> str:
        version = self._get_config_value(
            lambda: None, default=self.DEFAULT_VERSION, dict_key='MSWEP_VERSION')
        return self.GDRIVE_FOLDERS.get(version, f'MSWEP_{version}')

    def _download_rclone(
        self, output_dir: Path, file_list: List[dict],
        resolution: str, product: str, force: bool,
    ) -> List[Path]:
        """Download files via Rclone from Google Drive using batch sync."""
        rclone = shutil.which('rclone')
        if not rclone:
            self.logger.warning(
                "rclone not found in PATH. "
                "Install from https://rclone.org/downloads/")
            return []

        remote = self._get_rclone_remote()
        folder = self._get_gdrive_folder()

        if not self._verify_rclone_access(rclone, remote, folder):
            return []

        # Group files by remote directory for batch downloads
        batches = self._group_by_directory(file_list)
        downloaded: list[Path] = []

        for remote_subdir, batch_files in batches.items():
            needed = self._filter_needed(output_dir, batch_files, force)
            if not needed:
                downloaded.extend(
                    output_dir / f['local_name'] for f in batch_files
                    if (output_dir / f['local_name']).exists()
                )
                continue

            remote_path = f"{remote}:{folder}/{remote_subdir}"
            filenames = {Path(f['relative_path']).name for f in needed}

            batch_result = self._rclone_copy_batch(
                rclone, remote_path, output_dir, remote_subdir,
                filenames, batch_files)
            downloaded.extend(batch_result)

        return downloaded

    def _verify_rclone_access(
        self, rclone: str, remote: str, folder: str
    ) -> bool:
        """Check rclone remote is configured and MSWEP folder is accessible."""
        try:
            result = subprocess.run(
                [rclone, 'listremotes'],
                capture_output=True, text=True, timeout=15
            )
            if f"{remote}:" not in result.stdout:
                available = result.stdout.strip() or "(none)"
                self.logger.warning(
                    f"Rclone remote '{remote}' not configured. "
                    f"Available remotes: {available}")
                return False

            result = subprocess.run(
                [rclone, 'lsd', '--drive-shared-with-me',
                 f'{remote}:{folder}'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                self.logger.warning(
                    f"Cannot list MSWEP folder '{folder}' on Google Drive. "
                    f"Have you requested access at "
                    f"https://www.gloh2o.org/mswep/ ?\n"
                    f"rclone: {result.stderr.strip()}")
                return False

            return True
        except subprocess.TimeoutExpired:
            self.logger.warning("rclone timed out checking remote access")
            return False
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"rclone check failed: {e}")
            return False

    def _group_by_directory(
        self, file_list: List[dict]
    ) -> dict[str, List[dict]]:
        """Group files by their remote parent directory for batch copy."""
        batches: dict[str, List[dict]] = {}
        for f in file_list:
            parent = str(Path(f['relative_path']).parent)
            batches.setdefault(parent, []).append(f)
        return batches

    def _filter_needed(
        self, output_dir: Path, batch_files: List[dict], force: bool
    ) -> List[dict]:
        """Return only files that need downloading."""
        if force:
            return batch_files
        return [
            f for f in batch_files
            if not (output_dir / f['local_name']).exists()
        ]

    def _rclone_copy_batch(
        self, rclone: str, remote_path: str,
        output_dir: Path, remote_subdir: str,
        filenames: Set[str], batch_files: List[dict],
    ) -> List[Path]:
        """Copy a directory from the remote, filtering to needed files only."""
        # Build include filters for just the files we need
        filter_args = []
        for name in filenames:
            filter_args.extend(['--include', name])
        filter_args.extend(['--exclude', '*'])

        # rclone copy downloads into a mirrored directory structure;
        # use a temp subdir then move files to output with local names
        staging_dir = output_dir / '.mswep_staging' / remote_subdir
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                rclone, 'copy', '--drive-shared-with-me',
                *filter_args,
                remote_path, str(staging_dir),
            ]
            self.logger.info(
                f"Downloading {len(filenames)} files from {remote_subdir}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"rclone copy failed for {remote_subdir}: "
                    f"{result.stderr.strip()}")
                return []

        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"rclone timed out downloading {remote_subdir}")
            return []
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"rclone failed for {remote_subdir}: {e}")
            return []

        # Move staged files to output with local names
        downloaded = []
        for f in batch_files:
            src_name = Path(f['relative_path']).name
            staged = staging_dir / src_name
            dst = output_dir / f['local_name']

            if staged.exists():
                shutil.move(str(staged), str(dst))
                downloaded.append(dst)
            elif dst.exists():
                downloaded.append(dst)

        # Clean up staging
        staging_root = output_dir / '.mswep_staging'
        shutil.rmtree(staging_root, ignore_errors=True)

        return downloaded
