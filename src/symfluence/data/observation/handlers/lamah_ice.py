# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LamaH-ICE Observation Handlers

Provides handlers for LamaH-ICE (Iceland) streamflow data, with
optional auto-download from HydroShare when the local dataset is
missing.
"""
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.data.acquisition.utils import create_robust_session

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

# HydroShare record for LamaH-Ice (Helgason & Nijssen, 2024, ESSD).
# DOI: 10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91
_LAMAH_ICE_HS_RESOURCE_ID = "86117a5f36cc4b7c90a5d54e18161c91"
_LAMAH_ICE_DAILY_ZIP_NAME = "lamah_ice.zip"  # ~636 MB on HydroShare
_LAMAH_ICE_DAILY_ZIP_URL = (
    f"https://www.hydroshare.org/resource/{_LAMAH_ICE_HS_RESOURCE_ID}"
    f"/data/contents/{_LAMAH_ICE_DAILY_ZIP_NAME}"
)
_LAMAH_ICE_REQUIRED_SUBPATH = Path("D_gauges") / "2_timeseries" / "daily"


def ensure_lamah_ice_streamflow(lamah_path: Path, logger: logging.Logger, *,
                                 force: bool = False) -> Path:
    """Ensure ``lamah_path/D_gauges/`` is populated, downloading from
    HydroShare if needed.

    Args:
        lamah_path: Target root for the dataset
            (``LAMAH_ICE_PATH`` in configs). Created if missing.
        logger: Logger for progress / warning messages.
        force: Re-download even when the target subtree is present.

    Returns:
        The resolved ``lamah_path``.

    Raises:
        DataAcquisitionError: When the HydroShare fetch or extraction fails.
    """
    lamah_path = Path(lamah_path).expanduser().resolve()
    daily_dir = lamah_path / _LAMAH_ICE_REQUIRED_SUBPATH
    if not force and daily_dir.exists() and any(daily_dir.glob("ID_*.csv")):
        return lamah_path

    lamah_path.mkdir(parents=True, exist_ok=True)
    cache_dir = lamah_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / _LAMAH_ICE_DAILY_ZIP_NAME

    if force or not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        _download_lamah_ice_zip(zip_path, logger)

    _extract_d_gauges(zip_path, lamah_path, logger)

    # Drop the cached zip to save ~636 MB once we've extracted D_gauges.
    try:
        zip_path.unlink()
        cache_dir.rmdir()
    except OSError:
        pass

    if not any(daily_dir.glob("ID_*.csv")):
        raise DataAcquisitionError(
            f"LaMAH-Ice download finished but {daily_dir} contains no "
            "ID_*.csv files — the HydroShare archive layout may have "
            "changed. Report to SYMFLUENCE maintainers."
        )
    logger.info(f"LaMAH-Ice daily streamflow ready at {daily_dir}")
    return lamah_path


def _download_lamah_ice_zip(zip_path: Path, logger: logging.Logger) -> None:
    """Stream the 636 MB daily-zip from HydroShare into ``zip_path``."""
    tmp = zip_path.with_suffix(".zip.part")
    logger.info(
        f"Downloading LaMAH-Ice daily streamflow (~636 MB) from HydroShare: "
        f"{_LAMAH_ICE_DAILY_ZIP_URL}"
    )
    session = create_robust_session(max_retries=3, backoff_factor=2.0)
    try:
        with session.get(_LAMAH_ICE_DAILY_ZIP_URL, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_pct = -1
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1_048_576):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct and pct % 10 == 0:
                            logger.info(
                                f"LaMAH-Ice download progress: {pct}% ({downloaded/1e6:.0f} MB)"
                            )
                            last_pct = pct
        tmp.replace(zip_path)
    except Exception as e:  # noqa: BLE001 — wrap transport errors
        tmp.unlink(missing_ok=True)
        raise DataAcquisitionError(
            f"Failed to download LaMAH-Ice from HydroShare: {e}. "
            f"Manual fallback: visit https://doi.org/10.4211/hs.{_LAMAH_ICE_HS_RESOURCE_ID} "
            f"and extract {_LAMAH_ICE_DAILY_ZIP_NAME} into LAMAH_ICE_PATH."
        ) from e


def _extract_d_gauges(zip_path: Path, lamah_path: Path,
                      logger: logging.Logger) -> None:
    """Extract only ``D_gauges/*`` from the HydroShare zip (~57 MB of
    2 GB decompressed). The rest of the archive is polygon / stream
    data that streamflow calibration doesn't need."""
    logger.info("Extracting D_gauges from LaMAH-Ice archive...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members: Iterable[zipfile.ZipInfo] = [
                m for m in zf.infolist() if "D_gauges/" in m.filename
            ]
            if not members:
                raise DataAcquisitionError(
                    f"D_gauges not found inside {zip_path.name}; "
                    "HydroShare archive layout may have changed."
                )
            for m in members:
                # Strip any leading ``lamah_ice/`` prefix so the layout
                # under lamah_path matches what LAMAH_ICE_PATH expects.
                rel = m.filename.split("D_gauges/", 1)[1]
                target = lamah_path / "D_gauges" / rel if rel else lamah_path / "D_gauges"
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as e:
        raise DataAcquisitionError(
            f"LaMAH-Ice archive at {zip_path} is corrupt: {e}. "
            "Re-run with force=True to redownload."
        ) from e


@ObservationRegistry.register('lamah_ice_streamflow')
class LamahIceStreamflowHandler(BaseObservationHandler):
    """Handles LamaH-ICE streamflow data processing, with auto-download
    from HydroShare when the local dataset is missing."""

    obs_type = "streamflow"
    source_name = "LAMAH_ICE"
    SOURCE_INFO = {
        'source': 'LamaH-Ice',
        'source_doi': '10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91',
        'url': 'https://doi.org/10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91',
        'citation': (
            'Helgason, H. B. and Nijssen, B.: LamaH-Ice: LArge-SaMple DAta '
            'for Hydrology and Environmental Sciences for Iceland, Earth '
            'System Science Data, 16, 2741–2771, 2024. '
            'doi:10.5194/essd-16-2741-2024'
        ),
    }

    def acquire(self) -> Path:
        """
        Locates the raw LamaH-ICE file for the given station ID.

        Accepted config keys for the basin identifier (checked in order):
            LAMAH_ICE_DOMAIN_ID: 105   # preferred — matches LaMAH-ICE's
                                       # own D_gauges/.../ID_<n>.csv naming
                                       # and is what the 08_large_sample
                                       # paper configs (117 files) all use.
            STATION_ID: 105            # legacy alias kept for generic
                                       # cross-dataset configs.

        LAMAH_ICE_PATH points at the local extracted dataset:
            LAMAH_ICE_PATH: /path/to/lamah_ice
        """
        # LAMAH_ICE_DOMAIN_ID takes precedence. The 08_large_sample
        # paper configs write this key; the handler previously only
        # recognised STATION_ID, so every run silently failed at
        # acquire with the misleading error "STATION_ID required for
        # LAMAH_ICE acquisition".
        station_id = (
            self._get_config_value(
                lambda: self.config.evaluation.lamah_ice.domain_id,
                dict_key='LAMAH_ICE_DOMAIN_ID',
            )
            or self._get_config_value(
                lambda: self.config.evaluation.streamflow.station_id,
                dict_key='STATION_ID',
            )
        )
        lamah_path_str = self._get_config_value(lambda: self.config.data.lamah_ice_path, dict_key='LAMAH_ICE_PATH')

        if not station_id:
            raise ValueError(
                "LAMAH_ICE acquisition requires a basin identifier. Set "
                "LAMAH_ICE_DOMAIN_ID (preferred — matches LaMAH-ICE's "
                "own ID_<n>.csv naming) or STATION_ID in your config."
            )
        if not lamah_path_str:
            # Default to a SYMFLUENCE-managed cache so users don't need to
            # discover + manually download the dataset.
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'
            )
            if not data_dir:
                raise ValueError(
                    "LAMAH_ICE_PATH not set and SYMFLUENCE_DATA_DIR is not "
                    "configured — nowhere to download LaMAH-Ice to."
                )
            lamah_path = Path(data_dir) / "lamah_ice"
        else:
            lamah_path = Path(lamah_path_str)

        raw_file = lamah_path / "D_gauges" / "2_timeseries" / "daily" / f"ID_{station_id}.csv"
        if not raw_file.exists():
            ensure_lamah_ice_streamflow(lamah_path, self.logger)

        if not raw_file.exists():
            raise FileNotFoundError(
                f"LamaH-ICE file not found at {raw_file} after auto-download "
                f"attempt. Check LAMAH_ICE_DOMAIN_ID={station_id} is a valid "
                "LaMAH-Ice basin (1–111 per the published record)."
            )

        # Copy or link to project directory for processing consistency
        dest_dir = self.project_observations_dir / "streamflow" / "raw_data"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"lamah_ice_{station_id}_raw.csv"

        import shutil
        shutil.copy2(raw_file, dest_file)

        self.logger.info(f"Located and copied LamaH-ICE data to {dest_file}")
        return dest_file

    def process(self, input_path: Path) -> Path:
        """
        Process LamaH-ICE data into standard SYMFLUENCE format.
        LamaH-ICE format: YYYY;MM;DD;qobs;qc_flag
        """
        self.logger.info(f"Processing LamaH-ICE streamflow data from {input_path}")

        # Read semicolon-separated file
        df = pd.read_csv(input_path, sep=';')

        if not all(col in df.columns for col in ['YYYY', 'MM', 'DD', 'qobs']):
            raise DataAcquisitionError(f"Unexpected columns in LamaH-ICE file: {df.columns}")

        # Create datetime index
        df['datetime'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].rename(
            columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Discharge is in m3/s (qobs)
        df['discharge_cms'] = pd.to_numeric(df['qobs'], errors='coerce')

        # Filter by quality if requested (40.0 is usually 'original' or 'good')
        # We'll keep all for now but log if many are missing
        df = df.dropna(subset=['discharge_cms'])

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=30)

        # Save processed data
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"

        resampled.to_csv(output_file, header=True, index_label='datetime')

        self.logger.info(f"LamaH-ICE streamflow processing complete: {output_file}")
        return output_file
