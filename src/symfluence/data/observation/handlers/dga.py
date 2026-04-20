# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
DGA (Direccion General de Aguas) Observation Handler

Provides cloud-accessible streamflow data for 516 Chilean catchments via the
CAMELS-CL dataset hosted on PANGAEA. Data is downloaded automatically on
first use and cached locally.

Data source:
    Alvarez-Garreton, C., et al. (2018). The CAMELS-CL dataset: catchment
    attributes and meteorology for large sample studies - Chile dataset.
    Hydrology and Earth System Sciences, 22(11), 5817-5846.
    https://doi.org/10.5194/hess-22-5817-2018

    PANGAEA: https://doi.pangaea.de/10.1594/PANGAEA.894885

Station codes:
    DGA BNA codes (7-digit, e.g., '5710001' for Rio Maipo en El Manzano).
    The check digit suffix (e.g., '-6') is stripped automatically if provided.

Configuration:
    STATION_ID: DGA station code (e.g., '5710001')
    STREAMFLOW_DATA_PROVIDER: 'DGA'
    DOWNLOAD_DGA_DATA: True (default, downloads from PANGAEA if not cached)
"""

import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError, symfluence_error_handler

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

# PANGAEA direct download URL for CAMELS-CL streamflow (m3/s)
PANGAEA_STREAMFLOW_URL = (
    "https://store.pangaea.de/Publications/"
    "Alvarez-Garreton-etal_2018/2_CAMELScl_streamflow_m3s.zip"
)

# Well-known DGA stations for quick reference
DGA_STATIONS = {
    '5710001': 'Rio Maipo en El Manzano',
    '5716001': 'Rio Mapocho en Los Almendros',
    '4513001': 'Rio Aconcagua en Chacabuquito',
    '8317001': 'Rio Biobio en Rucalhue',
    '7104002': 'Rio Maule en Armerillo',
    '3410001': 'Rio Elqui en Algarrobal',
    '10356001': 'Rio Valdivia en Valdivia',
    '5707002': 'Rio Colorado antes junta Rio Maipo',
    '4530002': 'Rio Aconcagua en San Felipe',
    '8313001': 'Rio Biobio en Desembocadura',
}


def _normalize_station_id(station_id: str) -> str:
    """Strip check digit suffix (e.g., '5710001-6' -> '5710001')."""
    return str(station_id).split('-')[0].strip()


@ObservationRegistry.register('dga_streamflow')
class DGAStreamflowHandler(BaseObservationHandler):
    """Handles Chilean DGA streamflow data via CAMELS-CL (PANGAEA).

    Downloads the CAMELS-CL streamflow dataset from PANGAEA on first use,
    caches it locally, and extracts the requested station's time series.
    Covers 516 DGA stations with daily data from 1913 to 2018.

    Usage in config::

        station_id: '5710001'
        streamflow_data_provider: 'DGA'
        download_dga_data: true
    """

    obs_type = "streamflow"
    source_name = "DGA"
    SOURCE_INFO = {
        'source': 'DGA (via CAMELS-CL / PANGAEA)',
        'source_doi': '10.1594/PANGAEA.894885',
        'url': 'https://doi.pangaea.de/10.1594/PANGAEA.894885',
        'citation': (
            'Alvarez-Garreton, C., et al. (2018). The CAMELS-CL dataset. '
            'HESS, 22(11), 5817-5846.'
        ),
    }

    def acquire(self) -> Path:
        """Download DGA streamflow data from PANGAEA and extract station."""
        station_id = self._get_station_id()
        if not station_id:
            self.logger.debug("STATION_ID not found, skipping DGA acquisition")
            return self.project_observations_dir / "streamflow" / "raw_data"

        station_id = _normalize_station_id(station_id)

        raw_dir = self.project_observations_dir / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"dga_{station_id}_raw.csv"

        # Check if already extracted
        if raw_file.exists():
            self.logger.info(f"DGA data already available: {raw_file}")
            return raw_file

        download_enabled = self._get_config_value(
            lambda: self.config.data.download_dga_data, default=True
        )
        if not download_enabled:
            if raw_file.exists():
                return raw_file
            raise DataAcquisitionError(
                f"DGA data not found and download disabled: {raw_file}"
            )

        # Download and extract from PANGAEA
        cache_dir = self._get_cache_dir()
        cached_zip = cache_dir / "2_CAMELScl_streamflow_m3s.zip"

        if not cached_zip.exists():
            self._download_pangaea(cached_zip)

        self._extract_station(cached_zip, station_id, raw_file)
        return raw_file

    def process(self, input_path: Path) -> Path:
        """Process raw DGA CSV into standard SYMFLUENCE streamflow format."""
        if not input_path.exists():
            raise FileNotFoundError(f"DGA raw data not found: {input_path}")

        self.logger.info(f"Processing DGA streamflow from {input_path}")

        df = pd.read_csv(input_path, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        df['discharge_cms'] = pd.to_numeric(df['discharge_cms'], errors='coerce')
        df = df.dropna(subset=['discharge_cms'])

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        if df.empty:
            raise DataAcquisitionError(
                f"No DGA data in experiment period "
                f"({self.start_date} to {self.end_date}). "
                f"CAMELS-CL covers 1913-2018."
            )

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled = resampled.interpolate(
            method='time', limit_direction='both', limit=30
        )

        # Save
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        resampled.to_csv(output_file, header=True, index_label='datetime')

        self.logger.info(
            f"DGA streamflow processing complete: {output_file} "
            f"({len(resampled)} records)"
        )
        return output_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_station_id(self) -> Optional[str]:
        """Retrieve DGA station ID from config."""
        return (
            self._get_config_value(
                lambda: self.config.evaluation.streamflow.station_id
            )
            or self._get_config_value(
                lambda: self.config.data.streamflow_station_id
            )
        )

    def _get_cache_dir(self) -> Path:
        """Get or create cache directory for CAMELS-CL bulk download."""
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir, default=None
        )
        if data_dir:
            cache = Path(data_dir) / '.cache' / 'camels_cl'
        else:
            cache = self.project_observations_dir / '.cache' / 'camels_cl'
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _download_pangaea(self, output_path: Path) -> None:
        """Download CAMELS-CL streamflow ZIP from PANGAEA."""
        self.logger.info(
            "Downloading CAMELS-CL streamflow data from PANGAEA "
            "(~13 MB)..."
        )

        with symfluence_error_handler(
            "PANGAEA download", self.logger, error_type=DataAcquisitionError
        ):
            response = requests.get(
                PANGAEA_STREAMFLOW_URL, timeout=120, stream=True
            )
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = output_path.stat().st_size / 1e6
            self.logger.info(
                f"Downloaded CAMELS-CL streamflow: "
                f"{output_path} ({size_mb:.1f} MB)"
            )

    def _extract_station(
        self, zip_path: Path, station_id: str, output_path: Path
    ) -> None:
        """Extract a single station's time series from the CAMELS-CL ZIP.

        The ZIP contains a single large text file with all 516 stations as
        tab-separated columns. We extract just the requested station and
        save as a simple datetime,discharge_cms CSV.
        """
        self.logger.info(
            f"Extracting station {station_id} from CAMELS-CL archive..."
        )

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find the streamflow data file inside the ZIP
            data_files = [
                n for n in zf.namelist()
                if 'streamflow' in n.lower() and n.endswith('.txt')
            ]
            if not data_files:
                raise DataAcquisitionError(
                    f"No streamflow data file found in {zip_path}"
                )

            with zf.open(data_files[0]) as f:
                # Read header to find station column
                content = io.TextIOWrapper(f, encoding='utf-8')
                header_line = content.readline().strip()
                columns = header_line.split('\t')

                # Find station column (match with/without leading zeros)
                station_col_idx = None
                for i, col in enumerate(columns):
                    col_clean = col.strip().strip('"')
                    if col_clean == station_id or col_clean == station_id.lstrip('0'):
                        station_col_idx = i
                        break

                if station_col_idx is None:
                    available = [
                        c.strip().strip('"') for c in columns[1:11]
                    ]
                    raise DataAcquisitionError(
                        f"Station {station_id} not found in CAMELS-CL. "
                        f"First 10 available: {available}. "
                        f"Total stations: {len(columns) - 1}"
                    )

                # Parse data rows - only keep date and target station
                dates = []
                values = []
                for line in content:
                    parts = line.strip().split('\t')
                    if len(parts) <= station_col_idx:
                        continue
                    date_str = parts[0].strip().strip('"')
                    val_str = parts[station_col_idx].strip().strip('"')

                    try:
                        date = pd.Timestamp(date_str)
                    except (ValueError, TypeError):
                        continue

                    if val_str == '' or val_str == ' ':
                        values.append(float('nan'))
                    else:
                        try:
                            values.append(float(val_str))
                        except ValueError:
                            values.append(float('nan'))
                    dates.append(date)

        # Save extracted station data
        df = pd.DataFrame({'datetime': dates, 'discharge_cms': values})
        df = df.dropna(subset=['discharge_cms'])
        df.to_csv(output_path, index=False)

        station_name = DGA_STATIONS.get(station_id, 'unknown')
        self.logger.info(
            f"Extracted DGA station {station_id} ({station_name}): "
            f"{len(df)} records, "
            f"{df['datetime'].min()} to {df['datetime'].max()}"
        )
