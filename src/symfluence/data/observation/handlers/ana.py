# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ANA (Agencia Nacional de Aguas) Observation Handler

Provides cloud-accessible streamflow data for 897 Brazilian catchments via the
CAMELS-BR dataset hosted on Zenodo. Data is downloaded automatically on
first use and cached locally.

Data source:
    Chagas, V.B.P., et al. (2020). CAMELS-BR: hydrometeorological time series
    and landscape attributes for 897 catchments in Brazil. Earth System
    Science Data, 12(3), 2075-2098.
    https://doi.org/10.5194/essd-12-2075-2020

    Zenodo: https://doi.org/10.5281/zenodo.3709337

Station codes:
    ANA gauge codes (8-digit, e.g., '15930000').

Configuration:
    STATION_ID: ANA station code (e.g., '15930000')
    STREAMFLOW_DATA_PROVIDER: 'ANA'
    DOWNLOAD_ANA_DATA: True (default, downloads from Zenodo if not cached)
"""

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError, symfluence_error_handler

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

ZENODO_STREAMFLOW_URL = (
    "https://zenodo.org/api/records/15025488/files/"
    "03_CAMELS_BR_streamflow_m3s.zip/content"
)

ANA_STATIONS = {
    "15930000": "Rio Aripuana (Amazon, 98% forest)",
    "17345000": "Rio Teles Pires (Xingu headwaters, 88% forest)",
    "12400000": "Rio Jurua (Western Amazon, 100% forest)",
    "14350000": "Rio Negro (Central Amazon, 95% forest)",
    "10100000": "Rio Solimoes em Tabatinga",
}


@ObservationRegistry.register("ana_streamflow")
class ANAStreamflowHandler(BaseObservationHandler):
    """Handles Brazilian ANA streamflow data via CAMELS-BR (Zenodo).

    Downloads the CAMELS-BR streamflow dataset from Zenodo on first use,
    caches it locally, and extracts the requested station's time series.
    Covers 897 ANA stations with daily data (period varies by station).

    Usage in config::

        station_id: '15930000'
        streamflow_data_provider: 'ANA'
        download_ana_data: true
    """

    obs_type = "streamflow"
    source_name = "ANA"
    SOURCE_INFO = {
        "source": "ANA (via CAMELS-BR / Zenodo)",
        "source_doi": "10.5281/zenodo.3709337",
        "url": "https://doi.org/10.5281/zenodo.3709337",
        "citation": (
            "Chagas, V.B.P., et al. (2020). CAMELS-BR: hydrometeorological "
            "time series and landscape attributes for 897 catchments in "
            "Brazil. ESSD, 12(3), 2075-2098."
        ),
    }

    def acquire(self) -> Path:
        """Download ANA streamflow data from Zenodo and extract station."""
        station_id = self._get_station_id()
        if not station_id:
            self.logger.debug("STATION_ID not found, skipping ANA acquisition")
            return self.project_observations_dir / "streamflow" / "raw_data"

        station_id = str(station_id).strip()

        raw_dir = self.project_observations_dir / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"ana_{station_id}_raw.csv"

        if raw_file.exists():
            self.logger.info(f"ANA data already available: {raw_file}")
            return raw_file

        download_enabled = self._get_config_value(
            lambda: self.config.data.download_ana_data, default=True
        )
        if not download_enabled:
            if raw_file.exists():
                return raw_file
            raise DataAcquisitionError(
                f"ANA data not found and download disabled: {raw_file}"
            )

        cache_dir = self._get_cache_dir()
        cached_zip = cache_dir / "03_CAMELS_BR_streamflow_m3s.zip"

        if not cached_zip.exists():
            self._download_zenodo(cached_zip)

        self._extract_station(cached_zip, station_id, raw_file)
        return raw_file

    def process(self, input_path: Path) -> Path:
        """Process raw ANA CSV into standard SYMFLUENCE streamflow format."""
        if not input_path.exists():
            raise FileNotFoundError(f"ANA raw data not found: {input_path}")

        self.logger.info(f"Processing ANA streamflow from {input_path}")

        df = pd.read_csv(input_path, parse_dates=["datetime"])
        df.set_index("datetime", inplace=True)
        df["discharge_cms"] = pd.to_numeric(df["discharge_cms"], errors="coerce")
        df = df.dropna(subset=["discharge_cms"])

        df = df.loc[self.start_date : self.end_date]

        if df.empty:
            raise DataAcquisitionError(
                f"No ANA data in experiment period "
                f"({self.start_date} to {self.end_date})."
            )

        resample_freq = self._get_resample_freq()
        resampled = df["discharge_cms"].resample(resample_freq).mean()
        resampled = resampled.interpolate(
            method="time", limit_direction="both", limit=30
        )

        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        resampled.to_csv(output_file, header=True, index_label="datetime")

        self.logger.info(
            f"ANA streamflow processing complete: {output_file} "
            f"({len(resampled)} records)"
        )
        return output_file

    def _get_station_id(self) -> Optional[str]:
        """Retrieve ANA station ID from config."""
        return self._get_config_value(
            lambda: self.config.evaluation.streamflow.station_id
        ) or self._get_config_value(
            lambda: self.config.data.streamflow_station_id
        )

    def _get_cache_dir(self) -> Path:
        """Get or create cache directory for CAMELS-BR bulk download."""
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir, default=None
        )
        if data_dir:
            cache = Path(data_dir) / ".cache" / "camels_br"
        else:
            cache = self.project_observations_dir / ".cache" / "camels_br"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _download_zenodo(self, output_path: Path) -> None:
        """Download CAMELS-BR streamflow ZIP from Zenodo."""
        self.logger.info(
            "Downloading CAMELS-BR streamflow data from Zenodo..."
        )

        with symfluence_error_handler(
            "Zenodo download", self.logger, error_type=DataAcquisitionError
        ):
            response = requests.get(
                ZENODO_STREAMFLOW_URL, timeout=300, stream=True
            )
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = output_path.stat().st_size / 1e6
            self.logger.info(
                f"Downloaded CAMELS-BR streamflow: "
                f"{output_path} ({size_mb:.1f} MB)"
            )

    def _extract_station(
        self, zip_path: Path, station_id: str, output_path: Path
    ) -> None:
        """Extract a single station from the CAMELS-BR ZIP.

        CAMELS-BR stores each station as a separate file:
        3_CAMELS_BR_streamflow_m3s/{station_id}_streamflow_m3s.txt
        """
        self.logger.info(
            f"Extracting station {station_id} from CAMELS-BR archive..."
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            candidates = [
                f"3_CAMELS_BR_streamflow_m3s/{station_id}_streamflow_m3s.txt",
                f"{station_id}_streamflow_m3s.txt",
            ]

            target = None
            for c in candidates:
                if c in zf.namelist():
                    target = c
                    break

            if target is None:
                matching = [n for n in zf.namelist() if station_id in n]
                if matching:
                    target = matching[0]

            if target is None:
                available = [n for n in zf.namelist() if n.endswith(".txt")][:10]
                raise DataAcquisitionError(
                    f"Station {station_id} not found in CAMELS-BR archive. "
                    f"Sample files: {available}"
                )

            with zf.open(target) as f:
                df = None
                for sep in [r"\s+", ",", ";", "\t"]:
                    try:
                        f.seek(0)
                        df = pd.read_csv(f, sep=sep)
                        if len(df.columns) >= 2:  # noqa: PLR2004
                            break
                    except Exception:  # noqa: BLE001
                        continue

        if df is None or len(df.columns) < 2:  # noqa: PLR2004
            raise DataAcquisitionError(
                f"Could not parse streamflow file for station {station_id}"
            )

        date_cols = [c for c in df.columns if "date" in c.lower()]
        date_col = date_cols[0] if date_cols else df.columns[0]
        val_cols = [
            c for c in df.columns
            if any(k in c.lower() for k in ["streamflow", "discharge", "m3"])
        ]
        val_col = val_cols[0] if val_cols else df.columns[1]

        result = pd.DataFrame({
            "datetime": pd.to_datetime(df[date_col]),
            "discharge_cms": pd.to_numeric(df[val_col], errors="coerce"),
        })
        result = result.dropna(subset=["discharge_cms"])
        result = result[result["discharge_cms"] >= 0]
        result.to_csv(output_path, index=False)

        station_name = ANA_STATIONS.get(station_id, "unknown")
        self.logger.info(
            f"Extracted ANA station {station_id} ({station_name}): "
            f"{len(result)} records, "
            f"{result['datetime'].min()} to {result['datetime'].max()}"
        )
