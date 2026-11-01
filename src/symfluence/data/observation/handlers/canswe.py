# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CanSWE (Canadian Snow Water Equivalent) Observation Handler

Provides cloud acquisition and processing for the CanSWE historical dataset,
which contains in-situ snow water equivalent observations from across Canada.

CanSWE Overview:
    Data Type: In-situ station observations
    Variables: SWE (mm), snow depth (cm), bulk density (kg/m³)
    Coverage: Canada-wide (2921 stations in v6)
    Temporal: 1928-2023 (v6)
    Resolution: Point measurements (daily to monthly depending on station)
    Source: Zenodo (https://zenodo.org/records/10835278)
    Reference: Vionnet et al. (2021), ESSD

Data Access:
    Primary: Zenodo direct download (no authentication required)
    Format: NetCDF (CanSWE-CanEEN_1928-2023_v6.nc)
    Alternative: CSV format also available

Variables in NetCDF:
    - swe: Snow Water Equivalent (mm)
    - sd: Snow Depth (cm)
    - rho: Bulk Snow Density (kg/m³)
    - lat: Station latitude
    - lon: Station longitude
    - station_id: Unique station identifier
    - time: Observation timestamp
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.registries import R

from ..base import BaseObservationHandler

# Zenodo record IDs for CanSWE versions
CANSWE_ZENODO_RECORDS = {
    'v6': '10835278',  # 1928-2023
    'v5': '8250403',   # 1928-2022
    'v4': '5889352',   # 1928-2021
}

# Default Zenodo record (latest version)
CANSWE_DEFAULT_VERSION = 'v6'


@R.observation_handlers.add('canswe')
@R.observation_handlers.add('canswe_swe')
class CanSWEHandler(BaseObservationHandler):
    """
    Handles CanSWE data acquisition and processing.

    CanSWE provides historical in-situ SWE observations from manual snow surveys,
    snow pillows, and other measurement methods across Canada. This handler
    downloads the dataset from Zenodo and extracts stations within the domain
    bounding box for the experiment time period.

    Configuration Options:
        DOWNLOAD_CANSWE (bool): Enable/disable download (default: True)
        CANSWE_VERSION (str): Dataset version ('v4', 'v5', 'v6', default: 'v6')
        CANSWE_MIN_OBSERVATIONS (int): Minimum observations per station (default: 10)
        CANSWE_PATH (str): Override path to existing CanSWE NetCDF file

    Example Config:
        DOWNLOAD_CANSWE: true
        CANSWE_VERSION: v6
        CANSWE_MIN_OBSERVATIONS: 20
    """

    obs_type = "swe"
    source_name = "CanSWE"

    # Zenodo API base URL
    ZENODO_API = "https://zenodo.org/api/records"

    def acquire(self) -> Path:
        """
        Acquire CanSWE data from Zenodo.

        Returns:
            Path to the downloaded/located CanSWE NetCDF file.

        Raises:
            DataAcquisitionError: If download fails or file cannot be located.
        """
        download_enabled = self._get_config_value(
            lambda: self.config.evaluation.canswe.download,
            default=True,
            dict_key='DOWNLOAD_CANSWE'
        )
        if isinstance(download_enabled, str):
            download_enabled = download_enabled.lower() == 'true'

        # Check for existing file path override
        canswe_path = self._get_config_value(
            lambda: self.config.evaluation.canswe.path,
            default=None,
            dict_key='CANSWE_PATH'
        )
        if canswe_path and canswe_path != 'default' and Path(canswe_path).exists():
            self.logger.debug(f"Using existing CanSWE file: {canswe_path}")
            return Path(canswe_path)

        # Set up output directory
        raw_dir = self.project_observations_dir / "snow" / "canswe" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Get version
        version = self._get_config_value(
            lambda: self.config.evaluation.canswe.version,
            default=CANSWE_DEFAULT_VERSION,
            dict_key='CANSWE_VERSION'
        )
        if version not in CANSWE_ZENODO_RECORDS:
            self.logger.warning(f"Unknown CanSWE version '{version}', using {CANSWE_DEFAULT_VERSION}")
            version = CANSWE_DEFAULT_VERSION

        record_id = CANSWE_ZENODO_RECORDS[version]
        raw_file = raw_dir / f"CanSWE_{version}.nc"

        if raw_file.exists() and not download_enabled:
            self.logger.debug(f"Using existing CanSWE file: {raw_file}")
            return raw_file

        if download_enabled:
            return self._download_from_zenodo(record_id, raw_file, version)
        else:
            if raw_file.exists():
                return raw_file
            raise DataAcquisitionError(
                f"CanSWE file not found at {raw_file} and download is disabled. "
                "Set DOWNLOAD_CANSWE: true or provide CANSWE_PATH."
            )

    def _download_from_zenodo(self, record_id: str, output_path: Path, version: str) -> Path:
        """
        Download CanSWE dataset from Zenodo.

        Args:
            record_id: Zenodo record ID
            output_path: Where to save the downloaded file
            version: Version string for logging

        Returns:
            Path to downloaded file
        """
        import time

        self.logger.info(f"Downloading CanSWE {version} from Zenodo (record {record_id})")

        # Direct download URLs for known versions (bypasses API which can be slow)
        DIRECT_URLS = {
            'v6': 'https://zenodo.org/records/10835278/files/CanSWE-CanEEN_1928-2023_v6.nc?download=1',
            'v5': 'https://zenodo.org/records/8250403/files/CanSWE-CanEEN_1928-2022_v5.nc?download=1',
            'v4': 'https://zenodo.org/records/5889352/files/CanSWE-CanEEN_1928-2021_v4.nc?download=1',
        }

        # Try direct URL first (faster and more reliable)
        nc_file_url = DIRECT_URLS.get(version)

        if not nc_file_url:
            # Fall back to API discovery
            try:
                record_url = f"{self.ZENODO_API}/{record_id}"
                response = requests.get(record_url, timeout=60)
                response.raise_for_status()
                record_data = response.json()

                for file_info in record_data.get('files', []):
                    filename = file_info.get('key', '')
                    if filename.endswith('.nc'):
                        nc_file_url = file_info.get('links', {}).get('self')
                        self.logger.debug(f"Found CanSWE NetCDF via API: {filename}")
                        break
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Zenodo API failed: {e}, using direct URL pattern")
                nc_file_url = f"https://zenodo.org/records/{record_id}/files/CanSWE-CanEEN_1928-2023_{version}.nc?download=1"

        # Download with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Downloading CanSWE (~100 MB) from: {nc_file_url}")
                headers = {'User-Agent': 'SYMFLUENCE/1.0'}
                response = requests.get(nc_file_url, headers=headers, stream=True, timeout=(60, 1800))
                response.raise_for_status()

                # Save with progress indication
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                tmp_file = output_path.with_suffix('.nc.part')

                with open(tmp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                            progress = (downloaded / total_size) * 100
                            self.logger.debug(f"Download progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")

                tmp_file.replace(output_path)
                self.logger.info(f"Successfully downloaded CanSWE to {output_path}")
                return output_path

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # 30s, 60s, 90s backoff
                    self.logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise DataAcquisitionError(f"Failed to download CanSWE from Zenodo after {max_retries} attempts: {e}") from e

        # This should be unreachable, but satisfies mypy's return analysis
        raise DataAcquisitionError("Download loop completed without success or error")

    def process(self, input_path: Path) -> Path:
        """
        Process CanSWE data: filter by bounding box and time period.

        This method:
        1. Loads the full CanSWE NetCDF
        2. Filters stations within the domain bounding box
        3. Subsets to the experiment time period
        4. Aggregates multi-station data if needed
        5. Outputs processed CSV in SYMFLUENCE standard format

        Args:
            input_path: Path to the raw CanSWE NetCDF file

        Returns:
            Path to processed CSV file

        Raises:
            FileNotFoundError: If input file doesn't exist
            DataAcquisitionError: If no stations found in domain
        """
        if not input_path.exists():
            raise FileNotFoundError(f"CanSWE raw data file not found: {input_path}")

        self.logger.info(f"Processing CanSWE data from {input_path}")

        # Load dataset
        ds = self._open_dataset(input_path)

        # Get bounding box
        lat_min = min(self.bbox['lat_min'], self.bbox['lat_max'])
        lat_max = max(self.bbox['lat_min'], self.bbox['lat_max'])
        lon_min = min(self.bbox['lon_min'], self.bbox['lon_max'])
        lon_max = max(self.bbox['lon_min'], self.bbox['lon_max'])

        self.logger.debug(f"Filtering stations within bbox: lat [{lat_min:.2f}, {lat_max:.2f}], "
                        f"lon [{lon_min:.2f}, {lon_max:.2f}]")

        # Find stations within bounding box
        stations_in_bbox = self._find_stations_in_bbox(ds, lat_min, lat_max, lon_min, lon_max)

        if not stations_in_bbox:
            raise DataAcquisitionError(
                f"No CanSWE stations found within bounding box: "
                f"lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]. "
                "Consider expanding the domain or using a different snow data source."
            )

        self.logger.info(f"Found {len(stations_in_bbox)} CanSWE stations in domain")

        # Extract and process data for these stations
        df = self._extract_station_data(ds, stations_in_bbox)

        # Filter to experiment time period
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        if df.empty:
            raise DataAcquisitionError(
                f"No CanSWE observations in time period "
                f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
            )

        # Get minimum observations threshold
        min_obs = self._get_config_value(
            lambda: self.config.evaluation.canswe.min_observations,
            default=10,
            dict_key='CANSWE_MIN_OBSERVATIONS'
        )

        # Filter out stations with too few observations
        if 'station_id' in df.columns:
            station_counts = df.groupby('station_id').size()
            valid_stations = station_counts[station_counts >= min_obs].index
            df = df[df['station_id'].isin(valid_stations)]
            self.logger.debug(f"Retained {len(valid_stations)} stations with >= {min_obs} observations")

        if df.empty:
            raise DataAcquisitionError(
                f"No CanSWE stations have >= {min_obs} observations in the time period"
            )

        # Save station metadata
        self._save_station_metadata(ds, stations_in_bbox)

        # Save processed data
        output_dir = self._get_observation_dir('snow', 'preprocessed')
        output_file = output_dir / f"{self.domain_name}_canswe_swe_processed.csv"

        # Pivot or aggregate if multiple stations
        df_agg = self._aggregate_stations(df)

        # Create metadata
        metadata = self._create_metadata(
            units='mm',
            temporal_resolution='daily',
            spatial_aggregation='station_mean',
            station_id=','.join(map(str, stations_in_bbox[:5])) + ('...' if len(stations_in_bbox) > 5 else ''),
            quality_flags={'n_stations': len(stations_in_bbox)}
        )

        # Save with metadata
        self._save_with_metadata(df_agg, metadata, 'snow')

        # Also save the full multi-station version
        full_output = output_dir / f"{self.domain_name}_canswe_swe_all_stations.csv"
        df.to_csv(full_output, index_label='datetime')
        self.logger.debug(f"Saved full station data to {full_output}")

        # Create a symlink/copy for generic snow lookup
        snow_processed = output_dir / f"{self.domain_name}_swe_processed.csv"
        if not snow_processed.exists():
            df_agg.to_csv(snow_processed, index_label='datetime')

        self.logger.info(f"CanSWE processing complete: {output_file}")
        return output_file

    def _find_stations_in_bbox(
        self,
        ds: xr.Dataset,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> List[int]:
        """
        Find station indices within the bounding box.

        Args:
            ds: CanSWE xarray Dataset
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds

        Returns:
            List of station indices within the bounding box
        """
        # CanSWE NetCDF structure: lat/lon are per-station coordinates
        # Try different variable names for coordinates
        lat_var = None
        lon_var = None

        for lat_name in ['lat', 'latitude', 'Latitude', 'LAT']:
            if lat_name in ds:
                lat_var = lat_name
                break

        for lon_name in ['lon', 'longitude', 'Longitude', 'LON']:
            if lon_name in ds:
                lon_var = lon_name
                break

        if lat_var is None or lon_var is None:
            self.logger.warning("Could not find lat/lon variables in CanSWE file")
            # Try to find coordinates in data variables
            for var in ds.data_vars:
                if 'lat' in var.lower():
                    lat_var = var
                if 'lon' in var.lower():
                    lon_var = var

        if lat_var is None or lon_var is None:
            raise DataAcquisitionError("Cannot find latitude/longitude variables in CanSWE dataset")

        lats = ds[lat_var].values
        lons = ds[lon_var].values

        # Handle different array shapes
        if lats.ndim > 1:
            # If 2D (time x station), take first non-nan per station
            if 'station' in ds.dims:
                lats = np.nanmean(lats, axis=0) if lats.shape[0] != len(ds['station']) else lats
                lons = np.nanmean(lons, axis=0) if lons.shape[0] != len(ds['station']) else lons
            else:
                lats = lats.flatten()
                lons = lons.flatten()

        # Find stations in bbox
        in_bbox = (
            (lats >= lat_min) & (lats <= lat_max) &
            (lons >= lon_min) & (lons <= lon_max)
        )

        station_indices = np.where(in_bbox)[0].tolist()

        # Log station details
        for idx in station_indices[:5]:  # Log first 5
            self.logger.debug(f"Station {idx}: lat={lats[idx]:.4f}, lon={lons[idx]:.4f}")

        return station_indices

    def _extract_station_data(
        self,
        ds: xr.Dataset,
        station_indices: List[int]
    ) -> pd.DataFrame:
        """
        Extract SWE data for selected stations.

        Args:
            ds: CanSWE xarray Dataset
            station_indices: List of station indices to extract

        Returns:
            DataFrame with datetime index and columns for SWE data
        """
        # Find SWE variable
        swe_var = None
        for name in ['swe', 'SWE', 'snow_water_equivalent', 'snw']:
            if name in ds:
                swe_var = name
                break

        if swe_var is None:
            # Look in data_vars
            for var in ds.data_vars:
                if 'swe' in var.lower() or 'snow' in var.lower():
                    swe_var = var
                    break

        if swe_var is None:
            raise DataAcquisitionError("Cannot find SWE variable in CanSWE dataset")

        self.logger.debug(f"Extracting SWE from variable: {swe_var}")

        # Determine station dimension
        station_dim = None
        for dim in ['station', 'site', 'location', 'obs']:
            if dim in ds.dims:
                station_dim = dim
                break

        if station_dim is None:
            # Try to infer from SWE shape
            swe_data = ds[swe_var]
            if swe_data.ndim == 2:
                # Assume (time, station) or (station, time)
                if 'time' in swe_data.dims:
                    station_dim = [d for d in swe_data.dims if d != 'time'][0]

        records = []

        # Get station IDs if available
        station_id_var = None
        for name in ['station_id', 'site_id', 'station_name', 'id']:
            if name in ds:
                station_id_var = name
                break

        for idx in station_indices:
            try:
                # Select station data
                if station_dim:
                    station_data = ds[swe_var].isel({station_dim: idx})
                else:
                    station_data = ds[swe_var][:, idx] if ds[swe_var].ndim == 2 else ds[swe_var]

                # Get time coordinate
                if 'time' in station_data.dims:
                    times = station_data['time'].values
                else:
                    times = ds['time'].values if 'time' in ds else None

                if times is None:
                    continue

                swe_values = station_data.values

                # Get station ID
                if station_id_var:
                    try:
                        if station_dim:
                            sid = ds[station_id_var].isel({station_dim: idx}).values
                        else:
                            sid = ds[station_id_var][idx].values
                        # Handle bytes
                        if isinstance(sid, bytes):
                            sid = sid.decode('utf-8')
                        elif hasattr(sid, 'item'):
                            sid = sid.item()
                    except Exception:  # noqa: BLE001 — preprocessing resilience
                        sid = f"station_{idx}"
                else:
                    sid = f"station_{idx}"

                # Create records
                for t, swe in zip(times, swe_values):
                    if not np.isnan(swe):
                        records.append({
                            'datetime': pd.to_datetime(t),
                            'swe_mm': float(swe),
                            'station_id': str(sid),
                            'station_idx': idx
                        })

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed to extract station {idx}: {e}", exc_info=True)
                continue

        df = pd.DataFrame(records)
        if df.empty:
            raise DataAcquisitionError("No valid SWE observations extracted from CanSWE stations")

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        self.logger.info(f"Extracted {len(df)} SWE observations from {df['station_id'].nunique()} stations")
        return df

    def _aggregate_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multi-station data to a single time series.

        Uses daily mean of all available stations.

        Args:
            df: DataFrame with multi-station observations

        Returns:
            DataFrame with daily aggregated SWE
        """
        # Resample to daily and compute mean across stations
        if 'station_id' in df.columns:
            # Group by date, take mean of all stations
            df_daily = df.groupby(df.index.date).agg({
                'swe_mm': 'mean',
            })
            df_daily.index = pd.to_datetime(df_daily.index)
            df_daily.index.name = 'datetime'

            # Add station count per day
            station_counts = df.groupby(df.index.date)['station_id'].nunique()
            df_daily['n_stations'] = station_counts.values

        else:
            df_daily = df.resample('D').mean()

        return df_daily

    def _save_station_metadata(self, ds: xr.Dataset, station_indices: List[int]) -> None:
        """Save metadata about the selected stations."""
        output_dir = self._get_observation_dir('snow', 'preprocessed')
        meta_file = output_dir / f"{self.domain_name}_canswe_stations.csv"

        # Try to extract station info
        records = []
        for idx in station_indices:
            record = {'station_idx': idx}

            # Try to get various attributes
            for var in ['lat', 'latitude', 'Latitude']:
                if var in ds:
                    try:
                        val = ds[var].values
                        if val.ndim > 0:
                            record['lat'] = float(val[idx]) if idx < len(val) else np.nan
                        else:
                            record['lat'] = float(val)
                    except Exception:  # noqa: BLE001 — preprocessing resilience
                        pass

            for var in ['lon', 'longitude', 'Longitude']:
                if var in ds:
                    try:
                        val = ds[var].values
                        if val.ndim > 0:
                            record['lon'] = float(val[idx]) if idx < len(val) else np.nan
                        else:
                            record['lon'] = float(val)
                    except Exception:  # noqa: BLE001 — preprocessing resilience
                        pass

            for var in ['station_id', 'site_id', 'station_name', 'name']:
                if var in ds:
                    try:
                        val = ds[var].values
                        if val.ndim > 0 and idx < len(val):
                            sid = val[idx]
                            if isinstance(sid, bytes):
                                sid = sid.decode('utf-8')
                            record['station_id'] = str(sid)
                    except Exception:  # noqa: BLE001 — preprocessing resilience
                        pass

            for var in ['elevation', 'elev', 'alt', 'altitude']:
                if var in ds:
                    try:
                        val = ds[var].values
                        if val.ndim > 0 and idx < len(val):
                            record['elevation_m'] = float(val[idx])
                    except Exception:  # noqa: BLE001 — preprocessing resilience
                        pass

            records.append(record)

        if records:
            pd.DataFrame(records).to_csv(meta_file, index=False)
            self.logger.debug(f"Saved station metadata to {meta_file}")


# Also register NorSWE as an alias with extended coverage
@R.observation_handlers.add('norswe')
@R.observation_handlers.add('norswe_swe')
class NorSWEHandler(CanSWEHandler):
    """
    Handles NorSWE (Northern Hemisphere SWE) data acquisition and processing.

    NorSWE extends CanSWE to include stations from Russia, Finland, Norway,
    and Switzerland for the period 1979-2021.

    Note: NorSWE follows the same format as CanSWE, so this handler inherits
    from CanSWEHandler and only overrides the Zenodo record information.
    """

    source_name = "NorSWE"

    # NorSWE Zenodo record
    NORSWE_ZENODO_RECORD = '15263370'

    def acquire(self) -> Path:
        """Acquire NorSWE data from Zenodo."""
        download_enabled = self._get_config_value(
            lambda: None,
            default=True,
            dict_key='DOWNLOAD_NORSWE'
        )
        if isinstance(download_enabled, str):
            download_enabled = download_enabled.lower() == 'true'

        # Check for existing file path override
        norswe_path = self._get_config_value(
            lambda: None,
            default=None,
            dict_key='NORSWE_PATH'
        )
        if norswe_path and norswe_path != 'default' and Path(norswe_path).exists():
            self.logger.debug(f"Using existing NorSWE file: {norswe_path}")
            return Path(norswe_path)

        # Set up output directory
        raw_dir = self.project_observations_dir / "snow" / "norswe" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / "NorSWE_v1.nc"

        if raw_file.exists() and not download_enabled:
            self.logger.debug(f"Using existing NorSWE file: {raw_file}")
            return raw_file

        if download_enabled:
            return self._download_norswe(raw_file)
        else:
            if raw_file.exists():
                return raw_file
            raise DataAcquisitionError(
                f"NorSWE file not found at {raw_file} and download is disabled. "
                "Set DOWNLOAD_NORSWE: true or provide NORSWE_PATH."
            )

    def _download_norswe(self, output_path: Path) -> Path:
        """Download NorSWE dataset from Zenodo."""
        self.logger.info(f"Downloading NorSWE from Zenodo (record {self.NORSWE_ZENODO_RECORD})")

        try:
            # Get record metadata
            record_url = f"{self.ZENODO_API}/{self.NORSWE_ZENODO_RECORD}"
            response = requests.get(record_url, timeout=60)
            response.raise_for_status()
            record_data = response.json()

            # Find the NetCDF file
            nc_file_url = None
            for file_info in record_data.get('files', []):
                filename = file_info.get('key', '')
                if filename.endswith('.nc'):
                    nc_file_url = file_info.get('links', {}).get('self')
                    self.logger.debug(f"Found NorSWE NetCDF: {filename}")
                    break

            if not nc_file_url:
                # Try direct URL pattern
                nc_file_url = f"https://zenodo.org/records/{self.NORSWE_ZENODO_RECORD}/files/NorSWE_1979-2021_v1.nc"

            # Download
            self.logger.debug(f"Downloading from: {nc_file_url}")
            headers = {'User-Agent': 'SYMFLUENCE/1.0'}
            response = requests.get(nc_file_url, headers=headers, stream=True, timeout=600)
            response.raise_for_status()

            tmp_file = output_path.with_suffix('.nc.part')
            with open(tmp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            tmp_file.replace(output_path)
            self.logger.info(f"Successfully downloaded NorSWE to {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            raise DataAcquisitionError(f"Failed to download NorSWE from Zenodo: {e}") from e
