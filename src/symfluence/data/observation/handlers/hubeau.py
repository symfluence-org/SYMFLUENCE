# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Hub'Eau (France) Observation Handlers

Provides handlers for French hydrological data from the Hub'Eau API:
- Streamflow (discharge) from hydrometric stations
- Water levels from hydrometric stations

Hub'Eau is the official French government open data API for water data.
API Documentation: https://hubeau.eaufrance.fr/page/api-hydrometrie

Example stations:
- Seine at Paris (Austerlitz): H5920010
- Loire at Montjean: M5300010
- Garonne at Tonneins: O5550010
- Rhône at Beaucaire: V7200015
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.registries import R

from ..base import BaseObservationHandler

# Hub'Eau API configuration
HUBEAU_BASE_URL = "https://hubeau.eaufrance.fr/api/v1/hydrometrie"
HUBEAU_STATIONS_URL = f"{HUBEAU_BASE_URL}/referentiel/stations"
HUBEAU_OBS_TR_URL = f"{HUBEAU_BASE_URL}/observations_tr"  # Real-time observations
HUBEAU_OBS_ELAB_URL = f"{HUBEAU_BASE_URL}/obs_elab"  # Processed daily observations

# Default request parameters
DEFAULT_TIMEOUT = 60
MAX_RECORDS_PER_REQUEST = 10000  # Hub'Eau API limit

# Required headers for Hub'Eau API (prevents 403 errors)
HUBEAU_HEADERS = {
    'User-Agent': 'SYMFLUENCE-Hydrological-Modeling/1.0 (https://github.com/SYMFLUENCE)',
    'Accept': 'application/json',
}


class HubEauAPIError(DataAcquisitionError):
    """Hub'Eau API access error."""
    pass


def _hubeau_request(url: str, params: dict, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    """
    Make a request to Hub'Eau API with proper headers.

    Note: The Hub'Eau API may be geo-restricted to French IP addresses.
    If you're outside France, consider using a VPN or downloading data manually.

    Raises:
        HubEauAPIError: If API returns 403 Forbidden (likely geo-restriction)
        requests.HTTPError: For other HTTP errors
    """
    try:
        response = requests.get(url, params=params, headers=HUBEAU_HEADERS, timeout=timeout)

        if response.status_code == 403:
            raise HubEauAPIError(
                "Hub'Eau API returned 403 Forbidden. This API may be geo-restricted "
                "to French IP addresses. Options:\n"
                "  1. Use a French VPN\n"
                "  2. Download data manually from https://www.hydro.eaufrance.fr/\n"
                "  3. Use pre-downloaded CSV files with DOWNLOAD_HUBEAU_DATA: false"
            )

        response.raise_for_status()
        return response

    except requests.exceptions.ConnectionError as e:
        raise HubEauAPIError(f"Cannot connect to Hub'Eau API: {e}") from e
    except requests.exceptions.Timeout:
        raise HubEauAPIError(f"Hub'Eau API request timed out after {timeout}s") from None


@R.observation_handlers.add('hubeau_streamflow')
class HubEauStreamflowHandler(BaseObservationHandler):
    """
    Handles French streamflow (discharge) data from Hub'Eau API.

    The Hub'Eau hydrometry API provides:
    - Real-time observations (observations_tr): Sub-daily discharge data
    - Processed observations (obs_elab): Daily aggregated data with QC

    Configuration options:
        evaluation.streamflow.station_id: Hub'Eau station code (e.g., "H5920010")
        evaluation.hubeau.station_id: Alternative config path
        data.hubeau_station_code: Alternative config path
        data.download_hubeau_data: Enable/disable download (default: True)
        evaluation.hubeau.use_daily: Use daily processed data (default: True)

    Example config:
        evaluation:
          streamflow:
            station_id: "H5920010"  # Seine at Paris
          hubeau:
            download: true
            use_daily: true
    """

    obs_type = "streamflow"
    source_name = "Hub'Eau_Hydrometrie"

    def acquire(self) -> Path:
        """
        Acquire French streamflow data from Hub'Eau API or locate local raw file.

        Returns:
            Path to raw data file (JSON format)
        """
        # Get station ID from various config paths
        station_id = self._get_station_id()
        if not station_id:
            self.logger.debug("Hub'Eau station ID not found, skipping acquisition")
            return self.project_observations_dir / "streamflow" / "raw_data"

        # (The first term used to be a non-existent path that always yielded
        # its default True, so DOWNLOAD_HUBEAU_DATA: False was ignored.)
        download_enabled = self._get_config_value(
            lambda: self.config.evaluation.hubeau.download,
            dict_key='DOWNLOAD_HUBEAU_DATA',
        )

        raw_dir = self.project_observations_dir / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"hubeau_{station_id}_raw.json"

        if download_enabled:
            return self._download_data(station_id, raw_file)
        else:
            # Look for existing raw file
            if raw_file.exists():
                return raw_file

            # Check for CSV alternative
            csv_file = raw_dir / f"hubeau_{station_id}_raw.csv"
            if csv_file.exists():
                return csv_file

            self.logger.warning(f"Hub'Eau raw file not found and download disabled: {raw_file}")
            return raw_file

    def _get_station_id(self) -> Optional[str]:
        """Get Hub'Eau station ID from config."""
        return (
            self._get_config_value(lambda: self.config.evaluation.streamflow.station_id) or
            self._get_config_value(lambda: None, dict_key='HUBEAU_STATION_ID') or
            self._get_config_value(lambda: self.config.data.streamflow_station_id)
        )

    def _download_data(self, station_id: str, output_path: Path) -> Path:
        """
        Download discharge data from Hub'Eau API.

        Tries processed daily data first (obs_elab), falls back to real-time (observations_tr).

        Args:
            station_id: Hub'Eau station code
            output_path: Path to save raw data

        Returns:
            Path to downloaded data file
        """
        self.logger.info(f"Downloading Hub'Eau streamflow data for station {station_id}")

        # Use experiment time range
        start_date = self.start_date.strftime("%Y-%m-%d")
        end_date = self.end_date.strftime("%Y-%m-%d") if self.end_date else datetime.now().strftime("%Y-%m-%d")

        # Prefer daily processed data (more reliable, QC'd)
        use_daily = self._get_config_value(
            lambda: self.config.evaluation.hubeau.use_daily,
            dict_key='HUBEAU_USE_DAILY',
        )

        if use_daily:
            try:
                data = self._fetch_daily_data(station_id, start_date, end_date)
                if data:
                    return self._save_json(data, output_path, 'daily')
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Daily data fetch failed: {e}, trying real-time...", exc_info=True)

        # Fall back to real-time observations
        try:
            data = self._fetch_realtime_data(station_id, start_date, end_date)
            if data:
                return self._save_json(data, output_path, 'realtime')
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Real-time data fetch also failed: {e}", exc_info=True)

        raise DataAcquisitionError(f"Could not retrieve Hub'Eau data for station {station_id}")

    def _fetch_daily_data(self, station_id: str, start_date: str, end_date: str) -> List[dict]:
        """Fetch processed daily discharge data (QJO - débit journalier)."""
        all_data = []
        cursor = None

        while True:
            params = {
                'code_entite': station_id,
                'date_debut_obs_elab': start_date,
                'date_fin_obs_elab': end_date,
                'grandeur_hydro_elab': 'QmJ',  # Daily mean discharge
                'size': MAX_RECORDS_PER_REQUEST,
            }
            if cursor:
                params['cursor'] = cursor

            response = _hubeau_request(HUBEAU_OBS_ELAB_URL, params)
            response.raise_for_status()

            result = response.json()
            data = result.get('data', [])
            all_data.extend(data)

            # Check for pagination
            cursor = result.get('next')
            if not cursor or not data:
                break

            self.logger.debug(f"Fetched {len(all_data)} records so far...")

        self.logger.info(f"Downloaded {len(all_data)} daily records from Hub'Eau")
        return all_data

    def _fetch_realtime_data(self, station_id: str, start_date: str, end_date: str) -> List[dict]:
        """Fetch real-time discharge observations."""
        all_data = []
        cursor = None

        while True:
            params = {
                'code_entite': station_id,
                'date_debut_obs': start_date,
                'date_fin_obs': end_date,
                'grandeur_hydro': 'Q',  # Discharge
                'size': MAX_RECORDS_PER_REQUEST,
            }
            if cursor:
                params['cursor'] = cursor

            response = _hubeau_request(HUBEAU_OBS_TR_URL, params)
            response.raise_for_status()

            result = response.json()
            data = result.get('data', [])
            all_data.extend(data)

            cursor = result.get('next')
            if not cursor or not data:
                break

        self.logger.info(f"Downloaded {len(all_data)} real-time records from Hub'Eau")
        return all_data

    def _save_json(self, data: List[dict], output_path: Path, data_type: str) -> Path:
        """Save data as JSON file."""
        import json

        output = {
            'source': 'Hub\'Eau Hydrometrie API',
            'data_type': data_type,
            'download_date': datetime.now().isoformat(),
            'count': len(data),
            'data': data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved Hub'Eau {data_type} data to {output_path}")
        return output_path

    def process(self, input_path: Path) -> Path:
        """
        Process Hub'Eau JSON data into standard SYMFLUENCE streamflow CSV.

        Args:
            input_path: Path to raw JSON file

        Returns:
            Path to processed CSV file
        """
        import json

        if not input_path.exists():
            raise FileNotFoundError(f"Hub'Eau raw data file not found: {input_path}")

        self.logger.info(f"Processing Hub'Eau streamflow data from {input_path}")

        # Load JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        data_type = raw.get('data_type', 'unknown')
        records = raw.get('data', [])

        if not records:
            raise DataAcquisitionError(f"No records in Hub'Eau data file: {input_path}")

        # Parse based on data type
        if data_type == 'daily':
            df = self._parse_daily_data(records)
        else:
            df = self._parse_realtime_data(records)

        if df.empty:
            raise DataAcquisitionError("No valid discharge records after parsing")

        # Convert units: Hub'Eau uses L/s, convert to m³/s
        df['discharge_cms'] = df['resultat_obs'] / 1000.0

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        if data_type == 'daily':
            # QmJ daily means: step within each day, no interpolation/gap fill.
            resampled = self._daily_streamflow_to_target(df['discharge_cms'])
        else:
            resampled = df['discharge_cms'].resample(resample_freq).mean()
            # Interpolate small gaps in the sub-daily real-time series (up to 3 days)
            resampled = resampled.interpolate(method='time', limit_direction='both', limit=72)

        # Create metadata
        station_id = self._get_station_id()
        metadata = self._create_metadata(
            variable='streamflow',
            units='m3/s',
            temporal_resolution=resample_freq,
            spatial_aggregation='point',
            station_id=station_id,
            quality_flags={'source_units': 'L/s', 'converted_to': 'm3/s'}
        )

        # Save with metadata
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"

        resampled.to_csv(output_file, header=True, index_label='datetime')

        # Save metadata
        meta_file = output_file.with_suffix('.json')
        import json as json_module
        with open(meta_file, 'w', encoding='utf-8') as f:
            json_module.dump(metadata.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Hub'Eau streamflow processing complete: {output_file}")
        return output_file

    def _parse_daily_data(self, records: List[dict]) -> pd.DataFrame:
        """Parse daily processed data (obs_elab)."""
        rows = []
        for rec in records:
            try:
                # Daily data uses date_obs_elab
                date_str = rec.get('date_obs_elab')
                value = rec.get('resultat_obs_elab')

                if date_str and value is not None:
                    rows.append({
                        'datetime': pd.to_datetime(date_str),
                        'resultat_obs': float(value),
                        'code_station': rec.get('code_station', ''),
                        'libelle_qualification': rec.get('libelle_qualification', '')
                    })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
        return df

    def _parse_realtime_data(self, records: List[dict]) -> pd.DataFrame:
        """Parse real-time observations (observations_tr)."""
        rows = []
        for rec in records:
            try:
                # Real-time uses date_obs
                date_str = rec.get('date_obs')
                value = rec.get('resultat_obs')

                if date_str and value is not None:
                    rows.append({
                        'datetime': pd.to_datetime(date_str),
                        'resultat_obs': float(value),
                        'code_station': rec.get('code_station', ''),
                        'continuite_obs_hydro': rec.get('continuite_obs_hydro', '')
                    })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
        return df


@R.observation_handlers.add('hubeau_waterlevel')
class HubEauWaterLevelHandler(BaseObservationHandler):
    """
    Handles French water level data from Hub'Eau API.

    Water levels (hauteur d'eau) are useful for:
    - Rating curve development
    - Flood monitoring
    - Low-flow analysis

    Configuration:
        evaluation.waterlevel.station_id: Hub'Eau station code
        evaluation.hubeau.station_id: Alternative config path
    """

    obs_type = "waterlevel"
    source_name = "Hub'Eau_Hydrometrie"

    def acquire(self) -> Path:
        """Acquire water level data from Hub'Eau API."""
        station_id = (
            self._get_config_value(lambda: None, dict_key='WATERLEVEL_STATION_ID') or
            self._get_config_value(lambda: None, dict_key='HUBEAU_STATION_ID')
        )

        if not station_id:
            self.logger.debug("Hub'Eau station ID not found for water level")
            return self.project_observations_dir / "waterlevel" / "raw_data"

        download_enabled = self._get_config_value(
            lambda: self.config.evaluation.hubeau.download,
            dict_key='DOWNLOAD_HUBEAU_DATA',
        )

        raw_dir = self.project_observations_dir / "waterlevel" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"hubeau_wl_{station_id}_raw.json"

        if download_enabled:
            return self._download_water_level(station_id, raw_file)
        elif raw_file.exists():
            return raw_file
        else:
            self.logger.warning(f"Hub'Eau water level file not found: {raw_file}")
            return raw_file

    def _download_water_level(self, station_id: str, output_path: Path) -> Path:
        """Download water level data."""
        import json

        self.logger.info(f"Downloading Hub'Eau water level for station {station_id}")

        start_date = self.start_date.strftime("%Y-%m-%d")
        end_date = self.end_date.strftime("%Y-%m-%d") if self.end_date else datetime.now().strftime("%Y-%m-%d")

        all_data = []
        cursor = None

        while True:
            params = {
                'code_entite': station_id,
                'date_debut_obs': start_date,
                'date_fin_obs': end_date,
                'grandeur_hydro': 'H',  # Water level (Hauteur)
                'size': MAX_RECORDS_PER_REQUEST,
            }
            if cursor:
                params['cursor'] = cursor

            response = _hubeau_request(HUBEAU_OBS_TR_URL, params)
            response.raise_for_status()

            result = response.json()
            data = result.get('data', [])
            all_data.extend(data)

            cursor = result.get('next')
            if not cursor or not data:
                break

        if not all_data:
            raise DataAcquisitionError(f"No water level data for station {station_id}")

        output = {
            'source': 'Hub\'Eau Hydrometrie API',
            'variable': 'water_level',
            'download_date': datetime.now().isoformat(),
            'count': len(all_data),
            'data': all_data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Downloaded {len(all_data)} water level records")
        return output_path

    def process(self, input_path: Path) -> Path:
        """Process water level data into standard format."""
        import json

        if not input_path.exists():
            raise FileNotFoundError(f"Water level file not found: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        records = raw.get('data', [])
        if not records:
            raise DataAcquisitionError("No water level records in file")

        rows = []
        for rec in records:
            try:
                date_str = rec.get('date_obs')
                value = rec.get('resultat_obs')
                if date_str and value is not None:
                    rows.append({
                        'datetime': pd.to_datetime(date_str),
                        'water_level_mm': float(value),  # Hub'Eau uses mm
                        'code_station': rec.get('code_station', '')
                    })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            raise DataAcquisitionError("No valid water level records after parsing")

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Convert mm to meters
        df['water_level_m'] = df['water_level_mm'] / 1000.0

        # Resample to hourly
        resampled = df['water_level_m'].resample('h').mean()
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=24)

        output_dir = self.project_observations_dir / "waterlevel" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_waterlevel_processed.csv"

        resampled.to_csv(output_file, header=True, index_label='datetime')
        self.logger.info(f"Water level processing complete: {output_file}")
        return output_file


def search_hubeau_stations(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    river_name: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    Search for Hub'Eau hydrometric stations.

    Args:
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
        river_name: Filter by river name (partial match)
        department: French department code (e.g., "75" for Paris)
        limit: Maximum stations to return

    Returns:
        DataFrame with station information

    Example:
        >>> stations = search_hubeau_stations(river_name="Seine", limit=10)
        >>> print(stations[['code_station', 'libelle_station', 'libelle_cours_eau']])
    """
    params = {'size': min(limit, 1000)}

    if bbox:
        params['bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    if river_name:
        params['libelle_cours_eau'] = river_name
    if department:
        params['code_departement'] = department

    response = _hubeau_request(HUBEAU_STATIONS_URL, params)
    data = response.json().get('data', [])

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def get_station_info(station_id: str) -> dict:
    """
    Get detailed information about a Hub'Eau station.

    Args:
        station_id: Hub'Eau station code (e.g., "H5920010")

    Returns:
        Dictionary with station metadata

    Example:
        >>> info = get_station_info("H5920010")
        >>> print(f"Station: {info['libelle_station']} on {info['libelle_cours_eau']}")
    """
    params = {'code_station': station_id}
    response = _hubeau_request(HUBEAU_STATIONS_URL, params)
    data = response.json().get('data', [])
    if not data:
        raise ValueError(f"Station not found: {station_id}")

    return data[0]
