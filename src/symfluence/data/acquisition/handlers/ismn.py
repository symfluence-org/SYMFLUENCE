# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ISMN Soil Moisture Acquisition Handler

Provides cloud acquisition for ISMN (International Soil Moisture Network) data:
- Station metadata discovery (network name, location, data availability)
- Data filtering by spatial (bbox, radius) and temporal ranges
- Multi-depth measurements (0.05m, 0.2m, 0.5m, etc.)
- Multiple soil moisture variable types

ISMN Overview:
    Data Type: In-situ soil moisture observations
    Coverage: Global (sparse network stations)
    Source: ISMN (ismn.earth)

Authentication:
    - Configuration: ISMN_USERNAME / ISMN_PASSWORD
    - Environment: ISMN_USERNAME / ISMN_PASSWORD
    - netrc file with ismn.earth entry
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('ISMN')
class ISMNAcquirer(BaseAcquisitionHandler):
    """Acquires ISMN (International Soil Moisture Network) soil moisture data.

    ISMN provides in-situ soil moisture observations from global networks.
    This handler downloads data via an HTTP API that supports:
    - Station metadata discovery (network name, location, data availability)
    - Data filtering by spatial (bbox, radius) and temporal ranges
    - Multi-depth measurements (0.05m, 0.2m, 0.5m, etc.)
    - Multiple soil moisture variable types

    Configuration Parameters:
        ISMN_API_BASE: Base URL for ISMN API (default: https://ismn.earth/dataviewer)
        ISMN_METADATA_URL: URL for station metadata (JSON or CSV)
        ISMN_USERNAME/PASSWORD: Authentication credentials (or via netrc)
        ISMN_SEARCH_RADIUS_KM: Limit stations within radius of bbox center
        ISMN_MAX_STATIONS: Maximum number of stations to download (default: 3)

    Workflow:
        1. Load all available ISMN station metadata
        2. Filter to stations within bounding box (or search radius)
        3. For each selected station, fetch variable list for date range
        4. Download soil moisture data for each depth/sensor combination
        5. Store as CSV files with station_slug_depth_ID.csv naming

    Auth Strategy:
        - Check config ISMN_USERNAME/ISMN_PASSWORD
        - Fall back to ISMN_USERNAME/ISMN_PASSWORD environment variables
        - Fall back to netrc credentials matching ISMN API host
    """

    def download(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        force_download = bool(self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'))
        if any(output_dir.glob("*.csv")) and not force_download:
            return output_dir

        api_base = self._get_config_value(lambda: self.config.evaluation.ismn.api_base, default='https://ismn.earth/dataviewer', dict_key='ISMN_API_BASE').rstrip("/")
        metadata_url = self._get_config_value(lambda: self.config.evaluation.ismn.metadata_url, dict_key='ISMN_METADATA_URL') or "https://ismn.earth/static/dataviewer/network_station_details.json"
        variable_list_url = self._get_config_value(lambda: self.config.evaluation.ismn.variable_list_url, dict_key='ISMN_VARIABLE_LIST_URL')
        if not variable_list_url:
            variable_list_url = f"{api_base}/dataviewer_get_variable_list/"
        data_template = self._get_config_value(lambda: self.config.evaluation.ismn.data_url_template, dict_key='ISMN_DATA_URL_TEMPLATE')
        if not data_template:
            data_template = (
                f"{api_base}/dataviewer_load_variable/"
                "?station_id={station_id}&start={start_date}&end={end_date}"
                "&depth_id={depth_id}&sensor_id={sensor_id}&variable_id={variable_id}"
            )

        session = requests.Session()
        auth = self._resolve_auth(metadata_url)
        if auth:
            session.auth = auth

        stations = self._load_station_metadata(session, metadata_url)
        if stations is None or stations.empty:
            raise RuntimeError("ISMN station metadata could not be loaded.")

        stations = self._select_stations(stations)
        if stations.empty:
            raise RuntimeError("No ISMN stations found for the requested domain.")

        selection_file = output_dir / "ismn_station_selection.csv"
        stations.to_csv(selection_file, index=False)

        start_date = self.start_date.strftime("%Y/%m/%d")
        end_date = self.end_date.strftime("%Y/%m/%d")

        downloaded = 0
        for _, row in stations.iterrows():
            station_id = str(row["station_id"])
            station_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", station_id).strip("_")
            variables = self._fetch_variable_list(session, variable_list_url, station_id, start_date, end_date)
            if not variables:
                continue
            sm_vars = self._select_soil_moisture_variables(variables)
            if not sm_vars:
                continue

            for var in sm_vars:
                depth_id = var.get("depthId")
                sensor_id = var.get("sensorId")
                variable_id = var.get("variableId")
                if depth_id is None or sensor_id is None or variable_id is None:
                    continue

                depth_m = self._extract_depth_m(var.get("variableName", ""))
                out_file = output_dir / f"{station_slug}_depth_{depth_id}.csv"
                if out_file.exists() and not force_download:
                    downloaded += 1
                    continue

                data_url = data_template.format(
                    station_id=station_id,
                    start_date=start_date,
                    end_date=end_date,
                    depth_id=depth_id,
                    sensor_id=sensor_id,
                    variable_id=variable_id,
                )
                auth = self._resolve_auth(data_url)
                if auth:
                    session.auth = auth

                self.logger.info(f"Downloading ISMN station data: {station_id} depth_id={depth_id}")
                try:
                    resp = session.get(data_url, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()
                    if not isinstance(data, list) or len(data) < 2:
                        continue
                    dates, values = data[0], data[1]
                    df = pd.DataFrame({"DateTime": dates, "soil_moisture": values})
                    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce")
                    unit = str(var.get("unit", "")).lower()
                    if "* 100" in unit or unit.endswith("100") or df["soil_moisture"].max() > 1.5:
                        df["soil_moisture"] = df["soil_moisture"] / 100.0
                    if depth_m is not None:
                        df["depth_m"] = depth_m
                    df = df.dropna(subset=["soil_moisture"])
                    if df.empty:
                        continue
                    df.to_csv(out_file, index=False)
                    downloaded += 1
                except Exception as exc:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Failed to download ISMN station {station_id}: {exc}", exc_info=True)

        if downloaded == 0:
            raise RuntimeError("No ISMN station data downloaded.")

        return output_dir

    def _resolve_auth(self, url: str):
        """Resolve authentication credentials from multiple sources.

        Tries three strategies in order:
        1. Configuration dict (ISMN_USERNAME, ISMN_PASSWORD)
        2. Environment variables (ISMN_USERNAME, ISMN_PASSWORD)
        3. netrc file (using hostname from URL)

        The netrc approach allows storing credentials in ~/.netrc without
        setting environment variables, which is useful for shared systems.

        Args:
            url: URL to authenticate against (hostname extracted for netrc lookup)

        Returns:
            Tuple of (username, password) or None if no credentials found

        Note:
            An unreadable or malformed .netrc is skipped (debug log), not raised.
        """
        user = self._get_config_value(lambda: None, default=None, dict_key='ISMN_USERNAME') or os.environ.get("ISMN_USERNAME")
        password = self._get_config_value(lambda: None, default=None, dict_key='ISMN_PASSWORD') or os.environ.get("ISMN_PASSWORD")
        if user and password:
            return (user, password)

        from urllib.parse import urlparse

        from ..utils import read_netrc_credentials

        host = urlparse(url).hostname
        found = read_netrc_credentials([host], logger=self.logger) if host else None
        return (found[0], found[1]) if found else None

    def _load_station_metadata(self, session: requests.Session, metadata_url: str) -> Optional[pd.DataFrame]:
        try:
            resp = session.get(metadata_url, timeout=600)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to fetch ISMN station metadata: {exc}", exc_info=True)
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type or metadata_url.endswith(".json"):
            try:
                payload = resp.json()
            except ValueError as exc:
                self.logger.warning(f"Failed to parse ISMN metadata JSON: {exc}")
                return None
            if isinstance(payload, dict) and "Networks" in payload:
                df = self._parse_dataviewer_metadata(payload)
            else:
                records = payload
                if isinstance(payload, dict):
                    for key in ("data", "stations", "items"):
                        if key in payload:
                            records = payload[key]
                            break
                if not isinstance(records, list):
                    self.logger.warning("Unexpected ISMN metadata JSON structure.")
                    return None
                df = pd.DataFrame(records)
        else:
            from io import StringIO

            df = pd.read_csv(StringIO(resp.text))

        if df.empty:
            return None

        col_map = self._normalize_station_columns(df.columns)
        df = df.rename(columns=col_map)
        required = {"station_id", "latitude", "longitude"}
        if not required.issubset(df.columns):
            self.logger.warning("ISMN metadata missing station_id/latitude/longitude columns.")
            return None

        cols = ["station_id", "latitude", "longitude"]
        if "network" in df.columns:
            cols.append("network")
        if "start_date" in df.columns:
            cols.append("start_date")
        if "end_date" in df.columns:
            cols.append("end_date")
        return df[cols].copy()

    def _normalize_station_columns(self, columns):
        col_map = {}
        for col in columns:
            lower = col.lower()
            if lower in ("station_id", "station", "stationid", "id"):
                col_map[col] = "station_id"
            elif "lat" in lower:
                col_map[col] = "latitude"
            elif "lon" in lower or "lng" in lower:
                col_map[col] = "longitude"
            elif "network" in lower:
                col_map[col] = "network"
        return col_map

    def _parse_dataviewer_metadata(self, payload: dict) -> pd.DataFrame:
        records = []
        for network in payload.get("Networks", []) or []:
            network_id = network.get("networkID")
            for station in network.get("Stations", []) or []:
                lat = station.get("lat")
                lon = station.get("lng")
                station_id = station.get("stationID")
                if lat is None or lon is None or station_id is None:
                    continue
                records.append({
                    "station_id": station_id,
                    "latitude": lat,
                    "longitude": lon,
                    "network": network_id,
                    "start_date": station.get("minimum"),
                    "end_date": station.get("maximum"),
                })
        return pd.DataFrame(records)

    def _select_stations(self, stations: pd.DataFrame) -> pd.DataFrame:
        """Select and rank stations based on spatial/temporal coverage and proximity.

        Implements a multi-step filtering strategy:
        1. **Spatial filter**: Keep stations within bounding box
        2. **Fallback**: If no stations in bbox, use all stations (then rank by distance)
        3. **Temporal filter**: If date range provided, keep only overlapping stations
        4. **Fallback**: If no overlap, use all stations with any data
        5. **Distance ranking**: Compute Haversine distance to bbox center
        6. **Radius filter**: If configured, keep only stations within search radius
        7. **Top N selection**: Return closest N stations (default: 3)

        This multi-step approach is robust to incomplete metadata (missing dates)
        and prioritizes local stations when available.

        Args:
            stations: DataFrame with columns: station_id, latitude, longitude,
                     and optionally: network, start_date, end_date

        Returns:
            pd.DataFrame: Filtered and ranked stations, up to ISMN_MAX_STATIONS entries

        Algorithm Details:
            - Fallback logic: If bbox filter returns empty, all stations are retained
              and ranked by distance instead (useful for data-sparse regions)
            - Distance calculation: Haversine formula in kilometers
            - Search radius: Can be used to limit to nearest neighbors only
        """
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2

        all_stations = stations.copy()
        stations = all_stations[
            (stations["latitude"] >= lat_min) & (stations["latitude"] <= lat_max) &
            (stations["longitude"] >= lon_min) & (stations["longitude"] <= lon_max)
        ]

        if stations.empty:
            stations = all_stations.copy()

        if "start_date" in stations.columns and "end_date" in stations.columns:
            stations = stations.copy()
            stations["start_date"] = pd.to_datetime(stations["start_date"], errors="coerce")
            stations["end_date"] = pd.to_datetime(stations["end_date"], errors="coerce")
            start = pd.to_datetime(self.start_date, errors="coerce")
            end = pd.to_datetime(self.end_date, errors="coerce")
            if start is not pd.NaT and end is not pd.NaT:
                overlap = (
                    (stations["end_date"] >= start) &
                    (stations["start_date"] <= end)
                )
                stations = stations[overlap]
            if stations.empty:
                stations = all_stations.copy()

        stations["distance_km"] = self._haversine_km(
            stations["latitude"].astype(float),
            stations["longitude"].astype(float),
            lat_center,
            lon_center,
        )

        radius_km = self._get_config_value(lambda: self.config.evaluation.ismn.search_radius_km, dict_key='ISMN_SEARCH_RADIUS_KM')
        if radius_km:
            stations = stations[stations["distance_km"] <= float(radius_km)]

        max_stations = int(self._get_config_value(lambda: self.config.evaluation.ismn.max_stations, default=3, dict_key='ISMN_MAX_STATIONS'))
        stations = stations.sort_values("distance_km").head(max_stations)
        return stations

    def _fetch_variable_list(
        self,
        session: requests.Session,
        url: str,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        params = {
            "station_id": station_id,
            "start": start_date,
            "end": end_date,
        }
        try:
            resp = session.get(url, params=params, timeout=600)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to fetch ISMN variable list for {station_id}: {exc}", exc_info=True)
            return []
        return payload.get("variables", []) if isinstance(payload, dict) else []

    def _select_soil_moisture_variables(self, variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sm_vars = []
        for var in variables:
            name = str(var.get("variableName", "")).lower()
            quantity = str(var.get("quantityName", "")).lower()
            if "soil moisture" in quantity or "soil_moisture" in name:
                sm_vars.append(var)
        return sm_vars

    def _extract_depth_m(self, variable_name: str) -> Optional[float]:
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)m", variable_name)
        if not match:
            match = re.search(r"\s([0-9]+(?:\.[0-9]+)?)m", variable_name)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _haversine_km(self, lat_series, lon_series, lat0, lon0):
        """Calculate great-circle distance using Haversine formula.

        Computes the shortest distance between points on Earth's surface,
        accounting for spherical geometry. Uses Earth's mean radius (6371 km).

        This is preferred over Euclidean distance for geographic coordinates
        because Earth is approximately spherical. Haversine is accurate even
        for short distances (<1 km) unlike some approximations.

        The formula is numerically stable and avoids the tan(θ/2) singularity
        of similar methods.

        Args:
            lat_series: Numpy array or pandas Series of latitudes (degrees)
            lon_series: Numpy array or pandas Series of longitudes (degrees)
            lat0: Reference latitude (degrees) - scalar
            lon0: Reference longitude (degrees) - scalar

        Returns:
            np.ndarray: Distances in kilometers between each (lat, lon) pair
                       and the reference point (lat0, lon0)

        Note:
            Formula:
                a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
                c = 2·arcsin(√a)
                d = R·c  (R = Earth's radius)
        """
        lat1 = np.radians(lat_series.astype(float))
        lon1 = np.radians(lon_series.astype(float))
        lat2 = np.radians(float(lat0))
        lon2 = np.radians(float(lon0))

        dlat = lat1 - lat2
        dlon = lon1 - lon2

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * c
