# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WSC Observation Handlers

Provides handlers for Water Survey of Canada (WSC) streamflow data.
Supports both local HYDAT SQLite database extraction and web API acquisition.
"""
from pathlib import Path

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry


@ObservationRegistry.register('wsc_streamflow')
class WSCStreamflowHandler(BaseObservationHandler):
    """
    Handles WSC streamflow data acquisition and processing.
    """

    obs_type = "streamflow"
    source_name = "WSC_HYDAT"
    SOURCE_INFO = {
        'source': 'WSC HYDAT',
        'url': 'https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/',
    }

    def acquire(self) -> Path:
        self.logger.debug("WSCStreamflowHandler.acquire called")
        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='local', dict_key='DATA_ACCESS')
        download_enabled = self._get_config_value(lambda: self.config.evaluation.streamflow.download_wsc, default=False, dict_key='DOWNLOAD_WSC_DATA')
        station_id = self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID')

        # If this handler is being invoked at all, the workflow has
        # already decided WSC streamflow is needed (either via
        # ``streamflow_data_provider: WSC`` or an explicit entry in
        # ``additional_observations``). Treat ``download_wsc`` as opt-OUT,
        # not opt-in: the user should not need to set two flags to get
        # the obvious behaviour. This also prevents a silent fall-through
        # to the HYDAT path on machines without the HYDAT SQLite,
        # which was reported by NB/NV.
        streamflow_provider = (
            self._get_config_value(
                lambda: self.config.data.streamflow_data_provider,
                default='',
                dict_key='STREAMFLOW_DATA_PROVIDER',
            ) or ''
        )
        if not download_enabled and str(streamflow_provider).upper() == 'WSC':
            self.logger.info(
                "streamflow_data_provider=WSC implies download_wsc=True; "
                "enabling cloud GeoMet acquisition by default. "
                "Set DOWNLOAD_WSC_DATA: false to opt out (e.g. when using "
                "pre-staged HYDAT data)."
            )
            download_enabled = True

        self.logger.debug(f"WSC acquire - data_access={data_access}, download_enabled={download_enabled}, station_id={station_id}")

        if not station_id:
            self.logger.error("Missing STATION_ID in configuration for WSC streamflow")
            raise ValueError("STATION_ID required for WSC streamflow acquisition")

        raw_dir = self.project_observations_dir / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"wsc_{station_id}_raw.csv"

        # Cloud pathway: Use WSC GeoMet API. Default to GeoMet whenever
        # download is enabled, regardless of data_access — GeoMet is the
        # canonical public API and is reachable from any environment.
        # Only fall through to HYDAT when the user explicitly disabled
        # download (treating WSC as a local-data path).
        if download_enabled:
            self.logger.debug("Calling _download_from_geomet")
            return self._download_from_geomet(station_id, raw_file)

        self.logger.debug("WSC acquire - Falling back to local/default pathway")
        # Local/Default pathway: Use HYDAT or existing raw files
        if download_enabled:
            self.logger.info(f"WSC local access: will attempt HYDAT extraction if {raw_file} not found")
            return raw_file
        else:
            raw_name = self._get_config_value(lambda: self.config.evaluation.streamflow.raw_name, dict_key='STREAMFLOW_RAW_NAME')
            if raw_name and raw_name != 'default':
                custom_raw = raw_dir / raw_name
                if custom_raw.exists():
                    return custom_raw

            if raw_file.exists():
                return raw_file

            self.logger.warning(f"WSC raw file not found: {raw_file}")
            return raw_file

    def _download_from_geomet(self, station_id: str, output_path: Path) -> Path:
        self.logger.debug(f"_download_from_geomet called for station {station_id}")
        self.logger.info(f"Downloading WSC streamflow data for station {station_id} via GeoMet API")

        base_url = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
        page_limit = 10000

        try:
            all_rows = []
            offset = 0

            while True:
                params = {
                    'STATION_NUMBER': station_id,
                    'f': 'json',
                    'limit': page_limit,
                    'offset': offset
                }

                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()

                data = response.json()
                features = data.get('features', [])

                if not features:
                    if offset == 0:
                        raise DataAcquisitionError(
                            f"No data found for WSC station {station_id} in GeoMet API"
                        )
                    break

                for feat in features:
                    all_rows.append(feat.get('properties', {}))

                self.logger.debug(
                    f"WSC GeoMet page: offset={offset}, received={len(features)}, total={len(all_rows)}"
                )

                if len(features) < page_limit:
                    break

                offset += page_limit

            df = pd.DataFrame(all_rows)
            df.to_csv(output_path, index=False)

            self.logger.info(f"Successfully downloaded {len(df)} records to {output_path}")
            return output_path

        except DataAcquisitionError:
            raise
        except (requests.RequestException, OSError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to download WSC data from GeoMet: {e}")
            raise DataAcquisitionError(f"Could not retrieve WSC data for station {station_id}") from e

    def process(self, input_path: Path) -> Path:
        """
        Process WSC data (GeoMet JSON-to-CSV or legacy raw CSV) into standard SYMFLUENCE format.
        """
        if not input_path.exists():
            # Special case: check if we should try HYDAT extraction as a fallback
            hydat_path = self._get_config_value(lambda: self.config.evaluation.streamflow.hydat_path, dict_key='HYDAT_PATH')
            if hydat_path:
                return self._process_from_hydat()
            raise FileNotFoundError(f"WSC raw data file not found: {input_path}")

        self.logger.info(f"Processing WSC streamflow data from {input_path}")

        # Load the data
        try:
            df = pd.read_csv(input_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            self.logger.debug(f"Standard CSV parse failed: {e}, trying with comment='#'")
            # Try with '#' comments if it's a legacy RDB-like file
            df = pd.read_csv(input_path, comment='#')

        # Identify columns
        # GeoMet uses 'DATE' and 'VALUE' (Discharge)
        # Local files might use 'datetime' or 'Value'
        datetime_col = self._find_col(df.columns, ['date', 'datetime', 'ISO 8601 UTC', 'Timestamp'])
        discharge_col = self._find_col(df.columns, ['value', 'discharge', 'flow', 'discharge_cms'])

        if not datetime_col or not discharge_col:
            raise DataAcquisitionError(f"Could not identify required columns in WSC data: {input_path}")

        # Clean and convert
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df[discharge_col] = pd.to_numeric(df[discharge_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, discharge_col])

        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Standardize naming
        df['discharge_cms'] = df[discharge_col]  # WSC is already in cms (m3/s)

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=30)

        # Save processed data
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"

        resampled.to_csv(output_file, header=True, index_label='datetime')

        self.logger.info(f"WSC streamflow processing complete: {output_file}")
        return output_file

    def _process_from_hydat(self) -> Path:
        """
        Legacy fallback: Extract from local HYDAT database.
        """
        import sqlite3
        station_id = self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID')
        hydat_path = self._get_config_value(lambda: self.config.evaluation.streamflow.hydat_path, dict_key='HYDAT_PATH')
        if hydat_path == 'default':
            hydat_path = str(self.project_dir.parent.parent / 'geospatial-data' / 'hydat' / 'Hydat.sqlite3')

        if not Path(hydat_path).exists():
            raise FileNotFoundError(f"HYDAT database not found at: {hydat_path}")

        self.logger.info(f"Extracting WSC data from HYDAT: {hydat_path}")

        conn = sqlite3.connect(hydat_path)
        query = "SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER = ?"
        df_raw = pd.read_sql_query(query, conn, params=(station_id,))
        conn.close()

        if df_raw.empty:
            raise DataAcquisitionError(f"No data for station {station_id} in HYDAT")

        # Reshape HYDAT format (FLOW1...FLOW31) to time series
        ts_data = []
        for _, row in df_raw.iterrows():
            year, month = int(row['YEAR']), int(row['MONTH'])
            for day in range(1, 32):
                col = f'FLOW{day}'
                if col in row and not pd.isna(row[col]):
                    try:
                        date = f"{year}-{month:02d}-{day:02d}"
                        ts_data.append({'datetime': date, 'discharge_cms': row[col]})
                    except ValueError:
                        continue # Invalid date (e.g., Feb 30)

        df = pd.DataFrame(ts_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"

        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled.to_csv(output_file, header=True, index_label='datetime')

        return output_file
