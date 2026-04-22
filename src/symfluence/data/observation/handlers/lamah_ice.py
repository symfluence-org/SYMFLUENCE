# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LamaH-ICE Observation Handlers

Provides handlers for LamaH-ICE (Iceland) streamflow data from local files.
"""
from pathlib import Path

import pandas as pd

from symfluence.core.exceptions import DataAcquisitionError

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry


@ObservationRegistry.register('lamah_ice_streamflow')
class LamahIceStreamflowHandler(BaseObservationHandler):
    """
    Handles LamaH-ICE streamflow data processing from a local dataset.
    """

    obs_type = "streamflow"
    source_name = "LAMAH_ICE"
    SOURCE_INFO = {
        'source': 'LamaH-Ice',
        'source_doi': '10.5281/zenodo.4683950',
        'url': 'https://zenodo.org/record/4683950',
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
            raise ValueError("LAMAH_ICE_PATH required for LAMAH_ICE acquisition")

        lamah_path = Path(lamah_path_str)
        # Standard LamaH-ICE structure: D_gauges/2_timeseries/daily/ID_{station_id}.csv
        raw_file = lamah_path / "D_gauges" / "2_timeseries" / "daily" / f"ID_{station_id}.csv"

        if not raw_file.exists():
            self.logger.error(f"LamaH-ICE file not found at {raw_file}")
            raise FileNotFoundError(f"LamaH-ICE file not found: {raw_file}")

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
