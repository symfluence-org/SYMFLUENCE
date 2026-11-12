# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Streamflow from a WISKI KiWebPortal share channel.

Select with ``STREAMFLOW_DATA_PROVIDER: KIWIS`` (or list ``kiwis_streamflow``
in ``ADDITIONAL_OBSERVATIONS``).  The channel URL and time series id come from
``KIWIS_CHANNEL_URL`` / ``KIWIS_TS_ID`` in the configuration or, when unset
there, from environment variables of the same names; they are never logged.

``acquire`` keeps a raw archive (``time, value, quality``, UTC) and on later
calls refetches only from ``KIWIS_REVISE_DAYS`` before its newest value, since
recent values may still be edited in WISKI.  ``process`` averages the values
into the model time step with :func:`symfluence.wiski.kiwis.interval_mean`
(coverage rule, no interpolation).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.registries import R
from symfluence.wiski import kiwis

from ..base import BaseObservationHandler


@R.observation_handlers.add('kiwis')
@R.observation_handlers.add('kiwis_streamflow')
class KiWISStreamflowHandler(BaseObservationHandler):
    """Discharge from a KiWebPortal share channel into the standard streamflow CSV."""

    obs_type = "streamflow"
    source_name = "KiWIS"
    SOURCE_INFO = {
        'source': 'WISKI KiWebPortal share channel',
        'url': 'configured (KIWIS_CHANNEL_URL)',
    }

    # ------------------------------------------------------------------ config
    def _setting(self, attr: str, key: str, default=None):
        return self._get_config_value(
            lambda: getattr(self.config.evaluation.kiwis, attr), default=default, dict_key=key)

    def _channel(self) -> Tuple[str, str]:
        channel = self._setting('channel_url', 'KIWIS_CHANNEL_URL') or os.environ.get('KIWIS_CHANNEL_URL')
        ts_id = self._setting('ts_id', 'KIWIS_TS_ID') or os.environ.get('KIWIS_TS_ID')
        if not channel or not ts_id:
            raise DataAcquisitionError(
                'KiWIS streamflow needs KIWIS_CHANNEL_URL and KIWIS_TS_ID (config or environment)')
        return str(channel), str(ts_id)

    def _raw_file(self, ts_id: str) -> Path:
        # A digest keeps distinct series apart without writing the id into file names
        tag = hashlib.sha1(ts_id.encode('utf-8')).hexdigest()[:10]
        raw_dir = self.project_observations_dir / "streamflow" / "raw_data"
        return raw_dir / f"kiwis_{tag}_raw.csv"

    # ------------------------------------------------------------------ acquire
    def acquire(self) -> Path:
        channel, ts_id = self._channel()
        raw_file = self._raw_file(ts_id)
        if self.start_date is None or pd.isna(self.start_date) or self.end_date is None or pd.isna(self.end_date):
            raise DataAcquisitionError('KiWIS streamflow needs EXPERIMENT_TIME_START and EXPERIMENT_TIME_END')
        # Through the end of the last day, plus an hour so a value stamped at midnight (the mean of the
        # minutes before it) is fetched whether or not the server treats 'to' as inclusive
        start = pd.Timestamp(self.start_date)
        end = (pd.Timestamp(self.end_date) + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=1)

        force = self._get_config_value(lambda: self.config.data.force_download, default=False,
                                       dict_key='FORCE_DOWNLOAD')
        old = None
        if raw_file.exists() and not force:
            old = read_raw(raw_file)
            if len(old):
                revise = pd.Timedelta(days=float(self._setting('revise_days', 'KIWIS_REVISE_DAYS', 3.0)))
                start = max(start, old.index.max() - revise)
            if start >= end:
                self.logger.info(f"Using existing KiWIS data: {raw_file}")
                return raw_file

        self.logger.info(f"Fetching KiWIS streamflow {start} .. {end}")
        new = update_archive(old, channel, ts_id, start, end,
                             timeout=float(self._setting('timeout', 'KIWIS_TIMEOUT', 180.0)),
                             retries=int(self._setting('retries', 'KIWIS_RETRIES', 2)))
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw_file.with_name(raw_file.name + '.tmp')
        new.rename_axis('time').to_csv(tmp)
        tmp.replace(raw_file)
        self.logger.info(f"KiWIS archive holds {len(new)} values: {raw_file}")
        return raw_file

    # ------------------------------------------------------------------ process
    def process(self, input_path: Path) -> Path:
        if not input_path.exists():
            raise FileNotFoundError(f"KiWIS raw data file not found: {input_path}")
        raw = read_raw(input_path)
        if raw.empty:
            raise DataAcquisitionError(f"No KiWIS values in {input_path}")
        freq = self._get_resample_freq()
        span = pd.Timedelta(days=1) if freq == 'D' else pd.Timedelta(freq if freq[0].isdigit() else f'1{freq}')
        label = str(self._setting('interval_label', 'KIWIS_INTERVAL_LABEL', 'start'))
        coverage = float(self._setting('min_coverage', 'KIWIS_MIN_COVERAGE', kiwis.DEFAULT_MIN_COVERAGE))
        q = kiwis.interval_mean(raw['value'], span, label=label, min_coverage=coverage)
        if self.start_date is not None and not pd.isna(self.start_date):
            q = q[q.index >= pd.Timestamp(self.start_date)]
        if self.end_date is not None and not pd.isna(self.end_date):
            q = q[q.index <= pd.Timestamp(self.end_date)]
        q = q.rename('discharge_cms')

        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        q.to_csv(output_file, header=True, index_label='datetime')
        self.logger.info(f"KiWIS streamflow processing complete: {output_file} "
                         f"({int(q.notna().sum())} of {len(q)} intervals with data)")
        return output_file


def read_raw(path: Path) -> pd.DataFrame:
    """Read a raw archive written by :meth:`KiWISStreamflowHandler.acquire`."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index, name='time')
    return df


def update_archive(old: Optional[pd.DataFrame], channel: str, ts_id: str, start: pd.Timestamp,
                   end: pd.Timestamp, **fetch_kwargs) -> pd.DataFrame:
    """Fetch [start, end) year by year and replace that window of ``old`` with it."""
    parts = []
    for year in range(start.year, end.year + 1):
        a, b = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year + 1, 1, 1))
        if a < b:
            parts.append(kiwis.fetch(channel, ts_id, a, b, **fetch_kwargs))
    new = pd.concat(parts) if parts else kiwis.parse_payload({'columns': 'timestamp,value', 'data': []})
    if old is not None and len(old):
        new = pd.concat([old[old.index < start], new, old[old.index >= end]])
    new = new[~new.index.duplicated(keep='last')].sort_index()
    return new
