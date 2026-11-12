# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Read time series from a WISKI KiWebPortal share channel (KiWIS REST).

A KiWebPortal *share channel* exposes selected WISKI time series over HTTP
to anyone holding the channel URL::

    <channel>/timeSeries/<ts id>/data?from=...&to=...&addGaps=false&columns=timestamp,value,quality

and answers with JSON ``{"columns": "timestamp,value,quality", "data": [[...], ...]}``
(some servers wrap it in a one-element list, as KiWIS ``getTimeseriesValues``
does).  Because the URL itself grants access, it should be kept out of code
and version control: pass it from configuration or an environment variable.

Timestamps are ISO 8601 with an offset; a few carry odd seconds or
milliseconds.  They are returned as a naive UTC index.  The ``quality``
column is WISKI's quality code, returned as is (nothing is filtered here).

:func:`interval_mean` turns the (typically 5- or 15-minute) values into model
interval means with a coverage rule, for start- or end-stamped intervals.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = ('timestamp', 'value', 'quality')
DEFAULT_MIN_COVERAGE = 0.8
_QUERY_TIME = '%Y-%m-%dT%H:%M:%S'


def data_url(channel: str, ts_id: str, start: pd.Timestamp, end: pd.Timestamp,
             columns: Sequence[str] = DEFAULT_COLUMNS) -> str:
    """Data request URL for ``ts_id`` over [start, end) (naive times are UTC)."""
    if not channel or not str(channel).lower().startswith(('http://', 'https://')):
        raise ValidationError('KiWIS channel must be an http(s) URL')
    if not ts_id:
        raise ValidationError('KiWIS time series id is empty')
    a, b = (_naive_utc(pd.Timestamp(t)) for t in (start, end))
    return (f"{str(channel).rstrip('/')}/timeSeries/{ts_id}/data"
            f"?from={a.strftime(_QUERY_TIME)}&to={b.strftime(_QUERY_TIME)}"
            f"&addGaps=false&columns={','.join(columns)}")


def _naive_utc(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_convert('UTC').tz_localize(None) if t.tzinfo is not None else t


def _empty() -> pd.DataFrame:
    return pd.DataFrame({'value': pd.Series(dtype=float), 'quality': pd.Series(dtype=float)},
                        index=pd.DatetimeIndex([], name='time'))


def parse_payload(payload: Union[dict, list]) -> pd.DataFrame:
    """Values of a channel response as a frame [value, quality] on a naive UTC index.

    Rows without a numeric value are dropped; duplicate stamps keep the last
    value; the index is sorted.  ``quality`` is NaN when the column is absent.
    """
    if isinstance(payload, list):
        if not payload:
            return _empty()
        if len(payload) != 1:
            raise DataAcquisitionError(f'KiWIS response holds {len(payload)} series, expected one')
    body: Any = payload[0] if isinstance(payload, list) else payload
    try:
        columns = [c.strip() for c in str(body['columns']).split(',')]
        data = body.get('data') or []
    except (KeyError, TypeError, AttributeError) as e:
        raise DataAcquisitionError(f'unexpected KiWIS response: {e}') from e
    if not data:
        return _empty()
    df = pd.DataFrame(data, columns=columns)
    if 'timestamp' not in df or 'value' not in df:
        raise DataAcquisitionError(f'KiWIS response lacks timestamp/value columns: {columns}')
    # ISO 8601 with offsets; some stamps carry odd seconds or milliseconds
    stamps = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True).dt.tz_convert(None)
    quality = (pd.to_numeric(df['quality'], errors='coerce').to_numpy(dtype=float)
               if 'quality' in df else np.full(len(df), np.nan))
    out = pd.DataFrame({'value': pd.to_numeric(df['value'], errors='coerce').to_numpy(dtype=float),
                        'quality': quality},
                       index=pd.DatetimeIndex(stamps, name='time'))
    out = out[out['value'].notna()]
    return out[~out.index.duplicated(keep='last')].sort_index()


def fetch(
    channel: str,
    ts_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    timeout: float = 180,
    retries: int = 2,
    backoff: float = 10.0,
    get: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Values in [start, end) as a frame [value, quality] (see :func:`parse_payload`).

    Network errors, HTTP errors and malformed JSON are retried ``retries``
    times, waiting ``backoff * attempt`` seconds; the last error is raised as
    :class:`DataAcquisitionError`.  The channel URL is not included in log
    messages or errors (it grants access).
    """
    url = data_url(channel, ts_id, start, end)
    get = get or requests.get
    attempts = 1 + max(0, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            response = get(url, headers={'Accept': 'application/json'}, timeout=timeout)
            response.raise_for_status()
            return parse_payload(response.json())
        except (requests.RequestException, ValueError) as e:
            if attempt == attempts:
                raise DataAcquisitionError(
                    f'KiWIS request for {pd.Timestamp(start)}..{pd.Timestamp(end)} failed after '
                    f'{attempts} attempt(s): {type(e).__name__}') from e
            logger.warning('KiWIS request failed (%s), attempt %d of %d; retrying',
                           type(e).__name__, attempt, attempts)
            sleep(backoff * attempt)
    raise AssertionError('unreachable')


def interval_mean(
    samples: pd.Series,
    freq: Union[str, pd.Timedelta],
    *,
    label: str = 'start',
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    sample_step: Optional[Union[str, pd.Timedelta]] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """Mean of sub-interval samples over each interval of length ``freq``.

    A sample stamped ``s`` is taken as the mean over ``(s - sample_step, s]``
    (how WISKI stamps interval means), so an interval covers the samples
    stamped in ``(t0, t0 + freq]``.  ``label='start'`` stamps that interval
    ``t0`` (``[t, t + freq)``, pandas' usual label); ``label='end'`` stamps it
    ``t0 + freq`` (``(t - freq, t]``, how most models stamp their output).
    Intervals align to the Unix epoch, so daily intervals start at 00:00.

    Intervals holding fewer than ``min_coverage`` of the expected number of
    samples are NaN; nothing is interpolated.

    Args:
        samples: Sub-interval values on a naive (UTC) DatetimeIndex.
        freq: Interval length, e.g. ``'3h'`` or ``'1D'``.
        label: ``'start'`` or ``'end'``.
        min_coverage: Share of expected samples needed for a mean (0-1).
        sample_step: Sample spacing; defaults to the median spacing.
        index: Optional target stamps to reindex the result onto.
    """
    if label not in ('start', 'end'):
        raise ValidationError(f"label must be 'start' or 'end', got {label!r}")
    span = pd.Timedelta(freq)
    clean = samples.dropna().sort_index()
    clean = clean[~clean.index.duplicated(keep='last')]
    if len(clean) == 0:
        result = pd.Series(np.nan, index=index if index is not None else pd.DatetimeIndex([]), dtype=float)
        result.name = samples.name
        return result
    if sample_step is None:
        if len(clean) < 2:
            raise ValidationError('sample_step is needed with fewer than two samples')
        step = pd.Timedelta(np.median(np.diff(clean.index.asi8)), unit='ns')
    else:
        step = pd.Timedelta(sample_step)
    if step <= pd.Timedelta(0) or step > span:
        raise ValidationError(f'sample step {step} must be positive and at most the interval {span}')
    grouped = clean.resample(span, closed='right', label='right' if label == 'end' else 'left', origin='epoch')
    mean, count = grouped.mean(), grouped.count()
    mean[count < min_coverage * (span / step)] = np.nan
    if index is not None:
        mean = mean.reindex(index)
    mean.name = samples.name
    return mean
