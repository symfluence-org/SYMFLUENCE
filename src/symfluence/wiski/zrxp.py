# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ZRXP export of time series (e.g. discharge forecasts) for WISKI import.

ZRXP is the ASCII exchange format of KISTERS WISKI.  A file is one or more
blocks; each block starts with header lines of ``#KEYvalue|*|`` pairs and is
followed by one data row per time step.  This module writes one block per
file, with:

* ``#ZRXPVERSION`` / ``#ZRXPCREATOR``
* ``#REXCHANGE`` (the exchange id WISKI's import maps to a time series),
  optional ``#SNAME`` (station) and ``#CNAME`` (parameter / member name),
  ``#CUNIT``, ``#RINVAL`` (the value written for missing data) and ``#TZ``
* ``#LAYOUT(timestamp,value)`` or, when an issue time is given,
  ``#LAYOUT(timestamp,forecast,value)`` so that forecast runs stay apart by
  issue time if the import uses that column.

Timestamps are ``yyyymmddHHMMSS``.  The exchange id and the names are the
receiving system's configuration and are passed in by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from symfluence.core.exceptions import ValidationError

ZRXP_VERSION = '3014.03'
TIME_FORMAT = '%Y%m%d%H%M%S'
DEFAULT_INVALID = -777.0
_FORBIDDEN = ('|*|', '\n', '\r', ';')


@dataclass(frozen=True)
class ZrxpBlock:
    """One ZRXP block as read back: header keywords and the series."""

    header: Dict[str, str]
    values: pd.Series
    forecast: Optional[pd.Series] = None


def _naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert('UTC').tz_localize(None) if index.tz is not None else index


def _check_header_value(key: str, value: str) -> str:
    text = str(value)
    if not text or any(f in text for f in _FORBIDDEN):
        raise ValidationError(f"ZRXP header {key} must be non-empty without '|*|', ';' or line breaks: {text!r}")
    return text


def _format_value(v: float, invalid: float, decimals: int) -> str:
    if v is None or not math.isfinite(v):
        return f'{invalid:g}'
    return f'{round(float(v), decimals):.{decimals}f}'


def write_zrxp(
    path: Union[str, Path],
    series: pd.Series,
    exchange_id: str,
    *,
    issued: Optional[pd.Timestamp] = None,
    station: Optional[str] = None,
    name: Optional[str] = None,
    unit: str = 'm3/s',
    invalid: float = DEFAULT_INVALID,
    timezone: str = 'UTC',
    decimals: int = 2,
    creator: str = 'symfluence',
) -> Path:
    """Write ``series`` as one ZRXP block, atomically (temporary file + rename).

    Args:
        path: Output file.
        series: Values indexed by time; strictly increasing, no duplicates.
            A tz-aware index is converted to UTC (pass ``timezone='UTC'``);
            a naive index is written as is and labelled with ``timezone``.
        exchange_id: ``REXCHANGE`` id the WISKI import maps to a series.
        issued: Forecast issue time; adds the ``forecast`` column.
        station: ``SNAME`` (optional).
        name: ``CNAME`` (optional), e.g. model or ensemble member.
        unit: ``CUNIT``.
        invalid: ``RINVAL`` value written for NaN / inf.
        timezone: ``TZ`` label, e.g. ``UTC`` or ``UTC+1``.
        decimals: Decimals of the written values.
        creator: ``ZRXPCREATOR``.

    Returns:
        ``path``.
    """
    path = Path(path)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValidationError('ZRXP series needs a DatetimeIndex')
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValidationError('ZRXP series must be strictly increasing in time')
    index = _naive_utc(series.index)

    head = [('ZRXPVERSION', ZRXP_VERSION), ('ZRXPCREATOR', creator)]
    meta = [('REXCHANGE', exchange_id)]
    if station:
        meta.append(('SNAME', station))
    if name:
        meta.append(('CNAME', name))
    meta += [('CUNIT', unit), ('RINVAL', f'{invalid:g}'), ('TZ', timezone)]
    lines = [
        '#' + ''.join(f'{k}{_check_header_value(k, v)}|*|' for k, v in head),
        '#' + ''.join(f'{k}{_check_header_value(k, v)}|*|' for k, v in meta),
    ]
    if issued is not None:
        issued_ts = pd.Timestamp(issued)
        if issued_ts.tzinfo is not None:
            issued_ts = issued_ts.tz_convert('UTC').tz_localize(None)
        stamp = issued_ts.strftime(TIME_FORMAT)
        lines.append('#LAYOUT(timestamp,forecast,value)|*|')
        lines += [f'{t.strftime(TIME_FORMAT)} {stamp} {_format_value(v, invalid, decimals)}'
                  for t, v in zip(index, series.to_numpy(dtype=float))]
    else:
        lines.append('#LAYOUT(timestamp,value)|*|')
        lines += [f'{t.strftime(TIME_FORMAT)} {_format_value(v, invalid, decimals)}'
                  for t, v in zip(index, series.to_numpy(dtype=float))]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    # ZRXP is ISO-8859-1; '\n' line ends on every platform
    with open(tmp, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    tmp.replace(path)
    return path


def read_zrxp(path: Union[str, Path]) -> ZrxpBlock:
    """Read a single-block ZRXP file as written by :func:`write_zrxp`.

    ``RINVAL`` values become NaN.  Supports the ``(timestamp,value)`` and
    ``(timestamp,forecast,value)`` layouts.
    """
    header: Dict[str, str] = {}
    layout = ['timestamp', 'value']
    rows = []
    with open(path, encoding='latin-1') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                if line.startswith('#LAYOUT('):
                    layout = [c.strip() for c in line[len('#LAYOUT('):line.index(')')].split(',')]
                    continue
                for part in line[1:].split('|*|'):
                    if not part:
                        continue
                    for key in ('ZRXPVERSION', 'ZRXPCREATOR', 'REXCHANGE', 'SNAME', 'CNAME', 'CUNIT',
                                'RINVAL', 'TZ'):
                        if part.startswith(key):
                            header[key] = part[len(key):]
                            break
                continue
            rows.append(line.split())
    frame = pd.DataFrame(rows, columns=layout)
    index = pd.DatetimeIndex(pd.to_datetime(frame['timestamp'], format=TIME_FORMAT), name='time')
    values = pd.to_numeric(frame['value']).to_numpy(dtype=float)
    if 'RINVAL' in header:
        values[values == float(header['RINVAL'])] = float('nan')
    forecast = None
    if 'forecast' in frame:
        forecast = pd.Series(pd.to_datetime(frame['forecast'], format=TIME_FORMAT).to_numpy(), index=index)
    return ZrxpBlock(header=header, values=pd.Series(values, index=index, name='value'), forecast=forecast)
