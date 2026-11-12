# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""KiWebPortal share-channel client and interval means (synthetic payloads only)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import requests

from symfluence.core.exceptions import DataAcquisitionError, ValidationError
from symfluence.wiski import kiwis

pytestmark = pytest.mark.unit

CHANNEL = 'https://example.invalid/KiWebPortal/rest/shares/channels/abc'


def _payload(rows, columns='timestamp,value,quality'):
    return {'columns': columns, 'data': rows}


def test_data_url():
    url = kiwis.data_url(CHANNEL + '/', 'TS1', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02 06:00'))
    assert url == (CHANNEL + '/timeSeries/TS1/data?from=2026-01-01T00:00:00&to=2026-01-02T06:00:00'
                   '&addGaps=false&columns=timestamp,value,quality')
    with pytest.raises(ValidationError):
        kiwis.data_url('ftp://x', 'TS1', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02'))


def test_parse_iso8601_variants_quality_and_duplicates():
    df = kiwis.parse_payload(_payload([
        ['2026-01-01T00:10:00.000Z', 11.0, 200],
        ['2026-01-01T00:05:00+00:00', 10.0, 200],
        ['2026-01-01T00:15:07.250Z', None, 200],          # no value: dropped
        ['2026-01-01T00:20:03Z', '12.5', 40],              # odd seconds, string value
        ['2026-01-01T01:25:00+01:00', 13.0, None],         # offset -> 00:25 UTC
        ['2026-01-01T00:10:00Z', 11.5, 200],               # duplicate stamp: last wins
    ]))
    assert list(df.index) == [pd.Timestamp('2026-01-01 00:05'), pd.Timestamp('2026-01-01 00:10'),
                              pd.Timestamp('2026-01-01 00:20:03'), pd.Timestamp('2026-01-01 00:25')]
    assert df.index.tz is None
    np.testing.assert_allclose(df['value'], [10.0, 11.5, 12.5, 13.0])
    np.testing.assert_allclose(df['quality'], [200, 200, 40, np.nan])


def test_parse_list_wrapped_empty_and_no_quality():
    assert kiwis.parse_payload([]).empty
    assert kiwis.parse_payload(_payload([])).empty
    df = kiwis.parse_payload([_payload([['2026-01-01T00:05:00Z', 1.0]], 'timestamp,value')])
    assert len(df) == 1 and np.isnan(df['quality'].iloc[0])
    with pytest.raises(DataAcquisitionError):
        kiwis.parse_payload({'data': []})


class _Response:
    def __init__(self, payload=None, status=200, bad_json=False):
        self.payload, self.status, self.bad_json = payload, status, bad_json

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f'{self.status}')

    def json(self):
        if self.bad_json:
            raise ValueError('not json')
        return self.payload


def test_fetch_retries_then_succeeds():
    responses = [requests.ConnectionError('down'), _Response(status=503), _Response(bad_json=True),
                 _Response(_payload([['2026-01-01T00:05:00Z', 1.0, 200]]))]
    seen, waits = [], []

    def get(url, **kwargs):
        seen.append((url, kwargs))
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    df = kiwis.fetch(CHANNEL, 'TS1', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02'),
                     retries=3, backoff=1.0, get=get, sleep=waits.append)
    assert len(df) == 1 and len(seen) == 4 and waits == [1.0, 2.0, 3.0]
    assert seen[0][1]['headers'] == {'Accept': 'application/json'}


def test_fetch_gives_up_without_leaking_the_url(caplog):
    def get(url, **kwargs):
        raise requests.ConnectionError(f'cannot reach {url}')

    with caplog.at_level(logging.WARNING), pytest.raises(DataAcquisitionError) as err:
        kiwis.fetch(CHANNEL, 'TS1', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02'),
                    retries=1, get=get, sleep=lambda s: None)
    assert 'example.invalid' not in str(err.value)
    assert 'example.invalid' not in caplog.text


def _five_min(start, end, value=1.0):
    t = pd.date_range(start, end, freq='5min')
    return pd.Series(value, index=t, dtype=float)


def test_interval_mean_end_and_start_labels():
    # samples stamped s cover (s-5min, s]; values = hour-of-sample so the means are distinct
    s = _five_min('2026-01-01 00:05', '2026-01-01 06:00')
    s[:] = np.arange(len(s), dtype=float)
    end = kiwis.interval_mean(s, '3h', label='end')
    start = kiwis.interval_mean(s, '3h', label='start')
    # (00:00, 03:00] holds samples 00:05..03:00 = indices 0..35
    assert end[pd.Timestamp('2026-01-01 03:00')] == pytest.approx(np.arange(36).mean())
    assert start[pd.Timestamp('2026-01-01 00:00')] == pytest.approx(np.arange(36).mean())
    assert end[pd.Timestamp('2026-01-01 06:00')] == pytest.approx(np.arange(36, 72).mean())


def test_interval_mean_daily_coverage_rule():
    s = _five_min('2026-01-01 00:05', '2026-01-03 00:00', 2.0)
    s = s.drop(s.index[(s.index > '2026-01-02 00:00') & (s.index <= '2026-01-02 06:00')])  # 75 % of day 2
    day = kiwis.interval_mean(s, '1D', label='start', min_coverage=0.8)
    assert day[pd.Timestamp('2026-01-01')] == 2.0
    assert np.isnan(day[pd.Timestamp('2026-01-02')])
    day = kiwis.interval_mean(s, '1D', label='start', min_coverage=0.7)
    assert day[pd.Timestamp('2026-01-02')] == 2.0


def test_interval_mean_reindex_and_empty():
    idx = pd.date_range('2026-01-01', periods=3, freq='1D')
    out = kiwis.interval_mean(pd.Series(dtype=float), '1D', index=idx)
    assert out.isna().all() and (out.index == idx).all()
    with pytest.raises(ValidationError):
        kiwis.interval_mean(_five_min('2026-01-01', '2026-01-02'), '3h', label='middle')
