# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ZRXP writer for WISKI import."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ValidationError
from symfluence.wiski.zrxp import read_zrxp, write_zrxp

pytestmark = pytest.mark.unit


def _q() -> pd.Series:
    t = pd.date_range('2026-01-01 03:00', periods=4, freq='3h')
    return pd.Series([12.345, np.nan, 13.0, 14.999], index=t, name='q')


def test_forecast_layout_bytes(tmp_path):
    p = write_zrxp(tmp_path / 'out' / 'q.zrx', _q(), 'EX1', issued=pd.Timestamp('2026-01-01 00:00'),
                   station='Gauge A', name='model_x')
    assert p.read_bytes() == (
        b'#ZRXPVERSION3014.03|*|ZRXPCREATORsymfluence|*|\n'
        b'#REXCHANGEEX1|*|SNAMEGauge A|*|CNAMEmodel_x|*|CUNITm3/s|*|RINVAL-777|*|TZUTC|*|\n'
        b'#LAYOUT(timestamp,forecast,value)|*|\n'
        b'20260101030000 20260101000000 12.35\n'
        b'20260101060000 20260101000000 -777\n'
        b'20260101090000 20260101000000 13.00\n'
        b'20260101120000 20260101000000 15.00\n'
    )
    assert not list(tmp_path.glob('out/*.tmp'))


def test_roundtrip_without_issue_time(tmp_path):
    p = write_zrxp(tmp_path / 'q.zrx', _q(), 'EX1', decimals=3)
    block = read_zrxp(p)
    assert block.header['REXCHANGE'] == 'EX1' and block.header['RINVAL'] == '-777'
    assert block.forecast is None
    np.testing.assert_allclose(block.values.to_numpy(), [12.345, np.nan, 13.0, 14.999])
    assert (block.values.index == _q().index).all()


def test_roundtrip_with_issue_time(tmp_path):
    p = write_zrxp(tmp_path / 'q.zrx', _q(), 'EX1', issued=pd.Timestamp('2026-01-01'))
    block = read_zrxp(p)
    assert (block.forecast == pd.Timestamp('2026-01-01')).all()


def test_tz_aware_index_is_written_in_utc(tmp_path):
    q = _q()
    q.index = q.index.tz_localize('UTC').tz_convert('Etc/GMT-1')     # UTC+1
    p = write_zrxp(tmp_path / 'q.zrx', q, 'EX1')
    assert read_zrxp(p).values.index[0] == pd.Timestamp('2026-01-01 03:00')


@pytest.mark.parametrize('series', [
    pd.Series([1.0, 2.0], index=pd.DatetimeIndex(['2026-01-02', '2026-01-01'])),
    pd.Series([1.0, 2.0], index=pd.DatetimeIndex(['2026-01-01', '2026-01-01'])),
    pd.Series([1.0, 2.0]),
])
def test_rejects_unordered_or_non_time_index(tmp_path, series):
    with pytest.raises(ValidationError):
        write_zrxp(tmp_path / 'q.zrx', series, 'EX1')


def test_rejects_header_separators(tmp_path):
    with pytest.raises(ValidationError):
        write_zrxp(tmp_path / 'q.zrx', _q(), 'EX|*|1')
