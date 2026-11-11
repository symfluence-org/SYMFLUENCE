# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mizuRoute time fix looks at every step, not only the first."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from symfluence.models.mizuroute.runner import MizuRouteRunner

pytestmark = pytest.mark.unit


def _runner(freq='h'):
    fake = SimpleNamespace(logger=logging.getLogger('t'), _get_time_rounding_freq=lambda: freq)
    fake._round_stamps = lambda stamps, f: MizuRouteRunner._round_stamps(fake, stamps, f)
    return fake


def test_jitter_after_an_exact_first_step_is_detected():
    stamps = pd.date_range('2004-01-01', periods=72, freq='h')
    jittered = stamps + pd.to_timedelta([0] + [13e-6] * 71, unit='s')
    fake = _runner()
    assert MizuRouteRunner._times_misaligned(fake, jittered)
    assert not MizuRouteRunner._times_misaligned(fake, stamps)


def test_sub_hourly_steps_round_to_seconds_not_hours():
    stamps = pd.date_range('2004-01-01', periods=8, freq='15min') + pd.Timedelta(microseconds=7)
    rounded = MizuRouteRunner._round_stamps(_runner(), stamps, 'h')
    assert not rounded.has_duplicates
    assert (rounded == pd.date_range('2004-01-01', periods=8, freq='15min')).all()


def test_rounding_disabled_never_flags():
    stamps = pd.date_range('2004-01-01', periods=3, freq='h') + pd.Timedelta(microseconds=5)
    assert not MizuRouteRunner._times_misaligned(_runner(freq=None), stamps)
