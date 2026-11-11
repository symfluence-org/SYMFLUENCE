# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Forcing time window for the daily PRMS / GSFLOW inputs."""
from __future__ import annotations

from datetime import datetime
from typing import Tuple

import pandas as pd


def daily_forcing_window(start_date: datetime, end_date: datetime) -> Tuple[str, str]:
    """Label bounds selecting sub-daily forcing for a run from start to end day.

    PRMS runs whole days up to and including the end day (``end_time`` is
    written as year, month, day). The forcing aggregated to daily values must
    therefore cover the whole end day: slicing to ``end_date`` itself keeps a
    single timestep of that day when the configured end has no time of day
    (precipitation ~1/24 of the day's total, tmax = tmin).
    """
    last_instant = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return str(start_date), str(last_instant)
