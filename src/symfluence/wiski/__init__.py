# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Data interchange with KISTERS WISKI.

- :mod:`.zrxp`: write (and read back) ZRXP files for WISKI import, e.g.
  discharge forecasts with their issue time.
- :mod:`.kiwis`: read time series from a KiWebPortal share channel
  (KiWIS REST), and average sub-hourly values into model intervals.

The streamflow observation handler ``kiwis_streamflow``
(``STREAMFLOW_DATA_PROVIDER: KIWIS``) is built on :mod:`.kiwis`.
"""
from __future__ import annotations

from .kiwis import data_url, fetch, interval_mean, parse_payload
from .zrxp import ZrxpBlock, read_zrxp, write_zrxp

__all__ = [
    "ZrxpBlock",
    "data_url",
    "fetch",
    "interval_mean",
    "parse_payload",
    "read_zrxp",
    "write_zrxp",
]
