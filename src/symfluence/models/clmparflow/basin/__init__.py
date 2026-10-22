# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow-CLM basin mode: a regular grid over a real basin, chained water-year segments with spin-up.

Opt-in (``CLMPARFLOW_BASIN_MODE: true``). The preprocessor builds the grid (``grid.build_basin_grid``); the
runner runs the chain (``chain.run_chain``), writing each segment's inputs (``case``) and summarising its
outlet discharge, storage and glacier ledger. Glacier water and the runtime ice albedo need the patched
build (``symfluence binary install clmparflow --patched``).
"""
from __future__ import annotations

from .case import BasinParameters, parflow_keys, run_segment, write_segment
from .chain import ChainSettings, glacier_mass_balance, run_chain, summarise_segment
from .grid import BasinGrid, build_basin_grid

__all__ = ['BasinGrid', 'BasinParameters', 'ChainSettings', 'build_basin_grid', 'glacier_mass_balance',
           'parflow_keys', 'run_chain', 'run_segment', 'summarise_segment', 'write_segment']
