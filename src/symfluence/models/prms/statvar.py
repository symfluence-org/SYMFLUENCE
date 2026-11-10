# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared PRMS / GSFLOW ``statvar`` reader.

One parser for the calibration workers, postprocessors and extractors of
PRMS and GSFLOW so that calibration and the final ``results`` CSV read the
same column in the same units.

statvar text format::

    N                       <- number of output variables
    basin_cfs 1             <- name and element count, N lines
    seg_outflow 3
    ...
    1 2000 1 1 0 0 0 v1 v2 v3 v4 ...   <- step y m d h min s, then
                                          sum(counts) values in header order

A variable's columns therefore start at ``7 + sum(counts of the variables
before it)``, not at ``7 + its index``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ModelExecutionError

logger = logging.getLogger(__name__)

#: cubic feet per second -> cubic metres per second
CFS_TO_CMS = 0.028316846592

#: Streamflow variables in order of preference and whether they are in cfs.
#: ``basin_cfs`` is the basin outlet flow PRMS reports directly (what the
#: SYMFLUENCE preprocessors write first); ``seg_outflow`` is per segment.
STREAMFLOW_VARIABLES: Tuple[Tuple[str, bool], ...] = (
    ('basin_cfs', True),
    ('seg_outflow', True),
    ('basin_cms', False),
    ('streamflow_cfs', True),
    ('streamflow', False),
)

_CFS_UNITS = {'cfs', 'ft3 s-1', 'ft3/s', 'ft^3/s', 'cubic feet per second'}


def parse_statvar(path: Path) -> Tuple[List[Tuple[str, int]], pd.DataFrame]:
    """Parse a statvar file into its header and a dated value table.

    Returns:
        ``(header, df)`` where ``header`` is ``[(name, count), ...]`` and
        ``df`` has a DatetimeIndex and one column per element, named
        ``name`` for scalars and ``name_1..name_n`` for arrays.

    Raises:
        ModelExecutionError: if the file has no parseable header or data rows.
    """
    lines = Path(path).read_text(encoding='utf-8').strip().split('\n')
    if not lines:
        raise ModelExecutionError(f"Empty statvar file: {path}")
    try:
        n_vars = int(lines[0].split()[0])
    except (ValueError, IndexError) as e:
        raise ModelExecutionError(f"statvar {path} has no variable-count header line") from e

    header: List[Tuple[str, int]] = []
    for line in lines[1:n_vars + 1]:
        parts = line.split()
        if not parts:
            continue
        count = 1
        if len(parts) > 1:
            try:
                count = int(parts[1])
            except ValueError:
                count = 1
        header.append((parts[0], max(count, 1)))
    if len(header) != n_vars:
        raise ModelExecutionError(f"statvar {path}: expected {n_vars} header entries, found {len(header)}")

    columns: List[str] = []
    for name, count in header:
        columns.extend([name] if count == 1 else [f"{name}_{i + 1}" for i in range(count)])
    n_values = len(columns)

    dates, rows = [], []
    for line in lines[n_vars + 1:]:
        parts = line.split()
        if len(parts) < 7 + n_values:
            continue
        try:
            stamp = pd.Timestamp(
                year=int(parts[1]), month=int(parts[2]), day=int(parts[3]),
                hour=int(float(parts[4])), minute=int(float(parts[5])),
            )
            values = [float(v) for v in parts[7:7 + n_values]]
        except (ValueError, IndexError):
            continue
        dates.append(stamp)
        rows.append(values)
    if not rows:
        raise ModelExecutionError(f"statvar {path} contains no data rows")
    df = pd.DataFrame(np.asarray(rows, dtype=float), index=pd.DatetimeIndex(dates), columns=columns)
    return header, df


def _select_element(df: pd.DataFrame, name: str, count: int,
                    element: Optional[int]) -> pd.Series:
    if count == 1:
        return df[name]
    if element is None:
        logger.warning(
            f"statvar variable {name} has {count} elements and no element was "
            f"specified; using the last one ({name}_{count})"
        )
        element = count
    return df[f"{name}_{element}"]


def read_statvar_streamflow(
    path: Path,
    preferred: Sequence[str] = (),
    element: Optional[int] = None,
) -> pd.Series:
    """Basin streamflow (m3/s) from a statvar file.

    Picks the first available variable from ``preferred`` then
    :data:`STREAMFLOW_VARIABLES`, finds its real column from the header
    element counts and converts cfs to m3/s.

    Raises:
        ModelExecutionError: if no streamflow variable is present.
    """
    header, df = parse_statvar(path)
    counts = dict(header)
    cfs = dict(STREAMFLOW_VARIABLES)
    for name in [*preferred, *(n for n, _ in STREAMFLOW_VARIABLES)]:
        if name in counts:
            series = _select_element(df, name, counts[name], element)
            if cfs.get(name, name.endswith('_cfs')):
                series = series * CFS_TO_CMS
            return series.rename('discharge_cms')
    raise ModelExecutionError(
        f"No streamflow variable in statvar {path}; header has {[n for n, _ in header]}"
    )


def read_netcdf_streamflow(path: Path, element: Optional[int] = None) -> pd.Series:
    """Basin streamflow (m3/s) from PRMS NetCDF output (cfs converted)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        for name, default_cfs in STREAMFLOW_VARIABLES:
            if name not in ds.data_vars:
                continue
            flow = ds[name]
            extra = [d for d in flow.dims if d != 'time']
            for dim in extra:
                idx = (element - 1) if element is not None else -1
                flow = flow.isel({dim: idx})
            units = str(flow.attrs.get('units', '')).strip().lower()
            is_cfs = units in _CFS_UNITS if units else default_cfs
            series = flow.to_series()
            if is_cfs:
                series = series * CFS_TO_CMS
            return series.rename('discharge_cms')
    raise ModelExecutionError(f"No streamflow variable in PRMS NetCDF output {path}")
