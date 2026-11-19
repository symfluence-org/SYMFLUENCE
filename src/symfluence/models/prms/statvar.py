# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared PRMS / GSFLOW ``statvar`` reader.

One parser for the calibration workers, postprocessors, extractors and the
water-balance / cryosphere checks of PRMS and GSFLOW, so they all read the
same column in the same units.

statvar text format (PRMS ``statVar_names`` / ``statVar_element``)::

    N                       <- number of statistic entries
    basin_cfs 1             <- variable name and ONE element index, N lines
    hru_actet 1
    hru_actet 2             <- an HRU array appears once per element listed
    ...
    1 2000 1 1 0 0 0 v1 v2 ...   <- step y m d h min s, then exactly N values,
                                    one per header line, in header order

The second header token is the element index, not an element count: a
variable output for N HRUs contributes N header lines and N columns.
Every column is named ``<name>_<element>`` (``basin_cfs_1``,
``hru_actet_2``).
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


def column_name(name: str, element: int) -> str:
    """Column label of one statvar entry: ``<name>_<element>``."""
    return f"{name}_{element}"


def parse_statvar(path: Path) -> Tuple[List[Tuple[str, int]], pd.DataFrame]:
    """Parse a statvar file into its header and a dated value table.

    Returns:
        ``(header, df)`` where ``header`` is ``[(name, element), ...]`` in
        file order (one entry per value column) and ``df`` has a
        DatetimeIndex and one column per entry named ``<name>_<element>``.

    Raises:
        ModelExecutionError: if the header is malformed or repeats an entry,
            or the file has no complete data rows.
    """
    lines = Path(path).read_text(encoding='utf-8').strip().splitlines()
    if not lines:
        raise ModelExecutionError(f"Empty statvar file: {path}")
    try:
        n_vars = int(lines[0].split()[0])
    except (ValueError, IndexError) as e:
        raise ModelExecutionError(f"statvar {path} has no entry-count header line") from e

    header: List[Tuple[str, int]] = []
    for line in lines[1:n_vars + 1]:
        parts = line.split()
        if len(parts) == 1:
            header.append((parts[0], 1))
            continue
        try:
            header.append((parts[0], int(parts[1])))
        except (ValueError, IndexError) as e:
            raise ModelExecutionError(
                f"statvar {path}: header line {line!r} is not '<name> <element>'"
            ) from e
    if len(header) != n_vars:
        raise ModelExecutionError(f"statvar {path}: expected {n_vars} header entries, found {len(header)}")

    columns = [column_name(name, element) for name, element in header]
    if len(set(columns)) != len(columns):
        repeated = sorted({c for c in columns if columns.count(c) > 1})
        raise ModelExecutionError(f"statvar {path} lists entries more than once: {repeated}")

    dates, rows, skipped = [], [], 0
    for line in lines[n_vars + 1:]:
        parts = line.split()
        if len(parts) != 7 + n_vars:
            skipped += 1
            continue
        try:
            year, month, day, hour, minute = (int(float(t)) for t in parts[1:6])
            stamp = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
            values = [float(v) for v in parts[7:]]
        except (ValueError, IndexError):
            skipped += 1
            continue
        dates.append(stamp)
        rows.append(values)
    if not rows:
        raise ModelExecutionError(
            f"statvar {path} contains no data rows with {n_vars} values"
        )
    if skipped:
        logger.warning(f"statvar {path}: skipped {skipped} incomplete or malformed rows")
    df = pd.DataFrame(np.asarray(rows, dtype=float), index=pd.DatetimeIndex(dates), columns=columns)
    return header, df


def variable_elements(header: Sequence[Tuple[str, int]], name: str) -> List[int]:
    """Element indices of ``name`` present in the file, in file order."""
    return [element for n, element in header if n == name]


def variable_frame(header: Sequence[Tuple[str, int]], df: pd.DataFrame, name: str) -> pd.DataFrame:
    """All columns of ``name`` (one per listed element), in file order.

    Raises:
        ModelExecutionError: if ``name`` is not in the file.
    """
    elements = variable_elements(header, name)
    if not elements:
        raise ModelExecutionError(f"statvar has no variable {name!r}")
    return df[[column_name(name, e) for e in elements]]


def basin_variable(header: Sequence[Tuple[str, int]], df: pd.DataFrame, name: str) -> pd.Series:
    """A basin (scalar) variable such as ``basin_cfs`` as a Series named ``name``.

    Raises:
        ModelExecutionError: if ``name`` is not in the file or was output
            for several elements.
    """
    frame = variable_frame(header, df, name)
    if frame.shape[1] != 1:
        raise ModelExecutionError(
            f"statvar variable {name} has {frame.shape[1]} elements; it is not a basin scalar"
        )
    return frame.iloc[:, 0].rename(name)


def _select_element(header: Sequence[Tuple[str, int]], df: pd.DataFrame, name: str,
                    element: Optional[int]) -> pd.Series:
    elements = variable_elements(header, name)
    if element is None:
        if len(elements) > 1:
            logger.warning(
                f"statvar variable {name} has elements {elements} and no element was "
                f"specified; using the highest ({max(elements)})"
            )
        element = max(elements)
    elif element not in elements:
        raise ModelExecutionError(
            f"statvar variable {name} has no element {element}; file lists {elements}"
        )
    return df[column_name(name, element)]


def read_statvar_streamflow(
    path: Path,
    preferred: Sequence[str] = (),
    element: Optional[int] = None,
) -> pd.Series:
    """Basin streamflow (m3/s) from a statvar file.

    Picks the first available variable from ``preferred`` then
    :data:`STREAMFLOW_VARIABLES`, takes its column (``element``, else the
    only / highest listed element) and converts cfs to m3/s.

    Raises:
        ModelExecutionError: if no streamflow variable is present.
    """
    header, df = parse_statvar(path)
    names = {n for n, _ in header}
    cfs = dict(STREAMFLOW_VARIABLES)
    for name in [*preferred, *(n for n, _ in STREAMFLOW_VARIABLES)]:
        if name in names:
            series = _select_element(header, df, name, element)
            if cfs.get(name, name.endswith('_cfs')):
                series = series * CFS_TO_CMS
            return series.rename('discharge_cms')
    raise ModelExecutionError(
        f"No streamflow variable in statvar {path}; header has {sorted(names)}"
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
