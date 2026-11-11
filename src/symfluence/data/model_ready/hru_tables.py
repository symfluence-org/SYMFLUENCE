# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Id-keyed handling of per-HRU attribute tables.

The attribute processors write one CSV row per HRU.  Those tables must be
joined on the HRU id, never on row position: the order of rows in a CSV is
not the order of HRUs in the catchment shapefile or in a model's attribute
file.  This module holds the one implementation of that join, shared by the
CSV writer (``BaseAttributeProcessor._write_results_csv``) and its readers
(``AttributesNetCDFBuilder``, SUMMA transfer-function regionalization).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

#: Name of the id column in per-HRU attribute CSVs.
HRU_ID_COLUMN = 'hru_id'

#: CSVs written before the id column existed were string-sorted by HRU id
#: (1, 10, 11, 2, ...).  With single-digit ids that equals the numeric order,
#: so tables up to this size can still be mapped back to their ids; larger
#: ones cannot be trusted and must be regenerated.
LEGACY_POSITIONAL_MAX_ROWS = 9

#: File names written by the built-in attribute processors.
PROCESSOR_CSV_NAMES = frozenset({
    'worldclim_attributes.csv',
    'soil_extended_attributes.csv',
    'glhymps_attributes.csv',
    'landcover_extended_attributes.csv',
    'hydrology_attributes.csv',
})


def normalize_hru_id(value: Any) -> str:
    """Canonical string form of an HRU id (``12``, ``12.0``, ``' 12'`` -> ``'12'``)."""
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if np.isfinite(value) and float(value).is_integer() else str(value)
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    if np.isfinite(as_float) and as_float.is_integer() and 'e' not in text.lower():
        return str(int(as_float))
    return text


def order_hru_ids(ids: Iterable[Any], preferred_order: Optional[Sequence[Any]] = None) -> List[str]:
    """Order HRU ids numerically when all are integers, else by *preferred_order*.

    Ids that are not all integers keep the order of *preferred_order* (the
    catchment shapefile) when given; ids absent from it follow in first-seen
    order.
    """
    normalized = list(dict.fromkeys(normalize_hru_id(i) for i in ids))
    try:
        return sorted(normalized, key=int)
    except ValueError:
        pass
    if preferred_order is None:
        return normalized
    rank = {normalize_hru_id(v): k for k, v in enumerate(preferred_order)}
    return sorted(normalized, key=lambda i: rank.get(i, len(rank)))


def legacy_table_error(source: Union[str, Path], n_rows: int) -> ValidationError:
    """Error for a pre-``hru_id`` attribute table that cannot be joined safely."""
    return ValidationError(
        f"Per-HRU attribute table {source} has {n_rows} rows but no "
        f"'{HRU_ID_COLUMN}' column. Tables written by older SYMFLUENCE versions "
        "were sorted by the HRU id as text (1, 10, 11, 2, ...) and were joined to "
        "HRUs by row position, which assigns attributes to the wrong HRUs. "
        "Delete the file and regenerate the attributes (re-run the attribute "
        "processing step) so that it carries HRU ids."
    )


def read_hru_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read a per-HRU CSV, keeping the id column as normalized strings."""
    table = pd.read_csv(path)
    if HRU_ID_COLUMN in table.columns:
        table[HRU_ID_COLUMN] = [normalize_hru_id(v) for v in table[HRU_ID_COLUMN]]
    return table


def align_table_to_hru_ids(
    table: pd.DataFrame,
    hru_ids: Sequence[Any],
    source: Union[str, Path],
    *,
    legacy_order: str = 'string_sorted',
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """Return *table* with one row per entry of *hru_ids*, in that order.

    Args:
        table: Per-HRU table, ideally with an ``hru_id`` column.
        hru_ids: Target HRU ids, in the order the caller needs.
        source: Name of the table (for messages).
        legacy_order: How rows of a table WITHOUT an id column are ordered:
            ``'string_sorted'`` (written by an old attribute processor) or
            ``'positional'`` (a user-authored table documented as being in HRU
            order).
        logger: Logger for warnings.

    Returns:
        The aligned table without the id column (rows of HRUs absent from
        *table* are NaN), or ``None`` when the table cannot be matched to the
        HRUs at all (row-count mismatch without ids, or no id in common).

    Raises:
        ValidationError: for duplicate ids, or for a string-sorted legacy
            table with more than ``LEGACY_POSITIONAL_MAX_ROWS`` rows.
    """
    log = logger or logging.getLogger(__name__)
    target = [normalize_hru_id(i) for i in hru_ids]

    if HRU_ID_COLUMN in table.columns:
        ids = [normalize_hru_id(v) for v in table[HRU_ID_COLUMN]]
        if len(set(ids)) != len(ids):
            raise ValidationError(
                f"Per-HRU attribute table {source} has duplicate '{HRU_ID_COLUMN}' values."
            )
        keyed = table.drop(columns=[HRU_ID_COLUMN])
        keyed.index = pd.Index(ids)
        missing = [i for i in target if i not in set(ids)]
        if len(missing) == len(target):
            log.warning(
                f"Per-HRU attribute table {source} shares no HRU id with the domain "
                f"(table ids e.g. {ids[:5]}, domain ids e.g. {target[:5]}); not using it."
            )
            return None
        if missing:
            log.warning(
                f"Per-HRU attribute table {source} has no row for {len(missing)} of "
                f"{len(target)} HRUs (e.g. {missing[:5]}); their attributes are NaN."
            )
        return keyed.reindex(target).reset_index(drop=True)

    n_rows = len(table)
    if n_rows != len(target):
        log.warning(
            f"Per-HRU attribute table {source} has {n_rows} rows and no "
            f"'{HRU_ID_COLUMN}' column but the domain has {len(target)} HRUs; not using it."
        )
        return None
    if n_rows == 1 or legacy_order == 'positional':
        return table.reset_index(drop=True)
    if n_rows > LEGACY_POSITIONAL_MAX_ROWS:
        raise legacy_table_error(source, n_rows)

    # Old writer: rows sorted by str(hru_id). Recover each row's id from that.
    log.warning(
        f"Per-HRU attribute table {source} has no '{HRU_ID_COLUMN}' column (written "
        "by an older SYMFLUENCE version); assuming its rows are in text-sorted HRU id "
        "order. Regenerate the attributes to make the join explicit."
    )
    keyed = table.copy()
    keyed.index = pd.Index(sorted(target))
    return keyed.reindex(target).reset_index(drop=True)
