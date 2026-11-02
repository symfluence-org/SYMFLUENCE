# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical parsing of configured time periods and simulation dates.

``CALIBRATION_PERIOD``, ``EVALUATION_PERIOD`` and ``SPINUP_PERIOD`` are
``'start, end'`` strings (comma separator, whitespace around either half is
ignored). Every reader of those keys used to split them by hand, with small
differences in how many parts were accepted and what happened when the value
was missing. These helpers are the one implementation; the keyword arguments
reproduce the variants that callers rely on.

Pure functions: no config access, no logging, pandas only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Tuple, overload

import pandas as pd

__all__ = [
    'PERIOD_SEPARATOR',
    'is_missing_period',
    'split_period',
    'parse_period',
    'to_pydatetime',
]

PERIOD_SEPARATOR = ','

# Values treated as "no period configured". 'default' is the config-wide
# placeholder for "derive it"; a period cannot be derived from it here.
_MISSING_TOKENS = frozenset({'', 'default', 'none'})


def is_missing_period(value: Any) -> bool:
    """True when ``value`` means "no period configured".

    ``None``, empty / whitespace-only strings, ``'default'`` and ``'none'``
    (case-insensitive) are missing. Sequences are never missing here; their
    length is checked by :func:`split_period`.
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return str(value).strip().lower() in _MISSING_TOKENS


@overload
def split_period(
    value: Any, *, allow_extra: bool = ..., allow_single: bool = ..., strict: Literal[True],
) -> Tuple[str, str]: ...


@overload
def split_period(
    value: Any, *, allow_extra: bool = ..., allow_single: bool = ..., strict: bool = ...,
) -> Optional[Tuple[str, str]]: ...


def split_period(
    value: Any,
    *,
    allow_extra: bool = False,
    allow_single: bool = False,
    strict: bool = False,
) -> Optional[Tuple[str, str]]:
    """Split a ``'start, end'`` period into two stripped strings.

    Args:
        value: The configured period. A string is split on ``','``; a
            two-element list/tuple is accepted as already split.
        allow_extra: Accept more than two parts and keep the first two
            (what ``parts[0], parts[1]`` after a ``',' in value`` check did).
            By default exactly two parts are required.
        allow_single: Accept a single date and return it as both start and
            end (for readers that only use one half and used to index
            ``parts[0]`` without a count check).
        strict: Raise ``ValueError`` instead of returning ``None`` when the
            value is missing or has the wrong number of parts.

    Returns:
        ``(start, end)`` strings, or ``None`` when missing/malformed and not
        ``strict``. An empty half (``'2000-01-01,'``) is returned as ``''``,
        exactly as a hand ``split(',')`` did; it is not judged here.
    """
    if is_missing_period(value):
        if strict:
            raise ValueError(f"No period configured (got {value!r}); expected 'start, end'")
        return None

    if isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        parts = [p.strip() for p in str(value).split(PERIOD_SEPARATOR)]

    if len(parts) == 2 or (allow_extra and len(parts) > 2):
        return parts[0], parts[1]
    if allow_single and len(parts) == 1:
        return parts[0], parts[0]
    if strict:
        raise ValueError(
            f"Malformed period {value!r}: expected 'start, end' "
            f"(two comma-separated dates), got {len(parts)} part(s)"
        )
    return None


@overload
def parse_period(
    value: Any, *, allow_extra: bool = ..., allow_single: bool = ..., strict: Literal[True],
) -> Tuple[pd.Timestamp, pd.Timestamp]: ...


@overload
def parse_period(
    value: Any, *, allow_extra: bool = ..., allow_single: bool = ..., strict: bool = ...,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]: ...


def parse_period(
    value: Any,
    *,
    allow_extra: bool = False,
    allow_single: bool = False,
    strict: bool = False,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse a ``'start, end'`` period into two ``pd.Timestamp``.

    Missing / wrong-part-count handling is :func:`split_period`'s (``None``,
    or ``ValueError`` when ``strict``). Each half is then parsed with
    ``pd.Timestamp``; an unparseable half raises ``ValueError`` regardless of
    ``strict`` (wrap the call when a bad date must not raise), and an empty
    half yields ``NaT`` as ``pd.Timestamp('')`` does.
    """
    parts = split_period(value, allow_extra=allow_extra, allow_single=allow_single, strict=strict)
    if parts is None:
        return None
    return pd.Timestamp(parts[0]), pd.Timestamp(parts[1])


def to_pydatetime(value: Any) -> datetime:
    """Coerce a configured date (string, datetime or Timestamp) to ``datetime``.

    Strings go through ``pd.Timestamp`` (so ``'2000-01-01'`` and
    ``'2000-01-01 01:00'`` both work). Raises ``ValueError`` for ``None``.
    """
    if value is None:
        raise ValueError("No date configured")
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()
