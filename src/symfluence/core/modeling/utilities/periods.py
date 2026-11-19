# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical parsing of configured time periods and simulation dates.

``CALIBRATION_PERIOD``, ``EVALUATION_PERIOD``, ``TEST_PERIOD`` and ``SPINUP_PERIOD`` are
``'start, end'`` strings (comma separator, whitespace around either half is
ignored). Every reader of those keys used to split them by hand, with small
differences in how many parts were accepted and what happened when the value
was missing. These helpers are the one implementation; the keyword arguments
reproduce the variants that callers rely on.

Pure functions: no config access, no logging, pandas only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, overload

import pandas as pd

__all__ = [
    'PERIOD_SEPARATOR',
    'is_missing_period',
    'split_period',
    'parse_period',
    'to_pydatetime',
    'period_end_exclusive',
    'periods_overlap',
    'spinup_window',
    'split_sample_warnings',
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


def period_end_exclusive(end: Any) -> pd.Timestamp:
    """Exclusive upper bound of a period ending at ``end``.

    A bare date (00:00) covers the whole day, so the bound is the next
    midnight; an explicit time is inclusive as written (the bound is one
    nanosecond later). This matches how evaluators mask periods.
    """
    end = pd.Timestamp(end)
    if end == end.normalize():
        return end + pd.Timedelta(days=1)
    return end + pd.Timedelta(1, unit='ns')


def periods_overlap(a: Tuple[Any, Any], b: Tuple[Any, Any]) -> bool:
    """True when the inclusive periods ``a`` and ``b`` share any instant.

    Period ends follow :func:`period_end_exclusive`, so ``2005-12-31`` and
    ``2006-01-01`` are adjacent, not overlapping.
    """
    a_start, b_start = pd.Timestamp(a[0]), pd.Timestamp(b[0])
    return bool(
        a_start < period_end_exclusive(b[1]) and b_start < period_end_exclusive(a[1])
    )


def spinup_window(
    spinup_period: Any = None,
    spinup_years: Any = 0,
    simulation_start: Any = None,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """The single spin-up window evaluators exclude from every score.

    Spin-up is one notion: the stretch at the start of the run whose output is
    discarded. It is configured with ``SPINUP_PERIOD`` (``'start, end'``). The
    deprecated ``EVALUATION_SPINUP_YEARS`` (N years counted from the
    simulation start) maps onto the same window. When both are set the
    window extends to the later of the two ends.

    Returns ``(start, end_exclusive)`` or ``None`` when no spin-up is set.
    An unparseable ``SPINUP_PERIOD`` is ignored here (config validation
    reports it).
    """
    windows = []
    if isinstance(spinup_period, (str, list, tuple)) and not is_missing_period(spinup_period):
        try:
            parsed = parse_period(spinup_period, allow_extra=True)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and not pd.isna(parsed[0]) and not pd.isna(parsed[1]):
            windows.append((parsed[0], period_end_exclusive(parsed[1])))

    try:
        years = int(float(spinup_years)) if spinup_years is not None else 0
    except (TypeError, ValueError):
        years = 0
    if years > 0 and simulation_start is not None:
        try:
            sim_start = pd.Timestamp(simulation_start)
        except (TypeError, ValueError):
            sim_start = None
        if sim_start is not None and not pd.isna(sim_start):
            windows.append((sim_start, sim_start + pd.DateOffset(years=years)))

    if not windows:
        return None
    return min(w[0] for w in windows), max(w[1] for w in windows)


def split_sample_warnings(
    *,
    calibration: Any = None,
    evaluation: Any = None,
    test: Any = None,
    spinup: Any = None,
    spinup_years: Any = 0,
    simulation_start: Any = None,
    calibrating: bool = False,
) -> List[str]:
    """Problems with the split-sample design, as human-readable messages.

    Checks (each yields one message, nothing raises):

    * calibration, evaluation and test periods must not overlap each other;
    * spin-up (``SPINUP_PERIOD`` and/or ``EVALUATION_SPINUP_YEARS``) must end
      before every scored period starts;
    * without ``CALIBRATION_PERIOD``, metrics fall back to the full record,
      which contains the evaluation/test years. Reported when an evaluation
      or test period is set or ``calibrating`` is True.

    Malformed periods are skipped; format errors are reported elsewhere.
    """
    def _parse(value: Any) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        if not isinstance(value, (str, list, tuple)) or is_missing_period(value):
            return None
        try:
            parsed = parse_period(value, allow_extra=True)
        except (ValueError, TypeError):
            return None
        if parsed is None or pd.isna(parsed[0]) or pd.isna(parsed[1]):
            return None
        return parsed

    scored: Dict[str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]] = {
        'CALIBRATION_PERIOD': _parse(calibration),
        'EVALUATION_PERIOD': _parse(evaluation),
        'TEST_PERIOD': _parse(test),
    }
    raw = {'CALIBRATION_PERIOD': calibration, 'EVALUATION_PERIOD': evaluation, 'TEST_PERIOD': test}
    messages: List[str] = []

    names = list(scored)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = scored[a], scored[b]
            if pa is not None and pb is not None and periods_overlap(pa, pb):
                messages.append(
                    f"{a} ({raw[a]}) overlaps {b} ({raw[b]}): the split-sample "
                    f"periods are not independent, so {b.split('_')[0].lower()} "
                    f"scores are not out-of-sample."
                )

    window = spinup_window(spinup, spinup_years, simulation_start)
    if window is not None:
        for name, period in scored.items():
            if period is None:
                continue
            if period[0] < window[1]:
                messages.append(
                    f"Spin-up (ends {window[1] - pd.Timedelta(1, unit='ns'):%Y-%m-%d %H:%M}) "
                    f"does not precede {name} ({raw[name]}): output before the spin-up "
                    f"end is excluded from scoring, so part or all of this period is not scored."
                )

    if scored['CALIBRATION_PERIOD'] is None and (
        calibrating or scored['EVALUATION_PERIOD'] is not None or scored['TEST_PERIOD'] is not None
    ):
        messages.append(
            "CALIBRATION_PERIOD is not set: calibration metrics fall back to the full "
            "simulation/observation overlap, which includes any evaluation and test "
            "years. Set CALIBRATION_PERIOD to keep the evaluation independent."
        )
    return messages
