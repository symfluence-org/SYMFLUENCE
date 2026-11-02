# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical period parsing (``core.modeling.utilities.periods``)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from symfluence.core.modeling.utilities import TimeWindowManager
from symfluence.core.modeling.utilities.periods import (
    is_missing_period,
    parse_period,
    split_period,
    to_pydatetime,
)

# ---------------------------------------------------------------- split_period


@pytest.mark.parametrize("value", [
    "2000-01-01,2001-12-31",
    "2000-01-01, 2001-12-31",
    "  2000-01-01 ,\t2001-12-31  ",
    ("2000-01-01", " 2001-12-31"),
    ["2000-01-01", "2001-12-31"],
])
def test_split_period_separator_and_whitespace(value):
    assert split_period(value) == ("2000-01-01", "2001-12-31")


def test_split_period_keeps_time_of_day():
    assert split_period("2000-01-01 01:00, 2001-12-31 23:00") == ("2000-01-01 01:00", "2001-12-31 23:00")


@pytest.mark.parametrize("value", [None, "", "   ", "default", "DEFAULT", " Default ", "none", (), []])
def test_split_period_missing_is_none(value):
    assert is_missing_period(value)
    assert split_period(value) is None
    assert split_period(value, allow_extra=True, allow_single=True) is None


@pytest.mark.parametrize("value", [None, "", "default"])
def test_split_period_missing_strict_raises(value):
    with pytest.raises(ValueError, match="No period configured"):
        split_period(value, strict=True)


@pytest.mark.parametrize("value", ["2000-01-01", "2000-01-01 2001-12-31", "2000-01-01 to 2001-12-31"])
def test_split_period_without_comma_is_malformed(value):
    assert split_period(value) is None
    with pytest.raises(ValueError, match="Malformed period"):
        split_period(value, strict=True)


def test_split_period_extra_parts_need_allow_extra():
    value = "2000-01-01, 2001-12-31, 2005-01-01"
    assert split_period(value) is None
    with pytest.raises(ValueError, match="3 part"):
        split_period(value, strict=True)
    assert split_period(value, allow_extra=True) == ("2000-01-01", "2001-12-31")


def test_split_period_single_date_needs_allow_single():
    assert split_period("2000-01-01") is None
    assert split_period(" 2000-01-01 ", allow_single=True) == ("2000-01-01", "2000-01-01")
    # allow_single does not loosen the two-or-more rule
    assert split_period("a, b, c", allow_single=True) is None


def test_split_period_empty_half_is_returned_as_empty_string():
    # Not judged here: a hand split(',') returned '' too.
    assert split_period("2000-01-01,") == ("2000-01-01", "")
    assert split_period(", 2001-12-31") == ("", "2001-12-31")


# ---------------------------------------------------------------- parse_period


def test_parse_period_returns_timestamps():
    start, end = parse_period("2000-01-01, 2001-12-31 23:00")
    assert isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp)
    assert (start, end) == (pd.Timestamp("2000-01-01"), pd.Timestamp("2001-12-31 23:00"))


@pytest.mark.parametrize("value", [None, "", "default", "2000-01-01", "a, b, c"])
def test_parse_period_missing_or_malformed_is_none(value):
    assert parse_period(value) is None


@pytest.mark.parametrize("value", [None, "default", "2000-01-01"])
def test_parse_period_strict_raises(value):
    with pytest.raises(ValueError):
        parse_period(value, strict=True)


def test_parse_period_unparseable_date_raises_even_when_lenient():
    with pytest.raises(ValueError):
        parse_period("not-a-date, 2001-12-31")


def test_parse_period_options_pass_through():
    assert parse_period("2000-01-01, 2001-12-31, x", allow_extra=True) == (
        pd.Timestamp("2000-01-01"), pd.Timestamp("2001-12-31"))
    assert parse_period("2000-01-01", allow_single=True) == (
        pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-01"))


def test_parse_period_empty_half_is_nat():
    start, end = parse_period("2000-01-01,")
    assert start == pd.Timestamp("2000-01-01")
    assert pd.isna(end)


# ---------------------------------------------------------------- to_pydatetime


@pytest.mark.parametrize("value, expected", [
    ("2000-01-01", datetime(2000, 1, 1)),
    ("2000-01-01 01:00", datetime(2000, 1, 1, 1)),
    (pd.Timestamp("2000-01-02"), datetime(2000, 1, 2)),
    (datetime(2000, 1, 3, 4), datetime(2000, 1, 3, 4)),
])
def test_to_pydatetime(value, expected):
    result = to_pydatetime(value)
    assert type(result) is datetime
    assert result == expected


def test_to_pydatetime_none_raises():
    with pytest.raises(ValueError):
        to_pydatetime(None)


# ---------------------------------------------------------------- TimeWindowManager


def test_time_window_manager_prefers_explicit_start_end_keys():
    twm = TimeWindowManager({
        "CALIBRATION_PERIOD_START": "2001-01-01",
        "CALIBRATION_PERIOD_END": "2002-01-01",
        "CALIBRATION_PERIOD": "1990-01-01, 1991-01-01",
    })
    assert twm.get_calibration_period() == (pd.Timestamp("2001-01-01"), pd.Timestamp("2002-01-01"))


def test_time_window_manager_falls_back_to_period_string():
    twm = TimeWindowManager({
        "CALIBRATION_PERIOD": "2001-01-01, 2002-12-31",
        "EVALUATION_PERIOD": "2003-01-01,2004-12-31",
    })
    assert twm.get_calibration_period() == (pd.Timestamp("2001-01-01"), pd.Timestamp("2002-12-31"))
    assert twm.get_evaluation_period() == (pd.Timestamp("2003-01-01"), pd.Timestamp("2004-12-31"))


@pytest.mark.parametrize("config", [{}, {"EVALUATION_PERIOD": "default"}, {"EVALUATION_PERIOD": "2003-01-01"}])
def test_time_window_manager_missing_period_raises(config):
    with pytest.raises(ValueError, match="Evaluation period not defined"):
        TimeWindowManager(config).get_evaluation_period()
