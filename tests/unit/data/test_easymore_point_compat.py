# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""EASYMORE point-grid (case 3) row/column indices."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("easymore")

from symfluence.data.preprocessing.easymore_compat import NativePointEasymore  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def test_point_indices_are_scalar_and_preserve_native_order():
    remapper = NativePointEasymore()
    remapper.case = 3
    rows, cols = remapper.create_row_col_df(np.array([64., 65., 66.]), np.array([-18., -17., -16.]), [66., 64.], [-16., -18.])
    np.testing.assert_array_equal(rows, [2, 0])
    np.testing.assert_array_equal(cols, rows)
    assert rows.dtype.kind in "iu"


def test_ambiguous_point_is_rejected():
    remapper = NativePointEasymore()
    remapper.case = 3
    with pytest.raises(ValueError, match="Ambiguous"):
        remapper.create_row_col_df(np.array([64., 64.]), np.array([-18., -18.]), [64.], [-18.])


def test_invalid_point_coordinates_are_rejected():
    remapper = NativePointEasymore()
    remapper.case = 3
    with pytest.raises(ValueError, match="Invalid"):
        remapper.create_row_col_df(np.array([np.nan]), np.array([-18.]), [64.], [-18.])


def test_structured_grid_delegates_unchanged(monkeypatch):
    from symfluence.data.preprocessing import easymore_compat
    sentinel = (np.array([1]), np.array([2]))
    monkeypatch.setattr(easymore_compat._Base, "create_row_col_df", lambda self, *args: sentinel)
    remapper = NativePointEasymore()
    remapper.case = 2
    assert remapper.create_row_col_df([], [], [], []) is sentinel
