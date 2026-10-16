from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.exceptions import ValidationError
from symfluence.models.wflow.outlet import flow_to_gauge, retain_gauge_cell


def test_gauge_on_narrow_vector_edge_retains_its_cell():
    labels = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    # Gauge is within vector but its cell centre is outside it.
    geom = box(.8, 1., 3., 3.)
    result, pit, added = retain_gauge_cell(labels, from_origin(0, 3, 1, 1), [geom], (.9, 1.9))
    assert added and pit == 3 and result[1, 0] == 1
    assert labels[1, 0] == 0
    try:
        import pyflwdir  # noqa: F401
    except ImportError as exc:  # numba refuses some NumPy versions at import
        pytest.skip(f'pyflwdir unavailable: {exc}')
    dem = np.where(result, [[0., 8., 9.], [5., 7., 8.], [0., 0., 0.]], -9999.)
    flow, meta = flow_to_gauge(dem, from_origin(0, 3, 1, 1), pit)
    assert flow.idxs_pit.tolist() == [pit]
    assert meta['all_active_cells_reach_gauge']
    assert flow.accuflux((result > 0).astype(float)).ravel()[pit] == 5


def test_outside_gauge_is_rejected_not_snapped():
    with pytest.raises(ValidationError, match='outside'):
        retain_gauge_cell(np.ones((2, 2)), from_origin(0, 2, 1, 1), [box(0, 0, 2, 2)], (3., 1.))


def test_inactive_outlet_rejected():
    with pytest.raises(ValidationError, match='active'):
        flow_to_gauge(np.array([[-9999., 1.], [2., 3.]]), from_origin(0, 2, 1, 1), 0)
