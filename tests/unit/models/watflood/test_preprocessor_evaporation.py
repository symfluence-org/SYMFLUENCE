"""Bare classes must have finite native vegetation normalization."""
from types import SimpleNamespace

import numpy as np

from symfluence.models.watflood.preprocessor import WATFLOODPreProcessor


def test_generated_bare_class_evaporation_factor_is_finite(tmp_path):
    (tmp_path / 'basin').mkdir()
    processor = SimpleNamespace(
        settings_dir=tmp_path,
        _get_catchment_properties=lambda: {'lat': 64.0},
    )
    WATFLOODPreProcessor._generate_par_file(processor)
    rows = {}
    for line in (tmp_path / 'basin' / 'bow.par').read_text().splitlines():
        fields = line.split('#', 1)[0].split(',')
        if fields[0].strip() in [':a11', ':ftall'] or fields[0].startswith(':IntCap_'):
            rows[fields[0].strip()] = [float(v) for v in fields[1:] if v.strip()]
    floor = rows[':a11'][0]
    capacity = np.array([v for k, v in rows.items() if k.startswith(':IntCap_')])
    assert np.any(np.all(capacity == 0, axis=0)), 'Exercise a bare class'
    # This is the native reader's normalization; the old sentinel gives 0/0
    # for bare classes and lets NaN ET fall into its soil-draining branch.
    height = np.maximum(floor, capacity)
    with np.errstate(divide='raise', invalid='raise'):
        factor = np.array(rows[':ftall']) * height / height.max(axis=0)
    assert np.isfinite(factor).all()
    np.testing.assert_allclose(factor[:, np.all(capacity == 0, axis=0)],
                               np.broadcast_to(np.array(rows[':ftall'])[np.all(capacity == 0, axis=0)],
                                               factor[:, np.all(capacity == 0, axis=0)].shape))
