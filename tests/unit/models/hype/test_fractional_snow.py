from __future__ import annotations

import logging

import numpy as np
import pytest

from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.config_schema import HYPEConfig


def test_fractional_cover_is_opt_in_and_calibration_updates_every_class(tmp_path):
    assert not HYPEConfig().fractional_snow_cover
    manager = HYPEConfigManager({}, logging.getLogger('test'), tmp_path)
    manager.write_par_file(land_uses=np.array([15, 16, 17]))
    assert 'fscmax' not in (tmp_path/'par.txt').read_text()
    manager = HYPEConfigManager({'HYPE_FRACTIONAL_SNOW_COVER': True}, logging.getLogger('test'), tmp_path)
    manager.write_par_file(land_uses=np.array([15, 16, 17]), params={'fscdist0': .45})
    text = (tmp_path/'par.txt').read_text()
    rows = {line.split()[0]:line.split()[1:] for line in text.splitlines() if line and not line.startswith('!')}
    assert rows['fscmax'] == ['0.95']
    assert len(rows['fscdist0']) == 17
    assert [float(v) for v in rows['fscdist0']] == pytest.approx([.45]*17)
    assert [float(v) for v in rows['fscdistmax']] == pytest.approx([.8]*17)
