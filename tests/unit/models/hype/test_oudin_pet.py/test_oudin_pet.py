from __future__ import annotations

import logging

import numpy as np

from symfluence.models.hype.config_manager import HYPEConfigManager


def test_oudin_supplies_reference_constants_for_every_land_use(tmp_path):
    manager = HYPEConfigManager({'HYPE_PET_MODEL': 2}, logging.getLogger('test'), tmp_path)
    manager.write_par_file(land_uses=np.array([15, 16, 17]))
    rows = {line.split()[0]: line.split('!')[0].split()[1:]
            for line in (tmp_path / 'par.txt').read_text().splitlines()
            if line.strip() and not line.startswith('!')}
    assert rows['jhtadd'] == ['5.0']
    assert rows['jhtscale'] == ['100.0']
    assert list(map(float, rows['kc'])) == [1.0] * 17
    manager = HYPEConfigManager({}, logging.getLogger('test'), tmp_path)
    manager.write_par_file(land_uses=np.array([15, 16, 17]))
    assert '\njhtscale' not in (tmp_path / 'par.txt').read_text()
