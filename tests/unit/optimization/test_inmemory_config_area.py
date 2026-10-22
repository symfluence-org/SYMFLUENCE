"""Area conversion must honor the canonical config when no shapefile is staged."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from symfluence.core.calibration.workers.inmemory_worker import InMemoryModelWorker
from symfluence.core.config.models.domain import DomainConfig


@pytest.mark.parametrize('setting', [
    {'CATCHMENT_AREA': 1144.956},
    {'CATCHMENT_AREA_KM2': 1144.956},
    {'DOMAIN': {'catchment_area': 1144.956}},
    {'DOMAIN': {'catchment_area_km2': 1144.956}},
])
def test_config_area_without_shapefile(tmp_path, setting):
    config = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'DOMAIN_NAME': 'no_shapes', **setting}
    worker = SimpleNamespace(_catchment_area_km2=None, _cfg=config.get,
                             logger=logging.getLogger('area_test'))
    area = InMemoryModelWorker.get_catchment_area(worker)
    assert area == pytest.approx(1144.956)
    # The wrong 1000 km² fallback changes the observed mm/day used in calibration.
    assert 100 * 86400 / (area * 1000) == pytest.approx(7.546135)


def test_plugin_area_accessor_preserves_canonical_serialization():
    domain = DomainConfig(name='basin', catchment_area=1144.956,
                          experiment_id='test', time_start='2014-01-01',
                          time_end='2018-12-31', definition_method='lumped',
                          discretization='GRUs')
    assert domain.catchment_area_km2 == domain.catchment_area
    assert domain.model_dump(by_alias=True)['CATCHMENT_AREA'] == 1144.956
    assert 'catchment_area_km2' not in domain.model_dump()
