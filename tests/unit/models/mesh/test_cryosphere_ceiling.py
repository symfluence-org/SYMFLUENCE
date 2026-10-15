"""A better discharge fit cannot buy its way through declared physical limits."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from symfluence.models.mesh.calibration.worker import MESHWorker
from symfluence.models.mesh.config_schema import MESHConfig


def setup():
    cfg = {'model': {'mesh': {'cryosphere_constraints': True, 'snow_cover_max_rmse': .23, 'glacier_mb_max_rmse': .46}},
           'optimization': {'metric': 'COMPOSITE', 'composite_metric': {'kge': .6, 'snow_score': .2, 'glacier_score': .2}}}
    metrics = {'kge': .2, 'snow_score': -.15, 'glacier_score': .54, 'snow_rmse': .23, 'glacier_mb_rmse': .46}
    return cfg, metrics


@pytest.mark.parametrize('metric', ['snow_rmse', 'glacier_mb_rmse'])
def test_discharge_gain_cannot_offset_physical_regression(metric):
    cfg, metrics = setup()
    worker = MESHWorker(logger=Mock())
    assert worker._extract_primary_score(metrics, cfg) > worker.penalty_score
    metrics['kge'] = 1.
    metrics[metric] += .001
    assert worker._extract_primary_score(metrics, cfg) == worker.penalty_score


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -1., 'invalid'])
def test_missing_or_invalid_constraint_metric_fails_closed(value):
    cfg, metrics = setup()
    metrics['snow_rmse'] = value
    worker = MESHWorker(logger=Mock())
    assert worker._extract_primary_score(metrics, cfg) == worker.penalty_score


def test_flat_aliases_and_disabled_constraints():
    cfg, metrics = setup()
    cfg.pop('model')
    cfg.update(MESH_CRYOSPHERE_CONSTRAINTS=True, MESH_SNOW_COVER_MAX_RMSE=.23, MESH_GLACIER_MB_MAX_RMSE=.46)
    worker = MESHWorker(logger=Mock())
    assert worker._extract_primary_score(metrics, cfg) > worker.penalty_score
    cfg['MESH_CRYOSPHERE_CONSTRAINTS'] = False
    assert worker._extract_primary_score(metrics, cfg) == worker.penalty_score


def test_schema_rejects_limits_without_constraints():
    with pytest.raises(ValueError, match='ceilings require'):
        MESHConfig(snow_cover_max_rmse=.2)


def test_no_ceilings_preserves_legacy_composite():
    cfg, metrics = setup()
    cfg['model']['mesh'] = {'cryosphere_constraints': True}
    metrics.pop('snow_rmse')
    metrics.pop('glacier_mb_rmse')
    assert MESHWorker(logger=Mock())._extract_primary_score(metrics, cfg) == pytest.approx(.198)


def test_typed_config_enforces_same_limits(mesh_config):
    cfg, metrics = setup()
    from symfluence.core.config import SymfluenceConfig

    raw = mesh_config.to_dict()
    raw.update(MESH_CRYOSPHERE_CONSTRAINTS=True,
               MESH_SPATIAL_DIAGNOSTICS=True,
               MESH_SPATIAL_UNITS='units.json',
               MESH_SPATIAL_FORCING='forcing.nc',
               MESH_CRYOSPHERE_OBSERVATIONS='observations.json',
               MESH_GLACIER_MODE='finite_ice_guard',
               MESH_GLACIER_MANIFEST='manifest.json',
               MESH_GLACIER_UNIT_IDS=['1'],
               MESH_GLACIER_ALBEDO_VISIBLE=.3,
               MESH_GLACIER_ALBEDO_INFRARED=.2,
               MESH_GLACIER_TEMPERATURE_C=0.,
               MESH_SNOW_COVER_MAX_RMSE=.23,
               MESH_GLACIER_MB_MAX_RMSE=.46,
               COMPOSITE_METRIC=cfg['optimization']['composite_metric'])
    typed = SymfluenceConfig(**raw)
    worker = MESHWorker(logger=Mock())
    assert worker._extract_primary_score(metrics, typed) == pytest.approx(.198)
    metrics['glacier_mb_rmse'] = .461
    assert worker._extract_primary_score(metrics, typed) == worker.penalty_score


def test_absent_constraint_metric_fails_closed():
    cfg, metrics = setup()
    metrics.pop('snow_rmse')
    worker = MESHWorker(logger=Mock())
    assert worker._extract_primary_score(metrics, cfg) == worker.penalty_score
