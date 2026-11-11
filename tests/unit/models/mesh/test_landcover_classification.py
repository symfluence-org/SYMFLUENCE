# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MESH GRU labels follow the classification the class ids are actually in.

The framework's own land-cover statistics come from MODIS MCD12Q1 (IGBP ids,
written as ``IGBP_<id>`` columns), but MESH labelled every class id with the
NALCMS table: IGBP 15 (snow and ice) became 'Cropland'/crops, 17 (water)
became 'Urban', 11 (wetland) grass, 2 (evergreen broadleaf) needleleaf.
Prepared ``frac_<id>`` tables (NALCMS, gistool convention) are unchanged.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.mesh.errors import MESHConfigurationError
from symfluence.models.mesh.preprocessing.config_defaults import MESHConfigDefaults

STATS_NAME = 'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv'


def _csv(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n{row}\n")
    return path


# --- detection ----------------------------------------------------------------

def test_framework_statistics_from_modis_are_igbp(tmp_path):
    csv = _csv(tmp_path / STATS_NAME, 'IGBP_10,IGBP_15,IGBP_16,land_pixel_count', '0.2,0.1,0.7,100')
    assert MESHConfigDefaults.detect_landcover_classification(csv) == 'igbp'
    assert MESHConfigDefaults.detect_landcover_classification(csv, 'default', 'MODIS') == 'igbp'


def test_prepared_frac_tables_stay_nalcms(tmp_path):
    csv = _csv(tmp_path / 'mesh_landcover.csv', 'GRU_ID,frac_10,frac_14,frac_19,frac_16,frac_18',
               '1,0.06,0.001,0.09,0.82,0.025')
    assert MESHConfigDefaults.detect_landcover_classification(csv) == 'nalcms'


def test_configured_classification_decides(tmp_path):
    igbp = _csv(tmp_path / 'a.csv', 'IGBP_10,IGBP_15', '0.5,0.5')
    frac = _csv(tmp_path / 'b.csv', 'GRU_ID,frac_10,frac_15', '1,0.5,0.5')
    assert MESHConfigDefaults.detect_landcover_classification(igbp, 'nalcms') == 'nalcms'
    assert MESHConfigDefaults.detect_landcover_classification(frac, 'IGBP') == 'igbp'
    assert MESHConfigDefaults.detect_landcover_classification(frac, 'modis') == 'igbp'
    assert MESHConfigDefaults.detect_landcover_classification(None, 'igbp') == 'igbp'
    with pytest.raises(MESHConfigurationError, match='Unknown MESH land-cover classification'):
        MESHConfigDefaults.detect_landcover_classification(frac, 'corine')


def test_undeterminable_classification_is_an_error(tmp_path):
    detect = MESHConfigDefaults.detect_landcover_classification
    with pytest.raises(MESHConfigurationError, match="LAND_CLASS_SOURCE 'usgs_nlcd'"):
        detect(_csv(tmp_path / 'nlcd.csv', 'IGBP_11,IGBP_42', '0.5,0.5'), 'default', 'usgs_nlcd')
    with pytest.raises(MESHConfigurationError, match='both IGBP'):
        detect(_csv(tmp_path / 'both.csv', 'GRU_ID,IGBP_10,frac_10', '1,1,1'))
    with pytest.raises(MESHConfigurationError, match='no IGBP_<id> or frac_<id>'):
        detect(_csv(tmp_path / 'none.csv', 'GRU_ID,grassland', '1,1'))
    with pytest.raises(MESHConfigurationError, match='not found'):
        detect(tmp_path / 'absent.csv')
    with pytest.raises(MESHConfigurationError, match='not found'):
        detect(None)


def test_igbp_tables_cover_every_class_and_keep_meshflow_conventions():
    names, types = MESHConfigDefaults.IGBP_LANDCOVER_CLASSES, MESHConfigDefaults.IGBP_GRU_MAPPING
    assert sorted(names) == sorted(types) == list(range(1, 18))
    assert set(types.values()) <= set(MESHConfigDefaults.FULL_GRU_MAPPING.values())
    assert (types[15], types[16], types[17], types[11], types[12]) == (
        'barrenland', 'barrenland', 'water', 'wetland', 'crops')
    assert (types[1], types[2], types[3], types[4]) == ('needleleaf', 'broadleaf', 'needleleaf', 'broadleaf')
    # meshflow switches interflow off from the first word of the class name
    assert names[17].split()[0].lower() == 'water'
    assert names[11].split()[0].lower() == 'wetland'
    # ... and CLASS vegetation fixes key on the 4-letter short label
    assert 'needleleaf' in names[1] and 'shrubland' in names[6] and names[12].startswith('Crop')


def test_unknown_class_ids_are_an_error():
    MESHConfigDefaults.check_class_ids([10, 15, 17], 'igbp')
    MESHConfigDefaults.check_class_ids([10, 19], 'nalcms')
    with pytest.raises(MESHConfigurationError, match=r'\[18, 19\] are not IGBP classes'):
        MESHConfigDefaults.check_class_ids([10, 18, 19], 'igbp')
    with pytest.raises(MESHConfigurationError, match=r'\[42\] are not NALCMS classes'):
        MESHConfigDefaults.check_class_ids([42], 'nalcms')


# --- preprocessor ---------------------------------------------------------------

def _meshflow_config(tmp_path, header, row, name=STATS_NAME, **extra):
    from symfluence.models.mesh.preprocessor import MESHPreProcessor

    config = SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'mesh_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'MESH',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        **extra,
    })
    pre = MESHPreProcessor(config, logging.getLogger('mesh-pre'))
    _csv(pre.project_attributes_dir / 'gistool-outputs' / name, header, row)
    return pre._create_meshflow_config()


def test_modis_statistics_are_labelled_as_igbp(tmp_path):
    """Regression: IGBP 15/17/11 were labelled Cropland/Urban/shrub-lichen-moss."""
    cfg = _meshflow_config(
        tmp_path, 'IGBP_2,IGBP_10,IGBP_11,IGBP_15,IGBP_16,IGBP_17,land_pixel_count',
        '0.05,0.2,0.05,0.1,0.55,0.05,100')
    assert cfg['landcover_classes'] == {
        2: 'Evergreen broadleaf forest', 10: 'Grassland', 11: 'Wetland',
        15: 'Snow and Ice', 16: 'Barren or sparsely vegetated', 17: 'Water bodies'}
    assert cfg['settings']['class_params']['grus'] == {
        2: 'broadleaf', 10: 'grass', 11: 'wetland', 15: 'barrenland', 16: 'barrenland', 17: 'water'}


def test_nalcms_table_setup_is_unchanged(tmp_path):
    """A prepared frac_<id> table gets exactly the labels and types it always got."""
    cfg = _meshflow_config(
        tmp_path, 'GRU_ID,frac_10,frac_14,frac_19,frac_16,frac_18', '1,0.06,0.001,0.09,0.82,0.025',
        name='mesh_landcover.csv')
    assert cfg['landcover_classes'] == {
        10: 'Temperate or sub-polar grassland', 14: 'Wetland', 16: 'Barren lands',
        18: 'Water', 19: 'Snow and Ice'}
    assert cfg['settings']['class_params']['grus'] == {
        10: 'grass', 14: 'wetland', 16: 'barrenland', 18: 'water', 19: 'barrenland'}


def test_configured_classification_overrides_the_column_prefix(tmp_path):
    cfg = _meshflow_config(tmp_path, 'IGBP_10,IGBP_15,IGBP_19', '0.5,0.3,0.2',
                           MESH_LANDCOVER_CLASSIFICATION='nalcms')
    assert cfg['landcover_classes'][15] == 'Cropland'
    assert cfg['settings']['class_params']['grus'] == {10: 'grass', 15: 'crops', 19: 'barrenland'}


def test_class_ids_outside_the_classification_stop_the_setup(tmp_path):
    with pytest.raises(MESHConfigurationError, match=r'\[19\] are not IGBP classes'):
        _meshflow_config(tmp_path, 'IGBP_10,IGBP_19', '0.5,0.5')


def test_other_land_cover_sources_need_explicit_tables(tmp_path):
    header, row = 'IGBP_11,IGBP_42,IGBP_71', '0.1,0.5,0.4'
    with pytest.raises(MESHConfigurationError, match="LAND_CLASS_SOURCE 'usgs_nlcd'"):
        _meshflow_config(tmp_path / 'a', header, row, LAND_CLASS_SOURCE='usgs_nlcd')
    names = {11: 'Water open', 42: 'Evergreen needleleaf forest', 71: 'Grassland'}
    types = {11: 'water', 42: 'needleleaf', 71: 'grass'}
    cfg = _meshflow_config(tmp_path / 'b', header, row, LAND_CLASS_SOURCE='usgs_nlcd',
                           MESH_LANDCOVER_CLASSES=names, MESH_GRU_MAPPING=types)
    assert cfg['landcover_classes'] == names
    assert cfg['settings']['class_params']['grus'] == types


# --- calibration multipliers ----------------------------------------------------

def _manager(tmp_path, stats_header=None, **extra):
    settings = tmp_path / 'settings'
    settings.mkdir(parents=True, exist_ok=True)
    if stats_header is not None:
        _csv(tmp_path / 'domain_test_domain' / 'attributes' / 'gistool-outputs' / STATS_NAME,
             stats_header, ','.join('1' for _ in stats_header.split(',')))
    config = {'DOMAIN_NAME': 'test_domain', 'SYMFLUENCE_DATA_DIR': str(tmp_path),
              'MESH_PARAMS_TO_CALIBRATE': 'KSAT,SDEP,MANN_CLASS',
              'MESH_USE_LANDCOVER_MULTIPLIERS': True, **extra}
    return MESHParameterManager(config, logging.getLogger('mesh-pm'), settings)


def test_multiplier_table_follows_the_classification(tmp_path):
    nalcms = MESHParameterManager.DEFAULT_LANDCOVER_MULTIPLIERS
    igbp = _manager(tmp_path / 'igbp', 'IGBP_10,IGBP_15,IGBP_16').landcover_multipliers
    assert igbp[15] == nalcms[19]          # snow and ice
    assert 19 not in igbp and 5 not in igbp and 8 not in igbp
    assert igbp[4] == nalcms[5] and igbp[7] == nalcms[8] and igbp[10] == nalcms[10]

    assert _manager(tmp_path / 'nalcms', 'GRU_ID,frac_10,frac_19').landcover_multipliers == nalcms
    assert _manager(tmp_path / 'forced', 'IGBP_10,IGBP_15',
                    MESH_LANDCOVER_CLASSIFICATION='nalcms').landcover_multipliers == nalcms


def test_multipliers_without_a_known_classification_are_an_error(tmp_path):
    with pytest.raises(MESHConfigurationError, match='MESH_LANDCOVER_CLASSIFICATION'):
        _manager(tmp_path / 'none')
    # an explicit classification or an explicit table needs no statistics file
    assert _manager(tmp_path / 'key', MESH_LANDCOVER_CLASSIFICATION='igbp').landcover_multipliers[15]
    custom = _manager(tmp_path / 'custom', MESH_LANDCOVER_MULTIPLIERS={'7': {'KSAT': 2.0}})
    assert custom.landcover_multipliers == {7: {'KSAT': 2.0}}
    # switched off: nothing is detected and nothing is applied
    off = _manager(tmp_path / 'off', MESH_USE_LANDCOVER_MULTIPLIERS=False)
    assert off.landcover_multipliers == {}
