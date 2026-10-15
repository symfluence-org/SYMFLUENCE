"""Conservation and preservation contracts for staged MESH ice surfaces."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.geospatial.glacier_geometry import build_glacier_manifest
from symfluence.models.mesh.preprocessing.class_file_manager import CLASSFileManager
from symfluence.models.mesh.preprocessing.glacier_writer import write_glacier_bundle


@pytest.fixture
def geometry(tmp_path):
    units = gpd.GeoDataFrame({'id': ['a', 'b']}, geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=3057)
    ice = gpd.GeoDataFrame(geometry=[box(7.5, 0, 20, 10)], crs=3057)
    path = tmp_path/'thickness.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=2, height=1, count=1,
                       dtype='float64', crs=3057, transform=from_origin(0, 10, 10, 10), nodata=-9999) as out:
        out.write(np.array([[20., 40.]]), 1)
    return units, ice, path


def test_partial_cells_preserve_outline_area_volume(geometry):
    units, ice, path = geometry
    report = build_glacier_manifest(units, ice, [path], id_column='id')
    a, b = report['units']
    assert a['glacier_fraction'] == .25
    assert a['ice_volume_m3'] == 500
    assert b['glacier_fraction'] == 1
    assert b['ice_volume_m3'] == 4000
    assert sum(x['ice_water_volume_m3'] for x in report['units']) == 4500*.917


def test_missing_thickness_never_turns_ice_into_land(geometry):
    units, ice, path = geometry
    with rasterio.open(path, 'r+') as out:
        out.write(np.array([[20., -9999]]), 1)
    with pytest.raises(ValueError, match='Missing positive thickness'):
        build_glacier_manifest(units, ice, [path], id_column='id')


def test_overlapping_thickness_rejected(geometry):
    units, ice, path = geometry
    with pytest.raises(ValueError, match='Overlapping thickness'):
        build_glacier_manifest(units, ice, [path, path], id_column='id')


def test_zero_ice_needs_no_thickness(geometry):
    units, ice, _ = geometry
    ice.geometry = [box(30, 0, 40, 10)]
    report = build_glacier_manifest(units, ice, [], id_column='id')
    assert all(row['glacier_fraction'] == 0 for row in report['units'])


@pytest.fixture
def bundle(tmp_path, geometry):
    units, ice, path = geometry
    manifest = build_glacier_manifest(units, ice, [path], id_column='id')
    source = tmp_path/'source'
    source.mkdir()
    blocks = CLASSFileManager._get_default_class_block()
    text = ['title', '64 -18 10 2 50 -1 1 1 2 04 DEGLAT/DEGLON/ZRFM/ZRFH/ZBLD/GC/ILW/NL/NM']
    for j in range(2):
        text += [line.replace('100 Default', f'{100+j} Default') for line in blocks]
    text += ['0 20 footer']
    (source/'MESH_parameters_CLASS.ini').write_text('\n'.join(text)+'\n')
    (source/'MESH_parameters_hydrology.ini').write_text(
        'GRU class dependent hydrologic parameters\n2 # Number\nZSNL .1 .2\nIWF 1 0\nFRZTH .1 # uncounted\n')
    xr.Dataset({'GRU': (('subbasin', 'NGRU'), [[.5, .5, 0.]]),
                'GridArea': ('subbasin', [200.], {'units': 'm ** 2'}),
                'Rank': ('subbasin', [1])}).to_netcdf(source/'MESH_drainage_database.nc')
    (source/'forcing.txt').write_text('unchanged forcing')
    return source, manifest


def write(bundle, destination):
    source, manifest = bundle
    return write_glacier_bundle(source, destination, manifest, unit_ids=['a', 'b'],
                                albedo_visible=.4, albedo_infrared=.25, temperature_c=-3)


def test_split_conserves_native_area_forcing_and_hydrology(bundle, tmp_path):
    source, _ = bundle
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    dest = tmp_path/'ice'
    report = write(bundle, dest)
    assert report['finite_ice_storage'] is False
    with xr.open_dataset(dest/'MESH_drainage_database.nc') as ds:
        np.testing.assert_allclose(ds.GRU, [[.375, .125, .5, 0]])
        assert ds.GridArea.item() == 200
        assert ds.Rank.item() == 1
    hydro = (dest/'MESH_parameters_hydrology.ini').read_text()
    assert 'ZSNL  .1  .1  .2' in hydro
    assert 'IWF  1  1  1' in hydro
    assert 'FRZTH .1 # uncounted' in hydro
    assert (dest/'forcing.txt').read_bytes() == before['forcing.txt']
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_ice_fields_survive_terrestrial_fixers_twice(bundle, tmp_path):
    dest = tmp_path/'ice'
    write(bundle, dest)
    path = dest/'MESH_parameters_CLASS.ini'
    manager = CLASSFileManager(path, logging.getLogger(__name__))
    before = path.read_text().splitlines()
    indices = manager._ice_line_indices(before)
    assert len(indices) == 30
    # Native CLASSBG already stores bulk ice heat capacity in HCPS. Pore ice
    # must stay zero; THIC=1 caused a first-step CLASSZ energy-balance abort.
    for i in indices:
        if '18 3xTHLQ' in before[i]:
            assert [float(x) for x in before[i].split()[:7]] == [0]*7
    for _ in range(2):
        manager.fix_initial_conditions(latitude=64)
        manager.fix_vegetation_parameters()
        manager.apply_field_overrides({'sand': [40, 40, 40], 'tbar': [5, 5, 5], 'cmas': 3})
        manager.fix_nm()
    after = path.read_text().splitlines()
    assert [after[i] for i in indices] == [before[i] for i in indices]
    assert any('40.0' in line for line in after if '3xSAND' in line)
    first = min(indices)
    with pytest.raises(ValueError, match='before splitting'):
        manager._create_elevation_adjusted_block(after[first:first+15], 0, 700, .25)


def test_wrong_area_and_live_destination_rejected(bundle, tmp_path):
    with pytest.raises(ValueError, match='destination must be'):
        write(bundle, bundle[0])
    bundle[1]['units'][0]['area_m2'] = 101
    bundle[1]['units'][0]['glacier_area_m2'] = 25.25
    bundle[1]['units'][0]['land_area_m2'] = 75.75
    with pytest.raises(ValueError, match='areas disagree'):
        write(bundle, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_zero_glacier_preserves_all_native_files(bundle, tmp_path):
    source, manifest = bundle
    for row in manifest['units']:
        row.update(glacier_fraction=0, glacier_area_m2=0, land_area_m2=row['area_m2'],
                   ice_volume_m3=0, ice_water_volume_m3=0)
    dest = tmp_path/'zero'
    write(bundle, dest)
    for path in source.iterdir():
        assert path.read_bytes() == (dest/path.name).read_bytes()


def test_ids_are_explicit_and_ordered(bundle, tmp_path):
    source, manifest = bundle
    manifest['units'].reverse()
    write(bundle, tmp_path/'ordered')
    with xr.open_dataset(tmp_path/'ordered'/'MESH_drainage_database.nc') as ds:
        np.testing.assert_allclose(ds.GRU, [[.375, .125, .5, 0]])
    with pytest.raises(ValueError, match='IDs must match'):
        write_glacier_bundle(source, tmp_path/'bad', manifest, unit_ids=['a', 'a'],
                             albedo_visible=.4, albedo_infrared=.25, temperature_c=-3)


def test_geographic_crs_and_overlapping_units_rejected(geometry):
    units, ice, path = geometry
    with pytest.raises(ValueError, match='projected metre'):
        build_glacier_manifest(units.to_crs(4326), ice.to_crs(4326), [path], id_column='id')
    units.geometry = [box(0, 0, 15, 10), box(10, 0, 20, 10)]
    with pytest.raises(ValueError, match='units overlap'):
        build_glacier_manifest(units, ice, [path], id_column='id')


def test_workflow_hook_installs_only_validated_inputs(bundle, tmp_path):
    import json
    from types import SimpleNamespace

    from symfluence.models.mesh.config_schema import MESHConfig
    from symfluence.models.mesh.preprocessor import MESHPreProcessor

    source, manifest = bundle
    path = tmp_path/'geometry.json'
    path.write_text(json.dumps(manifest))
    config = MESHConfig(glacier_mode='native_ice_surface', glacier_manifest=str(path),
                        glacier_unit_ids=['a', 'b'], glacier_albedo_visible=.4,
                        glacier_albedo_infrared=.25, glacier_temperature_c=-3)
    obj = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(mesh=config)),
                          forcing_dir=source, logger=logging.getLogger(__name__),
                          _get_config_value=lambda getter, **kwargs: getter())
    before = (source/'forcing.txt').read_bytes()
    MESHPreProcessor._prepare_glacier_inputs(obj)
    assert (source/'glacier_manifest.json').exists()
    assert (source/'forcing.txt').read_bytes() == before
    with xr.open_dataset(source/'MESH_drainage_database.nc') as ds:
        np.testing.assert_allclose(ds.GRU, [[.375, .125, .5, 0]])


def test_glacier_config_requires_explicit_inputs():
    from symfluence.models.mesh.config_schema import MESHConfig

    assert MESHConfig().glacier_mode == 'none'
    with pytest.raises(ValueError, match='requires a geometry manifest'):
        MESHConfig(glacier_mode='native_ice_surface')
    with pytest.raises(ValueError):
        MESHConfig(glacier_mode='finite_ice')


def test_finite_storage_uses_public_volume_per_ice_area(bundle, tmp_path):
    source, manifest = bundle
    dest = tmp_path/'finite'
    report = write_glacier_bundle(source, dest, manifest, unit_ids=['a', 'b'],
                                 albedo_visible=.4, albedo_infrared=.25, temperature_c=-3,
                                 finite_storage=True)
    lines = (dest/'MESH_glacier_storage.ini').read_text().splitlines()
    assert int(lines[0]) == 2
    masses = [float(line.split()[1]) for line in lines[1:]]
    np.testing.assert_allclose(masses, [20*917, 40*917])
    assert report['finite_ice_storage'] is True


def test_explicit_spatial_units_preserve_templates_and_forcing_identity(bundle, geometry, tmp_path):
    from symfluence.models.mesh.preprocessing.spatial_units import check_spatial_output, prepare_spatial_units

    source, manifest = bundle
    units, _, _ = geometry
    units['unit_id'] = ['a', 'b']
    units['template_mid'] = [100, 101]
    units['elevation'] = [600., 1200.]
    units['area_m2'] = units.area
    path = tmp_path/'units.gpkg'
    units.to_file(path)
    forcing = xr.Dataset({k: (('time','subbasin'), [[1., 2.], [3., 4.]])
                          for k in ['TA','PRES','QA','UV','PRE','FSIN','FLIN']},
                         coords={'time': np.array(['2013-01-01T00', '2013-01-01T01'], dtype='datetime64[h]'), 'subbasin': [1, 2]})
    forcing['unit_id'] = ('subbasin', ['a', 'b'])
    forcing_path = tmp_path/'forcing.nc'
    forcing.to_netcdf(forcing_path)
    (source/'MESH_input_run_options.ini').write_text('RUNMODE noroute\n')
    before = (source/'MESH_parameters_CLASS.ini').read_bytes()
    forcing['unit_id'] = ('subbasin', ['b', 'a'])
    forcing.to_netcdf(forcing_path, mode='w')
    with pytest.raises(ValueError, match='forcing identity/order mismatch'):
        prepare_spatial_units(source, path, forcing_path)
    assert (source/'MESH_parameters_CLASS.ini').read_bytes() == before
    forcing['unit_id'] = ('subbasin', ['a', 'b'])
    forcing.to_netcdf(forcing_path, mode='w')
    prepare_spatial_units(source, path, forcing_path)
    with xr.open_dataset(source/'MESH_drainage_database.nc') as db:
        np.testing.assert_allclose(db.GRU, [[1,0,0],[0,1,0],[0,0,0]])
        np.testing.assert_allclose(db.GridArea, [100,100,0])
        np.testing.assert_array_equal(db.Next, [2,3,0])
        np.testing.assert_allclose(db.DA, [100,200,200])
    with xr.open_dataset(source/'MESH_forcing_safe.nc') as actual:
        np.testing.assert_allclose(actual.TA.isel(subbasin=slice(0,2)), forcing.TA)
        np.testing.assert_array_equal(actual.time, forcing.time)
    with xr.open_dataset(source/'MESH_forcing_safe.nc', decode_times=False) as actual:
        assert actual.time.attrs['units'] == 'hours since 1900-01-01 00:00:00'
        assert actual.time.attrs['calendar'] == 'gregorian'
    text = (source/'MESH_parameters_CLASS.ini').read_text()
    assert '1001' in text and '1002' in text
    assert 'ZSNL .1 .2' in (source/'MESH_parameters_hydrology.ini').read_text()
    report = write_glacier_bundle(source, tmp_path/'spatial_ice', manifest,
                                 unit_ids=['a', 'b'], albedo_visible=.4,
                                 albedo_infrared=.25, temperature_c=-3, finite_storage=True)
    assert report['finite_ice_storage']
    (source/'results').mkdir(exist_ok=True)
    basin_output = source/'results/Basin_average_water_balance.csv'
    basin_output.write_text('YEAR,JDAY,PREC\n2013,1,21600\n')
    with pytest.raises(ValueError, match='area-weighted forcing'):
        check_spatial_output(source)
    basin_output.write_text('YEAR,JDAY,PREC\n2013,1,18000\n')
    check_spatial_output(source)


def test_missing_dummy_column_does_not_drop_last_land_cover(bundle):
    from symfluence.models.mesh.preprocessing.spatial_units import preserve_template_grus

    source, _ = bundle
    path = source/'MESH_drainage_database.nc'
    original = (source/'MESH_parameters_CLASS.ini').read_bytes()
    with xr.open_dataset(path) as f:
        stripped = f.isel(NGRU=slice(0, 2)).load()
    stripped.to_netcdf(path, mode='w')
    preserve_template_grus(source)
    with xr.open_dataset(path) as db:
        np.testing.assert_allclose(db.GRU, [[.5, .5, 0]])
    assert (source/'MESH_parameters_CLASS.ini').read_bytes() == original


def test_spatial_baseflow_refreshes_grid_overrides_after_calibration(tmp_path):
    from symfluence.models.mesh.preprocessing.spatial_units import prepare_spatial_baseflow

    (tmp_path/'spatial_units.json').write_text('{}')
    options = tmp_path/'MESH_input_run_options.ini'
    options.write_text('BASEFLOWFLAG wf_lzs\nINPUTPARAMSFORMFLAG txt # formats\n')
    hydro = tmp_path/'MESH_parameters_hydrology.ini'
    hydro.write_text('FLZ .001\nPWR 2.37\n')
    xr.Dataset({'lat': ('subbasin', [64., 65., 65.]),
                'lon': ('subbasin', [-19., -18., -18.]),
                'crs': ((), 1, {'grid_mapping_name': 'latitude_longitude'})}).to_netcdf(
                    tmp_path/'MESH_drainage_database.nc')
    prepare_spatial_baseflow(tmp_path)
    assert options.read_text().split().count('nc') == 1
    with xr.open_dataset(tmp_path/'MESH_parameters.nc') as p:
        np.testing.assert_allclose(p.flz, [.001]*3)
        np.testing.assert_allclose(p.pwr, [2.37]*3)
    hydro.write_text('FLZ .002\nPWR 2.1\n')
    prepare_spatial_baseflow(tmp_path)
    assert options.read_text().split().count('nc') == 1
    with xr.open_dataset(tmp_path/'MESH_parameters.nc') as p:
        np.testing.assert_allclose(p.flz, [.002]*3)
        np.testing.assert_allclose(p.pwr, [2.1]*3)
