import pytest
from symfluence.models.vic.constants import target_snow_constants


def test_preserves_other_constants_and_worker_isolation(tmp_path):
    original = tmp_path/'original.txt'
    original.write_text('SNOW_ROUGH 0.02\nSNOW_MIN_RAIN_TEMP -0.5\n')
    workers = [tmp_path/'a', tmp_path/'b']
    for p in workers:p.mkdir()
    for p, lo in zip(workers, [-.5, 1.4454]):
        text = target_snow_constants('CONSTANTS original.txt\nMIN_RAIN_TEMP 99\nMAX_SNOW_TEMP 99\n', tmp_path, p,
                                    {'SNOW_MIN_RAIN_TEMP': lo, 'SNOW_MAX_SNOW_TEMP': 2.4341})
        assert 'MIN_RAIN_TEMP' not in text
        assert str(p/'vic_calibration_constants.txt') in text
        saved = (p/'vic_calibration_constants.txt').read_text()
        assert 'SNOW_ROUGH 0.02' in saved
        assert float(saved.split('SNOW_MIN_RAIN_TEMP ')[1].split()[0]) == lo
    assert original.read_text() == 'SNOW_ROUGH 0.02\nSNOW_MIN_RAIN_TEMP -0.5\n'


def test_rejects_silent_legacy_activation_and_bad_order(tmp_path):
    with pytest.raises(ValueError, match='ignored'):
        target_snow_constants('MIN_RAIN_TEMP 1.4',tmp_path,tmp_path,{})
    with pytest.raises(ValueError, match='minimum'):
        target_snow_constants('',tmp_path,tmp_path,{'SNOW_MIN_RAIN_TEMP': 1.4})


def test_worker_routes_fitted_thresholds_to_constants(tmp_path):
    import xarray as xr
    from symfluence.models.vic.calibration.worker import VICWorker
    source=tmp_path/'source';source.mkdir()
    worker_dir=tmp_path/'worker';worker_dir.mkdir()
    params=worker_dir/'parameters';params.mkdir()
    nc=params/'vic_params.nc'
    xr.Dataset({'infilt': (('lat','lon'), [[.2]])}).to_netcdf(nc)
    (source/'vic_global.txt').write_text('MIN_RAIN_TEMP 99\nMAX_SNOW_TEMP 99\nRESULT_DIR old\nPARAMETERS old.nc\n')
    worker=VICWorker()
    assert worker._update_params_file(nc,{'min_rain_temp':1.4454,'max_snow_temp':2.4341})
    global_file=worker._get_or_create_global_file({},source,worker_dir,tmp_path/'results')
    content=global_file.read_text()
    assert 'CONSTANTS ' in content and 'MIN_RAIN_TEMP ' not in content
    constants=(worker_dir/'vic_calibration_constants.txt').read_text()
    values={k:float(v) for k,v in (line.split() for line in constants.splitlines())}
    assert values=={'SNOW_MIN_RAIN_TEMP':1.4454,'SNOW_MAX_SNOW_TEMP':2.4341}
