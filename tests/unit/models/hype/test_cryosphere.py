from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.models.hype.cryosphere import calculate_constraints, ensure_observations


@pytest.fixture
def cryo_case(tmp_path):
    tmp_path = tmp_path / 'domain_test'
    setup = tmp_path / 'settings/HYPE'
    obs = tmp_path / 'data/observations/cryosphere'
    output = tmp_path / 'results'
    for path in (setup, obs, output):
        path.mkdir(parents=True)
    pd.DataFrame({'subid':[1], 'area':[1e8], 'glacier_fraction':[.1]}).to_csv(setup/'GeoData.txt', sep='\t', index=False)
    (setup/'GeoClass.txt').write_text('!! classes\n1 1 1 0 0 0 0 3\n')
    dates = pd.date_range('2010-01-01', '2012-12-31')
    for variable, values in [('CFSC', np.full(len(dates), .18)),
                             ('GLCA', np.full(len(dates), 10.)),
                             ('GLCV', 1 - np.arange(len(dates))*2*10/(.85*1000*365))]:
        frame = pd.DataFrame({'DATE': dates, '1': values})
        (output/f'time{variable}.txt').write_text('!! test\n'+frame.to_csv(sep='\t', index=False))
    pd.DataFrame({'YYYY': dates.year, 'MM': dates.month, 'DD':dates.day,
                  'fsca_outside_glaciers':20.}).to_csv(obs/'modis_fractional_snow_cover_and_glacier_albedo.csv',sep=';',index=False)
    pd.DataFrame({'annual_net_MB':[-2., -2.], 'g_area_dyn':[10.,10.]}, index=[2011,2012]).to_csv(obs/'glacier_timeseries.csv',sep=';')
    return tmp_path, output, obs


def test_matching_units_area_and_complete_calibration_years(cryo_case):
    project, output, obs = cryo_case
    metrics = calculate_constraints(output, project, '2010-01-01, 2011-12-31')
    assert metrics['snow_score'] == pytest.approx(1.)
    assert metrics['glacier_score'] == pytest.approx(1.)
    assert metrics['glacier_balance_year_count'] == 1
    assert metrics['snow_observation_count'] == 730
    # Held-out values must have no effect, even if wildly inconsistent.
    mb = pd.read_csv(obs/'glacier_timeseries.csv',sep=';',index_col=0)
    mb.loc[2012, 'annual_net_MB'] = 999
    mb.to_csv(obs/'glacier_timeseries.csv',sep=';')
    snow = pd.read_csv(obs/'modis_fractional_snow_cover_and_glacier_albedo.csv',sep=';')
    snow.loc[snow.YYYY==2012, 'fsca_outside_glaciers'] = 99
    snow.to_csv(obs/'modis_fractional_snow_cover_and_glacier_albedo.csv',sep=';',index=False)
    assert calculate_constraints(output, project, '2010-01-01, 2011-12-31') == metrics


def test_missing_output_does_not_silently_drop_constraint(cryo_case):
    project, output, _ = cryo_case
    (output/'timeGLCV.txt').unlink()
    with pytest.raises(FileNotFoundError):
        calculate_constraints(output, project, '2010-01-01, 2011-12-31')


def test_auxiliary_archive_extracts_requested_basin(tmp_path, monkeypatch):
    import logging
    import zipfile

    from symfluence.data.observation.handlers import lamah_ice
    from symfluence.data.observation.handlers.lamah_ice import FILES
    def download(path, logger):
        with zipfile.ZipFile(path, 'w') as z:
            for subtree in FILES.values():
                z.writestr(f'lamah_ice/A_basins_total_upstrm/2_timeseries/{subtree}/ID_86.csv', 'selected')
                z.writestr(f'lamah_ice/A_basins_total_upstrm/2_timeseries/{subtree}/ID_87.csv', 'wrong')
    monkeypatch.setattr(lamah_ice, '_download_lamah_ice_zip', download)
    result = ensure_observations(tmp_path, 86, logging.getLogger('test'))
    assert all((result/name).read_text()=='selected' for name in FILES)


def test_worker_uses_composite_and_preserves_raw_flow_metrics(cryo_case, monkeypatch):
    from symfluence.core.metrics import StreamflowMetrics
    from symfluence.models.hype.calibration.worker import HYPEWorker
    project, output, _ = cryo_case
    dates = pd.date_range('2010-01-01', '2012-12-31')
    discharge = 10 + np.sin(np.arange(len(dates))/20.)
    (output/'timeCOUT.txt').write_text('!! discharge\n'+pd.DataFrame(
        {'DATE':dates, '1':discharge}).to_csv(sep='\t',index=False))
    monkeypatch.setattr(StreamflowMetrics, 'load_observations', lambda *a, **kw: (discharge, dates))
    config = dict(DOMAIN_NAME='test', SYMFLUENCE_DATA_DIR=str(project.parent),
                  CALIBRATION_PERIOD='2010-01-01, 2011-12-31', HYPE_CRYOSPHERE_CONSTRAINTS=True,
                  OPTIMIZATION_METRIC='COMPOSITE', COMPOSITE_METRIC={'KGE':.6, 'SNOW_SCORE':.2, 'GLACIER_SCORE':.2})
    worker = HYPEWorker(config)
    metrics = worker.calculate_metrics(output, config)
    assert metrics['kge'] == pytest.approx(1.)
    assert worker._extract_primary_score(metrics, config) == pytest.approx(1.)
    assert worker._extract_primary_score(dict(metrics, snow_score=0.), config) == pytest.approx(.8)
    (output/'timeGLCV.txt').unlink()
    failed = worker.calculate_metrics(output, config)
    assert worker._extract_primary_score(failed, config) == worker.penalty_score
