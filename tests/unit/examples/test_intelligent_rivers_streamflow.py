"""Check the workshop's real data, daily weather and native training split."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

WORKSHOP = Path(__file__).resolve().parents[3] / 'examples' / '04_workshop_notebooks'
spec = importlib.util.spec_from_file_location('rivers_workshop', WORKSHOP / 'intelligent_rivers_streamflow.py')
workshop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workshop)


def test_bundled_observed_flow_and_boundary():
    basin, flow, provenance = workshop.load_bundle(WORKSHOP / 'data' / 'camels_ind_15007')
    assert len(flow) == 5844
    assert flow.loc['2000':'2015'].notna().all()
    assert (flow == 0).any()  # Measured zeros stay distinct from missing values.
    assert provenance['observations_source'].endswith('column 15007')
    assert basin.is_valid.all()
    assert 550 < basin.to_crs(basin.estimate_utm_crs()).area.sum() / 1e6 < 560


def make_hourly(project):
    folder = project / 'data' / 'forcing' / 'SUMMA_input'
    folder.mkdir(parents=True)
    settings = project / 'settings' / 'SUMMA'
    settings.mkdir(parents=True)
    (settings / 'forcingFileList.txt').write_text('"weather.nc" ! workshop input\n')
    times = pd.date_range('2000-01-01', periods=48, freq='h')
    hourly = pd.DataFrame({name: np.ones(48) for name in workshop.FORCING_UNITS}, index=times)
    hourly.airtemp = 273.15 + np.arange(48)
    hourly.pptrate = 1 / 86400
    path = folder / 'weather.nc'
    workshop.write_daily_forcing(hourly, path)
    return path


def test_daily_precipitation_conserves_water_and_keeps_seven_inputs(tmp_path):
    make_hourly(tmp_path)
    config = SimpleNamespace(domain=SimpleNamespace(time_start='2000-01-01 00:00', time_end='2000-01-02 23:00'),
                             forcing=SimpleNamespace(time_step_size=3600))
    daily, hashes = workshop.daily_summa_forcing(tmp_path, config)
    np.testing.assert_allclose(daily.pptrate * 86400, [1, 1])
    np.testing.assert_allclose(daily.airtemp, [284.65, 308.65])
    assert len(hashes) == 1
    output = tmp_path / 'daily.nc'
    workshop.write_daily_forcing(daily, output)
    with xr.open_dataset(output) as ds:
        assert set(ds.data_vars) == set(workshop.FORCING_UNITS)
        assert ds.sizes == {'time': 2, 'hru': 1}


def test_weather_gap_is_rejected_instead_of_compressing_time(tmp_path):
    path = make_hourly(tmp_path)
    with xr.open_dataset(path) as ds:
        damaged = ds.isel(time=[i for i in range(48) if i != 12]).load()
    damaged.to_netcdf(path)
    config = SimpleNamespace(domain=SimpleNamespace(time_start='2000-01-01 00:00', time_end='2000-01-02 23:00'),
                             forcing=SimpleNamespace(time_step_size=3600))
    with pytest.raises(ValueError, match='no missing timesteps'):
        workshop.daily_summa_forcing(tmp_path, config)


def test_comparison_uses_identical_dates_for_all_predictors():
    days = pd.date_range('2011-01-01', periods=4)
    frame = pd.DataFrame({'Observed': [0, 1, 2, 3], 'SUMMA': [0, np.nan, 2, 3],
                          'LSTM': [-1, 1, np.inf, 3]}, index=days)
    scores, paired = workshop.comparison_scores(frame, '2011-01-01,2011-01-04')
    assert paired.index.tolist() == [days[0], days[3]]
    assert scores.paired_days.eq(2).all()
    assert scores.loc['LSTM', 'negative_days'] == 1
    assert scores.loc['SUMMA', 'NSE'] == 1


def test_native_lstm_smoke_and_scalers_exclude_validation_and_holdout(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.models.lstm.runner import LSTMRunner
    import logging

    monkeypatch.setenv('SYMFLUENCE_LSTM_DEVICE', 'cpu')
    cfg = SymfluenceConfig.from_file(WORKSHOP / 'config_ayilam_streamflow.yaml', use_env=False, overrides={
        'SYMFLUENCE_DATA_DIR': str(tmp_path), 'HYDROLOGICAL_MODEL': 'LSTM',
        'EXPERIMENT_ID': 'workshop_test', 'EXPERIMENT_TIME_END': '2000-12-31 23:00',
        'SPINUP_PERIOD': '2000-01-01,2000-01-31',
        'CALIBRATION_PERIOD': '2000-02-01,2000-06-30',
        'EVALUATION_PERIOD': '2000-07-01,2000-12-31',
        'LSTM_HIDDEN_SIZE': 8, 'LSTM_LOOKBACK': 5, 'LSTM_EPOCHS': 1,
    })
    project = tmp_path / f'domain_{cfg.domain.name}'
    project.mkdir()
    days = pd.date_range('2000-01-01', '2000-12-31')
    # Synthetic meteorology is a test fixture, never a notebook fallback.
    daily = pd.DataFrame({name: 1 + np.arange(len(days)) / 100 for name in workshop.FORCING_UNITS}, index=days)
    daily.loc['2000-07-01':, 'airtemp'] = 1e5  # Must not affect fitted scalers.
    weather_file = project / 'daily' / 'weather.nc'
    workshop.write_daily_forcing(daily, weather_file)
    _, real_flow, _ = workshop.load_bundle(WORKSHOP / 'data' / 'camels_ind_15007')
    obs_dir = project / 'data' / 'observations' / 'streamflow' / 'preprocessed'
    obs_dir.mkdir(parents=True)
    real_flow.reindex(days).rename_axis('datetime').to_csv(obs_dir / f'{cfg.domain.name}_streamflow_processed.csv')
    runner = LSTMRunner(cfg, logging.getLogger('test.rivers'))
    runner.preprocessor.forcing_basin_path = weather_file.parent
    runner.run_lstm()
    fit_days = days[(days >= '2000-02-01') & (days <= '2000-06-30')]
    scaler_days = fit_days[:int(0.8 * len(fit_days))]
    np.testing.assert_allclose(runner.preprocessor.feature_scaler.mean_, daily.loc[scaler_days].mean())
    np.testing.assert_allclose(runner.preprocessor.target_scaler.mean_[0], real_flow.loc[scaler_days].mean())
    output = project / 'simulations' / 'workshop_test' / 'LSTM' / 'workshop_test_LSTM_output.nc'
    with xr.open_dataset(output) as ds:
        prediction = ds.predicted_streamflow.sel(time=slice('2000-07-01', '2000-12-31'))
        assert prediction.sizes['time'] == 184
        assert np.isfinite(prediction).all()
