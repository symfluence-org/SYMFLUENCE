from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ValidationError
from symfluence.models.raven.adapter import discharge
from symfluence.models.raven.balance import check
from symfluence.models.utilities.elevation_forcing import upper_ice_fractions


def test_ice_allocation_conserves_area():
    area = np.array([50.0, 30.0, 20.0])
    ice = upper_ice_fractions(area, [500, 1000, 1500], 0.25)
    np.testing.assert_allclose(ice, [0, 1 / 6, 1])
    assert np.dot(area, ice) == pytest.approx(25)


def test_discharge_is_interval_start_labelled(tmp_path):
    (tmp_path / "Hydrographs.csv").write_text(
        "time,date,hour,precip [mm/day],Tungnaa [m3/s]\n0,2009-07-01,00:00:00,0,0\n1,2009-07-02,00:00:00,1,12\n2,2009-07-03,00:00:00,0,14\n"
    )
    q = discharge(tmp_path)
    assert list(q.index) == list(pd.date_range("2009-07-01", periods=2))
    assert list(q) == [12, 14]


def test_stock_raven_cumulative_loss_is_rejected(tmp_path):
    (tmp_path / "geometry.json").write_text(json.dumps({"hrus": [{"ice": True, "area_km2": 1}], "ice_mm": 100000}))
    (tmp_path / "WatershedStorage.csv").write_text("MB Error [mm],Glacier Ice [mm]\n0,0\n0,-1\n")
    with pytest.raises(ValidationError, match="initialization"):
        check(tmp_path, tmp_path)


def test_forcing_corrections_written_to_every_gauge(tmp_path):
    from symfluence.models.raven import native

    bands = [dict(lat=64.0, lon=-18.0, elevation=600.0), dict(lat=64.1, lon=-18.1, elevation=900.0)]
    hrus = [dict(id=1, band=0, ice=False, area_km2=1.0, **bands[0]), dict(id=2, band=1, ice=True, area_km2=1.0, **bands[1])]
    (tmp_path / "geometry.json").write_text(json.dumps(dict(hrus=hrus, bands=bands, forcing_dir="f", ice_mm=1.0)))
    native.parameters(tmp_path, dict(PRECIP_CORR=1.2, SNOW_CORR=0.8, TEMP_CORR=0.5))
    rvt = (tmp_path / "model.rvt").read_text()
    assert rvt.count(":RainCorrection 1.2\n") == 2
    assert rvt.count(f":SnowCorrection {1.2 * 0.8}\n") == 2
    assert rvt.count(":TemperatureCorrection 0.5\n") == 2
    native.parameters(tmp_path, {})
    rvt = (tmp_path / "model.rvt").read_text()
    assert ":RainCorrection 1.0\n" in rvt and ":SnowCorrection 1.0\n" in rvt and ":TemperatureCorrection 0.0\n" in rvt


def test_config_bounds_override_only_touches_calibrated_names(tmp_path):
    import logging

    from symfluence.models.raven.adapter import RavenParameterManager

    config = {
        "RAVEN_PARAMS_TO_CALIBRATE": "MELT_FACTOR,PRECIP_CORR",
        "RAVEN_PARAM_BOUNDS": {"MELT_FACTOR": [2.5, 7], "GLACIER_CORR": [1, 2]},
    }
    pm = RavenParameterManager(config, logging.getLogger("test"), tmp_path)
    bounds = pm.get_parameter_bounds()
    assert set(bounds) == {"MELT_FACTOR", "PRECIP_CORR"}
    assert (bounds["MELT_FACTOR"]["min"], bounds["MELT_FACTOR"]["max"]) == (2.5, 7)
    assert (bounds["PRECIP_CORR"]["min"], bounds["PRECIP_CORR"]["max"]) == (0.9, 1.5)
