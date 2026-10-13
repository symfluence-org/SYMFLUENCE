# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Pin the CDS request variable lists for the CARRA/CERRA handlers.

The two regional reanalyses name their downwelling-longwave forecast
variable DIFFERENTLY in the CDS catalogs:

* CARRA: ``thermal_surface_radiation_downwards``
* CERRA: ``surface_thermal_radiation_downwards``  (ERA5-style)

CDS silently drops unknown variable names from a request, so getting this
split wrong does not error — it returns a file without longwave, which then
hard-fails the handler's required-variable validation (native CERRA was
unable to acquire at all until this was fixed; parity campaign exp12,
2026-06). These tests pin the full request variable lists so the name split
can never regress silently.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.cds_datasets import (
    CARRAAcquirer,
    CERRAAcquirer,
)

BBOX = "47.0/8.0/46.0/9.0"  # north/west/south/east


def _make(cls):
    config = {
        'DOMAIN_NAME': 'test_domain',
        'BOUNDING_BOX_COORDS': BBOX,
        'EXPERIMENT_TIME_START': '2018-06-01 00:00',
        'EXPERIMENT_TIME_END': '2018-06-03 00:00',
    }
    return cls(config, MagicMock())


@pytest.fixture
def carra():
    return _make(CARRAAcquirer)


@pytest.fixture
def cerra():
    return _make(CERRAAcquirer)


def test_converted_forcing_has_rate_and_specific_humidity_units(carra):
    ds = xr.Dataset({
        'air_temperature': ('time', [283.15], {'units': 'K'}),
        'relative_humidity': ('time', [75.0], {'units': '%'}),
        'surface_air_pressure': ('time', [100000.0], {'units': 'Pa'}),
        'precipitation_flux': ('time', [3.6], {'units': 'kg m-2'}),
        'surface_downwelling_shortwave_flux': ('time', [360000.0], {'units': 'J m-2'}),
        'surface_downwelling_longwave_flux': ('time', [1080000.0], {'units': 'J m-2'}),
    })
    with xr.set_options(keep_attrs=True):
        result = carra._convert_units(carra._calculate_derived_variables(ds))
    assert result.specific_humidity.attrs['units'] == 'kg kg-1'
    assert 0.005 < float(result.specific_humidity[0]) < 0.006
    for name, expected in [('surface_downwelling_shortwave_flux', 100.0),
                           ('surface_downwelling_longwave_flux', 300.0)]:
        assert result[name].attrs['units'] == 'W m-2'
        np.testing.assert_allclose(result[name], [expected])
    np.testing.assert_allclose(result.precipitation_flux, [0.001])


class TestForecastVariableNameSplit:
    """The CARRA/CERRA longwave name split must never regress."""

    def test_carra_forecast_variables_pinned(self, carra):
        assert carra._get_forecast_variables() == [
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "thermal_surface_radiation_downwards",
        ]

    def test_cerra_forecast_variables_pinned(self, cerra):
        assert cerra._get_forecast_variables() == [
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
        ]

    def test_carra_uses_thermal_first_longwave_name(self, carra):
        assert "thermal_surface_radiation_downwards" in carra._get_forecast_variables()
        assert "surface_thermal_radiation_downwards" not in carra._get_forecast_variables()

    def test_cerra_uses_era5_style_longwave_name(self, cerra):
        assert "surface_thermal_radiation_downwards" in cerra._get_forecast_variables()
        assert "thermal_surface_radiation_downwards" not in cerra._get_forecast_variables()


class TestAnalysisVariablesPinned:
    """Pin the analysis-product variable lists against the CDS catalogs."""

    def test_carra_analysis_variables(self, carra):
        assert carra._get_analysis_variables() == [
            "2m_temperature",
            "2m_relative_humidity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
        ]

    def test_cerra_analysis_variables(self, cerra):
        assert cerra._get_analysis_variables() == [
            "2m_temperature",
            "2m_relative_humidity",
            "surface_pressure",
            "10m_wind_speed",
        ]


class TestBuiltRequests:
    """The variable lists must flow unmodified into the actual CDS requests."""

    def test_cerra_forecast_request_carries_correct_longwave(self, cerra):
        req = cerra._build_forecast_request(
            ['2018'], ['06'], ['01'], ['00:00']
        )
        assert req['variable'] == [
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
        ]
        assert req['product_type'] == 'forecast'
        assert req['leadtime_hour'] == ['1']
        assert req['data_type'] == 'reanalysis'

    def test_carra_forecast_request_carries_correct_longwave(self, carra):
        req = carra._build_forecast_request(
            ['2018'], ['06'], ['01'], ['00:00']
        )
        assert req['variable'] == [
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "thermal_surface_radiation_downwards",
        ]
        assert req['domain'] == 'west_domain'

    def test_dataset_names(self, carra, cerra):
        assert carra._get_dataset_name() == "reanalysis-carra-single-levels"
        assert cerra._get_dataset_name() == "reanalysis-cerra-single-levels"
