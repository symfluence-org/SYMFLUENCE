# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Pin the NLDAS-2 v2.0 NetCDF variable names used by the acquisition handler.

The 2026-06 parity-validation sweep found the handler requesting legacy
GRIB-era names (TMP2m, SPFH2m, ...) that do not exist in the NLDAS-2 v2.0
NetCDF files at GES DISC — every OPeNDAP subset request returned HTTP 400.
The v2.0 names below were confirmed live against the dataset DDS.
"""
from __future__ import annotations

import pytest

from symfluence.core.modeling.variable_utils import VariableStandardizer
from symfluence.data.acquisition.handlers.nldas import (
    _DEFAULT_VARIABLES,
    _EXTENDED_VARIABLES,
)

pytestmark = [pytest.mark.unit, pytest.mark.data]

_LEGACY_GRIB_NAMES = {
    'TMP2m', 'SPFH2m', 'PRESsfc', 'UGRD10m', 'VGRD10m',
    'DLWRFsfc', 'DSWRFsfc', 'APCPsfc', 'CAPE180_0mb', 'PEVAPsfc', 'CONVfract',
}


def test_default_variables_are_v20_netcdf_names():
    assert _DEFAULT_VARIABLES == [
        'Tair', 'Qair', 'PSurf', 'Wind_E', 'Wind_N',
        'LWdown', 'SWdown', 'Rainf',
    ]


def test_extended_variables_are_v20_netcdf_names():
    assert _EXTENDED_VARIABLES == _DEFAULT_VARIABLES + [
        'CAPE', 'PotEvap', 'CRainf_frac',
    ]


def test_no_legacy_grib_names_remain():
    assert not _LEGACY_GRIB_NAMES & set(_EXTENDED_VARIABLES)


@pytest.mark.parametrize('alias', ['NLDAS', 'NLDAS2', 'NLDAS-2', 'nldas'])
def test_rename_map_covers_all_default_variables(alias):
    """Downstream variable standardization must accept the acquired names."""
    rename_map = VariableStandardizer().get_rename_map(alias)
    for var in _DEFAULT_VARIABLES:
        assert var in rename_map, f"{var} missing from {alias} rename map"


def test_rename_map_targets_standard_names():
    rename_map = VariableStandardizer().get_rename_map('NLDAS')
    assert rename_map['Tair'] == 'air_temperature'
    assert rename_map['Rainf'] == 'precipitation_flux'
    assert rename_map['SWdown'] == 'surface_downwelling_shortwave_flux'
    assert rename_map['LWdown'] == 'surface_downwelling_longwave_flux'
    assert rename_map['PSurf'] == 'surface_air_pressure'
    assert rename_map['Qair'] == 'specific_humidity'
    assert rename_map['Wind_E'] == 'eastward_wind'
    assert rename_map['Wind_N'] == 'northward_wind'
