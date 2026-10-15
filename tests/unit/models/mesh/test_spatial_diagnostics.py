"""Native spatial output contracts, independent of aggregate basin success."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import xarray as xr

from symfluence.models.mesh.preprocessing.spatial_units import (
    SPATIAL_DIAGNOSTIC_FIELDS,
    check_spatial_diagnostics,
    prepare_spatial_diagnostics,
)


def test_compact_fields_are_enabled_without_replacing_custom_outputs(tmp_path):
    (tmp_path/'spatial_units.json').write_text('{}')
    options = tmp_path/'MESH_input_run_options.ini'
    options.write_text('2\nOUTFILESFLAG daily\nRUNMODE noroute\n')
    # Bundled inactive defaults must be preserved before enabling our fields.
    (tmp_path/'outputs_balance.txt').write_text('inactive template\n')
    prepare_spatial_diagnostics(tmp_path)
    assert (tmp_path/'outputs_balance.pre_spatial.txt').read_text() == 'inactive template\n'
    assert options.read_text() == '2\nOUTFIELDSFLAG on\nRUNMODE noroute\n'
    prepare_spatial_diagnostics(tmp_path)
    assert len((tmp_path/'outputs_balance.txt').read_text().splitlines()) == 7
    (tmp_path/'outputs_balance.txt').write_text('custom fields\n')
    with pytest.raises(ValueError, match='independently configured'):
        prepare_spatial_diagnostics(tmp_path)


def test_diagnostic_coverage_and_aggregation_are_required(tmp_path):
    (tmp_path/'spatial_diagnostics.json').write_text(json.dumps({}))
    (tmp_path/'results').mkdir()
    xr.Dataset({'GridArea': ('subbasin', [1., 3., 0.])}).to_netcdf(tmp_path/'MESH_drainage_database.nc')
    dates = pd.date_range('2013-01-01', periods=2)
    basin = pd.DataFrame({v: [2.5, 3.5] for v in ['PREC', 'RFF', 'ET']})
    for v in SPATIAL_DIAGNOSTIC_FIELDS:
        values = [[1., 3., 0.], [2., 4., 0.]] if v != 'FSNO' else [[.1, .3, 0.], [.2, .4, 0.]]
        pd.DataFrame(values, index=dates).to_csv(tmp_path/'results'/f'{v}_D.csv', header=False)
    check_spatial_diagnostics(tmp_path, basin, dates)
    path = tmp_path/'results/RFF_D.csv'
    pd.DataFrame([[3., 1., 0.], [4., 2., 0.]], index=dates).to_csv(path, header=False)
    with pytest.raises(ValueError, match='aggregation mismatch: RFF'):
        check_spatial_diagnostics(tmp_path, basin, dates)
    pd.DataFrame([[1., 3., 0.]], index=dates[:1]).to_csv(path, header=False)
    with pytest.raises(ValueError, match='Incomplete or misaligned'):
        check_spatial_diagnostics(tmp_path, basin, dates)


def test_hourly_endpoint_contract_requires_native_hour_steps(tmp_path):
    (tmp_path/'spatial_units.json').write_text('{}')
    options = tmp_path/'MESH_input_run_options.ini'
    options.write_text('OUTFIELDSFLAG off\nTIMESTEPFLAG 60\n')
    prepare_spatial_diagnostics(tmp_path)
    prepare_spatial_diagnostics(tmp_path, cryosphere_endpoints=True)
    assert 'SNO h csv shedorder print_date' in (tmp_path/'outputs_balance.txt').read_text()
    prepare_spatial_diagnostics(tmp_path, cryosphere_endpoints=True)
    options.write_text(options.read_text().replace('TIMESTEPFLAG 60', 'TIMESTEPFLAG 30'))
    with pytest.raises(ValueError, match='60-minute'):
        prepare_spatial_diagnostics(tmp_path, cryosphere_endpoints=True)
