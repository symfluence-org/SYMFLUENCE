# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""VariableHandler must match CFIF-named forcing variables.

End-to-end run on the 09_large_domain config failed at FUSE
preprocessing with "Required variable temp not found in dataset CFIF"
because the ``DATASET_MAPPINGS['CFIF']`` entry only listed legacy
SUMMA short names (airtemp, pptrate, …) — but model-agnostic
preprocessing produces files with the canonical CF names
(air_temperature, precipitation_flux, …). Pin both forms.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.utils.variable_utils import VariableHandler

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _ds_with_cfif_canonical_names():
    times = pd.date_range("2015-01-01", periods=4, freq="h")
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars=dict(
            air_temperature=(("time",), rng.uniform(270, 290, 4).astype("float32")),
            precipitation_flux=(("time",), rng.uniform(0, 5e-5, 4).astype("float32")),
            specific_humidity=(("time",), rng.uniform(2e-3, 5e-3, 4).astype("float32")),
            wind_speed=(("time",), rng.uniform(1, 8, 4).astype("float32")),
            surface_downwelling_longwave_flux=(("time",), rng.uniform(200, 300, 4).astype("float32")),
            surface_downwelling_shortwave_flux=(("time",), rng.uniform(0, 200, 4).astype("float32")),
            surface_air_pressure=(("time",), rng.uniform(95000, 102000, 4).astype("float32")),
        ),
        coords=dict(time=times),
    )


def _ds_with_legacy_summa_names():
    times = pd.date_range("2015-01-01", periods=4, freq="h")
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars=dict(
            airtemp=(("time",), rng.uniform(270, 290, 4).astype("float32")),
            pptrate=(("time",), rng.uniform(0, 5e-5, 4).astype("float32")),
            spechum=(("time",), rng.uniform(2e-3, 5e-3, 4).astype("float32")),
            windspd=(("time",), rng.uniform(1, 8, 4).astype("float32")),
            LWRadAtm=(("time",), rng.uniform(200, 300, 4).astype("float32")),
            SWRadAtm=(("time",), rng.uniform(0, 200, 4).astype("float32")),
            airpres=(("time",), rng.uniform(95000, 102000, 4).astype("float32")),
        ),
        coords=dict(time=times),
    )


def _make_handler():
    return VariableHandler(
        config=MagicMock(),
        logger=logging.getLogger("test_cfif_handler"),
        dataset='CFIF',
        model='FUSE',
    )


def test_cfif_canonical_names_resolve():
    """The CFIF dataset_map must include canonical CF names so a
    file produced by the model-agnostic pipeline matches without
    relying on metadata attrs."""
    h = _make_handler()
    out = h.process_forcing_data(_ds_with_cfif_canonical_names())
    # FUSE's MODEL_REQUIREMENTS map to short names like 'temp', 'pr', ...
    # The exact short names depend on MODEL_REQUIREMENTS; what matters
    # is the call doesn't raise "Required variable not found".
    assert len(out.data_vars) > 0


def test_cfif_legacy_summa_names_still_resolve():
    """Older intermediate files use the SUMMA short names; the
    backwards-compat aliases keep them working."""
    h = _make_handler()
    out = h.process_forcing_data(_ds_with_legacy_summa_names())
    assert len(out.data_vars) > 0


def test_find_matching_variable_picks_canonical_first():
    """When both forms are available, _find_matching_variable should
    pick whichever matches on standard_name and exists in the data —
    canonical for canonical files, legacy for legacy files. No
    ordering dependency."""
    from symfluence.data.utils.variable_utils import VariableHandler
    h = VariableHandler(MagicMock(), logging.getLogger('t'), 'CFIF', 'FUSE')

    canonical_vars = {'air_temperature'}
    legacy_vars = {'airtemp'}

    cfif_map = h.DATASET_MAPPINGS['CFIF']
    canonical = h._find_matching_variable('air_temperature', cfif_map, canonical_vars)
    legacy = h._find_matching_variable('air_temperature', cfif_map, legacy_vars)

    assert canonical == 'air_temperature'
    assert legacy == 'airtemp'
