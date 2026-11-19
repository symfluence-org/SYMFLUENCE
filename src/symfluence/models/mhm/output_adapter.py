# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mHM standard output-variable adapter (see core.modeling.output_variables).

Reads ``mHM_Fluxes_States*.nc``. The default SYMFLUENCE ``mhm_outputs.nml``
writes only total runoff; enable the snowpack and aET flags
(``outputFlxState``) for these variables to exist. Cells are averaged with
equal weight (mHM's regular grid); mHM has no glacier output.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R


@R.output_adapters.add('MHM')
class MHMOutputAdapter(OutputVariableAdapter):
    """mHM gridded fluxes/states NetCDF."""

    MODEL_NAME = 'MHM'
    SUPPORTED = {
        'swe': 'snowpack / SWE (mm), cell mean (needs outputFlxState snowpack on)',
        'et': 'aET (mm per output step; daily output -> mm/d), cell mean (needs outputFlxState aET on)',
    }

    def _grid_mean(self, sim_dir: Path, names) -> pd.Series:
        import xarray as xr
        for path in sorted(Path(sim_dir).glob('*Fluxes_States*.nc')):
            with xr.open_dataset(path) as ds:
                for name in names:
                    if name in ds.variables:
                        da = ds[name].astype(float)
                        da = da.where(da > -9998)
                        dims = [d for d in da.dims if d != 'time']
                        return da.mean(dims, skipna=True).to_series()
        raise EvaluationError(f'None of {names} in mHM fluxes/states output under {sim_dir}')

    def _swe(self, sim_dir: Path) -> pd.Series:
        return self._grid_mean(sim_dir, ('snowpack', 'SWE'))

    def _et(self, sim_dir: Path) -> pd.Series:
        return self._grid_mean(sim_dir, ('aET',))
