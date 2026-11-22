# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow standard output-variable adapter (see core.modeling.output_variables).

``output.csv`` columns ``SWE`` (``snowpack__leq_depth``) and ``Ice``
(``glacier_ice__leq_depth``) are grid means in mm. Wflow stamps states at the
end of the step, so the index is shifted back one day (as the Wflow
cryosphere constraints do). Basin and glacier areas come from
``settings/WFLOW/wflow_staticmaps.nc``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R


@R.output_adapters.add('WFLOW')
class WflowOutputAdapter(OutputVariableAdapter):
    """Wflow ``output.csv`` + static maps."""

    MODEL_NAME = 'WFLOW'
    SUPPORTED = {
        'swe': 'output.csv SWE (snowpack__leq_depth, grid mean mm)',
        'glacier_storage': 'output.csv Ice (glacier_ice__leq_depth, grid mean mm) x basin area; bare ice only',
        'glacier_area': 'wflow_cellarea x wflow_glacierfrac in wflow_staticmaps.nc (fixed)',
    }

    def _csv(self, sim_dir: Path, column: str) -> pd.Series:
        path = Path(sim_dir) / 'output.csv'
        if not path.exists():
            raise EvaluationError(f'Wflow output.csv not in {sim_dir}')
        frame = pd.read_csv(path)
        if column not in frame.columns:
            raise EvaluationError(f'Column {column} not in Wflow output.csv')
        index = pd.DatetimeIndex(pd.to_datetime(frame['time'])) - pd.Timedelta(days=1)
        return pd.Series(frame[column].to_numpy(float), index=index)

    def _areas(self) -> Tuple[float, float]:
        import xarray as xr
        settings = self.settings_dir()
        path = settings / 'wflow_staticmaps.nc' if settings else None
        if path is None or not path.exists():
            raise EvaluationError('Wflow glacier storage needs settings/WFLOW/wflow_staticmaps.nc')
        with xr.open_dataset(path) as maps:
            mask = np.isfinite(maps['wflow_subcatch'].values)
            area = maps['wflow_cellarea'].values[mask] / 1e6
            frac = maps['wflow_glacierfrac'].values[mask] if 'wflow_glacierfrac' in maps else np.zeros_like(area)
        return float(area.sum()), float(np.dot(area, frac))

    def _swe(self, sim_dir: Path) -> pd.Series:
        return self._csv(sim_dir, 'SWE')

    def _glacier_storage(self, sim_dir: Path) -> pd.Series:
        basin_km2, _ = self._areas()
        return self._csv(sim_dir, 'Ice') * basin_km2 * 1e3  # mm km2 -> m3

    def _glacier_area(self, sim_dir: Path) -> pd.Series:
        _, glacier_km2 = self._areas()
        return self._csv(sim_dir, 'Ice') * 0.0 + glacier_km2
