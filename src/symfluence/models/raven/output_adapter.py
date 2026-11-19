# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven standard output-variable adapter (see core.modeling.output_variables).

Reads the ``:CustomOutput DAILY AVERAGE <VAR> BY_HRU`` files written by the
native Raven setup (``<VAR>_Daily_Average_ByHRU.csv``, mm per HRU) and the HRU
geometry ``settings/RAVEN/geometry.json`` (``hrus: [{id, area_km2, ice}]``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R

GLACIER_COMPONENTS = ('GLACIER_ICE', 'GLACIER', 'SNOW', 'SNOW_LIQ')


def read_by_hru(sim_dir: Path, variable: str) -> pd.DataFrame:
    """``*<VARIABLE>_Daily_Average_ByHRU.csv`` as a frame (columns = HRU ids)."""
    matches = sorted(Path(sim_dir).glob(f'{variable}_Daily_Average_ByHRU.csv')) or sorted(
        Path(sim_dir).glob(f'*_{variable}_Daily_Average_ByHRU.csv'))
    if not matches:
        raise EvaluationError(f'Raven custom output {variable}_Daily_Average_ByHRU.csv not in {sim_dir}')
    path = matches[0]
    header = pd.read_csv(path, header=None, nrows=1).iloc[0, 2:]
    ids = [int(float(h)) for h in header if pd.notna(h) and str(h).strip()]
    frame = pd.read_csv(path, skiprows=2, header=None)
    values = frame.iloc[:, 2:2 + len(ids)].to_numpy(float)
    return pd.DataFrame(values, index=pd.to_datetime(frame.iloc[:, 1]), columns=ids)


@R.output_adapters.add('RAVEN')
class RavenOutputAdapter(OutputVariableAdapter):
    """Raven per-HRU custom outputs + geometry.json."""

    MODEL_NAME = 'RAVEN'
    SUPPORTED = {
        'glacier_storage': 'GLACIER_ICE + GLACIER + SNOW + SNOW_LIQ (mm) on ice HRUs x HRU area',
        'glacier_area': 'sum of ice-HRU area_km2 in geometry.json (fixed geometry)',
        'swe': 'SNOW + SNOW_LIQ (mm), HRU-area weighted over all HRUs',
    }

    def _hrus(self) -> List[dict]:
        settings = self.settings_dir()
        path = settings / 'geometry.json' if settings else None
        if path is None or not path.exists():
            raise EvaluationError('Raven basin averages need settings/RAVEN/geometry.json')
        return json.loads(path.read_text())['hrus']

    def _weighted(self, frame: pd.DataFrame, weights: dict) -> pd.Series:
        missing = [h for h in weights if h not in frame.columns]
        if missing:
            raise EvaluationError(f'Raven output lacks HRUs {missing[:5]}')
        w = np.array([weights[h] for h in weights])
        return pd.Series(frame[list(weights)].to_numpy(float) @ w, index=frame.index)

    def _glacier_storage(self, sim_dir: Path) -> pd.Series:
        hrus = self._hrus()
        weights = {h['id']: float(h['area_km2']) for h in hrus if h['ice']}
        if not weights:
            raise EvaluationError('No ice HRUs in Raven geometry.json')
        total: Optional[pd.Series] = None
        for variable in GLACIER_COMPONENTS:
            try:
                part = self._weighted(read_by_hru(sim_dir, variable), weights)
            except EvaluationError:
                if variable == 'GLACIER_ICE':
                    raise
                continue
            total = part if total is None else total.add(part, fill_value=0.0)
        if total is None:
            raise EvaluationError('No Raven glacier storage output')
        return pd.Series(total * 1e3)  # mm km2 -> m3

    def _glacier_area(self, sim_dir: Path) -> pd.Series:
        area = sum(float(h['area_km2']) for h in self._hrus() if h['ice'])
        index = read_by_hru(sim_dir, 'GLACIER_ICE').index
        return pd.Series(area, index=index)

    def _swe(self, sim_dir: Path) -> pd.Series:
        hrus = self._hrus()
        total_area = sum(float(h['area_km2']) for h in hrus)
        weights = {h['id']: float(h['area_km2']) / total_area for h in hrus}
        swe = self._weighted(read_by_hru(sim_dir, 'SNOW'), weights)
        try:
            swe = swe.add(self._weighted(read_by_hru(sim_dir, 'SNOW_LIQ'), weights), fill_value=0.0)
        except EvaluationError:
            pass
        return swe
