# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE standard output-variable adapter (see core.modeling.output_variables).

Reads HYPE ``time<VAR>.txt`` basin-output files (one column per subid).
Sub-basin areas come from ``settings/HYPE/GeoData.txt`` (``area`` in m²);
without it only single-subid output can be averaged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter, area_weighted_mean
from symfluence.core.registries import R

#: HYPE's fixed glacier ice density used to convert GLCV (km3 ice) to water equivalent
GLACIER_ICE_DENSITY = 0.85


def read_time_output(sim_dir: Path, variable: str) -> pd.DataFrame:
    """``time<VARIABLE>.txt`` as a float frame indexed by date (columns = subids)."""
    path = Path(sim_dir) / f'time{variable}.txt'
    if not path.exists():
        raise EvaluationError(f'HYPE output {path.name} not found in {sim_dir}')
    with open(path, encoding='utf-8') as f:
        first = f.readline()
    skip = 0 if first.strip().upper().startswith('DATE') else 1
    frame = pd.read_csv(path, sep=r'\s+', skiprows=skip)
    date_col = next(c for c in frame.columns if str(c).upper() == 'DATE')
    frame = frame.set_index(pd.to_datetime(frame.pop(date_col)))
    frame.columns = [str(c) for c in frame.columns]
    return frame.astype(float).where(lambda f: f > -9998)


@R.output_adapters.add('HYPE')
class HYPEOutputAdapter(OutputVariableAdapter):
    """HYPE ``timeXXXX.txt`` outputs; request them in info.txt ``timeoutput variable``."""

    MODEL_NAME = 'HYPE'
    SUPPORTED = {
        'glacier_storage': 'timeGLCV.txt (km3 ice, x0.85 -> w.e.), summed over subids; bare ice only',
        'glacier_area': 'timeGLCA.txt (km2), summed over subids',
        'swe': 'timeSNOW.txt (mm), subid-area weighted',
        'snow_covered_fraction': 'timeCFSC.txt (fraction of land), re-normalised to the ice-free area when GLCA is written',
        'et': 'timeEVAP.txt (mm/d), subid-area weighted',
    }

    def _areas(self, columns) -> Dict[str, float]:
        settings = self.settings_dir()
        geo = settings / 'GeoData.txt' if settings else None
        if geo is not None and geo.exists():
            frame = pd.read_csv(geo, sep=r'\s+')
            frame['subid'] = frame['subid'].astype(int).astype(str)
            return dict(zip(frame['subid'], frame['area'].astype(float) / 1e6))
        if len(columns) == 1:
            return {columns[0]: 1.0}
        raise EvaluationError('HYPE basin averages need settings/HYPE/GeoData.txt for sub-basin areas')

    def _basin_mean(self, sim_dir: Path, variable: str) -> pd.Series:
        frame = read_time_output(sim_dir, variable)
        return area_weighted_mean(frame, self._areas(list(frame.columns)))

    def _glacier_storage(self, sim_dir: Path) -> pd.Series:
        return read_time_output(sim_dir, 'GLCV').sum(axis=1, min_count=1) * GLACIER_ICE_DENSITY * 1e9

    def _glacier_area(self, sim_dir: Path) -> pd.Series:
        return read_time_output(sim_dir, 'GLCA').sum(axis=1, min_count=1)

    def _swe(self, sim_dir: Path) -> pd.Series:
        return self._basin_mean(sim_dir, 'SNOW')

    def _et(self, sim_dir: Path) -> pd.Series:
        return self._basin_mean(sim_dir, 'EVAP')

    def _snow_covered_fraction(self, sim_dir: Path) -> pd.Series:
        frame = read_time_output(sim_dir, 'CFSC')
        areas = self._areas(list(frame.columns))
        cols = [c for c in frame.columns if c in areas]
        weights = pd.Series({c: areas[c] for c in cols})
        covered = frame[cols].mul(weights, axis=1).sum(axis=1, min_count=1)
        land = pd.Series(float(weights.sum()), index=frame.index)
        glacier: Optional[pd.Series] = None
        if (Path(sim_dir) / 'timeGLCA.txt').exists():
            glacier = read_time_output(sim_dir, 'GLCA').reindex(frame.index).sum(axis=1)
            land = land - glacier
        land = land.where(land > 0)
        return (covered / land).clip(0, 1)
