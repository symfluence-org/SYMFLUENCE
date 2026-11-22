# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PRMS / GSFLOW standard output-variable adapter (see core.modeling.output_variables).

Both models write the same ``statvar`` text file. ``basin_*`` variables are
basin area-weighted means in inches (or inches/day); HRU areas (acres) and
glacier fractions come from the parameter file for glacier storage and snow
cover.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R

INCH_TO_MM = 25.4
ACRE_TO_KM2 = 1.0 / 247.105381467


def read_statvar_elements(path: Path) -> pd.DataFrame:
    """statvar values with columns ``<name>_<element>`` (e.g. ``snowcov_area_3``).

    The header lines are ``name element`` pairs (one per written element), as
    the PRMS cryosphere constraints read them.
    """
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    names = ['_'.join(line.split()[:2]) for line in lines[1:n + 1]]
    values = np.loadtxt(path, skiprows=n + 1, ndmin=2)
    if values.size == 0:
        raise EvaluationError(f'statvar {path} has no data rows')
    dates = pd.to_datetime(dict(year=values[:, 1].astype(int), month=values[:, 2].astype(int),
                                day=values[:, 3].astype(int)))
    return pd.DataFrame(values[:, 7:7 + n], columns=names, index=pd.DatetimeIndex(dates))


def read_parameter_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Parameter arrays from a PRMS parameter file (``** Parameters **`` section)."""
    text = Path(path).read_text()
    if '** Parameters **' not in text:
        raise EvaluationError(f'{path} is not a PRMS parameter file')
    params: Dict[str, np.ndarray] = {}
    for block in text.split('** Parameters **')[1].split('####'):
        rows = block.strip().splitlines()
        if len(rows) < 4:
            continue
        try:
            n_dims = int(rows[1])
            params[rows[0].split()[0]] = np.asarray(rows[4 + n_dims:], float)
        except (ValueError, IndexError):
            continue
    return params


@R.output_adapters.add('PRMS')
class PRMSOutputAdapter(OutputVariableAdapter):
    """PRMS ``statvar*`` output."""

    MODEL_NAME = 'PRMS'
    PARAMETER_FILES = ('params.dat',)
    SUPPORTED = {
        'swe': 'basin_pweqv (inches) x 25.4',
        'et': 'basin_actet (inches/day, daily runs) x 25.4',
        'glacier_storage': 'basin_gl_storage (basin-mean inches) x basin area (hru_area); glacier reservoir',
        'glacier_area': 'hru_area x glacier_frac_init from the parameter file (fixed geometry)',
        'snow_covered_fraction': 'snowcov_area_i weighted by ice-free HRU area',
    }

    def _statvar(self, sim_dir: Path) -> pd.DataFrame:
        path = next(iter(sorted(Path(sim_dir).glob('statvar*'))), None)
        if path is None:
            raise EvaluationError(f'No statvar file in {sim_dir}')
        return read_statvar_elements(path)

    def _parameter_file(self) -> Optional[Path]:
        settings = self.settings_dir()
        if settings is None:
            return None
        configured = None
        if isinstance(self.config, dict):
            configured = self.config.get(f'{self.MODEL_NAME}_PARAMETER_FILE')
        for name in ([configured] if configured else []) + list(self.PARAMETER_FILES):
            if (settings / name).exists():
                return settings / name
        return next(iter(sorted(settings.glob('*param*'))), None)

    def _geometry(self):
        path = self._parameter_file()
        if path is None:
            raise EvaluationError(f'No {self.MODEL_NAME} parameter file for HRU areas')
        params = read_parameter_arrays(path)
        if 'hru_area' not in params:
            raise EvaluationError(f'hru_area missing from {path}')
        area = params['hru_area'] * ACRE_TO_KM2
        ice = params.get('glacier_frac_init', np.zeros_like(area))
        if ice.shape != area.shape:
            ice = np.zeros_like(area)
        return area, ice

    def _column(self, frame: pd.DataFrame, name: str) -> pd.Series:
        for key in (name, f'{name}_1'):
            if key in frame.columns:
                return frame[key]
        raise EvaluationError(f'{name} not in {self.MODEL_NAME} statvar output')

    def _swe(self, sim_dir: Path) -> pd.Series:
        return self._column(self._statvar(sim_dir), 'basin_pweqv') * INCH_TO_MM

    def _et(self, sim_dir: Path) -> pd.Series:
        return self._column(self._statvar(sim_dir), 'basin_actet') * INCH_TO_MM

    def _glacier_storage(self, sim_dir: Path) -> pd.Series:
        area, _ = self._geometry()
        inches = self._column(self._statvar(sim_dir), 'basin_gl_storage')
        return inches * 0.0254 * float(area.sum()) * 1e6

    def _glacier_area(self, sim_dir: Path) -> pd.Series:
        area, ice = self._geometry()
        frame = self._statvar(sim_dir)
        return pd.Series(float(np.dot(area, ice)), index=frame.index)

    def _snow_covered_fraction(self, sim_dir: Path) -> pd.Series:
        area, ice = self._geometry()
        frame = self._statvar(sim_dir)
        cols = [f'snowcov_area_{i + 1}' for i in range(len(area))]
        if not all(c in frame.columns for c in cols):
            raise EvaluationError('snowcov_area for every HRU is not in the statvar output')
        weights = area * (1.0 - ice)
        return pd.Series(frame[cols].to_numpy() @ weights / weights.sum(), index=frame.index).clip(0, 1)


@R.output_adapters.add('GSFLOW')
class GSFLOWOutputAdapter(PRMSOutputAdapter):
    """GSFLOW ``statvar*`` output (PRMS land-surface part)."""

    MODEL_NAME = 'GSFLOW'
    PARAMETER_FILES = ('params.dat',)
