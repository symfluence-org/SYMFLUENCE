# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GR standard output-variable adapter (see core.modeling.output_variables).

The lumped GR glacier mode writes one glacier-wide balance per water year
(Oct–Sep) to ``GR_glacier_diagnostics.json`` (``mass_balance_mwe``, m w.e.).
They are returned as a cumulative specific balance stamped at 30 September,
starting from 0 at the 30 September before the first complete water year.
CemaNeige SWE stays inside ``GR_results.Rdata`` and is not exposed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from symfluence.core.exceptions import EvaluationError
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R

DIAGNOSTICS = 'GR_glacier_diagnostics.json'


@R.output_adapters.add('GR')
class GROutputAdapter(OutputVariableAdapter):
    """Lumped GR glacier balances from ``GR_glacier_diagnostics.json``."""

    MODEL_NAME = 'GR'
    SUPPORTED = {
        'glacier_mass_balance': f'{DIAGNOSTICS} mass_balance_mwe per water year (annual, 30 Sep stamps)',
        'glacier_area': f'{DIAGNOSTICS} glacier_area_km2_outline (constant outline area)',
    }

    def _diagnostics(self, sim_dir: Path) -> dict:
        path = Path(sim_dir) / DIAGNOSTICS
        if not path.exists():
            raise EvaluationError(f'{DIAGNOSTICS} not found in {sim_dir} (GR glacier mode off?)')
        return json.loads(path.read_text())

    def _glacier_mass_balance(self, sim_dir: Path) -> pd.Series:
        balances = self._diagnostics(sim_dir).get('mass_balance_mwe') or {}
        if not balances:
            raise EvaluationError(f'No mass_balance_mwe in {DIAGNOSTICS}')
        years = sorted(int(y) for y in balances)
        stamps = [pd.Timestamp(years[0] - 1, 9, 30)] + [pd.Timestamp(y, 9, 30) for y in years]
        values = [0.0]
        for year in years:
            values.append(values[-1] + float(balances[str(year)]) * 1000.0)
        return pd.Series(values, index=pd.DatetimeIndex(stamps))

    def _glacier_area(self, sim_dir: Path) -> pd.Series:
        diag = self._diagnostics(sim_dir)
        area = float(diag.get('glacier_area_km2_outline') or 0.0)
        if area <= 0:
            raise EvaluationError(f'No positive glacier_area_km2_outline in {DIAGNOSTICS}')
        return self._glacier_mass_balance(sim_dir) * 0.0 + area
