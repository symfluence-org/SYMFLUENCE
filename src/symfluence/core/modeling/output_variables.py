# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-agnostic output-variable adapters.

Each model can declare how to read a small set of *standard variables* from
its own output files as **basin (area-weighted) time series in standard
units**. The model-agnostic evaluators (``GLACIER_MB``, ``BASIN_SWE``,
``BASIN_SCA``, ``BASIN_ET``) read model output only through these adapters, so
melt-versus-precipitation compensation can be checked the same way across
ensemble members.

Standard variables (name -> unit):

``glacier_storage``
    Total glacier water-equivalent storage in the basin, m³ w.e. What a
    model counts as glacier storage (bare ice only, or ice + firn + snow on
    the glacier) is listed in :data:`ADAPTER_NOTES` of each adapter; the
    difference cancels for multi-year geodetic periods but not for single
    seasons.
``glacier_area``
    Glacier area in the basin, km².
``glacier_mass_balance``
    Cumulative glacier-wide *specific* mass balance, mm w.e. (relative to the
    first time step). Derived from ``glacier_storage`` / ``glacier_area`` when
    a model provides those (``sum(dV_t / A_t)``), or supplied directly by
    models that only report balances (lumped GR glacier).
``swe``
    Basin-mean snow water equivalent, mm.
``snow_covered_fraction``
    Snow-covered fraction, 0–1. Where a model separates glacier ice from land
    it is the fraction of the ice-free area (the MODIS ``fsca_outside_glaciers``
    convention); otherwise the fraction of the basin.
``et``
    Basin-mean actual evapotranspiration, mm/d (positive = loss to the
    atmosphere).
``discharge``
    Outlet discharge, m³/s. Not served by adapters: the streamflow evaluator
    and the ``R.result_extractors`` registry already cover it.

Adapters register in ``R.output_adapters`` (keyed by model name) and raise
:class:`UnsupportedVariableError` for any variable they cannot provide.
"""
from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError

logger = logging.getLogger(__name__)

GLACIER_STORAGE = 'glacier_storage'
GLACIER_AREA = 'glacier_area'
GLACIER_MASS_BALANCE = 'glacier_mass_balance'
SWE = 'swe'
SNOW_COVERED_FRACTION = 'snow_covered_fraction'
ET = 'et'
DISCHARGE = 'discharge'

#: Standard variable -> unit.
STANDARD_VARIABLES: Dict[str, str] = {
    GLACIER_STORAGE: 'm3 w.e.',
    GLACIER_AREA: 'km2',
    GLACIER_MASS_BALANCE: 'mm w.e. (cumulative, specific)',
    SWE: 'mm',
    SNOW_COVERED_FRACTION: '1',
    ET: 'mm d-1',
    DISCHARGE: 'm3 s-1',
}

#: Variables served by adapters (discharge is served by the streamflow path).
ADAPTER_VARIABLES = (
    GLACIER_STORAGE, GLACIER_AREA, GLACIER_MASS_BALANCE,
    SWE, SNOW_COVERED_FRACTION, ET,
)


class UnsupportedVariableError(EvaluationError):
    """A model adapter cannot provide the requested standard variable."""


class OutputVariableAdapter(ABC):
    """Read standard variables from one model's native output directory.

    Subclasses implement ``_<variable>(sim_dir) -> pd.Series`` for each variable
    they support and list it in :attr:`SUPPORTED` (variable -> where it comes
    from). ``glacier_mass_balance`` is derived automatically from
    ``glacier_storage`` and ``glacier_area`` when both are supported.

    Args:
        project_dir: Domain directory (``<data>/domain_<name>``); adapters use
            it to find the model's settings (areas, HRU geometry).
        config: Flat config dict or typed config (optional).
    """

    MODEL_NAME: str = ''
    #: variable -> short description of the native source
    SUPPORTED: Mapping[str, str] = {}

    def __init__(self, project_dir: Optional[Path] = None, config: Any = None):
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.config = config

    # ------------------------------------------------------------------
    def supported_variables(self) -> List[str]:
        """Standard variables this adapter can provide (including derived ones)."""
        names = [v for v in ADAPTER_VARIABLES if v in self.SUPPORTED]
        if (GLACIER_MASS_BALANCE not in names and GLACIER_STORAGE in names
                and GLACIER_AREA in names):
            names.append(GLACIER_MASS_BALANCE)
        return names

    def supports(self, variable: str) -> bool:
        return variable in self.supported_variables()

    def extract(self, variable: str, sim_dir: Path) -> pd.Series:
        """Basin time series of ``variable`` in its standard unit.

        Raises:
            UnsupportedVariableError: the model does not provide the variable.
            EvaluationError (or a model error): the output could not be read.
        """
        if variable not in STANDARD_VARIABLES:
            raise UnsupportedVariableError(
                f"Unknown standard variable '{variable}'; expected one of "
                f"{sorted(STANDARD_VARIABLES)}")
        if not self.supports(variable):
            raise UnsupportedVariableError(
                f"{self.MODEL_NAME or type(self).__name__} adapter does not provide "
                f"'{variable}' (supported: {self.supported_variables() or 'none'})")
        if variable == GLACIER_MASS_BALANCE and GLACIER_MASS_BALANCE not in self.SUPPORTED:
            return cumulative_specific_balance(
                self.extract(GLACIER_STORAGE, sim_dir), self.extract(GLACIER_AREA, sim_dir))
        method: Callable[[Path], pd.Series] = getattr(self, f'_{variable}')
        series = method(Path(sim_dir))
        return clean_series(series, name=variable)

    # ------------------------------------------------------------------
    def settings_dir(self) -> Optional[Path]:
        """``<project>/settings/<MODEL>`` if it exists."""
        if self.project_dir is None:
            return None
        path = self.project_dir / 'settings' / self.MODEL_NAME
        return path if path.is_dir() else None


# ----------------------------------------------------------------------
# Helpers shared by adapters and evaluators
# ----------------------------------------------------------------------

def clean_series(series: pd.Series, name: Optional[str] = None) -> pd.Series:
    """Float series on a sorted, de-duplicated DatetimeIndex; fill values -> NaN."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s = s.where(s > -9998.0)
    if name:
        s.name = name
    return s


def area_weighted_mean(frame: pd.DataFrame, weights: Mapping[Any, float]) -> pd.Series:
    """Row-wise mean of ``frame`` columns weighted by ``weights`` (by column)."""
    cols = [c for c in frame.columns if c in weights]
    if not cols:
        raise EvaluationError('No output columns match the area weights')
    w = np.asarray([float(weights[c]) for c in cols])
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise EvaluationError('Area weights must be finite with a positive sum')
    values = frame[cols].to_numpy(dtype=float)
    return pd.Series(values @ w / w.sum(), index=frame.index)


def cumulative_specific_balance(storage_m3: pd.Series, area_km2: pd.Series) -> pd.Series:
    """Cumulative specific balance (mm w.e.) from storage (m³ w.e.) and area (km²).

    ``B_t = sum_i (V_i - V_{i-1}) / A_i`` with ``A_i`` the mean area over the
    step, so a shrinking glacier is not mistaken for mass change.
    """
    storage = clean_series(storage_m3)
    area = clean_series(area_km2).reindex(storage.index).interpolate(limit_direction='both')
    step_area = (area + area.shift(1)) / 2.0
    step_area = step_area.where(step_area > 0)
    increments = storage.diff() / (step_area * 1e6) * 1000.0
    increments.iloc[0] = 0.0
    out = increments.fillna(0.0).cumsum()
    out[storage.isna()] = np.nan
    out.name = GLACIER_MASS_BALANCE
    return out


# ----------------------------------------------------------------------
# Registry access
# ----------------------------------------------------------------------

def get_output_adapter(model: str, project_dir: Optional[Path] = None,
                       config: Any = None) -> Optional[OutputVariableAdapter]:
    """Instantiate the registered adapter for ``model``, or None if it has none."""
    from symfluence.core.registries import R
    if not model:
        return None
    try:
        cls = R.output_adapters.get(str(model).strip().upper())
    except (ImportError, AttributeError, KeyError) as e:
        logger.debug(f"Output adapter for {model} unavailable: {e}")
        return None
    if cls is None:
        return None
    return cls(project_dir=project_dir, config=config)


def support_matrix() -> Dict[str, Dict[str, bool]]:
    """``{MODEL: {variable: supported}}`` for every registered adapter."""
    from symfluence.core.registries import R
    matrix: Dict[str, Dict[str, bool]] = {}
    for key in sorted(R.output_adapters.keys()):
        try:
            cls = R.output_adapters.get(key)
        except ImportError:
            continue
        if cls is None:
            continue
        supported = set(cls().supported_variables())
        matrix[key] = {v: v in supported for v in ADAPTER_VARIABLES}
    return matrix
