# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""One forcing correction shared by every model (the FORCING_CORRECTION_* section).

Models used to carry their own precipitation (and sometimes temperature)
correction as ordinary calibration parameters, each with its own bounds
(GR ``PCORR`` 0.5-2.0 and ``TCORR`` +-3 K, MESH ``PRECIP_MULT`` 0.8-1.6, Wflow
``PCORR`` 0.8-1.6, SUMMA ``precip_mult`` 0.7-1.6, mHM
``precipCorrectionFactor`` 0.8-1.8, HYPE ``pcurain``/``pcusnow``). Calibrated
jointly with the model's own parameters they absorb structural error, so
ensemble members end up seeing different forcing.

With ``FORCING_CORRECTION_MODE`` set, one definition replaces them:

``off``
    No correction. Per-model precipitation/temperature correction parameters
    are dropped from calibration.
``fixed``
    The same values for every model: given in the config, or (precipitation)
    estimated once from the long-term water balance (:func:`estimate_precip_multiplier`).
``calibrated``
    The common parameters ``FC_PRECIP_MULT`` (and optionally ``FC_SNOW_MULT``,
    ``FC_TEMP_OFFSET``) with common bounds, plus an optional prior penalty.

The optimizer sees the common names; :class:`ForcingCorrectionAdapter`
translates them to the model's own mechanism (``PCORR``, ``precip_mult``, ...)
when parameters are denormalized, so workers, final evaluations and model
files are unchanged. Unset (the default), nothing here runs and every model
behaves as before.

This is the hydrological (undercatch / water-balance) correction. A
source-to-reference correction of the forcing product itself (e.g. NWP
towards the calibration reanalysis) is a separate, earlier step applied to
the forcing files; the two compose: the forcing is first mapped onto the
reference product, then this multiplier applies on top, identically for every
model.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.mixins.config import resolve_config_value

logger = logging.getLogger(__name__)

#: Quantities a forcing correction can set.
QUANTITIES = ('precip', 'snow', 'temp')

#: Common calibration parameter name per quantity.
COMMON_NAMES: Dict[str, str] = {
    'precip': 'FC_PRECIP_MULT',
    'snow': 'FC_SNOW_MULT',
    'temp': 'FC_TEMP_OFFSET',
}

#: Value of each quantity that leaves the forcing unchanged.
IDENTITY: Dict[str, float] = {'precip': 1.0, 'snow': 1.0, 'temp': 0.0}

#: Report written next to the calibration results.
REPORT_NAME = 'forcing_correction.json'

#: Task-dict key carrying the plan to calibration workers (prior penalty only).
TASK_KEY = 'forcing_correction'

#: Metric key under which workers report the prior penalty.
PENALTY_METRIC = 'FORCING_CORRECTION_PRIOR_PENALTY'


# =============================================================================
# Per-model adapters
# =============================================================================

@dataclass(frozen=True)
class ForcingCorrectionAdapter:
    """Translate the common quantities into one model's own correction parameters.

    ``style='multiplier'``: the native parameters are the multipliers / offset
    themselves (``snow`` multiplies snowfall on top of the total-precipitation
    multiplier, as SUMMA's ``frozenPrecipMultip`` does).

    ``style='undercatch'`` (HYPE): rainfall and snowfall are corrected by
    ``(1 + pcurain)`` and ``(1 + pcusnow)``, so ``pcurain = m_p - 1`` and
    ``pcusnow = m_p * m_s - 1``. Setting the precipitation multiplier then
    writes both.
    """

    model: str
    precip: Optional[str] = None
    snow: Optional[str] = None
    temp: Optional[str] = None
    style: str = 'multiplier'
    note: str = ''

    def supports(self, quantity: str) -> bool:
        return getattr(self, quantity) is not None

    def native_names(self, quantities: Iterable[str]) -> List[str]:
        """Native parameters written when ``quantities`` are set."""
        names: List[str] = []
        for q in quantities:
            if not self.supports(q):
                continue
            names.append(getattr(self, q))
            if self.style == 'undercatch' and q == 'precip' and self.snow:
                names.append(self.snow)
        return list(dict.fromkeys(names))

    def to_native(self, values: Mapping[str, float]) -> Dict[str, float]:
        """Native parameter values for the quantities in ``values``."""
        out: Dict[str, float] = {}
        if self.style == 'undercatch':
            if 'precip' in values or 'snow' in values:
                m_p = float(values.get('precip', IDENTITY['precip']))
                m_s = float(values.get('snow', IDENTITY['snow']))
                if 'precip' in values and self.precip:
                    out[self.precip] = m_p - 1.0
                if self.snow:
                    out[self.snow] = m_p * m_s - 1.0
        else:
            for q in ('precip', 'snow'):
                if q in values and self.supports(q):
                    out[getattr(self, q)] = float(values[q])
        if 'temp' in values and self.temp:
            out[self.temp] = float(values['temp'])
        return out

    def from_native(self, params: Mapping[str, Any], quantities: Iterable[str]) -> Dict[str, float]:
        """Inverse of :meth:`to_native` for ``quantities`` present in ``params``."""
        def scalar(v: Any) -> float:
            arr = np.asarray(v, dtype=float).ravel()
            return float(arr[0]) if arr.size else float('nan')

        quantities = list(quantities)
        out: Dict[str, float] = {}
        if self.style == 'undercatch':
            m_p = IDENTITY['precip']
            if self.precip and self.precip in params:
                m_p = 1.0 + scalar(params[self.precip])
                if 'precip' in quantities:
                    out['precip'] = m_p
            if 'snow' in quantities and self.snow and self.snow in params and m_p != 0:
                out['snow'] = (1.0 + scalar(params[self.snow])) / m_p
        else:
            for q in ('precip', 'snow'):
                name = getattr(self, q)
                if q in quantities and name and name in params:
                    out[q] = scalar(params[name])
        if 'temp' in quantities and self.temp and self.temp in params:
            out['temp'] = scalar(params[self.temp])
        return out


#: Models with a correction mechanism, keyed by the optimizer's model name.
ADAPTERS: Dict[str, ForcingCorrectionAdapter] = {
    'GR': ForcingCorrectionAdapter(
        'GR', precip='PCORR', temp='TCORR',
        note='Applied to BasinObs precipitation / temperature before the R run.'),
    'SUMMA': ForcingCorrectionAdapter(
        'SUMMA', precip='precip_mult', snow='frozenPrecipMultip',
        note='precip_mult rescales pptrate in copied forcing files; '
             'frozenPrecipMultip is SUMMA\'s snowfall multiplier.'),
    'MESH': ForcingCorrectionAdapter(
        'MESH', precip='PRECIP_MULT', temp='TEMP_OFFSET',
        note='Rescales PRE / offsets TA in a private copy of the run forcing file.'),
    'WFLOW': ForcingCorrectionAdapter(
        'WFLOW', precip='PCORR',
        note='Sets the precipitation scale in the Wflow TOML.'),
    'MHM': ForcingCorrectionAdapter(
        'MHM', precip='precipCorrectionFactor',
        note='Rescales pre.nc in the worker\'s adjusted forcing.'),
    'HYPE': ForcingCorrectionAdapter(
        'HYPE', precip='pcurain', snow='pcusnow', style='undercatch',
        note='Writes pcurain/pcusnow in par.txt; both must exist there '
             '(HYPE_EXTRA_PARAMETERS).'),
}


def get_adapter(model_name: str) -> Optional[ForcingCorrectionAdapter]:
    """Adapter for ``model_name`` (case-insensitive), or None."""
    return ADAPTERS.get(str(model_name or '').upper())


# =============================================================================
# Plan
# =============================================================================

@dataclass
class ForcingCorrectionPlan:
    """Resolved forcing correction for one model's calibration."""

    mode: str
    model: str
    fixed: Dict[str, float] = field(default_factory=dict)
    calibrated: List[str] = field(default_factory=list)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    prior_weight: float = 0.0
    prior_precip_median: float = 1.0
    prior_precip_sigma: float = 0.2
    prior_snow_sigma: float = 0.2
    prior_temp_sigma: float = 1.0
    estimate: Optional[Dict[str, Any]] = None

    @property
    def adapter(self) -> Optional[ForcingCorrectionAdapter]:
        return get_adapter(self.model)

    @property
    def quantities_set(self) -> List[str]:
        """Quantities this plan writes (fixed or calibrated)."""
        return [q for q in QUANTITIES if q in self.fixed or q in self.calibrated]

    # -- parameter space ----------------------------------------------------

    @property
    def calibrated_param_names(self) -> List[str]:
        return [COMMON_NAMES[q] for q in QUANTITIES if q in self.calibrated]

    def calibrated_param_bounds(self) -> Dict[str, Dict[str, float]]:
        return {COMMON_NAMES[q]: {'min': float(self.bounds[q][0]), 'max': float(self.bounds[q][1])}
                for q in self.calibrated}

    def superseded_native_names(self) -> List[str]:
        """Model parameters this plan replaces.

        Precipitation and temperature corrections are replaced in every mode
        (including ``off``); the snowfall parameter (e.g. SUMMA's
        ``frozenPrecipMultip``, a model parameter in its own right) only when
        the plan sets the snow quantity.
        """
        adapter = self.adapter
        if adapter is None:
            return []
        quantities = ['precip', 'temp'] + (['snow'] if 'snow' in self.quantities_set else [])
        return adapter.native_names(quantities)

    def initial_common_values(self, native_params: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
        """Starting value of each calibrated common parameter.

        Taken from the model's own correction parameter when ``native_params``
        holds one, otherwise the prior median (precipitation) or identity.
        """
        derived: Dict[str, float] = {}
        adapter = self.adapter
        if adapter is not None and native_params:
            derived = adapter.from_native(native_params, self.calibrated)
        out = {}
        for q in self.calibrated:
            default = self.prior_precip_median if q == 'precip' else IDENTITY[q]
            value = derived.get(q, default)
            lo, hi = self.bounds[q]
            out[COMMON_NAMES[q]] = float(min(max(value, lo), hi)) if np.isfinite(value) else default
        return out

    # -- translation --------------------------------------------------------

    def quantity_values(self, common_params: Mapping[str, Any]) -> Dict[str, float]:
        """Fixed values plus calibrated values found in ``common_params``."""
        values = dict(self.fixed)
        for q in self.calibrated:
            name = COMMON_NAMES[q]
            if name in common_params:
                values[q] = float(np.asarray(common_params[name], dtype=float).ravel()[0])
        return values

    def to_native(self, common_params: Mapping[str, Any]) -> Dict[str, float]:
        """Native parameters for the fixed values and the calibrated common values."""
        adapter = self.adapter
        if adapter is None:
            return {}
        return adapter.to_native(self.quantity_values(common_params))

    def values_from_native(self, native_params: Mapping[str, Any]) -> Dict[str, float]:
        """Quantity values recovered from a native parameter set (for reporting / the prior)."""
        values = dict(self.fixed)
        adapter = self.adapter
        if adapter is not None and self.calibrated:
            values.update(adapter.from_native(native_params, self.calibrated))
        return values

    def common_values_from_native(self, native_params: Mapping[str, Any]) -> Dict[str, float]:
        """Like :meth:`values_from_native`, keyed by common parameter name."""
        return {COMMON_NAMES[q]: v for q, v in self.values_from_native(native_params).items()}

    # -- prior --------------------------------------------------------------

    def prior_penalty(self, values: Mapping[str, float]) -> float:
        """Prior penalty for calibrated quantity values (0 when the prior is off).

        ``weight * 0.5 * z**2`` summed over calibrated quantities, with
        ``z = ln(m / median) / sigma`` for the precipitation multiplier
        (log-normal prior), ``z = ln(m) / sigma`` for the snowfall multiplier
        and ``z = offset / sigma`` for the temperature offset. The penalty is
        subtracted from the objective in its maximization form.
        """
        if self.mode != 'calibrated' or self.prior_weight <= 0:
            return 0.0
        total = 0.0
        for q in self.calibrated:
            if q not in values:
                continue
            v = float(values[q])
            if q == 'precip':
                if v <= 0:
                    return math.inf
                z = math.log(v / self.prior_precip_median) / self.prior_precip_sigma
            elif q == 'snow':
                if v <= 0:
                    return math.inf
                z = math.log(v) / self.prior_snow_sigma
            else:
                z = v / self.prior_temp_sigma
            total += 0.5 * z * z
        return self.prior_weight * total

    # -- transport / reporting ----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['bounds'] = {q: list(b) for q, b in self.bounds.items()}
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'ForcingCorrectionPlan':
        d = dict(data)
        d['bounds'] = {q: (float(b[0]), float(b[1])) for q, b in (d.get('bounds') or {}).items()}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def report(self, best_native_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """JSON-ready description of the correction, with applied values."""
        adapter = self.adapter
        out: Dict[str, Any] = {
            'mode': self.mode,
            'model': self.model,
            'native_parameters': (
                {q: adapter.native_names([q]) for q in self.quantities_set} if adapter else {}),
            'superseded_model_parameters': self.superseded_native_names(),
            'fixed': {COMMON_NAMES[q]: v for q, v in self.fixed.items()},
            'calibrated': self.calibrated_param_names,
            'bounds': {COMMON_NAMES[q]: list(b) for q, b in self.bounds.items() if q in self.calibrated},
        }
        if self.mode == 'calibrated':
            out['prior'] = {
                'weight': self.prior_weight,
                'precip_median': self.prior_precip_median,
                'precip_sigma_log': self.prior_precip_sigma,
                'snow_sigma_log': self.prior_snow_sigma,
                'temp_sigma_K': self.prior_temp_sigma,
            }
        if self.estimate is not None:
            out['water_balance_estimate'] = self.estimate
        if best_native_params is not None:
            values = self.values_from_native(best_native_params)
            out['applied'] = {COMMON_NAMES[q]: v for q, v in values.items()}
            out['applied_native'] = self.to_native(
                {COMMON_NAMES[q]: v for q, v in values.items()})
            if self.mode == 'calibrated':
                out['prior_penalty_at_best'] = self.prior_penalty(values)
        elif self.fixed:
            out['applied'] = {COMMON_NAMES[q]: v for q, v in self.fixed.items()}
            out['applied_native'] = self.to_native({})
        return out


def write_report(path: Path, plan: ForcingCorrectionPlan,
                 best_native_params: Optional[Mapping[str, Any]] = None) -> Path:
    """Write :meth:`ForcingCorrectionPlan.report` as JSON to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.report(best_native_params), indent=2, default=float),
                    encoding='utf-8')
    return path


def apply_prior_to_score(score: float, plan_data: Optional[Mapping[str, Any]],
                         native_params: Mapping[str, Any]) -> Tuple[float, float]:
    """``(score - penalty, penalty)`` for a worker result (maximization convention)."""
    if not plan_data:
        return score, 0.0
    plan = ForcingCorrectionPlan.from_dict(plan_data)
    penalty = plan.prior_penalty(plan.values_from_native(native_params))
    if penalty == 0.0:
        return score, 0.0
    return score - penalty, penalty


# =============================================================================
# Water-balance estimate
# =============================================================================

def budyko_et(precip: float, pet: float) -> float:
    """Long-term actual ET from the Budyko (1974) curve (same units as the inputs).

    ``E/P = sqrt(phi * tanh(1/phi) * (1 - exp(-phi)))`` with aridity
    ``phi = PET / P``.
    """
    if precip <= 0:
        return 0.0
    if pet <= 0:
        return 0.0
    phi = pet / precip
    return precip * math.sqrt(phi * math.tanh(1.0 / phi) * (1.0 - math.exp(-phi)))


def estimate_precip_multiplier(
    precip_mm_day: float,
    q_mm_day: float,
    et_mm_day: Optional[float] = None,
    pet_mm_day: Optional[float] = None,
) -> Dict[str, Any]:
    """Precipitation multiplier that closes the long-term water balance.

    With an actual-ET estimate: ``m = (Q + ET) / P``. Without one, ET follows
    the Budyko curve of the *corrected* precipitation and PET, and ``m``
    solves ``m P - E_Budyko(m P, PET) = Q`` (bisection; the left side grows
    monotonically with ``m``). Storage change is neglected, so use a period of
    several years.

    Args:
        precip_mm_day: Mean basin precipitation (mm/day) over the period.
        q_mm_day: Mean observed runoff (mm/day) over the same period.
        et_mm_day: Mean actual ET (mm/day), if known.
        pet_mm_day: Mean potential ET (mm/day), used when ``et_mm_day`` is None.

    Returns:
        Dict with ``multiplier`` and the terms used.
    """
    if not (precip_mm_day > 0 and np.isfinite(precip_mm_day)):
        raise ValueError(f'Mean precipitation must be positive, got {precip_mm_day}')
    if not (q_mm_day >= 0 and np.isfinite(q_mm_day)):
        raise ValueError(f'Mean runoff must be non-negative, got {q_mm_day}')
    out: Dict[str, Any] = {'precip_mm_day': float(precip_mm_day), 'q_mm_day': float(q_mm_day)}
    if et_mm_day is not None:
        out.update(method='config_et', et_mm_day=float(et_mm_day),
                   multiplier=(q_mm_day + et_mm_day) / precip_mm_day)
        return out
    if pet_mm_day is None or not np.isfinite(pet_mm_day) or pet_mm_day < 0:
        raise ValueError('Either an actual ET or a non-negative PET is required')

    def residual(m: float) -> float:
        p = m * precip_mm_day
        return p - budyko_et(p, pet_mm_day) - q_mm_day

    lo, hi = 1e-3, 1.0
    while residual(hi) < 0:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError('Water balance cannot be closed (runoff far exceeds precipitation)')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    m = 0.5 * (lo + hi)
    out.update(method='budyko', pet_mm_day=float(pet_mm_day),
               et_mm_day=float(budyko_et(m * precip_mm_day, pet_mm_day)), multiplier=m)
    return out


def _parse_period(text: Optional[str]) -> Tuple[Optional[Any], Optional[Any]]:
    import pandas as pd
    if not text:
        return None, None
    parts = [p.strip() for p in str(text).split(',')]
    if len(parts) != 2:
        return None, None
    return pd.Timestamp(parts[0]), pd.Timestamp(parts[1])


def _latitude(config: Any, ds: Any) -> float:
    for name in ('latitude', 'lat', 'hru_lat'):
        if name in ds.variables:
            vals = np.asarray(ds[name].values, dtype=float)
            if np.isfinite(vals).any():
                return float(np.nanmean(vals))
    coords = resolve_config_value(config, lambda: config.domain.pour_point_coords,
                                  dict_key='POUR_POINT_COORDS')
    if coords and '/' in str(coords):
        return float(str(coords).split('/')[0])
    raise ValueError('No latitude for the PET estimate (forcing has none and POUR_POINT_COORDS is unset)')


def water_balance_inputs(
    config: Any,
    project_dir: Path,
    domain_name: str,
    pet_method: str = 'oudin',
) -> Dict[str, Any]:
    """Mean P, Q and PET (mm/day) over the calibration period from project data.

    P and air temperature come from the basin-averaged forcing (``pptrate``
    in kg m-2 s-1, ``airtemp`` in K; HRU-mean when there are several), Q from
    the processed streamflow observations divided by the catchment area. PET
    uses the code base's Oudin or Hamon formula on daily mean temperature.
    """
    import pandas as pd
    import xarray as xr

    from symfluence.core.metrics.streamflow_metrics import (
        load_observations,
        require_catchment_area_km2,
    )
    from symfluence.core.mixins.project import resolve_forcing_basin_path
    from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin

    project_dir = Path(project_dir)
    files = sorted(resolve_forcing_basin_path(project_dir).glob('*.nc'))
    if not files:
        raise FileNotFoundError(f'No basin-averaged forcing under {project_dir}')
    with xr.open_mfdataset(files, combine='by_coords') as ds:
        p = ds['pptrate']
        t = ds['airtemp']
        extra = [d for d in p.dims if d != 'time']
        p_s = p.mean(dim=extra).to_series() if extra else p.to_series()
        t_s = t.mean(dim=extra).to_series() if extra else t.to_series()
        lat = _latitude(config, ds)
    p_day = (p_s * 86400.0).resample('D').mean()
    t_day = t_s.resample('D').mean() - 273.15

    obs, idx = load_observations(config, project_dir, domain_name, resample_freq='D')
    if obs is None or idx is None:
        raise FileNotFoundError(f'No streamflow observations for {domain_name}')
    area_km2 = require_catchment_area_km2(config, project_dir, domain_name)
    q_day = pd.Series(np.asarray(obs, dtype=float), index=idx) * 86400.0 / (area_km2 * 1e3)

    start, end = _parse_period(resolve_config_value(
        config, lambda: config.domain.calibration_period, dict_key='CALIBRATION_PERIOD'))
    frame = pd.DataFrame({'p': p_day, 't': t_day, 'q': q_day}).dropna()
    if start is not None:
        frame = frame.loc[start:end]
    if len(frame) < 365:
        raise ValueError(f'Only {len(frame)} days with P, T and Q; need at least a year')
    doy = frame.index.dayofyear.values
    if pet_method == 'hamon':
        pet = PETCalculatorMixin.hamon_pet_numpy(frame['t'].values, doy, lat)
    else:
        pet = PETCalculatorMixin.oudin_pet_numpy(frame['t'].values, doy, lat)
    return {
        'precip_mm_day': float(frame['p'].mean()),
        'q_mm_day': float(frame['q'].mean()),
        'pet_mm_day': float(np.mean(pet)),
        'pet_method': pet_method,
        'latitude': lat,
        'area_km2': float(area_km2),
        'n_days': int(len(frame)),
        'period': [str(frame.index[0].date()), str(frame.index[-1].date())],
    }


# =============================================================================
# Building the plan from config
# =============================================================================

def _cfg(config: Any, attr: str, key: str) -> Any:
    return resolve_config_value(
        config, lambda: getattr(config.optimization.forcing_correction, attr), dict_key=key)


def configured_mode(config: Any) -> Optional[str]:
    """``FORCING_CORRECTION_MODE`` (lower case), or None when the section is absent."""
    mode = _cfg(config, 'mode', 'FORCING_CORRECTION_MODE')
    if mode in (None, '', 'None', 'none'):
        return None
    mode = str(mode).lower()
    if mode not in ('off', 'fixed', 'calibrated'):
        raise ConfigurationError(
            f"FORCING_CORRECTION_MODE must be 'off', 'fixed' or 'calibrated', got {mode!r}")
    return mode


_DEFAULT_BOUNDS = {
    'FORCING_CORRECTION_PRECIP_BOUNDS': (0.7, 1.6),
    'FORCING_CORRECTION_SNOW_BOUNDS': (0.7, 1.6),
    'FORCING_CORRECTION_TEMP_BOUNDS': (-3.0, 3.0),
}


def _bounds(config: Any, attr: str, key: str, positive: bool) -> Tuple[float, float]:
    # Defaults mirror ForcingCorrectionConfig (list default_factory is not
    # visible to the schema-default fallback of a flat dict config).
    raw = _cfg(config, attr, key) or _DEFAULT_BOUNDS[key]
    if isinstance(raw, str):
        raw = [x for x in raw.replace('[', '').replace(']', '').split(',') if x.strip()]
    lo, hi = float(raw[0]), float(raw[1])
    if not lo < hi or (positive and lo <= 0):
        raise ConfigurationError(f'{key} must be [min, max] with {"0 < " if positive else ""}min < max')
    return lo, hi


def build_plan(
    config: Any,
    model_name: str,
    project_dir: Optional[Path] = None,
    domain_name: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[ForcingCorrectionPlan]:
    """Resolve the FORCING_CORRECTION section for ``model_name``.

    Returns None when ``FORCING_CORRECTION_MODE`` is unset (legacy behaviour).

    Raises:
        ConfigurationError: a correction other than identity is requested for
            a quantity the model has no mechanism for.
    """
    log = log or logger
    mode = configured_mode(config)
    if mode is None:
        return None
    model = str(model_name or '').upper()
    adapter = get_adapter(model)

    def num(attr: str, key: str) -> Optional[float]:
        v = _cfg(config, attr, key)
        return None if v in (None, '', 'None') else v

    plan = ForcingCorrectionPlan(
        mode=mode, model=model,
        prior_weight=float(num('prior_weight', 'FORCING_CORRECTION_PRIOR_WEIGHT') or 0.0),
        prior_precip_median=float(num('prior_precip_median', 'FORCING_CORRECTION_PRIOR_PRECIP_MEDIAN') or 1.0),
        prior_precip_sigma=float(num('prior_precip_sigma', 'FORCING_CORRECTION_PRIOR_PRECIP_SIGMA') or 0.2),
        prior_snow_sigma=float(num('prior_snow_sigma', 'FORCING_CORRECTION_PRIOR_SNOW_SIGMA') or 0.2),
        prior_temp_sigma=float(num('prior_temp_sigma', 'FORCING_CORRECTION_PRIOR_TEMP_SIGMA') or 1.0),
    )
    if mode == 'off':
        return plan

    if mode == 'calibrated':
        quantities = _cfg(config, 'quantities', 'FORCING_CORRECTION_QUANTITIES') or ['precip']
        if isinstance(quantities, str):
            quantities = [q.strip() for q in quantities.split(',') if q.strip()]
        quantities = [str(q).lower() for q in quantities]
        unknown = set(quantities) - set(QUANTITIES)
        if unknown:
            raise ConfigurationError(f'FORCING_CORRECTION_QUANTITIES: unknown {sorted(unknown)}')
        plan.calibrated = [q for q in QUANTITIES if q in quantities]
        plan.bounds = {
            'precip': _bounds(config, 'precip_bounds', 'FORCING_CORRECTION_PRECIP_BOUNDS', True),
            'snow': _bounds(config, 'snow_bounds', 'FORCING_CORRECTION_SNOW_BOUNDS', True),
            'temp': _bounds(config, 'temp_bounds', 'FORCING_CORRECTION_TEMP_BOUNDS', False),
        }

    # Fixed values: any quantity not calibrated that the config gives a value for.
    precip = num('precip_multiplier', 'FORCING_CORRECTION_PRECIP_MULTIPLIER')
    if 'precip' not in plan.calibrated:
        if isinstance(precip, str) and precip.lower() == 'water_balance':
            plan.estimate = _estimate_from_project(config, project_dir, domain_name)
            plan.fixed['precip'] = float(plan.estimate['multiplier'])
            log.info(f"Forcing correction: water-balance precipitation multiplier "
                     f"{plan.fixed['precip']:.3f} ({plan.estimate['method']})")
        elif precip is not None:
            plan.fixed['precip'] = float(precip)
        elif mode == 'fixed':
            log.info('FORCING_CORRECTION_MODE=fixed without FORCING_CORRECTION_PRECIP_MULTIPLIER: '
                     'precipitation multiplier 1.0')
            plan.fixed['precip'] = 1.0
    snow = num('snow_multiplier', 'FORCING_CORRECTION_SNOW_MULTIPLIER')
    if snow is not None and 'snow' not in plan.calibrated:
        plan.fixed['snow'] = float(snow)
    temp = num('temp_offset', 'FORCING_CORRECTION_TEMP_OFFSET')
    if temp is not None and 'temp' not in plan.calibrated:
        plan.fixed['temp'] = float(temp)
    for q, v in plan.fixed.items():
        if q in ('precip', 'snow') and not v > 0:
            raise ConfigurationError(f'Forcing correction {q} multiplier must be positive, got {v}')

    # Every requested quantity needs a mechanism in this model.
    missing = [q for q in plan.calibrated if not (adapter and adapter.supports(q))]
    missing += [q for q, v in plan.fixed.items()
                if v != IDENTITY[q] and not (adapter and adapter.supports(q))]
    if missing:
        supported = ', '.join(
            f"{m} ({'/'.join(q for q in QUANTITIES if a.supports(q))})" for m, a in ADAPTERS.items())
        raise ConfigurationError(
            f'FORCING_CORRECTION: {model} has no mechanism for {sorted(set(missing))}. '
            f'Supported: {supported}. Use FORCING_CORRECTION_MODE=off for this model, '
            f'or drop the quantity.')
    # Identity fixed values for unsupported quantities are no-ops.
    plan.fixed = {q: v for q, v in plan.fixed.items() if adapter and adapter.supports(q)}
    return plan


def _estimate_from_project(config: Any, project_dir: Optional[Path],
                           domain_name: Optional[str]) -> Dict[str, Any]:
    if project_dir is None or domain_name is None:
        raise ConfigurationError("FORCING_CORRECTION_PRECIP_MULTIPLIER='water_balance' needs project data")
    pet_method = str(_cfg(config, 'pet_method', 'FORCING_CORRECTION_PET_METHOD') or 'oudin').lower()
    try:
        inputs = water_balance_inputs(config, Path(project_dir), domain_name, pet_method)
    except (OSError, ValueError, KeyError) as e:
        raise ConfigurationError(f'Water-balance precipitation multiplier: {e}') from e
    et_year = _cfg(config, 'et_mm_per_year', 'FORCING_CORRECTION_ET_MM_PER_YEAR')
    et_day = None if et_year in (None, '', 'None') else float(et_year) / 365.25
    estimate = estimate_precip_multiplier(
        inputs['precip_mm_day'], inputs['q_mm_day'], et_mm_day=et_day,
        pet_mm_day=inputs['pet_mm_day'])
    estimate.update({k: v for k, v in inputs.items() if k not in estimate})
    return estimate


def conflicting_settings(config: Any, model_param_names: Sequence[str],
                         plan: ForcingCorrectionPlan) -> List[str]:
    """Per-model corrections that the plan supersedes (for a warning)."""
    superseded = set(plan.superseded_native_names())
    found = [n for n in model_param_names if n in superseded]
    if plan.model == 'MESH':
        mesh_mult = resolve_config_value(
            config, lambda: config.forcing.precip_multiplier, dict_key='MESH_PRECIP_MULTIPLIER')
        if mesh_mult not in (None, '') and float(mesh_mult) != 1.0:
            found.append(f'MESH_PRECIP_MULTIPLIER={mesh_mult} (applied at preprocessing, '
                         'on top of the shared correction)')
    return found


def supported_models_table() -> List[Dict[str, str]]:
    """Rows for documentation: model, native parameter per quantity, note."""
    rows = []
    for name, a in ADAPTERS.items():
        rows.append({'model': name, 'precip': a.precip or '-', 'snow': a.snow or '-',
                     'temp': a.temp or '-', 'note': a.note})
    return rows


# =============================================================================
# Forward (non-calibration) runs
# =============================================================================

#: Record written next to a forward run's output.
FORWARD_RECORD_NAME = 'forcing_correction.json'


def latest_calibration_report(project_dir: Path, model_name: str,
                              experiment_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Newest ``forcing_correction.json`` with applied values from a calibration of ``model_name``.

    Looks in ``optimization/<MODEL>/*_<experiment_id>/`` (any experiment when
    ``experiment_id`` is None).
    """
    base = Path(project_dir) / 'optimization' / str(model_name).upper()
    if not base.is_dir():
        return None
    pattern = f'*_{experiment_id}/{REPORT_NAME}' if experiment_id else f'*/{REPORT_NAME}'
    candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if data.get('mode') == 'calibrated' and data.get('applied'):
            data['report_path'] = str(path)
            return data
    return None


def forward_correction(
    config: Any,
    model_name: str,
    project_dir: Optional[Path] = None,
    domain_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Forcing correction for a forward (non-calibration) run of ``model_name``.

    ``fixed``: the plan's fixed values (water-balance estimate included).
    ``calibrated``: the values applied at the best parameter set of the
    newest calibration of this model and experiment (its
    ``forcing_correction.json``), so forward runs use what the final
    evaluation used. ``off``: identity values for the model's precipitation
    and temperature corrections, which resets settings an earlier fixed or
    calibrated run wrote.

    Returns:
        None when FORCING_CORRECTION_MODE is unset; otherwise a dict with
        ``mode``, ``model``, ``values`` (common names), ``native`` (model
        parameters) and ``source``.
    """
    log = log or logger
    plan = build_plan(config, model_name, project_dir=project_dir, domain_name=domain_name, log=log)
    if plan is None:
        return None
    adapter = plan.adapter
    values: Dict[str, float]
    source: str
    if plan.mode == 'fixed':
        values = dict(plan.fixed)
        source = 'water_balance' if plan.estimate is not None else 'config'
    elif plan.mode == 'calibrated':
        values = dict(plan.fixed)
        report = (latest_calibration_report(project_dir, plan.model, experiment_id)
                  if project_dir is not None else None)
        if report is None:
            log.warning(
                f'FORCING_CORRECTION_MODE=calibrated but no calibration report for {plan.model} '
                f'(experiment {experiment_id}); calibrated quantities stay uncorrected in this run')
            source = 'none'
        else:
            for q in plan.calibrated:
                v = report['applied'].get(COMMON_NAMES[q])
                if v is not None:
                    values[q] = float(v)
            source = report['report_path']
    else:
        values = {q: IDENTITY[q] for q in ('precip', 'temp') if adapter and adapter.supports(q)}
        source = 'off'
    native = adapter.to_native(values) if adapter else {}
    return {
        'mode': plan.mode,
        'model': plan.model,
        'values': {COMMON_NAMES[q]: v for q, v in values.items()},
        'native': native,
        'source': source,
        'water_balance_estimate': plan.estimate,
    }


def write_forward_record(directory: Path, info: Mapping[str, Any]) -> Path:
    """Write the forward-run correction ``info`` as ``forcing_correction.json`` in ``directory``."""
    path = Path(directory) / FORWARD_RECORD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(info), indent=2, default=float), encoding='utf-8')
    return path


def find_forward_applier(runner_class: Any) -> Optional[Any]:
    """``apply_forward`` of the model package that holds ``runner_class``.

    A model supports forward runs by providing ``forcing_correction.py`` with
    ``apply_forward(config, logger, project_dir, info) -> dict`` next to its
    runner module (inversion of control: core does not name model packages).
    """
    import importlib

    module = getattr(runner_class, '__module__', '') or ''
    if '.' not in module:
        return None
    package = module.rsplit('.', 1)[0]
    try:
        mod = importlib.import_module(package + '.forcing_correction')
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith('forcing_correction'):
            return None
        raise
    return getattr(mod, 'apply_forward', None)


def apply_forward_forcing_correction(
    model: str,
    config: Any,
    log: Optional[logging.Logger] = None,
    project_dir: Optional[Path] = None,
    runner_class: Any = None,
) -> Optional[Dict[str, Any]]:
    """Apply the shared forcing correction before a forward (non-calibration) run.

    Called by the workflow's model-run step for every model; code that
    constructs runners directly (forecast pipelines, scripts) should call it
    before running. No-op (returns None) when FORCING_CORRECTION_MODE is unset.

    Resolves the values with :func:`forward_correction` and hands the model's
    native parameters to the model's ``forcing_correction.apply_forward``,
    which sets them idempotently in the run settings the way a calibration's
    final evaluation does. The applied values are logged and written to
    ``simulations/<experiment>/<MODEL>/forcing_correction.json``.
    """
    log = log or logger
    domain_name = resolve_config_value(config, lambda: config.domain.name, dict_key='DOMAIN_NAME')
    experiment_id = resolve_config_value(
        config, lambda: config.domain.experiment_id, dict_key='EXPERIMENT_ID')
    if project_dir is None:
        data_dir = resolve_config_value(config, lambda: config.system.data_dir,
                                        dict_key='SYMFLUENCE_DATA_DIR')
        project_dir = Path(data_dir) / f'domain_{domain_name}'
    project_dir = Path(project_dir)

    info = forward_correction(config, model, project_dir=project_dir, domain_name=domain_name,
                              experiment_id=experiment_id, log=log)
    if info is None:
        return None
    applier = find_forward_applier(runner_class) if runner_class is not None else None
    if applier is None:
        if info['native'] and get_adapter(model) is not None:
            raise ConfigurationError(
                f'FORCING_CORRECTION: {info["model"]} has no forward-run applier '
                '(forcing_correction.apply_forward next to its runner)')
        info['details'] = {'applied': False, 'reason': 'no correction mechanism; identity only'}
    else:
        info['details'] = applier(config, log, project_dir, info) or {}
    log.info(f"Forcing correction ({info['mode']}) for {info['model']} forward run: "
             f"{info['values']} -> {info['native']} [{info['source']}]")
    record_dir = project_dir / 'simulations' / str(experiment_id) / str(model).upper()
    try:
        write_forward_record(record_dir, info)
    except OSError as e:
        log.warning(f'Could not write the forcing-correction record in {record_dir}: {e}')
    return info
