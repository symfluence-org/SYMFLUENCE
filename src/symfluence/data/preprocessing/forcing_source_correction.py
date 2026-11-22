# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Correct a forcing source (e.g. NWP forecasts) towards the forcing a model was calibrated on.

A model calibrated on one product (typically a reanalysis) and then driven by
another (an NWP forecast) sees the second product's biases as weather.  This
module fits simple, robust corrections from paired values of the two products
and applies them to a forcing dataset:

* per variable an additive (``kind='add'``) or multiplicative (``'mult'``)
  factor, ``reference_mean - source_mean`` or ``reference_mean / source_mean``;
* pooled at a chosen level: ``'none'`` (no correction), ``'annual'`` (one
  factor), ``'season'`` (DJF/MAM/JJA/SON) or ``'month'``; with short overlaps,
  monthly factors mostly fit one year's weather, so :func:`split_sample_test`
  reports out-of-sample skill (first half -> second half and the reverse) to
  choose the level;
* multiplicative factors are guarded for dry months: a group whose source
  mean is at or below ``min_source_mean`` stays at 1, and factors are clamped
  to ``factor_bounds`` (default 0.25-4); both are recorded in ``meta['limited']``;
* optionally (:class:`HumidityVariables`) specific humidity is recomputed
  after a temperature correction so that relative humidity is unchanged;
* stored as twelve monthly values per variable, with the source name and
  training period.  :meth:`ForcingSourceCorrection.apply` refuses a dataset from
  another source: a new source (or a new model generation of the same source)
  needs its own fit.

Pairs are best built after both products pass through the same mapping onto
the model's units (HRUs or cells), so the fit sees what the model sees.
Building them is source specific and left to the caller: :func:`fit_source_correction`
takes a table with a ``time`` column, the source variables and their
reference counterparts.

This is not the shared hydrological ``FORCING_CORRECTION_*`` section
(:mod:`symfluence.core.calibration.forcing_correction`).  The two compose in
order: this source correction first maps the forecast forcing onto the
reference product the model was calibrated on, then the model applies its
calibrated ``FORCING_CORRECTION_*`` values (precipitation multiplier, snow
multiplier, temperature offset) exactly as it did in calibration.  Any future
config keys for this module use the ``FORCING_SOURCE_CORRECTION_*`` prefix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.data.preprocessing.resampling.elevation_correction import correct_specific_humidity

Level = Literal['none', 'annual', 'season', 'month']
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
MONTHS = [str(m) for m in range(1, 13)]


@dataclass(frozen=True)
class CorrectionSpec:
    """How one variable is corrected.

    Attributes:
        variable: Name in the source pairs and in the forcing dataset.
        kind: ``'add'`` or ``'mult'``.
        level: Pooling of the fit: ``'none'``, ``'annual'``, ``'season'``, ``'month'``.
        reference: Reference column in the pairs (default ``ref_<variable>``).
        clip_min, clip_max: Bounds applied after correcting (e.g. 0 for
            precipitation, 0-100 for relative humidity).
        min_source_mean: ``'mult'`` only. A pooling group whose source mean is
            at or below this value stays uncorrected (factor 1): in dry months
            a ratio of two small means is noise. Set it in the variable's units
            (e.g. a fraction of a millimetre per time step for precipitation).
        factor_bounds: ``'mult'`` only. Fitted factors are clamped to
            ``(low, high)``; ``None`` disables the clamp.
    """

    variable: str
    kind: Literal['add', 'mult']
    level: Level = 'month'
    reference: Optional[str] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    min_source_mean: float = 0.0
    factor_bounds: Optional[Tuple[float, float]] = (0.25, 4.0)

    def __post_init__(self) -> None:
        if self.factor_bounds is not None:
            low, high = self.factor_bounds
            if not 0 < low <= 1 <= high:
                raise ValidationError(
                    f'factor_bounds for {self.variable} must satisfy 0 < low <= 1 <= high, got {self.factor_bounds}')

    @property
    def reference_column(self) -> str:
        return self.reference or f'ref_{self.variable}'

    @property
    def neutral(self) -> float:
        return 0.0 if self.kind == 'add' else 1.0


@dataclass(frozen=True)
class HumidityVariables:
    """Names used to keep specific humidity consistent with a temperature correction.

    Correcting air temperature alone changes the relative humidity the model
    sees (warmer air at the same specific humidity is drier).  When passed to
    :meth:`ForcingSourceCorrection.apply`, specific humidity is recomputed so
    that relative humidity is what it was before the correction.  Temperatures
    are in K, pressure in Pa, specific humidity in kg/kg.
    """

    temperature: str = 'airtemp'
    specific_humidity: str = 'spechum'
    pressure: str = 'airpres'


@dataclass
class ForcingSourceCorrection:
    """Fitted monthly factors for one forcing source."""

    source: str
    factors: Dict[str, Dict[str, float]]
    kinds: Dict[str, str]
    clips: Dict[str, List[Optional[float]]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, values in self.factors.items():
            if set(values) != set(MONTHS) or not np.isfinite(list(values.values())).all():
                raise ValidationError(f'correction for {name} must hold 12 finite monthly values')
            if self.kinds.get(name) not in ('add', 'mult'):
                raise ValidationError(f'correction for {name} has no kind (add/mult)')

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> Dict[str, Any]:
        return {'source': self.source, 'factors': self.factors, 'kinds': self.kinds,
                'clips': self.clips, 'meta': self.meta}

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ForcingSourceCorrection':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(source=d['source'], factors=d['factors'], kinds=d['kinds'],
                   clips=d.get('clips', {}), meta=d.get('meta', {}))

    # ---------------------------------------------------------------- apply
    def _monthly(self, name: str, months: np.ndarray) -> np.ndarray:
        table = self.factors[name]
        return np.array([table[str(int(m))] for m in months], dtype=float)

    def apply_frame(self, frame: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
        """Correct the source columns of a table (used for pairs and tests)."""
        out = frame.copy()
        months = pd.to_datetime(frame[time_column]).dt.month.to_numpy()
        for name, kind in self.kinds.items():
            if name not in out:
                continue
            f = self._monthly(name, months)
            v = out[name].to_numpy(dtype=float)
            out[name] = self._clip(name, v + f if kind == 'add' else v * f)
        return out

    def apply(self, ds: xr.Dataset, source: str,
              humidity: Optional[HumidityVariables] = None) -> xr.Dataset:
        """Correct a forcing dataset with a ``time`` dimension; refuses another source.

        Args:
            ds: Forcing dataset.
            source: Name of the product ``ds`` comes from; must match the fit.
            humidity: When given and its temperature variable is corrected (and
                its specific humidity is not), specific humidity is recomputed
                to keep relative humidity unchanged.  Without it, a temperature
                correction leaves specific humidity as it was.
        """
        if source != self.source:
            raise ValidationError(f'correction is for source {self.source!r}, forcing is {source!r}')
        missing = [n for n in self.kinds if n not in ds]
        if missing:
            raise ValidationError(f'forcing lacks corrected variables {missing}')
        out = ds.copy()
        months = ds['time'].dt.month.values
        for name, kind in self.kinds.items():
            f = xr.DataArray(self._monthly(name, months), dims='time', coords={'time': ds['time']})
            corrected = ds[name] + f if kind == 'add' else ds[name] * f
            lo, hi = (self.clips.get(name) or [None, None])
            if lo is not None or hi is not None:
                corrected = corrected.clip(min=lo, max=hi)
            out[name] = corrected.assign_attrs(ds[name].attrs)
        if humidity is not None:
            out = self._keep_relative_humidity(ds, out, humidity)
        stamp = self.meta.get('created', '')
        return out.assign_attrs(ds.attrs, forcing_source_correction=f'{self.source}:{stamp}')

    def _keep_relative_humidity(self, raw: xr.Dataset, out: xr.Dataset,
                                h: HumidityVariables) -> xr.Dataset:
        if h.temperature not in self.kinds or h.specific_humidity in self.kinds:
            return out
        missing = [n for n in (h.specific_humidity, h.pressure) if n not in raw]
        if missing:
            raise ValidationError(f'keeping relative humidity needs {missing} in the forcing')
        q = xr.apply_ufunc(correct_specific_humidity, raw[h.specific_humidity], out[h.pressure],
                           raw[h.temperature], out[h.pressure], out[h.temperature])
        out[h.specific_humidity] = q.clip(min=0.0).assign_attrs(raw[h.specific_humidity].attrs)
        return out

    def _clip(self, name: str, values: np.ndarray) -> np.ndarray:
        lo, hi = (self.clips.get(name) or [None, None])
        if lo is not None or hi is not None:
            values = np.clip(values, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
        return values


# ---------------------------------------------------------------- fitting
def _group(times: pd.Series, level: str) -> pd.Series:
    if level == 'annual':
        return pd.Series(1, index=times.index)
    if level == 'season':
        return times.dt.month.map(SEASON)
    if level == 'month':
        return times.dt.month
    raise ValidationError(f'unknown pooling level {level!r}')


def _factors(pairs: pd.DataFrame, spec: CorrectionSpec,
             limited: Optional[List[str]] = None) -> Dict[str, float]:
    if spec.level == 'none':
        return {m: spec.neutral for m in MONTHS}
    times = pd.to_datetime(pairs['time'])
    key = _group(times, spec.level)
    ref = pairs[spec.reference_column].groupby(key).mean()
    src = pairs[spec.variable].groupby(key).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        fitted = ref - src if spec.kind == 'add' else ref / src
    if spec.kind == 'mult':
        dry = src <= spec.min_source_mean
        if dry.any() and limited is not None:
            limited.extend(f'{spec.variable}[{k}]: source mean {src[k]:.4g} <= {spec.min_source_mean:g}, left at 1'
                           for k in src.index[dry])
        fitted = fitted.where(~dry, np.nan)
        if spec.factor_bounds is not None:
            low, high = spec.factor_bounds
            clamped = fitted.clip(lower=low, upper=high)
            hit = (clamped != fitted) & fitted.notna()
            if hit.any() and limited is not None:
                limited.extend(f'{spec.variable}[{k}]: factor {fitted[k]:.4g} clamped to {clamped[k]:.4g}'
                               for k in fitted.index[hit])
            fitted = clamped
    out = {}
    for m in range(1, 13):
        k = 1 if spec.level == 'annual' else SEASON[m] if spec.level == 'season' else m
        v = fitted.get(k, np.nan)
        # a group absent from the training data (or a zero source mean) is left uncorrected
        out[str(m)] = float(v) if np.isfinite(v) else spec.neutral
    return out


def fit_source_correction(pairs: pd.DataFrame, specs: Sequence[CorrectionSpec], source: str,
                          meta: Optional[Mapping[str, Any]] = None) -> ForcingSourceCorrection:
    """Fit monthly factors from paired source/reference values.

    Args:
        pairs: Table with ``time``, each spec's ``variable`` and its reference column.
        specs: One :class:`CorrectionSpec` per corrected variable.
        source: Name of the source (and generation) the correction belongs to.
        meta: Extra metadata stored with the correction.
    """
    needed = {'time'} | {s.variable for s in specs} | {s.reference_column for s in specs}
    missing = sorted(needed - set(pairs.columns))
    if missing:
        raise ValidationError(f'pairs lack columns {missing}')
    clean = pairs.dropna(subset=sorted(needed))
    if clean.empty:
        raise ValidationError('no complete pairs to fit')
    times = pd.to_datetime(clean['time'])
    limited: List[str] = []
    factors = {s.variable: _factors(clean, s, limited) for s in specs}
    info = {'created': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M'),
            'training_period': [str(times.min()), str(times.max())], 'n_pairs': int(len(clean)),
            'levels': {s.variable: s.level for s in specs}, 'limited': limited, **dict(meta or {})}
    return ForcingSourceCorrection(
        source=source,
        factors=factors,
        kinds={s.variable: s.kind for s in specs},
        clips={s.variable: [s.clip_min, s.clip_max] for s in specs},
        meta=info,
    )


def _scores(frame: pd.DataFrame, specs: Sequence[CorrectionSpec]) -> Dict[str, Dict[str, float]]:
    out = {}
    for s in specs:
        a, b = frame[s.variable].astype(float), frame[s.reference_column].astype(float)
        out[s.variable] = {'bias': float((a - b).mean()), 'rmse': float(np.sqrt(((a - b) ** 2).mean())),
                           'r': float(a.corr(b)), 'mean_ref': float(b.mean())}
    return out


def split_sample_test(pairs: pd.DataFrame, specs: Sequence[CorrectionSpec]) -> pd.DataFrame:
    """Out-of-sample skill of the specs: fit on one half in time, test on the other, both ways.

    Returns one row per variable and test (``forward``: first half -> second,
    ``reverse``: second -> first) with raw and corrected bias and RMSE, so that
    the pooling level of each variable can be chosen on evidence.
    """
    clean = pairs.dropna(subset=['time'] + [s.variable for s in specs] + [s.reference_column for s in specs])
    times = pd.to_datetime(clean['time'])
    middle = times.min() + (times.max() - times.min()) / 2
    a, b = clean[times < middle], clean[times >= middle]
    if a.empty or b.empty:
        raise ValidationError('split-sample test needs pairs in both halves of the period')
    rows = []
    for test, train, held in (('forward', a, b), ('reverse', b, a)):
        corr = fit_source_correction(train, specs, source='split-test')
        raw, fixed = _scores(held, specs), _scores(corr.apply_frame(held), specs)
        for s in specs:
            rows.append({'variable': s.variable, 'level': s.level, 'test': test, 'n_test': len(held),
                         'raw_bias': raw[s.variable]['bias'], 'raw_rmse': raw[s.variable]['rmse'],
                         'corrected_bias': fixed[s.variable]['bias'],
                         'corrected_rmse': fixed[s.variable]['rmse']})
    return pd.DataFrame(rows)
