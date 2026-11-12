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
* stored as twelve monthly values per variable, with the source name and
  training period.  :meth:`ForcingCorrection.apply` refuses a dataset from
  another source: a new source (or a new model generation of the same source)
  needs its own fit.

Pairs are best built after both products pass through the same mapping onto
the model's units (HRUs or cells), so the fit sees what the model sees.
Building them is source specific and left to the caller: :func:`fit_correction`
takes a table with a ``time`` column, the source variables and their
reference counterparts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ValidationError

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
    """

    variable: str
    kind: Literal['add', 'mult']
    level: Level = 'month'
    reference: Optional[str] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    @property
    def reference_column(self) -> str:
        return self.reference or f'ref_{self.variable}'

    @property
    def neutral(self) -> float:
        return 0.0 if self.kind == 'add' else 1.0


@dataclass
class ForcingCorrection:
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
    def load(cls, path: Union[str, Path]) -> 'ForcingCorrection':
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

    def apply(self, ds: xr.Dataset, source: str) -> xr.Dataset:
        """Correct a forcing dataset with a ``time`` dimension; refuses another source."""
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
        stamp = self.meta.get('created', '')
        return out.assign_attrs(ds.attrs, forcing_correction=f'{self.source}:{stamp}')

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


def _factors(pairs: pd.DataFrame, spec: CorrectionSpec) -> Dict[str, float]:
    if spec.level == 'none':
        return {m: spec.neutral for m in MONTHS}
    times = pd.to_datetime(pairs['time'])
    key = _group(times, spec.level)
    ref = pairs[spec.reference_column].groupby(key).mean()
    src = pairs[spec.variable].groupby(key).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        fitted = ref - src if spec.kind == 'add' else ref / src
    out = {}
    for m in range(1, 13):
        k = 1 if spec.level == 'annual' else SEASON[m] if spec.level == 'season' else m
        v = fitted.get(k, np.nan)
        # a group absent from the training data (or a zero source mean) is left uncorrected
        out[str(m)] = float(v) if np.isfinite(v) else spec.neutral
    return out


def fit_correction(pairs: pd.DataFrame, specs: Sequence[CorrectionSpec], source: str,
                   meta: Optional[Mapping[str, Any]] = None) -> ForcingCorrection:
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
    info = {'created': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M'),
            'training_period': [str(times.min()), str(times.max())], 'n_pairs': int(len(clean)),
            'levels': {s.variable: s.level for s in specs}, **dict(meta or {})}
    return ForcingCorrection(
        source=source,
        factors={s.variable: _factors(clean, s) for s in specs},
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
        corr = fit_correction(train, specs, source='split-test')
        raw, fixed = _scores(held, specs), _scores(corr.apply_frame(held), specs)
        for s in specs:
            rows.append({'variable': s.variable, 'level': s.level, 'test': test, 'n_test': len(held),
                         'raw_bias': raw[s.variable]['bias'], 'raw_rmse': raw[s.variable]['rmse'],
                         'corrected_bias': fixed[s.variable]['bias'],
                         'corrected_rmse': fixed[s.variable]['rmse']})
    return pd.DataFrame(rows)
