# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CF-aware classification of streamflow unit strings.

Streamflow reaches the evaluator in two physically different forms:

* **volumetric discharge** (``m3 s-1``, ``m3/s``, ``m^3 s^-1``, ``m3.s-1``,
  ``m³/s``, ``cms``, ``l/s`` ...) — already basin-scale; multiplying it by a
  catchment area is always wrong;
* **depth rates** (``m s-1``, ``mm/s``, ``mm/h``, ``mm/day``) and **mass
  fluxes** (``kg m-2 s-1``) — per unit area; they become discharge only after
  multiplying by the contributing area.

Matching hard-coded spellings (``'m3/s' in units``) let the CF spelling
``m3 s-1`` fall through to the depth-rate branch, which then multiplied an
already-volumetric flow by the catchment area. :func:`classify_streamflow_units`
parses the unit string into length/time/mass exponents instead, so every
spelling of the same unit lands in the same branch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

VOLUMETRIC = 'volumetric'
DEPTH_RATE = 'depth_rate'
UNKNOWN = 'unknown'

# Bare aliases that do not parse as exponent products.
_ALIASES: Dict[str, Tuple[str, float]] = {
    'cms': (VOLUMETRIC, 1.0),
    'cumec': (VOLUMETRIC, 1.0),
    'cumecs': (VOLUMETRIC, 1.0),
    'cfs': (VOLUMETRIC, 0.3048 ** 3),
}

# symbol -> (dimension, factor to SI base: m, s, kg)
_SYMBOLS: Dict[str, Tuple[str, float]] = {
    'm': ('L', 1.0), 'meter': ('L', 1.0), 'meters': ('L', 1.0), 'metre': ('L', 1.0), 'metres': ('L', 1.0),
    'mm': ('L', 1e-3), 'cm': ('L', 1e-2), 'km': ('L', 1e3), 'ft': ('L', 0.3048),
    's': ('T', 1.0), 'sec': ('T', 1.0), 'second': ('T', 1.0), 'seconds': ('T', 1.0),
    'min': ('T', 60.0), 'minute': ('T', 60.0),
    'h': ('T', 3600.0), 'hr': ('T', 3600.0), 'hour': ('T', 3600.0), 'hours': ('T', 3600.0),
    'd': ('T', 86400.0), 'day': ('T', 86400.0), 'days': ('T', 86400.0),
    'kg': ('M', 1.0),
}
# Litres are a volume: one symbol carrying L^3.
_VOLUME_SYMBOLS: Dict[str, float] = {'l': 1e-3, 'litre': 1e-3, 'liter': 1e-3, 'litres': 1e-3, 'liters': 1e-3}

_TOKEN = re.compile(r'^([a-z]+)(-?\d+)?$')
_WATER_DENSITY = 1000.0  # kg m-3


@dataclass(frozen=True)
class StreamflowUnits:
    """Result of classifying a unit string.

    Attributes:
        kind: ``'volumetric'`` (discharge), ``'depth_rate'`` (per unit area,
            including mass fluxes converted at water density) or ``'unknown'``.
        factor: Multiplier to m³/s for volumetric units, to m/s for depth rates.
        raw: The original unit string.
    """

    kind: str
    factor: float
    raw: str

    @property
    def is_volumetric(self) -> bool:
        return self.kind == VOLUMETRIC

    @property
    def is_depth_rate(self) -> bool:
        return self.kind == DEPTH_RATE


def _canonical(units: str) -> str:
    s = units.strip().lower()
    s = s.replace('³', '3').replace('²', '2').replace('⁻¹', '-1').replace('−', '-')
    s = s.replace('**', '').replace('^', '').replace('(', '').replace(')', '')
    s = re.sub(r'[\s.*·]+', ' ', s).strip()
    if s.count('/') == 1:
        num, den = s.split('/')
        den_tokens = []
        for tok in den.split():
            match = _TOKEN.match(tok)
            if match is None:
                return s
            exp = int(match.group(2)) if match.group(2) else 1
            den_tokens.append(f"{match.group(1)}{-exp}")
        s = ' '.join(num.split() + den_tokens)
    return s


def classify_streamflow_units(units: Optional[str]) -> StreamflowUnits:
    """Classify a streamflow unit string as volumetric, depth-rate or unknown.

    Accepted volumetric spellings include ``m3 s-1``, ``m3/s``, ``m^3 s^-1``,
    ``m3.s-1``, ``m³/s``, ``m**3/s``, ``cms``, ``l/s`` and ``cfs``. Depth rates
    include ``m s-1``, ``m/s``, ``mm/s``, ``mm/h``, ``mm/day``, ``mm d-1`` and
    the mass flux ``kg m-2 s-1`` (converted at 1000 kg m⁻³).
    """
    raw = '' if units is None else str(units)
    if not raw.strip():
        return StreamflowUnits(UNKNOWN, float('nan'), raw)

    s = _canonical(raw)
    if s in _ALIASES:
        kind, factor = _ALIASES[s]
        return StreamflowUnits(kind, factor, raw)

    dims = {'L': 0, 'T': 0, 'M': 0}
    factor = 1.0
    for tok in s.split():
        match = _TOKEN.match(tok)
        if match is None:
            return StreamflowUnits(UNKNOWN, float('nan'), raw)
        sym, exp_str = match.group(1), match.group(2)
        exp = int(exp_str) if exp_str else 1
        if sym in _VOLUME_SYMBOLS:
            dims['L'] += 3 * exp
            factor *= _VOLUME_SYMBOLS[sym] ** exp
        elif sym in _SYMBOLS:
            dim, sym_factor = _SYMBOLS[sym]
            dims[dim] += exp
            factor *= sym_factor ** exp
        else:
            return StreamflowUnits(UNKNOWN, float('nan'), raw)

    if dims == {'L': 3, 'T': -1, 'M': 0}:
        return StreamflowUnits(VOLUMETRIC, factor, raw)
    if dims == {'L': 1, 'T': -1, 'M': 0}:
        return StreamflowUnits(DEPTH_RATE, factor, raw)
    if dims == {'L': -2, 'T': -1, 'M': 1}:
        return StreamflowUnits(DEPTH_RATE, factor / _WATER_DENSITY, raw)
    return StreamflowUnits(UNKNOWN, float('nan'), raw)
