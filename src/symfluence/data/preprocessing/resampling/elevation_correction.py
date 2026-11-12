# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Elevation corrections of remapped forcing (the ``lapse`` config section).

Remapping averages each forcing variable over the grid cells an HRU overlaps,
so the remapped value represents the area-weighted elevation of those cells,
not the HRU's own elevation. The corrections here move the non-temperature
variables to the HRU elevation, using

    dz = HRU elevation - area-weighted elevation of the overlapping cells  (m)

Air temperature itself is NOT changed: model preprocessors apply the
temperature lapse rate (``APPLY_LAPSE_RATE`` / ``LAPSE_RATE``). The lapsed
temperature is only used internally, for the pressure and humidity terms.

Corrections (each behind its own toggle, all off by default):

- pressure: hypsometric equation with the mean temperature of the layer.
- humidity: specific humidity recomputed so that relative humidity is
  conserved at the corrected temperature and pressure.
- longwave: additive gradient (default -2.9 W m-2 per 100 m, Marty et al. 2002).
- shortwave, precipitation, wind: multiplicative, ``1 + gradient * dz``,
  floored at zero.

The corrections are applied by the remapper to each file before it is
published, and the active settings are part of the remapping signature, so
files written under other settings are regenerated rather than reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from symfluence.core.constants import PhysicalConstants
from symfluence.core.exceptions import ConfigurationError

#: NetCDF global attribute recording the corrections applied to a file (JSON).
ATTRIBUTE = 'symfluence_elevation_corrections'
VERSION = 'elevation-corrections-v1'

TOGGLES = ('pressure', 'humidity', 'longwave', 'shortwave', 'precipitation', 'wind')

_GRAVITY = 9.80665
_EPSILON = 0.622  # ratio of the gas constants of dry air and water vapour


@dataclass(frozen=True)
class ElevationCorrectionSettings:
    """The active part of the ``lapse`` section, in picklable form."""

    pressure: bool = False
    humidity: bool = False
    longwave: bool = False
    shortwave: bool = False
    precipitation: bool = False
    wind: bool = False
    lw_gradient: float = -0.029
    sw_gradient: float = 0.00002
    precip_gradient: float = 0.0004
    wind_gradient: float = 0.0
    #: Temperature lapse rate in K/m, positive = cooler with height; 0 when
    #: ``APPLY_LAPSE_RATE`` is off (temperature is then left at the forcing
    #: elevation by the models, and the same is assumed here).
    temperature_lapse_rate: float = 0.0

    @property
    def requested(self) -> List[str]:
        return [name for name in TOGGLES if getattr(self, name)]

    @property
    def active(self) -> bool:
        return bool(self.requested)

    def signature(self) -> str:
        """Hash of the active settings; empty when nothing is switched on.

        Empty-when-inactive keeps the remapping signature of every existing
        (uncorrected) cache unchanged.
        """
        if not self.active:
            return ''
        payload = json.dumps([VERSION, asdict(self)], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[Mapping[str, Any]]) -> 'ElevationCorrectionSettings':
        return cls(**dict(values or {}))

    @classmethod
    def from_config(cls, config: Any) -> 'ElevationCorrectionSettings':
        """Read the ``lapse`` section and the temperature lapse rate from a typed config."""
        forcing = getattr(config, 'forcing', None)
        lapse = getattr(forcing, 'lapse', None)
        if lapse is None:
            return cls()
        values = lapse if isinstance(lapse, Mapping) else lapse.model_dump()
        fields = {name: values[name] for name in cls.__dataclass_fields__ if name in values}
        fields['temperature_lapse_rate'] = (
            normalise_lapse_rate(getattr(forcing, 'lapse_rate', 0.0065))
            if getattr(forcing, 'apply_lapse_rate', True) else 0.0)
        return cls(**fields)


def normalise_lapse_rate(raw: float) -> float:
    """``LAPSE_RATE`` in K/m. Magnitudes of 0.1 or more are read as K/km."""
    raw = float(raw)
    return raw if abs(raw) < 0.1 else raw / 1000.0


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def saturation_vapour_pressure(temp_k):
    """Saturation vapour pressure over water (Pa), Magnus form (Alduchov & Eskridge 1996)."""
    temp_c = np.asarray(temp_k, dtype=float) - PhysicalConstants.KELVIN_OFFSET
    return 610.94 * np.exp(17.625 * temp_c / (temp_c + 243.04))


def correct_pressure(pressure, temp_k, dz, lapse_rate):
    """Pressure at the HRU elevation (hypsometric equation).

    ``temp_k`` is the temperature at the forcing elevation; the layer mean is
    taken as the average of that and the lapsed temperature.
    """
    layer_temp = np.asarray(temp_k, dtype=float) - 0.5 * lapse_rate * dz
    exponent = -_GRAVITY * PhysicalConstants.MOLAR_MASS_DRY_AIR * dz / (
        PhysicalConstants.UNIVERSAL_GAS_CONSTANT * layer_temp)
    return np.asarray(pressure, dtype=float) * np.exp(exponent)


def correct_specific_humidity(q, pressure_old, temp_old, pressure_new, temp_new):
    """Specific humidity that keeps relative humidity unchanged."""
    q = np.asarray(q, dtype=float)
    vapour = q * pressure_old / (_EPSILON + (1.0 - _EPSILON) * q)
    rel_hum = vapour / saturation_vapour_pressure(temp_old)
    vapour_new = rel_hum * saturation_vapour_pressure(temp_new)
    return _EPSILON * vapour_new / (pressure_new - (1.0 - _EPSILON) * vapour_new)


def correct_longwave(longwave, dz, gradient):
    """Additive longwave gradient (W m-2 per m), floored at zero."""
    return np.maximum(np.asarray(longwave, dtype=float) + gradient * dz, 0.0)


def scale_with_elevation(values, dz, gradient):
    """Multiplicative correction ``1 + gradient * dz``, floored at zero."""
    return np.asarray(values, dtype=float) * np.maximum(1.0 + gradient * dz, 0.0)


# ---------------------------------------------------------------------------
# Elevation differences from the remapping intersection
# ---------------------------------------------------------------------------

def find_intersection_table(remap_file: Path) -> Path:
    """The intersection table (HRU x forcing cell, with elevations) next to the weights."""
    directory = Path(remap_file).parent
    tables = sorted(directory.glob('*_intersected_shapefile.csv'))
    if tables:
        return tables[0]
    shapes = sorted(directory.glob('*_intersected_shapefile.shp'))
    if shapes:
        return shapes[0]
    raise ConfigurationError(
        f"Elevation corrections ('lapse' section) need the intersected shapefile of "
        f"the remapping, but none was found in {directory}.")


def _read_intersection(table: Path) -> pd.DataFrame:
    if table.suffix == '.csv':
        header = pd.read_csv(table, nrows=0).columns
        wanted = [c for c in header if c != 'geometry']
        return pd.read_csv(table, usecols=wanted)
    import geopandas as gpd
    frame = pd.DataFrame(gpd.read_file(table).drop(columns='geometry'))
    if 'weight' not in frame.columns and 'AP1' in frame.columns:
        frame['weight'] = frame['AP1']
    return frame


def hru_elevation_differences(
    table: Path, hru_field: str, logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """dz (m) per HRU id: HRU elevation minus the weighted forcing-cell elevation.

    Forcing cells whose elevation is missing (0, nodata, or implausibly far
    from the HRU) contribute no difference, as in the SUMMA temperature
    lapse: a fabricated elevation would otherwise become a large bias.
    """
    logger = logger or logging.getLogger(__name__)
    frame = _read_intersection(Path(table))
    hru_col = next((c for c in (f'S_1_{hru_field}', f'S_1_{hru_field}'[:10]) if c in frame.columns), None)
    hru_elev = next((c for c in ('S_1_elev_m', 'S_1_elev_mean', 'S_1_elev_me') if c in frame.columns), None)
    missing = [name for name, col in (
        (f'S_1_{hru_field}', hru_col), ('S_1_elev_m', hru_elev),
        ('S_2_elev_m', 'S_2_elev_m' if 'S_2_elev_m' in frame.columns else None),
        ('weight', 'weight' if 'weight' in frame.columns else None)) if col is None]
    if missing:
        raise ConfigurationError(
            f"Elevation corrections ('lapse' section) need column(s) {missing} in {table}. "
            "The catchment shapefile must carry a mean elevation (elev_mean) and the "
            "forcing shapefile a cell elevation.")

    catchment = frame[hru_elev].astype(float)
    forcing = frame['S_2_elev_m'].astype(float)
    weight = frame['weight'].astype(float)
    invalid = (~np.isfinite(forcing) | (forcing == 0.0) | (forcing <= -100.0)
               | ((forcing - catchment).abs() > 4000.0))
    if invalid.any():
        logger.warning(
            f"{int(invalid.sum())}/{len(frame)} forcing cell(s) have an invalid elevation "
            "(0, nodata, or implausible vs the catchment); no elevation correction is "
            "applied for those cells.")
    difference = (catchment - forcing).where(~invalid, 0.0)
    weighted = (weight * difference).groupby(frame[hru_col]).sum()
    total = weight.groupby(frame[hru_col]).sum()
    dz = (weighted / total.where(total > 0)).fillna(0.0)
    dz.index = dz.index.astype('int64')
    return dz


# ---------------------------------------------------------------------------
# Application to a remapped file
# ---------------------------------------------------------------------------

def apply_elevation_corrections(
    path: Path,
    dz_by_hru: pd.Series,
    settings: ElevationCorrectionSettings,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Correct the variables of a remapped (time, hru) forcing file in place.

    Returns the names of the corrections applied. A file already carrying the
    attribute is refused: applying the corrections twice would compound them.
    """
    from netCDF4 import Dataset

    logger = logger or logging.getLogger(__name__)
    if not settings.active:
        return []

    with Dataset(path, 'a') as nc:
        if ATTRIBUTE in nc.ncattrs():
            raise ValueError(f'{Path(path).name} already carries elevation corrections')
        if 'hruId' not in nc.variables:
            raise ValueError(f'{Path(path).name}: no hruId variable; cannot match HRU elevations')
        hru_ids = np.asarray(nc.variables['hruId'][:]).astype('int64')
        unknown = sorted(set(hru_ids.tolist()) - set(dz_by_hru.index.tolist()))
        if unknown:
            raise ValueError(
                f'{Path(path).name}: HRU id(s) {unknown[:10]} have no elevation in the '
                'remapping intersection')
        dz_hru = dz_by_hru.reindex(hru_ids).to_numpy(dtype=float)

        def shaped(name: str) -> np.ndarray:
            """dz broadcast against variable *name* along its hru dimension."""
            dims = nc.variables[name].dimensions
            shape = [len(dz_hru) if d == 'hru' else 1 for d in dims]
            return dz_hru.reshape(shape)

        def read(name: str) -> np.ndarray:
            return np.ma.filled(nc.variables[name][:].astype('float64'), np.nan)

        def write(name: str, values: np.ndarray) -> None:
            var = nc.variables[name]
            fill = getattr(var, '_FillValue', None)
            if fill is not None:
                values = np.where(np.isfinite(values), values, fill)
            var[:] = values.astype(var.dtype)

        def usable(name: str) -> bool:
            return name in nc.variables and 'hru' in nc.variables[name].dimensions

        applied: List[str] = []
        skipped: List[str] = []
        lapse = settings.temperature_lapse_rate
        has_temp = usable('air_temperature')
        temp_old = read('air_temperature') if has_temp else None
        temp_new = temp_old - lapse * shaped('air_temperature') if has_temp else None

        pressure_old = read('surface_air_pressure') if usable('surface_air_pressure') else None
        pressure_new = pressure_old
        if settings.pressure:
            if pressure_old is not None and has_temp:
                pressure_new = correct_pressure(
                    pressure_old, temp_old, shaped('surface_air_pressure'), lapse)
                write('surface_air_pressure', pressure_new)
                applied.append('pressure')
            else:
                skipped.append('pressure')

        if settings.humidity:
            if usable('specific_humidity') and pressure_old is not None and has_temp:
                write('specific_humidity', correct_specific_humidity(
                    read('specific_humidity'), pressure_old, temp_old, pressure_new, temp_new))
                applied.append('humidity')
            else:
                skipped.append('humidity')

        if settings.longwave:
            name = 'surface_downwelling_longwave_flux'
            if usable(name):
                write(name, correct_longwave(read(name), shaped(name), settings.lw_gradient))
                applied.append('longwave')
            else:
                skipped.append('longwave')

        for toggle, name, gradient in (
            ('shortwave', 'surface_downwelling_shortwave_flux', settings.sw_gradient),
            ('precipitation', 'precipitation_flux', settings.precip_gradient),
            ('wind', 'wind_speed', settings.wind_gradient),
        ):
            if not getattr(settings, toggle):
                continue
            if usable(name):
                write(name, scale_with_elevation(read(name), shaped(name), gradient))
                applied.append(toggle)
            else:
                skipped.append(toggle)

        if skipped:
            logger.warning(
                f"{Path(path).name}: elevation correction requested for {', '.join(skipped)} "
                "but the variable (or the temperature/pressure it needs) is not in the "
                "remapped forcing; not applied.")
        nc.setncattr(ATTRIBUTE, json.dumps(
            {'version': VERSION, 'applied': applied, 'settings': settings.to_dict()},
            sort_keys=True))
    return applied
