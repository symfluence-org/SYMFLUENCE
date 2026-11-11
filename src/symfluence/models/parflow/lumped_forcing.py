# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Lumped forcing for ParFlow and CLM-ParFlow.

One implementation of the pieces both lumped preprocessors (and their
calibration parameter managers) need: the simulation length, Oudin PET, the
elevation-band Snow-17 that turns basin-mean precipitation and temperature
into rain + melt, and the daily net-rainfall series written to the ``.pfidb``.

The Snow-17 elevation bands describe the hypsometry of the catchment being
modelled. They are derived from the domain's DEM inside the catchment outline
(five equal-area bands at the 10/30/50/70/90th elevation percentiles, lapsed
from the catchment mean elevation); when that is not available a single band
at the catchment mean elevation is used, i.e. no lapse adjustment. The bands
and latitude used are stored with the hourly forcing cache so calibration
re-runs Snow-17 with exactly the same geometry.

Basin-mode CLM-ParFlow does not use this module (CLM computes its own snow).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.mixins.project import resolve_data_subdir

DEFAULT_LAPSE_RATE = 0.0065  # degC/m, standard environmental lapse rate
BAND_PERCENTILES = (10.0, 30.0, 50.0, 70.0, 90.0)  # five equal-area bands
FORCING_CACHE_NAME = 'hourly_forcing_cache.npz'


@dataclass(frozen=True)
class SnowBands:
    """Elevation bands of the catchment for the lumped Snow-17.

    ``reference_elev`` is the elevation the basin-mean temperature represents
    (the catchment mean); band temperatures are lapsed from it.
    """

    elevations: Tuple[float, ...]
    fractions: Tuple[float, ...]
    reference_elev: float
    source: str

    @classmethod
    def single(cls, elevation: Optional[float], source: str) -> 'SnowBands':
        """One band at the catchment mean elevation (no lapse adjustment)."""
        elev = float('nan') if elevation is None else float(elevation)
        return cls((elev,), (1.0,), elev, source)

    def temperature_offsets(self, lapse_rate: float) -> List[float]:
        """Temperature offset of each band relative to the basin-mean series."""
        if len(self.elevations) == 1:
            return [0.0]
        return [-lapse_rate * (elev - self.reference_elev) for elev in self.elevations]


def simulation_time_info(start: Any, end: Any, timestep_hours: float) -> Tuple[int, float, int]:
    """(number of steps, step length in hours, simulation length in hours)."""
    start_dt = pd.Timestamp(str(start))
    end_dt = pd.Timestamp(str(end))
    n_hours = int((end_dt - start_dt).total_seconds() / 3600)
    if n_hours <= 0:
        n_hours = 8760
    n_steps = max(1, int(math.ceil(n_hours / timestep_hours)))
    return n_steps, timestep_hours, n_hours


def oudin_pet_mm_hr(temp_k: np.ndarray, dates: Any, latitude_deg: float) -> np.ndarray:
    """Oudin potential evapotranspiration as an hourly rate (mm/hr)."""
    temp_c = temp_k - 273.15
    doy = pd.DatetimeIndex(dates).dayofyear
    lat_rad = np.radians(latitude_deg)

    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    ws = np.clip(ws, 0.0, np.pi)
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )

    lam_rho = 2.45 * 1000.0
    pet_mm_day = np.where(
        temp_c > -5.0,
        ra / lam_rho * (temp_c + 5.0) / 100.0 * 1000.0,
        0.0,
    )
    pet_mm_day = np.maximum(pet_mm_day, 0.0)
    return pet_mm_day / 24.0


# ---------------------------------------------------------------------------
# Catchment geometry
# ---------------------------------------------------------------------------

ForcingSelector = Callable[[List[Path]], List[Path]]


def location_from_forcing(
    forcing_dir: Optional[Path],
    forcing_selector: Optional[ForcingSelector] = None,
) -> Optional[Tuple[float, float]]:
    """(lat, lon) of the first cell of the basin-averaged forcing, if it has them.

    ``forcing_selector`` narrows the files to this run's HRU set (the owning
    preprocessor's ``_select_run_forcing``, issue #339).
    """
    if not forcing_dir:
        return None
    import xarray as xr
    nc_files = sorted(Path(forcing_dir).glob('*.nc'))
    if forcing_selector is not None:
        nc_files = forcing_selector(nc_files)
    if not nc_files:
        return None
    with xr.open_dataset(nc_files[0]) as ds:
        if 'latitude' in ds and 'longitude' in ds:
            return float(ds['latitude'].values.flat[0]), float(ds['longitude'].values.flat[0])
    return None


def catchment_dem_elevations(dem_path: Path, catchment_path: Path) -> np.ndarray:
    """Valid DEM elevations (m) inside the catchment outline."""
    import geopandas as gpd
    import rasterio
    import rasterio.mask

    catchment = gpd.read_file(catchment_path)
    if catchment.crs is None:
        catchment = catchment.set_crs(epsg=4326)
    with rasterio.open(dem_path) as src:
        if src.crs is not None:
            catchment = catchment.to_crs(src.crs)
        data, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, filled=False)
    band = np.ma.masked_invalid(data[0].astype('float64'))
    values = band.compressed()
    return values[values > -1000.0]


def bands_from_elevations(elevations: np.ndarray, source: str) -> SnowBands:
    """Five equal-area bands at the 10/30/50/70/90th elevation percentiles."""
    values = np.asarray(elevations, dtype='float64')
    if values.size == 0:
        raise ConfigurationError('No valid elevations to derive Snow-17 elevation bands from')
    band_elevs = tuple(float(v) for v in np.percentile(values, BAND_PERCENTILES))
    fraction = 1.0 / len(band_elevs)
    return SnowBands(band_elevs, (fraction,) * len(band_elevs), float(values.mean()), source)


def resolve_snow_bands(
    dem_path: Optional[Path],
    catchment_path: Optional[Path],
    catchment_mean_elev: Optional[float],
    logger: logging.Logger,
) -> SnowBands:
    """Snow-17 elevation bands of this domain.

    DEM hypsometry inside the catchment outline when both files exist,
    otherwise a single band at the catchment mean elevation.
    """
    if dem_path is not None and catchment_path is not None \
            and Path(dem_path).exists() and Path(catchment_path).exists():
        bands = bands_from_elevations(
            catchment_dem_elevations(Path(dem_path), Path(catchment_path)),
            f'DEM hypsometry ({Path(dem_path).name})',
        )
        logger.info(
            f"Snow-17 elevation bands from {bands.source}: "
            f"{[round(e) for e in bands.elevations]} m, catchment mean {bands.reference_elev:.0f} m"
        )
        return bands
    logger.warning(
        "No DEM/catchment outline found to derive Snow-17 elevation bands "
        f"(DEM: {dem_path}, catchment: {catchment_path}); using a single band at the "
        "catchment mean elevation (no lapse adjustment)."
    )
    return SnowBands.single(catchment_mean_elev, 'single band at catchment mean elevation')


# ---------------------------------------------------------------------------
# Elevation-band Snow-17
# ---------------------------------------------------------------------------

def band_rain_plus_melt(
    daily_ppt: np.ndarray,
    daily_temp: np.ndarray,
    doy: np.ndarray,
    bands: SnowBands,
    latitude: float,
    lapse_rate: float = DEFAULT_LAPSE_RATE,
    snow17_params: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Area-weighted daily rain + melt (mm/day) from one Snow-17 per band.

    Low bands melt earlier and high bands later, which is what spreads the
    freshet of a snowmelt-dominated basin over spring and summer.
    """
    from jsnow17.bmi import Snow17BMI

    rpm_weighted = np.zeros(len(daily_ppt))
    info: List[str] = []
    offsets = bands.temperature_offsets(lapse_rate)
    for band_elev, area_frac, temp_offset in zip(bands.elevations, bands.fractions, offsets):
        if snow17_params is None:
            snow = Snow17BMI(latitude=latitude, dt=1.0)
        else:
            snow = Snow17BMI(params=snow17_params, latitude=latitude, dt=1.0)
        snow.initialize()
        rpm_band = snow.update_batch(daily_ppt, daily_temp + temp_offset, doy)
        rpm_weighted += rpm_band * area_frac

        peak_swe = max(
            float(snow.get_value('swe')),
            float(snow.get_value('w_i') + snow.get_value('w_q'))
        )
        info.append(f"{band_elev:.0f}m: offset={temp_offset:+.1f}°C, peak_SWE={peak_swe:.0f}mm")
    return rpm_weighted, info


def apply_snow_model(
    ppt_mm_hr: np.ndarray,
    temp_c_hourly: np.ndarray,
    times: Any,
    bands: SnowBands,
    latitude: float,
    logger: logging.Logger,
    lapse_rate: Optional[float] = None,
) -> np.ndarray:
    """Hourly liquid water input (rain + snowmelt, mm/hr) from hourly forcing.

    Snow-17 runs daily per elevation band; each day's area-weighted rain +
    melt is spread uniformly over that day's hours.
    """
    if lapse_rate is None:
        lapse_rate = DEFAULT_LAPSE_RATE

    time_idx = pd.DatetimeIndex(times)
    df_hourly = pd.DataFrame({'ppt': ppt_mm_hr, 'temp': temp_c_hourly}, index=time_idx)
    daily = df_hourly.resample('D').agg({
        'ppt': 'sum',    # total mm/day
        'temp': 'mean',  # mean daily temperature
    })
    daily['doy'] = daily.index.dayofyear

    rpm_weighted, band_info = band_rain_plus_melt(
        daily['ppt'].values, daily['temp'].values, daily['doy'].values,
        bands, latitude, lapse_rate,
    )

    # Map daily totals back to hourly rates (mm/day -> mm/hr)
    daily_rpm = pd.Series(rpm_weighted, index=daily.index)
    effective = np.zeros(len(df_hourly))
    for date, rpm_mm_day in daily_rpm.items():
        mask = df_hourly.index.normalize() == date
        n_hours = mask.sum()
        if n_hours > 0:
            effective[mask] = rpm_mm_day / n_hours

    logger.info(f"Snow-17 elevation bands (lapse={lapse_rate*1000:.1f} °C/km, {bands.source}):")
    for line in band_info:
        logger.info(f"  {line}")
    logger.info(
        f"  Area-weighted: mean rain+melt={np.mean(rpm_weighted):.2f} mm/day, "
        f"mean effective={np.mean(effective):.4f} mm/hr"
    )
    return effective


# ---------------------------------------------------------------------------
# Forcing cache (preprocessor -> calibration parameter manager)
# ---------------------------------------------------------------------------

def save_forcing_cache(
    directory: Path,
    ppt_mm_hr: np.ndarray,
    pet_mm_hr: np.ndarray,
    temp_c: np.ndarray,
    times: np.ndarray,
    bands: SnowBands,
    latitude: float,
) -> None:
    """Cache hourly forcing with the Snow-17 geometry it was processed with."""
    np.savez_compressed(
        Path(directory) / FORCING_CACHE_NAME,
        ppt_mm_hr=ppt_mm_hr, pet_mm_hr=pet_mm_hr,
        temp_c=temp_c, times=times.astype('int64'),
        band_elevations=np.asarray(bands.elevations, dtype='float64'),
        band_fractions=np.asarray(bands.fractions, dtype='float64'),
        band_reference_elev=np.float64(bands.reference_elev),
        latitude=np.float64(latitude),
    )


def load_cached_snow_setup(cache: Any, cache_path: Path) -> Tuple[SnowBands, float]:
    """Snow-17 bands and latitude stored with the hourly forcing cache.

    Raises:
        ConfigurationError: the cache predates the stored geometry (it was
            written with bands and a latitude hard-coded for another basin).
    """
    needed = ('band_elevations', 'band_fractions', 'band_reference_elev', 'latitude')
    missing = [name for name in needed if name not in cache]
    if missing:
        raise ConfigurationError(
            f"{cache_path} holds no Snow-17 elevation bands/latitude ({missing}); it was written by "
            "an older preprocessor that used bands hard-coded for another basin. Re-run model "
            "preprocessing to regenerate the forcing for this domain."
        )
    bands = SnowBands(
        tuple(float(v) for v in cache['band_elevations']),
        tuple(float(v) for v in cache['band_fractions']),
        float(cache['band_reference_elev']),
        'hourly forcing cache',
    )
    return bands, float(cache['latitude'])


def update_snow17_boundary_values(
    entries: Dict[str, str],
    snow_params: Dict[str, float],
    cache_path: Path,
) -> None:
    """Recompute the daily ``z_upper`` flux values of a ``.pfidb`` for new Snow-17 parameters.

    ``snow_params`` are the calibration parameters (``SNOW17_*`` and optionally
    ``SNOW_LAPSE_RATE``); the forcing, bands and latitude come from the cache
    the preprocessor wrote.
    """
    cache = np.load(cache_path)
    ppt_mm_hr = cache['ppt_mm_hr']
    pet_mm_hr = cache['pet_mm_hr']
    temp_c = cache['temp_c']
    times = pd.DatetimeIndex(cache['times'].astype('datetime64[ns]'))
    bands, latitude = load_cached_snow_setup(cache, cache_path)

    s17_params: Dict[str, float] = {}
    lapse_rate = DEFAULT_LAPSE_RATE
    for name, value in snow_params.items():
        if name == 'SNOW_LAPSE_RATE':
            lapse_rate = value
        else:
            s17_params[name.replace('SNOW17_', '')] = value

    df_hourly = pd.DataFrame({'ppt': ppt_mm_hr, 'temp': temp_c}, index=times)
    daily = df_hourly.resample('D').agg({'ppt': 'sum', 'temp': 'mean'})
    daily['doy'] = daily.index.dayofyear
    rpm_daily, _ = band_rain_plus_melt(
        daily['ppt'].values, daily['temp'].values, daily['doy'].values,
        bands, latitude, lapse_rate, s17_params,
    )

    daily_pet = pd.DataFrame({'pet': pet_mm_hr}, index=times).resample('D').mean()
    for i, (dt, pet_row) in enumerate(daily_pet.iterrows()):
        key = f"Patch.z_upper.BCPressure.d{dt.strftime('%Y%m%d')}.Value"
        if key in entries:
            # rpm is mm/day, pet is mm/hr; convert both to m/hr
            eff_mm_hr = rpm_daily[i] / 24
            net_m_hr = (eff_mm_hr - pet_row['pet']) * 0.001
            entries[key] = f'{-net_m_hr:g}'


# ---------------------------------------------------------------------------
# Daily net rainfall for the .pfidb
# ---------------------------------------------------------------------------

def prepare_daily_rainfall(
    *,
    forcing_dir: Optional[Path],
    output_dir: Path,
    start: Any,
    end: Any,
    latitude: Callable[[], float],
    snow_bands: Callable[[], SnowBands],
    logger: logging.Logger,
    model_label: str,
    on_hourly_forcing: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], None]] = None,
    forcing_selector: Optional[ForcingSelector] = None,
) -> Optional[Sequence[Tuple[str, int, float]]]:
    """Daily effective rainfall (rain + Snow-17 melt - Oudin PET) from basin-mean forcing.

    Args:
        forcing_dir: basin-averaged canonical forcing directory (None: no forcing).
        output_dir: where the hourly cache and ``daily_rainfall.npy`` are written.
        start, end: simulation window (end exclusive).
        latitude, snow_bands: resolved lazily, only once forcing has been read.
        on_hourly_forcing: called with (pptrate kg m-2 s-1, airtemp K, times)
            of the clipped hourly forcing (CLM-ParFlow writes its met file).
        forcing_selector: narrows the forcing files to this run's HRU set (the
            owning preprocessor's ``_select_run_forcing``, issue #339).

    Returns:
        List of (day_label, hours_in_day, net_rainfall_m_hr) tuples, or None
        if forcing data is unavailable.
    """
    if forcing_dir is None:
        logger.warning(f"No forcing data found. {model_label} will run without rainfall.")
        return None

    # Only the forcing remapped onto this run's HRUs (issue #339).
    run_files = sorted(Path(forcing_dir).glob('*.nc'))
    if forcing_selector is not None:
        run_files = forcing_selector(run_files)

    try:
        from symfluence.core.modeling.model_ready.forcing_reader import (
            open_canonical_forcing,
            resample_canonical_forcing,
        )
        nc_files = run_files
        era5 = [f for f in nc_files if 'ERA5' in f.name or 'remapped' in f.name]
        if era5:
            nc_files = era5
        if not nc_files:
            logger.warning("No forcing files found")
            return None

        logger.info(f"Reading forcing from {len(nc_files)} files")

        ds = open_canonical_forcing(nc_files)
        if 'precipitation_flux' not in ds:
            logger.warning("No precipitation_flux in forcing")
            return None

        # The pipeline treats one forcing step as one hour; resample any
        # non-hourly source (e.g. CARRA 3-hourly) to hourly so the cadence
        # assumption holds. Hourly source (ERA5) passes through unchanged.
        ds = resample_canonical_forcing(ds, 3600.0).sortby('time')

        def _series(name):
            arr = ds[name].values
            return arr[:, 0] if arr.ndim > 1 else np.asarray(arr)

        pptrate = _series('precipitation_flux')
        airtemp = _series('air_temperature')
        times = pd.DatetimeIndex(ds['time'].values).values

    except (OSError, ValueError, KeyError, RuntimeError) as e:
        logger.warning(f"Could not read forcing: {e}", exc_info=True)
        return None

    # Clip to simulation window
    start_dt = pd.Timestamp(str(start))
    end_dt = pd.Timestamp(str(end))
    time_idx = pd.DatetimeIndex(times)
    mask = (time_idx >= start_dt) & (time_idx < end_dt)
    pptrate, airtemp, times = pptrate[mask], airtemp[mask], times[mask]

    if len(pptrate) == 0:
        logger.warning("No forcing data in simulation period")
        return None

    # Hourly P and PET
    lat = latitude()
    bands = snow_bands()
    pet_mm_hr = oudin_pet_mm_hr(airtemp, times, lat)
    ppt_mm_hr = np.maximum(pptrate * 3600.0, 0.0)
    temp_c = airtemp - 273.15

    # Snow-17: rain/snow partition + snowmelt
    effective_mm_hr = apply_snow_model(ppt_mm_hr, temp_c, times, bands, lat, logger)

    # Cache hourly forcing for calibration (Snow-17 parameter updates)
    save_forcing_cache(output_dir, ppt_mm_hr, pet_mm_hr, temp_c, times, bands, lat)

    if on_hourly_forcing is not None:
        on_hourly_forcing(pptrate, airtemp, times)

    # Aggregate to daily averages
    df = pd.DataFrame({'effective': effective_mm_hr, 'pet': pet_mm_hr}, index=pd.DatetimeIndex(times))
    daily = df.resample('D').agg(['mean', 'count'])
    days = []
    for dt, row in daily.iterrows():
        n_hours = int(row[('effective', 'count')])
        if n_hours == 0:
            continue
        net_mm_hr = row[('effective', 'mean')] - row[('pet', 'mean')]
        days.append((f"d{dt.strftime('%Y%m%d')}", n_hours, net_mm_hr * 0.001))

    if not days:
        logger.warning("No daily data computed")
        return None

    # Save for reference
    np.save(Path(output_dir) / 'daily_rainfall.npy', np.array([(d[1], d[2]) for d in days]))

    total_eff = df['effective'].mean()
    total_pet = df['pet'].mean()
    logger.info(
        f"Prepared {len(days)} daily rainfall entries (with snow model): "
        f"mean effective={total_eff:.4f} mm/hr, mean PET={total_pet:.4f} mm/hr, "
        f"mean net={(total_eff - total_pet):.4f} mm/hr"
    )
    return days


# ---------------------------------------------------------------------------
# Preprocessor mixin
# ---------------------------------------------------------------------------

class LumpedForcingMixin:
    """Lumped forcing steps shared by the ParFlow and CLM-ParFlow preprocessors.

    The host class provides ``config``, ``logger``, ``project_dir``,
    ``domain_name``, ``_get_pf_cfg``, ``_get_forcing_dir``, ``_get_latitude``
    and the base preprocessor's ``get_catchment_path`` / ``get_catchment_location``.
    """

    LUMPED_MODEL_LABEL = 'ParFlow'
    DEFAULT_LAPSE_RATE = DEFAULT_LAPSE_RATE  # degC/m
    _snow_bands_cache: Optional[SnowBands] = None

    config: Any
    logger: Any
    project_dir: Path
    domain_name: str

    def _time_bound(self, name: str, default: str) -> Any:
        try:
            return getattr(self.config.domain, name) or default
        except (AttributeError, TypeError):
            return default

    def _get_time_info(self) -> Tuple[int, float, int]:
        return simulation_time_info(
            self._time_bound('time_start', '2000-01-01'),
            self._time_bound('time_end', '2001-01-01'),
            float(self._get_pf_cfg('TIMESTEP_HOURS', 1.0)),  # type: ignore[attr-defined]
        )

    def _dem_path(self) -> Optional[Path]:
        """The domain DEM (``attributes/elevation/dem``), if present."""
        dem_dir = resolve_data_subdir(self.project_dir, 'attributes') / 'elevation' / 'dem'
        preferred = dem_dir / f'domain_{self.domain_name}_elv.tif'
        if preferred.exists():
            return preferred
        candidates = sorted(dem_dir.glob('*_elv.tif')) if dem_dir.exists() else []
        return candidates[0] if candidates else None

    def _get_snow_bands(self) -> SnowBands:
        """Snow-17 elevation bands of this domain (DEM hypsometry, else one band)."""
        if self._snow_bands_cache is None:
            catchment_path: Optional[Path]
            mean_elev: Optional[float] = None
            try:
                catchment_path = Path(self.get_catchment_path())  # type: ignore[attr-defined]
            except (AttributeError, TypeError, ValueError, OSError):
                catchment_path = None
            if catchment_path is not None and catchment_path.exists():
                try:
                    mean_elev = self.get_catchment_location()['elev']  # type: ignore[attr-defined]
                except (FileNotFoundError, ValueError):
                    mean_elev = None
            self._snow_bands_cache = resolve_snow_bands(
                self._dem_path(), catchment_path, mean_elev, self.logger)
        return self._snow_bands_cache

    def _apply_snow_model(self, ppt_mm_hr, temp_c_hourly, times, lapse_rate=None):
        """Hourly rain + snowmelt (mm/hr) from the elevation-band Snow-17."""
        return apply_snow_model(
            ppt_mm_hr, temp_c_hourly, times, self._get_snow_bands(),
            self._get_latitude(), self.logger, lapse_rate=lapse_rate,  # type: ignore[attr-defined]
        )

    def _on_hourly_forcing(self, output_dir: Path, pptrate: np.ndarray, airtemp: np.ndarray,
                           times: np.ndarray) -> None:
        """Hook: called with the clipped hourly forcing (kg m-2 s-1, K)."""

    def _prepare_daily_rainfall(self, settings_dir):
        """Daily effective rainfall (rain + Snow-17 melt - PET) from basin-mean forcing.

        Returns:
            List of (day_label, hours_in_day, net_rainfall_m_hr) tuples,
            or None if forcing data is unavailable.
        """
        return prepare_daily_rainfall(
            forcing_dir=self._get_forcing_dir(),  # type: ignore[attr-defined]
            output_dir=settings_dir,
            start=self._time_bound('time_start', '2000-01-01'),
            end=self._time_bound('time_end', '2001-01-01'),
            latitude=self._get_latitude,  # type: ignore[attr-defined]
            snow_bands=self._get_snow_bands,
            logger=self.logger,
            model_label=self.LUMPED_MODEL_LABEL,
            on_hourly_forcing=lambda pptrate, airtemp, times: self._on_hourly_forcing(
                settings_dir, pptrate, airtemp, times),
            forcing_selector=self._select_run_forcing,  # type: ignore[attr-defined]
        )
