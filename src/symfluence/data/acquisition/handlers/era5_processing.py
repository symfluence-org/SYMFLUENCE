# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5 Processing Utilities for SYMFLUENCE.

This module provides shared functions for processing ERA5 data from both
ARCO (Google Cloud Zarr) and CDS (Copernicus Climate Data Store) pathways.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr

# Valid ranges for ERA5 variables after processing
ERA5_VARIABLE_RANGES = {
    'surface_downwelling_longwave_flux': {'min': 50.0, 'max': 600.0},
    'surface_downwelling_shortwave_flux': {'min': 0.0, 'max': 1500.0},
    'precipitation_flux': {'min': 0.0, 'max': 1.0},  # mm/s
}

# SUMMA variable attributes
SUMMA_VARIABLE_ATTRS = {
    'surface_air_pressure': {'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'},
    'air_temperature': {'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'},
    'wind_speed': {'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'},
    'specific_humidity': {'units': 'kg kg-1', 'long_name': 'specific humidity', 'standard_name': 'specific_humidity'},
    'precipitation_flux': {'units': 'mm/s', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'},
    'surface_downwelling_shortwave_flux': {'units': 'W m-2', 'long_name': 'shortwave radiation', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'},
    'surface_downwelling_longwave_flux': {'units': 'W m-2', 'long_name': 'longwave radiation', 'standard_name': 'surface_downwelling_longwave_flux_in_air'},
}

# Variable name mappings for different ERA5 sources
ARCO_VARIABLE_NAMES = {
    'temperature': '2m_temperature',
    'dewpoint': '2m_dewpoint_temperature',
    'pressure': 'surface_pressure',
    'wind_u': '10m_u_component_of_wind',
    'wind_v': '10m_v_component_of_wind',
    'precipitation': 'total_precipitation',
    'sw_radiation': 'surface_solar_radiation_downwards',
    'lw_radiation': 'surface_thermal_radiation_downwards',
}

CDS_VARIABLE_NAMES = {
    'temperature': ['t2m', '2m_temperature'],
    'dewpoint': ['d2m', '2m_dewpoint_temperature'],
    'pressure': ['sp', 'surface_pressure'],
    'wind_u': ['u10', '10m_u_component_of_wind'],
    'wind_v': ['v10', '10m_v_component_of_wind'],
    'precipitation': ['tp', 'total_precipitation'],
    'sw_radiation': ['ssrd', 'surface_solar_radiation_downwards'],
    'lw_radiation': ['strd', 'surface_thermal_radiation_downwards'],
}


def calculate_wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Calculate wind speed from u and v components.

    Args:
        u: Eastward wind component (m/s)
        v: Northward wind component (m/s)

    Returns:
        Wind speed magnitude (m/s)
    """
    windspd = ((u**2 + v**2)**0.5).astype('float32')
    windspd.attrs = SUMMA_VARIABLE_ATTRS['wind_speed']
    return windspd


def calculate_specific_humidity(dewpoint_K: xr.DataArray, pressure_Pa: xr.DataArray) -> xr.DataArray:
    """
    Calculate specific humidity from dewpoint temperature and pressure.

    Uses the Magnus formula for saturation vapor pressure.

    Args:
        dewpoint_K: Dewpoint temperature in Kelvin
        pressure_Pa: Surface pressure in Pascals

    Returns:
        Specific humidity (kg/kg)
    """
    Td_C = dewpoint_K - 273.15

    # Saturation vapor pressure (Pa) using Magnus formula
    es = 611.2 * np.exp((17.67 * Td_C) / (Td_C + 243.5))

    # Mixing ratio
    # Guard against division by zero or negative values
    denom = xr.where((pressure_Pa - es) <= 1.0, 1.0, pressure_Pa - es)
    r = 0.622 * es / denom

    # Specific humidity from mixing ratio
    spechum = (r / (1.0 + r)).astype('float32')
    spechum.attrs = SUMMA_VARIABLE_ATTRS['specific_humidity']
    return spechum


# ERA5 accumulated fields on both ARCO (``ar/full_37-1h-0p25deg``) and CDS
# single-levels are totals over the hour ending at the timestamp, not
# cumulative forecast accumulations (ERA5-Land is different). When the ARCO
# loader keeps every N-th hour it first sums the N hourly totals and records N
# in this dataset attribute, so the total always covers the whole interval.
ACCUMULATION_HOURS_ATTR = 'symfluence_accumulation_hours'

# ARCO accumulated variables (per-hour totals) that must be summed, not
# subsampled, when the time step is coarser than one hour.
ARCO_ACCUMULATED_VARIABLES = (
    'total_precipitation',
    'surface_solar_radiation_downwards',
    'surface_thermal_radiation_downwards',
)


def per_step_total_to_rate(
    total: xr.DataArray,
    step_seconds: float,
    scale_factor: float = 1.0,
    negate_if_negative: bool = False,
    var_name: str = '',
) -> xr.DataArray:
    """Convert an ERA5 per-step accumulated total to a mean rate over the step.

    Args:
        total: Accumulated total over each step ending at the timestamp
            (m for precipitation, J m-2 for radiation).
        step_seconds: Length of the accumulation interval in seconds.
        scale_factor: Scaling factor (e.g. 1000 for m -> mm).
        negate_if_negative: Flip sign if the field uses a negative-downward
            convention.
        var_name: Output variable name, for valid-range clamping.

    Returns:
        Mean rate over each step (float32).
    """
    val = total
    if negate_if_negative and float(val.min()) < 0.0:
        val = -val
    val = xr.where(np.isfinite(val), val, 0.0)
    rate = (val / float(step_seconds)) * scale_factor

    if var_name in ERA5_VARIABLE_RANGES:
        min_val = ERA5_VARIABLE_RANGES[var_name]['min']
        max_val = ERA5_VARIABLE_RANGES[var_name]['max']
        rate = xr.where(np.isfinite(rate), rate, min_val).clip(min=min_val, max=max_val)
    else:
        rate = xr.where(np.isfinite(rate), rate, 0.0).clip(min=0.0)
    return rate.astype('float32')


def apply_valid_range(data: xr.DataArray, var_name: str) -> xr.DataArray:
    """
    Clip data to valid physical range for the given variable.

    Args:
        data: Input data array
        var_name: Variable name (e.g., 'surface_downwelling_longwave_flux', 'surface_downwelling_shortwave_flux', 'precipitation_flux')

    Returns:
        Data clipped to valid range
    """
    if var_name in ERA5_VARIABLE_RANGES:
        min_val = ERA5_VARIABLE_RANGES[var_name]['min']
        max_val = ERA5_VARIABLE_RANGES[var_name]['max']
        return data.clip(min=min_val, max=max_val)
    return data


def find_variable(ds: xr.Dataset, var_type: str, source: str = 'arco') -> Optional[str]:
    """
    Find variable name in dataset based on variable type and source.

    Args:
        ds: xarray Dataset
        var_type: Variable type ('temperature', 'dewpoint', 'pressure', etc.)
        source: Data source ('arco' or 'cds')

    Returns:
        Variable name if found, None otherwise
    """
    if source == 'arco':
        var_name = ARCO_VARIABLE_NAMES.get(var_type)
        if var_name and var_name in ds.data_vars:
            return var_name
    else:
        # CDS source - try multiple possible names
        candidates = CDS_VARIABLE_NAMES.get(var_type, [])
        for candidate in candidates:
            if candidate in ds.data_vars or candidate in ds.variables:
                return candidate
    return None


def era5_to_summa_schema(
    ds: xr.Dataset,
    source: str = 'arco',
    logger: Optional[logging.Logger] = None
) -> xr.Dataset:
    """
    Convert ERA5 dataset to SUMMA forcing schema.

    This is the main processing function that handles both ARCO and CDS data sources.

    Args:
        ds: Input ERA5 dataset (must have 'time' dimension with at least 2 timesteps)
        source: Data source ('arco' or 'cds')
        logger: Optional logger for diagnostic messages

    Returns:
        Dataset with SUMMA-schema variables:
        - airpres: Air pressure (Pa)
        - airtemp: Air temperature (K)
        - windspd: Wind speed (m/s)
        - spechum: Specific humidity (kg/kg)
        - pptrate: Precipitation rate (mm/s)
        - SWRadAtm: Shortwave radiation (W/m2)
        - LWRadAtm: Longwave radiation (W/m2)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if 'time' not in ds.dims or ds.sizes['time'] < 2:
        logger.warning("Dataset has insufficient time steps for de-accumulation")
        return ds

    # Sort by time
    ds = ds.sortby('time')

    # For ARCO source, slice off first timestep after de-accumulation
    # For CDS source, we prepend the first value to maintain time dimension

    processed_vars = {}

    # === Instantaneous variables ===

    # Temperature
    temp_var = find_variable(ds, 'temperature', source)
    if temp_var:
        if source == 'arco':
            processed_vars['air_temperature'] = ds[temp_var].isel(time=slice(1, None)).astype('float32')
        else:
            processed_vars['air_temperature'] = ds[temp_var].astype('float32')
        processed_vars['air_temperature'].attrs = SUMMA_VARIABLE_ATTRS['air_temperature']
        logger.debug(f"Processed temperature from {temp_var}")

    # Pressure
    pres_var = find_variable(ds, 'pressure', source)
    if pres_var:
        if source == 'arco':
            processed_vars['surface_air_pressure'] = ds[pres_var].isel(time=slice(1, None)).astype('float32')
        else:
            processed_vars['surface_air_pressure'] = ds[pres_var].astype('float32')
        processed_vars['surface_air_pressure'].attrs = SUMMA_VARIABLE_ATTRS['surface_air_pressure']
        logger.debug(f"Processed pressure from {pres_var}")

    # Wind components -> wind speed
    u_var = find_variable(ds, 'wind_u', source)
    v_var = find_variable(ds, 'wind_v', source)
    if u_var and v_var:
        if source == 'arco':
            u = ds[u_var].isel(time=slice(1, None))
            v = ds[v_var].isel(time=slice(1, None))
        else:
            u = ds[u_var]
            v = ds[v_var]
        processed_vars['wind_speed'] = calculate_wind_speed(u, v)
        logger.debug(f"Calculated wind speed from {u_var}, {v_var}")

    # Specific humidity (from dewpoint and pressure)
    dew_var = find_variable(ds, 'dewpoint', source)
    if dew_var and pres_var:
        if source == 'arco':
            dewpoint = ds[dew_var].isel(time=slice(1, None))
            pressure = ds[pres_var].isel(time=slice(1, None))
        else:
            dewpoint = ds[dew_var]
            pressure = ds[pres_var]
        processed_vars['specific_humidity'] = calculate_specific_humidity(dewpoint, pressure)
        logger.debug(f"Calculated specific humidity from {dew_var}, {pres_var}")

    # ERA5 single-levels (CDS and ARCO) supplies per-hour totals, not
    # cumulative forecasts (ERA5-Land is different). Never infer this from
    # monotonicity.
    for logical, name, scale, negate in [
        ('precipitation', 'precipitation_flux', 1000.0, False),
        ('sw_radiation', 'surface_downwelling_shortwave_flux', 1.0, False),
        ('lw_radiation', 'surface_downwelling_longwave_flux', 1.0, True),
    ]:
        variable = find_variable(ds, logical, source)
        if not variable:
            continue
        if source == 'cds':
            rate = ds[variable] * (scale / 3600.0)
        else:
            # ARCO: per-hour totals, or N-hour sums when the loader kept every
            # N-th hour. The first step is dropped like the instantaneous
            # variables (it is the continuity step from the previous chunk).
            step_hours = float(ds.attrs.get(ACCUMULATION_HOURS_ATTR, 1))
            rate = per_step_total_to_rate(
                ds[variable].isel(time=slice(1, None)), step_hours * 3600.0,
                scale_factor=scale, negate_if_negative=negate, var_name=name,
            )
        processed_vars[name] = rate
        processed_vars[name].attrs = SUMMA_VARIABLE_ATTRS[name]

    if 'surface_downwelling_longwave_flux' in processed_vars:
        # Validate longwave radiation
        lw_mean = float(processed_vars['surface_downwelling_longwave_flux'].mean().values)
        if lw_mean < 50:
            # Only error if extremely low (likely data quality issue)
            raise ValueError(f"LW radiation critically low: {lw_mean:.1f} W/m^2 (min threshold: 50)")
        elif lw_mean < 80:
            logger.warning(f"LW radiation low: {lw_mean:.1f} W/m^2 (expected for winter/high latitudes)")
        elif lw_mean < 150:
            logger.warning(f"LW radiation relatively low: {lw_mean:.1f} W/m^2 (expected for cold climates)")
        else:
            logger.debug(f"LW radiation: mean={lw_mean:.1f} W/m^2")

    # Build output dataset
    if source == 'arco':
        # Use sliced time coordinates (after first timestep)
        ds_base = ds.isel(time=slice(1, None))
        out_coords = {c: ds_base.coords[c] for c in ds_base.coords}
    else:
        out_coords = {c: ds.coords[c] for c in ['time', 'latitude', 'longitude'] if c in ds.coords}

    ds_out = xr.Dataset(data_vars=processed_vars, coords=out_coords)

    # Ensure consistent dimension ordering
    if 'latitude' in ds_out.dims and 'longitude' in ds_out.dims:
        ds_out = ds_out.transpose('time', 'latitude', 'longitude', missing_dims='ignore')

    return ds_out
