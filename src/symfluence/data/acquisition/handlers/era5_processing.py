# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5 Processing Utilities for SYMFLUENCE.

This module provides shared functions for processing ERA5 data from both
ARCO (Google Cloud Zarr) and CDS (Copernicus Climate Data Store) pathways.
"""

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


def _is_cumulative_accumulation(data: xr.DataArray) -> bool:
    """
    Detect whether data is cumulatively accumulated (values increase
    monotonically within forecast periods with periodic resets) or
    per-step accumulated (each timestep is independent).

    ERA5 data from different sources may use different conventions:
    - CDS: Cumulatively accumulated from forecast init (values increase within periods)
    - ARCO: May provide per-step values (each hour's total independently)

    Returns:
        True if data appears cumulatively accumulated, False if per-step.
    """
    diff = data.diff('time')
    n_negative = int((diff < 0).sum())
    n_total = int(diff.size)
    if n_total == 0:
        return True
    negative_fraction = n_negative / n_total
    # Cumulative data has negative diffs only at forecast resets (~8% for 12h cycles).
    # Per-step data has negative diffs whenever the flux decreases (~30-50% typically).
    return negative_fraction < 0.15


def deaccumulate_to_rate(
    accumulated: xr.DataArray,
    time_seconds: xr.DataArray,
    scale_factor: float = 1.0,
    negate_if_negative: bool = False,
    var_name: str = ''
) -> xr.DataArray:
    """
    Convert accumulated ERA5 variable to instantaneous rate.

    ERA5 accumulated variables (precipitation, radiation) need to be
    de-accumulated by taking the time difference and dividing by the
    time step duration. This function auto-detects whether the data is
    cumulatively accumulated (values increase within forecast periods)
    or per-step accumulated (each timestep contains its own total).

    Args:
        accumulated: Accumulated variable values
        time_seconds: Time step durations in seconds
        scale_factor: Scaling factor to apply (e.g., 1000 for m to mm conversion)
        negate_if_negative: If True, negate values if minimum is negative
                           (handles ERA5's negative downward flux convention)
        var_name: Variable name for range clamping (e.g., 'precipitation_flux', 'surface_downwelling_longwave_flux')

    Returns:
        Instantaneous rate values
    """
    logger = logging.getLogger(__name__)
    val = accumulated

    # Handle ERA5's negative downward flux convention if needed
    if negate_if_negative:
        if float(val.min()) < 0.0:
            val = -val

    is_cumulative = _is_cumulative_accumulation(val)

    if is_cumulative:
        # Standard de-accumulation: take time difference
        diff = val.diff('time')

        # Handle accumulation resets (when diff is negative, indicating new accumulation period)
        # For resets, use the current value as the increment
        diff = xr.where(diff >= 0, diff, val.isel(time=slice(1, None)))

        # Handle NaN/inf
        diff = xr.where(np.isfinite(diff), diff, 0.0)

        # Convert to rate and scale
        rate = (diff / time_seconds) * scale_factor
    else:
        # Per-step data: each timestep is an independent accumulated total.
        # Just divide by the time step duration to get the rate.
        # Slice off the first timestep to match the time dimension of diff-based output.
        per_step = val.isel(time=slice(1, None))

        # Handle NaN/inf
        per_step = xr.where(np.isfinite(per_step), per_step, 0.0)

        # Convert to rate and scale
        rate = (per_step / time_seconds) * scale_factor
        logger.debug(
            f"Detected per-step accumulation for {var_name or 'variable'} "
            f"(using direct division instead of differencing)"
        )

    # Apply valid range if specified
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

    # Calculate time step duration in seconds
    dt = (ds['time'].diff('time') / np.timedelta64(1, 's')).astype('float32')

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

    # === Accumulated variables (need de-accumulation) ===

    # Precipitation
    precip_var = find_variable(ds, 'precipitation', source)
    if precip_var:
        pptrate = deaccumulate_to_rate(
            ds[precip_var], dt,
            scale_factor=1000.0,  # m to mm
            negate_if_negative=False,
            var_name='precipitation_flux'
        )
        if source == 'arco':
            processed_vars['precipitation_flux'] = pptrate
        else:
            # For CDS, prepend first value to maintain time dimension
            original_times = ds['time'].values
            first_val = pptrate.isel(time=0).drop_vars('time')
            pptrate_full = xr.concat([first_val.expand_dims('time'), pptrate.drop_vars('time')], dim='time')
            processed_vars['precipitation_flux'] = pptrate_full.assign_coords(time=original_times)
        processed_vars['precipitation_flux'].attrs = SUMMA_VARIABLE_ATTRS['precipitation_flux']
        logger.debug(f"Processed precipitation from {precip_var}")

    # Shortwave radiation
    sw_var = find_variable(ds, 'sw_radiation', source)
    if sw_var:
        sw_rad = deaccumulate_to_rate(
            ds[sw_var], dt,
            scale_factor=1.0,
            negate_if_negative=False,
            var_name='surface_downwelling_shortwave_flux'
        )
        if source == 'arco':
            processed_vars['surface_downwelling_shortwave_flux'] = sw_rad
        else:
            original_times = ds['time'].values
            first_val = sw_rad.isel(time=0).drop_vars('time')
            sw_rad_full = xr.concat([first_val.expand_dims('time'), sw_rad.drop_vars('time')], dim='time')
            processed_vars['surface_downwelling_shortwave_flux'] = sw_rad_full.assign_coords(time=original_times)
        processed_vars['surface_downwelling_shortwave_flux'].attrs = SUMMA_VARIABLE_ATTRS['surface_downwelling_shortwave_flux']
        logger.debug(f"Processed shortwave radiation from {sw_var}")

    # Longwave radiation - CRITICAL: ERA5 may encode downward flux as negative
    lw_var = find_variable(ds, 'lw_radiation', source)
    if lw_var:
        lw_rad = deaccumulate_to_rate(
            ds[lw_var], dt,
            scale_factor=1.0,
            negate_if_negative=True,  # Handle negative downward flux convention
            var_name='surface_downwelling_longwave_flux'
        )
        if source == 'arco':
            processed_vars['surface_downwelling_longwave_flux'] = lw_rad
        else:
            original_times = ds['time'].values
            first_val = lw_rad.isel(time=0).drop_vars('time')
            lw_rad_full = xr.concat([first_val.expand_dims('time'), lw_rad.drop_vars('time')], dim='time')
            processed_vars['surface_downwelling_longwave_flux'] = lw_rad_full.assign_coords(time=original_times)
        processed_vars['surface_downwelling_longwave_flux'].attrs = SUMMA_VARIABLE_ATTRS['surface_downwelling_longwave_flux']

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
