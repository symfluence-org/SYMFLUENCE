# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to HYPE format.
HYPE uses different variable names and may require temporal aggregation
from hourly to daily data.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import xarray as xr

from symfluence.core.modeling.adapters import ForcingAdapter
from symfluence.core.registries import R


@R.forcing_adapters.add('HYPE')
class HYPEForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for HYPE model.

    HYPE variable naming conventions:
        - Tair: Air temperature (typically °C for HYPE)
        - Prec: Precipitation (mm/timestep)
        - Tmax: Maximum temperature (°C)
        - Tmin: Minimum temperature (°C)

    Note:
        HYPE typically uses daily data with temperatures in Celsius
        and precipitation in mm/day. This adapter can handle both
        hourly and daily input data.
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to HYPE names.

        Returns:
            Dict mapping CFIF names to HYPE variable names
        """
        return {
            'air_temperature': 'Tair',
            'precipitation_flux': 'Prec',
            'air_temperature_max': 'Tmax',
            'air_temperature_min': 'Tmin',
            'potential_evapotranspiration': 'PET',
            'relative_humidity': 'RHum',
            'wind_speed': 'Wind',
            'surface_downwelling_shortwave_flux': 'SWRad',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by HYPE.

        Returns:
            List of required CFIF variable names
        """
        return [
            'air_temperature',
            'precipitation_flux',
        ]

    def get_optional_variables(self) -> List[str]:
        """
        Get optional variables for HYPE.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'air_temperature_max',
            'air_temperature_min',
            'potential_evapotranspiration',
            'relative_humidity',
            'wind_speed',
            'surface_downwelling_shortwave_flux',
        ]

    def get_unit_conversions(self) -> Dict[str, Callable]:
        """
        Get unit conversions for HYPE.

        HYPE typically uses:
            - Temperature in °C (CFIF uses K)
            - Precipitation in mm/day (CFIF uses kg m-2 s-1)

        Returns:
            Dict of conversion functions
        """
        # Determine timestep from config
        timestep_hours = self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            dict_key='FORCING_TIME_STEP_SIZE',
        ) / 3600

        # Precipitation: kg m-2 s-1 to mm/timestep
        # 1 kg m-2 s-1 = 1 mm/s = 3600 mm/hr
        precip_factor = 3600 * timestep_hours

        return {
            'air_temperature': lambda x: x - 273.15,  # K to °C
            'air_temperature_max': lambda x: x - 273.15,
            'air_temperature_min': lambda x: x - 273.15,
            'precipitation_flux': lambda x: x * precip_factor,  # to mm/timestep
        }

    def transform(self, cfif_data: xr.Dataset) -> xr.Dataset:
        """
        Transform CFIF data to HYPE format.

        This override handles the HYPE-specific requirement for
        daily data when hourly is provided.

        Args:
            cfif_data: xarray Dataset in CFIF format

        Returns:
            xarray Dataset in HYPE format
        """
        ds = super().transform(cfif_data)

        # HYPE typically expects daily data
        # If configured to aggregate to daily, do so here
        if self._get_config_value(lambda: None, default=False, dict_key='HYPE_AGGREGATE_TO_DAILY'):
            ds = self._aggregate_to_daily(ds)

        return ds

    def _aggregate_to_daily(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Aggregate hourly data to daily.

        Args:
            ds: Hourly dataset

        Returns:
            Daily aggregated dataset
        """
        daily_ds = xr.Dataset()

        for var in ds.data_vars:
            if var in ['Tair']:
                # Mean temperature
                daily_ds[var] = ds[var].resample(time='1D').mean()
            elif var in ['Tmax']:
                # Maximum temperature
                daily_ds[var] = ds[var].resample(time='1D').max()
            elif var in ['Tmin']:
                # Minimum temperature
                daily_ds[var] = ds[var].resample(time='1D').min()
            elif var in ['Prec']:
                # Total precipitation
                daily_ds[var] = ds[var].resample(time='1D').sum()
            elif var in ['SWRad', 'PET']:
                # Mean radiation/PET
                daily_ds[var] = ds[var].resample(time='1D').mean()
            else:
                # Default: mean
                daily_ds[var] = ds[var].resample(time='1D').mean()

        # Preserve coordinates
        daily_ds = daily_ds.assign_coords(ds.coords)
        daily_ds.attrs = ds.attrs.copy()
        daily_ds.attrs['temporal_resolution'] = 'daily'

        return daily_ds

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """Add HYPE-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['model_format'] = 'HYPE'
        return ds
