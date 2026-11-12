# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGen Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to NOAA NextGen format.
NGen uses specific variable names from the BMI standard.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import xarray as xr

from symfluence.core.modeling.adapters import ForcingAdapter
from symfluence.core.registries import R


@R.forcing_adapters.add('NGEN')
class NGENForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for NOAA NextGen framework.

    NGen variable naming conventions (BMI standard):
        - APCP_surface: Precipitation (kg m-2)
        - TMP_2maboveground: Temperature at 2m (K)
        - SPFH_2maboveground: Specific humidity (kg kg-1)
        - UGRD_10maboveground: U-component wind (m s-1)
        - VGRD_10maboveground: V-component wind (m s-1)
        - PRES_surface: Surface pressure (Pa)
        - DSWRF_surface: Downward shortwave radiation (W m-2)
        - DLWRF_surface: Downward longwave radiation (W m-2)

    Note:
        NGen uses the AORC/CFSv2 naming conventions.
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to NGen/BMI names.

        Returns:
            Dict mapping CFIF names to NGen variable names
        """
        return {
            'air_temperature': 'TMP_2maboveground',
            'precipitation_flux': 'APCP_surface',
            'surface_downwelling_shortwave_flux': 'DSWRF_surface',
            'surface_downwelling_longwave_flux': 'DLWRF_surface',
            'specific_humidity': 'SPFH_2maboveground',
            'eastward_wind': 'UGRD_10maboveground',
            'northward_wind': 'VGRD_10maboveground',
            'wind_speed': 'WIND_10maboveground',
            'surface_air_pressure': 'PRES_surface',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by NGen.

        Returns:
            List of required CFIF variable names
        """
        return [
            'air_temperature',
            'precipitation_flux',
        ]

    def get_optional_variables(self) -> List[str]:
        """
        Get optional variables for NGen.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'surface_downwelling_shortwave_flux',
            'surface_downwelling_longwave_flux',
            'specific_humidity',
            'eastward_wind',
            'northward_wind',
            'wind_speed',
            'surface_air_pressure',
        ]

    def get_unit_conversions(self) -> Dict[str, Callable]:
        """
        Get unit conversions for NGen.

        NGen precipitation is typically in kg m-2 (accumulated)
        rather than kg m-2 s-1 (rate).

        Returns:
            Dict of conversion functions
        """
        # Get timestep in seconds
        timestep_seconds = self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            dict_key='FORCING_TIME_STEP_SIZE',
        )

        return {
            # Convert rate to accumulated: multiply by timestep
            'precipitation_flux': lambda x: x * timestep_seconds,
        }

    def transform(self, cfif_data: xr.Dataset) -> xr.Dataset:
        """
        Transform CFIF data to NGen format.

        Args:
            cfif_data: xarray Dataset in CFIF format

        Returns:
            xarray Dataset in NGen format
        """
        ds = super().transform(cfif_data)

        # NGen expects specific coordinate names
        if 'lat' in ds.coords:
            ds = ds.rename({'lat': 'latitude'})
        if 'lon' in ds.coords:
            ds = ds.rename({'lon': 'longitude'})

        return ds

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """Add NGen-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['Conventions'] = 'CF-1.6'
        ds.attrs['model_format'] = 'NGEN'
        ds.attrs['featureType'] = 'timeSeries'
        return ds
