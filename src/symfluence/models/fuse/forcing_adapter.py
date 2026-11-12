# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to FUSE format.
FUSE uses a simplified set of forcing variables.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import xarray as xr

from symfluence.core.modeling.adapters import ForcingAdapter
from symfluence.core.registries import R


@R.forcing_adapters.add('FUSE')
class FUSEForcingAdapter(ForcingAdapter):
    """
    Forcing adapter for FUSE model.

    FUSE variable naming conventions:
        - temp: Air temperature (typically °C)
        - precip: Precipitation (mm/day)
        - pet: Potential evapotranspiration (mm/day)

    Note:
        FUSE is a lumped conceptual model that typically uses
        daily timesteps with simple forcing variables.
    """

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map CFIF names to FUSE names.

        Returns:
            Dict mapping CFIF names to FUSE variable names
        """
        return {
            'air_temperature': 'temp',
            'precipitation_flux': 'precip',
            'potential_evapotranspiration': 'pet',
        }

    def get_required_variables(self) -> List[str]:
        """
        Get variables required by FUSE.

        Returns:
            List of required CFIF variable names
        """
        return [
            'air_temperature',
            'precipitation_flux',
        ]

    def get_optional_variables(self) -> List[str]:
        """
        Get optional variables for FUSE.

        Returns:
            List of optional CFIF variable names
        """
        return [
            'potential_evapotranspiration',
        ]

    def get_unit_conversions(self) -> Dict[str, Callable]:
        """
        Get unit conversions for FUSE.

        FUSE typically uses:
            - Temperature in °C
            - Precipitation in mm/day

        Returns:
            Dict of conversion functions
        """
        # Determine timestep for precipitation conversion
        timestep_seconds = self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            dict_key='FORCING_TIME_STEP_SIZE',
        )

        return {
            'air_temperature': lambda x: x - 273.15,  # K to °C
            # kg m-2 s-1 to mm/timestep: multiply by timestep_seconds
            'precipitation_flux': lambda x: x * timestep_seconds,
            'potential_evapotranspiration': lambda x: x * timestep_seconds,
        }

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """Add FUSE-specific metadata."""
        ds = super().add_metadata(ds)
        ds.attrs['model_format'] = 'FUSE'
        return ds
