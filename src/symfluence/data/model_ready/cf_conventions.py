# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CF-1.8 convention helpers for the model-ready data store.

Provides a unified CF standard-name mapping for all variables used across
SYMFLUENCE (forcings, observations, attributes) and a builder for the
global attributes that every model-ready NetCDF file should carry.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

import symfluence

# ---------------------------------------------------------------------------
# CF standard-name mapping
# ---------------------------------------------------------------------------
# Each entry maps an *internal* SYMFLUENCE variable name to a dict with
# ``standard_name``, ``units``, and ``long_name`` following CF-1.8.
# This extends the existing STANDARD_VARIABLE_ATTRIBUTES from
# data.preprocessing.dataset_handlers.base_dataset and
# SUMMA_VARIABLE_ATTRS from data.acquisition.handlers.era5_processing.

CF_STANDARD_NAMES: Dict[str, Dict[str, str]] = {
    # --- Forcing variables ---
    'surface_air_pressure':    {'standard_name': 'air_pressure',
                   'units': 'Pa',
                   'long_name': 'air pressure at measurement height'},
    'air_temperature':    {'standard_name': 'air_temperature',
                   'units': 'K',
                   'long_name': 'air temperature at measurement height'},
    'precipitation_flux':    {'standard_name': 'precipitation_flux',
                   'units': 'kg m-2 s-1',
                   'long_name': 'precipitation rate'},
    'wind_speed':    {'standard_name': 'wind_speed',
                   'units': 'm s-1',
                   'long_name': 'wind speed at measurement height'},
    'specific_humidity':    {'standard_name': 'specific_humidity',
                   'units': 'kg kg-1',
                   'long_name': 'specific humidity'},
    'relative_humidity':     {'standard_name': 'relative_humidity',
                   'units': '%',
                   'long_name': 'relative humidity'},
    'surface_downwelling_shortwave_flux':   {'standard_name': 'surface_downwelling_shortwave_flux_in_air',
                   'units': 'W m-2',
                   'long_name': 'downward shortwave radiation at the surface'},
    'surface_downwelling_longwave_flux':   {'standard_name': 'surface_downwelling_longwave_flux_in_air',
                   'units': 'W m-2',
                   'long_name': 'downward longwave radiation at the surface'},

    # --- Extended CF aliases ---
    'surface_downwelling_shortwave_flux_in_air':     {'standard_name': 'surface_downwelling_shortwave_flux_in_air',
                                                      'units': 'W m-2',
                                                      'long_name': 'downward shortwave radiation'},
    'surface_downwelling_longwave_flux_in_air':      {'standard_name': 'surface_downwelling_longwave_flux_in_air',
                                                      'units': 'W m-2',
                                                      'long_name': 'downward longwave radiation'},

    # --- Observation variables ---
    'discharge_cms':   {'standard_name': 'water_volume_transport_in_river_channel',
                        'units': 'm3 s-1',
                        'long_name': 'river discharge'},
    'swe':             {'standard_name': 'lwe_thickness_of_surface_snow_amount',
                        'units': 'kg m-2',
                        'long_name': 'snow water equivalent'},
    'sca':             {'standard_name': 'surface_snow_area_fraction',
                        'units': '1',
                        'long_name': 'snow covered area fraction'},
    'et':              {'standard_name': 'water_evapotranspiration_flux',
                        'units': 'kg m-2 s-1',
                        'long_name': 'evapotranspiration'},
    'tws_anomaly':     {'standard_name': 'liquid_water_content_of_surface_snow',
                        'units': 'mm',
                        'long_name': 'terrestrial water storage anomaly'},
    'soil_moisture':   {'standard_name': 'volume_fraction_of_condensed_water_in_soil',
                        'units': 'm3 m-3',
                        'long_name': 'volumetric soil moisture'},

    # --- Attribute variables ---
    'elev_mean':       {'standard_name': 'surface_altitude',
                        'units': 'm',
                        'long_name': 'mean elevation of hydrological response unit'},
    'hru_area':        {'standard_name': 'area',
                        'units': 'm2',
                        'long_name': 'hydrological response unit area'},
    'latitude':        {'standard_name': 'latitude',
                        'units': 'degrees_north',
                        'long_name': 'centroid latitude'},
    'longitude':       {'standard_name': 'longitude',
                        'units': 'degrees_east',
                        'long_name': 'centroid longitude'},
}


# ---------------------------------------------------------------------------
# Global attribute builder
# ---------------------------------------------------------------------------

def build_global_attrs(
    domain_name: str,
    title: str,
    history: Optional[str] = None,
) -> Dict[str, str]:
    """Build CF-1.8 global attributes for a model-ready NetCDF file.

    Args:
        domain_name: Name of the hydrological domain.
        title: Human-readable title for the dataset.
        history: Optional processing history string.

    Returns:
        Dict of global attributes ready to write with netCDF4.
    """
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    version = getattr(symfluence, '__version__', 'dev')

    attrs: Dict[str, str] = {
        'Conventions': 'CF-1.8',
        'title': title,
        'institution': 'SYMFLUENCE',
        'source_software': f'SYMFLUENCE v{version}',
        'creation_date': now,
        'domain_name': domain_name,
    }
    if history:
        attrs['history'] = history
    return attrs
