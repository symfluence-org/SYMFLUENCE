# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Acquisition handlers for various datasets.
"""

import importlib as _importlib
import logging as _logging

_logger = _logging.getLogger(__name__)

# Import all handlers to trigger registration
# Use try/except for each to handle optional dependencies

_imported = []
_failed = []

_handler_modules = [
    'era5',
    'era5_cds',
    'era5_land',
    'aorc',
    'nex_gddp',
    'em_earth',
    'hrrr',
    'conus404',
    'cds_datasets',
    'daymet',
    'dem',
    'soilgrids',
    'landcover',
    'rdrs',
    'smap',
    'ismn',
    'esa_cci_sm',
    'fluxcom_et',
    'grace',
    'grdc',
    'glacier',
    'modis_sca',
    'modis_et',
    'modis_lai',
    'modis_lst',
    'mswep',
    'openet',
    'fluxnet',
    'gpm',
    'chirps',
    'sentinel1_sm',
    'smos_sm',
    'ascat_sm',
    'snodas',
    'jrc_water',
    'ssebop',
    'viirs_snow',
    'ims_snow',
    'sentinel2_snow',
    'cmc_snow',
    'gleam_et',
    'canopy_height',
    'hydrolakes',
    'gldas_tws',
    'cnes_grgs_tws',
    'merra2',
    'merit_hydro',
    'polaris',
    'soilgrids_properties',
    'gssurgo',
    'globsnow',
    'tdx_hydro',
    'merit_basins',
    'nws_hydrofabric',
    'hydrosheds',
    # New hydrological attribute handlers
    'modis_ndvi',
    'pelletier',
    'bedrock_depth',
    'glhymps',
    'aridity_index',
    'glwd',
    'root_zone_storage',
    'wokam',
    # NWM retrospective
    'nwm3_retrospective',
]

for _module_name in _handler_modules:
    try:
        _module = _importlib.import_module(f'.{_module_name}', __name__)
        globals()[_module_name] = _module
        _imported.append(_module_name)
    except Exception as _e:  # noqa: BLE001 — optional handler import resilience
        _failed.append((_module_name, str(_e)))
        _logger.debug("Failed to import acquisition handler '%s': %s", _module_name, _e)

# Clean up
del _handler_modules, _module_name
try:
    del _module, _e
except NameError:
    pass

__all__ = _imported
