# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Observation data handlers for various data sources.

This module provides handlers for acquiring and processing observation data
from multiple sources including satellite products, in-situ networks, and
reanalysis datasets.
"""

from .ana import ANAStreamflowHandler
from .camels import (
    CAMELSAUSStreamflowHandler,
    CAMELSBRStreamflowHandler,
    CAMELSCLStreamflowHandler,
    CAMELSGBStreamflowHandler,
    CAMELSUSStreamflowHandler,
    load_camels_attributes,
)
from .canopy_height import (
    CanopyHeightHandler,
    GEDICanopyHeightHandler,
    GLADTreeHeightHandler,
    MetaCanopyHeightHandler,
)
from .canswe import CanSWEHandler, NorSWEHandler
from .chirps import CHIRPSHandler
from .cmc_snow import CMCSnowHandler
from .cnes_grgs_tws import CNESGRGSHandler
from .daymet import DaymetHandler
from .dga import DGAStreamflowHandler
from .era5_land import ERA5LandHandler
from .fluxcom import FLUXCOMETHandler
from .fluxnet import FLUXNETObservationHandler
from .ggmn import GGMNHandler
from .gldas_tws import GLDASHandler
from .gleam import GLEAMETHandler
from .gpm import GPMIMERGHandler
from .grace import GRACEHandler
from .grdc import GRDCHandler
from .hubeau import (
    HubEauStreamflowHandler,
    HubEauWaterLevelHandler,
    get_station_info,
    search_hubeau_stations,
)
from .ims_snow import IMSSnowHandler
from .jrc_water import JRCWaterHandler
from .lamah_ice import LamahIceStreamflowHandler
from .modis_et import MODISETHandler
from .modis_lai import MODISLAIHandler
from .modis_lst import MODISLSTHandler
from .modis_snow import MODISSCAHandler, MODISSnowHandler
from .modis_utils import (
    CLOUD_VALUE,
    MODIS_ET_COLUMN_MAP,
    MODIS_FILL_VALUES,
    VALID_SNOW_RANGE,
    apply_modis_quality_filter,
    convert_cftime_to_datetime,
    extract_spatial_average,
    find_variable_in_dataset,
    interpolate_8day_to_daily,
    standardize_et_columns,
)
from .mswep import MSWEPHandler
from .openet import OpenETHandler
from .sentinel1_sm import Sentinel1SMHandler
from .smhi import SMHIStreamflowHandler
from .snodas import SNODASHandler
from .snotel import SNOTELHandler
from .soil_moisture import ASCATSMHandler, ESACCISMHandler, ISMNHandler, SMAPHandler, SMOSSMHandler
from .ssebop import SSEBopHandler
from .usgs import USGSGroundwaterHandler, USGSStreamflowHandler
from .viirs_snow import VIIRSSnowHandler
from .wsc import WSCStreamflowHandler

__all__ = [
    # CAMELS
    "CAMELSUSStreamflowHandler",
    "CAMELSBRStreamflowHandler",
    "CAMELSCLStreamflowHandler",
    "CAMELSAUSStreamflowHandler",
    "CAMELSGBStreamflowHandler",
    "load_camels_attributes",
    # CHIRPS
    "CHIRPSHandler",
    # Daymet
    "DaymetHandler",
    # ERA5-Land
    "ERA5LandHandler",
    # FLUXCOM
    "FLUXCOMETHandler",
    # FLUXNET
    "FLUXNETObservationHandler",
    # GGMN
    "GGMNHandler",
    # GLEAM
    "GLEAMETHandler",
    # GPM IMERG
    "GPMIMERGHandler",
    # GRACE
    "GRACEHandler",
    # GRDC
    "GRDCHandler",
    # Hub'Eau (France)
    "HubEauStreamflowHandler",
    "HubEauWaterLevelHandler",
    "search_hubeau_stations",
    "get_station_info",
    # JRC Global Surface Water
    "JRCWaterHandler",
    # LamaH-Ice
    "LamahIceStreamflowHandler",
    # MODIS ET
    "MODISETHandler",
    # MODIS LAI
    "MODISLAIHandler",
    # MODIS LST
    "MODISLSTHandler",
    # MODIS Snow
    "MODISSnowHandler",
    "MODISSCAHandler",
    # MODIS utilities
    "MODIS_FILL_VALUES",
    "CLOUD_VALUE",
    "VALID_SNOW_RANGE",
    "MODIS_ET_COLUMN_MAP",
    "convert_cftime_to_datetime",
    "standardize_et_columns",
    "interpolate_8day_to_daily",
    "apply_modis_quality_filter",
    "extract_spatial_average",
    "find_variable_in_dataset",
    # MSWEP
    "MSWEPHandler",
    # OpenET
    "OpenETHandler",
    # Sentinel-1 SM
    "Sentinel1SMHandler",
    # SMHI
    "SMHIStreamflowHandler",
    # SNODAS
    "SNODASHandler",
    # SNOTEL
    "SNOTELHandler",
    # SSEBop
    "SSEBopHandler",
    # Soil moisture
    "SMAPHandler",
    "ISMNHandler",
    "ESACCISMHandler",
    "SMOSSMHandler",
    "ASCATSMHandler",
    # ANA (Brazil / CAMELS-BR)
    "ANAStreamflowHandler",
    # DGA (Chile)
    "DGAStreamflowHandler",
    # USGS
    "USGSStreamflowHandler",
    "USGSGroundwaterHandler",
    # VIIRS Snow
    "VIIRSSnowHandler",
    # WSC
    "WSCStreamflowHandler",
    # Canopy Height
    "CanopyHeightHandler",
    "GEDICanopyHeightHandler",
    "MetaCanopyHeightHandler",
    "GLADTreeHeightHandler",
    # CanSWE / NorSWE
    "CanSWEHandler",
    "NorSWEHandler",
    # CMC Snow
    "CMCSnowHandler",
    # IMS Snow
    "IMSSnowHandler",
    # GLDAS TWS
    "GLDASHandler",
    # CNES/GRGS TWS
    "CNESGRGSHandler",
]
