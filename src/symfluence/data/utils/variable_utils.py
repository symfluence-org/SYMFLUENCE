# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Variable Utilities for SYMFLUENCE

Provides centralized variable name standardization and unit conversion:
- VariableStandardizer: Simple rename maps for dataset → standard names
- VariableHandler: Full unit conversion with pint

Variable Naming Formats:
- CFIF (CF-Intermediate Format): Model-neutral CF-compliant names (e.g., 'air_temperature')
- Legacy (SUMMA-style): Model-specific names used historically (e.g., 'airtemp')

Usage:
    from symfluence.data.utils import VariableStandardizer, VariableHandler

    # Simple renaming to CFIF format (recommended)
    standardizer = VariableStandardizer()
    rename_map = standardizer.get_cfif_rename_map('CONUS404')
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.data_vars})

    # Legacy renaming (backward compatible)
    rename_map = standardizer.get_rename_map('CONUS404')
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.data_vars})

    # Full processing with units
    handler = VariableHandler(config, logger, 'ERA5', 'SUMMA')
    ds = handler.process_forcing_data(ds)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import pint
import pint_xarray
import xarray as xr
import yaml

# Lazy import CFIF mappings to avoid circular imports
# These are imported in the methods that need them
if TYPE_CHECKING:
    pass  # Type hints only


def _get_cfif_mappings():
    """Lazy load CFIF mappings to avoid circular imports."""
    from symfluence.data.preprocessing.cfif.variables import (
        CFIF_TO_SUMMA_MAPPING,
        CFIF_VARIABLES,
        SUMMA_TO_CFIF_MAPPING,
    )
    return SUMMA_TO_CFIF_MAPPING, CFIF_TO_SUMMA_MAPPING, CFIF_VARIABLES


class VariableStandardizer:
    """
    Provides simple variable name standardization across datasets.

    This class consolidates all the rename_map dictionaries that were
    duplicated across acquisition handlers into a single source of truth.

    Standard variable names (target):
        - airtemp: Air temperature
        - airpres: Surface air pressure
        - spechum: Specific humidity
        - relhum: Relative humidity
        - windspd: Wind speed (magnitude)
        - windspd_u: Eastward wind component
        - windspd_v: Northward wind component
        - SWRadAtm: Downwelling shortwave radiation
        - LWRadAtm: Downwelling longwave radiation
        - pptrate: Precipitation rate

    Usage:
        standardizer = VariableStandardizer()

        # Get rename map for a dataset
        rename_map = standardizer.get_rename_map('CONUS404')
        ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.data_vars})

        # Standardize dataset (convenience method)
        ds = standardizer.standardize(ds, 'ERA5')

        # Check if a variable is standard
        if standardizer.is_standard_name('airtemp'):
            ...
    """

    # Mapping from source variable names to standard names
    # Organized by dataset for easy lookup
    RENAME_MAPS: Dict[str, Dict[str, str]] = {
        'ERA5': {
            't2m': 'airtemp',
            'tp': 'pptrate',
            'sp': 'airpres',
            'q': 'spechum',
            'u10': 'windspd_u',
            'v10': 'windspd_v',
            'ws10': 'windspd',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm',
        },
        'CARRA': {
            't2m': 'airtemp',
            'r2': 'relhum',
            'tp': 'pptrate',
            'sp': 'airpres',
            'q': 'spechum',
            'u10': 'windspd_u',
            'v10': 'windspd_v',
            'ws10': 'windspd',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm',
            # Long-form variable names from CDS API
            '2m_temperature': 'airtemp',
            '2m_relative_humidity': 'relhum',
            'total_precipitation': 'pptrate',
            'surface_pressure': 'airpres',
            '10m_u_component_of_wind': 'windspd_u',
            '10m_v_component_of_wind': 'windspd_v',
            'thermal_surface_radiation_downwards': 'LWRadAtm',
            'surface_thermal_radiation_downwards': 'LWRadAtm',  # Backwards compatibility
            'surface_solar_radiation_downwards': 'SWRadAtm',
        },
        'CERRA': {
            't2m': 'airtemp',
            'r2': 'relhum',
            'tp': 'pptrate',
            'sp': 'airpres',
            'q': 'spechum',
            'u10': 'windspd_u',
            'v10': 'windspd_v',
            'ws10': 'windspd',
            'si10': 'windspd',  # CERRA provides 10m wind speed directly
            '10m_wind_speed': 'windspd',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm',
            '2m_temperature': 'airtemp',
            '2m_relative_humidity': 'relhum',
            'total_precipitation': 'pptrate',
            'surface_pressure': 'airpres',
            '10m_u_component_of_wind': 'windspd_u',
            '10m_v_component_of_wind': 'windspd_v',
            'surface_solar_radiation_downwards': 'SWRadAtm',
            'surface_thermal_radiation_downwards': 'LWRadAtm',
            'thermal_surface_radiation_downwards': 'LWRadAtm',
        },
        'CONUS404': {
            'T2': 'airtemp',
            'Q2': 'spechum',
            'PSFC': 'airpres',
            'U10': 'windspd_u',
            'V10': 'windspd_v',
            'GLW': 'LWRadAtm',
            'SWDOWN': 'SWRadAtm',
            'ACSWDNB': 'SWRadAtm',
            'ACLWDNB': 'LWRadAtm',
            'LWDOWN': 'LWRadAtm',
            'RAINRATE': 'pptrate',
            'PREC_ACC_NC': 'pptrate',
            'ACDRIPR': 'pptrate',
            'PRATE': 'pptrate',
        },
        'HRRR': {
            'TMP': 'airtemp',
            'SPFH': 'spechum',
            'PRES': 'airpres',
            'DLWRF': 'LWRadAtm',
            'DSWRF': 'SWRadAtm',
            'UGRD': 'windspd_u',
            'VGRD': 'windspd_v',
            'APCP': 'pptrate',
        },
        'AORC': {
            'APCP_surface': 'pptrate',
            'TMP_2maboveground': 'airtemp',
            'SPFH_2maboveground': 'spechum',
            'PRES_surface': 'airpres',
            'DLWRF_surface': 'LWRadAtm',
            'DSWRF_surface': 'SWRadAtm',
            'UGRD_10maboveground': 'windspd_u',
            'VGRD_10maboveground': 'windspd_v',
        },
        'NEX-GDDP': {
            'pr': 'pptrate',
            'tas': 'airtemp',
            'tasmax': 'airtemp_max',
            'tasmin': 'airtemp_min',
            'hurs': 'relhum',
            'huss': 'spechum',
            'rlds': 'LWRadAtm',
            'rsds': 'SWRadAtm',
            'sfcWind': 'windspd',
            'ps': 'airpres',
        },
        'NEX-GDDP-CMIP6': {
            'pr': 'pptrate',
            'tas': 'airtemp',
            'huss': 'spechum',
            'ps': 'airpres',
            'rlds': 'LWRadAtm',
            'rsds': 'SWRadAtm',
            'sfcWind': 'windspd',
        },
        'GWF': {
            'PSFC': 'airpres',
            'Q2': 'spechum',
            'T2': 'airtemp',
            'U10': 'windspd_u',
            'V10': 'windspd_v',
            'PREC_ACC_NC': 'pptrate',
            'SWDOWN': 'SWRadAtm',
            'GLW': 'LWRadAtm',
        },
        'RDRS': {
            'RDRS_v2.1_P_TT_1.5m': 'airtemp',
            'RDRS_v2.1_P_P0_SFC': 'airpres',
            'RDRS_v2.1_P_HU_1.5m': 'spechum',
            'RDRS_v2.1_P_UVC_10m': 'windspd',
            'RDRS_v2.1_P_UUC_10m': 'windspd_u',
            'RDRS_v2.1_P_VVC_10m': 'windspd_v',
            'RDRS_v2.1_P_FI_SFC': 'LWRadAtm',
            'RDRS_v2.1_P_FB_SFC': 'SWRadAtm',
            'RDRS_v2.1_A_PR0_SFC': 'pptrate',
        },
        'RDRS_v3.1': {
            'TT': 'airtemp',
            'P0': 'airpres',
            'HU': 'spechum',
            'UVC': 'windspd',
            'UUC': 'windspd_u',
            'VVC': 'windspd_v',
            'FI': 'LWRadAtm',
            'FB': 'SWRadAtm',
            'PR0': 'pptrate',
        },
        'CASR_v3.1': {
            'CaSR_v3.1_A_TT_1.5m': 'airtemp',
            'CaSR_v3.1_P_TT_1.5m': 'airtemp',
            'CaSR_v3.1_A_PR0_SFC': 'pptrate',
            'CaSR_v3.1_P_PR0_SFC': 'pptrate',
            'CaSR_v3.1_P_P0_SFC': 'airpres',
            'CaSR_v3.1_P_HU_1.5m': 'spechum',
            'CaSR_v3.1_P_UVC_10m': 'windspd',
            'CaSR_v3.1_P_UUC_10m': 'windspd_u',
            'CaSR_v3.1_P_VVC_10m': 'windspd_v',
            'CaSR_v3.1_P_FB_SFC': 'SWRadAtm',
            'CaSR_v3.1_P_FI_SFC': 'LWRadAtm',
            'TT': 'airtemp',
            'P0': 'airpres',
            'HU': 'spechum',
            'UVC': 'windspd',
            'UUC': 'windspd_u',
            'VVC': 'windspd_v',
            'FI': 'LWRadAtm',
            'FB': 'SWRadAtm',
            'PR0': 'pptrate',
        },
        'CASR_v3.2': {
            'tas': 'airtemp',
            'ta': 'airtemp',
            'ps': 'airpres',
            'huss': 'spechum',
            'hus': 'spechum',
            'sfcWind': 'windspd',
            'uas': 'windspd_u',
            'vas': 'windspd_v',
            'rlds': 'LWRadAtm',
            'rsds': 'SWRadAtm',
            'pr': 'pptrate',
        },
        'DayMet': {
            'prcp': 'pptrate',
            'srad': 'SWRadAtm',
            'tmax': 'airtemp_max',
            'tmin': 'airtemp_min',
            'vp': 'water_vapor_pressure',
            'dayl': 'day_length',
            'swe': 'snow_water_equivalent',
        },
    }

    # Standard variable names recognized by SYMFLUENCE (legacy SUMMA-style)
    STANDARD_NAMES: Set[str] = {
        'airtemp', 'airtemp_max', 'airtemp_min',
        'airpres',
        'spechum', 'relhum',
        'windspd', 'windspd_u', 'windspd_v',
        'SWRadAtm', 'LWRadAtm',
        'pptrate',
        'day_length', 'snow_water_equivalent', 'water_vapor_pressure',
    }

    # CFIF (CF-Intermediate Format) variable names - populated lazily
    _cfif_names_cache: Optional[Set[str]] = None

    @classmethod
    def _get_cfif_names(cls) -> Set[str]:
        """Get CFIF variable names (lazy loaded)."""
        if cls._cfif_names_cache is None:
            _, _, cfif_vars = _get_cfif_mappings()
            cls._cfif_names_cache = set(cfif_vars.keys())
        return cls._cfif_names_cache

    @property
    def CFIF_NAMES(self) -> Set[str]:
        """CFIF variable names (lazy loaded property)."""
        return self._get_cfif_names()

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize VariableStandardizer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def get_rename_map(self, dataset: str) -> Dict[str, str]:
        """
        Get the variable rename map for a dataset.

        Args:
            dataset: Dataset name (e.g., 'ERA5', 'CONUS404', 'CASR')

        Returns:
            Dictionary mapping source variable names to standard names

        Raises:
            ValueError: If dataset is not supported
        """
        # Case-insensitive lookup against RENAME_MAPS keys
        dataset_key = self._find_dataset_key(dataset)

        if dataset_key is None:
            available = ', '.join(sorted(self.RENAME_MAPS.keys()))
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {available}"
            )

        return self.RENAME_MAPS[dataset_key].copy()

    def _find_dataset_key(self, dataset: str) -> Optional[str]:
        """Find the RENAME_MAPS key matching the dataset name (case-insensitive)."""
        # Exact match first
        if dataset in self.RENAME_MAPS:
            return dataset
        # Case-insensitive match
        dataset_lower = dataset.lower()
        for key in self.RENAME_MAPS:
            if key.lower() == dataset_lower:
                return key
        return None

    def standardize(
        self,
        ds: xr.Dataset,
        dataset: str,
        inplace: bool = False
    ) -> xr.Dataset:
        """
        Standardize variable names in a dataset.

        Args:
            ds: xarray Dataset with source variable names
            dataset: Source dataset name (e.g., 'ERA5', 'CONUS404')
            inplace: If True, modify and return the same dataset object

        Returns:
            Dataset with standardized variable names
        """
        rename_map = self.get_rename_map(dataset)

        # Only rename variables that exist in the dataset
        to_rename = {k: v for k, v in rename_map.items() if k in ds.data_vars}

        if not to_rename:
            self.logger.debug(f"No variables to rename for dataset {dataset}")
            return ds

        self.logger.debug(f"Renaming {len(to_rename)} variables: {to_rename}")

        if inplace:
            # xarray doesn't support true inplace, but we minimize copying
            return ds.rename(to_rename)
        else:
            return ds.rename(to_rename)

    def is_standard_name(self, name: str) -> bool:
        """Check if a variable name is a standard SYMFLUENCE name."""
        return name in self.STANDARD_NAMES

    def get_source_names(self, standard_name: str, dataset: str) -> List[str]:
        """
        Get possible source variable names for a standard name.

        Args:
            standard_name: Standard variable name (e.g., 'airtemp')
            dataset: Dataset to search in

        Returns:
            List of source variable names that map to this standard name
        """
        rename_map = self.get_rename_map(dataset)
        return [k for k, v in rename_map.items() if v == standard_name]

    def list_datasets(self) -> List[str]:
        """Return list of supported dataset names."""
        return sorted(self.RENAME_MAPS.keys())

    @classmethod
    def register_dataset(cls, name: str, rename_map: Dict[str, str]) -> None:
        """
        Register a new dataset's variable mapping.

        Args:
            name: Dataset name
            rename_map: Dictionary mapping source names to standard names
        """
        cls.RENAME_MAPS[name.upper()] = rename_map

    def get_cfif_rename_map(self, dataset: str) -> Dict[str, str]:
        """
        Get variable rename map that outputs CFIF names.

        This method returns a mapping from source dataset variable names
        directly to CFIF (CF-Intermediate Format) names, bypassing the
        legacy SUMMA-style intermediate names.

        Args:
            dataset: Dataset name (e.g., 'ERA5', 'CONUS404')

        Returns:
            Dictionary mapping source variable names to CFIF names

        Example:
            >>> standardizer = VariableStandardizer()
            >>> cfif_map = standardizer.get_cfif_rename_map('ERA5')
            >>> # {'t2m': 'air_temperature', 'tp': 'precipitation_flux', ...}
        """
        # First get the standard (SUMMA-style) mapping
        standard_map = self.get_rename_map(dataset)

        # Get CFIF mappings (lazy loaded)
        summa_to_cfif, _, _ = _get_cfif_mappings()

        # Convert SUMMA-style targets to CFIF names
        cfif_map = {}
        for source, summa_name in standard_map.items():
            cfif_name = summa_to_cfif.get(summa_name)
            if cfif_name:
                cfif_map[source] = cfif_name
            else:
                # Keep original if no CFIF mapping exists
                cfif_map[source] = summa_name
                self.logger.debug(
                    f"No CFIF mapping for '{summa_name}', keeping as-is"
                )

        return cfif_map

    def standardize_to_cfif(
        self,
        ds: xr.Dataset,
        dataset: str,
    ) -> xr.Dataset:
        """
        Standardize variable names to CFIF format.

        Args:
            ds: xarray Dataset with source variable names
            dataset: Source dataset name (e.g., 'ERA5', 'CONUS404')

        Returns:
            Dataset with CFIF variable names
        """
        cfif_map = self.get_cfif_rename_map(dataset)

        # Only rename variables that exist in the dataset
        to_rename = {k: v for k, v in cfif_map.items() if k in ds.data_vars}

        if not to_rename:
            self.logger.debug(f"No variables to rename for dataset {dataset}")
            return ds

        self.logger.debug(f"Renaming {len(to_rename)} variables to CFIF: {to_rename}")
        return ds.rename(to_rename)

    def is_cfif_name(self, name: str) -> bool:
        """Check if a variable name is a CFIF standard name."""
        return name in self.CFIF_NAMES

    def convert_summa_to_cfif(self, summa_name: str) -> Optional[str]:
        """Convert a SUMMA-style variable name to CFIF name."""
        summa_to_cfif, _, _ = _get_cfif_mappings()
        return summa_to_cfif.get(summa_name)

    def convert_cfif_to_summa(self, cfif_name: str) -> Optional[str]:
        """Convert a CFIF variable name to SUMMA-style name."""
        _, cfif_to_summa, _ = _get_cfif_mappings()
        return cfif_to_summa.get(cfif_name)


class VariableHandler:
    """
    Handles variable name mapping and unit conversion between different datasets and models.

    Attributes:
        variable_mappings (Dict): Dataset to model variable name mappings
        unit_registry (pint_xarray.UnitRegistry): Unit conversion registry
        logger (logging.Logger): SYMFLUENCE logger instance
    """

    # Dataset variable name mappings
    DATASET_MAPPINGS = {
        'CFIF': {
            # CF-Intermediate Format - used by model-agnostic preprocessing output
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'windspd_u': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'windspd_v': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'ERA5': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'CARRA': {
            '2m_temperature': {'standard_name': 'air_temperature', 'units': 'K'},
            'surface_pressure': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            '2m_specific_humidity': {'standard_name': 'specific_humidity', 'units': '1'},
            '10m_u_component_of_wind': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            '10m_v_component_of_wind': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'thermal_surface_radiation_downwards': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'surface_net_solar_radiation': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'total_precipitation': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            # Standardized variable names (after VariableStandardizer processing)
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'windspd_u': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'windspd_v': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'RDRS': {
            'RDRS_v2.1_P_TT_1.5m': {'standard_name': 'air_temperature', 'units': 'K'},
            'RDRS_v2.1_P_P0_SFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'RDRS_v2.1_P_HU_1.5m': {'standard_name': 'specific_humidity', 'units': '1'},
            'RDRS_v2.1_P_UVC_10m': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'RDRS_v2.1_P_FI_SFC': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'RDRS_v2.1_P_FB_SFC': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'RDRS_v2.1_A_PR0_SFC': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'RDRS_v3.1': {
            'TT': {'standard_name': 'air_temperature', 'units': 'K'},
            'P0': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'HU': {'standard_name': 'specific_humidity', 'units': '1'},
            'UVC': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'FI': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'FB': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'PR0': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'CASR_v3.1': {
            'CaSR_v3.1_A_TT_1.5m': {'standard_name': 'air_temperature', 'units': 'K'},
            'CaSR_v3.1_P_P0_SFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'CaSR_v3.1_P_HU_1.5m': {'standard_name': 'specific_humidity', 'units': '1'},
            'CaSR_v3.1_P_UVC_10m': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'CaSR_v3.1_P_FI_SFC': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'CaSR_v3.1_P_FB_SFC': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'CaSR_v3.1_P_PR0_SFC': {'standard_name': 'precipitation_flux', 'units': 'm'}
        },
        'CASR_v3.2': {
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'sfcWind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'uas': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'vas': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'kg/m^2/s'}
        },
        'DayMet': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'dayl': {'standard_name': 'day_length', 'units': 's/day'},
            'prcp': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'srad': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'swe': {'standard_name': 'snow_water_equivalent', 'units': 'kg/m^2'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC'},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC'},
            'vp': {'standard_name': 'water_vapor_pressure', 'units': 'Pa'}
        },
        'NEX-GDDP': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'tasmax': {'standard_name': 'air_temperature_max', 'units': 'K'},
            'tasmin': {'standard_name': 'air_temperature_min', 'units': 'K'},
            'hurs': {'standard_name': 'relative_humidity', 'units': '%'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'sfcWind': {'standard_name': 'wind_speed', 'units': 'm/s'}
        },
        'GWF-I': {
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'PREC_ACC_NC': {'standard_name': 'precipitation_flux', 'units': 'mm/hr'},
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'GWF-II': {
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'PREC_ACC_NC': {'standard_name': 'precipitation_flux', 'units': 'mm/hr'},
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'CCRN-CanRCM4': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'ta': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'hus': {'standard_name': 'specific_humidity', 'units': '1'},
            'wind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'lsds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'CCRN-WFDEI': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'ta': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'hus': {'standard_name': 'specific_humidity', 'units': '1'},
            'wind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'lsds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'Ouranos-ESPO': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'tasmax': {'standard_name': 'air_temperature_max', 'units': 'K'},
            'tasmin': {'standard_name': 'air_temperature_min', 'units': 'K'}
        },
        'Ouranos-MRCC5': {
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'uas': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'vas': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'AGCD': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC'},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC'}
        }

    }

    # Model variable requirements
    MODEL_REQUIREMENTS = {
        'SUMMA': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'FUSE': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'precip': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'pet': {'standard_name': 'potential_evapotranspiration', 'units': 'mm/day', 'required': False}
        },
        'GR': {
            'P': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'E': {'standard_name': 'potential_evapotranspiration', 'units': 'mm/day', 'required': False},
            'T': {'standard_name': 'air_temperature', 'units': 'degC', 'required': False}
        },
        'HYPE': {
            'Tair': {'standard_name': 'air_temperature', 'units': 'degC'},
            'Prec': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'Tmax': {'standard_name': 'air_temperature_max', 'units': 'degC', 'required': False},
            'Tmin': {'standard_name': 'air_temperature_min', 'units': 'degC', 'required': False},
            'PET': {'standard_name': 'potential_evapotranspiration', 'units': 'mm/day', 'required': False},
            'RHum': {'standard_name': 'relative_humidity', 'units': '1', 'required': False},
            'Wind': {'standard_name': 'wind_speed', 'units': 'm/s', 'required': False},
            'SWRad': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2', 'required': False}
        },
        'RHESSys': {
            'tavg': {'standard_name': 'air_temperature', 'units': 'degC'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC', 'required': False},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC', 'required': False},
            'rain': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'srad': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'vpd': {'standard_name': 'vapor_pressure_deficit', 'units': 'Pa', 'required': False},
            'dayl': {'standard_name': 'day_length', 'units': 's', 'required': False}
        },
        'MESH': {
            'TA': {'standard_name': 'air_temperature', 'units': 'K'},
            'PRE': {'standard_name': 'precipitation_flux', 'units': 'kg/m^2/s'},
            'FSIN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'FLIN': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'QA': {'standard_name': 'specific_humidity', 'units': '1'},
            'UV': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'PRES': {'standard_name': 'surface_air_pressure', 'units': 'Pa'}
        },
        'NGEN': {
            'TMP_2maboveground': {'standard_name': 'air_temperature', 'units': 'K'},
            'APCP_surface': {'standard_name': 'precipitation_flux', 'units': 'kg/m^2/s'},
            'DSWRF_surface': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2', 'required': False},
            'DLWRF_surface': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2', 'required': False},
            'SPFH_2maboveground': {'standard_name': 'specific_humidity', 'units': '1', 'required': False},
            'UGRD_10maboveground': {'standard_name': 'eastward_wind', 'units': 'm/s', 'required': False},
            'VGRD_10maboveground': {'standard_name': 'northward_wind', 'units': 'm/s', 'required': False},
            'WIND_10maboveground': {'standard_name': 'wind_speed', 'units': 'm/s', 'required': False},
            'PRES_surface': {'standard_name': 'surface_air_pressure', 'units': 'Pa', 'required': False}
        },
        'LSTM': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        }
    }

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, dataset: str, model: str):
        """
        Initialize VariableHandler with configuration settings.

        Args:
            config: SYMFLUENCE configuration dictionary
            logger: SYMFLUENCE logger instance
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset if dataset is not None else config.get('FORCING_DATASET')
        self.model = model if model is not None else config.get('HYDROLOGICAL_MODEL')

        # Initialize pint for unit handling
        self.ureg: pint.UnitRegistry = pint.UnitRegistry()
        pint_xarray.setup_registry(self.ureg)

        # Validate dataset and model are supported
        if self.dataset not in self.DATASET_MAPPINGS:
            self.logger.error(f"Unsupported dataset: {self.dataset}")
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.model not in self.MODEL_REQUIREMENTS:
            self.logger.error(f"Unsupported model: {self.model}")
            raise ValueError(f"Unsupported model: {self.model}")

    def get_dataset_variables(self, dataset: Optional[str] = None) -> str:
        """
        Get the forcing variable keys for a specified dataset as a comma-separated string.

        Args:
            dataset (Optional[str]): Name of the dataset. If None, uses the instance's dataset.

        Returns:
            str: Comma-separated string of variable keys for the specified dataset

        Raises:
            ValueError: If the specified dataset is not supported
        """
        # Use instance dataset if none provided
        dataset_name = dataset if dataset is not None else self.dataset

        # Check if dataset exists in mappings
        if dataset_name not in self.DATASET_MAPPINGS:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        return ','.join(self.DATASET_MAPPINGS[dataset_name].keys())

    def process_forcing_data(self, data: xr.Dataset) -> xr.Dataset:
        """Process forcing data by mapping variable names and converting units."""
        self.logger.debug("Starting forcing data unit processing")

        processed_data = data.copy()

        # Get dataset and model mappings
        dataset_map = self.DATASET_MAPPINGS[self.dataset]
        model_map = self.MODEL_REQUIREMENTS[self.model]

        # Get available variables in the data for matching
        available_vars = set(processed_data.data_vars)

        # Process each model variable
        for model_var, model_req in model_map.items():
            # Check if this variable is required (default to True if not specified)
            is_required = model_req.get('required', True)

            # Find corresponding dataset variable that exists in the data
            dataset_var = self._find_matching_variable(model_req['standard_name'], dataset_map, available_vars)

            if dataset_var is None:
                if is_required:
                    self.logger.error(f"Required variable {model_var} not found in dataset {self.dataset}")
                    raise ValueError(f"Required variable {model_var} not found in dataset {self.dataset}")
                else:
                    self.logger.debug(f"Optional variable {model_var} not found in dataset {self.dataset}, skipping")
                    continue

            # Rename variable
            if dataset_var in processed_data:
                self.logger.debug(f"Processing {dataset_var} -> {model_var}")

                # Get units: prioritize metadata from the DataArray over hardcoded mapping
                data_units = str(processed_data[dataset_var].attrs.get('units', '')).lower()
                source_units = dataset_map[dataset_var]['units']

                # If metadata exists and looks different from our mapping, trust metadata
                # BUT perform a range check for temperature to handle inconsistent files
                if data_units and data_units != source_units.lower():
                    # Handle minor string variations (e.g. 'degc' vs 'degC')
                    if data_units in ['degc', 'celsius', 'c']:
                        actual_source_units = 'degC'
                    elif data_units in ['k', 'kelvin']:
                        actual_source_units = 'K'
                    else:
                        actual_source_units = data_units

                    if actual_source_units != source_units:
                        # Perform range check for temperature
                        if model_req['standard_name'] == 'air_temperature':
                            temp_mean = float(processed_data[dataset_var].mean())
                            if temp_mean > 100 and actual_source_units == 'degC':
                                self.logger.warning(f"Metadata for {dataset_var} says 'degC' but mean value is {temp_mean:.2f}. Assuming 'K'.")
                                actual_source_units = 'K'
                            elif temp_mean < 100 and actual_source_units == 'K':
                                self.logger.warning(f"Metadata for {dataset_var} says 'K' but mean value is {temp_mean:.2f}. Assuming 'degC'.")
                                actual_source_units = 'degC'

                        # Perform range check for precipitation flux
                        # Common issue: metadata says 'm/s' or 'm s-1' but values are actually in mm/s
                        if model_req['standard_name'] == 'precipitation_flux':
                            precip_max = float(processed_data[dataset_var].max())
                            actual_lower = actual_source_units.lower().replace(' ', '')
                            # Check if metadata claims m/s but values look like mm/s
                            # Max precip rate of 0.01 m/s = 864 mm/day (extremely high)
                            # Typical max hourly precip: 50 mm/h = 0.014 mm/s
                            if ('m/s' in actual_lower or 'ms-1' in actual_lower or 'm s-1' in actual_source_units.lower()) and 'mm' not in actual_lower:
                                if precip_max < 0.1:  # If max < 0.1 m/s, likely mm/s
                                    self.logger.warning(
                                        f"Metadata for {dataset_var} says '{actual_source_units}' but max value is {precip_max:.6f}. "
                                        f"This looks like mm/s, not m/s. Using 'mm/s' to avoid ~1000x conversion error."
                                    )
                                    actual_source_units = 'mm/s'

                        if actual_source_units != source_units:
                            self.logger.info(f"Using metadata units '{actual_source_units}' instead of mapping '{source_units}' for {dataset_var}")
                            source_units = actual_source_units

                target_units = model_req['units']

                # Convert units if needed
                if source_units != target_units:
                    self.logger.debug(f"Converting units for {dataset_var}: {source_units} -> {target_units}")
                    try:
                        processed_data[dataset_var] = self._convert_units(
                            processed_data[dataset_var],
                            source_units,
                            target_units
                        )
                    except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                        self.logger.error(f"Unit conversion failed for {dataset_var}: {str(e)}")
                        raise

                # Rename after conversion
                processed_data = processed_data.rename({dataset_var: model_var})

        self.logger.debug("Forcing data unit processing completed")
        return processed_data

    def _find_matching_variable(self, standard_name: str, dataset_map: Dict, available_vars: Optional[set] = None) -> Optional[str]:
        """Find dataset variable matching the required standard_name.

        Args:
            standard_name: The CF standard name to match
            dataset_map: Mapping of variable names to their attributes
            available_vars: Optional set of variable names actually present in the data.
                           If provided, only returns variables that exist in this set.
        """
        for var, attrs in dataset_map.items():
            if attrs['standard_name'] == standard_name:
                # If available_vars provided, check if variable exists in data
                if available_vars is not None:
                    if var in available_vars:
                        return var
                    # Continue searching for another match that exists
                    continue
                return var
        self.logger.warning(f"No matching variable found for standard_name: {standard_name}")
        return None

    def _normalize_unit_string(self, unit_str: str) -> str:
        """
        Normalize unit strings to formats that Pint handles reliably.
        Example: 'mm hour-1' -> 'mm / hour'
        Example: 'm s-1' -> 'm / s'
        """
        if not unit_str:
            return unit_str

        import re
        norm = unit_str.strip()

        # Handle space-separated negative exponents: 'm s-1' -> 'm / s'
        # Process iteratively to handle multiple terms like 'kg m-2 s-1'
        while True:
            new_norm = re.sub(r'(\w+)\s+(\w+)-(\d+)', lambda m: f"{m.group(1)} / {m.group(2)}" + (f"**{m.group(3)}" if m.group(3) != '1' else ''), norm, count=1)
            if new_norm == norm:
                break
            norm = new_norm

        # Handle 'X**-1' -> '/ X'
        norm = re.sub(r'(\w+)\*\*-1\b', r'/ \1', norm)

        # Handle 'X^-1' -> '/ X'
        norm = re.sub(r'(\w+)\^-1\b', r'/ \1', norm)

        # Handle 'X**-N' -> '/ X**N' (for N > 1)
        norm = re.sub(r'(\w+)\*\*-(\d+)', r'/ \1**\2', norm)

        # Handle 'X^-N' -> '/ X^N' (for N > 1)
        norm = re.sub(r'(\w+)\^-(\d+)', r'/ \1^\2', norm)

        # Standardize spaces around operators
        norm = norm.replace('**', '__POW__')
        norm = norm.replace('/', ' / ')
        norm = norm.replace('*', ' * ')
        norm = norm.replace('__POW__', '**')

        # Final cleanup of any potential double slashes or extra spaces
        norm = ' '.join(norm.split())
        norm = norm.replace('/ /', '/')

        if norm != unit_str:
            self.logger.debug(f"Normalized units: '{unit_str}' -> '{norm}'")

        return norm

    def _convert_units(self, data: xr.DataArray, from_units: str, to_units: str) -> xr.DataArray:
        """
        Convert variable units using pint-xarray.

        Args:
            data: DataArray to convert
            from_units: Source units
            to_units: Target units

        Returns:
            DataArray with converted units
        """
        # Normalize unit strings for pint
        orig_from = from_units
        from_units = self._normalize_unit_string(from_units)
        to_units = self._normalize_unit_string(to_units)

        try:
            # Special case for precipitation flux conversions (very common source of errors)
            # Handle various kg/m²/s formats: 'kg/m2/s', 'kg m-2 s-1', 'kilogram / meter ** 2 / second'
            f_lower = from_units.lower().replace(' ', '')
            is_kg_m2_s = any(pattern in f_lower for pattern in ['kg/m2/s', 'kgm-2s-1', 'kg/m^2/s', 'kgm^-2s^-1', 'kg/m**2/s'])
            if not is_kg_m2_s:
                is_kg_m2_s = 'kilogram' in from_units.lower() and 'meter' in from_units.lower() and 'second' in from_units.lower()

            if is_kg_m2_s and 'mm' in to_units.lower() and 'day' in to_units.lower():
                # 1 kg/m² = 1 mm of water
                # Convert kg/m²/s to mm/s, then to mm/day
                converted = data * 86400  # multiply by seconds per day
                return converted

            # Additional manual check for common precipitation variants if pint might fail
            if 'mm' in from_units.lower() and 'hour' in from_units.lower() and 'mm' in to_units.lower() and 'day' in to_units.lower():
                return data * 24.0

            # Regular unit conversion
            try:
                data = data.pint.quantify(from_units)
                converted = data.pint.to(to_units)
                return converted.pint.dequantify()
            except Exception as pe:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"Pint conversion failed for {orig_from} -> {to_units}: {pe}. Trying manual fallback.")
                # Manual fallbacks for common meteorological variables
                f_low = from_units.lower()
                t_low = to_units.lower()

                # Temperature: Kelvin to Celsius
                if 'k' in f_low and 'c' in t_low and 'deg' in t_low:
                    return data - 273.15
                # Temperature: Celsius to Kelvin
                if 'c' in f_low and 'deg' in f_low and 'k' in t_low:
                    return data + 273.15
                # Precipitation: mm/h to mm/day
                if 'mm' in f_low and 'hour' in f_low and 'mm' in t_low and 'day' in t_low:
                    return data * 24.0
                # Precipitation: mm/s to mm/day
                if 'mm' in f_low and 's' in f_low and 'mm' in t_low and 'day' in t_low:
                    return data * 86400.0
                # Precipitation: kg/m2/s to mm/day
                if ('kg' in f_low and 'm' in f_low and 's' in f_low) and 'mm' in t_low and 'day' in t_low:
                    return data * 86400.0

                raise pe
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Unit conversion failed: {orig_from} -> {to_units}: {str(e)}")
            raise

    def save_mappings(self, filepath: Path):
        """Save current mappings to YAML file."""
        self.logger.info(f"Saving variable mappings to: {filepath}")
        mappings = {
            'dataset_mappings': self.DATASET_MAPPINGS,
            'model_requirements': self.MODEL_REQUIREMENTS
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(mappings, f)
            self.logger.info("Variable mappings saved successfully")
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Failed to save mappings: {str(e)}")
            raise

    @classmethod
    def load_mappings(cls, filepath: Path, logger: logging.Logger):
        """Load mappings from YAML file."""
        logger.info(f"Loading variable mappings from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                mappings = yaml.safe_load(f)

            cls.DATASET_MAPPINGS = mappings['dataset_mappings']
            cls.MODEL_REQUIREMENTS = mappings['model_requirements']
            logger.info("Variable mappings loaded successfully")
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            logger.error(f"Failed to load mappings: {str(e)}")
            raise
