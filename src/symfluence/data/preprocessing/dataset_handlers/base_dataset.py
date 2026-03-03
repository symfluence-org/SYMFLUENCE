# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base Dataset Handler for SYMFLUENCE

This module provides:
- BaseDatasetHandler: Abstract base class for dataset-specific handlers
- StandardVariableAttributes: Standard CF-compliant variable attribute definitions
- apply_standard_variable_attributes: Helper to apply standard attributes to datasets

Variable Naming:
- CFIF (CF-Intermediate Format): Model-neutral names (e.g., 'air_temperature')
- Legacy (SUMMA-style): Backward-compatible names (e.g., 'airtemp')

New code should use CFIF names; legacy names are maintained for compatibility.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr

from symfluence.core.mixins.config import ConfigMixin


def _get_cfif_variable_attributes() -> Dict[str, Dict[str, str]]:
    """Lazy load CFIF variable attributes to avoid circular imports."""
    from symfluence.data.preprocessing.cfif.variables import CFIF_VARIABLES
    return {
        name: {
            'units': attrs['units'],
            'long_name': attrs['long_name'],
            'standard_name': attrs['cf_standard_name'],
        }
        for name, attrs in CFIF_VARIABLES.items()
    }


# CFIF variable attributes - lazy loaded on first access
_cfif_attrs_cache: Optional[Dict[str, Dict[str, str]]] = None


def _get_cached_cfif_attrs() -> Dict[str, Dict[str, str]]:
    """Get cached CFIF attributes."""
    global _cfif_attrs_cache
    if _cfif_attrs_cache is None:
        _cfif_attrs_cache = _get_cfif_variable_attributes()
    return _cfif_attrs_cache


# Legacy SUMMA-style variable attribute definitions (for backward compatibility)
# These can be overridden by individual handlers if needed
STANDARD_VARIABLE_ATTRIBUTES: Dict[str, Dict[str, str]] = {
    'airpres': {
        'units': 'Pa',
        'long_name': 'air pressure',
        'standard_name': 'air_pressure',
    },
    'airtemp': {
        'units': 'K',
        'long_name': 'air temperature',
        'standard_name': 'air_temperature',
    },
    'pptrate': {
        'units': 'kg m-2 s-1',
        'long_name': 'precipitation rate',
        'standard_name': 'precipitation_flux',
    },
    'windspd': {
        'units': 'm s-1',
        'long_name': 'wind speed',
        'standard_name': 'wind_speed',
    },
    'LWRadAtm': {
        'units': 'W m-2',
        'long_name': 'downward longwave radiation at the surface',
        'standard_name': 'surface_downwelling_longwave_flux_in_air',
    },
    'SWRadAtm': {
        'units': 'W m-2',
        'long_name': 'downward shortwave radiation at the surface',
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
    },
    'spechum': {
        'units': 'kg kg-1',
        'long_name': 'specific humidity',
        'standard_name': 'specific_humidity',
    },
    'relhum': {
        'units': '%',
        'long_name': 'relative humidity',
        'standard_name': 'relative_humidity',
    },
    'windspd_u': {
        'units': 'm s-1',
        'long_name': 'eastward wind component',
        'standard_name': 'eastward_wind',
    },
    'windspd_v': {
        'units': 'm s-1',
        'long_name': 'northward wind component',
        'standard_name': 'northward_wind',
    },
}

def _get_all_variable_attributes() -> Dict[str, Dict[str, str]]:
    """Get combined CFIF + legacy attributes (lazy loaded)."""
    return {
        **_get_cached_cfif_attrs(),
        **STANDARD_VARIABLE_ATTRIBUTES,
    }


def apply_standard_variable_attributes(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
    use_cfif: bool = True
) -> xr.Dataset:
    """
    Apply standard CF-compliant attributes to dataset variables.

    This function centralizes the attribute-setting logic that was previously
    duplicated across all dataset handlers. It supports both CFIF and legacy
    SUMMA-style variable names.

    Args:
        ds: xarray Dataset to modify
        variables: List of variable names to process. If None, processes all
                  variables that have standard definitions.
        overrides: Optional dict of {var_name: {attr: value}} to override defaults
        use_cfif: If True (default), also check CFIF variable names

    Returns:
        Modified dataset with standardized attributes

    Example:
        >>> ds = apply_standard_variable_attributes(ds)
        >>> ds = apply_standard_variable_attributes(ds, variables=['air_temperature', 'precipitation_flux'])
        >>> ds = apply_standard_variable_attributes(ds, overrides={'pptrate': {'units': 'mm/s'}})
    """
    # Use combined attributes (CFIF + legacy) for maximum compatibility
    attrs_to_apply = _get_all_variable_attributes().copy() if use_cfif else STANDARD_VARIABLE_ATTRIBUTES.copy()

    # Merge overrides with defaults
    if overrides:
        for var, var_overrides in overrides.items():
            if var in attrs_to_apply:
                attrs_to_apply[var] = {**attrs_to_apply[var], **var_overrides}
            else:
                attrs_to_apply[var] = var_overrides

    # Determine which variables to process
    if variables is None:
        variables = list(attrs_to_apply.keys())

    # Apply attributes to each variable present in the dataset
    for var_name in variables:
        if var_name in ds.data_vars and var_name in attrs_to_apply:
            ds[var_name].attrs.update(attrs_to_apply[var_name])

    return ds


def apply_cfif_variable_attributes(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None
) -> xr.Dataset:
    """
    Apply CFIF (CF-Intermediate Format) attributes to dataset variables.

    This function is specifically for datasets using CFIF variable naming.

    Args:
        ds: xarray Dataset to modify
        variables: List of CFIF variable names to process. If None, processes all
                  CFIF variables present in the dataset.
        overrides: Optional dict of {var_name: {attr: value}} to override defaults

    Returns:
        Modified dataset with CFIF-compliant attributes

    Example:
        >>> ds = apply_cfif_variable_attributes(ds)
        >>> ds = apply_cfif_variable_attributes(ds, variables=['air_temperature', 'precipitation_flux'])
    """
    attrs_to_apply = _get_cached_cfif_attrs().copy()

    if overrides:
        for var, var_overrides in overrides.items():
            if var in attrs_to_apply:
                attrs_to_apply[var] = {**attrs_to_apply[var], **var_overrides}
            else:
                attrs_to_apply[var] = var_overrides

    if variables is None:
        variables = list(attrs_to_apply.keys())

    for var_name in variables:
        if var_name in ds.data_vars and var_name in attrs_to_apply:
            ds[var_name].attrs.update(attrs_to_apply[var_name])

    return ds


class BaseDatasetHandler(ABC, ConfigMixin):
    """
    Abstract base class for dataset-specific forcing data handlers.

    Provides a standardized interface and common functionality for processing different
    meteorological datasets (ERA5, RDRS, CARRA, CONUS404, etc.) into model-ready
    forcing files using CFIF (CF-Intermediate Format) naming conventions.

    Output Format:
        Handlers output data in CFIF format by default, using CF-compliant variable
        names (e.g., 'air_temperature', 'precipitation_flux'). Model-specific adapters
        in the models/adapters/ package convert CFIF to model-specific formats.

    Common Functionality:
        - Variable name standardization (dataset-specific → CFIF standard)
        - CF-compliant attribute management
        - Time encoding standardization
        - NetCDF metadata handling
        - Missing value conventions
        - Coordinate name resolution

    Required Subclass Methods (Abstract):
        - get_variable_mapping(): Dataset-specific to standard variable names
        - process_dataset(): Apply dataset-specific transformations
        - get_coordinate_names(): Return (lat_name, lon_name) tuple
        - create_shapefile(): Generate forcing station shapefile
        - merge_forcings(): Merge/process raw files to standardized format
        - needs_merging(): Whether dataset requires file merging

    Optional Override Methods:
        - get_file_pattern(): Pattern for raw forcing files
        - get_merged_file_pattern(): Pattern for merged output files
        - setup_time_encoding(): Custom time encoding
        - add_metadata(): Custom metadata
        - clean_variable_attributes(): Custom attribute cleaning

    Attributes:
        config (Dict): Configuration dictionary
        logger: Logger instance
        project_dir (Path): Project root directory
        domain_name (str): Domain identifier

    Example Subclass:
        >>> @DatasetRegistry.register('my_dataset')
        >>> class MyDatasetHandler(BaseDatasetHandler):
        ...     def get_variable_mapping(self):
        ...         return {'t2m': 'airtemp', 'tp': 'pptrate'}
        ...     def process_dataset(self, ds):
        ...         # Apply conversions
        ...         return ds
        ...     # ... implement other abstract methods

    See Also:
        - ERA5Handler: ERA5 reanalysis data
        - RDRSHandler: Canadian Regional Deterministic Reanalysis System
        - CARRAHandler: Arctic regional reanalysis
    """

    def __init__(self, config: Any, logger: Any, project_dir: Path, **kwargs: Any) -> None:
        """
        Initialize the dataset handler.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for progress and error reporting
            project_dir: Path to project root directory
            **kwargs: Additional handler-specific parameters
                     (e.g., forcing_timestep_seconds)
        """
        self.config = config
        self.logger = logger
        self.project_dir = project_dir
        # Support both typed SymfluenceConfig (.domain.name) and plain dict
        if hasattr(config, 'domain') and not isinstance(config, dict):
            self.domain_name = config.domain.name
        else:
            self.domain_name = config['DOMAIN_NAME']
        # Store extra kwargs like forcing_timestep_seconds if provided
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_variable_mapping(self) -> Dict[str, str]:
        """Return mapping of dataset-native variable names to CFIF standard names.

        Returns:
            Dict mapping source names (e.g. ``'t2m'``) to standard names
            (e.g. ``'air_temperature'``).
        """

    @abstractmethod
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply dataset-specific transformations (unit conversion, renaming, etc.).

        Args:
            ds: Raw xarray Dataset loaded from a forcing file.

        Returns:
            Transformed Dataset with standardised variable names and units.
        """

    @abstractmethod
    def get_coordinate_names(self) -> Tuple[str, str]:
        """Return the latitude and longitude coordinate names used by this dataset.

        Returns:
            ``(lat_name, lon_name)`` tuple (e.g. ``('latitude', 'longitude')``).
        """

    @abstractmethod
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """Generate a forcing-grid shapefile for remapping weight computation.

        Args:
            shapefile_path: Directory to write the output shapefile.
            merged_forcing_path: Path to a merged forcing file for grid extraction.
            dem_path: Path to the DEM raster for elevation assignment.
            elevation_calculator: Callable that computes mean elevation per polygon.

        Returns:
            Path to the created shapefile.
        """

    @abstractmethod
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """Merge or reorganise raw forcing files into a standardised layout.

        Args:
            raw_forcing_path: Directory containing downloaded raw files.
            merged_forcing_path: Directory for merged output files.
            start_year: First year to process (inclusive).
            end_year: Last year to process (inclusive).
        """

    @abstractmethod
    def needs_merging(self) -> bool:
        """Whether this dataset requires a merge step before remapping.

        Returns:
            ``True`` if :meth:`merge_forcings` must be called.
        """

    def get_file_pattern(self) -> str:
        """Return the glob pattern for raw forcing files.

        Returns:
            Glob pattern string (e.g. ``'domain_test_*.nc'``).
        """
        return f"domain_{self.domain_name}_*.nc"

    def get_merged_file_pattern(self, year: int, month: int) -> str:
        """Return the filename pattern for a merged monthly output file.

        Args:
            year: Calendar year.
            month: Calendar month (1-12).

        Returns:
            Filename string (e.g. ``'ERA5_monthly_200401.nc'``).
        """
        dataset_name = self.__class__.__name__.replace('Handler', '').upper()
        return f"{dataset_name}_monthly_{year}{month:02d}.nc"

    def setup_time_encoding(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Set up standard time encoding for NetCDF output.

        Applies CF-compliant time encoding using hours since 1900-01-01
        with Gregorian calendar.

        Args:
            ds: Dataset with time coordinate

        Returns:
            Dataset with time encoding configured

        Note:
            - Uses 'hours since 1900-01-01' for broad compatibility
            - Gregorian calendar is CF-compliant standard
            - This encoding is recognized by most hydrological models
        """
        ds['time'].encoding['units'] = 'hours since 1900-01-01'
        ds['time'].encoding['calendar'] = 'gregorian'
        return ds

    def add_metadata(self, ds: xr.Dataset, description: str) -> xr.Dataset:
        """
        Add standard metadata attributes to dataset.

        Args:
            ds: Dataset to annotate
            description: Description of processing/purpose

        Returns:
            Dataset with added metadata attributes

        Note:
            Adds 'History' with creation timestamp and 'Reason' with description.
        """
        import time
        ds.attrs.update({'History': f'Created {time.ctime(time.time())}', 'Reason': description})
        return ds

    def clean_variable_attributes(self, ds: xr.Dataset, missing_value: float = -999.0) -> xr.Dataset:
        """
        Standardize missing value handling across all variables.

        Removes missing value attributes from variable attrs dict and sets them
        in the encoding instead, following NetCDF best practices.

        Args:
            ds: Dataset to clean
            missing_value: Value to use for missing data (default: -999.0)

        Returns:
            Dataset with cleaned variable attributes

        Note:
            - Removes 'missing_value' and '_FillValue' from attrs
            - Sets both in encoding for each variable
            - Prevents attribute/encoding conflicts
            - Ensures consistent missing value handling across variables
        """
        for var in ds.data_vars:
            # Remove from attributes to avoid conflicts
            if 'missing_value' in ds[var].attrs: del ds[var].attrs['missing_value']
            if '_FillValue' in ds[var].attrs: del ds[var].attrs['_FillValue']

            # Set in encoding for consistent NetCDF output
            ds[var].encoding['missing_value'] = missing_value
            ds[var].encoding['_FillValue'] = missing_value
        return ds

    def apply_standard_attributes(
        self,
        ds: xr.Dataset,
        overrides: Optional[Dict[str, Dict[str, str]]] = None
    ) -> xr.Dataset:
        """
        Apply standard variable attributes to the dataset.

        Convenience method that wraps apply_standard_variable_attributes.
        Subclasses can override this to customize attribute handling.

        Args:
            ds: Dataset to modify
            overrides: Optional attribute overrides per variable

        Returns:
            Modified dataset
        """
        return apply_standard_variable_attributes(ds, overrides=overrides)

    # Patterns in OSError messages that indicate HDF5/netCDF engine issues
    # where falling back to h5netcdf may help.
    _HDF_ERROR_PATTERNS = (
        "HDF error",
        "HDF5 error",
        "Errno -101",
        "unable to lock file",      # errno 11 on parallel filesystems
        "Resource temporarily",      # EAGAIN variant
        "unable to synchronously",   # HDF5 1.14+ error format
    )

    def open_dataset(self, path: Path, **kwargs) -> xr.Dataset:
        """
        Open a NetCDF dataset with automatic engine fallback.

        Tries the default engine first. If that fails with an HDF5/locking
        error (common on HPC where system modules load a conflicting libhdf5,
        or the parallel filesystem does not support POSIX file locking),
        retries with h5netcdf using ``lock=False``.

        If a libhdf5 build conflict between h5py and netCDF4 was detected at
        startup, the h5netcdf fallback is skipped (importing h5py would
        corrupt the netcdf4 backend for the rest of the process).

        Args:
            path: Path to the NetCDF file
            **kwargs: Additional arguments passed to xr.open_dataset

        Returns:
            Opened xarray Dataset
        """
        try:
            return xr.open_dataset(path, **kwargs)
        except OSError as e:
            msg = str(e).lower()
            if not any(pat.lower() in msg for pat in self._HDF_ERROR_PATTERNS):
                raise

            # If h5py and netCDF4 bundle different libhdf5 builds, falling
            # back to h5netcdf would import h5py and poison the netcdf4
            # backend for the rest of the process.  Fail fast with a
            # clear message instead.
            from symfluence.core.hdf5_safety import hdf5_library_conflict
            if hdf5_library_conflict:
                raise OSError(
                    f"netcdf4 engine failed on {Path(path).name} and the "
                    f"h5netcdf fallback is disabled because h5py and netCDF4 "
                    f"link against different libhdf5 builds. Fix: "
                    f"pip uninstall h5py netCDF4 -y && "
                    f"conda install h5py netcdf4"
                ) from e

            self.logger.warning(
                f"netcdf4 engine failed on {Path(path).name} (likely HDF5 "
                f"library conflict or file-locking issue), retrying with "
                f"h5netcdf engine (lock=False)"
            )
            # lock=False avoids Python-level fcntl locking that also fails
            # on parallel filesystems (Lustre, GPFS, BeeGFS).
            fallback_kw = {**kwargs, "lock": False}
            return xr.open_dataset(path, engine="h5netcdf", **fallback_kw)
