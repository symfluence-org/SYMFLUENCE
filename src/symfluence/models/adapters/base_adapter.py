# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base Forcing Adapter for SYMFLUENCE.

This module defines the abstract base class for forcing data adapters.
Each hydrological model implements its own adapter to convert CFIF
(CF-Intermediate Format) data to its specific format.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import xarray as xr

from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)


class ForcingAdapter(ConfigMixin, ABC):
    """
    Abstract base class for model forcing adapters.

    Forcing adapters convert CFIF (CF-Intermediate Format) forcing data
    to model-specific formats. Each model implements its own adapter
    with specific variable mappings and transformations.

    Responsibilities:
        - Map CFIF variable names to model-specific names
        - Convert units if model requires different units
        - Apply any model-specific transformations
        - Handle temporal aggregation (e.g., hourly to daily)
        - Add model-specific metadata

    Example Implementation:
        >>> @ForcingAdapterRegistry.register_adapter('SUMMA')
        >>> class SUMMAForcingAdapter(ForcingAdapter):
        ...     def get_variable_mapping(self):
        ...         return {
        ...             'air_temperature': 'air_temperature',
        ...             'precipitation_flux': 'precipitation_flux',
        ...             # ...
        ...         }
        ...
        ...     def transform(self, cfif_data):
        ...         ds = self.rename_variables(cfif_data)
        ...         return ds

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
    """

    def __init__(self, config: Union[Dict[str, Any], 'SymfluenceConfig'], logger: Optional[logging.Logger] = None):
        """
        Initialize the forcing adapter.

        Args:
            config: Configuration dictionary or SymfluenceConfig instance with model settings
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Get mapping from CFIF variable names to model variable names.

        Returns:
            Dict mapping CFIF names to model-specific names.
            Keys are CFIF names (e.g., 'air_temperature'),
            values are model names (e.g., 'air_temperature' for SUMMA).

        Example:
            >>> return {
            ...     'air_temperature': 'air_temperature',
            ...     'precipitation_flux': 'precipitation_flux',
            ...     'surface_downwelling_shortwave_flux': 'surface_downwelling_shortwave_flux',
            ... }
        """
        pass

    @abstractmethod
    def get_required_variables(self) -> List[str]:
        """
        Get list of CFIF variables required by this model.

        Returns:
            List of CFIF variable names that must be present.

        Example:
            >>> return [
            ...     'air_temperature',
            ...     'precipitation_flux',
            ...     'surface_downwelling_shortwave_flux',
            ... ]
        """
        pass

    def get_optional_variables(self) -> List[str]:
        """
        Get list of CFIF variables that are optional for this model.

        Override this method if your model can use additional variables
        that are not strictly required.

        Returns:
            List of optional CFIF variable names.
        """
        return []

    def get_unit_conversions(self) -> Dict[str, Callable]:
        """
        Get unit conversion functions for variables.

        Override this method if the model requires different units
        than the CFIF standard units.

        Returns:
            Dict mapping CFIF variable names to conversion functions.
            Conversion functions take array data and return converted data.

        Example:
            >>> return {
            ...     'air_temperature': lambda x: x - 273.15,  # K to °C
            ...     'precipitation_flux': lambda x: x * 3600,  # to mm/hr
            ... }
        """
        return {}

    def transform(self, cfif_data: xr.Dataset) -> xr.Dataset:
        """
        Transform CFIF dataset to model-specific format.

        This is the main entry point for forcing conversion.
        The default implementation:
        1. Validates required variables are present
        2. Renames variables to model names
        3. Applies unit conversions
        4. Updates metadata attributes

        Override this method for custom transformation logic.

        Args:
            cfif_data: xarray Dataset in CFIF format

        Returns:
            xarray Dataset in model-specific format

        Raises:
            ValueError: If required variables are missing
        """
        # Validate required variables
        self._validate_input(cfif_data)

        # Rename variables
        ds = self.rename_variables(cfif_data)

        # Apply unit conversions
        ds = self.apply_unit_conversions(ds)

        # Add model-specific metadata
        ds = self.add_metadata(ds)

        return ds

    def _validate_input(self, cfif_data: xr.Dataset) -> None:
        """
        Validate that required CFIF variables are present.

        Args:
            cfif_data: Input dataset

        Raises:
            ValueError: If required variables are missing
        """
        required = set(self.get_required_variables())
        present = set(cfif_data.data_vars)

        missing = required - present
        if missing:
            raise ValueError(
                f"Missing required CFIF variables for {self.__class__.__name__}: "
                f"{sorted(missing)}"
            )

    def rename_variables(self, cfif_data: xr.Dataset) -> xr.Dataset:
        """
        Rename CFIF variables to model-specific names.

        Args:
            cfif_data: Dataset with CFIF variable names

        Returns:
            Dataset with model-specific variable names
        """
        mapping = self.get_variable_mapping()

        # Only rename variables that are present
        rename_dict = {
            cfif_name: model_name
            for cfif_name, model_name in mapping.items()
            if cfif_name in cfif_data.data_vars
        }

        if rename_dict:
            return cfif_data.rename(rename_dict)
        return cfif_data

    def apply_unit_conversions(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply unit conversions to variables.

        Args:
            ds: Dataset (already renamed to model names)

        Returns:
            Dataset with converted units
        """
        conversions = self.get_unit_conversions()
        mapping = self.get_variable_mapping()

        # Map CFIF conversion keys to model variable names
        for cfif_name, converter in conversions.items():
            model_name = mapping.get(cfif_name, cfif_name)
            if model_name in ds.data_vars:
                ds[model_name] = converter(ds[model_name])
                self.logger.debug(f"Applied unit conversion to {model_name}")

        return ds

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add model-specific metadata to dataset.

        Override this method to add custom metadata attributes.

        Args:
            ds: Dataset to annotate

        Returns:
            Dataset with added metadata
        """
        ds.attrs['forcing_adapter'] = self.__class__.__name__
        return ds

    def get_model_name(self) -> str:
        """
        Get the model name this adapter is for.

        Returns:
            Model name string (e.g., 'SUMMA', 'HYPE')
        """
        # Extract from class name by convention
        name = self.__class__.__name__
        if name.endswith('ForcingAdapter'):
            return name[:-len('ForcingAdapter')].upper()
        return name.upper()
