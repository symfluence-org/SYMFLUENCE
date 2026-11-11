# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parameter transformation adapters for model-specific calibration.

Provides transformers that apply calibration parameters to model-specific
formats (e.g., soil depth multipliers for SUMMA NetCDF files).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import netCDF4 as nc
import numpy as np

from symfluence.core.calibration.parameters.soil_depth import (
    DEPTH_PSEUDO_PARAMS,
    calculate_soil_depths,
    has_depth_params,
    update_soil_depths,
)
from symfluence.core.mixins import ConfigMixin


class ParameterTransformer(ConfigMixin, ABC):
    """Base class for parameter transformers."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Import here to avoid circular imports

        from symfluence.core.config.models import SymfluenceConfig



        # Auto-convert dict to typed config for backward compatibility

        if isinstance(config, dict):

            try:

                self._config = SymfluenceConfig(**config)

            except (AttributeError, KeyError, TypeError):

                # Fallback for partial configs (e.g., in tests)

                self._config = config

        else:

            self._config = config
        self.logger = logger

    @abstractmethod
    def apply(self, params: Dict[str, Any], settings_dir: Path) -> bool:
        """Apply parameter transformations to model configuration files.

        Args:
            params: Dictionary of parameter names and their calibrated values.
            settings_dir: Path to the model settings directory containing files to modify.

        Returns:
            True if transformation succeeded, False otherwise.
        """

class SoilDepthTransformer(ParameterTransformer):
    """Handles transformation of soil depth parameters.

    The arithmetic and the coldState update are the shared routines in
    ``symfluence.core.calibration.parameters.soil_depth``, the same ones the
    SUMMA parameter manager and worker use; this class only finds the file and
    remembers the original layer depths.
    """

    SPECIAL_PARAMS = list(DEPTH_PSEUDO_PARAMS)

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.original_depths = None

    def apply(self, params: Dict[str, Any], settings_dir: Path) -> bool:
        """Apply soil depth multiplier transformations to SUMMA's coldState file.

        Modifies mLayerDepth and iLayerHeight variables in the NetCDF file
        based on total_soil_depth_multiplier and shape_factor parameters.
        Shape factor controls exponential stretching of layer depths.

        Args:
            params: Parameter dict, may contain 'total_soil_depth_multiplier',
                'total_mult', and/or 'shape_factor' keys.
            settings_dir: Path containing the coldState file
                (``SETTINGS_SUMMA_COLDSTATE``, default coldState.nc).

        Returns:
            True if transformation succeeded or no special params present,
            False if the coldState file is missing or the NetCDF update failed.
        """
        # Check if any special parameters are present
        if not has_depth_params(params):
            return True

        try:
            coldstate_path = Path(settings_dir) / self._get_config_value(lambda: self.config.model.summa.coldstate, default='coldState.nc', dict_key='SETTINGS_SUMMA_COLDSTATE')
            if not coldstate_path.exists():
                self.logger.error(f"{coldstate_path.name} not found at {coldstate_path}")
                return False

            # Load original depths if not already loaded
            if self.original_depths is None:
                self.original_depths = self._get_original_depths(coldstate_path)

            if self.original_depths is None:
                return False

            return update_soil_depths(
                coldstate_path, self.original_depths, params, self.logger,
                component='soil_depth_transformer')

        except (OSError, IOError, KeyError, ValueError) as e:
            self.logger.error(f"Error in SoilDepthTransformer: {str(e)}")
            return False

    def _get_original_depths(self, path: Path) -> Optional[np.ndarray]:
        try:
            with nc.Dataset(path, 'r') as ds:
                return ds.variables['mLayerDepth'][:, 0].copy()
        except (OSError, IOError, KeyError) as e:
            self.logger.debug(f"Could not read mLayerDepth from {path}: {e}")
            return None

    def _calculate_new_depths(self, original: np.ndarray, total_mult: float, shape_factor: float) -> np.ndarray:
        """Rescaled layer depths (the shared ``calculate_soil_depths``)."""
        return calculate_soil_depths(original, total_mult, shape_factor)

class TransformationManager(ConfigMixin):
    """Orchestrates all parameter transformations."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.transformers = [
            SoilDepthTransformer(config, logger)
        ]

    def transform(self, params: Dict[str, Any], settings_dir: Path) -> bool:
        """Apply all registered transformations to model settings.

        Iterates through all registered transformers and applies each
        in sequence. Stops on first failure.

        Args:
            params: Calibration parameter dictionary.
            settings_dir: Path to model settings directory.

        Returns:
            True if all transformations succeeded, False if any failed.
        """
        for transformer in self.transformers:
            if not transformer.apply(params, settings_dir):
                return False
        return True
