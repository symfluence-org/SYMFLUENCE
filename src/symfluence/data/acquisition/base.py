# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base Acquisition Handler for SYMFLUENCE.

Provides the abstract base class that all data acquisition handlers inherit from.
Centralizes common functionality like bounding box parsing, temporal subsetting,
caching logic, and diagnostic visualization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import pandas as pd

from symfluence.core import ConfigurableMixin
from symfluence.geospatial.coordinate_utils import CoordinateUtilsMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

    from .earthdata_auth import EarthdataAuth


class BaseAcquisitionHandler(ABC, ConfigurableMixin, CoordinateUtilsMixin):
    """
    Abstract base class for all data acquisition handlers.

    Provides the common infrastructure for downloading meteorological forcing
    data, geospatial attributes, and observations from various data sources.
    Each handler implements the download() method for its specific data source.

    Inherited Capabilities:
        - ConfigurableMixin: Configuration access, project paths, domain info
        - CoordinateUtilsMixin: Bounding box parsing, CRS handling

    Common Functionality:
        - Temporal range parsing from EXPERIMENT_TIME_START/END
        - Bounding box coordinate parsing and validation
        - Skip-if-exists caching logic with force override
        - Diagnostic plotting hooks for acquired data
        - Automatic directory creation and path management

    Attributes:
        config: SymfluenceConfig instance (auto-converted from dict if needed)
        logger: Logger for acquisition progress messages
        bbox: Parsed bounding box tuple (lat_min, lon_min, lat_max, lon_max)
        start_date: Simulation start timestamp
        end_date: Simulation end timestamp
        reporting_manager: Optional manager for diagnostic visualization

    Abstract Methods:
        download(output_dir): Must be implemented by subclasses to perform
                              the actual data download and return output path
    """
    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger,
        reporting_manager: Any = None
    ):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Standard attributes via typed config access
        self.bbox = self._parse_bbox(self._get_config_value(
            lambda: self.config.domain.bounding_box_coords, default=None,
            dict_key='BOUNDING_BOX_COORDS'))
        if not self.bbox:
            # Point domains may set only POUR_POINT_COORDS; derive the same square
            # extent the delineator and AcquisitionService use (shared ConfigMixin
            # helper) so cloud handlers (DEM, forcing, …) have a bbox to fetch.
            self.bbox = self._parse_bbox(self._resolve_point_bbox())
        self.start_date = pd.to_datetime(self._get_config_value(
            lambda: self.config.domain.time_start, default=None,
            dict_key='EXPERIMENT_TIME_START'))
        self.end_date = pd.to_datetime(self._get_config_value(
            lambda: self.config.domain.time_end, default=None,
            dict_key='EXPERIMENT_TIME_END'))

        if not self.bbox:
            self.logger.warning(
                "No bounding box resolved for acquisition handler %s. "
                "Set BOUNDING_BOX_COORDS ('north/west/south/east') in your "
                "configuration file.",
                type(self).__name__,
            )

    @property
    def domain_dir(self) -> Path:
        """Alias for project_dir (backward compatibility)."""
        return self.ensure_dir(self.project_dir)

    def _attribute_dir(self, subdir: str) -> Path:
        """Get attribute subdirectory, ensuring it exists."""
        return self.ensure_dir(self.project_attributes_dir / subdir)

    @abstractmethod
    def download(self, output_dir: Path) -> Path:
        pass

    def plot_diagnostics(self, file_path: Path):
        """
        Create diagnostic plots for the acquired data.
        Can be overridden by subclasses for specific plotting needs.
        """
        if not self.reporting_manager:
            return

        try:
            if file_path.suffix in ['.tif', '.nc']:
                # Raster data
                self.reporting_manager.visualize_spatial_coverage(
                    file_path,
                    variable_name=file_path.stem,
                    stage='acquisition'
                )
            elif file_path.suffix == '.csv':
                # Tabular data - try to read and plot distribution
                df = pd.read_csv(file_path)
                # Assume numeric columns are interesting
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if 'id' not in col.lower() and 'date' not in col.lower():
                        self.reporting_manager.visualize_data_distribution(
                            df[col],
                            variable_name=col,
                            stage='acquisition'
                        )
        except Exception as e:  # noqa: BLE001 — diagnostic plots are non-critical
            self.logger.warning(f"Failed to create diagnostic plots for {file_path}: {e}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _skip_if_exists(self, path: Path, force: bool = None) -> bool:
        """
        Check if a file exists and should be skipped.

        Args:
            path: Path to check
            force: Override for FORCE_DOWNLOAD config (uses config if None)

        Returns:
            True if file exists and should be skipped, False otherwise
        """
        if force is None:
            force = self._get_config_value(lambda: self.config.data.force_download, default=False)

        if path.exists() and not force:
            self.logger.info(f"Using existing file: {path}")
            return True
        return False

    def _earthdata_auth(self) -> 'EarthdataAuth':
        """NASA Earthdata auth helper bound to this handler's config and logger.

        Lookup order (see :mod:`~symfluence.data.acquisition.earthdata_auth`):
        config -> environment -> .netrc -> in-process earthaccess login; the
        token comes from config, then the environment.
        """
        from .earthdata_auth import EarthdataAuth
        return EarthdataAuth(
            logger=self.logger,
            config_lookup=lambda key: self._get_config_value(lambda: None, default=None, dict_key=key),
        )

    def _get_earthdata_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """NASA Earthdata ``(username, password)``, or ``(None, None)``.

        Order: config, ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD`` env vars,
        .netrc, in-process earthaccess login.
        """
        return self._earthdata_auth().credentials()

    def _get_earthdata_token(self) -> Optional[str]:
        """NASA Earthdata Bearer token (config, then ``EARTHDATA_TOKEN`` env var), or None."""
        return self._earthdata_auth().token()
