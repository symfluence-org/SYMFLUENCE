# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base class for geofabric delineators.

Provides shared infrastructure for all delineation modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation
- Common initialization patterns
- Logger integration

Following the BaseModelPreProcessor pattern from model refactoring.

Refactored from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.path_resolver import PathResolverMixin


class BaseGeofabricDelineator(ABC, PathResolverMixin):
    """
    Abstract base class for all geofabric delineators.

    Provides common initialization, path management, and utility methods
    that are shared across different geofabric delineation strategies.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        num_processes: Number of parallel processes for TauDEM
        max_retries: Maximum number of command retries
        retry_delay: Delay between retries (seconds)
        min_gru_size: Minimum GRU size (km²)
        taudem_dir: Path to TauDEM binary directory
        dem_path: Path to DEM file
        interim_dir: Directory for interim TauDEM files
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize base delineator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        # Set config and logger for mixins (ConfigMixin expects _config attribute)
        self._config = config
        self.logger = logger

        # Base paths (use convenience properties from mixin where available)
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default=None,
            dict_key='SYMFLUENCE_DATA_DIR'
        )
        # Fall back to DATA_DIR if SYMFLUENCE_DATA_DIR not found
        if data_dir is None:
            import tempfile
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir,
                default=tempfile.gettempdir(),  # nosec B108
                dict_key='DATA_DIR'
            )
        self.data_dir = Path(data_dir)
        # domain_name is provided by ConfigMixin via ProjectContextMixin
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Common configuration
        self.num_processes = self._get_config_value(
            lambda: self.config.system.num_processes,
            default=1
        )
        self.max_retries = self._get_config_value(
            lambda: self.config.domain.delineation.max_retries,
            dict_key='MAX_RETRIES',
        )
        self.retry_delay = self._get_config_value(
            lambda: self.config.domain.delineation.retry_delay,
            dict_key='RETRY_DELAY',
        )
        self.min_gru_size = self._get_config_value(lambda: self.config.domain.min_gru_size, dict_key='MIN_GRU_SIZE')
        try:
            min_gru_size_set = float(self.min_gru_size or 0) > 0
        except (TypeError, ValueError):
            min_gru_size_set = False
        if min_gru_size_set:
            self.logger.warning(
                f"MIN_GRU_SIZE={self.min_gru_size} is set but is not applied: delineated "
                "GRUs are never merged, whatever their size. Set it to 0 to silence this "
                "warning (MIN_HRU_SIZE, in km², does merge undersized HRUs within a GRU)."
            )

        # TauDEM configuration
        self.taudem_dir = self._get_taudem_dir()
        self._set_taudem_path()

        # DEM path
        self.dem_path = self._get_dem_path()

        # Interim directory (subclasses can override)
        self.interim_dir = self.project_dir / "taudem-interim-files"

    def _get_taudem_dir(self) -> str:
        """
        Get TauDEM installation directory.

        Returns:
            Path to TauDEM bin directory
        """
        # Try typed config first
        taudem_dir = self._get_config_value(lambda: self.config.paths.taudem_dir, dict_key='TAUDEM_DIR')
        if taudem_dir == "default":
            return str(self.data_dir / 'installs' / 'TauDEM' / 'bin')
        return str(taudem_dir)

    def _set_taudem_path(self):
        """Add TauDEM directory to system PATH."""
        os.environ['PATH'] = f"{os.environ['PATH']}:{self.taudem_dir}"

    def _get_dem_path(self) -> Path:
        """
        Get DEM file path with default handling.

        Returns:
            Path to DEM file
        """
        dem_path = self._get_config_value(
            lambda: self.config.paths.dem_path,
            default=None,
            dict_key='DEM_PATH'
        )
        dem_name = self._get_config_value(
            lambda: self.config.paths.dem_name,
            default=None,
            dict_key='DEM_NAME'
        )

        if dem_name is None or dem_name == "default":
            dem_name = f"domain_{self.domain_name}_elv.tif"

        if dem_path is None or dem_path == 'default':
            attr_dir = resolve_data_subdir(self.project_dir, 'attributes')
            return attr_dir / 'elevation' / 'dem' / dem_name

        return Path(dem_path) / dem_name

    # -----------------------------------------------------------------
    # DEM conditioning (stream burning)
    # -----------------------------------------------------------------

    def _condition_dem(self) -> Path:
        """
        Apply DEM conditioning if configured.

        Returns the path to the conditioned DEM, or ``self.dem_path``
        unchanged when conditioning is disabled (the default).
        """
        method = self._get_config_value(
            lambda: self.config.domain.delineation.dem_conditioning_method,
            dict_key='DEM_CONDITIONING_METHOD',
        )
        if isinstance(method, str):
            method = method.lower().strip()

        if method == 'none':
            return self.dem_path

        if method == 'burn_streams':
            return self._burn_streams_into_dem()

        self.logger.warning(
            f"Unknown DEM_CONDITIONING_METHOD '{method}' — using raw DEM"
        )
        return self.dem_path

    def _burn_streams_into_dem(self) -> Path:
        """
        Burn a stream network into the DEM and return the path to the
        burned raster. Falls back to the original DEM on any error.
        """
        try:
            from ..processors.stream_burner import StreamBurner

            burn_depth = self._get_config_value(
                lambda: self.config.domain.delineation.stream_burn_depth,
                dict_key='STREAM_BURN_DEPTH',
            )
            source = self._get_config_value(
                lambda: self.config.domain.delineation.stream_burn_source,
                dict_key='STREAM_BURN_SOURCE',
            )

            stream_path = self._find_stream_burn_source(source)
            if stream_path is None:
                self.logger.warning(
                    "No stream vector file found for burning — using raw DEM"
                )
                return self.dem_path

            output_path = self.interim_dir / "dem_burned.tif"
            self.interim_dir.mkdir(parents=True, exist_ok=True)

            burner = StreamBurner(self.logger)
            result = burner.burn_streams(
                self.dem_path, stream_path, output_path, burn_depth
            )
            self.logger.info(
                f"DEM stream burning complete (depth={burn_depth}m): {result}"
            )
            return result

        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(
                f"Stream burning failed, falling back to raw DEM: {exc}"
            , exc_info=True)
            return self.dem_path

    def _find_stream_burn_source(self, source: str) -> Optional[Path]:
        """
        Locate a stream vector file for burning.

        Args:
            source: One of 'auto', 'merit', 'tdx', 'custom'.

        Returns:
            Path to the stream vector file, or None if not found.
        """
        source = source.lower().strip()

        if source == 'custom':
            custom = self._get_config_value(
                lambda: self.config.domain.delineation.stream_burn_custom_path,
                dict_key='STREAM_BURN_CUSTOM_PATH',
            )
            if custom and custom != 'default':
                p = Path(custom)
                if p.exists():
                    return p
                self.logger.warning(f"Custom stream burn path not found: {p}")
            return None

        attr_dir = resolve_data_subdir(self.project_dir, 'attributes')
        geofabric_dir = attr_dir / 'geofabric'

        if source == 'merit':
            return self._glob_first(
                geofabric_dir / 'merit', '*rivernet*.shp'
            )

        if source == 'tdx':
            return self._glob_first(
                geofabric_dir / 'tdx', '*rivers*.parquet'
            )

        # 'auto': search all geofabric subdirs, then shapefiles/river_network
        if source == 'auto':
            # Try each geofabric source in preference order
            for subdir, pattern in [
                ('merit', '*rivernet*.shp'),
                ('tdx', '*rivers*.parquet'),
                ('tdx', '*rivers*.shp'),
                ('nws', '*flowpath*.gpkg'),
            ]:
                hit = self._glob_first(geofabric_dir / subdir, pattern)
                if hit is not None:
                    return hit

            # Fallback: river_network shapefiles directory
            river_net_dir = self.project_dir / 'shapefiles' / 'river_network'
            for pattern in ['*.shp', '*.gpkg', '*.parquet']:
                hit = self._glob_first(river_net_dir, pattern)
                if hit is not None:
                    return hit

            return None

        self.logger.warning(f"Unknown STREAM_BURN_SOURCE '{source}'")
        return None

    @staticmethod
    def _glob_first(directory: Path, pattern: str) -> Optional[Path]:
        """Return the first file matching *pattern* in *directory*, or None."""
        if not directory.is_dir():
            return None
        hits = sorted(directory.glob(pattern))
        return hits[0] if hits else None

    def _get_pour_point_path(self) -> Path:
        """
        Get pour point shapefile path.

        Returns:
            Path to pour point shapefile
        """
        pour_point_path = self._get_config_value(
            lambda: self.config.paths.pour_point_path,
            dict_key='POUR_POINT_SHP_PATH',
        )
        if pour_point_path is None or pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(pour_point_path)

        pour_point_name = self._get_config_value(
            lambda: self.config.paths.pour_point_name,
            dict_key='POUR_POINT_SHP_NAME',
        )
        if pour_point_name is None or pour_point_name == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"
        else:
            pour_point_path = pour_point_path / pour_point_name

        return pour_point_path

    def _validate_inputs(self):
        """
        Validate required input files exist.

        Raises:
            FileNotFoundError: If DEM file doesn't exist
        """
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")

    def create_directories(self):
        """Create necessary directories for delineation."""
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created interim directory: {self.interim_dir}")

    def cleanup(self):
        """
        Clean up interim files after processing.

        Only removes files if CLEANUP_INTERMEDIATE_FILES is True.
        """
        if self._get_config_value(
            lambda: self.config.domain.delineation.cleanup_intermediate_files,
            dict_key='CLEANUP_INTERMEDIATE_FILES',
        ):
            if hasattr(self, 'interim_dir') and self.interim_dir.exists():
                shutil.rmtree(self.interim_dir.parent, ignore_errors=True)
                self.logger.info(f"Cleaned up interim files: {self.interim_dir.parent}")

    @abstractmethod
    def _get_delineation_method_name(self) -> str:
        """
        Return the delineation method name for output files.

        This is used to construct output filenames like:
        - {domain_name}_riverBasins_{method}.shp
        - {domain_name}_riverNetwork_{method}.shp

        Returns:
            Method name string (e.g., 'delineate', 'lumped', 'subset_MERIT')
        """
        pass
