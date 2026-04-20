# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Project management for SYMFLUENCE hydrological modeling setups.

Handles project directory structure creation, pour point generation,
and project metadata management for hydrological model domains.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import geopandas as gpd
from shapely.geometry import Point

from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class ProjectManager(ConfigurableMixin):
    """
    Manages project-level operations including directory structure and initialization.

    The ProjectManager is responsible for creating and managing the project directory
    structure, handling pour point creation, and maintaining project metadata. It serves
    as the foundation for all other SYMFLUENCE components by establishing the physical
    file organization that the workflow depends on.

    Key responsibilities:
    - Creating the project directory structure
    - Generating pour point shapefiles from coordinates
    - Validating project structure integrity
    - Providing project metadata to other components

    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing project settings
        logger (logging.Logger): Logger instance for recording operations
    """

    def __init__(self, config: 'SymfluenceConfig', logger: logging.Logger):
        """
        Initialize the ProjectManager.

        Args:
            config: SymfluenceConfig instance
            logger: Logger instance

        Raises:
            TypeError: If config is not a SymfluenceConfig instance
        """
        # Import here to avoid circular imports at module level
        from symfluence.core.config.models import SymfluenceConfig

        if not isinstance(config, SymfluenceConfig):
            raise TypeError(
                f"config must be SymfluenceConfig, got {type(config).__name__}. "
                "Use SymfluenceConfig.from_file() to load configuration."
            )

        # Set config via the ConfigMixin property
        self._config = config
        self.logger = logger

    def setup_project(self) -> Path:
        """
        Set up the project directory structure.

        Creates the main project directory and all required subdirectories.
        Project layout (canonical, post-2026):

            {project_dir}/
              shapefiles/{pour_point, catchment, river_network, river_basins}/
              data/
                attributes/      <- DEM, soil, landclass, etc.
                forcing/         <- raw + merged + basin-averaged forcing
                observations/    <- streamflow, snotel, etc.
                model_ready/     <- model-agnostic store

        The ``data/`` prefix is created up-front so that
        ``resolve_data_subdir`` consistently resolves to the new layout
        for fresh projects. Without this, ``setup_project`` used to
        create the legacy ``attributes/`` directory directly, which made
        ``resolve_data_subdir`` pick the legacy path on subsequent
        reads. Some downstream callers (e.g. TauDEM in
        ``define_domain``) construct the path from string templates
        anchored at ``data/attributes/...`` and then fail to find the
        DEM that was written into the legacy ``attributes/...`` tree.

        Legacy projects with a pre-existing ``attributes/`` directory
        continue to work via the backward-compat branch in
        ``resolve_data_subdir`` — this change only affects fresh
        ``setup_project`` runs.

        Returns:
            Path: Path to the created project directory

        Raises:
            OSError: If directory creation fails due to permission or disk space issues
        """
        self.logger.info(f"Setting up project for domain: {self.domain_name}")

        # Create main project directory
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Top-level shapefile structure (unchanged — shapefiles live
        # outside data/ by convention since they are domain-defining
        # artefacts produced before any data acquisition).
        shapefile_subdirs = ['pour_point', 'catchment', 'river_network', 'river_basins']
        shapefiles_path = self.project_dir / 'shapefiles'
        shapefiles_path.mkdir(parents=True, exist_ok=True)
        for subdir in shapefile_subdirs:
            (shapefiles_path / subdir).mkdir(parents=True, exist_ok=True)

        # Canonical data subtree. We only create ``data/{subdir}`` when
        # no legacy ``{subdir}`` already exists at the project root.
        # Pre-staged test fixtures and legacy domains keep working
        # because resolve_data_subdir still finds their legacy paths.
        # Fresh projects get the canonical layout because neither path
        # exists yet, so we create ``data/{subdir}`` and resolve_data_subdir
        # picks it on every read.
        data_subdirs = ['attributes', 'forcing', 'observations', 'model_ready']
        data_path = self.project_dir / 'data'
        data_path.mkdir(parents=True, exist_ok=True)
        for subdir in data_subdirs:
            legacy_path = self.project_dir / subdir
            if legacy_path.exists():
                self.logger.debug(
                    f"Legacy '{subdir}/' directory present at {legacy_path}; "
                    f"skipping creation of data/{subdir} so resolve_data_subdir "
                    f"keeps using the legacy path."
                )
                continue
            (data_path / subdir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Project directory created at: {self.project_dir}")
        return self.project_dir

    def create_pour_point(self) -> Optional[Path]:
        """
        Create pour point shapefile from coordinates if specified.

        If pour point coordinates are specified in the configuration, creates
        a GeoDataFrame with a single point geometry and saves it as a shapefile
        at the appropriate location. If 'default' is specified, assumes a
        user-provided pour point shapefile exists.

        Returns:
            Optional[Path]: Path to the created pour point shapefile if successful,
                          None if using a user-provided shapefile or if creation fails

        Raises:
            ValueError: If the pour point coordinates are in an invalid format
            Exception: For other errors during shapefile creation
        """
        # Check if using user-provided shapefile
        pour_point_coords = self._get_config_value(
            lambda: self.config.domain.pour_point_coords,
            'default'
        )
        if str(pour_point_coords).lower() == 'default':
            self.logger.info("Using user-provided pour point shapefile")
            return None

        try:
            # Parse coordinates
            lat, lon = map(float, str(pour_point_coords).split('/'))
            point = Point(lon, lat)  # Note: Point takes (lon, lat) order

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")

            # Determine output path
            output_path = self.project_dir / "shapefiles" / "pour_point"
            shp_path = self._get_config_value(
                lambda: self.config.paths.pour_point_shp_path,
                'default'
            )
            if shp_path != 'default' and shp_path is not None:
                output_path = Path(shp_path)

            # Determine shapefile name
            pour_point_shp_name = f"{self.domain_name}_pourPoint.shp"
            shp_name = self._get_config_value(
                lambda: self.config.paths.pour_point_shp_name,
                'default'
            )
            if shp_name != 'default' and shp_name is not None:
                pour_point_shp_name = shp_name

            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / pour_point_shp_name

            # Save shapefile
            gdf.to_file(output_file)
            self.logger.info(f"Pour point shapefile created successfully: {output_file}")
            return output_file

        except ValueError:
            self.logger.error("Invalid pour point coordinates format. Expected 'lat/lon'.")
            return None
        except Exception as e:  # noqa: BLE001 — configuration resilience
            self.logger.error(f"Error creating pour point shapefile: {str(e)}")
            return None

    def get_project_info(self) -> Dict[str, Any]:
        """
        Get information about the project configuration.

        Collects key project metadata into a dictionary for reporting,
        logging, or providing status information to other components.

        The returned information includes:
        - Domain name
        - Experiment ID
        - Project directory path
        - Data directory path
        - Pour point coordinates

        Returns:
            Dict[str, Any]: Dictionary containing project information
        """
        info = {
            'domain_name': self.domain_name,
            'experiment_id': self.experiment_id,
            'project_dir': str(self.project_dir),
            'data_dir': str(self.data_dir),
            'pour_point_coords': self._get_config_value(
                lambda: self.config.domain.pour_point_coords
            )
        }

        return info
