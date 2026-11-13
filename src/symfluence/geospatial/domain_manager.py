# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Domain management facade for SYMFLUENCE geospatial operations.

Coordinates domain definition, delineation, and discretization workflows
with integrated visualization and artifact tracking.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from symfluence.core.exceptions import GeospatialError, symfluence_error_handler
from symfluence.core.mixins import ConfigurableMixin
from symfluence.geospatial.delineation import (
    DelineationArtifacts,
    DomainDelineator,
    create_point_domain_shapefile,
)
from symfluence.geospatial.discretization import DiscretizationArtifacts, DomainDiscretizationRunner

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class DomainManager(ConfigurableMixin):
    """
    Orchestrates all geospatial domain operations for hydrological modeling setup.

    This manager coordinates domain definition (delineation), spatial discretization,
    and artifact tracking for hydrological modeling workflows. It provides a unified
    interface for creating HRU (Hydrologic Response Unit) configurations from various
    spatial data sources and discretization strategies.

    Architecture:
        Facade Pattern - Coordinates specialized geospatial services:
        - **DomainDelineator**: Watershed boundary extraction and network topology
        - **DomainDiscretizationRunner**: HRU creation via spatial disaggregation
        - **Artifact Tracking**: Maintains references to all created shapefiles
        - **Visualization Integration**: Optional reporting for QA/QC

    Domain Definition Methods:
        **point**:
            - Creates square bounding box domain from coordinates
            - Use case: FLUXNET sites, point-scale modeling
            - Output: Single polygon shapefile

        **lumped**:
            - Single-basin watershed delineation from pour point
            - Use case: Traditional lumped hydrological modeling
            - Output: Single watershed polygon + optional delineated routing network
            - Special: Supports lumped-to-distributed routing workflow
            - With subset_from_geofabric=True: Dissolves geofabric basins to single polygon

        **semidistributed**:
            - Full TauDEM-based watershed delineation from DEM
            - Use case: Detailed distributed modeling with subcatchments
            - Output: River network + subcatchment polygons
            - Optional: Coastal watershed handling
            - With subset_from_geofabric=True: Extracts from existing geofabric

        **distributed**:
            - Regular grid domain with D8 flow direction
            - Use case: Grid-based land surface models (VIC, MESH, CLM)
            - Output: Grid cells as both HRUs and routing segments
            - grid_source='generate': Create grid from bounding box
            - grid_source='native': Match forcing data resolution

    Discretization Strategies:
        **lumped**:
            - Single HRU representing entire basin
            - No spatial disaggregation

        **elevation**:
            - Elevation bands (e.g., 100m intervals)
            - Use case: Snow modeling, orographic effects

        **landclass**:
            - Land cover types (forest, urban, agriculture, etc.)
            - Use case: Land surface heterogeneity

        **soilclass**:
            - Soil classification types
            - Use case: Infiltration and runoff variability

        **aspect**:
            - Slope aspect classes (N, NE, E, SE, S, SW, W, NW)
            - Use case: Solar radiation and snow redistribution

        **radiation**:
            - Potential solar radiation classes
            - Use case: Energy balance modeling

        **combined**:
            - Multiple attributes combined (e.g., elevation × landclass)
            - Use case: Capturing complex spatial heterogeneity
            - Handles attribute interactions and MultiPolygons

    Workflow Sequence:
        1. **define_domain()**: Create/extract watershed boundaries
           → Produces DelineationArtifacts (river basins, network, pour point)

        2. **discretize_domain()**: Subdivide into HRUs
           → Produces DiscretizationArtifacts (HRU shapefile, attributes)

        3. **Visualization** (optional): Spatial QA/QC plots
           → Generated via reporting_manager if available

    Artifact Tracking:
        DelineationArtifacts:
            - method: Domain definition method used
            - river_basins_path: Path to basin shapefile
            - river_network_path: Path to river network shapefile
            - pour_point_path: Path to pour point shapefile
            - metadata: Additional delineation metadata

        DiscretizationArtifacts:
            - method: Discretization method used
            - hru_shapefile_path: Path to HRU shapefile
            - attributes: HRU attributes DataFrame
            - statistics: Discretization statistics (HRU count, min/max areas)

    Configuration Dependencies:
        Domain Definition:
            - DOMAIN_DEFINITION_METHOD: point/lumped/semidistributed/distributed
            - DOMAIN_NAME: Basin identifier
            - SUBSET_FROM_GEOFABRIC: Extract from existing geofabric (default: False)
            - GRID_SOURCE (distributed): 'generate' or 'native'
            - NATIVE_GRID_DATASET (distributed + native): Dataset identifier (default: 'era5')
            - POUR_POINT_SHP_PATH (lumped/semidistributed): Pour point location
            - RIVER_NETWORK_SHP_PATH (subset): Existing river network
            - DOMAIN_BOUNDING_BOX: Bbox coordinates
            - GRID_CELL_SIZE (distributed): Grid spacing in meters

        Discretization:
            - SUB_GRID_DISCRETIZATION: Discretization method
            - DEM_PATH: Elevation data (for elevation/aspect/radiation)
            - LAND_CLASS_PATH: Land cover data (for landclass)
            - SOIL_CLASS_PATH: Soil data (for soilclass)
            - ELEVATION_BAND_SIZE: Band interval in meters (default: 100)

        Delineation (TauDEM):
            - DELINEATE_COASTAL_WATERSHEDS: Coastal handling (True/False)
            - ROUTING_DELINEATION: Routing network strategy

    Output Files:
        Shapefiles created in ``project_dir/shapefiles/``.

        Delineation outputs: river_basins, river_network, pour_point shapefiles.

        Discretization outputs: catchment HRU shapefiles.

        Example structure::

            shapefiles/
            ├── river_basins/
            │   └── bow_river_riverBasins_lumped.shp
            ├── river_network/
            │   └── bow_river_riverNetwork_lumped.shp
            ├── pour_point/
            │   └── bow_river_pourPoint.shp
            └── catchment/
                └── bow_river_HRUs_elevation.shp

    Special Workflows:
        Lumped-to-Distributed Routing:
            1. Define lumped domain (single watershed polygon)
            2. Internally delineate subcatchments within lumped domain
            3. Create area-weighted remapping (lumped HRU to distributed routing)
            4. Enables distributed routing with lumped hydrology

        Coastal Watershed Delineation:
            - Special handling for basins draining to ocean
            - Avoids river network artifacts at coastline
            - Uses modified TauDEM workflow

        Grid-Based Distributed:
            - Creates regular grid cells
            - Assigns D8 flow direction from DEM
            - Detects and fixes routing cycles
            - Each cell is both HRU and routing segment

    Visualization Integration:
        If reporting_manager available:
        - Delineation: Watershed boundary maps, river network plots
        - Discretization: HRU spatial distribution, attribute histograms
        - QA/QC: Identifies potential issues (small HRUs, disconnected polygons)

    Error Handling:
        - Validates configuration before execution
        - Raises descriptive errors for missing required shapefiles
        - Logs warnings for non-critical issues
        - Provides context for TauDEM failures

    Example Workflow:
        >>> from symfluence.geospatial.domain_manager import DomainManager
        >>> config = SymfluenceConfig.from_file('config.yaml')
        >>> logger = setup_logger()
        >>> reporting = ReportingManager(config, logger)
        >>>
        >>> # Initialize manager
        >>> domain_mgr = DomainManager(config, logger, reporting)
        >>>
        >>> # Define watershed boundaries
        >>> domain_mgr.define_domain()
        >>> print(domain_mgr.delineation_artifacts.river_basins_path)
        # ./shapefiles/river_basins/bow_river_riverBasins_lumped.shp
        >>>
        >>> # Discretize into elevation bands
        >>> domain_mgr.discretize_domain()
        >>> print(domain_mgr.discretization_artifacts.statistics)
        # {'hru_count': 8, 'min_area_km2': 120.5, 'max_area_km2': 450.2}

    Performance Considerations:
        - Delineation: ~1-30 minutes (depends on DEM resolution, TauDEM)
        - Discretization: ~10 seconds - 5 minutes (depends on attribute resolution)
        - Grid generation: ~1-10 minutes (depends on grid cell count)
        - Memory: Peak during raster operations (~2-8 GB for high-res DEMs)

    Notes:
        - DomainDelineator initialized eagerly
        - DomainDiscretizationRunner initialized lazily when needed
        - Artifacts tracked for downstream workflows (preprocessing, modeling)
        - Reporting integration provides visual validation
        - Supports both simple (lumped) and complex (combined attributes) setups

    See Also:
        - geospatial.delineation.DomainDelineator: Watershed delineation
        - geospatial.discretization.DomainDiscretizationRunner: HRU creation
        - geospatial.discretization.core.DomainDiscretizer: Discretization engine
        - geospatial.geofabric: Geofabric delineation backends
    """

    def __init__(self, config: 'SymfluenceConfig', logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the Domain Manager.

        Args:
            config: SymfluenceConfig instance
            logger: Logger instance
            reporting_manager: ReportingManager instance

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
        self.reporting_manager = reporting_manager

        # Initialize domain workflows
        self.domain_delineator = DomainDelineator(config, self.logger, self.reporting_manager)
        self.domain_discretizer = None  # Initialized when needed
        self.delineation_artifacts: Optional[DelineationArtifacts] = None
        self.discretization_artifacts: Optional[DiscretizationArtifacts] = None

        # Create point domain shapefile if method is 'point'
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method
        )
        if domain_method == 'point':
            self.create_point_domain_shapefile()

    def create_point_domain_shapefile(self) -> Optional[Path]:
        """
        Create a square basin shapefile from bounding box coordinates for point modelling.

        This method creates a rectangular polygon from the BOUNDING_BOX_COORDS and saves it
        as a shapefile for point-based modelling approaches.

        Returns:
            Path to the created shapefile or None if failed
        """
        return create_point_domain_shapefile(self.config, self.logger)

    def define_domain(
        self,
    ) -> Tuple[Optional[Union[Path, Tuple[Path, Path]]], DelineationArtifacts]:
        """
        Define the domain using the configured method.

        Returns:
            Tuple of the domain result and delineation artifacts
        """
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method
        )
        self.logger.debug(f"Domain definition workflow starting with: {domain_method}")

        result, artifacts = self.domain_delineator.define_domain()
        self.delineation_artifacts = artifacts

        if not result:
            # The delineator returned no domain (e.g. TauDEM watershed
            # delineation produced nothing). Fail loudly here instead of
            # silently marking define_domain complete and letting the next
            # step (discretize_domain) die with a confusing "no such file"
            # for the river-basins shapefile that was never created.
            raise GeospatialError(
                f"Domain definition failed for method '{domain_method}': the "
                f"delineator produced no domain boundary. For watershed, lumped, "
                f"or semi-distributed domains this usually means TauDEM is not "
                f"installed or failed to run — install it with "
                f"'symfluence binary install taudem' (then 'symfluence binary "
                f"doctor') and review the delineation errors logged above."
            )

        self.logger.info(f"Domain definition completed using method: {domain_method}")

        # Generate diagnostic plots if enabled
        if self.reporting_manager and artifacts.river_basins_path:
            with symfluence_error_handler(
                "generating domain definition diagnostics",
                self.logger,
                reraise=False,
                error_type=GeospatialError
            ):
                import geopandas as gpd
                basin_gdf = gpd.read_file(artifacts.river_basins_path)
                dem_path = self.project_attributes_dir / 'elevation' / 'dem' / f"{self.domain_name}_elv.tif"
                self.reporting_manager.diagnostic_domain_definition(
                    basin_gdf=basin_gdf,
                    dem_path=dem_path if dem_path.exists() else None
                )

        self.logger.debug("Domain definition workflow finished")

        return result, artifacts


    def discretize_domain(
        self,
    ) -> Tuple[Optional[Union[Path, dict]], DiscretizationArtifacts]:
        """
        Discretize the domain into HRUs or GRUs.

        Returns:
            Tuple of HRU shapefile(s) and discretization artifacts
        """
        with symfluence_error_handler(
            "domain discretization",
            self.logger,
            error_type=GeospatialError
        ):
            discretization_method = self._get_config_value(
                lambda: self.config.domain.discretization
            )
            self.logger.debug(f"Discretizing domain using method: {discretization_method}")

            # Initialize discretizer if not already done
            if self.domain_discretizer is None:
                self.domain_discretizer = DomainDiscretizationRunner(self.config, self.logger)

            # Perform discretization
            hru_shapefile, artifacts = self.domain_discretizer.discretize_domain()
            self.discretization_artifacts = artifacts

            # Visualize the discretized domain
            self.visualize_discretized_domain()

            # Generate diagnostic plots if enabled
            if self.reporting_manager and artifacts.hru_paths:
                with symfluence_error_handler(
                    "generating discretization diagnostics",
                    self.logger,
                    reraise=False,
                    error_type=GeospatialError
                ):
                    import geopandas as gpd
                    # hru_paths can be Path or Dict[str, Path]
                    hru_path = artifacts.hru_paths
                    if isinstance(hru_path, dict):
                        # Take first path from dict
                        hru_path = next(iter(hru_path.values()))
                    hru_gdf = gpd.read_file(hru_path)
                    self.reporting_manager.diagnostic_discretization(
                        hru_gdf=hru_gdf,
                        method=discretization_method
                    )

            return hru_shapefile, artifacts

    def visualize_domain(self) -> Optional[Path]:
        """
        Create visualization of the domain.

        Returns:
            Path to the created plot or None if failed
        """
        if self.reporting_manager:
            return self.reporting_manager.visualize_domain()
        return None

    def visualize_discretized_domain(self) -> Optional[Path]:
        """
        Create visualization of the discretized domain.

        Returns:
            Path to the created plot or None if failed
        """
        if self.reporting_manager:
            discretization_method = self._get_config_value(
                lambda: self.config.domain.discretization
            )
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method
            )
            if domain_method != 'point':
                return self.reporting_manager.visualize_discretized_domain(discretization_method)
            else:
                self.logger.info('Point scale model, not creating visualisation')
                return None
        return None

    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get information about the current domain configuration.

        Returns:
            Dictionary containing domain information
        """
        info = {
            'domain_name': self.domain_name,
            'domain_method': self._get_config_value(
                lambda: self.config.domain.definition_method
            ),
            'spatial_mode': self._get_config_value(
                lambda: self.config.domain.definition_method
            ),
            'discretization_method': self._get_config_value(
                lambda: self.config.domain.discretization
            ),
            'pour_point_coords': self._get_config_value(
                lambda: self.config.domain.pour_point_coords
            ),
            'bounding_box': self._get_config_value(
                lambda: self.config.domain.bounding_box_coords
            ),
            'project_dir': str(self.project_dir),
        }

        # Add shapefile paths if they exist
        river_basins_path = self.project_dir / "shapefiles" / "river_basins"
        catchment_path = self.project_dir / "shapefiles" / "catchment"

        if river_basins_path.exists():
            info['river_basins_path'] = str(river_basins_path)

        if catchment_path.exists():
            info['catchment_path'] = str(catchment_path)

        return info

    def validate_domain_configuration(self) -> bool:
        """
        Validate domain configuration settings.

        .. deprecated::
            Use :meth:`validate_readiness` instead. Config-level checks
            (required keys, definition method, bounding box format) are now
            handled by Pydantic validators at config construction time.

        Returns:
            True if configuration is valid, False otherwise
        """
        import warnings
        warnings.warn(
            "validate_domain_configuration() is deprecated, use validate_readiness()",
            DeprecationWarning,
            stacklevel=2,
        )
        readiness = self.validate_readiness()
        return all(readiness.values()) if readiness else True

    def validate_readiness(self) -> Dict[str, bool]:
        """
        Validate that this manager is ready for execution.

        Checks runtime prerequisites that Pydantic cannot verify at
        config construction time: subset geofabric config and grid
        source for distributed method.

        Returns:
            Dict mapping check names to pass/fail booleans.
        """
        results = {}

        # Validate subset configuration
        subset_from_geofabric = self._get_config_value(
            lambda: self.config.domain.subset_from_geofabric,
            dict_key='SUBSET_FROM_GEOFABRIC',
        )
        if subset_from_geofabric:
            geofabric_type = self._get_config_value(
                lambda: self.config.domain.delineation.geofabric_type,
                dict_key='GEOFABRIC_TYPE',
            )
            ok = geofabric_type not in ('na', '', None)
            if not ok:
                self.logger.error(
                    "subset_from_geofabric=True requires GEOFABRIC_TYPE to be set. "
                    "Valid values: merit, tdx, nws, hydrosheds, etc."
                )
            results['subset_config'] = ok

        # Validate grid_source for distributed method
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method
        )
        if domain_method == 'distributed':
            grid_source = self._get_config_value(lambda: self.config.domain.grid_source, dict_key='GRID_SOURCE')
            valid_grid_sources = ['generate', 'native']
            ok = grid_source in valid_grid_sources
            if not ok:
                self.logger.error(
                    f"Invalid grid_source: {grid_source}. Must be one of {valid_grid_sources}"
                )
            results['grid_source'] = ok

        return results
