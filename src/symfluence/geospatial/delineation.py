# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Domain delineation module for watershed boundary extraction and spatial discretization.

Provides flexible domain delineation workflows supporting four primary definition methods
with optional subsetting: point-scale (FLUXNET), lumped watershed (conceptual),
semi-distributed (subcatchments), and distributed (grid-based). Integrates with
geofabric providers (MERIT-Basins, HydroSHEDS) and DEM-based delineation.

Architecture:
    The delineation module enables multiple domain definition workflows through the
    DomainDelineator orchestrator class:

    1. Four Domain Definition Methods:
       - POINT: Single-polygon square domain for point-scale modeling (FLUXNET, weather stations)
       - LUMPED: Single-basin watershed from pour point (traditional hydrological modeling)
       - SEMIDISTRIBUTED: Full subcatchment delineation via TauDEM (distributed models like SUMMA)
       - DISTRIBUTED: Regular grid cells with D8 flow routing (grid-based models like VIC, MESH)

    2. Subsetting Option (subset_from_geofabric):
       - When True, extracts domain from existing geofabric (MERIT-Basins, HydroSHEDS)
       - Available for lumped, semidistributed, and distributed methods
       - Lumped + subset: Dissolves subset basins to single polygon, preserves original for routing
       - Semidistributed + subset: Uses subset basins directly
       - Distributed + subset: Creates grid cells clipped to geofabric extent

    3. Grid Source Options (for distributed only):
       - 'generate': Create grid from bounding box with specified cell size
       - 'native': Match forcing data grid resolution (e.g., ERA5)

    4. Lumped-to-Distributed Routing (Special Workflow):
       - Hydrological model runs as lumped (single basin, 1 HRU)
       - Routing model runs as distributed (subcatchment-level)
       - Requires area-weighted remapping from lumped model outputs
       - Configuration: domain.delineation.routing = "river_network"

    5. Artifact Tracking:
       DelineationArtifacts dataclass tracks:
       - method: Domain definition method used (point, lumped, semidistributed, distributed)
       - river_basins_path: Subcatchment polygons shapefile
       - river_network_path: River network polyline shapefile
       - pour_point_path: Outlet point coordinate
       - original_basins_path: Original basins before aggregation (for lumped+subset)
       - metadata: Method-specific configuration (grid cell size, coastal delineation flags)

    6. Component Integration:
       - GeofabricDelineator: TauDEM-based full delineation
       - GeofabricSubsetter: Spatial intersection with existing shapefiles
       - LumpedWatershedDelineator: Single-watershed creation
       - PointDelineator: Bounding box domain
       - GridDelineator: Regular grid discretization

Configuration Parameters:
    domain.definition_method: Delineation method (point, lumped, semidistributed, distributed)
    domain.subset_from_geofabric: Extract from existing geofabric (default: False)
    domain.grid_source: Grid creation method for distributed (generate, native)
    domain.native_grid_dataset: Dataset identifier for native grid (default: era5)
    domain.delineation.routing: Routing mode for lumped domain (lumped, river_network)
    domain.delineation.geofabric_type: Source geofabric (merit_basins, hydrosheds)
    domain.delineation.delineate_coastal_watersheds: Enable coastal special handling (bool)
    domain.grid_cell_size: Grid spacing in meters (distributed method, default 1000.0)
    domain.clip_grid_to_watershed: Clip grid cells to watershed boundary (bool)
    paths.river_basins_name: Pre-existing shapefile basename (skip delineation if provided)

Output File Naming Convention:
    | Method           | Subset | Grid Source | Suffix                              |
    |------------------|--------|-------------|-------------------------------------|
    | point            | —      | —           | point                               |
    | lumped           | false  | —           | lumped                              |
    | lumped           | true   | —           | lumped_subset_{geofabric}           |
    | semidistributed  | false  | —           | semidistributed                     |
    | semidistributed  | true   | —           | semidistributed_subset_{geofabric}  |
    | distributed      | false  | generate    | distributed_{cellsize}m             |
    | distributed      | true   | —           | distributed_subset_{geofabric}      |
    | distributed      | —      | native      | distributed_native_{dataset}        |

Use Cases by Method:
    1. POINT: FLUXNET tower calibration, single weather station forcing, small-scale studies
    2. LUMPED: Traditional bucket-model calibration (GR4J, GR6J, conceptual models)
    3. LUMPED + subset: Use geofabric topology for routing with lumped hydrology
    4. SEMIDISTRIBUTED: Physically-distributed LSM calibration (SUMMA, HYPE with TauDEM)
    5. SEMIDISTRIBUTED + subset: Work within larger geofabric (MERIT-Basins, HydroSHEDS)
    6. DISTRIBUTED: Grid-based land surface models (VIC, MESH, CLM at 1 km resolution)

Example Workflows:
    Point-scale: FLUXNET site → Point domain (10×10 km) → Model forcing
    Lumped: Pour point → Single-basin domain → Simple routing (usually no routing)
    Lumped + subset: Geofabric subset → Dissolve to single basin → Keep original for routing
    Semidistributed: Pour point → TauDEM delineation → Subcatchments → SUMMA + MizuRoute
    Semidistributed + subset: Geofabric subset → Subcatchments → Model + Routing
    Distributed: Domain extent → Regular grid → D8 routing → VIC/MESH

Backwards Compatibility:
    Old method names are automatically mapped to new values:
    - 'delineate' → 'semidistributed'
    - 'distribute' → 'distributed'
    - 'subset' → 'semidistributed' (with subset_from_geofabric implied)
    - 'discretized' → 'semidistributed' (deprecated, shows warning)

References:
    - TauDEM Delineation: https://hydrology.unsw.edu.au/download/TauDEM/
    - MERIT-Basins: Yamazaki et al., 2019 (https://doi.org/10.1029/2019WR024873)
    - HydroSHEDS: Lehner et al., 2008 (https://doi.org/10.1029/2008HY04215)
    - D8 Flow Routing: O'Callaghan & Mark, 1984
    - Lumped-to-Distributed Routing: Getenet et al., 2021 (MizuRoute paper)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from symfluence.core.path_resolver import PathResolverMixin
from symfluence.geospatial.geofabric import (
    GeofabricDelineator,
    GeofabricSubsetter,
    GridDelineator,
    LumpedWatershedDelineator,
    PointDelineator,
)


@dataclass
class DelineationArtifacts:
    """Tracks delineation outputs and configuration for a domain definition workflow.

    This dataclass stores all artifacts created during domain delineation, enabling
    downstream components to locate and access shapefile outputs. Provides transparent
    audit trail of which method was used and optional metadata for method-specific config.

    Attributes:
        method (str): Domain definition method used (point, lumped, semidistributed, distributed).
            Used to identify workflow configuration and output naming scheme.

        river_basins_path (Optional[Path]): Path to subcatchment polygon shapefile (*.shp).
            - POINT: Single-polygon bounding box
            - LUMPED: Single polygon (entire watershed, possibly dissolved from subset)
            - SEMIDISTRIBUTED: TauDEM-delineated or subset subcatchments
            - DISTRIBUTED: Grid cells as polygons

        river_network_path (Optional[Path]): Path to river network polyline shapefile (*.shp).
            - POINT: Usually None (point-scale has no network)
            - LUMPED: Simplified river network (main stem only) or preserved from subset
            - SEMIDISTRIBUTED: Full stream network from TauDEM or geofabric
            - DISTRIBUTED: D8 flow paths connecting grid cells

        pour_point_path (Optional[Path]): Path to outlet point shapefile (*.shp).
            Single point geometry representing watershed outlet. Used by some models
            (e.g., HYPE) and for visualization.

        original_basins_path (Optional[Path]): Path to original basins before aggregation.
            Only populated for lumped + subset workflows where multiple basins are
            dissolved into a single polygon. The original basins are preserved for
            creating remap files for distributed routing.

        metadata (Dict[str, str]): Method-specific configuration and outputs.
            Examples:
                - 'grid_cell_size': '1000.0' (from DISTRIBUTED method)
                - 'clip_to_watershed': 'True' (from DISTRIBUTED method)
                - 'delineated_river_network_path': path (from LUMPED with routing='river_network')
                - 'delineated_river_basins_path': path (from LUMPED with routing='river_network')
                - 'geofabric_type': 'merit_basins' (from subset workflows)

    Examples:
        >>> # Point-scale domain
        >>> artifacts = DelineationArtifacts(
        ...     method='point',
        ...     river_basins_path=Path('shapefiles/river_basins/site_basins.shp'),
        ...     pour_point_path=Path('shapefiles/pour_point/site_point.shp')
        ... )

        >>> # Lumped with subset (dissolved basins, original preserved)
        >>> artifacts = DelineationArtifacts(
        ...     method='lumped',
        ...     river_basins_path=Path('shapefiles/river_basins/domain_riverBasins_lumped_subset_merit.shp'),
        ...     river_network_path=Path('shapefiles/river_network/domain_riverNetwork_lumped_subset_merit.shp'),
        ...     original_basins_path=Path('shapefiles/river_basins/domain_riverBasins_lumped_subset_merit_original.shp'),
        ...     metadata={'geofabric_type': 'merit'}
        ... )

        >>> # Lumped with distributed routing
        >>> artifacts = DelineationArtifacts(
        ...     method='lumped',
        ...     river_basins_path=Path('shapefiles/river_basins/lumped_basin.shp'),
        ...     river_network_path=Path('shapefiles/river_network/lumped_network.shp'),
        ...     metadata={
        ...         'delineated_river_basins_path': 'shapefiles/delineated_basins.shp',
        ...         'delineated_river_network_path': 'shapefiles/delineated_network.shp'
        ...     }
        ... )
    """
    method: str
    river_basins_path: Optional[Path] = None
    river_network_path: Optional[Path] = None
    pour_point_path: Optional[Path] = None
    original_basins_path: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def create_point_domain_shapefile(
    config: Dict[str, Any],
    logger: Any,
) -> Optional[Path]:
    """
    Create a square basin shapefile from bounding box coordinates for point modeling.
    """
    delineator = PointDelineator(config, logger)
    return delineator.create_point_domain_shapefile()


class DomainDelineator(PathResolverMixin):
    """Orchestrates domain delineation workflows supporting multiple modeling paradigms.

    Central delineation orchestrator enabling flexible domain definition strategies. Routes
    to appropriate delineator based on configuration (point, subset, lumped, delineate, distribute)
    and tracks all outputs via DelineationArtifacts. Integrates five specialized delineators
    through composition, supporting everything from point-scale FLUXNET sites to fully
    distributed watershed models.

    The DomainDelineator implements the Facade Pattern, providing a unified interface
    (define_domain()) that routes to component delineators while managing artifact tracking,
    configuration lookup, logging, and error handling.

    Key Responsibilities:
        1. Domain Method Routing: Select appropriate delineator based on configuration
        2. Artifact Tracking: Collect and return all shapefile paths via DelineationArtifacts
        3. Configuration Management: Lazy configuration lookup with safe defaults
        4. Lumped-to-Distributed Support: Optionally delineate subcatchments within lumped domain
        5. Path Resolution: Resolve output paths using PathResolverMixin

    Five Delineation Methods:

        POINT: Single-polygon bounding box domain for point-scale modeling
            - Use case: FLUXNET sites, weather stations, single-point forcing
            - Delineator: PointDelineator
            - Output: Single polygon (typically 10×10 km bounding box)
            - No river network (attributes uniform across domain)
            - Configuration: domain.definition_method = 'point'

        SUBSET: Extract domain from existing geofabric via spatial intersection
            - Use case: Continental studies, work within larger geofabric (MERIT-Basins, HydroSHEDS)
            - Delineator: GeofabricSubsetter
            - Output: Subcatchment polygons + river network from geofabric source
            - Supports geofabric_type selection (merit_basins, hydrosheds, etc.)
            - Configuration: domain.definition_method = 'subset'

        LUMPED: Single-basin watershed from pour point (traditional lumped modeling)
            - Use case: GR4J/GR6J calibration, lumped conceptual models, simple routing
            - Delineator: LumpedWatershedDelineator
            - Output: Single polygon representing entire watershed
            - Optional: Delineate internal subcatchments for distributed routing (lumped-to-distributed)
            - Configuration: domain.definition_method = 'lumped'
            - Special feature: domain.delineation.routing = 'river_network' enables distributed routing

        DELINEATE: Full watershed delineation using TauDEM and DEM
            - Use case: SUMMA, HYPE with distributed physics, detailed subcatchment process modeling
            - Delineator: GeofabricDelineator
            - Output: Multiple subcatchment polygons + full river network
            - Optional: Coastal delineation for basins draining to ocean
            - Configuration: domain.definition_method = 'delineate'

        DISTRIBUTE: Regular grid cells with D8 flow routing
            - Use case: Grid-based LSMs (VIC, MESH, CLM) at 1 km resolution
            - Delineator: GridDelineator
            - Output: Grid cells as polygons with D8 connectivity
            - Configurable grid spacing (default 1000 m) and watershed clipping
            - Configuration: domain.definition_method = 'distribute'

    Lumped-to-Distributed Routing Workflow (Special Case):
        Enables hybrid workflow where:
        - Hydrological model runs as lumped (single HRU, fast)
        - Routing model runs as distributed (subcatchments, accurate)
        - Requires area-weighted remapping from lumped outputs to subcatchments

        Triggered by: domain.delineation.routing = 'river_network'
        When triggered in lumped method:
            1. Primary domain: Single-basin lumped domain
            2. Delineation: Internal subcatchment delineation via GeofabricDelineator
            3. Output: Metadata includes 'delineated_river_basins_path' and 'delineated_river_network_path'
            4. Routing: MizuRoute uses delineated network and area-weighted lumped outputs

    Component Integration:
        - GeofabricDelineator: TauDEM + DEM-based delineation
        - GeofabricSubsetter: Spatial subsetting of existing shapefiles
        - LumpedWatershedDelineator: Single-basin creation (usually via subsetting)
        - PointDelineator: Bounding box geometry creation
        - GridDelineator: Regular grid discretization with D8 routing

    Artifact Management:
        define_domain() returns tuple of (result_object, DelineationArtifacts):
        - result_object: Varies by method (Path or tuple of paths)
        - DelineationArtifacts: Standardized container with river_basins_path, river_network_path,
          pour_point_path, and method-specific metadata

    Configuration Parameters:
        domain.definition_method: Delineation method (required, defaults to 'lumped')
        domain.delineation.routing: Routing mode for lumped domain (default 'lumped')
        domain.delineation.geofabric_type: Source geofabric (default from config)
        domain.delineation.delineate_coastal_watersheds: Enable coastal handling (bool)
        domain.grid_cell_size: Grid spacing in meters (distribute only, default 1000.0)
        domain.clip_grid_to_watershed: Clip grid to boundary (distribute only, default True)
        paths.river_basins_name: Pre-existing shapefile (skip delineation if provided)

    Attributes:
        config (Dict[str, Any]): Configuration object with nested structure
        logger (Any): Logger instance for recording delineation progress/errors
        reporting_manager (Optional[Any]): Optional reporting manager for diagnostics
        delineator (GeofabricDelineator): TauDEM-based delineator
        lumped_delineator (LumpedWatershedDelineator): Lumped watershed creator
        subsetter (GeofabricSubsetter): Geofabric spatial subsetter
        point_delineator (PointDelineator): Point-scale domain creator
        grid_delineator (GridDelineator): Grid-based domain creator
        project_dir (Path): From PathResolverMixin
        domain_name (str): From PathResolverMixin
        data_dir (Path): From PathResolverMixin

    Examples:
        >>> config = load_config('config.yaml')
        >>> logger = setup_logger()
        >>> delineator = DomainDelineator(config, logger)

        >>> # Point-scale delineation
        >>> config.domain.definition_method = 'point'
        >>> result, artifacts = delineator.define_domain()
        >>> # artifacts.river_basins_path: FLUXNET site bounding box
        >>> # artifacts.pour_point_path: Site coordinate

        >>> # Lumped watershed
        >>> config.domain.definition_method = 'lumped'
        >>> (network_path, basin_path), artifacts = delineator.define_domain()
        >>> # artifacts.river_basins_path: Single polygon
        >>> # artifacts.river_network_path: Main stem river

        >>> # Lumped with distributed routing
        >>> config.domain.delineation.routing = 'river_network'
        >>> (network_path, basin_path), artifacts = delineator.define_domain()
        >>> # artifacts.river_basins_path: Lumped domain (for model)
        >>> # artifacts.metadata['delineated_river_basins_path']: Subcatchments (for routing)
        >>> # Remapping: Model output (1 HRU) → MizuRoute (subcatchments)

        >>> # Distributed delineation with TauDEM
        >>> config.domain.definition_method = 'delineate'
        >>> (network_path, basin_path), artifacts = delineator.define_domain()
        >>> # artifacts.river_basins_path: Multiple subcatchment polygons
        >>> # artifacts.river_network_path: Full stream network from DEM

        >>> # Grid-based domain
        >>> config.domain.definition_method = 'distribute'
        >>> config.domain.grid_cell_size = 1000.0  # 1 km cells
        >>> (network_path, basin_path), artifacts = delineator.define_domain()
        >>> # artifacts.river_basins_path: Regular grid cells
        >>> # artifacts.metadata['grid_cell_size']: '1000.0'

    Error Handling:
        - Graceful fallback if shapefile pre-exists (skip delineation)
        - Traceback logging for delineation failures
        - Critical error for lumped-to-distributed if delineation fails (required for routing)
        - Warning for unknown delineation methods

    References:
        - TauDEM Manual: https://hydrology.unsw.edu.au/download/TauDEM/
        - D8 Steepest Descent: O'Callaghan & Mark, 1984
        - MERIT-Basins: Yamazaki et al., 2019
        - HydroSHEDS: Lehner et al., 2008
        - MizuRoute Routing: Getenet et al., 2021
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager

        # properties from mixins: self.project_dir, self.domain_name, self.data_dir are available

        self.delineator = GeofabricDelineator(self.config, self.logger, self.reporting_manager)
        self.lumped_delineator = LumpedWatershedDelineator(self.config, self.logger)
        self.subsetter = GeofabricSubsetter(self.config, self.logger)
        self.point_delineator = PointDelineator(self.config, self.logger)
        self.grid_delineator = GridDelineator(self.config, self.logger)

    def _get_pour_point_path(self) -> Optional[Path]:
        """Resolve outlet point shapefile path from configuration.

        Determines output path for outlet point shapefile. Supports configuration
        overrides via POUR_POINT_SHP_PATH/POUR_POINT_SHP_NAME or uses domain-specific
        default naming.

        Configuration Parameters:
            paths.POUR_POINT_SHP_PATH: Custom absolute path to output file (optional)
            paths.POUR_POINT_SHP_NAME: Custom filename (optional)
            Default: shapefiles/pour_point/{domain_name}_pourPoint.shp

        Returns:
            Path: Resolved path to pour point shapefile (*.shp).
                Will be created by delineation methods.
                Used by: HYPE model, visualization, routing validation.

        Example:
            >>> pour_point_path = delineator._get_pour_point_path()
            >>> # Returns: Path('project/domain_name/shapefiles/pour_point/domain_name_pourPoint.shp')
        """
        return self._get_file_path(
            path_key="POUR_POINT_SHP_PATH",
            name_key="POUR_POINT_SHP_NAME",
            default_subpath="shapefiles/pour_point",
            default_name=f"{self.domain_name}_pourPoint.shp"
        )

    def _get_subset_paths(self) -> Tuple[Path, Path]:
        """Resolve output paths for geofabric subsetting method.

        Determines output paths for river basins and river network shapefiles created
        by geofabric subsetting. Includes geofabric_type in filename to distinguish
        outputs from different geofabric sources (e.g., merit_basins vs hydrosheds).

        Configuration Parameters:
            domain.delineation.geofabric_type: Source geofabric (merit_basins, hydrosheds, etc.)
            paths.OUTPUT_BASINS_PATH: Custom absolute path for basins shapefile (optional)
            paths.OUTPUT_RIVERS_PATH: Custom absolute path for rivers shapefile (optional)

            Default paths:
                basins: shapefiles/river_basins/{domain_name}_riverBasins_subset_{geofabric_type}.shp
                rivers: shapefiles/river_network/{domain_name}_riverNetwork_subset_{geofabric_type}.shp

        Returns:
            Tuple[Path, Path]: (basins_path, rivers_path)
                - basins_path: Subset subcatchment polygon shapefile
                  * Format: {domain_name}_riverBasins_subset_{geofabric_type}.shp
                  * Example: site_riverBasins_subset_merit_basins.shp

                - rivers_path: Subset river network polyline shapefile
                  * Format: {domain_name}_riverNetwork_subset_{geofabric_type}.shp
                  * Example: site_riverNetwork_subset_merit_basins.shp

        Example:
            >>> config.domain.delineation.geofabric_type = 'merit_basins'
            >>> basins, rivers = delineator._get_subset_paths()
            >>> # basins: Path('project/domain/shapefiles/river_basins/domain_riverBasins_subset_merit_basins.shp')
            >>> # rivers: Path('project/domain/shapefiles/river_network/domain_riverNetwork_subset_merit_basins.shp')
        """
        geofabric_type = self._get_config_value(lambda: self.config.domain.delineation.geofabric_type)

        basins_path = self._get_default_path(
            config_key="OUTPUT_BASINS_PATH",
            default_subpath=f"shapefiles/river_basins/{self.domain_name}_riverBasins_subset_{geofabric_type}.shp"
        )

        rivers_path = self._get_default_path(
            config_key="OUTPUT_RIVERS_PATH",
            default_subpath=f"shapefiles/river_network/{self.domain_name}_riverNetwork_subset_{geofabric_type}.shp"
        )

        return basins_path, rivers_path

    def _delineate_lumped_domain(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Delineate lumped domain into subcatchments for distributed routing (internal helper).

        Special workflow for lumped-to-distributed routing: Creates subcatchment delineation
        within the primary lumped domain to enable area-weighted remapping from lumped model
        outputs to distributed routing network.

        This method is called internally by define_domain() only when:
            1. domain.definition_method = 'lumped'
            2. domain.delineation.routing = 'river_network'

        The workflow enables a hybrid approach:
            - Hydrological Model: Runs as lumped (1 HRU, fast, single-parameter set)
            - Routing Model: Runs as distributed (subcatchments, accurate network routing)
            - Remapping: MizuRoute uses area-weighted remapping from lumped HRU to subcatchments

        Rationale for Lumped-to-Distributed Routing:
            - Computational efficiency: Model runs as lumped (much faster than distributed)
            - Physical accuracy: Routing uses distributed network (subcatchment-level flows)
            - Parameter simplicity: Single parameter set for lumped model
            - Routing accuracy: MizuRoute uses proper flow routing on distributed network
            - Example: GR4J model (lumped) with MizuRoute (distributed) on subcatchments

        Delineation Workflow:
            1. Calls GeofabricDelineator.delineate_geofabric() to delineate subcatchments
            2. Generates TauDEM-based river network and subcatchment polygons
            3. Returns delineated network and basin paths
            4. Primary paths (lumped domain) remain unchanged in artifacts
            5. Delineated paths stored in artifacts.metadata for routing use

        Output Files:
            - delineated_river_network_path: River network polyline shapefile
              * Format: shapefiles/river_network/{domain_name}_riverNetwork_delineated.shp
              * Geometry: Polyline features representing delineated stream segments
              * Used by: MizuRoute for distributed routing

            - delineated_river_basins_path: Subcatchment polygon shapefile
              * Format: shapefiles/river_basins/{domain_name}_riverBasins_delineated.shp
              * Geometry: Polygon features representing delineated subcatchments
              * Attributes: Subcatchment IDs, area, etc.
              * Used by: MizuRoute for area-weighted remapping

        Error Handling:
            - Logs detailed error with traceback if delineation fails
            - Logs warning if delineation returns None or incomplete outputs
            - Returns (None, None) on failure (checked by caller)
            - Raises RuntimeError in caller if delineation fails (critical for routing)

        Returns:
            Tuple of (delineated_river_network_path, delineated_river_basins_path):
                - delineated_river_network_path (Path): River network polyline shapefile
                  Returns None if delineation fails

                - delineated_river_basins_path (Path): Subcatchment polygon shapefile
                  Returns None if delineation fails

        Example:
            >>> # Lumped-to-distributed routing scenario
            >>> config.domain.definition_method = 'lumped'
            >>> config.domain.delineation.routing = 'river_network'
            >>> result, artifacts = delineator.define_domain()
            >>> # define_domain() internally calls _delineate_lumped_domain()
            >>> # artifacts.river_basins_path: Lumped domain (1 polygon for GR4J)
            >>> # artifacts.metadata['delineated_river_basins_path']: Subcatchments
            >>> # Downstream:
            >>> #   1. Run GR4J model → 1 HRU output (streamflow, evap, etc.)
            >>> #   2. MizuRoute remaps 1 HRU → subcatchments (area-weighted)
            >>> #   3. MizuRoute routes on delineated_river_network

        Implementation Notes:
            - Called only if lumped domain routing='river_network' (rare special case)
            - Separate from primary lumped domain (which is single polygon)
            - Uses GeofabricDelineator (same as delineate method)
            - Returns output paths for metadata storage (not returned to user directly)
            - Critical for routing workflow (fails if cannot delineate)

        References:
            - Lumped-to-distributed routing: Getenet et al., 2021 (MizuRoute paper)
            - Area-weighted remapping: Hydrological ensemble methods literature
            - TauDEM Delineation: https://hydrology.unsw.edu.au/download/TauDEM/
        """
        try:
            self.logger.info("Delineating lumped domain into subcatchments")

            # Back up original lumped shapefiles before delineation overwrites them
            river_net_dir = self.project_dir / "shapefiles" / "river_network"
            river_bas_dir = self.project_dir / "shapefiles" / "river_basins"
            lumped_net = river_net_dir / f"{self.domain_name}_riverNetwork_lumped.shp"
            lumped_bas = river_bas_dir / f"{self.domain_name}_riverBasins_lumped.shp"

            import shutil
            lumped_net_bak = lumped_net.with_suffix('.shp.bak') if lumped_net.exists() else None
            lumped_bas_bak = lumped_bas.with_suffix('.shp.bak') if lumped_bas.exists() else None
            if lumped_net_bak:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = lumped_net.with_suffix(ext)
                    if src.exists():
                        shutil.copy2(src, src.with_suffix(ext + '.bak'))
            if lumped_bas_bak:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = lumped_bas.with_suffix(ext)
                    if src.exists():
                        shutil.copy2(src, src.with_suffix(ext + '.bak'))

            # delineate_geofabric() writes to _lumped suffix (domain_definition_method)
            delineated_network, delineated_basins = self.delineator.delineate_geofabric()

            if delineated_network and delineated_basins:
                # Rename delineated output from _lumped to _delineate suffix
                # so topology generators can find them without conflicting with the originals
                final_net = river_net_dir / f"{self.domain_name}_riverNetwork_delineate.shp"
                final_bas = river_bas_dir / f"{self.domain_name}_riverBasins_delineate.shp"

                for src_base, dst_base in [(delineated_network, final_net), (delineated_basins, final_bas)]:
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        src = src_base.with_suffix(ext)
                        if src.exists():
                            shutil.copy2(src, dst_base.with_suffix(ext))

                # Restore original lumped shapefiles
                if lumped_net_bak:
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        bak = lumped_net.with_suffix(ext + '.bak')
                        if bak.exists():
                            shutil.move(str(bak), str(lumped_net.with_suffix(ext)))
                if lumped_bas_bak:
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        bak = lumped_bas.with_suffix(ext + '.bak')
                        if bak.exists():
                            shutil.move(str(bak), str(lumped_bas.with_suffix(ext)))

                self.logger.info(f"Delineated river network: {final_net}")
                self.logger.info(f"Delineated river basins: {final_bas}")
                return final_net, final_bas
            else:
                self.logger.warning("Geofabric delineation did not produce expected outputs")
                return None, None

        except Exception as e:  # noqa: BLE001 — geospatial resilience
            self.logger.error(f"Error delineating lumped domain: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def define_domain(self) -> Tuple[Optional[Union[Path, Tuple[Path, Path]]], DelineationArtifacts]:
        """Define the spatial domain using the configured delineation method (main entry point).

        Central orchestration method that routes to the appropriate delineation workflow based on
        domain.definition_method configuration. Manages artifact tracking, error handling, and
        special workflows (lumped-to-distributed routing).

        This method implements the Strategy Pattern: selects a concrete delineation strategy
        (point, subset, lumped, delineate, distribute) at runtime based on configuration.

        Workflow:
            1. Load domain definition method from config (default: 'lumped')
            2. Create DelineationArtifacts to track outputs
            3. Check for pre-existing shapefiles (short-circuit if provided)
            4. Route to appropriate delineation method
            5. Return (result_object, artifacts) tuple

        Method Routing:
            point → PointDelineator.create_point_domain_shapefile()
                Returns: Single polygon Path
                Output paths: river_basins_path only (point-scale)

            subset → GeofabricSubsetter.subset_geofabric()
                Returns: Subset result object
                Output paths: river_basins_path, river_network_path from geofabric

            lumped → LumpedWatershedDelineator.delineate_lumped_watershed()
                Returns: Tuple of (river_network_path, river_basins_path)
                Output paths: river_basins_path, river_network_path
                Special: If routing='river_network', calls _delineate_lumped_domain()
                    to create internal subcatchments for distributed routing

            delineate → GeofabricDelineator.delineate_geofabric()
                Returns: Tuple of (river_network_path, river_basins_path)
                Output paths: river_basins_path, river_network_path
                Optional: If delineate_coastal_watersheds=True, uses coastal delineation

            distribute → GridDelineator.create_grid_domain()
                Returns: Tuple of (river_network_path, river_basins_path)
                Output paths: river_basins_path (grid cells), river_network_path (D8 paths)
                Metadata: Stores grid_cell_size and clip_to_watershed config

        Special Workflows:

            Pre-existing Shapefile (Short-circuit):
                If paths.river_basins_name is explicitly set (not 'default'):
                    → Skip all delineation
                    → Return None with method in artifacts
                    → Allows pre-computed shapefile reuse

            Lumped-to-Distributed Routing:
                Triggered by: domain.delineation.routing = 'river_network'
                Only applies to: lumped method
                Workflow:
                    1. Create primary lumped domain (single basin)
                    2. Internally call _delineate_lumped_domain() for subcatchments
                    3. Store lumped paths in artifacts (primary)
                    4. Store delineated paths in artifacts.metadata (routing)
                    5. MizuRoute uses metadata paths with area-weighted remapping

                Example configuration:
                    domain:
                      definition_method: lumped
                      delineation:
                        routing: river_network  # Enables lumped-to-distributed

                Output:
                    - artifacts.river_basins_path: Lumped domain (for hydrological model)
                    - artifacts.metadata['delineated_river_basins_path']: Subcatchments (for routing)
                    - MizuRoute remaps lumped model output (1 HRU) to subcatchments

            Coastal Delineation:
                Triggered by: domain.delineation.delineate_coastal_watersheds = True
                Only applies to: delineate method
                Replaces standard TauDEM delineation with coastal-specific workflow
                Useful for: Basins draining to ocean (avoids river network artifacts)

        Configuration Parameters:
            domain.definition_method: str (required)
                - Selects delineation method
                - Default: 'lumped'
                - Valid: 'point', 'subset', 'lumped', 'delineate', 'distribute'

            domain.delineation.routing: str (optional, lumped only)
                - Routing mode for lumped domain
                - Default: 'lumped' (no distributed routing)
                - 'river_network': Enable lumped-to-distributed routing

            domain.delineation.geofabric_type: str (subset and delineate)
                - Source geofabric for subsetting/delineation
                - Examples: 'merit_basins', 'hydrosheds'

            domain.delineation.delineate_coastal_watersheds: bool (delineate only)
                - Enable coastal delineation special handling
                - Default: False

            domain.grid_cell_size: float (distribute only)
                - Grid spacing in meters
                - Default: 1000.0 (1 km cells)

            domain.clip_grid_to_watershed: bool (distribute only)
                - Clip grid cells to watershed boundary
                - Default: True

            paths.river_basins_name: str (all methods)
                - Pre-existing shapefile basename
                - If != 'default', skips delineation entirely

        Artifact Tracking:
            DelineationArtifacts returned includes:
                - method: Delineation method used
                - river_basins_path: Subcatchment polygons (None for point)
                - river_network_path: River network polylines (None for point/some methods)
                - pour_point_path: Outlet point location
                - metadata: Method-specific config (grid size, coastal flags, delineated paths)

        Error Handling:
            - ValueError: Unknown delineation method (logged as error)
            - RuntimeError: Delineation fails for lumped-to-distributed (critical for routing)
            - Exception: Generic delineation failures logged with traceback
            - Warning: Geofabric delineation doesn't produce expected outputs

        Returns:
            Tuple of (result_object, artifacts):

                result_object: Varies by method
                    point → Path to bounding box polygon
                    subset → Subsetting result object
                    lumped → (river_network_path, river_basins_path) tuple
                    delineate → (river_network_path, river_basins_path) tuple
                    distribute → (river_network_path, river_basins_path) tuple
                    pre-existing → None (skip delineation)

                artifacts: DelineationArtifacts with:
                    - method: Delineation method (point, subset, lumped, delineate, distribute)
                    - river_basins_path: Subcatchment shapefile path
                    - river_network_path: River network shapefile path
                    - pour_point_path: Outlet point shapefile path
                    - metadata: Method-specific configuration and paths

        Examples:
            >>> # Point-scale modeling
            >>> config.domain.definition_method = 'point'
            >>> result, artifacts = delineator.define_domain()
            >>> # result: Path to bounding box shapefile
            >>> # artifacts.river_basins_path: FLUXNET site domain
            >>> # artifacts.river_network_path: None (point-scale)

            >>> # Lumped watershed (GR4J style)
            >>> config.domain.definition_method = 'lumped'
            >>> (network, basins), artifacts = delineator.define_domain()
            >>> # network: Main stem river network
            >>> # basins: Single polygon representing entire catchment
            >>> # Downstream: Use single-HRU model (e.g., GR4J) with simple routing

            >>> # Lumped-to-distributed routing
            >>> config.domain.definition_method = 'lumped'
            >>> config.domain.delineation.routing = 'river_network'
            >>> (network, basins), artifacts = delineator.define_domain()
            >>> # basins: Lumped domain (for hydrological model, 1 HRU)
            >>> # metadata['delineated_river_basins_path']: Subcatchments
            >>> # Workflow: SUMMA (1 HRU lumped) → MizuRoute (subcatchments)

            >>> # Distributed delineation (SUMMA style)
            >>> config.domain.definition_method = 'delineate'
            >>> (network, basins), artifacts = delineator.define_domain()
            >>> # basins: Multiple subcatchment polygons
            >>> # network: Full TauDEM stream network
            >>> # Downstream: Use distributed model (e.g., SUMMA) with complex routing

            >>> # Grid-based discretization (VIC style)
            >>> config.domain.definition_method = 'distribute'
            >>> config.domain.grid_cell_size = 1000.0
            >>> (network, basins), artifacts = delineator.define_domain()
            >>> # basins: 1 km × 1 km grid cells
            >>> # network: D8 flow paths connecting grid cells
            >>> # Downstream: Use grid-based model (e.g., VIC)

            >>> # Geofabric subsetting (continental scale)
            >>> config.domain.definition_method = 'subset'
            >>> config.domain.delineation.geofabric_type = 'merit_basins'
            >>> result, artifacts = delineator.define_domain()
            >>> # Extracts domain from MERIT-Basins using spatial intersection
            >>> # Preserves geofabric network topology

        Time Tracking:
            Uses time_limit context manager to track delineation execution time.
            Logs overall domain definition timing.

        References:
            - Strategy Pattern: Gang of Four design patterns
            - TauDEM: https://hydrology.unsw.edu.au/download/TauDEM/
            - MERIT-Basins: Yamazaki et al., 2019
            - MizuRoute: Getenet et al., 2021
        """
        with self.time_limit("Domain Definition"):
            # Get the domain definition method from configuration
            # Valid methods: point, lumped, semidistributed, distributed
            # Legacy methods (delineate, distribute, subset, discretized) are mapped by validator
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='semidistributed'
            )

            # Get subset and grid options
            subset_from_geofabric = self._get_config_value(
                lambda: self.config.domain.subset_from_geofabric,
                dict_key='SUBSET_FROM_GEOFABRIC',
            )
            grid_source = self._get_config_value(lambda: self.config.domain.grid_source, dict_key='GRID_SOURCE')
            geofabric_type = self._get_config_value(
                lambda: self.config.domain.delineation.geofabric_type,
                dict_key='GEOFABRIC_TYPE',
            )

            artifacts = DelineationArtifacts(method=domain_method)

            # Early exit: User provided pre-existing shapefiles, skip all delineation
            # Check if RIVER_BASINS_NAME is explicitly set (not "default")
            if self._get_config_value(lambda: self.config.paths.river_basins_name, dict_key='RIVER_BASINS_NAME') != "default":
                self.logger.info("Shapefile provided, skipping domain definition")
                return None, artifacts

            # =========================================================================
            # Method 1: POINT - Create square bounding box domain for point-scale modeling
            # Use case: FLUXNET sites, single point forcing data
            # Creates: Single polygon shapefile from bounding box coordinates
            # =========================================================================
            if domain_method == "point":
                output_path = self.point_delineator.create_point_domain_shapefile()
                artifacts.river_basins_path = output_path
                artifacts.pour_point_path = self._get_pour_point_path()
                return output_path, artifacts

            # =========================================================================
            # Method 2: LUMPED - Single-basin watershed
            # Use case: Traditional lumped hydrological modeling with single HRU
            # Variants:
            #   - Default: Delineate from pour point
            #   - subset_from_geofabric=True: Subset from geofabric, dissolve to single polygon
            # =========================================================================
            if domain_method == "lumped":
                if subset_from_geofabric:
                    # Lumped + subset: Extract from geofabric, dissolve to single polygon
                    self.logger.info(f"Creating lumped domain from {geofabric_type} geofabric subset")

                    # Step 1: Subset the geofabric to get basins and rivers
                    subset_basins, subset_rivers = self.subsetter.subset_geofabric()

                    if subset_basins is None or len(subset_basins) == 0:
                        self.logger.error("Geofabric subsetting returned no basins")
                        return None, artifacts

                    # Step 2: Get output paths using new naming convention
                    method_suffix = self._get_method_suffix()
                    basins_path = (
                        self.project_dir / "shapefiles" / "river_basins" /
                        f"{self.domain_name}_riverBasins_{method_suffix}.shp"
                    )
                    original_basins_path = (
                        self.project_dir / "shapefiles" / "river_basins" /
                        f"{self.domain_name}_riverBasins_{method_suffix}_original.shp"
                    )
                    rivers_path = (
                        self.project_dir / "shapefiles" / "river_network" /
                        f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
                    )

                    # Ensure directories exist
                    basins_path.parent.mkdir(parents=True, exist_ok=True)
                    rivers_path.parent.mkdir(parents=True, exist_ok=True)

                    # Step 3: Dissolve basins to single polygon, preserving original
                    dissolved_basins = self.subsetter.aggregate_to_lumped(
                        subset_basins, original_basins_path
                    )
                    dissolved_basins.to_file(basins_path)
                    self.logger.info(f"Saved dissolved lumped basin to: {basins_path}")

                    # Step 4: Save rivers (preserved as-is for routing)
                    if subset_rivers is not None:
                        subset_rivers.to_file(rivers_path)
                        artifacts.river_network_path = rivers_path

                    artifacts.river_basins_path = basins_path
                    artifacts.original_basins_path = original_basins_path
                    artifacts.pour_point_path = self._get_pour_point_path()
                    artifacts.metadata['geofabric_type'] = geofabric_type

                    return (rivers_path, basins_path), artifacts  # type: ignore[return-value]

                else:
                    # Default lumped: Delineate from pour point
                    river_network_path, river_basins_path = (
                        self.lumped_delineator.delineate_lumped_watershed()
                    )
                    artifacts.river_network_path = river_network_path
                    artifacts.river_basins_path = river_basins_path
                    artifacts.pour_point_path = self._get_pour_point_path()

                    # Check if we need delineated catchments for distributed routing
                    routing_delineation = self._get_config_value(
                        lambda: self.config.domain.delineation.routing,
                        dict_key='ROUTING_DELINEATION',
                    )
                    if routing_delineation == "river_network":
                        self.logger.info("Creating delineated catchments for lumped-to-distributed routing")
                        delineated_river_network, delineated_river_basins = self._delineate_lumped_domain()

                        if delineated_river_network and delineated_river_basins:
                            artifacts.metadata['delineated_river_network_path'] = str(delineated_river_network)
                            artifacts.metadata['delineated_river_basins_path'] = str(delineated_river_basins)
                            self.logger.info("Delineated catchments created successfully")
                        else:
                            self.logger.error("Failed to create delineated catchments for lumped domain")
                            raise RuntimeError("Delineation of lumped domain failed")

                    return (river_network_path, river_basins_path), artifacts  # type: ignore[return-value]

            # =========================================================================
            # Method 3: SEMIDISTRIBUTED - Multiple subcatchment polygons
            # Use case: Distributed hydrological modeling with subcatchment-level HRUs
            # Variants:
            #   - Default: TauDEM delineation from DEM
            #   - subset_from_geofabric=True: Subset from existing geofabric
            # =========================================================================
            if domain_method == "semidistributed":
                if subset_from_geofabric:
                    # Semidistributed + subset: Extract subcatchments from geofabric
                    self.logger.info(f"Creating semidistributed domain from {geofabric_type} geofabric subset")
                    result = self.subsetter.subset_geofabric()
                    basins_path, rivers_path = self._get_subset_paths()
                    artifacts.river_basins_path = basins_path
                    artifacts.river_network_path = rivers_path
                    artifacts.pour_point_path = self._get_pour_point_path()
                    artifacts.metadata['geofabric_type'] = geofabric_type
                    return result, artifacts  # type: ignore[return-value]

                else:
                    # Default semidistributed: TauDEM delineation
                    river_network_path, river_basins_path = self.delineator.delineate_geofabric()

                    # Optional coastal delineation for basins draining to ocean
                    if self._get_config_value(
                        lambda: self.config.domain.delineation.delineate_coastal_watersheds,
                        dict_key='DELINEATE_COASTAL_WATERSHEDS',
                    ):
                        coastal_result = self.delineator.delineate_coastal()
                        if coastal_result and all(coastal_result):
                            river_network_path, river_basins_path = coastal_result
                            self.logger.info(f"Coastal delineation completed: {coastal_result}")

                    artifacts.river_network_path = river_network_path
                    artifacts.river_basins_path = river_basins_path
                    artifacts.pour_point_path = self._get_pour_point_path()
                    return (river_network_path, river_basins_path), artifacts  # type: ignore[return-value]

            # =========================================================================
            # Method 4: DISTRIBUTED - Regular grid cells with D8 flow routing
            # Use case: Grid-based distributed modeling (VIC, MESH, CLM)
            # Variants:
            #   - grid_source='generate': Create grid from bounding box
            #   - grid_source='native': Match forcing data resolution
            #   - subset_from_geofabric=True: Clip grid to geofabric extent
            # =========================================================================
            if domain_method == "distributed":
                # Store grid configuration in metadata
                artifacts.metadata['grid_cell_size'] = str(
                    self._get_config_value(lambda: self.config.domain.grid_cell_size, dict_key='GRID_CELL_SIZE')
                )
                artifacts.metadata['clip_to_watershed'] = str(
                    self._get_config_value(
                        lambda: self.config.domain.clip_grid_to_watershed,
                        dict_key='CLIP_GRID_TO_WATERSHED',
                    )
                )
                artifacts.metadata['grid_source'] = grid_source

                if subset_from_geofabric:
                    # Distributed + subset: Create grid clipped to geofabric extent
                    self.logger.info(f"Creating distributed grid domain from {geofabric_type} geofabric subset")
                    artifacts.metadata['geofabric_type'] = geofabric_type

                    # First subset the geofabric
                    subset_basins, subset_rivers = self.subsetter.subset_geofabric()
                    if subset_basins is None:
                        self.logger.error("Geofabric subsetting returned no basins")
                        return None, artifacts

                    # Then create grid clipped to subset extent
                    grid_gdf = self.grid_delineator._subset_grid_from_geofabric(subset_basins)
                    if grid_gdf is None:
                        self.logger.error("Failed to create grid from geofabric subset")
                        return None, artifacts

                    # Save the grid
                    method_suffix = self._get_method_suffix()
                    river_basins_path = (
                        self.project_dir / "shapefiles" / "river_basins" /
                        f"{self.domain_name}_riverBasins_{method_suffix}.shp"
                    )
                    river_network_path = (
                        self.project_dir / "shapefiles" / "river_network" /
                        f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
                    )
                    river_basins_path.parent.mkdir(parents=True, exist_ok=True)
                    river_network_path.parent.mkdir(parents=True, exist_ok=True)

                    grid_gdf.to_file(river_basins_path)

                    # Create river network from grid
                    network_gdf = self.grid_delineator._create_river_network_from_grid(grid_gdf)
                    network_gdf.to_file(river_network_path)

                    artifacts.river_basins_path = river_basins_path
                    artifacts.river_network_path = river_network_path
                    artifacts.pour_point_path = self._get_pour_point_path()

                    return (river_network_path, river_basins_path), artifacts  # type: ignore[return-value]

                elif grid_source == 'native':
                    # Distributed + native: Match forcing data grid resolution
                    self.logger.info("Creating distributed grid domain from native forcing resolution")
                    native_dataset = self._get_config_value(
                        lambda: self.config.domain.native_grid_dataset,
                        dict_key='NATIVE_GRID_DATASET',
                    )
                    artifacts.metadata['native_grid_dataset'] = native_dataset

                    grid_gdf = self.grid_delineator._create_native_grid()
                    if grid_gdf is None:
                        self.logger.error(f"Native grid creation not yet implemented for {native_dataset}")
                        return None, artifacts

                    # If native grid is implemented, save it
                    method_suffix = self._get_method_suffix()
                    river_basins_path = (
                        self.project_dir / "shapefiles" / "river_basins" /
                        f"{self.domain_name}_riverBasins_{method_suffix}.shp"
                    )
                    grid_gdf.to_file(river_basins_path)
                    artifacts.river_basins_path = river_basins_path
                    artifacts.pour_point_path = self._get_pour_point_path()

                    return (None, river_basins_path), artifacts  # type: ignore[return-value]

                else:
                    # Default distributed: Generate grid from bounding box
                    river_network_path, river_basins_path = self.grid_delineator.create_grid_domain()
                    artifacts.river_network_path = river_network_path
                    artifacts.river_basins_path = river_basins_path
                    artifacts.pour_point_path = self._get_pour_point_path()

                    return (river_network_path, river_basins_path), artifacts  # type: ignore[return-value]

            # Fallback: Unknown domain definition method - configuration error
            # Valid methods: point, lumped, semidistributed, distributed
            self.logger.error(f"Unknown domain definition method: {domain_method}")
            return None, artifacts
