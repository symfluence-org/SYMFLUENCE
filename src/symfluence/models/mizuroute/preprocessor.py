# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MizuRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the mizuRoute routing model.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional

from symfluence.core.geometry_utils import GeospatialUtilsMixin
from symfluence.core.modeling.base import BaseModelPreProcessor
from symfluence.core.registries import R
from symfluence.models.mizuroute.control_writer import ControlFileWriter
from symfluence.models.mizuroute.mixins import MizuRouteConfigMixin


@R.preprocessors.add('MIZUROUTE')
class MizuRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin, MizuRouteConfigMixin):  # type: ignore[misc]
    """
    Spatial preprocessor and configuration generator for the mizuRoute river routing model.

    This preprocessor handles all spatial setup tasks required to run mizuRoute, including
    network topology file creation, remapping file generation, and control file writing.
    It supports multiple domain discretization strategies (lumped, semi-distributed,
    distributed, grid-based) and integrates with various hydrological models as runoff
    sources (SUMMA, FUSE, GR, NextGen).

    Supported Domain Types:
        Lumped:
            - Single HRU draining to river network
            - Optional distributed routing via delineated subcatchments
            - Area-weighted remapping for lumped-to-distributed conversion

        Semi-distributed:
            - Multiple HRUs per GRU routing at GRU level
            - Reads SUMMA attributes file to determine HRU/GRU structure
            - GRU-aggregated runoff routing

        Distributed:
            - Elevation bands or attribute-based discretization
            - Routing at finest spatial resolution
            - Optional remapping between catchment scales

        Grid-based:
            - Regular grid cells with D8 flow direction
            - Each cell is both HRU and routing segment
            - Cycle detection and fixing via graph algorithms

    Supported Source Models:
        - SUMMA: Physics-based snow hydrology (HRU or GRU runoff)
        - FUSE: Framework for Understanding Structural Errors
        - GR: Parsimonious hydrological models (GR4J, GR5J, GR6J)
        - NextGen (NGEN): NOAA modular BMI framework

    Processing Workflow:
        1. **Initialization**: Set up directories, handle custom paths for parallel runs
        2. **Base Settings**: Copy template parameter and control files
        3. **Network Topology**: Create NetCDF topology file from river network shapefiles
           - Handle headwater basins (synthetic network generation)
           - Detect and fix routing cycles using DFS graph algorithms
           - Support lumped-to-distributed routing via delineated subcatchments
        4. **Remapping** (optional): Create NetCDF remapping file
           - Area-weighted remapping for lumped-to-distributed conversion
           - Equal-weight remapping for uniform distribution
           - Spatial intersection-based remapping for multi-scale modeling
        5. **Control File**: Generate model-specific control file
           - SUMMA control: HRU vs GRU runoff handling
           - FUSE control: Basin-scale runoff routing
           - GR control: Daily timestep alignment, midnight forcing

    Key Methods (36 total):
        Main Workflow:
            run_preprocessing(): Orchestrates all preprocessing steps
            copy_base_settings(): Copy template files to setup directory

        Topology Creation (delegated to MizuRouteTopologyGenerator):
            create_network_topology_file(): Main topology file creation (208 lines)
            _create_grid_topology_file(): Grid-based distributed topology
            _check_if_headwater_basin(): Detect headwater basins with no river network
            _create_synthetic_river_network(): Generate single-segment network for headwaters
            _fix_routing_cycles(): Graph algorithm to detect and fix cycles
            _find_closest_segment_to_pour_point(): Locate segment nearest to basin outlet

        Remapping (delegated to MizuRouteRemapGenerator):
            create_area_weighted_remap_file(): Area-based weights from delineated catchments
            remap_summa_catchments_to_routing(): Spatial intersection remapping

        Control File Generation:
            create_control_file(): SUMMA-specific control file
            create_fuse_control_file(): FUSE-specific control file
            create_gr_control_file(): GR-specific control file

    Configuration Dependencies:
        Required:
            - DOMAIN_NAME: Basin identifier
            - SUB_GRID_DISCRETIZATION: Domain definition method (lumped/TBL/distribute)
            - RIVER_NETWORK_SHP_PATH: Path to river network shapefile
            - RIVER_NETWORK_SHP_NAME: River network shapefile name
            - RIVER_BASINS_PATH: Path to river basin shapefile
            - RIVER_BASINS_NAME: River basin shapefile name
            - EXPERIMENT_ID: Experiment identifier
            - EXPERIMENT_OUTPUT_MIZUROUTE: mizuRoute output directory

        Optional:
            - SETTINGS_MIZU_PATH: Custom setup directory (for parallel runs)
            - SETTINGS_MIZU_TOPOLOGY: Topology file name (default: mizuRoute_topology.nc)
            - SETTINGS_MIZU_REMAP: Remapping file name (default: remap_file.nc)
            - SETTINGS_MIZU_PARAMETERS: Parameter file name
            - SETTINGS_MIZU_NEEDS_REMAP: Enable remapping (T/F)
            - SETTINGS_MIZU_MAKE_OUTLET: Comma-separated segment IDs to force as outlets
            - SETTINGS_MIZU_WITHIN_BASIN: Hillslope routing option (0/1)
            - ROUTING_DELINEATION: Routing delineation method (river_network/basin)
            - GRID_CELL_SIZE: Grid cell size in meters (for distribute mode)
            - MODEL_MIZUROUTE_FROM_MODEL: Source model name (SUMMA/FUSE/GR)

        Shapefile Column Names:
            River Network:
                - RIVER_NETWORK_SHP_SEGID: Segment ID column
                - RIVER_NETWORK_SHP_DOWNSEGID: Downstream segment ID column
                - RIVER_NETWORK_SHP_LENGTH: Segment length column (m)
                - RIVER_NETWORK_SHP_SLOPE: Segment slope column (-)

            River Basins:
                - RIVER_BASIN_SHP_RM_GRUID: GRU ID column
                - RIVER_BASIN_SHP_RM_HRUID: HRU ID column
                - RIVER_BASIN_SHP_RM_AREA: Basin area column (m^2)
                - RIVER_BASIN_SHP_RM_HRU2SEG: HRU-to-segment mapping column

    Output Files:
        Network Topology (NetCDF):
            Dimensions: seg, hru
            Segment Variables:
                - segId: Unique segment IDs
                - downSegId: Downstream segment IDs (0 = outlet)
                - slope: Segment slopes
                - length: Segment lengths (m)
            HRU Variables:
                - hruId: Unique HRU IDs
                - hruToSegId: HRU-to-segment drainage mapping
                - area: HRU areas (m^2)

        Remapping File (NetCDF, optional):
            Dimensions: hru, data
            Variables:
                - RN_hruId: River network HRU IDs
                - nOverlaps: Number of overlapping source HRUs per routing HRU
                - HM_hruId: Source model HRU/GRU IDs
                - weight: Areal weights for remapping

        Control File (text):
            Sections:
                - Simulation controls (start/end times, routing options)
                - Directory paths (input/output/ancillary)
                - Topology file configuration
                - Remapping configuration (if enabled)
                - Parameter file reference
                - Miscellaneous settings (hillslope routing, output frequency)

    Special Handling:
        Headwater Basins:
            - Detects basins with None/null river network data
            - Creates synthetic single-segment network
            - Uses first HRU ID as segment ID, outlet downstream ID = 0

        Lumped-to-Distributed Routing:
            - Delineates subcatchments within lumped domain
            - Creates area-weighted remapping from single SUMMA GRU to N routing HRUs
            - Enables distributed routing for lumped hydrological models

        Routing Cycles:
            - Detects cycles using iterative DFS graph traversal
            - Breaks cycles by forcing lowest-elevation segment to outlet (downSegId = 0)
            - Logs number of cycles detected and fixed

        GRU-level Runoff:
            - Detects SUMMA simulations with multiple HRUs per GRU
            - Reads SUMMA attributes.nc to determine structure
            - Aggregates HRU areas to GRU level for topology

        Grid-based Distributed:
            - Reads D8 flow direction from grid shapefile
            - Each grid cell becomes both HRU and segment
            - Segment length = grid cell size
            - Fixes cycles in D8 topology

    Integration Patterns:
        SUMMA Integration:
            - Reads attributes.nc to detect HRU/GRU structure
            - Handles both hru/hruId and gru/gruId output formats
            - Sets summa_uses_gru_runoff flag for control file

        FUSE Integration:
            - Basin-scale runoff routing
            - Control file references FUSE output files

        GR Integration:
            - Daily timestep alignment (midnight forcing)
            - R/rpy2 interface output handling
            - Forces simulation times to 00:00 alignment

    Error Handling:
        - Validates shapefile existence before processing
        - Handles missing pour point shapefiles (fallback to outlet segment)
        - Fills missing/null length and slope values with defaults
        - Detects and logs warnings for outlet segment mismatches
        - Raises FileNotFoundError for critical missing files

    Example:
        >>> config = {
        ...     'DOMAIN_NAME': 'bow_river',
        ...     'SUB_GRID_DISCRETIZATION': 'lumped',
        ...     'RIVER_NETWORK_SHP_PATH': './shapefiles/river_network',
        ...     'RIVER_NETWORK_SHP_NAME': 'bow_river_riverNetwork_lumped.shp',
        ...     'RIVER_BASINS_PATH': './shapefiles/river_basins',
        ...     'RIVER_BASINS_NAME': 'bow_river_riverBasins_lumped.shp',
        ...     'EXPERIMENT_ID': 'bow_calibration',
        ...     'SETTINGS_MIZU_TOPOLOGY': 'mizuRoute_topology.nc',
        ...     'SETTINGS_MIZU_NEEDS_REMAP': False,
        ...     'MODEL_MIZUROUTE_FROM_MODEL': 'SUMMA'
        ... }
        >>> preprocessor = MizuRoutePreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates:
        # - ./settings/mizuRoute/mizuRoute_topology.nc (network topology)
        # - ./settings/mizuRoute/mizuRoute.control (control file)
        # - ./settings/mizuRoute/*.param (parameter files)

    Notes:
        - Topology file must be created before control file generation
        - Remapping is optional and only needed when source and routing HRUs differ
        - Control file references are model-specific (SUMMA uses different variable names than FUSE/GR)
        - Grid-based distributed mode requires D8 flow direction in shapefile
        - Parallel runs can specify custom setup directory via SETTINGS_MIZU_PATH
        - Cycle detection uses O(V+E) iterative DFS to avoid recursion depth issues
        - Minimum segment length enforced (1m) to prevent numerical instabilities
        - Minimum slope enforced (0.001) for routing calculations

    See Also:
        - models.mizuroute.topology_generator.MizuRouteTopologyGenerator: Topology generation
        - models.mizuroute.remap_generator.MizuRouteRemapGenerator: Remapping file generation
        - models.mizuroute.control_writer.ControlFileWriter: Control file generation
        - models.mizuroute.mixins.MizuRouteConfigMixin: Configuration accessors
        - core.geometry_utils.GeospatialUtilsMixin: Spatial utilities
        - models.base.BaseModelPreProcessor: Base preprocessor interface
    """

    MODEL_NAME = "mizuRoute"
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the mizuRoute preprocessor.

        Sets up directory paths for routing configuration, including optional
        custom settings path for isolated parallel runs during calibration.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                mizuRoute settings, topology paths, and routing parameters.
            logger: Logger instance for status messages and debugging.
        """
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        # Lazy-init backing fields for sub-modules
        self._topology_generator = None
        self._remap_generator = None

        self.logger.debug(f"MizuRoutePreProcessor initialized. Default setup_dir: {self.setup_dir}")

        # Override setup_dir if SETTINGS_MIZU_PATH is provided (for isolated parallel runs)
        mizu_settings_path = self.mizu_settings_path
        if mizu_settings_path and mizu_settings_path != 'default':
            self.setup_dir: Path = Path(mizu_settings_path)
            self.logger.debug(f"MizuRoutePreProcessor using custom setup_dir from SETTINGS_MIZU_PATH: {self.setup_dir}")

        # Ensure setup directory exists
        if not self.setup_dir.exists():
            self.logger.info(f"Creating mizuRoute setup directory: {self.setup_dir}")
            self.setup_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.logger.debug(f"mizuRoute setup directory already exists: {self.setup_dir}")

    # =========================================================================
    # Lazy-init sub-module properties
    # =========================================================================

    @property
    def topology_generator(self):
        """Lazy-init topology generator sub-module."""
        if self._topology_generator is None:
            from symfluence.models.mizuroute.topology_generator import MizuRouteTopologyGenerator
            self._topology_generator = MizuRouteTopologyGenerator(self)
        return self._topology_generator

    @property
    def remap_generator(self):
        """Lazy-init remap generator sub-module."""
        if self._remap_generator is None:
            from symfluence.models.mizuroute.remap_generator import MizuRouteRemapGenerator
            self._remap_generator = MizuRouteRemapGenerator(self)
        return self._remap_generator

    # =========================================================================
    # Main workflow
    # =========================================================================

    def run_preprocessing(self):
        """
        Run the complete mizuRoute preprocessing workflow.

        Executes all steps needed to prepare mizuRoute for routing:
        1. Copy base settings files from templates
        2. Create network topology file from river/catchment shapefiles
        3. Create remapping file if source model uses different spatial units
        4. Generate appropriate control file based on source model (SUMMA/FUSE/GR)

        The workflow adapts based on configuration, supporting both lumped-to-distributed
        remapping and distributed model coupling.
        """
        self.logger.debug("Starting mizuRoute spatial preprocessing")
        self.copy_base_settings()
        self.topology_generator.create_network_topology_file()

        # Get config values using typed config
        needs_remap = self._get_config_value(
            lambda: self.config.model.mizuroute.needs_remap if self.config.model and self.config.model.mizuroute else None,
            False
        )
        from_model = self._get_config_value(
            lambda: self.config.model.mizuroute.from_model if self.config.model and self.config.model.mizuroute else None
        )

        # Infer source model from HYDROLOGICAL_MODEL when MIZU_FROM_MODEL is not set
        if not from_model or from_model == 'default':
            from_model = self._get_config_value(
                lambda: self.config.model.hydrological_model if self.config.model else None,
                default=None,
                dict_key='HYDROLOGICAL_MODEL'
            )
            if from_model:
                from_model = from_model.upper()
                self.logger.info(f"MIZU_FROM_MODEL not set, inferred '{from_model}' from HYDROLOGICAL_MODEL")

        fuse_routing = self._get_config_value(
            lambda: self.config.model.fuse.routing_integration if self.config.model and self.config.model.fuse else None
        )
        gr_routing = self._get_config_value(
            lambda: self.config.model.gr.routing_integration if self.config.model and self.config.model.gr else None
        )

        # Check if lumped-to-distributed remapping is needed (set during topology creation)
        if getattr(self, 'needs_remap_lumped_distributed', False):
            self.logger.info("Creating area-weighted remap file for lumped-to-distributed routing")
            self.remap_generator.create_area_weighted_remap_file()
            needs_remap = True  # Override to enable remapping in control file

        self.logger.info(f"Should we remap?: {needs_remap}")
        if needs_remap and not getattr(self, 'needs_remap_lumped_distributed', False):
            self.remap_generator.remap_summa_catchments_to_routing()

        # Choose control writer based on source model
        if from_model == 'FUSE' or fuse_routing == 'mizuRoute':
            self.create_fuse_control_file()
        elif from_model == 'GR' or gr_routing == 'mizuRoute':
            self.create_gr_control_file()
        elif from_model == 'NGEN':
            self.create_ngen_control_file()
        else:
            self.create_control_file()

        self.logger.info("mizuRoute spatial preprocessing completed")

    # =========================================================================
    # Base settings
    # =========================================================================

    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[List[str]] = None):
        """
        Copy mizuRoute base settings from package resources.

        Copies template configuration files (parameter anchors, routing method
        settings) from symfluence resources to the setup directory, providing
        starting points that will be customized during preprocessing.
        """
        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)

        self.logger.info("Copying mizuRoute base settings")
        from symfluence.core.modeling.base_settings import get_base_settings_dir
        base_settings_path = get_base_settings_dir('mizuRoute')
        self.setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.setup_dir / file)
        self.logger.info("mizuRoute base settings copied")

    # =========================================================================
    # Control file generation
    # =========================================================================

    def create_control_file(self):
        """
        Create mizuRoute control file for SUMMA runoff input.

        Generates the mizuRoute control file (*.control) that configures the
        routing simulation when using SUMMA as the source hydrological model.
        The control file specifies input/output paths, topology files, routing
        scheme parameters, and simulation time controls.

        The control file includes sections for:
        - Directory and file paths (topology, runoff input, output)
        - Simulation period (start/end times from config)
        - Routing scheme selection (IRF, KWT, DW)
        - Spatial configuration (segments, HRUs, remapping)
        - Output variable selection and frequency

        Uses ControlFileWriter to generate SUMMA-specific settings that account
        for SUMMA's GRU-level runoff output format and time conventions.

        File is written to: {setup_dir}/{experiment_id}.control

        See Also:
            create_fuse_control_file: For FUSE model input configuration.
        """
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        writer.write_control_file(model_type='summa', mizu_config=mizu_config)

    def create_fuse_control_file(self):
        """
        Create mizuRoute control file for FUSE runoff input.

        Generates the mizuRoute control file (*.control) configured for routing
        FUSE model output. FUSE produces runoff in a different format than SUMMA,
        requiring specific variable mappings and time handling in the control file.

        Key FUSE-specific configurations:
        - Variable name mapping for FUSE runoff output variables
        - Time dimension handling (FUSE uses different time conventions)
        - Appropriate runoff flux units conversion if needed

        The control file includes the same structural sections as SUMMA routing:
        - Directory and file paths (topology, runoff input, output)
        - Simulation period matching the FUSE run
        - Routing scheme selection (IRF, KWT, DW)
        - Spatial configuration with appropriate remapping

        File is written to: {setup_dir}/{experiment_id}.control

        See Also:
            create_control_file: For SUMMA model input configuration.
        """
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        writer.write_control_file(model_type='fuse', mizu_config=mizu_config)

    def create_gr_control_file(self):
        """Create mizuRoute control file specifically for GR4J input."""
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        writer.write_control_file(model_type='gr', mizu_config=mizu_config)

    def create_ngen_control_file(self):
        """Create mizuRoute control file for NGEN runoff input.

        NGEN modules like SAC-SMA and TOPMODEL produce raw runoff without
        built-in routing (unlike CFE which has GIUH). The NGEN postprocessor
        converts nexus CSV outputs to a mizuRoute-compatible NetCDF with
        runoff in m/s. This control file points mizuRoute at that NetCDF.

        doesBasinRoute is set to 1 (IRF hillslope routing) since NGEN's
        SAC-SMA/TOPMODEL output is unrouted.
        """
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        mizu_config['within_basin'] = 1
        writer.write_control_file(model_type='ngen', mizu_config=mizu_config)

    def _get_control_writer(self) -> ControlFileWriter:
        """Get a configured ControlFileWriter instance."""
        writer = ControlFileWriter(
            config=self.config_dict,
            setup_dir=self.setup_dir,
            project_dir=self.project_dir,
            experiment_id=self.experiment_id,
            domain_name=self.domain_name,
            logger=self.logger
        )
        # Transfer state flags
        writer.summa_uses_gru_runoff = getattr(self, 'summa_uses_gru_runoff', False)
        writer.needs_remap_lumped_distributed = getattr(self, 'needs_remap_lumped_distributed', False)
        return writer

    def _get_mizu_config(self) -> dict:
        """Get mizuRoute configuration values for the control writer."""
        return {
            'topology_file': self.mizu_topology_file,
            'remap_file': self.mizu_remap_file,
            'parameters_file': self.mizu_parameters_file,
            'within_basin': self.mizu_within_basin,
        }
