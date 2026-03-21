# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Distributed geofabric delineation using TauDEM.

Fully refactored implementation using extracted stream methods and shared utilities.

Refactored from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd

from ....geospatial.delineation_registry import DelineationRegistry
from ..base.base_delineator import BaseGeofabricDelineator
from ..methods.curvature import CurvatureMethod
from ..methods.multi_scale import MultiScaleMethod
from ..methods.slope_area import SlopeAreaMethod
from ..methods.stream_threshold import StreamThresholdMethod
from ..processors.gdal_processor import GDALProcessor
from ..processors.geometry_processor import GeometryProcessor
from ..processors.graph_processor import RiverGraphProcessor
from ..processors.taudem_executor import TauDEMExecutor
from ..utils.crs_utils import CRSUtils
from ..utils.io_utils import GeofabricIOUtils
from ..utils.validation import GeofabricValidator
from .coastal_delineator import CoastalWatershedDelineator


@DelineationRegistry.register('semidistributed')
class GeofabricDelineator(BaseGeofabricDelineator):
    """
    Main geofabric delineation class for distributed and semi-distributed domains.

    Supports multiple stream delineation methods:
    - stream_threshold: Threshold-based (with optional drop analysis)
    - curvature: Peuker-Douglas curvature-based
    - slope_area: Slope-area relationship
    - multi_scale: Multi-scale hierarchical
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        """
        Initialize distributed geofabric delineator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger)
        self.reporting_manager = reporting_manager

        # Stream delineation method
        self.delineation_method = self._get_config_value(lambda: self.config.domain.delineation.method, default='stream_threshold', dict_key='DELINEATION_METHOD').lower()

        # Validate method
        valid_methods = ['stream_threshold', 'curvature', 'slope_area', 'multi_scale']
        if self.delineation_method not in valid_methods:
            self.logger.warning(f"Unknown delineation method '{self.delineation_method}'. Using 'stream_threshold'.")
            self.delineation_method = 'stream_threshold'

        # Interim directory for d8 workflow
        self.interim_dir = self.project_dir / "taudem-interim-files" / "d8"

        # Initialize processors
        self.taudem = TauDEMExecutor(config, logger, self.taudem_dir)
        self.gdal = GDALProcessor(logger)
        self.graph = RiverGraphProcessor()

        # Initialize stream method instances
        self.stream_methods = {
            'stream_threshold': StreamThresholdMethod(self.taudem, config, logger, self.interim_dir, self.reporting_manager),
            'curvature': CurvatureMethod(self.taudem, config, logger, self.interim_dir),
            'slope_area': SlopeAreaMethod(self.taudem, config, logger, self.interim_dir),
            'multi_scale': MultiScaleMethod(self.taudem, config, logger, self.interim_dir),
        }

        # Coastal delineator
        self.coastal_delineator = CoastalWatershedDelineator(config, logger)

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files.

        Uses config-based _get_method_suffix() for consistency with model preprocessors.
        This ensures delineation output filenames match what preprocessors expect when loading.
        """
        return self._get_method_suffix()

    def delineate_geofabric(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Main delineation workflow.

        Returns:
            Tuple of (river_network_path, river_basins_path)
        """
        try:
            self.logger.info(
                f"Starting geofabric delineation for {self.domain_name} "
                f"using method: {self.delineation_method}"
            )

            # Validate inputs
            self._validate_inputs()

            # Create directories
            self.create_directories()

            # Get pour point path
            pour_point_path = self._get_pour_point_path()

            # Run TauDEM workflow
            self._run_taudem_workflow(pour_point_path)

            # Convert raster watersheds to polygon shapefile
            self.gdal.run_gdal_processing(self.interim_dir)

            # Subset upstream geofabric
            river_network_path, river_basins_path = self._subset_upstream_geofabric(pour_point_path)

            # Cleanup
            self.cleanup()

            self.logger.info(f"Geofabric delineation completed for {self.domain_name}")
            return river_network_path, river_basins_path

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error in geofabric delineation: {str(e)}")
            self.cleanup()
            raise

    def _validate_inputs(self) -> None:
        """Validate required input files."""
        GeofabricValidator.validate_dem_exists(self.dem_path)
        GeofabricValidator.validate_delineation_method(self.delineation_method)

    def _get_pour_point_path(self) -> Path:
        """
        Get the pour point shapefile path.

        Returns:
            Path to pour point shapefile
        """
        pour_point_path = self._get_config_value(lambda: self.config.paths.pour_point_path, dict_key='POUR_POINT_SHP_PATH')
        if pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(pour_point_path)

        if self._get_config_value(lambda: self.config.paths.pour_point_name, dict_key='POUR_POINT_SHP_NAME') == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"

        if not pour_point_path.exists():
            raise FileNotFoundError(f"Pour point file not found: {pour_point_path}")

        return pour_point_path

    def _run_taudem_workflow(self, pour_point_path: Path) -> None:
        """
        Run complete TauDEM workflow with selected stream identification method.

        Args:
            pour_point_path: Path to pour point shapefile
        """
        # Determine MPI command
        mpi_cmd = self.taudem.get_mpi_command()
        if mpi_cmd:
            mpi_prefix = f"{mpi_cmd} -n {self._get_config_value(lambda: self.config.system.num_processes, default=1, dict_key='NUM_PROCESSES')} "
        else:
            mpi_prefix = ""

        # Run common initial TauDEM steps
        self._run_common_taudem_steps(mpi_prefix)

        # Run selected stream identification method
        method = self.stream_methods[self.delineation_method]
        self.logger.info(f"Running {self.delineation_method} stream identification")
        method.run(self.dem_path, pour_point_path, mpi_prefix)

        self.logger.info("Completed all TauDEM steps")

    def _run_common_taudem_steps(self, mpi_prefix: str) -> None:
        """
        Run common TauDEM preprocessing steps (same for all methods).

        Args:
            mpi_prefix: MPI command prefix
        """
        taudem_dir = self.taudem.taudem_dir

        # Apply DEM conditioning (stream burning) if configured
        dem_input = self._condition_dem()

        common_steps = [
            f"{mpi_prefix}{taudem_dir}/pitremove -z {dem_input} -fel {self.interim_dir}/elv-fel.tif -v",
            f"{mpi_prefix}{taudem_dir}/d8flowdir -fel {self.interim_dir}/elv-fel.tif -sd8 {self.interim_dir}/elv-sd8.tif -p {self.interim_dir}/elv-fdir.tif",
            f"{mpi_prefix}{taudem_dir}/aread8 -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -nc",
        ]

        for step in common_steps:
            self.taudem.run_command(step)
            self.logger.info("Completed TauDEM common step")

    def _subset_upstream_geofabric(self, pour_point_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Subset geofabric to upstream contributing area.

        Args:
            pour_point_path: Path to pour point shapefile

        Returns:
            Tuple of (river_network_path, river_basins_path)
        """
        try:
            basins_path = self.interim_dir / "basin-watersheds.shp"
            rivers_path = self.interim_dir / "basin-streams.shp"

            # Load shapefiles
            pour_point = GeofabricIOUtils.load_geopandas(pour_point_path, self.logger)
            basins = GeofabricIOUtils.load_geopandas(basins_path, self.logger)
            rivers = GeofabricIOUtils.load_geopandas(rivers_path, self.logger)

            # Process geofabric attributes
            basins, rivers = self._process_geofabric(basins, rivers)

            # Get output paths
            subset_basins_path, subset_rivers_path = self._get_output_paths()

            # Subset by pour point if requested
            if self._get_config_value(lambda: self.config.domain.delineation.delineate_by_pourpoint, default=True, dict_key='DELINEATE_BY_POURPOINT'):
                # Ensure CRS consistency
                basins, rivers, pour_point = CRSUtils.ensure_crs_consistency(
                    basins, rivers, pour_point, self.logger
                )

                # Find basin containing pour point
                downstream_basin_id = CRSUtils.find_basin_for_pour_point(
                    pour_point, basins, logger=self.logger,
                )

                # Build river network graph
                river_graph = self.graph.build_river_graph(rivers, self._get_fabric_config(rivers))

                # Find upstream basins
                upstream_basin_ids = self.graph.find_upstream_basins(
                    downstream_basin_id, river_graph, self.logger
                )

                # Subset to upstream area
                subset_basins = basins[basins['GRU_ID'].isin(upstream_basin_ids)].copy()
                subset_rivers = rivers[rivers['GRU_ID'].isin(upstream_basin_ids)].copy()
            else:
                subset_basins, subset_rivers = basins, rivers

            # Save geofabric
            self._save_geofabric(subset_basins, subset_rivers, subset_basins_path, subset_rivers_path)

            return subset_rivers_path, subset_basins_path

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error during geofabric subsetting: {str(e)}")
            return None, None

    def _process_geofabric(
        self,
        basins: gpd.GeoDataFrame,
        rivers: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Process geofabric attributes.

        Args:
            basins: Basin GeoDataFrame
            rivers: River network GeoDataFrame

        Returns:
            Tuple of processed (basins, rivers)
        """
        # Set GRU_ID from DN field
        if 'GRU_ID' in basins.columns:
            pass
        elif 'DN' in basins.columns:
            basins['GRU_ID'] = basins['DN']
        elif 'value' in basins.columns:
            basins['GRU_ID'] = basins['value']
        elif 'ID' in basins.columns:
            basins['GRU_ID'] = basins['ID']
        else:
            self.logger.warning(f"'DN' column not found in basins. Available columns: {list(basins.columns)}")
            # Fallback to first column if it exists, hoping it's the ID
            if len(basins.columns) > 0:
                # Avoid using geometry column
                cols = [c for c in basins.columns if c != 'geometry']
                if cols:
                    first_col = cols[0]
                    self.logger.warning(f"Using '{first_col}' as GRU_ID fallback.")
                    basins['GRU_ID'] = basins[first_col]
                else:
                     raise KeyError("No suitable ID column found in basins shapefile")
            else:
                raise KeyError("No suitable ID column found in basins shapefile ('DN', 'value', 'ID')")

        river_id_col = self._get_fabric_config(rivers)['river_id_col']
        rivers['GRU_ID'] = rivers[river_id_col]

        # Calculate areas in UTM
        utm_crs = basins.estimate_utm_crs()
        basins_utm = basins.to_crs(utm_crs)
        basins['GRU_area'] = basins_utm.geometry.area
        basins['gru_to_seg'] = basins['GRU_ID']

        # Drop DN column
        if 'DN' in basins.columns:
            basins = basins.drop(columns=['DN'])

        # Handle duplicate GRU_IDs
        len(basins)
        duplicated_ids = basins['GRU_ID'].duplicated(keep=False)
        duplicate_count = duplicated_ids.sum()

        if duplicate_count > 0:
            self.logger.info(f"Found {duplicate_count} rows with duplicate GRU_ID values")

            # Keep only the largest area for each GRU_ID
            basins = basins.sort_values(['GRU_ID', 'GRU_area'], ascending=[True, False])
            basins = basins.drop_duplicates(subset=['GRU_ID'], keep='first')

            self.logger.info("Removed duplicates, keeping largest area for each GRU_ID")
            self.logger.info(f"Remaining GRUs: {len(basins)}")

        return basins, rivers

    def _get_output_paths(self) -> Tuple[Path, Path]:
        """
        Get output file paths for geofabric.

        Returns:
            Tuple of (basins_path, rivers_path)
        """
        subset_basins_path = self._get_config_value(lambda: self.config.paths.output_basins_path, dict_key='OUTPUT_BASINS_PATH')
        subset_rivers_path = self._get_config_value(lambda: self.config.paths.output_rivers_path, dict_key='OUTPUT_RIVERS_PATH')
        # Use config-based method suffix for consistency with model preprocessors
        method_suffix = self._get_delineation_method_name()

        if subset_basins_path == 'default':
            subset_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{method_suffix}.shp"
        else:
            subset_basins_path = Path(self._get_config_value(lambda: self.config.paths.output_basins_path, dict_key='OUTPUT_BASINS_PATH'))

        if subset_rivers_path == 'default':
            subset_rivers_path = self.project_dir / "shapefiles" / "river_network" / f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
        else:
            subset_rivers_path = Path(self._get_config_value(lambda: self.config.paths.output_rivers_path, dict_key='OUTPUT_RIVERS_PATH'))

        return subset_basins_path, subset_rivers_path

    def _save_geofabric(
        self,
        basins: gpd.GeoDataFrame,
        rivers: gpd.GeoDataFrame,
        basins_path: Path,
        rivers_path: Path
    ) -> None:
        """
        Save geofabric files with corrected geometries.

        Args:
            basins: Basin GeoDataFrame
            rivers: River network GeoDataFrame
            basins_path: Output path for basins
            rivers_path: Output path for rivers
        """
        basins_path.parent.mkdir(parents=True, exist_ok=True)
        rivers_path.parent.mkdir(parents=True, exist_ok=True)

        # Fix polygon winding order
        basins['geometry'] = basins['geometry'].apply(GeometryProcessor.fix_polygon_winding)

        # Process geofabric one more time to ensure consistency
        basins, rivers = self._process_geofabric(basins, rivers)

        # Save files
        basins.to_file(basins_path)
        rivers.to_file(rivers_path)

        self.logger.info(f"Subset basins shapefile saved to: {basins_path}")
        self.logger.info(f"Subset rivers shapefile saved to: {rivers_path}")

    def _get_fabric_config(self, rivers: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Get hydrofabric configuration for graph processing.

        Detects the river ID column from the available columns,
        preferring WSNO (watershed number) over LINKNO (link number)
        for TauDEM-derived geofabrics.

        Args:
            rivers: River network GeoDataFrame

        Returns:
            Dictionary with column name mappings
        """
        if 'WSNO' in rivers.columns:
            river_id_col = 'WSNO'
        elif 'LINKNO' in rivers.columns:
            river_id_col = 'LINKNO'
        else:
            raise KeyError("No suitable river ID column found ('WSNO' or 'LINKNO')")

        return {
            'river_id_col': river_id_col,
            'upstream_cols': ['DSLINKNO'],
            'upstream_default': -1,
            'direction': 'downstream'
        }

    def delineate_coastal(self, work_log_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Delineate coastal watersheds.

        Identifies and delineates coastal areas that drain directly to the ocean,
        not captured by standard watershed delineation.

        Args:
            work_log_dir: Optional directory for logging

        Returns:
            Tuple of (river_network_path, river_basins_path) with coastal areas included
        """
        return self.coastal_delineator.delineate_coastal(work_log_dir)

    def delineate_point_buffer_shape(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Create a small square buffer around the pour point for point-scale simulations.

        Returns:
            Tuple of (river_network_path, river_basins_path) for point buffer
        """
        return self.coastal_delineator.delineate_point_buffer_shape()
