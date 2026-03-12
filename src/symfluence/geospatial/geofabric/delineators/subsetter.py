# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Subset existing geofabric data based on pour points.

Supports MERIT, TDX, NWS, and HydroSHEDS hydrofabric formats.
Uses graph-based upstream tracing to subset basins and rivers.

Refactored from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd

from ..base.base_delineator import BaseGeofabricDelineator
from ..processors.graph_processor import RiverGraphProcessor
from ..utils.crs_utils import CRSUtils
from ..utils.io_utils import GeofabricIOUtils


class GeofabricSubsetter(BaseGeofabricDelineator):
    """
    Subsets geofabric data based on pour points and upstream basins.

    Supports four hydrofabric formats with different column naming conventions:
    - MERIT: COMID-based with up1, up2, up3 upstream columns
    - TDX: streamID/LINKNO with USLINKNO1, USLINKNO2 upstream columns
    - NWS: divide_id-based with toid (reverse direction)
    - HYDROSHEDS: HYBAS_ID with NEXT_DOWN (reverse direction)
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize geofabric subsetter.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Hydrofabric type configurations
        self.hydrofabric_types = {
            'MERIT': {
                'basin_id_col': 'COMID',
                'river_id_col': 'COMID',
                'upstream_cols': ['up1', 'up2', 'up3'],
                'upstream_default': -9999
            },
            'TDX': {
                'basin_id_col': 'streamID',
                'river_id_col': 'LINKNO',
                'upstream_cols': ['USLINKNO1', 'USLINKNO2'],
                'upstream_default': -9999
            },
            'NWS': {
                'basin_id_col': 'divide_id',
                'river_id_col': 'divide_id',
                'upstream_cols': ['toid'],
                'upstream_default': 0
            },
            'HYDROSHEDS': {
                'basin_id_col': 'HYBAS_ID',
                'river_id_col': 'HYBAS_ID',
                'upstream_cols': ['NEXT_DOWN'],
                'upstream_default': 0
            }
        }

        # Initialize graph processor
        self.graph = RiverGraphProcessor()

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files.

        Uses the new naming convention based on definition_method and subset_from_geofabric.
        """
        return self._get_method_suffix()

    def subset_geofabric(self) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Subset the geofabric based on configuration.

        If source geofabric paths are not set or files don't exist,
        automatically downloads the appropriate regional data.

        Returns:
            Tuple of (subset_basins, subset_rivers) GeoDataFrames
        """
        hydrofabric_type = self._get_config_value(lambda: self.config.domain.delineation.geofabric_type, dict_key='GEOFABRIC_TYPE').upper()
        if hydrofabric_type not in self.hydrofabric_types:
            self.logger.error(f"Unknown hydrofabric type: {hydrofabric_type}")
            return None, None

        fabric_config = self.hydrofabric_types[hydrofabric_type]

        # Auto-download geofabric if source paths are missing or default
        basins_path, rivers_path = self._ensure_geofabric_available(hydrofabric_type)

        # Load data using shared utility
        basins = GeofabricIOUtils.load_geopandas(basins_path, self.logger)
        rivers = GeofabricIOUtils.load_geopandas(rivers_path, self.logger)
        pour_point = GeofabricIOUtils.load_geopandas(
            self._get_pour_point_path(),
            self.logger
        )

        # Ensure CRS consistency using shared utility
        basins, rivers, pour_point = CRSUtils.ensure_crs_consistency(
            basins, rivers, pour_point, self.logger
        )

        # Find downstream basin using shared utility
        downstream_basin_id = CRSUtils.find_basin_for_pour_point(
            pour_point, basins, fabric_config['basin_id_col'],
            logger=self.logger,
        )

        # Build graph and find upstream using shared processor
        river_graph = self.graph.build_river_graph(rivers, fabric_config)
        upstream_basin_ids = self.graph.find_upstream_basins(
            downstream_basin_id, river_graph, self.logger
        )

        # Subset basins and rivers
        subset_basins = basins[basins[fabric_config['basin_id_col']].isin(upstream_basin_ids)].copy()
        subset_rivers = rivers[rivers[fabric_config['river_id_col']].isin(upstream_basin_ids)].copy()

        # Add SYMFLUENCE-specific columns
        self._add_symfluence_columns(subset_basins, subset_rivers, hydrofabric_type)

        # Save using custom paths
        basins_path, rivers_path = self._get_output_paths()
        GeofabricIOUtils.save_geofabric(
            subset_basins, subset_rivers,
            basins_path, rivers_path,
            self.logger
        )

        return subset_basins, subset_rivers

    def _ensure_geofabric_available(self, hydrofabric_type: str) -> Tuple[Path, Path]:
        """Ensure geofabric source files exist, downloading if necessary.

        Checks configured source paths. If paths are 'default', missing, or
        point to non-existent files, triggers auto-download of the appropriate
        regional data.

        Args:
            hydrofabric_type: Uppercase hydrofabric type (MERIT, TDX, NWS)

        Returns:
            Tuple of (basins_path, rivers_path) to existing files
        """
        basins_cfg = self._get_config_value(
            lambda: self.config.paths.source_geofabric_basins_path,
            dict_key='SOURCE_GEOFABRIC_BASINS_PATH'
        )
        rivers_cfg = self._get_config_value(
            lambda: self.config.paths.source_geofabric_rivers_path,
            dict_key='SOURCE_GEOFABRIC_RIVERS_PATH'
        )

        basins_path = Path(basins_cfg) if basins_cfg and basins_cfg != 'default' else None
        rivers_path = Path(rivers_cfg) if rivers_cfg and rivers_cfg != 'default' else None

        # If both paths exist, use them directly
        if (basins_path and basins_path.exists() and
                rivers_path and rivers_path.exists()):
            return basins_path, rivers_path

        # Auto-download required
        self.logger.info(
            f"Source geofabric files not found for {hydrofabric_type}, "
            "attempting automatic download..."
        )
        return self._auto_download_geofabric(hydrofabric_type)

    def _auto_download_geofabric(self, hydrofabric_type: str) -> Tuple[Path, Path]:
        """Download geofabric data using the appropriate acquisition handler.

        Args:
            hydrofabric_type: Uppercase hydrofabric type (MERIT, TDX, NWS)

        Returns:
            Tuple of (basins_path, rivers_path) to downloaded files

        Raises:
            ValueError: If hydrofabric type has no registered handler
            FileNotFoundError: If download succeeds but expected files not found
        """
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        # Map hydrofabric type to acquisition handler name
        handler_map = {
            'TDX': 'TDX_HYDRO',
            'MERIT': 'MERIT_BASINS',
            'NWS': 'NWS_HYDROFABRIC',
            'HYDROSHEDS': 'HYDROSHEDS',
        }

        handler_name = handler_map.get(hydrofabric_type)
        if not handler_name:
            raise ValueError(
                f"No automatic download handler for hydrofabric type "
                f"'{hydrofabric_type}'. Please set SOURCE_GEOFABRIC_BASINS_PATH "
                f"and SOURCE_GEOFABRIC_RIVERS_PATH manually."
            )

        handler = AcquisitionRegistry.get_handler(
            handler_name, self.config, self.logger
        )
        output_dir = handler.download(self.project_dir)

        # Locate the downloaded files
        basins_path, rivers_path = self._find_downloaded_geofabric(
            output_dir, hydrofabric_type
        )
        return basins_path, rivers_path

    def _find_downloaded_geofabric(
        self, download_dir: Path, hydrofabric_type: str
    ) -> Tuple[Path, Path]:
        """Locate basins and rivers files in the download directory.

        Args:
            download_dir: Directory where handler saved files
            hydrofabric_type: Uppercase hydrofabric type

        Returns:
            Tuple of (basins_path, rivers_path)

        Raises:
            FileNotFoundError: If expected files not found
        """
        basins_path = None
        rivers_path = None

        if hydrofabric_type == 'TDX':
            # Look for merged first, then individual parquets
            merged_cat = download_dir / "tdx_catchments_merged.parquet"
            merged_riv = download_dir / "tdx_rivers_merged.parquet"
            if merged_cat.exists() and merged_riv.exists():
                return merged_cat, merged_riv
            # Find individual VPU files
            cat_files = sorted(download_dir.glob("tdx_catchments_*.parquet"))
            riv_files = sorted(download_dir.glob("tdx_rivers_*.parquet"))
            if cat_files:
                basins_path = cat_files[0]
            if riv_files:
                rivers_path = riv_files[0]

        elif hydrofabric_type == 'MERIT':
            # Find shapefile pairs
            cat_files = sorted(download_dir.glob("**/pfaf_*_Basins_*.shp"))
            riv_files = sorted(download_dir.glob("**/pfaf_*_rivernet*.shp"))
            # Also check for MERIT naming pattern
            if not cat_files:
                cat_files = sorted(download_dir.glob("**/merit_cat_*.shp"))
            if not riv_files:
                riv_files = sorted(download_dir.glob("**/merit_riv_*.shp"))
            if cat_files:
                basins_path = cat_files[0]
            if riv_files:
                rivers_path = riv_files[0]

        elif hydrofabric_type == 'NWS':
            cat_files = sorted(download_dir.glob("nws_catchments_*.gpkg"))
            riv_files = sorted(download_dir.glob("nws_flowpaths_*.gpkg"))
            if cat_files:
                basins_path = cat_files[0]
            if riv_files:
                rivers_path = riv_files[0]

        elif hydrofabric_type == 'HYDROSHEDS':
            # HydroBASINS provides both catchment polygons and topology
            # (HYBAS_ID + NEXT_DOWN), so we use the same file for both
            shp_files = sorted(download_dir.glob("hybas_*_v1c.shp"))
            if shp_files:
                basins_path = shp_files[0]
                rivers_path = shp_files[0]  # Same file — topology is in basins

        if basins_path is None or rivers_path is None:
            raise FileNotFoundError(
                f"Could not locate downloaded {hydrofabric_type} geofabric files "
                f"in {download_dir}. Check download logs for errors."
            )

        self.logger.info(f"Using downloaded basins: {basins_path}")
        self.logger.info(f"Using downloaded rivers: {rivers_path}")
        return basins_path, rivers_path

    def _add_symfluence_columns(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, hydrofabric_type: str):
        """
        Add SYMFLUENCE-specific columns based on hydrofabric type.

        Modifies GeoDataFrames in place.

        Args:
            basins: Basin GeoDataFrame to modify
            rivers: River GeoDataFrame to modify
            hydrofabric_type: Type of hydrofabric (NWS, TDX, Merit)
        """
        if hydrofabric_type == 'NWS':
            basins['GRU_ID'] = basins['divide_id']
            basins['gru_to_seg'] = basins['divide_id']
            # Calculate area in metric
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area
            # Rivers
            rivers['LINKNO'] = rivers['divide_id']
            rivers['DSLINKNO'] = rivers['toid']

        elif hydrofabric_type == 'TDX':
            basins['GRU_ID'] = basins['fid']
            basins['gru_to_seg'] = basins['streamID']
            # Calculate area in metric
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area

        elif hydrofabric_type in ['Merit', 'MERIT']:
            basins['GRU_ID'] = basins['COMID']
            basins['gru_to_seg'] = basins['COMID']
            # Calculate area in metric
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area
            # Rivers
            rivers['LINKNO'] = rivers['COMID']
            rivers['DSLINKNO'] = rivers['NextDownID']
            rivers_metric = rivers.to_crs('epsg:3763')
            rivers['Length'] = rivers_metric.geometry.length
            rivers.rename(columns={'slope': 'Slope'}, inplace=True)

        elif hydrofabric_type == 'HYDROSHEDS':
            basins['GRU_ID'] = basins['HYBAS_ID']
            basins['gru_to_seg'] = basins['HYBAS_ID']
            # HydroBASINS provides SUB_AREA in km², but also compute from geometry
            basins_metric = basins.to_crs('epsg:3763')
            basins['GRU_area'] = basins_metric.geometry.area
            # Rivers (same data as basins for HydroSHEDS)
            rivers['LINKNO'] = rivers['HYBAS_ID']
            rivers['DSLINKNO'] = rivers['NEXT_DOWN']

    def aggregate_to_lumped(
        self,
        basins: gpd.GeoDataFrame,
        preserve_path: Path
    ) -> gpd.GeoDataFrame:
        """
        Aggregate subset basins to single lumped polygon.

        This method dissolves multiple subset basins into a single polygon,
        preserving the original basins for use in remap files.

        Args:
            basins: Subset basins GeoDataFrame
            preserve_path: Path to save original (unaggregated) basins

        Returns:
            GeoDataFrame with single dissolved polygon
        """
        # Save original basins for remap files
        basins.to_file(preserve_path)
        self.logger.info(f"Preserved original basins to: {preserve_path}")

        # Dissolve to single polygon
        dissolved = basins.dissolve()

        # Set lumped attributes
        dissolved = dissolved.reset_index(drop=True)
        dissolved['GRU_ID'] = 1
        dissolved['gru_to_seg'] = 1

        # Calculate area in metric CRS
        dissolved_metric = dissolved.to_crs('EPSG:3763')
        dissolved['GRU_area'] = dissolved_metric.geometry.area.values[0]

        self.logger.info(f"Aggregated {len(basins)} basins to single lumped polygon")
        return dissolved

    def _get_output_paths(self) -> Tuple[Path, Path]:
        """
        Get output paths for subset shapefiles.

        Returns:
            Tuple of (basins_path, rivers_path)
        """
        method_suffix = self._get_method_suffix()

        if self._get_config_value(lambda: self.config.paths.output_basins_path, dict_key='OUTPUT_BASINS_PATH') == 'default':
            basins_path = (
                self.project_dir / "shapefiles" / "river_basins" /
                f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            )
        else:
            basins_path = Path(self._get_config_value(lambda: self.config.paths.output_basins_path, dict_key='OUTPUT_BASINS_PATH'))

        if self._get_config_value(lambda: self.config.paths.output_rivers_path, dict_key='OUTPUT_RIVERS_PATH') == 'default':
            rivers_path = (
                self.project_dir / "shapefiles" / "river_network" /
                f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
            )
        else:
            rivers_path = Path(self._get_config_value(lambda: self.config.paths.output_rivers_path, dict_key='OUTPUT_RIVERS_PATH'))

        return basins_path, rivers_path
