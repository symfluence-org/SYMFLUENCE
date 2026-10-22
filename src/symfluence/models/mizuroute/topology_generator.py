# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MizuRoute topology generation sub-module.

Handles network topology file creation, including grid-based and point-scale
topologies, headwater basin detection, synthetic network generation,
and routing cycle detection/fixing.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import netCDF4 as nc4
import numpy as np

from symfluence.core.exceptions import FileOperationError
from symfluence.core.path_resolver import find_catchment_subfile

if TYPE_CHECKING:
    from symfluence.models.mizuroute.preprocessor import MizuRoutePreProcessor

logger = logging.getLogger(__name__)


class MizuRouteTopologyGenerator:
    """
    Generates network topology NetCDF files for mizuRoute.

    Delegates to the parent preprocessor for config access, logger,
    directory paths, and mixin properties (shapefile columns, etc.).

    Args:
        preprocessor: Parent MizuRoutePreProcessor instance.
    """

    def __init__(self, preprocessor: 'MizuRoutePreProcessor'):
        self.pp = preprocessor

    # =========================================================================
    # Main topology creation
    # =========================================================================

    def create_network_topology_file(self):
        """
        Create the network topology NetCDF file for mizuRoute.

        Generates a topology file containing river segment IDs, downstream connectivity,
        HRU assignments, and channel properties. Supports multiple modes:
        - Standard distributed: Uses river network and basin shapefiles
        - Lumped-to-distributed: Creates synthetic network for single-GRU to multi-segment
        - Grid-based: Creates topology from regular grid cells
        - Point-scale: Creates minimal single-segment topology

        The topology file is required by mizuRoute to route water through the network.
        """
        self.pp.logger.info("Creating network topology file")

        # Check for grid-based distribute mode
        is_grid_distribute = self.pp.domain_definition_method == 'distribute'
        if is_grid_distribute:
            self._create_grid_topology_file()
            return

        # Check for point-scale mode
        is_point_scale = self.pp.domain_definition_method == 'point'
        if is_point_scale:
            self._create_point_topology_file()
            return

        river_network_path = self.pp._get_config_value(lambda: self.pp.config.paths.river_network_path, default='default')
        river_network_name = self.pp._get_config_value(lambda: self.pp.config.paths.river_network_name, default='default')
        method_suffix = self.pp._get_method_suffix()

        # Check if this is lumped domain with distributed routing
        # If so, use the delineated river network (from distributed delineation)
        routing_delineation = self.pp._get_config_value(
            lambda: self.pp.config.domain.delineation.routing,
            default='lumped',
            dict_key='ROUTING_DELINEATION',
        )
        is_lumped_to_distributed = (
            self.pp.domain_definition_method == 'lumped' and
            routing_delineation == 'river_network'
        )

        # For lumped-to-distributed, use delineated river network and catchments
        routing_suffix = 'delineate' if is_lumped_to_distributed else method_suffix

        if river_network_name == 'default':
            river_network_name = f"{self.pp.domain_name}_riverNetwork_{routing_suffix}.shp"

        if river_network_path == 'default':
            river_network_path = self.pp.project_dir / 'shapefiles/river_network'
        else:
            river_network_path = Path(river_network_path)

        river_basin_path = self.pp._get_config_value(lambda: self.pp.config.paths.river_basins_path, default='default')
        river_basin_name = self.pp._get_config_value(lambda: self.pp.config.paths.river_basins_name, default='default')

        if river_basin_name == 'default':
            river_basin_name = f"{self.pp.domain_name}_riverBasins_{routing_suffix}.shp"

        if river_basin_path == 'default':
            river_basin_path = self.pp.project_dir / 'shapefiles/river_basins'
        else:
            river_basin_path = Path(river_basin_path)

        topology_name = self.pp.mizu_topology_file
        if not topology_name:
            topology_name = "mizuRoute_topology.nc"
            self.pp.logger.warning(f"SETTINGS_MIZU_TOPOLOGY not found in config, using default: {topology_name}")

        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)

        if is_lumped_to_distributed:
            self.pp.logger.info("Using delineated catchments for lumped-to-distributed routing")

            # For lumped-to-distributed, SUMMA output is converted to gru/gruId format
            # by the spatial_orchestrator, so mizuRoute control file should use gru/gruId
            self.pp.summa_uses_gru_runoff = True

            # Enable remapping: map single lumped SUMMA GRU to 25 routing HRUs with area weights
            self.pp.needs_remap_lumped_distributed = True

            # Load the delineated catchments shapefile (nested experiment layout)
            catchment_path = find_catchment_subfile(
                self.pp.project_dir / 'shapefiles',
                self.pp.domain_definition_method,
                self.pp.experiment_id,
                f"{self.pp.domain_name}_catchment_delineated.shp",
                logger=self.pp.logger,
            )
            if catchment_path is None:
                raise FileOperationError(
                    f"Delineated catchment shapefile not found for {self.pp.domain_name}"
                )

            shp_catchments = gpd.read_file(catchment_path)
            self.pp.logger.info(f"Loaded {len(shp_catchments)} delineated subcatchments")

            # Extract HRU data from delineated catchments
            hru_ids = shp_catchments['GRU_ID'].values.astype(int)

            # Check if we have a headwater basin (None values in river network)
            if self._check_if_headwater_basin(shp_river):
                # Create synthetic river network for headwater basin
                shp_river = self._create_synthetic_river_network(shp_river, hru_ids)

            # Use the delineated catchments as HRUs
            num_seg = len(shp_river)
            num_hru = len(shp_catchments)

            hru_to_seg_ids = shp_catchments[self.pp.basin_hru_to_seg_col].values.astype(int)
            if not np.isin(hru_to_seg_ids, shp_river[self.pp.river_segid_col].values).all():
                raise ValueError('Delineated routing subcatchments reference missing river segments')

            # Convert fractional areas to actual areas (multiply by total basin area)
            total_basin_area = shp_basin[self.pp.basin_area_col].sum()
            hru_areas = shp_catchments['avg_subbas'].values * total_basin_area

            # Store fractional areas for remapping
            self.pp.subcatchment_weights = shp_catchments['avg_subbas'].values
            self.pp.subcatchment_gru_ids = hru_ids

            self.pp.logger.info(f"Created {num_hru} HRUs from delineated catchments")
            self.pp.logger.info(f"Weight range: {self.pp.subcatchment_weights.min():.4f} to {self.pp.subcatchment_weights.max():.4f}")

        else:
            # Check if we have SUMMA attributes file with multiple HRUs per GRU
            attributes_path = self.pp.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'

            if attributes_path.exists():
                with nc4.Dataset(attributes_path, 'r') as attrs:
                    n_hrus = len(attrs.dimensions['hru'])
                    n_grus = len(attrs.dimensions['gru'])

                    if n_hrus > n_grus:
                        # Multiple HRUs per GRU - SUMMA will output GRU-level runoff
                        # mizuRoute should route at GRU level
                        self.pp.logger.info(f"Distributed SUMMA with {n_hrus} HRUs across {n_grus} GRUs")
                        self.pp.logger.info("Creating GRU-level topology for mizuRoute (SUMMA outputs averageRoutedRunoff at GRU level)")

                        # Read GRU information from SUMMA attributes file
                        gru_ids = attrs.variables['gruId'][:].astype(int)

                        # For distributed SUMMA, GRU IDs should match segment IDs
                        hru_ids = gru_ids  # mizuRoute will read GRU-level data
                        hru_to_seg_ids = gru_ids  # Each GRU drains to segment with same ID

                        # Calculate GRU areas by summing HRU areas within each GRU
                        hru2gru = attrs.variables['hru2gruId'][:].astype(int)
                        hru_areas_all = attrs.variables['HRUarea'][:].astype(float)

                        # Sum areas for each GRU
                        gru_areas = np.zeros(n_grus)
                        for i, gru_id in enumerate(gru_ids):
                            gru_mask = hru2gru == gru_id
                            gru_areas[i] = hru_areas_all[gru_mask].sum()

                        hru_areas = gru_areas

                        num_seg = len(shp_river)
                        num_hru = n_grus  # mizuRoute sees GRUs as HRUs

                        # Store flag for control file generation
                        self.pp.summa_uses_gru_runoff = True

                        self.pp.logger.info(f"Created topology with {num_hru} GRUs for mizuRoute routing")
                    else:
                        # Lumped modeling: use original logic
                        self.pp.summa_uses_gru_runoff = False
                        closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)

                        if len(shp_basin) == 1:
                            shp_basin.loc[0, self.pp.basin_hru_to_seg_col] = closest_segment_id
                            self.pp.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")

                        num_seg = len(shp_river)
                        num_hru = len(shp_basin)

                        hru_ids = shp_basin[self.pp.basin_gruid_col].values.astype(int)
                        hru_to_seg_ids = shp_basin[self.pp.basin_hru_to_seg_col].values.astype(int)
                        hru_areas = shp_basin[self.pp.basin_area_col].values.astype(float)
            else:
                # No attributes file: use original logic
                self.pp.summa_uses_gru_runoff = False
                closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)

                if len(shp_basin) == 1:
                    shp_basin.loc[0, self.pp.basin_hru_to_seg_col] = closest_segment_id
                    self.pp.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")

                num_seg = len(shp_river)
                num_hru = len(shp_basin)

                hru_ids = shp_basin[self.pp.basin_gruid_col].values.astype(int)
                hru_to_seg_ids = shp_basin[self.pp.basin_hru_to_seg_col].values.astype(int)
                hru_areas = shp_basin[self.pp.basin_area_col].values.astype(float)

        # Ensure minimum segment length - now safe from None values
        length_col = self.pp.river_length_col
        if length_col in shp_river.columns:
            # Convert None/null values to 0 first, then set minimum
            shp_river[length_col] = shp_river[length_col].fillna(0)
            shp_river.loc[shp_river[length_col] == 0, length_col] = 1

        # Ensure slope column has valid values
        slope_col = self.pp.river_slope_col
        if slope_col in shp_river.columns:
            shp_river[slope_col] = shp_river[slope_col].fillna(0.001)  # Default slope
            shp_river.loc[shp_river[slope_col] == 0, slope_col] = 0.001

        # Enforce outlets if specified
        make_outlet = self.pp.mizu_make_outlet
        if make_outlet and make_outlet != 'n/a':
            river_outlet_ids = [int(id) for id in make_outlet.split(',')]
            seg_id_col = self.pp.river_segid_col
            downseg_id_col = self.pp.river_downsegid_col

            for outlet_id in river_outlet_ids:
                if outlet_id in shp_river[seg_id_col].values:
                    shp_river.loc[shp_river[seg_id_col] == outlet_id, downseg_id_col] = 0
                else:
                    self.pp.logger.warning(f"Outlet ID {outlet_id} not found in river network")

        # Validate downstream segment references
        seg_id_col = self.pp.river_segid_col
        downseg_id_col = self.pp.river_downsegid_col
        valid_seg_ids = set(shp_river[seg_id_col].values.astype(int))

        invalid_refs = []
        for idx, row in shp_river.iterrows():
            seg_id = int(row[seg_id_col])
            down_seg_id = int(row[downseg_id_col])

            # Check if downstream ID is valid (either -9999 for outlet, or exists in segment list)
            if down_seg_id not in valid_seg_ids and down_seg_id != -9999:
                invalid_refs.append((seg_id, down_seg_id))
                # Fix: set invalid downstream references to -9999 (mizuRoute outlet convention)
                shp_river.loc[idx, downseg_id_col] = -9999

        if invalid_refs:
            self.pp.logger.warning(f"Fixed {len(invalid_refs)} invalid downstream segment references:")
            for seg_id, invalid_down_id in invalid_refs:
                self.pp.logger.warning(f"  Segment {seg_id} had invalid downstream ID {invalid_down_id} -> set to -9999 (outlet)")

        # Validate HRU-to-segment mapping
        invalid_hru_refs = []
        for i, hru_to_seg in enumerate(hru_to_seg_ids):
            if hru_to_seg not in valid_seg_ids:
                invalid_hru_refs.append((hru_ids[i], hru_to_seg))
                # Find the closest valid segment or use the first segment
                if len(valid_seg_ids) > 0:
                    # Use the segment with closest ID
                    closest_seg = min(valid_seg_ids, key=lambda x: abs(x - hru_to_seg))
                    hru_to_seg_ids[i] = closest_seg
                    self.pp.logger.warning(f"  HRU {hru_ids[i]} had invalid segment reference {hru_to_seg} -> set to {closest_seg}")

        if invalid_hru_refs:
            self.pp.logger.warning(f"Fixed {len(invalid_hru_refs)} invalid HRU-to-segment references")

        # Create the netCDF file
        with nc4.Dataset(self.pp.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_seg, num_hru)

            # Create segment variables (now safe from None values)
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.pp.river_segid_col].values.astype(int), 'Unique ID of each stream segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.pp.river_downsegid_col].values.astype(int), 'ID of the downstream segment', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.pp.river_slope_col].values.astype(float), 'Segment slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.pp.river_length_col].values.astype(float), 'Segment length', 'm')

            # Create HRU variables (using our computed values)
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids, 'Unique hru ID', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids, 'ID of the stream segment to which the HRU discharges', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas, 'HRU area', 'm^2')

        self.pp.logger.info(f"Network topology file created at {self.pp.setup_dir / topology_name}")

    # =========================================================================
    # Grid-based topology
    # =========================================================================

    def _create_grid_topology_file(self):
        """
        Create mizuRoute topology for grid-based distributed modeling.

        Each grid cell becomes both an HRU and a segment. D8 flow direction
        determines segment connectivity.
        """
        self.pp.logger.info("Creating grid-based network topology for distributed mode")

        # Load grid shapefile with D8 topology
        grid_path = self.pp.project_dir / 'shapefiles' / 'river_basins' / f"{self.pp.domain_name}_riverBasins_distribute.shp"

        if not grid_path.exists():
            self.pp.logger.error(f"Grid basins shapefile not found: {grid_path}")
            raise FileOperationError(f"Grid basins not found: {grid_path}")

        grid_gdf = gpd.read_file(grid_path)
        num_cells = len(grid_gdf)

        self.pp.logger.info(f"Loaded {num_cells} grid cells from {grid_path}")

        topology_name = self.pp.mizu_topology_file

        # Extract topology data from grid shapefile
        seg_ids = grid_gdf['GRU_ID'].values.astype(int)

        # Get downstream IDs from D8 topology
        # Note: shapefile truncates column names to 10 chars, so downstream_id becomes downstream
        if 'downstream_id' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream_id'].values.astype(int)
        elif 'downstream' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream'].values.astype(int)
        elif 'DSLINKNO' in grid_gdf.columns:
            down_seg_ids = grid_gdf['DSLINKNO'].values.astype(int)
        else:
            self.pp.logger.warning("No D8 topology found, setting all cells as outlets")
            down_seg_ids = np.zeros(num_cells, dtype=int)

        # Get slopes from grid
        if 'slope' in grid_gdf.columns:
            slopes = grid_gdf['slope'].values.astype(float)
            # Ensure minimum slope
            slopes = np.maximum(slopes, 0.001)
        else:
            self.pp.logger.warning("No slope data found, using default 0.01")
            slopes = np.full(num_cells, 0.01)

        # Get elevations from grid (for cycle breaking)
        if 'elev_mean' in grid_gdf.columns:
            elevations = grid_gdf['elev_mean'].values.astype(float)
        else:
            self.pp.logger.warning("No elevation data found, using 0.0")
            elevations = np.zeros(num_cells)

        # Fix cycles in topology
        down_seg_ids = self._fix_routing_cycles(seg_ids, down_seg_ids, elevations)

        # Validate downstream segment references
        valid_seg_ids = set(seg_ids)
        invalid_count = 0
        for i, down_seg_id in enumerate(down_seg_ids):
            # Check if downstream ID is valid (either -9999 for outlet, or exists in segment list)
            if down_seg_id not in valid_seg_ids and down_seg_id != -9999:
                invalid_count += 1
                down_seg_ids[i] = -9999  # Fix: set to outlet (mizuRoute convention)
                self.pp.logger.warning(f"Segment {seg_ids[i]} had invalid downstream ID {down_seg_id} -> set to -9999 (outlet)")

        if invalid_count > 0:
            self.pp.logger.warning(f"Fixed {invalid_count} invalid downstream segment references in grid topology")

        # Get cell size for segment length
        grid_cell_size = self.pp._get_config_value(lambda: self.pp.config.model.mizuroute.grid_cell_size, default=1000.0)
        lengths = np.full(num_cells, float(grid_cell_size))

        # HRU variables (each cell is also an HRU)
        hru_ids = seg_ids.copy()
        hru_to_seg_ids = seg_ids.copy()  # Each HRU drains to its own segment

        # Get HRU areas
        if 'GRU_area' in grid_gdf.columns:
            hru_areas = grid_gdf['GRU_area'].values.astype(float)
        else:
            self.pp.logger.warning("No area data found, using cell size squared")
            hru_areas = np.full(num_cells, grid_cell_size ** 2)

        # Create the netCDF topology file
        with nc4.Dataset(self.pp.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_cells, num_cells)

            # Create segment variables
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', seg_ids,
                                         'Unique ID of each grid cell segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', down_seg_ids,
                                         'ID of downstream grid cell (0=outlet)', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', slopes,
                                         'Grid cell slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', lengths,
                                         'Grid cell length (cell size)', 'm')

            # Create HRU variables
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids,
                                         'Unique HRU ID (=grid cell ID)', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids,
                                         'Segment to which HRU drains (=cell ID)', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas,
                                         'HRU area', 'm^2')

        # Count outlets for logging
        n_outlets = np.sum(down_seg_ids == 0)
        self.pp.logger.info(f"Grid topology created: {num_cells} cells, {n_outlets} outlets")
        self.pp.logger.info(f"Topology file: {self.pp.setup_dir / topology_name}")

        # Set flag for control file - grid cells use GRU-level runoff
        self.pp.summa_uses_gru_runoff = True

    # =========================================================================
    # Point-scale topology
    # =========================================================================

    def _create_point_topology_file(self):
        """
        Create mizuRoute topology for point-scale modeling.

        Point-scale domains have a single HRU and a single segment (outlet).
        """
        self.pp.logger.info("Creating point-scale network topology")

        topology_name = self.pp.mizu_topology_file
        if not topology_name:
            topology_name = "mizuRoute_topology.nc"

        # Single segment and HRU for point-scale domain
        seg_id = 1
        down_seg_id = 0  # Outlet
        hru_id = 1

        # Default values for point-scale
        slope = 0.01  # 1% slope default
        length = 100.0  # 100m default segment length
        area = 10000.0  # 1 hectare default area

        # Create the netCDF topology file
        with nc4.Dataset(self.pp.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, 1, 1)  # 1 segment, 1 HRU

            # Create segment variables
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', np.array([seg_id]),
                                         'Unique ID of segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', np.array([down_seg_id]),
                                         'ID of downstream segment (0=outlet)', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', np.array([slope]),
                                         'Segment slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', np.array([length]),
                                         'Segment length', 'm')

            # Create HRU variables
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', np.array([hru_id]),
                                         'Unique HRU ID', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', np.array([seg_id]),
                                         'Segment to which HRU drains', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', np.array([area]),
                                         'HRU area', 'm^2')

        self.pp.logger.info("Point-scale topology created: 1 HRU, 1 outlet segment")
        self.pp.logger.info(f"Topology file: {self.pp.setup_dir / topology_name}")

        # Set flag for control file - point-scale uses GRU-level runoff
        self.pp.summa_uses_gru_runoff = True

    # =========================================================================
    # Headwater basin handling
    # =========================================================================

    def _check_if_headwater_basin(self, shp_river):
        """
        Check if this is a headwater basin with None/invalid river network data.

        Args:
            shp_river: GeoDataFrame of river network

        Returns:
            bool: True if this appears to be a headwater basin with invalid network data
        """
        # Check for critical None values in key columns
        seg_id_col = self.pp.river_segid_col
        downseg_id_col = self.pp.river_downsegid_col

        if seg_id_col in shp_river.columns and downseg_id_col in shp_river.columns:
            # Check if all segment IDs are None/null
            seg_ids_null = shp_river[seg_id_col].isna().all()
            downseg_ids_null = shp_river[downseg_id_col].isna().all()

            if seg_ids_null and downseg_ids_null:
                self.pp.logger.info("Detected headwater basin: all river network IDs are None/null")
                return True

            # Also check for string 'None' values (sometimes shapefiles store None as string)
            if shp_river[seg_id_col].dtype == 'object':
                seg_ids_none_str = (shp_river[seg_id_col] == 'None').all()
                downseg_ids_none_str = (shp_river[downseg_id_col] == 'None').all()

                if seg_ids_none_str and downseg_ids_none_str:
                    self.pp.logger.info("Detected headwater basin: all river network IDs are 'None' strings")
                    return True

        return False

    def _create_synthetic_river_network(self, shp_river, hru_ids):
        """
        Create a synthetic single-segment river network for headwater basins.

        Args:
            shp_river: Original GeoDataFrame (with None values)
            hru_ids: Array of HRU IDs from delineated catchments

        Returns:
            GeoDataFrame: Modified river network with synthetic single segment
        """
        self.pp.logger.info("Creating synthetic river network for headwater basin")

        # Use the first HRU ID as the segment ID (should be reasonable identifier)
        synthetic_seg_id = int(hru_ids[0]) if len(hru_ids) > 0 else 1

        # Create synthetic values for the single segment
        synthetic_data = {
            self.pp.river_segid_col: synthetic_seg_id,
            self.pp.river_downsegid_col: 0,  # Outlet (downstream ID = 0)
            self.pp.river_length_col: 1000.0,  # Default 1 km length
            self.pp.river_slope_col: 0.001,  # Default 0.1% slope
        }

        # Get the geometry column name (usually 'geometry')
        geom_col = shp_river.geometry.name

        # Create a simple point geometry at the centroid of the original (if it exists)
        if not shp_river.empty and shp_river.geometry.iloc[0] is not None:
            # Use the centroid of the first geometry, handling CRS projection via mixin
            synthetic_geom = self.pp.calculate_feature_centroids(shp_river.iloc[[0]]).iloc[0]
        else:
            # Create a default point geometry (this won't be used for actual routing)
            from shapely.geometry import Point
            synthetic_geom = Point(0, 0)

        synthetic_data[geom_col] = synthetic_geom

        # Create new GeoDataFrame with single row
        synthetic_gdf = gpd.GeoDataFrame([synthetic_data], crs=shp_river.crs)

        self.pp.logger.info(f"Created synthetic river network: segment ID {synthetic_seg_id} (outlet)")

        return synthetic_gdf

    # =========================================================================
    # Pour point and segment lookup
    # =========================================================================

    def _find_closest_segment_to_pour_point(self, shp_river):
        """
        Find the river segment closest to the pour point.

        Args:
            shp_river: GeoDataFrame of river network

        Returns:
            int: Segment ID of closest segment to pour point
        """

        # Find pour point shapefile
        pour_point_dir = self.pp.project_dir / 'shapefiles' / 'pour_point'
        pour_point_files = list(pour_point_dir.glob('*.shp'))

        if not pour_point_files:
            self.pp.logger.error(f"No pour point shapefiles found in {pour_point_dir}")
            # Fallback: use outlet segment (downSegId == 0)
            outlet_mask = shp_river[self.pp.river_downsegid_col] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.pp.river_segid_col].iloc[0]
                self.pp.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                # Last resort: use first segment
                fallback_seg = shp_river[self.pp.river_segid_col].iloc[0]
                self.pp.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg

        # Load first pour point file
        pour_point_file = pour_point_files[0]
        self.pp.logger.info(f"Loading pour point from {pour_point_file}")

        try:
            shp_pour_point = gpd.read_file(pour_point_file)

            # Ensure both are in the same CRS
            if shp_river.crs != shp_pour_point.crs:
                shp_pour_point = shp_pour_point.to_crs(shp_river.crs)

            # Get pour point coordinates (assume first/only point)
            shp_pour_point.geometry.iloc[0]

            # Calculate distances from pour point to all river segments
            shp_river_proj = shp_river.to_crs(shp_river.estimate_utm_crs())
            # Use mixin to get pour point centroid safely if needed (though it's a point)
            pour_point_centroids = self.pp.calculate_feature_centroids(shp_pour_point.iloc[[0]])
            pour_point_proj = pour_point_centroids.to_crs(shp_river_proj.crs)
            distances = shp_river_proj.geometry.distance(pour_point_proj.iloc[0])

            # Find closest segment
            closest_idx = distances.idxmin()
            closest_segment_id = shp_river.loc[closest_idx, self.pp.river_segid_col]

            self.pp.logger.info(f"Closest segment to pour point: {closest_segment_id} (distance: {distances.iloc[closest_idx]:.1f} units)")

            return closest_segment_id

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.pp.logger.error(f"Error finding closest segment: {str(e)}", exc_info=True)
            # Fallback to outlet segment
            outlet_mask = shp_river[self.pp.river_downsegid_col] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.pp.river_segid_col].iloc[0]
                self.pp.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                fallback_seg = shp_river[self.pp.river_segid_col].iloc[0]
                self.pp.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg

    # =========================================================================
    # Cycle detection and fixing
    # =========================================================================

    def _fix_routing_cycles(self, seg_ids, down_seg_ids, elevations):
        """
        Detect and fix cycles in the routing graph.

        For each cycle found, the node with the lowest elevation is forced
        to be an outlet (downSegId = 0).

        Args:
            seg_ids: Array of segment IDs
            down_seg_ids: Array of downstream segment IDs
            elevations: Array of segment elevations

        Returns:
            Fixed down_seg_ids array
        """
        self.pp.logger.info("Checking for cycles in routing topology...")

        # Create mapping from ID to index
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        # Adjacency list (node_idx -> downstream_node_idx)
        # Use -1 for outlet/external
        adj = {}
        for i, down_sid in enumerate(down_seg_ids):
            if down_sid in id_to_idx:
                adj[i] = id_to_idx[down_sid]
            else:
                adj[i] = -1

        visited = set()
        path_set = set()
        path_stack = []
        cycles_found = 0
        fixed_down_ids = down_seg_ids.copy()

        def visit(u):
            nonlocal cycles_found

            [(u, iter(adj.get(u, []) if u in adj and adj[u] != -1 else []))]
            path_set.add(u)
            path_stack.append(u)
            visited.add(u)

            # Iterative DFS to avoid recursion depth issues
            curr = u
            while True:
                neighbor = adj.get(curr, -1)

                if neighbor == -1:
                    # End of path
                    path_set.remove(curr)
                    path_stack.pop()
                    if not path_stack:
                        break
                    curr = path_stack[-1]
                    continue

                if neighbor in path_set:
                    # Cycle detected
                    cycle_nodes_idx = []
                    # Extract cycle from path_stack
                    try:
                        start_pos = path_stack.index(neighbor)
                        cycle_nodes_idx = path_stack[start_pos:]
                    except ValueError:
                        pass # Should not happen

                    if cycle_nodes_idx:
                        cycles_found += 1

                        # Find node with lowest elevation in cycle
                        min_elev = float('inf')
                        sink_node_idx = -1

                        for idx in cycle_nodes_idx:
                            elev = elevations[idx]
                            if elev < min_elev:
                                min_elev = elev
                                sink_node_idx = idx

                        # Break cycle: make sink_node an outlet
                        if sink_node_idx != -1:
                            fixed_down_ids[sink_node_idx] = 0
                            # Update adjacency to reflect break for future traversals
                            adj[sink_node_idx] = -1

                    # Backtrack
                    path_set.remove(curr)
                    path_stack.pop()
                    if not path_stack:
                        break
                    curr = path_stack[-1]
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    path_set.add(neighbor)
                    path_stack.append(neighbor)
                    curr = neighbor
                else:
                    # Already visited, not a cycle
                    path_set.remove(curr)
                    path_stack.pop()
                    if not path_stack:
                        break
                    curr = path_stack[-1]

        # Iterative DFS wrapper
        # The above nested function approach was a bit mix of recursive/iterative thinking.
        # Let's implement a clean iterative DFS.

        visited = set()
        path_set = set()

        for start_node_idx in range(len(seg_ids)):
            if start_node_idx in visited:
                continue

            stack = [(start_node_idx, 0)] # node_idx, state (0: enter, 1: exit)

            while stack:
                u, state = stack[-1]

                if state == 0:
                    visited.add(u)
                    path_set.add(u)
                    stack[-1] = (u, 1) # Next time we see u, we are exiting

                    v = adj.get(u, -1)
                    if v != -1:
                        if v in path_set:
                            # Cycle detected
                            cycles_found += 1

                            # Trace back stack to find cycle
                            cycle_indices = []
                            for node, _ in reversed(stack):
                                cycle_indices.append(node)
                                if node == v:
                                    break

                            # Find lowest elevation
                            min_elev = float('inf')
                            sink_idx = -1
                            for idx in cycle_indices:
                                if elevations[idx] < min_elev:
                                    min_elev = elevations[idx]
                                    sink_idx = idx

                            # Break cycle
                            fixed_down_ids[sink_idx] = 0
                            adj[sink_idx] = -1 # Update graph

                            # No need to continue this path as it's broken
                            # But we continue DFS to find other components

                        elif v not in visited:
                            stack.append((v, 0))
                else:
                    path_set.remove(u)
                    stack.pop()

        if cycles_found > 0:
            self.pp.logger.warning(f"Detected and fixed {cycles_found} cycles in routing topology.")
        else:
            self.pp.logger.info("No cycles detected in routing topology.")

        return fixed_down_ids

    # =========================================================================
    # NetCDF helpers
    # =========================================================================

    def _set_topology_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a river network .nc file for mizuRoute routing')

    def _create_topology_dimensions(self, ncid, num_seg, num_hru):
        ncid.createDimension('seg', num_seg)
        ncid.createDimension('hru', num_hru)

    def _create_topology_variables(self, ncid, shp_river, shp_basin):
        self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.pp.river_segid_col].values.astype(int), 'Unique ID of each stream segment', '-')
        self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.pp.river_downsegid_col].values.astype(int), 'ID of the downstream segment', '-')
        self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.pp.river_slope_col].values.astype(float), 'Segment slope', '-')
        self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.pp.river_length_col].values.astype(float), 'Segment length', 'm')
        self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', shp_basin[self.pp.basin_gruid_col].values.astype(int), 'Unique hru ID', '-')
        self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', shp_basin[self.pp.basin_hru_to_seg_col].values.astype(int), 'ID of the stream segment to which the HRU discharges', '-')
        self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', shp_basin[self.pp.basin_area_col].values.astype(float), 'HRU area', 'm^2')

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, fill_data, long_name, units):
        ncvar = ncid.createVariable(var_name, var_type, (dim,))
        ncvar[:] = fill_data
        ncvar.long_name = long_name
        ncvar.units = units
