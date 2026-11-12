# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Grid-based domain delineation for fully distributed modeling.

Creates a regular mesh grid for distributed SUMMA modeling with D8 flow
direction routing via mizuRoute.

Refactored from plan (2026-01-06)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from symfluence.core.exceptions import DiscretizationError
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from ....geospatial.raster_utils import _scipy_mode_compat
from ..base.base_delineator import BaseGeofabricDelineator
from ..processors.gdal_processor import GDALProcessor
from ..processors.taudem_executor import TauDEMExecutor

# D8 flow direction encoding (TauDEM convention: 1=E, 2=NE, 3=N, ... 8=SE,
# counter-clockwise from east).
#
# Maps direction code to (row_offset, col_offset) of the downstream GRID cell.
# The grid ``row`` index is NOT a raster row: every grid built in this module
# numbers rows from the southern edge upward (``y = miny + row * size`` in
# ``_create_grid_from_bbox``; ascending latitude in ``_create_native_grid``),
# so ``row + 1`` is the cell to the NORTH and ``col + 1`` the cell to the EAST.
D8_OFFSETS = {
    1: (0, 1),    # East
    2: (1, 1),    # Northeast
    3: (1, 0),    # North
    4: (1, -1),   # Northwest
    5: (0, -1),   # West
    6: (-1, -1),  # Southwest
    7: (-1, 0),   # South
    8: (-1, 1),   # Southeast
}


@R.delineation_strategies.add('distributed')
class GridDelineator(BaseGeofabricDelineator):
    """
    Creates a regular mesh grid for fully distributed modeling.

    Grid cells serve as both GRUs and HRUs for SUMMA, with D8 flow direction
    routing topology for mizuRoute.

    Attributes:
        grid_cell_size: Size of grid cells in meters
        clip_to_watershed: Whether to clip grid to watershed boundary
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize grid delineator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Grid-specific configuration
        self.grid_cell_size = resolve_config_value(
            config,
            lambda: config.domain.grid_cell_size,
            dict_key='GRID_CELL_SIZE',
        )  # meters
        self.clip_to_watershed = resolve_config_value(
            config,
            lambda: config.domain.clip_grid_to_watershed,
            dict_key='CLIP_GRID_TO_WATERSHED',
        )
        self.grid_source = resolve_config_value(config, lambda: config.domain.grid_source, dict_key='GRID_SOURCE')
        self.native_grid_dataset = resolve_config_value(
            config,
            lambda: config.domain.native_grid_dataset,
            dict_key='NATIVE_GRID_DATASET',
        )

        # Specific paths for grid delineation
        self.output_dir = self.project_dir / "shapefiles/tempdir"
        self.interim_dir = self.project_dir / "taudem-interim-files" / "grid"

        # Initialize processors
        self.taudem = TauDEMExecutor(config, logger, self.taudem_dir)
        self.gdal = GDALProcessor(logger)

    def _get_delineation_method_name(self) -> str:
        """Return method name for output files.

        Encodes grid configuration in the suffix:
        - generate mode: distributed_{cellsize}m
        - native mode: distributed_native_{dataset}
        - subset mode: distributed_subset_{geofabric}
        """
        return self._get_method_suffix()

    def create_grid_domain(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Create a grid-based domain for distributed modeling.

        Returns:
            Tuple of (river_network_path, river_basins_path)
        """
        self.logger.info(f"Creating grid domain for: {self.domain_name}")
        self.logger.info(f"Grid cell size: {self.grid_cell_size}m")
        self.logger.info(f"Clip to watershed: {self.clip_to_watershed}")

        # Define output paths
        method_suffix = self._get_method_suffix()
        river_basins_path = (
            self.project_dir / "shapefiles" / "river_basins" /
            f"{self.domain_name}_riverBasins_{method_suffix}.shp"
        )
        river_network_path = (
            self.project_dir / "shapefiles" / "river_network" /
            f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
        )

        # Create directories
        river_basins_path.parent.mkdir(parents=True, exist_ok=True)
        river_network_path.parent.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Create grid from bounding box
        grid_gdf = self._create_grid_from_bbox()
        if grid_gdf is None or len(grid_gdf) == 0:
            self.logger.error("Failed to create grid from bounding box")
            return None, None

        self.logger.info(f"Created initial grid with {len(grid_gdf)} cells")

        # Step 2: Optionally clip to watershed boundary
        if self.clip_to_watershed:
            grid_gdf = self._clip_grid_to_watershed(grid_gdf)
            if grid_gdf is None or len(grid_gdf) == 0:
                self.logger.error("Failed to clip grid to watershed")
                return None, None
            self.logger.info(f"Grid after clipping: {len(grid_gdf)} cells")

        # Step 3: Compute D8 flow directions
        d8_path = self._compute_d8_flow_directions()
        if d8_path is None:
            self.logger.error("Failed to compute D8 flow directions")
            return None, None

        # Step 4: Extract D8 topology for each grid cell
        grid_gdf = self._extract_grid_d8_topology(grid_gdf, d8_path)

        # Step 5: Add additional grid attributes (elevation, slope, centroids)
        grid_gdf = self._add_grid_attributes(grid_gdf)

        # Step 5b: The dominant-direction topology is not guaranteed acyclic
        # (two neighbouring cells can each have the other as dominant
        # direction). Break any cycle before the network is written.
        grid_gdf = self._resolve_topology_cycles(grid_gdf)

        # Step 6: Save grid as river basins shapefile
        grid_gdf.to_file(river_basins_path)
        self.logger.info(f"Saved grid basins to: {river_basins_path}")

        # Step 7: Create synthetic river network from grid topology
        network_gdf = self._create_river_network_from_grid(grid_gdf)
        network_gdf.to_file(river_network_path)
        self.logger.info(f"Saved grid network to: {river_network_path}")

        # Cleanup intermediate files if requested
        if self._get_config_value(
            lambda: self.config.domain.delineation.cleanup_intermediate_files,
            dict_key='CLEANUP_INTERMEDIATE_FILES',
        ):
            self.cleanup()

        return river_network_path, river_basins_path

    def _create_grid_from_bbox(self) -> Optional[gpd.GeoDataFrame]:
        """
        Create regular grid cells from bounding box coordinates.

        Returns:
            GeoDataFrame with grid cell polygons
        """
        try:
            # Parse bounding box (format: lat_max/lon_min/lat_min/lon_max)
            bbox_coords = self._get_config_value(lambda: self.config.domain.bounding_box_coords, default="", dict_key='BOUNDING_BOX_COORDS')
            if not bbox_coords:
                self.logger.error("BOUNDING_BOX_COORDS not found in configuration")
                return None

            try:
                lat_max, lon_min, lat_min, lon_max = map(float, bbox_coords.split("/"))
            except ValueError:
                self.logger.error(
                    f"Invalid bounding box format: {bbox_coords}. "
                    "Expected format: lat_max/lon_min/lat_min/lon_max"
                )
                return None

            # Create bounding box polygon
            bbox_polygon = box(lon_min, lat_min, lon_max, lat_max)
            bounds_gdf = gpd.GeoDataFrame(
                geometry=[bbox_polygon],
                crs='EPSG:4326'
            )

            # Convert to UTM for accurate cell sizing
            utm_crs = bounds_gdf.estimate_utm_crs()
            bounds_utm = bounds_gdf.to_crs(utm_crs)

            # Get UTM bounds
            minx, miny, maxx, maxy = bounds_utm.total_bounds

            # Generate grid cells
            cells = []
            cell_id = 1

            # Calculate number of rows and columns
            n_cols = int(np.ceil((maxx - minx) / self.grid_cell_size))
            n_rows = int(np.ceil((maxy - miny) / self.grid_cell_size))

            self.logger.info(f"Creating grid: {n_rows} rows x {n_cols} cols")

            for row in range(n_rows):
                for col in range(n_cols):
                    x = minx + col * self.grid_cell_size
                    y = miny + row * self.grid_cell_size

                    cell = box(
                        x, y,
                        x + self.grid_cell_size,
                        y + self.grid_cell_size
                    )

                    cells.append({
                        'geometry': cell,
                        'GRU_ID': cell_id,
                        'row': row,
                        'col': col
                    })
                    cell_id += 1

            # Create GeoDataFrame in UTM
            grid_gdf = gpd.GeoDataFrame(cells, crs=utm_crs)

            # Calculate area in square meters
            grid_gdf['GRU_area'] = grid_gdf.geometry.area

            # Convert back to WGS84
            grid_gdf = grid_gdf.to_crs('EPSG:4326')

            return grid_gdf

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating grid from bounding box: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _clip_grid_to_watershed(self, grid_gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
        """
        Clip grid cells to watershed boundary.

        First delineates the watershed using the lumped method, then clips
        grid cells to fit within the watershed.

        Args:
            grid_gdf: GeoDataFrame with grid cells

        Returns:
            Clipped GeoDataFrame
        """
        try:
            from .lumped_delineator import LumpedWatershedDelineator

            self.logger.info("Delineating watershed for grid clipping")

            # Delineate watershed
            lumped = LumpedWatershedDelineator(self.config, self.logger)
            _, basin_path = lumped.delineate_lumped_watershed()

            if basin_path is None or not basin_path.exists():
                # Falling back to the full bounding-box grid silently produces a
                # DIFFERENT DOMAIN than the one requested — the bbox grid is far
                # larger than the watershed (observed: 6900 cells instead of
                # 2332, a 3x-too-large domain) — and the workflow then reports
                # success. That is a wrong result presented as a right one: the
                # published cell count is a headline number of the domain
                # experiment. Clipping was explicitly requested, so a failed
                # delineation is a hard error.
                raise DiscretizationError(
                    "Grid clipping was requested (CLIP_GRID_TO_WATERSHED) but "
                    "the watershed could not be delineated, so the grid cannot "
                    "be clipped. Refusing to fall back to the unclipped "
                    "bounding-box grid, which would silently yield a much "
                    "larger domain than requested. Fix the delineation (check "
                    "the TauDEM executables and the DEM) and re-run, or set "
                    "CLIP_GRID_TO_WATERSHED=false to intentionally model the "
                    "full bounding box."
                )

            # Load watershed
            watershed = gpd.read_file(basin_path)

            # Ensure same CRS
            if watershed.crs != grid_gdf.crs:
                watershed = watershed.to_crs(grid_gdf.crs)

            # Clip grid cells to watershed
            clipped = gpd.overlay(grid_gdf, watershed[['geometry']], how='intersection')

            if len(clipped) == 0:
                self.logger.warning("No grid cells intersect watershed, using full grid")
                return grid_gdf

            # Recalculate areas after clipping
            utm_crs = clipped.estimate_utm_crs()
            clipped_utm = clipped.to_crs(utm_crs)
            clipped['GRU_area'] = clipped_utm.geometry.area

            # Remove tiny slivers (less than 10% of target cell size)
            min_area = 0.1 * (self.grid_cell_size ** 2)
            original_count = len(clipped)
            clipped = clipped[clipped['GRU_area'] >= min_area]
            removed_count = original_count - len(clipped)

            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} sliver cells (area < {min_area:.0f} m^2)")

            # Reassign sequential IDs
            clipped = clipped.reset_index(drop=True)
            clipped['GRU_ID'] = range(1, len(clipped) + 1)

            return clipped

        except DiscretizationError:
            raise
        except Exception as e:  # noqa: BLE001 — re-raised as a domain error below
            # Returning the unclipped grid here was the silent-failure path:
            # ANY clip error yielded the full bounding-box mesh — a different,
            # much larger domain than requested — and the workflow still
            # reported success (observed: 6900 cells instead of 2332 when the
            # TauDEM executables were missing). A domain that silently changes
            # size is a wrong result presented as a right one, so surface it.
            self.logger.error(f"Error clipping grid to watershed: {str(e)}", exc_info=True)
            raise DiscretizationError(
                f"Failed to clip the grid to the watershed ({type(e).__name__}: {e}). "
                "Refusing to fall back to the unclipped bounding-box grid, which "
                "would silently model a much larger domain than requested. Fix the "
                "delineation and re-run, or set CLIP_GRID_TO_WATERSHED=false to "
                "intentionally model the full bounding box."
            ) from e

    def _compute_d8_flow_directions(self) -> Optional[Path]:
        """
        Compute D8 flow directions using TauDEM.

        Returns:
            Path to D8 flow direction raster
        """
        try:
            self.logger.info("Computing D8 flow directions")

            # Create interim directory
            self.interim_dir.mkdir(parents=True, exist_ok=True)

            # Check DEM exists
            if not self.dem_path.exists():
                self.logger.error(f"DEM not found: {self.dem_path}")
                return None

            # Get MPI command if available
            mpi_cmd = self.taudem.get_mpi_command()
            mpi_prefix = f"{mpi_cmd} -n {self.num_processes} " if mpi_cmd else ""

            # TauDEM processing steps
            fel_path = self.interim_dir / "elv-fel.tif"
            sd8_path = self.interim_dir / "elv-sd8.tif"
            d8_path = self.interim_dir / "elv-fdir.tif"

            steps = [
                # Pit removal
                f"{mpi_prefix}{self.taudem_dir}/pitremove -z {self.dem_path} -fel {fel_path}",
                # D8 flow direction
                f"{mpi_prefix}{self.taudem_dir}/d8flowdir -fel {fel_path} -sd8 {sd8_path} -p {d8_path}",
            ]

            for step in steps:
                self.taudem.run_command(step)
                self.logger.debug(f"Completed TauDEM step: {step}")

            if not d8_path.exists():
                self.logger.error("D8 flow direction raster not created")
                return None

            self.logger.info(f"D8 flow directions computed: {d8_path}")
            return d8_path

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error computing D8 flow directions: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _extract_grid_d8_topology(
        self,
        grid_gdf: gpd.GeoDataFrame,
        d8_path: Path
    ) -> gpd.GeoDataFrame:
        """
        Extract dominant D8 flow direction for each grid cell.

        Args:
            grid_gdf: GeoDataFrame with grid cells
            d8_path: Path to D8 flow direction raster

        Returns:
            GeoDataFrame with downstream_id column added
        """
        try:
            self.logger.info("Extracting D8 topology for grid cells")

            # Create lookup from (row, col) to GRU_ID
            cell_lookup = {
                (int(row['row']), int(row['col'])): int(row['GRU_ID'])
                for _, row in grid_gdf.iterrows()
            }

            # Read D8 flow direction raster
            with rasterio.open(d8_path) as src:
                d8_data = src.read(1)
                transform = src.transform
                d8_crs = src.crs

            # Ensure grid is in same CRS as raster
            grid_raster_crs = grid_gdf.to_crs(d8_crs)

            downstream_ids = []
            n_leaving_domain = 0

            for idx, cell in grid_raster_crs.iterrows():
                # Get cell bounds
                bounds = cell.geometry.bounds

                # Convert bounds to pixel coordinates
                col_start = int((bounds[0] - transform.c) / transform.a)
                col_end = int((bounds[2] - transform.c) / transform.a)
                row_start = int((bounds[3] - transform.f) / transform.e)
                row_end = int((bounds[1] - transform.f) / transform.e)

                # Ensure valid bounds
                col_start = max(0, col_start)
                row_start = max(0, row_start)
                col_end = min(d8_data.shape[1], col_end)
                row_end = min(d8_data.shape[0], row_end)

                # Extract D8 values within cell
                if col_end > col_start and row_end > row_start:
                    cell_d8 = d8_data[row_start:row_end, col_start:col_end]

                    # Get mode (most common direction), excluding nodata (0)
                    valid_d8 = cell_d8[(cell_d8 > 0) & (cell_d8 <= 8)]

                    if len(valid_d8) > 0:
                        # Use compatibility wrapper for scipy version differences
                        mode_result = _scipy_mode_compat(valid_d8.flatten(), axis=0)
                        dominant_direction = int(mode_result.flat[0] if mode_result.size > 0 else 0)
                    else:
                        dominant_direction = 0
                else:
                    dominant_direction = 0

                # Get original grid row/col
                grid_row = int(grid_gdf.loc[idx, 'row'])
                grid_col = int(grid_gdf.loc[idx, 'col'])

                # Determine downstream cell based on D8 direction
                if dominant_direction in D8_OFFSETS:
                    drow, dcol = D8_OFFSETS[dominant_direction]
                    downstream_row = grid_row + drow
                    downstream_col = grid_col + dcol
                    downstream_id = cell_lookup.get((downstream_row, downstream_col), 0)
                    if downstream_id == 0:
                        # Flows into a cell outside the grid (domain edge, or a
                        # sliver removed by the watershed clip): becomes an outlet.
                        n_leaving_domain += 1
                else:
                    downstream_id = 0  # Outlet or no data

                downstream_ids.append(downstream_id)

            # Use 'downstream' to fit shapefile 10-char column name limit
            grid_gdf['downstream'] = downstream_ids

            # Count outlets
            n_outlets = sum(1 for d in downstream_ids if d == 0)
            self.logger.info(f"Grid topology: {len(grid_gdf)} cells, {n_outlets} outlets")
            if n_leaving_domain > 0:
                self.logger.info(
                    f"{n_leaving_domain} grid cell(s) drain to a cell outside the grid "
                    "(domain edge or clipped-away sliver) and were made outlets; "
                    f"{n_outlets - n_leaving_domain} cell(s) have no valid D8 direction."
                )

            return grid_gdf

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error extracting D8 topology: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            # Set all cells as outlets on error
            grid_gdf['downstream'] = 0
            return grid_gdf

    def _resolve_topology_cycles(self, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Detect and break cycles in the cell-to-cell ``downstream`` topology.

        Each cell has at most one downstream cell, so every cycle is a simple
        loop. A loop is broken at its lowest cell (ties: smallest GRU_ID): that
        cell is redirected to its lowest 8-neighbour that does not drain back
        into the loop and is not higher than the cell itself; when there is no
        such neighbour the cell becomes an outlet (``downstream = 0``).

        Raises:
            DiscretizationError: if cycles remain after resolution.
        """
        if 'downstream' not in grid_gdf.columns or len(grid_gdf) == 0:
            return grid_gdf

        ids = [int(v) for v in grid_gdf['GRU_ID']]
        down = dict(zip(ids, (int(v) for v in grid_gdf['downstream'])))

        def find_cycles() -> list:
            state: Dict[int, int] = {}  # 1 = on current walk, 2 = finished
            cycles = []
            for start in ids:
                if state.get(start):
                    continue
                path = []
                node = start
                while node in down and node != 0 and not state.get(node):
                    state[node] = 1
                    path.append(node)
                    node = down[node]
                if node in down and state.get(node) == 1:
                    cycles.append(path[path.index(node):])
                for visited in path:
                    state[visited] = 2
            return cycles

        cycles = find_cycles()
        if not cycles:
            return grid_gdf

        has_elev = 'elev_mean' in grid_gdf.columns
        elev = (
            dict(zip(ids, (float(v) for v in grid_gdf['elev_mean'])))
            if has_elev else {i: 0.0 for i in ids}
        )
        has_rc = 'row' in grid_gdf.columns and 'col' in grid_gdf.columns
        rc_of: Dict[int, Tuple[int, int]] = {}
        id_at: Dict[Tuple[int, int], int] = {}
        if has_rc:
            for gid, r, c in zip(ids, grid_gdf['row'], grid_gdf['col']):
                rc_of[gid] = (int(r), int(c))
                id_at[(int(r), int(c))] = gid

        def drains_into(start: int, targets: set) -> bool:
            seen = set()
            node = start
            while node in down and node != 0 and node not in seen:
                if node in targets:
                    return True
                seen.add(node)
                node = down[node]
            return False

        n_redirected = 0
        n_outlets = 0
        for cycle in cycles:
            members = set(cycle)
            breaker = min(cycle, key=lambda g: (elev[g], g))
            new_down = 0
            if has_rc:
                r0, c0 = rc_of[breaker]
                candidates = []
                for drow, dcol in D8_OFFSETS.values():
                    nb = id_at.get((r0 + drow, c0 + dcol))
                    if nb is None or nb in members or elev[nb] > elev[breaker]:
                        continue
                    if drains_into(nb, members):
                        continue
                    candidates.append((elev[nb], nb))
                if candidates:
                    new_down = min(candidates)[1]
            down[breaker] = new_down
            if new_down == 0:
                n_outlets += 1
            else:
                n_redirected += 1

        remaining = find_cycles()
        if remaining:
            raise DiscretizationError(
                f"Grid D8 topology still contains {len(remaining)} cycle(s) after "
                f"resolution (e.g. GRU_IDs {remaining[0][:10]}); refusing to write a "
                "cyclic routing network."
            )

        self.logger.warning(
            f"Grid D8 topology contained {len(cycles)} cycle(s) (cells whose dominant "
            f"flow directions point at each other). Broke each at its lowest cell: "
            f"{n_redirected} redirected to a lower neighbour, {n_outlets} turned into "
            "internal outlets (no lower neighbour outside the loop). Example loop "
            f"GRU_IDs: {cycles[0][:10]}"
        )
        grid_gdf['downstream'] = [down[i] for i in ids]
        return grid_gdf

    def _add_grid_attributes(self, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add additional attributes to grid cells.

        Calculates elevation, slope, and centroid coordinates for each cell.

        Args:
            grid_gdf: GeoDataFrame with grid cells

        Returns:
            GeoDataFrame with additional attributes
        """
        try:
            self.logger.info("Adding grid cell attributes")

            # Calculate centroids (reproject to projected CRS if geographic)
            if grid_gdf.crs and grid_gdf.crs.is_geographic:
                # Use Web Mercator for centroid calculation
                grid_projected = grid_gdf.to_crs('EPSG:3857')
                centroids_projected = grid_projected.geometry.centroid
                centroids = centroids_projected.to_crs(grid_gdf.crs)
            else:
                centroids = grid_gdf.geometry.centroid
            grid_gdf['center_lon'] = centroids.x
            grid_gdf['center_lat'] = centroids.y

            # Add gru_to_seg (same as GRU_ID for grid cells)
            grid_gdf['gru_to_seg'] = grid_gdf['GRU_ID']

            # Extract elevation from DEM if available
            if self.dem_path.exists():
                with rasterio.open(self.dem_path) as src:
                    dem_data = src.read(1)
                    transform = src.transform
                    dem_crs = src.crs
                    nodata = src.nodata

                # Convert grid to DEM CRS
                grid_dem_crs = grid_gdf.to_crs(dem_crs)

                elevations = []
                slopes = []

                for idx, cell in grid_dem_crs.iterrows():
                    bounds = cell.geometry.bounds

                    # Convert to pixel coordinates
                    col_start = int((bounds[0] - transform.c) / transform.a)
                    col_end = int((bounds[2] - transform.c) / transform.a)
                    row_start = int((bounds[3] - transform.f) / transform.e)
                    row_end = int((bounds[1] - transform.f) / transform.e)

                    # Ensure valid bounds
                    col_start = max(0, col_start)
                    row_start = max(0, row_start)
                    col_end = min(dem_data.shape[1], col_end)
                    row_end = min(dem_data.shape[0], row_end)

                    if col_end > col_start and row_end > row_start:
                        cell_dem = dem_data[row_start:row_end, col_start:col_end]

                        # Exclude nodata
                        if nodata is not None:
                            valid_dem = cell_dem[cell_dem != nodata]
                        else:
                            valid_dem = cell_dem.flatten()

                        if len(valid_dem) > 0:
                            mean_elev = float(np.mean(valid_dem))
                            # Estimate slope from elevation range
                            elev_range = float(np.max(valid_dem) - np.min(valid_dem))
                            cell_slope = elev_range / self.grid_cell_size
                        else:
                            mean_elev = 0.0
                            cell_slope = 0.01
                    else:
                        mean_elev = 0.0
                        cell_slope = 0.01

                    elevations.append(mean_elev)
                    slopes.append(max(cell_slope, 0.001))  # Minimum slope

                grid_gdf['elev_mean'] = elevations
                grid_gdf['slope'] = slopes
            else:
                self.logger.warning("DEM not found, using default elevation/slope")
                grid_gdf['elev_mean'] = 0.0
                grid_gdf['slope'] = 0.01

            return grid_gdf

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error adding grid attributes: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            # Add default values on error
            grid_gdf['center_lon'] = grid_gdf.geometry.centroid.x
            grid_gdf['center_lat'] = grid_gdf.geometry.centroid.y
            grid_gdf['gru_to_seg'] = grid_gdf['GRU_ID']
            grid_gdf['elev_mean'] = 0.0
            grid_gdf['slope'] = 0.01
            return grid_gdf

    def _create_river_network_from_grid(
        self,
        grid_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Create synthetic river network from grid topology.

        Each grid cell becomes a "segment" with connectivity based on D8.

        Args:
            grid_gdf: GeoDataFrame with grid cells and D8 topology

        Returns:
            GeoDataFrame with river network segments
        """
        try:
            self.logger.info("Creating river network from grid topology")

            # Create line segments from cell centroids to downstream cell centroids
            segments = []

            # Create centroid lookup
            {
                int(row['GRU_ID']): row.geometry.centroid
                for _, row in grid_gdf.iterrows()
            }

            for _, cell in grid_gdf.iterrows():
                gru_id = int(cell['GRU_ID'])
                downstream_id = int(cell['downstream'])

                # Get cell centroid
                centroid = cell.geometry.centroid

                # For river network, use cell centroid as point geometry
                # (like the lumped delineator does with pour point)
                segments.append({
                    'geometry': centroid,
                    'LINKNO': gru_id,
                    'DSLINKNO': downstream_id,
                    'Length': self.grid_cell_size,
                    'Slope': cell.get('slope', 0.01),
                    'GRU_ID': gru_id
                })

            network_gdf = gpd.GeoDataFrame(segments, crs=grid_gdf.crs)

            self.logger.info(f"Created river network with {len(network_gdf)} segments")
            return network_gdf

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating river network: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            # Return empty network on error
            return gpd.GeoDataFrame(
                columns=['LINKNO', 'DSLINKNO', 'Length', 'Slope', 'GRU_ID', 'geometry'],
                crs=grid_gdf.crs
            )

    def _create_native_grid(self) -> Optional[gpd.GeoDataFrame]:
        """
        Create grid matching native forcing data resolution.

        Reads forcing file metadata to determine grid structure and creates
        polygons matching the forcing data grid cells. Filters cells to the
        domain bounding box.

        Returns:
            GeoDataFrame with grid cells matching native forcing resolution,
            or None if forcing data cannot be read.

        Supported datasets: any forcing file on a regular latitude/longitude
        grid with 1-D coordinate variables (e.g. ERA5, RDRS on a regular grid).
        Forcing on a projected/curvilinear grid with 2-D latitude/longitude
        (native CARRA / CERRA Lambert grids) is not supported and raises.

        The returned ``row`` index increases northward and ``col`` eastward
        (see ``D8_OFFSETS``); ``native_lat_idx`` / ``native_lon_idx`` index the
        forcing file's own coordinate arrays, whatever their order.

        Raises:
            DiscretizationError: if the latitude/longitude coordinates are not 1-D.
        """
        self.logger.info(f"Creating native grid from dataset: {self.native_grid_dataset}")

        try:
            # Step 1: Locate the forcing file
            forcing_path = self._locate_forcing_file()
            if forcing_path is None:
                self.logger.warning(
                    f"Could not locate forcing file for '{self.native_grid_dataset}'. "
                    "Use grid_source='generate' with explicit grid_cell_size instead."
                )
                return None

            self.logger.info(f"Reading grid structure from: {forcing_path}")

            # Step 2: Read grid structure from netCDF
            import xarray as xr

            with xr.open_dataset(forcing_path) as ds:
                # Detect coordinate names
                lat_name = self._detect_coord_name(ds, ['latitude', 'lat', 'y', 'rlat'])
                lon_name = self._detect_coord_name(ds, ['longitude', 'lon', 'x', 'rlon'])

                if lat_name is None or lon_name is None:
                    self.logger.error(
                        f"Could not detect lat/lon coordinates in {forcing_path}. "
                        f"Available coords: {list(ds.coords)}"
                    )
                    return None

                self.logger.info(f"Detected coordinates: lat='{lat_name}', lon='{lon_name}'")

                # Get coordinate arrays
                lat_vals = np.asarray(ds[lat_name].values)
                lon_vals = np.asarray(ds[lon_name].values)

            if lat_vals.ndim != 1 or lon_vals.ndim != 1:
                raise DiscretizationError(
                    f"GRID_SOURCE='native' needs a regular latitude/longitude grid with "
                    f"1-D coordinates, but '{lat_name}'/'{lon_name}' in {forcing_path} "
                    f"have shapes {lat_vals.shape}/{lon_vals.shape} (a projected or "
                    "curvilinear grid such as native CARRA/CERRA). Regrid the forcing "
                    "to a regular lat/lon grid, or use GRID_SOURCE='generate' with an "
                    "explicit GRID_CELL_SIZE."
                )

            # Process latitude in ascending order (row index increases
            # northward), remembering each value's index in the file.
            n_lat_native = len(lat_vals)
            lat_ascending = lat_vals[0] < lat_vals[-1] if n_lat_native > 1 else True
            if not lat_ascending:
                lat_vals = lat_vals[::-1]

            # Step 3: Parse bounding box to filter grid cells
            bbox_coords = self._get_config_value(
                lambda: self.config.domain.bounding_box_coords,
                default="",
                dict_key='BOUNDING_BOX_COORDS'
            )

            if bbox_coords:
                try:
                    lat_max, lon_min, lat_min, lon_max = map(float, bbox_coords.split("/"))
                except ValueError:
                    self.logger.warning(f"Invalid bounding box format: {bbox_coords}, using full grid")
                    lat_min, lat_max = lat_vals.min(), lat_vals.max()
                    lon_min, lon_max = lon_vals.min(), lon_vals.max()
            else:
                lat_min, lat_max = lat_vals.min(), lat_vals.max()
                lon_min, lon_max = lon_vals.min(), lon_vals.max()

            # Step 4: Calculate grid cell size (resolution)
            if len(lat_vals) > 1:
                lat_res = abs(lat_vals[1] - lat_vals[0])
            else:
                lat_res = 0.25  # Default ERA5 resolution

            if len(lon_vals) > 1:
                lon_res = abs(lon_vals[1] - lon_vals[0])
            else:
                lon_res = 0.25

            self.logger.info(f"Grid resolution: lat={lat_res:.4f}, lon={lon_res:.4f} degrees")

            # Step 5: Filter coordinates to bounding box with buffer
            buffer = max(lat_res, lon_res)  # One cell buffer
            lat_mask = (lat_vals >= lat_min - buffer) & (lat_vals <= lat_max + buffer)
            lon_mask = (lon_vals >= lon_min - buffer) & (lon_vals <= lon_max + buffer)

            filtered_lats = lat_vals[lat_mask]
            filtered_lons = lon_vals[lon_mask]

            if len(filtered_lats) == 0 or len(filtered_lons) == 0:
                self.logger.error("No grid cells within bounding box")
                return None

            self.logger.info(
                f"Filtered grid: {len(filtered_lats)} lats x {len(filtered_lons)} lons "
                f"= {len(filtered_lats) * len(filtered_lons)} cells"
            )

            # Step 6: Create grid cell polygons
            cells = []
            cell_id = 1

            # Store original indices for native grid referencing
            lat_indices = np.where(lat_mask)[0]
            if not lat_ascending:
                # lat_vals was reversed above; map back to the file's own index
                lat_indices = (n_lat_native - 1) - lat_indices
            lon_indices = np.where(lon_mask)[0]

            for i, lat in enumerate(filtered_lats):
                for j, lon in enumerate(filtered_lons):
                    # Create cell polygon (cell centers to cell edges)
                    half_lat = lat_res / 2
                    half_lon = lon_res / 2

                    cell = box(
                        lon - half_lon, lat - half_lat,
                        lon + half_lon, lat + half_lat
                    )

                    # Calculate approximate area in square meters
                    # Using spherical approximation at cell center latitude
                    lat_m = lat_res * 111000  # ~111 km per degree latitude
                    lon_m = lon_res * 111000 * np.cos(np.radians(lat))
                    area_m2 = lat_m * lon_m

                    cells.append({
                        'geometry': cell,
                        'GRU_ID': cell_id,
                        'GRU_area': area_m2,
                        'gru_to_seg': cell_id,
                        'center_lat': lat,
                        'center_lon': lon,
                        'native_lat_idx': int(lat_indices[i]),
                        'native_lon_idx': int(lon_indices[j]),
                        'row': i,
                        'col': j,
                    })
                    cell_id += 1

            # Step 7: Create GeoDataFrame
            grid_gdf = gpd.GeoDataFrame(cells, crs='EPSG:4326')

            self.logger.info(
                f"Created native grid with {len(grid_gdf)} cells "
                f"from {self.native_grid_dataset} forcing data"
            )

            return grid_gdf

        except DiscretizationError:
            raise
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating native grid: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _locate_forcing_file(self) -> Optional[Path]:
        """
        Locate forcing netCDF file for native grid extraction.

        Searches for forcing files in the configured forcing directory
        based on the native_grid_dataset setting.

        Returns:
            Path to forcing file, or None if not found.
        """
        # Get forcing directory from config
        forcing_dir = self._get_config_value(
            lambda: self.config.paths.forcing_path,
            default=str(self.project_forcing_dir),
            dict_key='FORCING_PATH'
        )

        if forcing_dir == 'default':
            forcing_dir = self.project_forcing_dir
        else:
            forcing_dir = Path(forcing_dir)

        if not forcing_dir.exists():
            self.logger.warning(f"Forcing directory not found: {forcing_dir}")
            return None

        # Dataset-specific file patterns
        dataset_patterns = {
            'era5': ['era5*.nc', 'ERA5*.nc', '*era5*.nc', '*ERA5*.nc'],
            'cerra': ['cerra*.nc', 'CERRA*.nc', '*cerra*.nc', '*CERRA*.nc'],
            'carra': ['carra*.nc', 'CARRA*.nc', '*carra*.nc', '*CARRA*.nc'],
            'rdrs': ['rdrs*.nc', 'RDRS*.nc', '*rdrs*.nc', '*RDRS*.nc'],
        }

        patterns = dataset_patterns.get(
            self.native_grid_dataset.lower(),
            [f'{self.native_grid_dataset}*.nc', f'*{self.native_grid_dataset}*.nc']
        )

        # Search for matching files
        for pattern in patterns:
            matches = list(forcing_dir.glob(pattern))
            if matches:
                # Return the first match (or could prioritize by date/size)
                return matches[0]

        # Also try subdirectories
        for subdir in forcing_dir.iterdir():
            if subdir.is_dir():
                for pattern in patterns:
                    matches = list(subdir.glob(pattern))
                    if matches:
                        return matches[0]

        # Fallback: try any netCDF file in the forcing directory
        all_nc_files = list(forcing_dir.glob('*.nc'))
        if all_nc_files:
            self.logger.info(
                f"No {self.native_grid_dataset}-specific file found, "
                f"using first available: {all_nc_files[0].name}"
            )
            return all_nc_files[0]

        return None

    def _detect_coord_name(self, ds, candidates: list) -> Optional[str]:
        """
        Detect coordinate name from dataset using candidate names.

        Args:
            ds: xarray Dataset
            candidates: List of possible coordinate names (e.g., ['lat', 'latitude', 'y'])

        Returns:
            Detected coordinate name, or None if not found.
        """
        # Check coordinates first
        for name in candidates:
            if name in ds.coords:
                return name

        # Check dimensions as fallback
        for name in candidates:
            if name in ds.dims:
                return name

        # Check data variables (some datasets store coords as vars)
        for name in candidates:
            if name in ds.data_vars:
                return name

        return None

    def _subset_grid_from_geofabric(
        self,
        geofabric_basins: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Create grid cells that intersect with geofabric basins.

        Args:
            geofabric_basins: GeoDataFrame with basin polygons from geofabric

        Returns:
            GeoDataFrame with grid cells clipped to geofabric extent
        """
        try:
            self.logger.info("Subsetting grid from geofabric basins")

            # Create full grid from bounding box of geofabric
            full_grid = self._create_grid_from_bbox()
            if full_grid is None:
                return None

            # Ensure same CRS
            if geofabric_basins.crs != full_grid.crs:
                geofabric_basins = geofabric_basins.to_crs(full_grid.crs)

            # Clip grid to geofabric extent
            clipped = gpd.overlay(full_grid, geofabric_basins[['geometry']], how='intersection')

            if len(clipped) == 0:
                self.logger.warning("No grid cells intersect geofabric basins")
                return None

            # Recalculate areas
            utm_crs = clipped.estimate_utm_crs()
            clipped_utm = clipped.to_crs(utm_crs)
            clipped['GRU_area'] = clipped_utm.geometry.area

            # Remove slivers
            min_area = 0.1 * (self.grid_cell_size ** 2)
            clipped = clipped[clipped['GRU_area'] >= min_area]

            # Reassign IDs
            clipped = clipped.reset_index(drop=True)
            clipped['GRU_ID'] = range(1, len(clipped) + 1)

            self.logger.info(f"Created {len(clipped)} grid cells from geofabric subset")
            return clipped

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error subsetting grid from geofabric: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None
