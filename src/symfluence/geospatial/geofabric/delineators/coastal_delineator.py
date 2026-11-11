# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coastal watershed delineation module.

Handles delineation of coastal watersheds that drain directly to the ocean,
and creation of point buffer shapes for point-scale simulations.

Extracted from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
from shapely.geometry import Point, Polygon

from ..base.base_delineator import BaseGeofabricDelineator
from ..processors.geometry_processor import GeometryProcessor
from ..utils.io_utils import GeofabricIOUtils


class CoastalWatershedDelineator(BaseGeofabricDelineator):
    """
    Delineator for coastal watersheds and point buffer shapes.

    Handles:
    - Coastal watershed identification (land areas outside existing watersheds)
    - Point buffer creation for point-scale simulations
    - GRU merging for small watersheds
    """

    def _get_delineation_method_name(self) -> str:
        """Return the delineation method name."""
        return "coastal"

    def delineate_coastal(self, work_log_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Delineate coastal watersheds that drain directly to the ocean.

        This method:
        1. Creates a land mask from the DEM
        2. Finds the difference between the land mask and existing watersheds
        3. Divides the coastal strip into individual watersheds by extending the boundaries
        of adjacent inland watersheds

        Args:
            work_log_dir: Directory for logging. Defaults to None.

        Returns:
            Tuple[Optional[Path], Optional[Path]]: Paths to the updated river_network and river_basins shapefiles.
        """
        try:
            self.logger.info(f"Starting coastal watershed delineation for {self.domain_name}")

            # Get paths to existing delineated river basins and network
            method_suffix = self._get_method_suffix()
            river_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            river_network_path = self.project_dir / "shapefiles" / "river_network" / f"{self.domain_name}_riverNetwork_{method_suffix}.shp"

            if not river_basins_path.exists() or not river_network_path.exists():
                self.logger.error("River basins or network files not found. Run delineate_geofabric first.")
                return None, None

            # Load existing delineation
            river_basins = GeofabricIOUtils.load_geopandas(river_basins_path, self.logger)
            GeofabricIOUtils.load_geopandas(river_network_path, self.logger)

            # Create interim directory for coastal delineation
            coastal_interim_dir = self.project_dir / "taudem-interim-files" / "coastal"
            coastal_interim_dir.mkdir(parents=True, exist_ok=True)

            # ---------- STEP 1: IDENTIFY COASTAL AREAS ---------- #

            # Create a land polygon from the DEM (areas with elevation > 0)
            land_polygon = self._create_land_polygon_from_dem()
            if land_polygon is None or land_polygon.empty:
                self.logger.error("Failed to create land polygon from DEM.")
                return river_network_path, river_basins_path

            # Create a single polygon from all existing watersheds
            try:
                watersheds_polygon = gpd.GeoDataFrame(
                    geometry=[river_basins.union_all()],
                    crs=river_basins.crs
                )
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error creating watersheds polygon: {str(e)}", exc_info=True)
                return river_network_path, river_basins_path

            # Find land areas not covered by existing watersheds
            try:
                # Make sure CRS matches
                if land_polygon.crs != watersheds_polygon.crs:
                    land_polygon = land_polygon.to_crs(watersheds_polygon.crs)

                # Find the difference between land and watersheds
                coastal_strip = gpd.overlay(land_polygon, watersheds_polygon, how='difference')

                if coastal_strip.empty:
                    self.logger.info("No coastal areas found outside existing watersheds.")
                    return river_network_path, river_basins_path

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error finding coastal strip: {str(e)}", exc_info=True)
                return river_network_path, river_basins_path

            # ---------- STEP 2: DIVIDE COASTAL STRIP INTO INDIVIDUAL WATERSHEDS ---------- #

            # All geometric division (Voronoi, buffers, overlays) is done in a
            # projected CRS (metres). In geographic degrees these operations shed
            # thin sliver/tentacle polygons, especially at high latitude where a
            # degree of longitude and latitude differ greatly.
            utm_crs = river_basins.estimate_utm_crs()
            coastal_strip = coastal_strip.to_crs(utm_crs)
            river_basins_proj = river_basins.to_crs(utm_crs)

            # First, create Voronoi polygons for each river basin
            try:
                # Get centroids of each river basin (in projected CRS)
                river_basins_centroids = river_basins_proj.copy()
                river_basins_centroids.geometry = river_basins_proj.geometry.centroid

                # Buffer centroids slightly (1 m) to avoid geopandas Voronoi issues
                river_basins_centroids.geometry = river_basins_centroids.geometry.buffer(1.0)

                # Create Voronoi polygons
                voronoi_gdf = self._create_voronoi_tessellation(
                    river_basins_centroids, cover=coastal_strip
                )

                if voronoi_gdf is None or voronoi_gdf.empty:
                    self.logger.warning("Failed to create Voronoi tessellation. Using alternative approach.")
                    # Use watershed boundaries to create extended lines to the coast
                    coastal_watersheds = self._divide_coastal_strip_by_extending_boundaries(coastal_strip, river_basins_proj)
                else:
                    # Intersect Voronoi polygons with coastal strip to create coastal watersheds
                    voronoi_gdf = gpd.GeoDataFrame(
                        geometry=voronoi_gdf.geometry,
                        crs=utm_crs
                    )
                    coastal_watersheds = gpd.overlay(coastal_strip, voronoi_gdf, how='intersection')
                    self._check_strip_area_conserved(coastal_strip, coastal_watersheds)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error dividing coastal strip: {str(e)}", exc_info=True)
                # Fallback to simpler approach: use a buffer method
                coastal_watersheds = self._divide_coastal_strip_by_buffer_method(coastal_strip, river_basins_proj)

            # If we still don't have valid coastal watersheds, use a simpler approach
            if coastal_watersheds is None or coastal_watersheds.empty:
                self.logger.warning("Failed to create divided coastal watersheds. Using buffer method.")
                coastal_watersheds = self._divide_coastal_strip_by_buffer_method(coastal_strip, river_basins_proj)

            if coastal_watersheds is None or coastal_watersheds.empty:
                self.logger.info("No valid coastal watersheds created.")
                return river_network_path, river_basins_path

            # ---------- STEP 3: PROCESS AND CLEAN COASTAL WATERSHEDS ---------- #

            # Calculate areas and filter small fragments
            utm_crs = coastal_watersheds.estimate_utm_crs()
            coastal_watersheds_utm = coastal_watersheds.to_crs(utm_crs)

            # Calculate area
            coastal_watersheds_utm['area_km2'] = coastal_watersheds_utm.geometry.area / 1_000_000

            # Remove tiny fragments
            min_coastal_area = 0.1  # 0.1 km²
            coastal_watersheds_utm = coastal_watersheds_utm[coastal_watersheds_utm['area_km2'] > min_coastal_area]

            if coastal_watersheds_utm.empty:
                self.logger.info("No significant coastal watersheds after size filtering.")
                return river_network_path, river_basins_path

            # Add required attributes
            max_gru_id = river_basins['GRU_ID'].max() if 'GRU_ID' in river_basins.columns else 0
            coastal_watersheds_utm = coastal_watersheds_utm.reset_index(drop=True)
            coastal_watersheds_utm['GRU_ID'] = range(max_gru_id + 1, max_gru_id + 1 + len(coastal_watersheds_utm))
            coastal_watersheds_utm['gru_to_seg'] = 0  # No river segment
            coastal_watersheds_utm['GRU_area'] = coastal_watersheds_utm.geometry.area
            coastal_watersheds_utm['is_coastal'] = True

            # Convert back to original CRS
            coastal_watersheds = coastal_watersheds_utm.to_crs(river_basins.crs)

            # ---------- STEP 4: MERGE WITH EXISTING WATERSHEDS ---------- #

            # Add coastal attribute to existing basins
            river_basins['is_coastal'] = False

            # Ensure required columns exist
            required_cols = ['GRU_ID', 'gru_to_seg', 'GRU_area', 'is_coastal', 'geometry']
            for col in required_cols:
                if col not in coastal_watersheds.columns and col != 'geometry':
                    coastal_watersheds[col] = None if col == 'is_coastal' else 0

            # Get only needed columns
            coastal_cols = [col for col in coastal_watersheds.columns if col in required_cols or col == 'geometry']

            # Merge with existing river basins
            combined_basins = pd.concat([
                river_basins,
                coastal_watersheds[coastal_cols]
            ])

            # Optionally despike: the coastal-strip division (buffer/overlay in a
            # geographic CRS) and the inherited base watersheds can carry thin
            # tentacle artifacts. Strip them before writing the final geofabric.
            despike = self._get_config_value(
                lambda: None,
                default=False,
                dict_key='DESPIKE_BASIN_GEOMETRY',
            )
            if despike:
                n_before = len(combined_basins)
                combined_basins = GeometryProcessor.despike_geodataframe(combined_basins)
                self.logger.info(f"Despiked combined basin geometry ({n_before} basins)")

            # Save combined results
            combined_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_with_coastal.shp"
            combined_basins.to_file(combined_basins_path)

            self.logger.info(f"Added {len(coastal_watersheds)} coastal watersheds to the delineation.")
            self.logger.info(f"Combined river basins saved to: {combined_basins_path}")

            # Cleanup if requested
            if self._get_config_value(lambda: self.config.domain.delineation.cleanup_intermediate_files, default=True, dict_key='CLEANUP_INTERMEDIATE_FILES'):
                shutil.rmtree(coastal_interim_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up coastal interim files: {coastal_interim_dir}")

            return river_network_path, combined_basins_path

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error in coastal watershed delineation: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return river_network_path, river_basins_path

    def delineate_point_buffer_shape(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Create a small square buffer around the pour point for point-scale simulations.

        This method creates a simple square buffer with 0.01 degree (~1km) around the pour point
        specified in the configuration. It saves the buffer as shapefiles in both the river_basins
        and catchment directories to satisfy SYMFLUENCE's requirements.

        Returns:
            Tuple[Optional[Path], Optional[Path]]: Paths to the created river_basins and catchment shapefiles
        """
        try:
            self.logger.info(f"Creating point buffer shape for point-scale simulation at {self.domain_name}")

            # Get pour point coordinates
            pour_point_coords = self._get_config_value(lambda: self.config.domain.pour_point_coords, default='', dict_key='POUR_POINT_COORDS').split('/')
            if len(pour_point_coords) != 2:
                self.logger.error(f"Invalid pour point coordinates: {self._get_config_value(lambda: self.config.domain.pour_point_coords, dict_key='POUR_POINT_COORDS')}")
                return None, None

            # Convert to floats
            try:
                lat, lon = float(pour_point_coords[0]), float(pour_point_coords[1])
            except ValueError:
                self.logger.error(f"Invalid pour point coordinates format: {self._get_config_value(lambda: self.config.domain.pour_point_coords, dict_key='POUR_POINT_COORDS')}")
                return None, None

            # Define buffer distance (0.01 degrees, approximately 1km at the equator)
            buffer_dist = self._get_config_value(
                lambda: self.config.domain.delineation.point_buffer_distance,
                default=None,
                dict_key='POINT_BUFFER_DISTANCE'
            )

            # Create a square buffer around the point
            min_lon = lon - buffer_dist
            max_lon = lon + buffer_dist
            min_lat = lat - buffer_dist
            max_lat = lat + buffer_dist

            # Create polygon geometry
            polygon = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat)
            ])

            # Create GeoDataFrame with the polygon
            gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

            # Add required attributes
            gdf['GRU_ID'] = 1
            gdf['gru_to_seg'] = 1

            # Convert to UTM for area calculation
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf['GRU_area'] = gdf_utm.geometry.area

            # Create a simple point feature at the pour point for river network
            point = Point(lon, lat)
            river_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')
            river_gdf['LINKNO'] = 1
            river_gdf['DSLINKNO'] = 0
            river_gdf['Length'] = 0
            river_gdf['Slope'] = 0
            river_gdf['GRU_ID'] = 1

            # Create directories if they don't exist
            river_basins_dir = self.project_dir / "shapefiles" / "river_basins"
            catchment_dir = self.project_dir / "shapefiles" / "catchment"
            river_network_dir = self.project_dir / "shapefiles" / "river_network"

            river_basins_dir.mkdir(parents=True, exist_ok=True)
            catchment_dir.mkdir(parents=True, exist_ok=True)
            river_network_dir.mkdir(parents=True, exist_ok=True)

            # Define output paths
            method_suffix = self._get_method_suffix()
            river_basins_path = river_basins_dir / f"{self.domain_name}_riverBasins_{method_suffix}.shp"
            catchment_path = catchment_dir / f"{self.domain_name}_HRUs_{method_suffix}.shp"
            river_network_path = river_network_dir / f"{self.domain_name}_riverNetwork_{method_suffix}.shp"

            # Save shapefiles
            gdf.to_file(river_basins_path)
            gdf.to_file(catchment_path)
            river_gdf.to_file(river_network_path)

            self.logger.info("Point buffer shapefiles created successfully at:")
            self.logger.info(f"  - River basins: {river_basins_path}")
            self.logger.info(f"  - Catchment: {catchment_path}")
            self.logger.info(f"  - River network: {river_network_path}")

            return river_network_path, river_basins_path

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating point buffer shape: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def _create_land_polygon_from_dem(self) -> Optional[gpd.GeoDataFrame]:
        """
        Create a polygon representing land areas based on the DEM.

        Returns:
            GeoDataFrame with land polygon or None if failed
        """
        try:
            # Open the DEM
            with rasterio.open(str(self.dem_path)) as src:
                # Read DEM data
                dem_data = src.read(1)
                nodata_value = src.nodata
                transform = src.transform
                crs = src.crs

                # Create binary mask where elevation > 0
                land_mask = np.where(
                    (dem_data > 0) & (dem_data != nodata_value),
                    1,  # Land
                    0   # Ocean or nodata
                ).astype(np.uint8)

                # Use rasterio features to extract land polygons
                shapes = rasterio.features.shapes(
                    land_mask,
                    mask=land_mask == 1,
                    transform=transform
                )

                # Convert shapes to shapely geometries
                land_polygons = [shapely.geometry.shape(shape) for shape, value in shapes if value == 1]

                if not land_polygons:
                    self.logger.warning("No land areas detected in DEM.")
                    return None

                # Create a GeoDataFrame with dissolved geometry
                land_gdf = gpd.GeoDataFrame(
                    geometry=[shapely.ops.unary_union(land_polygons)],
                    crs=crs
                )

                self.logger.info(f"Created land polygon from DEM with {len(land_polygons)} original features.")
                return land_gdf

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating land polygon from DEM: {str(e)}", exc_info=True)
            return None

    def _check_strip_area_conserved(
        self,
        coastal_strip: gpd.GeoDataFrame,
        coastal_watersheds: gpd.GeoDataFrame,
        tolerance: float = 1e-3,
    ) -> bool:
        """Log an error if dividing the coastal strip lost or duplicated area."""
        strip_area = float(coastal_strip.geometry.area.sum())
        divided_area = float(coastal_watersheds.geometry.area.sum()) if len(coastal_watersheds) else 0.0
        if strip_area <= 0:
            return True
        rel = abs(divided_area - strip_area) / strip_area
        if rel > tolerance:
            self.logger.error(
                f"Coastal strip division does not conserve area: {divided_area / 1e6:.3f} km2 "
                f"assigned of {strip_area / 1e6:.3f} km2 ({100 * divided_area / strip_area:.1f}%)."
            )
            return False
        return True

    def _create_voronoi_tessellation(
        self,
        points_gdf: gpd.GeoDataFrame,
        cover: Optional[gpd.GeoDataFrame] = None,
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Create Voronoi polygons from point data.

        Every input point gets a polygon and together the polygons tile the
        bounding area of the points and of *cover* (the area to be divided):
        far-away helper points are added before the tessellation so that the
        regions of the outermost points — which are unbounded in a plain
        Voronoi diagram, and are exactly the ones facing the coast — are
        closed instead of being discarded.

        Args:
            points_gdf: Points to create Voronoi diagram from
            cover: Geometries the tessellation must cover completely

        Returns:
            GeoDataFrame with Voronoi polygons or None if failed
        """
        try:
            from scipy.spatial import Voronoi

            # Extract points coordinates - ensure we're getting actual points
            coords = []
            for geom in points_gdf.geometry:
                # Get centroid if it's not already a point
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    point = geom.centroid
                else:
                    point = geom

                coords.append((point.x, point.y))

            coords = np.unique(np.array(coords, dtype=float), axis=0)

            if len(coords) < 1:
                self.logger.warning("No points for Voronoi tessellation.")
                return None

            # Area the tessellation has to cover
            minx, miny = coords.min(axis=0)
            maxx, maxy = coords.max(axis=0)
            if cover is not None and len(cover) > 0:
                cminx, cminy, cmaxx, cmaxy = cover.total_bounds
                minx, miny = min(minx, cminx), min(miny, cminy)
                maxx, maxy = max(maxx, cmaxx), max(maxy, cmaxy)
            margin = 1000.0  # projected CRS, metres
            envelope = shapely.geometry.box(minx - margin, miny - margin, maxx + margin, maxy + margin)

            # Far helper points close every real point's region
            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
            reach = 10.0 * max(maxx - minx, maxy - miny, 10000.0)
            helpers = np.array([
                (cx + dx * reach, cy + dy * reach)
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            ])
            vor = Voronoi(np.vstack([coords, helpers]))

            # One polygon per real point
            regions = []
            for point_idx in range(len(coords)):
                region = vor.regions[vor.point_region[point_idx]]
                if len(region) < 3 or -1 in region:
                    self.logger.warning(
                        f"Voronoi region of point {point_idx} is unbounded; tessellation incomplete."
                    )
                    return None
                polygon = shapely.geometry.Polygon([vor.vertices[i] for i in region])
                regions.append(polygon.intersection(envelope))

            # Create GeoDataFrame
            voronoi_gdf = gpd.GeoDataFrame(geometry=regions, crs=points_gdf.crs)

            # Filter out empty geometries
            voronoi_gdf = voronoi_gdf[~voronoi_gdf.geometry.is_empty]

            self.logger.info(f"Created {len(voronoi_gdf)} Voronoi polygons.")
            return voronoi_gdf

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error creating Voronoi tessellation: {str(e)}", exc_info=True)
            return None

    def _divide_coastal_strip_by_extending_boundaries(
        self,
        coastal_strip: gpd.GeoDataFrame,
        river_basins: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Divide coastal strip by extending the external boundaries of river basins.

        Args:
            coastal_strip: The coastal areas to divide
            river_basins: Existing river basins

        Returns:
            GeoDataFrame with divided coastal watersheds or None if failed
        """
        try:
            # Use a buffer-based approach to divide the coastal strip
            coastal_geom = coastal_strip.geometry.union_all()
            divided_coastal = []

            for idx, basin in river_basins.iterrows():
                try:
                    # Create a buffer around the basin (projected CRS, metres)
                    buffer_dist = 1000.0  # ~1 km
                    buffer = basin.geometry.buffer(buffer_dist)

                    # Intersect with coastal strip
                    coastal_part = buffer.intersection(coastal_geom)

                    if not coastal_part.is_empty:
                        if coastal_part.geom_type == 'GeometryCollection':
                            # Extract polygons from collection
                            for geom in coastal_part.geoms:
                                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                    divided_coastal.append({
                                        'geometry': geom,
                                        'basin_id': basin['GRU_ID']
                                    })
                        elif coastal_part.geom_type in ['Polygon', 'MultiPolygon']:
                            divided_coastal.append({
                                'geometry': coastal_part,
                                'basin_id': basin['GRU_ID']
                            })
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Error processing basin {basin['GRU_ID']}: {str(e)}", exc_info=True)

            # Create GeoDataFrame from divided coastal watersheds
            if divided_coastal:
                coastal_watersheds = gpd.GeoDataFrame(
                    {
                        'geometry': [item['geometry'] for item in divided_coastal],
                        'parent_basin': [item['basin_id'] for item in divided_coastal]
                    },
                    crs=river_basins.crs
                )
                return coastal_watersheds
            else:
                self.logger.warning("No coastal watersheds created by boundary extension method.")
                return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error dividing coastal strip by extending boundaries: {str(e)}", exc_info=True)
            return None

    def _divide_coastal_strip_by_buffer_method(
        self,
        coastal_strip: gpd.GeoDataFrame,
        river_basins: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Divide the coastal strip using a buffer-based method.

        This is a more robust fallback method that uses buffers around each basin
        to claim portions of the coastal strip: progressively wider bands (up to
        1 km) around each basin claim first-come-first-served, and whatever the
        bands do not reach is split into contiguous fragments, each assigned
        whole to its nearest basin. A contiguous fragment equidistant to several
        basins (e.g. the outer part of a strip wider than 1 km) therefore goes
        entirely to the first spatial-index match; the choice is deterministic
        for a given input. Fine-grained partitioning is the job of the primary
        Voronoi path — this method only guarantees no coastal area is lost.

        Args:
            coastal_strip: Coastal areas to divide (projected CRS, metres)
            river_basins: Existing river basins

        Returns:
            GeoDataFrame with divided coastal watersheds or None if failed
        """
        try:
            # The strip is reprojected to UTM upstream while basins may arrive in
            # the original (possibly geographic) CRS; every intersection and
            # distance below is meaningless across mixed CRSes, so align first.
            if river_basins.crs != coastal_strip.crs:
                river_basins = river_basins.to_crs(coastal_strip.crs)

            coastal_geom = coastal_strip.geometry.union_all()

            # Create an empty list to store divided coastal watersheds
            divided_coastal = []

            # Progressively wider claim bands, in metres (the strip is in a
            # projected CRS). These match the intent of the original
            # degree-sized bands (~0.001-0.01 deg) from when this method
            # operated on geographic coordinates.
            buffer_sizes = [100.0, 200.0, 300.0, 500.0, 800.0, 1000.0]
            remaining_coastal = coastal_geom

            for buffer_size in buffer_sizes:
                if remaining_coastal.is_empty:
                    break

                # Process each basin to claim its portion of the coastal strip
                for idx, basin in river_basins.iterrows():
                    try:
                        # Create a buffer around the basin with gradient size
                        buffer = basin.geometry.buffer(buffer_size)

                        # Intersect with remaining coastal strip
                        claimed_area = buffer.intersection(remaining_coastal)

                        if not claimed_area.is_empty:
                            # Add to divided coastal watersheds
                            divided_coastal.append({
                                'geometry': claimed_area,
                                'basin_id': basin['GRU_ID']
                            })

                            # Remove claimed area from remaining coastal strip
                            remaining_coastal = remaining_coastal.difference(claimed_area)
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.warning(f"Error processing basin {basin['GRU_ID']} with buffer {buffer_size}: {str(e)}", exc_info=True)

            # Handle any remaining coastal strip by assigning each contiguous
            # fragment to its nearest basin (vectorized via the spatial index).
            if not remaining_coastal.is_empty:
                self.logger.info("Assigning remaining coastal areas to nearest basins.")

                remaining_gdf = gpd.GeoDataFrame(
                    geometry=[remaining_coastal],
                    crs=coastal_strip.crs
                )

                # Explode to get individual polygons
                remaining_gdf = remaining_gdf.explode(index_parts=True).reset_index(drop=True)

                joined = gpd.sjoin_nearest(
                    remaining_gdf,
                    river_basins[['GRU_ID', 'geometry']],
                    how='left',
                )
                # Exact-tie matches duplicate the fragment row; keep the first.
                joined = joined[~joined.index.duplicated(keep='first')]

                for geometry, basin_id in zip(joined.geometry, joined['GRU_ID']):
                    if pd.notna(basin_id):
                        divided_coastal.append({
                            'geometry': geometry,
                            'basin_id': basin_id
                        })

            # Create GeoDataFrame from divided coastal watersheds
            if divided_coastal:
                coastal_watersheds = gpd.GeoDataFrame(
                    {
                        'geometry': [item['geometry'] for item in divided_coastal],
                        'parent_basin': [item['basin_id'] for item in divided_coastal]
                    },
                    crs=river_basins.crs
                )

                # Dissolve by parent_basin to merge adjacent pieces
                coastal_watersheds = coastal_watersheds.dissolve(by='parent_basin').reset_index()

                return coastal_watersheds
            else:
                self.logger.warning("No coastal watersheds created by buffer method.")
                return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error dividing coastal strip by buffer method: {str(e)}", exc_info=True)
            return None

    def _find_neighbors(self, geometry: Any, gdf: gpd.GeoDataFrame, exclude_idx: int) -> gpd.GeoDataFrame:
        """
        Find neighboring GRUs that share a boundary.

        Args:
            geometry: Geometry to find neighbors for
            gdf: GeoDataFrame containing all GRUs
            exclude_idx: Index to exclude (self)

        Returns:
            GeoDataFrame with neighboring GRUs
        """
        return gdf[
            (gdf.index != exclude_idx) &
            (gdf.geometry.boundary.intersects(geometry.boundary))
        ]

    def merge_small_grus(self, gru_gdf: gpd.GeoDataFrame, min_gru_size: float) -> gpd.GeoDataFrame:
        """
        Merge GRUs smaller than the minimum size threshold with their neighbors.
        Optimized version with spatial indexing and vectorized operations.

        Args:
            gru_gdf: GeoDataFrame with GRUs to merge
            min_gru_size: Minimum GRU size in km²

        Returns:
            GeoDataFrame with merged GRUs
        """
        self.logger.info(f"Starting GRU merging process (minimum size: {min_gru_size} km²)")
        initial_count = len(gru_gdf)

        # Ensure CRS is geographic and convert to UTM for area calculations
        gru_gdf.set_crs(epsg=4326, inplace=True)
        utm_crs = gru_gdf.estimate_utm_crs()
        gru_gdf_utm = gru_gdf.to_crs(utm_crs)

        # Clean geometries (vectorized)
        gru_gdf_utm['geometry'] = gru_gdf_utm['geometry'].apply(GeometryProcessor.clean_geometries)
        gru_gdf_utm = gru_gdf_utm[gru_gdf_utm['geometry'].notnull()]

        # Store original boundary
        from shapely.ops import unary_union
        original_boundary = unary_union(gru_gdf_utm.geometry)

        # Calculate areas in km² (vectorized)
        gru_gdf_utm['area'] = gru_gdf_utm.geometry.area / 1_000_000

        # Create spatial index for faster neighbor finding
        spatial_index = gru_gdf_utm.sindex

        merged_count = 0
        while True:
            small_grus = gru_gdf_utm[gru_gdf_utm['area'] < min_gru_size]
            if len(small_grus) == 0:
                break

            # Process multiple small GRUs in parallel
            small_grus_to_merge = small_grus.head(100)  # Process in batches
            if len(small_grus_to_merge) == 0:
                break

            for idx, small_gru in small_grus_to_merge.iterrows():
                try:
                    small_gru_geom = GeometryProcessor.clean_geometries(small_gru.geometry)
                    if small_gru_geom is None:
                        gru_gdf_utm = gru_gdf_utm.drop(idx)
                        continue

                    # Use spatial index to find potential neighbors
                    possible_matches_idx = list(spatial_index.intersection(small_gru_geom.bounds))
                    possible_matches = gru_gdf_utm.iloc[possible_matches_idx]

                    # Filter actual neighbors
                    neighbors = possible_matches[
                        (possible_matches.index != idx) &
                        (possible_matches.geometry.boundary.intersects(small_gru_geom.boundary))
                    ]

                    if len(neighbors) > 0:
                        largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
                        merged_geometry = unary_union([small_gru_geom, largest_neighbor.geometry])
                        merged_geometry = GeometryProcessor.simplify_geometry(merged_geometry)

                        if merged_geometry and merged_geometry.is_valid:
                            gru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
                            gru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
                            gru_gdf_utm = gru_gdf_utm.drop(idx)
                            merged_count += 1

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.error(f"Error merging GRU {idx}: {str(e)}", exc_info=True)

            # Update spatial index after batch processing
            spatial_index = gru_gdf_utm.sindex

        # Handle gaps (vectorized where possible)
        current_coverage = unary_union(gru_gdf_utm.geometry)
        gaps = original_boundary.difference(current_coverage)
        if not gaps.is_empty:
            gap_geoms = list(gaps.geoms) if gaps.geom_type == 'MultiPolygon' else [gaps]

            for gap in gap_geoms:
                if gap.area > 0:
                    # Use spatial index to find nearest GRU
                    possible_matches_idx = list(spatial_index.nearest(gap.bounds))
                    nearest_gru = possible_matches_idx[0]
                    merged_geom = GeometryProcessor.clean_geometries(
                        unary_union([gru_gdf_utm.iloc[nearest_gru].geometry, gap])
                    )
                    if merged_geom and merged_geom.is_valid:
                        gru_gdf_utm.iloc[nearest_gru, gru_gdf_utm.columns.get_loc('geometry')] = merged_geom
                        gru_gdf_utm.iloc[nearest_gru, gru_gdf_utm.columns.get_loc('area')] = merged_geom.area / 1_000_000

        # Reset index and update IDs (vectorized)
        gru_gdf_utm = gru_gdf_utm.reset_index(drop=True)
        gru_gdf_utm['GRU_ID'] = range(1, len(gru_gdf_utm) + 1)
        gru_gdf_utm['gru_to_seg'] = gru_gdf_utm['GRU_ID']

        # Convert back to original CRS
        gru_gdf_merged = gru_gdf_utm.to_crs(gru_gdf.crs)

        self.logger.info("GRU merging statistics:")
        self.logger.info(f"- Initial GRUs: {initial_count}")
        self.logger.info(f"- Merged {merged_count} small GRUs")
        self.logger.info(f"- Final GRUs: {len(gru_gdf_merged)}")
        self.logger.info(f"- Reduction: {((initial_count - len(gru_gdf_merged)) / initial_count) * 100:.1f}%")

        return gru_gdf_merged
