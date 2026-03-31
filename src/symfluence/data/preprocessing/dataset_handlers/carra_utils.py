# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CARRA Dataset Handler for SYMFLUENCE

This module provides the CARRA-specific implementation for forcing data processing.
CARRA uses a polar stereographic projection and requires special coordinate handling.
"""

from pathlib import Path
from typing import Dict, Tuple

import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from ...utils import VariableStandardizer
from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('carra')
class CARRAHandler(BaseDatasetHandler):
    """Handler for CARRA (Copernicus Arctic Regional Reanalysis) dataset."""

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        CARRA variable name mapping to standard names.

        Uses centralized VariableStandardizer for consistency across the codebase.

        Returns:
            Dictionary mapping CARRA variable names to standard names
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('CARRA')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CARRA dataset with variable renaming if needed.

        CARRA data typically comes in standard units.

        Args:
            ds: Input CARRA dataset

        Returns:
            Processed dataset with standardized variables
        """
        # Rename variables using mapping
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}

        if existing_vars:
            ds = ds.rename(existing_vars)

        # Apply standard CF-compliant attributes (uses centralized definitions)
        # CARRA precipitation follows ECMWF conventions: kg m-2 s-1 (equivalent to mm/s)
        ds = self.apply_standard_attributes(ds, overrides={
            'precipitation_flux': {'units': 'kg m-2 s-1', 'standard_name': 'precipitation_rate'}
        })

        return ds

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CARRA uses latitude/longitude coordinates.

        Returns:
            Tuple of ('latitude', 'longitude')
        """
        return ('latitude', 'longitude')

    def needs_merging(self) -> bool:
        """CARRA data typically doesn't require merging."""
        return False

    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        CARRA typically doesn't require merging.

        This method is a no-op for CARRA but is required by the interface.
        """
        self.logger.info("CARRA data does not require merging. Skipping merge step.")
        pass

    def _normalize_coordinates(self, merged_forcing_path: Path):
        """
        Normalize CARRA file coordinates: convert lon from 0-360 to -180/180
        and sort latitude ascending.

        This must run before shapefile creation and EASYMORE remapping to ensure
        consistency between the forcing shapefile (which uses WGS84 -180/180 for
        HRU intersection) and EASYMORE's internal meshgrid (built from the NetCDF
        coordinate arrays).

        Without this, CARRA's native 0-360 longitudes cause EASYMORE to assign
        cols=0 for all entries because it cannot match -180/180 shapefile centroids
        to 0-360 meshgrid values.
        """
        carra_files = sorted(merged_forcing_path.glob('*.nc'))
        if not carra_files:
            return

        # Check first file to see if normalization is needed
        with self.open_dataset(carra_files[0]) as ds:
            lons = ds.longitude.values
            lats = ds.latitude.values
            needs_lon_fix = lons.max() > 180
            needs_lat_sort = len(lats) > 1 and lats[0] > lats[-1]

        if not needs_lon_fix and not needs_lat_sort:
            self.logger.info("CARRA coordinates already normalized (lon -180/180, lat ascending)")
            return

        fixes = []
        if needs_lon_fix:
            fixes.append(f"longitude 0-360 -> -180/180 (range: {lons.min():.1f}-{lons.max():.1f})")
        if needs_lat_sort:
            fixes.append(f"latitude descending -> ascending ({lats[0]:.2f}-{lats[-1]:.2f})")
        self.logger.info(f"Normalizing CARRA coordinates in {len(carra_files)} files: {', '.join(fixes)}")

        for nc_file in carra_files:
            # Load into memory and close file handle before writing
            with self.open_dataset(nc_file) as ds:
                ds_loaded = ds.load()

            modified = False

            if needs_lon_fix:
                new_lons = ds_loaded.longitude.values.copy()
                new_lons[new_lons > 180] -= 360
                ds_loaded = ds_loaded.assign_coords(longitude=new_lons)
                ds_loaded = ds_loaded.sortby('longitude')
                modified = True

            if needs_lat_sort:
                ds_loaded = ds_loaded.sortby('latitude')
                modified = True

            if modified:
                ds_loaded.to_netcdf(nc_file, mode='w')
            ds_loaded.close()

        self.logger.info("CARRA coordinate normalization complete")

    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create CARRA grid shapefile.

        CARRA uses a polar stereographic projection which requires special handling.
        The grid is defined in stereographic coordinates but must be converted to lat/lon.

        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to CARRA data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics

        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating CARRA grid shapefile")

        # Normalize coordinates before shapefile creation and EASYMORE remapping.
        # CARRA data may have 0-360 longitudes and descending latitude which causes
        # mismatches with EASYMORE's coordinate matching.
        self._normalize_coordinates(merged_forcing_path)

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        # Import geopandas here - needed for GeoDataFrame creation below
        import geopandas as gpd

        try:
            # Find a processed CARRA file
            carra_files = list(merged_forcing_path.glob('*.nc'))
            if not carra_files:
                raise FileNotFoundError("No processed CARRA files found")
            carra_file = carra_files[0]

            self.logger.info(f"Using CARRA file: {carra_file}")

            # Read CARRA data
            with self.open_dataset(carra_file) as ds:
                lats = ds.latitude.values
                lons = ds.longitude.values

            self.logger.info(f"CARRA dimensions: lat={lats.shape}, lon={lons.shape}")
            self.logger.info(f"CARRA lat range: {lats.min():.4f} to {lats.max():.4f}")
            self.logger.info(f"CARRA lon range: {lons.min():.4f} to {lons.max():.4f}")

            # Check if this is a regular lat/lon grid (1D) or curvilinear (2D)
            is_regular_grid = (lats.ndim == 1)
            if is_regular_grid:
                self.logger.info("Detected regular lat/lon grid (1D coordinates)")

            # Get HRU bounding box for spatial filtering
            # Read the HRU shapefile to get its extent
            # shapefile_path is .../shapefiles/forcing, so parent is .../shapefiles
            hru_shapefile_dir = shapefile_path.parent / 'catchment'
            domain_name = self._get_config_value(lambda: self.config.domain.name, default='domain')
            hru_shapefile = hru_shapefile_dir / f"{domain_name}_HRUs_GRUs.shp"

            bbox_filter = None
            if hru_shapefile.exists():
                try:
                    hru_gdf = gpd.read_file(hru_shapefile)
                    # Reproject to WGS84 if needed before extracting bounds
                    if hru_gdf.crs and hru_gdf.crs != 'EPSG:4326':
                        self.logger.info(f"Reprojecting HRU from {hru_gdf.crs} to EPSG:4326 for spatial filtering")
                        hru_gdf = hru_gdf.to_crs('EPSG:4326')
                    # Get bounding box with small buffer to ensure we capture nearby cells
                    bbox = hru_gdf.total_bounds  # [minx, miny, maxx, maxy]
                    buffer = 0.1  # ~10km buffer
                    bbox_filter = {
                        'lon_min': bbox[0] - buffer,
                        'lon_max': bbox[2] + buffer,
                        'lat_min': bbox[1] - buffer,
                        'lat_max': bbox[3] + buffer
                    }
                    self.logger.info("Applying spatial filter based on HRU extent:")
                    self.logger.info(f"  Lon: {bbox_filter['lon_min']:.2f} to {bbox_filter['lon_max']:.2f}")
                    self.logger.info(f"  Lat: {bbox_filter['lat_min']:.2f} to {bbox_filter['lat_max']:.2f}")
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Could not read HRU shapefile for spatial filtering: {e}")
                    self.logger.warning("Will create shapefile for full domain (may be slow)")

            # Create geometries
            self.logger.info("Creating CARRA grid cell geometries")

            geometries = []
            ids = []
            center_lats = []
            center_lons = []

            if is_regular_grid:
                # Regular lat/lon grid - create rectangular cells directly

                # Calculate grid spacing
                lat_spacing = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.025
                lon_spacing = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.025

                self.logger.info(f"Grid spacing: lat={lat_spacing:.4f}°, lon={lon_spacing:.4f}°")

                # Create meshgrid for all combinations
                cell_id = 0
                cells_created = 0

                for lat_idx, center_lat in enumerate(lats):
                    for lon_idx, center_lon_raw in enumerate(lons):
                        # Convert longitude from 0-360 to -180/180 range
                        if center_lon_raw > 180:
                            center_lon = float(center_lon_raw - 360)
                        else:
                            center_lon = float(center_lon_raw)

                        center_lat = float(center_lat)

                        # Apply spatial filter if available
                        if bbox_filter is not None:
                            if (center_lon < bbox_filter['lon_min'] or center_lon > bbox_filter['lon_max'] or
                                center_lat < bbox_filter['lat_min'] or center_lat > bbox_filter['lat_max']):
                                cell_id += 1
                                continue

                        # Create rectangular cell
                        half_dlat = lat_spacing / 2.0
                        half_dlon = lon_spacing / 2.0

                        vertices = [
                            (center_lon - half_dlon, center_lat - half_dlat),
                            (center_lon - half_dlon, center_lat + half_dlat),
                            (center_lon + half_dlon, center_lat + half_dlat),
                            (center_lon + half_dlon, center_lat - half_dlat),
                            (center_lon - half_dlon, center_lat - half_dlat)
                        ]

                        geometries.append(Polygon(vertices))
                        ids.append(cell_id)
                        center_lats.append(center_lat)
                        center_lons.append(center_lon)
                        cells_created += 1
                        cell_id += 1

                self.logger.info(f"Created {cells_created} grid cells")

                if cells_created == 0 and bbox_filter is not None:
                    self.logger.warning("No grid cells created! CARRA grid does not overlap with HRU extent.")
                    self.logger.warning("This usually means the CARRA download used a different bounding box than the HRU shapefile.")
                    self.logger.warning("Check that BOUNDING_BOX_COORDS in config matches your study area.")

            else:
                # Curvilinear grid - use stereographic projection
                # Define CARRA projection (polar stereographic)
                carra_proj = CRS('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356752.3142 +units=m +no_defs')
                wgs84 = CRS('EPSG:4326')

                transformer = Transformer.from_crs(carra_proj, wgs84, always_xy=True)
                transformer_to_carra = Transformer.from_crs(wgs84, carra_proj, always_xy=True)

                # Flatten 2D lat/lon arrays to 1D for iteration
                lats_flat = lats.flatten()
                lons_flat = lons.flatten()

                # Process in batches
                batch_size = 100
                total_cells = len(lats_flat)
                num_batches = (total_cells + batch_size - 1) // batch_size
                cells_created = 0

                self.logger.info(f"Processing {total_cells} CARRA grid cells in {num_batches} batches")

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_cells)

                    self.logger.info(f"Processing grid cell batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")

                    for i in range(start_idx, end_idx):
                        # Get cell center coordinates
                        center_lat_raw = float(lats_flat[i])
                        center_lon_raw = float(lons_flat[i])

                        # Convert longitude from 0-360 to -180/180 range for consistency
                        if center_lon_raw > 180:
                            center_lon_normalized = center_lon_raw - 360
                        else:
                            center_lon_normalized = center_lon_raw

                        # Apply spatial filter if available (using normalized longitude)
                        if bbox_filter is not None:
                            if (center_lon_normalized < bbox_filter['lon_min'] or center_lon_normalized > bbox_filter['lon_max'] or
                                center_lat_raw < bbox_filter['lat_min'] or center_lat_raw > bbox_filter['lat_max']):
                                continue

                        # Convert lat/lon to CARRA coordinates (scalars now)
                        # Use normalized longitude (-180/180) instead of raw (0-360)
                        x, y = transformer_to_carra.transform(center_lon_normalized, center_lat_raw)

                        # Define grid cell (assuming 2.5 km resolution)
                        half_dx = 1250  # meters
                        half_dy = 1250  # meters

                        vertices = [
                            (x - half_dx, y - half_dy),
                            (x - half_dx, y + half_dy),
                            (x + half_dx, y + half_dy),
                            (x + half_dx, y - half_dy),
                            (x - half_dx, y - half_dy)
                        ]

                        # Convert vertices back to lat/lon
                        # Ensure values are scalars, not arrays
                        lat_lon_vertices = []
                        for vx, vy in vertices:
                            lon, lat = transformer.transform(vx, vy)
                            # Extract scalar values if they're arrays
                            if hasattr(lon, 'item'):
                                lon = lon.item()
                            if hasattr(lat, 'item'):
                                lat = lat.item()
                            lat_lon_vertices.append((float(lon), float(lat)))

                        geometries.append(Polygon(lat_lon_vertices))
                        ids.append(i)

                        center_lon, center_lat = transformer.transform(x, y)
                        # Extract scalar values if they're arrays
                        if hasattr(center_lon, 'item'):
                            center_lon = center_lon.item()
                        if hasattr(center_lat, 'item'):
                            center_lat = center_lat.item()
                        center_lats.append(float(center_lat))
                        center_lons.append(float(center_lon))
                        cells_created += 1

                if bbox_filter is not None:
                    self.logger.info(f"Spatial filtering: created {cells_created} cells (from {total_cells} total)")

            # Create GeoDataFrame
            self.logger.info(f"Creating GeoDataFrame with {cells_created} grid cells")
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): center_lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): center_lons,
            }, crs='EPSG:4326')

            # Calculate elevation using the safe method
            self.logger.info("Calculating elevation values using safe method")
            elevations = elevation_calculator(gdf, dem_path, batch_size=50)
            gdf['elev_m'] = elevations

            # Save the shapefile
            self.logger.info(f"Saving CARRA shapefile to {output_shapefile}")
            gdf.to_file(output_shapefile)
            self.logger.info(f"CARRA grid shapefile created and saved to {output_shapefile}")

            return output_shapefile

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error in create_carra_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
