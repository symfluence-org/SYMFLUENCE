# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CERRA Dataset Handler for SYMFLUENCE

This module provides the CERRA-specific implementation for forcing data processing.
CERRA (Copernicus European Regional Reanalysis) covers Europe at 5.5 km resolution.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from shapely.geometry import Polygon

from ...utils import VariableStandardizer
from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('cerra')
class CERRAHandler(BaseDatasetHandler):
    """Handler for CERRA (Copernicus European Regional Reanalysis) dataset."""

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        CERRA variable name mapping to standard names.

        Uses centralized VariableStandardizer for consistency across the codebase.

        Returns:
            Dictionary mapping CERRA variable names to standard names
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('CERRA')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CERRA dataset with variable renaming and unit conversions.

        CERRA data typically comes in standard units but may need some adjustments.

        Args:
            ds: Input CERRA dataset

        Returns:
            Processed dataset with standardized variables
        """
        # Handle time coordinate (CERRA sometimes uses valid_time)
        if 'time' not in ds and 'valid_time' in ds:
            ds = ds.rename({'valid_time': 'time'})

        # Rename variables
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}

        if existing_vars:
            ds = ds.rename(existing_vars)

        # Calculate wind speed from components if not present
        if 'wind_speed' not in ds and 'eastward_wind' in ds and 'northward_wind' in ds:
            u = ds['eastward_wind']
            v = ds['northward_wind']
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = 'wind_speed'
            ds['wind_speed'] = windspd

        # Convert total precipitation from accumulated to rate if needed
        if 'precipitation_flux' in ds:
            p = ds['precipitation_flux']
            if 'units' in p.attrs:
                units = p.attrs['units'].lower()
                # Check if it's an accumulation (kg m-2 or m) and not already a rate (s-1 or rate)
                if ('m' in units or 'kg' in units) and 's-1' not in units and 'rate' not in units and 'hour' not in units:
                    # Accumulated meters or kg/m2 - convert to rate
                    time_diff = ds.time.diff('time').median()
                    if time_diff:
                        seconds = time_diff.values / np.timedelta64(1, 's')
                        ds['precipitation_flux'] = p / float(seconds)

        # Apply standard CF-compliant attributes (uses centralized definitions)
        # CERRA precipitation is in kg m-2 s-1 (equiv to mm/s) after conversion
        ds = self.apply_standard_attributes(ds, overrides={
            'precipitation_flux': {'units': 'kg m-2 s-1', 'standard_name': 'precipitation_rate'}
        })

        # Add metadata via base helpers
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "CERRA data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CERRA uses latitude/longitude coordinates.

        Returns:
            Tuple of ('latitude', 'longitude') or ('lat', 'lon')
        """
        # CERRA from CDS typically uses 'latitude' and 'longitude'
        return ('latitude', 'longitude')

    def needs_merging(self) -> bool:
        """CERRA data requires standardization/processing."""
        return True

    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Standardize CERRA forcings.

        This method processes CERRA data by:
        - Applying variable mapping
        - Converting units
        - Deriving additional variables (wind speed, specific humidity)
        """
        self.logger.info("Standardizing CERRA forcing files")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns = [
            f"{self.domain_name}_CERRA_*.nc",
            f"domain_{self.domain_name}_CERRA_*.nc",
            "*CERRA*.nc",
        ]

        files = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} CERRA file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No CERRA forcing files found in {raw_forcing_path} with patterns {patterns}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Filter to files whose year range overlaps the configured period
        all_files = files
        files = [
            f for f in all_files
            if self._file_overlaps_period(f, start_year, end_year)
        ]
        skipped = len(all_files) - len(files)
        if skipped:
            self.logger.info(
                f"Skipped {skipped} CERRA file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        if not files:
            self.logger.error(
                f"No CERRA files match the configured period {start_year}-{end_year}"
            )
            raise FileNotFoundError(
                f"No CERRA forcing files match the configured period "
                f"{start_year}-{end_year} in {raw_forcing_path}"
            )

        for f in files:
            # Skip intermediate analysis/forecast files that are not full forcing
            if '_analysis_' in f.name or '_forecast_' in f.name:
                self.logger.debug(f"Skipping intermediate file: {f.name}")
                continue

            self.logger.info(f"Processing CERRA file: {f}")
            try:
                ds = self.open_dataset(f)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error opening CERRA file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed CERRA forcing: {out_name}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error processing CERRA dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("CERRA forcing standardization completed")

    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create CERRA grid shapefile.

        CERRA uses a regular latitude-longitude grid over Europe.

        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to CERRA data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics

        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating CERRA grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        try:
            # Find a processed CERRA file
            cerra_files = list(merged_forcing_path.glob('*.nc'))
            if not cerra_files:
                raise FileNotFoundError("No processed CERRA files found")
            cerra_file = cerra_files[0]

            self.logger.info(f"Using CERRA file: {cerra_file}")

            # Read CERRA data
            with self.open_dataset(cerra_file) as ds:
                var_lat, var_lon = self.get_coordinate_names()

                # Handle both 1D and 2D coordinates
                if var_lat in ds.coords:
                    lats = ds.coords[var_lat].values
                elif var_lat in ds.variables:
                    lats = ds[var_lat].values
                else:
                    # Try alternative names
                    if 'lat' in ds.coords:
                        lats = ds.coords['lat'].values
                    elif 'lat' in ds.variables:
                        lats = ds['lat'].values
                    else:
                        raise KeyError("Latitude coordinate not found in CERRA file")

                if var_lon in ds.coords:
                    lons = ds.coords[var_lon].values
                elif var_lon in ds.variables:
                    lons = ds[var_lon].values
                else:
                    # Try alternative names
                    if 'lon' in ds.coords:
                        lons = ds.coords['lon'].values
                    elif 'lon' in ds.variables:
                        lons = ds['lon'].values
                    else:
                        raise KeyError("Longitude coordinate not found in CERRA file")

            self.logger.info(f"CERRA grid dimensions: lat={lats.shape}, lon={lons.shape}")

            # Get HRU bounding box for spatial filtering
            # Read the HRU shapefile to get its extent
            # shapefile_path is .../shapefiles/forcing, so parent is .../shapefiles
            hru_shapefile_dir = shapefile_path.parent / 'catchment'
            domain_name = self._get_config_value(lambda: self.config.domain.name, default='domain')
            hru_shapefile = hru_shapefile_dir / f"{domain_name}_HRUs_GRUs.shp"

            self.logger.info(f"Looking for HRU shapefile at: {hru_shapefile}")

            bbox_filter = None
            if hru_shapefile.exists():
                try:
                    import geopandas as gpd
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
                    self.logger.info("✓ Applying spatial filter based on HRU extent:")
                    self.logger.info(f"  Lon: {bbox_filter['lon_min']:.2f} to {bbox_filter['lon_max']:.2f}")
                    self.logger.info(f"  Lat: {bbox_filter['lat_min']:.2f} to {bbox_filter['lat_max']:.2f}")
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Could not read HRU shapefile for spatial filtering: {e}")
                    self.logger.warning("Will create shapefile for full domain (may be slow)")
            else:
                self.logger.warning(f"HRU shapefile not found: {hru_shapefile}")
                self.logger.warning("Will create shapefile for full domain (may be slow)")

            # Create geometries
            self.logger.info("Creating CERRA grid cell geometries")

            geometries = []
            ids = []
            center_lats = []
            center_lons = []

            # Handle 1D grid (regular lat/lon)
            if lats.ndim == 1 and lons.ndim == 1:
                # SPECIAL CASE: Single point forcing (1x1 grid)
                if len(lats) == 1 and len(lons) == 1:
                    self.logger.info("Detected 1x1 CERRA grid. Creating catchment-covering polygon.")

                    # If we have the HRU shapefile, use it to ensure full coverage
                    poly_created = False
                    if hru_shapefile.exists():
                        try:
                            # Re-read or use existing if efficient (safest to re-read to be sure)
                            if 'hru_gdf' not in locals():
                                hru_gdf = gpd.read_file(hru_shapefile)
                                # Reproject to WGS84 if needed
                                if hru_gdf.crs and hru_gdf.crs != 'EPSG:4326':
                                    hru_gdf = hru_gdf.to_crs('EPSG:4326')

                            minx, miny, maxx, maxy = hru_gdf.total_bounds

                            # Add a generous buffer (0.1 deg ~ 10km) to ensure full coverage
                            # This forces the single forcing point to map to ALL HRUs
                            buffer = 0.1
                            verts = [
                                [minx - buffer, miny - buffer],
                                [minx - buffer, maxy + buffer],
                                [maxx + buffer, maxy + buffer],
                                [maxx + buffer, miny - buffer],
                                [minx - buffer, miny - buffer],
                            ]
                            geometries.append(Polygon(verts))
                            self.logger.info("Created polygon matching HRU extent for 1x1 forcing.")
                            poly_created = True
                        except Exception as e:  # noqa: BLE001 — preprocessing resilience
                            self.logger.warning(f"Failed to use HRU bounds for 1x1 forcing: {e}")

                    if not poly_created:
                        # Fallback to small box around point if HRU shapefile fails
                        self.logger.warning("Using default small box for 1x1 grid (may not intersect catchment)")
                        center_lat, center_lon = float(lats[0]), float(lons[0])
                        d = 0.025
                        verts = [
                            [center_lon - d, center_lat - d],
                            [center_lon - d, center_lat + d],
                            [center_lon + d, center_lat + d],
                            [center_lon + d, center_lat - d],
                            [center_lon - d, center_lat - d],
                        ]
                        geometries.append(Polygon(verts))

                    ids.append(0)
                    center_lats.append(float(lats[0]))
                    center_lons.append(float(lons[0]))
                    cells_created = 1

                # Regular grid (more than 1x1)
                else:
                    # Regular grid - create cell boundaries
                    half_dlat = abs(lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.025
                    half_dlon = abs(lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.025

                    cell_id = 0
                    total_cells = len(lats) * len(lons)
                    cells_created = 0

                    for i, center_lon in enumerate(lons):
                        for j, center_lat in enumerate(lats):
                            # Apply spatial filter if available
                            if bbox_filter is not None:
                                if (center_lon < bbox_filter['lon_min'] or center_lon > bbox_filter['lon_max'] or
                                    center_lat < bbox_filter['lat_min'] or center_lat > bbox_filter['lat_max']):
                                    continue

                            verts = [
                                [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                                [float(center_lon) - half_dlon, float(center_lat) + half_dlat],
                                [float(center_lon) + half_dlon, float(center_lat) + half_dlat],
                                [float(center_lon) + half_dlon, float(center_lat) - half_dlat],
                                [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                            ]
                            geometries.append(Polygon(verts))
                            ids.append(cell_id)
                            center_lats.append(float(center_lat))
                            center_lons.append(float(center_lon))
                            cells_created += 1

                            if cells_created % 1000 == 0:
                                self.logger.info(f"Created {cells_created} CERRA grid cells (filtered from {cell_id}/{total_cells})")

                            cell_id += 1

                    if bbox_filter is not None:
                        self.logger.info(f"Spatial filtering: created {cells_created} cells (from {total_cells} total)")
            else:
                # 2D grid (Lambert Conformal - CERRA uses this)
                ny, nx = lats.shape
                total_cells = ny * nx
                cells_created = 0

                # Pre-filter grid indices if bounding box is available
                if bbox_filter is not None:
                    # Create mask for cells within bounding box
                    mask = ((lons >= bbox_filter['lon_min']) & (lons <= bbox_filter['lon_max']) &
                            (lats >= bbox_filter['lat_min']) & (lats <= bbox_filter['lat_max']))
                    # Get indices of cells within bbox
                    indices = np.where(mask)
                    indices_list = list(zip(indices[0], indices[1]))
                    self.logger.info(f"Spatial filtering: {len(indices_list)} cells within bbox (from {total_cells} total)")
                else:
                    # No filter - use all indices
                    indices_list = [(i, j) for i in range(ny) for j in range(nx)]

                for i, j in indices_list:
                    # Get cell center coordinates
                    center_lat = float(lats[i, j])
                    center_lon = float(lons[i, j])

                    # Create cell from corners
                    lat_corners = [
                        lats[i, j],
                        lats[i, j + 1] if j + 1 < nx else lats[i, j],
                        lats[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lats[i, j],
                        lats[i + 1, j] if i + 1 < ny else lats[i, j],
                    ]
                    lon_corners = [
                        lons[i, j],
                        lons[i, j + 1] if j + 1 < nx else lons[i, j],
                        lons[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lons[i, j],
                        lons[i + 1, j] if i + 1 < ny else lons[i, j],
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * nx + j)
                    center_lats.append(center_lat)
                    center_lons.append(center_lon)
                    cells_created += 1

                    if cells_created % 1000 == 0:
                        self.logger.info(f"Created {cells_created} CERRA grid cells")

                if bbox_filter is not None:
                    self.logger.info(f"Spatial filtering: created {cells_created} cells (from {total_cells} total)")

            # Create GeoDataFrame
            import geopandas as gpd
            self.logger.info("Creating GeoDataFrame")
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): center_lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): center_lons,
            }, crs='EPSG:4326')

            # Calculate elevation
            self.logger.info("Calculating elevation values")
            elevations = elevation_calculator(gdf, dem_path, batch_size=50)
            gdf['elev_m'] = elevations

            # Save the shapefile
            self.logger.info(f"Saving CERRA shapefile to {output_shapefile}")
            shapefile_path.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_shapefile)
            self.logger.info(f"CERRA grid shapefile created and saved to {output_shapefile}")

            return output_shapefile

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error in create_cerra_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
