# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GDAL raster processing operations.

Provides raster to vector conversion for watershed delineation.
Used by both distributed and lumped delineators.

Refactored from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path
from typing import Any

import geopandas as gpd

try:
    from osgeo import gdal, ogr
except ImportError:
    gdal = None
    ogr = None


class GDALProcessor:
    """
    GDAL operations for raster processing.

    Handles raster-to-vector conversion using both GDAL library
    and command-line tools as fallback.
    """

    def __init__(self, logger: Any):
        """
        Initialize GDAL processor.

        Args:
            logger: Logger instance
        """
        if gdal is None:
            raise ImportError(
                "GDAL Python bindings (osgeo) are required for GDALProcessor. "
                "Install with: pip install GDAL"
            )
        self.logger = logger

    def _mask_ocean_watersheds(self, interim_dir: Path, sea_level: float) -> str:
        """Set watershed cells at/below ``sea_level`` to nodata, return new raster path.

        Uses the pit-filled DEM (``elv-fel.tif``), which is co-registered with the
        watershed grid. If the DEM is missing, returns the original raster
        unchanged so delineation still proceeds.
        """
        import rasterio

        ws_path = interim_dir / "elv-watersheds.tif"
        fel_path = interim_dir / "elv-fel.tif"
        if not fel_path.exists():
            self.logger.warning(f"{fel_path.name} not found; cannot mask ocean, polygonizing as-is")
            return str(ws_path)

        with rasterio.open(ws_path) as src:
            watersheds = src.read(1)
            profile = src.profile
            nodata = src.nodata
        with rasterio.open(fel_path) as src:
            elevation = src.read(1)

        if nodata is None:
            nodata = -2147483647
            profile.update(nodata=nodata)

        ocean = elevation <= sea_level
        n_ocean = int(ocean.sum())
        watersheds = watersheds.copy()
        watersheds[ocean] = nodata

        masked_path = interim_dir / "elv-watersheds-landmasked.tif"
        with rasterio.open(masked_path, "w", **profile) as dst:
            dst.write(watersheds, 1)

        pct = 100.0 * n_ocean / watersheds.size if watersheds.size else 0.0
        self.logger.info(
            f"Masked {n_ocean} ocean cells ({pct:.1f}%, elev<={sea_level} m) from watershed "
            f"raster before polygonization to remove cross-ocean tentacle basins"
        )
        return str(masked_path)

    def run_gdal_processing(self, interim_dir: Path, mask_ocean: bool = False, sea_level: float = 0.0):
        """
        Convert watershed raster to polygon shapefile.

        Used by distributed delineation. Attempts direct GDAL polygonization
        first, falls back to command-line tool if that fails.

        Args:
            interim_dir: Directory containing interim TauDEM outputs
            mask_ocean: When True, set watershed cells at or below ``sea_level``
                (per the pit-filled DEM ``elv-fel.tif``) to nodata before
                polygonizing. Coastal DEMs encode the sea as flat 0-elevation,
                which TauDEM flow-routes in straight D8 lines to the domain edge
                and assigns to watersheds — producing thin radial "tentacle"
                basins across the ocean. Masking the sea removes the cause.
            sea_level: Elevation (m) at/below which a cell is treated as ocean.

        Raises:
            RuntimeError: If all polygonization attempts fail
        """
        # Ensure output directory exists
        interim_dir.mkdir(parents=True, exist_ok=True)

        input_raster = str(interim_dir / "elv-watersheds.tif")
        output_shapefile = str(interim_dir / "basin-watersheds.shp")

        if mask_ocean:
            input_raster = self._mask_ocean_watersheds(interim_dir, sea_level)

        try:
            # First attempt: Using gdal.Polygonize directly
            src_ds = gdal.Open(input_raster)
            if src_ds is None:
                raise RuntimeError(f"Could not open input raster: {input_raster}")

            srcband = src_ds.GetRasterBand(1)

            # Create output shapefile
            drv = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(output_shapefile):
                drv.DeleteDataSource(output_shapefile)

            dst_ds = drv.CreateDataSource(output_shapefile)
            if dst_ds is None:
                raise RuntimeError(f"Could not create output shapefile: {output_shapefile}")

            # Carry the raster's CRS so the shapefile is not later assumed to
            # be EPSG:4326 (the DEM may be projected).
            raster_wkt = src_ds.GetProjection()
            layer_srs = None
            if raster_wkt:
                from osgeo import osr
                layer_srs = osr.SpatialReference()
                layer_srs.ImportFromWkt(raster_wkt)
            dst_layer = dst_ds.CreateLayer("watersheds", srs=layer_srs)
            if dst_layer is None:
                raise RuntimeError("Could not create output layer")

            # Add field for raster value
            fd = ogr.FieldDefn("DN", ogr.OFTInteger)
            dst_layer.CreateField(fd)

            # Run polygonize
            gdal.Polygonize(srcband, srcband.GetMaskBand(), dst_layer, 0)

            # Cleanup
            dst_ds = None
            src_ds = None

            self.logger.info("Completed GDAL polygonization using direct method")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Direct polygonization failed: {str(e)}, trying command line method...", exc_info=True)
            try:
                # Second attempt: Using command line tool without MPI
                command = [
                    "gdal_polygonize.py",
                    "-f", "ESRI Shapefile",
                    str(input_raster),
                    str(output_shapefile)
                ]
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.logger.info("Completed GDAL polygonization using command line method")

            except subprocess.CalledProcessError as cmd_err:
                error_msg = f"gdal_polygonize.py failed with exit code {cmd_err.returncode}"
                if cmd_err.stderr:
                    error_msg += f"\nstderr: {cmd_err.stderr}"
                if cmd_err.stdout:
                    error_msg += f"\nstdout: {cmd_err.stdout}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from cmd_err
            except FileNotFoundError:
                self.logger.error("gdal_polygonize.py not found in PATH")
                raise RuntimeError(
                    "gdal_polygonize.py not found. Ensure GDAL is installed and in PATH. "
                    "On macOS, try: brew install gdal"
                ) from None
            except Exception as fallback_err:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"All polygonization attempts failed: {str(fallback_err)}")
                raise

    def raster_to_polygon(self, raster_path: Path, output_shp_path: Path):
        """
        Convert a watershed raster to a polygon shapefile.

        Keeps every polygon covering valid (non-nodata) cells, carrying the cell
        value through in the ``ID`` field so callers can select by basin ID.

        This used to filter to ``ID == 1``, which silently broke the pixel-exact
        lumped path: TauDEM's ``gagewatershed`` labels each basin with the *gauge's*
        id from ``moveoutletstostreams``, and that id is 0-based — a single-gauge
        run produces a watershed of value 0, never 1. The filter therefore raised on
        every lumped delineation and the caller fell back to dissolving streamnet
        sub-watersheds, whose outlet polygon extends downstream past the gauge
        (Bow at Banff: 2248 km^2 instead of 2208 km^2, gauge 3.7 km inside the
        boundary rather than on it).

        Args:
            raster_path: Path to the input raster file
            output_shp_path: Path to save the output shapefile

        Raises:
            ValueError: If the raster contains no valid (non-nodata) cells
        """
        gdal.UseExceptions()
        ogr.UseExceptions()

        # Open the raster
        raster = gdal.Open(str(raster_path))
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        raster_wkt = raster.GetProjection()

        # Create a temporary shapefile
        temp_shp_path = output_shp_path.with_name(output_shp_path.stem + "_temp.shp")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        temp_ds = driver.CreateDataSource(str(temp_shp_path))
        temp_layer = temp_ds.CreateLayer("watershed", srs=None)

        # Add a field to the layer
        field_def = ogr.FieldDefn("ID", ogr.OFTInteger)
        temp_layer.CreateField(field_def)

        # Polygonize the raster. The band's own validity mask excludes nodata
        # cells, so only real watershed cells become polygons.
        gdal.Polygonize(band, band.GetMaskBand(), temp_layer, 0, [], callback=None)

        # Close the temporary datasource
        temp_ds = None
        raster = None

        # Read the temporary shapefile with geopandas
        gdf = gpd.read_file(temp_shp_path)

        # Drop anything that slipped through as nodata (older GDAL builds hand back
        # a mask band that does not honour the nodata value on every driver).
        if nodata is not None:
            gdf = gdf[gdf['ID'] != int(nodata)]
        filtered_gdf = gdf.sort_values('ID').reset_index(drop=True)
        filtered_gdf = filtered_gdf.set_crs(raster_wkt or 'epsg:4326')

        if filtered_gdf.empty:
            self.logger.error(f"No valid (non-nodata) cells found in watershed raster: {raster_path}")
            raise ValueError(f"No valid (non-nodata) cells found in watershed raster: {raster_path}")

        self.logger.debug(
            f"Polygonized {raster_path.name} to {len(filtered_gdf)} polygon(s) "
            f"with ID(s): {sorted(filtered_gdf['ID'].unique().tolist())[:10]}"
        )

        # Save the filtered GeoDataFrame to the final shapefile
        filtered_gdf.to_file(output_shp_path)

        # Remove all temporary files
        temp_files = glob.glob(str(temp_shp_path.with_suffix(".*")))
        for temp_file in temp_files:
            Path(temp_file).unlink()
            self.logger.debug(f"Removed temporary file: {temp_file}")

        self.logger.debug(f"Filtered watershed shapefile created at: {output_shp_path}")
