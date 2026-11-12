# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Glacier Data Acquisition Handler.

Acquires glacier data from the Randolph Glacier Inventory (RGI) 7.0
and processes it into rasters for SUMMA glacier simulations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# RGI 7.0 region information
# Regions are defined by NSIDC: https://nsidc.org/data/nsidc-0770
RGI_REGIONS = {
    1: {"name": "Alaska", "bbox": (52.0, -176.0, 72.0, -126.0)},
    2: {"name": "Western Canada and USA", "bbox": (37.0, -130.0, 60.0, -100.0)},
    3: {"name": "Arctic Canada North", "bbox": (74.0, -130.0, 84.0, -55.0)},
    4: {"name": "Arctic Canada South", "bbox": (58.0, -100.0, 75.0, -55.0)},
    5: {"name": "Greenland Periphery", "bbox": (59.0, -75.0, 84.0, -10.0)},
    6: {"name": "Iceland", "bbox": (63.0, -25.0, 67.0, -13.0)},
    7: {"name": "Svalbard and Jan Mayen", "bbox": (70.0, -10.0, 82.0, 35.0)},
    8: {"name": "Scandinavia", "bbox": (58.0, 4.0, 72.0, 32.0)},
    9: {"name": "Russian Arctic", "bbox": (68.0, 30.0, 82.0, 180.0)},
    10: {"name": "North Asia", "bbox": (42.0, 65.0, 78.0, 180.0)},
    11: {"name": "Central Europe", "bbox": (42.0, 5.0, 48.0, 18.0)},
    12: {"name": "Caucasus and Middle East", "bbox": (32.0, 38.0, 46.0, 55.0)},
    13: {"name": "Central Asia", "bbox": (27.0, 67.0, 50.0, 100.0)},
    14: {"name": "South Asia West", "bbox": (25.0, 66.0, 40.0, 82.0)},
    15: {"name": "South Asia East", "bbox": (25.0, 80.0, 40.0, 105.0)},
    16: {"name": "Low Latitudes", "bbox": (-20.0, -82.0, 20.0, 120.0)},
    17: {"name": "Southern Andes", "bbox": (-56.0, -76.0, -17.0, -62.0)},
    18: {"name": "New Zealand", "bbox": (-47.0, 166.0, -43.0, 175.0)},
    19: {"name": "Antarctic and Subantarctic", "bbox": (-90.0, -180.0, -60.0, 180.0)},
    20: {"name": "Subantarctic and Antarctic Islands", "bbox": (-60.0, -180.0, -45.0, 180.0)},
}

# NSIDC RGI 7.0 download URLs (requires EarthData authentication)
# Note: NSIDC requires authentication, so we primarily use GLIMS WFS as fallback
RGI_BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/"

# GLIMS WFS configuration
GLIMS_WFS_URL = "https://www.glims.org/geoserver/GLIMS/wfs"
GLIMS_WFS_TIMEOUT = 300  # Increased timeout for slow server


@R.acquisition_handlers.add('GLACIER')
@R.acquisition_handlers.add('RGI')
class GlacierAcquirer(BaseAcquisitionHandler):
    """
    Glacier data acquisition handler.

    Downloads RGI 7.0 glacier outlines and processes them into
    rasters required for SUMMA glacier simulations:
    - domain_type.tif: Glacier domain classification
    - hru_id.tif: HRU mapping
    - rgi_id.tif: RGI glacier IDs

    Also creates catchment intersection shapefiles with glacier
    domain statistics per HRU.
    """

    # Domain type classifications
    DOMAIN_UPLAND = 1
    DOMAIN_GLACIER_CLEAN_1 = 2  # Accumulation zone
    DOMAIN_GLACIER_CLEAN_2 = 3  # Ablation zone (clean)
    DOMAIN_GLACIER_DEBRIS = 4
    DOMAIN_WETLAND = 5

    # Default resolution for glacier rasters (degrees)
    DEFAULT_RESOLUTION = 0.0001  # ~10m at equator

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Any = None):
        super().__init__(config, logger, reporting_manager)
        self.session = self._create_nsidc_session()

    def _create_nsidc_session(self):
        """Session for NSIDC (NASA Earthdata): earthaccess first, then token /
        credentials, else unauthenticated (RGI/GLIMS mirrors need no login)."""
        return self._earthdata_auth().session(prefer_earthaccess=True, max_retries=5)

    def download(self, output_dir: Path) -> Path:
        """
        Download and process glacier data for the domain.

        Args:
            output_dir: Output directory (typically project_dir/attributes/glaciers)

        Returns:
            Path to glacier data directory
        """
        glacier_dir = self._attribute_dir("glaciers")
        glacier_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        domain_type_file = glacier_dir / f"domain_{self.domain_name}_domain_type.tif"
        if domain_type_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            self.logger.info(f"Glacier data already exists: {glacier_dir}")
            return glacier_dir

        self.logger.info("Starting glacier data acquisition from RGI 7.0")

        # 1. Determine which RGI regions to download
        regions = self._determine_rgi_regions()
        if not regions:
            self.logger.warning("No RGI regions intersect with domain bounding box")
            return glacier_dir

        self.logger.info(f"RGI regions to download: {regions}")

        # 2. Download RGI data for relevant regions
        rgi_gdf = self._download_rgi_regions(regions, glacier_dir)
        if rgi_gdf is None or len(rgi_gdf) == 0:
            self.logger.warning("No glaciers found in domain bounding box")
            self.logger.info(
                "To manually add glacier data:\n"
                "  1. Download RGI 7.0 from https://nsidc.org/data/nsidc-0770 (requires EarthData login)\n"
                "  2. Or use GLIMS data from https://www.glims.org/maps/glims\n"
                f"  3. Place glacier shapefiles in: {glacier_dir}\n"
                "  4. Run the preprocessing step again"
            )
            return glacier_dir

        self.logger.info(f"Found {len(rgi_gdf)} glaciers in domain")

        # 3. Create glacier rasters
        self._create_glacier_rasters(rgi_gdf, glacier_dir)

        # 4. Create catchment intersection shapefiles
        self._create_intersection_shapefiles(rgi_gdf, glacier_dir)

        self.logger.info(f"Glacier data acquisition complete: {glacier_dir}")
        return glacier_dir

    def _determine_rgi_regions(self) -> List[int]:
        """Determine which RGI regions intersect with the domain bounding box."""
        regions = []
        domain_box = box(
            self.bbox['lon_min'],
            self.bbox['lat_min'],
            self.bbox['lon_max'],
            self.bbox['lat_max']
        )

        for region_id, region_info in RGI_REGIONS.items():
            region_bbox = region_info['bbox']
            region_box = box(region_bbox[1], region_bbox[0], region_bbox[3], region_bbox[2])
            if domain_box.intersects(region_box):
                regions.append(region_id)

        return regions

    def _download_rgi_regions(self, regions: List[int], glacier_dir: Path) -> Optional[gpd.GeoDataFrame]:
        """Download RGI data for specified regions and clip to domain."""
        cache_dir = glacier_dir / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_glaciers = []

        for region_id in regions:
            try:
                region_gdf = self._download_single_region(region_id, cache_dir)
                if region_gdf is not None and len(region_gdf) > 0:
                    all_glaciers.append(region_gdf)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to download RGI region {region_id}: {e}", exc_info=True)
                continue

        if not all_glaciers:
            return None

        # Combine all regions
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_glaciers, ignore_index=True))

        # Clip to domain bounding box
        domain_box = box(
            self.bbox['lon_min'],
            self.bbox['lat_min'],
            self.bbox['lon_max'],
            self.bbox['lat_max']
        )
        clipped_gdf = combined_gdf[combined_gdf.intersects(domain_box)]

        return clipped_gdf

    def _download_single_region(self, region_id: int, cache_dir: Path) -> Optional[gpd.GeoDataFrame]:
        """Download RGI data for a single region."""

        # RGI 7.0 file naming: RGI2000-v7.0-G-02_western_canada_usa.zip
        region_name = RGI_REGIONS[region_id]['name'].lower().replace(' and ', '_').replace(' ', '_')

        filename = f"RGI2000-v7.0-G-{region_id:02d}_{region_name}.zip"

        cached_file = cache_dir / filename.replace('.zip', '.shp')

        if cached_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            self.logger.debug(f"Using cached RGI data: {cached_file}")
            return gpd.read_file(cached_file)

        # Try to download from NSIDC (requires EarthData auth)
        url = f"{RGI_BASE_URL}{filename}"
        self.logger.info(f"Downloading RGI region {region_id} from {url}")

        try:
            response = self.session.get(url, timeout=300, stream=True)

            if response.status_code in [401, 302]:
                # NSIDC requires authentication - try alternative sources
                self.logger.warning("NSIDC requires authentication, trying alternative source")
                return self._download_from_alternative(region_id, cache_dir)

            response.raise_for_status()

            # Download zip file
            zip_path = cache_dir / filename
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            # Extract shapefile from zip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                from symfluence.core.archive_extraction import safe_zip_extract
                safe_zip_extract(zf, cache_dir)

            # Find the extracted shapefile
            for shp_file in cache_dir.glob(f"RGI2000-v7.0-G-{region_id:02d}*.shp"):
                return gpd.read_file(shp_file)

            self.logger.warning(f"No shapefile found in {zip_path}")
            return self._download_from_alternative(region_id, cache_dir)

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to download from NSIDC: {e}", exc_info=True)
            return self._download_from_alternative(region_id, cache_dir)

    def _download_from_alternative(self, region_id: int, cache_dir: Path) -> Optional[gpd.GeoDataFrame]:
        """
        Try alternative sources for RGI data.

        Falls back to GLIMS WFS service or pre-downloaded local files.
        """
        # Check for local RGI data in cache dir first
        for pattern in [f"*{region_id:02d}*.shp", f"*{region_id:02d}*.gpkg"]:
            for local_file in cache_dir.glob(pattern):
                self.logger.info(f"Using cached RGI file: {local_file}")
                return gpd.read_file(local_file)

        # Check for local RGI data in config-specified directory
        local_rgi_dir = Path(self._get_config_value(lambda: None, default='', dict_key='RGI_LOCAL_DIR'))
        if local_rgi_dir.exists():
            for pattern in [f"*{region_id:02d}*.shp", f"*{region_id:02d}*.gpkg"]:
                for local_file in local_rgi_dir.glob(pattern):
                    self.logger.info(f"Using local RGI file: {local_file}")
                    return gpd.read_file(local_file)

        # Try GLIMS WFS service (limited but public)
        try:
            self.logger.info("Attempting GLIMS WFS download (may take several minutes)...")
            return self._download_from_glims_wfs()
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GLIMS WFS failed: {e}", exc_info=True)

        return None

    def _download_from_glims_wfs(self) -> Optional[gpd.GeoDataFrame]:
        """Download glacier outlines from GLIMS WFS service."""
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typeName': 'GLIMS:glacier_outlines',
            'outputFormat': 'application/json',
            'bbox': f"{self.bbox['lat_min']},{self.bbox['lon_min']},{self.bbox['lat_max']},{self.bbox['lon_max']},EPSG:4326",
        }

        self.logger.info("Downloading glacier data from GLIMS WFS")
        self.logger.info(f"GLIMS WFS bbox: {params['bbox']}")

        try:
            response = self.session.get(
                GLIMS_WFS_URL,
                params=params,
                timeout=GLIMS_WFS_TIMEOUT
            )
            response.raise_for_status()

            geojson_data = response.json()

            if not geojson_data.get('features'):
                self.logger.warning("No glaciers found in GLIMS WFS response")
                return None

            self.logger.info(f"Found {len(geojson_data['features'])} glaciers from GLIMS")
            return gpd.GeoDataFrame.from_features(geojson_data['features'], crs='EPSG:4326')

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GLIMS WFS request failed: {e}", exc_info=True)
            return None

    def _create_glacier_rasters(self, rgi_gdf: gpd.GeoDataFrame, glacier_dir: Path):
        """Create glacier rasters from RGI data."""
        self.logger.info("Creating glacier rasters")

        # Determine resolution and bounds
        resolution = self._get_config_value(lambda: None, default=self.DEFAULT_RESOLUTION, dict_key='GLACIER_RASTER_RESOLUTION')

        # Calculate raster dimensions
        width = int((self.bbox['lon_max'] - self.bbox['lon_min']) / resolution)
        height = int((self.bbox['lat_max'] - self.bbox['lat_min']) / resolution)

        # Ensure reasonable dimensions
        max_dim = 5000
        if width > max_dim or height > max_dim:
            scale = max(width, height) / max_dim
            width = int(width / scale)
            height = int(height / scale)
            resolution = (self.bbox['lon_max'] - self.bbox['lon_min']) / width
            self.logger.warning(f"Reduced raster resolution to {resolution:.6f} degrees")

        transform = from_bounds(
            self.bbox['lon_min'], self.bbox['lat_min'],
            self.bbox['lon_max'], self.bbox['lat_max'],
            width, height
        )

        # Create domain_type raster
        domain_type_file = glacier_dir / f"domain_{self.domain_name}_domain_type.tif"
        self._create_domain_type_raster(rgi_gdf, domain_type_file, transform, width, height)

        # Create rgi_id raster
        rgi_id_file = glacier_dir / f"domain_{self.domain_name}_rgi_id.tif"
        self._create_rgi_id_raster(rgi_gdf, rgi_id_file, transform, width, height)

        # Create hru_id raster (placeholder - will be updated during preprocessing)
        hru_id_file = glacier_dir / f"domain_{self.domain_name}_hru_id.tif"
        self._create_hru_id_raster(hru_id_file, transform, width, height)

        self.logger.info("Glacier rasters created successfully")

    def _create_domain_type_raster(
        self,
        rgi_gdf: gpd.GeoDataFrame,
        output_file: Path,
        transform,
        width: int,
        height: int
    ):
        """Create domain type raster from glacier outlines."""
        # Initialize with upland (domain type 1)
        domain_type = np.full((height, width), self.DOMAIN_UPLAND, dtype=np.int16)

        # Determine glacier domain type based on debris cover
        for idx, glacier in rgi_gdf.iterrows():
            # Check for debris cover information
            debris_frac = glacier.get('debris_frac', 0.0)
            if debris_frac is None:
                debris_frac = 0.0

            # Determine domain type
            if debris_frac > 0.5:
                domain_type_val = self.DOMAIN_GLACIER_DEBRIS
            else:
                # Split clean glacier into accumulation/ablation zones
                # This is a simplification - actual zones should come from DEM analysis
                domain_type_val = self.DOMAIN_GLACIER_CLEAN_1

            # Rasterize this glacier
            glacier_mask = features.rasterize(
                [(glacier.geometry, domain_type_val)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.int16
            )

            # Update where glacier exists
            domain_type = np.where(glacier_mask > 0, glacier_mask, domain_type)

        # Write raster
        profile = {
            'driver': 'GTiff',
            'dtype': 'int16',
            'width': width,
            'height': height,
            'count': 1,
            'crs': CRS.from_epsg(4326),
            'transform': transform,
            'compress': 'lzw',
            'nodata': 0
        }

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(domain_type, 1)

        self.logger.debug(f"Created domain_type raster: {output_file}")

    def _create_rgi_id_raster(
        self,
        rgi_gdf: gpd.GeoDataFrame,
        output_file: Path,
        transform,
        width: int,
        height: int
    ):
        """Create RGI ID raster from glacier outlines."""
        # Initialize with no glacier (0)
        rgi_ids = np.zeros((height, width), dtype=np.float32)

        # Map RGI IDs to numeric values
        rgi_id_map = {}
        for idx, glacier in rgi_gdf.iterrows():
            # Get RGI ID - try different column names
            rgi_id = glacier.get('rgi_id', glacier.get('RGIId', glacier.get('glac_id', idx + 1)))

            # Convert string ID to numeric
            if isinstance(rgi_id, str):
                # RGI IDs are like "RGI2000-v7.0-G-01-00001"
                # Extract numeric part
                try:
                    numeric_id = int(rgi_id.split('-')[-1])
                except (ValueError, IndexError):
                    numeric_id = idx + 1
            else:
                numeric_id = int(rgi_id) if rgi_id else idx + 1

            rgi_id_map[idx] = numeric_id

            # Rasterize this glacier
            glacier_mask = features.rasterize(
                [(glacier.geometry, numeric_id)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.float32
            )

            # Update where glacier exists
            rgi_ids = np.where(glacier_mask > 0, glacier_mask, rgi_ids)

        # Write raster
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': width,
            'height': height,
            'count': 1,
            'crs': CRS.from_epsg(4326),
            'transform': transform,
            'compress': 'lzw',
            'nodata': 0
        }

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(rgi_ids, 1)

        self.logger.debug(f"Created rgi_id raster: {output_file}")

    def _create_hru_id_raster(
        self,
        output_file: Path,
        transform,
        width: int,
        height: int
    ):
        """Create placeholder HRU ID raster."""
        # Initialize with single HRU (1) - will be updated during preprocessing
        hru_ids = np.ones((height, width), dtype=np.float32)

        # Write raster
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': width,
            'height': height,
            'count': 1,
            'crs': CRS.from_epsg(4326),
            'transform': transform,
            'compress': 'lzw',
            'nodata': 0
        }

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(hru_ids, 1)

        self.logger.debug(f"Created hru_id raster: {output_file}")

    def _create_intersection_shapefiles(self, rgi_gdf: gpd.GeoDataFrame, glacier_dir: Path):
        """Create catchment intersection shapefiles with glacier statistics."""
        self.logger.info("Creating catchment intersection shapefiles")

        # Find catchment shapefile
        catchment_file = self._find_catchment_shapefile()
        if catchment_file is None:
            self.logger.warning("Catchment shapefile not found - skipping intersection creation")
            return

        try:
            catchment_gdf = gpd.read_file(catchment_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to read catchment shapefile: {e}", exc_info=True)
            return

        # Ensure same CRS
        if rgi_gdf.crs != catchment_gdf.crs:
            rgi_gdf = rgi_gdf.to_crs(catchment_gdf.crs)

        # Create intersection directory
        intersect_dir = self.project_dir / 'shapefiles' / 'catchment_intersection'
        domain_type_dir = intersect_dir / 'with_domain_type'
        domain_type_dir.mkdir(parents=True, exist_ok=True)

        # Calculate glacier statistics per catchment
        intersected = self._calculate_glacier_intersection(catchment_gdf, rgi_gdf)

        # Save intersection shapefile
        output_file = domain_type_dir / 'catchment_with_domain_type.shp'
        intersected.to_file(output_file)
        self.logger.info(f"Created intersection shapefile: {output_file}")

    def _find_catchment_shapefile(self) -> Optional[Path]:
        """Find the catchment shapefile in the project directory."""
        shapefile_dir = self.project_dir / 'shapefiles' / 'catchment'

        if shapefile_dir.exists():
            for shp_file in shapefile_dir.glob('*.shp'):
                return shp_file

        # Try alternate location
        river_basins_dir = self.project_dir / 'shapefiles' / 'river_basins'
        if river_basins_dir.exists():
            for shp_file in river_basins_dir.glob('*.shp'):
                return shp_file

        return None

    def _calculate_glacier_intersection(
        self,
        catchment_gdf: gpd.GeoDataFrame,
        rgi_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Calculate glacier statistics per catchment unit."""
        result = catchment_gdf.copy()

        # Initialize domain type columns
        for i in range(1, 6):
            result[f'domType_{i}'] = 0.0

        # Calculate intersection for each catchment
        for idx, catchment in result.iterrows():
            catchment_area = catchment.geometry.area

            # Find intersecting glaciers
            intersecting = rgi_gdf[rgi_gdf.intersects(catchment.geometry)]

            if len(intersecting) == 0:
                # No glaciers - all upland
                result.loc[idx, 'domType_1'] = 1.0
                continue

            # Calculate glacier coverage fractions
            glacier_area = 0.0
            debris_area = 0.0

            for _, glacier in intersecting.iterrows():
                intersection = glacier.geometry.intersection(catchment.geometry)
                if not intersection.is_empty:
                    area = intersection.area
                    glacier_area += area

                    # Check for debris cover
                    debris_frac = glacier.get('debris_frac', 0.0)
                    if debris_frac is None:
                        debris_frac = 0.0
                    if debris_frac > 0.5:
                        debris_area += area

            # Calculate fractions
            glacier_frac = min(glacier_area / catchment_area, 1.0) if catchment_area > 0 else 0.0
            debris_frac = min(debris_area / catchment_area, 1.0) if catchment_area > 0 else 0.0
            clean_frac = max(glacier_frac - debris_frac, 0.0)
            upland_frac = max(1.0 - glacier_frac, 0.0)

            # Set domain type fractions
            result.loc[idx, 'domType_1'] = upland_frac  # Upland
            result.loc[idx, 'domType_2'] = clean_frac * 0.5  # Clean glacier (accumulation)
            result.loc[idx, 'domType_3'] = clean_frac * 0.5  # Clean glacier (ablation)
            result.loc[idx, 'domType_4'] = debris_frac  # Debris-covered
            # domType_5 (wetland) remains 0

        return result


# Import pandas for the handler
import pandas as pd
