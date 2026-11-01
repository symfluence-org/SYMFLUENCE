# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
JRC Global Surface Water Observation Handler.

Provides acquisition and preprocessing of JRC (Joint Research Centre) Global
Surface Water data for hydrological model validation.

JRC Global Surface Water Overview:
    Data Type: Landsat-derived surface water extent and dynamics
    Resolution: 30m
    Coverage: Global
    Variables: occurrence, recurrence, seasonality, change, transitions, extent
    Units: Percentage (0-100) or count (seasonality: 0-12 months)

Output Format:
    CSV with spatial statistics: mean, std, min, max water occurrence/extent
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('jrc_water')
@R.observation_handlers.add('jrc_gsw')
@R.observation_handlers.add('surface_water')
class JRCWaterHandler(BaseObservationHandler):
    """
    Handles JRC Global Surface Water data acquisition and processing.

    Provides basin-level surface water statistics from Landsat-derived
    JRC Global Surface Water dataset for reservoir/lake monitoring and
    model validation.
    """

    obs_type = "surface_water"
    source_name = "JRC_GSW"

    def acquire(self) -> Path:
        """
        Locate or download JRC Global Surface Water data.

        Returns:
            Path to directory containing JRC GeoTIFF files
        """
        data_access = self._get_config_value(
            lambda: self.config.domain.data_access,
            default='local'
        )
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        jrc_path = self._get_config_value(
            lambda: None,
            default='default', dict_key='JRC_WATER_PATH'
        )
        if isinstance(jrc_path, str) and jrc_path.lower() == 'default':
            jrc_dir = self.project_observations_dir / "surface_water" / "jrc"
        else:
            jrc_dir = Path(jrc_path)

        jrc_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        force_download = self._get_config_value(
            lambda: self.config.data.force_download,
            default=False
        )

        existing_files = list(jrc_dir.glob("*.tif"))
        if existing_files and not force_download:
            self.logger.info(f"Using existing JRC data: {len(existing_files)} files")
            return jrc_dir

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for JRC Global Surface Water")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('JRC_WATER', self.config, self.logger)
            return acquirer.download(jrc_dir)

        return jrc_dir

    def process(self, input_path: Path) -> Path:
        """
        Process JRC Global Surface Water GeoTIFF data to basin statistics.

        Args:
            input_path: Path to directory containing JRC GeoTIFF files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing JRC Global Surface Water for domain: {self.domain_name}")

        # Get dataset type from config
        dataset = self._get_config_value(
            lambda: None,
            default='occurrence',
            dict_key='JRC_WATER_DATASET'
        )
        if isinstance(dataset, str):
            dataset = dataset.lower()

        # Find GeoTIFF files
        self.logger.debug(f"Looking for JRC GeoTIFF files in {input_path}")
        tif_files = list(input_path.glob(f"*{dataset}*.tif"))
        if not tif_files:
            tif_files = list(input_path.glob("*.tif"))
        if not tif_files:
            self.logger.warning(f"No JRC GeoTIFF files found in {input_path}")
            return input_path

        self.logger.info(f"Processing {len(tif_files)} JRC files for dataset: {dataset}")

        # Get optional catchment mask
        catchment_path = self._get_catchment_path()

        results = []

        for tif_file in sorted(tif_files):
            try:
                stats = self._compute_basin_stats(tif_file, dataset, catchment_path)
                if stats:
                    results.append(stats)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {tif_file.name}: {e}", exc_info=True)

        if not results:
            self.logger.warning(f"No JRC surface water data could be extracted from {len(tif_files)} files")
            self.logger.debug(f"TIF files attempted: {[f.name for f in tif_files]}")
            return input_path

        # Combine results
        df = pd.DataFrame(results)

        # Save output
        output_dir = self._get_observation_dir('surface_water')
        output_file = output_dir / f"{self.domain_name}_jrc_{dataset}_processed.csv"
        df.to_csv(output_file, index=False)

        # Also save to product-specific location
        product_dir = self.project_observations_dir / "surface_water" / "jrc" / "processed"
        product_dir.mkdir(parents=True, exist_ok=True)
        product_file = product_dir / f"{self.domain_name}_jrc_{dataset}_processed.csv"
        df.to_csv(product_file, index=False)

        self.logger.info(f"JRC processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df)}")

        return output_file

    def _get_catchment_path(self) -> Optional[Path]:
        """Get path to catchment shapefile if available.

        Delegates to the shared resolver, which handles the nested
        discretization layout and river_basins fallback.
        """
        return self._resolve_catchment_shapefile()

    def _compute_basin_stats(
        self,
        tif_path: Path,
        dataset: str,
        catchment_path: Optional[Path]
    ) -> Optional[dict]:
        """
        Compute basin-level statistics from JRC raster.

        Args:
            tif_path: Path to GeoTIFF file
            dataset: Dataset name (occurrence, seasonality, etc.)
            catchment_path: Optional path to catchment shapefile for masking

        Returns:
            Dictionary of statistics or None if failed
        """
        try:
            import rasterio
            from rasterio.mask import mask as rio_mask

            with rasterio.open(tif_path) as src:
                if catchment_path:
                    # Mask to catchment
                    import geopandas as gpd
                    gdf = gpd.read_file(catchment_path)
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)

                    data, _ = rio_mask(src, gdf.geometry, crop=True)
                    data = data[0]  # First band
                else:
                    data = src.read(1)

                # Get nodata value
                nodata = src.nodata if src.nodata is not None else 255

                # Create mask for valid data
                valid_mask = (data != nodata) & (data >= 0)

                if not np.any(valid_mask):
                    self.logger.warning(f"No valid data in {tif_path.name}")
                    return None

                valid_data = data[valid_mask].astype(float)

                # Compute statistics
                stats = {
                    'source_file': tif_path.name,
                    'dataset': dataset,
                    f'{dataset}_mean': float(np.mean(valid_data)),
                    f'{dataset}_std': float(np.std(valid_data)),
                    f'{dataset}_min': float(np.min(valid_data)),
                    f'{dataset}_max': float(np.max(valid_data)),
                    f'{dataset}_median': float(np.median(valid_data)),
                    'valid_pixels': int(np.sum(valid_mask)),
                    'total_pixels': int(data.size),
                    'coverage_pct': float(np.sum(valid_mask) / data.size * 100),
                }

                # For occurrence dataset, compute water area fraction
                if dataset == 'occurrence':
                    # Occurrence > 0 means water was present at least once
                    water_pixels = np.sum(valid_data > 0)
                    stats['water_presence_pct'] = float(water_pixels / len(valid_data) * 100)

                    # Permanent water (occurrence > 80%)
                    permanent_pixels = np.sum(valid_data > 80)
                    stats['permanent_water_pct'] = float(permanent_pixels / len(valid_data) * 100)

                    # Seasonal water (20-80% occurrence)
                    seasonal_pixels = np.sum((valid_data > 20) & (valid_data <= 80))
                    stats['seasonal_water_pct'] = float(seasonal_pixels / len(valid_data) * 100)

                return stats

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            import traceback
            self.logger.warning(f"Error computing stats for {tif_path.name}: {e}", exc_info=True)
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
