# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tree Canopy Height Observation Handler

Processes tree canopy height data from multiple sources (GEDI, Meta/WRI, GLAD)
for use in hydrological modeling and vegetation analysis.

Canopy height is important for:
- Interception modeling (storage capacity)
- Roughness length estimation
- Evapotranspiration partitioning
- Snow interception modeling
- Forest type classification
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

from symfluence.core.registries import R

from ..base import BaseObservationHandler

# Scale factors and valid ranges
CANOPY_HEIGHT_VALID_RANGE = (0, 100)  # meters
GEDI_RH98_SCALE = 1.0  # Already in meters
META_HEIGHT_SCALE = 1.0  # Already in meters
GLAD_COVER_SCALE = 1.0  # Percentage for tree cover


@R.observation_handlers.add('canopy_height')
@R.observation_handlers.add('tree_height')
@R.observation_handlers.add('vegetation_height')
class CanopyHeightHandler(BaseObservationHandler):
    """
    Unified handler for tree canopy height data from multiple sources.

    Acquires and processes canopy height data from:
    - GEDI L2A: NASA LiDAR-based canopy height
    - Meta/WRI: AI-derived global 10m canopy height
    - GLAD/UMD: Tree cover and height from Landsat

    Unlike time-varying observations (ET, streamflow), canopy height is
    typically treated as a static or slowly-varying attribute. This handler
    extracts basin statistics (mean, max, std) for use in parameterization.

    Configuration:
        CANOPY_HEIGHT_SOURCE: Data source ('gedi', 'meta', 'glad', 'all')
        CANOPY_HEIGHT_METRIC: Height metric for GEDI ('rh98', 'rh95', 'rh75')
        CANOPY_HEIGHT_COMPUTE_STATS: Compute spatial statistics (default: True)

    Output:
        - Basin-averaged canopy height (mean, max, std)
        - Optional: Gridded canopy height raster clipped to domain
    """

    obs_type = "canopy_height"
    source_name = "MULTI_SOURCE"

    def acquire(self) -> Path:
        """
        Acquire canopy height data from configured source(s).

        Returns:
            Path to directory containing acquired data
        """
        canopy_dir = self.project_observations_dir / "vegetation" / "canopy_height"
        canopy_dir.mkdir(parents=True, exist_ok=True)

        # Check instance attribute first, then derive from source_name
        source = getattr(self, 'canopy_height_source', None)
        if source is None:
            source = {
                'NASA_GEDI': 'gedi', 'META_WRI': 'meta', 'UMD_GLAD': 'glad',
            }.get(self.source_name, 'meta') if self.source_name != 'MULTI_SOURCE' else 'all'
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        acquired_sources = []

        # Acquire from specified source(s)
        if source in ['gedi', 'all']:
            gedi_path = self._acquire_gedi(canopy_dir, force_download)
            if gedi_path:
                acquired_sources.append(('gedi', gedi_path))

        if source in ['meta', 'all']:
            meta_path = self._acquire_meta(canopy_dir, force_download)
            if meta_path:
                acquired_sources.append(('meta', meta_path))

        if source in ['glad', 'all']:
            glad_path = self._acquire_glad(canopy_dir, force_download)
            if glad_path:
                acquired_sources.append(('glad', glad_path))

        if not acquired_sources:
            self.logger.warning("No canopy height data could be acquired")

        return canopy_dir

    def _acquire_gedi(self, output_dir: Path, force: bool) -> Optional[Path]:
        """Acquire GEDI canopy height data."""
        gedi_dir = output_dir / 'gedi'
        gedi_dir.mkdir(parents=True, exist_ok=True)
        expected_file = gedi_dir / f"{self.domain_name}_gedi_canopy_height.tif"

        # Also check attributes location (where acquisition handlers write)
        attr_file = (
            self.project_attributes_dir / "vegetation" / "canopy_height" / "gedi"
            / f"{self.domain_name}_gedi_canopy_height.tif"
        )

        if expected_file.exists() and not force:
            self.logger.info(f"Using existing GEDI canopy height: {expected_file}")
            return expected_file

        if attr_file.exists() and not force:
            self.logger.info(f"Using existing GEDI canopy height from attributes: {attr_file}")
            import shutil
            shutil.copy(attr_file, expected_file)
            return expected_file

        try:
            from ...acquisition.handlers.canopy_height import GEDICanopyHeightAcquirer
            acquirer = GEDICanopyHeightAcquirer(self.config, self.logger)
            acq_result = acquirer.download(gedi_dir)

            # Copy from attributes location if acquisition wrote there
            if acq_result and acq_result.exists() and acq_result != expected_file:
                import shutil
                expected_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(acq_result, expected_file)
                return expected_file
            return acq_result
        except ImportError as e:
            self.logger.warning(f"GEDI acquirer not available: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GEDI acquisition failed: {e}", exc_info=True)
            return None

    def _acquire_meta(self, output_dir: Path, force: bool) -> Optional[Path]:
        """Acquire Meta/WRI canopy height data."""
        meta_dir = output_dir / 'meta_wri'
        meta_dir.mkdir(parents=True, exist_ok=True)
        expected_file = meta_dir / f"{self.domain_name}_meta_canopy_height.tif"

        # Also check attributes location (where acquisition handlers write)
        attr_file = (
            self.project_attributes_dir / "vegetation" / "canopy_height" / "meta_wri"
            / f"{self.domain_name}_meta_canopy_height.tif"
        )

        if expected_file.exists() and not force:
            self.logger.info(f"Using existing Meta/WRI canopy height: {expected_file}")
            return expected_file

        if attr_file.exists() and not force:
            self.logger.info(f"Using existing Meta/WRI canopy height from attributes: {attr_file}")
            import shutil
            shutil.copy(attr_file, expected_file)
            return expected_file

        try:
            from ...acquisition.handlers.canopy_height import MetaCanopyHeightAcquirer
            acquirer = MetaCanopyHeightAcquirer(self.config, self.logger)
            acq_result = acquirer.download(meta_dir)

            # Copy from attributes location if acquisition wrote there
            if acq_result and acq_result.exists() and acq_result != expected_file:
                import shutil
                expected_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(acq_result, expected_file)
                return expected_file
            return acq_result
        except ImportError as e:
            self.logger.warning(f"Meta/WRI acquirer not available: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Meta/WRI acquisition failed: {e}", exc_info=True)
            return None

    def _acquire_glad(self, output_dir: Path, force: bool) -> Optional[Path]:
        """Acquire GLAD/UMD tree height data."""
        glad_dir = output_dir / 'glad'
        glad_dir.mkdir(parents=True, exist_ok=True)
        expected_file = glad_dir / f"{self.domain_name}_glad_tree_height.tif"

        # Also check attributes location (where acquisition handlers write)
        attr_file = (
            self.project_attributes_dir / "vegetation" / "canopy_height" / "glad"
            / f"{self.domain_name}_glad_tree_height.tif"
        )

        if expected_file.exists() and not force:
            self.logger.info(f"Using existing GLAD tree height: {expected_file}")
            return expected_file

        if attr_file.exists() and not force:
            self.logger.info(f"Using existing GLAD tree height from attributes: {attr_file}")
            # Copy to observations location
            import shutil
            shutil.copy(attr_file, expected_file)
            return expected_file

        try:
            from ...acquisition.handlers.canopy_height import GLADTreeHeightAcquirer
            acquirer = GLADTreeHeightAcquirer(self.config, self.logger)
            acq_result = acquirer.download(glad_dir)

            # Copy from attributes location if acquisition wrote there
            if acq_result and acq_result.exists() and acq_result != expected_file:
                import shutil
                expected_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(acq_result, expected_file)
                return expected_file
            return acq_result
        except ImportError as e:
            self.logger.warning(f"GLAD acquirer not available: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GLAD acquisition failed: {e}", exc_info=True)
            return None

    def process(self, input_path: Path) -> Path:
        """
        Process canopy height data for the current domain.

        Args:
            input_path: Path to canopy height data directory

        Returns:
            Path to processed output file (CSV with statistics)
        """
        self.logger.info(f"Processing canopy height data for domain: {self.domain_name}")

        # Load catchment shapefile for masking
        basin_gdf = self._load_catchment_shapefile()

        # Find all canopy height files
        results = {}

        # Process each source
        for source_name in ['gedi', 'meta_wri', 'glad']:
            source_dir = input_path / source_name
            if not source_dir.exists():
                continue

            tif_files = list(source_dir.glob("*.tif"))
            for tif_file in tif_files:
                try:
                    stats = self._extract_basin_statistics(tif_file, basin_gdf)
                    if stats:
                        results[source_name] = stats
                        self.logger.info(
                            f"{source_name} canopy height - "
                            f"mean: {stats['mean']:.1f}m, max: {stats['max']:.1f}m"
                        )
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Failed to process {tif_file.name}: {e}", exc_info=True)

        if not results:
            self.logger.warning("No canopy height data could be processed")
            return input_path

        # Create summary DataFrame
        summary_data = []
        for source, stats in results.items():
            for stat_name, value in stats.items():
                summary_data.append({
                    'source': source,
                    'statistic': stat_name,
                    'value': value,
                    'unit': 'm' if stat_name != 'coverage_fraction' else 'fraction'
                })

        df = pd.DataFrame(summary_data)

        # Save summary statistics
        output_dir = self.project_observations_dir / "vegetation" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_canopy_height_stats.csv"

        df.to_csv(output_file, index=False)
        self.logger.info(f"Canopy height processing complete: {output_file}")

        # Also save a simple summary for easy access
        self._save_simple_summary(results, output_dir)

        return output_file

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """Load the catchment shapefile for spatial masking.

        Delegates to the shared resolver, which handles the nested
        discretization layout and river_basins fallback. Returns None when
        no basin is found (callers fall back to the bounding box).
        """
        return self._load_catchment_gdf()

    def _extract_basin_statistics(
        self,
        raster_path: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ) -> Optional[Dict[str, float]]:
        """
        Extract basin-averaged statistics from canopy height raster.

        Args:
            raster_path: Path to canopy height GeoTIFF
            basin_gdf: Catchment boundary for masking

        Returns:
            Dictionary with statistics (mean, max, std, etc.)
        """
        with rasterio.open(raster_path) as src:
            if basin_gdf is not None:
                # Reproject shapefile if needed
                if basin_gdf.crs != src.crs:
                    basin_gdf = basin_gdf.to_crs(src.crs)

                # Mask to catchment boundary
                try:
                    out_image, _ = rio_mask(
                        src,
                        basin_gdf.geometry,
                        crop=True,
                        nodata=np.nan
                    )
                    data = out_image[0]  # First band
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Masking failed, using full extent: {e}", exc_info=True)
                    data = src.read(1)
            else:
                # Use bounding box
                data = src.read(1)

                # Mask nodata values
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)

        # Apply valid range filter
        data = np.where(
            (data >= CANOPY_HEIGHT_VALID_RANGE[0]) &
            (data <= CANOPY_HEIGHT_VALID_RANGE[1]),
            data,
            np.nan
        )

        # Compute statistics
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            return None

        stats = {
            'mean': float(np.mean(valid_data)),
            'median': float(np.median(valid_data)),
            'max': float(np.max(valid_data)),
            'min': float(np.min(valid_data)),
            'std': float(np.std(valid_data)),
            'p25': float(np.percentile(valid_data, 25)),
            'p75': float(np.percentile(valid_data, 75)),
            'p90': float(np.percentile(valid_data, 90)),
            'coverage_fraction': float(len(valid_data) / data.size),
        }

        return stats

    def _save_simple_summary(
        self,
        results: Dict[str, Dict[str, float]],
        output_dir: Path
    ):
        """Save a simple summary CSV with one row per source."""
        summary_rows = []

        for source, stats in results.items():
            row = {'source': source}
            row.update(stats)
            summary_rows.append(row)

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            output_file = output_dir / f"{self.domain_name}_canopy_height_summary.csv"
            df.to_csv(output_file, index=False)

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed canopy height statistics."""
        processed_path = (
            self.project_observations_dir / "vegetation" / "preprocessed"
            / f"{self.domain_name}_canopy_height_stats.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path)
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading canopy height data: {e}", exc_info=True)
            return None

    def get_mean_canopy_height(self, source: str = 'meta_wri') -> Optional[float]:
        """
        Get mean canopy height for a specific source.

        Args:
            source: Data source ('gedi', 'meta_wri', 'glad')

        Returns:
            Mean canopy height in meters, or None if not available
        """
        df = self.get_processed_data()
        if df is None:
            return None

        mask = (df['source'] == source) & (df['statistic'] == 'mean')
        if mask.any():
            return float(df.loc[mask, 'value'].iloc[0])

        return None


@R.observation_handlers.add('gedi_canopy_height')
class GEDICanopyHeightHandler(CanopyHeightHandler):
    """
    GEDI-specific canopy height handler.

    Convenience handler that defaults to GEDI data source.
    """

    source_name = "NASA_GEDI"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.canopy_height_source = 'gedi'


@R.observation_handlers.add('meta_canopy_height')
@R.observation_handlers.add('wri_canopy_height')
class MetaCanopyHeightHandler(CanopyHeightHandler):
    """
    Meta/WRI-specific canopy height handler.

    Convenience handler that defaults to Meta/WRI data source.
    """

    source_name = "META_WRI"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.canopy_height_source = 'meta'


@R.observation_handlers.add('glad_tree_height')
@R.observation_handlers.add('umd_tree_height')
class GLADTreeHeightHandler(CanopyHeightHandler):
    """
    GLAD/UMD-specific tree height handler.

    Convenience handler that defaults to GLAD data source.
    """

    source_name = "UMD_GLAD"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.canopy_height_source = 'glad'
