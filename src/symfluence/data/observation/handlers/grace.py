# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GRACE total water storage observation handler.

Provides acquisition and preprocessing of GRACE/GRACE-FO satellite data
for total water storage anomaly validation with adaptive basin extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('grace')
class GRACEHandler(BaseObservationHandler):
    """
    Handles GRACE Total Water Storage anomaly data.
    Implements adaptive extraction based on basin size.
    """

    obs_type = "tws"
    source_name = "NASA_GRACE"
    SOURCE_INFO = {
        'source': 'GRACE/GRACE-FO',
        'source_doi': '10.5067/TEMSC-3JC62',
        'url': 'https://podaac.jpl.nasa.gov/GRACE',
    }

    # Basin size thresholds for extraction strategy
    STRATEGY_CONFIG = {
        'large_basin_threshold': 5000,      # > 5000 km²: bounding box
        'medium_basin_threshold': 1000,     # 1000-5000 km²: buffered bounding box
        'buffer_medium': 0.5,               # Buffer for medium basins (degrees)
    }

    def acquire(self) -> Path:
        """Locate GRACE data or download if possible."""
        grace_dir_cfg = self._get_config_value(lambda: None, default='default', dict_key='GRACE_DATA_DIR')
        if grace_dir_cfg == 'default' or not grace_dir_cfg:
            grace_dir = self.project_observations_dir / "grace"
        else:
            grace_dir = Path(grace_dir_cfg)

        # Check if we need to download
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        # Count only usable data files. An interrupted download leaves a
        # "<target>.nc.part" staging file behind (the acquirer's cleanup is
        # skipped on SIGKILL), and a bare .part satisfied the old
        # any(iterdir()) test — so the directory looked populated, the
        # re-download was skipped forever, and the downstream *.nc glob found
        # nothing. That combination silently starved a multi-objective
        # calibration of its GRACE observations for an entire run.
        has_files = grace_dir.exists() and any(
            p.is_file() and p.suffix == ".nc" for p in grace_dir.iterdir()
        )

        if not has_files or force_download:
            self.logger.info("Acquiring GRACE data...")
            # Use the Acquisition handler
            try:
                from symfluence.data.acquisition.handlers.grace import GRACEAcquirer
                acquirer = GRACEAcquirer(self.config, self.logger)
                acquirer.download(grace_dir)
            except ImportError as e:
                self.logger.error(f"Could not import GRACEAcquirer: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"GRACE acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing GRACE data in {grace_dir}")

        return grace_dir

    def process(self, input_path: Path) -> Path:
        """Process GRACE data for the current domain."""
        self.logger.info(f"Processing GRACE TWS for domain: {self.domain_name}")

        # GRACE TWS averaging needs a basin polygon. The shared resolver handles
        # the nested discretization layout and river_basins fallback.
        basin_shp = self._resolve_catchment_shapefile(required=True)
        basin_gdf = gpd.read_file(basin_shp)
        basin_area_km2 = self._calculate_area(basin_gdf)
        self.logger.info(f"Basin area: {basin_area_km2:.1f} km²")

        # Find GRACE files
        grace_files = self._find_grace_files(input_path)
        if not grace_files:
            self.logger.error("No GRACE NetCDF files found")
            return input_path

        results = {}
        for name, file_path in grace_files.items():
            with xr.open_dataset(file_path) as ds:
                ts = self._extract_for_basin(ds, basin_gdf, name, basin_area_km2)
                if ts is not None:
                    # Calculate anomalies (2003-2008 baseline as default)
                    ts_anomaly = self._calculate_anomalies(ts)
                    results[f'grace_{name}'] = ts
                    results[f'grace_{name}_anomaly'] = ts_anomaly

        if not results:
            self.logger.warning("No GRACE data could be extracted")
            return input_path

        # Save to CSV
        df = pd.DataFrame(results)
        output_dir = self.project_observations_dir / "grace" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_grace_tws_processed.csv"
        df.to_csv(output_file)

        self.logger.info(f"GRACE processing complete: {output_file}")
        return output_file

    def _calculate_area(self, gdf: gpd.GeoDataFrame) -> float:
        # Use equal area projection for calculation
        return gdf.to_crs('EPSG:6933').geometry.area.sum() / 1e6

    def _find_grace_files(self, grace_dir: Path) -> Dict[str, Path]:
        files = {}
        # Exact filenames matching GRACEAcquirer
        filenames = {
            'jpl': 'GRCTellus.JPL.200204_202511.GLO.RL06.3M.MSCNv04CRI.nc',
            'csr': 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc',
            'gsfc': 'gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc'
        }

        subset_patterns = {
            'jpl': '*JPL*subset*.nc',
            'csr': '*CSR*subset*.nc',
            'gsfc': '*gsfc*subset*.nc'
        }

        for name, pattern in subset_patterns.items():
            found = list(grace_dir.rglob(pattern))
            if found:
                files[name] = found[0]

        patterns = {'jpl': '*JPL*.nc', 'csr': '*CSR*.nc', 'gsfc': '*gsfc*.nc'}

        for name, filename in filenames.items():
            if name in files:
                continue
            file_path = grace_dir / filename
            if file_path.exists():
                files[name] = file_path
                continue
            found = list(grace_dir.rglob(patterns[name]))
            if found:
                files[name] = found[0]

        return files

    def _extract_for_basin(self, ds: xr.Dataset, gdf: gpd.GeoDataFrame, name: str, area: float) -> Optional[pd.Series]:
        # Project to UTM for accurate centroid, then get coordinates in geographic CRS
        dissolved = gdf.dissolve()
        utm_crs = dissolved.estimate_utm_crs()
        centroid = dissolved.to_crs(utm_crs).centroid.to_crs(gdf.crs).iloc[0]

        # Adaptive strategy
        if area <= self.STRATEGY_CONFIG['medium_basin_threshold']:
            # Point sampling
            lons, lats = ds.lon.values, ds.lat.values
            c_lon = centroid.x + 360 if centroid.x < 0 and lons.max() > 180 else centroid.x
            idx_lon = np.argmin(np.abs(lons - c_lon))
            idx_lat = np.argmin(np.abs(lats - centroid.y))
            data = ds.lwe_thickness.isel(lon=idx_lon, lat=idx_lat)
        else:
            # Spatial averaging
            bounds = gdf.total_bounds
            if bounds[0] < 0 and ds.lon.values.max() > 180:
                bounds[0] += 360
                bounds[2] += 360

            lon_mask = (ds.lon >= bounds[0]) & (ds.lon <= bounds[2])
            lat_mask = (ds.lat >= bounds[1]) & (ds.lat <= bounds[3])
            subset = ds.lwe_thickness.where(lon_mask & lat_mask, drop=True)
            data = subset.mean(dim=[d for d in subset.dims if d != 'time'])

        time_idx = self._get_time_index(ds, name)
        return pd.Series(data.values, index=time_idx).resample('MS').mean()

    def _get_time_index(self, ds: xr.Dataset, name: str) -> pd.DatetimeIndex:
        """Robustly get time index, handling decoding issues."""
        # Check if already decoded (datetime64)
        if np.issubdtype(ds.time.dtype, np.datetime64):
            return pd.to_datetime(ds.time.values)

        # Look for units attribute (case-insensitive)
        units_attr = None
        for key in ds.time.attrs:
            if key.lower() == 'units':
                units_attr = ds.time.attrs[key]
                break

        if units_attr and 'days since' in units_attr:
            origin_str = units_attr.split('since')[1].strip()
            # Clean origin string to remove time and timezone for robustness
            # e.g. "2002-01-01T00:00:00Z" -> "2002-01-01"
            if 'T' in origin_str:
                origin_str = origin_str.split('T')[0]

            return pd.to_datetime(ds.time.values, unit='D', origin=origin_str)

        # Fallback: assume standard decoding or let pandas handle it
        return pd.to_datetime(ds.time.values)

    def _calculate_anomalies(self, ts: pd.Series) -> pd.Series:
        baseline = ts.loc['2003-01-01':'2008-12-31']  # type: ignore[misc]
        mean = baseline.mean() if not baseline.empty else ts.mean()
        return ts - mean
