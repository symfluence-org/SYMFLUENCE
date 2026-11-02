# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Model Preprocessor

Handles preparation of SWAT model inputs including:
- TxtInOut directory structure
- Forcing files (.pcp and .tmp) from ERA5 NetCDF data
- Basin file (.bsn) with default snow/surface parameters
- file.cio master control file

The heavy lifting is delegated to sub-module generators:
- SWATForcingGenerator: .pcp and .tmp forcing files
- SWATBasinGenerator: .bsn, .wgn, .pnd, .wus, .chm files
- SWATSubbasinGenerator: .sub, .hru, .gw, .mgt, .sol files
- SWATRoutingGenerator: file.cio, fig.fig, .rte, .swq, database stubs
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.core.exceptions import FileOperationError
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("SWAT")
class SWATPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a SWAT model run.

    SWAT requires a TxtInOut directory containing:
    - file.cio: Master control file
    - .pcp files: Precipitation data
    - .tmp files: Temperature data (min/max)
    - .bsn: Basin-level parameters
    - .sub: Sub-basin files
    - .hru: HRU files
    - .gw: Groundwater files
    - .mgt: Management files
    - .sol: Soil files
    """


    MODEL_NAME = "SWAT"
    def __init__(self, config, logger):
        """
        Initialize the SWAT preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Standard paths (from base class):
        #   self.setup_dir   = project_dir / settings / SWAT
        #   self.forcing_dir = project_dir / data / forcing / SWAT_input
        # Settings files (file.cio, .bsn, .sub, etc.) go to setup_dir.
        # Forcing files (.pcp, .tmp) go to forcing_dir.
        # The runner assembles both into a TxtInOut for execution.
        self.txtinout_dir = self.setup_dir

        # Lazy-init backing fields for sub-module generators
        self._forcing_generator = None
        self._basin_generator = None
        self._subbasin_generator = None
        self._routing_generator = None
        self._catch_props: Optional[Dict] = None   # cached catchment properties

    # ------------------------------------------------------------------
    # Lazy-init properties for sub-module generators
    # ------------------------------------------------------------------

    @property
    def forcing_generator(self):
        """Lazy-init SWATForcingGenerator."""
        if self._forcing_generator is None:
            from .forcing_generator import SWATForcingGenerator
            self._forcing_generator = SWATForcingGenerator(self)
        return self._forcing_generator

    @property
    def basin_generator(self):
        """Lazy-init SWATBasinGenerator."""
        if self._basin_generator is None:
            from .basin_generator import SWATBasinGenerator
            self._basin_generator = SWATBasinGenerator(self)
        return self._basin_generator

    @property
    def subbasin_generator(self):
        """Lazy-init SWATSubbasinGenerator."""
        if self._subbasin_generator is None:
            from .subbasin_generator import SWATSubbasinGenerator
            self._subbasin_generator = SWATSubbasinGenerator(self)
        return self._subbasin_generator

    @property
    def routing_generator(self):
        """Lazy-init SWATRoutingGenerator."""
        if self._routing_generator is None:
            from .routing_generator import SWATRoutingGenerator
            self._routing_generator = SWATRoutingGenerator(self)
        return self._routing_generator

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_preprocessing(self) -> bool:
        """
        Run the complete SWAT preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting SWAT preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate forcing files from ERA5
            self.forcing_generator.generate_forcing_files(start_date, end_date)

            # Generate basin file
            self.basin_generator.generate_basin_file()

            # Generate sub-basin, HRU, groundwater, management, and soil files
            self.subbasin_generator.generate_subbasin_files()

            # Generate watershed routing file (fig.fig) and reach files
            self.routing_generator.generate_fig_file()
            self.routing_generator.generate_route_files()

            # Generate minimal database stub files
            self.routing_generator.generate_database_stubs()

            # Generate file.cio (must be last -- references all other files)
            self.routing_generator.generate_file_cio(start_date, end_date)

            logger.info("SWAT preprocessing complete.")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"SWAT preprocessing failed: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Helper methods (stay on orchestrator, used by sub-modules via self.pp)
    # ------------------------------------------------------------------

    def _create_directory_structure(self) -> None:
        """Create SWAT directory structure (settings + forcing)."""
        self.setup_dir.mkdir(parents=True, exist_ok=True)
        self.forcing_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created SWAT settings directory at {self.setup_dir}")
        logger.info(f"Created SWAT forcing directory at {self.forcing_dir}")

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from the model-ready datastore (shapefile +
        DEM + land/soil rasters). Cached: the DEM/raster reads run once.

        Returns:
            Dict with centroid lat/lon, area, elevation, mean slope, and the
            dominant land-use / soil-hydrologic-group / CN2 for the lumped HRU.
        """
        if self._catch_props is not None:
            return self._catch_props
        try:
            import geopandas as gpd
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)

                # Centroid in GEOGRAPHIC degrees. The shapefile is often in a
                # projected CRS (e.g. EPSG:3057 metres); a centroid taken there
                # yields lon/lat in metres -> an absurd UTM zone and wrong area.
                if gdf.crs.is_geographic:
                    cpt = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326).iloc[0]
                else:
                    cpt = gdf.geometry.centroid.to_crs(epsg=4326).iloc[0]
                lon, lat = cpt.x, cpt.y

                # Accurate area via the UTM zone derived from the degree centroid.
                utm_zone = int((lon + 180) / 6) + 1
                utm_crs = f"EPSG:{(32600 if lat >= 0 else 32700) + utm_zone}"
                area_m2 = gdf.to_crs(utm_crs).geometry.area.sum()

                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else None
                if elev is None:
                    elev = self._mean_dem_elev()

                cls = self._dominant_classes()
                self._catch_props = {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'area_km2': area_m2 / 1e6,
                    'elev': elev,
                    'slope': self._mean_dem_slope(),
                    'lulc': cls['lulc'],
                    'hydgrp': cls['hydgrp'],
                    'cn2': cls['cn2'],
                }
                return self._catch_props
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read catchment properties: {e}", exc_info=True)

        return {
            'lat': 65.0,
            'lon': -19.0,
            'area_m2': 1e8,
            'area_km2': 100.0,
            'elev': 500.0,
            'slope': 0.05,
            'lulc': 'RNGE',
            'hydgrp': 'C',
            'cn2': 78.0,
        }

    def _mean_dem_elev(self) -> float:
        """Mean catchment elevation from the model-ready DEM (fallback 500 m)."""
        try:
            import rasterio
            dd = self.project_dir / 'attributes' / 'elevation' / 'dem'
            dems = sorted(dd.glob('*_elv.tif')) if dd.exists() else []
            if dems:
                with rasterio.open(dems[0]) as src:
                    a = src.read(1).astype('float64')
                    nd = src.nodata
                m = np.isfinite(a) & (a > -100.0)
                if nd is not None:
                    m &= a != nd
                if m.any():
                    return float(a[m].mean())
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read DEM elevation: {e}", exc_info=True)
        return 500.0

    def _mean_dem_slope(self) -> float:
        """Mean catchment slope (m/m) from the DEM (fallback 0.05)."""
        try:
            import rasterio
            dd = self.project_dir / 'attributes' / 'elevation' / 'dem'
            dems = sorted(dd.glob('*_elv.tif')) if dd.exists() else []
            if dems:
                with rasterio.open(dems[0]) as src:
                    a = src.read(1).astype('float64')
                    nd = src.nodata
                    xr_, yr_ = src.res
                    geo = src.crs is not None and src.crs.is_geographic
                    bnds = src.bounds
                m = np.isfinite(a) & (a > -100.0)
                if nd is not None:
                    m &= a != nd
                if m.sum() > 4:
                    if geo:
                        lat0 = np.radians((bnds.bottom + bnds.top) / 2.0)
                        xm = abs(xr_) * 111320.0 * max(np.cos(lat0), 0.1)
                        ym = abs(yr_) * 111320.0
                    else:
                        xm, ym = abs(xr_), abs(yr_)
                    z = np.where(m, a, np.nan)
                    dy, dx = np.gradient(z, ym, xm)
                    slope = np.hypot(dx, dy)
                    return float(np.clip(np.nanmean(slope[m]), 0.001, 1.0))
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read DEM slope: {e}", exc_info=True)
        return 0.05

    def _dominant_classes(self) -> dict:
        """Dominant land-use (SWAT 4-char LULC) and soil hydrologic group from
        the model-ready land/soil rasters, for the lumped HRU."""
        out = {'lulc': 'RNGE', 'hydgrp': 'C', 'cn2': 78.0}
        try:
            import rasterio
            lc = self.project_dir / 'attributes' / 'landclass'
            tifs = sorted(lc.glob('domain_*_land_classes.tif')) if lc.exists() else []
            if tifs:
                with rasterio.open(tifs[0]) as s:
                    a = s.read(1); nd = s.nodata
                v = a[a != nd] if nd is not None else a.ravel()
                v = v[(v > 0) & (v != 17)]
                if v.size:
                    dom = int(np.bincount(v).argmax())
                    # MODIS IGBP -> SWAT LULC + a representative CN2 (HSG C).
                    if dom in (1, 2, 3, 4, 5):
                        out.update(lulc='FRST', cn2=70.0)
                    elif dom in (6, 7):
                        out.update(lulc='RNGB', cn2=74.0)   # shrubland
                    elif dom in (8, 9, 10):
                        out.update(lulc='RNGE', cn2=79.0)   # grassland/savanna
                    elif dom in (12, 14):
                        out.update(lulc='AGRL', cn2=82.0)   # cropland
                    elif dom in (11,):
                        out.update(lulc='WETL', cn2=80.0)
                    else:                                    # 15 ice, 16 barren
                        out.update(lulc='BARR', cn2=88.0)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read land class: {e}", exc_info=True)
        return out

    def _load_forcing_data(self):
        """Load basin-averaged forcing from the canonical model-ready store.

        Returns the dataset under the canonical vocabulary (pptrate/airtemp/...)
        with ds.attrs['timestep_seconds'] set, so the forcing generator never
        re-parses raw variable names.
        """
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            merged_path = self.project_forcing_dir / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path}")

        # The basin-averaged store is domain-shared and can hold several remaps
        # (e.g. a 1-HRU lumped basin average beside a 12-band elevation split).
        # open_canonical_forcing refuses to merge mixed spatial shapes and raises
        # ("hru {1, 12}"), which the forcing generator would otherwise swallow and
        # replace with FABRICATED synthetic weather -- silently calibrating on
        # random precipitation. Pick the single spatially-compatible remap first
        # (mirrors mHM/RHESSys #376).
        forcing_files = self._select_forcing_files(forcing_files)

        logger.info(f"Loading forcing from {len(forcing_files)} files")
        from symfluence.core.modeling.model_ready.forcing_reader import open_canonical_forcing
        ds = open_canonical_forcing(forcing_files)
        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _select_forcing_files(self, forcing_files: List[Path]) -> List[Path]:
        """Deterministically choose the spatially-compatible forcing file(s).

        SWAT basin-averages its forcing to a single gage, so the single-HRU
        (1 spatial point) remap is always the correct representation. When the
        shared store also holds a multi-HRU remap (a different discretization),
        prefer the canonical ``{domain}_{dataset}_remapped_*`` name, then drop
        multi-point candidates so ``open_canonical_forcing`` is handed one shape.

        Raises:
            FileOperationError: if incompatibly-shaped candidates remain.
        """
        files = sorted(forcing_files)
        if len(files) <= 1:
            return files

        domain = self.domain_name.lower()
        dataset = str(self.forcing_dataset or '').lower()

        candidates = [f for f in files if f.name.lower().startswith(f"{domain}_{dataset}_remapped")]
        if not candidates and dataset:
            candidates = [f for f in files if dataset in f.name.lower()]
        if not candidates:
            candidates = [f for f in files if f.name.lower().startswith(domain)]
        if not candidates:
            candidates = files
        if len(candidates) == 1:
            return candidates

        # SWAT basin-averages to one gage -> the 1-point remap is always correct.
        sizes = {f: self._forcing_spatial_size(f) for f in candidates}
        single = [f for f in candidates if sizes[f] == 1]
        if single:
            candidates = single

        distinct = {sizes[f] for f in candidates}
        if len(distinct) > 1:
            listing = "\n".join(
                f"  {f.name}: {sizes[f]} spatial point(s)" for f in candidates
            )
            raise FileOperationError(
                f"Ambiguous SWAT forcing in {self.forcing_basin_path}: "
                f"{len(candidates)} files match (domain={self.domain_name}, "
                f"forcing={self.forcing_dataset}) but have incompatible spatial "
                f"shapes; refusing to merge them.\n{listing}\n"
                "Remove the stale/incompatible forcing file(s)."
            )
        return candidates

    @staticmethod
    def _forcing_spatial_size(path: Path) -> int:
        """Return the number of spatial points (product of non-time dims)."""
        with xr.open_dataset(path) as ds:
            size = 1
            for dim, length in ds.sizes.items():
                if dim != 'time':
                    size *= int(length)
            return size

    def _extract_variable(self, ds, candidates, default_val=0.0):
        """Extract a variable from dataset by trying multiple candidate names."""
        for candidate in candidates:
            if candidate in ds:
                data = ds[candidate].values
                # Average over spatial dims if present
                while data.ndim > 1:
                    data = np.nanmean(data, axis=-1)
                return data, candidate
        return None, None

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
