# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM Model Preprocessor

Handles preparation of mHM model inputs including:
- Forcing data conversion from ERA5 NetCDF to mHM grid format
- Namelist generation (mhm.nml, mrm.nml, mhm_parameter.nml, mhm_outputs.nml)
- Directory structure setup
- Grid and morphological input preparation
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigurationError, FileOperationError, GeospatialError
from symfluence.core.geometry_utils import calculate_catchment_centroid
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("MHM")
class MHMPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a mHM model run.

    mHM requires:
    - Meteorological forcing grids (NetCDF): precipitation, temperature, PET
    - Morphological data (DEM, soil, land cover grids)
    - Namelist files:
        - mhm.nml (main config: directories, time periods, process selection)
        - mrm.nml (routing config)
        - mhm_parameter.nml (parameter values, bounds, flags, scaling)
        - mhm_outputs.nml (output control: which fluxes/states to write)
    - Gauge data for calibration/validation
    """

    MODEL_NAME = "MHM"

    # Grid resolution for the 2-row lumped layout.  Two cells at this
    # size give approximately the same total area as a single 0.5° cell.
    LUMPED_CELLSIZE: float = 0.35

    def __init__(self, config, logger):
        """
        Initialize the mHM preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Use standard base-class paths:
        #   self.setup_dir   -> {project_dir}/settings/MHM   (namelists, morph, lcover, gauge)
        #   self.forcing_dir -> {project_dir}/data/forcing/MHM_input  (forcing NetCDFs)
        self.settings_dir = self.setup_dir
        self.morph_dir = self.setup_dir / "morph"
        self.lcover_dir = self.setup_dir / "lcover"
        self.gauge_dir = self.setup_dir / "gauge"
        self.output_mhm_dir = self.setup_dir / "output"

        # Resolve spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.mhm.spatial_mode,
            default=None,
            dict_key='MHM_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped',
                dict_key='DOMAIN_DEFINITION_METHOD'
            )
            if domain_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        logger.info(f"mHM spatial mode: {self.spatial_mode}")

        # Distributed-morphology mode: build a real gridded domain (DEM, soil,
        # land cover, slope, flow network) at MHM_GRID_RES so Multiscale
        # Parameter Regionalization actually regionalises parameters over
        # spatially varying morphology -- mHM's distinctive capability, inert in
        # the lumped 2-cell layout. The flow network is derived with TauDEM
        # (the framework's geofabric engine), consistent with delineation.
        self.distributed_morph = bool(self._get_config_value(
            lambda: self.config.model.mhm.distributed_morph,
            default=False, dict_key='MHM_DISTRIBUTED'))
        self.grid_res = float(self._get_config_value(
            lambda: self.config.model.mhm.grid_res,
            default=0.02, dict_key='MHM_GRID_RES'))   # degrees (~2 km lat)
        if not self.distributed_morph and self.config.model.mhm.match_lumped_area:
            from scipy.optimize import brentq
            props = self._get_catchment_properties()
            # Same two-cell spherical area calculation as mo_common_grid.f90.
            metres_per_degree = 6371228.0 * np.pi / 180.0
            def area(cell):
                return (cell * metres_per_degree) ** 2 * (
                    np.cos(np.radians(props['lat'])) + np.cos(np.radians(props['lat'] - cell)))
            self.LUMPED_CELLSIZE = brentq(lambda cell: area(cell) - props['area_m2'], 0.001, 5.0)
            logger.info(f"mHM lumped cells matched to {props['area_m2'] / 1e6:.3f} km2")
        self._dist_grid = None  # cache of distributed-grid arrays/metadata
        if self.distributed_morph:
            logger.info(f"mHM distributed morphology ON (grid res {self.grid_res} deg, TauDEM flow)")

    def _taudem_dir(self) -> str:
        """Resolve a *working* TauDEM binary directory.

        The flow network is built in-process with pyflwdir, so this is only a
        documented fallback for callers that want TauDEM. Multiple installs can
        coexist (e.g. a per-domain copy with broken shared libraries alongside a
        good shared install), so candidate locations are validated by checking
        that ``d8flowdir`` exists and actually loads (a copy with missing dylibs
        exits 127). The configured TAUDEM_DIR wins if set; otherwise the first
        runnable candidate is returned, falling back to the framework default.
        """
        td = self._get_config_value(lambda: self.config.paths.taudem_dir,
                                    default='default', dict_key='TAUDEM_DIR')
        if td != 'default':
            return str(td)
        data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        # Candidate install roots: per-domain data dir, its SYMFLUENCE_data
        # parent, and the sibling code/<->data SYMFLUENCE_data layout.
        candidates = [
            data_dir / 'installs' / 'TauDEM' / 'bin',
            data_dir.parent / 'installs' / 'TauDEM' / 'bin',
        ]
        s = str(data_dir)
        if 'code/SYMFLUENCE_data' in s:
            alt = Path(s.replace('code/SYMFLUENCE_data', 'data/SYMFLUENCE_data'))
            candidates.append(alt.parent / 'installs' / 'TauDEM' / 'bin')
            candidates.append(Path(s.split('code/SYMFLUENCE_data')[0])
                              / 'data' / 'SYMFLUENCE_data' / 'installs' / 'TauDEM' / 'bin')
        default = candidates[0]
        for cand in candidates:
            exe = cand / 'd8flowdir'
            if not exe.exists():
                continue
            try:
                rc = run_subprocess([str(exe)], capture_output=True).returncode
            except OSError:
                continue
            if rc not in (126, 127):   # 127/126 => dylib/load failure
                return str(cand)
        return str(default)

    def _build_distributed_grids(self) -> dict:
        """Build the distributed mHM grid: resample the DEM to grid_res, derive
        the D8 flow network in-process with pyflwdir, mask to the basin upstream
        of the outlet, and resample soil/land classes. Returns a dict of 2D
        arrays + grid metadata (cached)."""
        if self._dist_grid is not None:
            return self._dist_grid
        import pyflwdir
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.warp import Resampling, reproject

        res = self.grid_res
        dem_path = self.project_attributes_dir / 'elevation' / 'dem'
        dem_files = sorted(dem_path.glob('*_elv.tif'))

        # 1. Resample DEM to the (geographic) grid at grid_res.
        with rasterio.open(dem_files[0]) as src:
            b = src.bounds
            ncol = max(3, int(np.ceil((b.right - b.left) / res)))
            nrow = max(3, int(np.ceil((b.top - b.bottom) / res)))
            from rasterio.transform import from_origin
            dst_t = from_origin(b.left, b.top, res, res)
            dem = np.full((nrow, ncol), -9999.0, 'float32')
            reproject(rasterio.band(src, 1), dem, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=dst_t, dst_crs=src.crs,
                      resampling=Resampling.average, dst_nodata=-9999.0)
            crs = src.crs
        # 2. Catchment mask. The river-basin polygon does not depend on the
        # experiment id (the HRU catchment path does), so prefer it. A missing
        # polygon is an error: falling back to the DEM extent picks whichever
        # basin drains the most of the bounding box, not this catchment.
        basin_shp = (self.project_dir / 'shapefiles' / 'river_basins'
                     / f'{self.domain_name}_riverBasins_lumped.shp')
        gdf = gpd.read_file(basin_shp if basin_shp.exists() else self._catchment_polygon_path())
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        polygon_km2 = float(gdf.to_crs(gdf.estimate_utm_crs()).area.sum() / 1e6)
        gdf = gdf.to_crs(crs)
        inside = ~geometry_mask(gdf.geometry, (nrow, ncol), dst_t, invert=False)

        # 3. Flow network via pyflwdir (in-process; depression-filled D8),
        # conditioned on the polygon: the DEM outside it is nodata and only the
        # lowest edge cell is an outlet, so every polygon cell drains to one
        # outlet. Deriving the basin from the raw coarse DEM instead loses ice
        # divide and flat headwater cells that drain "elsewhere" at grid_res
        # (14% of Tungnaa at 0.02 deg, mostly glacier). Masking the network
        # after the fact would cut it and make mRM routing loop forever.
        # pyflwdir's 'd8' encoding IS the ESRI/mHM convention
        # (1=E,2=SE,4=S,...,128=NE); latlon=True uses geographic distances.
        # (TauDEM produces the same network but fails in the workflow
        # subprocess due to macOS stripping its GDAL/MPI dyld paths.)
        dem_filled = np.where(dem > -100.0, dem, np.nan).astype('float32')
        flw = pyflwdir.from_dem(data=np.where(inside, dem_filled, np.nan), nodata=np.nan,
                                transform=dst_t, latlon=True, outlets='min')
        fdir = flw.to_array(ftype='d8').astype('int32')
        fdir = np.where(fdir > 0, fdir, 0)
        ad8 = flw.upstream_area(unit='cell').astype('float64')
        ad8 = np.where(ad8 < 0, 0, ad8)
        mask = inside & np.isfinite(dem_filled)
        ad8m = np.where(mask, ad8, -1)
        oi, oj = np.unravel_index(int(np.argmax(ad8m)), ad8m.shape)
        # Polygon fragments not 8-connected to the main basin drain to their
        # own pits; drop them, but refuse if that loses a material area.
        draining = flw.basins(idxs=np.array([oi * ncol + oj])) > 0
        dropped = int((mask & ~draining).sum())
        if dropped > 0.02 * mask.sum():
            raise GeospatialError(f"{dropped} of {int(mask.sum())} catchment cells do not drain "
                                  "to the mHM outlet; check MHM_GRID_RES and the catchment polygon")
        if dropped:
            logger.info(f"mHM dropped {dropped} catchment cells disconnected from the outlet")
        mask &= draining
        # Slope (m/m) from the unconditioned DEM gradient, geographic cell sizes.
        latc = (b.top + b.bottom) / 2.0
        ym = res * 111320.0
        xm = res * 111320.0 * max(np.cos(np.radians(latc)), 0.1)
        dy, dx = np.gradient(np.where(np.isnan(dem_filled), np.nanmean(dem_filled),
                                      dem_filled), ym, xm)
        sd8 = np.hypot(dx, dy)

        # The outlet pit drains off-domain; give it a defined edge direction
        # so mHM sees a clear basin exit rather than a self-pointing pit.
        if fdir[oi, oj] <= 0:
            fdir[oi, oj] = 4  # south / off-grid

        # 6. Resample soil / land-cover classes to the grid (nearest).
        def _resample_class(path):
            out = np.zeros((nrow, ncol), 'int32')
            try:
                with rasterio.open(path) as s:
                    reproject(rasterio.band(s, 1), out, src_transform=s.transform,
                              src_crs=s.crs, dst_transform=dst_t, dst_crs=crs,
                              resampling=Resampling.nearest)
            except Exception:  # noqa: BLE001
                pass
            return out
        soil = _resample_class(self.project_attributes_dir / 'soilclass'
                               / f'domain_{self.domain_name}_soil_classes.tif')
        land = _resample_class(self.project_attributes_dir / 'landclass'
                               / f'domain_{self.domain_name}_land_classes.tif')
        glacier_frac = np.where(mask, self._glacier_fraction_grid(nrow, ncol, dst_t, crs), 0.0)
        min_frac = float(self._get_config_value(
            lambda: self.config.model.mhm.glacier_min_fraction,
            default=0.5, dict_key='MHM_GLACIER_MIN_FRACTION'))
        glacier = glacier_frac >= max(min_frac, 1e-9)

        slope = np.clip(np.abs(sd8), 0.0001, 2.0)
        self._dist_grid = dict(
            nrow=nrow, ncol=ncol, res=res, transform=dst_t, crs=crs,
            dem=dem, fdir=fdir, facc=ad8, slope=slope, mask=mask,
            soil=soil, land=land, glacier=glacier, glacier_frac=glacier_frac,
            outlet=(int(oi), int(oj)))
        cell_km2 = (res * 111.32) ** 2 * np.cos(np.radians(
            dst_t.f + dst_t.e * (np.arange(nrow) + 0.5)))[:, None]
        basin_km2 = float((cell_km2 * mask).sum())
        logger.info(f"mHM distributed basin {basin_km2:.0f} km2 vs catchment polygon "
                    f"{polygon_km2:.0f} km2")
        if abs(basin_km2 / polygon_km2 - 1.0) > 0.15:
            logger.warning("mHM distributed basin area differs from the catchment "
                           "polygon by more than 15%; check MHM_GRID_RES and the outlet")
        logger.info(f"mHM distributed grid: {nrow}x{ncol} @ {res} deg, "
                    f"{int(mask.sum())} active cells, outlet=({oi},{oj}), "
                    f"max_acc={int(np.nanmax(ad8m))}, glacier cells={int(glacier.sum())}")
        if glacier.any():
            logger.info(f"mHM glacier: {float((cell_km2 * glacier).sum()):.1f} km2 in glacier cells, "
                        f"{float((cell_km2 * glacier_frac * glacier).sum()):.1f} km2 of outline within them, "
                        f"{float((cell_km2 * glacier_frac).sum()):.1f} km2 of outline in the basin")
        return self._dist_grid

    def _glacier_fraction_grid(self, nrow, ncol, transform, crs) -> np.ndarray:
        """Glacierised fraction of each cell, from the configured glacier
        outline.

        Fractions come from rasterising the outline on a 10x finer grid and
        block-averaging, so partially glacierised edge cells are resolved.
        Returns all zeros when no outline is configured.
        """
        outline = self._get_config_value(
            lambda: self.config.model.mhm.glacier_outline,
            default=None, dict_key='MHM_GLACIER_OUTLINE')
        if not outline:
            return np.zeros((nrow, ncol))
        from rasterio.features import rasterize
        from rasterio.transform import Affine
        gdf = gpd.read_file(outline)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(crs)
        k = 10
        fine_t = transform * Affine.scale(1.0 / k)
        fine = rasterize(((g, 1) for g in gdf.geometry), out_shape=(nrow * k, ncol * k),
                         transform=fine_t, fill=0, dtype='uint8')
        return fine.reshape(nrow, k, ncol, k).mean(axis=(1, 3))

    def _model_timestep(self) -> int:
        """MHM_TIMESTEP, refused for a non-hourly step unless the installed mHM
        carries the SYMFLUENCE timestep patch.

        Upstream mHM picks the forcing record with ``ceil(tt / (24 h per
        forcing step))``, i.e. it assumes ``tt`` counts hours. At a 6 h or
        24 h step each forcing day is reused for 6 or 24 model steps and the
        run silently consumes only the start of the forcing.
        """
        from .build_instructions import TIMESTEP_PATCH_MARKER

        timestep = int(self._get_config_value(
            lambda: self.config.model.mhm.timestep, default=1, dict_key='MHM_TIMESTEP'))
        if timestep == 1:
            return timestep
        install = self._get_config_value(
            lambda: self.config.model.mhm.install_path, default='default',
            dict_key='MHM_INSTALL_PATH')
        if not install or install == 'default':
            data_dir = Path(self._get_config_value(
                lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
            install = data_dir / 'installs' / 'mhm' / 'bin'
        install = Path(install)
        if not any((d / TIMESTEP_PATCH_MARKER).exists()
                   for d in (install, install / 'bin', install.parent)):
            raise ConfigurationError(
                f"MHM_TIMESTEP={timestep} needs the patched mHM build "
                f"(`symfluence binary install mhm --patched`); upstream mHM misreads "
                f"the forcing at any step other than 1 h. No {TIMESTEP_PATCH_MARKER} "
                f"found under {install}.")
        return timestep

    def run_preprocessing(self) -> bool:
        """
        Run the complete mHM preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting mHM preprocessing...")
            self._model_timestep()

            # Create directory structure
            self._create_directory_structure()

            # Generate forcing files
            self._generate_forcing_files()

            # Generate morphological inputs
            self._generate_morph_files()

            # Generate lookup tables, gauge data, and latlon file
            self._generate_lookup_tables()
            self._generate_gauge_data()
            self._generate_latlon_file()

            # Generate namelist files
            self._generate_mhm_namelist()
            self._generate_mrm_namelist()
            self._generate_mhm_parameter_namelist()
            self._generate_mhm_outputs_namelist()

            logger.info("mHM preprocessing complete.")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"mHM preprocessing failed: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create mHM input directory structure."""
        dirs = [
            self.settings_dir,
            self.forcing_dir,
            self.morph_dir,
            self.lcover_dir,
            self.gauge_dir,
            self.output_mhm_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created mHM input directories at {self.setup_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from configuration."""
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _catchment_polygon_path(self) -> Path:
        """Catchment polygon: the experiment's HRU file when present, else the
        experiment-independent river-basin polygon."""
        for path in (self.get_catchment_path(),
                     self.project_dir / 'shapefiles' / 'river_basins'
                     / f'{self.domain_name}_riverBasins_lumped.shp'):
            if path.exists():
                return path
        raise FileOperationError(
            f"No catchment polygon for {self.domain_name}: neither "
            f"{self.get_catchment_path()} nor the lumped river-basin shapefile exists")

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        A missing polygon raises: the old placeholder (a 100 km2 basin at
        51N, 115W) silently set the lumped cell size and coordinates.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        gdf = gpd.read_file(self._catchment_polygon_path())

        # Get centroid using proper CRS-aware calculation
        lon, lat = calculate_catchment_centroid(gdf, logger=logger)

        # Project to UTM for accurate area
        utm_zone = int((lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        area_m2 = gdf.to_crs(f"EPSG:{utm_epsg}").geometry.area.sum()

        # Get elevation if available
        elev = float(gdf['elev_mean'].iloc[0]) if 'elev_mean' in gdf.columns else 1000.0

        return {'lat': lat, 'lon': lon, 'area_m2': area_m2, 'elev': elev}

    def _generate_forcing_files(self) -> None:
        """
        Generate mHM forcing files from ERA5 or basin-averaged data.

        mHM requires meteorological input as NetCDF grids:
        - Precipitation (pre): mm/day
        - Temperature (tavg): deg C
        - PET (pet): mm/day (or temperature for internal PET calculation)
        """
        logger.info("Generating mHM forcing files...")

        start_date, end_date = self._get_simulation_dates()

        # Load the real forcing. Fabricating weather when it cannot be read is
        # never acceptable in a real run: a stray, incompatibly-shaped forcing
        # file once made this fall back silently, and mHM then calibrated 1500
        # iterations against a sine-wave temperature and random-exponential
        # precipitation, reporting a plausible-looking KGE of -0.17 with no
        # sign that the weather was fake. Let the read error propagate so the
        # bad input is fixed, not papered over. Synthetic forcing stays
        # available for tests behind an explicit opt-in.
        if os.environ.get("SYMFLUENCE_MHM_ALLOW_SYNTHETIC_FORCING") != "1":
            # Default (real) path: any read failure propagates with its full
            # traceback. No blanket except — a bad input must be fixed, and a
            # forcing that cannot be read must never be replaced by fabricated
            # weather.
            forcing_ds = self._load_forcing_data()
            self._write_mhm_forcing(forcing_ds, start_date, end_date)
            return

        # Opt-in smoke-test path only: fall back to synthetic forcing.
        try:
            forcing_ds = self._load_forcing_data()
            self._write_mhm_forcing(forcing_ds, start_date, end_date)
        except Exception as e:  # noqa: BLE001 — smoke-test fallback only
            logger.warning(
                "Could not load mHM forcing (%s); SYMFLUENCE_MHM_ALLOW_"
                "SYNTHETIC_FORCING=1 is set, generating SYNTHETIC forcing — "
                "results are meaningless for anything but a smoke test.",
                e, exc_info=True,
            )
            self._generate_synthetic_forcing(start_date, end_date)

    def _load_forcing_data(self) -> xr.Dataset:
        """Load basin-averaged forcing data."""
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            merged_path = self.project_forcing_dir / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path}")

        # The basin forcing store is shared across every model in a domain and
        # may hold more than one .nc. Choose deterministically by (active forcing
        # dataset, spatial mode) instead of merging whatever globs -- merging a
        # stray remap crashes on hru alignment ({1, 12}) and once triggered a
        # silent synthetic-forcing fallback.
        forcing_files = self._select_forcing_files(forcing_files)
        logger.info(f"Loading forcing from {len(forcing_files)} file(s)")

        # Read through the canonical model-ready forcing reader: variables come
        # back under the canonical vocabulary (pptrate/airtemp/...) with the
        # forcing timestep on ds.attrs['timestep_seconds'] -- no per-model alias
        # lists or timestep guessing.
        from symfluence.core.modeling.model_ready.forcing_reader import open_canonical_forcing
        ds = open_canonical_forcing(forcing_files)
        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _select_forcing_files(self, forcing_files: List[Path]) -> List[Path]:
        """Deterministically choose the forcing file(s) for this domain and mode.

        The basin-averaged forcing store is a domain-shared product that can
        accumulate several NetCDFs: a different forcing dataset (RDRS alongside
        ERA5), a remap produced under a different discretization (a 12-band
        elevation split alongside the 1-HRU lumped basin), or a stale artefact
        from an earlier run. ``open_canonical_forcing`` merges/aligns everything
        it is handed, so an incompatible file either corrupts the merge or
        crashes on dimension alignment (the ``{1, 12}`` hru mismatch that once
        made mHM fall back to fabricated weather).

        Selection reuses the framework's established remap naming convention
        ``{domain}_{forcing_dataset}_remapped_*`` (see
        ``resampling.file_processor.determine_output_filename`` and SUMMA's
        ``startswith(domain)`` file discovery) and the domain's spatial mode.
        Genuine multi-file forcing of one shape (e.g. per-period time chunks) is
        still merged; genuinely ambiguous, incompatibly-shaped candidates raise a
        clear error listing them rather than being merged blindly.

        Args:
            forcing_files: every ``*.nc`` discovered in the forcing store.

        Returns:
            The spatially-compatible subset to hand to ``open_canonical_forcing``.

        Raises:
            FileOperationError: if more than one incompatibly-shaped candidate
                remains after filtering.
        """
        files = sorted(forcing_files)
        if len(files) == 1:
            return files

        domain = self.domain_name.lower()
        dataset = self.forcing_dataset.lower()

        # 1. Prefer the canonical remap naming for the active (domain, dataset).
        #    Progressive fallbacks keep non-standard stores working: any file
        #    naming this dataset, then any naming this domain, then everything.
        candidates = [f for f in files if f.name.lower().startswith(f"{domain}_{dataset}_remapped")]
        if not candidates and dataset:
            candidates = [f for f in files if dataset in f.name.lower()]
        if not candidates:
            candidates = [f for f in files if f.name.lower().startswith(domain)]
        if not candidates:
            candidates = files
        if len(candidates) == 1:
            return candidates

        # 2. Disambiguate by spatial mode. A lumped domain must resolve to a
        #    single spatial point, so drop any candidate remapped onto a
        #    multi-HRU geofabric when a single-HRU one is present.
        sizes = {f: self._forcing_spatial_size(f) for f in candidates}
        if self.spatial_mode == 'lumped':
            lumped = [f for f in candidates if sizes[f] == 1]
            if lumped:
                candidates = lumped

        # 3. Whatever remains must be spatially compatible to merge (e.g. genuine
        #    per-period time chunks of one grid). Refuse to merge mixed shapes.
        distinct = {sizes[f] for f in candidates}
        if len(distinct) > 1:
            listing = "\n".join(
                f"  {f.name}: {sizes[f]} spatial point(s)" for f in candidates
            )
            raise FileOperationError(
                f"Ambiguous mHM forcing in {self.forcing_basin_path}: "
                f"{len(candidates)} files match (domain={self.domain_name}, "
                f"forcing={self.forcing_dataset}, spatial_mode={self.spatial_mode}) "
                f"but have incompatible spatial shapes; refusing to merge them.\n"
                f"{listing}\n"
                "Remove the stale/incompatible forcing file(s), or set the "
                "correct forcing dataset for this run."
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

    def _write_mhm_forcing_nc(
        self,
        filepath: Path,
        varname: str,
        data_1d: np.ndarray,
        times: np.ndarray,
        props: Dict,
        long_name: str,
        units: str,
        cellsize: float = 0.35,  # kept for API compat; overridden by LUMPED_CELLSIZE
    ) -> None:
        """
        Write a single mHM-format forcing NetCDF file using netCDF4.

        mHM v5.13 expects forcing NetCDF files with:
        - Dimensions: time (unlimited), yc (nrows), xc (ncols)
        - Coordinate variables: xc(xc), yc(yc), lon(yc,xc), lat(yc,xc)
        - Data variable with fill_value=-9999.0
        - Time in 'days since YYYY-01-01 00:00:00', calendar 'standard'

        For a lumped setup the grid is 2 rows × 1 col (upstream + outlet)
        with identical forcing in both cells.
        """
        nrows, ncols = 2, 1
        cellsize = self.LUMPED_CELLSIZE
        xllcorner = props['lon'] - cellsize / 2
        yllcorner = props['lat'] - cellsize / 2 - cellsize  # 2 rows

        xc_vals = np.array([xllcorner + cellsize * (i + 0.5) for i in range(ncols)])
        yc_vals = np.array([yllcorner + cellsize * (nrows - 1 - i + 0.5) for i in range(nrows)])

        # Build time coordinate as days since start-of-year
        time_index = pd.DatetimeIndex(times)
        ref_year = time_index[0].year
        time_units = f'days since {ref_year}-01-01 00:00:00'
        time_num = nc4.date2num(time_index.to_pydatetime(), units=time_units, calendar='standard')

        # Replace NaN with fill value
        fill = -9999.0
        data_clean = np.where(np.isnan(data_1d), fill, data_1d)

        ds = nc4.Dataset(str(filepath), 'w', format='NETCDF4')
        try:
            ds.createDimension('time', None)  # unlimited
            ds.createDimension('yc', nrows)
            ds.createDimension('xc', ncols)

            # Time variable — use int32 to match mHM reference format;
            # mHM's get_time_vector_and_select reads into integer(i8).
            t_var = ds.createVariable('time', 'i4', ('time',))
            t_var.units = time_units
            t_var.calendar = 'standard'
            t_var[:] = np.round(time_num).astype(np.int32)

            # xc / yc coordinate variables
            xc_var = ds.createVariable('xc', 'f8', ('xc',))
            xc_var.units = 'degrees_east'
            xc_var[:] = xc_vals

            yc_var = ds.createVariable('yc', 'f8', ('yc',))
            yc_var.units = 'degrees_north'
            yc_var[:] = yc_vals

            # 2-D lon / lat (yc, xc) — matches mHM reference test data.
            # mHM's read_header_ascii swaps ncols↔nrows internally, so
            # Fortran getShape returns (xc, yc, time) and the shape check
            # passes with var_shape(1)==level2%nrows==ncols, var_shape(2)==level2%ncols==nrows.
            lon_2d = np.full((nrows, ncols), props['lon'])
            lat_rows = [props['lat'] - i * cellsize for i in range(nrows)]
            lat_2d = np.array(lat_rows).reshape(nrows, ncols)

            lon_var = ds.createVariable('lon', 'f8', ('yc', 'xc'))
            lon_var.units = 'degrees_east'
            lon_var[:] = lon_2d

            lat_var = ds.createVariable('lat', 'f8', ('yc', 'xc'))
            lat_var.units = 'degrees_north'
            lat_var[:] = lat_2d

            # Data variable — replicate 1-D time series to both cells.
            # Dimensions (time, yc, xc): matches mHM reference convention.
            # Fortran reverses to (xc, yc, time); getData reads with
            # cnt=(nRows_internal, nCols_internal, time_cnt) in Fortran order.
            data_var = ds.createVariable(varname, 'f8', ('time', 'yc', 'xc'), fill_value=fill)
            data_var.long_name = long_name
            data_var.units = units
            data_2d = np.asarray(data_clean)[:, np.newaxis, np.newaxis]
            data_var[:] = np.broadcast_to(data_2d, (len(data_clean), nrows, ncols)).copy()
        finally:
            ds.close()

    def _write_forcing_header(self, props: Dict) -> None:
        """
        Write header.txt in the forcing directory.

        mHM reads this ESRI ASCII grid header to determine the spatial
        extent of the forcing grid.  Must match the 2-row lumped layout.
        """
        cellsize = self.LUMPED_CELLSIZE
        xllcorner = props['lon'] - cellsize / 2
        yllcorner = props['lat'] - cellsize / 2 - cellsize  # 2 rows
        content = (
            f"ncols         1\n"
            f"nrows         2\n"
            f"xllcorner     {xllcorner:.6f}\n"
            f"yllcorner     {yllcorner:.6f}\n"
            f"cellsize      {cellsize}\n"
            f"NODATA_value  -9999\n"
        )
        header_path = self.forcing_dir / 'header.txt'
        header_path.write_text(content, encoding='utf-8')
        logger.info(f"Forcing header.txt written: {header_path}")

    def _write_mhm_forcing(
        self,
        forcing_ds: xr.Dataset,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Write forcing data in mHM NetCDF grid format."""
        props = self._get_catchment_properties()
        times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='D')

        # Forcing arrives under the canonical CFIF vocabulary (precipitation_flux
        # kg m-2 s-1 == mm s-1, air_temperature K) from open_canonical_forcing.
        if 'precipitation_flux' in forcing_ds:
            precip = forcing_ds['precipitation_flux'].values * 86400.0  # mm/s -> mm/day rate
        else:
            logger.warning("No canonical precip (precipitation_flux); using zeros")
            precip = np.zeros(len(times))

        if 'air_temperature' in forcing_ds:
            temp = forcing_ds['air_temperature'].values
            if np.nanmean(temp) > 100:   # Kelvin
                temp = temp - 273.15
        else:
            logger.warning("No canonical airtemp; estimating")
            day_frac = np.arange(len(times)) / 365.0
            temp = 10 + 10 * np.sin(2 * np.pi * day_frac)

        # Flatten to 1D if needed
        if precip.ndim > 1:
            precip = precip.reshape(len(times), -1).mean(axis=1)
        if temp.ndim > 1:
            temp = temp.reshape(len(times), -1).mean(axis=1)

        # Aggregate sub-daily forcing to daily (mHM expects daily timesteps)
        time_index = pd.DatetimeIndex(times)
        if len(time_index) > 1 and (time_index[1] - time_index[0]) < pd.Timedelta('1D'):
            logger.info("Aggregating sub-daily forcing to daily for mHM...")
            precip_series = pd.Series(precip, index=time_index)
            temp_series = pd.Series(temp, index=time_index)
            # Precipitation: sum hourly values (already in mm/day rate, so average)
            precip_daily = precip_series.resample('D').mean()
            # Temperature: daily mean
            temp_daily = temp_series.resample('D').mean()
            precip = precip_daily.values
            temp = temp_daily.values
            times = precip_daily.index.values
            logger.info(f"Aggregated to {len(times)} daily timesteps")

        doy = pd.DatetimeIndex(times).dayofyear.values

        if self.distributed_morph:
            # Basin-mean forcing broadcast across the distributed grid (L2 meteo
            # grid == L0/L1 grid, so resolutions align exactly). With
            # MHM_TEMP_LAPSE_RATE set, temperature is lapsed from the basin-mean
            # DEM elevation to each cell and PET follows the cell temperature;
            # precipitation stays uniform.
            g = self._build_distributed_grids()
            temp_grid = self._lapse_temperature(temp, g)
            lat2d, _ = self._grid_latlon(g)
            pet_grid = self._compute_pet(temp_grid, doy[:, None, None], lat2d[None])
            self._write_distributed_forcing(
                {'pre': precip, 'tavg': temp_grid, 'pet': pet_grid}, times)
            logger.info(f"mHM distributed forcing written to {self.forcing_dir}")
            return

        pet = self._compute_pet(temp, doy, props['lat'])

        # Write forcing NetCDF files in mHM-expected format
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pre.nc', 'pre', precip, times, props,
            'precipitation', 'mm/day')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'tavg.nc', 'tavg', temp, times, props,
            'average temperature', 'degC')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pet.nc', 'pet', pet, times, props,
            'potential evapotranspiration', 'mm/day')

        # Write header.txt for the forcing grid
        self._write_forcing_header(props)

        logger.info(f"mHM forcing files written to {self.forcing_dir}")

    def _lapse_temperature(self, temp: np.ndarray, g: dict) -> np.ndarray:
        """Return temperature as (time, nrow, ncol), lapsed to each cell's
        elevation when MHM_TEMP_LAPSE_RATE (K/km) is configured."""
        lapse = self._get_config_value(
            lambda: self.config.model.mhm.temp_lapse_rate,
            default=None, dict_key='MHM_TEMP_LAPSE_RATE')
        temp = np.asarray(temp, 'f8')
        if lapse is None:
            return np.broadcast_to(temp[:, None, None], (len(temp), g['nrow'], g['ncol'])).copy()
        dem = np.where(g['mask'], g['dem'], np.nan)
        z_ref = float(np.nanmean(dem))
        offset = np.nan_to_num(float(lapse) * (dem - z_ref) / 1000.0)
        logger.info(f"mHM temperature lapsed at {lapse} K/km from basin-mean "
                    f"{z_ref:.0f} m (cell offsets {np.nanmin(offset[g['mask']]):+.1f}"
                    f" to {np.nanmax(offset[g['mask']]):+.1f} K)")
        return temp[:, None, None] + offset[None]

    def _compute_pet(self, temp, doy, lat_deg):
        """Daily PET (mm/day) from temperature with Hamon or Oudin
        (MHM_PET_METHOD). Arguments broadcast, so this serves both the lumped
        series and the (time, nrow, ncol) distributed grid."""
        method = self._get_config_value(
            lambda: self.config.model.mhm.pet_method,
            default='hamon', dict_key='MHM_PET_METHOD')
        lat_rad = np.radians(lat_deg)
        declination = 0.4093 * np.sin(2 * np.pi / 365 * doy - 1.405)
        ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(declination), -1.0, 1.0))
        if method == 'oudin':
            # Oudin et al. (2005): PET = Re / (lambda rho) * (T + 5) / 100.
            dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
            re = (24 * 60 / np.pi) * 0.0820 * dr * (
                ws * np.sin(lat_rad) * np.sin(declination)
                + np.cos(lat_rad) * np.cos(declination) * np.sin(ws))
            pet = re / 2.45 * (temp + 5.0) / 100.0
        else:
            # Hamon: D = day length in 12-hour units, RHOSAT = saturated
            # absolute humidity (g/m3).
            D = (24.0 / np.pi * ws) / 12.0
            es = 6.108 * np.exp(17.27 * temp / (temp + 237.3))
            pet = 0.1651 * D * D * 216.7 * es / (temp + 273.15)
        return np.maximum(pet, 0.0)

    def _grid_latlon(self, g):
        """2D lat/lon (yc,xc) of cell centres for a distributed grid dict."""
        t = g['transform']
        nrow, ncol = g['nrow'], g['ncol']
        xs = t.c + t.a * (np.arange(ncol) + 0.5)
        ys = t.f + t.e * (np.arange(nrow) + 0.5)   # t.e negative -> descending
        lon2d, lat2d = np.meshgrid(xs, ys)
        return lat2d, lon2d

    def _write_distributed_forcing(self, var_data: Dict, times) -> None:
        """Write pre/tavg/pet NetCDFs + header.txt on the distributed grid.
        Each variable is a daily series broadcast uniformly across active
        cells, or an already-gridded (time, nrow, ncol) array."""
        g = self._build_distributed_grids()
        nrow, ncol = g['nrow'], g['ncol']
        lat2d, lon2d = self._grid_latlon(g)
        fill = -9999.0
        t0 = pd.Timestamp(times[0])
        tnum = (pd.DatetimeIndex(times) - t0).days.values.astype('f8')
        units = {'pre': 'mm/day', 'tavg': 'degC', 'pet': 'mm/day'}
        for name, series in var_data.items():
            fp = self.forcing_dir / f'{name}.nc'
            ds = nc4.Dataset(fp, 'w', format='NETCDF4')
            try:
                ds.createDimension('time', None)
                ds.createDimension('yc', nrow)
                ds.createDimension('xc', ncol)
                tv = ds.createVariable('time', 'f8', ('time',))
                tv.units = f"days since {t0.strftime('%Y-%m-%d')} 00:00:00"
                tv.calendar = 'standard'
                tv[:] = tnum
                lonv = ds.createVariable('lon', 'f8', ('yc', 'xc')); lonv.units = 'degrees_east'; lonv[:] = lon2d
                latv = ds.createVariable('lat', 'f8', ('yc', 'xc')); latv.units = 'degrees_north'; latv[:] = lat2d
                dv = ds.createVariable(name, 'f8', ('time', 'yc', 'xc'), fill_value=fill)
                dv.units = units[name]
                arr = np.nan_to_num(np.asarray(series, 'f8'), nan=0.0)
                if arr.ndim == 1:
                    arr = arr[:, None, None]
                grid = np.broadcast_to(arr, (arr.shape[0], nrow, ncol)).copy()
                grid[:, ~g['mask']] = fill
                dv[:] = grid
            finally:
                ds.close()
        # header.txt for the meteo grid (== distributed grid)
        t = g['transform']
        xll = t.c
        yll = t.f + t.e * nrow
        (self.forcing_dir / 'header.txt').write_text(
            f"ncols         {ncol}\nnrows         {nrow}\n"
            f"xllcorner     {xll:.6f}\nyllcorner     {yll:.6f}\n"
            f"cellsize      {t.a:.8f}\nNODATA_value  -9999\n", encoding='utf-8')

    def _generate_synthetic_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """Generate synthetic forcing data for smoke tests ONLY.

        Sine-wave temperature and random precipitation — physically meaningless.
        Only reachable via SYMFLUENCE_MHM_ALLOW_SYNTHETIC_FORCING=1; a real run
        raises instead of calling this (see _generate_forcing_files).
        """
        props = self._get_catchment_properties()
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)

        day_frac = np.arange(n) / 365.0
        precip = np.random.exponential(2.0, n)
        temp = 10 + 10 * np.sin(2 * np.pi * day_frac)
        pet = np.maximum(0, 2 + 3 * np.sin(2 * np.pi * (day_frac - 0.25)))

        # Write forcing NetCDF files in mHM-expected format
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pre.nc', 'pre', precip, dates.values, props,
            'precipitation', 'mm/day')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'tavg.nc', 'tavg', temp, dates.values, props,
            'average temperature', 'degC')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pet.nc', 'pet', pet, dates.values, props,
            'potential evapotranspiration', 'mm/day')

        # Write header.txt for the forcing grid
        self._write_forcing_header(props)

        logger.info(f"Synthetic mHM forcing files written to {self.forcing_dir}")

    def _write_asc_grid(self, filepath, value, props, cellsize=None,
                        value_row2=None):
        """
        Write a 2-row × 1-col ASCII grid file in ESRI .asc format.

        mRM needs ≥ 2 cells to create routing reaches.  For a lumped
        setup the grid has two rows (upstream cell → outlet cell) so
        that one reach exists and discharge can be routed to the gauge.

        Args:
            filepath: Path to the output .asc file.
            value: Data value for the upstream cell (row 1, printed first).
            props: Dict with 'lat' and 'lon' for the cell centre.
            cellsize: Grid cell size in degrees (default ``LUMPED_CELLSIZE``).
            value_row2: Optional different value for the outlet cell
                        (row 2).  Defaults to *value*.
        """
        cellsize = cellsize or self.LUMPED_CELLSIZE
        if value_row2 is None:
            value_row2 = value
        xllcorner = props['lon'] - cellsize / 2
        yllcorner = props['lat'] - cellsize / 2 - cellsize  # 2 rows
        content = (
            f"ncols         1\n"
            f"nrows         2\n"
            f"xllcorner     {xllcorner:.6f}\n"
            f"yllcorner     {yllcorner:.6f}\n"
            f"cellsize      {cellsize}\n"
            f"NODATA_value  -9999\n"
            f"{value}\n"        # row 1 (top / upstream)
            f"{value_row2}\n"   # row 2 (bottom / outlet)
        )
        Path(filepath).write_text(content, encoding='utf-8')

    def _generate_morph_files(self) -> None:
        """
        Generate morphological input files as ASCII grid (.asc) files.

        mHM v5.13 requires ESRI ASCII grid format for all morphological
        inputs. For a lumped 1x1 model each file contains a single cell.

        Generates in morph_dir:
            dem.asc, slope.asc, aspect.asc, soil_class.asc, facc.asc,
            fdir.asc, idgauges.asc, LAI_class.asc, geology_class.asc

        Generates in lcover_dir:
            lc_{start_year}.asc
        """
        logger.info("Generating mHM morphological inputs (ASCII grids)...")

        props = self._get_catchment_properties()
        start_date, _ = self._get_simulation_dates()

        if self.distributed_morph:
            self._generate_distributed_morph(start_date)
            return

        # Remove any old .nc morph files (mHM v5.13 requires .asc only)
        for old_nc in self.morph_dir.glob('*.nc'):
            if old_nc.name != 'latlon.nc':
                old_nc.unlink()
                logger.info(f"Removed old NetCDF morph file: {old_nc.name}")

        # --- Morph directory grids (2-row lumped layout) ---
        # Row 1 = upstream cell, row 2 = outlet cell with gauge.
        # Both cells share the same properties so the model is effectively
        # lumped, but mRM can create one routing reach.
        self._write_asc_grid(self.morph_dir / 'dem.asc', props['elev'], props)
        self._write_asc_grid(self.morph_dir / 'slope.asc', 0.05, props)
        self._write_asc_grid(self.morph_dir / 'aspect.asc', 180.0, props)
        self._write_asc_grid(self.morph_dir / 'soil_class.asc', 1, props)
        # Flow accumulation: upstream=1, outlet=2 (collects from upstream)
        self._write_asc_grid(self.morph_dir / 'facc.asc', 1, props,
                             value_row2=2)
        # D8 flow direction: both cells flow South (4).
        # Row 1 → row 2 (within domain), row 2 → outside (outlet).
        self._write_asc_grid(self.morph_dir / 'fdir.asc', 4, props)
        # Gauge only on the outlet cell (row 2)
        self._write_asc_grid(self.morph_dir / 'idgauges.asc', -9999, props,
                             value_row2=1)
        self._write_asc_grid(self.morph_dir / 'LAI_class.asc', 1, props)
        self._write_asc_grid(self.morph_dir / 'geology_class.asc', 1, props)

        # --- Land cover directory grid ---
        # Class 3 = pervious (appropriate for lumped basins — forest class
        # over-extracts water via rootFractionCoefficient).
        self._write_asc_grid(
            self.lcover_dir / f'lc_{start_date.year}.asc', 3, props
        )

        logger.info(f"Morphological ASCII grids written to {self.morph_dir}")

    def _write_asc_array(self, filepath, grid, mask, transform, intfmt=False):
        """Write a 2D numpy array as an ESRI .asc grid (masked cells = -9999)."""
        nrow, ncol = grid.shape
        xll = transform.c
        yll = transform.f + transform.e * nrow   # transform.e is negative
        cellsize = transform.a
        out = np.where(mask, grid, -9999)
        lines = [f"ncols         {ncol}", f"nrows         {nrow}",
                 f"xllcorner     {xll:.6f}", f"yllcorner     {yll:.6f}",
                 f"cellsize      {cellsize:.8f}", "NODATA_value  -9999"]
        for r in range(nrow):
            if intfmt:
                lines.append(" ".join(str(int(v)) if v != -9999 else "-9999"
                                      for v in out[r]))
            else:
                lines.append(" ".join(f"{v:.4f}" if v != -9999 else "-9999"
                                      for v in out[r]))
        Path(filepath).write_text("\n".join(lines) + "\n", encoding='utf-8')

    def _generate_distributed_morph(self, start_date) -> None:
        """Write all morphological .asc grids from the TauDEM-derived
        distributed grid (real DEM/soil/land/flow over the catchment)."""
        g = self._build_distributed_grids()
        m, t = g['mask'], g['transform']
        oi, oj = g['outlet']
        md = self.morph_dir
        self._write_asc_array(md / 'dem.asc', g['dem'], m, t)
        self._write_asc_array(md / 'slope.asc', g['slope'], m, t)
        self._write_asc_array(md / 'aspect.asc', np.full(g['dem'].shape, 180.0), m, t)
        self._write_asc_array(md / 'fdir.asc', g['fdir'], m, t, intfmt=True)
        self._write_asc_array(md / 'facc.asc', g['facc'], m, t, intfmt=True)
        # Soil/geology/LAI classes: remap to dense 1..N ids the class
        # definition files declare. Soil from the datastore; geology uniform
        # (no Iceland geology raster); LAI from a single class.
        soil_ids = self._dense_class_grid(g['soil'], m)
        self._write_asc_array(md / 'soil_class.asc', soil_ids, m, t, intfmt=True)
        self._write_asc_array(md / 'geology_class.asc',
                              np.ones(g['dem'].shape, 'int32'), m, t, intfmt=True)
        self._write_asc_array(md / 'LAI_class.asc',
                              np.ones(g['dem'].shape, 'int32'), m, t, intfmt=True)
        # Gauge cell at the outlet only.
        idg = np.full(g['dem'].shape, -9999, 'int32')
        idg[oi, oj] = 1
        self._write_asc_array(md / 'idgauges.asc', idg, m, t, intfmt=True)
        # Land cover -> mHM 3-class scheme (1=forest,2=impervious,3=pervious).
        lc = self._landcover_to_mhm(g['land'])
        # Glacier cells become class 2 (sealed, fracSealed_cityArea = 1): no
        # soil store, own degree-day factors. glacier_mask.asc is not an mHM
        # input; the calibration worker reads it to add external ice melt.
        lc[g['glacier']] = 2
        self._write_asc_array(md / 'glacier_mask.asc', g['glacier'].astype('int32'),
                              m, t, intfmt=True)
        # Glacierised fraction of each glacier cell, for MHM_GLACIER_AREA_FROM_OUTLINE.
        self._write_asc_array(md / 'glacier_fraction.asc',
                              np.where(g['glacier'], g['glacier_frac'], 0.0), m, t)
        self._write_asc_array(self.lcover_dir / f'lc_{start_date.year}.asc',
                              lc, m, t, intfmt=True)
        self._n_soil_classes = int(soil_ids[m].max()) if m.any() else 1
        logger.info(f"Distributed morph grids written ({self._n_soil_classes} soil classes)")

    def _has_glacier_cells(self) -> bool:
        return bool(self.distributed_morph and self._build_distributed_grids()['glacier'].any())

    @staticmethod
    def _dense_class_grid(grid, mask):
        """Remap arbitrary class codes within the mask to dense ids 1..N."""
        out = np.ones(grid.shape, 'int32')
        vals = sorted(set(int(v) for v in grid[mask].tolist()) - {0})
        remap = {v: i + 1 for i, v in enumerate(vals)} if vals else {}
        for v, i in remap.items():
            out[grid == v] = i
        return out

    @staticmethod
    def _landcover_to_mhm(land):
        """MODIS IGBP -> mHM 3-class (1=forest, 2=impervious, 3=pervious)."""
        out = np.full(land.shape, 3, 'int32')   # default pervious
        out[np.isin(land, [1, 2, 3, 4, 5])] = 1  # forests
        out[land == 13] = 2                       # urban/impervious
        return out

    def _generate_mhm_namelist(self) -> None:
        """
        Generate the mhm.nml Fortran namelist file for mHM v5.13.

        Writes ALL required namelist blocks in the order expected by mHM:
        project_description, mainconfig, mainconfig_mhm_mrm,
        mainconfig_mrm, directories_general, directories_mHM,
        directories_mRM, optional_data, processSelection, LCover,
        time_periods, soildata, LAI_data_information, LCover_MPR,
        directories_MPR, evaluation_gauges, inflow_gauges, panEvapo,
        nightDayRatio, Optimization, baseflow_config.

        Note: Parameter values (bounds, defaults, flags) are written to the
        separate ``mhm_parameter.nml`` file by ``_generate_mhm_parameter_namelist``.
        Output control is handled by ``_generate_mhm_outputs_namelist``.
        """
        logger.info("Generating mhm.nml...")

        start_date, end_date = self._get_simulation_dates()
        self._get_catchment_properties()

        output_dir = str(self.output_mhm_dir)
        morph_dir = str(self.morph_dir)
        lcover_dir = str(self.lcover_dir)
        forcing_dir = str(self.forcing_dir)
        gauge_dir = str(self.gauge_dir)

        # Compute warmup days for mHM.
        # mHM warming_Days goes BACKWARD from eval_Per start, so we shift
        # eval_Per to start at the calibration period and set warming_Days
        # to the gap back to the experiment start (where forcing data begins).
        cal_period = self._get_config_value(
            lambda: self.config.domain.calibration_period, default='')
        if cal_period:
            cal_start_str = str(cal_period).split(',')[0].strip()
            cal_start = pd.to_datetime(cal_start_str)
            warmup_days = (cal_start - pd.to_datetime(start_date)).days
            eval_start = cal_start.to_pydatetime()
        else:
            warmup_days = 0
            eval_start = start_date

        namelist_content = f"""\
!-- mHM Namelist File (mHM v5.13) --
!-- Generated by SYMFLUENCE on {datetime.now().isoformat()} --

&project_description
  project_details = "SYMFLUENCE mHM model run"
  setup_description = "Lumped mHM simulation for {self.domain_name}"
  simulation_type = "historical simulation"
  Conventions = "CF-1.6"
  contact = "SYMFLUENCE"
  mHM_details = "mHM v5.13"
  history = "auto-generated"
/

&mainconfig
  iFlag_cordinate_sys = 1     ! lat/lon coordinates
  nDomains = 1
  resolution_Hydrology(1) = {self.grid_res if self.distributed_morph else self.LUMPED_CELLSIZE}
  L0Domain(1) = 1
  write_restart = .FALSE.
  read_opt_domain_data(1) = 0
/

&mainconfig_mhm_mrm
  mhm_file_RestartIn(1) = ""
  mrm_file_RestartIn(1) = ""
  resolution_Routing(1) = {self.grid_res if self.distributed_morph else self.LUMPED_CELLSIZE}
  timestep = {self._model_timestep()}
  read_restart = .FALSE.
  optimize = .FALSE.
  optimize_restart = .FALSE.
  opti_method = 1
  opti_function = 9
/

&mainconfig_mrm
  ALMA_convention = .TRUE.
  varnametotalrunoff = 'total_runoff'
  filenametotalrunoff = 'total_runoff'
  gw_coupling = .false.
/

&directories_general
  dirConfigOut = "{output_dir}/"
  dirCommonFiles = "{morph_dir}/"
  dir_Morpho(1) = "{morph_dir}/"
  dir_LCover(1) = "{lcover_dir}/"
  mhm_file_RestartOut(1) = "{output_dir}/mHM_restart_001.nc"
  mrm_file_RestartOut(1) = "{output_dir}/mRM_restart_001.nc"
  dir_Out(1) = "{output_dir}/"
  file_LatLon(1) = "{morph_dir}/latlon.nc"
/

&directories_mHM
  inputFormat_meteo_forcings = "nc"
  bound_error = .FALSE.
  dir_Precipitation(1) = "{forcing_dir}/"
  dir_Temperature(1) = "{forcing_dir}/"
  dir_ReferenceET(1) = "{forcing_dir}/"
  dir_MinTemperature(1) = "{forcing_dir}/"
  dir_MaxTemperature(1) = "{forcing_dir}/"
  dir_NetRadiation(1) = "{forcing_dir}/"
  dir_absVapPressure(1) = "{forcing_dir}/"
  dir_windspeed(1) = "{forcing_dir}/"
  dir_Radiation(1) = "{forcing_dir}/"
  time_step_model_inputs(1) = 0
/

&directories_mRM
  dir_Gauges(1) = "{gauge_dir}/"
  dir_Total_Runoff(1) = "{output_dir}/"
  dir_Bankfull_Runoff(1) = "{output_dir}/"
/

&optional_data
  dir_soil_moisture(1) = "{output_dir}/"
  nSoilHorizons_sm_input = 1
  timeStep_sm_input = -2
  dir_neutrons(1) = "{output_dir}/"
  dir_evapotranspiration(1) = "{output_dir}/"
  timeStep_et_input = -2
  dir_tws(1) = "{output_dir}/"
  timeStep_tws_input = -2
/

&processSelection
  processCase(1)  = 1    ! Interception
  processCase(2)  = 1    ! Snowpack
  processCase(3)  = 1    ! Soil moisture
  processCase(4)  = 1    ! Direct runoff
  processCase(5)  = 0    ! PET (0 = PET is input)
  processCase(6)  = 1    ! Interflow
  processCase(7)  = 1    ! Percolation
  processCase(8)  = 1    ! Routing (Muskingum)
  processCase(9)  = 1    ! Baseflow
  processCase(10) = 0    ! Neutrons (off)
  processCase(11) = 0    ! River temperature (off)
/

&LCover
  nLCoverScene = 1
  LCoverYearStart(1) = {start_date.year}
  LCoverYearEnd(1) = {end_date.year}
  LCoverfName(1) = 'lc_{start_date.year}.asc'
/

&time_periods
  warming_Days(1) = {warmup_days}
  eval_Per(1)%yStart = {eval_start.year}
  eval_Per(1)%mStart = {eval_start.month}
  eval_Per(1)%dStart = {eval_start.day}
  eval_Per(1)%yEnd = {end_date.year}
  eval_Per(1)%mEnd = {end_date.month}
  eval_Per(1)%dEnd = {end_date.day}
/

&soildata
  iFlag_soilDB = 0
  tillageDepth = 200
  nSoilHorizons_mHM = 2
  soil_Depth(1) = 200
/

&LAI_data_information
  timeStep_LAI_input = 0
  inputFormat_gridded_LAI = "nc"
/

&LCover_MPR
  fracSealed_cityArea = {1.0 if self._has_glacier_cells() else 0.6}
/

&directories_MPR
  dir_gridded_LAI(1) = "{morph_dir}/"
/

&evaluation_gauges
  nGaugesTotal = 1
  NoGauges_domain(1) = 1
  Gauge_id(1,1) = 1
  gauge_filename(1,1) = "gauge_001.txt"
/

&inflow_gauges
  nInflowGaugesTotal = 0
  NoInflowGauges_domain(1) = 0
  InflowGauge_id(1,1) = -9
  InflowGauge_filename(1,1) = ""
  InflowGauge_Headwater(1,1) = .FALSE.
/

&panEvapo
  evap_coeff = 1.30, 1.20, 0.72, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.50
/

&nightDayRatio
  read_meteo_weights = .FALSE.
  fnight_prec = 0.46, 0.50, 0.52, 0.51, 0.48, 0.50, 0.49, 0.48, 0.52, 0.56, 0.50, 0.47
  fnight_pet  = 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10
  fnight_temp = -0.76, -1.30, -1.88, -2.38, -2.72, -2.75, -2.74, -3.04, -2.44, -1.60, -0.94, -0.53
  fnight_ssrd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  fnight_strd = 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45
/

&Optimization
  nIterations = 7
  seed = 1235876
  dds_r = 0.2
  sa_temp = -9.0
  sce_ngs = 2
  sce_npg = -9
  sce_nps = -9
  mcmc_opti = .false.
  mcmc_error_params = 0.01, 0.6
/

&baseflow_config
  BFI_calc = .true.
/
"""
        namelist_path = self.settings_dir / 'mhm.nml'
        namelist_path.write_text(namelist_content, encoding='utf-8')
        logger.info(f"mhm.nml written: {namelist_path}")

    def _generate_lookup_tables(self) -> None:
        """
        Generate mHM lookup table text files in the morph directory.

        Creates:
            soil_classdefinition.txt  - Soil horizon properties
            LAI_classdefinition.txt   - Monthly LAI per land-use class
            geology_classdefinition.txt - Geology / aquifer properties
        """
        logger.info("Generating mHM lookup table files...")

        # --- soil_classdefinition.txt ---
        # Declare one MU_GLOBAL per soil class present on the grid (distributed
        # mode may resolve several). Two horizons each; Andosol-leaning texture
        # (low clay / high sand, typical of Icelandic volcanic soils). MPR's
        # pedotransfer functions + calibrated coefficients set the actual Ks.
        n_soil = getattr(self, '_n_soil_classes', 1) if self.distributed_morph else 1
        rows = ["nSoil_Types  %d" % n_soil,
                "MU_GLOBAL\tHORIZON\tUD[mm]\tLD[mm]\tCLAY[%]\tSAND[%]\tBD[gcm-3]"]
        for c in range(1, n_soil + 1):
            rows.append(f"{c}\t1\t0\t200\t12.0\t60.0\t0.90")
            rows.append(f"{c}\t2\t200\t1500\t15.0\t55.0\t1.00")
        (self.morph_dir / 'soil_classdefinition.txt').write_text(
            "\n".join(rows) + "\n", encoding='utf-8'
        )

        # --- LAI_classdefinition.txt ---
        lai_content = (
            "NoLAIclasses           3\n"
            "ID   LAND-USE                  Jan.   Feb.    Mar.    Apr.    May"
            "    Jun.    Jul.    Aug.    Sep.    Oct.    Nov.    Dec.\n"
            " 1    Coniferous-forest         11     11      11      11      11"
            "     11      11      11      11      11      11      11\n"
            " 2    Deciduous-forest          0.5    0.5     1.5     4.0     7.0"
            "    11      12      12      11      8.0     1.5     0.5\n"
            " 3    Sealed-Water-bodies       0.01   0.01    0.02    0.04    0.06"
            "   0.09    0.09    0.07    0.06    0.04    0.02    0.02\n"
        )
        (self.morph_dir / 'LAI_classdefinition.txt').write_text(
            lai_content, encoding='utf-8'
        )

        # --- geology_classdefinition.txt ---
        geo_content = (
            "nGeo_Formations  1\n"
            "GeoParam(i)   ClassUnit     Karstic      Description\n"
            "         1            1           0      GeoUnit-1\n"
            "!<-END\n"
        )
        (self.morph_dir / 'geology_classdefinition.txt').write_text(
            geo_content, encoding='utf-8'
        )

        logger.info(f"Lookup tables written to {self.morph_dir}")

    def _generate_gauge_data(self) -> None:

        """
        Generate gauge observation data file for mHM.

        Attempts to load observed streamflow from the project using the
        ObservationLoaderMixin. If no observations are available, writes
        a file filled with the mHM missing-value marker (-9999).

        Output:
            gauge_dir / gauge_001.txt  (one daily discharge value per line, m3/s)
        """
        logger.info("Generating mHM gauge data file...")

        start_date, end_date = self._get_simulation_dates()
        n_days = (end_date - start_date).days + 1

        # Try to load real observations via the mixin
        values = None
        try:
            obs_series = self.load_streamflow_observations(
                output_format='series',
                target_units='cms',
                return_none_on_error=True
            )
            if obs_series is not None and len(obs_series) > 0:
                date_index = pd.date_range(start_date, end_date, freq='D')
                obs_reindexed = obs_series.reindex(date_index)
                values = obs_reindexed.values
                n_obs = int(obs_reindexed.notna().sum())
                logger.info(
                    f"Loaded {n_obs} observed streamflow values for gauge file"
                )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not load observed streamflow: {e}", exc_info=True)

        if values is None:
            values = np.full(n_days, -9999.0)
            logger.info("No observations available; gauge file filled with -9999")

        # Replace NaN with missing-value marker
        values = np.where(np.isnan(values), -9999.0, values)

        # Build date range for data lines
        date_index = pd.date_range(start_date, end_date, freq='D')

        # Write gauge file with mHM 5-line header format
        header_lines = [
            f"# Gauge observation data for {self.domain_name}",
            "nodata_value  -9999",
            "measurements_per_day  1",
            f"start  {start_date.year} {start_date.month:02d} {start_date.day:02d}",
            f"end  {end_date.year} {end_date.month:02d} {end_date.day:02d}",
        ]
        data_lines = []
        for dt, v in zip(date_index, values):
            val_str = f"{v:.4f}" if v != -9999.0 else "-9999"
            data_lines.append(f"{dt.year} {dt.month:02d} {dt.day:02d} 00 00 {val_str}")

        gauge_path = self.gauge_dir / 'gauge_001.txt'
        gauge_path.write_text(
            "\n".join(header_lines + data_lines) + "\n", encoding='utf-8'
        )
        logger.info(f"Gauge data written: {gauge_path}")

    def _generate_latlon_file(self) -> None:
        """
        Generate the latlon.nc NetCDF file required by mHM.

        mHM reads latitude and longitude fields from a NetCDF file
        referenced by file_LatLon(1) in the namelist.  For the 2-row
        lumped layout this contains two cells (upstream + outlet).

        Variables written (all float64, dims (nrows, ncols)):
            lat, lon       -- cell-centre coordinates
            lat_l0, lon_l0 -- level-0 coordinates
            lat_l1, lon_l1 -- level-1 (hydrology) coordinates
            lat_l11, lon_l11 -- level-11 (routing) coordinates

        Output:
            morph_dir / latlon.nc
        """
        logger.info("Generating latlon.nc...")

        props = self._get_catchment_properties()
        if self.distributed_morph:
            g = self._build_distributed_grids()
            lat2d, lon2d = self._grid_latlon(g)
            lat_val, lon_val = lat2d, lon2d
            nlat, nlon = g['nrow'], g['ncol']
        else:
            cellsize = self.LUMPED_CELLSIZE
            lat_val = np.array(
                [[props['lat']], [props['lat'] - cellsize]], dtype=np.float64)
            lon_val = np.full((2, 1), props['lon'], dtype=np.float64)
            nlat, nlon = 2, 1

        latlon_path = self.morph_dir / 'latlon.nc'
        ds = nc4.Dataset(str(latlon_path), 'w', format='NETCDF4')
        try:
            ds.createDimension('nrows', nlat)
            ds.createDimension('ncols', nlon)

            for vname, data in [
                ('lat', lat_val), ('lon', lon_val),
                ('lat_l0', lat_val), ('lon_l0', lon_val),
                ('lat_l1', lat_val), ('lon_l1', lon_val),
                ('lat_l11', lat_val), ('lon_l11', lon_val),
            ]:
                var = ds.createVariable(vname, 'f8', ('nrows', 'ncols'))
                if 'lat' in vname:
                    var.units = 'degrees_north'
                    var.long_name = 'latitude'
                else:
                    var.units = 'degrees_east'
                    var.long_name = 'longitude'
                var[:] = data
        finally:
            ds.close()

        logger.info(f"latlon.nc written: {latlon_path}")

    def _generate_mrm_namelist(self) -> None:
        """
        Generate the mrm.nml Fortran namelist file for mRM routing.

        This namelist controls the mRM (mesoscale Routing Model) routing
        component of mHM.
        """
        logger.info("Generating mrm.nml...")

        props = self._get_catchment_properties()
        if self.distributed_morph:
            g = self._build_distributed_grids()
            lat2d, lon2d = self._grid_latlon(g)
            oi, oj = g['outlet']
            gauge_lat = float(lat2d[oi, oj])
            gauge_lon = float(lon2d[oi, oj])
        else:
            gauge_lat = props['lat'] - self.LUMPED_CELLSIZE
            gauge_lon = props['lon']

        routing_content = f"""!-- mRM Routing Namelist File --
!-- Generated by SYMFLUENCE on {datetime.now().isoformat()} --

&routing1
  gaugeID(1)          = 1
  gauge_lat(1)        = {gauge_lat:.6f}
  gauge_lon(1)        = {gauge_lon:.6f}
/

&routing_general
  routing_model       = 1    ! Muskingum-Cunge routing
  timeStep_model_outputs_mrm = -2    ! daily
/
"""
        routing_path = self.settings_dir / 'mrm.nml'
        routing_path.write_text(routing_content, encoding='utf-8')
        logger.info(f"mrm.nml written: {routing_path}")

    def _generate_mhm_parameter_namelist(self) -> None:
        """
        Generate the ``mhm_parameter.nml`` Fortran namelist file.

        This file contains all mHM process parameters with their lower bound,
        upper bound, default value, optimisation flag, and scaling factor.
        The mHM binary reads this file at startup alongside ``mhm.nml`` and
        ``mrm.nml``.

        The format for each parameter line is::

            parameterName = lower, upper, value, FLAG, SCALING
        """
        logger.info("Generating mhm_parameter.nml...")

        from .parameters import MHM_PARAMETER_NML_SPEC

        lines = [
            "!> \\file mhm_parameter.nml",
            "!> \\brief Parameters for mHM.",
            f"!> \\details Generated by SYMFLUENCE on {datetime.now().isoformat()}",
            "!!",
            "!! PARAMETER = lower_bound, upper_bound, value, FLAG, SCALING",
            "!!   FLAG: 1 = parameter is optimised, 0 = parameter is fixed",
            "!!   SCALING: typically 1",
            "",
        ]

        for block_name, params in MHM_PARAMETER_NML_SPEC:
            lines.append(f"&{block_name}")
            for entry in params:
                pname, lower, upper, value, flag, scaling = entry
                # Format numbers consistently for Fortran readability
                lower_s = self._format_param_nml_value(lower)
                upper_s = self._format_param_nml_value(upper)
                value_s = self._format_param_nml_value(value)
                lines.append(
                    f"  {pname:45s} = {lower_s}, {upper_s}, {value_s}, {flag}, {scaling}"
                )
            lines.append("/")
            lines.append("")

        content = "\n".join(lines)
        param_nml_path = self.settings_dir / 'mhm_parameter.nml'
        param_nml_path.write_text(content, encoding='utf-8')
        logger.info(f"mhm_parameter.nml written: {param_nml_path}")

    def _generate_mhm_outputs_namelist(self) -> None:
        """
        Generate the ``mhm_outputs.nml`` Fortran namelist file.

        This file controls which mHM output variables are written and their
        format.  The ``outputFlxState`` logical array (21 elements) selects
        individual fluxes and states.  Element 11 (``total_runoff``) is enabled
        so that simulated discharge is available for evaluation.

        References:
            mHM source files ``mo_namelists.f90`` and ``mo_mhm_read_config.f90``.
        """
        logger.info("Generating mhm_outputs.nml...")

        # Build the outputFlxState array: only element 11 (total_runoff) is .true.
        n_elements = 21
        total_runoff_index = 11
        flx_lines = []
        for i in range(1, n_elements + 1):
            value = ".true." if i == total_runoff_index else ".false."
            flx_lines.append(f"  outputFlxState({i}) = {value}")

        namelist_content = (
            f"!-- mHM Output Namelist File --\n"
            f"!-- Generated by SYMFLUENCE on {datetime.now().isoformat()} --\n"
            f"\n"
            f"&nloutputresults\n"
            f"  output_deflate_level = 6\n"
            f"  output_double_precision = .true.\n"
            f"  timeStep_model_outputs = 0\n"
            f"  output_time_reference = 0\n"
            + "\n".join(flx_lines) + "\n"
            "/\n"
        )

        outputs_path = self.settings_dir / 'mhm_outputs.nml'
        outputs_path.write_text(namelist_content, encoding='utf-8')
        logger.info(f"mhm_outputs.nml written: {outputs_path}")

    @staticmethod
    def _format_param_nml_value(value: float) -> str:
        """
        Format a numeric value for the ``mhm_parameter.nml`` file.

        Uses fixed-point notation with 4 decimal places for most values,
        and scientific notation for very small numbers.

        Args:
            value: Numeric value to format.

        Returns:
            Formatted string suitable for Fortran namelist.
        """
        abs_val = abs(value)
        if abs_val == 0.0:
            return "0.0000"
        elif abs_val < 0.0001:
            return f"{value:.4E}"
        elif abs_val == int(abs_val) and abs_val < 1e6:
            return f"{value:.1f}"
        else:
            return f"{value:.4f}"

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
