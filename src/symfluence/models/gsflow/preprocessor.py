# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Pre-Processor.

Prepares the full GSFLOW input file suite:
  - PRMS data file (data.dat) from ERA5 basin-averaged forcing
  - PRMS parameter file (params.dat) with HRU and GVR definitions
  - MODFLOW-NWT packages: NAM, DIS, BAS, UPW, NWT, SFR, OC
  - GSFLOW control file (control.dat) tying PRMS + MODFLOW together

For a lumped domain the MODFLOW grid is a single cell representing
the bulk aquifer.  The SFR package provides stream-aquifer coupling.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor

logger = logging.getLogger(__name__)


class GSFLOWPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Pre-processor for GSFLOW model setup."""


    MODEL_NAME = "GSFLOW"
    def __init__(self, config, logger):
        super().__init__(config, logger)
        # Base class provides:
        #   self.setup_dir = project_dir / "settings" / "GSFLOW"
        #   self.forcing_dir = project_forcing_dir / "GSFLOW_input"
        self.modflow_dir = self.setup_dir / 'modflow'

        # Simulation output directory (matches base runner convention)
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default='run_1'
        )
        self.sim_output_dir = self.project_dir / 'simulations' / experiment_id / self.model_name

        # Relative path from setup_dir (CWD at runtime) to sim_output_dir
        self._output_relpath = os.path.relpath(self.sim_output_dir, self.setup_dir)

        # Distributed-groundwater mode: lumped PRMS surface (1 HRU) coupled to a
        # multi-cell MODFLOW grid carrying real per-cell DEM elevations, with the
        # stream confined to a single flow path. This creates genuine head
        # gradients (topography + distance-to-stream) so hydraulic conductivity
        # and spatial groundwater routing actually matter -- unlike the lumped
        # single-value-elevation grid where K is inert. Isolates the
        # groundwater-routing contribution (surface side identical to PRMS).
        self.distributed_gw = bool(self._get_config_value(
            lambda: self.config.model.gsflow.distributed_gw,
            default=False, dict_key='GSFLOW_DISTRIBUTED_GW'))
        self.gw_grid_n = int(self._get_config_value(
            lambda: self.config.model.gsflow.gw_grid_n,
            default=10, dict_key='GSFLOW_GW_GRID_N'))
        if self.distributed_gw:
            logger.info(f"GSFLOW distributed GW ON ({self.gw_grid_n}x{self.gw_grid_n} "
                        "MODFLOW grid, lumped PRMS surface)")

    def _sample_dem_to_grid(self, nrow: int, ncol: int):
        """Resample the catchment DEM to an nrow x ncol grid of mean elevations.

        Returns a 2D numpy array (nrow x ncol) of cell-mean elevations, or None
        if no DEM is available (caller falls back to a constant surface).
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
            dd = self.project_attributes_dir / 'elevation' / 'dem'
            dems = sorted(dd.glob('*_elv.tif')) if dd.exists() else []
            if not dems:
                return None
            with rasterio.open(dems[0]) as src:
                arr = src.read(
                    1, out_shape=(nrow, ncol), resampling=Resampling.average
                ).astype('float64')
                nd = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            arr = np.where(np.isfinite(arr) & (arr > -100.0), arr, np.nan)
            if np.isnan(arr).all():
                return None
            # Fill any NaN cells with the grid mean so the surface is complete.
            arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
            return arr
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not resample DEM to grid: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run_preprocessing(self) -> bool:
        """Generate all GSFLOW input files from scratch."""
        try:
            for d in (self.setup_dir, self.modflow_dir, self.forcing_dir,
                      self.sim_output_dir, self.sim_output_dir / 'modflow',
                      self.sim_output_dir / 'prms'):
                d.mkdir(parents=True, exist_ok=True)

            gsflow_mode = self._get_config_value(
                lambda: self.config.model.gsflow.gsflow_mode,
                default='COUPLED', dict_key='GSFLOW_MODE'
            ).upper()
            logger.info(f"GSFLOW preprocessing in {gsflow_mode} mode")

            start_date, end_date = self._get_simulation_dates()

            # 1. PRMS data file (forcing)
            self._generate_data_file(start_date, end_date)

            # 2. PRMS parameter file
            self._generate_parameter_file()

            # 3. MODFLOW-NWT packages
            self._generate_modflow_packages(start_date, end_date)

            # 4. Master GSFLOW control file
            self._generate_control_file(start_date, end_date, gsflow_mode)

            logger.info(f"GSFLOW preprocessing complete: {self.setup_dir}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"GSFLOW preprocessing failed: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------
    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        start = self._get_config_value(
            lambda: self.config.domain.time_start, default='2002-01-01')
        end = self._get_config_value(
            lambda: self.config.domain.time_end, default='2009-12-31')
        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        if isinstance(end, str):
            end = pd.Timestamp(end).to_pydatetime()
        return start, end

    # ------------------------------------------------------------------
    # 0.  Per-domain geometry from the model-ready datastore
    # ------------------------------------------------------------------
    def _get_catchment_properties(self) -> dict:
        """Catchment lat/lon/area (shapefile) and mean elev/slope/aspect (DEM).

        Resolve existing geometry independently of the output experiment ID.
        Missing geometry must not silently create a fictitious 100 km2 basin.
        """
        props = {'elev': 500.0, 'slope': 5.0, 'aspect': 180.0}
        try:
            cp = self._find_basin_shapefile(required=True)
            if cp and cp.exists():
                gdf = gpd.read_file(cp)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                if gdf.crs.is_geographic:
                    pt = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326).iloc[0]
                else:
                    pt = gdf.geometry.centroid.to_crs(epsg=4326).iloc[0]
                lon, lat = pt.x, pt.y
                utm = int((lon + 180) / 6) + 1
                area_m2 = gdf.to_crs(f"EPSG:{(32600 if lat >= 0 else 32700)+utm}").geometry.area.sum()
                if not np.isfinite(area_m2) or area_m2 <= 0:
                    raise ValueError('Catchment area must be finite and positive')
                expected = self._get_config_value(lambda: self.config.domain.catchment_area, default=None)
                if expected is not None and not np.isclose(area_m2 / 1e6, expected, rtol=0.02):
                    raise ValueError(f'Geometry area {area_m2 / 1e6:g} km2 disagrees with CATCHMENT_AREA={expected}')
                props.update({'lat': lat, 'lon': lon, 'area_km2': area_m2 / 1e6})
            else:
                raise FileNotFoundError(f'Catchment geometry not found: {cp}')
        except Exception as e:
            raise ValueError('GSFLOW requires readable catchment geometry; refusing default basin area/location') from e
        try:
            import rasterio
            dd = self.project_attributes_dir / 'elevation' / 'dem'
            dems = sorted(dd.glob('*_elv.tif')) if dd.exists() else []
            if dems:
                with rasterio.open(dems[0]) as src:
                    z = src.read(1).astype('float64'); nd = src.nodata
                    xres, yres = src.res; geo = src.crs is not None and src.crs.is_geographic
                    bnds = src.bounds
                m = np.isfinite(z) & (z > -100.0)
                if nd is not None:
                    m &= z != nd
                if m.any():
                    props['elev'] = float(z[m].mean())
                    if geo:
                        lat0 = np.radians((bnds.bottom + bnds.top) / 2.0)
                        xm = abs(xres) * 111320.0 * max(np.cos(lat0), 0.1); ym = abs(yres) * 111320.0
                    else:
                        xm, ym = abs(xres), abs(yres)
                    dy, dx = np.gradient(np.where(m, z, np.nan), ym, xm)
                    props['slope'] = float(np.nanmean(np.degrees(np.arctan(np.hypot(dx, dy)))[m]))
                    props['relief'] = float(np.nanpercentile(z[m], 95) - np.nanpercentile(z[m], 5))
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read DEM attributes: {e}; defaults", exc_info=True)
        props.setdefault('relief', 200.0)
        return props

    # ------------------------------------------------------------------
    # 1.  PRMS data file  (ERA5 → daily P / Tmax / Tmin)
    # ------------------------------------------------------------------
    def _generate_data_file(self, start_date: datetime, end_date: datetime) -> None:
        logger.info("Generating GSFLOW data file from ERA5 forcing...")

        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            raise FileNotFoundError(
                f"Basin-averaged forcing not found: {forcing_path}")

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            raise FileNotFoundError(
                f"No NetCDF forcing files in {forcing_path}")

        logger.info(f"Loading forcing ({len(forcing_files)} files)")
        # Canonical model-ready forcing: variables under the canonical vocabulary
        # (pptrate kg m-2 s-1, airtemp K) with the timestep on
        # ds.attrs['timestep_seconds'] -- no per-model alias lists or hardcoded step.
        from symfluence.core.modeling.model_ready.forcing_reader import (
            forcing_timestep_seconds,
            open_canonical_forcing,
        )
        ds = open_canonical_forcing(forcing_files)
        ds = ds.sel(time=slice(str(start_date), str(end_date)))

        airtemp = ds['air_temperature'].values.squeeze()      # K
        pptrate = ds['precipitation_flux'].values.squeeze()   # mm s-1 (== kg m-2 s-1)
        times = pd.DatetimeIndex(ds['time'].values)
        dt_seconds = forcing_timestep_seconds(ds)

        sub = pd.DataFrame({
            'airtemp_C': airtemp - 273.15,
            'precip_mm': pptrate * dt_seconds,
        }, index=times)

        daily = pd.DataFrame({
            'precip': sub['precip_mm'].resample('D').sum(),
            'tmax': sub['airtemp_C'].resample('D').max(),
            'tmin': sub['airtemp_C'].resample('D').min(),
        }).dropna(how='all')

        ds.close()
        logger.info(f"ERA5 forcing: {len(daily)} days, "
                     f"P [{daily['precip'].min():.1f}–{daily['precip'].max():.1f}] mm/d, "
                     f"T [{daily['tmin'].min():.1f}–{daily['tmax'].max():.1f}] °C")

        out = self.forcing_dir / 'data.dat'
        with open(out, 'w') as f:
            f.write("PRMS data file generated by SYMFLUENCE (ERA5)\n")
            f.write("precip 1\ntmax 1\ntmin 1\nrunoff 1\n####\n")
            for date, row in daily.iterrows():
                d = pd.Timestamp(date)
                f.write(f"{d.year} {d.month} {d.day} 0 0 0"
                        f" {row['precip']:.4f}"
                        f" {row['tmax']:.2f}"
                        f" {row['tmin']:.2f}"
                        f" 0.0\n")
        logger.info(f"Wrote data file: {out}")
        self._daily_forcing = daily      # cache for period length calc

    # ------------------------------------------------------------------
    # 2.  PRMS parameter file
    # ------------------------------------------------------------------
    def _generate_parameter_file(self) -> None:
        logger.info("Generating GSFLOW parameter file...")

        nmonths = 12
        nsegment = 1
        if self.distributed_gw:
            # Lumped PRMS surface (1 HRU) over an N x N MODFLOW grid; the stream
            # runs down a single column (N reaches), so off-stream cells must
            # transmit groundwater laterally to reach it (K-dependent).
            nrow = ncol = self.gw_grid_n
            ncells = nrow * ncol
            nhru = 1
            nreach = nrow            # single-column stream path
            nhrucell = ncells        # N^2 GVRs, all -> the single HRU
        else:
            nrow, ncol = 3, 3
            ncells = nrow * ncol
            nhru = ncells            # 1 HRU per MODFLOW cell (1:1 GVR mapping)
            nreach = ncells          # serpentine: 1 SFR reach per cell
            nhrucell = ncells

        # Geometry from the per-domain model-ready datastore (cached so the
        # MODFLOW package builder reuses the same values).
        self._props = self._get_catchment_properties()
        lat = self._props['lat']
        lon = self._props['lon']
        elev = self._props['elev']
        area_km2 = self._props['area_km2']
        hru_area_acres = area_km2 * 247.105 / nhru  # each HRU = 1/nhru of basin
        logger.info(
            f"GSFLOW HRU from datastore: lat={lat:.3f} lon={lon:.3f} "
            f"elev={elev:.0f} m area={area_km2:.1f} km2")

        out = self.setup_dir / 'params.dat'
        with open(out, 'w') as f:
            f.write("PRMS/GSFLOW parameter file generated by SYMFLUENCE\n")
            f.write("Version: 1.0\n")

            # ---- Dimensions ----
            f.write("** Dimensions **\n")
            dims = {
                'nhru': nhru, 'ngw': nhru, 'nssr': nhru, 'nsub': 1,
                'nsegment': nsegment, 'nmonths': nmonths, 'one': 1,
                'nobs': 1, 'ndepl': 1, 'ndeplval': 11,
                'nrain': 1, 'ntemp': 1, 'nsol': 0,
                'nhrucell': nhrucell, 'ncascade': nhru, 'ncascdgw': nhru,
                'nreach': nreach, 'ngwcell': ncells,
            }
            for name, size in dims.items():
                f.write(f"####\n{name}\n{size}\n")

            # ---- Parameters ----
            f.write("** Parameters **\n")

            def wp(name, ndim, dim_name, dim_size, dtype, values):
                f.write(f"####\n{name}\n{ndim}\n{dim_name}\n"
                        f"{dim_size}\n{dtype}\n")
                if isinstance(values, list):
                    for v in values:
                        f.write(f"{v}\n")
                else:
                    f.write(f"{values}\n")

            # -- Unit system --
            wp("precip_units", 1, "one", 1, 1, 1)  # mm
            wp("temp_units", 1, "one", 1, 1, 1)    # Celsius
            wp("elev_units", 1, "one", 1, 1, 1)    # metres

            # -- HRU geometry (all 9 HRUs identical) --
            wp("hru_area", 1, "nhru", nhru, 2,
               [f"{hru_area_acres:.1f}"] * nhru)
            wp("hru_elev", 1, "nhru", nhru, 2, [elev] * nhru)
            wp("hru_slope", 1, "nhru", nhru, 2, [0.15] * nhru)
            wp("hru_lat", 1, "nhru", nhru, 2, [lat] * nhru)
            wp("hru_lon", 1, "nhru", nhru, 2, [lon] * nhru)
            wp("hru_aspect", 1, "nhru", nhru, 2, [180.0] * nhru)
            wp("hru_type", 1, "nhru", nhru, 1, [1] * nhru)
            wp("cov_type", 1, "nhru", nhru, 1, [3] * nhru)
            wp("soil_type", 1, "nhru", nhru, 1, [2] * nhru)
            wp("hru_percent_imperv", 1, "nhru", nhru, 2, [0.0] * nhru)

            # -- Connectivity (all HRUs → segment 1) --
            wp("hru_segment", 1, "nhru", nhru, 1, [1] * nhru)
            wp("hru_subbasin", 1, "nhru", nhru, 1, [1] * nhru)
            wp("hru_deplcrv", 1, "nhru", nhru, 1, [1] * nhru)

            # -- Canopy interception --
            wp("covden_sum", 1, "nhru", nhru, 2, [0.5] * nhru)
            wp("covden_win", 1, "nhru", nhru, 2, [0.3] * nhru)
            wp("snow_intcp", 1, "nhru", nhru, 2, [0.05] * nhru)
            wp("srain_intcp", 1, "nhru", nhru, 2, [0.05] * nhru)
            wp("wrain_intcp", 1, "nhru", nhru, 2, [0.05] * nhru)

            # -- Snow --
            wp("rad_trncf", 1, "nhru", nhru, 2, [0.5] * nhru)
            wp("potet_sublim", 1, "nhru", nhru, 2, [0.5] * nhru)
            wp("emis_noppt", 1, "nhru", nhru, 2, [0.757] * nhru)
            wp("freeh2o_cap", 1, "nhru", nhru, 2, [0.05] * nhru)
            wp("cecn_coef", 1, "nmonths", nmonths, 2, [5.0] * nmonths)
            wp("snarea_curve", 1, "ndeplval", 11, 2,
               [0.05, 0.24, 0.40, 0.53, 0.65, 0.73,
                0.80, 0.87, 0.93, 0.97, 1.00])
            wp("snarea_thresh", 1, "nhru", nhru, 2, [50.0] * nhru)

            # -- Transpiration --
            wp("transp_beg", 1, "nhru", nhru, 1, [4] * nhru)
            wp("transp_end", 1, "nhru", nhru, 1, [10] * nhru)
            wp("transp_tmax", 1, "nhru", nhru, 2, [1.0] * nhru)

            # -- PET (Jensen-Haise) --
            wp("jh_coef", 1, "nmonths", nmonths, 2, [0.014] * nmonths)
            wp("jh_coef_hru", 1, "nhru", nhru, 2, [13.0] * nhru)

            # -- Solar radiation --
            wp("dday_slope", 1, "nmonths", nmonths, 2,
               [0.30, 0.32, 0.34, 0.36, 0.38, 0.38,
                0.36, 0.35, 0.33, 0.31, 0.30, 0.29])
            wp("dday_intcp", 1, "nmonths", nmonths, 2,
               [-20.0, -18.0, -14.0, -10.0, -6.0, -3.0,
                -2.0, -3.0, -6.0, -12.0, -16.0, -20.0])

            # -- Surface runoff --
            wp("carea_max", 1, "nhru", nhru, 2, [0.6] * nhru)
            wp("smidx_coef", 1, "nhru", nhru, 2, [0.01] * nhru)
            wp("smidx_exp", 1, "nhru", nhru, 2, [0.3] * nhru)
            wp("imperv_stor_max", 1, "nhru", nhru, 2, [0.05] * nhru)

            # -- Soil zone --
            wp("soil_moist_max", 1, "nhru", nhru, 2, [6.0] * nhru)
            # GSFLOW5 uses fractions of capacity, not PRMS4 depths.
            wp("soil_rechr_max_frac", 1, "nhru", nhru, 2, [1.0 / 3.0] * nhru)
            wp("soil_moist_init_frac", 1, "nhru", nhru, 2, [0.5] * nhru)
            wp("soil_rechr_init_frac", 1, "nhru", nhru, 2, [0.5] * nhru)
            wp("soil2gw_max", 1, "nhru", nhru, 2, [0.5] * nhru)
            wp("ssr2gw_rate", 1, "nssr", nhru, 2, [0.1] * nhru)
            wp("ssr2gw_exp", 1, "nssr", nhru, 2, [1.0] * nhru)
            wp("ssrcoef_lin", 1, "nssr", nhru, 2, [0.1] * nhru)
            wp("ssrcoef_sq", 1, "nssr", nhru, 2, [0.1] * nhru)
            wp("slowcoef_lin", 1, "nhru", nhru, 2, [0.015] * nhru)
            wp("slowcoef_sq", 1, "nhru", nhru, 2, [0.1] * nhru)
            wp("pref_flow_den", 1, "nhru", nhru, 2, [0.0] * nhru)
            wp("fastcoef_lin", 1, "nhru", nhru, 2, [0.1] * nhru)
            wp("fastcoef_sq", 1, "nhru", nhru, 2, [0.1] * nhru)

            # -- Groundwater --
            wp("gwflow_coef", 1, "ngw", nhru, 2, [0.015] * nhru)
            wp("gwstor_init", 1, "ngw", nhru, 2, [2.0] * nhru)
            wp("gwstor_min", 1, "ngw", nhru, 2, [0.0] * nhru)
            wp("gwsink_coef", 1, "ngw", nhru, 2, [0.0] * nhru)
            wp("gw_seep_coef", 1, "ngw", nhru, 2, [0.02] * nhru)

            # -- Temp / precip adjustments --
            wp("tmax_allrain", 1, "nmonths", nmonths, 2, [3.3] * nmonths)
            wp("tmax_allsnow", 1, "nmonths", nmonths, 2, [0.0] * nmonths)
            wp("adjmix_rain", 1, "nmonths", nmonths, 2, [1.0] * nmonths)
            wp("tmax_adj", 1, "nhru", nhru, 2, [0.0] * nhru)
            wp("tmin_adj", 1, "nhru", nhru, 2, [0.0] * nhru)
            wp("rain_adj", 1, "nmonths", nmonths, 2, [1.0] * nmonths)
            wp("snow_adj", 1, "nmonths", nmonths, 2, [1.0] * nmonths)

            # -- Station parameters --
            wp("tsta_elev", 1, "ntemp", 1, 2, elev)
            wp("psta_elev", 1, "nrain", 1, 2, elev)
            wp("hru_tsta", 1, "nhru", nhru, 1, [1] * nhru)
            wp("hru_psta", 1, "nhru", nhru, 1, [1] * nhru)
            wp("basin_tsta", 1, "one", 1, 1, 1)

            # -- Segment routing --
            wp("tosegment", 1, "nsegment", nsegment, 1, 0)
            wp("seg_length", 1, "nsegment", nsegment, 2, 47000.0)
            wp("K_coef", 1, "nsegment", nsegment, 2, 0.1)
            wp("x_coef", 1, "nsegment", nsegment, 2, 0.2)
            wp("obsin_segment", 1, "nsegment", nsegment, 1, 0)

            # -- GVR mapping --
            if self.distributed_gw:
                # Every cell is fully covered by the single surface HRU.
                # gvr_cell_pct is a fraction of CELL area, not HRU area.
                wp("gvr_hru_id", 1, "nhrucell", nhrucell, 1, [1] * nhrucell)
                wp("gvr_cell_id", 1, "nhrucell", nhrucell, 1,
                   list(range(1, ncells + 1)))
                wp("gvr_cell_pct", 1, "nhrucell", nhrucell, 2,
                   ["1.00000000"] * nhrucell)
                # The complementary mapping partitions the HRU across cells.
                # Omitting it gives all-zero defaults and a 0/0 normalization
                # in native prms2mfinit, contaminating storage and exchange.
                wp("gvr_hru_pct", 1, "nhrucell", nhrucell, 2,
                   [1.0 / ncells] * nhrucell)
            else:
                # 1:1 — each HRU maps to one cell.
                wp("gvr_hru_id", 1, "nhrucell", nhrucell, 1,
                   list(range(1, nhru + 1)))
                wp("gvr_cell_id", 1, "nhrucell", nhrucell, 1,
                   list(range(1, ncells + 1)))
                wp("gvr_cell_pct", 1, "nhrucell", nhrucell, 2,
                   [f"{1.0:.6f}"] * nhrucell)

            # -- Cascade routing (each HRU → segment 1) --
            wp("hru_up_id", 1, "ncascade", nhru, 1,
               list(range(1, nhru + 1)))
            wp("hru_down_id", 1, "ncascade", nhru, 1, [0] * nhru)
            wp("hru_pct_up", 1, "ncascade", nhru, 2, [1.0] * nhru)
            wp("hru_strmseg_down_id", 1, "ncascade", nhru, 1,
               [1] * nhru)
            wp("cascade_tol", 1, "one", 1, 2, 5.0)
            wp("cascade_flg", 1, "one", 1, 1, 0)

            # GW cascade: each GW reservoir → segment 1
            wp("gw_up_id", 1, "ncascdgw", nhru, 1,
               list(range(1, nhru + 1)))
            wp("gw_down_id", 1, "ncascdgw", nhru, 1, [0] * nhru)
            wp("gw_pct_up", 1, "ncascdgw", nhru, 2, [1.0] * nhru)
            wp("gw_strmseg_down_id", 1, "ncascdgw", nhru, 1,
               [1] * nhru)

        logger.info(f"Wrote parameter file: {out}")

    # ------------------------------------------------------------------
    # 3.  MODFLOW-NWT packages (lumped 1-cell grid)
    # ------------------------------------------------------------------
    def _generate_modflow_packages(self, start_date: datetime,
                                    end_date: datetime) -> None:
        logger.info("Generating MODFLOW-NWT packages...")

        ndays = (end_date - start_date).days
        # Grid geometry from the datastore (cached by _generate_parameter_file;
        # fall back to a fresh read if packages are generated standalone).
        props = getattr(self, '_props', None) or self._get_catchment_properties()
        area_m2 = props['area_km2'] * 1e6
        nrow = ncol = self.gw_grid_n if self.distributed_gw else 3
        cell_size = float(np.sqrt(area_m2) / nrow)   # m
        aquifer_thick = max(100.0, float(props.get('relief', 200.0)))

        # Per-cell land-surface elevations. Distributed mode samples the DEM onto
        # the grid (the head-gradient source that makes K matter); lumped mode
        # uses a single mean elevation. top_grid is an (nrow x ncol) array.
        top_grid = None
        if self.distributed_gw:
            top_grid = self._sample_dem_to_grid(nrow, ncol)
        if top_grid is None:
            top_grid = np.full((nrow, ncol), float(props['elev']))
        top_elev = float(np.mean(top_grid))          # representative surface
        bot_grid = top_grid - aquifer_thick          # aquifer base per cell
        bot_elev = float(np.mean(bot_grid))
        init_head = top_elev - 5.0   # shallow water table (5 m below surface)

        def _mf_array(grid, fmt='%.2f'):
            """Format a 2D array as a MODFLOW INTERNAL block (free format)."""
            lines = ["  INTERNAL  1.0  (FREE)  0"]
            for r in range(grid.shape[0]):
                lines.append("    " + " ".join(fmt % v for v in grid[r]))
            return "\n".join(lines) + "\n"
        hk = 1.0             # m/d  (calibrated as K)
        sy = 0.15            # (-)  (calibrated as SY)
        stream_elev = top_elev - 15.0  # stream bed 15 m below land surface
        stream_slope = 0.0001  # gentle slope; total drop ≈ 12.6 m over 9 reaches
        stream_width = 5.0
        strmbed_k = 1.0      # streambed hydraulic conductivity (m/d)

        mf = self.modflow_dir

        # -- NAM  (paths relative to setup_dir = CWD when GSFLOW runs) --
        nam = mf / 'modflow.nam'
        with open(nam, 'w') as f:
            f.write(f"LIST  26  {self._output_relpath}/modflow/gsflow_mf.list\n")
            f.write("BAS6   8  modflow/bow.bas\n")
            f.write("DIS   11  modflow/bow.dis\n")
            f.write("UPW   35  modflow/bow.upw\n")
            f.write("NWT   34  modflow/bow.nwt\n")
            f.write("UZF   14  modflow/bow.uzf\n")
            f.write("SFR   15  modflow/bow.sfr\n")
            f.write("OC     9  modflow/bow.oc\n")
            f.write(f"DATA(BINARY)  58  {self._output_relpath}/modflow/heads.out\n")
        logger.info(f"  NAM: {nam}")

        # -- DIS  (1 layer, 3x3 grid; 2 stress periods) --
        dis = mf / 'bow.dis'
        with open(dis, 'w') as f:
            f.write("# GSFLOW MODFLOW-NWT Discretization (3x3 lumped)\n")
            f.write(f"         1         {nrow}         {ncol}         2"
                    f"         4         2\n")
            f.write(" 0\n")
            f.write(f"  CONSTANT  {cell_size:.1f}\n")
            f.write(f"  CONSTANT  {cell_size:.1f}\n")
            if self.distributed_gw:
                f.write(_mf_array(top_grid))     # per-cell land surface (DEM)
                f.write(_mf_array(bot_grid))     # per-cell aquifer base
            else:
                f.write(f"  CONSTANT  {top_elev:.1f}\n")
                f.write(f"  CONSTANT  {bot_elev:.1f}\n")
            f.write("    1.0  1  1.0  SS\n")
            f.write(f"    {float(ndays):.1f}  {ndays}  1.0  TR\n")
        logger.info(f"  DIS: {dis}  ({nrow}x{ncol}, {ndays} days)")

        # -- BAS --
        bas = mf / 'bow.bas'
        with open(bas, 'w') as f:
            f.write("# GSFLOW Basic Package (lumped)\n")
            f.write("FREE\n")
            # IBOUND layer 1
            f.write("  CONSTANT  1\n")
            # HNOFLO (head assigned to no-flow cells)
            f.write("  -999.99\n")
            # STRT (starting heads) layer 1 — start ~5 m below the local surface
            # in distributed mode so the water table follows topography.
            if self.distributed_gw:
                f.write(_mf_array(top_grid - 5.0))
            else:
                f.write(f"  CONSTANT  {init_head:.1f}\n")

        # -- UPW  (K, SY — updated during calibration) --
        upw = mf / 'bow.upw'
        with open(upw, 'w') as f:
            f.write("# GSFLOW UPW Package (lumped)\n")
            # ILPFCB  HDRY  NPLPF  <HDRYFLG>
            f.write("  0  -9999.0  0  1\n")
            # LAYTYP (1 = convertible)
            f.write("  1\n")
            # LAYAVG
            f.write("  0\n")
            # CHANI
            f.write("  1.0\n")
            # LAYVKA
            f.write("  0\n")
            # LAYWET
            f.write("  0\n")
            # Layer 1 arrays: HK, VKA, SS, SY
            f.write(f"  CONSTANT  {hk:.6e}\n")   # HK
            f.write(f"  CONSTANT  {hk:.6e}\n")   # VKA
            f.write("  CONSTANT  1.0e-05\n")     # SS
            f.write(f"  CONSTANT  {sy:.6e}\n")    # SY
        logger.info(f"  UPW: {upw}  (K={hk}, SY={sy})")

        # -- NWT --
        nwt = mf / 'bow.nwt'
        with open(nwt, 'w') as f:
            f.write("# GSFLOW NWT Solver\n")
            # HEADTOL FLUXTOL MAXITEROUT THICKFACT LINMETH IPRNWT
            # IBOTAV OPTIONS
            # SPECIFIED requires nonlinear damping/backtracking coefficients
            # on the first record. Omitting them silently disables safeguards
            # and can produce NaN heads during steady-state initialization.
            f.write("1.0E-4  1.0  500  1.0E-5  1  0  1  COMPLEX\n")

        # -- UZF  (unsaturated zone flow — required for GSFLOW coupling) --
        uzf = mf / 'bow.uzf'
        with open(uzf, 'w') as f:
            f.write("# GSFLOW UZF Package (lumped)\n")
            # NUZTOP IUZFOPT IRUNFLG IETFLG IUZFCB1 IUZFCB2
            #   NTRAIL2 NSETS2 NUZGAG SURFDEP
            # NUZTOP=1 (recharge to top layer)
            # IUZFOPT=1 (specify VKS for unsaturated vertical K)
            # IRUNFLG=1 (rejected recharge → SFR)
            # IETFLG=0  (no ET from UZF)
            f.write("  1  1  1  0  0  0  15  20  0  1.0\n")
            # IUZFBND (active UZF cells: 1=active)
            f.write("  CONSTANT  1\n")
            # IRUNBND (SFR segment receiving rejected recharge)
            f.write("  CONSTANT  1\n")
            # VKS (vertical hydraulic conductivity of unsaturated zone)
            f.write(f"  CONSTANT  {hk:.6e}\n")
            # EPS (Brooks-Corey epsilon)
            f.write("  CONSTANT  3.5\n")
            # THTS (saturated water content)
            thts = self._get_config_value(
                lambda: self.config.model.gsflow.uzf_saturated_water_content, default=0.30)
            if thts <= sy:
                raise ValueError('GSFLOW saturated water content must exceed specific yield')
            f.write(f"  CONSTANT  {thts:.8g}\n")
            # Stress period 1 (SS)
            ncells = nrow * ncol
            f.write(f"  {ncells}\n")               # NUZF1 (read FINF)
            recharge = self._get_config_value(
                lambda: self.config.model.gsflow.initial_recharge, default=1.e-8)
            f.write(f"  CONSTANT  {recharge:.10g}\n")  # FINF, m/day
            # Stress period 2 (TR) — reuse previous
            f.write("  -1\n")                       # NUZF1 < 0 = reuse
        logger.info(f"  UZF: {uzf}")

        # -- SFR  (1 segment) --
        # Distributed mode: a single stream down the lowest (valley) column so
        # off-stream cells must transmit groundwater laterally to it. Lumped
        # mode: serpentine path visiting every cell (no lateral gradient).
        if self.distributed_gw:
            # Valley column = column with the lowest mean surface elevation.
            col_means = top_grid.mean(axis=0)
            jcol = int(np.argmin(col_means)) + 1  # 1-based
            # Reaches ordered downstream = descending land-surface elevation.
            rows_sorted = sorted(range(1, nrow + 1),
                                 key=lambda r: -top_grid[r - 1, jcol - 1])
            reach_cells = [(r, jcol) for r in rows_sorted]
            # Streambed top: 2 m below local surface, forced strictly decreasing.
            strtops = []
            prev = None
            for (r, c) in reach_cells:
                s = float(top_grid[r - 1, c - 1]) - 2.0
                if prev is not None and s >= prev:
                    s = prev - 0.5
                strtops.append(s)
                prev = s
        else:
            reach_cells = []
            for row in range(1, nrow + 1):
                cols = range(1, ncol + 1) if row % 2 == 1 else range(ncol, 0, -1)
                for col in cols:
                    reach_cells.append((row, col))
            strtops = [stream_elev + stream_slope * (len(reach_cells) - i) * cell_size
                       for i in range(1, len(reach_cells) + 1)]

        nsfr_reach = len(reach_cells)
        sfr = mf / 'bow.sfr'
        with open(sfr, 'w') as f:
            f.write(f"# GSFLOW SFR Package ({nrow}x{ncol} grid — {nsfr_reach} reaches)\n")
            f.write("OPTIONS\nREACHINPUT\nEND\n")
            f.write(f"    {nsfr_reach}    1    0    0  86400.0  0.0001"
                    f"  -1  0  3  10  1  40  0\n")
            rchlen = cell_size
            for ireach, ((row, col), strtop) in enumerate(zip(reach_cells, strtops), 1):
                slope_r = max(stream_slope, 1.0e-4)
                # KRCH IRCH JRCH ISEG IREACH RCHLEN STRTOP SLOPE
                #   STRTHICK STRMBD_K THTS THTI EPS UHC
                f.write(f"    1    {row}    {col}    1    {ireach}"
                        f"  {rchlen:.1f}  {strtop:.1f}"
                        f"  {slope_r:.6f}  1.0"
                        f"  {strmbed_k:.1f}  0.30  0.20  3.5  0.3\n")
            # Stress period 1 (SS): segment data
            f.write("    1    0    0\n")
            f.write("    1    1    0    0  0.5  0.0  0.0  0.0"
                    "  0.04  0.0  0.0  0.0  0.0  0.0\n")
            f.write(f"  {stream_width:.1f}\n  {stream_width:.1f}\n")
            # Stress period 2 (TR) — reuse
            f.write("   -1    0    0\n")
        logger.info(f"  SFR: {sfr}  ({nsfr_reach} reaches)")

        # -- OC --
        oc = mf / 'bow.oc'
        with open(oc, 'w') as f:
            f.write("# GSFLOW Output Control\n")
            f.write("HEAD PRINT FORMAT  0\n")
            f.write("HEAD SAVE UNIT 58\n")
            f.write("PERIOD  1 STEP  1\n")
            f.write("  SAVE HEAD\n  SAVE BUDGET\n")
            f.write(f"PERIOD  2 STEP  {ndays}\n")
            f.write("  SAVE HEAD\n  SAVE BUDGET\n")

        logger.info("MODFLOW-NWT packages complete")

    # ------------------------------------------------------------------
    # 4.  GSFLOW control file
    # ------------------------------------------------------------------
    def _generate_control_file(self, start_date: datetime,
                                end_date: datetime,
                                mode: str) -> None:
        logger.info("Generating GSFLOW control file...")

        control_name = self._get_config_value(
            lambda: self.config.model.gsflow.control_file,
            default='control.dat', dict_key='GSFLOW_CONTROL_FILE')

        out = self.setup_dir / control_name

        with open(out, 'w') as f:
            f.write("GSFLOW Control File generated by SYMFLUENCE\n")

            def wc(name, nvals, dtype, values):
                """Write one control-file parameter block."""
                f.write(f"####\n{name}\n{nvals}\n{dtype}\n")
                if isinstance(values, list):
                    for v in values:
                        f.write(f"{v}\n")
                else:
                    f.write(f"{values}\n")

            # -- Model mode --
            if mode == 'COUPLED':
                wc("model_mode", 1, 4, "GSFLOW5")
            elif mode == 'PRMS':
                wc("model_mode", 1, 4, "PRMS5")
            else:
                wc("model_mode", 1, 4, "MODFLOW")

            # -- Time --
            wc("modflow_time_zero", 6, 1,
               [start_date.year, start_date.month, start_date.day,
                0, 0, 0])
            wc("start_time", 6, 1,
               [start_date.year, start_date.month, start_date.day,
                0, 0, 0])
            wc("end_time", 6, 1,
               [end_date.year, end_date.month, end_date.day, 0, 0, 0])

            # -- File paths (relative to setup_dir = CWD) --
            data_relpath = os.path.relpath(
                self.forcing_dir / 'data.dat', self.setup_dir)
            wc("data_file", 1, 4, data_relpath)
            wc("param_file", 1, 4, "params.dat")
            wc("modflow_name", 1, 4, "modflow/modflow.nam")

            # -- Module selections --
            wc("precip_module", 1, 4, "precip_1sta")
            wc("temp_module", 1, 4, "temp_1sta")
            wc("solrad_module", 1, 4, "ddsolrad")
            wc("et_module", 1, 4, "potet_jh")
            wc("srunoff_module", 1, 4, "srunoff_smidx")
            wc("strmflow_module", 1, 4, "strmflow")
            wc("soilzone_module", 1, 4, "soilzone")

            # -- Output (relative to setup_dir = CWD → simulations dir) --
            wc("gsflow_output_file", 1, 4,
               f"{self._output_relpath}/gsflow.out")
            wc("model_output_file", 1, 4,
               f"{self._output_relpath}/prms/gsflow_prms.out")
            wc("csv_output_file", 1, 4,
               f"{self._output_relpath}/gsflow.csv")
            wc("gsf_rpt", 1, 1, 1)
            wc("rpt_days", 1, 1, 7)

            # -- Stat vars (streamflow) --
            wc("statsON_OFF", 1, 1, 1)
            wc("nstatVars", 1, 1, 2)
            wc("stat_var_file", 1, 4,
               f"{self._output_relpath}/statvar.dat")
            wc("statVar_names", 2, 4, ["basin_cfs", "basin_actet"])
            wc("statVar_element", 2, 4, [1, 1])

            # -- Flags --
            wc("print_debug", 1, 1, 0)
            wc("parameter_check_flag", 1, 1, 0)
            wc("init_vars_from_file", 1, 1, 0)
            wc("save_vars_to_file", 1, 1, 0)
            wc("subbasin_flag", 1, 1, 1)
            wc("cascadegw_flag", 1, 1, 1)
            wc("soilzone_aet_flag", 1, 1, 1)

        logger.info(f"Wrote GSFLOW control file: {out}")
