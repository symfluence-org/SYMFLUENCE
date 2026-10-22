# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD Pre-Processor.

Generates a complete WATFLOOD/CHARM input file suite from ERA5 forcing
for a lumped single-cell basin:
  - Watershed definition  (_shd.r2c)
  - Parameter file        (.par)
  - Event files           (.evt)  — one per month, chained
  - Forcing files         (.rag / .tag)  — one per month
  - Output spec           (wfo_spec.txt)
  - Streamflow obs        (_str.tb0)
  - Directory structure   (basin/, event/, raing/, tempg/, strfw/, results/, debug/)
"""
from __future__ import annotations

import calendar
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor

logger = logging.getLogger(__name__)


class WATFLOODPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Pre-processor for WATFLOOD model setup (lumped 1-cell basin)."""

    MODEL_NAME = "WATFLOOD"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.watflood_dir = self.project_dir / 'WATFLOOD_input'
        self.settings_dir = self.watflood_dir / 'settings'
        self._catch_props: Optional[dict] = None   # cached catchment properties

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run_preprocessing(self) -> bool:
        """Generate all WATFLOOD input files from scratch."""
        try:
            # Create directory tree
            for d in ('basin', 'event', 'raing', 'tempg', 'strfw',
                      'results', 'debug', 'moist', 'snow1', 'radcl', 'tempr'):
                (self.settings_dir / d).mkdir(parents=True, exist_ok=True)

            start, end = self._get_simulation_dates()
            logger.info(f"WATFLOOD preprocessing: {start:%Y-%m-%d} to {end:%Y-%m-%d}")

            # Load ERA5 forcing
            hourly = self._load_era5_forcing(start, end)

            # 1. Watershed definition
            self._generate_shd_file()

            # 2. Parameter file
            self._generate_par_file()

            # 2b. Snow-cover depletion curve (snwflg=y requires it; rdsdc).
            self._generate_sdc_file()

            # CHARM's read_shd_ef opens the shd/par by bare name from the run cwd
            # (settings/), not the basin/ subdir referenced in the event file, so
            # mirror them to the settings root (matching a working WATFLOOD layout).
            import shutil
            for name in ('bow_shd.r2c', 'bow.par', 'bow.sdc'):
                srcf = self.settings_dir / 'basin' / name
                if srcf.exists():
                    shutil.copy2(srcf, self.settings_dir / name)

            # 3. Monthly forcing + event files
            self._generate_monthly_files(hourly, start, end)

            # 4. Output spec
            self._generate_wfo_spec()

            # 5. Observation streamflow (for WATFLOOD stats)
            self._generate_streamflow_tb0(start, end)

            logger.info(f"WATFLOOD preprocessing complete: {self.settings_dir}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"WATFLOOD preprocessing failed: {e}", exc_info=True)
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
    # Catchment geometry (from the model-ready datastore)
    # ------------------------------------------------------------------
    def _get_catchment_properties(self) -> dict:
        """Area / centroid lat-lon / mean elevation / projected origin from the
        catchment shapefile + DEM (replaces the Bow-at-Banff hardcoded values).
        Cached on first call."""
        if self._catch_props is not None:
            return self._catch_props
        props = {'area_km2': 2210.0, 'lat': 65.0, 'lon': -19.0, 'elev': 800.0,
                 'x_origin': 0.0, 'y_origin': 0.0, 'epsg': 32627}
        try:
            import geopandas as gpd
            catch = self.get_catchment_path()
            if catch and catch.exists():
                gdf = gpd.read_file(catch)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                if gdf.crs.is_geographic:
                    cpt = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326).iloc[0]
                else:
                    cpt = gdf.geometry.centroid.to_crs(epsg=4326).iloc[0]
                props['lon'], props['lat'] = float(cpt.x), float(cpt.y)
                utm = int((props['lon'] + 180) / 6) + 1
                props['epsg'] = (32600 if props['lat'] >= 0 else 32700) + utm
                gproj = gdf.to_crs(epsg=props['epsg'])
                props['area_km2'] = float(gproj.geometry.area.sum()) / 1e6
                pc = gproj.geometry.centroid.iloc[0]
                # 3x3 grid of 5 km cells centred on the catchment.
                props['x_origin'] = float(pc.x) - 1.5 * 5000.0
                props['y_origin'] = float(pc.y) - 1.5 * 5000.0
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"WATFLOOD catchment properties failed: {e}", exc_info=True)
        props['elev'] = self._mean_dem_elev(props['elev'])
        self._catch_props = props
        return props

    def _mean_dem_elev(self, default: float) -> float:
        """Mean catchment elevation (m) from the model-ready DEM."""
        try:
            import numpy as np
            import rasterio
            dd = self.project_dir / 'attributes' / 'elevation' / 'dem'
            dems = sorted(dd.glob('*_elv.tif')) if dd.exists() else []
            if dems:
                with rasterio.open(dems[0]) as src:
                    a = src.read(1).astype('float64'); nd = src.nodata
                m = np.isfinite(a) & (a > -100.0)
                if nd is not None:
                    m &= a != nd
                if m.any():
                    return float(a[m].mean())
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"WATFLOOD DEM elevation failed: {e}", exc_info=True)
        return default

    # ------------------------------------------------------------------
    # ERA5 loading
    # ------------------------------------------------------------------
    def _load_era5_forcing(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load ERA5 basin-averaged forcing → hourly P (mm) and T (°C)."""
        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            raise FileNotFoundError(f"Forcing not found: {forcing_path}")

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            raise FileNotFoundError(f"No NetCDF files in {forcing_path}")

        logger.info(f"Loading forcing ({len(forcing_files)} files)")
        # open_canonical_forcing guarantees the canonical CFIF vocabulary —
        # air_temperature (K) and precipitation_flux (kg m-2 s-1 == mm s-1),
        # renaming aliases like pptrate/airtemp — and records the real timestep
        # on ds.attrs so precip integrates over the ACTUAL step (e.g. 3-hourly
        # CARRA) rather than a hardcoded hour.
        from symfluence.core.modeling.model_ready.forcing_reader import (
            forcing_timestep_seconds,
            open_canonical_forcing,
        )
        ds = open_canonical_forcing(forcing_files)
        ds = ds.sel(time=slice(str(start), str(end)))

        airtemp = ds['air_temperature'].values.squeeze()      # K
        pptrate = ds['precipitation_flux'].values.squeeze()   # mm/s
        dt_seconds = forcing_timestep_seconds(ds)
        times = pd.DatetimeIndex(ds['time'].values)

        # Resample to a strict hourly series WATFLOOD expects: temperature
        # interpolated, precip distributed evenly across each forcing interval.
        raw = pd.DataFrame({
            'temp_C': airtemp - 273.15,
            'precip_mm': pptrate * dt_seconds,   # mm per forcing timestep
        }, index=times)
        ds.close()
        step_h = max(int(round(dt_seconds / 3600.0)), 1)
        if step_h > 1:
            hourly = raw.resample('1h').ffill()
            hourly['temp_C'] = raw['temp_C'].resample('1h').interpolate()
            # spread each interval's precip evenly over its hours
            hourly['precip_mm'] = raw['precip_mm'].resample('1h').ffill() / step_h
        else:
            hourly = raw

        logger.info(f"ERA5: {len(hourly)} hours, "
                     f"P [{hourly['precip_mm'].min():.2f}–{hourly['precip_mm'].max():.2f}] mm/h, "
                     f"T [{hourly['temp_C'].min():.1f}–{hourly['temp_C'].max():.1f}] °C")
        return hourly

    # ------------------------------------------------------------------
    # 1. Watershed definition  (_shd.r2c)
    # ------------------------------------------------------------------
    def _generate_shd_file(self) -> None:
        """Generate a lumped watershed definition in r2c format.

        Uses a 3x3 grid with only the center cell (row=2, col=2) active.
        Data is written as 2D grid blocks matching the standard EnSim r2c
        format that CHARM expects:
          rank, next, DA, bankfull, slope, elevation, channel_length,
          IAK, int_slope, chnl, reach, then one grid per land class.
        """
        props = self._get_catchment_properties()
        area_km2 = props['area_km2']
        cell_m = 5000.0   # 5 km cells
        elev = props['elev']
        x_origin = props['x_origin']
        y_origin = props['y_origin']
        da = area_km2 / ((cell_m / 1000.0) ** 2)  # drainage area in grid units
        bankfull = 20.0
        slope = 0.005
        ch_len = cell_m
        nc = 3  # grid dimension

        def _grid_line(vals):
            """Format one row of a 3x3 grid."""
            return ' '.join(f'{v:5d}' for v in vals) + '\n'

        def _grid_line_f(vals, fmt='.7E'):
            """Format one row of a 3x3 float grid."""
            return ' '.join(f' {v:{fmt}}' for v in vals) + ' \n'

        def _grid_line_fx(vals, fmt='10.3f'):
            """Format one row of a 3x3 float grid (fixed)."""
            return ' '.join(f'{v:{fmt}}' for v in vals) + ' \n'

        # Active cell is (row=2, col=2) in 1-indexed → index (1,1) in 0-indexed
        z3 = [0, 0, 0]

        out = self.settings_dir / 'basin' / 'bow_shd.r2c'
        with open(out, 'w') as f:
            # Header
            f.write("########################################\n")
            f.write(":FileType r2c  ASCII  EnSim 1.0         \n")
            f.write("#                                       \n")
            f.write("# DataType               2D Rect Cell   \n")
            f.write("#                                       \n")
            f.write(":Application             EnSimHydrologic\n")
            f.write(":Version                 2.1.23         \n")
            f.write(":WrittenBy          SYMFLUENCE          \n")
            f.write(":CreationDate       2026-01-01  00:00\n")
            f.write("#                                       \n")
            f.write(":SourceFileName                bow.map  \n")
            f.write(f":NominalGridSize_AL     {cell_m:.3f}\n")
            f.write(":ContourInterval           1.000\n")
            f.write(":ImperviousArea            0.000\n")
            f.write(":ClassCount                    3\n")
            f.write(":NumRiverClasses               1\n")
            f.write(":ElevConversion            1.000\n")
            f.write(":TotalNumOfGrids               2\n")
            f.write(":numGridsInBasin               2\n")
            f.write(":DebugGridNo                   1\n")
            f.write("#                                       \n")
            f.write(":Projection         CARTESIAN \n")
            f.write(":Ellipsoid          unknown   \n")
            f.write("#                                       \n")
            f.write(f":xOrigin              {x_origin:.6f}\n")
            f.write(f":yOrigin             {y_origin:.6f}\n")
            f.write("#                                       \n")
            f.write(":AttributeName 1 Rank         \n")
            f.write(":AttributeName 2 Next         \n")
            f.write(":AttributeName 3 DA           \n")
            f.write(":AttributeName 4 Bankfull     \n")
            f.write(":AttributeName 5 ChnlSlope    \n")
            f.write(":AttributeName 6 Elev         \n")
            f.write(":AttributeName 7 ChnlLength   \n")
            f.write(":AttributeName 8 IAK          \n")
            f.write(":AttributeName 9 IntSlope     \n")
            f.write(":AttributeName 10 Chnl        \n")
            f.write(":AttributeName 11 Reach       \n")
            f.write(":AttributeName 12 GridArea    \n")
            f.write(":AttributeName 13 conifer     \n")
            f.write(":AttributeName 14 water       \n")
            f.write(":AttributeName 15 impervious  \n")
            f.write("#                                       \n")
            f.write(f":xCount                       {nc}\n")
            f.write(f":yCount                       {nc}\n")
            f.write(f":xDelta                 {cell_m:.6f}\n")
            f.write(f":yDelta                 {cell_m:.6f}\n")
            f.write("#                                       \n")
            f.write(":EndHeader                              \n")

            # TWO active cells in the middle row: (2,1)=upstream (rank 1) drains
            # east into (2,2)=outlet (rank 2). CHARM cannot run a single-cell
            # watershed -- it computes xxx(naa/2), which is xxx(0) when naa=1
            # (process_temp.f) -> out-of-bounds. naa=2 gives naa/2=1.
            z3f = [0.0, 0.0, 0.0]

            def _row3i(up, out):
                return _grid_line([up, out, 0])

            def _row3f(up, out, fmt='10.3f'):
                return _grid_line_fx([up, out, 0.0], fmt)

            # 1. Rank: upstream=1, outlet=2
            f.write(_grid_line(z3)); f.write(_row3i(1, 2)); f.write(_grid_line(z3))
            # 2. Next: upstream -> rank 2; outlet -> 0
            f.write(_grid_line(z3)); f.write(_row3i(2, 0)); f.write(_grid_line(z3))
            # 3. DA (grid units): upstream half, outlet accumulates full
            f.write(_grid_line_f(z3f))
            f.write(_grid_line_f([da / 2.0, da, 0.0]))
            f.write(_grid_line_f(z3f))
            # 4. Bankfull
            f.write(_row3f(0, 0)); f.write(_row3f(bankfull, bankfull)); f.write(_row3f(0, 0))
            # 5. Channel slope
            f.write(_row3f(0, 0, '10.7f')); f.write(_row3f(slope, slope, '10.7f')); f.write(_row3f(0, 0, '10.7f'))
            # 6. Elevation
            f.write(_row3f(0, 0)); f.write(_row3f(elev, elev)); f.write(_row3f(0, 0))
            # 7. Channel length
            f.write(_row3f(0, 0)); f.write(_row3f(ch_len, ch_len)); f.write(_row3f(0, 0))
            # 8. IAK (interflow active key)
            f.write(_grid_line(z3)); f.write(_row3i(1, 1)); f.write(_grid_line(z3))
            # 9. Internal slope
            f.write(_row3f(0, 0, '10.7f')); f.write(_row3f(slope, slope, '10.7f')); f.write(_row3f(0, 0, '10.7f'))
            # 10. Channel class (1=river)
            f.write(_grid_line(z3)); f.write(_row3i(1, 1)); f.write(_grid_line(z3))
            # 11. Reach (0=no reach)
            for _ in range(nc):
                f.write(_grid_line(z3))
            # 12. GridArea (m^2) on both active cells
            cell_area = cell_m * cell_m
            f.write(_grid_line_f(z3f))
            f.write(_grid_line_f([cell_area, cell_area, 0.0]))
            f.write(_grid_line_f(z3f))
            # 13-15. Land-class fractions (par's 3 classes): conifer=1, rest=0
            for frac in (1.0, 0.0, 0.0):
                f.write(_row3f(0, 0)); f.write(_row3f(frac, frac)); f.write(_row3f(0, 0))

        logger.info(f"Wrote watershed file: {out}")

    # ------------------------------------------------------------------
    # 2. Parameter file (.par)
    # ------------------------------------------------------------------
    def _generate_par_file(self) -> None:
        """Generate WATFLOOD .par file for lumped basin (3 classes minimum).

        Format matches CHARM's read_par_parser.f (version 10.x) which uses
        colon-prefixed keywords: `:keyword, value, # comment`.
        Section markers (:GlobalParameters etc.) are found by substring match.

        WATFLOOD requires at minimum 3 land classes: land, water, impervious.
        The second-to-last class must be water (ak<0), last is impervious.
        """
        out = self.settings_dir / 'basin' / 'bow.par'
        lat = self._get_catchment_properties()['lat']

        def sv(v):
            """Format scalar value in g12.3-like notation."""
            if v == 0.0:
                return f"{0.0:12.3f}"
            elif abs(v) < 0.01 or abs(v) >= 1000:
                return f"{v:12.3E}"
            else:
                return f"{v:12.3f}"

        def cv(*vals):
            """Format comma-separated class values."""
            return ','.join(sv(v) for v in vals) + ','

        with open(out, 'w') as f:
            # ── Header (FileType + CreationDate + comments) ──
            f.write(":FileType, WatfloodParameter     10.10,# parameter file version number\n")
            from datetime import datetime
            now = datetime.now()
            f.write(f":CreationDate ,{now:%Y-%m-%d  %H:%M:%S}\n")
            f.write("# WATFLOOD parameter file generated by SYMFLUENCE\n")
            f.write("# Bow at Banff - lumped 1-cell, 1-class\n")

            # ── :GlobalParameters ──
            f.write(":GlobalParameters\n")
            f.write(f":iopt,           {0:7d},# debug level\n")
            f.write(f":itype,          {0:7d},# channel type\n")
            f.write(f":itrace,         {0:7d},# Tracer choice\n")
            f.write(f":a1,          {-999.999:10.3f},# ice cover weighting factor\n")
            f.write(f":a2,          {-999.999:10.3f},# swe correction threshold\n")
            f.write(f":a3,          {-999.999:10.3f},# error penalty coefficient\n")
            f.write(f":a4,          {-999.999:10.3f},# error penalty threshold\n")
            f.write(f":a5,          {0.983:10.3f},# API coefficient\n")
            f.write(f":a6,          {900.000:10.3f},# Minimum routing time step in seconds\n")
            f.write(f":a7,          {0.750:10.3f},# weighting - old vs. new sca value\n")
            f.write(f":a8,          {0.000:10.3f},# min temperature time offset\n")
            f.write(f":a9,          {0.500:10.3f},# max heat deficit /swe ratio\n")
            f.write(f":a10,         {1.500:10.3f},# exponent on uz discharge function\n")
            # The native reader normalizes monthly vegetation factors by the
            # maximum of IntCap and a11. A missing-value sentinel permits 0/0
            # for bare classes with zero interception, corrupting soil ET.
            f.write(f":a11,         {0.100:10.3f},# bare ground equiv. veg height for ev\n")
            f.write(f":a12,         {0.000:10.3f},# min precip rate for smearing\n")
            f.write(f":a13,         {0.000:10.3f},# \n")
            f.write(f":fmadjust,    {0.000:10.3f},# snowmelt ripening rate\n")
            f.write(f":fmalow,      {0.000:10.3f},# min melt factor multiplier\n")
            f.write(f":fmahigh,     {0.000:10.3f},# max melt factor multiplier\n")
            f.write(f":gladjust,    {0.000:10.3f},# glacier melt factor multiplier\n")
            f.write(f":rlapse,      {0.000000:10.6f},# precip lapse rate mm/m\n")
            f.write(f":tlapse,      {6.500000:10.6f},# temperature lapse rate dC/m\n")
            f.write(f":rainsnowtemp,{0.000:10.3f},# rain/snow temperature\n")
            f.write(f":radiusinflce,{0.000:10.3f},# radius of influence km\n")
            f.write(f":smoothdist,  {0.000:10.3f},# smoothing distance km\n")
            f.write(f":elvref,      {1600.000:10.3f},# reference elevation\n")
            f.write(f":flgevp2  ,   {2.000:10.3f},# 1=pan;4=Hargreaves;3=Priestley-Taylor\n")
            f.write(f":albe  ,      {1.000:10.3f},# albedo\n")
            f.write(f":tempa2,      {1.000:10.3f},# \n")
            f.write(f":tempa3,      {3.000:10.3f},# \n")
            f.write(f":tton  ,      {200.000:10.3f},# \n")
            f.write(f":lat   ,      {lat:10.3f},# latitude\n")
            f.write(f":chnl(1),     {1.000:10.3f},# manning`s n multiplier\n")
            f.write(f":chnl(2),     {1.000:10.3f},# manning`s n multiplier\n")
            f.write(f":chnl(3),     {1.000:10.3f},# manning`s n multiplier\n")
            f.write(f":chnl(4),     {1.000:10.3f},# manning`s n multiplier\n")
            f.write(f":chnl(5),     {1.000:10.3f},# manning`s n multiplier\n")
            f.write(":EndGlobalParameters\n")
            f.write("#\n")

            # ── :OptimizationSwitches ──
            f.write(":OptimizationSwitches\n")
            f.write(f":numa,  {0:7d},# PS optimization 1=yes 0=no\n")
            f.write(f":nper,  {1:7d},# opt 1=delta 0=absolute\n")
            f.write(f":kc,    {5:7d},# no of times delta halved\n")
            f.write(f":maxn,  {2000:7d},# max no of trials\n")
            f.write(f":ddsflg,{0:7d},# 0=single run  1=DDS\n")
            f.write(f":errflg,{1:7d},# 1=wMSE 2=SSE 3=wSSE 4=VOL\n")
            f.write(":EndOptimizationSwitches\n")
            f.write("#\n")

            # ── :RoutingParameters ──
            f.write(":RoutingParameters\n")
            f.write(f":RiverClasses,{1:12d}\n")
            f.write(":RiverClassName,  meander   ,\n")
            f.write(f":flz,             {sv(1.0e-4)},# lower zone coefficient\n")
            f.write(f":pwr,             {sv(2.0)},# lower zone exponent\n")
            f.write(f":r2n,             {sv(0.04)},# channel Manning`s n\n")
            f.write(f":theta,           {sv(0.50)},# wetland or bank porosity\n")
            f.write(f":kcond,           {sv(1.0)},# wetland/bank lateral conductivity\n")
            f.write(f":rlake,           {sv(0.0)},# in channel lake retardation coefficient\n")
            f.write(f":r1n,             {sv(0.10)},# overbank Manning`s n\n")
            f.write(f":aa2,             {sv(0.11)},# channel area intercept\n")
            f.write(f":aa3,             {sv(0.043)},# channel area coefficient\n")
            f.write(f":widep,           {sv(20.0)},# channel width to depth ratio\n")
            f.write(f":pool,            {sv(0.0)},# average area of zero flow pools\n")
            f.write(f":mndr,            {sv(1.20)},# meander channel length multiplier\n")
            f.write(f":aa4,             {sv(1.0)},# channel area exponent\n")
            f.write(":EndRoutingParameters\n")
            f.write("#\n")

            # ── :HydrologicalParameters (3 classes: conifer, water, impervious) ──
            # Water class: ak<0 signals water; impervious: last class
            f.write(":HydrologicalParameters\n")
            f.write(f":LandCoverClasses,{3:12d}\n")
            f.write(":ClassName       ,conifer   ,water     ,impervious,\n")
            f.write("#Vegetationparameters\n")
            f.write(f":fpet,            {cv(3.0, 1.0, 1.0)}# interception evap factor\n")
            f.write(f":ftall,           {cv(0.50, 1.0, 0.50)}# reduction in PET\n")
            f.write(f":fratio,          {cv(1.0, 1.0, 1.0)}# int. capacity multiplier\n")
            f.write("#SoilParameters\n")
            f.write(f":rec,             {cv(0.30, 0.0, 0.0)}# interflow coefficient\n")
            f.write(f":ak,              {cv(30.0, -1.0, 100.0)}# infiltration coeff\n")
            f.write(f":akfs,            {cv(20.0, -1.0, 100.0)}# infiltration coeff snow\n")
            f.write(f":retn,            {cv(100.0, 0.0, 10.0)}# upper zone retention mm\n")
            f.write(f":ak2,             {cv(0.05, 0.0, 0.01)}# recharge coeff bare\n")
            f.write(f":ak2fs,           {cv(0.01, 0.0, 0.01)}# recharge coeff snow\n")
            f.write(f":r3,              {cv(30.0, 0.0, 5.0)}# overland flow roughness\n")
            f.write(f":ds,              {cv(5.0, 0.0, 1.0)}# depression storage mm\n")
            f.write(f":dsfs,            {cv(5.0, 0.0, 1.0)}# depression storage snow\n")
            f.write(f":r3fs,            {cv(30.0, 0.0, 5.0)}# overland flow rough snow\n")
            f.write(f":r4,              {cv(10.0, 0.0, 2.0)}# overland flow rough imperv\n")
            f.write(f":flint,           {cv(1.0, 0.0, 0.0)}# interception flag\n")
            f.write(f":fcap,            {cv(0.25, 0.0, 0.0)}# not used\n")
            f.write(f":ffcap,           {cv(0.10, 0.0, 0.0)}# wilting point\n")
            f.write(f":spore,           {cv(0.40, 1.0, 0.10)}# soil porosity\n")
            f.write(":EndHydrologicalParameters\n")
            f.write("#\n")

            # ── :SnowParameters (3 classes) ──
            def cf(*vals):
                """Format comma-separated fixed-point class values."""
                return ','.join(f"{v:12.3f}" for v in vals) + ','

            def ci(*vals):
                """Format comma-separated integer class values."""
                return ','.join(f"{v:12d}" for v in vals) + ','

            f.write(":SnowParameters\n")
            f.write(f":fm,              {cf(0.090, 0.0, 0.0)}# melt factor mm/dC/hour\n")
            f.write(f":base,            {cf(-1.0, 0.0, 0.0)}# base temperature dC\n")
            f.write(f":sublim_factor,   {cf(0.0, 0.0, 0.0)}# sublimation factor ratio\n")
            f.write(f":sdcd,            {cf(25.0, 1.0, 1.0)}# swe for 100% sca\n")
            f.write(f":fmn,             {cf(0.0, 0.0, 0.0)}# -ve melt factor\n")
            f.write(f":uadj,            {cf(0.0, 0.0, 0.0)}# not used\n")
            f.write(f":tipm,            {cf(0.10, 0.10, 0.10)}# coefficient for ati\n")
            f.write(f":rho,             {cf(0.333, 0.333, 0.333)}# snow density\n")
            f.write(f":whcl,            {cf(0.035, 0.035, 0.035)}# fraction swe as water\n")
            f.write(f":alb,             {cf(0.80, 0.10, 0.30)}# albedo\n")
            f.write(f":idump,           {ci(0, 0, 0)}# receiving class for redistrib\n")
            f.write(f":snocap,          {cf(500.0, 500.0, 500.0)}# max swe before redistrib\n")
            f.write(f":nsdc,            {ci(1, 1, 1)}# no of points on scd curve\n")
            f.write(f":sdcsca,          {cf(1.0, 1.0, 1.0)}# snow covered area\n")
            f.write(":EndSnowParameters\n")
            f.write("#\n")

            # ── :InterceptionCapacityTable (3 classes) ──
            f.write(":InterceptionCapacityTable \n")
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for mon in months:
                # conifer=1.8mm, water=0.0mm, impervious=0.0mm
                f.write(f":IntCap_{mon},      {cf(1.80, 0.0, 0.0)}# interception capacity {mon.lower()} mm\n")
            f.write(":EndInterceptionCapacityTable\n")
            f.write("#\n")

            # ── :MonthlyEvapotranspirationTable (3 classes) ──
            # Monthly PET mm for Bow at Banff (Hargreaves-estimated)
            monthly_et = [0.0, 0.0, 5.0, 15.0, 40.0, 60.0,
                          70.0, 55.0, 30.0, 10.0, 0.0, 0.0]
            f.write(":MonthlyEvapotranspirationTable \n")
            for i, mon in enumerate(months):
                # Same ET for all classes (Hargreaves is spatially uniform)
                v = monthly_et[i]
                f.write(f":Montly_ET_{mon},   {v:12.1f},{v:12.1f},{v:12.1f},# evapotranspiration {mon.lower()} mm\n")
            f.write(":EndMonthlyEvapotranspirationTable\n")
            f.write("#\n")

            # ── :GlobalSnowParLimits ──
            f.write(":GlobalSnowParLimits\n")
            f.write("# snowmelt ripening rate\n")
            f.write(f":fmadjustdlt,       {sv(0.0)}\n")
            f.write(f":fmadjustlow,       {sv(0.0)}\n")
            f.write(f":fmadjusthgh,       {sv(1.0)}\n")
            f.write("# min melt factor multiplier\n")
            f.write(f":fmalowdlt,         {sv(0.0)}\n")
            f.write(f":fmalowlow,         {sv(0.0)}\n")
            f.write(f":fmalowhgh,         {sv(1.0)}\n")
            f.write("# max melt factor multiplier\n")
            f.write(f":fmahighdlt,        {sv(0.0)}\n")
            f.write(f":fmahighlow,        {sv(0.0)}\n")
            f.write(f":fmahighhgh,        {sv(1.0)}\n")
            f.write("# glacier melt factor multiplier\n")
            f.write(f":gladjustdlt,       {sv(0.0)}\n")
            f.write(f":gladjustlow,       {sv(0.0)}\n")
            f.write(f":gladjusthgh,       {sv(1.0)}\n")
            f.write(":EndGlobalSnowParLimits\n")
            f.write("#\n")

            # ── :GlobalParLimits ──
            f.write(":GlobalParLimits\n")
            f.write("# precip lapse rate\n")
            f.write(f":rlapsedlt,       {sv(0.0)}\n")
            f.write(f":rlapselow,       {sv(0.0)}\n")
            f.write(f":rlapsehgh,       {sv(0.01)}\n")
            f.write("# temperature lapse rate\n")
            f.write(f":tlapsedlt,       {sv(1.0)}\n")
            f.write(f":tlapselow,       {sv(3.0)}\n")
            f.write(f":tlapsehgh,       {sv(10.0)}\n")
            f.write("# rain/snow temperature\n")
            f.write(f":rainsnowtempdlt, {sv(0.0)}\n")
            f.write(f":rainsnowtemplow, {sv(-2.0)}\n")
            f.write(f":rainsnowtemphgh, {sv(2.0)}\n")
            f.write("# radius of influence\n")
            f.write(f":radinfldlt,      {sv(0.0)}\n")
            f.write(f":radinfllow,      {sv(0.0)}\n")
            f.write(f":radinflhgh,      {sv(100.0)}\n")
            f.write("# smoothing distance\n")
            f.write(f":smoothdisdlt,    {sv(0.0)}\n")
            f.write(f":smoothdislow,    {sv(0.0)}\n")
            f.write(f":smoothdishgh,    {sv(100.0)}\n")
            f.write(":EndGlobalParLimits\n")
            f.write("#\n")

            # ── :APILimits ──
            f.write(":APILimits\n")
            f.write(f":a5dlt,             {sv(0.1)}\n")
            f.write(f":a5low,             {sv(0.8)}\n")
            f.write(f":a5hgh,             {sv(0.999)}\n")
            f.write(":EndAPILimits\n")
            f.write("#\n")

            # ── :RoutingParLimits ──
            f.write(":RoutingParLimits\n")
            f.write(":RiverClassName,  meander   ,\n")
            f.write("# lower zone coefficient\n")
            f.write(f":flzdlt,          {sv(1.0e-5)},\n")
            f.write(f":flzlow,          {sv(1.0e-6)},\n")
            f.write(f":flzhgh,          {sv(1.0e-2)},\n")
            f.write("# lower zone exponent\n")
            f.write(f":pwrdlt,          {sv(0.5)},\n")
            f.write(f":pwrlow,          {sv(1.0)},\n")
            f.write(f":pwrhgh,          {sv(4.0)},\n")
            f.write("# channel Manning`s n\n")
            f.write(f":r2ndlt,          {sv(0.01)},\n")
            f.write(f":r2nlow,          {sv(0.01)},\n")
            f.write(f":r2nhgh,          {sv(0.30)},\n")
            f.write("# wetland or bank porosity\n")
            f.write(f":thetadlt,        {sv(0.1)},\n")
            f.write(f":thetalow,        {sv(0.1)},\n")
            f.write(f":thetahgh,        {sv(0.9)},\n")
            f.write("# wetland/bank lateral conductivity\n")
            f.write(f":kconddlt,        {sv(0.1)},\n")
            f.write(f":kcondlow,        {sv(0.01)},\n")
            f.write(f":kcondhgh,        {sv(10.0)},\n")
            f.write("# in channel lake retardation\n")
            f.write(f":rlakedlt,        {sv(0.0)},\n")
            f.write(f":rlakelow,        {sv(0.0)},\n")
            f.write(f":rlakehgh,        {sv(1.0)},\n")
            f.write(":EndRoutingParLimits\n")
            f.write("#\n")

            # ── :HydrologicalParLimits (3 classes: conifer, water, impervious) ──
            # Water & impervious limits set to no-op (dlt=-1 disables optimization)
            f.write(":HydrologicalParLimits\n")
            f.write("# infiltration coefficient bare ground\n")
            f.write(f":akdlt,           {cv(5.0, -1.0, -1.0)}\n")
            f.write(f":aklow,           {cv(1.0, -1.0, 1.0)}\n")
            f.write(f":akhgh,           {cv(100.0, -1.0, 100.0)}\n")
            f.write("# infiltration coefficient snow covered\n")
            f.write(f":akfsdlt,         {cv(5.0, -1.0, -1.0)}\n")
            f.write(f":akfslow,         {cv(1.0, -1.0, 1.0)}\n")
            f.write(f":akfshgh,         {cv(100.0, -1.0, 100.0)}\n")
            f.write("# interflow coefficient\n")
            f.write(f":recdlt,          {cv(0.1, -1.0, -1.0)}\n")
            f.write(f":reclow,          {cv(0.01, 0.0, 0.0)}\n")
            f.write(f":rechgh,          {cv(1.0, 0.0, 0.0)}\n")
            f.write("# overland flow roughness\n")
            f.write(f":r3dlt,           {cv(5.0, -1.0, -1.0)}\n")
            f.write(f":r3low,           {cv(1.0, 0.0, 1.0)}\n")
            f.write(f":r3hgh,           {cv(100.0, 0.0, 100.0)}\n")
            f.write("# interception evaporation factor\n")
            f.write(f":fpetdlt,         {cv(1.0, -1.0, -1.0)}\n")
            f.write(f":fpetlow,         {cv(0.5, 1.0, 1.0)}\n")
            f.write(f":fpethgh,         {cv(5.0, 1.0, 1.0)}\n")
            f.write("# reduction in PET for tall vegetation\n")
            f.write(f":ftalldlt,        {cv(0.1, -1.0, -1.0)}\n")
            f.write(f":ftalllow,        {cv(0.1, 1.0, 0.5)}\n")
            f.write(f":ftallhgh,        {cv(1.0, 1.0, 0.5)}\n")
            f.write("# upper zone retention\n")
            f.write(f":retndlt,         {cv(10.0, -1.0, -1.0)}\n")
            f.write(f":retnlow,         {cv(10.0, 0.0, 0.0)}\n")
            f.write(f":retnhgh,         {cv(500.0, 0.0, 0.0)}\n")
            f.write("# recharge coefficient bare ground\n")
            f.write(f":ak2dlt,          {cv(0.01, -1.0, -1.0)}\n")
            f.write(f":ak2low,          {cv(0.001, 0.0, 0.0)}\n")
            f.write(f":ak2hgh,          {cv(1.0, 0.0, 0.0)}\n")
            f.write("# recharge coefficient snow covered\n")
            f.write(f":ak2fsdlt,        {cv(0.01, -1.0, -1.0)}\n")
            f.write(f":ak2fslow,        {cv(0.001, 0.0, 0.0)}\n")
            f.write(f":ak2fshgh,        {cv(1.0, 0.0, 0.0)}\n")
            f.write(":EndHydrologicalParLimits\n")
            f.write("#\n")

            # ── :SnowParLimits (3 classes) ──
            f.write(":SnowParLimits\n")
            f.write("# melt factor\n")
            f.write(f":fmdlt,           {cv(0.01, -1.0, -1.0)}\n")
            f.write(f":fmlow,           {cv(0.01, 0.0, 0.0)}\n")
            f.write(f":fmhgh,           {cv(0.50, 0.0, 0.0)}\n")
            f.write("# base temperature\n")
            f.write(f":basedlt,         {cv(0.5, -1.0, -1.0)}\n")
            f.write(f":baselow,         {cv(-3.0, 0.0, 0.0)}\n")
            f.write(f":basehgh,         {cv(2.0, 0.0, 0.0)}\n")
            f.write("# sublimation\n")
            f.write(f":subdlt,          {cv(0.0, 0.0, 0.0)}\n")
            f.write(f":sublow,          {cv(0.0, 0.0, 0.0)}\n")
            f.write(f":subhgh,          {cv(1.0, 1.0, 1.0)}\n")
            f.write(":EndSnowParLimits\n")

        logger.info(f"Wrote parameter file: {out}")

    # ------------------------------------------------------------------
    # 3. Monthly event + forcing files
    def _generate_sdc_file(self) -> None:
        """Snow-cover depletion curve file (.sdc) — one curve per land class.

        Required by CHARM (rdsdc) whenever snwflg=y. Per class, rdsdc reads a
        header line ``nsdc idump snocap`` (number of curve points, dump flag,
        max SWE mm) then ``nsdc`` point lines ``swe_ratio  snow_covered_frac``.
        A single-point curve (nsdc=1) is degenerate: CHARM cannot interpolate it
        and divides by a zero range, raising IEEE_INVALID and aborting. Emit a
        proper monotonic depletion curve from (0,0) to (1,1) — snow-covered area
        rises quickly then saturates as SWE accumulates."""
        # (swe/swe_full ratio, snow-covered-area fraction) — concave, monotonic.
        curve = [(0.00, 0.00), (0.05, 0.30), (0.10, 0.50), (0.20, 0.66),
                 (0.30, 0.76), (0.50, 0.87), (0.70, 0.94), (1.00, 1.00)]
        out = self.settings_dir / 'basin' / 'bow.sdc'
        with open(out, 'w') as f:
            f.write("Snow Depletion Curves - SYMFLUENCE\n")
            for _ in range(3):  # one per land class (ClassCount = 3)
                f.write(f"{len(curve):5d}{0:5d}{500.0:10.1f}\n")
                for ratio, sca in curve:
                    f.write(f"{ratio:10.3f}{sca:10.3f}\n")
        logger.info(f"Wrote snow depletion curve ({len(curve)} pts/class): {out}")

    # ------------------------------------------------------------------
    def _generate_monthly_files(self, hourly: pd.DataFrame,
                                start: datetime, end: datetime) -> None:
        """Generate per-month .evt, .rag, .tag files."""
        # Forcing-grid headers MUST match the shd grid origin, or CHARM indexes
        # the gauge onto a cell outside the watershed grid and segfaults. Derive
        # the grid (km) from the catchment's projected origin, not Bow's.
        props = self._get_catchment_properties()
        x_km = int(round(props['x_origin'] / 1000.0))
        y_km = int(round(props['y_origin'] / 1000.0))
        ymin, ymax = y_km, y_km + 15  # 3 cells * 5km = 15km span
        xmin, xmax = x_km, x_km + 15

        # Build list of months
        months = pd.date_range(start, end, freq='MS')
        logger.info(f"Generating {len(months)} monthly event files")

        evt_files = []
        datestrs = []
        for i, month_start in enumerate(months):
            year = month_start.year
            month = month_start.month
            ndays = calendar.monthrange(year, month)[1]
            nhours = ndays * 24
            datestr = f"{year:04d}{month:02d}01"

            # Extract this month's hourly data
            month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta('23:59:59')
            mdata = hourly.loc[month_start:month_end]

            if len(mdata) == 0:
                logger.warning(f"No data for {datestr}, skipping")
                continue

            # Pad/trim to exact nhours
            precip_vals = mdata['precip_mm'].values[:nhours]
            temp_vals = mdata['temp_C'].values[:nhours]
            if len(precip_vals) < nhours:
                precip_vals = np.pad(precip_vals, (0, nhours - len(precip_vals)),
                                     constant_values=0.0)
                temp_vals = np.pad(temp_vals, (0, nhours - len(temp_vals)),
                                   constant_values=temp_vals[-1] if len(temp_vals) > 0 else 0.0)

            # Write .rag file (precipitation)
            rag_path = self.settings_dir / 'raing' / f'{datestr}.rag'
            with open(rag_path, 'w') as f:
                f.write(f"    2 {ymin} {ymax}  {xmin}  {xmax}\n")
                f.write(f"    1  {nhours} 1.00\n")
                # 1 station at basin centroid (y, x order per WATFLOOD convention)
                f.write(f" {y_km + 7}  {x_km + 7} SYMFLUENCE\n")
                for h in range(nhours):
                    f.write(f"    {precip_vals[h]:.2f}\n")

            # Write .tag file (temperature)
            tag_path = self.settings_dir / 'tempg' / f'{datestr}.tag'
            with open(tag_path, 'w') as f:
                f.write(f"    2 {ymin} {ymax}  {xmin}  {xmax}\n")
                f.write(f"    1  {nhours}    1\n")
                f.write(f" {y_km + 7}  {x_km + 7} SYMFLUENCE\n")
                for h in range(nhours):
                    f.write(f"    {temp_vals[h]:.2f}\n")

            # Gridded met (CHARM with modelflg=n reads these as fln(10)/temp):
            # hourly r2c frames, value in the active centre cell of the 3x3 grid.
            x0, y0 = props['x_origin'], props['y_origin']
            self._write_gridded_r2c(
                self.settings_dir / 'radcl' / f'{datestr}_met.r2c',
                'Precipitation', 'mm', precip_vals, month_start, nhours, x0, y0)
            self._write_gridded_r2c(
                self.settings_dir / 'tempr' / f'{datestr}_tem.r2c',
                'Temperature', 'dC', temp_vals, month_start, nhours, x0, y0)

            # Each monthly event is a leaf (no chained follower); the master
            # event.evt lists them all as followers (WATFLOOD's flat structure).
            evt_path = self.settings_dir / 'event' / f'{datestr}.evt'
            self._write_evt_file(evt_path, year, month, nhours,
                                 datestr, True, None)
            evt_files.append(evt_path)
            datestrs.append(datestr)

        # Master event.evt = the first month + a flat list of all the rest as
        # followers (matches a working WATFLOOD setup; a recursive 1-follower
        # chain mis-sizes CHARM's event arrays and segfaults).
        if evt_files:
            first_datestr = datestrs[0]
            master_evt = self.settings_dir / 'event' / 'event.evt'
            first_month = months[0]
            ndays_first = calendar.monthrange(first_month.year, first_month.month)[1]
            nhours_first = ndays_first * 24
            self._write_evt_file(master_evt, first_month.year, first_month.month,
                                 nhours_first, first_datestr, True, None,
                                 followers=datestrs[1:])

        logger.info(f"Wrote {len(evt_files)} monthly event/forcing files")

    def _write_gridded_r2c(self, path: Path, name: str, units: str,
                           vals, month_start, nhours: int,
                           x0: float, y0: float) -> None:
        """Write a gridded met r2c (hourly frames on the 3x3 grid). The value
        goes in the active centre cell (row 2, col 2); other cells are 0."""
        cell_m = 5000.0
        with open(path, 'w') as f:
            f.write("########################################\n")
            f.write(":FileType r2c  ASCII  EnSim 1.0\n")
            f.write("#\n# DataType               2D Rect Cell\n#\n")
            f.write(":Application             EnSimHydrologic\n")
            f.write(":Version                 2.1.23\n")
            f.write(":WrittenBy          SYMFLUENCE\n")
            f.write("#\n")
            f.write(f":Name               {name}\n")
            f.write(f":AttributeUnits     {units}\n")
            f.write(":UnitConversion            1.000\n")
            f.write("#\n:Projection         CARTESIAN\n:Ellipsoid          unknown\n#\n")
            f.write(f":xOrigin              {x0:.6f}\n")
            f.write(f":yOrigin             {y0:.6f}\n")
            f.write("#\n:xCount                       3\n:yCount                       3\n")
            f.write(f":xDelta                {cell_m:.6f}\n")
            f.write(f":yDelta                {cell_m:.6f}\n")
            f.write("#\n:EndHeader\n")
            ts = month_start
            for h in range(nhours):
                stamp = (ts + pd.Timedelta(hours=h)).strftime("%Y/%m/%d %H:00:00.000")
                v = float(vals[h])
                f.write(f":Frame {h + 1:9d} {h + 1:9d}   \"{stamp}\"\n")
                f.write("   0.000   0.000   0.000\n")
                f.write(f"   {v:.3f}   {v:.3f}   0.000\n")  # both active cells
                f.write("   0.000   0.000   0.000\n")
                f.write(":EndFrame\n")

    def _write_evt_file(self, path: Path, year: int, month: int,
                        nhours: int, datestr: str,
                        is_last: bool, next_month, followers=None) -> None:
        """Write a single .evt file.

        If ``followers`` (a list of YYYYMMDD strings) is given, this is the
        master event and lists them all under :noeventstofollow.
        """
        with open(path, 'w') as f:
            f.write("#\n")
            f.write(":filetype                     .evt\n")
            f.write(":fileversionno                9.4\n")
            f.write(f":year                         {year}\n")
            f.write(f":month                        {month:02d}\n")
            f.write(":day                          01\n")
            f.write(":hour                          0\n")
            f.write("#\n")
            f.write(":snwflg                       y\n")
            f.write(":sedflg                       n\n")
            f.write(":vapflg                       y\n")
            f.write(":smrflg                       n\n")
            f.write(":resinflg                     n\n")
            f.write(":tbcflg                       n\n")
            f.write(":resumflg                     n\n")
            # Continue from previous month (except first month)
            is_continuation = (path.name != 'event.evt' and
                               not (year == 2002 and month == 1))
            f.write(f":contflg                      {'y' if is_continuation else 'n'}\n")
            f.write(":routeflg                     n\n")
            f.write(":crseflg                      n\n")
            f.write(":ensimflg                     n\n")
            f.write(":picflg                       n\n")
            f.write(":wetflg                       n\n")
            f.write(":modelflg                     n\n")
            f.write(":shdflg                       n\n")
            f.write(":trcflg                       n\n")
            f.write(":frcflg                       n\n")
            f.write("#\n")
            f.write(":intsoilmoisture              0.25\n")
            f.write(":rainconvfactor                1.00\n")
            f.write(":eventprecipscalefactor        1.00\n")
            f.write(":precipscalefactor             0.00\n")
            f.write(":eventsnowscalefactor          0.00\n")
            f.write(":snowscalefactor               0.00\n")
            f.write(":eventtempscalefactor          0.00\n")
            f.write(":tempscalefactor               0.00\n")
            f.write("#\n")
            f.write(f":hoursraindata                 {nhours}\n")
            f.write(f":hoursflowdata                 {nhours}\n")
            f.write("#\n")
            f.write(":basinfilename                basin/bow_shd.r2c\n")
            f.write(":parfilename                  basin/bow.par\n")
            # CHARM's rdsdc reads the .sdc filename with a list-directed read,
            # which terminates at the '/' in "basin/bow.sdc" — it then tries to
            # open "basin" and aborts. The shd/par/sdc are mirrored to the run
            # cwd (settings root), so reference the .sdc by its bare name.
            f.write(":snowcoverdepletioncurve      bow.sdc\n")
            f.write("#\n")
            f.write(f":pointprecip                  raing/{datestr}.rag\n")
            f.write(f":griddedrainfile              radcl/{datestr}_met.r2c\n")
            f.write(f":pointtemps                   tempg/{datestr}.tag\n")
            f.write(f":griddedtemperaturefile       tempr/{datestr}_tem.r2c\n")
            f.write(":pointnetradiation\n")
            f.write(":pointhumidity\n")
            f.write(":pointwind\n")
            f.write(":pointlongwave\n")
            f.write(":pointshortwave\n")
            f.write(":pointatmpressure\n")
            f.write("#\n")
            f.write(f":streamflowdatafile           strfw/{datestr}_str.tb0\n")
            f.write("#\n")
            if followers:
                f.write(f":noeventstofollow            {len(followers):6d}\n")
                f.write("#\n")
                for fd in followers:
                    f.write(f"event/{fd}.evt\n")
            else:
                f.write(":noeventstofollow                 00\n")
            f.write("eof\n")

    # ------------------------------------------------------------------
    # 4. Output spec
    # ------------------------------------------------------------------
    def _generate_wfo_spec(self) -> None:
        """Generate wfo_spec.txt controlling WATFLOOD output."""
        out = self.settings_dir / 'wfo_spec.txt'
        with open(out, 'w') as f:
            f.write("  5.0 Version Number\n")
            f.write("   10 AttributeCount\n")
            f.write("   24 ReportingTimeStep Hours\n")
            f.write("    0 Start Reporting Time for GreenKenue (hr)\n")
            f.write("    0 End Reporting Time for GreenKenue (hr)\n")
            f.write("1   1 Temperature\n")
            f.write("1   2 Precipitation\n")
            f.write("1   3 Cumulative Precipitation\n")
            f.write("0   4 Lower Zone Storage Class\n")
            f.write("0   5 Ground Water Discharge m^3/s\n")
            f.write("0   6 Grid Runoff\n")
            f.write("1   7 Observed Outflow\n")
            f.write("1   8 Computed Outflow\n")
            f.write("1   9 Weighted SWE\n")
            f.write("1  10 Cumulative ET\n")
        logger.info(f"Wrote output spec: {out}")

    # ------------------------------------------------------------------
    # 5. Streamflow observation .tb0
    # ------------------------------------------------------------------
    def _generate_streamflow_tb0(self, start: datetime, end: datetime) -> None:
        """Generate streamflow .tb0 files from observations for each month."""
        try:
            props = self._get_catchment_properties()
            # Gauge sits at the active (centre) cell of the 3x3 grid.
            gauge_x = int(round(props['x_origin'] + 1.5 * 5000.0))
            gauge_y = int(round(props['y_origin'] + 1.5 * 5000.0))
            utm_zone = int(props['epsg']) - (32600 if props['lat'] >= 0 else 32700)
            obs_path = self._find_observation_file()
            if obs_path is None:
                logger.warning("No observation file found, skipping .tb0 generation")
                return

            obs_df = pd.read_csv(obs_path, parse_dates=[0], index_col=0)
            flow_col = None
            for col in obs_df.columns:
                if 'discharge' in col.lower() or 'flow' in col.lower():
                    flow_col = col
                    break
            if flow_col is None and len(obs_df.columns) > 0:
                flow_col = obs_df.columns[0]

            if flow_col is None:
                logger.warning("No flow column found in observations")
                return

            obs_daily = obs_df[flow_col].resample('D').mean()

            months = pd.date_range(start, end, freq='MS')
            for month_start in months:
                datestr = f"{month_start.year:04d}{month_start.month:02d}01"
                month_end = month_start + pd.offsets.MonthEnd(0)
                month_obs = obs_daily.loc[month_start:month_end]

                tb0_path = self.settings_dir / 'strfw' / f'{datestr}_str.tb0'
                with open(tb0_path, 'w') as f:
                    f.write("########################################\n")
                    f.write(":FileType tb0  ASCII  EnSim 1.0\n")
                    f.write("#\n")
                    f.write("# DataType               EnSim Table\n")
                    f.write("#\n")
                    f.write(":Application             EnSimHydrologic\n")
                    f.write(":Version                 2.1.23\n")
                    f.write(":WrittenBy          SYMFLUENCE\n")
                    f.write(f":CreationDate       {datetime.now():%Y-%m-%d  %H:%M}\n")
                    f.write("#\n")
                    f.write(":Name               Streamflow\n")
                    f.write("#\n")
                    f.write(":Projection         UTM\n")
                    f.write(":Ellipsoid          WGS84\n")
                    f.write(f":Zone                       {utm_zone}\n")
                    f.write("#\n")
                    f.write(":StartTime         00:00:00.00\n")
                    f.write(f":StartDate            {month_start:%Y/%m/%d}\n")
                    f.write(":DeltaT                        1\n")
                    f.write(":RoutingDeltaT                 1\n")
                    f.write("#\n")
                    f.write(":ColumnMetaData\n")
                    f.write("   :ColumnUnits             m3/s\n")
                    f.write("   :ColumnType             float\n")
                    f.write("   :ColumnName          GAUGE001\n")
                    f.write(f"   :ColumnLocationX      {gauge_x}\n")
                    f.write(f"   :ColumnLocationY     {gauge_y}\n")
                    # CHARM's read_flow_ef reads these per-column function
                    # coefficients (colCoeff1..4) + nopt (colValue1); without
                    # them the coeff arrays are size 0 and it crashes at l=1.
                    f.write("   :Coeff1                  0.0\n")
                    f.write("   :Coeff2                  0.0\n")
                    f.write("   :Coeff3                  0.0\n")
                    f.write("   :Coeff4                  0.0\n")
                    f.write("   :Value1                    1\n")
                    f.write(":EndColumnMetaData\n")
                    f.write("#\n")
                    f.write(":endHeader\n")
                    # One value per timestep (DeltaT=1 h), single column to match
                    # the 1-gauge header. Daily obs are held across the 24 hours
                    # of the day; -1 = missing.
                    ndays = calendar.monthrange(month_start.year, month_start.month)[1]
                    for day in range(1, ndays + 1):
                        date = pd.Timestamp(year=month_start.year,
                                            month=month_start.month, day=day)
                        val = month_obs.get(date, -1.0)
                        if pd.isna(val):
                            val = -1.0
                        for _ in range(24):   # hourly
                            f.write(f"    {val:10.3f}\n")

            logger.info(f"Wrote {len(months)} streamflow .tb0 files")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not generate streamflow .tb0: {e}", exc_info=True)

    def _find_observation_file(self):
        """Find observation streamflow file."""
        search_dirs = [
            self.project_observations_dir / 'streamflow' / 'preprocessed',
            self.project_observations_dir / 'streamflow',
            self.project_observations_dir,
        ]
        for obs_dir in search_dirs:
            if not obs_dir.exists():
                continue
            for pattern in ['*streamflow*.csv', '*discharge*.csv', '*.csv']:
                matches = list(obs_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None
