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

import calendar
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor

logger = logging.getLogger(__name__)


class WATFLOODPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Pre-processor for WATFLOOD model setup (lumped 1-cell basin)."""

    MODEL_NAME = "WATFLOOD"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.watflood_dir = self.project_dir / 'WATFLOOD_input'
        self.settings_dir = self.watflood_dir / 'settings'

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run_preprocessing(self) -> bool:
        """Generate all WATFLOOD input files from scratch."""
        try:
            # Create directory tree
            for d in ('basin', 'event', 'raing', 'tempg', 'strfw',
                      'results', 'debug', 'moist', 'snow1'):
                (self.settings_dir / d).mkdir(parents=True, exist_ok=True)

            start, end = self._get_simulation_dates()
            logger.info(f"WATFLOOD preprocessing: {start:%Y-%m-%d} to {end:%Y-%m-%d}")

            # Load ERA5 forcing
            hourly = self._load_era5_forcing(start, end)

            # 1. Watershed definition
            self._generate_shd_file()

            # 2. Parameter file
            self._generate_par_file()

            # 3. Monthly forcing + event files
            self._generate_monthly_files(hourly, start, end)

            # 4. Output spec
            self._generate_wfo_spec()

            # 5. Observation streamflow (for WATFLOOD stats)
            self._generate_streamflow_tb0(start, end)

            logger.info(f"WATFLOOD preprocessing complete: {self.settings_dir}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"WATFLOOD preprocessing failed: {e}")
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

        logger.info(f"Loading ERA5 forcing ({len(forcing_files)} files)")
        try:
            ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
        except Exception:  # noqa: BLE001 — model execution resilience
            datasets = [xr.open_dataset(f) for f in forcing_files]
            ds = xr.concat(datasets, dim='time')

        ds = ds.sel(time=slice(str(start), str(end)))

        airtemp = ds['air_temperature'].values.squeeze()   # K
        pptrate = ds['precipitation_flux'].values.squeeze()    # mm/s
        times = pd.DatetimeIndex(ds['time'].values)

        hourly = pd.DataFrame({
            'temp_C': airtemp - 273.15,
            'precip_mm': pptrate * 3600.0,
        }, index=times)
        ds.close()

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
        area_km2 = 2210.0
        cell_m = 5000.0   # 5 km cells
        elev = 1600.0
        x_origin = 560000.0
        y_origin = 5670000.0
        da = area_km2 / ((cell_m / 1000.0) ** 2)  # ~88.4 grid units
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
            f.write(":ClassCount                    1\n")
            f.write(":NumRiverClasses               1\n")
            f.write(":ElevConversion            1.000\n")
            f.write(":TotalNumOfGrids               1\n")
            f.write(":numGridsInBasin               1\n")
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
            f.write(":AttributeName 12 conifer     \n")
            f.write("#                                       \n")
            f.write(f":xCount                       {nc}\n")
            f.write(f":yCount                       {nc}\n")
            f.write(f":xDelta                 {cell_m:.6f}\n")
            f.write(f":yDelta                 {cell_m:.6f}\n")
            f.write("#                                       \n")
            f.write(":EndHeader                              \n")

            # 1. Rank grid (3x3 integers) — cell (2,2) = 1
            f.write(_grid_line(z3))
            f.write(_grid_line([0, 1, 0]))
            f.write(_grid_line(z3))

            # 2. Next grid (downstream cell) — 0 everywhere (outlet)
            for _ in range(nc):
                f.write(_grid_line(z3))

            # 3. DA grid (drainage area in grid units, scientific notation)
            z3f = [0.0, 0.0, 0.0]
            f.write(_grid_line_f(z3f))
            f.write(_grid_line_f([0.0, da, 0.0]))
            f.write(_grid_line_f(z3f))

            # 4. Bankfull grid (fixed format)
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, bankfull, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

            # 5. Channel slope grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, slope, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))

            # 6. Elevation grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, elev, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

            # 7. Channel length grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, ch_len, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

            # 8. IAK grid (interflow active key: 1=active)
            f.write(_grid_line(z3))
            f.write(_grid_line([0, 1, 0]))
            f.write(_grid_line(z3))

            # 9. Internal slope grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, slope, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))

            # 10. Channel class grid (1=river)
            f.write(_grid_line(z3))
            f.write(_grid_line([0, 1, 0]))
            f.write(_grid_line(z3))

            # 11. Reach grid (0=no reach)
            for _ in range(nc):

                f.write(_grid_line(z3))

            # 12. Land class fraction: conifer = 1.0 at active cell
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, 1.0, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

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
        lat = 51.17

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
            f.write(f":a11,         {-999.999:10.3f},# bare ground equiv. veg height for ev\n")
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
    # ------------------------------------------------------------------
    def _generate_monthly_files(self, hourly: pd.DataFrame,
                                start: datetime, end: datetime) -> None:
        """Generate per-month .evt, .rag, .tag files."""
        # Coordinate info for forcing headers (UTM zone 11N, km)
        y_km = 5670
        x_km = 560
        ymin, ymax = y_km, y_km + 15  # 3 cells * 5km = 15km span
        xmin, xmax = x_km, x_km + 15

        # Build list of months
        months = pd.date_range(start, end, freq='MS')
        logger.info(f"Generating {len(months)} monthly event files")

        evt_files = []
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

            # Write .evt file
            evt_path = self.settings_dir / 'event' / f'{datestr}.evt'
            is_last = (i == len(months) - 1)
            next_month = months[i + 1] if not is_last else None
            self._write_evt_file(evt_path, year, month, nhours,
                                 datestr, is_last, next_month)
            evt_files.append(evt_path)

        # Write the master event.evt pointing to the first month
        if evt_files:
            first_datestr = f"{months[0].year:04d}{months[0].month:02d}01"
            master_evt = self.settings_dir / 'event' / 'event.evt'
            first_month = months[0]
            ndays_first = calendar.monthrange(first_month.year, first_month.month)[1]
            nhours_first = ndays_first * 24
            self._write_evt_file(master_evt, first_month.year, first_month.month,
                                 nhours_first, first_datestr,
                                 len(months) <= 1,
                                 months[1] if len(months) > 1 else None)

        logger.info(f"Wrote {len(evt_files)} monthly event/forcing files")

    def _write_evt_file(self, path: Path, year: int, month: int,
                        nhours: int, datestr: str,
                        is_last: bool, next_month) -> None:
        """Write a single .evt file."""
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
            f.write("#\n")
            f.write(f":pointprecip                  raing/{datestr}.rag\n")
            f.write(f":pointtemps                   tempg/{datestr}.tag\n")
            f.write(":pointnetradiation\n")
            f.write(":pointhumidity\n")
            f.write(":pointwind\n")
            f.write(":pointlongwave\n")
            f.write(":pointshortwave\n")
            f.write(":pointatmpressure\n")
            f.write("#\n")
            f.write(f":streamflowdatafile           strfw/{datestr}_str.tb0\n")
            f.write("#\n")
            if is_last:
                f.write(":noeventstofollow                 00\n")
            else:
                next_datestr = f"{next_month.year:04d}{next_month.month:02d}01"
                f.write(":noeventstofollow                 01\n")
                f.write("#\n")
                f.write(f"event/{next_datestr}.evt\n")
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
                    f.write(":Zone                       11\n")
                    f.write("#\n")
                    f.write(":StartTime         00:00:00.00\n")
                    f.write(f":StartDate            {month_start:%Y/%m/%d}\n")
                    f.write(":DeltaT                        1\n")
                    f.write(":RoutingDeltaT                 1\n")
                    f.write("#\n")
                    f.write(":ColumnMetaData\n")
                    f.write("   :ColumnUnits             m3/s\n")
                    f.write("   :ColumnType             float\n")
                    f.write("   :ColumnName          05BB001\n")
                    f.write("   :ColumnLocationX      583000\n")
                    f.write("   :ColumnLocationY     5673000\n")
                    f.write(":EndColumnMetaData\n")
                    f.write("#\n")
                    f.write(":endHeader\n")
                    # Write daily obs at hour 0
                    ndays = calendar.monthrange(month_start.year, month_start.month)[1]
                    for day in range(1, ndays + 1):
                        date = pd.Timestamp(year=month_start.year,
                                            month=month_start.month, day=day)
                        val = month_obs.get(date, -1.0)
                        if pd.isna(val):
                            val = -1.0
                        f.write(f" {month_start.year} {month_start.month:2d}"
                                f" {day:2d}  0 {val:10.3f}\n")

            logger.info(f"Wrote {len(months)} streamflow .tb0 files")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not generate streamflow .tb0: {e}")

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
