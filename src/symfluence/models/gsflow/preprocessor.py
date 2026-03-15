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

import logging
import os
from datetime import datetime
from typing import Tuple

import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor

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
            logger.error(f"GSFLOW preprocessing failed: {e}")
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

        logger.info(f"Loading ERA5 forcing ({len(forcing_files)} files)")
        try:
            ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
        except Exception:  # noqa: BLE001 — model execution resilience
            datasets = [xr.open_dataset(f) for f in forcing_files]
            ds = xr.concat(datasets, dim='time')

        ds = ds.sel(time=slice(str(start_date), str(end_date)))

        airtemp = ds['air_temperature'].values.squeeze()   # K
        pptrate = ds['precipitation_flux'].values.squeeze()    # mm/s
        times = pd.DatetimeIndex(ds['time'].values)

        hourly = pd.DataFrame({
            'airtemp_C': airtemp - 273.15,
            'precip_mm': pptrate * 3600.0,
        }, index=times)

        daily = pd.DataFrame({
            'precip': hourly['precip_mm'].resample('D').sum(),
            'tmax': hourly['airtemp_C'].resample('D').max(),
            'tmin': hourly['airtemp_C'].resample('D').min(),
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

        nrow, ncol = 3, 3
        ncells = nrow * ncol
        nhru = ncells          # 1 HRU per MODFLOW cell (1:1 GVR mapping)
        nsegment = 1
        nreach = ncells         # 1 SFR reach per cell (full coverage)
        nmonths = 12
        nhrucell = ncells      # 1:1 mapping → 9 HRU-cell pairs

        lat = 51.17
        lon = -115.57
        elev = 1500.0
        area_km2 = 2210.0
        hru_area_acres = area_km2 * 247.105 / nhru  # each HRU = 1/9 of basin

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
            wp("soil_rechr_max", 1, "nhru", nhru, 2, [2.0] * nhru)
            wp("soil_moist_init", 1, "nhru", nhru, 2, [3.0] * nhru)
            wp("soil_rechr_init", 1, "nhru", nhru, 2, [1.0] * nhru)
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

            # -- GVR mapping (1:1, each HRU maps to one cell) --
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
        nrow, ncol = 3, 3
        cell_size = 15700.0  # ~15.7 km → 9 cells ≈ 2210 km²
        top_elev = 1600.0
        bot_elev = 1400.0
        init_head = 1595.0   # shallow water table (5 m below surface)
        hk = 1.0             # m/d  (calibrated as K)
        sy = 0.15            # (-)  (calibrated as SY)
        stream_elev = 1585.0 # stream bed 15 m below land surface
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
            # STRT (starting heads) layer 1
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
            f.write("1.0E-3  100.0  500  1.0E-5  2  1  1  SPECIFIED\n")
            # XMD solver line
            f.write("2  1  3  3  1  0.0  1  5.0E-4  5.0E-4  200  XMD\n")

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
            f.write("  CONSTANT  0.30\n")
            # Stress period 1 (SS)
            ncells = nrow * ncol
            f.write(f"  {ncells}\n")               # NUZF1 (read FINF)
            f.write("  CONSTANT  1.0E-8\n")        # FINF array
            # Stress period 2 (TR) — reuse previous
            f.write("  -1\n")                       # NUZF1 < 0 = reuse
        logger.info(f"  UZF: {uzf}")

        # -- SFR  (1 segment, 9 reaches — serpentine through all cells) --
        nsfr_reach = nrow * ncol
        sfr = mf / 'bow.sfr'
        with open(sfr, 'w') as f:
            f.write(f"# GSFLOW SFR Package (3x3 grid — {nsfr_reach} reaches)\n")
            f.write("OPTIONS\nREACHINPUT\nEND\n")
            f.write(f"    {nsfr_reach}    1    0    0  86400.0  0.0001"
                    f"  -1  0  3  10  1  40  0\n")
            # Serpentine path: row1 L→R, row2 R→L, row3 L→R
            reach_cells = []
            for row in range(1, nrow + 1):
                cols = range(1, ncol + 1) if row % 2 == 1 else range(ncol, 0, -1)
                for col in cols:
                    reach_cells.append((row, col))
            # Write reach data with decreasing STRTOP
            rchlen = cell_size
            for ireach, (row, col) in enumerate(reach_cells, 1):
                strtop = stream_elev + stream_slope * (nsfr_reach - ireach) * cell_size
                # KRCH IRCH JRCH ISEG IREACH RCHLEN STRTOP SLOPE
                #   STRTHICK STRMBD_K THTS THTI EPS UHC
                f.write(f"    1    {row}    {col}    1    {ireach}"
                        f"  {rchlen:.1f}  {strtop:.1f}"
                        f"  {stream_slope:.6f}  1.0"
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
