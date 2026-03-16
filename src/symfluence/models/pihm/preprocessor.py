# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Preprocessor for MM-PIHM

Generates all required MM-PIHM input files for a lumped single-element mesh
model. The single equilateral triangle represents the entire catchment.

MM-PIHM expects files at: input/<project_name>/<project_name>.*

Input files generated:
    <project>.mesh  -- Triangular mesh (single triangle for lumped)
    <project>.att   -- Element attributes (soil/geol/lc/meteo/lai indices)
    <project>.soil  -- Soil properties table
    <project>.geol  -- Geology (deep subsurface) properties
    <project>.riv   -- River network (single segment for lumped)
    <project>.meteo -- Meteorological forcing (from ERA5 basin-averaged data)
    <project>.lai   -- Leaf area index time series (seasonal cycle)
    <project>.para  -- Solver/output control parameters
    <project>.calib -- Calibration multipliers (all 1.0 = no change)
    <project>.bc    -- Boundary conditions (empty for lumped)
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("PIHM")
class PIHMPreProcessor:
    """Generates MM-PIHM input files for lumped groundwater simulation."""

    # Project name used for all MM-PIHM files
    PROJECT_NAME = "pihm_lumped"

    def __init__(self, config, logger, **kwargs):
        self.config = config
        self.logger = logger
        self.config_dict = config.to_dict(flatten=True) if hasattr(config, 'to_dict') else (config if isinstance(config, dict) else {})

        self.domain_name = self._get_cfg('DOMAIN_NAME', 'unknown')
        self.experiment_id = self._get_cfg('EXPERIMENT_ID', 'default')

        project_dir = getattr(config, 'get', lambda k, d=None: d)('PROJECT_DIR')
        if not project_dir or project_dir == '.':
            try:
                data_dir = str(config.system.data_dir)
            except (AttributeError, TypeError):
                data_dir = '.'
            project_dir = Path(data_dir) / f"domain_{self.domain_name}"
        self.project_dir = Path(project_dir)

    # -------------------------------------------------------------------------
    # Config helpers
    # -------------------------------------------------------------------------

    def _get_cfg(self, key, default=None):
        """Get config value with typed config first."""
        try:
            if key == 'DOMAIN_NAME':
                return self.config.domain.name
            elif key == 'EXPERIMENT_ID':
                return self.config.domain.experiment_id
        except (AttributeError, TypeError):
            pass
        return default

    def _get_pihm_cfg(self, key, default=None):
        """Get PIHM-specific config value with typed config first."""
        try:
            pihm_cfg = self.config.model.pihm
            if pihm_cfg:
                attr = key.lower()
                if hasattr(pihm_cfg, attr):
                    pydantic_val = getattr(pihm_cfg, attr)
                    if pydantic_val is not None:
                        return pydantic_val
        except (AttributeError, TypeError):
            pass
        return default

    def _get_catchment_area_m2(self) -> float:
        """Get catchment area in m2."""
        try:
            area_km2 = self.config.domain.catchment_area
        except (AttributeError, TypeError):
            area_km2 = None
        if area_km2 is None:
            area_km2 = self._get_pihm_cfg('CATCHMENT_AREA')
        if area_km2 is None:
            self.logger.warning("CATCHMENT_AREA not set, using default 2210 km2")
            area_km2 = 2210.0
        return float(area_km2) * 1e6

    def _get_time_info(self):
        """Get simulation time range as pandas Timestamps."""
        try:
            start = self.config.domain.time_start
        except (AttributeError, TypeError):
            start = None
        if not start:
            start = '2000-01-01'
        try:
            end = self.config.domain.time_end
        except (AttributeError, TypeError):
            end = None
        if not end:
            end = '2001-01-01'

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        return start_dt, end_dt

    # -------------------------------------------------------------------------
    # Forcing data helpers
    # -------------------------------------------------------------------------

    def _find_forcing_files(self, start_dt, end_dt):
        """Find ERA5 basin-averaged forcing NetCDF files covering the time range."""
        forcing_dir = resolve_data_subdir(self.project_dir, 'forcing') / "basin_averaged_data"
        if not forcing_dir.exists():
            self.logger.warning(
                f"Forcing directory not found: {forcing_dir}. "
                "Will generate synthetic meteorological data."
            )
            return []

        nc_files = sorted(forcing_dir.glob("*.nc"))
        if not nc_files:
            self.logger.warning(
                f"No NetCDF files found in {forcing_dir}. "
                "Will generate synthetic meteorological data."
            )
            return []

        # Filter files that overlap with our time range.
        # File names contain date like: *_YYYY-MM-DD-HH-MM-SS.nc
        selected = []
        for f in nc_files:
            # Extract date from filename
            # Format: *_YYYY-MM-DD-HH-MM-SS.nc (date is in the last underscore-delimited part)
            parts = f.stem.split('_')
            date_str = None
            for p in reversed(parts):
                # Look for a part starting with a 4-digit year (e.g., 2002-01-01-00-00-00)
                if len(p) >= 10 and p[:4].isdigit() and p[4] == '-':
                    date_str = p
                    break
            if date_str is None:
                continue
            try:
                # Parse YYYY-MM-DD from the date string
                file_date = pd.Timestamp(date_str[:10])
            except (ValueError, IndexError):
                continue

            # Monthly files: include if the file month overlaps with our range
            # A file dated YYYY-MM-01 covers that entire month
            file_month_end = file_date + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
            if file_date <= end_dt and file_month_end >= start_dt:
                selected.append(f)

        self.logger.info(
            f"Found {len(selected)} forcing files covering "
            f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        )
        return selected

    def _load_forcing_data(self, forcing_files, start_dt, end_dt):
        """Load and concatenate ERA5 forcing data from NetCDF files.

        Returns a DataFrame with columns:
            time, prcp (kg/m2/s), temp (K), rh (%), wind (m/s),
            solar (W/m2), longwave (W/m2), pres (Pa)
        """
        try:
            import xarray as xr
        except ImportError:
            self.logger.warning("xarray not available; using synthetic forcing data")
            return None

        datasets = []
        for f in sorted(forcing_files):
            try:
                ds = xr.open_dataset(f)
                datasets.append(ds)
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Could not read {f.name}: {e}")

        if not datasets:
            return None

        combined = xr.concat(datasets, dim='time')
        # Select time range
        combined = combined.sel(time=slice(str(start_dt), str(end_dt)))

        if combined.time.size == 0:
            self.logger.warning("No forcing data within simulation time range")
            return None

        # Extract variables -- squeeze the hru dimension (lumped = single HRU)
        def _get_var(ds, name):
            if name in ds:
                arr = ds[name].values
                if arr.ndim > 1:
                    arr = arr[:, 0]  # first HRU
                return arr
            return None

        times = pd.DatetimeIndex(combined.time.values)
        prcp = _get_var(combined, 'precipitation_flux')        # mm/s -> need kg/m2/s (same numerically)
        temp = _get_var(combined, 'air_temperature')         # K
        spechum = _get_var(combined, 'specific_humidity')      # kg/kg
        wind = _get_var(combined, 'wind_speed')          # m/s
        solar = _get_var(combined, 'surface_downwelling_shortwave_flux')        # W/m2
        longwave = _get_var(combined, 'surface_downwelling_longwave_flux')     # W/m2
        pres = _get_var(combined, 'surface_air_pressure')          # Pa

        # Convert specific humidity to relative humidity (%)
        # Using Tetens formula for saturation vapor pressure
        rh = self._spechum_to_rh(spechum, temp, pres)

        # Convert precipitation from mm/s to kg/m2/s
        # 1 mm/s water = 1 kg/m2/s (density of water = 1000 kg/m3, 1mm = 0.001m)
        # So mm/s and kg/m2/s are numerically identical
        prcp_kgm2s = prcp if prcp is not None else np.zeros(len(times))

        df = pd.DataFrame({
            'time': times,
            'prcp': prcp_kgm2s,
            'temp': temp if temp is not None else np.full(len(times), 273.15),
            'rh': rh if rh is not None else np.full(len(times), 50.0),
            'wind': wind if wind is not None else np.full(len(times), 2.0),
            'solar': solar if solar is not None else np.full(len(times), 200.0),
            'longwave': longwave if longwave is not None else np.full(len(times), 300.0),
            'pres': pres if pres is not None else np.full(len(times), 101325.0),
        })

        # Close datasets
        for ds in datasets:
            ds.close()

        return df

    @staticmethod
    def _spechum_to_rh(spechum, temp_k, pres_pa):
        """Convert specific humidity (kg/kg) to relative humidity (%).

        Uses the Tetens formula for saturation vapor pressure:
            es = 610.78 * exp(17.27 * (T - 273.15) / (T - 35.86))

        Then:
            e = q * P / (0.622 + 0.378 * q)
            RH = 100 * e / es
        """
        if spechum is None or temp_k is None or pres_pa is None:
            return None

        q = np.array(spechum, dtype=np.float64)
        T = np.array(temp_k, dtype=np.float64)
        P = np.array(pres_pa, dtype=np.float64)

        # Saturation vapor pressure (Pa) -- Tetens formula
        es = 610.78 * np.exp(17.27 * (T - 273.15) / (T - 35.86))

        # Actual vapor pressure from specific humidity
        e = q * P / (0.622 + 0.378 * q)

        # Relative humidity (%)
        rh = 100.0 * e / es
        rh = np.clip(rh, 0.0, 100.0)
        return rh

    def _generate_synthetic_forcing(self, start_dt, end_dt):
        """Generate simple synthetic hourly forcing data for testing.

        Returns a DataFrame with the same structure as real forcing data.
        """
        times = pd.date_range(start_dt, end_dt, freq='h')
        n = len(times)
        day_of_year = times.dayofyear.values
        hour = times.hour.values

        # Simple diurnal/seasonal cycles
        temp_mean = 273.15 + 5.0  # 5 C mean
        temp_seasonal = 15.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365.0)
        temp_diurnal = 5.0 * np.cos(2 * np.pi * (hour - 14) / 24.0)
        temp = temp_mean + temp_seasonal + temp_diurnal

        # Solar radiation (simple bell curve during day)
        solar_max = 400.0 + 200.0 * np.cos(2 * np.pi * (day_of_year - 172) / 365.0)
        solar_hour = np.maximum(0, np.cos(2 * np.pi * (hour - 12) / 24.0))
        solar = solar_max * solar_hour

        # Longwave radiation (roughly 200-350 W/m2)
        longwave = 250.0 + 50.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365.0)

        # Precipitation (random events)
        rng = np.random.default_rng(42)
        prcp = np.zeros(n)
        rain_mask = rng.random(n) < 0.05  # 5% chance each hour
        prcp[rain_mask] = rng.exponential(5e-5, rain_mask.sum())  # kg/m2/s

        df = pd.DataFrame({
            'time': times,
            'prcp': prcp,
            'temp': temp,
            'rh': np.full(n, 60.0),
            'wind': np.full(n, 2.0),
            'solar': solar,
            'longwave': longwave,
            'pres': np.full(n, 90000.0),
        })
        return df

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run_preprocessing(self):
        """Generate all MM-PIHM input files."""
        settings_dir = self.project_dir / "settings" / "PIHM"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating MM-PIHM input files in {settings_dir}")

        # Gather parameters
        area_m2 = self._get_catchment_area_m2()
        k_sat = float(self._get_pihm_cfg('K_SAT', 1e-5))
        porosity = float(self._get_pihm_cfg('POROSITY', 0.4))
        vg_alpha = float(self._get_pihm_cfg('VG_ALPHA', 1.0))
        vg_n = float(self._get_pihm_cfg('VG_N', 2.0))
        macropore_k = float(self._get_pihm_cfg('MACROPORE_K', 1e-4))
        macropore_depth = float(self._get_pihm_cfg('MACROPORE_DEPTH', 0.5))
        soil_depth = float(self._get_pihm_cfg('SOIL_DEPTH', 2.0))
        solver_reltol = float(self._get_pihm_cfg('SOLVER_RELTOL', 1e-3))
        solver_abstol = float(self._get_pihm_cfg('SOLVER_ABSTOL', 1e-4))
        timestep = int(self._get_pihm_cfg('TIMESTEP_SECONDS', 60))
        lsm_step = int(self._get_pihm_cfg('LSM_STEP', 900))

        start_dt, end_dt = self._get_time_info()
        name = self.PROJECT_NAME

        self.logger.info(
            f"MM-PIHM lumped mesh: area={area_m2/1e6:.0f} km2, "
            f"K_sat={k_sat}, porosity={porosity}, soil_depth={soil_depth}m, "
            f"period={start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        )

        # Load forcing data
        forcing_files = self._find_forcing_files(start_dt, end_dt)
        if forcing_files:
            forcing_df = self._load_forcing_data(forcing_files, start_dt, end_dt)
        else:
            forcing_df = None

        if forcing_df is None or forcing_df.empty:
            self.logger.warning("Using synthetic forcing data")
            forcing_df = self._generate_synthetic_forcing(start_dt, end_dt)

        # Write all MM-PIHM input files
        self._write_mesh(settings_dir, name, area_m2, soil_depth)
        self._write_att(settings_dir, name)
        self._write_soil(settings_dir, name, k_sat, porosity,
                         vg_alpha, vg_n, macropore_k, macropore_depth, soil_depth)
        self._write_geol(settings_dir, name, k_sat)
        self._write_riv(settings_dir, name, k_sat)
        self._write_meteo(settings_dir, name, forcing_df)
        self._write_lai(settings_dir, name, start_dt, end_dt)
        self._write_para(settings_dir, name, start_dt, end_dt,
                         timestep, lsm_step, solver_reltol, solver_abstol)
        self._write_calib(settings_dir, name)
        self._write_bc(settings_dir, name)
        self._write_ic(settings_dir, name, soil_depth)
        self._write_lsm(settings_dir, name, soil_depth)

        self.logger.info(f"MM-PIHM (Flux-PIHM) input files generated in {settings_dir}")

    # -------------------------------------------------------------------------
    # File writers
    # -------------------------------------------------------------------------

    def _write_mesh(self, d: Path, name: str, area_m2: float, soil_depth: float):
        """Write .mesh file -- two-element mesh for lumped mode.

        MM-PIHM requires LEFT != RIGHT for river segments so that the
        elem<->river water exchange is correctly reflected back to the
        element ODEs.  A single-element mesh with LEFT=RIGHT=1 causes
        the ``nabr_river`` mapping to fail (init_river.c), breaking
        mass conservation.

        Layout (4 nodes, 2 triangles sharing edge 1-2 = river):

            Node 3
            /|\\
           / | \\
          /  |  \\       Element 1: nodes 1,2,3  (left of river)
         /   |   \\      Element 2: nodes 2,1,4  (right of river)
        /    |    \\
       Node1 ---- Node2   <-- river runs along this edge
        \\    |    /
         \\   |   /
          \\  |  /
           \\ | /
            \\|/
            Node 4

        Each triangle has area = catchment_area / 2.
        """
        # River length (edge 1-2).  Use sqrt(area) as characteristic length.
        # Each triangle: area = 0.5 * L * (H/2) = L*H/4.
        # Two triangles: total = L*H/2 = area_m2  =>  H = 2*area_m2/L.
        L = math.sqrt(area_m2)          # ~ 47 km for 2210 km²
        H = 2.0 * area_m2 / L          # = 2*L  (each triangle apex at ±H/2)

        # Topographic gradient (gentle slope along the river toward outlet)
        mean_slope = 0.001
        elev_outlet = 1500.0
        elev_drop = L * mean_slope
        elev_upstream = elev_outlet + elev_drop
        elev_mid = elev_outlet + elev_drop * 0.5

        lines = [
            "NUMELE\t2",
            "INDEX\tNODE1\tNODE2\tNODE3\tNABR1\tNABR2\tNABR3",
            # Element 1 (left bank): nodes 1,2,3.  Edge 1-2 faces element 2.
            "1\t1\t2\t3\t2\t0\t0",
            # Element 2 (right bank): nodes 2,1,4.  Edge 2-1 faces element 1.
            "2\t2\t1\t4\t1\t0\t0",
            "NUMNODE\t4",
            "INDEX\tX\tY\tZMIN\tZMAX",
            f"1\t0.0\t0.0\t{elev_upstream - soil_depth:.1f}\t{elev_upstream:.1f}",
            f"2\t{L:.2f}\t0.0\t{elev_outlet - soil_depth:.1f}\t{elev_outlet:.1f}",
            f"3\t{L/2:.2f}\t{H/2:.2f}\t{elev_mid - soil_depth:.1f}\t{elev_mid:.1f}",
            f"4\t{L/2:.2f}\t{-H/2:.2f}\t{elev_mid - soil_depth:.1f}\t{elev_mid:.1f}",
        ]
        (d / f"{name}.mesh").write_text("\n".join(lines) + "\n")
        centroid_dist = H / 2 / 3  # dist from centroid to river (y=0)
        self.logger.debug(
            f"Wrote {name}.mesh (2 elements, L={L:.0f}m, H={H:.0f}m, "
            f"centroid-to-river={centroid_dist:.0f}m, "
            f"elev={elev_outlet:.0f}-{elev_upstream:.0f}m)"
        )

    def _write_att(self, d: Path, name: str):
        """Write .att file -- element attributes.

        Format:
            INDEX   SOIL    GEOL    LC      METEO   LAI     BC1     BC2     BC3

        BC values: 0 = no boundary condition on that edge.
        Two elements share the same soil, geology, land cover, forcing, and LAI.
        """
        lines = [
            "INDEX\tSOIL\tGEOL\tLC\tMETEO\tLAI\tBC1\tBC2\tBC3",
            "1\t1\t1\t1\t1\t1\t0\t0\t0",
            "2\t1\t1\t1\t1\t1\t0\t0\t0",
        ]
        (d / f"{name}.att").write_text("\n".join(lines) + "\n")
        self.logger.debug(f"Wrote {name}.att (2 elements)")

    def _write_soil(self, d: Path, name: str, k_sat, porosity,
                    vg_alpha, vg_n, macropore_k, macropore_depth, soil_depth):
        """Write .soil file -- soil properties table.

        Format:
            NUMSOIL     <n>
            INDEX   SILT    CLAY    OM      BD      KINF        KSATV       KSATH
                    MAXSMC  MINSMC  ALPHA   BETA    MACHF       MACVF       DMAC    QTZ
            ...
            DINF        <depth>
            KMACV_RO    <ratio>
            KMACH_RO    <ratio>
        """
        # Residual moisture ~ 0.05 for most soils
        min_smc = 0.05
        lines = [
            "NUMSOIL\t1",
            ("INDEX\tSILT\tCLAY\tOM\tBD\tKINF\tKSATV\tKSATH\t"
             "MAXSMC\tMINSMC\tALPHA\tBETA\tMACHF\tMACVF\tDMAC\tQTZ"),
            (f"1\t30.0\t15.0\t3.0\t1.4\t{k_sat:.6e}\t{k_sat:.6e}\t{k_sat:.6e}\t"
             f"{porosity:.4f}\t{min_smc:.2f}\t{vg_alpha:.4f}\t{vg_n:.4f}\t"
             f"{macropore_k:.6e}\t{macropore_k:.6e}\t{macropore_depth:.2f}\t0.25"),
            "DINF\t0.10",
            "KMACV_RO\t100.0",
            "KMACH_RO\t1000.0",
        ]
        (d / f"{name}.soil").write_text("\n".join(lines) + "\n")
        self.logger.debug(f"Wrote {name}.soil")

    def _write_geol(self, d: Path, name: str, k_sat):
        """Write .geol file -- deep subsurface (geology) properties.

        Format:
            NUMGEOL     <n>
            INDEX   KSATV       KSATH       MAXSMC  MINSMC  ALPHA   BETA
                    MACHF       MACVF       DMAC
            ...
            KMACV_RO    <ratio>
            KMACH_RO    <ratio>
        """
        # Deep geology: much lower K, lower porosity, larger pore size param
        geol_k = k_sat / 100.0
        lines = [
            "NUMGEOL\t1",
            ("INDEX\tKSATV\tKSATH\tMAXSMC\tMINSMC\tALPHA\tBETA\t"
             "MACHF\tMACVF\tDMAC"),
            (f"1\t{geol_k:.6e}\t{geol_k:.6e}\t0.037\t0.00\t10.0\t2.0\t"
             f"0.01\t0.01\t1.0"),
            "KMACV_RO\t100.0",
            "KMACH_RO\t1000.0",
        ]
        (d / f"{name}.geol").write_text("\n".join(lines) + "\n")
        self.logger.debug(f"Wrote {name}.geol")

    def _write_riv(self, d: Path, name: str, k_sat):
        """Write .riv file -- river network.

        Even lumped mode needs at least one river segment.

        Format matches ShaleHills example: NUMRIV, river segments, SHAPE section,
        MATERIAL section (INDEX ROUGH CWR KH), BC and RES sections.

        DOWN = -3 means outflow point (outlet).

        Channel dimensions are scaled from catchment area using hydraulic
        geometry relationships (Leopold & Maddock, 1953):
            bankfull width  W ≈ 1.5 * A_km2^0.5   (m)
            bankfull depth  D ≈ 0.25 * A_km2^0.35  (m)
        These give realistic river dimensions for any catchment size.
        """
        # Scale channel geometry from catchment area
        area_km2 = self._get_catchment_area_m2() / 1e6
        channel_width = 1.5 * area_km2 ** 0.5     # ~70 m for 2210 km2
        channel_depth = 0.25 * area_km2 ** 0.35    # ~2.5 m for 2210 km2
        # Ensure reasonable minimums
        channel_width = max(channel_width, 2.0)
        channel_depth = max(channel_depth, 0.3)

        self.logger.info(
            f"River channel geometry: width={channel_width:.1f}m, "
            f"depth={channel_depth:.1f}m (scaled from {area_km2:.0f} km2)"
        )

        lines = [
            "NUMRIV\t1",
            "INDEX\tFROM\tTO\tDOWN\tLEFT\tRIGHT\tSHAPE\tMATL\tBC\tRES",
            "1\t1\t2\t-3\t1\t2\t1\t1\t0\t0",
            "SHAPE\t1",
            "INDEX\tDPTH\tOINT\tCWID",
            "#-\tm\t-\t-",
            f"1\t{channel_depth:.2f}\t1\t{channel_width:.1f}",
            "MATERIAL\t1",
            "INDEX\tROUGH\tCWR\tKH",
            "#-\ts/m1/3\t-\tm/s",
            f"1\t0.05\t0.6\t{k_sat:.3E}",
            "BC\t0",
            "RES\t0",
        ]
        (d / f"{name}.riv").write_text("\n".join(lines) + "\n")
        self.logger.debug(
            f"Wrote {name}.riv (W={channel_width:.1f}m, D={channel_depth:.1f}m)"
        )

    def _write_meteo(self, d: Path, name: str, forcing_df: pd.DataFrame):
        """Write .meteo file -- meteorological forcing time series.

        Format:
            METEO_TS        1       WIND_LVL    10.0
            TIME                    PRCP            SFCTMP  RH      SFCSPD
                                    SOLAR           LONGWV  PRES
            #TS                     kg/m2/s         K       %       m/s
                                    W/m2            W/m2    Pa
            YYYY-MM-DD HH:MM       <prcp>          <temp>  <rh>    <wind>
                                    <solar>         <longwv> <pres>
            ...
        """
        lines = [
            "METEO_TS\t1\tWIND_LVL\t10.0",
            "TIME\t\tPRCP\t\tSFCTMP\tRH\tSFCSPD\tSOLAR\tLONGWV\tPRES",
            "#TS\t\tkg/m2/s\t\tK\t%\tm/s\tW/m2\tW/m2\tPa",
        ]

        for _, row in forcing_df.iterrows():
            ts = row['time']
            if isinstance(ts, pd.Timestamp):
                time_str = ts.strftime('%Y-%m-%d %H:%M')
            else:
                time_str = str(ts)[:16]

            lines.append(
                f"{time_str}\t"
                f"{row['prcp']:.8e}\t"
                f"{row['temp']:.2f}\t"
                f"{row['rh']:.1f}\t"
                f"{row['wind']:.2f}\t"
                f"{row['solar']:.2f}\t"
                f"{row['longwave']:.2f}\t"
                f"{row['pres']:.1f}"
            )

        (d / f"{name}.meteo").write_text("\n".join(lines) + "\n")
        self.logger.info(
            f"Wrote {name}.meteo ({len(forcing_df)} timesteps, "
            f"{forcing_df['time'].iloc[0]} to {forcing_df['time'].iloc[-1]})"
        )

    def _write_lai(self, d: Path, name: str, start_dt, end_dt):
        """Write .lai file -- leaf area index time series.

        Simple seasonal cycle: LAI=1.0 in winter, LAI=4.0 in summer.

        Format:
            LAI_TS          1
            TIME                    LAI
            #TS                     m2/m2
            YYYY-MM-DD HH:MM       <lai>
            ...
        """
        lines = [
            "LAI_TS\t1",
            "TIME\t\tLAI",
            "#TS\t\tm2/m2",
        ]

        # Generate monthly LAI values over the simulation period
        # Extend slightly beyond to ensure coverage
        lai_start = start_dt.replace(day=1)
        lai_end = end_dt + pd.offsets.MonthBegin(1)
        months = pd.date_range(lai_start, lai_end, freq='MS')

        for dt in months:
            # Simple sinusoidal seasonal cycle (Northern Hemisphere)
            # Peak LAI (~4.0) in July (month 7), minimum (~1.0) in January
            lai = 2.5 + 1.5 * math.cos(2 * math.pi * (dt.month - 7) / 12.0)
            lines.append(f"{dt.strftime('%Y-%m-%d %H:%M')}\t{lai:.2f}")

        (d / f"{name}.lai").write_text("\n".join(lines) + "\n")
        self.logger.debug(f"Wrote {name}.lai ({len(months)} monthly values)")

    def _write_para(self, d: Path, name: str, start_dt, end_dt,
                    timestep, lsm_step, reltol, abstol):
        """Write .para file -- solver and output control parameters.

        Format uses keyword-value pairs matching ShaleHills reference.
        Output intervals use DAILY/MONTHLY/HOURLY keywords or integer seconds.
        """
        # Format times as YYYY-MM-DD HH:MM
        start_str = start_dt.strftime('%Y-%m-%d %H:%M')
        end_str = end_dt.strftime('%Y-%m-%d %H:%M')

        lines = [
            "SIMULATION_MODE     0",
            "INIT_MODE           1",
            "ASCII_OUTPUT        1",
            "WATBAL_OUTPUT       0",
            "WRITE_IC            1",
            f"START               {start_str}",
            f"END                 {end_str}",
            "MAX_SPINUP_YEAR     50",
            f"MODEL_STEPSIZE      {timestep}",
            f"LSM_STEP            {lsm_step}",
            f"{'#' * 80}",
            f"# CVode parameters{' ' * 60}#",
            f"{'#' * 80}",
            f"ABSTOL              {abstol:.1E}",
            f"RELTOL              {reltol:.1E}",
            "INIT_SOLVER_STEP    1E-5",
            "NUM_NONCOV_FAIL     0.0",
            "MAX_NONLIN_ITER     5.0",
            "MIN_NONLIN_ITER     1.0",
            "DECR_FACTOR         1.5",
            "INCR_FACTOR         1.5",
            "MIN_MAXSTEP         10.0",
            f"{'#' * 80}",
            f"# OUTPUT CONTROL{' ' * 62}#",
            f"{'#' * 80}",
            "SURF                DAILY",
            "UNSAT               DAILY",
            "GW                  DAILY",
            "RIVSTG              DAILY",
            "SNOW                DAILY",
            "CMC                 DAILY",
            "INFIL               DAILY",
            "RECHARGE            DAILY",
            "EC                  DAILY",
            "ETT                 DAILY",
            "EDIR                DAILY",
            "RIVFLX0             DAILY",
            "RIVFLX1             DAILY",
            "RIVFLX2             DAILY",
            "RIVFLX3             DAILY",
            "RIVFLX4             DAILY",
            "RIVFLX5             DAILY",
            "SUBFLX              DAILY",
            "SURFFLX             DAILY",
            "IC                  MONTHLY",
        ]
        (d / f"{name}.para").write_text("\n".join(lines) + "\n")
        self.logger.debug(
            f"Wrote {name}.para (dt={timestep}s, output=DAILY)"
        )

    def _write_calib(self, d: Path, name: str):
        """Write .calib file -- calibration multipliers (all 1.0 = no change).

        These are multiplicative factors applied to the base parameter values.
        Includes LSM_CALIBRATION section for Flux-PIHM (Noah LSM).
        """
        lines = [
            "KSATH\t\t1.0",
            "KSATV\t\t1.0",
            "KINF\t\t1.0",
            "KMACSATH\t1.0",
            "KMACSATV\t1.0",
            "DROOT\t\t1.0",
            "DMAC\t\t1.0",
            "POROSITY\t1.0",
            "ALPHA\t\t1.0",
            "BETA\t\t1.0",
            "MACVF\t\t1.0",
            "MACHF\t\t1.0",
            "VEGFRAC\t\t1.0",
            "ALBEDO\t\t1.0",
            "ROUGH\t\t1.0",
            "ROUGH_RIV\t1.0",
            "KRIVH\t\t1.0",
            "RIV_DPTH\t1.0",
            "RIV_WDTH\t1.0",
            "LSM_CALIBRATION",
            "DRIP\t\t1.0",
            "CMCMAX\t\t1.0",
            "RS\t\t1.0",
            "CZIL\t\t1.0",
            "FXEXP\t\t1.0",
            "CFACTR\t\t1.0",
            "RGL\t\t1.0",
            "HS\t\t1.0",
            "REFSMC\t\t1.0",
            "WLTSMC\t\t1.0",
            "SCENARIO",
            "PRCP\t\t1.0",
            "SFCTMP\t\t0.0",
        ]
        (d / f"{name}.calib").write_text("\n".join(lines) + "\n")
        self.logger.debug(f"Wrote {name}.calib")

    def _write_ic(self, d: Path, name: str, soil_depth: float):
        """Write .ic file -- initial conditions (binary).

        The IC file is a binary file containing ic_struct per element followed
        by river_ic_struct per river.

        For the Flux-PIHM (Noah LSM) build with 2 elements and 1 river:
            ic_struct = {cmc, sneqv, surf, unsat, gw,
                         t1, snowh, stc[11], smc[11], swc[11]}  (40 doubles)
            river_ic  = {stage}                                   (1 double)

        Total: 2 * 40 + 1 = 81 doubles = 648 bytes.
        """
        import struct

        # Mountain catchment initial conditions
        init_gw_frac = float(self._get_pihm_cfg('INIT_GW_FRAC', 0.5))
        init_satn = 0.5   # 50% saturation of unsaturated zone
        gw = soil_depth * init_gw_frac
        deficit = soil_depth - gw
        unsat = init_satn * deficit

        # Noah LSM initial state
        MAXLYR = 11
        porosity = float(self._get_pihm_cfg('POROSITY', 0.4))
        min_smc = 0.05
        init_smc = min_smc + init_satn * (porosity - min_smc)
        init_temp = 277.15  # ~4°C (annual mean for mountain catchment)

        def _pack_one_elem():
            # Pack: cmc, sneqv, surf, unsat, gw
            d_ = struct.pack('ddddd', 0.0, 0.0, 0.0, unsat, gw)
            # Noah fields: t1, snowh
            d_ += struct.pack('dd', init_temp, 0.0)
            # stc[11] - soil temperature for each layer
            d_ += struct.pack(f'{MAXLYR}d', *([init_temp] * MAXLYR))
            # smc[11] - total soil moisture (m3/m3) for each layer
            d_ += struct.pack(f'{MAXLYR}d', *([init_smc] * MAXLYR))
            # swc[11] - unfrozen soil water content (= smc when above freezing)
            d_ += struct.pack(f'{MAXLYR}d', *([init_smc] * MAXLYR))
            return d_

        # Two elements with identical initial conditions
        data = _pack_one_elem() + _pack_one_elem()

        # River IC: stage
        data += struct.pack('d', 0.1)

        (d / f"{name}.ic").write_bytes(data)
        self.logger.info(
            f"Wrote {name}.ic (Noah format, 2 elements, {len(data)} bytes, "
            f"GW={gw:.2f}m, unsat={unsat:.2f}m, smc={init_smc:.3f})"
        )

    def _write_lsm(self, d: Path, name: str, soil_depth: float):
        """Write .lsm file -- Noah land surface model parameters.

        Required by Flux-PIHM for energy-balance ET computation.
        Defines soil layer structure, radiation mode, and Noah constants.
        """
        # Get site coordinates
        lat = float(self._get_pihm_cfg('LATITUDE', 51.17))
        lon = float(self._get_pihm_cfg('LONGITUDE', -115.57))

        # Define soil layers that sum to soil_depth
        # Use 4 layers with increasing thickness
        if soil_depth <= 1.0:
            layers = [soil_depth / 4] * 4
        else:
            # Layers: 0.1m, 0.3m, 0.6m, remainder
            layers = [0.10, 0.30, 0.60, soil_depth - 1.0]
        nlayers = len(layers)
        layer_str = "  ".join(f"{lyr:.3f}" for lyr in layers)

        # Bottom temperature: annual mean air temp for mountain site
        tbot = float(self._get_pihm_cfg('TBOT', 277.15))  # ~4°C

        lines = [
            f"LATITUDE        {lat:.6f}",
            f"LONGITUDE       {lon:.6f}",
            f"NSOIL           {nlayers}",
            f"SLDPTH_DATA     {layer_str}",
            "RAD_MODE_DATA   0",      # 0=uniform, 1=topographic
            "SBETA_DATA      -2.0",
            "FXEXP_DATA      2.0",
            "CSOIL_DATA      2E6",
            "SALP_DATA       2.6",
            "FRZK_DATA       0.15",
            "ZBOT_DATA       -8.0",
            f"TBOT_DATA       {tbot:.2f}",
            "CZIL_DATA       0.05",
            "LVCOEF_DATA     0.5",
            f"{'#' * 22}",
            "# Print Control      #",
            f"{'#' * 22}",
            "T1              DAILY",
            "STC             DAILY",
            "SMC             DAILY",
            "SH2O            DAILY",
            "SNOWH           DAILY",
            "ALBEDO          DAILY",
            "LE              DAILY",
            "SH              DAILY",
            "G               DAILY",
            "ETP             DAILY",
            "ESNOW           DAILY",
            "ROOTW           DAILY",
            "SOILM           DAILY",
            "SOLAR           DAILY",
            "CH              DAILY",
        ]
        (d / f"{name}.lsm").write_text("\n".join(lines) + "\n")
        self.logger.info(
            f"Wrote {name}.lsm (lat={lat:.2f}, lon={lon:.2f}, "
            f"{nlayers} layers, tbot={tbot:.1f}K)"
        )

    def _write_bc(self, d: Path, name: str):
        """Write .bc file -- boundary conditions.

        Empty for lumped mode with no external boundary conditions.
        """
        (d / f"{name}.bc").write_text("")
        self.logger.debug(f"Wrote {name}.bc (empty -- no external BCs)")
