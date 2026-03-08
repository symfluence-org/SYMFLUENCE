# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Preprocessor

Generates ParFlow-CLM input for a lumped-catchment variably-saturated
groundwater + overland flow + CLM land surface model.

Extends the standalone ParFlow preprocessor with CLM-specific input generation:
- ParFlow .pfidb database with Solver.CLM.* keys enabled
- drv_clmin.dat (CLM driver input file)
- drv_vegm.dat (vegetation map — IGBP land cover fractions)
- drv_vegp.dat (vegetation parameters — LAI, roughness, displacement per PFT)
- CLM-compatible met forcing (1D text files from ERA5)

Output:
    settings/CLMPARFLOW/<runname>.pfidb       -- ParFlow database with CLM keys
    settings/CLMPARFLOW/drv_clmin.dat         -- CLM driver input
    settings/CLMPARFLOW/drv_vegm.dat          -- Vegetation map
    settings/CLMPARFLOW/drv_vegp.dat          -- Vegetation parameters
    data/forcing/CLMPARFLOW_input/forcing.1d            -- Met forcing for CLM
    data/forcing/CLMPARFLOW_input/daily_rainfall.npy    -- daily net rainfall [m/hr]
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("CLMPARFLOW")
class CLMParFlowPreProcessor:
    """Generates ParFlow-CLM input files for integrated hydrologic simulation."""

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

    def _get_cfg(self, key, default=None):
        try:
            if key == 'DOMAIN_NAME':
                return self.config.domain.name
            elif key == 'EXPERIMENT_ID':
                return self.config.domain.experiment_id
        except (AttributeError, TypeError):
            pass
        return default

    def _get_pf_cfg(self, key, default=None):
        """Get config value with CLMPARFLOW_ prefix, typed config first."""
        try:
            pf_cfg = self.config.model.clmparflow
            if pf_cfg:
                attr = key.lower()
                if hasattr(pf_cfg, attr):
                    pydantic_val = getattr(pf_cfg, attr)
                    if pydantic_val is not None:
                        return pydantic_val
        except (AttributeError, TypeError):
            pass
        return default

    def _get_forcing_dir(self) -> Optional[Path]:
        basin_avg = resolve_data_subdir(self.project_dir, 'forcing') / 'basin_averaged_data'
        if basin_avg.exists():
            nc_files = list(basin_avg.glob('*_remapped*.nc'))
            if nc_files:
                return basin_avg

        try:
            forcing_dir = self.config.paths.forcing_path
        except (AttributeError, TypeError):
            forcing_dir = None
        if forcing_dir and forcing_dir != 'default':
            fp = Path(forcing_dir)
            if fp.exists():
                sub = fp / 'basin_averaged_data'
                if sub.exists():
                    return sub
                nc_files = list(fp.glob('*_remapped*.nc'))
                if nc_files:
                    return fp
        return None

    def _get_latitude(self) -> float:
        try:
            lat = self.config.paths.catchment_lat
            if lat and lat != 'center_lat':
                return float(lat)
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            lat = self.config.domain.latitude
        except (AttributeError, TypeError):
            lat = None
        if lat is not None:
            return float(lat)
        try:
            return float(self.config.domain.latitude)
        except (AttributeError, TypeError):
            pass
        forcing_dir = self._get_forcing_dir()
        if forcing_dir:
            try:
                import xarray as xr
                nc_files = sorted(forcing_dir.glob('*.nc'))
                if nc_files:
                    ds = xr.open_dataset(nc_files[0])
                    if 'latitude' in ds:
                        lat = float(ds['latitude'].values.flat[0])
                        ds.close()
                        return lat
                    ds.close()
            except Exception:  # noqa: BLE001 — model execution resilience
                pass
        return 51.36

    def _get_time_info(self):
        import pandas as pd
        try:
            start = self.config.domain.time_start
        except (AttributeError, TypeError):
            start = '2000-01-01'
        try:
            end = self.config.domain.time_end
        except (AttributeError, TypeError):
            end = '2001-01-01'

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        n_hours = int((end_dt - start_dt).total_seconds() / 3600)
        if n_hours <= 0:
            n_hours = 8760

        timestep_hours = float(self._get_pf_cfg('TIMESTEP_HOURS', 1.0))
        n_steps = max(1, int(math.ceil(n_hours / timestep_hours)))

        return n_steps, timestep_hours, n_hours

    def _compute_oudin_pet(self, temp_k, dates, latitude_deg):
        import pandas as pd
        temp_c = temp_k - 273.15
        doy = pd.DatetimeIndex(dates).dayofyear
        lat_rad = np.radians(latitude_deg)

        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
        delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        ws = np.clip(ws, 0.0, np.pi)
        ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta)
            + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )

        lam_rho = 2.45 * 1000.0
        pet_mm_day = np.where(
            temp_c > -5.0,
            ra / lam_rho * (temp_c + 5.0) / 100.0 * 1000.0,
            0.0,
        )
        pet_mm_day = np.maximum(pet_mm_day, 0.0)
        return pet_mm_day / 24.0

    # Elevation bands for multi-band Snow-17 (same as ParFlow)
    ELEVATION_BANDS = [
        (1521, 0.2),
        (1890, 0.2),
        (2099, 0.2),
        (2340, 0.2),
        (2614, 0.2),
    ]
    BASIN_MEAN_ELEV = 2099.0
    DEFAULT_LAPSE_RATE = 0.0065

    def _apply_snow_model(self, ppt_mm_hr, temp_c_hourly, times,
                          lapse_rate=None):
        """Apply elevation-band Snow-17 model to hourly forcing."""
        import pandas as pd
        from jsnow17.bmi import Snow17BMI

        if lapse_rate is None:
            lapse_rate = self.DEFAULT_LAPSE_RATE

        latitude = self._get_latitude()

        time_idx = pd.DatetimeIndex(times)
        df_hourly = pd.DataFrame({
            'ppt': ppt_mm_hr, 'temp': temp_c_hourly
        }, index=time_idx)

        daily = df_hourly.resample('D').agg({
            'ppt': 'sum',
            'temp': 'mean',
        })
        daily['doy'] = daily.index.dayofyear

        n_days = len(daily)
        rpm_weighted = np.zeros(n_days)
        band_info = []

        for band_elev, area_frac in self.ELEVATION_BANDS:
            temp_offset = -lapse_rate * (band_elev - self.BASIN_MEAN_ELEV)
            temp_band = daily['temp'].values + temp_offset

            snow = Snow17BMI(latitude=latitude, dt=1.0)
            snow.initialize()

            rpm_band = snow.update_batch(
                daily['ppt'].values, temp_band, daily['doy'].values,
            )
            rpm_weighted += rpm_band * area_frac

            peak_swe = max(
                float(snow.get_value('swe')),
                float(snow.get_value('w_i') + snow.get_value('w_q'))
            )
            band_info.append(
                f"{band_elev}m: offset={temp_offset:+.1f}°C, "
                f"peak_SWE={peak_swe:.0f}mm"
            )

        daily_rpm = pd.Series(rpm_weighted, index=daily.index)

        effective = np.zeros(len(df_hourly))
        for date, rpm_mm_day in daily_rpm.items():
            mask = df_hourly.index.normalize() == date
            n_hours = mask.sum()
            if n_hours > 0:
                effective[mask] = rpm_mm_day / n_hours

        self.logger.info(
            f"Snow-17 elevation bands (lapse={lapse_rate*1000:.1f} °C/km):"
        )
        for info in band_info:
            self.logger.info(f"  {info}")
        self.logger.info(
            f"  Area-weighted: mean rain+melt={np.mean(rpm_weighted):.2f} mm/day, "
            f"mean effective={np.mean(effective):.4f} mm/hr"
        )
        return effective

    def _prepare_daily_rainfall(self, settings_dir):
        """Compute daily effective rainfall from ERA5 with snow model.

        Reuses same logic as ParFlow preprocessor for consistency.
        """
        import pandas as pd

        forcing_dir = self._get_forcing_dir()
        if forcing_dir is None:
            self.logger.warning(
                "No forcing data found. CLMParFlow will run without rainfall."
            )
            return None

        try:
            import xarray as xr
            nc_files = sorted(forcing_dir.glob('*.nc'))
            nc_files = [f for f in nc_files
                        if 'ERA5' in f.name or 'remapped' in f.name]
            if not nc_files:
                nc_files = sorted(forcing_dir.glob('*.nc'))

            self.logger.info(
                f"Reading forcing from {len(nc_files)} files"
            )

            all_ppt, all_temp, all_times = [], [], []
            for nc_file in nc_files:
                ds = xr.open_dataset(nc_file)
                if 'pptrate' not in ds:
                    ds.close()
                    continue
                all_ppt.append(ds['pptrate'].values[:, 0])
                all_temp.append(ds['airtemp'].values[:, 0])
                all_times.append(ds['time'].values)
                ds.close()

            if not all_ppt:
                self.logger.warning("No ERA5 files with pptrate found")
                return None

            pptrate = np.concatenate(all_ppt)
            airtemp = np.concatenate(all_temp)
            times = np.concatenate(all_times)
            sort_idx = np.argsort(times)
            pptrate, airtemp, times = pptrate[sort_idx], airtemp[sort_idx], times[sort_idx]

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not read ERA5 forcing: {e}")
            return None

        try:
            start = self.config.domain.time_start
        except (AttributeError, TypeError):
            start = '2000-01-01'
        try:
            end = self.config.domain.time_end
        except (AttributeError, TypeError):
            end = '2001-01-01'

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        time_idx = pd.DatetimeIndex(times)
        mask = (time_idx >= start_dt) & (time_idx < end_dt)
        pptrate, airtemp, times = pptrate[mask], airtemp[mask], times[mask]

        if len(pptrate) == 0:
            self.logger.warning("No forcing data in simulation period")
            return None

        latitude = self._get_latitude()
        pet_mm_hr = self._compute_oudin_pet(airtemp, times, latitude)
        ppt_mm_hr = np.maximum(pptrate * 3600.0, 0.0)
        temp_c = airtemp - 273.15

        effective_mm_hr = self._apply_snow_model(ppt_mm_hr, temp_c, times)

        np.savez_compressed(
            settings_dir / 'hourly_forcing_cache.npz',
            ppt_mm_hr=ppt_mm_hr, pet_mm_hr=pet_mm_hr,
            temp_c=temp_c, times=times.astype('int64'),
        )

        # Also save raw forcing for CLM met file generation
        self._write_clm_met_forcing(settings_dir, pptrate, airtemp, times)

        time_idx = pd.DatetimeIndex(times)
        df = pd.DataFrame({
            'effective': effective_mm_hr, 'pet': pet_mm_hr
        }, index=time_idx)

        daily = df.resample('D').agg(['mean', 'count'])
        days = []
        for dt, row in daily.iterrows():
            eff_mean = row[('effective', 'mean')]
            pet_mean = row[('pet', 'mean')]
            n_hours = int(row[('effective', 'count')])
            if n_hours == 0:
                continue
            net_mm_hr = eff_mean - pet_mean
            net_m_hr = net_mm_hr * 0.001
            label = f"d{dt.strftime('%Y%m%d')}"
            days.append((label, n_hours, net_m_hr))

        if not days:
            self.logger.warning("No daily data computed")
            return None

        rainfall_arr = np.array([(d[1], d[2]) for d in days])
        np.save(settings_dir / 'daily_rainfall.npy', rainfall_arr)

        total_eff = df['effective'].mean()
        total_pet = df['pet'].mean()
        self.logger.info(
            f"Prepared {len(days)} daily rainfall entries (with snow model): "
            f"mean effective={total_eff:.4f} mm/hr, mean PET={total_pet:.4f} mm/hr, "
            f"mean net={(total_eff - total_pet):.4f} mm/hr"
        )
        return days

    def _write_clm_met_forcing(self, settings_dir, pptrate, airtemp, times):
        """Write CLM 1D met forcing file from ERA5 data.

        CLM expects hourly forcing with columns:
        DSWR DLWR APCP TEMP UGRD VGRD Press SPFH
        (shortwave, longwave, precip, temp, u-wind, v-wind, pressure, humidity)
        """
        n = len(pptrate)
        # Approximate missing variables with defaults
        dswr = np.full(n, 200.0)     # W/m2 shortwave (approximate)
        dlwr = np.full(n, 300.0)     # W/m2 longwave (approximate)
        apcp = pptrate * 3600.0      # mm/hr precip
        temp = airtemp               # K temperature
        ugrd = np.full(n, 2.0)       # m/s u-wind
        vgrd = np.full(n, 0.5)       # m/s v-wind
        press = np.full(n, 80000.0)  # Pa surface pressure (~800 hPa for mountain)
        spfh = np.full(n, 0.004)     # kg/kg specific humidity

        forcing_data = np.column_stack([dswr, dlwr, apcp, temp, ugrd, vgrd, press, spfh])

        forcing_file = settings_dir / 'forcing.1d'
        # ParFlow-CLM 1D forcing: no header line, pure numeric data
        np.savetxt(forcing_file, forcing_data, fmt='%.6e')

        self.logger.info(f"Wrote CLM met forcing: {forcing_file} ({n} timesteps)")

    def _write_drv_clmin(self, settings_dir, nx, ny, n_steps, timestep_hours):
        """Write CLM driver input file (drv_clmin.dat).

        Uses Fortran fixed-format: 15-char variable name field, then value.
        Variable names must match exactly what drv_readclmin.F90 expects.
        """
        import pandas as pd

        try:
            start = self.config.domain.time_start
        except (AttributeError, TypeError):
            start = '2000-01-01'
        start_dt = pd.Timestamp(str(start))

        try:
            end = self.config.domain.time_end
        except (AttributeError, TypeError):
            end = '2009-12-31'
        end_dt = pd.Timestamp(str(end))

        metfile = str(self._get_pf_cfg('METFILE', 'forcing.1d'))

        def f(name, val, desc=''):
            """Format a drv_clmin.dat line: 15-char name field + value + description."""
            return f"{name:<15s}{val!s:<40s}{desc}"

        lines = [
            "!=========================================================================",
            "! drv_clmin.dat: CLM driver input for SYMFLUENCE CLMParFlow",
            "! Generated by CLMParFlowPreProcessor",
            "!=========================================================================",
            "!",
            "! CLM Domain (Read into 1D drv_module variables):",
            "!",
            f("maxt", 1, "Maximum tiles per grid"),
            f("mina", 0.05, "Min grid area for tile (%)"),
            f("udef", -9999., "Undefined value"),
            f("vclass", 2, "Vegetation Classification (1=UMD, 2=IGBP)"),
            "!",
            "! CLM Files:",
            "!",
            f("vegtf", "drv_vegm.dat", "Vegetation Tile Specification File"),
            f("vegpf", "drv_vegp.dat", "Vegetation Type Parameter File"),
            f("metf1d", metfile, "Meteorological forcing file (1D)"),
            f("outf1d", "clm_output.txt", "CLM output file"),
            f("poutf1d", "clm_para.out.dat", "CLM 1D Parameter Output File"),
            f("rstf", "clm_rst.", "CLM active restart file"),
            "!",
            "! Run timing parameters:",
            "!",
            f("startcode", 2, "1=restart file, 2=defined"),
            f("sss", "00", "Starting Second"),
            f("smn", "00", "Starting Minute"),
            f("shr", f"{start_dt.hour:02d}", "Starting Hour"),
            f("sda", f"{start_dt.day:02d}", "Starting Day"),
            f("smo", f"{start_dt.month:02d}", "Starting Month"),
            f("syr", start_dt.year, "Starting Year"),
            "!",
            f("ess", "00", "Ending Second"),
            f("emn", "00", "Ending Minute"),
            f("ehr", "00", "Ending Hour"),
            f("eda", f"{end_dt.day:02d}", "Ending Day"),
            f("emo", f"{end_dt.month:02d}", "Ending Month"),
            f("eyr", end_dt.year, "Ending Year"),
            "!",
            "! IC Source: (1) restart file, (2) drv_clmin.dat (this file)",
            "!",
            f("clm_ic", 2, "1=restart file, 2=defined"),
            "!",
            "! CLM initial conditions:",
            "!",
            f("t_ini", 280., "Initial temperature [K]"),
            f("h2osno_ini", 0., "Initial snow cover [mm]"),
            "!",
            "! Diagnostic output:",
            "!",
            f("surfind", 2, "Number of surface diagnostic vars"),
            f("soilind", 1, "Number of soil layer diagnostic vars"),
            f("snowind", 0, "Number of snow layer diagnostic vars"),
            "!",
            "! CLM Forcing parameters:",
            "!",
            f("forc_hgt_u", 10.0, "Observational height of wind [m]"),
            f("forc_hgt_t", 2.0, "Observational height of temperature [m]"),
            f("forc_hgt_q", 2.0, "Observational height of humidity [m]"),
            "!",
            "! CLM Vegetation parameters:",
            "!",
            f("dewmx", 0.1, "Maximum allowed dew [mm]"),
            f("rootfr", -9999.0, "Root Fraction (use default)"),
            "!",
            "! Roughness lengths:",
            "!",
            f("zlnd", 0.01, "Roughness length for soil [m]"),
            f("zsno", 0.0024, "Roughness length for snow [m]"),
            f("csoilc", 0.0025, "Drag coefficient for soil under canopy [-]"),
            "!",
            "! Numerical parameters:",
            "!",
            f("capr", 0.34, "Tuning factor for surface T"),
            f("cnfac", 0.5, "Crank Nicholson factor [0-1]"),
            f("smpmin", -1.e8, "Min soil potential restriction (mm)"),
            f("ssi", 0.033, "Irreducible water saturation of snow"),
            f("wimp", 0.05, "Water impermeable if porosity < wimp"),
        ]

        clmin_path = settings_dir / 'drv_clmin.dat'
        clmin_path.write_text('\n'.join(lines) + '\n')
        self.logger.info(f"Wrote CLM driver input: {clmin_path}")

    def _write_drv_vegm(self, settings_dir, nx, ny):
        """Write vegetation map file (drv_vegm.dat).

        IGBP land cover fractions per grid cell.
        For lumped Bow River catchment: mixed forest + alpine meadow.
        18 IGBP types + bare soil.
        """
        lines = [
            f"! Vegetation map for CLMParFlow ({nx}x{ny} grid, IGBP land cover fractions)",
            "! x  y  lat  lon  sand clay color fractional_coverage[1:18] bare",
        ]

        lat = self._get_latitude()
        lon = -115.57  # Bow River at Banff approximate longitude

        for j in range(ny):
            for i in range(nx):
                # Bow River: mixed forest (type 5) + open shrub (type 7) + barren/alpine (type 16)
                fracs = [0.0] * 18
                fracs[4] = 0.35   # Type 5: Mixed forest
                fracs[6] = 0.20   # Type 7: Open shrublands
                fracs[9] = 0.15   # Type 10: Grasslands
                fracs[12] = 0.10  # Type 13: Urban (bare rock/alpine)
                fracs[15] = 0.10  # Type 16: Barren
                fracs[0] = 0.10   # Type 1: Evergreen needleleaf

                bare = 1.0 - sum(fracs)
                frac_str = ' '.join(f'{f:.4f}' for f in fracs)
                lines.append(
                    f"{i+1:3d} {j+1:3d} {lat:.4f} {lon:.4f} "
                    f"0.40 0.20 4 {frac_str} {bare:.4f}"
                )

        vegm_path = settings_dir / 'drv_vegm.dat'
        vegm_path.write_text('\n'.join(lines) + '\n')
        self.logger.info(f"Wrote vegetation map: {vegm_path}")

    def _write_drv_vegp(self, settings_dir):
        """Write vegetation parameters file (drv_vegp.dat).

        Uses the standard CLM format: 15-char variable name line followed by
        values for all 18 IGBP plant functional types. Comment lines start with !.
        Format matches drv_readvegpf.F90 which reads: read(9,'(a15)')vname.
        """
        # Standard CLM vegetation parameters from reference drv_vegp.dat
        # Each entry: (variable_name, description, values_for_18_types)
        params = [
            ("itypwat", "(1-soil, 2-land ice, 3-deep lake, 4-shallow lake, 5-wetland)",
             "1 1 1 1 1 1 1 1 1 1 5 1 1 1 2 1 3 1"),
            ("lai", "Maximum leaf area index [-]",
             "6.00 6.00 6.00 6.00 6.00 6.00 6.00 6.00 6.00 2.00 6.00 6.00 5.00 6.00 0.00 6.00 0.00 0.00"),
            ("lai0", "Minimum leaf area index [-]",
             "5.00 5.00 1.00 1.00 3.00 2.00 1.00 2.00 1.00 0.50 0.50 0.50 1.00 2.00 0.00 0.50 0.00 0.00"),
            ("sai", "Stem area index [-]",
             "2.00 2.00 2.00 2.00 2.00 2.00 2.00 2.00 2.00 4.00 2.00 0.50 2.00 2.00 2.00 2.00 2.00 0.00"),
            ("z0m", "Aerodynamic roughness length [m]",
             "1.00 2.20 1.00 0.80 0.80 0.10 0.10 0.70 0.10 0.03 0.03 0.06 0.50 0.06 0.01 0.05 0.002 0.01"),
            ("displa", "Displacement height [m]",
             "11.0 23.00 11.0 13.0 13.0 0.30 0.30 6.50 0.70 0.30 0.30 0.30 3.00 0.30 0.00 0.10 0.00 0.00"),
            ("dleaf", "Leaf dimension [m]",
             "0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.00"),
            ("roota", "Fitted numerical index of rooting distribution",
             "7.00 7.00 7.00 6.00 5.00 6.00 5.00 6.00 5.00 1.00 6.00 6.00 5.00 5.00 0.00 5.00 0.00 0.00"),
            ("rootb", "Fitted numerical index of rooting distribution",
             "2.00 1.00 2.00 2.00 1.50 1.50 2.50 2.50 1.00 2.50 2.00 2.50 2.00 2.00 0.00 2.00 0.00 0.00"),
            ("rhol_vis", "Leaf reflectance vis",
             "0.07 0.10 0.07 0.10 0.08 0.08 0.08 0.09 0.11 0.11 0.11 0.11 0.09 0.09 -99. 0.09 -99. -99."),
            ("rhol_nir", "Leaf reflectance nir",
             "0.35 0.45 0.35 0.45 0.40 0.40 0.40 0.49 0.58 0.58 0.35 0.58 0.47 0.47 -99. 0.47 -99. -99."),
            ("rhos_vis", "Stem reflectance vis",
             "0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.26 0.36 0.36 0.36 0.36 0.24 0.24 -99. 0.24 -99. -99."),
            ("rhos_nir", "Stem reflectance nir",
             "0.39 0.39 0.39 0.39 0.39 0.39 0.39 0.48 0.58 0.58 0.39 0.58 0.47 0.47 -99. 0.47 -99. -99."),
            ("taul_vis", "Leaf transmittance vis",
             "0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.06 0.07 0.07 0.07 0.07 0.06 0.06 -99. 0.06 -99. -99."),
            ("taul_nir", "Leaf transmittance nir",
             "0.10 0.25 0.10 0.25 0.17 0.17 0.17 0.21 0.25 0.25 0.10 0.25 0.20 0.20 -99. 0.20 -99. -99."),
            ("taus_vis", "Stem transmittance vis",
             "0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.11 0.22 0.22 0.22 0.22 0.09 0.09 -99. 0.09 -99. -99."),
            ("taus_nir", "Stem transmittance nir",
             "0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.19 0.38 0.38 0.001 0.38 0.15 0.15 -99. 0.15 -99. -99."),
            ("xl", "Leaf/stem orientation index",
             "0.01 0.10 0.01 0.25 0.13 0.13 0.13 -0.08 -0.3 -0.3 -0.3 -0.3 -0.07 -0.07 -99. -0.07 -99. -99."),
            ("vw", "btran exponent",
             "1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. -99. 1. -99. -99."),
            ("irrig", "(irrig=0 -> no irrigation, irrig=1 -> irrigate)",
             "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"),
        ]

        lines = [
            "! Vegetation parameters for CLMParFlow",
            "! 18 IGBP plant functional types",
            "!",
        ]

        for vname, desc, vals in params:
            # 15-char left-justified variable name followed by description
            lines.append(f"{vname:<15s}!{desc}")
            lines.append(vals)
            lines.append("!")

        vegp_path = settings_dir / 'drv_vegp.dat'
        vegp_path.write_text('\n'.join(lines) + '\n')
        self.logger.info(f"Wrote vegetation parameters: {vegp_path}")

    def run_preprocessing(self):
        """Generate ParFlow-CLM .pfidb input file with CLM keys + CLM input files."""
        settings_dir = self.project_dir / "settings" / "CLMPARFLOW"
        settings_dir.mkdir(parents=True, exist_ok=True)

        forcing_output_dir = resolve_data_subdir(
            self.project_dir, 'forcing'
        ) / 'CLMPARFLOW_input'
        forcing_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating CLMParFlow settings in {settings_dir}")
        self.logger.info(f"Generating CLMParFlow forcing in {forcing_output_dir}")

        nx = int(self._get_pf_cfg('NX', 3))
        ny = int(self._get_pf_cfg('NY', 1))
        nz = int(self._get_pf_cfg('NZ', 1))
        dx = float(self._get_pf_cfg('DX', 1000.0))
        dy = float(self._get_pf_cfg('DY', 1000.0))
        dz = float(self._get_pf_cfg('DZ', 2.0))
        top = float(self._get_pf_cfg('TOP', 2.0))
        bot = float(self._get_pf_cfg('BOT', 0.0))
        k_sat = float(self._get_pf_cfg('K_SAT', 5.0))
        porosity = float(self._get_pf_cfg('POROSITY', 0.4))
        vg_alpha = float(self._get_pf_cfg('VG_ALPHA', 1.0))
        vg_n = float(self._get_pf_cfg('VG_N', 2.0))
        s_res = float(self._get_pf_cfg('S_RES', 0.1))
        s_sat = float(self._get_pf_cfg('S_SAT', 1.0))
        ss = float(self._get_pf_cfg('SS', 1e-5))
        mannings_n = float(self._get_pf_cfg('MANNINGS_N', 0.03))
        slope_x = float(self._get_pf_cfg('SLOPE_X', 0.01))
        initial_pressure = self._get_pf_cfg('INITIAL_PRESSURE')
        solver = str(self._get_pf_cfg('SOLVER', 'Richards'))

        n_steps, timestep_hours, n_hours = self._get_time_info()
        dump_interval = float(self._get_pf_cfg('DUMP_INTERVAL_HOURS', 24.0))

        if initial_pressure is None:
            initial_pressure = -1.0
        else:
            initial_pressure = float(initial_pressure)

        self.logger.info(
            f"CLMParFlow grid: {nx}x{ny}x{nz}, dx={dx:.0f}m, dz={dz:.1f}m, "
            f"K_sat={k_sat}, porosity={porosity}, slope={slope_x}"
        )
        self.logger.info(f"Timesteps: {n_steps} x {timestep_hours} hours")

        # Prepare daily rainfall (also writes CLM met forcing to forcing dir)
        rainfall_entries = self._prepare_daily_rainfall(forcing_output_dir)

        runname = self.domain_name

        common_params = dict(
            nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
            top=top, bot=bot, k_sat=k_sat, porosity=porosity,
            vg_alpha=vg_alpha, vg_n=vg_n, s_res=s_res, s_sat=s_sat,
            ss=ss, mannings_n=mannings_n, slope_x=slope_x,
            initial_pressure=initial_pressure,
            solver=solver, n_steps=n_steps, timestep_hours=timestep_hours,
            dump_interval=dump_interval, rainfall_entries=rainfall_entries,
        )

        # Write .pfidb with CLM keys enabled
        self._write_pfidb_text(settings_dir, runname, **common_params)

        # Write CLM-specific input files
        self._write_drv_clmin(settings_dir, nx, ny, n_steps, timestep_hours)
        self._write_drv_vegm(settings_dir, nx, ny)
        self._write_drv_vegp(settings_dir)

        self.logger.info(f"CLMParFlow input files generated in {settings_dir}")

    def _write_pfidb_text(self, settings_dir, runname, **params):
        """Write plain-text .pfidb with daily rainfall cycle + CLM keys."""
        entries = []

        def add(key, value):
            if isinstance(value, bool):
                entries.append((key, 'True' if value else 'False'))
            elif isinstance(value, float):
                entries.append((key, f'{value:g}'))
            else:
                entries.append((key, str(value)))

        add('FileVersion', 4)
        add('Process.Topology.P', 1)
        add('Process.Topology.Q', 1)
        add('Process.Topology.R', 1)

        add('ComputationalGrid.Lower.X', 0.0)
        add('ComputationalGrid.Lower.Y', 0.0)
        add('ComputationalGrid.Lower.Z', params['bot'])
        add('ComputationalGrid.NX', params['nx'])
        add('ComputationalGrid.NY', params['ny'])
        add('ComputationalGrid.NZ', params['nz'])
        add('ComputationalGrid.DX', params['dx'])
        add('ComputationalGrid.DY', params['dy'])
        add('ComputationalGrid.DZ', params['dz'])

        add('GeomInput.Names', 'domain_input')
        add('GeomInput.domain_input.GeomName', 'domain')
        add('GeomInput.domain_input.InputType', 'Box')
        add('Geom.domain.Lower.X', 0.0)
        add('Geom.domain.Lower.Y', 0.0)
        add('Geom.domain.Lower.Z', params['bot'])
        add('Geom.domain.Upper.X', params['nx'] * params['dx'])
        add('Geom.domain.Upper.Y', params['ny'] * params['dy'])
        add('Geom.domain.Upper.Z', params['top'])
        add('Geom.domain.Patches',
            'x_lower x_upper y_lower y_upper z_lower z_upper')

        add('Geom.Perm.Names', 'domain')
        add('Geom.domain.Perm.Type', 'Constant')
        add('Geom.domain.Perm.Value', params['k_sat'])
        add('Perm.TensorType', 'TensorByGeom')
        add('Geom.Perm.TensorByGeom.Names', 'domain')
        add('Geom.domain.Perm.TensorValX', 1.0)
        add('Geom.domain.Perm.TensorValY', 1.0)
        add('Geom.domain.Perm.TensorValZ', 1.0)

        add('Geom.Porosity.GeomNames', 'domain')
        add('Geom.domain.Porosity.Type', 'Constant')
        add('Geom.domain.Porosity.Value', params['porosity'])

        add('SpecificStorage.Type', 'Constant')
        add('SpecificStorage.GeomNames', 'domain')
        add('Geom.domain.SpecificStorage.Value', params['ss'])

        add('Phase.RelPerm.Type', 'VanGenuchten')
        add('Phase.RelPerm.GeomNames', 'domain')
        add('Geom.domain.RelPerm.Alpha', params['vg_alpha'])
        add('Geom.domain.RelPerm.N', params['vg_n'])

        add('Phase.Saturation.Type', 'VanGenuchten')
        add('Phase.Saturation.GeomNames', 'domain')
        add('Geom.domain.Saturation.Alpha', params['vg_alpha'])
        add('Geom.domain.Saturation.N', params['vg_n'])
        add('Geom.domain.Saturation.SRes', params['s_res'])
        add('Geom.domain.Saturation.SSat', params['s_sat'])

        add('Phase.Names', 'water')
        add('Phase.water.Density.Type', 'Constant')
        add('Phase.water.Density.Value', 1.0)
        add('Phase.water.Viscosity.Type', 'Constant')
        add('Phase.water.Viscosity.Value', 1.0)
        add('Contaminants.Names', '')
        add('Gravity', 1.0)

        add('Solver', params['solver'])
        add('Solver.MaxIter', 100000)
        add('Solver.Drop', 1e-20)
        add('Solver.AbsTol', 1e-8)
        add('Solver.MaxConvergenceFailures', 8)
        add('Solver.Nonlinear.MaxIter', 80)
        add('Solver.Nonlinear.ResidualTol', 1e-6)
        add('Solver.Nonlinear.EtaChoice', 'Walker1')
        add('Solver.Nonlinear.EtaValue', 0.001)
        add('Solver.Nonlinear.UseJacobian', True)
        add('Solver.Nonlinear.DerivativeEpsilon', 1e-14)
        add('Solver.Nonlinear.StepTol', 1e-30)
        add('Solver.Nonlinear.Globalization', 'LineSearch')
        add('Solver.Linear.KrylovDimension', 80)
        add('Solver.Linear.MaxRestarts', 2)
        add('Solver.Linear.Preconditioner', 'MGSemi')

        # === CLM-specific solver keys ===
        add('Solver.LSM', 'CLM')
        add('Solver.CLM.MetForcing', '1D')
        add('Solver.CLM.MetFileName', 'forcing.1d')
        add('Solver.CLM.MetFilePath', './')
        add('Solver.CLM.IstepStart', 1)
        add('Solver.CLM.EvapBeta', 'Linear')
        add('Solver.CLM.VegWaterStress', 'Saturation')
        add('Solver.CLM.ResSat', params['s_res'])
        add('Solver.CLM.WiltingPoint', 0.1)
        add('Solver.CLM.FieldCapacity', 0.98)
        add('Solver.CLM.IrrigationType', 'none')
        add('Solver.CLM.RootZoneNZ', params['nz'])
        add('Solver.CLM.SoiLayer', 7)
        add('Solver.CLM.WriteLogs', False)
        add('Solver.CLM.WriteLastRST', True)
        add('Solver.CLM.DailyRST', False)
        add('Solver.CLM.SingleFile', True)
        add('Solver.CLM.BinaryOutDir', False)

        stop_time = float(params['n_steps'] * params['timestep_hours'])
        add('TimingInfo.BaseUnit', 1.0)
        add('TimingInfo.StartCount', 0)
        add('TimingInfo.StartTime', 0.0)
        add('TimingInfo.StopTime', stop_time)
        add('TimingInfo.DumpInterval', params['dump_interval'])

        add('TimeStep.Type', 'Constant')
        add('TimeStep.Value', params['timestep_hours'])

        # Build cycles
        rainfall = params.get('rainfall_entries')
        if rainfall and len(rainfall) > 0:
            rain_labels = [r[0] for r in rainfall]
            add('Cycle.Names', 'constant raincycle')
            add('Cycle.constant.Names', 'alltime')
            add('Cycle.constant.alltime.Length', 1)
            add('Cycle.constant.Repeat', -1)
            add('Cycle.raincycle.Names', ' '.join(rain_labels))
            for label, hours, _rate in rainfall:
                add(f'Cycle.raincycle.{label}.Length', hours)
            add('Cycle.raincycle.Repeat', -1)
        else:
            rain_labels = None
            add('Cycle.Names', 'constant')
            add('Cycle.constant.Names', 'alltime')
            add('Cycle.constant.alltime.Length', 1)
            add('Cycle.constant.Repeat', -1)

        # Boundary conditions
        add('BCPressure.PatchNames',
            'x_lower x_upper y_lower y_upper z_lower z_upper')

        for patch in ['x_lower', 'x_upper', 'y_lower', 'y_upper', 'z_lower']:
            add(f'Patch.{patch}.BCPressure.Type', 'FluxConst')
            add(f'Patch.{patch}.BCPressure.Cycle', 'constant')
            add(f'Patch.{patch}.BCPressure.alltime.Value', 0.0)

        add('Patch.z_upper.BCPressure.Type', 'OverlandFlow')
        if rain_labels:
            add('Patch.z_upper.BCPressure.Cycle', 'raincycle')
            for label, _hours, rate in rainfall:
                add(f'Patch.z_upper.BCPressure.{label}.Value', -rate)
        else:
            add('Patch.z_upper.BCPressure.Cycle', 'constant')
            add('Patch.z_upper.BCPressure.alltime.Value', 0.0)

        add('Mannings.Type', 'Constant')
        add('Mannings.GeomNames', 'domain')
        add('Mannings.Geom.domain.Value', params['mannings_n'])

        add('ICPressure.Type', 'HydroStaticPatch')
        add('ICPressure.GeomNames', 'domain')
        add('Geom.domain.ICPressure.Value', params['initial_pressure'])
        add('Geom.domain.ICPressure.RefGeom', 'domain')
        add('Geom.domain.ICPressure.RefPatch', 'z_lower')

        add('Domain.GeomName', 'domain')
        add('Geom.Retardation.GeomNames', '')

        add('PhaseSources.water.Type', 'Constant')
        add('PhaseSources.water.GeomNames', 'domain')
        add('PhaseSources.water.Geom.domain.Value', 0.0)

        add('TopoSlopesX.Type', 'Constant')
        add('TopoSlopesX.GeomNames', 'domain')
        add('TopoSlopesX.Geom.domain.Value', params['slope_x'])
        add('TopoSlopesY.Type', 'Constant')
        add('TopoSlopesY.GeomNames', 'domain')
        add('TopoSlopesY.Geom.domain.Value', 0.0)

        add('KnownSolution', 'NoKnownSolution')
        add('Wells.Names', '')

        add('Solver.PrintSubsurfData', True)
        add('Solver.PrintPressure', True)
        add('Solver.PrintSaturation', True)
        add('Solver.PrintVelocities', False)
        add('Solver.PrintOverlandSum', True)
        add('Solver.PrintCLM', True)

        parts = [str(len(entries))]
        for key, val in entries:
            parts.append(str(len(key)))
            parts.append(key)
            parts.append(str(len(val)))
            parts.append(val)

        pfidb_path = settings_dir / f'{runname}.pfidb'
        pfidb_path.write_text('\n'.join(parts) + '\n')

        (settings_dir / 'runname.txt').write_text(runname)

        n_entries = len(entries)
        self.logger.info(
            f"Generated CLMParFlow .pfidb (text): {pfidb_path} "
            f"({n_entries} entries, CLM enabled)"
        )
