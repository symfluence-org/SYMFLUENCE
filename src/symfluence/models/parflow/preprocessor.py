# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Preprocessor

Generates ParFlow input for a lumped-catchment variably-saturated
groundwater + overland flow model.

Uses a 3×1×1 horizontal grid (NX=3) to avoid ParFlow's isolated-cell
issue with single-cell domains.  Time-varying rainfall is applied via
the OverlandFlow boundary condition at z_upper using a daily cycle
derived from ERA5 basin-averaged forcing.

Output:
    settings/PARFLOW/<runname>.pfidb       -- ParFlow database file
    settings/PARFLOW/daily_rainfall.npy    -- daily net rainfall [m/hr]
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("PARFLOW")
class ParFlowPreProcessor:
    """Generates ParFlow input files for integrated hydrologic simulation."""

    def __init__(self, config, logger, **kwargs):
        self.config = config
        self.logger = logger

        try:
            self.domain_name = config.domain.name or 'unknown'
        except (AttributeError, TypeError):
            self.domain_name = 'unknown'
        try:
            self.experiment_id = config.domain.experiment_id or 'default'
        except (AttributeError, TypeError):
            self.experiment_id = 'default'

        # PROJECT_DIR is a derived path — check dict sources first, then compute
        _dict = config.to_dict(flatten=True) if hasattr(config, 'to_dict') else (config if isinstance(config, dict) else {})
        project_dir = (_dict or {}).get('PROJECT_DIR')
        if not project_dir or project_dir == '.':
            try:
                data_dir = str(config.system.data_dir)
            except (AttributeError, TypeError):
                data_dir = '.'
            project_dir = str(Path(data_dir) / f"domain_{self.domain_name}")
        self.project_dir = Path(project_dir)

    def _get_pf_cfg(self, key, default=None):
        """Get ParFlow-specific config value from typed config."""
        try:
            pf_cfg = self.config.model.parflow
            if pf_cfg:
                attr = key.lower()
                if hasattr(pf_cfg, attr):
                    val = getattr(pf_cfg, attr)
                    if val is not None:
                        return val
        except (AttributeError, TypeError):
            pass
        return default

    def _get_forcing_dir(self) -> Optional[Path]:
        basin_avg = resolve_data_subdir(self.project_dir, 'forcing') / 'basin_averaged_data'
        if basin_avg.exists():
            nc_files = list(basin_avg.glob('*_remapped*.nc'))
            if nc_files:
                return basin_avg

        forcing_dir = getattr(self.config.paths, 'forcing_path', None)
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
        lat = getattr(self.config.domain, 'latitude', None)
        if lat is not None:
            return float(lat)
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
        start = self.config.domain.time_start or '2000-01-01'
        end = self.config.domain.time_end or '2001-01-01'

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

    # Elevation bands for multi-band Snow-17 (Bow at Banff: 787–3528 m)
    # 5 equal-area bands from hypsometry (P10/P30/P50/P70/P90)
    ELEVATION_BANDS = [
        (1521, 0.2),  # Band 1: low elevation, rain-dominated
        (1890, 0.2),  # Band 2
        (2099, 0.2),  # Band 3: basin mean (reference)
        (2340, 0.2),  # Band 4
        (2614, 0.2),  # Band 5: high elevation, snow-dominated
    ]
    BASIN_MEAN_ELEV = 2099.0  # m, reference elevation for lapse rate
    DEFAULT_LAPSE_RATE = 0.0065  # °C/m, standard environmental lapse rate

    def _apply_snow_model(self, ppt_mm_hr, temp_c_hourly, times,
                          lapse_rate=None):
        """Apply elevation-band Snow-17 model to hourly forcing.

        Runs 5 Snow-17 instances at different elevations with lapse-rate
        adjusted temperatures. Low-elevation bands melt earlier (March-April),
        high-elevation bands melt later (June-July), producing a broad
        spring-summer freshet.

        Args:
            ppt_mm_hr: hourly precipitation (mm/hr)
            temp_c_hourly: hourly temperature at basin mean elevation (°C)
            times: array of timestamps
            lapse_rate: temperature lapse rate (°C/m), default 0.0065

        Returns:
            effective_mm_hr: hourly liquid water input (rain + snowmelt), mm/hr
        """
        import pandas as pd
        from jsnow17.bmi import Snow17BMI

        if lapse_rate is None:
            lapse_rate = self.DEFAULT_LAPSE_RATE

        latitude = self._get_latitude()

        # Aggregate to daily for Snow-17 (operates on daily timestep)
        time_idx = pd.DatetimeIndex(times)
        df_hourly = pd.DataFrame({
            'ppt': ppt_mm_hr, 'temp': temp_c_hourly
        }, index=time_idx)

        daily = df_hourly.resample('D').agg({
            'ppt': 'sum',    # total mm/day
            'temp': 'mean',  # mean daily temperature
        })
        daily['doy'] = daily.index.dayofyear

        # Run Snow-17 for each elevation band with lapse-rate adjusted temps
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

        # Distribute daily rain+melt back to hourly (uniform within each day)
        daily_rpm = pd.Series(rpm_weighted, index=daily.index)

        # Map daily totals back to hourly rates (mm/day -> mm/hr)
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

        Uses a degree-day snow model to partition rain/snow and melt
        the snowpack, producing realistic seasonal timing for
        snowmelt-dominated basins.

        Returns:
            List of (day_label, hours_in_day, net_rainfall_m_hr) tuples,
            or None if forcing data is unavailable.
        """
        import pandas as pd

        forcing_dir = self._get_forcing_dir()
        if forcing_dir is None:
            self.logger.warning(
                "No forcing data found. ParFlow will run without rainfall."
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
                f"Reading ERA5 forcing from {len(nc_files)} files"
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

        # Clip to simulation window
        start = self.config.domain.time_start or '2000-01-01'
        end = self.config.domain.time_end or '2001-01-01'

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        time_idx = pd.DatetimeIndex(times)
        mask = (time_idx >= start_dt) & (time_idx < end_dt)
        pptrate, airtemp, times = pptrate[mask], airtemp[mask], times[mask]

        if len(pptrate) == 0:
            self.logger.warning("No forcing data in simulation period")
            return None

        # Compute hourly P and PET
        latitude = self._get_latitude()
        pet_mm_hr = self._compute_oudin_pet(airtemp, times, latitude)
        ppt_mm_hr = np.maximum(pptrate * 3600.0, 0.0)
        temp_c = airtemp - 273.15

        # Apply Snow-17 model: rain/snow partition + snowmelt
        effective_mm_hr = self._apply_snow_model(ppt_mm_hr, temp_c, times)

        # Cache hourly forcing for calibration (Snow17 param updates)
        np.savez_compressed(
            settings_dir / 'hourly_forcing_cache.npz',
            ppt_mm_hr=ppt_mm_hr, pet_mm_hr=pet_mm_hr,
            temp_c=temp_c, times=times.astype('int64'),
        )

        # Aggregate to daily averages
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

        # Save for reference
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

    def run_preprocessing(self):
        """Generate ParFlow .pfidb input file with daily rainfall cycle."""
        settings_dir = self.project_dir / "settings" / "PARFLOW"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating ParFlow input files in {settings_dir}")

        # NX=3 avoids ParFlow isolated-cell issue on 1x1x1 grids
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

        # Default: unsaturated start (negative pressure head)
        if initial_pressure is None:
            initial_pressure = -1.0
        else:
            initial_pressure = float(initial_pressure)

        self.logger.info(
            f"ParFlow grid: {nx}x{ny}x{nz}, dx={dx:.0f}m, dz={dz:.1f}m, "
            f"K_sat={k_sat}, porosity={porosity}, slope={slope_x}"
        )
        self.logger.info(f"Timesteps: {n_steps} x {timestep_hours} hours")

        # Prepare daily rainfall from ERA5
        rainfall_entries = self._prepare_daily_rainfall(settings_dir)

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

        self._write_pfidb_text(settings_dir, runname, **common_params)
        self.logger.info(f"ParFlow input files generated in {settings_dir}")

    def _write_pfidb_text(self, settings_dir, runname, **params):
        """Write plain-text .pfidb with daily rainfall cycle."""
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

        stop_time = float(params['n_steps'] * params['timestep_hours'])
        add('TimingInfo.BaseUnit', 1.0)
        add('TimingInfo.StartCount', 0)
        add('TimingInfo.StartTime', 0.0)
        add('TimingInfo.StopTime', stop_time)
        add('TimingInfo.DumpInterval', params['dump_interval'])

        add('TimeStep.Type', 'Constant')
        add('TimeStep.Value', params['timestep_hours'])

        # Build cycles: 'constant' for zero-flux patches, 'raincycle' for z_upper
        rainfall = params.get('rainfall_entries')
        if rainfall and len(rainfall) > 0:
            rain_labels = [r[0] for r in rainfall]
            add('Cycle.Names', 'constant raincycle')
            # Simple constant cycle for zero-flux boundaries
            add('Cycle.constant.Names', 'alltime')
            add('Cycle.constant.alltime.Length', 1)
            add('Cycle.constant.Repeat', -1)
            # Daily rainfall cycle for z_upper
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

        # Zero-flux lateral and bottom patches use 'constant' cycle
        for patch in ['x_lower', 'x_upper', 'y_lower', 'y_upper', 'z_lower']:
            add(f'Patch.{patch}.BCPressure.Type', 'FluxConst')
            add(f'Patch.{patch}.BCPressure.Cycle', 'constant')
            add(f'Patch.{patch}.BCPressure.alltime.Value', 0.0)

        # z_upper: OverlandFlow with daily rainfall values
        add('Patch.z_upper.BCPressure.Type', 'OverlandFlow')
        if rain_labels:
            add('Patch.z_upper.BCPressure.Cycle', 'raincycle')
            for label, _hours, rate in rainfall:
                # Negative Value = rainfall (water addition to surface)
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

        # Non-zero slope enables overland flow routing
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
            f"Generated ParFlow .pfidb (text): {pfidb_path} "
            f"({n_entries} entries)"
        )
