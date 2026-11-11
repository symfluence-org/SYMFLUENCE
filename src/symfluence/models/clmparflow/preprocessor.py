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
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from symfluence.core.exceptions import ShapefileError
from symfluence.core.mixins.project import resolve_data_subdir, resolve_forcing_basin_path
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R
from symfluence.models.parflow.lumped_forcing import (
    LumpedForcingMixin,
    location_from_forcing,
)

logger = logging.getLogger(__name__)


@R.preprocessors.add("CLMPARFLOW")
class CLMParFlowPreProcessor(LumpedForcingMixin, BaseModelPreProcessor):  # type: ignore[misc]
    """Generates ParFlow-CLM input files for integrated hydrologic simulation."""

    MODEL_NAME = "CLMPARFLOW"
    _location_cache: Optional[Tuple[float, float]] = None

    # This preprocessor keeps its own defensive __init__ (tolerates dict / coupled
    # contexts) rather than calling super().__init__, whose strict typed-config
    # access doesn't fit how the coupled groundwater models are wired.
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
        basin_avg = resolve_forcing_basin_path(self.project_dir)
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

    def _get_location(self) -> Tuple[float, float]:
        """(lat, lon) of the catchment, from the catchment shapefile centroid.

        Falls back to the first cell of the basin-averaged forcing if the
        shapefile is missing; raises if neither is available. (This used to
        fall back to a hard-coded Bow-at-Banff latitude and always used the
        Banff longitude, which CLM uses for solar geometry.)
        """
        if self._location_cache is not None:
            return self._location_cache
        try:
            props = self.get_catchment_location()
            self._location_cache = (props['lat'], props['lon'])
            return self._location_cache
        except FileNotFoundError as e:
            shp_error = e
        location = location_from_forcing(self._get_forcing_dir())
        if location is not None:
            self._location_cache = location
            return self._location_cache
        raise ShapefileError(
            f"Cannot determine CLM-ParFlow site location: {shp_error}; and no "
            "basin-averaged forcing with latitude/longitude was found."
        )

    def _get_latitude(self) -> float:
        return self._get_location()[0]

    LUMPED_MODEL_LABEL = 'CLMParFlow'

    def _on_hourly_forcing(self, output_dir, pptrate, airtemp, times):
        # Raw forcing for the CLM met file
        self._write_clm_met_forcing(output_dir, pptrate, airtemp, times)

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

        lat, lon = self._get_location()

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

    def _basin_mode(self) -> bool:
        return bool(self._get_pf_cfg('BASIN_MODE', False))

    def _run_basin_preprocessing(self) -> bool:
        """Basin mode: build the regular grid over the basin (segment inputs are written by the runner)."""
        from .basin import build_basin_grid
        from .basin.settings import basin_inputs, grid_dir

        cfg = self.config.model.clmparflow
        inputs = basin_inputs(cfg, self.project_dir, self.domain_name)
        grid = build_basin_grid(inputs.outline, inputs.units, inputs.unit_field, inputs.dem, inputs.pour_point,
                                inputs.crs, inputs.dx, inputs.ice_field, inputs.ice_value, inputs.min_slope,
                                rotation=inputs.rotation, k_zone=inputs.k_zone, lake=inputs.lake)
        out = grid_dir(cfg, self.project_dir)
        grid.save(out)
        info = grid.info
        extra = ''.join(f", {info[k]} {label}" for k, label in (('kzone_cells', 'K-zone'), ('lake_cells', 'lake'))
                        if k in info)
        if grid.rotation:
            extra += f", axes rotated {grid.rotation:g} deg"
        if grid.lake is not None and cfg.basin_lake_manning is None:
            self.logger.warning("CLMPARFLOW_BASIN_LAKE levels the lake cells, but without "
                                "CLMPARFLOW_BASIN_LAKE_MANNING they store no more water than other cells")
        if grid.kzone is not None and cfg.basin_k_zone_factor == 1.0 and not cfg.basin_k_zone_soil:
            self.logger.warning("CLMPARFLOW_BASIN_K_ZONE is set but CLMPARFLOW_BASIN_K_ZONE_FACTOR is 1: no effect")
        self.logger.info(
            f"CLMParFlow basin grid: {info['nx']}x{info['ny']} cells of {inputs.dx:.0f} m, {info['active']} active, "
            f"{info['ice_cells']} glacier{extra}; outlet {info['outlet_rc_north_up']} -> {out}")
        return True

    def run_preprocessing(self):
        """Generate ParFlow-CLM .pfidb input file with CLM keys + CLM input files."""
        if self._basin_mode():
            return self._run_basin_preprocessing()
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
        from symfluence.models.parflow.extractor import resolve_dump_interval_hours
        dump_interval = resolve_dump_interval_hours(self.config, 'clmparflow')

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
