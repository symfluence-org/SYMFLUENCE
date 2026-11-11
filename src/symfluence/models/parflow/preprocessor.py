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
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ShapefileError
from symfluence.core.mixins.project import resolve_forcing_basin_path
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R

from .lumped_forcing import (
    LumpedForcingMixin,
    location_from_forcing,
)

logger = logging.getLogger(__name__)


@R.preprocessors.add("PARFLOW")
class ParFlowPreProcessor(LumpedForcingMixin, BaseModelPreProcessor):  # type: ignore[misc]
    """Generates ParFlow input files for integrated hydrologic simulation."""

    MODEL_NAME = "PARFLOW"

    # Keeps its own defensive __init__ (tolerates dict / MagicMock coupled-test
    # configs) rather than calling super().__init__, whose strict typed-config
    # access doesn't fit how the coupled groundwater models are wired.
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
        basin_avg = resolve_forcing_basin_path(self.project_dir)
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
        """Catchment latitude: config, catchment centroid, else forcing; never a default.

        (This used to fall back to 51.36, the latitude of Bow at Banff.)
        """
        lat = getattr(self.config.domain, 'latitude', None)
        if lat is not None:
            return float(lat)
        try:
            return float(self.get_catchment_location()['lat'])
        except (FileNotFoundError, ValueError, AttributeError, TypeError) as e:
            shp_error = e
        location = location_from_forcing(self._get_forcing_dir(), self._select_run_forcing)
        if location is not None:
            return location[0]
        raise ShapefileError(
            f"Cannot determine the ParFlow catchment latitude: {shp_error}; and no "
            "basin-averaged forcing with latitude/longitude was found."
        )

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
        from symfluence.models.parflow.extractor import resolve_dump_interval_hours
        dump_interval = resolve_dump_interval_hours(self.config, 'parflow')

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
