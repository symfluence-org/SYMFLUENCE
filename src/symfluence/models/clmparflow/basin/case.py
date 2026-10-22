# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Write and run one basin-mode ParFlow-CLM segment (normally one water year).

Vertical structure: 14 terrain-following layers (70 m), the top 10 being CLM's soil layers (7 m). Lateral
boundaries and the base are no-flow; water leaves only as overland flow across the outlet face (OverlandFlow
BC, the only ParFlow BC whose boundary outflow ``PrintOverlandSum`` records). Permeability is a 3-D field: soil K
in the CLM layers, aquifer K below, and optionally ``glacier_bed_k`` in the CLM layers of glacier cells (the bed
under the ice is ice, so melt runs off at the surface instead of recharging the aquifer).

Forcing: hourly values per forcing unit, each cell taking its unit's column, written as 3-D CLM met files of
24 hours. Vegetation: IGBP class 15 (snow/ice, CLM land ice) on glacier cells, 16 (barren) elsewhere.

Segments chain by restart: the previous segment's last pressure field (``ICPressure`` PFBFile) and its CLM
restart files (``clm_ic`` 1). ``startcode`` 2 takes the calendar from ``drv_clmin.dat``, so spin-up can rewind the
dates while carrying the state.

Glacier water and the runtime ice albedo need the SYMFLUENCE-patched build
(``symfluence binary install clmparflow --patched``; see ``patches/README.md``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import pfio
from .grid import BasinGrid

TEMPLATES = Path(__file__).with_name('templates')
CLM_OUTPUT_DIRS = ('qflx_evap_grnd eflx_lh_tot qflx_evap_tot qflx_tran_veg correct_output qflx_infl swe_out '
                   'eflx_lwrad_out t_grnd diag_out qflx_evap_soi eflx_soil_grnd eflx_sh_tot qflx_evap_veg '
                   'qflx_top_soil').split()
# CLM 3-D met file variables and the default (SYMFLUENCE basin-averaged) forcing names they are read from
MET_VARIABLES = ('DSWR', 'DLWR', 'APCP', 'Temp', 'UGRD', 'VGRD', 'Press', 'SPFH')
DEFAULT_FORCING_NAMES = {'DSWR': 'SWRadAtm', 'DLWR': 'LWRadAtm', 'APCP': 'pptrate', 'Temp': 'airtemp',
                         'UGRD': 'windspd', 'Press': 'airpres', 'SPFH': 'spechum'}


@dataclass
class BasinParameters:
    """Subsurface, surface and forcing parameters of a basin run (lengths in m, K in m/h, Manning in h m^-1/3)."""

    dz_top_down: List[float] = field(default_factory=lambda: [.1, .1, .2, .2, .4, .6, .8, 1., 1.6, 2., 5., 10., 20., 28.])
    soil_k: float = 0.1
    soil_porosity: float = 0.40
    aquifer_k: float = 1.0
    aquifer_porosity: float = 0.25
    vg_alpha: float = 3.5
    vg_n: float = 2.0
    s_res: float = 0.05
    specific_storage: float = 1e-5
    manning: float = 3e-6
    glacier_bed_k: Optional[float] = None
    precip_scale: float = 1.0
    ice_albedo: Optional[Tuple[float, float]] = None
    initial_water_table_depth: float = 3.0

    @property
    def nz(self) -> int:
        return len(self.dz_top_down)


def parflow_keys(grid: BasinGrid, params: BasinParameters, name: str, hours: int, P: int = 1, Q: int = 1,
                 dump_hours: int = 24, ic_file: Optional[str] = None) -> Dict[str, object]:
    """The ParFlow key database of a basin segment."""
    ny, nx = grid.shape
    dx = grid.dx
    nz = params.nz
    k: Dict[str, object] = {
        'FileVersion': 4, 'Process.Topology.P': P, 'Process.Topology.Q': Q, 'Process.Topology.R': 1,
        'ComputationalGrid.Lower.X': 0., 'ComputationalGrid.Lower.Y': 0., 'ComputationalGrid.Lower.Z': 0.,
        'ComputationalGrid.DX': dx, 'ComputationalGrid.DY': dx, 'ComputationalGrid.DZ': 1.,
        'ComputationalGrid.NX': nx, 'ComputationalGrid.NY': ny, 'ComputationalGrid.NZ': nz,
        'GeomInput.Names': 'domain_input soil_input',
        'GeomInput.domain_input.InputType': 'SolidFile', 'GeomInput.domain_input.GeomNames': 'domain',
        'GeomInput.domain_input.FileName': 'basin.pfsol', 'Geom.domain.Patches': 'top bottom side',
        'GeomInput.soil_input.InputType': 'Box', 'GeomInput.soil_input.GeomName': 'soil',
        'Geom.soil.Lower.X': 0., 'Geom.soil.Lower.Y': 0., 'Geom.soil.Lower.Z': float(nz - 10),
        'Geom.soil.Upper.X': nx * dx, 'Geom.soil.Upper.Y': ny * dx, 'Geom.soil.Upper.Z': float(nz),
        'Solver.Nonlinear.VariableDz': True, 'dzScale.GeomNames': 'domain', 'dzScale.Type': 'nzList',
        'dzScale.nzListNumber': nz,
    }
    for i, d in enumerate(params.dz_top_down[::-1]):
        k[f'Cell.{i}.dzScale.Value'] = d
    k.update({
        'Geom.Perm.Names': 'domain', 'Geom.domain.Perm.Type': 'PFBFile', 'Geom.domain.Perm.FileName': 'perm.pfb',
        'Perm.TensorType': 'TensorByGeom', 'Geom.Perm.TensorByGeom.Names': 'domain',
        'Geom.domain.Perm.TensorValX': 1., 'Geom.domain.Perm.TensorValY': 1., 'Geom.domain.Perm.TensorValZ': 1.,
        'SpecificStorage.Type': 'Constant', 'SpecificStorage.GeomNames': 'domain',
        'Geom.domain.SpecificStorage.Value': params.specific_storage,
        'Phase.Names': 'water', 'Phase.water.Density.Type': 'Constant', 'Phase.water.Density.Value': 1.,
        'Phase.water.Viscosity.Type': 'Constant', 'Phase.water.Viscosity.Value': 1., 'Contaminants.Names': '',
        'Gravity': 1., 'Geom.Porosity.GeomNames': 'domain soil', 'Geom.domain.Porosity.Type': 'Constant',
        'Geom.domain.Porosity.Value': params.aquifer_porosity, 'Geom.soil.Porosity.Type': 'Constant',
        'Geom.soil.Porosity.Value': params.soil_porosity, 'Domain.GeomName': 'domain',
        'Phase.water.Mobility.Type': 'Constant', 'Phase.water.Mobility.Value': 1.,
        'Phase.RelPerm.Type': 'VanGenuchten', 'Phase.RelPerm.GeomNames': 'domain soil',
        'Phase.Saturation.Type': 'VanGenuchten', 'Phase.Saturation.GeomNames': 'domain soil',
    })
    for g in ('domain', 'soil'):
        k.update({f'Geom.{g}.RelPerm.Alpha': params.vg_alpha, f'Geom.{g}.RelPerm.N': params.vg_n,
                  f'Geom.{g}.Saturation.Alpha': params.vg_alpha, f'Geom.{g}.Saturation.N': params.vg_n,
                  f'Geom.{g}.Saturation.SRes': params.s_res, f'Geom.{g}.Saturation.SSat': 1.})
    k.update({
        'Wells.Names': '', 'Cycle.Names': 'constant', 'Cycle.constant.Names': 'alltime',
        'Cycle.constant.alltime.Length': 1, 'Cycle.constant.Repeat': -1, 'BCPressure.PatchNames': 'top bottom side',
    })
    for patch in ('bottom', 'side'):
        k.update({f'Patch.{patch}.BCPressure.Type': 'FluxConst', f'Patch.{patch}.BCPressure.Cycle': 'constant',
                  f'Patch.{patch}.BCPressure.alltime.Value': 0.})
    k.update({
        'Patch.top.BCPressure.Type': 'OverlandFlow', 'Patch.top.BCPressure.Cycle': 'constant',
        'Patch.top.BCPressure.alltime.Value': 0.,
        'TopoSlopesX.Type': 'PFBFile', 'TopoSlopesX.GeomNames': 'domain', 'TopoSlopesX.FileName': 'slope_x.pfb',
        'TopoSlopesY.Type': 'PFBFile', 'TopoSlopesY.GeomNames': 'domain', 'TopoSlopesY.FileName': 'slope_y.pfb',
        'Mannings.Type': 'Constant', 'Mannings.GeomNames': 'domain', 'Mannings.Geom.domain.Value': params.manning,
        'PhaseSources.water.Type': 'Constant', 'PhaseSources.water.GeomNames': 'domain',
        'PhaseSources.water.Geom.domain.Value': 0., 'KnownSolution': 'NoKnownSolution', 'Solver': 'Richards',
        'Solver.TerrainFollowingGrid': True, 'Solver.MaxIter': 10000000, 'Solver.Nonlinear.MaxIter': 100,
        'Solver.Nonlinear.ResidualTol': 1e-6, 'Solver.Nonlinear.EtaChoice': 'EtaConstant',
        'Solver.Nonlinear.EtaValue': .001, 'Solver.Nonlinear.UseJacobian': True,
        'Solver.Nonlinear.DerivativeEpsilon': 1e-16, 'Solver.Nonlinear.StepTol': 1e-20,
        'Solver.Nonlinear.Globalization': 'LineSearch', 'Solver.Linear.KrylovDimension': 70,
        'Solver.Linear.MaxRestarts': 2, 'Solver.Linear.Preconditioner': 'PFMG',
        'Solver.MaxConvergenceFailures': 8, 'Solver.Drop': 1e-20, 'Solver.AbsTol': 1e-9,
        'Solver.LSM': 'CLM', 'Solver.CLM.MetForcing': '3D', 'Solver.CLM.MetFileName': 'CARRA',
        'Solver.CLM.MetFilePath': './forcing', 'Solver.CLM.MetFileNT': 24, 'Solver.CLM.IstepStart': 1,
        'Solver.CLM.CLMDumpInterval': dump_hours, 'Solver.CLM.EvapBeta': 'Linear',
        'Solver.CLM.VegWaterStress': 'Saturation', 'Solver.CLM.ResSat': params.s_res,
        'Solver.CLM.WiltingPoint': .12, 'Solver.CLM.FieldCapacity': .98, 'Solver.CLM.IrrigationType': 'none',
        'Solver.CLM.RootZoneNZ': 10, 'Solver.CLM.SoiLayer': 7, 'Solver.CLM.WriteLogs': False,
        'Solver.CLM.WriteLastRST': True, 'Solver.CLM.DailyRST': True, 'Solver.CLM.SingleFile': True,
        'Solver.CLM.Print1dOut': False, 'Solver.PrintSubsurfData': True, 'Solver.PrintPressure': True,
        'Solver.PrintSaturation': True, 'Solver.PrintMask': True, 'Solver.PrintOverlandSum': True,
        'Solver.PrintCLM': True, 'Solver.PrintEvapTrans': False,
        'TimingInfo.BaseUnit': 1., 'TimingInfo.StartCount': 0, 'TimingInfo.StartTime': 0.,
        'TimingInfo.StopTime': float(hours), 'TimingInfo.DumpInterval': float(dump_hours),
        'TimeStep.Type': 'Constant', 'TimeStep.Value': 1.,
    })
    if ic_file:
        k.update({'ICPressure.Type': 'PFBFile', 'ICPressure.GeomNames': 'domain',
                  'Geom.domain.ICPressure.FileName': ic_file})
    else:
        k.update({'ICPressure.Type': 'HydroStaticPatch', 'ICPressure.GeomNames': 'domain',
                  'Geom.domain.ICPressure.Value': sum(params.dz_top_down) - params.initial_water_table_depth,
                  'Geom.domain.ICPressure.RefGeom': 'domain', 'Geom.domain.ICPressure.RefPatch': 'bottom'})
    return k


def open_forcing(path: Path):
    """Open the unit forcing: one NetCDF file, or every ``*.nc`` in a directory (e.g. SYMFLUENCE basin-averaged)."""
    import xarray as xr

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.nc'))
        if not files:
            raise FileNotFoundError(f'no NetCDF forcing in {path}')
        return xr.open_mfdataset(files, combine='by_coords')
    return xr.open_dataset(path)


def permeability(grid: BasinGrid, params: BasinParameters) -> np.ndarray:
    """3-D permeability [z, y, x] (z = 0 at the base): aquifer, soil in the 10 CLM layers, glacier bed."""
    ny, nx = grid.shape
    nz = params.nz
    K = np.full((nz, ny, nx), params.aquifer_k)
    K[nz - 10:] = params.soil_k
    if params.glacier_bed_k is not None:
        K[nz - 10:, grid.ice] = params.glacier_bed_k
    return K


def write_static(run: Path, grid: BasinGrid, params: BasinParameters, parflow_dir: Path, P: int = 1,
                 Q: int = 1) -> None:
    """Slopes, permeability, the mask and the solid file (``pfmask-to-pfsol``: top 1, bottom 2, side 3)."""
    dx = grid.dx
    pfio.write_pfb(run / 'slope_x.pfb', grid.sx[None], dx=dx, dy=dx, P=P, Q=Q)
    pfio.write_pfb(run / 'slope_y.pfb', grid.sy[None], dx=dx, dy=dx, P=P, Q=Q)
    pfio.write_pfb(run / 'perm.pfb', permeability(grid, params), dx=dx, dy=dx, P=P, Q=Q)
    pfio.write_pfb(run / 'mask.pfb', grid.mask[None].astype(float), dx=dx, dy=dx)
    tool = Path(parflow_dir) / 'bin' / 'pfmask-to-pfsol'
    r = subprocess.run([str(tool), '--mask', 'mask.pfb', '--pfsol', 'basin.pfsol', '--vtk', 'basin.vtk',
                        '--bottom-patch-label', '2', '--side-patch-label', '3', '--z-bottom', '0',
                        '--z-top', str(float(params.nz)), '--dx', str(dx), '--dy', str(dx)],
                       cwd=run, capture_output=True, text=True)
    if r.returncode != 0 or not (run / 'basin.pfsol').exists():
        raise RuntimeError(f'pfmask-to-pfsol failed: {r.stdout}{r.stderr}')


def write_forcing(out: Path, grid: BasinGrid, params: BasinParameters, forcing_file: Path, start: str, end: str,
                  unit_variable: str = 'unit_id', names: Optional[Dict[str, str]] = None, P: int = 1,
                  Q: int = 1) -> int:
    """Write CLM 3-D met files (24 h each) for [start, end); returns the number of hours.

    ``forcing_file`` holds hourly series on (time, unit) with ``unit_variable`` naming each unit; ``names`` maps
    CLM met variables to its variable names (``VGRD`` is written as zero when not mapped: wind speed goes in
    ``UGRD``); values are hourly (W m-2, mm s-1, K, m s-1, Pa, kg kg-1). Precipitation is multiplied by ``params.precip_scale``. Inactive cells take a land unit's column;
    their values are never used but must be finite.
    """
    names = {**DEFAULT_FORCING_NAMES, **(names or {})}
    out.mkdir(parents=True, exist_ok=True)
    unit = grid.unit
    with open_forcing(forcing_file) as ds:  # closed after loading: repeated open handles have segfaulted HDF5
        ids = [str(u) for u in ds[unit_variable].values]
        col_of = [ids.index(u) for u in grid.units]
        data = ds.sel(time=slice(start, pd.Timestamp(end) - pd.Timedelta(hours=1))).load()
    fill = col_of[int(unit[grid.mask & ~grid.ice][0])] if (grid.mask & ~grid.ice).any() else col_of[0]
    idx = np.array([col_of[u] if u >= 0 else fill for u in unit.ravel()])
    n = data.sizes['time']
    if n % 24:
        raise ValueError(f'forcing for {start}..{end} has {n} hours, not whole days')
    ny, nx = grid.shape
    for var in MET_VARIABLES:
        if var in names:
            v = data[names[var]].transpose('time', ...)
            # scale in the file's precision before widening, as the promoted Tungnaá member was run
            a = (v * params.precip_scale if var == 'APCP' else v).values.astype(float)
        elif var == 'VGRD':
            a = np.zeros((n, len(ids)))
        else:
            raise KeyError(f'no forcing variable mapped to {var}')
        a = a[:, idx].reshape(n, ny, nx)
        if not np.isfinite(a).all():
            raise ValueError(f'non-finite forcing in {var}')
        for s in range(0, n, 24):
            pfio.write_pfb(out / f'CARRA.{var}.{s + 1:06d}_to_{s + 24:06d}.pfb', a[s:s + 24], dx=grid.dx,
                           dy=grid.dx, P=P, Q=Q)
    return n


def write_clm_files(run: Path, grid: BasinGrid, params: BasinParameters, start: str, end: str, restart: bool,
                    ranks: int) -> None:
    """``drv_clmin.dat`` (dates, restart flags), ``drv_vegp.dat``, ``drv_vegm.dat`` and per-rank copies."""
    from pyproj import Transformer

    txt = (TEMPLATES / 'drv_clmin.dat').read_text()
    s, e = pd.Timestamp(start), pd.Timestamp(end)

    def put(name: str, value: object) -> None:
        nonlocal txt
        txt, count = re.subn(rf'(?m)^{name}(\s+)\S+', lambda m: f'{name}{m.group(1)}{value}', txt, count=1)
        if count != 1:
            raise KeyError(name)

    for name, value in [('shr', f'{s.hour:02d}'), ('sda', f'{s.day:02d}'), ('smo', f'{s.month:02d}'), ('syr', s.year),
                        ('ehr', f'{e.hour:02d}'), ('eda', f'{e.day:02d}'), ('emo', f'{e.month:02d}'), ('eyr', e.year),
                        ('t_ini', '273.'), ('h2osno_ini', '0.'), ('rstf', 'clm.rst.'), ('startcode', 2),
                        ('clm_ic', 1 if restart else 2)]:
        put(name, value)
    (run / 'drv_clmin.dat').write_text(txt)
    shutil.copy(TEMPLATES / 'drv_vegp.dat', run / 'drv_vegp.dat')
    to_lonlat = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    lines = [' x  y  lat    lon    sand clay color  fractional coverage of grid by vegetation class (Must/Should Add to 1.0)',
             '       (Deg)\t (Deg)  (%/100)   index  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18']
    ny, nx = grid.shape
    for j in range(ny):
        for i in range(nx):
            lon, lat = to_lonlat.transform(grid.x0 + (i + .5) * grid.dx, grid.y0 + (j + .5) * grid.dx)
            frac = np.zeros(18)
            frac[14 if grid.ice[j, i] else 15] = 1
            lines.append(f'{i + 1:4d}{j + 1:4d}{lat:9.3f}{lon:9.3f}  0.65 0.07   4  ' + ' '.join(f'{x:.1f}' for x in frac))
    (run / 'drv_vegm.dat').write_text('\n'.join(lines) + '\n')
    for r in range(ranks):
        for f in ('drv_clmin.dat', 'drv_vegm.dat'):
            shutil.copy(run / f, run / f'{f}.{r}')
    for d in CLM_OUTPUT_DIRS:  # CLM opens these even with SingleFile output
        (run / d).mkdir(exist_ok=True)
    if params.ice_albedo is not None:
        (run / 'clm_albice.txt').write_text('%.3f %.3f\n' % tuple(params.ice_albedo))


def run_parflow(run: Path, name: str, parflow_dir: Path, ranks: int, mpirun: Optional[str] = None,
                timeout: Optional[int] = None) -> Tuple[int, str]:
    """Run ParFlow in ``run``; returns (exit code, tail of stdout+stderr)."""
    exe = Path(parflow_dir) / 'bin' / 'parflow'
    cmd = [mpirun or 'mpirun', '-np', str(ranks), str(exe), name] if ranks > 1 or mpirun else [str(exe), name]
    env = dict(os.environ, PARFLOW_DIR=str(parflow_dir))
    r = subprocess.run(cmd, cwd=run, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def write_segment(run: Path, name: str, grid: BasinGrid, params: BasinParameters, parflow_dir: Path,
                  forcing_file: Path, start: str, end: str, prev: Optional[Path] = None, P: int = 1, Q: int = 1,
                  forcing_unit_variable: str = 'unit_id', forcing_names: Optional[Dict[str, str]] = None) -> Dict:
    """Write every input of one segment into ``run`` (emptied first); returns its metadata."""
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    write_static(run, grid, params, parflow_dir, P, Q)
    hours = write_forcing(run / 'forcing', grid, params, forcing_file, start, end, forcing_unit_variable,
                          forcing_names, P, Q)
    ic = None
    if prev is not None:
        last = sorted(prev.glob(f'{prev.name}.out.press.*.pfb'))[-1]
        shutil.copy(last, run / 'ic_press.pfb')
        shutil.copy(str(last) + '.dist', run / 'ic_press.pfb.dist')
        ic = 'ic_press.pfb'
        for r in range(P * Q):
            shutil.copy(prev / f'clm.rst.00000.{r}', run / f'clm.rst.00000.{r}')
    write_clm_files(run, grid, params, start, end, restart=prev is not None, ranks=P * Q)
    pfio.write_pfidb(run / f'{name}.pfidb', parflow_keys(grid, params, name, hours, P, Q, ic_file=ic))
    meta = dict(seg=name, start=start, end=end, hours=hours, prev=prev.name if prev else None, P=P, Q=Q,
                nx=grid.shape[1], ny=grid.shape[0], params=asdict(params))
    (run / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


def run_segment(run: Path, name: str, grid: BasinGrid, params: BasinParameters, parflow_dir: Path,
                forcing_file: Path, start: str, end: str, prev: Optional[Path] = None, P: int = 1, Q: int = 1,
                keep_forcing: bool = False, mpirun: Optional[str] = None, timeout: Optional[int] = None,
                forcing_unit_variable: str = 'unit_id', forcing_names: Optional[Dict[str, str]] = None) -> Dict:
    """Write and run one segment; raises if ParFlow does not finish. Forcing and CLM output folders are removed."""
    t0 = time.time()
    meta = write_segment(run, name, grid, params, parflow_dir, forcing_file, start, end, prev, P, Q,
                         forcing_unit_variable, forcing_names)
    t1 = time.time()
    rc, out = run_parflow(run, name, parflow_dir, P * Q, mpirun, timeout)
    meta.update(rc=rc, build_s=round(t1 - t0), run_s=round(time.time() - t1))
    (run / 'meta.json').write_text(json.dumps(meta, indent=1))
    (run / 'run.log').write_text(out)
    if rc != 0 or 'Problem solved' not in out:
        raise RuntimeError(f'{name} failed (exit {rc}):\n{out[-2000:]}')
    if not keep_forcing:
        shutil.rmtree(run / 'forcing')
    for d in CLM_OUTPUT_DIRS:
        shutil.rmtree(run / d, ignore_errors=True)
    return meta


def water_year_segments(first_wy: int, last_wy: int) -> Sequence[Tuple[int, str, str]]:
    """(water year, start, end) for each October-September water year."""
    return [(wy, f'{wy - 1}-10-01', f'{wy}-10-01') for wy in range(first_wy, last_wy + 1)]
