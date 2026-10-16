# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Real-basin MODFLOW 6 groundwater model driven by land-model recharge.

The grid is the active-cell grid of the GSFLOW real-basin geometry (square
cells over the basin, land surface from the DEM, river reaches from the
delineated network). The model has one convertible layer:

* RCH: one time series per land-model recharge zone. Each (cell, zone) overlap
  is a separate entry whose auxiliary multiplier is chosen so that the zone's
  volume is conserved exactly, whatever the zone/grid mismatch.
* DRN ``RIV``: the river reaches, at the streambed top, conductance
  ``river_leakance`` x reach length.
* DRN ``SEEP``: every active cell at its land surface, conductance
  ``seepage_leakance`` x the cell's basin area (springs and saturated ground).
* Stress period 1 is steady state at the mean recharge; then one transient
  period per day.

Both drains return water to the river, so simulated groundwater discharge to
the stream is their summed outflow. The model writes everything it needs to
``basin_grid.json`` so calibration can rewrite parameter packages without
rebuilding the geometry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ConfigurationError, ValidationError

GRID_FILE = 'basin_grid.json'
WEIGHTS_FILE = 'recharge_weights.csv'
RECHARGE_TS = 'gwf.rch.ts'
BUDGET_CSV = 'gwf.bud.csv'
COUPLING_FILE = 'coupling.json'   # land model, its settings/output dirs, recharge start date


@dataclass
class MF6BasinParams:
    """Calibratable groundwater parameters (units: m, days)."""
    k: float = 5.0                    # horizontal hydraulic conductivity (m/d)
    sy: float = 0.1                   # specific yield (-)
    ss: float = 1e-5                  # specific storage (1/m)
    thickness: float = 100.0          # aquifer thickness below land surface / streambed (m)
    river_leakance: float = 10.0      # river conductance per reach length (m2/d per m)
    seepage_leakance: float = 0.1     # land-surface seepage conductance per area (1/d)

    # Calibration names -> attribute
    NAMES = {'K': 'k', 'SY': 'sy', 'SS': 'ss', 'AQUIFER_THICKNESS': 'thickness',
             'RIVER_LEAKANCE': 'river_leakance', 'SEEPAGE_LEAKANCE': 'seepage_leakance'}

    def updated(self, values: Dict[str, float]) -> 'MF6BasinParams':
        kw = dict(self.__dict__)
        for name, v in values.items():
            attr = self.NAMES.get(name.upper())
            if attr is not None:
                kw[attr] = float(v)
        out = MF6BasinParams(**kw)
        out.validate()
        return out

    def validate(self) -> None:
        for name, attr in self.NAMES.items():
            if not getattr(self, attr) > 0:
                raise ConfigurationError(f'MODFLOW basin parameter {name} must be positive, got {getattr(self, attr)}')


@dataclass
class MF6BasinGrid:
    """Active-cell grid with land surface, river reaches and basin cell fractions."""
    nrow: int
    ncol: int
    cell_size: float
    origin: tuple                 # (x left edge, y top edge) in ``crs``
    crs: str
    active: np.ndarray            # (nrow, ncol) bool
    top: np.ndarray               # (nrow, ncol) land surface; NaN where inactive
    basin_frac: np.ndarray        # (nrow, ncol) fraction of cell inside the basin
    river: pd.DataFrame           # row, col (1-based), length, strtop, seg
    notes: list = field(default_factory=list)

    @property
    def cell_area(self) -> float:
        return self.cell_size * self.cell_size

    def cell_polygons(self):
        """GeoDataFrame of active cells (row, col, geometry) in ``crs``."""
        import geopandas as gpd
        from shapely.geometry import box
        x0, y0 = self.origin
        s = self.cell_size
        rows, cols = np.nonzero(self.active)
        return gpd.GeoDataFrame(
            {'row': rows + 1, 'col': cols + 1},
            geometry=[box(x0 + c * s, y0 - (r + 1) * s, x0 + (c + 1) * s, y0 - r * s) for r, c in zip(rows, cols)],
            crs=self.crs)

    def to_json(self, path: Path) -> None:
        d = {'nrow': self.nrow, 'ncol': self.ncol, 'cell_size': self.cell_size, 'origin': list(self.origin),
             'crs': self.crs, 'active': self.active.astype(int).tolist(),
             'top': np.where(self.active, self.top, 0.0).round(3).tolist(),
             'basin_frac': self.basin_frac.round(6).tolist(),
             'river': self.river.to_dict(orient='list'), 'notes': self.notes}
        Path(path).write_text(json.dumps(d))

    @classmethod
    def from_json(cls, path: Path) -> 'MF6BasinGrid':
        d = json.loads(Path(path).read_text())
        active = np.array(d['active'], bool)
        return cls(nrow=d['nrow'], ncol=d['ncol'], cell_size=d['cell_size'], origin=tuple(d['origin']),
                   crs=d['crs'], active=active, top=np.where(active, np.array(d['top'], float), np.nan),
                   basin_frac=np.array(d['basin_frac'], float), river=pd.DataFrame(d['river']),
                   notes=d.get('notes', []))


def grid_from_basin_geometry(geo) -> MF6BasinGrid:
    """Convert a GSFLOW ``BasinGeometry`` to the MODFLOW-only grid description."""
    frac = np.zeros((geo.nrow, geo.ncol))
    per_cell = geo.gvr.groupby('cell_id').cell_pct.sum()
    for cid, f in per_cell.items():
        r, c = divmod(int(cid) - 1, geo.ncol)
        frac[r, c] = min(float(f), 1.0)
    river = geo.reaches[['seg', 'row', 'col', 'length', 'strtop']].copy()
    crs = geo.crs.to_string() if hasattr(geo.crs, 'to_string') else f'EPSG:{geo.crs}' if isinstance(geo.crs, int) else str(geo.crs)
    return MF6BasinGrid(nrow=geo.nrow, ncol=geo.ncol, cell_size=geo.cell_size, origin=tuple(geo.origin), crs=crs,
                        active=geo.active.copy(), top=geo.top.copy(), basin_frac=frac, river=river,
                        notes=list(geo.notes))


def zone_cell_weights(grid: MF6BasinGrid, zones, zone_area_m2: Optional[Dict] = None) -> pd.DataFrame:
    """Recharge multipliers for every (zone, active cell) overlap.

    ``zones`` is a GeoDataFrame with ``zone_id``. The multiplier turns a zone
    recharge rate (m/d) into the cell's rate such that the zone's total volume
    (rate x ``zone_area_m2``, the land model's own area) is preserved:
    ``mult = overlap / cell_area * zone_area / sum(overlaps of zone)``.
    """
    import geopandas as gpd
    if 'zone_id' not in zones:
        raise ConfigurationError('Recharge zones need a zone_id column')
    z = zones[['zone_id', 'geometry']].to_crs(grid.crs)
    cells = grid.cell_polygons()
    ov = gpd.overlay(z, cells, how='intersection', keep_geom_type=True)
    ov['a'] = ov.geometry.area
    ov = ov[ov.a > 1.0].groupby(['zone_id', 'row', 'col'], as_index=False).a.sum()
    missing = set(z.zone_id) - set(ov.zone_id)
    if missing:
        raise ValidationError(f'Recharge zones {sorted(missing)} do not overlap the active MODFLOW grid')
    covered = ov.groupby('zone_id').a.sum()
    target = covered.copy()
    if zone_area_m2:
        target = pd.Series({k: float(zone_area_m2[k]) for k in covered.index})
    ov['mult'] = ov.a / grid.cell_area * ov.zone_id.map(target / covered)
    ov['coverage'] = ov.zone_id.map(covered / z.set_index('zone_id').geometry.area)
    return ov[['zone_id', 'row', 'col', 'mult', 'coverage']].sort_values(['zone_id', 'row', 'col']).reset_index(drop=True)


def _array(a: np.ndarray, fmt: str) -> str:
    return '\n'.join('    ' + ' '.join(fmt.format(v) for v in row) for row in a)


def _ts_name(zone_id) -> str:
    return f'z{zone_id}'


def write_parameter_packages(sim_dir: Path, grid: MF6BasinGrid, params: MF6BasinParams) -> None:
    """(Re)write the parameter-dependent packages: DIS, IC, NPF, STO and both DRNs."""
    params.validate()
    d = Path(sim_dir)
    top = np.where(grid.active, grid.top, 0.0)
    # Thickness is measured below the lower of land surface and streambed, so
    # deeply incised reaches stay inside the aquifer.
    base = top.copy()
    rr = grid.river.row.to_numpy(int) - 1
    rc = grid.river.col.to_numpy(int) - 1
    np.minimum.at(base, (rr, rc), grid.river.strtop.to_numpy(float))
    bot = base - params.thickness
    idomain = grid.active.astype(int)
    (d / 'gwf.dis').write_text(
        'BEGIN OPTIONS\n  LENGTH_UNITS METERS\n'
        f'  XORIGIN {grid.origin[0]:.3f}\n  YORIGIN {grid.origin[1] - grid.nrow * grid.cell_size:.3f}\n'
        'END OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  NLAY 1\n  NROW {grid.nrow}\n  NCOL {grid.ncol}\nEND DIMENSIONS\n\n'
        f'BEGIN GRIDDATA\n  DELR\n    CONSTANT {grid.cell_size}\n  DELC\n    CONSTANT {grid.cell_size}\n'
        f'  TOP\n    INTERNAL\n{_array(top, "{:.3f}")}\n'
        f'  BOTM\n    INTERNAL\n{_array(bot, "{:.3f}")}\n'
        f'  IDOMAIN\n    INTERNAL\n{_array(idomain, "{:d}")}\n'
        'END GRIDDATA\n')
    # Start near the land surface; the steady-state first period sets the heads.
    (d / 'gwf.ic').write_text(
        f'BEGIN GRIDDATA\n  STRT\n    INTERNAL\n{_array(top - 0.1 * params.thickness, "{:.3f}")}\nEND GRIDDATA\n')
    (d / 'gwf.npf').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN GRIDDATA\n  ICELLTYPE\n    CONSTANT 1\n  K\n    CONSTANT {params.k:.8g}\nEND GRIDDATA\n')
    (d / 'gwf.sto').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN GRIDDATA\n  ICONVERT\n    CONSTANT 1\n  SS\n    CONSTANT {params.ss:.8g}\n'
        f'  SY\n    CONSTANT {params.sy:.8g}\nEND GRIDDATA\n\n'
        'BEGIN PERIOD 1\n  STEADY-STATE\nEND PERIOD 1\n\nBEGIN PERIOD 2\n  TRANSIENT\nEND PERIOD 2\n')
    riv = grid.river
    lines = [f'  1 {int(r.row)} {int(r.col)} {r.strtop:.3f} {params.river_leakance * r.length:.6g}'
             for r in riv.itertuples()]
    (d / 'gwf.riv.drn').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  MAXBOUND {len(lines)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIOD 1\n' + '\n'.join(lines) + '\nEND PERIOD 1\n')
    rows, cols = np.nonzero(grid.active)
    seep = [f'  1 {r + 1} {c + 1} {top[r, c]:.3f} '
            f'{params.seepage_leakance * grid.basin_frac[r, c] * grid.cell_area:.6g}'
            for r, c in zip(rows, cols)]
    (d / 'gwf.seep.drn').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  MAXBOUND {len(seep)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIOD 1\n' + '\n'.join(seep) + '\nEND PERIOD 1\n')


def write_simulation(sim_dir: Path, grid: MF6BasinGrid, weights: pd.DataFrame, n_days: int,
                     params: MF6BasinParams, save_heads: str = 'last') -> None:
    """Write a complete MODFLOW 6 simulation (recharge series written separately)."""
    if n_days < 1:
        raise ConfigurationError('MODFLOW basin simulation needs at least one day')
    if save_heads not in ('none', 'last', 'all'):
        raise ConfigurationError(f"MODFLOW_SAVE_HEADS must be none, last or all, got {save_heads!r}")
    d = Path(sim_dir)
    d.mkdir(parents=True, exist_ok=True)
    grid.to_json(d / GRID_FILE)
    weights.to_csv(d / WEIGHTS_FILE, index=False)
    (d / 'mfsim.nam').write_text(
        'BEGIN OPTIONS\nEND OPTIONS\n\nBEGIN TIMING\n  TDIS6 gwf.tdis\nEND TIMING\n\n'
        'BEGIN MODELS\n  GWF6 gwf.nam gwf\nEND MODELS\n\nBEGIN EXCHANGES\nEND EXCHANGES\n\n'
        'BEGIN SOLUTIONGROUP 1\n  IMS6 gwf.ims gwf\nEND SOLUTIONGROUP 1\n')
    set_simulation_length(d, n_days)
    (d / 'gwf.nam').write_text(
        'BEGIN OPTIONS\n  NEWTON UNDER_RELAXATION\nEND OPTIONS\n\n'
        'BEGIN PACKAGES\n  DIS6 gwf.dis\n  IC6 gwf.ic\n  NPF6 gwf.npf\n  STO6 gwf.sto\n'
        '  RCH6 gwf.rch RCH\n  DRN6 gwf.riv.drn RIV\n  DRN6 gwf.seep.drn SEEP\n  OC6 gwf.oc\nEND PACKAGES\n')
    rch = [f'  1 {int(w.row)} {int(w.col)} {_ts_name(w.zone_id)} {w.mult:.8g}' for w in weights.itertuples()]
    (d / 'gwf.rch').write_text(
        f'BEGIN OPTIONS\n  AUXILIARY mult\n  AUXMULTNAME mult\n  TS6 FILEIN {RECHARGE_TS}\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  MAXBOUND {len(rch)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIOD 1\n' + '\n'.join(rch) + '\nEND PERIOD 1\n')
    head_save = {'none': '', 'last': '  SAVE HEAD LAST\n', 'all': '  SAVE HEAD ALL\n'}[save_heads]
    (d / 'gwf.oc').write_text(
        f'BEGIN OPTIONS\n  HEAD FILEOUT gwf.hds\n  BUDGETCSV FILEOUT {BUDGET_CSV}\nEND OPTIONS\n\n'
        f'BEGIN PERIOD 1\n{head_save}  PRINT BUDGET LAST\nEND PERIOD 1\n')
    (d / 'gwf.ims').write_text(
        'BEGIN OPTIONS\n  PRINT_OPTION SUMMARY\n  COMPLEXITY COMPLEX\nEND OPTIONS\n\n'
        'BEGIN NONLINEAR\n  OUTER_DVCLOSE 1.0e-3\n  OUTER_MAXIMUM 200\nEND NONLINEAR\n\n'
        'BEGIN LINEAR\n  LINEAR_ACCELERATION BICGSTAB\n  INNER_DVCLOSE 1.0e-4\n'
        '  INNER_RCLOSE 1.0e-1\n  INNER_MAXIMUM 300\nEND LINEAR\n')
    write_parameter_packages(d, grid, params)


def set_simulation_length(sim_dir: Path, n_days: int) -> None:
    """Rewrite TDIS for one steady-state period plus ``n_days`` daily periods."""
    if n_days < 1:
        raise ConfigurationError('MODFLOW basin simulation needs at least one day')
    nper = n_days + 1
    Path(sim_dir, 'gwf.tdis').write_text(
        f'BEGIN OPTIONS\n  TIME_UNITS DAYS\nEND OPTIONS\n\nBEGIN DIMENSIONS\n  NPER {nper}\nEND DIMENSIONS\n\n'
        'BEGIN PERIODDATA\n' + '  1.0 1 1.0\n' * nper + 'END PERIODDATA\n')


def write_recharge_timeseries(sim_dir: Path, recharge: pd.DataFrame, zone_ids) -> pd.DatetimeIndex:
    """Write zone recharge (m/d, daily rows, one column per zone) as the RCH time series.

    Time 0 (the steady-state period) carries the record mean; day ``i`` of the
    record drives transient period ``i + 2`` (model time ``i + 1``).
    Returns the dates of the transient periods.
    """
    ids = list(zone_ids)
    missing = [z for z in ids if z not in recharge.columns]
    if missing:
        raise ValidationError(f'Recharge series lack zones {missing}')
    rec = recharge[ids].astype(float)
    if rec.isna().any().any():
        raise ValidationError('Recharge series contain gaps')
    if (np.diff(rec.index.values).astype('timedelta64[D]') != np.timedelta64(1, 'D')).any():
        raise ValidationError('Recharge series must be continuous daily')
    names = [_ts_name(z) for z in ids]
    body = ['  0.0 ' + ' '.join(f'{v:.6e}' for v in rec.mean().to_numpy())]
    vals = rec.to_numpy()
    body += [f'  {i + 1}.0 ' + ' '.join(f'{v:.6e}' for v in row) for i, row in enumerate(vals)]
    # MODFLOW integrates over each period, so the series must reach the end time.
    body.append(f'  {len(vals) + 1}.0 ' + ' '.join(f'{v:.6e}' for v in vals[-1]))
    Path(sim_dir, RECHARGE_TS).write_text(
        'BEGIN ATTRIBUTES\n  NAMES ' + ' '.join(names) + '\n  METHODS ' + ' '.join(['STEPWISE'] * len(names))
        + '\nEND ATTRIBUTES\n\nBEGIN TIMESERIES\n' + '\n'.join(body) + '\nEND TIMESERIES\n')
    return pd.DatetimeIndex(rec.index)


def read_budget(sim_dir: Path, start_date) -> pd.DataFrame:
    """Daily groundwater budget (m3/d) from the budget CSV, transient periods only.

    Columns: recharge, river (discharge to river reaches), seepage (discharge at
    the land surface), storage_change (+ = into storage), discharge (river + seepage).
    """
    df = pd.read_csv(Path(sim_dir) / BUDGET_CSV)
    df.columns = [c.strip().upper() for c in df.columns]

    def col(pkg: str, direction: str) -> pd.Series:
        hits = [c for c in df.columns if c.endswith(f'({pkg})_{direction}') or c == f'{pkg}_{direction}']
        if len(hits) != 1:
            raise ValidationError(f'Budget column for {pkg} {direction} not found in {list(df.columns)}')
        return df[hits[0]].astype(float)

    out = pd.DataFrame({
        'recharge': col('RCH', 'IN') - col('RCH', 'OUT'),
        'river': col('RIV', 'OUT') - col('RIV', 'IN'),
        'seepage': col('SEEP', 'OUT') - col('SEEP', 'IN'),
    })
    sto = [c for c in df.columns if c.startswith('STO-')]
    out['storage_change'] = sum(df[c] for c in sto if c.endswith('_OUT')) - sum(df[c] for c in sto if c.endswith('_IN'))
    out['discharge'] = out.river + out.seepage
    out['totim'] = df['TIME'].astype(float).to_numpy()
    out['pct_discrepancy'] = df['PERCENT_DIFFERENCE'].astype(float).to_numpy()
    out = out[out.totim > 1.0 + 1e-9]
    out.index = pd.Timestamp(start_date).normalize() + pd.to_timedelta(np.round(out.totim - 2.0).astype(int), unit='D')
    return out.drop(columns='totim')


def check_run(sim_dir: Path) -> None:
    """Raise unless MODFLOW terminated normally with every step converged."""
    d = Path(sim_dir)
    sim_lst = d / 'mfsim.lst'
    if not sim_lst.exists() or 'Normal termination' not in sim_lst.read_text(errors='replace'):
        raise ValidationError(f'MODFLOW did not terminate normally in {d}')
    lst = d / 'gwf.lst'
    failed = lst.read_text(errors='replace').count('FAILED TO MEET SOLVER CONVERGENCE') if lst.exists() else 0
    if failed:
        raise ValidationError(f'MODFLOW: {failed} time steps failed to converge in {d}')


def params_from_config(cfg) -> MF6BasinParams:
    """Basin parameters from a typed ``MODFLOWConfig`` (or None for defaults)."""
    if cfg is None:
        return MF6BasinParams()
    return MF6BasinParams(k=float(cfg.k), sy=float(cfg.sy), ss=float(cfg.ss), thickness=float(cfg.aquifer_thickness),
                          river_leakance=float(cfg.river_leakance), seepage_leakance=float(cfg.seepage_leakance))


def params_from_flat(config: Dict) -> MF6BasinParams:
    """Basin parameters from a flat (``MODFLOW_*``) config dict."""
    d = MF6BasinParams()
    get = (config or {}).get
    return MF6BasinParams(k=float(get('MODFLOW_K', d.k)), sy=float(get('MODFLOW_SY', d.sy)),
                          ss=float(get('MODFLOW_SS', d.ss)),
                          thickness=float(get('MODFLOW_AQUIFER_THICKNESS', d.thickness)),
                          river_leakance=float(get('MODFLOW_RIVER_LEAKANCE', d.river_leakance)),
                          seepage_leakance=float(get('MODFLOW_SEEPAGE_LEAKANCE', d.seepage_leakance)))


def write_coupling_record(sim_dir: Path, land: str, land_settings: Path, land_output: Path, start) -> None:
    Path(sim_dir, COUPLING_FILE).write_text(json.dumps({
        'land_model': land, 'land_settings': str(land_settings), 'land_output': str(land_output),
        'recharge_start': pd.Timestamp(start).strftime('%Y-%m-%d')}, indent=1))


def read_coupling_record(sim_dir: Path) -> Dict:
    path = Path(sim_dir, COUPLING_FILE)
    if not path.exists():
        raise ValidationError(f'{path} missing; was the run prepared with land-model recharge?')
    return json.loads(path.read_text())


def read_groundwater_discharge(sim_dir: Path) -> pd.Series:
    """Daily groundwater discharge to the river network (m3/s) of a finished basin run."""
    start = read_coupling_record(sim_dir)['recharge_start']
    return (read_budget(sim_dir, start).discharge / 86400.0).rename('gw_discharge_m3s')


def apply_basin_parameters(sim_dir: Path, values: Dict[str, float], base: MF6BasinParams) -> bool:
    """Rewrite parameter packages of a basin-mode simulation; False if ``sim_dir`` is not one."""
    grid_file = Path(sim_dir) / GRID_FILE
    if not grid_file.exists():
        return False
    write_parameter_packages(sim_dir, MF6BasinGrid.from_json(grid_file), base.updated(values))
    return True
