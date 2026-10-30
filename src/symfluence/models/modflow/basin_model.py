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
* GHB ``LAKE`` (optional): cells under a lake polygon exchange with the lake
  stage, conductance ``lake_leakance`` x the lake area in the cell. Seepage and
  river drains are scaled out under the lake so nothing drains below its stage.
* Optional horizontal anisotropy: ``K22 = k * k_anisotropy`` with the major
  axis rotated ``k_angle`` degrees counter-clockwise from the grid x axis
  (e.g. along a rift or fissure swarm).
* Optional reduced recharge zones (e.g. glacier beds): their recharge is scaled
  by ``zone_recharge_factor``; the rejected part is returned to the river the
  same day, so the land model's water is conserved.
* Stress period 1 is steady state at the mean recharge; then one transient
  period per day.

Both drains and the lake return water to the river, so simulated groundwater
discharge to the stream is their summed net outflow (plus any rejected recharge). The model writes everything it needs to
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
PARAMS_FILE = 'basin_params.json'  # parameters of the current packages (read back for rejected recharge)


@dataclass
class MF6BasinParams:
    """Calibratable groundwater parameters (units: m, days)."""
    k: float = 5.0                    # horizontal hydraulic conductivity (m/d)
    sy: float = 0.1                   # specific yield (-)
    ss: float = 1e-5                  # specific storage (1/m)
    thickness: float = 100.0          # aquifer thickness below land surface / streambed (m)
    river_leakance: float = 10.0      # river conductance per reach length (m2/d per m)
    seepage_leakance: float = 0.1     # land-surface seepage conductance per area (1/d)
    lake_leakance: float = 1.0        # lake-bed conductance per lake area (1/d); used only with a lake
    k_anisotropy: float = 1.0         # K22 / K (-); 1 = isotropic
    k_angle: float = 0.0              # direction of K, degrees counter-clockwise from the grid x axis
    zone_recharge_factor: float = 1.0  # recharge kept in the grid's reduced recharge zones (0-1)

    # Calibration names -> attribute
    NAMES = {'K': 'k', 'SY': 'sy', 'SS': 'ss', 'AQUIFER_THICKNESS': 'thickness',
             'RIVER_LEAKANCE': 'river_leakance', 'SEEPAGE_LEAKANCE': 'seepage_leakance',
             'LAKE_LEAKANCE': 'lake_leakance', 'K_ANISOTROPY': 'k_anisotropy', 'K_ANGLE': 'k_angle',
             'ZONE_RECHARGE_FACTOR': 'zone_recharge_factor'}
    POSITIVE = ('K', 'SY', 'SS', 'AQUIFER_THICKNESS', 'RIVER_LEAKANCE', 'SEEPAGE_LEAKANCE', 'LAKE_LEAKANCE',
                'K_ANISOTROPY')

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
        for name in self.POSITIVE:
            attr = self.NAMES[name]
            if not getattr(self, attr) > 0:
                raise ConfigurationError(f'MODFLOW basin parameter {name} must be positive, got {getattr(self, attr)}')
        if not 0.0 <= self.zone_recharge_factor <= 1.0:
            raise ConfigurationError(f'ZONE_RECHARGE_FACTOR must be in [0, 1], got {self.zone_recharge_factor}')
        if not -180.0 <= self.k_angle <= 180.0:
            raise ConfigurationError(f'K_ANGLE must be in [-180, 180] degrees, got {self.k_angle}')

    def to_dict(self) -> Dict[str, float]:
        return {a: float(getattr(self, a)) for a in self.NAMES.values()}


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
    lake_frac: Optional[np.ndarray] = None   # (nrow, ncol) fraction of cell under the lake; None = no lake
    lake_stage: Optional[float] = None       # lake stage (m, DEM datum)
    reduced_zones: list = field(default_factory=list)  # recharge zone ids scaled by zone_recharge_factor

    @property
    def has_lake(self) -> bool:
        return self.lake_frac is not None and bool((self.lake_frac > 0).any())

    def lake_array(self) -> np.ndarray:
        """Per-cell lake fraction; zeros when there is no lake."""
        return self.lake_frac if self.lake_frac is not None else np.zeros((self.nrow, self.ncol))

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
             'river': self.river.to_dict(orient='list'), 'notes': self.notes,
             'reduced_zones': [int(z) if isinstance(z, (int, np.integer)) else z for z in self.reduced_zones]}
        if self.lake_frac is not None:
            d.update(lake_frac=self.lake_frac.round(6).tolist(), lake_stage=self.lake_stage)
        Path(path).write_text(json.dumps(d))

    @classmethod
    def from_json(cls, path: Path) -> 'MF6BasinGrid':
        d = json.loads(Path(path).read_text())
        active = np.array(d['active'], bool)
        return cls(nrow=d['nrow'], ncol=d['ncol'], cell_size=d['cell_size'], origin=tuple(d['origin']),
                   crs=d['crs'], active=active, top=np.where(active, np.array(d['top'], float), np.nan),
                   basin_frac=np.array(d['basin_frac'], float), river=pd.DataFrame(d['river']),
                   notes=d.get('notes', []),
                   lake_frac=np.array(d['lake_frac'], float) if 'lake_frac' in d else None,
                   lake_stage=d.get('lake_stage'), reduced_zones=d.get('reduced_zones', []))

    def set_lake(self, lake, stage: float) -> None:
        """Lake polygon(s) (GeoDataFrame/GeoSeries) at ``stage`` (m, DEM datum) -> per-cell lake fraction."""
        cells = self.cell_polygons()
        geom = lake.to_crs(self.crs).union_all()
        frac = np.zeros((self.nrow, self.ncol))
        for r, c, g in zip(cells.row, cells.col, cells.geometry):
            a = g.intersection(geom).area
            if a > 0:
                frac[r - 1, c - 1] = min(a / self.cell_area, 1.0)
        if not (frac > 0).any():
            raise ValidationError('The lake polygon does not overlap the active MODFLOW grid')
        self.lake_frac, self.lake_stage = frac, float(stage)


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
    """(Re)write the parameter-dependent packages: DIS, IC, NPF, STO, both DRNs, the lake GHB and RCH."""
    params.validate()
    d = Path(sim_dir)
    top = np.where(grid.active, grid.top, 0.0)
    lake = grid.lake_array()
    # Thickness is measured below the lower of land surface and streambed, so
    # deeply incised reaches stay inside the aquifer.
    base = top.copy()
    rr = grid.river.row.to_numpy(int) - 1
    rc = grid.river.col.to_numpy(int) - 1
    np.minimum.at(base, (rr, rc), grid.river.strtop.to_numpy(float))
    if grid.has_lake:   # and below the lake stage, so the lake boundary sits inside the aquifer
        base = np.where(lake > 0, np.minimum(base, grid.lake_stage), base)
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
    aniso = ''
    if params.k_anisotropy != 1.0:
        # MF6 orients K along ANGLE1 and K22 perpendicular to it (ellipse formulation)
        aniso = (f'  K22\n    CONSTANT {params.k * params.k_anisotropy:.8g}\n'
                 f'  ANGLE1\n    CONSTANT {params.k_angle:.6g}\n')
    (d / 'gwf.npf').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\n  SAVE_SPECIFIC_DISCHARGE\nEND OPTIONS\n\n'
        f'BEGIN GRIDDATA\n  ICELLTYPE\n    CONSTANT 1\n  K\n    CONSTANT {params.k:.8g}\n{aniso}END GRIDDATA\n')
    (d / 'gwf.sto').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN GRIDDATA\n  ICONVERT\n    CONSTANT 1\n  SS\n    CONSTANT {params.ss:.8g}\n'
        f'  SY\n    CONSTANT {params.sy:.8g}\nEND GRIDDATA\n\n'
        'BEGIN PERIOD 1\n  STEADY-STATE\nEND PERIOD 1\n\nBEGIN PERIOD 2\n  TRANSIENT\nEND PERIOD 2\n')
    riv = grid.river
    # Reaches drawn through a lake are the lake itself: the GHB handles them
    riv = riv[lake[riv.row.to_numpy(int) - 1, riv.col.to_numpy(int) - 1] < 0.5]
    lines = [f'  1 {int(r.row)} {int(r.col)} {r.strtop:.3f} {params.river_leakance * r.length:.6g}'
             for r in riv.itertuples()]
    (d / 'gwf.riv.drn').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  MAXBOUND {len(lines)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIOD 1\n' + '\n'.join(lines) + '\nEND PERIOD 1\n')
    rows, cols = np.nonzero(grid.active)
    seep = [f'  1 {r + 1} {c + 1} {top[r, c]:.3f} '
            f'{params.seepage_leakance * max(grid.basin_frac[r, c] - lake[r, c], 0.0) * grid.cell_area:.6g}'
            for r, c in zip(rows, cols) if grid.basin_frac[r, c] - lake[r, c] > 1e-6]
    (d / 'gwf.seep.drn').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  MAXBOUND {len(seep)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIOD 1\n' + '\n'.join(seep) + '\nEND PERIOD 1\n')
    if grid.has_lake:
        lr, lc = np.nonzero(lake * grid.active > 0)
        ghb = [f'  1 {r + 1} {c + 1} {grid.lake_stage:.3f} {params.lake_leakance * lake[r, c] * grid.cell_area:.6g}'
               for r, c in zip(lr, lc)]
        (d / 'gwf.lake.ghb').write_text(
            'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\n'
            f'BEGIN DIMENSIONS\n  MAXBOUND {len(ghb)}\nEND DIMENSIONS\n\n'
            'BEGIN PERIOD 1\n' + '\n'.join(ghb) + '\nEND PERIOD 1\n')
    weights = pd.read_csv(d / WEIGHTS_FILE)
    factor = {z: params.zone_recharge_factor for z in grid.reduced_zones}
    rch = [f'  1 {int(w.row)} {int(w.col)} {_ts_name(w.zone_id)} {w.mult * factor.get(w.zone_id, 1.0):.8g}'
           for w in weights.itertuples()]
    (d / 'gwf.rch').write_text(
        f'BEGIN OPTIONS\n  AUXILIARY mult\n  AUXMULTNAME mult\n  TS6 FILEIN {RECHARGE_TS}\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  MAXBOUND {len(rch)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIOD 1\n' + '\n'.join(rch) + '\nEND PERIOD 1\n')
    Path(d, PARAMS_FILE).write_text(json.dumps(params.to_dict(), indent=1))


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
        '  RCH6 gwf.rch RCH\n  DRN6 gwf.riv.drn RIV\n  DRN6 gwf.seep.drn SEEP\n'
        + ('  GHB6 gwf.lake.ghb LAKE\n' if grid.has_lake else '') + '  OC6 gwf.oc\nEND PACKAGES\n')
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
    has_lake = any(c.endswith('(LAKE)_OUT') or c == 'LAKE_OUT' for c in df.columns)
    out['lake'] = (col('LAKE', 'OUT') - col('LAKE', 'IN')) if has_lake else 0.0
    sto = [c for c in df.columns if c.startswith('STO-')]
    out['storage_change'] = sum(df[c] for c in sto if c.endswith('_OUT')) - sum(df[c] for c in sto if c.endswith('_IN'))
    out['discharge'] = out.river + out.seepage + out.lake
    out['totim'] = df['TIME'].astype(float).to_numpy()
    out['pct_discrepancy'] = df['PERCENT_DIFFERENCE'].astype(float).to_numpy()
    out = out[out.totim > 1.0 + 1e-9]
    out.index = pd.Timestamp(start_date).normalize() + pd.to_timedelta(np.round(out.totim - 2.0).astype(int), unit='D')
    out = out.drop(columns='totim')
    out['rejected_recharge'] = rejected_recharge(sim_dir, out.index)
    out['discharge'] += out.rejected_recharge
    return out


def rejected_recharge(sim_dir: Path, dates: pd.DatetimeIndex) -> pd.Series:
    """Recharge (m3/d) the reduced zones did not pass to the aquifer; it returns to the river the same day."""
    d = Path(sim_dir)
    zero = pd.Series(0.0, index=dates)
    if not (d / GRID_FILE).exists() or not (d / PARAMS_FILE).exists():
        return zero
    zones = json.loads((d / GRID_FILE).read_text()).get('reduced_zones', [])
    f = json.loads((d / PARAMS_FILE).read_text()).get('zone_recharge_factor', 1.0)
    if not zones or f >= 1.0:
        return zero
    w = pd.read_csv(d / WEIGHTS_FILE)
    cell_area = json.loads((d / GRID_FILE).read_text())['cell_size'] ** 2
    area = w[w.zone_id.isin(zones)].groupby('zone_id').mult.sum() * cell_area   # zone area (m2) as recharged
    names, rows = None, []
    for line in (d / RECHARGE_TS).read_text().splitlines():
        t = line.split()
        if t[:1] == ['NAMES']:
            names = t[1:]
        elif names and len(t) == len(names) + 1:
            try:
                rows.append([float(v) for v in t])
            except ValueError:
                continue
    ts = pd.DataFrame(rows, columns=['time'] + names).set_index('time')
    rate = ts.loc[ts.index >= 1.0].iloc[:len(dates)]   # day i of the record is at model time i + 1
    vol = sum((1.0 - f) * rate[_ts_name(z)].to_numpy() * area[z] for z in zones if z in area.index)
    return pd.Series(np.asarray(vol, float), index=dates[:len(rate)]).reindex(dates).fillna(0.0)


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
                          river_leakance=float(cfg.river_leakance), seepage_leakance=float(cfg.seepage_leakance),
                          lake_leakance=float(cfg.lake_leakance), k_anisotropy=float(cfg.k_anisotropy),
                          k_angle=float(cfg.k_angle), zone_recharge_factor=float(cfg.zone_recharge_factor))


def params_from_flat(config: Dict) -> MF6BasinParams:
    """Basin parameters from a flat (``MODFLOW_*``) config dict."""
    d = MF6BasinParams()
    get = (config or {}).get
    return MF6BasinParams(k=float(get('MODFLOW_K', d.k)), sy=float(get('MODFLOW_SY', d.sy)),
                          ss=float(get('MODFLOW_SS', d.ss)),
                          thickness=float(get('MODFLOW_AQUIFER_THICKNESS', d.thickness)),
                          river_leakance=float(get('MODFLOW_RIVER_LEAKANCE', d.river_leakance)),
                          seepage_leakance=float(get('MODFLOW_SEEPAGE_LEAKANCE', d.seepage_leakance)),
                          lake_leakance=float(get('MODFLOW_LAKE_LEAKANCE', d.lake_leakance)),
                          k_anisotropy=float(get('MODFLOW_K_ANISOTROPY', d.k_anisotropy)),
                          k_angle=float(get('MODFLOW_K_ANGLE', d.k_angle)),
                          zone_recharge_factor=float(get('MODFLOW_ZONE_RECHARGE_FACTOR', d.zone_recharge_factor)))


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
