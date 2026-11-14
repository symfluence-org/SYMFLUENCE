# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE inputs for a whole-country domain from a delivered hydrofabric (landslíkan, Iceland).

The fabric is consumed as delivered: no delineation, no zonal statistics, no remapping. Every table is keyed on
the fabric's unit id, which becomes the HYPE subid. See docs/adr/0012-national-hype-mf6.md.

Layers read from the fabric GeoPackage (``landslikan_hydrofabric_<version>.gpkg`` in the build directory):

* ``divides``: id, toid (0 = terminal), area_km2, terminal_type, geometry (EPSG:3057)
* ``reaches``: id, length_m, slope, q_prior_m3s
* ``attributes``: elev_mean, elev_std, glacier_area_km2, ice_volume_km3
* ``hru_classes``: id, land, soil, band, area_km2, elev_mean (joint land x soil x 200 m band)
* ``lakes_routing``: lake_id, unit_id (outlet unit), area_km2, reservoir, max_depth_m_wfd
* ``transfers``: transfer_id, type, from_unit, to_unit, fraction, capacity_m3s, status

Files written: GeoData.txt, GeoClass.txt, GlacierData.txt, BranchData.txt (every transfer ever operated),
ForcKey.txt, ``national_transfers.csv`` (with commissioning years, for ``branch_eras``) and ``national_fabric.json``
(what was assumed, e.g. default branch capacities and merged transfers).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Fabric hydrological land classes (landslikan.hru.LAND) -> IGBP land-use ids, so the existing par.txt generator
# (HYPEConfigManager._generate_landuse_params) gives each class its usual defaults and its own calibration column.
LAND_TO_IGBP = {1: 15,   # glacier            -> snow and ice
                2: 17,   # lake               -> water (outlet lake / internal lake classes)
                3: 16,   # bare rock, sand, lava -> barren
                4: 7,    # sparse vegetation (moss) -> open shrubland
                5: 10,   # heath, grass, shrub -> grassland
                6: 11,   # wetland            -> permanent wetland
                7: 4,    # forest (birch)     -> deciduous broadleaf
                8: 12,   # agriculture, artificial -> cropland
                9: 17}   # other open water   -> water
SPECIAL_ILAKE, SPECIAL_OLAKE, SPECIAL_GLACIER = 1, 2, 3
WATER_LAND = (2, 9)
# Soil layers (m) as in the Tungnaá member; per-soil depths are calibration targets, not data.
SOIL_LAYERS = {1: (0.091, 0.493, 2.296), 2: (0.091, 0.493, 2.296), 3: (0.091, 0.493, 2.296)}
STREAM_DEPTH = 2.296
DEFAULT_LAKE_DEPTH_M = 5.0
# Default headrace/diversion capacity: this multiple of the source reach's prior mean flow (fabric q_prior_m3s).
DEFAULT_CAPACITY_X_MEAN = 1.5
# HYPE glacier volume-area scaling for ice caps (glactype 1), Radić & Hock (2010), V [m3] = c * A[m2]^e
GLAC_COEF_ICECAP, GLAC_EXP_ICECAP = 1.701, 1.25


@dataclass
class NationalHYPEInputs:
    geodata: pd.DataFrame
    geoclass: pd.DataFrame
    glacierdata: pd.DataFrame
    branchdata: pd.DataFrame          # all transfers ever operated (routing order)
    transfers: pd.DataFrame = None    # HYPE-routed transfers with commissioned/decommissioned years
    notes: Dict[str, list] = field(default_factory=dict)


def fabric_gpkg(fabric_dir: Path) -> Path:
    hits = sorted(Path(fabric_dir).glob('landslikan_hydrofabric_*.gpkg'))
    if len(hits) != 1:
        raise ValidationError(f'Expected one landslikan_hydrofabric_*.gpkg in {fabric_dir}, found {len(hits)}')
    return hits[0]


def _read(fabric_dir: Path, name: str) -> pd.DataFrame:
    import pyogrio
    gpkg = fabric_gpkg(fabric_dir)
    if name not in {lyr[0] for lyr in pyogrio.list_layers(gpkg)}:
        raise ValidationError(f'Fabric layer {name} not in {gpkg}')
    if name == 'divides':
        import geopandas as gpd
        return gpd.read_file(gpkg, layer=name, engine='pyogrio')
    df = pyogrio.read_dataframe(gpkg, layer=name, read_geometry=False)
    return df.replace({'nan': np.nan, 'None': np.nan, '': np.nan}) if name in ('transfers', 'lakes_routing') else df


def topological_order(ids: np.ndarray, edges: List[tuple]) -> List[int]:
    """Kahn order with every edge source before its target (surface tree plus branches). Raises on cycles."""
    succ, indeg = defaultdict(list), {int(i): 0 for i in ids}
    for a, b in edges:
        if b in indeg and a in indeg:
            succ[a].append(b)
            indeg[b] += 1
    q = deque(sorted(i for i, d in indeg.items() if d == 0))
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(indeg):
        raise ValidationError(f'Routing graph with branches has a cycle ({len(indeg) - len(out)} subbasins unordered)')
    return out


def build_slc(classes: pd.DataFrame, outlet_lake_units: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SLC definitions and per-unit fractions + elevations.

    One SLC per (land, soil) combination, except: glacier is one class (special 3, soil ignored); water in a unit
    whose outlet is a lake is the outlet lake (special 2, one class); all other water is internal lake (special 1).
    Returns (slc table, long table id/slc/frac/elev).
    """
    c = classes.copy()
    c['kind'] = 'land'
    is_water = c.land.isin(WATER_LAND)
    c.loc[c.land == 1, 'kind'] = 'glacier'
    c.loc[is_water & c.id.isin(outlet_lake_units) & (c.land == 2), 'kind'] = 'olake'
    c.loc[is_water & (c.kind == 'land'), 'kind'] = 'ilake'
    c['soil_k'] = np.where(c.kind == 'land', c.soil, 0)
    c['land_k'] = np.where(c.kind.isin(['olake', 'ilake']), 2, c.land)
    c['elev_x_area'] = c.elev_mean.fillna(0) * c.area_km2
    c['has_elev'] = np.where(c.elev_mean.notna(), c.area_km2, 0.0)
    g = c.groupby(['id', 'kind', 'land_k', 'soil_k'], as_index=False)[['area_km2', 'elev_x_area', 'has_elev']].sum()
    g['elev'] = g.elev_x_area / g.has_elev.replace(0, np.nan)
    keys = g[['kind', 'land_k', 'soil_k']].drop_duplicates().sort_values(['kind', 'land_k', 'soil_k'])
    order = {'glacier': 0, 'olake': 1, 'ilake': 2, 'land': 3}
    keys = keys.assign(o=keys.kind.map(order)).sort_values(['o', 'land_k', 'soil_k']).drop(columns='o')
    keys['slc'] = np.arange(1, len(keys) + 1)
    g = g.merge(keys, on=['kind', 'land_k', 'soil_k'])
    g['frac'] = g.area_km2 / g.groupby('id').area_km2.transform('sum')
    special = {'glacier': SPECIAL_GLACIER, 'olake': SPECIAL_OLAKE, 'ilake': SPECIAL_ILAKE, 'land': 0}
    slc = pd.DataFrame({
        'slc': keys.slc.values,
        'landuse': keys.land_k.map(LAND_TO_IGBP).values,
        'soil': np.where(keys.soil_k > 0, keys.soil_k, 1),
        'special': keys.kind.map(special).values,
        'kind': keys.kind.values, 'land': keys.land_k.values})
    return slc, g[['id', 'slc', 'frac', 'elev']]


def branch_rows(tr: pd.DataFrame, notes: Optional[Dict[str, list]] = None) -> pd.DataFrame:
    """BranchData rows from transfers (one branch per source subbasin: HYPE allows one; the lowest-numbered
    transfer from a source is kept). Headraces/diversions take flow up to their capacity (default
    DEFAULT_CAPACITY_X_MEAN x the source's prior mean flow); natural bifurcations split by fraction (default 0.5)."""
    notes = notes if notes is not None else defaultdict(list)
    rows = []
    tr = tr.assign(_n=tr.transfer_id.str.extract(r'(\d+)$')[0].astype(float))
    for src, grp in tr.groupby('from_unit'):
        grp = grp.sort_values(['_n', 'transfer_id'])
        t = grp.iloc[0]
        if len(grp) > 1:
            notes['transfers_merged_one_branch_per_source'].append(
                f"{'/'.join(grp.transfer_id)} from unit {int(src)}: kept {t.transfer_id} -> unit {int(t.to_unit)}")
        if t.type == 'natural_bifurcation':
            frac = float(t.fraction) if pd.notna(t.fraction) else 0.5
            if pd.isna(t.fraction):
                notes['default_bifurcation_fraction'].append(f'{t.transfer_id}: 0.5 to branch')
            rows.append(dict(sourceid=int(src), branchid=int(t.to_unit), mainpart=1.0 - frac, maxqmain=0.0, maxqbranch=0.0))
        else:   # headrace / diversion: the branch takes the flow up to its capacity, the rest stays in the river
            cap = float(t.capacity_m3s) if pd.notna(t.capacity_m3s) else DEFAULT_CAPACITY_X_MEAN * float(t.q_prior_m3s)
            if pd.isna(t.capacity_m3s):
                notes['default_branch_capacity'].append(f'{t.transfer_id}: {cap:.1f} m3/s = {DEFAULT_CAPACITY_X_MEAN} x prior mean')
            rows.append(dict(sourceid=int(src), branchid=int(t.to_unit), mainpart=0.0, maxqmain=0.0, maxqbranch=cap))
    return pd.DataFrame(rows, columns=['sourceid', 'branchid', 'mainpart', 'maxqmain', 'maxqbranch'])


def branch_eras(transfers: pd.DataFrame, start, end) -> List[tuple]:
    """Split [start, end] at 1 January of every commissioning/decommissioning year inside it.

    Returns (era_start, era_end, BranchData, active transfer ids). A transfer is active in year y when
    commissioned <= y (or unknown: natural bifurcations) and y < decommissioned (or none).
    HYPE's BranchData is static, so a period crossing a commissioning date is run as consecutive eras with the
    model state carried across (national_coupling.run_eras)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    years = pd.concat([transfers.commissioned, transfers.decommissioned]).dropna().astype(int)
    cuts = sorted({pd.Timestamp(f'{y}-01-01') for y in years if start < pd.Timestamp(f'{y}-01-01') <= end})
    bounds = [start] + cuts + [end + pd.Timedelta(days=1)]
    eras = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        on = ((transfers.commissioned.isna() | (transfers.commissioned <= a.year))
              & (transfers.decommissioned.isna() | (a.year < transfers.decommissioned)))
        eras.append((a, b - pd.Timedelta(days=1), branch_rows(transfers[on]), sorted(transfers.transfer_id[on])))
    return eras


def build_inputs(fabric_dir: Path) -> NationalHYPEInputs:
    fabric_dir = Path(fabric_dir)
    notes: Dict[str, list] = defaultdict(list)
    div = _read(fabric_dir, 'divides')
    rch = _read(fabric_dir, 'reaches').set_index('id')
    att = _read(fabric_dir, 'attributes').set_index('id')
    cls = _read(fabric_dir, 'hru_classes')
    lakes = _read(fabric_dir, 'lakes_routing')
    tr = _read(fabric_dir, 'transfers')
    ids = div.id.to_numpy(int)

    # --- branches: every transfer ever operated (the routing order must hold for all of them); the BranchData
    # active on a given date comes from branch_eras(). Groundwater edges belong to MODFLOW, not HYPE routing.
    for t in tr.query("type == 'groundwater'").transfer_id:
        notes['transfers_left_to_groundwater_model'].append(str(t))
    tr = tr[tr.status.isin(['resolved', 'ok']) & (tr.type != 'groundwater')].copy()
    for col in ('fraction', 'capacity_m3s', 'commissioned', 'decommissioned'):
        tr[col] = pd.to_numeric(tr[col], errors='coerce')
    tr['q_prior_m3s'] = rch.q_prior_m3s.reindex(tr.from_unit.astype(int)).to_numpy(float)
    branch = branch_rows(tr, notes)

    # --- order: upstream before downstream, sources before branch targets
    toid = div.set_index('id').toid.astype(int)
    edges = [(int(a), int(b)) for a, b in toid.items() if b != 0] + list(zip(branch.sourceid, branch.branchid))
    order = topological_order(ids, edges)

    # --- classes
    olake_units = set(lakes.unit_id.astype(int))
    slc, frac = build_slc(cls, olake_units)
    F = frac.pivot(index='id', columns='slc', values='frac').reindex(index=order, columns=slc.slc).fillna(0.0)
    E = frac.pivot(index='id', columns='slc', values='elev').reindex(index=order, columns=slc.slc)

    gd = pd.DataFrame(index=pd.Index(order, name='subid'))
    gd['maindown'] = toid.reindex(order).values
    gd['grwdown'] = gd.maindown
    gd['area'] = (div.set_index('id').area_km2.reindex(order) * 1e6).values
    cen = div.set_index('id').geometry.reindex(order).representative_point().set_crs(div.crs).to_crs(4326)
    gd['latitude'], gd['longitude'] = cen.y.values, cen.x.values
    gd['elev_mean'] = att.elev_mean.reindex(order).values
    gd['elev_std'] = att.elev_std.reindex(order).fillna(0).values
    gd['slope_mean'] = rch.slope.reindex(order).clip(lower=1e-4).values
    gd['rivlen'] = rch.length_m.reindex(order).values
    depth = lakes.drop_duplicates('unit_id').set_index('unit_id').max_depth_m_wfd
    gd['lake_depth'] = 0.0
    has_olake = F[slc.slc[slc.kind == 'olake']].sum(axis=1) > 0
    gd.loc[has_olake.index[has_olake], 'lake_depth'] = (
        depth.reindex(has_olake.index[has_olake]).mul(0.5).fillna(DEFAULT_LAKE_DEPTH_M).clip(lower=1.0).values)
    for s in slc.slc:
        gd[f'slc_{s}'] = F[s].values
    dh = E.sub(gd.elev_mean.values, axis=0).fillna(0.0)
    for s in slc.slc:
        gd[f'dhslc_{s}'] = dh[s].values
    gd = gd.reset_index()
    sums = gd[[f'slc_{s}' for s in slc.slc]].sum(axis=1)
    if (sums - 1).abs().max() > 1e-6:
        raise ValidationError(f'SLC fractions do not sum to 1 (max error {(sums - 1).abs().max():.2e})')
    if not np.isfinite(gd[['area', 'elev_mean', 'rivlen']].to_numpy()).all():
        bad = gd.subid[~np.isfinite(gd[['area', 'elev_mean', 'rivlen']].to_numpy()).all(axis=1)].tolist()
        raise ValidationError(f'Non-finite area/elevation/river length for subids {bad[:10]}')

    # --- glaciers: volume-area correction so HYPE starts from the fabric's ice volume
    gslc = slc.slc[slc.kind == 'glacier']
    glac = pd.DataFrame(columns=['subid', 'glactype', 'logvolcor'])
    if len(gslc):
        a_m2 = (F[gslc.iloc[0]] * gd.set_index('subid').area.reindex(F.index)).rename('a')
        vol = att.ice_volume_km3.reindex(F.index).fillna(0) * 1e9
        on = a_m2 > 0
        v_scale = GLAC_COEF_ICECAP * a_m2[on] ** GLAC_EXP_ICECAP
        lvc = np.log(np.where(vol[on] > 0, vol[on], v_scale) / v_scale)
        glac = pd.DataFrame({'subid': a_m2.index[on], 'glactype': 1, 'logvolcor': np.round(lvc, 6)})
        notes['glacier'].append(f'{on.sum()} subbasins with glacier; logvolcor matches fabric ice volume '
                                f'({vol[on].sum() / 1e9:,.0f} km3)')
    notes['lakes'].append(f'{int(has_olake.sum())} outlet lakes (natural rating, default parameters; '
                          f'{int(lakes.reservoir.sum())} reservoirs not regulated yet)')
    geoclass = slc.copy()
    return NationalHYPEInputs(gd, geoclass, glac, branch, tr, dict(notes))


def write_inputs(inp: NationalHYPEInputs, out_dir: Path) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    inp.geodata.to_csv(d / 'GeoData.txt', sep='\t', index=False, float_format='%.6g')
    hdr = ('!          SLC\tLULC\tSOIL TYPE\tMain crop cropid\tSecond crop cropid\tCrop rotation group\t'
           'Vegetation type\tSpecial class code\tTile depth\tStream depth\tNumber of soil layers\t'
           'Soil layer depth 1\tSoil layer depth 2\tSoil layer depth 3')
    lines = [hdr]
    for r in inp.geoclass.itertuples():
        veg = 3 if r.kind in ('olake', 'ilake') else 2 if r.landuse in (1, 2, 3, 4, 5) else 1
        nl = 1 if r.kind in ('olake', 'ilake') else 3
        depths = SOIL_LAYERS[int(r.soil)]
        lines.append('\t'.join(str(v) for v in (
            r.slc, r.landuse, r.soil, 0, 0, 0, veg, r.special, 0, STREAM_DEPTH, nl, *depths)))
    (d / 'GeoClass.txt').write_text('\n'.join(lines) + '\n')
    inp.glacierdata.to_csv(d / 'GlacierData.txt', sep='\t', index=False)
    if len(inp.branchdata):
        inp.branchdata.to_csv(d / 'BranchData.txt', sep='\t', index=False, float_format='%.6g')
    if inp.transfers is not None:   # read back by national_coupling to split runs at commissioning dates
        inp.transfers[['transfer_id', 'type', 'from_unit', 'to_unit', 'fraction', 'capacity_m3s', 'q_prior_m3s',
                       'commissioned', 'decommissioned']].to_csv(d / 'national_transfers.csv', index=False)
    pd.DataFrame({'subid': inp.geodata.subid, 'stationid': inp.geodata.subid}).to_csv(
        d / 'ForcKey.txt', sep='\t', index=False)
    (d / 'national_fabric.json').write_text(json.dumps(
        {'n_subbasins': len(inp.geodata), 'n_slc': len(inp.geoclass),
         'slc': inp.geoclass.to_dict(orient='records'), 'notes': inp.notes}, indent=1, default=int))
