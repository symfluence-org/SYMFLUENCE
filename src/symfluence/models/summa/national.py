# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""National SUMMA (glacier branch) inputs from the landslíkan hydrofabric (ADR 0015).

The fabric is consumed as delivered, as for national HYPE (``models/hype/national_fabric.py``): no shapefile or
zonal-statistics pipeline.

* GRU = fabric unit (subid = unit id, so routing needs no remapping).
* HRU = elevation band of a unit: 200 m bands in units with ice, 400 m elsewhere, from ``hru_classes``; a band
  under ``MIN_BAND_FRAC`` of its unit is merged into its smaller neighbour. hruId = unit id * 100 + k.
* Two runs share decisions and parameters: ``ice`` (the units with glacier cells; glacier attributes and a
  two-domain cold state, ice geometry on a per-GRU grid of at most ``MAX_GRID`` cells a side) and ``land`` (all
  others, ordinary attributes and cold state). The glacier build stores grids densely over every GRU, so keeping
  the 9,000-odd ice-free units out of the ice run keeps those arrays at a few hundred MB.
* Forcing: one NetCDF per water year from the 3-h CARRA land-cell archive (``landslikan.carra 3h``) and the
  fabric's ``forcing_carra`` weights; temperature lapsed from CARRA orography to the HRU elevation, pressure
  hypsometrically, specific humidity capped at saturation, longwave scaled with T^4 (constant emissivity).

    python -m symfluence.models.summa.national build  <fabric_dir> <domain_dir>
    python -m symfluence.models.summa.national forcing <fabric_dir> <domain_dir> <carra_3h_dir> <wy0> [<wy1>]
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_BAND_FRAC = 0.05
BAND_ICE_M, BAND_LAND_M, CLASS_BAND_M = 200, 400, 200        # hru_classes bands are 200 m
MAX_GRID, MIN_DX, FABRIC_DX = 150, 100.0, 25.0
LAPSE_K_PER_M = -0.0065
G, RD = 9.80665, 287.05
CARRA_SHAPE = (221, 301)
GLACIER_LAND, WATER_LAND = 1, (2, 9)
# fabric land class -> MODIFIED_IGBP_MODIS_NOAH (as national HYPE's LAND_TO_IGBP); water is never an HRU's class:
# lakes are routed by dRoute, so an HRU takes its largest non-water class
LAND_TO_IGBP = {1: 15, 3: 16, 4: 7, 5: 10, 6: 11, 7: 4, 8: 12}
# fabric soil group -> ROSETTA class (priors; the Tungnaá runs used SANDY LOAM throughout and calibrated k_soil
# and theta_sat): coarse (sand, gravel, lava) -> LOAMY SAND, andosol -> SANDY LOAM, organic (histosol) -> SILT LOAM
SOIL_TO_ROSETTA = {1: 4, 2: 8, 3: 12}
TEMPLATE = (Path(__file__).resolve().parents[5] / 'SYMFLUENCE_data'
            / 'domain_Tungnaa_Mariufossar_SUMMA_V2_19GRU_GLACIER' / 'settings' / 'SUMMA')
COPY = ('modelDecisions.txt', 'localParamInfo.txt', 'basinParamInfo.txt', 'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL',
        'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL')
OUTPUT_CONTROL = """! national SUMMA output (ADR 0015): daily means, GRU runoff for dRoute
hruId
gruId
HRUarea | 1
averageRoutedRunoff | day
pptrate | day
scalarRainfall | day
scalarSnowfall | day
scalarTotalET | day
scalarSWE | day
scalarTotalSoilWat | day
scalarAquiferStorage | day
scalarSurfaceRunoff | day
{glacier}"""
OUTPUT_GLACIER = """scalarGlceMelt | day
scalarGlceWE | day
basin__GlacierStorage | day
basin__GlacierArea | day
glacierRoutedRunoff | day
DOMarea | day
"""


@dataclass
class NationalSUMMA:
    hrus: pd.DataFrame          # one row per HRU: hruId, gruId, run, elevation, area_m2, ice_m2, soil, veg, ...
    bands: pd.DataFrame         # (unit id, 200 m class band) -> hruId, for rasters and class tables


def _table(fabric_dir: Path, name: str) -> pd.DataFrame:
    """A fabric table: parquet where pyarrow is available, else the GeoPackage layer (as national HYPE reads it)."""
    try:
        return pd.read_parquet(Path(fabric_dir) / f'{name}.parquet')
    except ImportError:
        import pyogrio
        gpkg = sorted(Path(fabric_dir).glob('landslikan_hydrofabric_*.gpkg'))[0]
        return pyogrio.read_dataframe(gpkg, layer=name, read_geometry=False)


# ------------------------------------------------------------------------------------------------ HRUs
def _merge_small(areas: np.ndarray) -> np.ndarray:
    """Group labels for consecutive bands after merging every band under MIN_BAND_FRAC into its smaller
    neighbour (repeatedly, smallest first)."""
    groups = [[i] for i in range(len(areas))]
    a = list(areas.astype(float))
    total = sum(a)
    while len(a) > 1:
        k = int(np.argmin(a))
        if a[k] >= MIN_BAND_FRAC * total:
            break
        if k == 0:
            j = 1
        elif k == len(a) - 1:
            j = k - 1
        else:
            j = k - 1 if a[k - 1] <= a[k + 1] else k + 1
        lo, hi = min(j, k), max(j, k)
        groups[lo] = groups[lo] + groups[hi]
        a[lo] = a[lo] + a[hi]
        del groups[hi], a[hi]
    lab = np.empty(len(areas), int)
    for g, members in enumerate(groups):
        lab[members] = g
    return lab


def build_hrus(fabric_dir: Path) -> NationalSUMMA:
    fabric_dir = Path(fabric_dir)
    cls = _table(fabric_dir, 'hru_classes')
    att = _table(fabric_dir, 'attributes').set_index('id')
    wts = _table(fabric_dir, 'forcing_carra')
    ice_units = set(cls.loc[cls.land == GLACIER_LAND, 'id'])
    # cells without DEM (band -1) join the unit's largest band
    big = cls[cls.band >= 0].groupby(['id', 'band']).area_km2.sum().reset_index()
    big = big.loc[big.groupby('id').area_km2.idxmax()].set_index('id').band
    cls['band'] = np.where(cls.band < 0, cls.id.map(big).fillna(0), cls.band).astype(int)
    cls['coarse'] = np.where(cls.id.isin(ice_units), cls.band, cls.band // (BAND_LAND_M // CLASS_BAND_M))
    per = cls.groupby(['id', 'coarse']).area_km2.sum().reset_index().sort_values(['id', 'coarse'])
    lab = np.empty(len(per), int)
    for _, idx in per.groupby('id').indices.items():
        lab[idx] = _merge_small(per.area_km2.to_numpy()[idx])
    per['k'] = lab + 1
    cls = cls.merge(per[['id', 'coarse', 'k']], on=['id', 'coarse'])
    cls['hruId'] = cls.id * 100 + cls.k
    if cls.k.max() >= 100:
        raise ValueError('more than 99 HRUs in a unit')

    def dominant(df, col, exclude=()):
        d = df[~df[col].isin(exclude)].groupby(['hruId', col]).area_km2.sum().reset_index()
        return d.loc[d.groupby('hruId').area_km2.idxmax()].set_index('hruId')[col]

    w = cls.assign(ea=cls.area_km2 * cls.elev_mean.fillna(0), has_e=cls.elev_mean.notna() * cls.area_km2)
    g = w.groupby('hruId')
    h = pd.DataFrame({'gruId': g.id.first(), 'area_km2': g.area_km2.sum(),
                      'elevation': g.ea.sum() / g.has_e.sum().replace(0, np.nan),
                      'ice_km2': w[w.land == GLACIER_LAND].groupby('hruId').area_km2.sum(),
                      'band_lo_m': g.band.min() * CLASS_BAND_M, 'band_hi_m': (g.band.max() + 1) * CLASS_BAND_M})
    h['ice_km2'] = h.ice_km2.fillna(0.0)
    h['elevation'] = h.elevation.fillna(h.gruId.map(att.elev_mean)).fillna(0.0)
    land = dominant(cls, 'land', exclude=WATER_LAND + (GLACIER_LAND,))
    h['vegTypeIndex'] = land.reindex(h.index).map(LAND_TO_IGBP)
    h.loc[h.vegTypeIndex.isna() & (h.ice_km2 > 0), 'vegTypeIndex'] = 16     # all-ice band: barren upland domain
    h['vegTypeIndex'] = h.vegTypeIndex.fillna(16).astype(int)                # all-water band (lake units)
    h['soilTypeIndex'] = dominant(cls, 'soil').reindex(h.index).map(SOIL_TO_ROSETTA).fillna(8).astype(int)
    slope = np.tan(np.radians(h.gruId.map(att.slope_mean_deg).fillna(2.0).clip(0.5, 60)))
    h['tan_slope'] = slope
    h['aspect'] = np.degrees(np.arctan2(h.gruId.map(att.eastness).fillna(0), h.gruId.map(att.northness).fillna(0))) % 360
    wl = wts.assign(wlat=wts.weight * wts.carra_lat, wlon=wts.weight * wts.carra_lon).groupby('unit_id')
    h['latitude'] = h.gruId.map(wl.wlat.sum())
    h['longitude'] = h.gruId.map(wl.wlon.sum())
    h['run'] = np.where(h.gruId.isin(ice_units), 'ice', 'land')
    h = h.reset_index().sort_values(['run', 'gruId', 'hruId']).reset_index(drop=True)
    bands = cls[['id', 'band', 'hruId']].drop_duplicates(['id', 'band']).rename(columns={'id': 'unit'})
    return NationalSUMMA(h, bands)


# ------------------------------------------------------------------------------------------------ files
def _attributes(h: pd.DataFrame):
    import xarray as xr
    grus = h.gruId.drop_duplicates().to_numpy()
    return xr.Dataset({
        'hruId': ('hru', h.hruId.to_numpy('i4')), 'gruId': ('gru', grus.astype('i4')),
        'hru2gruId': ('hru', h.gruId.to_numpy('i4')), 'downHRUindex': ('hru', np.zeros(len(h), 'i4')),
        'longitude': ('hru', h.longitude.to_numpy('f8')), 'latitude': ('hru', h.latitude.to_numpy('f8')),
        'elevation': ('hru', h.elevation.to_numpy('f8')), 'HRUarea': ('hru', h.area_km2.to_numpy('f8') * 1e6),
        'tan_slope': ('hru', h.tan_slope.to_numpy('f8')), 'contourLength': ('hru', np.full(len(h), 100.0)),
        'slopeTypeIndex': ('hru', np.ones(len(h), 'i4')), 'soilTypeIndex': ('hru', h.soilTypeIndex.to_numpy('i4')),
        'vegTypeIndex': ('hru', h.vegTypeIndex.to_numpy('i4')), 'mHeight': ('hru', np.full(len(h), 2.0)),
        'aspect': ('hru', h.aspect.to_numpy('f8'))})


def _coldstate(h: pd.DataFrame):
    """Ordinary cold state as in the Tungnaá runs: three soil layers (0.428, 0.585, 0.889 m), no snow, 283.16 K,
    volumetric liquid 0.2, matric head -1 m, aquifer 2.5 m (spun up before any scored period)."""
    import xarray as xr
    n = len(h)
    depth = np.array([0.428, 0.585, 0.889])
    one = lambda v, dt='f8': (('scalarv', 'hru'), np.full((1, n), v, dt))
    lay = lambda v: (('midToto', 'hru'), np.tile(np.asarray(v, 'f8')[:, None], (1, n)))
    return xr.Dataset({
        'hruId': ('hru', h.hruId.to_numpy('i4')), 'dt_init': one(10800.0), 'nSoil': one(3, 'i4'),
        'nSnow': one(0, 'i4'), 'scalarCanopyIce': one(0.0), 'scalarCanopyLiq': one(0.0), 'scalarSnowDepth': one(0.0),
        'scalarSWE': one(0.0), 'scalarSfcMeltPond': one(0.0), 'scalarAquiferStorage': one(2.5),
        'scalarSnowAlbedo': one(0.0), 'scalarCanairTemp': one(283.16), 'scalarCanopyTemp': one(283.16),
        'mLayerTemp': lay([283.16] * 3), 'mLayerVolFracIce': lay([0.0] * 3), 'mLayerVolFracLiq': lay([0.2] * 3),
        'mLayerMatricHead': (('midSoil', 'hru'), np.full((3, n), -1.0)),        # glacier build reads midSoil
        'mLayerDepth': lay(depth),
        'iLayerHeight': (('ifcToto', 'hru'), np.tile(np.r_[0.0, depth.cumsum()][:, None], (1, n)))})


def _trial_params(h: pd.DataFrame, params: dict):
    import xarray as xr
    grus = h.gruId.drop_duplicates().to_numpy('i4')
    ds = xr.Dataset({'hruId': ('hru', h.hruId.to_numpy('i4')), 'gruId': ('gru', grus)})
    # dimensions as in the template: only routingGamma* are basin (GRU) parameters there; aquifer and glacier
    # storage parameters are local. A local parameter on 'gru' is read at HRU indices, past the end of the GRU
    # dimension once a block's HRUs pass nGRU (read_param fails with a bare "netcdf-file_open/").
    for k, (v, dim) in params.items():
        ds[k] = (dim, np.full(len(grus) if dim == 'gru' else len(h), v))
    return ds


def template_params(template: Path = TEMPLATE) -> dict:
    """The best Tungnaá glacier calibration (summa_v2_19gru_glacier) as national priors: uniform values,
    {name: (value, dimension)}."""
    import xarray as xr
    with xr.open_dataset(template / 'trialParams.nc') as t:
        return {k: (float(np.asarray(v.values).ravel()[0]), v.dims[0]) for k, v in t.data_vars.items()
                if k not in ('hruId', 'gruId')}


def raw_dem(fabric_dir: Path) -> Path:
    """The DEM the fabric's class bands were cut from (landslikan config `dem`, IslandsDEM 25 m on the fabric
    grid). Not dem_filled.tif: that is the conditioned routing surface (hydraulic potential under ice)."""
    return Path(fabric_dir).resolve().parents[2] / 'data' / 'raw' / 'rasters' / 'IslandsDEMv1.0_25x25m_isn93_zmasl.tif'


def glacier_grids(fabric_dir: Path, h: pd.DataFrame, bands: pd.DataFrame, dem_path: Path | None = None) -> dict:
    """Per ice GRU: grid of block size dx (a multiple of 25 m, >= MIN_DX, <= MAX_GRID cells a side) over the
    unit's bounding box, with cell2hruId (-999 outside the unit), glacierMask, surface (block-mean raw DEM) and ice
    thickness (block mean). Reads the 25 m fabric rasters window by window."""
    import rasterio
    from rasterio.windows import Window
    from scipy import ndimage
    fabric_dir = Path(fabric_dir)
    ice_grus = np.sort(h.loc[h.run == 'ice', 'gruId'].unique())
    with rasterio.open(fabric_dir / 'units.tif') as src:
        units = src.read(1)
        transform = src.transform
    objs = ndimage.find_objects(units)        # index k-1 -> slices of unit k
    lut = bands.set_index(['unit', 'band']).hruId
    out = {}
    with rasterio.open(dem_path or raw_dem(fabric_dir)) as dem_src, \
            rasterio.open(fabric_dir / 'glacier.tif') as gl_src, \
            rasterio.open(fabric_dir / 'ice_thickness.tif') as th_src:
        for gid in ice_grus:
            sl = objs[gid - 1]
            ny25, nx25 = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            f = int(max(MIN_DX / FABRIC_DX, np.ceil(max(ny25, nx25) / MAX_GRID)))
            ny, nx = -(-ny25 // f), -(-nx25 // f)
            u = np.zeros((ny * f, nx * f), units.dtype)
            r = units[sl[0].start:sl[0].start + ny * f, sl[1].start:sl[1].start + nx * f]
            u[:r.shape[0], :r.shape[1]] = r
            win = Window(sl[1].start, sl[0].start, r.shape[1], r.shape[0])    # inside the raster (no boundless
                                                                               # reads: they crash with two GDALs)

            def read(s):
                a = np.full((ny * f, nx * f), np.nan)
                v = s.read(1, window=win).astype('f8')
                if s.nodata is not None:
                    v[v == s.nodata] = np.nan
                v[v < -1000] = np.nan
                a[:v.shape[0], :v.shape[1]] = v
                return a
            dem, gl, th = read(dem_src), read(gl_src), read(th_src)
            inside = u == gid
            blk = lambda a: a.reshape(ny, f, nx, f)
            n_in = blk(inside).sum((1, 3))
            keep = n_in * 2 >= f * f                                   # block mostly inside the unit
            demv = np.where(inside & np.isfinite(dem), dem, 0.0)
            surface = blk(demv).sum((1, 3)) / np.maximum(blk(inside & np.isfinite(dem)).sum((1, 3)), 1)
            ice = inside & (np.nan_to_num(gl) > 0)
            ice_frac = blk(ice).sum((1, 3)) / np.maximum(n_in, 1)
            thick = blk(np.where(ice, np.nan_to_num(th), 0.0)).sum((1, 3)) / np.maximum(blk(ice).sum((1, 3)), 1)
            # HRU of each 25 m cell from its band (nearest band of the unit where the class table lacks it), then
            # the block's majority HRU, so grid ice stays in the HRUs the class table puts it in
            known = lut.loc[gid]
            bkeys = known.index.to_numpy()
            band25 = np.where(np.isfinite(dem), dem // CLASS_BAND_M, bkeys[0]).astype(int)
            near = bkeys[np.abs(band25.ravel()[:, None] - bkeys[None, :]).argmin(1)].reshape(band25.shape)
            hru25 = np.where(inside, known.reindex(near.ravel()).to_numpy().reshape(near.shape), 0)
            cands = np.unique(known.to_numpy())
            counts = np.stack([blk(hru25 == k).sum((1, 3)) for k in cands])
            hid = np.full((ny, nx), -999, 'i4')
            ok = keep & np.isfinite(surface)
            hid[ok] = cands[counts.argmax(0)][ok]
            mask = ok & (ice_frac >= 0.5)
            out[gid] = dict(nx=nx, ny=ny, dx=f * FABRIC_DX, hid=hid, mask=mask.astype('i4'),
                            surface=surface, thick=np.where(mask, thick, 0.0))
    return out


def _glacier_files(attrs, cold, h: pd.DataFrame, grids: dict):
    """Extend ordinary attributes/cold state to the glacier build's two-domain form (as glacier_geometry.py, but
    from fabric grids and with ice area per HRU from the grid)."""
    import xarray as xr
    grus = attrs.gruId.values
    ids, mapping, areas = attrs.hruId.values, attrs.hru2gruId.values, attrs.HRUarea.values
    n, ng = len(ids), len(grus)
    ice_area, ice_elev = np.zeros(n), attrs.elevation.values.copy()
    pos = pd.Series(np.arange(n), index=ids)
    for gid, g in grids.items():
        cell = g['dx'] ** 2
        for hid_ in np.unique(g['hid'][g['mask'] > 0]):
            sel = (g['hid'] == hid_) & (g['mask'] > 0)
            j = pos[hid_]
            ice_area[j] = min(sel.sum() * cell, areas[j] * 0.999)
            ice_elev[j] = g['surface'][sel].mean()
    nx, ny = max(g['nx'] for g in grids.values()), max(g['ny'] for g in grids.values())
    gl = [grids[gid] for gid in grus]
    has = np.array([int(g['mask'].any()) for g in gl], 'i4')
    attrs['nGlac'] = ('gru', has)
    attrs['nWtld'] = ('gru', np.zeros(ng, 'i4'))
    attrs['nGrid'] = ('gru', has)
    attrs['gridId'] = (('grid', 'gru'), grus.reshape(1, -1).astype('i4'))
    attrs['nx'] = (('grid', 'gru'), np.array([[g['nx'] for g in gl]], 'i4'))
    attrs['ny'] = (('grid', 'gru'), np.array([[g['ny'] for g in gl]], 'i4'))
    attrs['dx'] = (('grid', 'gru'), np.array([[g['dx'] for g in gl]], 'f8'))
    attrs['dy'] = attrs['dx']
    dims = ('ygrid', 'xgrid', 'grid', 'gru')

    def stack(fn, dtype, fill):
        a = np.full((ny, nx, 1, ng), fill, dtype)
        for j, g in enumerate(gl):
            a[:g['ny'], :g['nx'], 0, j] = fn(g)
        return a
    attrs['bed_elev'] = (dims, stack(lambda g: g['surface'] - g['thick'], 'f4', 0))
    attrs['glacierMask'] = (dims, stack(lambda g: g['mask'], 'i1', 0))
    attrs['cell2hruId'] = (dims, stack(lambda g: g['hid'], 'i4', -999))

    c = xr.Dataset()
    for name, var in cold.data_vars.items():
        if name == 'hruId':
            c[name] = var
            continue
        dims_ = tuple('midToto' if d == 'midToto' else d for d in var.dims)
        shape = tuple(6 if d == 'ifcToto' else 5 if d == 'midToto' else var.sizes[d] for d in var.dims)
        v = np.zeros(shape + (2,), var.dtype)
        v[tuple(slice(0, k) for k in var.shape) + (0,)] = var.values
        c[name] = (dims_ + ('dom',), v)
    c['gruId'] = ('gru', grus)
    heads = np.full((3, n, 2), -1.0)
    heads[:, :, 0] = cold.mLayerMatricHead.values[:3]
    c = c.drop_vars('mLayerMatricHead')
    c['mLayerMatricHead'] = (('midSoil', 'hru', 'dom'), heads)

    def scalar(name, landv, icev, dtype='f8'):
        a = np.zeros((1, n, 2), dtype)
        a[0, :, 0], a[0, :, 1] = landv, icev
        c[name] = (('scalarv', 'hru', 'dom'), a)
    scalar('domType', 1, np.where(ice_area > 0, 3, -999), 'i4')
    scalar('DOMarea', areas - ice_area, ice_area)
    scalar('DOMelev', attrs.elevation.values, ice_elev)
    scalar('DOMtan_slope', attrs.tan_slope.values, attrs.tan_slope.values)
    scalar('DOMaspect', attrs.aspect.values, attrs.aspect.values)
    scalar('DOMcontourLength', 100.0, 100.0)
    scalar('scalarAblFrac', 0, 1)
    scalar('nLake', 0, 0, 'i4')
    scalar('nGlce', 0, 5, 'i4')
    scalar('glacMass4AreaChange', 0, 0)
    scalar('scalarGlceWE', 0, 0)
    for name in ('nSnow', 'nSoil'):
        c[name].values[:, :, 1] = 0
    depths = np.array([.15, .45, 2.25, 7., 30.])
    c['iLayerHeight'].values[:, :, 1] = np.r_[0., depths.cumsum()][:, None]
    c['mLayerDepth'].values[:, :, 1] = depths[:, None]
    for name, val in [('mLayerTemp', 268.15), ('mLayerVolFracIce', 1.), ('mLayerVolFracLiq', 0.)]:
        c[name].values[:, :, 1] = val
    for name, val in [('scalarCanairTemp', 268.15), ('scalarCanopyTemp', 268.15), ('scalarSnowAlbedo', .8),
                      ('dt_init', 10800.), ('scalarAquiferStorage', 0.0)]:
        c[name].values[:, :, 1] = val
    glarea = np.array([ice_area[mapping == gid].sum() for gid in grus])
    c['glacierAblArea'] = (('glac', 'gru'), glarea[None, :])
    c['glacierAccArea'] = (('glac', 'gru'), np.zeros((1, ng)))
    c['glacId'] = (('glac', 'gru'), grus[None, :].astype('i4'))
    c['gridId'] = attrs.gridId
    c['basin__GlacierStorage'] = ('gru', np.array([(g['thick'] * g['dx'] ** 2).sum() * 917.0 / 1e12 for g in gl]))
    c['surface_elev'] = (dims, stack(lambda g: g['surface'], 'f4', 0))
    c['debris_thick'] = (dims, np.zeros((ny, nx, 1, ng), 'f4'))
    assert np.allclose(c.DOMarea.sum('dom').values.ravel(), areas)
    return attrs, c, ice_area


def write_settings(nat: NationalSUMMA, fabric_dir: Path, domain_dir: Path, template: Path = TEMPLATE) -> dict:
    """settings/SUMMA/{ice,land}/ with attributes, cold state, trial parameters, decisions and tables."""
    domain_dir = Path(domain_dir)
    params = template_params(template)
    report = {'template': str(template), 'params': params}
    grids = glacier_grids(fabric_dir, nat.hrus, nat.bands)
    for run, h in nat.hrus.groupby('run'):
        d = domain_dir / 'settings' / 'SUMMA' / run
        d.mkdir(parents=True, exist_ok=True)
        for f in COPY:
            shutil.copy2(template / f, d / f)
        attrs, cold = _attributes(h), _coldstate(h)
        if run == 'ice':
            attrs, cold, ice_area = _glacier_files(attrs, cold, h, grids)
            report['ice'] = dict(grus=int(attrs.sizes['gru']), hrus=len(h),
                                 grid_max=[int(attrs.sizes['ygrid']), int(attrs.sizes['xgrid'])],
                                 ice_km2_grid=float(ice_area.sum() / 1e6), ice_km2_classes=float(h.ice_km2.sum()),
                                 ice_gt=float(cold.basin__GlacierStorage.sum()))
        else:
            report['land'] = dict(grus=int(attrs.sizes['gru']), hrus=len(h))
        # uncompressed, contiguous: SUMMA reads these GRU by GRU, and a compressed variable is decompressed whole
        # on every read (setup ~1 s per GRU)
        attrs.to_netcdf(d / 'attributes.nc', format='NETCDF4_CLASSIC' if run == 'land' else 'NETCDF4')
        cold.to_netcdf(d / 'coldState.nc')
        _trial_params(h, params).to_netcdf(d / 'trialParams.nc')
        (d / 'outputControl.txt').write_text(OUTPUT_CONTROL.format(glacier=OUTPUT_GLACIER if run == 'ice' else ''))
    nat.hrus.to_csv(domain_dir / 'settings' / 'SUMMA' / 'hrus.csv', index=False)
    (domain_dir / 'settings' / 'SUMMA' / 'build.json').write_text(json.dumps(report, indent=2, default=str))
    return report


def write_filemanager(domain_dir: Path, run: str, start: str, end: str, forcing_list: str, out_dir: Path,
                      prefix: str, init: str = 'coldState.nc') -> Path:
    d = Path(domain_dir) / 'settings' / 'SUMMA' / run
    fm = d / f'fileManager_{prefix}.txt'
    fm.write_text(f"""controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'
simStartTime         '{start}'
simEndTime           '{end}'
tmZoneInfo           'utcTime'
outFilePrefix        '{prefix}'
settingsPath         '{d}/'
forcingPath          '{Path(domain_dir) / "forcing" / "SUMMA"}/'
outputPath           '{out_dir}/'
initConditionFile    '{init}'
attributeFile        'attributes.nc'
trialParamFile       'trialParams.nc'
forcingListFile      '{forcing_list}'
decisionsFile        'modelDecisions.txt'
outputControlFile    'outputControl.txt'
globalHruParamFile   'localParamInfo.txt'
globalGruParamFile   'basinParamInfo.txt'
vegTableFile         'TBL_VEGPARM.TBL'
soilTableFile        'TBL_SOILPARM.TBL'
generalTableFile     'TBL_GENPARM.TBL'
noahmpTableFile      'TBL_MPTABLE.TBL'
""")
    return fm


# ------------------------------------------------------------------------------------------------ forcing
def carra_orography(monthly_dir: Path) -> np.ndarray:
    """CARRA model orography (m) on the flat 221 x 301 grid, from a monthly reduction (landslikan.carra)."""
    f = sorted(Path(monthly_dir).glob('*.npz'))[0]
    return np.load(f)['zs'].ravel().astype('f8')


def write_forcing(fabric_dir: Path, domain_dir: Path, h3_dir: Path, wy: int | None = None,
                  hrus: pd.DataFrame | None = None, runs=('ice', 'land'), period: tuple | None = None,
                  tag: str | None = None) -> list:
    """forcing/SUMMA/<run>_WY<wy>.nc: the 3-h periods ending in (Oct 1 00Z of wy-1, Oct 1 00Z of wy], per HRU
    (or ending in (period[0], period[1]], written as <run>_<tag>.nc)."""
    import netCDF4
    from scipy import sparse
    sys.path.insert(0, str(Path(h3_dir).resolve().parents[3] / 'src'))
    from landslikan.carra import read_3h          # noqa: E402  (landslikan repo, same machine)
    domain_dir = Path(domain_dir)
    hrus = hrus if hrus is not None else pd.read_csv(domain_dir / 'settings' / 'SUMMA' / 'hrus.csv')
    t0, t1 = period or (f'{wy - 1}-10-01T00:00', f'{wy}-10-01T00:00')
    tag = tag or f'WY{wy}'
    end, cells, x = read_3h(t0, t1)
    wts = _table(fabric_dir, 'forcing_carra')
    zs = carra_orography(Path(h3_dir).parent / 'monthly')
    col = pd.Series(np.arange(len(cells)), index=cells)
    flat = wts.carra_j.to_numpy() * CARRA_SHAPE[1] + wts.carra_i.to_numpy()
    files = []
    out = domain_dir / 'forcing' / 'SUMMA'
    out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        h = hrus[hrus.run == run].reset_index(drop=True)
        units = h.gruId.drop_duplicates().to_numpy()
        row = pd.Series(np.arange(len(units)), index=units)
        w = wts.unit_id.isin(units).to_numpy()
        W = sparse.csr_matrix((wts.weight.to_numpy()[w], (row[wts.unit_id[w]].to_numpy(), col[flat[w]].to_numpy())),
                              shape=(len(units), len(cells)))
        U = sparse.csr_matrix((np.ones(len(h)), (np.arange(len(h)), row[h.gruId].to_numpy())), shape=(len(h), len(units)))
        M = (U @ W).tocsr()                                           # HRU x cell
        z0 = M @ zs[cells]
        dz = h.elevation.to_numpy() - z0
        ta = (M @ x['t'].T).T + 273.15                               # [time, hru] K at CARRA orography
        tb = ta + LAPSE_K_PER_M * dz
        ps = (M @ x['ps'].T).T * 100.0 * np.exp(-G * dz / (RD * 0.5 * (ta + tb)))
        es = 611.2 * np.exp(17.67 * (tb - 273.15) / (tb - 29.65))
        qsat = 0.622 * es / (ps - 0.378 * es)
        q = np.minimum((M @ np.nan_to_num(x['q'], nan=np.nanmedian(x['q'])).T).T, qsat)
        lw = (M @ x['lw'].T).T * (tb / ta) ** 4
        data = {'airtemp': tb, 'airpres': ps, 'spechum': q, 'LWRadAtm': lw,
                'SWRadAtm': np.clip((M @ x['sw'].T).T, 0, None), 'windspd': np.clip((M @ x['u10'].T).T, 0.1, None),
                'pptrate': np.clip((M @ x['p'].T).T, 0, None) / 3600.0}
        bad = {k: int((~np.isfinite(v)).sum()) for k, v in data.items() if not np.isfinite(v).all()}
        if bad:
            raise ValueError(f'non-finite forcing in {tag} {run}: {bad}')
        f = out / f'{run}_{tag}.nc'
        with netCDF4.Dataset(f.with_suffix('.tmp'), 'w') as ds:
            ds.createDimension('time', None)
            ds.createDimension('hru', len(h))
            t = ds.createVariable('time', 'f8', ('time',))
            t.units, t.calendar = 'hours since 1990-01-01 00:00:00', 'standard'
            t[:] = (end - np.datetime64('1990-01-01T00:00:00')).astype('timedelta64[s]').astype(float) / 3600.0
            hv = ds.createVariable('hruId', 'i4', ('hru',))
            hv[:] = h.hruId.to_numpy()
            ds.createVariable('data_step', 'f8')[...] = 10800.0
            units_ = {'airtemp': 'K', 'airpres': 'Pa', 'spechum': 'g g-1', 'LWRadAtm': 'W m-2', 'SWRadAtm': 'W m-2',
                      'windspd': 'm s-1', 'pptrate': 'kg m-2 s-1'}
            for k, v in data.items():
                var = ds.createVariable(k, 'f4', ('time', 'hru'), zlib=True, complevel=4, chunksizes=(248, len(h)))
                var.units = units_[k]
                var.long_name = 'mean over the 3 h ending at time' if k in ('pptrate', 'SWRadAtm', 'LWRadAtm') else k
                var[:] = v.astype('f4')
            ds.source = 'CARRA (Belgingur archive, 3 h) via landslikan.carra 3h; fabric forcing_carra weights'
        f.with_suffix('.tmp').rename(f)
        files.append(f)
    return files


def main(argv: list) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cmd, fabric, domain = argv[0], Path(argv[1]), Path(argv[2])
    if cmd == 'build':
        nat = build_hrus(fabric)
        rep = write_settings(nat, fabric, domain)
        print(json.dumps({k: v for k, v in rep.items() if k != 'params'}, indent=2))
    elif cmd == 'forcing':
        h3, wy0 = Path(argv[3]), int(argv[4])
        for wy in range(wy0, int(argv[5]) + 1 if len(argv) > 5 else wy0 + 1):
            print([str(f) for f in write_forcing(fabric, domain, h3, wy)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
