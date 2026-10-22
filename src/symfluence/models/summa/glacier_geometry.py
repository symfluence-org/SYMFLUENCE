"""Build explicit multi-GRU glacier inputs from georeferenced ice thickness.

Each GRU contains one glacier complex; HRUs retain fractional clean-ice/land
surface domains. Geometry and volume come from supplied rasters, never a default
ice thickness. Glacier complexes are hydrologic partitions, not ice-flow models.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


def prepare_glacier_geometry(attributes: Path, coldstate: Path, catchment: Path,
                             dem: Path, outlines: Path, thickness_dir: Path,
                             resolution: float = 100.0, hru_id: str = 'HRU_ID',
                             target_area_km2: float | None = None) -> dict:
    """Extend base files in place, rejecting missing/invalid geometry.

    ``target_area_km2`` shrinks an outline older than the simulation start: the
    thinnest ice cells (margins, where retreat removes ice first) are dropped
    until the total glacier area is at most the target.

    The supplied thickness is an estimate of full ice geometry. SUMMA's five
    active thermal ice layers are distinct from that full glacier-volume store.
    Initial ice temperature/phase are explicit cold-start assumptions; they are
    equilibrated during the unscored spinup.
    """
    with xr.open_dataset(attributes) as src:
        attrs = src.load()
    with xr.open_dataset(coldstate) as src:
        base = src.load()
    if 'dom' in base.dims or 'grid' in attrs.dims:
        raise ValueError('Glacier preparation requires fresh ordinary SUMMA inputs')
    hrus = gpd.read_file(catchment)
    hrus.geometry = hrus.geometry.make_valid()
    hrus = hrus.to_crs(hrus.estimate_utm_crs())
    hrus = hrus.set_index(hru_id).loc[attrs.hruId.values]
    ice = gpd.read_file(outlines).to_crs(hrus.crs)
    ice.geometry = ice.geometry.make_valid()
    tiles = sorted(Path(thickness_dir).glob('*thickness.tif'))
    if not tiles:
        raise ValueError('No thickness rasters supplied')
    ids, grus = attrs.hruId.values, attrs.gruId.values
    areas = attrs.HRUarea.values
    mapping = attrs.hru2gruId.values
    grids, ice_area, ice_elev = [], np.zeros(len(ids)), attrs.elevation.values.copy()
    coverage_rows = []

    def warp(path, shape, transform):
        target = np.full(shape, np.nan, dtype='f8')
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), target, src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform,
                      dst_crs=hrus.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        return target

    for gru in grus:
        rows = hrus.loc[ids[mapping == gru]]
        xmin, ymin, xmax, ymax = rows.total_bounds
        xmin, ymax = np.floor(xmin/resolution)*resolution, np.ceil(ymax/resolution)*resolution
        nx, ny = int(np.ceil((xmax-xmin)/resolution)), int(np.ceil((ymax-ymin)/resolution))
        transform = from_origin(xmin, ymax, resolution, resolution)
        shape = (ny, nx)
        hid = rasterize([(geom, int(i)) for i, geom in rows.geometry.items()],
                        out_shape=shape, transform=transform, fill=-999, dtype='int32')
        mask = rasterize([(geom, 1) for geom in ice.geometry], out_shape=shape,
                         transform=transform, fill=0).astype(bool) & (hid > 0)
        thick = np.zeros(shape)
        if mask.any():
            for tile in tiles:
                values = warp(tile, shape, transform)
                thick = np.maximum(thick, np.nan_to_num(values, nan=0.0))
            covered = mask & (thick > 0)
            coverage = covered.sum()/mask.sum()
            if coverage < .95:
                raise ValueError(f'GRU {gru}: thickness covers only {coverage:.1%} of glacier cells')
            mask = covered
        surface = warp(dem, shape, transform)
        if not np.isfinite(surface[hid > 0]).all():
            raise ValueError(f'GRU {gru}: DEM missing inside HRUs')
        surface = np.nan_to_num(surface)
        grids.append(dict(gru=gru, hid=hid, mask=mask, thick=np.where(mask, thick, 0.), surface=surface,
                          nx=nx, ny=ny))
    trimmed_area = 0.
    if target_area_km2 is not None:
        keep = int(np.floor(target_area_km2*1e6/resolution**2))
        if keep < 1:
            raise ValueError('target_area_km2 leaves no glacier cells')
        # Rank every ice cell across GRUs by thickness and drop the thinnest beyond `keep`.
        cells = [(j, idx) for j, g in enumerate(grids) for idx in zip(*np.nonzero(g['mask']))]
        thickness = np.array([grids[j]['thick'][idx] for j, idx in cells])
        for k in np.argsort(thickness, kind='stable')[:max(0, len(cells)-keep)]:
            j, idx = cells[k]
            grids[j]['mask'][idx] = False
            grids[j]['thick'][idx] = 0.
            trimmed_area += resolution**2
    for g in grids:
        for j in np.flatnonzero(mapping == g['gru']):
            cells = g['mask'] & (g['hid'] == ids[j])
            ice_area[j] = min(float(cells.sum()*resolution**2), float(areas[j]))
            if cells.any():
                ice_elev[j] = g['surface'][cells].mean()
        coverage_rows.append(dict(gru=int(g['gru']), ice_grid_area_m2=float(g['mask'].sum()*resolution**2),
                                  ice_volume_m3=float(g['thick'].sum()*resolution**2)))
    if not ice_area.any():
        raise ValueError('No positive glacier area')

    nx, ny = max(g['nx'] for g in grids), max(g['ny'] for g in grids)
    ng = len(grus)
    has = np.array([g['mask'].any() for g in grids], dtype='i4')
    attrs['nGlac'] = ('gru', has)
    attrs['nWtld'] = ('gru', np.zeros(ng, dtype='i4'))
    attrs['nGrid'] = ('gru', has)
    attrs['gridId'] = (('grid', 'gru'), grus.reshape(1, -1))
    for name in ['nx', 'ny']:
        attrs[name] = (('grid', 'gru'), np.array([[g[name] for g in grids]], dtype='i4'))
    for name in ['dx', 'dy']:
        attrs[name] = (('grid', 'gru'), np.full((1, ng), resolution))
    grid_dims = ('ygrid', 'xgrid', 'grid', 'gru')
    for name, key in [('bed_elev', 'surface'), ('glacierMask', 'mask'), ('cell2hruId', 'hid')]:
        data = np.zeros((ny, nx, 1, ng), dtype='f8')
        for j, g in enumerate(grids):
            values = g[key] - g['thick'] if name == 'bed_elev' else g[key]
            data[:g['ny'], :g['nx'], 0, j] = values
        attrs[name] = (grid_dims, data)

    # Preserve the entire upland cold-start state and its actual layer counts.
    cold = xr.Dataset()
    n = len(ids)
    maxlayer = max(5, base.sizes['midToto'])
    for name, var in base.data_vars.items():
        if name in ('hruId', 'gruId'):
            cold[name] = var
        elif 'hru' in var.dims:
            target_shape = tuple(maxlayer+1 if dim == 'ifcToto' else
                                 maxlayer if dim == 'midToto' else base.sizes[dim]
                                 for dim in var.dims)
            values = np.zeros(target_shape+(2,))
            sl = tuple(slice(0, k) for k in var.shape)+(0,)
            values[sl] = var.values
            cold[name] = (var.dims+('dom',), values)
        else:
            cold[name] = var
    cold['hruId'] = ('hru', ids)
    cold['gruId'] = ('gru', grus)
    # The glacier reader distinguishes soil-only from total-layer dimensions;
    # ordinary SUMMA cold states often store matric head on midToto instead.
    if 'mLayerMatricHead' in base:
        nsoil = int(base.nSoil.max())
        heads = np.full((nsoil, n, 2), -1.)
        heads[:, :, 0] = base.mLayerMatricHead.values[:nsoil, :]
        cold = cold.drop_vars('mLayerMatricHead')
        cold['mLayerMatricHead'] = (('midSoil', 'hru', 'dom'), heads)
    scalar_dims = ('scalarv', 'hru', 'dom')
    def scalar(name, land, glacial):
        a = np.zeros((1, n, 2))
        a[0, :, 0], a[0, :, 1] = land, glacial
        cold[name] = (scalar_dims, a)
    scalar('domType', 1, np.where(ice_area > 0, 3, -999))
    scalar('DOMarea', areas-ice_area, ice_area)
    scalar('DOMelev', attrs.elevation.values, ice_elev)
    for name, attr, default in [('DOMtan_slope', 'tan_slope', .01),
                               ('DOMaspect', 'aspect', 180.),
                               ('DOMcontourLength', 'contourLength', resolution)]:
        val = attrs[attr].values if attr in attrs else np.full(n, default)
        scalar(name, val, val)
    scalar('scalarAblFrac', 0, 1)
    scalar('nLake', 0, 0)
    scalar('nGlce', 0, 5)
    for name in ('nSnow', 'nSoil'):
        cold[name].values[:, :, 1] = 0
    depths = np.array([.15, .45, 2.25, 7., 30.])
    cold['iLayerHeight'].values[:6, :, 1] = np.r_[0., depths.cumsum()][:, None]
    if 'mLayerDepth' not in cold:
        cold['mLayerDepth'] = (('midToto', 'hru', 'dom'), np.diff(cold.iLayerHeight.values, axis=0))
    else:
        cold['mLayerDepth'].values[:5, :, 1] = depths[:, None]
    for name, val in [('mLayerTemp', 268.15), ('mLayerVolFracIce', 1.), ('mLayerVolFracLiq', 0.)]:
        cold[name].values[:5, :, 1] = val
    for name, val in [('scalarCanairTemp', 268.15), ('scalarCanopyTemp', 268.15),
                      ('scalarSnowAlbedo', .8), ('dt_init', 3600.)]:
        if name in cold:
            cold[name].values[:, :, 1] = val
        else:
            scalar(name, val, val)
    for name in ['glacMass4AreaChange', 'scalarGlceWE']:
        scalar(name, 0, 0)
    # Surface routing reservoirs are initialized empty; no glacier aquifer.
    cold['scalarAquiferStorage'].values[:, :, 1] = 0
    glarea = np.array([ice_area[mapping == gid].sum() for gid in grus])
    cold['glacierAblArea'] = (('glac', 'gru'), glarea[None, :])
    cold['glacierAccArea'] = (('glac', 'gru'), np.zeros((1, ng)))
    cold['glacId'] = (('glac', 'gru'), grus[None, :])
    cold['gridId'] = attrs.gridId
    density = 917.
    cold['basin__GlacierStorage'] = ('gru', np.array([g['thick'].sum()*resolution**2*density/1e12 for g in grids]))
    for name in ['surface_elev', 'debris_thick']:
        data = np.zeros((ny, nx, 1, ng))
        if name == 'surface_elev':
            for j, g in enumerate(grids):
                data[:g['ny'], :g['nx'], 0, j] = g['surface']
        cold[name] = (grid_dims, data)
    assert np.allclose(cold.DOMarea.sum('dom').values.ravel(), areas)
    attrs.to_netcdf(attributes)
    cold.to_netcdf(coldstate)
    report = dict(grus=ng, hrus=n, glacier_grus=int(has.sum()), glacier_hrus=int((ice_area > 0).sum()),
                  glacier_area_km2=float(ice_area.sum()/1e6), initial_ice_storage_gt=float(cold.basin__GlacierStorage.sum()),
                  resolution_m=resolution, crs=str(hrus.crs), grids=coverage_rows,
                  target_area_km2=target_area_km2, trimmed_area_km2=trimmed_area/1e6,
                  assumptions='Clean-ice complex per GRU; all ice routed as ablation-zone ice/snow; no debris/firn classification. '
                              'Five active thermal layers; full geometry from thickness raster. Cold start 268.15 K, fully frozen ice (zero liquid).')
    (attributes.parent/'glacier_geometry.json').write_text(json.dumps(report, indent=2))
    return report
