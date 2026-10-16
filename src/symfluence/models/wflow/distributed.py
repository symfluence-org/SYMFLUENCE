"""DEM-derived Wflow grid with explicit ice storage and native routing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from symfluence.core.exceptions import ValidationError
from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
from symfluence.core.modeling.utilities.elevation_forcing import daily_bands, upper_ice_fractions

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def offset_temperature(forcing, bands, offsets):
    """Apply fixed seasonal temperature offsets (deg C) and recompute Oudin PET consistently."""
    if set(offsets) != set(SEASONS) or not np.isfinite(list(offsets.values())).all():
        raise ValidationError("Temperature offsets need finite DJF, MAM, JJA and SON values")
    month = forcing.time.dt.month.values
    delta = np.zeros(len(month))
    for season, months in SEASONS.items():
        delta[np.isin(month, months)] = offsets[season]
    out = forcing.copy()
    for k in ("temp", "tmin", "tmax"):
        out[k] = forcing[k] + xr.DataArray(delta, dims="time")
    doy = forcing.time.dt.dayofyear.values
    out["pet"] = (("time", "hru"), np.column_stack([
        PETCalculatorMixin.oudin_pet_numpy(out.temp.values[:, i], doy, float(bands.center_lat.iloc[i]))
        for i in range(len(bands))
    ]))
    return out


def prepare(pre):
    c = pre.config
    s = c.model.wflow
    bands, forcing = daily_bands(c.paths.forcing_path, s.hru_file, c.domain.time_start, c.domain.time_end)
    if s.temperature_offsets:
        forcing = offset_temperature(forcing, bands, s.temperature_offsets)
    projected = bands.to_crs(3057)
    res = s.grid_resolution
    x0, y0, x1, y1 = projected.total_bounds
    x0 = np.floor(x0 / res) * res - res
    y1 = np.ceil(y1 / res) * res + res
    nx = int(np.ceil((x1 - x0) / res)) + 1
    ny = int(np.ceil((y1 - y0) / res)) + 1
    transform = from_origin(x0, y1, res, res)
    labels = rasterize(
        [(g, i + 1) for i, g in enumerate(projected.geometry)], out_shape=(ny, nx), transform=transform, fill=0
    )
    gauge_index = None
    gauge_meta = {}
    if s.outlet_mode == 'gauge':
        from pyproj import Transformer

        from .outlet import retain_gauge_cell
        lat, lon = map(float, c.domain.pour_point_coords.split('/'))
        gauge_xy = Transformer.from_crs(4326, 3057, always_xy=True).transform(lon, lat)
        labels, gauge_index, added = retain_gauge_cell(
            labels, transform, projected.geometry.make_valid(), gauge_xy)
        gauge_meta = {'outlet_mode': 'gauge', 'gauge_cell_added': added,
                      'gauge_cell_added_area_km2': res**2 / 1e6 if added else 0.0}
    mask = labels > 0
    dem = np.full((ny, nx), -9999.0)
    with rasterio.open(s.dem_file) as src:
        reproject(
            rasterio.band(src, 1),
            dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=3057,
            dst_nodata=-9999,
            resampling=Resampling.average,
        )
    if np.any(dem[mask] == -9999):
        raise ValidationError("DEM does not cover grid")
    dem[~mask] = -9999
    if gauge_index is None:
        flow = pyflwdir.from_dem(dem, transform=transform, nodata=-9999, outlets="min")
    else:
        from .outlet import flow_to_gauge
        flow, conditioning = flow_to_gauge(dem, transform, gauge_index)
        gauge_meta.update(conditioning)
    ldd = flow.to_array(ftype="ldd").astype(float)
    if len(flow.idxs_pit) != 1:
        raise ValidationError("Expected one basin outlet")
    outlet = int(flow.idxs_pit[0])
    oy, ox = np.unravel_index(outlet, mask.shape)
    area = mask.sum() * res**2
    reference = float(bands.HRU_area.sum())
    if abs(area / reference - 1) > 0.02:
        raise ValidationError("Grid area differs from HRU geometry by >2%")
    upstream = flow.upstream_area(unit="km2")
    river = mask & (upstream >= s.river_threshold_km2)
    river[oy, ox] = True
    ice = np.zeros(mask.shape)
    ice[mask] = upper_ice_fractions(np.full(mask.sum(), res**2), dem[mask], s.glacier_fraction)
    ice_store = np.where(ice > 0, s.glacier_ice_mm, 0)
    glacier_meta = {}
    if s.glacier_outline_file or s.glacier_thickness_dir:
        if not (s.glacier_outline_file and s.glacier_thickness_dir):
            raise ValidationError('Both glacier outlines and thickness directory are required')
        from .glacier_geometry import mapped_glaciers
        ice, ice_store, glacier_meta = mapped_glaciers(
            mask, transform, s.glacier_outline_file, s.glacier_thickness_dir,
            projected.geometry.make_valid().union_all())
    # Raster slopes from the DEM, bounded only to keep flat floodplain cells routable.
    filled = np.where(mask, dem, np.nan)
    from scipy.ndimage import distance_transform_edt

    inds = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    filled = dem[tuple(inds)]
    dy, dx = np.gradient(filled, res)
    slope = np.maximum(np.hypot(dx, dy), 0.001)
    river_length = np.full(mask.shape, res)
    river_slope = slope.copy()
    outlet_meta = gauge_meta.copy()
    if s.corrected_routing:
        from pyproj import Transformer
        conditioned = flow.dem_adjust(dem)
        indices = np.flatnonzero(mask)
        downstream = flow.idxs_ds[indices]
        row, col = np.unravel_index(indices, mask.shape)
        dr, dc = np.unravel_index(downstream, mask.shape)
        lengths = np.hypot(row - dr, col - dc) * res
        lengths[lengths == 0] = res  # terminal half/whole reach unresolved at grid scale
        river_length.ravel()[indices] = lengths
        drops = conditioned.ravel()[indices] - conditioned.ravel()[downstream]
        river_slope.ravel()[indices] = np.maximum(drops / lengths, .0001)
        # The terminal reach has no downstream DEM sample; inherit its upstream channel slope.
        parents = indices[(downstream == outlet) & (indices != outlet)]
        if len(parents):
            river_slope.ravel()[outlet] = np.median(river_slope.ravel()[parents])
        lat, lon = map(float, c.domain.pour_point_coords.split('/'))
        gx, gy = Transformer.from_crs(4326, 3057, always_xy=True).transform(lon, lat)
        distance = float(np.hypot(x0 + (ox + .5)*res - gx, y1 - (oy + .5)*res - gy))
        if distance > 1.5 * res:
            raise ValidationError(f'DEM outlet is {distance:.0f} m from gauge; check delineation')
        outlet_meta.update(dict(gauge_distance_m=distance, discharge_definition='river_only',
                           other_outlet_fluxes='Q_land (m3/s), Q_ssf_m3day (m3/day)',
                           routing_geometry='D8 lengths and conditioned downstream DEM slopes'))
    # Get the standard parameter vocabulary from the lumped builder, then replace topology.
    pre._generate_staticmaps()
    path = pre.settings_dir / s.staticmaps_file
    with xr.open_dataset(path) as original:
        old = original.load()
    coords = {"y": y1 - (np.arange(ny) + 0.5) * res, "x": x0 + (np.arange(nx) + 0.5) * res}
    ds = xr.Dataset(coords=coords)
    for name, v in old.data_vars.items():
        val = float(np.nanmean(v.values))
        if "layer" in v.dims:
            ds[name] = xr.DataArray(
                np.where(mask[None, :, :], val, np.nan) * np.ones((v.sizes["layer"], 1, 1)), dims=["layer", "y", "x"]
            )
        else:
            ds[name] = xr.DataArray(np.where(mask, val, np.nan), dims=["y", "x"])
    values = dict(
        wflow_dem=dem,
        wflow_ldd=ldd,
        wflow_river=river.astype(float),
        wflow_subcatch=np.ones(mask.shape),
        wflow_pits=(ldd == 5).astype(float),
        wflow_cellarea=np.full(mask.shape, res**2),
        wflow_gauges=np.where((np.indices(mask.shape)[0] == oy) & (np.indices(mask.shape)[1] == ox), 1, 0),
        wflow_riverlength=river_length,
        wflow_riverwidth=np.maximum(5, 2 * np.sqrt(np.maximum(upstream, 1))),
        Slope=slope,
        RiverSlope=river_slope,
        wflow_glacierfrac=ice,
        wflow_glacierstore=ice_store,
        G_Cfmax=np.full(mask.shape, 5.0),
        G_TT=np.zeros(mask.shape),
        G_SIfrac=np.full(mask.shape, 0.001),
        CanopyGapFraction=np.ones(mask.shape),
        Cmax=np.zeros(mask.shape),
        LAI=np.zeros(mask.shape),
    )
    for k, v in values.items():
        ds[k] = xr.DataArray(np.where(mask, v, np.nan), dims=["y", "x"])
    ds.attrs.update(crs="EPSG:3057", native_area_m2=float(area), reference_area_m2=reference)
    ds.to_netcdf(path)
    grid = xr.Dataset(coords={"time": forcing.time.values + np.timedelta64(1, "D"), **coords})
    for k in ["precip", "temp", "pet"]:
        grid[k] = xr.DataArray(forcing[k].values[:, np.maximum(labels - 1, 0)], dims=["time", "y", "x"])
    grid.to_netcdf(pre.forcing_out_dir / "forcing.nc")
    pre._generate_toml_config()
    toml = pre.settings_dir / s.config_file
    text = toml.read_text()
    import re

    text = re.sub(
        r"starttime = .*", f"starttime = {pd.Timestamp(forcing.time.values[0]).strftime('%Y-%m-%dT00:00:00')}", text
    )
    text = re.sub(
        r"endtime = .*",
        f"endtime = {(pd.Timestamp(forcing.time.values[-1]) + pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')}",
        text,
    )
    text = re.sub(r"timestepsecs = .*", "timestepsecs = 86400", text)
    text = text.replace(
        "[input.static]",
        '[input.static]\nglacier_surface__area_fraction = "wflow_glacierfrac"\nglacier_ice__initial_leq_depth = "wflow_glacierstore"\nglacier_ice__degree_day_coefficient = "G_Cfmax"\nglacier_ice__melting_temperature_threshold = "G_TT"\nglacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction = "G_SIfrac"',
    )
    text = text.replace(
        "[model]",
        f"[model]\ncell_length_in_meter__flag = true\nwater_mass_balance__flag = true\nglacier__flag = {str(bool(ice.any())).lower()}"
        + ("\nsoil_infiltration_reduction__flag = true" if s.frozen_soil else ""),
    )
    text = re.sub(r'reducer = "mean"', f"index.x = {ox + 1}\nindex.y = {ny - oy}", text)
    # NetCDF axes are read ascending by Wflow (y index reversed).
    for header, param, reducer in [
        ("P", "atmosphere_water__precipitation_volume_flux", "mean"),
        ("WB_max", "land_water_mass_balance_error__volume_flux", "maximum"),
        ("WB_min", "land_water_mass_balance_error__volume_flux", "minimum"),
        ("SWE", "snowpack__leq_depth", "mean"),
        ("RiverWB_max", "river_water_mass_balance_error__volume_flow_rate", "maximum"),
        ("RiverWB_min", "river_water_mass_balance_error__volume_flow_rate", "minimum"),
        ("SSF_WB_max", "subsurface_water_mass_balance_error__volume_flow_rate", "maximum"),
        ("SSF_WB_min", "subsurface_water_mass_balance_error__volume_flow_rate", "minimum"),
    ]:
        text += f'\n[[output.csv.column]]\nheader = "{header}"\nparameter = "{param}"\nreducer = "{reducer}"\n'
    if ice.any():
        text += "\n[state.variables]\n" if "[state.variables]" not in text else ""
        text = text.replace("[state.variables]", '[state.variables]\nglacier_ice__leq_depth = "ice"')
        text += '\n[[output.csv.column]]\nheader = "Ice"\nparameter = "glacier_ice__leq_depth"\nreducer = "mean"\n'
    if s.corrected_routing:
        for header, parameter in [('Q_land', 'land_surface_water__volume_flow_rate'),
                                  ('Q_ssf_m3day', 'subsurface_water__volume_flow_rate')]:
            text += f'\n[[output.csv.column]]\nheader = "{header}"\nparameter = "{parameter}"\nindex.x = {ox + 1}\nindex.y = {ny - oy}\n'
    if s.cryosphere_constraints:
        if not ice.any():
            raise ValidationError('Glacier constraints require positive mapped glacier area')
        text += '\n[output.netcdf_grid]\npath = "cryosphere.nc"\ncompressionlevel = 2\n[output.netcdf_grid.variables]\n'
        for parameter, name in [('glacier_ice__leq_depth', 'ice'),
                                ('snowpack_dry_snow__leq_depth', 'snow'),
                                ('snowpack_liquid_water__depth', 'snowwater'),
                                ('land_water_storage__total_depth', 'storage'),
                                ('land_surface__evapotranspiration_volume_flux', 'et')]:
            text += f'{parameter} = "{name}"\n'
    toml.write_text(text)
    (pre.settings_dir / "geometry.json").write_text(
        json.dumps(
            dict(
                cells=int(mask.sum()),
                area_km2=area / 1e6,
                reference_area_km2=reference / 1e6,
                glacier_fraction=float(ice[mask].mean()),
                glacier_ice_mm=s.glacier_ice_mm,
                outlet_xy=[float(coords["x"][ox]), float(coords["y"][oy])],
                resolution_m=res,
                days=len(forcing.time),
                mapped_glacier=glacier_meta,
                outlet=outlet_meta,
            ),
            indent=2,
        )
    )
    if s.initial_parameters:
        from .calibration.parameter_manager import WflowParameterManager
        manager = WflowParameterManager(c, pre.logger, pre.settings_dir)
        initial = manager.get_initial_parameters()
        if not manager.update_model_files(initial, pre.settings_dir):
            raise ValidationError('Could not apply configured Wflow initial parameters')
    return True
