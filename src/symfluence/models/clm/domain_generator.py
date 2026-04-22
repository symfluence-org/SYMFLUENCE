# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Domain and ESMF Mesh Generator.

Generates the single-point domain file and ESMF unstructured mesh file
required by the NUOPC coupling driver for CLM5.  Also provides catchment
geometry helpers used by other CLM sub-modules.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .preprocessor import CLMPreProcessor

logger = logging.getLogger(__name__)


class CLMDomainGenerator:
    """Generates CLM domain and ESMF mesh files.

    Also owns catchment geometry helpers (centroid, area, elevation)
    used by other CLM sub-modules.

    Parameters
    ----------
    preprocessor : CLMPreProcessor
        Parent preprocessor providing config access and project paths.
    """

    def __init__(self, preprocessor: CLMPreProcessor) -> None:
        self.pp = preprocessor

    # ------------------------------------------------------------------ #
    #  Catchment geometry helpers
    # ------------------------------------------------------------------ #

    def get_catchment_centroid(self) -> Tuple[float, float, float]:
        """Get catchment centroid lat/lon and area from shapefile."""
        shapefile_path = self.pp._get_config_value(
            lambda: self.pp.config.domain.shapefile_path,
            default=None, dict_key='CATCHMENT_SHP_PATH'
        )

        if shapefile_path and shapefile_path != 'default':
            shapefile = Path(shapefile_path)
        else:
            shapefile = (
                self.pp.project_dir / 'shapefiles' / 'catchment'
                / f'{self.pp.domain_name}_catchment.shp'
            )

        if not shapefile.exists():
            for pattern in ['**/*catchment*.shp', '**/*.shp']:
                shps = list(self.pp.project_dir.glob(pattern))
                if shps:
                    shapefile = shps[0]
                    break

        if shapefile.exists():
            import geopandas as gpd
            gdf = gpd.read_file(shapefile)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            centroid = gdf.geometry.unary_union.centroid
            area_km2 = self._get_catchment_area_from_gdf(gdf)
            return centroid.y, centroid.x, area_km2
        else:
            lat = float(self.pp._get_config_value(
                lambda: self.pp.config.domain.latitude,
                default=51.17, dict_key='LATITUDE'
            ))
            lon = float(self.pp._get_config_value(
                lambda: self.pp.config.domain.longitude,
                default=-115.57, dict_key='LONGITUDE'
            ))
            area = float(self.pp._get_config_value(
                lambda: self.pp.config.domain.catchment_area,
                default=2210.0, dict_key='CATCHMENT_AREA'
            ))
            return lat, lon, area

    @staticmethod
    def _get_catchment_area_from_gdf(gdf) -> float:
        """Calculate catchment area in km2 from GeoDataFrame."""
        try:
            for col in ['Area_km2', 'area_km2', 'AREA_KM2', 'Area']:
                if col in gdf.columns:
                    area_attr = float(gdf[col].iloc[0])
                    if area_attr > 0:
                        return area_attr
            gdf_proj = gdf.to_crs(epsg=3857)
            area_m2 = gdf_proj.geometry.area.sum()
            return area_m2 / 1e6
        except Exception:  # noqa: BLE001 — model execution resilience
            return 2210.0

    def get_mean_elevation(self) -> float:
        """Get mean elevation from HRU shapefile or DEM (GeoTIFF/NetCDF)."""
        # 1. Try HRU shapefile elev_mean attribute
        try:
            gdf = self._load_hru_shapefile()
            if gdf is not None and 'elev_mean' in gdf.columns:
                val = float(gdf['elev_mean'].mean())
                if val > 0:
                    logger.info(f"Elevation from HRU shapefile: {val:.1f} m")
                    return val
        except Exception:  # noqa: BLE001
            pass

        # 2. Try DEM GeoTIFF
        elev_dir = self.pp.project_attributes_dir / 'elevation'
        if elev_dir.exists():
            import rasterio
            for pattern in ['**/*elv*.tif', '**/*dem*.tif', '**/*elevation*.tif']:
                for f in elev_dir.glob(pattern):
                    try:
                        with rasterio.open(f) as src:
                            data = src.read(1)
                            valid = data[(data > -9999) & (data < 9000)]
                            if len(valid) > 0:
                                val = float(valid.mean())
                                logger.info(f"Elevation from DEM GeoTIFF: {val:.1f} m")
                                return val
                    except Exception:  # noqa: BLE001
                        continue

        # 3. Try DEM NetCDF
        if elev_dir.exists():
            for f in elev_dir.glob('*dem*.nc'):
                try:
                    ds = xr.open_dataset(f)
                    for var in ['elev', 'elevation', 'dem', 'z']:
                        if var in ds.data_vars:
                            val = float(ds[var].mean().values)
                            ds.close()
                            logger.info(f"Elevation from DEM NetCDF: {val:.1f} m")
                            return val
                    ds.close()
                except Exception:  # noqa: BLE001
                    continue

        default = float(self.pp._get_config_value(
            lambda: None, default=300.0, dict_key='MEAN_ELEVATION'
        ))
        logger.warning(f"Could not read DEM; using default elevation: {default} m")
        return default

    def get_elevation_stats(self) -> tuple:
        """Get elevation std dev and mean slope from DEM GeoTIFF."""
        elev_dir = self.pp.project_attributes_dir / 'elevation'
        if elev_dir.exists():
            import rasterio
            for pattern in ['**/*elv*.tif', '**/*dem*.tif', '**/*elevation*.tif']:
                for f in elev_dir.glob(pattern):
                    try:
                        with rasterio.open(f) as src:
                            data = src.read(1).astype(float)
                            valid_mask = (data > -9999) & (data < 9000)
                            valid = data[valid_mask]
                            if len(valid) < 4:
                                continue
                            std_elev = float(np.std(valid))
                            res = abs(src.res[0]) * 111000
                            gy, gx = np.gradient(data, res)
                            slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
                            slope_deg = float(np.degrees(slope_rad[valid_mask].mean()))
                            logger.info(f"DEM stats: std_elev={std_elev:.1f} m, slope={slope_deg:.1f} deg")
                            return std_elev, slope_deg
                    except Exception:  # noqa: BLE001
                        continue
        return 100.0, 5.0

    def _load_hru_shapefile(self):
        """Load HRU shapefile from standard project paths."""
        import geopandas as gpd
        search_dirs = [
            self.pp.project_dir / 'shapefiles' / 'catchment' / 'lumped',
            self.pp.project_dir / 'shapefiles' / 'river_basins',
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            for f in d.rglob('*.shp'):
                try:
                    return gpd.read_file(f)
                except Exception:  # noqa: BLE001
                    continue
        return None

    # ------------------------------------------------------------------ #
    #  Domain and mesh file generation
    # ------------------------------------------------------------------ #

    def generate_domain_file(self) -> None:
        """Generate single-point CLM domain file."""
        lat, lon, area_km2 = self.get_catchment_centroid()
        area_m2 = area_km2 * 1e6
        R_earth = 6.371e6
        cell_area_rad = area_m2 / (R_earth ** 2)

        delta = 0.005
        xv = np.array([[lon - delta, lon + delta, lon + delta, lon - delta]])
        yv = np.array([[lat - delta, lat - delta, lat + delta, lat + delta]])

        ds = xr.Dataset({
            'xc': xr.DataArray(np.array([[lon]]), dims=['nj', 'ni'],
                               attrs={'units': 'degrees_east'}),
            'yc': xr.DataArray(np.array([[lat]]), dims=['nj', 'ni'],
                               attrs={'units': 'degrees_north'}),
            'xv': xr.DataArray(xv.reshape(1, 1, 4), dims=['nj', 'ni', 'nv'],
                               attrs={'units': 'degrees_east'}),
            'yv': xr.DataArray(yv.reshape(1, 1, 4), dims=['nj', 'ni', 'nv'],
                               attrs={'units': 'degrees_north'}),
            'mask': xr.DataArray(np.array([[1]], dtype=np.int32), dims=['nj', 'ni']),
            'frac': xr.DataArray(np.array([[1.0]]), dims=['nj', 'ni']),
            'area': xr.DataArray(np.array([[cell_area_rad]]), dims=['nj', 'ni'],
                                 attrs={'units': 'radians^2'}),
        }, attrs={'title': f'CLM domain file for {self.pp.domain_name}',
                  'Conventions': 'CF-1.6', 'source': 'SYMFLUENCE'})

        domain_file = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.domain_file,
            default='domain.nc', dict_key='CLM_DOMAIN_FILE'
        )
        filepath = self.pp.params_dir / domain_file
        ds.to_netcdf(filepath, format='NETCDF4')
        logger.info(f"Generated CLM domain file: {filepath}")
        logger.info(f"  Centroid: ({lat:.4f}, {lon:.4f}), Area: {area_km2:.1f} km2")

    def generate_esmf_mesh(self) -> None:
        """Generate ESMF unstructured mesh file for NUOPC coupling."""
        lat, lon, area_km2 = self.get_catchment_centroid()
        area_m2 = area_km2 * 1e6
        R_earth = 6.371e6
        cell_area_rad = area_m2 / (R_earth ** 2)

        delta = 0.5  # half-degree box for ESMF mesh element
        node_lons = [lon - delta, lon + delta, lon + delta, lon - delta]
        node_lats = [lat - delta, lat - delta, lat + delta, lat + delta]

        ds = xr.Dataset({
            'nodeCoords': xr.DataArray(
                data=np.array([[node_lons[i], node_lats[i]] for i in range(4)]),
                dims=['nodeCount', 'coordDim'],
                attrs={'units': 'degrees'},
            ),
            'elementConn': xr.DataArray(
                data=np.array([[1, 2, 3, 4]], dtype=np.int32),
                dims=['elementCount', 'maxNodePElement'],
                attrs={'long_name': 'Node indices per element', 'start_index': 1},
            ),
            'numElementConn': xr.DataArray(
                data=np.array([4], dtype=np.int32), dims=['elementCount'],
            ),
            'centerCoords': xr.DataArray(
                data=np.array([[lon, lat]]), dims=['elementCount', 'coordDim'],
                attrs={'units': 'degrees'},
            ),
            'elementArea': xr.DataArray(
                data=np.array([cell_area_rad]), dims=['elementCount'],
                attrs={'units': 'radians^2'},
            ),
            'elementMask': xr.DataArray(
                data=np.array([1], dtype=np.int32), dims=['elementCount'],
            ),
            'origGridDims': xr.DataArray(
                data=np.array([1, 1], dtype=np.int32), dims=['origGridRank'],
            ),
        }, attrs={'gridType': 'unstructured', 'version': '0.9',
                  'title': f'ESMF mesh for {self.pp.domain_name}'})

        filepath = self.pp.params_dir / 'esmf_mesh.nc'
        ds.to_netcdf(filepath, format='NETCDF4')
        logger.info(f"Generated ESMF mesh file: {filepath}")
