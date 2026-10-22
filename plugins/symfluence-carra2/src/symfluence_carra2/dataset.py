from symfluence.data.preprocessing.dataset_handlers.carra_utils import CARRAHandler

class CARRA2Handler(CARRAHandler):
    """Use the supplied native GRIB projection, never a CARRA1 default."""
    def get_variable_mapping(self):
        return {}  # Acquisition already writes SYMFLUENCE standard names.

    def process_dataset(self, ds):
        if 'MAP_PROJ4_STR' not in ds.attrs:
            raise ValueError('CARRA2 requires the projection decoded from its GRIB grid')
        return super().process_dataset(ds)

    def create_shapefile(self, shapefile_path, merged_forcing_path, dem_path, elevation_calculator):
        """Packed native cells are independent polygons, not a lat/lon meshgrid."""
        from pathlib import Path
        import numpy as np
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Polygon
        files = sorted(Path(merged_forcing_path).glob('*.nc'))
        if not files:
            raise FileNotFoundError('No processed CARRA2 files')
        with self.open_dataset(files[0]) as ds:
            if ds.attrs.get('spatial_layout') != 'packed_native_cells':
                return super().create_shapefile(shapefile_path, merged_forcing_path, dem_path, elevation_calculator)
            if ds.latitude.dims != ('point',) or ds.longitude.dims != ('point',):
                raise ValueError('Packed CARRA2 requires point coordinates')
            lat, lon = ds.latitude.values, ds.longitude.values
            proj = ds.attrs['MAP_PROJ4_STR']
            dx, dy = float(ds.attrs['DX']), float(ds.attrs['DY'])
        if not np.isfinite([dx, dy]).all() or min(dx,dy) <= 0:
            raise ValueError('Invalid CARRA2 native grid spacing')
        forward = Transformer.from_crs(4326, proj, always_xy=True)
        inverse = Transformer.from_crs(proj, 4326, always_xy=True)
        x, y = forward.transform(lon, lat)
        if not np.isfinite([x,y]).all():
            raise ValueError('Invalid projected CARRA2 coordinates')
        geometries = []
        for xx, yy in zip(x, y):
            vx = [xx-dx/2,xx-dx/2,xx+dx/2,xx+dx/2]
            vy = [yy-dy/2,yy+dy/2,yy+dy/2,yy-dy/2]
            lng, lt = inverse.transform(vx,vy)
            geometries.append(Polygon(zip(lng,lt)))
        frame = gpd.GeoDataFrame({'ID':np.arange(len(lat)),
                                  self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'):lat,
                                  self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'):lon},
                                 geometry=geometries,crs=4326)
        if not frame.geometry.is_valid.all():
            raise ValueError('Invalid native CARRA2 cell polygons')
        frame['elev_m'] = elevation_calculator(frame,dem_path,batch_size=50)
        target=Path(shapefile_path)/'forcing_CARRA2.shp'
        target.parent.mkdir(parents=True,exist_ok=True)
        frame.to_file(target)
        return target
