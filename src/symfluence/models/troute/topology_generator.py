# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TRoute topology generator.

Subclass of BaseTopologyGenerator that writes NWM-format NetCDF topology
with t-route's expected variable schema (comid, to_node, link_id_hru, etc.).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4 as nc4
import numpy as np

from symfluence.core.modeling.utilities.base_topology_generator import BaseTopologyGenerator, TopologyData

if TYPE_CHECKING:
    from symfluence.models.troute.preprocessor import TRoutePreProcessor


class TRouteTopologyGenerator(BaseTopologyGenerator):
    """
    Generates NWM-format network topology NetCDF files for t-route.

    Produces files with dimensions link/nhru and variables: comid, to_node,
    length, slope, link_id_hru, hru_area_m2, lat, lon, alt, n, channel_width.
    """

    def __init__(self, preprocessor: 'TRoutePreProcessor'):
        super().__init__(preprocessor)

    def get_topology_output_path(self) -> Path:
        topology_name = self.pp._get_config_value(
            lambda: self.pp.config.model.troute.topology_file,
            dict_key='SETTINGS_TROUTE_TOPOLOGY',
        )
        return self.pp.setup_dir / topology_name

    def write_topology_file(self, topology_data: TopologyData, output_path: Path) -> None:
        """Write topology in t-route's NWM NetCDF format."""
        with nc4.Dataset(output_path, 'w', format='NETCDF4') as ncid:
            ncid.setncattr('Author', "Created by SYMFLUENCE workflow for t-route")
            ncid.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            ncid.setncattr('Conventions', "CF-1.6")

            ncid.createDimension('link', topology_data.num_seg)
            ncid.createDimension('nhru', topology_data.num_hru)
            ncid.createDimension('gages', None)

            self._write_var(ncid, 'comid', 'i4', 'link', topology_data.seg_ids, 'Unique segment ID')
            self._write_var(ncid, 'to_node', 'i4', 'link', topology_data.down_seg_ids, 'Downstream segment ID')
            self._write_var(ncid, 'length', 'f8', 'link', topology_data.lengths, 'Segment length', 'meters')
            self._write_var(ncid, 'slope', 'f8', 'link', topology_data.slopes, 'Segment slope', 'm/m')
            # HRU ids, so the runner can match runoff columns by id, not position.
            self._write_var(ncid, 'hruId', 'i8', 'nhru', topology_data.hru_ids, 'Unique HRU ID')
            self._write_var(ncid, 'link_id_hru', 'i4', 'nhru', topology_data.hru_to_seg_ids, 'Segment ID for HRU discharge')
            self._write_var(ncid, 'hru_area_m2', 'f8', 'nhru', topology_data.hru_areas, 'HRU area', 'm^2')

            # Manning's n (uniform)
            mannings_n = self.pp._get_config_value(
                lambda: self.pp.config.model.troute.mannings_n,
                dict_key='TROUTE_MANNINGS_N',
            )
            self._write_var(ncid, 'n', 'f8', 'link', np.full(topology_data.num_seg, float(mannings_n)), 'Mannings roughness coefficient')

            # Elevations if available
            if topology_data.elevations is not None:
                self._write_var(ncid, 'alt', 'f8', 'link', topology_data.elevations, 'Mean elevation of segment', 'meters')
            else:
                self._write_var(ncid, 'alt', 'f8', 'link', np.zeros(topology_data.num_seg), 'Mean elevation of segment', 'meters')

            # Lat/lon from centroids if we have the river shapefile
            self._write_var(ncid, 'lat', 'f8', 'link', np.zeros(topology_data.num_seg), 'Latitude of segment midpoint', 'degrees_north')
            self._write_var(ncid, 'lon', 'f8', 'link', np.zeros(topology_data.num_seg), 'Longitude of segment midpoint', 'degrees_east')
            self._write_var(ncid, 'from_node', 'i4', 'link', np.zeros(topology_data.num_seg, dtype=int), 'Upstream node ID')

            # Channel width from hydraulic geometry (W = a * A^b)
            self._write_channel_width(ncid, topology_data)

        self.pp.logger.info(f"t-route topology file created at {output_path}")

    def write_topology_with_shapefiles(self) -> TopologyData:
        """
        Build topology from shapefiles and write to NWM-format NetCDF.

        Also adds lat/lon centroids and drainage area from the river shapefile.
        Returns the TopologyData for use by the preprocessor.
        """
        topology_data = self.build_topology()
        output_path = self.get_topology_output_path()

        # Try to enrich with shapefile geometry
        try:
            self._enrich_with_shapefile_data(topology_data, output_path)
        except Exception:  # noqa: BLE001
            self.write_topology_file(topology_data, output_path)

        return topology_data

    def _enrich_with_shapefile_data(self, topology_data: TopologyData, output_path: Path) -> None:
        """Write topology with additional data from shapefiles (centroids, drainage area)."""
        import geopandas as gpd

        # Load shapefiles to get centroids
        domain_type = self.detect_domain_type()
        if domain_type == 'point':
            self.write_topology_file(topology_data, output_path)
            return

        try:
            if topology_data.domain_type == 'grid':
                shp_river = gpd.read_file(self.find_grid_basins_path())
            else:
                routing_suffix = 'delineate' if domain_type == 'lumped_to_distributed' else self._get_method_suffix()
                river_network_path = self.pp.project_dir / 'shapefiles/river_network'
                river_network_name = f"{self.pp.domain_name}_riverNetwork_{routing_suffix}.shp"
                shp_river = gpd.read_file(river_network_path / river_network_name)
        except Exception:  # noqa: BLE001
            self.write_topology_file(topology_data, output_path)
            return

        centroids = self.pp.calculate_feature_centroids(shp_river)

        with nc4.Dataset(output_path, 'w', format='NETCDF4') as ncid:
            ncid.setncattr('Author', "Created by SYMFLUENCE workflow for t-route")
            ncid.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            ncid.setncattr('Conventions', "CF-1.6")

            ncid.createDimension('link', topology_data.num_seg)
            ncid.createDimension('nhru', topology_data.num_hru)
            ncid.createDimension('gages', None)

            self._write_var(ncid, 'comid', 'i4', 'link', topology_data.seg_ids, 'Unique segment ID')
            self._write_var(ncid, 'to_node', 'i4', 'link', topology_data.down_seg_ids, 'Downstream segment ID')
            self._write_var(ncid, 'length', 'f8', 'link', topology_data.lengths, 'Segment length', 'meters')
            self._write_var(ncid, 'slope', 'f8', 'link', topology_data.slopes, 'Segment slope', 'm/m')
            # HRU ids, so the runner can match runoff columns by id, not position.
            self._write_var(ncid, 'hruId', 'i8', 'nhru', topology_data.hru_ids, 'Unique HRU ID')
            self._write_var(ncid, 'link_id_hru', 'i4', 'nhru', topology_data.hru_to_seg_ids, 'Segment ID for HRU discharge')
            self._write_var(ncid, 'hru_area_m2', 'f8', 'nhru', topology_data.hru_areas, 'HRU area', 'm^2')

            mannings_n = self.pp._get_config_value(
                lambda: self.pp.config.model.troute.mannings_n,
                dict_key='TROUTE_MANNINGS_N',
            )
            self._write_var(ncid, 'n', 'f8', 'link', np.full(topology_data.num_seg, float(mannings_n)), 'Mannings roughness coefficient')

            # Centroids from shapefile
            lats = centroids.y.values[:topology_data.num_seg] if len(centroids) >= topology_data.num_seg else np.zeros(topology_data.num_seg)
            lons = centroids.x.values[:topology_data.num_seg] if len(centroids) >= topology_data.num_seg else np.zeros(topology_data.num_seg)
            self._write_var(ncid, 'lat', 'f8', 'link', lats, 'Latitude of segment midpoint', 'degrees_north')
            self._write_var(ncid, 'lon', 'f8', 'link', lons, 'Longitude of segment midpoint', 'degrees_east')

            # Elevation
            if topology_data.elevations is not None:
                self._write_var(ncid, 'alt', 'f8', 'link', topology_data.elevations, 'Mean elevation of segment', 'meters')
            elif 'elev_mean' in shp_river.columns:
                self._write_var(ncid, 'alt', 'f8', 'link', shp_river['elev_mean'].values[:topology_data.num_seg].astype(float), 'Mean elevation', 'meters')
            else:
                self._write_var(ncid, 'alt', 'f8', 'link', np.zeros(topology_data.num_seg), 'Mean elevation of segment', 'meters')

            self._write_var(ncid, 'from_node', 'i4', 'link', np.zeros(topology_data.num_seg, dtype=int), 'Upstream node ID')

            # Drainage area and channel width
            if 'DSContArea' in shp_river.columns:
                da_km2 = shp_river['DSContArea'].values[:topology_data.num_seg].astype(np.float64) / 1e6
                self._write_var(ncid, 'drainage_area_km2', 'f8', 'link', da_km2, 'Downstream contributing area', 'km^2')
            self._write_channel_width(ncid, topology_data, shp_river=shp_river)

            if 'strmOrder' in shp_river.columns:
                self._write_var(ncid, 'stream_order', 'i4', 'link', shp_river['strmOrder'].values[:topology_data.num_seg].astype(int), 'Strahler stream order')

        self.pp.logger.info(f"t-route topology file created at {output_path}")

    def _write_channel_width(self, ncid, topology_data: TopologyData, shp_river=None) -> None:
        """Compute channel width from hydraulic geometry: W = a * A^b."""
        hg_a = float(self.pp._get_config_value(
            lambda: self.pp.config.model.troute.hg_width_coeff,
            dict_key='TROUTE_HG_WIDTH_COEFF',
        ))
        hg_b = float(self.pp._get_config_value(
            lambda: self.pp.config.model.troute.hg_width_exp,
            dict_key='TROUTE_HG_WIDTH_EXP',
        ))

        da_km2 = None
        if shp_river is not None and 'DSContArea' in shp_river.columns:
            da_km2 = shp_river['DSContArea'].values[:topology_data.num_seg].astype(np.float64) / 1e6

        if da_km2 is not None:
            da_clamped = np.maximum(da_km2, 0.01)
            channel_width = np.maximum(hg_a * da_clamped ** hg_b, 1.0)
        else:
            # Fallback: estimate from HRU areas
            hru_areas_km2 = topology_data.hru_areas[:topology_data.num_seg] / 1e6 if len(topology_data.hru_areas) >= topology_data.num_seg else np.ones(topology_data.num_seg)
            da_clamped = np.maximum(hru_areas_km2, 0.01)
            channel_width = np.maximum(hg_a * da_clamped ** hg_b, 1.0)

        self._write_var(ncid, 'channel_width', 'f8', 'link', channel_width, 'Channel width from hydraulic geometry', 'meters')

    @staticmethod
    def _write_var(ncid, name, dtype, dim, data, long_name, units='-'):
        """Helper to create and fill a NetCDF variable."""
        var = ncid.createVariable(name, dtype, (dim,))
        var[:] = data.values if hasattr(data, 'values') else data
        var.long_name = long_name
        var.units = units
