# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MizuRoute remapping file generation sub-module.

Handles creation of remapping NetCDF files that map between source model
spatial units and routing network HRUs, supporting area-weighted and
spatial intersection-based remapping approaches.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import easymore
import geopandas as gpd
import netCDF4 as nc4
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.path_resolver import (
    find_basin_shapefile,
    find_catchment_subfile,
    find_river_basins_shapefile,
)

if TYPE_CHECKING:
    from symfluence.models.mizuroute.preprocessor import MizuRoutePreProcessor

logger = logging.getLogger(__name__)


def _create_easymore_instance():
    """Create an EASYMORE instance handling different module structures."""
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise ModelExecutionError("easymore module does not expose an Easymore class")


class MizuRouteRemapGenerator:
    """
    Generates remapping NetCDF files for mizuRoute.

    Supports area-weighted, equal-weight, and spatial intersection-based
    remapping between source model HRUs and routing network HRUs.

    Delegates to the parent preprocessor for config access, logger,
    directory paths, and mixin properties.

    Args:
        preprocessor: Parent MizuRoutePreProcessor instance.
    """

    def __init__(self, preprocessor: 'MizuRoutePreProcessor'):
        self.pp = preprocessor

    # =========================================================================
    # Area-weighted remapping
    # =========================================================================

    def create_area_weighted_remap_file(self):
        """Create remapping file with area-based weights from delineated catchments"""
        self.pp.logger.info("Creating area-weighted remapping file")

        # Load topology to get HRU information
        topology_file = self.pp.setup_dir / self.pp.mizu_topology_file
        with xr.open_dataset(topology_file) as topo:
            hru_ids = topo['hruId'].values

        n_hrus = len(hru_ids)

        # Use the weights stored during topology creation
        if hasattr(self.pp, 'subcatchment_weights') and hasattr(self.pp, 'subcatchment_gru_ids'):
            weights = self.pp.subcatchment_weights
            gru_ids = self.pp.subcatchment_gru_ids
        else:
            # Fallback: load from delineated catchments shapefile (nested layout)
            catchment_path = find_catchment_subfile(
                self.pp.project_dir / 'shapefiles',
                self.pp.domain_definition_method,
                self.pp.experiment_id,
                f"{self.pp.domain_name}_catchment_delineated.shp",
                logger=self.pp.logger,
            )
            if catchment_path is None:
                raise ModelExecutionError(
                    f"Delineated catchment shapefile not found for {self.pp.domain_name}"
                )
            shp_catchments = gpd.read_file(catchment_path)
            weights = shp_catchments['avg_subbas'].values
            # Same ids, same order, as the topology's hruId for this layout
            # (create_network_topology_file takes them from GRU_ID too).
            gru_ids = shp_catchments['GRU_ID'].values.astype(int)
        lumped_gru_id = self._lumped_summa_gru_id()
        remap_name = self.pp.mizu_remap_file
        if not remap_name:
            remap_name = "remap_file.nc"
            self.pp.logger.warning(f"SETTINGS_MIZU_REMAP not found in config, using default: {remap_name}")

        with nc4.Dataset(self.pp.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Area-weighted remapping for lumped to distributed routing')

            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU

            # Create variables
            # RN_hruId: The routing HRU IDs (from delineated catchments)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = gru_ids
            rn_hru.long_name = 'River network HRU ID'

            # nOverlaps: Each HRU gets input from the single lumped SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'

            # HM_hruId: the lumped SUMMA GRU id for each entry. mizuRoute
            # matches it against the runoff file's gruId (<vname_hruid>), which
            # SUMMA copies from attributes.nc, so it must be that id, not 1.
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [lumped_gru_id] * n_hrus
            hm_hru.long_name = 'ID of overlapping HM_HRUs'

            # weight: Area-based weights from delineated catchments
            weight_var = ncid.createVariable('weight', 'f8', ('data',))
            weight_var[:] = weights
            weight_var.long_name = 'Areal weights based on delineated subcatchment areas'

        self.pp.logger.info(f"Area-weighted remapping file created with {n_hrus} HRUs")
        self.pp.logger.info(f"Weight range: {weights.min():.4f} to {weights.max():.4f}")
        self.pp.logger.info(f"Weight sum: {weights.sum():.4f}")

    def _lumped_summa_gru_id(self) -> int:
        """GRU id of the lumped SUMMA domain, read from SUMMA's attributes file.

        Falls back to 1 (with a warning) when the attributes file is
        unavailable or has no gruId.
        """
        settings_dir = self.pp._get_config_value(
            lambda: self.pp.config.model.summa.settings_path,
            default='default', dict_key='SETTINGS_SUMMA_PATH',
        )
        if not settings_dir or settings_dir == 'default':
            settings_dir = self.pp.project_dir / 'settings' / 'SUMMA'
        attr_name = self.pp._get_config_value(
            lambda: self.pp.config.model.summa.attributes,
            default='attributes.nc', dict_key='SETTINGS_SUMMA_ATTRIBUTES',
        ) or 'attributes.nc'
        attributes_path = Path(settings_dir) / attr_name
        try:
            with xr.open_dataset(attributes_path) as attrs:
                gru_ids = attrs['gruId'].values.astype(int).ravel()
        except (OSError, KeyError, ValueError) as exc:
            self.pp.logger.warning(
                f"Could not read gruId from SUMMA attributes {attributes_path} ({exc}); "
                "assuming lumped GRU id 1 for HM_hruId"
            )
            return 1
        if gru_ids.size == 0:
            self.pp.logger.warning(
                f"SUMMA attributes {attributes_path} has no gruId; assuming lumped GRU id 1"
            )
            return 1
        if gru_ids.size > 1:
            self.pp.logger.warning(
                f"Lumped-to-distributed remap expects 1 SUMMA GRU but {attributes_path} has "
                f"{gru_ids.size}; using the first ({gru_ids[0]})"
            )
        return int(gru_ids[0])

    # =========================================================================
    # Equal-weight remapping
    # =========================================================================

    # =========================================================================
    # SUMMA catchment-to-routing remapping
    # =========================================================================

    def remap_summa_catchments_to_routing(self):
        """
        Create remapping file from SUMMA catchments to routing network HRUs.

        Computes spatial intersection between hydrological model (HM) catchments
        and routing model (RM) basins to create area-weighted remapping. This
        enables routing when the source model uses different spatial units than
        the river network topology.

        For lumped domains with river_network delineation, creates area-weighted
        remapping that distributes runoff proportionally to HRU areas. For
        distributed domains, performs full spatial intersection using EASYMORE.

        The workflow:
        1. Load HM catchment and RM basin shapefiles
        2. Perform spatial intersection (reproject to EPSG:6933 for accuracy)
        3. Calculate area weights for each HM-RM overlap
        4. Write remapping NetCDF with overlap counts and weights

        Configuration keys used:
        - CATCHMENT_PATH, CATCHMENT_SHP_NAME: Source model catchments
        - RIVER_BASINS_PATH, RIVER_BASINS_NAME: Routing network basins
        - INTERSECT_ROUTING_PATH, INTERSECT_ROUTING_NAME: Intersection output
        - SETTINGS_MIZU_REMAP: Output remapping file name

        Note:
            Requires geopandas and EASYMORE for spatial intersection operations.
        """
        self.pp.logger.info("Remapping SUMMA catchments to routing catchments")
        routing_delineation = self.pp._get_config_value(
            lambda: self.pp.config.domain.delineation.routing,
            dict_key='ROUTING_DELINEATION',
        )
        if self.pp.domain_definition_method == 'lumped' and routing_delineation == 'river_network':
            self.pp.logger.info("Area-weighted mapping for SUMMA catchments to routing catchments")
            self.create_area_weighted_remap_file()
            return

        hm_path_cfg = self.pp._get_config_value(
            lambda: self.pp.config.paths.catchment_path,
            dict_key='CATCHMENT_PATH',
        )
        hm_name_cfg = self.pp._get_config_value(
            lambda: self.pp.config.paths.catchment_name,
            dict_key='CATCHMENT_SHP_NAME',
        )
        rm_path_cfg = self.pp._get_config_value(
            lambda: self.pp.config.paths.river_basins_path,
            dict_key='RIVER_BASINS_PATH',
        )
        rm_name_cfg = self.pp._get_config_value(
            lambda: self.pp.config.paths.river_basins_name,
            dict_key='RIVER_BASINS_NAME',
        )

        intersect_path_value = self.pp._get_config_value(
            lambda: self.pp.config.paths.intersect_routing_path,
            dict_key='INTERSECT_ROUTING_PATH',
        )
        intersect_name = self.pp._get_config_value(
            lambda: self.pp.config.paths.intersect_routing_name,
            dict_key='INTERSECT_ROUTING_NAME',
        )
        if intersect_name == 'default':
            intersect_name = 'catchment_with_routing_basins.shp'

        if intersect_path_value == 'default':
            intersect_path = self.pp.project_dir / 'shapefiles/catchment_intersection'
        else:
            intersect_path = Path(intersect_path_value)

        remap_name = self.pp.mizu_remap_file
        if not remap_name:
            remap_name = "remap_file.nc"
            self.pp.logger.warning(f"SETTINGS_MIZU_REMAP not found in config, using default: {remap_name}")

        # Resolve the hydrologic-model catchment (HRUs) and routing-model basins
        # via the shared finders (nested layout + casing drift), honouring any
        # explicit CATCHMENT_PATH / RIVER_BASINS_PATH overrides.
        shapefiles_dir = self.pp.project_dir / 'shapefiles'
        hm_shapefile = find_basin_shapefile(
            shapefiles_dir, self.pp.domain_name,
            self.pp.domain_definition_method, self.pp.experiment_id,
            explicit_path=hm_path_cfg, explicit_name=hm_name_cfg,
            include_river_basins=False, required=True, logger=self.pp.logger,
        )
        if rm_path_cfg and rm_path_cfg != 'default' and rm_name_cfg and rm_name_cfg != 'default':
            rm_shapefile = Path(rm_path_cfg) / rm_name_cfg
        else:
            rm_shapefile = find_river_basins_shapefile(
                shapefiles_dir, self.pp.domain_name, required=True, logger=self.pp.logger,
            )

        # Load shapefiles
        hm_shape = gpd.read_file(hm_shapefile)
        rm_shape = gpd.read_file(rm_shapefile)

        # Create intersection
        esmr_obj = _create_easymore_instance()
        hm_shape = hm_shape.to_crs('EPSG:6933')
        rm_shape = rm_shape.to_crs('EPSG:6933')
        intersected_shape = esmr_obj.intersection_shp(rm_shape, hm_shape)
        intersected_shape = intersected_shape.to_crs('EPSG:4326')
        intersected_shape.to_file(intersect_path / intersect_name)

        # Process variables for remapping file
        self._process_remap_variables(intersected_shape)

        # Create remapping netCDF file
        self._create_remap_file(intersected_shape, remap_name)

        self.pp.logger.info(f"Remapping file created at {self.pp.setup_dir / remap_name}")

    # =========================================================================
    # Remap NetCDF helpers
    # =========================================================================

    def _routing_basin_id_field(self) -> str:
        """GRU id column of the routing (river) basins shapefile."""
        return self.pp._get_config_value(
            lambda: self.pp.config.paths.river_basin_rm_gruid,
            default='GRU_ID', dict_key='RIVER_BASIN_SHP_RM_GRUID',
        ) or 'GRU_ID'

    def _process_remap_variables(self, intersected_shape):
        int_rm_id = f"S_1_{self._routing_basin_id_field()}"
        int_hm_id = f"S_2_{self.pp._get_config_value(lambda: self.pp.config.paths.catchment_gruid, default='GRU_ID')}"
        int_weight = 'AP1N'

        intersected_shape = intersected_shape.sort_values(by=[int_rm_id, int_hm_id])

        self.nc_rnhruid = intersected_shape.groupby(int_rm_id).agg({int_rm_id: 'first'}).values.astype(int)
        self.nc_noverlaps = intersected_shape.groupby(int_rm_id).agg({int_hm_id: 'count'}).values.astype(int)

        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_hm_id: list}).values.tolist()
        self.nc_hmgruid = [item for sublist in multi_nested_list for item in sublist[0]]

        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_weight: list}).values.tolist()
        self.nc_weight = [item for sublist in multi_nested_list for item in sublist[0]]

    def _create_remap_file(self, intersected_shape, remap_name):
        num_hru = len(intersected_shape[f"S_1_{self._routing_basin_id_field()}"].unique())
        num_data = len(intersected_shape)

        with nc4.Dataset(self.pp.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            self._set_remap_attributes(ncid)
            self._create_remap_dimensions(ncid, num_hru, num_data)
            self._create_remap_variables(ncid)

    def _set_remap_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a remapping .nc file for mizuRoute routing')

    def _create_remap_dimensions(self, ncid, num_hru, num_data):
        ncid.createDimension('hru', num_hru)
        ncid.createDimension('data', num_data)

    def _create_remap_variables(self, ncid):
        self._create_and_fill_nc_var(ncid, 'RN_hruId', 'int', 'hru', self.nc_rnhruid, 'River network HRU ID', '-')
        self._create_and_fill_nc_var(ncid, 'nOverlaps', 'int', 'hru', self.nc_noverlaps, 'Number of overlapping HM_HRUs for each RN_HRU', '-')
        self._create_and_fill_nc_var(ncid, 'HM_hruId', 'int', 'data', self.nc_hmgruid, 'ID of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')
        self._create_and_fill_nc_var(ncid, 'weight', 'f8', 'data', self.nc_weight, 'Areal weight of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, fill_data, long_name, units):
        ncvar = ncid.createVariable(var_name, var_type, (dim,))
        ncvar[:] = fill_data
        ncvar.long_name = long_name
        ncvar.units = units
