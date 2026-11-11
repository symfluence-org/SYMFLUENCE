# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH Drainage Database Handler

Handles drainage database topology fixes, completeness, and normalization.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr

from symfluence.core.mixins import ConfigMixin
from symfluence.models.mesh.errors import MESHInputError
from symfluence.models.mesh.preprocessing.config_defaults import (
    is_elevation_band_mode,
    should_force_single_gru,
)


class MESHDrainageDatabase(ConfigMixin):
    """
    Manages MESH drainage database fixes and completeness.

    Handles:
    - Topology fixes when meshflow fails to build routing
    - Adding missing required variables (IREACH, IAK, AL, DA)
    - Reordering by Rank and normalizing GRU fractions
    """

    def __init__(
        self,
        forcing_dir: Path,
        rivers_path: Path,
        rivers_name: str,
        catchment_path: Path,
        catchment_name: str,
        config: Dict[str, Any],
        logger: logging.Logger = None
    ):
        """
        Initialize drainage database handler.

        Args:
            forcing_dir: Directory containing MESH files
            rivers_path: Path to river network directory
            rivers_name: River network shapefile name
            catchment_path: Path to catchment directory
            catchment_name: Catchment shapefile name
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.forcing_dir = forcing_dir
        self.rivers_path = rivers_path
        self.rivers_name = rivers_name
        self.catchment_path = catchment_path
        self.catchment_name = catchment_name
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)

    @property
    def ddb_path(self) -> Path:
        """Path to drainage database file."""
        return self.forcing_dir / "MESH_drainage_database.nc"

    def _get_landcover_class_ids(self, n_land: int) -> Optional[list]:
        """Get actual landcover class IDs from landcover stats file.

        Reads the frac_* column names from the landcover stats CSV to determine
        which NALCMS/IGBP class IDs are present.

        Args:
            n_land: Expected number of landcover classes

        Returns:
            List of integer class IDs, or None if not determinable
        """
        import re

        import pandas as pd

        # Look for landcover stats file in forcing directory or attributes
        possible_paths = [
            self.forcing_dir / "temp_modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv",
            self.forcing_dir.parent.parent / "attributes" / "gistool-outputs" / "modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv",
        ]

        for lc_path in possible_paths:
            if lc_path.exists():
                try:
                    df = pd.read_csv(lc_path)
                    # Extract class IDs from frac_* columns
                    frac_cols = [col for col in df.columns if col.startswith('frac_')]
                    class_ids = []
                    for col in frac_cols:
                        match = re.match(r'frac_(\d+)', col)
                        if match:
                            class_ids.append(int(match.group(1)))

                    if class_ids:
                        self.logger.debug(f"Found landcover class IDs from {lc_path.name}: {sorted(class_ids)}")
                        return sorted(class_ids)
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Failed to read landcover classes from {lc_path}: {e}", exc_info=True)

        return None

    def fix_drainage_topology(self) -> None:
        """
        Fix drainage database topology if meshflow failed to build it properly.

        When meshflow fails to establish routing connectivity, all Next values
        become 0 (all GRUs treated as outlets). This method rebuilds the topology
        from the river network shapefile.
        """
        if not self.ddb_path.exists():
            self.logger.warning("Drainage database not found, skipping topology fix")
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    return

                n_size = ds.sizes[n_dim]

                # Check if topology is broken
                if 'Next' not in ds:
                    self.logger.warning("Next array missing from drainage database")
                    needs_fix = True
                else:
                    next_arr = ds['Next'].values
                    needs_fix = np.all(next_arr <= 0)
                    if needs_fix:
                        self.logger.warning(f"Topology broken: all {n_size} GRUs have Next=0")

                if not needs_fix:
                    self.logger.debug("Drainage topology appears valid")
                    return

            # Rebuild topology
            self._rebuild_topology(n_size)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            import traceback
            self.logger.error(f"Failed to fix drainage topology: {e}", exc_info=True)
            self.logger.debug(traceback.format_exc())

    def _rebuild_topology(self, n_size: int) -> None:
        """Rebuild topology from river network."""
        self.logger.info("Rebuilding drainage topology from river network")

        # Load river network
        river_shp = self.rivers_path / self.rivers_name
        if not river_shp.exists():
            self.logger.error(f"River network shapefile not found: {river_shp}")
            return

        riv_gdf = gpd.read_file(river_shp)

        if 'LINKNO' not in riv_gdf.columns or 'DSLINKNO' not in riv_gdf.columns:
            self.logger.error("River network missing LINKNO or DSLINKNO columns")
            return

        # Build mappings
        linkno_to_dslink = dict(zip(riv_gdf['LINKNO'], riv_gdf['DSLINKNO']))
        valid_linknos = set(riv_gdf['LINKNO'].values)

        # Load river basins
        basin_shp = self.catchment_path / self.catchment_name
        if not basin_shp.exists():
            self.logger.error(f"River basins shapefile not found: {basin_shp}")
            return

        basin_gdf = gpd.read_file(basin_shp)

        # Find GRU ID field
        gru_id_field = None
        for field in ['GRU_ID', 'GRUID', 'gru_id', 'LINKNO']:
            if field in basin_gdf.columns:
                gru_id_field = field
                break

        if gru_id_field is None:
            self.logger.error("Could not find GRU ID field in river basins")
            return

        gru_ids = basin_gdf[gru_id_field].values
        n_grus = len(gru_ids)

        if n_grus != n_size:
            self.logger.warning(f"GRU count mismatch: basins={n_grus}, DDB={n_size}")

        # Build GRU_ID -> index mapping
        gru_to_idx = {int(gid): i for i, gid in enumerate(gru_ids)}
        outlet_value = self._get_config_value(lambda: self.config.model.mesh.outlet_value, default=-9999, dict_key='MESH_OUTLET_VALUE')

        # Build downstream GRU mapping
        gru_to_ds_gru = {}
        for gid in gru_ids:
            gid = int(gid)
            ds_linkno = linkno_to_dslink.get(gid)
            if ds_linkno is None or ds_linkno == outlet_value or ds_linkno not in valid_linknos:
                gru_to_ds_gru[gid] = 0
            elif ds_linkno in gru_to_idx:
                gru_to_ds_gru[gid] = int(ds_linkno)
            else:
                gru_to_ds_gru[gid] = 0

        # Build DA-based connectivity if no downstream links
        if all(v == 0 for v in gru_to_ds_gru.values()):
            gru_to_ds_gru = self._build_da_connectivity(gru_ids, n_grus)

        # Compute topological levels
        levels = self._compute_levels(gru_ids, gru_to_ds_gru, n_grus)

        # Sort and assign ranks
        sorted_grus = sorted(gru_ids, key=lambda g: (-levels.get(int(g), 0), int(g)))
        gru_to_rank = {int(g): i + 1 for i, g in enumerate(sorted_grus)}

        # Build arrays
        self._update_ddb_topology(gru_ids, gru_to_ds_gru, gru_to_rank, levels, n_grus)

    def _build_da_connectivity(self, gru_ids: np.ndarray, n_grus: int) -> Dict[int, int]:
        """Build connectivity based on drainage area."""
        self.logger.warning("River network has no downstream links; building DA-based connectivity")

        gru_to_ds_gru = {int(gid): 0 for gid in gru_ids}

        with xr.open_dataset(self.ddb_path) as ds:
            if 'DA' in ds:
                da_vals = ds['DA'].values
            elif 'GridArea' in ds:
                da_vals = ds['GridArea'].values
            else:
                return gru_to_ds_gru

            n_dim = self._get_spatial_dim(ds)
            if n_dim in ds:
                ddb_ids = [int(v) for v in ds[n_dim].values]
            elif 'N' in ds:
                ddb_ids = [int(v) for v in ds['N'].values]
            else:
                ddb_ids = [int(v) for v in gru_ids]

            da_map = {ddb_ids[i]: float(da_vals[i]) for i in range(min(len(ddb_ids), len(da_vals)))}
            sorted_by_da = sorted(da_map.items(), key=lambda x: x[1], reverse=True)

            for i, (gid, _) in enumerate(sorted_by_da):
                if i == 0:
                    gru_to_ds_gru[gid] = 0
                else:
                    gru_to_ds_gru[gid] = int(sorted_by_da[i - 1][0])

        return gru_to_ds_gru

    def _compute_levels(
        self,
        gru_ids: np.ndarray,
        gru_to_ds_gru: Dict[int, int],
        n_grus: int
    ) -> Dict[int, int]:
        """Compute topological levels using iterative propagation."""
        levels = {int(gid): 0 for gid in gru_ids}
        changed = True
        max_iter = n_grus + 1
        iteration = 0

        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for gid in gru_ids:
                gid = int(gid)
                ds_gid = gru_to_ds_gru.get(gid, 0)
                if ds_gid == 0:
                    if levels[gid] != 1:
                        levels[gid] = 1
                        changed = True
                else:
                    ds_level = levels.get(ds_gid, 0)
                    if ds_level > 0:
                        new_level = ds_level + 1
                        if levels[gid] != new_level:
                            levels[gid] = new_level
                            changed = True

        return levels

    def _update_ddb_topology(
        self,
        gru_ids: np.ndarray,
        gru_to_ds_gru: Dict[int, int],
        gru_to_rank: Dict[int, int],
        levels: Dict[int, int],
        n_grus: int
    ) -> None:
        """Update drainage database with corrected topology."""
        with xr.open_dataset(self.ddb_path) as ds:
            n_dim = self._get_spatial_dim(ds)

            if n_dim in ds:
                ddb_ids = [int(v) for v in ds[n_dim].values]
            elif 'N' in ds:
                ddb_ids = [int(v) for v in ds['N'].values]
            else:
                ddb_ids = [int(v) for v in gru_ids]

        next_arr = np.zeros(n_grus, dtype=np.int32)
        rank_arr = np.zeros(n_grus, dtype=np.int32)

        for i, gid in enumerate(ddb_ids):
            rank_arr[i] = gru_to_rank.get(gid, i + 1)
            ds_gid = gru_to_ds_gru.get(gid, 0)
            if ds_gid == 0:
                next_arr[i] = 0
            else:
                next_arr[i] = gru_to_rank.get(ds_gid, 0)

        n_outlets = np.sum(next_arr == 0)
        max_level = max(levels.values()) if levels else 0

        self.logger.info(f"Rebuilt topology: {n_grus} GRUs, {n_outlets} outlet(s), max level={max_level}")

        # Reorder by rank and remap Next values
        with xr.open_dataset(self.ddb_path) as ds:
            n_dim = self._get_spatial_dim(ds)
            order_idx = np.argsort(rank_arr)
            ds_new = ds.isel({n_dim: order_idx}).copy(deep=True)

            # After sorting, ranks become sequential [1, 2, 3, ...]
            old_ranks_sorted = rank_arr[order_idx]
            new_ranks = np.arange(1, n_grus + 1, dtype=np.int32)

            # Build mapping from old rank -> new rank
            rank_remap = {int(old): int(new) for old, new in zip(old_ranks_sorted, new_ranks)}

            # Remap Next values to point to new ranks
            next_arr_sorted = next_arr[order_idx]
            next_arr_remapped = np.array([
                rank_remap.get(int(val), 0) if val > 0 else 0
                for val in next_arr_sorted
            ], dtype=np.int32)

            # CRITICAL: For single-cell (lumped) domains, MESH uses max(Next) to
            # determine the number of active cells for array sizing. With Next=0,
            # max(Next)=0, so arrays are sized to 0 and nothing can be read.
            # Fix: Set Next=1 (self-reference) for single-cell domains.
            if n_grus == 1 and next_arr_remapped[0] == 0:
                next_arr_remapped[0] = 1
                self.logger.debug("Single-cell domain: set Next=1 (self-reference) for MESH array sizing")

            ds_new['Next'] = xr.DataArray(
                next_arr_remapped,
                dims=[n_dim],
                attrs={'long_name': 'Downstream cell rank', 'units': '1'}
            )
            ds_new['Rank'] = xr.DataArray(
                new_ranks,
                dims=[n_dim],
                attrs={'long_name': 'Cell rank in topological order', 'units': '1'}
            )

            if 'GRU' in ds_new and 'NGRU' in ds_new.dims:
                # Ensure GRU fractions sum to 1.0 for each subbasin
                # gru is (N, NGRU)
                gru_da = ds_new['GRU']
                gru_sums = gru_da.sum('NGRU')
                # Avoid division by zero, default to 1.0 if all zero
                safe_sums = xr.where(gru_sums == 0, 1.0, gru_sums)
                ds_new['GRU'] = gru_da / safe_sums

                # If sum was 0, set the first GRU to 1.0 as a fallback
                if (gru_sums == 0).any():
                    self.logger.warning("Found subbasins with 0 GRU coverage during topology update. Setting first GRU to 1.0.")
                    vals = ds_new['GRU'].values
                    zero_indices = np.where(gru_sums.values == 0)[0]
                    for idx in zero_indices:
                        vals[idx, 0] = 1.0
                    ds_new['GRU'].values = vals

            temp_path = self.ddb_path.with_suffix('.tmp.nc')
            ds_new.to_netcdf(temp_path)
            os.replace(temp_path, self.ddb_path)

        self.logger.info("Updated drainage database with corrected topology")

    def ensure_completeness(self) -> None:
        """Ensure drainage database has all required variables for MESH."""
        if not self.ddb_path.exists():
            return
        self._ensure_completeness_impl()

    def _ensure_completeness_impl(self) -> None:
        """Internal implementation for drainage database completeness checks."""
        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    return

                n_size = ds.sizes[n_dim]
                modified = False

                ds, n_dim, modified = self._prepare_ddb_dimensions(ds, n_dim, n_size, modified)
                ds, modified = self._ensure_gru_coverage(ds, n_dim, n_size, modified)
                ds, modified = self._normalize_ddb_variables(ds, n_dim, n_size, modified)
                ds, modified = self._normalize_lat_lon_coordinates(ds, n_dim, n_size, modified)
                ds, modified = self._postprocess_gru_dimensions(ds, n_dim, n_size, modified)
                ds, modified = self._ensure_required_mesh_variables(ds, n_dim, n_size, modified)
                ds, modified = self._normalize_integer_fields(ds, modified)

                self._write_ddb_if_modified(ds, modified)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to ensure DDB completeness: {e}", exc_info=True)

    def _prepare_ddb_dimensions(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        modified: bool,
    ) -> Tuple[xr.Dataset, str, bool]:
        target_n_dim = 'subbasin'
        if n_dim != target_n_dim:
            self.logger.debug(f"Renaming spatial dimension '{n_dim}' to '{target_n_dim}'")
            ds = ds.rename({n_dim: target_n_dim})
            n_dim = target_n_dim
            modified = True

        for old_name in ['N']:
            if old_name in ds.coords or old_name in ds.data_vars:
                ds = ds.rename({old_name: target_n_dim})
                modified = True

        ds[target_n_dim] = xr.DataArray(
            np.arange(1, n_size + 1, dtype=np.int32),
            dims=[target_n_dim],
            attrs={'long_name': 'Grid index', 'units': '1'}
        )
        modified = True

        old_lc_dim = 'land' if 'land' in ds.dims else None
        if old_lc_dim:
            self.logger.debug(f"Renaming dimension '{old_lc_dim}' to 'NGRU'")
            ds = ds.rename({old_lc_dim: 'NGRU'})
            modified = True

        return ds, n_dim, modified

    def _collapse_to_single_gru(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        message: str,
    ) -> xr.Dataset:
        self.logger.info(message)
        gru_data = np.array([[0.998, 0.002]], dtype=np.float64)
        if n_size > 1:
            gru_data = np.tile(gru_data, (n_size, 1))

        vars_to_drop = [name for name in ds.data_vars if 'NGRU' in ds[name].dims]
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop, errors='ignore')
        if 'NGRU' in ds.coords:
            ds = ds.drop_vars('NGRU', errors='ignore')

        ds['GRU'] = xr.DataArray(
            gru_data,
            dims=[n_dim, 'NGRU'],
            attrs={'long_name': 'Group Response Unit', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
        )
        return ds

    def _ensure_gru_coverage(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        modified: bool,
    ) -> Tuple[xr.Dataset, bool]:
        if 'GRU' in ds:
            if ds['GRU'].dims != (n_dim, 'NGRU'):
                ds['GRU'] = ((n_dim, 'NGRU'), ds['GRU'].values, ds['GRU'].attrs)
                modified = True
            ds['GRU'].attrs['grid_mapping'] = 'crs'

            if should_force_single_gru(self.config_dict) and 'NGRU' in ds.dims and ds.sizes['NGRU'] > 2:
                ds = self._collapse_to_single_gru(
                    ds,
                    n_dim,
                    n_size,
                    f"Enforcing lumped mode: collapsing {ds.sizes['NGRU']} GRUs to 1 "
                    "(NGRU=2 for MESH off-by-one workaround)"
                )
                modified = True

        elif 'NGRU' in ds:
            self.logger.debug("Found 'NGRU' variable without GRU; renaming to GRU")
            ds['GRU'] = ((n_dim, 'NGRU'), ds['NGRU'].values, ds['NGRU'].attrs)
            ds = ds.drop_vars('NGRU')
            ds['GRU'].attrs['grid_mapping'] = 'crs'
            modified = True

            if should_force_single_gru(self.config_dict) and 'NGRU' in ds.dims and ds.sizes['NGRU'] > 2:
                ds = self._collapse_to_single_gru(
                    ds,
                    n_dim,
                    n_size,
                    f"Enforcing lumped mode: collapsing {ds.sizes['NGRU']} GRUs to 1 "
                    "(NGRU=2 for MESH off-by-one workaround)"
                )
                modified = True
        else:
            self.logger.debug("Creating GRU variable for lumped mode (1 landclass, 100% coverage)")
            ds['GRU'] = ((n_dim, 'NGRU'), np.ones((n_size, 1), dtype=np.float64), {
                'long_name': 'Land cover class fractions',
                'units': '1',
                'grid_mapping': 'crs'
            })
            modified = True

        return ds, modified

    def _normalize_ddb_variables(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        modified: bool,
    ) -> Tuple[xr.Dataset, bool]:
        for var_name in ['landclass_dim', 'time']:
            if var_name in ds:
                ds = ds.drop_vars(var_name)

        if 'time' in ds.dims:
            ds = ds.isel(time=0, drop=True)

        if 'crs' not in ds:
            ds['crs'] = xr.DataArray(
                np.array(0, dtype=np.int32),
                attrs={
                    'grid_mapping_name': 'latitude_longitude',
                    'semi_major_axis': 6378137.0,
                    'inverse_flattening': 298.257223563,
                    'longitude_of_prime_meridian': 0.0
                }
            )
            modified = True
        elif ds['crs'].dtype != np.int32:
            ds['crs'] = ds['crs'].astype(np.int32)
            modified = True

        for var_name in list(ds.data_vars):
            if var_name == 'crs':
                continue

            if var_name == 'GRU':
                ds[var_name].attrs['grid_mapping'] = 'crs'
                continue

            if n_dim not in ds[var_name].dims:
                self.logger.warning(f"Variable {var_name} missing dimension {n_dim}. Forcing it.")
                temp_data = ds[var_name].values.flatten()[0] if ds[var_name].values.size > 0 else 0
                ds[var_name] = xr.DataArray(
                    np.full(n_size, temp_data, dtype=ds[var_name].dtype),
                    dims=[n_dim],
                    attrs=ds[var_name].attrs
                )
                modified = True
            elif len(ds[var_name].dims) > 1:
                other_dims = [dim for dim in ds[var_name].dims if dim != n_dim]
                self.logger.debug(f"Squeezing {var_name} to 1D over {n_dim} (removing {other_dims})")
                ds[var_name] = ds[var_name].isel({dim: 0 for dim in other_dims}, drop=True)
                modified = True

            ds[var_name] = ds[var_name].transpose(n_dim)
            ds[var_name].attrs['grid_mapping'] = 'crs'
            if 'coordinates' in ds[var_name].attrs:
                coords_str = ds[var_name].attrs['coordinates']
                new_coords = " ".join([coord for coord in coords_str.split() if coord in ['lat', 'lon', 'N']])
                if new_coords:
                    ds[var_name].attrs['coordinates'] = new_coords
                else:
                    del ds[var_name].attrs['coordinates']

        return ds, modified

    def _normalize_lat_lon_coordinates(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        modified: bool,
    ) -> Tuple[xr.Dataset, bool]:
        for coord in ['lat', 'lon']:
            if coord not in ds:
                continue

            if coord in ds.coords:
                ds = ds.reset_coords(coord)

            vals = ds[coord].values.flatten()
            if len(vals) >= n_size:
                coord_vals = vals[:n_size].astype(np.float64)
            elif len(vals) == 1:
                coord_vals = np.full(n_size, vals[0], dtype=np.float64)
            else:
                coord_vals = np.zeros(n_size, dtype=np.float64)
                coord_vals[:len(vals)] = vals
                coord_vals[len(vals):] = vals[-1]
                self.logger.warning(
                    f"Lat/lon array has {len(vals)} values but need {n_size}, padding with last value"
                )

            ds[coord] = (n_dim, coord_vals, {
                'units': 'degrees_north' if coord == 'lat' else 'degrees_east',
                'long_name': 'latitude' if coord == 'lat' else 'longitude',
                'standard_name': 'latitude' if coord == 'lat' else 'longitude',
                'grid_mapping': 'crs'
            })
            modified = True

        return ds, modified

    def _postprocess_gru_dimensions(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        modified: bool,
    ) -> Tuple[xr.Dataset, bool]:
        if 'NGRU' in ds.dims and 'GRU' in ds and not is_elevation_band_mode(self.config_dict):
            gru_vals = ds['GRU'].values
            gru_sums = gru_vals.sum(axis=0) if gru_vals.ndim == 2 else gru_vals
            nonzero_mask = gru_sums > 0.001
            n_nonzero = nonzero_mask.sum()
            n_original = len(nonzero_mask)

            if n_nonzero < n_original:
                self.logger.debug(
                    f"Trimming {n_original - n_nonzero} zero-fraction GRU(s) (keeping {n_nonzero} of {n_original})"
                )
                ds = ds.isel(NGRU=nonzero_mask)
                if 'GRU' in ds:
                    gru_trimmed = ds['GRU'].values
                    if gru_trimmed.ndim == 2:
                        row_sums = gru_trimmed.sum(axis=1, keepdims=True)
                        row_sums = np.where(row_sums == 0, 1, row_sums)
                        ds['GRU'] = (ds['GRU'].dims, gru_trimmed / row_sums)
                    else:
                        total = gru_trimmed.sum()
                        if total > 0:
                            ds['GRU'] = (ds['GRU'].dims, gru_trimmed / total)
                modified = True

        if 'NGRU' in ds.dims and 'NGRU' in ds.coords:
            ds = ds.drop_vars('NGRU')
            self.logger.debug("Removed NGRU coordinate variable (MESH expects dimension only)")
            modified = True

        if should_force_single_gru(self.config_dict) and 'NGRU' in ds.dims and 'GRU' in ds and ds.sizes['NGRU'] > 2:
            ds = self._collapse_to_single_gru(
                ds,
                n_dim,
                n_size,
                f"Enforcing lumped mode: collapsing {ds.sizes['NGRU']} GRUs to 1 "
                "(set MESH_FORCE_SINGLE_GRU=false to use multiple GRUs)"
            )
            modified = True

        if 'GRU' in ds:
            ds['GRU'].attrs.update({
                'long_name': 'Group Response Unit',
                'standard_name': 'GRU',
                'grid_mapping': 'crs'
            })
            modified = True

        for attr in ['crs', 'grid_mapping', 'featureType']:
            if attr in ds.attrs:
                del ds.attrs[attr]

        return ds, modified

    def _correct_gridarea_scale(self, ds: xr.Dataset) -> None:
        """Rescale a uniformly mis-projected GridArea to the true catchment area.

        meshflow occasionally emits subbasin areas off by a constant factor (a
        projection/units mismatch). When the total departs from the authoritative
        catchment area — the sum of ``GRU_area`` in the riverBasins shapefile
        meshflow consumed — by more than 5%, and the per-cell error is uniform,
        every cell is rescaled by the single correcting factor so ``GridArea``
        sums to the true area while preserving relative subbasin proportions.
        Mutates ``ds['GridArea']`` in place; no-op when already correct or when
        the reference area cannot be read.
        """
        import glob

        try:
            grid_area = np.asarray(ds['GridArea'].values, dtype=float)
            current_total = float(np.nansum(grid_area))
            if current_total <= 0:
                return

            candidates = glob.glob(str(self.forcing_dir / '*riverBasins*.shp'))
            if not candidates:
                return
            import geopandas as gpd
            gdf = gpd.read_file(candidates[0])
            if 'GRU_area' in gdf.columns:
                true_total = float(np.nansum(gdf['GRU_area'].values))
            else:
                # Fall back to an equal-area geometric measure.
                true_total = float(gdf.to_crs('EPSG:6933').geometry.area.sum())
            if true_total <= 0:
                return

            ratio = true_total / current_total
            if abs(ratio - 1.0) <= 0.05:
                return  # already consistent

            ds['GridArea'] = xr.DataArray(
                grid_area * ratio,
                dims=ds['GridArea'].dims,
                attrs=dict(ds['GridArea'].attrs),
            )
            self.logger.info(
                f"Corrected GridArea scale by x{ratio:.3f}: "
                f"{current_total / 1e6:.1f} km2 -> {true_total / 1e6:.1f} km2 "
                f"(matched to riverBasins shapefile area)"
            )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not correct GridArea scale: {e}", exc_info=True)

    def _ensure_required_mesh_variables(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int,
        modified: bool,
    ) -> Tuple[xr.Dataset, bool]:
        if 'IREACH' not in ds:
            ds['IREACH'] = xr.DataArray(
                np.zeros(n_size, dtype=np.int32),
                dims=[n_dim],
                attrs={'long_name': 'Reservoir ID', 'units': '1'}
            )
            modified = True
            self.logger.debug("Added IREACH to drainage database")
        elif '_FillValue' in ds['IREACH'].attrs:
            ireach_vals = ds['IREACH'].values.copy()
            fill_val = ds['IREACH'].attrs['_FillValue']
            ireach_vals = np.where(ireach_vals == fill_val, 0, ireach_vals)
            ireach_vals = np.where(np.isnan(ireach_vals.astype(float)), 0, ireach_vals)
            ds['IREACH'] = xr.DataArray(
                ireach_vals.astype(np.int32),
                dims=[n_dim],
                attrs={'long_name': 'Reservoir ID', 'units': '1'}
            )
            modified = True
            self.logger.debug("Fixed IREACH encoding (removed problematic _FillValue)")

        if 'IAK' not in ds:
            ds['IAK'] = xr.DataArray(
                np.ones(n_size, dtype=np.int32),
                dims=[n_dim],
                attrs={'long_name': 'River class', 'units': '1'}
            )
            modified = True
            self.logger.debug("Added IAK to drainage database")
        elif '_FillValue' in ds['IAK'].attrs:
            iak_vals = ds['IAK'].values.copy()
            fill_val = ds['IAK'].attrs['_FillValue']
            iak_vals = np.where(iak_vals == fill_val, 1, iak_vals)
            ds['IAK'] = xr.DataArray(
                iak_vals.astype(np.int32),
                dims=[n_dim],
                attrs={'long_name': 'River class', 'units': '1'}
            )
            modified = True
            self.logger.debug("Fixed IAK encoding (removed problematic _FillValue)")

        # Correct GridArea if meshflow computed subbasin areas in the wrong
        # projection. meshflow's areas can come out uniformly scaled (observed
        # ~5.2x too small on a UTM/degree mismatch), which leaves the basin-average
        # water-balance DEPTHS correct (the scaling cancels in the area weights) but
        # makes every mm->m3/s conversion — routed streamflow AND the calibration
        # worker's discharge — wrong by that factor. Rescale to the authoritative
        # catchment area from the riverBasins shapefile meshflow itself consumed.
        if 'GridArea' in ds:
            self._correct_gridarea_scale(ds)

        if 'AL' not in ds and 'GridArea' in ds:
            grid_area = ds['GridArea'].values
            if 'Perimeter' in ds:
                char_length = ds['Perimeter'].values / (2 * np.pi)
                method = 'perimeter-based'
            elif 'ChnlLength' in ds:
                char_length = ds['ChnlLength'].values
                method = 'channel length'
            elif 'ChnlLen' in ds:
                char_length = ds['ChnlLen'].values
                method = 'channel length'
            else:
                char_length = np.sqrt(grid_area)
                method = 'sqrt(area) fallback'
                self.logger.warning(
                    "AL computed from sqrt(area) - may be inaccurate for irregular catchments. "
                    "Consider adding Perimeter or ChnlLength to input shapefiles."
                )

            char_length = np.maximum(char_length, 1.0)
            ds['AL'] = xr.DataArray(
                char_length,
                dims=[n_dim],
                attrs={'long_name': 'Characteristic length of grid', 'units': 'm', '_FillValue': np.nan}
            )
            modified = True
            self.logger.debug(f"Added AL ({method}): min={char_length.min():.1f}m, max={char_length.max():.1f}m")

        if 'DA' not in ds and 'GridArea' in ds and 'Next' in ds and 'Rank' in ds:
            ds['DA'] = self._compute_drainage_area(ds, n_dim, n_size)
            modified = True

        if 'Area' not in ds and 'GridArea' in ds:
            ds['Area'] = xr.DataArray(
                ds['GridArea'].values,
                dims=[n_dim],
                attrs={'long_name': 'Grid area', 'units': 'm2'}
            )
            modified = True
            self.logger.debug("Added Area (alias of GridArea) to drainage database")

        if 'Slope' not in ds:
            slope_vals = ds['ChnlSlope'].values if 'ChnlSlope' in ds else np.full(n_size, 0.001, dtype=np.float64)
            slope_vals = np.where(np.isnan(slope_vals), 0.001, slope_vals)
            slope_vals = np.where(slope_vals <= 0, 0.001, slope_vals)
            ds['Slope'] = xr.DataArray(
                slope_vals,
                dims=[n_dim],
                attrs={'long_name': 'Mean basin slope', 'units': '1'}
            )
            modified = True
            self.logger.debug("Added Slope to drainage database")

        if 'ChnlLen' not in ds and 'ChnlLength' in ds:
            ds['ChnlLen'] = xr.DataArray(
                ds['ChnlLength'].values,
                dims=[n_dim],
                attrs={'long_name': 'Channel length', 'units': 'm'}
            )
            modified = True
            self.logger.debug("Added ChnlLen (alias of ChnlLength) to drainage database")

        return ds, modified

    def _normalize_integer_fields(
        self,
        ds: xr.Dataset,
        modified: bool,
    ) -> Tuple[xr.Dataset, bool]:
        for name in ['Rank', 'Next', 'IAK', 'IREACH', 'N']:
            if name in ds and ds[name].dtype != np.int32:
                ds[name] = ds[name].astype(np.int32)
                modified = True
                self.logger.debug(f"Coerced {name} to int32 in drainage database")
        return ds, modified

    def _write_ddb_if_modified(self, ds: xr.Dataset, modified: bool) -> None:
        if not modified:
            return

        lc_var = 'NGRU' if 'NGRU' in ds else 'GRU' if 'GRU' in ds else None
        lc_dim = 'NGRU' if 'NGRU' in ds.dims else 'NGRU' if 'NGRU' in ds.dims else None
        if lc_var and lc_dim:
            lc_da = ds[lc_var]
            lc_sums = lc_da.sum(lc_dim)
            safe_sums = xr.where(lc_sums == 0, 1.0, lc_sums)
            ds[lc_var] = lc_da / safe_sums

            if (lc_sums == 0).any():
                self.logger.warning(
                    f"Found subbasins with 0 {lc_var} coverage during completeness check. Setting first entry to 1.0."
                )
                vals = ds[lc_var].values
                zero_indices = np.where(lc_sums.values == 0)[0]
                for idx in zero_indices:
                    vals[idx, 0] = 1.0
                ds[lc_var].values = vals

        temp_path = self.ddb_path.with_suffix('.tmp.nc')
        encoding = {}
        if 'NGRU' in ds.coords:
            encoding['NGRU'] = {'dtype': 'float64'}
        ds.to_netcdf(temp_path, encoding=encoding)
        os.replace(temp_path, self.ddb_path)

    def _compute_drainage_area(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int
    ) -> xr.DataArray:
        """Compute accumulated drainage area.

        Each cell's area is its own GridArea plus the accumulated area of every
        cell draining into it. Cells are visited upstream to downstream (a
        cell only after all of its contributors), so each cell's total is
        passed on exactly once.
        """
        grid_area = ds['GridArea'].values
        next_arr = ds['Next'].values.astype(int)
        rank_arr = ds['Rank'].values.astype(int)

        da = grid_area.astype(float)

        # Index of the cell each cell drains to (-1: outlet, self-reference or
        # a Next that matches no Rank).
        index_of_rank: dict = {}
        for i, rank in enumerate(rank_arr):
            index_of_rank.setdefault(int(rank), i)
        downstream = np.full(n_size, -1, dtype=int)
        for i in range(n_size):
            if next_arr[i] > 0:
                j = index_of_rank.get(int(next_arr[i]), -1)
                if j != i:
                    downstream[i] = j

        n_upstream = np.zeros(n_size, dtype=int)
        for j in downstream[downstream >= 0]:
            n_upstream[j] += 1
        ready = [i for i in range(n_size) if n_upstream[i] == 0]
        visited = 0
        while ready:
            i = ready.pop()
            visited += 1
            j = downstream[i]
            if j >= 0:
                da[j] += da[i]
                n_upstream[j] -= 1
                if n_upstream[j] == 0:
                    ready.append(j)
        if visited != n_size:
            raise MESHInputError(
                'Drainage database Rank/Next topology contains a cycle; '
                'cannot accumulate drainage area'
            )

        self.logger.debug(f"Added DA: max={da.max()/1e6:.1f} km²")

        return xr.DataArray(
            da,
            dims=[n_dim],
            attrs={
                'long_name': 'Drainage area',
                'units': 'm**2',
                'coordinates': 'lon lat',
                '_FillValue': np.nan
            }
        )

    def reorder_by_rank_and_normalize(self) -> None:
        """Reorder drainage database by Rank and normalize GRU fractions."""
        if not self.ddb_path.exists():
            self.logger.warning("MESH_drainage_database.nc not found")
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    self.logger.warning("No spatial dimension found")
                    return

                if 'Rank' not in ds or 'Next' not in ds:
                    self.logger.warning("Rank/Next not found, skipping reorder")
                    return

                rank_arr = ds['Rank'].values.astype(int)
                next_arr = ds['Next'].values.astype(int)
                order_idx = np.argsort(rank_arr)

                ds_new = ds.isel({n_dim: order_idx}).copy(deep=True)
                old_rank_sorted = rank_arr[order_idx]
                new_rank_arr = np.arange(1, len(old_rank_sorted) + 1, dtype=np.int32)
                rank_map = {int(old): int(new) for old, new in zip(old_rank_sorted, new_rank_arr)}

                next_sorted = next_arr[order_idx]
                next_remap = np.array(
                    [rank_map.get(int(val), 0) if int(val) > 0 else 0 for val in next_sorted],
                    dtype=np.int32
                )

                # CRITICAL: For single-cell (lumped) domains, set Next=1 (self-reference)
                # so MESH can properly size its internal arrays
                n_grus = len(new_rank_arr)
                if n_grus == 1 and next_remap[0] == 0:
                    next_remap[0] = 1
                    self.logger.debug("Single-cell domain: preserving Next=1 for MESH")

                ds_new['Rank'] = xr.DataArray(
                    new_rank_arr,
                    dims=[n_dim],
                    attrs={'long_name': 'Cell rank in topological order', 'units': '1'}
                )
                ds_new['Next'] = xr.DataArray(
                    next_remap,
                    dims=[n_dim],
                    attrs={'long_name': 'Downstream cell rank', 'units': '1'}
                )

                if 'GRU' in ds_new and 'NGRU' in ds_new.dims:
                    # Ensure GRU fractions sum to 1.0 for each subbasin
                    gru_da = ds_new['GRU']
                    gru_sums = gru_da.sum('NGRU')
                    # Avoid division by zero, default to 1.0 if all zero
                    safe_sums = xr.where(gru_sums == 0, 1.0, gru_sums)
                    ds_new['GRU'] = gru_da / safe_sums

                    # If sum was 0, set the first GRU to 1.0 as a fallback
                    if (gru_sums == 0).any():
                        self.logger.warning("Found subbasins with 0 GRU coverage during reorder. Setting first GRU to 1.0.")
                        vals = ds_new['GRU'].values
                        zero_indices = np.where(gru_sums.values == 0)[0]
                        for idx in zero_indices:
                            vals[idx, 0] = 1.0
                        ds_new['GRU'].values = vals

                # MESH does NOT expect a coordinate variable for the NGRU dimension.
                # Having NGRU(NGRU) confuses MESH - it tries to map it as a data variable.
                # Remove any NGRU coordinate variable if present.
                encoding: dict[str, dict[str, bool]] = {}
                if 'NGRU' in ds_new.dims and 'NGRU' in ds_new.coords:
                    ds_new = ds_new.drop_vars('NGRU')

                # Collapse to single GRU when lumped mode is active.
                # Use GRU shape as fallback when dimension metadata is unreliable.
                # Threshold > 2: NGRU=2 is already single-GRU format.
                ngru_count = 1
                if 'GRU' in ds_new:
                    gru_shape = ds_new['GRU'].shape
                    ngru_count = gru_shape[-1] if len(gru_shape) >= 2 else 1

                if should_force_single_gru(self.config_dict) and ngru_count > 2:
                    n_size = ds_new.sizes[n_dim] if n_dim else 1
                    self.logger.info(
                        f"Enforcing lumped mode: collapsing {ngru_count} GRUs to 1 "
                        f"(set MESH_FORCE_SINGLE_GRU=false to use multiple GRUs)"
                    )
                    # MESH has an off-by-one issue: it reads NGRU-1 GRUs.
                    # For MESH to see 1 GRU, we need NGRU=2: [0.998, 0.002]
                    # The second column is padding with a small value.
                    gru_data = np.array([[0.998, 0.002]], dtype=np.float64)
                    if n_size > 1:
                        gru_data = np.tile(gru_data, (n_size, 1))

                    # Drop all vars with NGRU dimension and the NGRU coord if present
                    vars_to_drop = [v for v in ds_new.data_vars if 'NGRU' in ds_new[v].dims]
                    if vars_to_drop:
                        ds_new = ds_new.drop_vars(vars_to_drop, errors='ignore')
                    if 'NGRU' in ds_new.coords:
                        ds_new = ds_new.drop_vars('NGRU', errors='ignore')

                    # Create new GRU with NGRU=2 for MESH off-by-one workaround
                    ds_new['GRU'] = xr.DataArray(
                        gru_data,
                        dims=[n_dim, 'NGRU'],
                        attrs={'long_name': 'Group Response Unit', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
                    )
                    self.logger.debug(f"Created GRU with shape {gru_data.shape} (NGRU=2 for MESH to see 1)")

                # Ensure GRU has proper attributes that MESH expects
                if 'GRU' in ds_new:
                    ds_new['GRU'].attrs.update({
                        'long_name': 'Group Response Unit',
                        'standard_name': 'GRU',
                        'grid_mapping': 'crs'
                    })

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path, encoding=encoding)
                os.replace(temp_path, self.ddb_path)

            self.logger.info("Reordered by Rank and normalized GRU fractions")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to reorder: {e}", exc_info=True)

    def _get_spatial_dim(self, ds: xr.Dataset) -> Optional[str]:
        """Get the spatial dimension name."""
        if 'N' in ds.dims:
            return 'N'
        elif 'subbasin' in ds.dims:
            return 'subbasin'
        return None

    def _as_bool(self, value: Any, default: bool = False) -> bool:
        """Parse a truthy/falsey config value with a safe default."""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ('true', '1', 'yes', 'y', 'on'):
                return True
            if val in ('false', '0', 'no', 'n', 'off'):
                return False
        return default

    def _should_force_single_gru(self) -> bool:
        """Determine whether to collapse GRUs to a single class.

        .. deprecated::
            Use :func:`should_force_single_gru` from ``config_defaults``
            instead.  This method is kept for backward compatibility only.
        """
        return should_force_single_gru(self.config_dict)

    def convert_to_elevation_band_grus(self, hru_shapefile: Path) -> tuple:
        """Convert DDB to use elevation bands as GRUs instead of landcover classes.

        This method reads the elevation band HRU shapefile created during discretization
        and rebuilds the DDB GRU variable to have one GRU per elevation band.

        Args:
            hru_shapefile: Path to the elevation band HRU shapefile

        Returns:
            Tuple of (n_elev_bands, elevation_info) where elevation_info is a list of
            dicts with 'elevation', 'fraction' for each band. Returns (0, []) on failure.
        """
        if not hru_shapefile.exists():
            self.logger.warning(f"Elevation band shapefile not found: {hru_shapefile}")
            return 0, []

        if not self.ddb_path.exists():
            self.logger.warning("MESH_drainage_database.nc not found")
            return 0, []

        try:
            # Read elevation band HRU shapefile
            gdf = gpd.read_file(hru_shapefile)
            self.logger.debug(f"Read elevation band shapefile with {len(gdf)} HRUs")

            # Get area column name
            area_col = 'HRU_area' if 'HRU_area' in gdf.columns else 'area'
            if area_col not in gdf.columns:
                # Calculate areas
                gdf['HRU_area'] = gdf.geometry.area
                area_col = 'HRU_area'

            # Calculate total area and fraction for each elevation band
            total_area = gdf[area_col].sum()
            n_elev_bands = len(gdf)

            elev_fractions = (gdf[area_col] / total_area).values
            self.logger.debug(
                f"Elevation band fractions: {[f'{f:.3f}' for f in elev_fractions]}"
            )

            # Get mean elevation for each band
            elev_col = 'avg_elevcl' if 'avg_elevcl' in gdf.columns else 'elev_mean'
            if elev_col in gdf.columns:
                mean_elevs = gdf[elev_col].values
            else:
                # Default elevations if not available
                mean_elevs = [1500 + i * 400 for i in range(n_elev_bands)]

            self.logger.debug(
                f"Elevation band means: {[f'{e:.0f}m' for e in mean_elevs]}"
            )

            # Build elevation info for CLASS block creation
            elevation_info = [
                {'elevation': float(mean_elevs[i]), 'fraction': float(elev_fractions[i])}
                for i in range(n_elev_bands)
            ]

            # Update DDB with elevation band GRUs
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    self.logger.warning("Could not determine spatial dimension")
                    return 0, []

                n_size = int(ds.sizes[n_dim])

                # Create new GRU array with elevation bands
                # MESH reads NGRU-1, so we need n_elev_bands + 1 columns
                n_gru_cols = n_elev_bands + 1
                new_gru = np.zeros((n_size, n_gru_cols), dtype=np.float64)

                # Set elevation band fractions (first n_elev_bands columns)
                for i in range(n_elev_bands):
                    new_gru[:, i] = elev_fractions[i]

                # Create new dataset
                ds_new = ds.drop_vars(['GRU'] if 'GRU' in ds else [], errors='ignore')
                if 'NGRU' in ds_new.dims:
                    ds_new = ds_new.drop_dims('NGRU')

                # Add new GRU variable
                ds_new['GRU'] = xr.DataArray(
                    new_gru,
                    dims=[n_dim, 'NGRU'],
                    attrs={'long_name': 'Elevation band fractions'}
                )

                # Save updated DDB
                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

            self.logger.info(
                f"Converted DDB to {n_elev_bands} elevation band GRUs "
                f"(NGRU={n_gru_cols}, MESH reads {n_elev_bands})"
            )
            return n_elev_bands, elevation_info

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to convert to elevation band GRUs: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0, []

    def convert_to_multi_subbasin_elevation_bands(self, hru_shapefile: Path) -> tuple:
        """Convert DDB from 1 subbasin × N GRUs to N subbasins × (N+1) GRUs.

        Each elevation band becomes its own subbasin with its own forcing,
        enabling temperature lapsing. Uses an identity GRU matrix so each
        subbasin has exactly one active GRU.

        Args:
            hru_shapefile: Path to the elevation band HRU shapefile

        Returns:
            Tuple of (n_elev_bands, elevation_info) where elevation_info is a list of
            dicts with 'elevation', 'fraction' for each band. Returns (0, []) on failure.
        """
        if not hru_shapefile.exists():
            self.logger.warning(f"Elevation band shapefile not found: {hru_shapefile}")
            return 0, []

        if not self.ddb_path.exists():
            self.logger.warning("MESH_drainage_database.nc not found")
            return 0, []

        try:
            # Read elevation band HRU shapefile
            gdf = gpd.read_file(hru_shapefile)
            self.logger.debug(f"Read elevation band shapefile with {len(gdf)} HRUs")

            # Get area column name
            area_col = 'HRU_area' if 'HRU_area' in gdf.columns else 'area'
            if area_col not in gdf.columns:
                gdf['HRU_area'] = gdf.geometry.area
                area_col = 'HRU_area'

            total_area_from_shp = gdf[area_col].sum()
            n_bands = len(gdf)
            band_fractions = (gdf[area_col] / total_area_from_shp).values

            # Get mean elevation for each band
            elev_col = 'avg_elevcl' if 'avg_elevcl' in gdf.columns else 'elev_mean'
            if elev_col in gdf.columns:
                mean_elevs = gdf[elev_col].values
            else:
                mean_elevs = np.array([1500 + i * 400 for i in range(n_bands)])

            self.logger.debug(
                f"Elevation bands: {n_bands} bands, "
                f"elevations={[f'{e:.0f}m' for e in mean_elevs]}, "
                f"fractions={[f'{f:.3f}' for f in band_fractions]}"
            )

            elevation_info = [
                {'elevation': float(mean_elevs[i]), 'fraction': float(band_fractions[i])}
                for i in range(n_bands)
            ]

            # Read existing DDB (single subbasin)
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    self.logger.warning("Could not determine spatial dimension")
                    return 0, []

                # Get original values from the single subbasin
                orig_total_area = float(ds['GridArea'].values[0]) if 'GridArea' in ds else total_area_from_shp
                orig_lat = float(ds['lat'].values[0]) if 'lat' in ds else 0.0
                orig_lon = float(ds['lon'].values[0]) if 'lon' in ds else 0.0

                # Collect other 1D variables to replicate
                chnl_slope = float(ds['ChnlSlope'].values[0]) if 'ChnlSlope' in ds else 0.001
                chnl_length = float(ds['ChnlLength'].values[0]) if 'ChnlLength' in ds else 1000.0
                # Check for optional variables
                has_slope = 'Slope' in ds
                slope_val = float(ds['Slope'].values[0]) if has_slope else chnl_slope
                has_area = 'Area' in ds

            # Build new multi-subbasin DDB
            target_dim = 'subbasin'

            # GRU matrix: identity pattern with padding column for MESH off-by-one
            # MESH reads NGRU-1 GRUs, so we need n_bands + 1 columns
            n_gru_cols = n_bands + 1
            gru_data = np.zeros((n_bands, n_gru_cols), dtype=np.float64)
            for i in range(n_bands):
                gru_data[i, i] = 1.0  # Each subbasin has exactly one active GRU

            # Spatial arrays. Use the catchment area from the HRU shapefile, not
            # the single-subbasin DDB's GridArea: meshflow can compute the latter
            # in the wrong projection (observed ~5x too small), which would make
            # every band area — and thus all downstream mm->m3/s discharge — wrong.
            # total_area_from_shp * band_fractions == the shapefile HRU areas.
            if total_area_from_shp > 0 and abs(orig_total_area - total_area_from_shp) / total_area_from_shp > 0.05:
                self.logger.info(
                    f"Band GridArea from HRU shapefile area {total_area_from_shp/1e6:.1f} km² "
                    f"(single-subbasin DDB GridArea was {orig_total_area/1e6:.1f} km² — "
                    f"meshflow projection mismatch, corrected)"
                )
            area_basis = total_area_from_shp if total_area_from_shp > 0 else orig_total_area
            grid_area = area_basis * band_fractions
            lat_arr = np.full(n_bands, orig_lat, dtype=np.float64)
            lon_arr = np.full(n_bands, orig_lon, dtype=np.float64)

            # Rank and Next: all outlets in noroute mode
            rank_arr = np.arange(1, n_bands + 1, dtype=np.int32)
            next_arr = np.zeros(n_bands, dtype=np.int32)

            # Build dataset
            ds_new = xr.Dataset()

            # CRS
            ds_new['crs'] = xr.DataArray(
                np.array(0, dtype=np.int32),
                attrs={
                    'grid_mapping_name': 'latitude_longitude',
                    'semi_major_axis': 6378137.0,
                    'inverse_flattening': 298.257223563,
                    'longitude_of_prime_meridian': 0.0
                }
            )

            # Coordinate
            ds_new[target_dim] = xr.DataArray(
                np.arange(1, n_bands + 1, dtype=np.int32),
                dims=[target_dim],
                attrs={'long_name': 'Grid index', 'units': '1'}
            )

            # GRU
            ds_new['GRU'] = xr.DataArray(
                gru_data,
                dims=[target_dim, 'NGRU'],
                attrs={'long_name': 'Elevation band fractions', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
            )

            # 1D variables replicated/scaled across subbasins
            ds_new['GridArea'] = xr.DataArray(
                grid_area, dims=[target_dim],
                attrs={'long_name': 'Grid area', 'units': 'm2', 'grid_mapping': 'crs'}
            )
            ds_new['lat'] = xr.DataArray(
                lat_arr, dims=[target_dim],
                attrs={'long_name': 'latitude', 'standard_name': 'latitude',
                       'units': 'degrees_north', 'grid_mapping': 'crs'}
            )
            ds_new['lon'] = xr.DataArray(
                lon_arr, dims=[target_dim],
                attrs={'long_name': 'longitude', 'standard_name': 'longitude',
                       'units': 'degrees_east', 'grid_mapping': 'crs'}
            )
            ds_new['Rank'] = xr.DataArray(
                rank_arr, dims=[target_dim],
                attrs={'long_name': 'Cell rank in topological order', 'units': '1'}
            )
            ds_new['Next'] = xr.DataArray(
                next_arr, dims=[target_dim],
                attrs={'long_name': 'Downstream cell rank', 'units': '1'}
            )
            ds_new['ChnlSlope'] = xr.DataArray(
                np.full(n_bands, chnl_slope, dtype=np.float64), dims=[target_dim],
                attrs={'long_name': 'Channel slope', 'units': '1', 'grid_mapping': 'crs'}
            )
            ds_new['ChnlLength'] = xr.DataArray(
                np.full(n_bands, chnl_length, dtype=np.float64), dims=[target_dim],
                attrs={'long_name': 'Channel length', 'units': 'm', 'grid_mapping': 'crs'}
            )
            ds_new['AL'] = xr.DataArray(
                np.sqrt(grid_area), dims=[target_dim],
                attrs={'long_name': 'Characteristic length of grid', 'units': 'm', 'grid_mapping': 'crs'}
            )
            ds_new['DA'] = xr.DataArray(
                grid_area.copy(), dims=[target_dim],
                attrs={'long_name': 'Drainage area', 'units': 'm**2', 'grid_mapping': 'crs'}
            )
            ds_new['IAK'] = xr.DataArray(
                np.ones(n_bands, dtype=np.int32), dims=[target_dim],
                attrs={'long_name': 'River class', 'units': '1'}
            )
            ds_new['IREACH'] = xr.DataArray(
                np.zeros(n_bands, dtype=np.int32), dims=[target_dim],
                attrs={'long_name': 'Reservoir ID', 'units': '1'}
            )

            if has_slope:
                ds_new['Slope'] = xr.DataArray(
                    np.full(n_bands, slope_val, dtype=np.float64), dims=[target_dim],
                    attrs={'long_name': 'Mean basin slope', 'units': '1', 'grid_mapping': 'crs'}
                )
            if has_area:
                ds_new['Area'] = xr.DataArray(
                    grid_area.copy(), dims=[target_dim],
                    attrs={'long_name': 'Grid area', 'units': 'm2', 'grid_mapping': 'crs'}
                )
            ds_new['ChnlLen'] = xr.DataArray(
                np.full(n_bands, chnl_length, dtype=np.float64), dims=[target_dim],
                attrs={'long_name': 'Channel length', 'units': 'm', 'grid_mapping': 'crs'}
            )

            # Save
            temp_path = self.ddb_path.with_suffix('.tmp.nc')
            ds_new.to_netcdf(temp_path)
            os.replace(temp_path, self.ddb_path)

            self.logger.info(
                f"Converted DDB to multi-subbasin elevation bands: "
                f"subbasin={n_bands}, NGRU={n_gru_cols}, "
                f"total GridArea={grid_area.sum():.0f} m² "
                f"(original={orig_total_area:.0f} m²)"
            )
            return n_bands, elevation_info

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to convert to multi-subbasin elevation bands: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0, []
