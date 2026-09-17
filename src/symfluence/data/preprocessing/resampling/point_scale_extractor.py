# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Point Scale Forcing Extractor

Simplified forcing extraction for point-scale or small grid domains.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from symfluence.core.mixins import ConfigMixin


class PointScaleForcingExtractor(ConfigMixin):
    """
    Extracts forcing data for point-scale or tiny grid domains.

    Bypasses EASYMORE remapping for 1x1 or small grids where intersection
    calculations would fail or be meaningless.
    """

    def __init__(
        self,
        config: dict,
        project_dir: Path,
        dataset_handler,
        logger: logging.Logger = None
    ):
        """
        Initialize point scale extractor.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            dataset_handler: Dataset-specific handler for coordinate names
            logger: Optional logger instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.project_dir = project_dir
        self.dataset_handler = dataset_handler
        self.logger = logger or logging.getLogger(__name__)

    def should_use_point_scale(self, merged_forcing_path: Path) -> bool:
        """
        Check if the forcing grid is too small for EASYMORE remapping.

        Args:
            merged_forcing_path: Path to merged forcing files

        Returns:
            True if point-scale extraction should be used
        """
        try:
            # Find a sample forcing file
            exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params']
            all_nc_files = list(merged_forcing_path.glob('*.nc'))
            forcing_files = [
                f for f in all_nc_files
                if not any(pattern in f.name.lower() for pattern in exclude_patterns)
            ]

            if not forcing_files:
                return False

            sample_file = forcing_files[0]
            var_lat, var_lon = self.dataset_handler.get_coordinate_names()

            with xr.open_dataset(sample_file, engine="h5netcdf") as ds:
                lat_vals = ds[var_lat].values
                lon_vals = ds[var_lon].values

                # Determine grid size
                if lat_vals.ndim == 1:
                    lat_size = len(lat_vals)
                    lon_size = len(lon_vals)
                elif lat_vals.ndim == 2:
                    lat_size, lon_size = lat_vals.shape
                else:
                    lat_size = lon_size = 1

                # EASYMORE requires at least 3 values in each dimension
                is_tiny = (lat_size <= 2 or lon_size <= 2)

                if is_tiny:
                    self.logger.info(
                        f"Detected small forcing grid: {lat_size}x{lon_size} "
                        f"(EASYMORE requires >= 3 in each dimension)"
                    )
                    return True

                return False

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Could not check forcing grid size: {e}", exc_info=True)
            return False

    def process(
        self,
        forcing_files: List[Path],
        output_dir: Path,
        catchment_file_path: Path,
        output_filename_func,
        dem_path: Optional[Path] = None
    ) -> None:
        """
        Process forcing files using simplified point-scale extraction.

        Args:
            forcing_files: List of forcing files to process
            output_dir: Output directory for processed files
            catchment_file_path: Full path to catchment shapefile (resolved by caller)
            output_filename_func: Function to determine output filename
            dem_path: Optional path to DEM raster for forcing grid elevation lookup
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)

        # Create minimal intersection CSV
        case_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_{self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')}"
        intersect_csv = intersect_path / f"{case_name}_intersected_shapefile.csv"

        # Recompute the elevation metadata for the same cell selected below.
        # Older intersection files used the grid centre, even though extraction
        # selected the first cell. Do not carry those corrections forward.
        self._create_intersection_csv(intersect_csv, catchment_file_path, forcing_files, dem_path)

        # Process each file
        for file in forcing_files:
            output_file = output_filename_func(file)
            if output_file.exists() and not self._get_config_value(lambda: self.config.system.force_run_all_steps, default=False, dict_key='FORCE_RUN_ALL_STEPS'):
                with xr.open_dataset(output_file) as existing:
                    target = self._target_coordinates()
                    if (existing.attrs.get('point_extraction_method') == 'nearest_coordinate_v1'
                            and existing.attrs.get('point_target_latitude') == target[0]
                            and existing.attrs.get('point_target_longitude') == target[1]
                            and output_file.stat().st_mtime >= file.stat().st_mtime):
                        continue

            self._process_single_file(file, output_file, intersect_csv)

    def _create_intersection_csv(
        self,
        intersect_csv: Path,
        catchment_file_path: Path,
        forcing_files: Optional[List[Path]] = None,
        dem_path: Optional[Path] = None
    ) -> None:
        """Create minimal intersection artifact for SUMMA preprocessor.

        Computes the forcing grid elevation (S_2_elev_m) from the DEM by
        sampling at the forcing grid cell centre.  Falls back to the catchment
        mean elevation when the DEM is unavailable, so that no lapse-rate
        correction is applied rather than applying a spurious one.
        """
        self.logger.info(f"Creating minimal intersection artifact: {intersect_csv.name}")

        target_gdf = gpd.read_file(catchment_file_path)
        hru_id_field = self._get_config_value(lambda: self.config.paths.catchment_hruid, dict_key='CATCHMENT_SHP_HRUID')

        hru_id_field_val = target_gdf[hru_id_field].values
        catchment_elevs = target_gdf['elev_mean'].values if 'elev_mean' in target_gdf.columns else None

        # Determine forcing grid elevation from DEM
        forcing_elev = self._get_forcing_elevation_from_dem(forcing_files, dem_path)

        if forcing_elev is None and catchment_elevs is not None:
            # No DEM available — use catchment elevation so lapse correction is zero
            forcing_elev_arr = catchment_elevs
            self.logger.info(
                "DEM not available for forcing elevation; setting S_2_elev_m = S_1_elev_m "
                "(no lapse-rate correction will be applied)"
            )
        elif forcing_elev is not None:
            forcing_elev_arr = [forcing_elev] * len(target_gdf)
            self.logger.info(f"Forcing grid elevation from DEM: {forcing_elev:.1f} m")
        else:
            # Neither DEM nor catchment elevation available — use 0 as neutral default
            forcing_elev_arr = [0.0] * len(target_gdf)
            self.logger.warning(
                "Neither DEM nor catchment elevation available; setting S_2_elev_m = 0"
            )

        df_int = pd.DataFrame({
            hru_id_field: hru_id_field_val,
            'S_1_HRU_ID': hru_id_field_val,
            'S_1_GRU_ID': target_gdf['GRU_ID'].values if 'GRU_ID' in target_gdf.columns else [1],
            'ID': [1] * len(target_gdf),
            'weight': [1.0] * len(target_gdf),
            'S_1_elev_m': catchment_elevs if catchment_elevs is not None else [0.0] * len(target_gdf),
            'S_2_elev_m': forcing_elev_arr
        })
        df_int.to_csv(intersect_csv, index=False)

    def _get_forcing_elevation_from_dem(
        self,
        forcing_files: Optional[List[Path]],
        dem_path: Optional[Path]
    ) -> Optional[float]:
        """Sample the DEM at the forcing grid cell centre to get forcing elevation.

        Returns the DEM sample at the selected forcing-cell centre, or None
        when that centre is outside the DEM or inputs are unavailable. This
        terrain sample is not the atmospheric model's orography.
        """
        if dem_path is None or not Path(dem_path).exists():
            return None
        if not forcing_files:
            return None

        try:
            # Get forcing grid cell coordinates from first file
            var_lat, var_lon = self.dataset_handler.get_coordinate_names()
            with xr.open_dataset(forcing_files[0], engine="h5netcdf") as ds:
                selected = ds.isel(self._point_indexers(ds))
                lat_center = float(selected[var_lat])
                lon_center = float(selected[var_lon])

            # Sample DEM at the forcing grid centre
            with rasterio.open(dem_path) as src:
                # A coarse forcing grid (ERA5, CONUS404, RDRS) centred outside a
                # small domain DEM must NOT be sampled: rasterio returns the fill
                # value for out-of-bounds points, and when the DEM has no nodata
                # set that fill is 0.0, which would masquerade as a sea-level
                # forcing elevation and drive a spurious ~lapse_rate*S_1 cold
                # correction. Reject out-of-bounds centres so the caller falls
                # back to the catchment elevation (zero lapse correction).
                bounds = src.bounds
                if not (
                    bounds.left <= lon_center <= bounds.right
                    and bounds.bottom <= lat_center <= bounds.top
                ):
                    self.logger.warning(
                        f"Forcing grid centre ({lat_center:.4f}, {lon_center:.4f}) "
                        f"is outside the domain DEM extent; falling back to the "
                        f"catchment elevation (no lapse-rate correction)."
                    )
                    return None

                # rasterio.sample expects (x, y) = (lon, lat) for geographic CRS
                samples = list(src.sample([(lon_center, lat_center)]))
                if samples:
                    value = float(samples[0][0])
                    if np.isfinite(value) and value != src.nodata and value != -9999:
                        return value

                self.logger.warning(
                    f"DEM returned nodata at forcing grid centre "
                    f"({lat_center:.4f}, {lon_center:.4f})"
                )
                return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Could not extract forcing elevation from DEM: {e}", exc_info=True)
            return None

    def _standardize_if_raw(self, ds: xr.Dataset, file: Path) -> xr.Dataset:
        """Apply dataset-handler standardisation when *ds* still carries raw names.

        Already-standardised forcing (CF names such as ``air_temperature``)
        exposes no source names from the handler's rename map, so this is a
        no-op for the normal merged monthly files. When raw dataset variables
        are present (e.g. ``CaSR_v3.2_P_TT_1.5m``) the per-handler
        ``process_dataset`` step was bypassed upstream; run it here so the
        extracted point forcing gets CF names and the unit conversions
        (deg_C->K, mb->Pa, m/hr->kg m-2 s-1, kts->m/s) rather than passing raw
        names straight through to the model preprocessor.

        Returns the standardised dataset, or *ds* unchanged when no raw names
        are found or no dataset handler is available.
        """
        handler = getattr(self, 'dataset_handler', None)
        if handler is None or not hasattr(handler, 'get_variable_mapping'):
            return ds

        try:
            full_map = handler.get_variable_mapping() or {}
        except Exception as e:  # noqa: BLE001 — defensive guard, don't kill extraction
            self.logger.debug(f"get_variable_mapping failed for {file.name}: {e}")
            return ds

        raw_present = [k for k in full_map if k in ds.variables]
        if not raw_present:
            return ds

        self.logger.warning(
            f"{file.name} carries raw dataset variables ({sorted(raw_present)}); "
            "the per-handler standardisation step was skipped upstream. "
            "Applying dataset-handler standardisation before point extraction "
            "so the remapped forcing carries CF names + correct units."
        )
        try:
            return handler.process_dataset(ds)
        except Exception as e:  # noqa: BLE001 — defensive guard, don't kill extraction
            self.logger.error(
                f"Failed to standardise raw forcing variables in {file.name}: {e}"
            )
            return ds

    def _process_single_file(
        self,
        file: Path,
        output_file: Path,
        intersect_csv: Path
    ) -> None:
        """Process a single forcing file."""
        self.logger.info(f"Extracting point forcing: {file.name}")

        with xr.open_dataset(file, engine="h5netcdf") as ds:
            # Standardise any file that still carries raw dataset variable
            # names before extraction. The point-scale path trusts merged_path
            # to hold already-standardised files, but a raw acquisition
            # artifact (e.g. the consolidated ``domain_*_RDRS_*.nc`` that is
            # dataset-tagged and so globbed alongside the standardised monthly
            # files) can reach here unprocessed. This mirrors the gridded
            # weight-applier guard so the extracted point forcing always
            # carries CF names + correct units.
            ds = self._standardize_if_raw(ds, file)

            # Select the closest geographic cell, independent of grid ordering.
            spatial_dims = {d: 0 for d in ds.dims if d not in ['time', 'hru']}
            spatial_dims.update(self._point_indexers(ds))

            # Check for empty spatial dimensions
            for dim_name, idx in spatial_dims.items():
                if dim_name in ds.dims and ds.sizes[dim_name] == 0:
                    raise ValueError(
                        f"Cannot extract point forcing from {file.name}: "
                        f"dimension '{dim_name}' has size 0."
                    )

            ds_point = ds.isel(spatial_dims)
            lat_name, lon_name = self.dataset_handler.get_coordinate_names()
            target_lat, target_lon = self._target_coordinates()
            ds_point.attrs.update(point_extraction_method='nearest_coordinate_v1',
                                  point_target_latitude=target_lat,
                                  point_target_longitude=target_lon)
            if lat_name in ds and lon_name in ds:
                ds_point.attrs.update(point_forcing_latitude=float(ds_point[lat_name]),
                                      point_forcing_longitude=float(ds_point[lon_name]))

            # Determine HRU IDs
            hru_ids = [1]
            if intersect_csv.exists():
                try:
                    df_int = pd.read_csv(intersect_csv)
                    hru_ids = df_int[self._get_config_value(lambda: self.config.paths.catchment_hruid, dict_key='CATCHMENT_SHP_HRUID')].values.astype('int32')
                except (pd.errors.ParserError, KeyError, ValueError) as e:
                    self.logger.debug(f"Could not read HRU IDs from intersection CSV: {e}")

            n_hrus = len(hru_ids)

            # Add HRU dimension if missing
            if 'hru' not in ds_point.dims:
                ds_point = ds_point.expand_dims(hru=range(n_hrus))

            if 'hruId' not in ds_point.data_vars:
                ds_point['hruId'] = (('hru',), hru_ids)

            # Ensure correct dimension order (time, hru)
            for var in ds_point.data_vars:
                if 'time' in ds_point[var].dims and 'hru' in ds_point[var].dims:
                    ds_point[var] = ds_point[var].transpose('time', 'hru')

            # Drop irrelevant coordinates
            coords_to_drop = ['latitude', 'longitude', 'lat', 'lon', 'expver']
            ds_point = ds_point.drop_vars(
                [c for c in coords_to_drop if c in ds_point.coords or c in ds_point.data_vars]
            )

            # Clear encoding
            for var in ds_point.variables:
                ds_point[var].encoding = {}
                if 'missing_value' in ds_point[var].attrs:
                    del ds_point[var].attrs['missing_value']
                if '_FillValue' in ds_point[var].attrs:
                    del ds_point[var].attrs['_FillValue']

            ds_point.to_netcdf(output_file, engine="h5netcdf")
            self.logger.info(f"Created point forcing: {output_file.name}")

    def _target_coordinates(self):
        """Get the requested point, or the centre of a small-domain bbox."""
        point = self._get_config_value(lambda: self.config.domain.pour_point_coords, dict_key='POUR_POINT_COORDS')
        if point:
            return tuple(float(v) for v in point.split('/'))
        bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords, dict_key='BOUNDING_BOX_COORDS')
        if bbox:
            north, west, south, east = (float(v) for v in bbox.split('/'))
            return (north + south) / 2, (west + east) / 2
        raise ValueError('Point forcing extraction requires POUR_POINT_COORDS or BOUNDING_BOX_COORDS')

    def _point_indexers(self, ds):
        """Nearest cell on regular or curvilinear geographic grids."""
        lat_name, lon_name = self.dataset_handler.get_coordinate_names()
        if lat_name not in ds or lon_name not in ds:
            if all(size == 1 for dim, size in ds.sizes.items() if dim not in {'time', 'hru'}):
                return {}
            raise ValueError('Cannot locate point forcing without geographic coordinates')
        lat, lon = xr.broadcast(ds[lat_name], ds[lon_name])
        target_lat, target_lon = self._target_coordinates()
        # Great-circle distance, including longitude wrap and descending axes.
        dlat = np.deg2rad(lat.values - target_lat)
        dlon = np.deg2rad((lon.values - target_lon + 180) % 360 - 180)
        distance = np.sin(dlat / 2)**2 + np.cos(np.deg2rad(target_lat)) * np.cos(np.deg2rad(lat.values)) * np.sin(dlon / 2)**2
        index = np.unravel_index(np.nanargmin(distance), distance.shape)
        return dict(zip(lat.dims, index))
