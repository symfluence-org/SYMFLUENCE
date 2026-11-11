# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Point Scale Forcing Extractor

Forcing extraction for domains whose forcing grid is too small for EASYMORE.

Two modes:

- **point**: a point-scale domain, or a forcing grid with a single cell. Every
  HRU takes the cell nearest to the target point.
- **area-weighted**: a small multi-cell grid (2 x N, N x 1, ...) under a real
  catchment. Cell polygons are built from the grid coordinates and every HRU
  gets the area-weighted mean of the cells it overlaps — the same result the
  regular remapper gives on larger grids.
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

    Bypasses EASYMORE remapping for grids with fewer than three rows or
    columns. Single-cell grids and point domains use the nearest cell; small
    multi-cell grids are area-weighted onto the HRUs.
    """

    POINT_METHOD = 'nearest_coordinate_v1'
    AREA_METHOD = 'area_weighted_v1'


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

        True does not mean "use one cell for everything": :meth:`process`
        area-weights small multi-cell grids and only falls back to the
        nearest cell for single-cell grids and point domains.

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

        force = self._get_config_value(lambda: self.config.system.force_run_all_steps, default=False, dict_key='FORCE_RUN_ALL_STEPS')

        if forcing_files and self._use_area_weighting(forcing_files[0]):
            self.logger.info(
                "Small multi-cell forcing grid under a non-point domain: area-weighting "
                "the forcing cells onto the HRUs."
            )
            weights = self._create_area_weighted_intersection(
                intersect_csv, catchment_file_path, forcing_files[0], dem_path
            )
            for file in forcing_files:
                output_file = output_filename_func(file)
                if output_file.exists() and not force:
                    with xr.open_dataset(output_file) as existing:
                        if (existing.attrs.get('point_extraction_method') == self.AREA_METHOD
                                and existing.sizes.get('hru') == weights['matrix'].shape[0]
                                and output_file.stat().st_mtime >= file.stat().st_mtime
                                and output_file.stat().st_mtime >= intersect_csv.stat().st_mtime):
                            continue
                self._process_single_file_area_weighted(file, output_file, weights)
            return

        # Recompute the elevation metadata for the same cell selected below.
        # Older intersection files used the grid centre, even though extraction
        # selected the first cell. Do not carry those corrections forward.
        self._create_intersection_csv(intersect_csv, catchment_file_path, forcing_files, dem_path)

        # Process each file
        for file in forcing_files:
            output_file = output_filename_func(file)
            if output_file.exists() and not force:
                with xr.open_dataset(output_file) as existing:
                    target = self._target_coordinates()
                    if (existing.attrs.get('point_extraction_method') == self.POINT_METHOD
                            and existing.attrs.get('point_target_latitude') == target[0]
                            and existing.attrs.get('point_target_longitude') == target[1]
                            and output_file.stat().st_mtime >= file.stat().st_mtime):
                        continue

            self._process_single_file(file, output_file, intersect_csv)

    # ------------------------------------------------------------------
    # Area-weighted extraction for small multi-cell grids
    # ------------------------------------------------------------------

    def _is_point_domain(self) -> bool:
        method = self._get_config_value(
            lambda: self.config.domain.definition_method, default='', dict_key='DOMAIN_DEFINITION_METHOD'
        )
        return str(method or '').lower() == 'point'

    def _use_area_weighting(self, sample_file: Path) -> bool:
        """True for a multi-cell grid under a non-point domain."""
        if self._is_point_domain():
            return False
        lat_name, lon_name = self.dataset_handler.get_coordinate_names()
        with xr.open_dataset(sample_file, engine="h5netcdf") as ds:
            if lat_name not in ds or lon_name not in ds:
                return False
            n_cells = ds[lat_name].size * (ds[lon_name].size if ds[lat_name].ndim == 1 else 1)
            return n_cells > 1

    @staticmethod
    def _edges(centres: np.ndarray, fallback_step: Optional[float]) -> np.ndarray:
        """Cell edges midway between 1-D centres (outer edges extrapolated)."""
        centres = np.asarray(centres, dtype=float)
        if centres.size == 1:
            if not fallback_step:
                raise ValueError('Cannot infer the forcing cell size from a single coordinate')
            half = abs(fallback_step) / 2.0
            return np.array([centres[0] - half, centres[0] + half])
        mid = (centres[:-1] + centres[1:]) / 2.0
        return np.concatenate([[centres[0] - (mid[0] - centres[0])], mid,
                               [centres[-1] + (centres[-1] - mid[-1])]])

    def _forcing_cell_polygons(self, ds: xr.Dataset):
        """Forcing cell footprints in EPSG:4326.

        Returns ``(polygons, spatial_dims, shape)`` with the polygons in
        row-major order over *spatial_dims*.
        """
        from shapely.geometry import box

        from .grid_geometry import projected_cell_polygons

        lat_name, lon_name = self.dataset_handler.get_coordinate_names()
        lat = ds[lat_name]
        lon = ds[lon_name]

        if lat.ndim == 1 and lon.ndim == 1:
            lat_v = np.asarray(lat.values, dtype=float)
            lon_v = np.asarray(lon.values, dtype=float)
            lat_step = float(np.mean(np.abs(np.diff(lat_v)))) if lat_v.size > 1 else None
            lon_step = float(np.mean(np.abs(np.diff(lon_v)))) if lon_v.size > 1 else None
            lat_edges = self._edges(lat_v, lon_step)
            lon_edges = self._edges(lon_v, lat_step)
            polygons = [
                box(min(lon_edges[j], lon_edges[j + 1]), min(lat_edges[i], lat_edges[i + 1]),
                    max(lon_edges[j], lon_edges[j + 1]), max(lat_edges[i], lat_edges[i + 1]))
                for i in range(lat_v.size) for j in range(lon_v.size)
            ]
            return polygons, (lat.dims[0], lon.dims[0]), (lat_v.size, lon_v.size)

        if lat.ndim == 2 and lat.dims == lon.dims:
            polygons = list(projected_cell_polygons(lon.values, lat.values))
            return polygons, tuple(lat.dims), lat.shape

        raise ValueError(
            f"Cannot build forcing cell polygons from coordinates '{lat_name}' {lat.dims} / "
            f"'{lon_name}' {lon.dims}"
        )

    def _cell_mean_dem(self, cells: gpd.GeoDataFrame, dem_path: Optional[Path]) -> np.ndarray:
        """Mean DEM elevation of each forcing cell (NaN where the DEM has no data)."""
        out = np.full(len(cells), np.nan)
        if dem_path is None or not Path(dem_path).exists():
            return out
        from symfluence.geospatial.zonal import zonal_stats_covered

        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            nodata = src.nodata
            dem_crs = src.crs
            pixel_area = abs(transform.a * transform.e)
        cells_dem = cells.to_crs(dem_crs) if dem_crs is not None and cells.crs != dem_crs else cells
        stats, _ = zonal_stats_covered(
            cells_dem.geometry, dem, affine=transform, stats=['mean'],
            nodata=nodata if nodata is not None else -9999,
        )
        for k, (stat, geom) in enumerate(zip(stats, cells_dem.geometry)):
            if stat.get('mean') is None:
                continue
            out[k] = float(stat['mean'])
            covered = stat['count'] * pixel_area / geom.area if geom.area > 0 else 1.0
            if covered < 0.5:
                self.logger.warning(
                    f"The DEM covers only {100 * min(covered, 1.0):.0f}% of forcing cell {k}; its "
                    f"elevation ({out[k]:.0f} m) is the mean of the covered part, not of the cell."
                )
        return out

    def _create_area_weighted_intersection(
        self,
        intersect_csv: Path,
        catchment_file_path: Path,
        sample_file: Path,
        dem_path: Optional[Path],
    ) -> dict:
        """Area weights of forcing cells per HRU; writes the intersection CSV.

        The CSV has one row per (HRU, overlapping cell) with the weight and the
        cell-mean DEM elevation — the layout the regular remapper writes.
        """
        hru_id_field = self._get_config_value(
            lambda: self.config.paths.catchment_hruid, default='HRU_ID', dict_key='CATCHMENT_SHP_HRUID'
        ) or 'HRU_ID'
        target = gpd.read_file(catchment_file_path)
        if target.crs is None:
            target = target.set_crs('EPSG:4326')
        target_wgs = target.to_crs('EPSG:4326')

        with xr.open_dataset(sample_file, engine="h5netcdf") as ds:
            polygons, spatial_dims, shape = self._forcing_cell_polygons(ds)
        cells = gpd.GeoDataFrame({'cell': range(len(polygons))}, geometry=polygons, crs='EPSG:4326')

        # Forcing on 0..360 longitudes under a catchment on -180..180
        if cells.total_bounds[2] > 180.0 and target_wgs.total_bounds[0] < 0.0:
            cells['geometry'] = cells.geometry.translate(xoff=-360.0)

        equal_area = 'EPSG:6933'
        cells_ea = cells.to_crs(equal_area)
        target_ea = target_wgs.to_crs(equal_area)

        n_hru = len(target)
        matrix = np.zeros((n_hru, len(cells)), dtype=float)
        for h, geom in enumerate(target_ea.geometry):
            for c, cell in enumerate(cells_ea.geometry):
                if geom.intersects(cell):
                    matrix[h, c] = geom.intersection(cell).area
            total = matrix[h].sum()
            if total <= 0:
                nearest = int(np.argmin([geom.distance(cell) for cell in cells_ea.geometry]))
                matrix[h, nearest] = 1.0
                self.logger.warning(
                    f"HRU {target[hru_id_field].iloc[h]} does not overlap any forcing cell; "
                    f"using the nearest cell ({nearest})."
                )
            else:
                covered = total / geom.area if geom.area > 0 else 1.0
                if covered < 0.999:
                    self.logger.warning(
                        f"Forcing grid covers only {100 * covered:.1f}% of HRU "
                        f"{target[hru_id_field].iloc[h]}; weights are normalised over the covered part."
                    )
                matrix[h] /= total

        cell_elev = self._cell_mean_dem(cells, dem_path)
        catchment_elevs = (
            np.asarray(target['elev_mean'].values, dtype=float) if 'elev_mean' in target.columns else None
        )
        if np.isnan(cell_elev).any():
            self.logger.info(
                "DEM does not cover every forcing cell; those cells use the HRU elevation "
                "(no lapse-rate correction)."
            )

        hru_ids = target[hru_id_field].values
        gru_ids = target['GRU_ID'].values if 'GRU_ID' in target.columns else np.ones(n_hru, dtype=int)
        rows = []
        for h in range(n_hru):
            s1 = float(catchment_elevs[h]) if catchment_elevs is not None else 0.0
            for c in np.nonzero(matrix[h])[0]:
                rows.append({
                    hru_id_field: hru_ids[h],
                    'S_1_HRU_ID': hru_ids[h],
                    'S_1_GRU_ID': gru_ids[h],
                    'ID': int(c) + 1,
                    'S_2_ID': int(c) + 1,
                    'weight': matrix[h, c],
                    'S_1_elev_m': s1,
                    'S_2_elev_m': float(cell_elev[c]) if np.isfinite(cell_elev[c]) else s1,
                })
        pd.DataFrame(rows).to_csv(intersect_csv, index=False)
        self.logger.info(
            f"Area-weighted {len(cells)} forcing cells ({shape[0]}x{shape[1]}) onto {n_hru} HRUs: "
            f"{intersect_csv.name}"
        )
        return {'matrix': matrix, 'dims': spatial_dims, 'shape': shape,
                'hru_ids': self._hru_ids_int32(hru_ids)}

    @staticmethod
    def _hru_ids_int32(hru_ids) -> np.ndarray:
        """HRU ids as int32, refusing values that would overflow."""
        ids = np.asarray(hru_ids).astype('int64')
        if ids.size and (ids.max() > np.iinfo('int32').max or ids.min() < np.iinfo('int32').min):
            raise ValueError(
                f"HRU ids up to {ids.max()} do not fit the 32-bit hruId of the forcing files. "
                "Use compact HRU/GRU ids (re-run the domain definition, which now assigns them)."
            )
        return ids.astype('int32')

    def _process_single_file_area_weighted(self, file: Path, output_file: Path, weights: dict) -> None:
        """Write (time, hru) forcing as the area-weighted mean of the forcing cells."""
        self.logger.info(f"Area-weighting forcing: {file.name}")
        matrix = weights['matrix']
        dims = tuple(weights['dims'])
        n_hru = matrix.shape[0]
        lat_name, lon_name = self.dataset_handler.get_coordinate_names()

        with xr.open_dataset(file, engine="h5netcdf") as ds:
            ds = self._standardize_if_raw(ds, file)
            if tuple(ds.sizes[d] for d in dims) != tuple(weights['shape']):
                raise ValueError(
                    f"{file.name} has forcing grid {tuple(ds.sizes[d] for d in dims)}, but the "
                    f"area weights were built for {tuple(weights['shape'])}."
                )

            data_vars = {}
            for name, var in ds.data_vars.items():
                if name in (lat_name, lon_name):
                    continue
                if not all(d in var.dims for d in dims):
                    if not any(d in var.dims for d in dims):
                        data_vars[name] = var
                    continue
                other = [d for d in var.dims if d not in dims]
                values = var.transpose(*other, *dims).values.astype(float)
                flat = values.reshape(values.shape[:len(other)] + (-1,))
                finite = np.isfinite(flat)
                weighted = np.where(finite, flat, 0.0) @ matrix.T
                weight_sum = finite.astype(float) @ matrix.T
                with np.errstate(invalid='ignore', divide='ignore'):
                    result = np.where(weight_sum > 0, weighted / weight_sum, np.nan)
                data_vars[name] = xr.DataArray(
                    result.astype('float32') if var.dtype.kind == 'f' and var.dtype.itemsize <= 4 else result,
                    dims=(*other, 'hru'),
                    coords={d: ds[d] for d in other if d in ds.coords},
                    attrs=dict(var.attrs),
                )

            out = xr.Dataset(data_vars, attrs=dict(ds.attrs))
            out['hruId'] = (('hru',), weights['hru_ids'])
            out.attrs.update(point_extraction_method=self.AREA_METHOD, forcing_cells=int(matrix.shape[1]))
            for name in list(out.data_vars):
                if 'time' in out[name].dims and 'hru' in out[name].dims:
                    out[name] = out[name].transpose('time', 'hru', ...)
            for name in out.variables:
                out[name].encoding = {}
                out[name].attrs.pop('missing_value', None)
                out[name].attrs.pop('_FillValue', None)
            assert out.sizes['hru'] == n_hru
            out.to_netcdf(output_file, engine="h5netcdf")
            self.logger.info(f"Created area-weighted forcing: {output_file.name}")

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

                # Prefer the mean DEM elevation of the whole forcing cell over a
                # single pixel at its centre (possible when the grid has more
                # than one cell, so the cell size is known).
                cell_mean = self._selected_cell_mean_dem(forcing_files[0], dem_path)
                if cell_mean is not None:
                    return cell_mean

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

    def _selected_cell_mean_dem(self, forcing_file: Path, dem_path: Path) -> Optional[float]:
        """Mean DEM elevation of the selected forcing cell, or None if unknown."""
        try:
            with xr.open_dataset(forcing_file, engine="h5netcdf") as ds:
                polygons, dims, shape = self._forcing_cell_polygons(ds)
                indexers = self._point_indexers(ds)
            flat = int(np.ravel_multi_index(tuple(int(indexers[d]) for d in dims), shape))
        except (ValueError, KeyError, IndexError):
            return None  # single-cell or irregular grid: cell footprint unknown
        cell = gpd.GeoDataFrame(geometry=[polygons[flat]], crs='EPSG:4326')
        value = self._cell_mean_dem(cell, dem_path)[0]
        return float(value) if np.isfinite(value) else None

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
            ds_point.attrs.update(point_extraction_method=self.POINT_METHOD,
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
                    hru_ids = df_int[self._get_config_value(lambda: self.config.paths.catchment_hruid, dict_key='CATCHMENT_SHP_HRUID')].values
                except (pd.errors.ParserError, KeyError, ValueError) as e:
                    self.logger.debug(f"Could not read HRU IDs from intersection CSV: {e}")
            hru_ids = self._hru_ids_int32(hru_ids)

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
