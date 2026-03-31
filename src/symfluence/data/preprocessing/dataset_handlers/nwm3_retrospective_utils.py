# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NWM3 Retrospective forcing dataset handler.

Processes NOAA NWM v3.0 Retrospective forcing data downloaded from AWS S3
into model-ready format. The data is on a Lambert Conformal Conic projected
grid with 2D latitude/longitude auxiliary coordinates.

Variable mapping (NWM3 → CFIF):
    T2D      → air_temperature
    Q2D      → specific_humidity
    PSFC     → surface_air_pressure
    SWDOWN   → surface_downwelling_shortwave_flux
    LWDOWN   → surface_downwelling_longwave_flux
    RAINRATE → precipitation_flux (hourly accumulation kg/m² → rate mm/s)
    U2D      → eastward_wind
    V2D      → northward_wind
"""

from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

from symfluence.core.constants import UnitConversion

from ...utils import VariableStandardizer
from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('nwm3_retrospective')
class NWM3RetrospectiveHandler(BaseDatasetHandler):
    """
    Handler for NWM v3.0 Retrospective forcing dataset.

    The NWM3 forcing data is on a ~1 km Lambert Conformal Conic grid with
    2D latitude/longitude auxiliary coordinates. Variables are stored with
    NWM-native names (T2D, Q2D, etc.) and must be mapped to CFIF standard
    names for downstream processing.
    """

    def __init__(self, config, logger, project_dir, **kwargs):
        super().__init__(config, logger, project_dir, **kwargs)
        self.forcing_timestep_seconds: float = float(
            self.forcing_timestep_seconds
            if hasattr(self, 'forcing_timestep_seconds')
            else float(UnitConversion.SECONDS_PER_HOUR)
        )

    # ---------- Variable mapping / standardization ----------

    def get_variable_mapping(self) -> Dict[str, str]:
        """NWM3 → CFIF variable mapping via centralized VariableStandardizer."""
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('NWM3_RETROSPECTIVE')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process NWM3 forcing with variable standardization and unit conversions.

        Steps:
            1. Rename NWM3 variables to CFIF names
            2. Derive wind_speed from U2D/V2D components
            3. Convert precipitation from hourly accumulation (kg/m²) to rate (mm/s)
            4. Fill NaN values using nearest-neighbor interpolation
            5. Apply standard CF-compliant attributes
        """
        var_map = self.get_variable_mapping()

        # Keep original precip attrs before rename
        precip_attrs = {}
        if 'RAINRATE' in ds.variables:
            precip_attrs = dict(ds['RAINRATE'].attrs)

        existing = {old: new for old, new in var_map.items() if old in ds.variables}
        ds = ds.rename(existing)

        # ---- Derive wind speed magnitude from components ----
        if 'eastward_wind' in ds and 'northward_wind' in ds:
            u = ds['eastward_wind']
            v = ds['northward_wind']
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = 'wind_speed'
            windspd.attrs = {
                'units': 'm s-1',
                'long_name': 'wind speed',
                'standard_name': 'wind_speed',
            }
            ds['wind_speed'] = windspd

        # ---- Precip: convert accumulation -> rate (mm/s) ----
        if 'precipitation_flux' in ds:
            p = ds['precipitation_flux']

            units = (precip_attrs.get('units') or p.attrs.get('units', '')).lower()
            units_no_space = units.replace(' ', '')
            dt_seconds = float(self.forcing_timestep_seconds)

            # Clean inputs
            p = xr.where(np.isfinite(p), p, 0.0)
            p = xr.where(p < 0.0, 0.0, p)

            # The acquisition handler converts mm/s rate to kg/m² accumulation.
            # Convert back to rate for the model pipeline.
            accum_keywords = [
                'kgm-2', 'kgm^-2', 'kg/m^2', 'kg/m2', 'kgm**-2', 'mm'
            ]
            if any(k in units_no_space for k in accum_keywords):
                p_rate = p / dt_seconds
            elif any(k in units_no_space for k in ['kgm-2s-1', 'kgm^-2s^-1', 'kg/m^2s', 'mm/s']):
                p_rate = p  # already a rate
            else:
                self.logger.warning(
                    f"Unrecognized precip units for NWM3: '{units}', assuming accumulation."
                )
                p_rate = p / dt_seconds

            p_rate = xr.where(np.isfinite(p_rate), p_rate, 0.0)
            p_rate = xr.where(p_rate < 0.0, 0.0, p_rate)

            p_rate.attrs = {
                'units': 'mm/s',
                'long_name': 'Mean total precipitation rate',
                'standard_name': 'precipitation_flux',
            }
            ds['precipitation_flux'] = p_rate

        # Fill NaN values (NWM land-only grid typically has none, but be safe)
        ds = self._fill_nan_forcing_variables(ds)

        # Apply standard CF-compliant attributes
        ds = self.apply_standard_attributes(ds)

        return ds

    def _fill_nan_forcing_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Replace NaN values using nearest-neighbor interpolation."""
        spatial_dims: List[str] = [
            d for d in ('y', 'x', 'latitude', 'longitude') if d in ds.dims
        ]

        for var_name in list(ds.data_vars):
            if ds[var_name].dtype.kind != 'f':
                continue

            nan_count = int(ds[var_name].isnull().sum())
            if nan_count == 0:
                continue

            self.logger.info(
                f"Filling {nan_count} NaN values in '{var_name}' "
                f"using nearest-neighbor interpolation"
            )

            filled = ds[var_name]
            for dim in spatial_dims:
                if dim in filled.dims:
                    filled = filled.interpolate_na(
                        dim=dim, method='nearest', fill_value='extrapolate'
                    )

            remaining_nan = int(filled.isnull().sum())
            if remaining_nan > 0:
                spatial_mean = float(filled.mean(skipna=True))
                filled = filled.fillna(spatial_mean)
                self.logger.warning(
                    f"Filled {remaining_nan} remaining NaN in '{var_name}' "
                    f"with spatial mean ({spatial_mean:.4g})"
                )

            ds[var_name] = filled

        return ds

    # ---------- Coordinate info ----------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """NWM3 uses 2D latitude/longitude auxiliary coordinates on an LCC grid."""
        return ('latitude', 'longitude')

    # ---------- Merging (standardization) ----------

    def needs_merging(self) -> bool:
        """NWM3 needs standardization from NWM variable names to CFIF."""
        return True

    def merge_forcings(
        self,
        raw_forcing_path: Path,
        merged_forcing_path: Path,
        start_year: int,
        end_year: int,
    ) -> None:
        """Standardize NWM3 forcing files: rename variables, convert units."""
        self.logger.info("Standardizing NWM3 retrospective forcing files")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns = [
            f"{self.domain_name}_NWM3_forcing_*.nc",
            f"domain_{self.domain_name}_NWM3_*.nc",
            "*NWM3*forcing*.nc",
        ]

        files: list[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} NWM3 file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            self.logger.error(
                f"No NWM3 files found in {raw_forcing_path} with any of patterns: {patterns}"
            )
            raise FileNotFoundError(
                f"No NWM3 forcing files found in {raw_forcing_path}"
            )

        # Filter to files overlapping configured period
        all_files = files
        files = [
            f for f in all_files
            if self._file_overlaps_period(f, start_year, end_year)
        ]
        skipped = len(all_files) - len(files)
        if skipped:
            self.logger.info(
                f"Skipped {skipped} NWM3 file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        if not files:
            raise FileNotFoundError(
                f"No NWM3 forcing files match the configured period "
                f"{start_year}-{end_year} in {raw_forcing_path}"
            )

        for f in files:
            self.logger.info(f"Processing NWM3 file: {f}")
            try:
                ds = xr.open_dataset(f, engine="h5netcdf")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Error opening NWM3 file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                ds_proc = self.setup_time_encoding(ds_proc)
                ds_proc = self.add_metadata(
                    ds_proc,
                    "NWM3 retrospective forcing standardized for SYMFLUENCE",
                )
                ds_proc = self.clean_variable_attributes(ds_proc)

                for var_name in ds_proc.data_vars:
                    ds_proc[var_name].attrs.pop("missing_value", None)

                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name, engine="h5netcdf")
                self.logger.info(f"Saved processed NWM3 forcing: {out_name}")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Error processing NWM3 dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("NWM3 forcing standardization completed")

    # ---------- Shapefile for NWM3 grid ----------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """Create NWM3 grid shapefile from the 2D LCC projected grid."""
        self.logger.info("Creating NWM3 grid shapefile")

        output_shapefile = shapefile_path / "forcing_NWM3_RETROSPECTIVE.shp"

        nwm3_files = list(merged_forcing_path.glob('*.nc'))
        if not nwm3_files:
            raise FileNotFoundError(
                f"No NWM3 processed files found in {merged_forcing_path}"
            )

        nwm3_file = nwm3_files[0]
        self.logger.info(f"Using NWM3 file for grid: {nwm3_file}")

        # Check if point domain or small bbox
        is_point_domain = (
            self._get_config_value(
                lambda: self.config.domain.definition_method, default=''
            ) or ''
        ).lower() == 'point'

        bbox_coords = self._get_config_value(
            lambda: self.config.domain.bounding_box_coords, default=''
        ) or ''

        is_small_bbox = False
        domain_center_lat = None
        domain_center_lon = None

        if bbox_coords:
            try:
                coords = [float(x) for x in str(bbox_coords).split('/')]
                if len(coords) == 4:
                    lat_max, lon_max, lat_min, lon_min = coords
                    bbox_area = abs(lat_max - lat_min) * abs(lon_max - lon_min)
                    is_small_bbox = bbox_area < 0.01
                    domain_center_lat = (lat_max + lat_min) / 2
                    domain_center_lon = (lon_max + lon_min) / 2
                    if is_small_bbox:
                        self.logger.info(
                            f"Detected small bounding box (area={bbox_area:.6f} sq deg), "
                            f"will use nearest cell extraction"
                        )
            except (ValueError, AttributeError):
                pass

        if domain_center_lat is None:
            pour_point = self._get_config_value(
                lambda: self.config.domain.pour_point_coords, default=''
            ) or ''
            if pour_point:
                try:
                    coords = [float(x) for x in str(pour_point).split('/')]
                    if len(coords) == 2:
                        domain_center_lat, domain_center_lon = coords
                        is_point_domain = True
                except (ValueError, AttributeError):
                    pass

        with xr.open_dataset(nwm3_file, engine="h5netcdf") as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in NWM3 file. "
                    f"Available: {list(ds.coords)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in NWM3 file. "
                    f"Available: {list(ds.coords)}"
                )

        geometries = []
        ids = []
        lats = []
        lons = []

        # NWM3 always has 2D lat/lon (projected grid)
        if (is_point_domain or is_small_bbox) and domain_center_lat is not None:
            self.logger.info(
                f"Finding nearest NWM3 grid cell to domain center "
                f"({domain_center_lat:.4f}, {domain_center_lon:.4f})"
            )
            distances = np.sqrt(
                (lat - domain_center_lat)**2 + (lon - domain_center_lon)**2
            )
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            i, j = min_idx
            ny, nx = lat.shape

            lat_corners = [
                lat[i, j],
                lat[i, j+1] if j+1 < nx else lat[i, j],
                lat[i+1, j+1] if i+1 < ny and j+1 < nx else lat[i, j],
                lat[i+1, j] if i+1 < ny else lat[i, j],
            ]
            lon_corners = [
                lon[i, j],
                lon[i, j+1] if j+1 < nx else lon[i, j],
                lon[i+1, j+1] if i+1 < ny and j+1 < nx else lon[i, j],
                lon[i+1, j] if i+1 < ny else lon[i, j],
            ]

            geometries.append(Polygon(zip(lon_corners, lat_corners)))
            ids.append(0)
            lats.append(float(lat[i, j]))
            lons.append(float(lon[i, j]))
            self.logger.info(
                f"Selected nearest cell: lat={lat[i,j]:.4f}, lon={lon[i,j]:.4f}"
            )
        else:
            # Full 2D grid
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(
                f"NWM3 grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}"
            )

            cell_count = 0
            for i in range(ny):
                for j in range(nx):
                    lat_corners = [
                        lat[i, j],
                        lat[i, j+1] if j+1 < nx else lat[i, j],
                        lat[i+1, j+1] if i+1 < ny and j+1 < nx else lat[i, j],
                        lat[i+1, j] if i+1 < ny else lat[i, j],
                    ]
                    lon_corners = [
                        lon[i, j],
                        lon[i, j+1] if j+1 < nx else lon[i, j],
                        lon[i+1, j+1] if i+1 < ny and j+1 < nx else lon[i, j],
                        lon[i+1, j] if i+1 < ny else lon[i, j],
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * nx + j)
                    lats.append(float(lat[i, j]))
                    lons.append(float(lon[i, j]))

                    cell_count += 1
                    if cell_count % 5000 == 0 or cell_count == total_cells:
                        self.logger.info(
                            f"Created {cell_count}/{total_cells} NWM3 grid cells"
                        )

        lat_name = self._get_config_value(
            lambda: self.config.forcing.shape_lat_name, default='lat'
        )
        lon_name = self._get_config_value(
            lambda: self.config.forcing.shape_lon_name, default='lon'
        )

        gdf = gpd.GeoDataFrame(
            {
                'geometry': geometries,
                'ID': ids,
                lat_name: lats,
                lon_name: lons,
            },
            crs='EPSG:4326',
        )

        self.logger.info("Calculating elevation values for NWM3 grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf['elev_m'] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"NWM3 shapefile created at {output_shapefile}")

        return output_shapefile
