# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
AORC (Analysis of Record for Calibration) dataset handler.

Processes NOAA AORC atmospheric forcing data with spatial subsetting,
variable standardization, and unit conversion support.
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


@DatasetRegistry.register('aorc')
class AORCHandler(BaseDatasetHandler):
    """
    Handler for AORC (Analysis of Record for Calibration) dataset.

    This assumes AORC has been downloaded via CloudForcingDownloader into
    forcing/raw_data/domain_<DOMAIN_NAME>_AORC_<startYear>-<endYear>.nc
    and stored on a regular lat/lon grid with coordinates:
        latitude, longitude
    """
    def __init__(self, config, logger, project_dir, **kwargs):
        super().__init__(config, logger, project_dir, **kwargs)
        self.forcing_timestep_seconds: float = float(self.forcing_timestep_seconds) if hasattr(self, 'forcing_timestep_seconds') else float(UnitConversion.SECONDS_PER_HOUR)
    # ---------- Variable mapping / standardization ----------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        AORC → SUMMA/standard variable mapping.

        Uses centralized VariableStandardizer for consistency across the codebase.
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('AORC')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process AORC dataset with variable standardization and unit conversions.

        Transforms raw AORC forcing data into SUMMA-compatible format by renaming
        variables, deriving wind speed from components, converting accumulated
        precipitation to rates, and cleaning NetCDF attributes.

        Args:
            ds: Raw AORC xarray Dataset with variables:
                - APCP_surface: Accumulated precipitation (kg/m²)
                - UGRD_10maboveground: U-component wind (m/s)
                - VGRD_10maboveground: V-component wind (m/s)
                - TMP_2maboveground: Air temperature (K)
                - SPFH_2maboveground: Specific humidity (kg/kg)
                - PRES_surface: Surface pressure (Pa)
                - DSWRF_surface: Downward shortwave radiation (W/m²)
                - DLWRF_surface: Downward longwave radiation (W/m²)

        Returns:
            Processed xarray Dataset with SUMMA-compatible variables:
                - airtemp: Air temperature (K)
                - spechum: Specific humidity (kg/kg)
                - airpres: Surface pressure (Pa)
                - SWRadAtm: Shortwave radiation (W/m²)
                - LWRadAtm: Longwave radiation (W/m²)
                - pptrate: Precipitation rate (mm/s)
                - windspd: Derived wind speed magnitude (m/s)

        Processing Steps:
            1. **Variable Renaming**: Apply centralized AORC → SUMMA mapping
            2. **Wind Derivation**: Calculate windspd = sqrt(u² + v²)
            3. **Precipitation Conversion**:
               - Read original APCP units (kg/m²)
               - Detect accumulation period (hourly assumed)
               - Convert accumulated kg/m² to rate mm/s
               - Formula: rate = (accum / timestep) * (1000 kg/m³ / water_density)
            4. **Attribute Cleaning**: Remove conflicting NetCDF attributes

        Precipitation Unit Conversion:
            AORC provides hourly accumulated precipitation in kg/m²:
            - 1 kg/m² = 1 mm of water (equivalent depth)
            - Hourly accumulation → rate conversion:
              rate [mm/s] = (accum [kg/m²] / 3600 s) * 1 [mm/(kg/m²)]

            Example:
                Hourly accum = 3.6 kg/m² → rate = 3.6 / 3600 = 0.001 mm/s

        Wind Speed Derivation:
            Magnitude from orthogonal components:
            windspd = sqrt(u² + v²)

            Where:
                u = UGRD_10maboveground (eastward wind)
                v = VGRD_10maboveground (northward wind)

            Result units: m/s (unchanged from components)

        Attribute Handling:
            - Preserves essential attributes (units, long_name, standard_name)
            - Removes fill_value, missing_value to avoid NetCDF conflicts
            - Adds derived variable attributes (windspd)
            - Cleans coordinate attributes via clean_variable_attributes()

        Variable Mapping Details:
            Via VariableStandardizer('AORC'):
                AORC Name → SUMMA Name
                TMP_2maboveground → airtemp
                SPFH_2maboveground → spechum
                PRES_surface → airpres
                DSWRF_surface → SWRadAtm
                DLWRF_surface → LWRadAtm
                APCP_surface → pptrate (with conversion)
                UGRD_10maboveground → windspd_u
                VGRD_10maboveground → windspd_v

        Example:
            >>> ds = xr.open_dataset('AORC_2015-2016.nc')
            >>> handler = AORCHandler(config, logger, project_dir)
            >>> ds_processed = handler.process_dataset(ds)
            >>> print(ds_processed.data_vars)
            # Variables: airtemp, spechum, airpres, SWRadAtm, LWRadAtm, pptrate, windspd
            >>> print(ds_processed['precipitation_flux'].units)
            # 'mm s-1'

        Notes:
            - Assumes hourly AORC timestep (3600 seconds)
            - Wind components retained for reference but windspd used by models
            - Precipitation conversion critical for water balance accuracy
            - Lambert Conformal projection coordinates preserved
            - Time encoding follows CF conventions

        See Also:
            - data.utils.VariableStandardizer: Centralized variable mapping
            - data.preprocessing.dataset_handlers.base_dataset: Base handler
            - core.constants.UnitConversion: Unit conversion factors
        """
        var_map = self.get_variable_mapping()

        # Keep original APCP attrs BEFORE rename so we can inspect units
        apcp_attrs = {}
        if 'APCP_surface' in ds.variables:
            apcp_attrs = dict(ds['APCP_surface'].attrs)

        existing = {old: new for old, new in var_map.items() if old in ds.variables}
        ds = ds.rename(existing)

        # ---- Derive wind speed magnitude if components are present ----
        if 'eastward_wind' in ds and 'northward_wind' in ds:
            u = ds['eastward_wind']
            v = ds['northward_wind']
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = 'wind_speed'
            windspd.attrs = {
                'units': 'm s-1',
                'long_name': 'wind speed',
                'standard_name': 'wind_speed'
            }
            ds['wind_speed'] = windspd

        # ---- Precip: convert accumulation -> rate (mm/s) ----
        if 'precipitation_flux' in ds:
            p = ds['precipitation_flux']

            # Units before rename
            units = (apcp_attrs.get('units') or p.attrs.get('units', '')).lower()
            units_no_space = units.replace(' ', '')

            dt_seconds = float(self.forcing_timestep_seconds)

            # Clean inputs: no NaNs, no negatives
            p = xr.where(np.isfinite(p), p, 0.0)
            p = xr.where(p < 0.0, 0.0, p)

            # Detect accumulated precip
            accum_keywords = [
                'kgm-2', 'kgm^-2', 'kg/m^2', 'kg/m2', 'kgm**-2', 'mm'
            ]

            if any(k in units_no_space for k in accum_keywords):
                p_rate = p / dt_seconds
            elif any(k in units_no_space for k in ['kgm-2s-1', 'kgm^-2s^-1', 'kg/m^2s']):
                p_rate = p    # already a rate
            else:
                self.logger.warning(
                    f"Unrecognized precip units for AORC: '{units}', assuming accumulation."
                )
                p_rate = p / dt_seconds

            # Final cleanup
            p_rate = xr.where(np.isfinite(p_rate), p_rate, 0.0)
            p_rate = xr.where(p_rate < 0.0, 0.0, p_rate)

            p_rate.attrs = {
                'units': 'mm/s',
                'long_name': 'Mean total precipitation rate',
                'standard_name': 'precipitation_flux',
            }
            ds['precipitation_flux'] = p_rate


        # Clean NaN values across all forcing variables.
        # The AORC lat-lon grid contains NaN cells outside CONUS coverage
        # (ocean/border regions). These must be handled to prevent NaN
        # propagation through the spatial remapping step.
        ds = self._fill_nan_forcing_variables(ds)

        # Apply standard CF-compliant attributes (uses centralized definitions)
        # Skip pptrate since we already set it with custom units above
        ds = self.apply_standard_attributes(ds)

        return ds

    def _fill_nan_forcing_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Replace NaN values in all forcing variables using nearest-neighbor interpolation.

        The AORC lat-lon gridded dataset (v1.1) stores NaN in the rectangular
        bounding box outside the actual CONUS data coverage. For catchments
        near the edges of CONUS, the spatial subsetting can include these NaN
        cells. If left untreated, NaN propagates through the EASMORE
        remapping weights into the model forcing.

        Strategy:
            1. Interpolate along each spatial dimension using nearest-neighbor
            2. Fill any remaining NaN with the variable's spatial mean
        """
        lat_name, lon_name = self.get_coordinate_names()
        # Support both lat-lon grid (latitude/longitude dims) and
        # NWM projected grid (y/x dims)
        spatial_dims: List[str] = [
            d for d in (lat_name, lon_name, 'y', 'x') if d in ds.dims
        ]

        for var_name in list(ds.data_vars):
            if ds[var_name].dtype.kind != 'f':  # skip non-float variables
                continue

            nan_count = int(ds[var_name].isnull().sum())
            if nan_count == 0:
                continue

            self.logger.info(
                f"Filling {nan_count} NaN values in '{var_name}' "
                f"using nearest-neighbor interpolation"
            )

            # Interpolate along each spatial dimension
            filled = ds[var_name]
            for dim in spatial_dims:
                if dim in filled.dims:
                    filled = filled.interpolate_na(
                        dim=dim, method='nearest', fill_value='extrapolate'
                    )

            # Fallback: fill any remaining NaN with spatial mean
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
        """
        AORC cloud path uses latitude/longitude coordinates, same as ERA5. :contentReference[oaicite:4]{index=4}
        """
        return ('latitude', 'longitude')

    # ---------- Merging (standardization) ----------

    def needs_merging(self) -> bool:
        """
        We *do* mark AORC as needing "merging", but it's really a
        standardization step: we take the raw cloud-downloaded file(s),
        apply variable mapping, and write processed NetCDF to forcing/merged_path.

        This hooks into forcingResampler's existing logic that calls
        dataset_handler.merge_forcings() when needs_merging() is True. :contentReference[oaicite:5]{index=5}
        """
        return True

    def merge_forcings(
        self,
        raw_forcing_path: Path,
        merged_forcing_path: Path,
        start_year: int,
        end_year: int,
    ) -> None:
        """
        "Merge" AORC forcings: for each cloud-downloaded file in raw_data,
        apply process_dataset() and save a standardized file to merged_path.

        No temporal slicing or monthly splitting here: we just 1:1 transform.
        """
        self.logger.info("Standardizing AORC forcing files (no temporal merging)")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        # CloudForcingDownloader._download_aorc currently writes:
        #   <DOMAIN_NAME>_AORC_<startYear>-<endYear>.nc
        # e.g., paradise_AORC_2000-2002.nc
        # but we also support potential future patterns.
        patterns = [
            f"{self.domain_name}_AORC_*.nc",          # paradise_AORC_2000-2002.nc (current)
            f"domain_{self.domain_name}_AORC_*.nc",   # domain_paradise_AORC_*.nc (future-proof)
            "*AORC*.nc",                              # last-resort catch-all
        ]

        files: list[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} AORC file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            self.logger.error(
                f"No AORC files found in {raw_forcing_path} with any of patterns: {patterns}"
            )
            raise FileNotFoundError(
                f"No AORC forcing files found in {raw_forcing_path}"
            )

        # Filter to files whose year range overlaps the configured period
        all_files = files
        files = [
            f for f in all_files
            if self._file_overlaps_period(f, start_year, end_year)
        ]
        skipped = len(all_files) - len(files)
        if skipped:
            self.logger.info(
                f"Skipped {skipped} AORC file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        if not files:
            self.logger.error(
                f"No AORC files match the configured period {start_year}-{end_year}"
            )
            raise FileNotFoundError(
                f"No AORC forcing files match the configured period "
                f"{start_year}-{end_year} in {raw_forcing_path}"
            )

        for f in files:
            self.logger.info(f"Processing AORC file: {f}")
            try:
                ds = xr.open_dataset(f, engine="h5netcdf")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error opening AORC file {f}: {e}")
                continue

            try:
                # Dataset-specific standardization
                ds_proc = self.process_dataset(ds)

                # Common BaseDatasetHandler helpers (time encoding, metadata, attrs)
                ds_proc = self.setup_time_encoding(ds_proc)
                ds_proc = self.add_metadata(
                    ds_proc,
                    "AORC data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
                )
                ds_proc = self.clean_variable_attributes(ds_proc)

                # 🔧 Fix: xarray complains if 'missing_value' is left in attrs,
                # because it's treated as an encoding key. Strip it explicitly.
                for var_name in ds_proc.data_vars:
                    if "missing_value" in ds_proc[var_name].attrs:
                        self.logger.debug(
                            f"Removing 'missing_value' attribute from variable '{var_name}'"
                        )
                        ds_proc[var_name].attrs.pop("missing_value", None)

                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name, engine="h5netcdf")
                self.logger.info(f"Saved processed AORC forcing: {out_name}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error processing AORC dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("AORC forcing standardization completed")

    # ---------- Shapefile for AORC grid ----------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """
        Create AORC grid shapefile.

        We mirror the ERA5/CASR/RDRS logic:
        - open a processed AORC file in merged_forcing_path
        - build polygons from latitude/longitude
        - compute elevation via provided elevation_calculator

        For point domains or very small bounding boxes, we find the nearest
        grid cell to the domain center instead of trying to intersect geometries.
        """
        self.logger.info("Creating AORC grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        # Choose a file to infer grid from
        aorc_files = list(merged_forcing_path.glob('*.nc'))
        if not aorc_files:
            raise FileNotFoundError(f"No AORC processed files found in {merged_forcing_path}")

        aorc_file = aorc_files[0]
        self.logger.info(f"Using AORC file for grid: {aorc_file}")

        # Check if this is a point domain or very small bounding box
        is_point_domain = (self._get_config_value(lambda: self.config.domain.definition_method, default='') or '').lower() == 'point'
        bbox_coords = self._get_config_value(lambda: self.config.domain.bounding_box_coords, default='') or ''
        is_small_bbox = False
        domain_center_lat = None
        domain_center_lon = None

        if bbox_coords:
            try:
                coords = [float(x) for x in str(bbox_coords).split('/')]
                if len(coords) == 4:
                    lat_max, lon_max, lat_min, lon_min = coords
                    bbox_area = abs(lat_max - lat_min) * abs(lon_max - lon_min)
                    is_small_bbox = bbox_area < 0.01  # Less than 0.01 square degrees
                    domain_center_lat = (lat_max + lat_min) / 2
                    domain_center_lon = (lon_max + lon_min) / 2
                    if is_small_bbox:
                        self.logger.info(f"Detected small bounding box (area={bbox_area:.6f} sq deg), will use nearest cell extraction")
            except (ValueError, AttributeError):
                pass

        # If no bounding box center, try pour point coords
        if domain_center_lat is None:
            pour_point = self._get_config_value(lambda: self.config.domain.pour_point_coords, default='') or ''
            if pour_point:
                try:
                    coords = [float(x) for x in str(pour_point).split('/')]
                    if len(coords) == 2:
                        domain_center_lat, domain_center_lon = coords
                        is_point_domain = True
                        self.logger.info(f"Using pour point coordinates: {domain_center_lat}/{domain_center_lon}")
                except (ValueError, AttributeError):
                    pass

        with xr.open_dataset(aorc_file, engine="h5netcdf") as ds:
            var_lat, var_lon = self.get_coordinate_names()

            # Try coords first, then data_vars, then fail
            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in AORC file {aorc_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in AORC file {aorc_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        # 1D or 2D grid handling
        geometries = []
        ids = []
        lats = []
        lons = []

        # For point domains or small bboxes, find the nearest cell
        if (is_point_domain or is_small_bbox) and domain_center_lat is not None:
            self.logger.info(f"Finding nearest AORC grid cell to domain center ({domain_center_lat:.4f}, {domain_center_lon:.4f})")

            if lat.ndim == 1 and lon.ndim == 1:
                # Regular 1D grid - find nearest indices
                lat_idx = np.argmin(np.abs(lat - domain_center_lat))
                lon_idx = np.argmin(np.abs(lon - domain_center_lon))

                center_lat = float(lat[lat_idx])
                center_lon = float(lon[lon_idx])

                half_dlat = abs(lat[1] - lat[0]) / 2 if len(lat) > 1 else 0.005
                half_dlon = abs(lon[1] - lon[0]) / 2 if len(lon) > 1 else 0.005

                verts = [
                    [center_lon - half_dlon, center_lat - half_dlat],
                    [center_lon - half_dlon, center_lat + half_dlat],
                    [center_lon + half_dlon, center_lat + half_dlat],
                    [center_lon + half_dlon, center_lat - half_dlat],
                    [center_lon - half_dlon, center_lat - half_dlat],
                ]
                geometries.append(Polygon(verts))
                ids.append(0)
                lats.append(center_lat)
                lons.append(center_lon)

                self.logger.info(f"Selected nearest cell: lat={center_lat:.4f}, lon={center_lon:.4f}")

            elif lat.ndim == 2:
                # 2D grid - find nearest cell
                distances = np.sqrt((lat - domain_center_lat)**2 + (lon - domain_center_lon)**2)
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

                self.logger.info(f"Selected nearest cell: lat={lat[i,j]:.4f}, lon={lon[i,j]:.4f}")

        elif lat.ndim == 1 and lon.ndim == 1:
            # Regular lat/lon grid (ERA5-style) :contentReference[oaicite:10]{index=10}
            half_dlat = abs(lat[1] - lat[0]) / 2 if len(lat) > 1 else 0.005
            half_dlon = abs(lon[1] - lon[0]) / 2 if len(lon) > 1 else 0.005

            for i, center_lon in enumerate(lon):
                for j, center_lat in enumerate(lat):
                    verts = [
                        [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                        [float(center_lon) - half_dlon, float(center_lat) + half_dlat],
                        [float(center_lon) + half_dlon, float(center_lat) + half_dlat],
                        [float(center_lon) + half_dlon, float(center_lat) - half_dlat],
                        [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                    ]
                    geometries.append(Polygon(verts))
                    ids.append(i * len(lat) + j)
                    lats.append(float(center_lat))
                    lons.append(float(center_lon))
        else:
            # 2D lat/lon grid (CASR/RDRS-style) :contentReference[oaicite:11]{index=11}
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(f"AORC grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

            cell_count = 0
            for i in range(ny):
                for j in range(nx):
                    # Corner indices with edge-safe fallbacks
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
                        self.logger.info(f"Created {cell_count}/{total_cells} AORC grid cells")

        gdf = gpd.GeoDataFrame(
            {
                'geometry': geometries,
                'ID': ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): lons,
            },
            crs='EPSG:4326',
        )

        # Elevation using the safe helper used by other handlers
        self.logger.info("Calculating elevation values for AORC grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf['elev_m'] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"AORC shapefile created at {output_shapefile}")

        return output_shapefile
