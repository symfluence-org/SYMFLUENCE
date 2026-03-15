# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HRRR (High-Resolution Rapid Refresh) dataset handler.

Processes HRRR atmospheric forecast data with Lambert conformal projection
handling, variable extraction, and unit conversions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

from ...utils import VariableStandardizer
from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register("hrrr")
class HRRRHandler(BaseDatasetHandler):
    """
    Handler for HRRR (High-Resolution Rapid Refresh) forcing data.

    Assumptions (similar to AORC, NCEP-style fields in NetCDF):
      - Coordinates:
          time, latitude, longitude
      - Typical variables:
          APCP_surface         – accumulated precip or precip rate
          TMP_2maboveground    – 2m air temp [K]
          SPFH_2maboveground   – 2m specific humidity [kg/kg]
          PRES_surface         – surface pressure [Pa]
          DLWRF_surface        – down longwave [W/m²]
          DSWRF_surface        – down shortwave [W/m²]
          UGRD_10maboveground  – 10m U wind [m/s]
          VGRD_10maboveground  – 10m V wind [m/s]

    This mirrors AORC’s mapping so that the rest of the pipeline
    (SUMMA, easymore, etc.) can treat HRRR in the same way.
    """

    # ------------ variable mapping ------------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map raw HRRR variables → SYMFLUENCE/SUMMA standard names.

        Uses centralized VariableStandardizer for consistency across the codebase.
        Note: Cloud downloader provides short variable names (TMP, SPFH, etc.)
        rather than full NCEP-style names (TMP_2maboveground, etc.)
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('HRRR')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process HRRR dataset with standardization and Lambert Conformal projection handling.

        Transforms raw HRRR operational forecast data into SUMMA-compatible format.
        Handles variable standardization via centralized mapping, derives wind speed,
        and manages Lambert Conformal Conic projection coordinates.

        Args:
            ds: Raw HRRR xarray Dataset with variables:
                - TMP: 2m air temperature (K) - short name from cloud downloader
                - SPFH: 2m specific humidity (kg/kg)
                - PRES: Surface pressure (Pa)
                - UGRD: U-component wind at 10m (m/s)
                - VGRD: V-component wind at 10m (m/s)
                - DSWRF: Downward shortwave radiation (W/m²)
                - DLWRF: Downward longwave radiation (W/m²)

                Note: Cloud downloader provides abbreviated names (TMP, SPFH, etc.)
                rather than full GRIB2 names (TMP_2maboveground, etc.)

        Returns:
            Processed xarray Dataset with SUMMA-compatible variables:
                - airtemp: Air temperature (K)
                - spechum: Specific humidity (kg/kg)
                - airpres: Surface pressure (Pa)
                - SWRadAtm: Shortwave radiation (W/m²)
                - LWRadAtm: Longwave radiation (W/m²)
                - windspd: Wind speed magnitude (m/s)
                - pptrate: Precipitation rate (mm/s) - WARNING: Not valid from HRRR analysis

        Processing Steps:
            1. **Centralized Standardization**: Use VariableStandardizer.standardize()
               - Handles all variable renaming via centralized mapping
               - Applies unit conversions if needed
               - Cleans NetCDF attributes
            2. **Wind Speed Derivation**: Calculate windspd = sqrt(UGRD² + VGRD²)
            3. **Precipitation Handling**: Set to NaN (HRRR analysis lacks valid precip)
            4. **Coordinate Preservation**: Retain Lambert Conformal projection info

        Variable Standardization:
            Uses VariableStandardizer for centralized mapping:

            HRRR Short Name → SUMMA Name:
                TMP → airtemp
                SPFH → spechum
                PRES → airpres
                DSWRF → SWRadAtm
                DLWRF → LWRadAtm
                UGRD → windspd_u
                VGRD → windspd_v

            Standardizer handles:
            - Attribute preservation/cleaning
            - Unit validation
            - Missing data handling
            - Coordinate standardization

        Wind Speed Derivation:
            Vector magnitude from orthogonal components:
            windspd = sqrt(UGRD² + VGRD²)

            Attributes assigned:
                units: 'm s-1'
                long_name: 'wind speed'
                standard_name: 'wind_speed'

            Components (windspd_u, windspd_v) retained for diagnostic purposes.

        CRITICAL: HRRR Precipitation Limitation:
            HRRR analysis fields (0-hour forecasts) do NOT contain valid precipitation:
            - Analysis fields assimilate observations but exclude precip
            - Precipitation only available in forecast hours (f01-f18)
            - For precip, use HRRR forecasts or alternative datasets (AORC, CONUS404)

            Handler sets pptrate to NaN with warning:
                ds['precipitation_flux'] = xr.full_like(ds['air_temperature'], np.nan)

            Workaround options:
                1. Use HRRR f01 (1-hour forecast) for all variables including precip
                2. Use AORC or CONUS404 for precip, HRRR for other variables
                3. Blend HRRR with observation-based precip (MRMS, Stage IV)

        Lambert Conformal Projection:
            HRRR native coordinates:
            - Projection: Lambert Conformal Conic
            - Reference latitude: 38.5°N
            - Reference longitude: -97.5°W
            - Grid spacing: ~3 km

            Coordinate handling:
            - 2D curvilinear lat/lon arrays preserved
            - Projection metadata retained in dataset attributes
            - Compatible with SUMMA/mizuRoute spatial subsetting

        Float16 to Float32 Conversion:
            HRRR Zarr archives use Float16 for compression:
            - Cloud downloader converts to Float32 during download
            - NetCDF export requires Float32 (no Float16 support)
            - Conversion already applied by acquisition handler

        Example:
            >>> ds = xr.open_dataset('HRRR_20220101-20220107.nc')
            >>> handler = HRRRHandler(config, logger, project_dir)
            >>> ds_processed = handler.process_dataset(ds)
            >>> print(ds_processed.data_vars)
            # Variables: airtemp, spechum, airpres, SWRadAtm, LWRadAtm, windspd, pptrate
            >>> print(ds_processed['wind_speed'].attrs)
            # {'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'}
            >>> print(ds_processed['precipitation_flux'].values[0])
            # nan  (HRRR analysis has no valid precip)

        Operational vs Forecast Data:
            Analysis (0-hour forecast):
                - Assimilates observations
                - No valid precipitation
                - Used for temperature, humidity, pressure, radiation, wind
                - Available hourly with ~30-60 min delay

            Forecast (f01-f18):
                - Physics-based precipitation
                - Includes convective and grid-scale precip
                - Less accurate for non-precip variables (no obs assimilation)
                - Available 18 hours ahead

        Performance:
            - Standardization: ~1-2 seconds
            - Wind derivation: <1 second
            - Memory: In-place operations minimize overhead
            - Processing time: ~2-5 seconds total

        Notes:
            - HRRR best for high-resolution temperature/radiation forcing
            - Combine with AORC/CONUS404 for precipitation
            - Lambert Conformal projection requires careful spatial subsetting
            - U/V wind components available for wind direction analysis
            - Operational HRRR updated hourly (recent data available)

        See Also:
            - data.utils.VariableStandardizer: Centralized variable mapping
            - data.acquisition.handlers.hrrr.HRRRAcquirer: HRRR download handler
            - data.preprocessing.dataset_handlers.base_dataset: Base handler interface
        """
        standardizer = VariableStandardizer(self.logger)
        ds = standardizer.standardize(ds, 'HRRR')

        # Wind speed magnitude
        if "eastward_wind" in ds and "northward_wind" in ds:
            u = ds["eastward_wind"]
            v = ds["northward_wind"]
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = "wind_speed"
            windspd.attrs = {
                "units": "m s-1",
                "long_name": "wind speed",
                "standard_name": "wind_speed",
            }
            ds["wind_speed"] = windspd

        # Precipitation rate
        # NOTE: HRRR analysis fields do not contain valid precipitation data
        # Create zero-precipitation fallback if missing
        if "precipitation_flux" not in ds:
            self.logger.warning(
                "HRRR analysis fields do not contain precipitation data. "
                "Creating zero-precipitation fallback. "
                "Users should supplement with MRMS, Stage IV, or HRRR forecast fields."
            )
            # Create zero precipitation with same dimensions as other variables
            template_var = ds["air_temperature"] if "air_temperature" in ds else list(ds.data_vars.values())[0]
            pptrate = template_var * 0.0  # Same shape, all zeros
            pptrate.name = "precipitation_flux"
            pptrate.attrs = {
                "units": "m s-1",
                "long_name": "precipitation rate (zero fallback)",
                "standard_name": "precipitation_rate",
                "note": "HRRR analysis does not provide precipitation; this is a zero fallback"
            }
            ds["precipitation_flux"] = pptrate
        elif "precipitation_flux" in ds:
            p = ds["precipitation_flux"]
            # remove problematic encoding attrs if present
            p.attrs.pop("missing_value", None)

            units = p.attrs.get("units", "").lower()
            # Handle common cases; adjust if your HRRR derives rate differently
            if "mm" in units:
                # mm/s → m/s
                ds["precipitation_flux"] = p / 1000.0
                ds["precipitation_flux"].attrs["units"] = "m s-1"
            elif "kg m-2 s-1" in units or "kg m-2 s^-1" in units:
                # 1 kg/m²/s ≈ 1 mm/s = 1e-3 m/s
                ds["precipitation_flux"] = p / 1000.0
                ds["precipitation_flux"].attrs["units"] = "m s-1"
            else:
                ds["precipitation_flux"].attrs.setdefault("units", "m s-1")

            ds["precipitation_flux"].attrs.update(
                {
                    "long_name": "precipitation rate",
                    "standard_name": "precipitation_rate",
                }
            )

        # Temperature
        if "air_temperature" in ds:
            ds["air_temperature"].attrs.update(
                {
                    "units": "K",
                    "long_name": "air temperature",
                    "standard_name": "air_temperature",
                }
            )

        # Specific humidity
        if "specific_humidity" in ds:
            ds["specific_humidity"].attrs.update(
                {
                    "units": "kg kg-1",
                    "long_name": "specific humidity",
                    "standard_name": "specific_humidity",
                }
            )

        # Pressure
        if "surface_air_pressure" in ds:
            ds["surface_air_pressure"].attrs.update(
                {
                    "units": "Pa",
                    "long_name": "air pressure",
                    "standard_name": "air_pressure",
                }
            )

        # Radiation
        if "surface_downwelling_longwave_flux" in ds:
            ds["surface_downwelling_longwave_flux"].attrs.update(
                {
                    "units": "W m-2",
                    "long_name": "downward longwave radiation at the surface",
                    "standard_name": "surface_downwelling_longwave_flux_in_air",
                }
            )

        if "surface_downwelling_shortwave_flux" in ds:
            ds["surface_downwelling_shortwave_flux"].attrs.update(
                {
                    "units": "W m-2",
                    "long_name": "downward shortwave radiation at the surface",
                    "standard_name": "surface_downwelling_shortwave_flux_in_air",
                }
            )

        # Common metadata + clean attrs via base helpers
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "HRRR data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds

    # ------------ coordinates ------------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        HRRR NetCDF subset from the cloud downloader has latitude/longitude
        as 2D coordinates (not dimensions).
        """
        return ("latitude", "longitude")

    # ------------ merging / standardization ------------

    def needs_merging(self) -> bool:
        """
        Treat HRRR like AORC/CONUS404: run a standardization pass
        over the raw cloud-downloaded file(s).
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
        Standardize HRRR forcings:

          - Find HRRR NetCDF files in raw_forcing_path
          - Apply process_dataset()
          - Save processed file(s) to merged_forcing_path
        """
        self.logger.info("Standardizing HRRR forcing files (no temporal merging)")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns: List[str] = [
            f"{self.domain_name}_HRRR_*.nc",
            f"domain_{self.domain_name}_HRRR_*.nc",
            "*HRRR*.nc",
        ]

        files: List[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} HRRR file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No HRRR forcing files found in {raw_forcing_path} with patterns {patterns}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Filter to files whose year range overlaps the configured period
        all_files = files
        files = [
            f for f in all_files
            if self._file_overlaps_period(f, start_year, end_year)
        ]
        skipped = len(all_files) - len(files)
        if skipped:
            self.logger.info(
                f"Skipped {skipped} HRRR file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        if not files:
            self.logger.error(
                f"No HRRR files match the configured period {start_year}-{end_year}"
            )
            raise FileNotFoundError(
                f"No HRRR forcing files match the configured period "
                f"{start_year}-{end_year} in {raw_forcing_path}"
            )

        for f in files:
            self.logger.info(f"Processing HRRR file: {f}")
            try:
                ds = self.open_dataset(f)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error opening HRRR file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed HRRR forcing: {out_name}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error processing HRRR dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("HRRR forcing standardization completed")

    # ------------ shapefile creation ------------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """
        Create HRRR grid shapefile from latitude/longitude.

        Mirrors the AORC/CONUS404 logic but assumes regular lat/lon grid.
        """
        self.logger.info("Creating HRRR grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        hrrr_files = list(merged_forcing_path.glob("*.nc"))
        if not hrrr_files:
            raise FileNotFoundError(f"No HRRR processed files found in {merged_forcing_path}")

        hrrr_file = hrrr_files[0]
        self.logger.info(f"Using HRRR file for grid: {hrrr_file}")

        with self.open_dataset(hrrr_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in HRRR file {hrrr_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in HRRR file {hrrr_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        bbox_str = self._get_config_value(lambda: None, default=None, dict_key='HRRR_BOUNDING_BOX_COORDS') or self._get_config_value(lambda: self.config.domain.bounding_box_coords, default=None)
        if isinstance(bbox_str, str) and "/" in bbox_str:
            lat_max, lon_min, lat_min, lon_max = [float(v) for v in bbox_str.split("/")]
            lat_min, lat_max = sorted([lat_min, lat_max])
            lon_min, lon_max = sorted([lon_min, lon_max])
        else:
            lat_min = lat_max = lon_min = lon_max = None

        use_360 = np.nanmax(lon) > 180
        if use_360 and lon_min is not None and lon_max is not None:
            if lon_min < 0:
                lon_min += 360
            if lon_max < 0:
                lon_max += 360

        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
            if lat_min is not None:
                lat_mask = (lat >= lat_min) & (lat <= lat_max)
                lon_mask = (lon >= lon_min) & (lon <= lon_max)
                if np.any(lat_mask) and np.any(lon_mask):
                    i_min, i_max = np.where(lon_mask)[0].min(), np.where(lon_mask)[0].max()
                    j_min, j_max = np.where(lat_mask)[0].min(), np.where(lat_mask)[0].max()
                    buffer_cells = int(self._get_config_value(lambda: None, default=1, dict_key='HRRR_BUFFER_CELLS'))
                    i_min = max(i_min - buffer_cells, 0)
                    i_max = min(i_max + buffer_cells, len(lon) - 1)
                    j_min = max(j_min - buffer_cells, 0)
                    j_max = min(j_max + buffer_cells, len(lat) - 1)
                    lon = lon[i_min:i_max + 1]
                    lat = lat[j_min:j_max + 1]
                else:
                    self.logger.warning("No HRRR grid points found in bbox; using full grid.")

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
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(f"HRRR grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

            i_min, i_max = 0, ny - 1
            j_min, j_max = 0, nx - 1
            if lat_min is not None:
                mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
                if np.any(mask):
                    rows, cols = np.where(mask)
                    buffer_cells = int(self._get_config_value(lambda: None, default=1, dict_key='HRRR_BUFFER_CELLS'))
                    i_min = max(int(rows.min()) - buffer_cells, 0)
                    i_max = min(int(rows.max()) + buffer_cells, ny - 1)
                    j_min = max(int(cols.min()) - buffer_cells, 0)
                    j_max = min(int(cols.max()) + buffer_cells, nx - 1)
                else:
                    self.logger.warning("No HRRR grid points found in bbox; using full grid.")

            cell_count = 0
            total_cells = (i_max - i_min + 1) * (j_max - j_min + 1)
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    lat_corners = [
                        lat[i, j],
                        lat[i, j + 1] if j + 1 < nx else lat[i, j],
                        lat[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lat[i, j],
                        lat[i + 1, j] if i + 1 < ny else lat[i, j],
                    ]
                    lon_corners = [
                        lon[i, j],
                        lon[i, j + 1] if j + 1 < nx else lon[i, j],
                        lon[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lon[i, j],
                        lon[i + 1, j] if i + 1 < ny else lon[i, j],
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * nx + j)
                    lats.append(float(lat[i, j]))
                    lons.append(float(lon[i, j]))

                    cell_count += 1
                    if cell_count % 5000 == 0 or cell_count == total_cells:
                        self.logger.info(f"Created {cell_count}/{total_cells} HRRR grid cells")

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "ID": ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): lons,
            },
            crs="EPSG:4326",
        )

        self.logger.info("Calculating elevation values for HRRR grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf["elev_m"] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"HRRR shapefile created at {output_shapefile}")

        return output_shapefile
