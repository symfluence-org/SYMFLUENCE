# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Daymet dataset handler for daily surface weather forcing.

Processes Daymet gridded climate data (1 km, daily) with variable
standardization, unit conversions, and derived variable estimation.

Daymet provides: tmax, tmin, prcp, srad, vp, dayl, swe
Daymet does NOT provide: longwave radiation, pressure, specific humidity, wind speed.
These are estimated from available variables when possible.
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


@DatasetRegistry.register("daymet")
class DaymetHandler(BaseDatasetHandler):
    """
    Handler for Daymet daily surface weather forcing data.

    Daymet variables and native units:
        tmax  – daily maximum 2m temperature (deg C)
        tmin  – daily minimum 2m temperature (deg C)
        prcp  – daily total precipitation (mm/day)
        srad  – incident shortwave radiation (W/m^2, daylight average)
        vp    – water vapor pressure (Pa)
        dayl  – day length (seconds/day)
        swe   – snow water equivalent (kg/m^2)

    Limitations:
        Daymet does NOT provide longwave radiation, surface pressure,
        specific humidity, or wind speed. This handler estimates:
        - airtemp as mean of tmax and tmin, converted to K
        - airpres from elevation using the barometric formula
        - spechum from vapor pressure and estimated surface pressure
        - LWRadAtm from air temperature and vapor pressure (Brutsaert 1975)
        - windspd set to a default climatological value (2 m/s)

    Coverage: North America, 1 km, daily, 1980-present.
    """

    # Default wind speed (m/s) when no wind data available
    DEFAULT_WIND_SPEED = 2.0

    # Standard atmosphere constants for pressure estimation
    SEA_LEVEL_PRESSURE = 101325.0  # Pa
    LAPSE_RATE_STANDARD = 0.0065   # K/m
    GRAVITY = 9.80665              # m/s^2
    MOLAR_MASS_AIR = 0.0289644     # kg/mol
    GAS_CONSTANT = 8.31447         # J/(mol*K)
    SEA_LEVEL_TEMP = 288.15        # K

    # Stefan-Boltzmann constant
    STEFAN_BOLTZMANN = 5.67e-8     # W/(m^2 K^4)

    def get_variable_mapping(self) -> Dict[str, str]:
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('DayMet')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process Daymet dataset: rename, convert units, derive missing variables.

        Steps:
            1. Rename raw Daymet variables to standard names
            2. Derive mean air temperature (K) from tmax/tmin
            3. Convert precipitation mm/day -> kg m-2 s-1
            4. Adjust shortwave radiation from daylight-average to 24h-average
            5. Estimate surface pressure from elevation
            6. Estimate specific humidity from vapor pressure
            7. Estimate longwave radiation (Brutsaert 1975)
            8. Set default wind speed
        """
        ds = ds.copy()

        # Clean missing_value attrs
        for vname in list(ds.variables):
            if "missing_value" in ds[vname].attrs:
                ds[vname].attrs.pop("missing_value", None)

        # Rename using centralized mapping
        var_map = self.get_variable_mapping()
        rename_map = {src: tgt for src, tgt in var_map.items() if src in ds.data_vars}
        if rename_map:
            self.logger.debug(f"Renaming Daymet variables: {rename_map}")
            ds = rename_map_apply(ds, rename_map)

        # --- Derive mean air temperature (K) ---
        if 'air_temperature_max' in ds and 'air_temperature_min' in ds:
            tmax_c = ds['air_temperature_max']
            tmin_c = ds['air_temperature_min']
            airtemp_k = (tmax_c + tmin_c) / 2.0 + 273.15
            airtemp_k.attrs = {
                'units': 'K',
                'long_name': 'air temperature (mean of tmax and tmin)',
                'standard_name': 'air_temperature',
            }
            ds['air_temperature'] = airtemp_k
            self.logger.info("Derived mean air temperature from tmax/tmin")

        # --- Precipitation: mm/day -> kg m-2 s-1 ---
        if 'precipitation_flux' in ds:
            prcp = ds['precipitation_flux'].astype('float32')
            prcp = xr.where(np.isfinite(prcp), prcp, 0.0)
            prcp = xr.where(prcp < 0.0, 0.0, prcp)
            # mm/day -> kg m-2 s-1 (1 mm = 1 kg/m^2)
            ds['precipitation_flux'] = prcp / 86400.0
            ds['precipitation_flux'].attrs = {
                'units': 'kg m-2 s-1',
                'long_name': 'precipitation rate',
                'standard_name': 'precipitation_flux',
            }

        # --- Shortwave radiation: daylight average -> 24h average ---
        if 'surface_downwelling_shortwave_flux' in ds and 'day_length' in ds:
            dayl_s = ds['day_length']
            # Daymet srad is average over daylight hours; convert to 24h average
            ds['surface_downwelling_shortwave_flux'] = ds['surface_downwelling_shortwave_flux'] * (dayl_s / 86400.0)
            ds['surface_downwelling_shortwave_flux'].attrs = {
                'units': 'W m-2',
                'long_name': 'downward shortwave radiation (24h average)',
                'standard_name': 'surface_downwelling_shortwave_flux_in_air',
            }
            self.logger.info("Converted Daymet shortwave from daylight-average to 24h-average")
        elif 'surface_downwelling_shortwave_flux' in ds:
            ds['surface_downwelling_shortwave_flux'].attrs = {
                'units': 'W m-2',
                'long_name': 'downward shortwave radiation at the surface',
                'standard_name': 'surface_downwelling_shortwave_flux_in_air',
            }

        # --- Estimate surface pressure from elevation ---
        if 'surface_air_pressure' not in ds:
            elevation = self._get_elevation(ds)
            if 'air_temperature' in ds:
                t_k = ds['air_temperature']
            else:
                t_k = self.SEA_LEVEL_TEMP

            exponent = (self.GRAVITY * self.MOLAR_MASS_AIR) / (self.GAS_CONSTANT * self.LAPSE_RATE_STANDARD)
            airpres = self.SEA_LEVEL_PRESSURE * (1 - self.LAPSE_RATE_STANDARD * elevation / self.SEA_LEVEL_TEMP) ** exponent
            if isinstance(airpres, xr.DataArray):
                airpres.attrs = {
                    'units': 'Pa',
                    'long_name': 'surface air pressure (estimated from elevation)',
                    'standard_name': 'air_pressure',
                }
            else:
                airpres = xr.DataArray(
                    np.full_like(ds['air_temperature'].values, airpres) if 'air_temperature' in ds else airpres,
                    dims=ds['air_temperature'].dims if 'air_temperature' in ds else (),
                    attrs={
                        'units': 'Pa',
                        'long_name': 'surface air pressure (estimated from elevation)',
                        'standard_name': 'air_pressure',
                    }
                )
            ds['surface_air_pressure'] = airpres
            self.logger.info("Estimated surface pressure from elevation (barometric formula)")

        # --- Estimate specific humidity from vapor pressure ---
        if 'specific_humidity' not in ds and 'water_vapor_pressure' in ds and 'surface_air_pressure' in ds:
            vp = ds['water_vapor_pressure']  # Pa
            p = ds['surface_air_pressure']                # Pa
            # q = 0.622 * e / (p - 0.378 * e)
            epsilon = 0.622
            spechum = epsilon * vp / (p - (1 - epsilon) * vp)
            spechum = xr.where(spechum < 0, 0.0, spechum)
            spechum.attrs = {
                'units': 'kg kg-1',
                'long_name': 'specific humidity (estimated from vapor pressure)',
                'standard_name': 'specific_humidity',
            }
            ds['specific_humidity'] = spechum
            self.logger.info("Estimated specific humidity from vapor pressure")

        # --- Estimate longwave radiation (Brutsaert 1975) ---
        if 'surface_downwelling_longwave_flux' not in ds and 'air_temperature' in ds:
            t_k = ds['air_temperature']
            if 'water_vapor_pressure' in ds:
                vp_hpa = ds['water_vapor_pressure'] / 100.0  # Pa -> hPa
                # Brutsaert (1975) clear-sky emissivity
                emissivity = 1.24 * (vp_hpa / t_k) ** (1.0 / 7.0)
            else:
                # Fallback: assume emissivity ~ 0.85 (typical clear-sky)
                emissivity = 0.85

            lw = emissivity * self.STEFAN_BOLTZMANN * t_k ** 4
            lw.attrs = {
                'units': 'W m-2',
                'long_name': 'downward longwave radiation (estimated, Brutsaert 1975)',
                'standard_name': 'surface_downwelling_longwave_flux_in_air',
            }
            ds['surface_downwelling_longwave_flux'] = lw
            self.logger.info("Estimated longwave radiation using Brutsaert (1975)")

        # --- Default wind speed ---
        if 'wind_speed' not in ds and 'air_temperature' in ds:
            template = ds['air_temperature']
            windspd = xr.full_like(template, self.DEFAULT_WIND_SPEED)
            windspd.attrs = {
                'units': 'm s-1',
                'long_name': f'wind speed (default {self.DEFAULT_WIND_SPEED} m/s)',
                'standard_name': 'wind_speed',
                'note': 'Daymet does not provide wind data; using climatological default',
            }
            ds['wind_speed'] = windspd
            self.logger.warning(
                f"Daymet does not provide wind speed. "
                f"Using default value of {self.DEFAULT_WIND_SPEED} m/s. "
                f"Consider supplementing with another dataset (ERA5, HRRR) for wind."
            )

        # Standard metadata + encoding
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(ds, "Daymet data standardized for forcing (SYMFLUENCE)")
        ds = self.clean_variable_attributes(ds)

        return ds

    def _get_elevation(self, ds: xr.Dataset) -> float:
        """Extract elevation from config or dataset, or use a default."""
        # Try config
        elev = self._get_config_value(lambda: None, default=None, dict_key='CATCHMENT_MEAN_ELEVATION')
        if elev is not None:
            return float(elev)

        # Try dataset variable
        for elev_name in ('elevation', 'elev', 'dem', 'alt'):
            if elev_name in ds:
                return float(ds[elev_name].mean())

        # Default: 500m (reasonable mid-range)
        self.logger.warning(
            "No elevation found for pressure estimation. "
            "Set CATCHMENT_MEAN_ELEVATION in config for better accuracy. Using 500m default."
        )
        return 500.0

    def get_coordinate_names(self) -> Tuple[str, str]:
        """Daymet NetCDF files from THREDDS use 'lat' and 'lon'."""
        return ('lat', 'lon')

    def needs_merging(self) -> bool:
        """Daymet downloads are per-variable per-year; they need merging."""
        return True

    def merge_forcings(
        self,
        raw_forcing_path: Path,
        merged_forcing_path: Path,
        start_year: int,
        end_year: int,
    ) -> None:
        """
        Merge per-variable, per-year Daymet files into annual multi-variable files.

        The acquisition handler downloads files as daymet_{var}_{year}.nc.
        This method merges all variables for each year, then processes them.
        """
        self.logger.info("Merging and standardizing Daymet forcing files")
        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        for year in range(start_year, end_year + 1):
            out_file = merged_forcing_path / f"Daymet_{year}_processed.nc"
            if out_file.exists():
                self.logger.info(f"Daymet {year} already processed: {out_file}")
                continue

            # Collect per-variable files for this year
            var_datasets = []
            patterns = [
                f"daymet_*_{year}.nc",
                f"*daymet*{year}*.nc",
                f"*Daymet*{year}*.nc",
                f"*DAYMET*{year}*.nc",
            ]

            found_files: List[Path] = []
            for pattern in patterns:
                candidates = sorted(raw_forcing_path.glob(pattern))
                if candidates:
                    found_files = candidates
                    break

            if not found_files:
                # Try single combined file
                combined_patterns = [
                    f"daymet_*_{start_year}*.nc",
                    f"{self.domain_name}_*daymet*.nc",
                    "*daymet*.nc",
                ]
                for pattern in combined_patterns:
                    candidates = sorted(raw_forcing_path.glob(pattern))
                    if candidates:
                        found_files = candidates
                        break

            if not found_files:
                self.logger.warning(f"No Daymet files found for year {year} in {raw_forcing_path}")
                continue

            self.logger.info(f"Found {len(found_files)} Daymet file(s) for {year}")

            for f in found_files:
                try:
                    ds = self.open_dataset(f)
                    var_datasets.append(ds)
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Error opening {f}: {e}")

            if not var_datasets:
                self.logger.warning(f"No valid Daymet datasets for year {year}")
                continue

            # Merge all variables into one dataset
            try:
                if len(var_datasets) == 1:
                    merged = var_datasets[0]
                else:
                    merged = xr.merge(var_datasets, compat='override', join='inner')

                # Select year if time dimension spans multiple years
                if 'time' in merged.dims:
                    time_vals = merged['time'].dt.year
                    year_mask = time_vals == year
                    if year_mask.any() and not year_mask.all():
                        merged = merged.sel(time=year_mask)

                # Process
                processed = self.process_dataset(merged)
                processed.to_netcdf(out_file)
                self.logger.info(f"Saved processed Daymet forcing: {out_file}")

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error processing Daymet data for {year}: {e}")
            finally:
                for ds in var_datasets:
                    ds.close()

        self.logger.info("Daymet forcing merge and standardization completed")

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """Create Daymet grid shapefile from lat/lon coordinates."""
        self.logger.info("Creating Daymet grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='DAYMET')}.shp"

        nc_files = sorted(merged_forcing_path.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No Daymet processed files in {merged_forcing_path}")

        sample_file = nc_files[0]
        self.logger.info(f"Using Daymet file for grid: {sample_file}")

        with self.open_dataset(sample_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            lat = self._extract_coord(ds, var_lat, 'latitude')
            lon = self._extract_coord(ds, var_lon, 'longitude')

        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
            half_dlat = abs(lat[1] - lat[0]) / 2 if len(lat) > 1 else 0.005
            half_dlon = abs(lon[1] - lon[0]) / 2 if len(lon) > 1 else 0.005

            for i, clon in enumerate(lon):
                for j, clat in enumerate(lat):
                    verts = [
                        [float(clon) - half_dlon, float(clat) - half_dlat],
                        [float(clon) - half_dlon, float(clat) + half_dlat],
                        [float(clon) + half_dlon, float(clat) + half_dlat],
                        [float(clon) + half_dlon, float(clat) - half_dlat],
                        [float(clon) - half_dlon, float(clat) - half_dlat],
                    ]
                    geometries.append(Polygon(verts))
                    ids.append(i * len(lat) + j)
                    lats.append(float(clat))
                    lons.append(float(clon))
        else:
            # 2D coordinate arrays (Daymet native projection)
            ny, nx = lat.shape
            self.logger.info(f"Daymet grid (2D): ny={ny}, nx={nx}")

            for i in range(ny):
                for j in range(nx):
                    clat = float(lat[i, j])
                    clon = float(lon[i, j])
                    if np.isnan(clat) or np.isnan(clon):
                        continue

                    # Approximate cell size from neighbors
                    dlat = 0.01  # ~1km default
                    dlon = 0.01
                    if i + 1 < ny:
                        dlat = abs(float(lat[i + 1, j]) - clat) / 2
                    if j + 1 < nx:
                        dlon = abs(float(lon[i, j + 1]) - clon) / 2

                    verts = [
                        [clon - dlon, clat - dlat],
                        [clon - dlon, clat + dlat],
                        [clon + dlon, clat + dlat],
                        [clon + dlon, clat - dlat],
                        [clon - dlon, clat - dlat],
                    ]
                    geometries.append(Polygon(verts))
                    ids.append(i * nx + j)
                    lats.append(clat)
                    lons.append(clon)

        lat_col = self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat')
        lon_col = self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon')

        gdf = gpd.GeoDataFrame(
            {'geometry': geometries, 'ID': ids, lat_col: lats, lon_col: lons},
            crs='EPSG:4326',
        )

        self.logger.info("Calculating elevation values for Daymet grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf['elev_m'] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"Daymet shapefile created at {output_shapefile}")

        return output_shapefile

    def _extract_coord(self, ds: xr.Dataset, name: str, desc: str):
        """Extract a coordinate array from dataset, trying coords then variables."""
        if name in ds.coords:
            return ds.coords[name].values
        if name in ds.variables:
            return ds[name].values
        # Try alternatives
        for alt in ('latitude', 'lat', 'y') if 'lat' in name else ('longitude', 'lon', 'x'):
            if alt in ds.coords:
                return ds.coords[alt].values
            if alt in ds.variables:
                return ds[alt].values
        raise KeyError(
            f"{desc} coordinate '{name}' not found. "
            f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
        )


def rename_map_apply(ds: xr.Dataset, rename_map: Dict[str, str]) -> xr.Dataset:
    """Apply rename map, handling duplicate target names gracefully."""
    # Check for duplicate targets
    seen = {}
    safe_map = {}
    for src, tgt in rename_map.items():
        if tgt in seen:
            # Skip duplicate mapping
            continue
        if src in ds.data_vars:
            safe_map[src] = tgt
            seen[tgt] = src
    return ds.rename(safe_map)
