# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NEX-GDDP-CMIP6 dataset handler for climate projections.

Processes downscaled climate model outputs with variable standardization,
coordinate transformation, and multi-model ensemble support.
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


@DatasetRegistry.register("nex-gddp-cmip6")
@DatasetRegistry.register("nex-gddp")
class NEXGDDPCMIP6Handler(BaseDatasetHandler):
    """
    Handler for NEX-GDDP-CMIP6 downscaled climate data.

    Assumptions (align with typical NEX-GDDP-CMIP6 conventions):
      - Variables:
          pr      – precipitation flux [kg m-2 s-1]
          tas     – near-surface air temperature [K]
          huss    – near-surface specific humidity [1]
          ps      – surface air pressure [Pa] (may be absent in some configs)
          rlds    – surface downwelling longwave radiation [W m-2]
          rsds    – surface downwelling shortwave radiation [W m-2]
          sfcWind – near-surface wind speed [m/s]

      - Coordinates: lat, lon (1D or 2D), time, possibly an ensemble/realization
        dimension that we currently do not collapse; we just preserve it.

    This handler:
      - renames variables to SUMMA standard names
      - converts pr [kg m-2 s-1] → pptrate [m s-1]
      - ensures attributes are consistent
      - writes processed files into forcing/merged_path
      - builds a grid shapefile from lat/lon
    """

    # --------------------- Variable mapping ---------------------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map raw NEX-GDDP-CMIP6 variables to standard forcing names.

        Uses centralized VariableStandardizer for consistency across the codebase.
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('NEX-GDDP-CMIP6')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy()

        # Remove problematic missing_value attrs (already there)
        for vname, var in ds.variables.items():
            if "missing_value" in var.attrs:
                var.attrs.pop("missing_value", None)

        var_map = self.get_variable_mapping()
        rename_map = {src: tgt for src, tgt in var_map.items() if src in ds.data_vars}
        if rename_map:
            self.logger.debug(f"Renaming NEX-GDDP-CMIP6 variables: {rename_map}")
            ds = ds.rename(rename_map)

        # ---- Precipitation: pr -> pptrate [kg m-2 s-1] ----
        if "pptrate" in ds.data_vars:
            pr = ds["pptrate"].astype("float32")

            # clean NaNs / negatives
            pr = xr.where(np.isfinite(pr), pr, 0.0)
            pr = xr.where(pr < 0.0, 0.0, pr)

            pr.attrs.update(
                long_name="Total precipitation rate",
                units="kg m-2 s-1",    # ≡ mm/s
                standard_name="precipitation_flux",
            )
            ds["pptrate"] = pr

        # ---- Longwave radiation ----
        if "LWRadAtm" in ds.data_vars:
            lw = ds["LWRadAtm"].astype("float32")
            lw = xr.where(np.isfinite(lw), lw, np.nan)
            lw.attrs.update(
                long_name="Downwelling longwave radiation at surface",
                units="W m-2",
                standard_name="surface_downwelling_longwave_flux_in_air",
            )
            ds["LWRadAtm"] = lw

        # ---- Shortwave radiation ----
        if "SWRadAtm" in ds.data_vars:
            sw = ds["SWRadAtm"].astype("float32")
            sw = xr.where(np.isfinite(sw), sw, np.nan)
            sw.attrs.update(
                long_name="Downwelling shortwave radiation at surface",
                units="W m-2",
                standard_name="surface_downwelling_shortwave_flux_in_air",
            )
            ds["SWRadAtm"] = sw

        # ---- Air temperature ----
        if "airtemp" in ds.data_vars:
            ta = ds["airtemp"].astype("float32")
            ta = xr.where(np.isfinite(ta), ta, np.nan)
            ta.attrs.update(
                long_name="Near-surface air temperature",
                units="K",
                standard_name="air_temperature",
            )
            ds["airtemp"] = ta

        # ---- Specific humidity / relative humidity ----
        if "spechum" in ds.data_vars:
            q = ds["spechum"].astype("float32")
            q = xr.where(np.isfinite(q), q, np.nan)
            q.attrs.update(
                long_name="Near-Surface Specific Humidity",
                units="kg kg-1",
                standard_name="specific_humidity",
            )
            ds["spechum"] = q
        elif "hurs" in ds.data_vars:
            # Keep as relative humidity; downstream forcing processor will
            # convert to specific humidity via Magnus formula
            hurs = ds["hurs"].astype("float32")
            hurs = xr.where(np.isfinite(hurs), hurs, np.nan)
            hurs.attrs.update(
                long_name="Near-surface relative humidity",
                units="percent",
            )
            ds["relhum"] = hurs
            ds = ds.drop_vars("hurs")

        # ---- Wind speed ----
        if "windspd" in ds.data_vars:
            ws = ds["windspd"].astype("float32")
            ws = xr.where(np.isfinite(ws), ws, np.nan)
            ws.attrs.update(
                long_name="Near-surface wind speed",
                units="m s-1",
                standard_name="wind_speed",
            )
            ds["windspd"] = ws

        # ---- Air pressure (if present) ----
        if "airpres" in ds.data_vars:
            ap = ds["airpres"].astype("float32")
            ap = xr.where(np.isfinite(ap), ap, np.nan)
            ap.attrs.update(
                long_name="surface air pressure",
                units="Pa",
                standard_name="air_pressure",
            )
            ds["airpres"] = ap

        keep_vars = [
            v for v in [
                "pptrate", "LWRadAtm", "SWRadAtm",
                "airtemp", "spechum", "relhum", "windspd", "airpres",
            ]
            if v in ds.data_vars
        ]

        if not keep_vars:
            self.logger.warning(
                "NEX-GDDP-CMIP6 process_dataset produced no forcing variables; dataset will be empty."
            )
            return ds

        ds_out = ds[keep_vars]

        # preserve lat/lon coords
        for coord in ("lat", "lon"):
            if coord in ds.coords and coord not in ds_out.coords:
                ds_out = ds_out.assign_coords({coord: ds.coords[coord]})

        # Interpolate daily data to hourly if needed
        try:
            ds_out = self._interpolate_daily_to_hourly(ds_out)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to interpolate NEX-GDDP to hourly: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return ds_out

    def _interpolate_daily_to_hourly(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Convert daily NEX-GDDP-CMIP6 data to hourly resolution for SUMMA compatibility.

        Strategy:
        - Precipitation: Uniformly distribute daily total over 24 hours
        - Temperature, humidity, pressure, wind: Linear interpolation
        - Shortwave radiation: Apply diurnal cycle (zero at night, peak at solar noon)
        - Longwave radiation: Linear interpolation (less diurnal variation)
        """
        import pandas as pd

        if "time" not in ds.dims:
            self.logger.debug("No time dimension found; skipping temporal interpolation")
            return ds

        # Check if data is already sub-daily (hourly or finer)
        if len(ds.time) > 1:
            time_diff = (ds.time.values[1] - ds.time.values[0])
            timestep_seconds = time_diff / np.timedelta64(1, 's')

            if timestep_seconds < 86400:  # Already sub-daily (< 24 hours)
                self.logger.debug(f"Data already sub-daily (timestep={timestep_seconds}s); skipping interpolation")
                return ds

        self.logger.info("Converting NEX-GDDP-CMIP6 daily data to hourly resolution")

        # Create hourly time index
        # Start from midnight of the first day and end at midnight of the last day
        # This ensures SUMMA can run from 00:00
        start_time = pd.Timestamp(ds.time.values[0]).normalize()  # Start of first day (00:00)
        end_time = pd.Timestamp(ds.time.values[-1]).normalize() + pd.Timedelta(days=1)  # Start of day after last day
        hourly_times = pd.date_range(
            start=start_time,
            end=end_time,
            freq='h',  # Use 'h' instead of 'H' (deprecated)
            inclusive='left'  # Exclude the final endpoint (start of next day)
        )

        # Convert to numpy datetime64
        hourly_times_np = hourly_times.to_numpy()

        # Initialize output dataset
        ds_hourly = xr.Dataset(coords=ds.coords)
        ds_hourly['time'] = hourly_times_np

        self.logger.debug(f"Original timesteps: {len(ds.time)}, Hourly timesteps: {len(hourly_times_np)}")

        # Process each variable
        for var_name in ds.data_vars:
            if var_name not in ['pptrate', 'airtemp', 'spechum', 'relhum', 'windspd', 'airpres', 'LWRadAtm', 'SWRadAtm']:
                # Keep non-forcing variables as-is
                ds_hourly[var_name] = ds[var_name]
                continue

            var_data = ds[var_name]

            if var_name == 'pptrate':
                # Precipitation: Uniformly distribute daily total over 24 hours
                # Use nearest neighbor interpolation with extrapolation
                ds_hourly[var_name] = var_data.interp(
                    time=hourly_times_np,
                    method='nearest',
                    kwargs={'fill_value': 'extrapolate'}
                )
                self.logger.debug(f"  {var_name}: uniform distribution (nearest neighbor with extrapolation)")

            elif var_name == 'SWRadAtm':
                # Shortwave radiation: Apply simple diurnal cycle
                # Interpolate daily values with extrapolation, then apply diurnal pattern
                interp_sw = var_data.interp(
                    time=hourly_times_np,
                    method='linear',
                    kwargs={'fill_value': 'extrapolate'}
                )

                # Create diurnal cycle (simplified: sinusoidal, zero at night)
                hours = np.array([t.hour for t in pd.to_datetime(hourly_times_np)])
                # Simple diurnal cycle: peak at hour 12, zero from hour 18-6
                diurnal_factor = np.where(
                    (hours >= 6) & (hours <= 18),
                    np.sin((hours - 6) * np.pi / 12),  # Sine curve from 6am to 6pm
                    0.0
                )

                # Normalize so daily average matches original
                # Apply diurnal pattern (broadcast over spatial dims)
                sw_hourly = interp_sw.copy(deep=True)
                sw_hourly.values = sw_hourly.values * diurnal_factor.reshape(-1, *([1] * (sw_hourly.ndim - 1)))

                # Rescale to preserve daily mean
                # Group by day and rescale
                daily_groups = np.array([pd.Timestamp(t).date() for t in hourly_times_np])
                for day_val in np.unique(daily_groups):
                    day_mask = daily_groups == day_val
                    day_mean_original = interp_sw.isel(time=day_mask).mean(dim='time')
                    day_mean_diurnal = sw_hourly.isel(time=day_mask).mean(dim='time')
                    # Avoid division by zero
                    scale_factor = xr.where(day_mean_diurnal > 0, day_mean_original / day_mean_diurnal, 1.0)
                    sw_hourly.values[day_mask] = sw_hourly.values[day_mask] * scale_factor.values.reshape(1, *scale_factor.shape)

                ds_hourly[var_name] = sw_hourly
                self.logger.debug(f"  {var_name}: diurnal cycle applied")

            else:
                # All other variables: linear interpolation with extrapolation
                # Use kwargs to enable extrapolation for points outside the data range
                ds_hourly[var_name] = var_data.interp(
                    time=hourly_times_np,
                    method='linear',
                    kwargs={'fill_value': 'extrapolate'}
                )
                self.logger.debug(f"  {var_name}: linear interpolation with extrapolation")

            # Preserve attributes
            ds_hourly[var_name].attrs = var_data.attrs

        self.logger.info(f"Successfully interpolated to hourly: {len(ds.time)} daily → {len(ds_hourly.time)} hourly steps")

        return ds_hourly



    # --------------------- Coordinates -------------------------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        NEX-GDDP-CMIP6 NetCDF typically uses 'lat' and 'lon' for coordinates.
        """
        return ("lat", "lon")

    # --------------------- Merging / standardization ----------

    def needs_merging(self) -> bool:
        """
        As with AORC and CONUS404, we treat this as needing a ‘standardization’
        pass over the raw files, even if there is only one big file.
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
        Standardize NEX-GDDP-CMIP6 forcings:

        - Find NEX-GDDP-CMIP6 NetCDF files in raw_forcing_path
        - Apply process_dataset()
        - Save processed files into merged_forcing_path

        We do not (yet) merge across different models/scenarios/ensembles;
        each file is processed independently and later remapped.
        """
        self.logger.info("Standardizing NEX-GDDP-CMIP6 forcing files")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        # Support both the "domain + dataset" naming convention *and*
        # the NEXGDDP_all_YYYYMM.nc files produced by the downloader.
        patterns: List[str] = [
            f"{self.domain_name}_NEX-GDDP-CMIP6_*.nc",  # future / more explicit naming
            "NEXGDDP_all_*.nc",                         # current downloader output
            "*NEXGDDP*.nc",                             # any other NEXGDDP-style names
            "*NEX-GDDP-CMIP6*.nc",
            "*nex-gddp*.nc",
        ]

        files: List[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} NEX-GDDP-CMIP6 file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = (
                f"No NEX-GDDP-CMIP6 forcing files found in {raw_forcing_path} "
                f"with patterns {patterns}"
            )
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
                f"Skipped {skipped} NEX-GDDP-CMIP6 file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        if not files:
            self.logger.error(
                f"No NEX-GDDP-CMIP6 files match the configured period {start_year}-{end_year}"
            )
            raise FileNotFoundError(
                f"No NEX-GDDP-CMIP6 forcing files match the configured period "
                f"{start_year}-{end_year} in {raw_forcing_path}"
            )

        for f in files:
            self.logger.info(f"Processing NEX-GDDP-CMIP6 file: {f}")
            try:
                ds = self.open_dataset(f)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error opening NEX-GDDP-CMIP6 file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed NEX-GDDP-CMIP6 forcing: {out_name}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(
                    f"Error processing NEX-GDDP-CMIP6 dataset from {f}: {e}"
                )
            finally:
                ds.close()

        self.logger.info("NEX-GDDP-CMIP6 forcing standardization completed")


    # --------------------- Shapefile creation -----------------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """
        Create NEX-GDDP-CMIP6 grid shapefile from lat/lon coordinates.
        """
        self.logger.info("Creating NEX-GDDP-CMIP6 grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        nex_files = list(merged_forcing_path.glob("*.nc"))
        if not nex_files:
            raise FileNotFoundError(f"No NEX-GDDP-CMIP6 processed files found in {merged_forcing_path}")

        nex_file = nex_files[0]
        self.logger.info(f"Using NEX-GDDP-CMIP6 file for grid: {nex_file}")

        with self.open_dataset(nex_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in NEX-GDDP-CMIP6 file {nex_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in NEX-GDDP-CMIP6 file {nex_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
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
            self.logger.info(f"NEX-GDDP-CMIP6 grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

            cell_count = 0
            for i in range(ny):
                for j in range(nx):
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
                        self.logger.info(f"Created {cell_count}/{total_cells} NEX-GDDP-CMIP6 grid cells")

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "ID": ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): lons,
            },
            crs="EPSG:4326",
        )

        self.logger.info("Calculating elevation values for NEX-GDDP-CMIP6 grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf["elev_m"] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"NEX-GDDP-CMIP6 shapefile created at {output_shapefile}")

        return output_shapefile
