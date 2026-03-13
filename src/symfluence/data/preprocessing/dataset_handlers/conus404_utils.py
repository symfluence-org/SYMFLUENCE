# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CONUS404 WRF reanalysis dataset handler.

Processes CONUS404 high-resolution atmospheric reanalysis data from
the HyTEST catalog with spatial subsetting and variable mapping.
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


@DatasetRegistry.register("conus404")
class CONUS404Handler(BaseDatasetHandler):
    """
    Handler for CONUS404 WRF reanalysis.

    Cloud downloader currently writes a domain file:
        <DOMAIN_NAME>_CONUS404_<startYear>-<endYear>.nc

    Variables (from cloud_downloader) typically include:
        T2       – 2m temperature [K]
        Q2       – 2m mixing ratio or specific humidity [kg/kg]
        PSFC     – surface pressure [Pa]
        U10,V10  – 10m wind components [m/s]
        GLW      – downward longwave radiation [W/m2]
        SWDOWN   – downward shortwave radiation [W/m2]
        RAINRATE – precipitation rate [mm/s or kg/m2/s, depending on source]

    This handler:
      - renames those to SYMFLUENCE/SUMMA standard names
      - derives wind speed magnitude
      - cleans attrs
      - writes processed files into forcing/merged_path
      - creates a grid shapefile from lat/lon
    """

    # --------------------- Variable mapping ---------------------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map raw CONUS404 variables to standard forcing names expected by SUMMA.

        Uses centralized VariableStandardizer for consistency across the codebase.
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('CONUS404')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CONUS404 WRF reanalysis with flexible variable detection and unit conversions.

        Transforms raw CONUS404 data from HyTEST catalog into SUMMA-compatible format.
        Handles multiple variable name conventions (accumulated vs instantaneous), derives
        wind speed, converts units, and cleans NetCDF attributes.

        Args:
            ds: Raw CONUS404 xarray Dataset with potential variables:
                Core meteorology:
                    - T2: 2m air temperature (K)
                    - Q2: 2m specific humidity (kg/kg)
                    - PSFC: Surface pressure (Pa)
                    - U10: U-component wind at 10m (m/s)
                    - V10: V-component wind at 10m (m/s)

                Radiation (accumulated OR instantaneous):
                    - ACSWDNB: Accumulated downward shortwave (W/m²)
                    - SWDOWN: Instantaneous downward shortwave (W/m²)
                    - ACLWDNB: Accumulated downward longwave (W/m²)
                    - LWDOWN: Instantaneous downward longwave (W/m²)
                    - GLW: Alias for longwave (W/m²)

                Precipitation (multiple formats):
                    - PREC_ACC_NC: Accumulated non-convective precipitation (mm)
                    - RAINRATE: Instantaneous rain rate (mm/s or kg/m²/s)
                    - PRATE: Precipitation rate (mm/s)
                    - ACDRIPR: Accumulated driving rain (mm)

        Returns:
            Processed xarray Dataset with SUMMA-compatible variables:
                - airtemp: Air temperature (K)
                - spechum: Specific humidity (kg/kg)
                - airpres: Surface pressure (Pa)
                - SWRadAtm: Shortwave radiation (W/m²)
                - LWRadAtm: Longwave radiation (W/m²)
                - pptrate: Precipitation rate (mm/s)
                - windspd: Wind speed magnitude (m/s)

        Processing Steps:
            1. **Core Variable Renaming**: T2/Q2/PSFC/U10/V10 → standard names
            2. **Shortwave Detection & Conversion**:
               - Priority: ACSWDNB (accumulated) → flux via _convert_accumulated_to_flux()
               - Fallback: SWDOWN (instantaneous) → use directly
            3. **Longwave Detection & Conversion**:
               - Priority: ACLWDNB (accumulated) → flux conversion
               - Fallback: LWDOWN or GLW (instantaneous)
            4. **Precipitation Detection & Conversion**:
               - Priority order: PREC_ACC_NC, RAINRATE, PRATE, ACDRIPR
               - Accumulated vars → rate conversion
               - Instantaneous vars → direct use or unit conversion
            5. **Wind Speed Derivation**: windspd = sqrt(U10² + V10²)
            6. **Attribute Cleaning**: Remove conflicting NetCDF attributes

        Variable Detection Strategy:
            Uses flexible fallback logic to handle different CONUS404 versions:
            1. Check for accumulated variables first (ACSWDNB, ACLWDNB, PREC_ACC_NC)
            2. If accumulated: convert to flux/rate using timestep
            3. If not found: check for instantaneous equivalents
            4. If neither: skip variable (not critical for all models)

        Accumulated-to-Flux Conversion:
            For radiation (SW/LW):
                - Accumulated W/m² over timestep dt
                - Conversion: flux = accumulated_value / dt
                - Result units: W/m² (average flux over period)

            For precipitation:
                - Accumulated mm over timestep dt (seconds)
                - Conversion: rate = accumulated_mm / dt
                - Result units: mm/s

        Wind Speed Derivation:
            Magnitude from WRF wind components:
            windspd = sqrt(U10² + V10²)

            Where:
                U10 = eastward wind component (m/s)
                V10 = northward wind component (m/s)

            Attributes set:
                units: 'm s-1'
                long_name: 'wind speed'
                standard_name: 'wind_speed'

        Precipitation Handling:
            Multiple CONUS404 precipitation formats handled:

            1. PREC_ACC_NC (accumulated non-convective):
               - Convert to rate: accum_mm / timestep_s
               - Units: mm → mm/s

            2. RAINRATE (instantaneous):
               - May be mm/s or kg/m²/s
               - Convert kg/m²/s to mm/s if needed

            3. PRATE (rate):
               - Typically already in mm/s
               - Use directly or convert units

            4. ACDRIPR (accumulated driving rain):
               - Convert to rate: accum_mm / timestep_s

        WRF Curvilinear Coordinates:
            - CONUS404 uses curvilinear lat/lon (2D arrays)
            - Coordinates preserved from HyTEST download
            - Projection: Lambert Conformal Conic
            - Grid spacing: ~4 km

        Example:
            >>> ds = xr.open_dataset('CONUS404_2015-2016.nc')
            >>> handler = CONUS404Handler(config, logger, project_dir)
            >>> ds_processed = handler.process_dataset(ds)
            >>> print(ds_processed.data_vars)
            # Variables: airtemp, spechum, airpres, SWRadAtm, LWRadAtm, pptrate, windspd
            >>> print(ds_processed['SWRadAtm'].attrs)
            # {'units': 'W m-2', 'long_name': 'downward shortwave radiation'}

        Notes:
            - Variable availability depends on HyTEST catalog version
            - Fallback logic ensures robustness to catalog changes
            - Radiation variables require accumulation period knowledge
            - Precipitation unit consistency critical for water balance
            - U10/V10 components retained alongside derived windspd

        See Also:
            - _convert_accumulated_to_flux(): Accumulation-to-flux conversion helper
            - data.utils.VariableStandardizer: Centralized variable mapping
            - data.preprocessing.dataset_handlers.base_dataset: Base handler
        """
        # --- Core met variables using VariableStandardizer ---
        standardizer = VariableStandardizer(self.logger)
        core_vars = {'T2', 'Q2', 'PSFC', 'U10', 'V10'}
        full_rename_map = standardizer.get_rename_map('CONUS404')
        rename_map = {old: new for old, new in full_rename_map.items()
                      if old in ds.data_vars and old in core_vars}
        ds = ds.rename(rename_map)

        # ============================
        # Quality control: detect and interpolate corrupt values in core variables
        # ============================
        # CONUS404 source data occasionally contains near-zero fill values
        # (e.g., airpres ≈ 0 Pa) that are physically impossible.
        # Replace these with time-interpolated values before further processing.
        qc_ranges = {
            'airpres': (20000.0, 120000.0),   # Pa - wide range to catch only clearly bad values
            'airtemp': (150.0, 370.0),         # K
            'spechum': (0.0, 0.2),             # kg/kg
        }
        for var, (qc_min, qc_max) in qc_ranges.items():
            if var not in ds:
                continue
            bad_mask = (ds[var] < qc_min) | (ds[var] > qc_max)
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                total = int(ds[var].size)
                self.logger.warning(
                    f"CONUS404 QC: {var} has {n_bad}/{total} values outside "
                    f"[{qc_min}, {qc_max}]. Replacing with time-interpolated values."
                )
                ds[var] = ds[var].where(~bad_mask).interpolate_na(
                    dim='time', method='linear', fill_value='extrapolate'
                )

        # ============================
        # Shortwave radiation → SWRadAtm
        # ============================
        # Handle accumulated radiation variables (may already be renamed by acquirer)
        if "ACSWDNB" in ds:
            # Not yet renamed - convert and assign
            sw_flux = self._convert_accumulated_to_flux(ds["ACSWDNB"])
            sw_flux.name = "SWRadAtm"
            ds["SWRadAtm"] = sw_flux
        elif "SWRadAtm" in ds:
            # Already renamed by acquirer - check if it needs conversion
            # Accumulated values are typically very large (>1e6)
            if float(ds["SWRadAtm"].mean()) > 1e6:
                self.logger.info("SWRadAtm appears to be accumulated - converting to flux")
                sw_flux = self._convert_accumulated_to_flux(ds["SWRadAtm"])
                ds["SWRadAtm"] = sw_flux
        elif "SWDOWN" in ds:
            # Instantaneous shortwave - just rename
            ds = ds.rename({"SWDOWN": "SWRadAtm"})

        if "SWRadAtm" in ds:
            ds["SWRadAtm"].attrs.update({
                "units": "W m-2",
                "long_name": "downward shortwave radiation at the surface",
            })


        # ============================
        # Longwave radiation → LWRadAtm
        # ============================
        if "ACLWDNB" in ds:
            # Not yet renamed - convert and assign
            lw_flux = self._convert_accumulated_to_flux(ds["ACLWDNB"])
            lw_flux.name = "LWRadAtm"
            ds["LWRadAtm"] = lw_flux
        elif "LWRadAtm" in ds:
            # Already renamed by acquirer - check if it needs conversion
            # Accumulated values are typically very large (>1e6)
            if float(ds["LWRadAtm"].mean()) > 1e6:
                self.logger.info("LWRadAtm appears to be accumulated - converting to flux")
                lw_flux = self._convert_accumulated_to_flux(ds["LWRadAtm"])
                ds["LWRadAtm"] = lw_flux
        elif "LWDOWN" in ds:
            # Instantaneous longwave - just rename
            ds = ds.rename({"LWDOWN": "LWRadAtm"})
        elif "GLW" in ds:
            # Alternative longwave name - just rename
            ds = ds.rename({"GLW": "LWRadAtm"})

        if "LWRadAtm" in ds:
            ds["LWRadAtm"].attrs.update({
                "units": "W m-2",
                "long_name": "downward longwave radiation at the surface",
            })

        # ============================
        # Precipitation rate → pptrate [kg m-2 s-1] (mm/s)
        # ============================
        if "ACDRIPR" in ds:
            pr_rate = self._convert_accumulated_to_flux(ds["ACDRIPR"])  # mm/s
            pr_rate.name = "pptrate"
            ds["pptrate"] = pr_rate

        elif "PREC_ACC_NC" in ds:
            # Accumulated total precip in mm
            pr_rate = self._convert_accumulated_to_flux(ds["PREC_ACC_NC"])  # mm/s
            pr_rate.name = "pptrate"
            ds["pptrate"] = pr_rate

        elif "RAINRATE" in ds:
            ds["pptrate"] = ds["RAINRATE"] # mm/s

        elif "pptrate" in ds:
            # Handle case where pptrate is already present but in mm (accumulated per step)
            attrs = ds["pptrate"].attrs
            units = attrs.get("units", "")
            desc = attrs.get("description", "").lower()
            long_name = attrs.get("long_name", "").lower()

            if units == "mm" and ("accumulated" in desc or "accumulated" in long_name):
                 self.logger.warning("Found 'pptrate' in mm (interval accumulated). Converting to rate mm/s.")
                 # Calculate dt
                 time_coord = "time"
                 dt = (ds[time_coord].diff(time_coord) / np.timedelta64(1, "s")).astype("float32")
                 dt = dt.reindex({time_coord: ds[time_coord]}, method="bfill")

                 # Convert mm/step -> mm/s
                 ds["pptrate"] = ds["pptrate"] / dt

        ds["pptrate"].attrs.update({
            "units": "kg m-2 s-1",  # SUMMA standard mass flux unit (equal to mm/s)
            "long_name": "precipitation rate",
            "standard_name": "precipitation_rate"
        })


        # ============================
        # Wind speed from components
        # ============================
        if "windspd_u" in ds and "windspd_v" in ds:
            u = ds["windspd_u"]
            v = ds["windspd_v"]
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = "windspd"
            windspd.attrs = {
                "units": "m s-1",
                "long_name": "wind speed",
                "standard_name": "wind_speed",
            }
            ds["windspd"] = windspd
        else:
            self.logger.error("Missing U10 and/or V10 for wind speed in CONUS404 dataset")
            raise KeyError("windspd")

        # ============================
        # Attributes for other core variables
        # ============================
        if "airtemp" in ds:
            ds["airtemp"].attrs.update(
                {
                    "units": "K",
                    "long_name": "air temperature",
                    "standard_name": "air_temperature",
                }
            )
        if "spechum" in ds:
            ds["spechum"].attrs.update(
                {
                    "units": "kg kg-1",
                    "long_name": "specific humidity",
                    "standard_name": "specific_humidity",
                }
            )
        if "airpres" in ds:
            ds["airpres"].attrs.update(
                {
                    "units": "Pa",
                    "long_name": "air pressure",
                    "standard_name": "air_pressure",
                }
            )

        # Common metadata + clean attributes
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "CONUS404 data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds

    def _convert_accumulated_to_flux(self, da, time_coord="time"):
        """
        Convert WRF accumulated variables to fluxes.
        Handles daily accumulation resets properly.
        """
        # Time delta in seconds
        dt = (da[time_coord].diff(time_coord) / np.timedelta64(1, "s")).astype("float32")

        # Accumulated difference
        dA = da.diff(time_coord)

        # Detect accumulation resets (values go backwards)
        reset = dA < 0

        # When reset occurs, the post-reset value IS the accumulated amount since reset
        # da.isel(time=slice(1,None)) aligns with dA (which lost first timestep from diff)
        dA = xr.where(reset, da.isel({time_coord: slice(1, None)}), dA)

        # Flux = deltaAccum / deltaTime
        flux = dA / dt

        # Ensure non-negative (numerical precision)
        flux = flux.clip(min=0)

        # Restore original time dimension length by padding the first timestep
        flux = flux.reindex({time_coord: da[time_coord]}, method="bfill")

        return flux



    # --------------------- Coordinates -------------------------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CONUS404 (HyTEST Zarr) exposes 2D lat/lon grids named 'lat' and 'lon'.
        """
        return ("lat", "lon")

    # --------------------- Merging / standardization ----------

    def needs_merging(self) -> bool:
        """
        We mark CONUS404 as needing 'merging' in the same sense as AORC:
        a standardization pass over raw cloud-downloaded files.
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
        Standardize CONUS404 forcings:

          - Look for domain-level CONUS404 NetCDF files in raw_forcing_path
          - Apply process_dataset()
          - Save processed files into merged_forcing_path
        """
        self.logger.info("Standardizing CONUS404 forcing files (no temporal merging)")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns: List[str] = [
            f"{self.domain_name}_CONUS404_*.nc",
            f"domain_{self.domain_name}_CONUS404_*.nc",
            "*CONUS404*.nc",
        ]

        files: List[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} CONUS404 file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No CONUS404 forcing files found in {raw_forcing_path} with patterns {patterns}"
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
                f"Skipped {skipped} CONUS404 file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        if not files:
            self.logger.error(
                f"No CONUS404 files match the configured period {start_year}-{end_year}"
            )
            raise FileNotFoundError(
                f"No CONUS404 forcing files match the configured period "
                f"{start_year}-{end_year} in {raw_forcing_path}"
            )

        for f in files:
            self.logger.info(f"Processing CONUS404 file: {f}")
            try:
                ds = self.open_dataset(f)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error opening CONUS404 file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)

                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed CONUS404 forcing: {out_name}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Error processing CONUS404 dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("CONUS404 forcing standardization completed")

    # --------------------- Shapefile creation -----------------

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
        """
        self.logger.info("Creating CONUS404 grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        # 🔧 Only use CONUS404 processed files, not ANY .nc
        patterns = [
            f"{self.domain_name}_CONUS404_*_processed.nc",
            f"domain_{self.domain_name}_CONUS404_*_processed.nc",
            "*CONUS404*_processed.nc",
        ]
        conus_files = []
        for pattern in patterns:
            candidates = sorted(merged_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} CONUS404 processed file(s) "
                    f"in {merged_forcing_path} with pattern '{pattern}'"
                )
                conus_files = candidates
                break

        if not conus_files:
            raise FileNotFoundError(
                f"No CONUS404 processed files found in {merged_forcing_path} "
                f"with patterns {patterns}"
            )

        conus_file = conus_files[0]
        self.logger.info(f"Using CONUS404 file for grid: {conus_file}")

        with self.open_dataset(conus_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in CONUS404 file {conus_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in CONUS404 file {conus_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        # Parse bounding box if available to filter grid
        bbox = None
        bbox_str = self._get_config_value(lambda: self.config.domain.bounding_box_coords, default=None)
        if isinstance(bbox_str, str) and "/" in bbox_str:
            try:
                # Format: lat_max/lon_min/lat_min/lon_max
                parts = [float(v) for v in bbox_str.split("/")]
                lat_min, lat_max = sorted([parts[0], parts[2]])
                lon_min, lon_max = sorted([parts[1], parts[3]])

                # Add a small buffer (approx 10km) to ensure we cover the domain
                # CONUS404 is ~4km resolution
                buffer = 0.1  # degrees
                lat_min -= buffer
                lat_max += buffer
                lon_min -= buffer
                lon_max += buffer

                bbox = (lat_min, lat_max, lon_min, lon_max)
                self.logger.info(f"Filtering CONUS404 grid by bbox (with buffer): {bbox}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to parse BOUNDING_BOX_COORDS '{bbox_str}': {e}. Processing entire grid.")

        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
            # Regular 1D lat/lon axes
            half_dlat = abs(lat[1] - lat[0]) / 2 if len(lat) > 1 else 0.005
            half_dlon = abs(lon[1] - lon[0]) / 2 if len(lon) > 1 else 0.005

            for i, center_lon in enumerate(lon):
                # Optimization: Skip longitudes outside bbox
                if bbox and not (bbox[2] <= float(center_lon) <= bbox[3]):
                    continue

                for j, center_lat in enumerate(lat):
                    # Optimization: Skip latitudes outside bbox
                    if bbox and not (bbox[0] <= float(center_lat) <= bbox[1]):
                        continue

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
            # 2D lat/lon grid
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(f"CONUS404 grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

            cell_count = 0
            filtered_count = 0

            # Pre-calculate approximate grid spacing to speed up loop
            # Check center of domain or use a default if small
            mid_y, mid_x = ny // 2, nx // 2
            if ny > 1 and nx > 1:
                default_dlat = abs(lat[mid_y + 1, mid_x] - lat[mid_y, mid_x]) if mid_y + 1 < ny else 0.04
                default_dlon = abs(lon[mid_y, mid_x + 1] - lon[mid_y, mid_x]) if mid_x + 1 < nx else 0.04
            else:
                default_dlat = 0.04
                default_dlon = 0.04

            for i in range(ny):
                for j in range(nx):
                    center_lat = float(lat[i, j])
                    center_lon = float(lon[i, j])

                    # Optimization: Skip cells outside bbox
                    if bbox and not (bbox[0] <= center_lat <= bbox[1] and bbox[2] <= center_lon <= bbox[3]):
                        filtered_count += 1
                        continue

                    # Robust local spacing calculation
                    # Try to use next neighbor, else previous neighbor, else default
                    if i < ny - 1:
                        dlat = abs(lat[i+1, j] - lat[i, j])
                    elif i > 0:
                        dlat = abs(lat[i, j] - lat[i-1, j])
                    else:
                        dlat = default_dlat

                    if j < nx - 1:
                        dlon = abs(lon[i, j+1] - lon[i, j])
                    elif j > 0:
                        dlon = abs(lon[i, j] - lon[i, j-1])
                    else:
                        dlon = default_dlon

                    # Ensure we have non-zero dimensions
                    if dlat < 1e-6: dlat = default_dlat
                    if dlon < 1e-6: dlon = default_dlon

                    half_dlat = dlat / 2.0
                    half_dlon = dlon / 2.0

                    # Create a rectangle centered on the point
                    # This avoids degenerate polygons at edges and is robust for zonal stats
                    verts = [
                        [center_lon - half_dlon, center_lat - half_dlat],
                        [center_lon - half_dlon, center_lat + half_dlat],
                        [center_lon + half_dlon, center_lat + half_dlat],
                        [center_lon + half_dlon, center_lat - half_dlat],
                        [center_lon - half_dlon, center_lat - half_dlat],
                    ]

                    geometries.append(Polygon(verts))
                    ids.append(i * nx + j)
                    lats.append(center_lat)
                    lons.append(center_lon)

                    cell_count += 1
                    if cell_count % 5000 == 0:
                        self.logger.info(f"Created {cell_count} geometries (filtered {filtered_count} so far)")

            self.logger.info(f"Finished grid processing. Created {len(geometries)} cells, skipped {filtered_count} cells.")

        if not geometries:
            msg = "No grid cells found within the specified bounding box! Check BOUNDING_BOX_COORDS."
            self.logger.error(msg)
            raise ValueError(msg)

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "ID": ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): lons,
            },
            crs="EPSG:4326",
        )

        self.logger.info("Calculating elevation values for CONUS404 grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf["elev_m"] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"CONUS404 shapefile created at {output_shapefile}")

        return output_shapefile
