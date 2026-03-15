# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM Model Preprocessor

Handles preparation of mHM model inputs including:
- Forcing data conversion from ERA5 NetCDF to mHM grid format
- Namelist generation (mhm.nml, mrm.nml, mhm_parameter.nml, mhm_outputs.nml)
- Directory structure setup
- Grid and morphological input preparation
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.geospatial.geometry_utils import calculate_catchment_centroid
from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("MHM")
class MHMPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a mHM model run.

    mHM requires:
    - Meteorological forcing grids (NetCDF): precipitation, temperature, PET
    - Morphological data (DEM, soil, land cover grids)
    - Namelist files:
        - mhm.nml (main config: directories, time periods, process selection)
        - mrm.nml (routing config)
        - mhm_parameter.nml (parameter values, bounds, flags, scaling)
        - mhm_outputs.nml (output control: which fluxes/states to write)
    - Gauge data for calibration/validation
    """

    MODEL_NAME = "MHM"

    # Grid resolution for the 2-row lumped layout.  Two cells at this
    # size give approximately the same total area as a single 0.5° cell.
    LUMPED_CELLSIZE: float = 0.35

    def __init__(self, config, logger):
        """
        Initialize the mHM preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Use standard base-class paths:
        #   self.setup_dir   -> {project_dir}/settings/MHM   (namelists, morph, lcover, gauge)
        #   self.forcing_dir -> {project_dir}/data/forcing/MHM_input  (forcing NetCDFs)
        self.settings_dir = self.setup_dir
        self.morph_dir = self.setup_dir / "morph"
        self.lcover_dir = self.setup_dir / "lcover"
        self.gauge_dir = self.setup_dir / "gauge"
        self.output_mhm_dir = self.setup_dir / "output"

        # Resolve spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.mhm.spatial_mode,
            default=None,
            dict_key='MHM_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped',
                dict_key='DOMAIN_DEFINITION_METHOD'
            )
            if domain_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        logger.info(f"mHM spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        """
        Run the complete mHM preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting mHM preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate forcing files
            self._generate_forcing_files()

            # Generate morphological inputs
            self._generate_morph_files()

            # Generate lookup tables, gauge data, and latlon file
            self._generate_lookup_tables()
            self._generate_gauge_data()
            self._generate_latlon_file()

            # Generate namelist files
            self._generate_mhm_namelist()
            self._generate_mrm_namelist()
            self._generate_mhm_parameter_namelist()
            self._generate_mhm_outputs_namelist()

            logger.info("mHM preprocessing complete.")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"mHM preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create mHM input directory structure."""
        dirs = [
            self.settings_dir,
            self.forcing_dir,
            self.morph_dir,
            self.lcover_dir,
            self.gauge_dir,
            self.output_mhm_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created mHM input directories at {self.setup_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from configuration."""
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)

                # Get centroid using proper CRS-aware calculation
                lon, lat = calculate_catchment_centroid(gdf, logger=logger)

                # Project to UTM for accurate area
                utm_zone = int((lon + 180) / 6) + 1
                utm_epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
                gdf_proj = gdf.to_crs(f"EPSG:{utm_epsg}")
                area_m2 = gdf_proj.geometry.area.sum()

                # Get elevation if available
                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0

                return {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'elev': elev
                }
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read catchment properties: {e}")

        # Defaults
        return {
            'lat': 51.0,
            'lon': -115.0,
            'area_m2': 1e8,
            'elev': 1000.0
        }

    def _generate_forcing_files(self) -> None:
        """
        Generate mHM forcing files from ERA5 or basin-averaged data.

        mHM requires meteorological input as NetCDF grids:
        - Precipitation (pre): mm/day
        - Temperature (tavg): deg C
        - PET (pet): mm/day (or temperature for internal PET calculation)
        """
        logger.info("Generating mHM forcing files...")

        start_date, end_date = self._get_simulation_dates()

        # Try to load forcing data
        try:
            forcing_ds = self._load_forcing_data()
            self._write_mhm_forcing(forcing_ds, start_date, end_date)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not load forcing data: {e}, using synthetic")
            self._generate_synthetic_forcing(start_date, end_date)

    def _load_forcing_data(self) -> xr.Dataset:
        """Load basin-averaged forcing data."""
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            merged_path = self.project_forcing_dir / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path}")

        logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')
        except ValueError:
            try:
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
            except Exception:  # noqa: BLE001 — model execution resilience
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _write_mhm_forcing_nc(
        self,
        filepath: Path,
        varname: str,
        data_1d: np.ndarray,
        times: np.ndarray,
        props: Dict,
        long_name: str,
        units: str,
        cellsize: float = 0.35,  # kept for API compat; overridden by LUMPED_CELLSIZE
    ) -> None:
        """
        Write a single mHM-format forcing NetCDF file using netCDF4.

        mHM v5.13 expects forcing NetCDF files with:
        - Dimensions: time (unlimited), yc (nrows), xc (ncols)
        - Coordinate variables: xc(xc), yc(yc), lon(yc,xc), lat(yc,xc)
        - Data variable with fill_value=-9999.0
        - Time in 'days since YYYY-01-01 00:00:00', calendar 'standard'

        For a lumped setup the grid is 2 rows × 1 col (upstream + outlet)
        with identical forcing in both cells.
        """
        nrows, ncols = 2, 1
        cellsize = self.LUMPED_CELLSIZE
        xllcorner = props['lon'] - cellsize / 2
        yllcorner = props['lat'] - cellsize / 2 - cellsize  # 2 rows

        xc_vals = np.array([xllcorner + cellsize * (i + 0.5) for i in range(ncols)])
        yc_vals = np.array([yllcorner + cellsize * (nrows - 1 - i + 0.5) for i in range(nrows)])

        # Build time coordinate as days since start-of-year
        time_index = pd.DatetimeIndex(times)
        ref_year = time_index[0].year
        time_units = f'days since {ref_year}-01-01 00:00:00'
        time_num = nc4.date2num(time_index.to_pydatetime(), units=time_units, calendar='standard')

        # Replace NaN with fill value
        fill = -9999.0
        data_clean = np.where(np.isnan(data_1d), fill, data_1d)

        ds = nc4.Dataset(str(filepath), 'w', format='NETCDF4')
        try:
            ds.createDimension('time', None)  # unlimited
            ds.createDimension('yc', nrows)
            ds.createDimension('xc', ncols)

            # Time variable — use int32 to match mHM reference format;
            # mHM's get_time_vector_and_select reads into integer(i8).
            t_var = ds.createVariable('time', 'i4', ('time',))
            t_var.units = time_units
            t_var.calendar = 'standard'
            t_var[:] = np.round(time_num).astype(np.int32)

            # xc / yc coordinate variables
            xc_var = ds.createVariable('xc', 'f8', ('xc',))
            xc_var.units = 'degrees_east'
            xc_var[:] = xc_vals

            yc_var = ds.createVariable('yc', 'f8', ('yc',))
            yc_var.units = 'degrees_north'
            yc_var[:] = yc_vals

            # 2-D lon / lat (yc, xc) — matches mHM reference test data.
            # mHM's read_header_ascii swaps ncols↔nrows internally, so
            # Fortran getShape returns (xc, yc, time) and the shape check
            # passes with var_shape(1)==level2%nrows==ncols, var_shape(2)==level2%ncols==nrows.
            lon_2d = np.full((nrows, ncols), props['lon'])
            lat_rows = [props['lat'] - i * cellsize for i in range(nrows)]
            lat_2d = np.array(lat_rows).reshape(nrows, ncols)

            lon_var = ds.createVariable('lon', 'f8', ('yc', 'xc'))
            lon_var.units = 'degrees_east'
            lon_var[:] = lon_2d

            lat_var = ds.createVariable('lat', 'f8', ('yc', 'xc'))
            lat_var.units = 'degrees_north'
            lat_var[:] = lat_2d

            # Data variable — replicate 1-D time series to both cells.
            # Dimensions (time, yc, xc): matches mHM reference convention.
            # Fortran reverses to (xc, yc, time); getData reads with
            # cnt=(nRows_internal, nCols_internal, time_cnt) in Fortran order.
            data_var = ds.createVariable(varname, 'f8', ('time', 'yc', 'xc'), fill_value=fill)
            data_var.long_name = long_name
            data_var.units = units
            data_2d = np.asarray(data_clean)[:, np.newaxis, np.newaxis]
            data_var[:] = np.broadcast_to(data_2d, (len(data_clean), nrows, ncols)).copy()
        finally:
            ds.close()

    def _write_forcing_header(self, props: Dict) -> None:
        """
        Write header.txt in the forcing directory.

        mHM reads this ESRI ASCII grid header to determine the spatial
        extent of the forcing grid.  Must match the 2-row lumped layout.
        """
        cellsize = self.LUMPED_CELLSIZE
        xllcorner = props['lon'] - cellsize / 2
        yllcorner = props['lat'] - cellsize / 2 - cellsize  # 2 rows
        content = (
            f"ncols         1\n"
            f"nrows         2\n"
            f"xllcorner     {xllcorner:.6f}\n"
            f"yllcorner     {yllcorner:.6f}\n"
            f"cellsize      {cellsize}\n"
            f"NODATA_value  -9999\n"
        )
        header_path = self.forcing_dir / 'header.txt'
        header_path.write_text(content, encoding='utf-8')
        logger.info(f"Forcing header.txt written: {header_path}")

    def _write_mhm_forcing(
        self,
        forcing_ds: xr.Dataset,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Write forcing data in mHM NetCDF grid format."""
        props = self._get_catchment_properties()
        times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='D')

        # Map ERA5 variables to mHM variables
        precip_candidates = ['precipitation_flux', 'precipitation', 'pr', 'precip', 'tp', 'PREC']
        temp_candidates = ['air_temperature', 'temperature', 'tas', 'temp', 't2m', 'AIR_TEMP']

        # Extract precipitation
        precip = None
        for candidate in precip_candidates:
            if candidate in forcing_ds:
                precip = forcing_ds[candidate].values
                src_units = forcing_ds[candidate].attrs.get('units', '')
                # Convert to mm/day
                if 'mm/s' in src_units or candidate == 'precipitation_flux':
                    precip = precip * 86400.0
                    logger.info(f"Converted {candidate} from mm/s to mm/day")
                elif src_units == 'm' or candidate == 'tp':
                    precip = precip * 1000.0
                    logger.info(f"Converted {candidate} from m to mm")
                break

        if precip is None:
            logger.warning("No precipitation variable found, using zeros")
            precip = np.zeros(len(times))

        # Extract temperature
        temp = None
        for candidate in temp_candidates:
            if candidate in forcing_ds:
                temp = forcing_ds[candidate].values
                src_units = forcing_ds[candidate].attrs.get('units', '')
                if src_units == 'K' or np.nanmean(temp) > 100:
                    temp = temp - 273.15
                    logger.info(f"Converted {candidate} from K to deg C")
                break

        if temp is None:
            logger.warning("No temperature variable found, estimating")
            day_frac = np.arange(len(times)) / 365.0
            temp = 10 + 10 * np.sin(2 * np.pi * day_frac)

        # Flatten to 1D if needed
        if precip.ndim > 1:
            precip = precip.reshape(len(times), -1).mean(axis=1)
        if temp.ndim > 1:
            temp = temp.reshape(len(times), -1).mean(axis=1)

        # Aggregate sub-daily forcing to daily (mHM expects daily timesteps)
        time_index = pd.DatetimeIndex(times)
        if len(time_index) > 1 and (time_index[1] - time_index[0]) < pd.Timedelta('1D'):
            logger.info("Aggregating sub-daily forcing to daily for mHM...")
            precip_series = pd.Series(precip, index=time_index)
            temp_series = pd.Series(temp, index=time_index)
            # Precipitation: sum hourly values (already in mm/day rate, so average)
            precip_daily = precip_series.resample('D').mean()
            # Temperature: daily mean
            temp_daily = temp_series.resample('D').mean()
            precip = precip_daily.values
            temp = temp_daily.values
            times = precip_daily.index.values
            logger.info(f"Aggregated to {len(times)} daily timesteps")

        # Estimate PET using Hamon method from temperature
        doy = pd.DatetimeIndex(times).dayofyear
        lat_rad = np.radians(props['lat'])
        # Solar declination
        declination = 0.4093 * np.sin(2 * np.pi / 365 * doy - 1.405)
        # Day length in hours
        hour_angle = np.arccos(-np.tan(lat_rad) * np.tan(declination))
        day_length = 24.0 / np.pi * hour_angle
        # Saturated vapor pressure (hPa)
        es = 6.108 * np.exp(17.27 * temp / (temp + 237.3))
        # Hamon PET (mm/day)
        # D = day_length normalised to 12-hour units
        # RHOSAT = saturated absolute humidity (g/m³) via ideal gas approximation
        D = day_length / 12.0
        RHOSAT = 216.7 * es / (temp + 273.15)
        pet = 0.1651 * D * D * RHOSAT
        pet = np.maximum(pet, 0.0)

        # Write forcing NetCDF files in mHM-expected format
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pre.nc', 'pre', precip, times, props,
            'precipitation', 'mm/day')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'tavg.nc', 'tavg', temp, times, props,
            'average temperature', 'degC')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pet.nc', 'pet', pet, times, props,
            'potential evapotranspiration', 'mm/day')

        # Write header.txt for the forcing grid
        self._write_forcing_header(props)

        logger.info(f"mHM forcing files written to {self.forcing_dir}")

    def _generate_synthetic_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """Generate synthetic forcing data for testing."""
        props = self._get_catchment_properties()
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)

        day_frac = np.arange(n) / 365.0
        precip = np.random.exponential(2.0, n)
        temp = 10 + 10 * np.sin(2 * np.pi * day_frac)
        pet = np.maximum(0, 2 + 3 * np.sin(2 * np.pi * (day_frac - 0.25)))

        # Write forcing NetCDF files in mHM-expected format
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pre.nc', 'pre', precip, dates.values, props,
            'precipitation', 'mm/day')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'tavg.nc', 'tavg', temp, dates.values, props,
            'average temperature', 'degC')
        self._write_mhm_forcing_nc(
            self.forcing_dir / 'pet.nc', 'pet', pet, dates.values, props,
            'potential evapotranspiration', 'mm/day')

        # Write header.txt for the forcing grid
        self._write_forcing_header(props)

        logger.info(f"Synthetic mHM forcing files written to {self.forcing_dir}")

    def _write_asc_grid(self, filepath, value, props, cellsize=None,
                        value_row2=None):
        """
        Write a 2-row × 1-col ASCII grid file in ESRI .asc format.

        mRM needs ≥ 2 cells to create routing reaches.  For a lumped
        setup the grid has two rows (upstream cell → outlet cell) so
        that one reach exists and discharge can be routed to the gauge.

        Args:
            filepath: Path to the output .asc file.
            value: Data value for the upstream cell (row 1, printed first).
            props: Dict with 'lat' and 'lon' for the cell centre.
            cellsize: Grid cell size in degrees (default ``LUMPED_CELLSIZE``).
            value_row2: Optional different value for the outlet cell
                        (row 2).  Defaults to *value*.
        """
        cellsize = cellsize or self.LUMPED_CELLSIZE
        if value_row2 is None:
            value_row2 = value
        xllcorner = props['lon'] - cellsize / 2
        yllcorner = props['lat'] - cellsize / 2 - cellsize  # 2 rows
        content = (
            f"ncols         1\n"
            f"nrows         2\n"
            f"xllcorner     {xllcorner:.6f}\n"
            f"yllcorner     {yllcorner:.6f}\n"
            f"cellsize      {cellsize}\n"
            f"NODATA_value  -9999\n"
            f"{value}\n"        # row 1 (top / upstream)
            f"{value_row2}\n"   # row 2 (bottom / outlet)
        )
        Path(filepath).write_text(content, encoding='utf-8')

    def _generate_morph_files(self) -> None:
        """
        Generate morphological input files as ASCII grid (.asc) files.

        mHM v5.13 requires ESRI ASCII grid format for all morphological
        inputs. For a lumped 1x1 model each file contains a single cell.

        Generates in morph_dir:
            dem.asc, slope.asc, aspect.asc, soil_class.asc, facc.asc,
            fdir.asc, idgauges.asc, LAI_class.asc, geology_class.asc

        Generates in lcover_dir:
            lc_{start_year}.asc
        """
        logger.info("Generating mHM morphological inputs (ASCII grids)...")

        props = self._get_catchment_properties()
        start_date, _ = self._get_simulation_dates()

        # Remove any old .nc morph files (mHM v5.13 requires .asc only)
        for old_nc in self.morph_dir.glob('*.nc'):
            if old_nc.name != 'latlon.nc':
                old_nc.unlink()
                logger.info(f"Removed old NetCDF morph file: {old_nc.name}")

        # --- Morph directory grids (2-row lumped layout) ---
        # Row 1 = upstream cell, row 2 = outlet cell with gauge.
        # Both cells share the same properties so the model is effectively
        # lumped, but mRM can create one routing reach.
        self._write_asc_grid(self.morph_dir / 'dem.asc', props['elev'], props)
        self._write_asc_grid(self.morph_dir / 'slope.asc', 0.05, props)
        self._write_asc_grid(self.morph_dir / 'aspect.asc', 180.0, props)
        self._write_asc_grid(self.morph_dir / 'soil_class.asc', 1, props)
        # Flow accumulation: upstream=1, outlet=2 (collects from upstream)
        self._write_asc_grid(self.morph_dir / 'facc.asc', 1, props,
                             value_row2=2)
        # D8 flow direction: both cells flow South (4).
        # Row 1 → row 2 (within domain), row 2 → outside (outlet).
        self._write_asc_grid(self.morph_dir / 'fdir.asc', 4, props)
        # Gauge only on the outlet cell (row 2)
        self._write_asc_grid(self.morph_dir / 'idgauges.asc', -9999, props,
                             value_row2=1)
        self._write_asc_grid(self.morph_dir / 'LAI_class.asc', 1, props)
        self._write_asc_grid(self.morph_dir / 'geology_class.asc', 1, props)

        # --- Land cover directory grid ---
        # Class 3 = pervious (appropriate for lumped basins — forest class
        # over-extracts water via rootFractionCoefficient).
        self._write_asc_grid(
            self.lcover_dir / f'lc_{start_date.year}.asc', 3, props
        )

        logger.info(f"Morphological ASCII grids written to {self.morph_dir}")

    def _generate_mhm_namelist(self) -> None:
        """
        Generate the mhm.nml Fortran namelist file for mHM v5.13.

        Writes ALL required namelist blocks in the order expected by mHM:
        project_description, mainconfig, mainconfig_mhm_mrm,
        mainconfig_mrm, directories_general, directories_mHM,
        directories_mRM, optional_data, processSelection, LCover,
        time_periods, soildata, LAI_data_information, LCover_MPR,
        directories_MPR, evaluation_gauges, inflow_gauges, panEvapo,
        nightDayRatio, Optimization, baseflow_config.

        Note: Parameter values (bounds, defaults, flags) are written to the
        separate ``mhm_parameter.nml`` file by ``_generate_mhm_parameter_namelist``.
        Output control is handled by ``_generate_mhm_outputs_namelist``.
        """
        logger.info("Generating mhm.nml...")

        start_date, end_date = self._get_simulation_dates()
        self._get_catchment_properties()

        output_dir = str(self.output_mhm_dir)
        morph_dir = str(self.morph_dir)
        lcover_dir = str(self.lcover_dir)
        forcing_dir = str(self.forcing_dir)
        gauge_dir = str(self.gauge_dir)

        # Compute warmup days for mHM.
        # mHM warming_Days goes BACKWARD from eval_Per start, so we shift
        # eval_Per to start at the calibration period and set warming_Days
        # to the gap back to the experiment start (where forcing data begins).
        cal_period = self._get_config_value(
            lambda: self.config.domain.calibration_period, default='')
        if cal_period:
            cal_start_str = str(cal_period).split(',')[0].strip()
            cal_start = pd.to_datetime(cal_start_str)
            warmup_days = (cal_start - pd.to_datetime(start_date)).days
            eval_start = cal_start.to_pydatetime()
        else:
            warmup_days = 0
            eval_start = start_date

        namelist_content = f"""\
!-- mHM Namelist File (mHM v5.13) --
!-- Generated by SYMFLUENCE on {datetime.now().isoformat()} --

&project_description
  project_details = "SYMFLUENCE mHM model run"
  setup_description = "Lumped mHM simulation for {self.domain_name}"
  simulation_type = "historical simulation"
  Conventions = "CF-1.6"
  contact = "SYMFLUENCE"
  mHM_details = "mHM v5.13"
  history = "auto-generated"
/

&mainconfig
  iFlag_cordinate_sys = 1     ! lat/lon coordinates
  nDomains = 1
  resolution_Hydrology(1) = {self.LUMPED_CELLSIZE}   ! 2-cell lumped layout
  L0Domain(1) = 1
  write_restart = .FALSE.
  read_opt_domain_data(1) = 0
/

&mainconfig_mhm_mrm
  mhm_file_RestartIn(1) = ""
  mrm_file_RestartIn(1) = ""
  resolution_Routing(1) = {self.LUMPED_CELLSIZE}
  timestep = 1
  read_restart = .FALSE.
  optimize = .FALSE.
  optimize_restart = .FALSE.
  opti_method = 1
  opti_function = 9
/

&mainconfig_mrm
  ALMA_convention = .TRUE.
  varnametotalrunoff = 'total_runoff'
  filenametotalrunoff = 'total_runoff'
  gw_coupling = .false.
/

&directories_general
  dirConfigOut = "{output_dir}/"
  dirCommonFiles = "{morph_dir}/"
  dir_Morpho(1) = "{morph_dir}/"
  dir_LCover(1) = "{lcover_dir}/"
  mhm_file_RestartOut(1) = "{output_dir}/mHM_restart_001.nc"
  mrm_file_RestartOut(1) = "{output_dir}/mRM_restart_001.nc"
  dir_Out(1) = "{output_dir}/"
  file_LatLon(1) = "{morph_dir}/latlon.nc"
/

&directories_mHM
  inputFormat_meteo_forcings = "nc"
  bound_error = .FALSE.
  dir_Precipitation(1) = "{forcing_dir}/"
  dir_Temperature(1) = "{forcing_dir}/"
  dir_ReferenceET(1) = "{forcing_dir}/"
  dir_MinTemperature(1) = "{forcing_dir}/"
  dir_MaxTemperature(1) = "{forcing_dir}/"
  dir_NetRadiation(1) = "{forcing_dir}/"
  dir_absVapPressure(1) = "{forcing_dir}/"
  dir_windspeed(1) = "{forcing_dir}/"
  dir_Radiation(1) = "{forcing_dir}/"
  time_step_model_inputs(1) = 0
/

&directories_mRM
  dir_Gauges(1) = "{gauge_dir}/"
  dir_Total_Runoff(1) = "{output_dir}/"
  dir_Bankfull_Runoff(1) = "{output_dir}/"
/

&optional_data
  dir_soil_moisture(1) = "{output_dir}/"
  nSoilHorizons_sm_input = 1
  timeStep_sm_input = -2
  dir_neutrons(1) = "{output_dir}/"
  dir_evapotranspiration(1) = "{output_dir}/"
  timeStep_et_input = -2
  dir_tws(1) = "{output_dir}/"
  timeStep_tws_input = -2
/

&processSelection
  processCase(1)  = 1    ! Interception
  processCase(2)  = 1    ! Snowpack
  processCase(3)  = 1    ! Soil moisture
  processCase(4)  = 1    ! Direct runoff
  processCase(5)  = 0    ! PET (0 = PET is input)
  processCase(6)  = 1    ! Interflow
  processCase(7)  = 1    ! Percolation
  processCase(8)  = 1    ! Routing (Muskingum)
  processCase(9)  = 1    ! Baseflow
  processCase(10) = 0    ! Neutrons (off)
  processCase(11) = 0    ! River temperature (off)
/

&LCover
  nLCoverScene = 1
  LCoverYearStart(1) = {start_date.year}
  LCoverYearEnd(1) = {end_date.year}
  LCoverfName(1) = 'lc_{start_date.year}.asc'
/

&time_periods
  warming_Days(1) = {warmup_days}
  eval_Per(1)%yStart = {eval_start.year}
  eval_Per(1)%mStart = {eval_start.month}
  eval_Per(1)%dStart = {eval_start.day}
  eval_Per(1)%yEnd = {end_date.year}
  eval_Per(1)%mEnd = {end_date.month}
  eval_Per(1)%dEnd = {end_date.day}
/

&soildata
  iFlag_soilDB = 0
  tillageDepth = 200
  nSoilHorizons_mHM = 2
  soil_Depth(1) = 200
/

&LAI_data_information
  timeStep_LAI_input = 0
  inputFormat_gridded_LAI = "nc"
/

&LCover_MPR
  fracSealed_cityArea = 0.6
/

&directories_MPR
  dir_gridded_LAI(1) = "{morph_dir}/"
/

&evaluation_gauges
  nGaugesTotal = 1
  NoGauges_domain(1) = 1
  Gauge_id(1,1) = 1
  gauge_filename(1,1) = "gauge_001.txt"
/

&inflow_gauges
  nInflowGaugesTotal = 0
  NoInflowGauges_domain(1) = 0
  InflowGauge_id(1,1) = -9
  InflowGauge_filename(1,1) = ""
  InflowGauge_Headwater(1,1) = .FALSE.
/

&panEvapo
  evap_coeff = 1.30, 1.20, 0.72, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.50
/

&nightDayRatio
  read_meteo_weights = .FALSE.
  fnight_prec = 0.46, 0.50, 0.52, 0.51, 0.48, 0.50, 0.49, 0.48, 0.52, 0.56, 0.50, 0.47
  fnight_pet  = 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10
  fnight_temp = -0.76, -1.30, -1.88, -2.38, -2.72, -2.75, -2.74, -3.04, -2.44, -1.60, -0.94, -0.53
  fnight_ssrd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  fnight_strd = 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45
/

&Optimization
  nIterations = 7
  seed = 1235876
  dds_r = 0.2
  sa_temp = -9.0
  sce_ngs = 2
  sce_npg = -9
  sce_nps = -9
  mcmc_opti = .false.
  mcmc_error_params = 0.01, 0.6
/

&baseflow_config
  BFI_calc = .true.
/
"""
        namelist_path = self.settings_dir / 'mhm.nml'
        namelist_path.write_text(namelist_content, encoding='utf-8')
        logger.info(f"mhm.nml written: {namelist_path}")

    def _generate_lookup_tables(self) -> None:
        """
        Generate mHM lookup table text files in the morph directory.

        Creates:
            soil_classdefinition.txt  - Soil horizon properties
            LAI_classdefinition.txt   - Monthly LAI per land-use class
            geology_classdefinition.txt - Geology / aquifer properties
        """
        logger.info("Generating mHM lookup table files...")

        # --- soil_classdefinition.txt ---
        soil_content = (
            "nSoil_Types  1\n"
            "MU_GLOBAL\tHORIZON\tUD[mm]\tLD[mm]\tCLAY[%]\tSAND[%]\tBD[gcm-3]\n"
            "1\t1\t0\t200\t20.0\t50.0\t1.45\n"
            "1\t2\t200\t1500\t25.0\t45.0\t1.50\n"
        )
        (self.morph_dir / 'soil_classdefinition.txt').write_text(
            soil_content, encoding='utf-8'
        )

        # --- LAI_classdefinition.txt ---
        lai_content = (
            "NoLAIclasses           3\n"
            "ID   LAND-USE                  Jan.   Feb.    Mar.    Apr.    May"
            "    Jun.    Jul.    Aug.    Sep.    Oct.    Nov.    Dec.\n"
            " 1    Coniferous-forest         11     11      11      11      11"
            "     11      11      11      11      11      11      11\n"
            " 2    Deciduous-forest          0.5    0.5     1.5     4.0     7.0"
            "    11      12      12      11      8.0     1.5     0.5\n"
            " 3    Sealed-Water-bodies       0.01   0.01    0.02    0.04    0.06"
            "   0.09    0.09    0.07    0.06    0.04    0.02    0.02\n"
        )
        (self.morph_dir / 'LAI_classdefinition.txt').write_text(
            lai_content, encoding='utf-8'
        )

        # --- geology_classdefinition.txt ---
        geo_content = (
            "nGeo_Formations  1\n"
            "GeoParam(i)   ClassUnit     Karstic      Description\n"
            "         1            1           0      GeoUnit-1\n"
            "!<-END\n"
        )
        (self.morph_dir / 'geology_classdefinition.txt').write_text(
            geo_content, encoding='utf-8'
        )

        logger.info(f"Lookup tables written to {self.morph_dir}")

    def _generate_gauge_data(self) -> None:

        """
        Generate gauge observation data file for mHM.

        Attempts to load observed streamflow from the project using the
        ObservationLoaderMixin. If no observations are available, writes
        a file filled with the mHM missing-value marker (-9999).

        Output:
            gauge_dir / gauge_001.txt  (one daily discharge value per line, m3/s)
        """
        logger.info("Generating mHM gauge data file...")

        start_date, end_date = self._get_simulation_dates()
        n_days = (end_date - start_date).days + 1

        # Try to load real observations via the mixin
        values = None
        try:
            obs_series = self.load_streamflow_observations(
                output_format='series',
                target_units='cms',
                return_none_on_error=True
            )
            if obs_series is not None and len(obs_series) > 0:
                date_index = pd.date_range(start_date, end_date, freq='D')
                obs_reindexed = obs_series.reindex(date_index)
                values = obs_reindexed.values
                n_obs = int(obs_reindexed.notna().sum())
                logger.info(
                    f"Loaded {n_obs} observed streamflow values for gauge file"
                )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not load observed streamflow: {e}")

        if values is None:
            values = np.full(n_days, -9999.0)
            logger.info("No observations available; gauge file filled with -9999")

        # Replace NaN with missing-value marker
        values = np.where(np.isnan(values), -9999.0, values)

        # Build date range for data lines
        date_index = pd.date_range(start_date, end_date, freq='D')

        # Write gauge file with mHM 5-line header format
        header_lines = [
            f"# Gauge observation data for {self.domain_name}",
            "nodata_value  -9999",
            "measurements_per_day  1",
            f"start  {start_date.year} {start_date.month:02d} {start_date.day:02d}",
            f"end  {end_date.year} {end_date.month:02d} {end_date.day:02d}",
        ]
        data_lines = []
        for dt, v in zip(date_index, values):
            val_str = f"{v:.4f}" if v != -9999.0 else "-9999"
            data_lines.append(f"{dt.year} {dt.month:02d} {dt.day:02d} 00 00 {val_str}")

        gauge_path = self.gauge_dir / 'gauge_001.txt'
        gauge_path.write_text(
            "\n".join(header_lines + data_lines) + "\n", encoding='utf-8'
        )
        logger.info(f"Gauge data written: {gauge_path}")

    def _generate_latlon_file(self) -> None:
        """
        Generate the latlon.nc NetCDF file required by mHM.

        mHM reads latitude and longitude fields from a NetCDF file
        referenced by file_LatLon(1) in the namelist.  For the 2-row
        lumped layout this contains two cells (upstream + outlet).

        Variables written (all float64, dims (nrows, ncols)):
            lat, lon       -- cell-centre coordinates
            lat_l0, lon_l0 -- level-0 coordinates
            lat_l1, lon_l1 -- level-1 (hydrology) coordinates
            lat_l11, lon_l11 -- level-11 (routing) coordinates

        Output:
            morph_dir / latlon.nc
        """
        logger.info("Generating latlon.nc...")

        props = self._get_catchment_properties()
        cellsize = self.LUMPED_CELLSIZE
        lat_val = np.array(
            [[props['lat']], [props['lat'] - cellsize]], dtype=np.float64
        )
        lon_val = np.full((2, 1), props['lon'], dtype=np.float64)

        latlon_path = self.morph_dir / 'latlon.nc'
        ds = nc4.Dataset(str(latlon_path), 'w', format='NETCDF4')
        try:
            ds.createDimension('nrows', 2)
            ds.createDimension('ncols', 1)

            for vname, data in [
                ('lat', lat_val), ('lon', lon_val),
                ('lat_l0', lat_val), ('lon_l0', lon_val),
                ('lat_l1', lat_val), ('lon_l1', lon_val),
                ('lat_l11', lat_val), ('lon_l11', lon_val),
            ]:
                var = ds.createVariable(vname, 'f8', ('nrows', 'ncols'))
                if 'lat' in vname:
                    var.units = 'degrees_north'
                    var.long_name = 'latitude'
                else:
                    var.units = 'degrees_east'
                    var.long_name = 'longitude'
                var[:] = data
        finally:
            ds.close()

        logger.info(f"latlon.nc written: {latlon_path}")

    def _generate_mrm_namelist(self) -> None:
        """
        Generate the mrm.nml Fortran namelist file for mRM routing.

        This namelist controls the mRM (mesoscale Routing Model) routing
        component of mHM.
        """
        logger.info("Generating mrm.nml...")

        props = self._get_catchment_properties()

        routing_content = f"""!-- mRM Routing Namelist File --
!-- Generated by SYMFLUENCE on {datetime.now().isoformat()} --

&routing1
  gaugeID(1)          = 1
  gauge_lat(1)        = {props['lat'] - self.LUMPED_CELLSIZE:.6f}
  gauge_lon(1)        = {props['lon']:.6f}
/

&routing_general
  routing_model       = 1    ! Muskingum-Cunge routing
  timeStep_model_outputs_mrm = -2    ! daily
/
"""
        routing_path = self.settings_dir / 'mrm.nml'
        routing_path.write_text(routing_content, encoding='utf-8')
        logger.info(f"mrm.nml written: {routing_path}")

    def _generate_mhm_parameter_namelist(self) -> None:
        """
        Generate the ``mhm_parameter.nml`` Fortran namelist file.

        This file contains all mHM process parameters with their lower bound,
        upper bound, default value, optimisation flag, and scaling factor.
        The mHM binary reads this file at startup alongside ``mhm.nml`` and
        ``mrm.nml``.

        The format for each parameter line is::

            parameterName = lower, upper, value, FLAG, SCALING
        """
        logger.info("Generating mhm_parameter.nml...")

        from .parameters import MHM_PARAMETER_NML_SPEC

        lines = [
            "!> \\file mhm_parameter.nml",
            "!> \\brief Parameters for mHM.",
            f"!> \\details Generated by SYMFLUENCE on {datetime.now().isoformat()}",
            "!!",
            "!! PARAMETER = lower_bound, upper_bound, value, FLAG, SCALING",
            "!!   FLAG: 1 = parameter is optimised, 0 = parameter is fixed",
            "!!   SCALING: typically 1",
            "",
        ]

        for block_name, params in MHM_PARAMETER_NML_SPEC:
            lines.append(f"&{block_name}")
            for entry in params:
                pname, lower, upper, value, flag, scaling = entry
                # Format numbers consistently for Fortran readability
                lower_s = self._format_param_nml_value(lower)
                upper_s = self._format_param_nml_value(upper)
                value_s = self._format_param_nml_value(value)
                lines.append(
                    f"  {pname:45s} = {lower_s}, {upper_s}, {value_s}, {flag}, {scaling}"
                )
            lines.append("/")
            lines.append("")

        content = "\n".join(lines)
        param_nml_path = self.settings_dir / 'mhm_parameter.nml'
        param_nml_path.write_text(content, encoding='utf-8')
        logger.info(f"mhm_parameter.nml written: {param_nml_path}")

    def _generate_mhm_outputs_namelist(self) -> None:
        """
        Generate the ``mhm_outputs.nml`` Fortran namelist file.

        This file controls which mHM output variables are written and their
        format.  The ``outputFlxState`` logical array (21 elements) selects
        individual fluxes and states.  Element 11 (``total_runoff``) is enabled
        so that simulated discharge is available for evaluation.

        References:
            mHM source files ``mo_namelists.f90`` and ``mo_mhm_read_config.f90``.
        """
        logger.info("Generating mhm_outputs.nml...")

        # Build the outputFlxState array: only element 11 (total_runoff) is .true.
        n_elements = 21
        total_runoff_index = 11
        flx_lines = []
        for i in range(1, n_elements + 1):
            value = ".true." if i == total_runoff_index else ".false."
            flx_lines.append(f"  outputFlxState({i}) = {value}")

        namelist_content = (
            f"!-- mHM Output Namelist File --\n"
            f"!-- Generated by SYMFLUENCE on {datetime.now().isoformat()} --\n"
            f"\n"
            f"&nloutputresults\n"
            f"  output_deflate_level = 6\n"
            f"  output_double_precision = .true.\n"
            f"  timeStep_model_outputs = 0\n"
            f"  output_time_reference = 0\n"
            + "\n".join(flx_lines) + "\n"
            "/\n"
        )

        outputs_path = self.settings_dir / 'mhm_outputs.nml'
        outputs_path.write_text(namelist_content, encoding='utf-8')
        logger.info(f"mhm_outputs.nml written: {outputs_path}")

    @staticmethod
    def _format_param_nml_value(value: float) -> str:
        """
        Format a numeric value for the ``mhm_parameter.nml`` file.

        Uses fixed-point notation with 4 decimal places for most values,
        and scientific notation for very small numbers.

        Args:
            value: Numeric value to format.

        Returns:
            Formatted string suitable for Fortran namelist.
        """
        abs_val = abs(value)
        if abs_val == 0.0:
            return "0.0000"
        elif abs_val < 0.0001:
            return f"{value:.4E}"
        elif abs_val == int(abs_val) and abs_val < 1e6:
            return f"{value:.1f}"
        else:
            return f"{value:.4f}"

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
