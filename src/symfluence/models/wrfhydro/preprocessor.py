# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WRF-Hydro Model Preprocessor

Handles preparation of WRF-Hydro model inputs including:
- HRLDAS namelist (namelist.hrldas) for Noah-MP LSM
- Hydro namelist (hydro.namelist) for routing configuration
- Geogrid/wrfinput files (domain definition)
- Fulldom routing grid (channel network)
- Forcing files (LDASIN NetCDF)
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("WRFHYDRO")
class WRFHydroPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a WRF-Hydro model run.

    WRF-Hydro requires:
    - namelist.hrldas: HRLDAS/Noah-MP control file
    - hydro.namelist: Hydrological routing control file
    - wrfinput_d01.nc / geo_em.d01.nc: Domain definition files
    - Fulldom_hires.nc: High-resolution routing grid
    - LDASIN forcing files: Meteorological forcing in NetCDF format
    - Restart files (optional): For warm-start simulations
    """


    MODEL_NAME = "WRFHYDRO"
    def __init__(self, config, logger):
        """
        Initialize the WRF-Hydro preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # WRF-Hydro-specific subdirectories under the standard layout
        # setup_dir  = project_dir/settings/WRFHYDRO   (inherited)
        # forcing_dir = project_forcing_dir/WRFHYDRO_input (inherited)
        self.settings_dir = self.setup_dir
        self.routing_dir = self.setup_dir / "routing"
        self.restart_dir = self.setup_dir / "restart"

        # Resolve spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.wrfhydro.spatial_mode,
            default=None,
            dict_key='WRFHYDRO_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            self.spatial_mode = 'distributed'
        logger.info(f"WRF-Hydro spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        """
        Run the complete WRF-Hydro preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting WRF-Hydro preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate forcing files
            self._generate_forcing_files(start_date, end_date)

            # Generate domain/geogrid files
            self._generate_domain_files()

            # Generate routing files
            self._generate_routing_files()

            # Generate HRLDAS namelist
            self._generate_hrldas_namelist(start_date, end_date)

            # Generate hydro namelist
            self._generate_hydro_namelist()

            # Copy TBL lookup tables from WRF-Hydro install
            self._copy_tbl_files()

            logger.info("WRF-Hydro preprocessing completed successfully")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"WRF-Hydro preprocessing failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create WRF-Hydro input directory structure."""
        for d in [self.settings_dir, self.forcing_dir,
                  self.routing_dir, self.restart_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created WRF-Hydro directory structure (settings={self.settings_dir}, forcing={self.forcing_dir})")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from config."""
        start = self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01'
        )
        end = self._get_config_value(
            lambda: self.config.domain.time_end,
            default='2001-12-31'
        )
        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        if isinstance(end, str):
            end = pd.Timestamp(end).to_pydatetime()
        return start, end

    def _generate_forcing_files(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate WRF-Hydro LDASIN forcing files from ERA5 data.

        Writes aggregated monthly forcing files first (compact, fast), then
        unpacks to individual hourly LDASIN files as required by WRF-Hydro.
        Individual files are cached — re-running skips existing files.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        logger.info("Generating WRF-Hydro forcing files...")

        forcing_data = self._load_forcing_data()

        if forcing_data is not None:
            self._write_aggregated_forcing(start_date, end_date)
            self._unpack_forcing_to_ldasin()
        else:
            logger.warning("No forcing data found, generating synthetic forcing")
            self._generate_synthetic_forcing(start_date, end_date)

    def _load_forcing_data(self):
        """
        Check that basin-averaged ERA5 forcing data exists.

        Uses the same basin-averaged forcing data as all other models in the
        ensemble (from self.forcing_basin_path inherited from BaseModelPreProcessor).

        Returns:
            True if forcing data available, None otherwise.
            Actual data is processed file-by-file in _write_ldasin_files.
        """
        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            logger.warning(f"Forcing path does not exist: {forcing_path}")
            return None

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            logger.warning(f"No NetCDF files found in {forcing_path}")
            return None

        logger.info(f"Found ERA5 forcing in {forcing_path} ({len(forcing_files)} files)")
        return True  # Data processed file-by-file in _write_ldasin_files

    def _write_aggregated_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """
        Write aggregated monthly forcing NetCDF files from ERA5 data.

        Creates one file per month in forcing_dir/aggregated/ with all hourly
        timesteps stored along a time dimension.  This is ~100x faster than
        writing individual LDASIN files (96 monthly files vs 70k+ hourly).

        Maps ERA5 variables to WRF-Hydro LDASIN format:
          airtemp (K)      -> T2D (K)
          spechum (kg/kg)  -> Q2D (kg/kg)
          windspd (m/s)    -> U2D (m/s), V2D = 0
          airpres (Pa)     -> PSFC (Pa)
          pptrate (mm/s)   -> RAINRATE (mm/s = kg/m^2/s)
          SWRadAtm (W/m^2) -> SWDOWN (W/m^2)
          LWRadAtm (W/m^2) -> LWDOWN (W/m^2)
        """
        import xarray as xr
        from netCDF4 import Dataset as NC4Dataset

        aggregated_dir = self.forcing_dir / 'aggregated'
        aggregated_dir.mkdir(parents=True, exist_ok=True)

        var_map = {
            'air_temperature': ('T2D', 280.0),
            'specific_humidity': ('Q2D', 0.005),
            'wind_speed': ('U2D', 2.0),
            'surface_air_pressure': ('PSFC', 101325.0),
            'precipitation_flux': ('RAINRATE', 0.0),
            'surface_downwelling_shortwave_flux': ('SWDOWN', 0.0),
            'surface_downwelling_longwave_flux': ('LWDOWN', 300.0),
        }

        grid_shape = (3, 3)
        total_months = 0

        forcing_files = sorted(self.forcing_basin_path.glob("*.nc"))
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        for fpath in forcing_files:
            ds = xr.open_dataset(fpath).load()

            times = pd.DatetimeIndex(ds['time'].values)
            mask = (times >= start_ts) & (times <= end_ts)
            if not mask.any():
                ds.close()
                continue

            ds = ds.isel(time=mask)
            times = pd.DatetimeIndex(ds['time'].values)
            n_times = len(times)

            first_ts = pd.Timestamp(times[0])
            out_name = f"forcing_{first_ts.strftime('%Y%m')}.nc"
            out_path = aggregated_dir / out_name

            nc = NC4Dataset(str(out_path), 'w', format='NETCDF4')
            nc.TITLE = "WRF-Hydro aggregated forcing"

            nc.createDimension('time', n_times)
            nc.createDimension('south_north', grid_shape[0])
            nc.createDimension('west_east', grid_shape[1])

            # Time coordinate as hours since epoch
            time_var = nc.createVariable('time', 'f8', ('time',))
            time_var.units = 'hours since 2000-01-01 00:00:00'
            time_var.calendar = 'standard'
            epoch = pd.Timestamp('2000-01-01')
            time_var[:] = np.array(
                [(t - epoch).total_seconds() / 3600.0 for t in times]
            )

            for era5_var, (ldasin_var, default) in var_map.items():
                var = nc.createVariable(
                    ldasin_var, 'f4',
                    ('time', 'south_north', 'west_east'),
                    zlib=True
                )
                if era5_var in ds:
                    arr = ds[era5_var].values.squeeze()
                    for i in range(n_times):
                        val = float(arr[i]) if arr.ndim > 0 else float(arr)
                        if np.isnan(val):
                            val = default
                        var[i, :, :] = np.full(grid_shape, val, dtype=np.float32)
                else:
                    var[:] = np.full(
                        (n_times,) + grid_shape, default, dtype=np.float32
                    )

            v2d = nc.createVariable(
                'V2D', 'f4',
                ('time', 'south_north', 'west_east'),
                zlib=True
            )
            v2d[:] = np.zeros((n_times,) + grid_shape, dtype=np.float32)

            nc.close()
            ds.close()
            total_months += 1
            logger.info(f"  Wrote aggregated forcing: {out_name} ({n_times} timesteps)")

        logger.info(f"Generated {total_months} aggregated monthly forcing files")

    def _unpack_forcing_to_ldasin(self) -> int:
        """
        Unpack aggregated monthly forcing to individual LDASIN files.

        Reads from forcing_dir/aggregated/ and writes hourly files to
        forcing_dir/ as YYYYMMDDHH.LDASIN_DOMAIN1 (required by WRF-Hydro).
        Existing files are skipped (cached), so re-runs are fast.

        Returns:
            Number of new LDASIN files created.
        """
        from netCDF4 import Dataset as NC4Dataset
        from netCDF4 import num2date

        aggregated_dir = self.forcing_dir / 'aggregated'
        if not aggregated_dir.exists():
            logger.warning("No aggregated forcing directory found")
            return 0

        aggregated_files = sorted(aggregated_dir.glob("forcing_*.nc"))
        if not aggregated_files:
            logger.warning("No aggregated forcing files found")
            return 0

        grid_shape = (3, 3)
        ldasin_vars = [
            'T2D', 'Q2D', 'U2D', 'V2D', 'PSFC', 'RAINRATE', 'SWDOWN', 'LWDOWN'
        ]

        created = 0
        skipped = 0

        for agg_path in aggregated_files:
            agg = NC4Dataset(str(agg_path), 'r')

            time_var = agg.variables['time']
            times = num2date(time_var[:], time_var.units, time_var.calendar)

            # Bulk-read all variable data for this month
            var_data = {}
            for vname in ldasin_vars:
                if vname in agg.variables:
                    var_data[vname] = agg.variables[vname][:]

            for i, t in enumerate(times):
                t_pd = pd.Timestamp(t.isoformat())
                fname = t_pd.strftime('%Y%m%d%H') + '.LDASIN_DOMAIN1'
                out_path = self.forcing_dir / fname

                if out_path.exists():
                    skipped += 1
                    continue

                nc = NC4Dataset(str(out_path), 'w', format='NETCDF4')
                nc.TITLE = "OUTPUT FROM HRLDAS v20110427"
                nc.createDimension('south_north', grid_shape[0])
                nc.createDimension('west_east', grid_shape[1])

                for vname in ldasin_vars:
                    var = nc.createVariable(
                        vname, 'f4', ('south_north', 'west_east')
                    )
                    if vname in var_data:
                        var[:] = var_data[vname][i]
                    else:
                        var[:] = np.zeros(grid_shape, dtype=np.float32)

                nc.close()
                created += 1

            agg.close()

        logger.info(
            f"Unpacked LDASIN files: {created} created, {skipped} already cached"
        )
        return created

    def _generate_synthetic_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """Raise error — synthetic forcing should never be used."""
        raise FileNotFoundError(
            f"No basin-averaged ERA5 forcing data found in {self.forcing_basin_path}. "
            "WRF-Hydro requires real forcing data. Ensure the domain has been set up "
            "with forcing data before running preprocessing."
        )

    def _generate_domain_files(self) -> None:
        """
        Generate wrfinput_d01.nc domain file for WRF-Hydro/Noah-MP.

        Critical requirements discovered through debugging:
        - TITLE must be "OUTPUT FROM HRLDAS v20110427" for version detection
        - All 2D variables must have Time dimension: (Time, south_north, west_east)
        - Soil variables must have dims: (Time, soil_layers_stag, south_north, west_east)
        - SEAICE must be included (=0.0) or uninitialized memory causes sea-ice
          skip path to bypass all physics (SH2O→1.0, LAI→0.01, energy→-9999)
        - MAPFAC_MX/MY must be present (=1.0) or map factor warnings
        - SH2O (liquid soil moisture) must be present alongside SMOIS
        - NUM_LAND_CAT must be 20 (MODIFIED_IGBP_MODIS_NOAH)
        """
        logger.info("Generating WRF-Hydro domain files...")

        from netCDF4 import Dataset as NC4Dataset

        # Bow at Banff basin parameters
        lat = 51.17
        lon = -115.57
        elev = 1500.0
        dx = 1000.0  # Grid spacing in meters (3x3 lumped grid)

        wrfinput_path = self.settings_dir / 'wrfinput_d01.nc'
        ds = NC4Dataset(str(wrfinput_path), 'w', format='NETCDF4')

        ny, nx, nsoil = 3, 3, 4

        # Global attributes required by WRF-Hydro
        # TITLE must match "HRLDAS v20110427" for HRLDAS version detection
        ds.TITLE = "OUTPUT FROM HRLDAS v20110427"
        ds.DX = dx
        ds.DY = dx
        ds.TRUELAT1 = lat
        ds.TRUELAT2 = lat
        ds.STAND_LON = lon
        ds.CEN_LAT = lat
        ds.CEN_LON = lon
        ds.MOAD_CEN_LAT = lat
        ds.MAP_PROJ = 1  # Lambert Conformal
        ds.ISWATER = 17
        ds.ISLAKE = 21
        ds.ISICE = 15
        ds.ISURBAN = 13
        ds.ISOILWATER = 14
        ds.GRID_ID = 1
        ds.MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
        ds.NUM_LAND_CAT = 20
        ds.WEST_EAST_GRID_DIMENSION = nx + 1
        ds.SOUTH_NORTH_GRID_DIMENSION = ny + 1

        # Dimensions — Time must be unlimited for HRLDAS I/O
        ds.createDimension('Time', None)
        ds.createDimension('south_north', ny)
        ds.createDimension('west_east', nx)
        ds.createDimension('soil_layers_stag', nsoil)
        ds.createDimension('DateStrLen', 19)

        # Helper to create a 2D variable with Time dimension
        def create_2d(name, dtype, values):
            v = ds.createVariable(name, dtype, ('Time', 'south_north', 'west_east'))
            v[0, :, :] = np.full((ny, nx), values, dtype=dtype)
            return v

        # Helper to create a soil variable with Time dimension
        def create_soil(name, dtype, values):
            v = ds.createVariable(name, dtype,
                                  ('Time', 'soil_layers_stag', 'south_north', 'west_east'))
            if hasattr(values, '__len__'):
                for k in range(nsoil):
                    v[0, k, :, :] = np.full((ny, nx), values[k], dtype=dtype)
            else:
                v[0, :, :, :] = np.full((nsoil, ny, nx), values, dtype=dtype)
            return v

        # Terrain and coordinates
        create_2d('HGT', 'f4', elev)
        create_2d('XLAT', 'f4', lat)
        create_2d('XLONG', 'f4', lon)
        create_2d('HGT_M', 'f4', elev)
        create_2d('XLAT_M', 'f4', lat)
        create_2d('XLONG_M', 'f4', lon)

        # Land use and soil type (integer)
        create_2d('IVGTYP', 'i4', 1)    # 1=Evergreen Needleleaf Forest
        create_2d('ISLTYP', 'i4', 4)    # 4=Silt Loam

        # Deep soil temperature
        create_2d('TMN', 'f4', 275.0)

        # Vegetation parameters
        create_2d('LAI', 'f4', 4.0)
        create_2d('VEGFRA', 'f4', 60.0)
        create_2d('SHDMIN', 'f4', 10.0)
        create_2d('SHDMAX', 'f4', 80.0)

        # Surface state
        create_2d('TSK', 'f4', 260.0)     # Skin temperature (K)
        create_2d('CANWAT', 'f4', 0.0)    # Canopy water (kg/m2)
        create_2d('SNOW', 'f4', 50.0)     # Snow water equivalent (mm)
        create_2d('SNOWH', 'f4', 0.25)    # Snow depth (m)

        # Land mask and sea ice
        create_2d('XLAND', 'f4', 1.0)     # 1=land, 2=water
        create_2d('SEAICE', 'f4', 0.0)    # CRITICAL: must be 0.0 for land points

        # Map factors (1.0 for small domain)
        create_2d('MAPFAC_MX', 'f4', 1.0)
        create_2d('MAPFAC_MY', 'f4', 1.0)

        # Soil variables — 4 layers
        create_soil('SMOIS', 'f4', [0.30, 0.32, 0.35, 0.38])   # Total soil moisture
        create_soil('SH2O', 'f4', [0.20, 0.25, 0.30, 0.35])    # Liquid soil moisture
        create_soil('TSLB', 'f4', [265.0, 270.0, 273.0, 275.0]) # Soil temperature

        # Soil layer thicknesses
        dzs = ds.createVariable('DZS', 'f4', ('Time', 'soil_layers_stag'))
        dzs[0, :] = np.array([0.10, 0.30, 0.60, 1.00], dtype=np.float32)

        ds.close()
        logger.info("Generated wrfinput_d01.nc domain file")

    def _generate_routing_files(self) -> None:
        """Generate Fulldom routing grid and channel routing files."""
        logger.info("Generating WRF-Hydro routing files...")

        from netCDF4 import Dataset as NC4Dataset

        # Fulldom_hires.nc (routing grid - same resolution as LSM for lumped)
        fulldom_path = self.routing_dir / 'Fulldom_hires.nc'
        ds = NC4Dataset(str(fulldom_path), 'w', format='NETCDF4')
        ds.DX = 1000.0
        ds.DY = 1000.0

        ds.createDimension('y', 3)
        ds.createDimension('x', 3)

        topo = ds.createVariable('TOPOGRAPHY', 'f4', ('y', 'x'))
        topo[:] = np.full((3, 3), 1500.0)

        flowdir = ds.createVariable('FLOWDIRECTION', 'i4', ('y', 'x'))
        # Flow direction: center cell flows out (value=0 for outlet)
        flowdir[:] = np.array([[4, 8, 8], [4, 0, 8], [2, 2, 4]], dtype=np.int32)

        chgrid = ds.createVariable('CHANNELGRID', 'i4', ('y', 'x'))
        chgrid[:] = np.full((3, 3), -1, dtype=np.int32)
        chgrid[1, 1] = 0  # Channel at center cell

        strmord = ds.createVariable('STREAMORDER', 'i4', ('y', 'x'))
        strmord[:] = np.full((3, 3), -1, dtype=np.int32)
        strmord[1, 1] = 1

        lakegrid = ds.createVariable('LAKEGRID', 'i4', ('y', 'x'))
        lakegrid[:] = np.full((3, 3), -1, dtype=np.int32)

        lat = ds.createVariable('LATITUDE', 'f4', ('y', 'x'))
        lat[:] = np.full((3, 3), 51.17)

        lon = ds.createVariable('LONGITUDE', 'f4', ('y', 'x'))
        lon[:] = np.full((3, 3), -115.57)

        ds.close()

        # Route_Link.nc (channel routing parameters)
        rl_path = self.routing_dir / 'Route_Link.nc'
        ds = NC4Dataset(str(rl_path), 'w', format='NETCDF4')

        ds.createDimension('feature_id', 1)

        link = ds.createVariable('link', 'i4', ('feature_id',))
        link[:] = [1]

        frm = ds.createVariable('from', 'i4', ('feature_id',))
        frm[:] = [0]

        to = ds.createVariable('to', 'i4', ('feature_id',))
        to[:] = [0]

        length = ds.createVariable('Length', 'f4', ('feature_id',))
        length[:] = [1000.0]

        n = ds.createVariable('n', 'f4', ('feature_id',))
        n[:] = [0.035]

        chslp = ds.createVariable('ChSlp', 'f4', ('feature_id',))
        chslp[:] = [0.01]

        btmwdth = ds.createVariable('BtmWdth', 'f4', ('feature_id',))
        btmwdth[:] = [5.0]

        ds.close()
        logger.info("Generated Fulldom_hires.nc and Route_Link.nc routing files")

    def _generate_hrldas_namelist(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate the HRLDAS namelist (namelist.hrldas).

        This controls the Noah-MP land surface model component.
        """
        namelist_file = self._get_config_value(
            lambda: self.config.model.wrfhydro.namelist_file,
            default='namelist.hrldas'
        )

        restart_freq = self._get_config_value(
            lambda: self.config.model.wrfhydro.restart_frequency,
            default='monthly'
        )

        # Map restart frequency to output steps
        restart_minutes = {'hourly': 60, 'daily': 1440, 'monthly': 43200}.get(
            restart_freq, 43200
        )

        content = f"""&NOAHLSM_OFFLINE

 HRLDAS_SETUP_FILE = '{self.settings_dir}/wrfinput_d01.nc'
 INDIR = '{self.forcing_dir}'
 OUTDIR = '{self.project_dir / "simulations"}'

 START_YEAR  = {start_date.year}
 START_MONTH = {start_date.month:02d}
 START_DAY   = {start_date.day:02d}
 START_HOUR  = {max(start_date.hour, 1):02d}
 START_MIN   = 00

 ! Simulation length in hours
 KHOUR = {int((end_date - start_date).total_seconds() // 3600)}

 ! Physics options (simple/robust defaults for standalone HRLDAS)
 DYNAMIC_VEG_OPTION                = 1
 CANOPY_STOMATAL_RESISTANCE_OPTION = 1
 BTR_OPTION                        = 1
 RUNOFF_OPTION                     = 1
 SURFACE_DRAG_OPTION               = 1
 SUPERCOOLED_WATER_OPTION          = 1
 FROZEN_SOIL_OPTION                = 1
 RADIATIVE_TRANSFER_OPTION         = 1
 SNOW_ALBEDO_OPTION                = 1
 PCP_PARTITION_OPTION              = 1
 TBOT_OPTION                       = 2
 TEMP_TIME_SCHEME_OPTION           = 1
 GLACIER_OPTION                    = 1
 SURFACE_RESISTANCE_OPTION         = 1

 ! Output — daily is sufficient for streamflow evaluation and
 ! reduces output from ~70k files to ~2900 over a typical run
 OUTPUT_TIMESTEP = 86400
 RESTART_FREQUENCY_HOURS = {restart_minutes // 60}
 SPLIT_OUTPUT_COUNT = 1

 ! Forcing
 FORCING_TIMESTEP = 3600
 NOAH_TIMESTEP    = 3600

 ! Soil layers
 NSOIL = 4
 soil_thick_input(1) = 0.10
 soil_thick_input(2) = 0.30
 soil_thick_input(3) = 0.60
 soil_thick_input(4) = 1.00

 ZLVL = 10.0

/

&WRF_HYDRO_OFFLINE
 finemesh        = 0
 finemesh_factor = 1
 forc_typ        = 1
 snow_assim      = 0
/
"""
        out_path = self.settings_dir / namelist_file
        out_path.write_text(content)
        logger.info(f"Generated HRLDAS namelist: {out_path}")

    def _generate_hydro_namelist(self) -> None:
        """
        Generate the hydro namelist (hydro.namelist).

        This controls the hydrological routing component.
        """
        hydro_namelist_file = self._get_config_value(
            lambda: self.config.model.wrfhydro.hydro_namelist,
            default='hydro.namelist'
        )

        content = f"""&HYDRO_nlist

 ! System coupling
 sys_cpl = 1
 IGRID = 1

 ! Routing: disabled for lumped basin
 CHANRTSWCRT    = 0
 channel_option = 0
 SUBRTSWCRT     = 0
 OVRTSWCRT      = 0
 GWBASESWCRT    = 0

 ! Routing grid parameters
 AGGFACTRT = 1
 dtrt_ter  = 10
 dtrt_ch   = 10
 dxrt      = 1000.0
 NSOIL     = 4
 ZSOIL8(1) = -0.10
 ZSOIL8(2) = -0.40
 ZSOIL8(3) = -1.00
 ZSOIL8(4) = -2.00

 ! File paths
 GEO_STATIC_FLNM  = '{self.settings_dir}/wrfinput_d01.nc'
 GEO_FINEGRID_FLNM = '{self.routing_dir}/Fulldom_hires.nc'
 route_link_f      = '{self.routing_dir}/Route_Link.nc'

 ! Output control — daily output (matches HRLDAS OUTPUT_TIMESTEP)
 SPLIT_OUTPUT_COUNT = 1
 out_dt             = 1440
 rst_dt             = 1440
 rst_typ            = 1
 rst_bi_in          = 0
 rst_bi_out         = 0
 RSTRT_SWC          = 0
 GW_RESTART         = 0
 order_to_write     = 1
 io_form_outputs    = 0
 io_config_outputs  = 0
 t0OutputFlag       = 1
 output_channelBucket_influx = 0
 TERADJ_SOLAR       = 0
 bucket_loss        = 0
 UDMP_OPT           = 0
 imperv_adj         = 0

 ! Output switches
 CHRTOUT_DOMAIN     = 0
 CHANOBS_DOMAIN     = 0
 CHRTOUT_GRID       = 0
 LSMOUT_DOMAIN      = 1
 RTOUT_DOMAIN       = 0
 output_gw          = 0
 outlake            = 0
 frxst_pts_out      = 0
 GW_RESTART  = 0

/

&NUDGING_nlist
 nudgingParamFile = ''
 netwkReExFile    = ''
/
"""
        out_path = self.settings_dir / hydro_namelist_file
        out_path.write_text(content)
        logger.info(f"Generated hydro namelist: {out_path}")

    def _copy_tbl_files(self) -> None:
        """
        Copy .TBL lookup tables from the WRF-Hydro install to settings_dir.

        WRF-Hydro requires MPTABLE.TBL, SOILPARM.TBL, VEGPARM.TBL,
        GENPARM.TBL, etc. in the run directory.  The runner symlinks them
        from settings_dir at runtime, so they must be staged here.

        Search order for TBL source directory:
        1. {install}/Run or {install}/run (standard WRF build layout)
        2. {exe_parent} (bin/ directory)
        3. Recursive search under {install}/src for parameter_tables
        """
        import shutil

        install_path = self._get_config_value(
            lambda: self.config.model.wrfhydro.install_path,
            default=None,
            dict_key='WRFHYDRO_INSTALL_PATH'
        )
        if not install_path or install_path == 'default':
            install_path = self.data_dir / 'installs' / 'wrfhydro'
        else:
            install_path = Path(install_path)

        tbl_source_dir = None

        # 1. Standard WRF build layout
        for subdir in ['Run', 'run']:
            candidate = install_path / subdir
            if candidate.exists() and list(candidate.glob('*.TBL')):
                tbl_source_dir = candidate
                break

        # 2. Alongside the executable
        if tbl_source_dir is None:
            bin_dir = install_path / 'bin'
            if bin_dir.exists() and list(bin_dir.glob('*.TBL')):
                tbl_source_dir = bin_dir

        # 3. Recursive search — find MPTABLE.TBL, prefer CONUS, then latest version
        if tbl_source_dir is None:
            mptable_hits = sorted(install_path.rglob('MPTABLE.TBL'))
            # Prefer CONUS variant, then highest version directory
            conus_hits = [p for p in mptable_hits if 'CONUS' in str(p)]
            if conus_hits:
                tbl_source_dir = conus_hits[-1].parent
            elif mptable_hits:
                tbl_source_dir = mptable_hits[-1].parent

        if tbl_source_dir is None:
            logger.warning(
                f"No .TBL files found under {install_path}. "
                "WRF-Hydro will fail without MPTABLE.TBL and other lookup tables."
            )
            return

        copied = 0
        for tbl_file in tbl_source_dir.glob('*.TBL'):
            dest = self.settings_dir / tbl_file.name
            if not dest.exists():
                shutil.copy2(tbl_file, dest)
                copied += 1

        logger.info(f"Copied {copied} TBL files from {tbl_source_dir} to {self.settings_dir}")
