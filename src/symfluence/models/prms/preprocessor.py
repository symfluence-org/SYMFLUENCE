# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Model Preprocessor

Handles preparation of PRMS model inputs including:
- Control file (control.dat) with simulation settings
- Parameter file (params.dat) with HRU definitions and soil parameters
- Data file (data.dat) with forcing time series
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("PRMS")
class PRMSPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a PRMS model run.

    PRMS requires:
    - control.dat: Master control file with simulation settings
    - params.dat: Parameter file with HRU definitions
    - data.dat: Data file with forcing time series (precip, temp)
    """


    MODEL_NAME = "PRMS"
    def __init__(self, config, logger):
        """
        Initialize the PRMS preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Use standard SYMFLUENCE directory layout (inherited from base):
        #   self.setup_dir   -> {project_dir}/settings/PRMS
        #   self.forcing_dir -> {project_dir}/data/forcing/PRMS_input
        self.settings_dir = self.setup_dir

        # Resolve spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.prms.spatial_mode,
            default=None,
            dict_key='PRMS_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            self.spatial_mode = 'semi_distributed'
        logger.info(f"PRMS spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        """
        Run the complete PRMS preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting PRMS preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate data file (forcing time series)
            self._generate_data_file(start_date, end_date)

            # Generate parameter file
            self._generate_parameter_file()

            # Generate control file
            self._generate_control_file(start_date, end_date)

            logger.info("PRMS preprocessing completed successfully")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"PRMS preprocessing failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create PRMS input directory structure."""
        for d in [self.settings_dir, self.forcing_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created PRMS directory structure: settings={self.settings_dir}, forcing={self.forcing_dir}")

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

    def _generate_data_file(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate the PRMS data file with forcing time series.

        The data file contains daily precipitation and temperature
        (min/max) for each station or HRU.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        logger.info("Generating PRMS data file...")

        data_file_name = self._get_config_value(
            lambda: self.config.model.prms.data_file,
            default='data.dat'
        )

        # Try to load forcing data
        forcing_data = self._load_forcing_data(start_date, end_date)

        out_path = self.forcing_dir / data_file_name

        if forcing_data is not None:
            self._write_data_file(out_path, forcing_data, start_date, end_date)
        else:
            logger.warning("No forcing data found, generating synthetic data file")
            self._generate_synthetic_data_file(out_path, start_date, end_date)

    def _load_forcing_data(self, start_date: datetime, end_date: datetime):
        """
        Load basin-averaged ERA5 forcing data and convert to daily PRMS format.

        Uses the same basin-averaged forcing data as all other models in the
        ensemble (from self.forcing_basin_path inherited from BaseModelPreProcessor).

        Returns:
            pd.DataFrame with columns: precip (mm/day), tmax (°C), tmin (°C)
            indexed by date, or None if no forcing data found.
        """

        # Use inherited forcing_basin_path from BaseModelPreProcessor
        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            logger.warning(f"Forcing path does not exist: {forcing_path}")
            return None

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            logger.warning(f"No NetCDF files found in {forcing_path}")
            return None

        logger.info(f"Loading ERA5 forcing from {forcing_path} ({len(forcing_files)} files)")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
        except Exception:  # noqa: BLE001 — model execution resilience
            datasets = [xr.open_dataset(f) for f in forcing_files]
            ds = xr.concat(datasets, dim='time')

        # Subset to simulation time window
        ds = ds.sel(time=slice(str(start_date), str(end_date)))

        # Extract variables (ERA5 basin-averaged format)
        # airtemp in K, pptrate in mm/s
        airtemp = ds['air_temperature'].values.squeeze()  # (time,) in K
        pptrate = ds['precipitation_flux'].values.squeeze()   # (time,) in mm/s

        # Convert to pandas for daily resampling
        times = pd.DatetimeIndex(ds['time'].values)
        hourly = pd.DataFrame({
            'airtemp_C': airtemp - 273.15,          # K → °C
            'precip_mm': pptrate * 3600.0,           # mm/s → mm/hour
        }, index=times)

        # Resample hourly to daily
        daily = pd.DataFrame({
            'precip': hourly['precip_mm'].resample('D').sum(),    # mm/day
            'tmax': hourly['airtemp_C'].resample('D').max(),      # °C
            'tmin': hourly['airtemp_C'].resample('D').min(),      # °C
        })

        # Drop any days with all NaN
        daily = daily.dropna(how='all')

        ds.close()
        logger.info(f"Loaded ERA5 forcing: {len(daily)} days, "
                     f"precip range [{daily['precip'].min():.1f}, {daily['precip'].max():.1f}] mm/day, "
                     f"temp range [{daily['tmin'].min():.1f}, {daily['tmax'].max():.1f}] °C")
        return daily

    def _write_data_file(self, out_path: Path, forcing_data: pd.DataFrame,
                         start_date, end_date) -> None:
        """
        Write PRMS data file from loaded ERA5 forcing data.

        Args:
            out_path: Path to output data.dat file
            forcing_data: DataFrame with columns precip, tmax, tmin indexed by date
            start_date: Simulation start date
            end_date: Simulation end date
        """
        nhru = 1  # Lumped mode

        with open(out_path, 'w') as f:
            # Header: variable names and counts
            f.write("PRMS data file generated by SYMFLUENCE from ERA5\n")
            f.write(f"precip {nhru}\n")
            f.write(f"tmax {nhru}\n")
            f.write(f"tmin {nhru}\n")
            f.write("runoff 1\n")
            f.write("####\n")

            # Write daily records: date + all values on one line (PRMS MMF format)
            for date, row in forcing_data.iterrows():
                d = pd.Timestamp(date)
                f.write(f"{d.year} {d.month} {d.day} 0 0 0"
                        f" {row['precip']:.4f}"
                        f" {row['tmax']:.2f}"
                        f" {row['tmin']:.2f}"
                        f" 0.0\n")

        logger.info(f"Generated PRMS data file with ERA5 data: {out_path}")

    def _generate_synthetic_data_file(self, out_path: Path,
                                       start_date: datetime,
                                       end_date: datetime) -> None:
        """Raise error — synthetic forcing should never be used."""
        raise FileNotFoundError(
            f"No basin-averaged ERA5 forcing data found in {self.forcing_basin_path}. "
            "PRMS requires real forcing data. Ensure the domain has been set up "
            "with forcing data before running preprocessing."
        )

    def _generate_parameter_file(self) -> None:
        """
        Generate the PRMS parameter file.

        Contains all required dimensions and parameters for PRMS 5.x
        lumped-mode operation including HRU, soil, snow, runoff,
        groundwater, and PET (Jensen-Haise) parameters.
        """
        logger.info("Generating PRMS parameter file...")

        param_file_name = self._get_config_value(
            lambda: self.config.model.prms.parameter_file,
            default='params.dat'
        )

        out_path = self.settings_dir / param_file_name

        nhru = 1  # Lumped mode
        nsegment = 1
        nmonths = 12

        # Bow River at Banff parameters
        lat = 51.17
        lon = -115.57
        elev = 1500.0  # mean basin elevation (m)

        with open(out_path, 'w') as f:
            f.write("PRMS parameter file generated by SYMFLUENCE\n")
            f.write("Version: 1.0\n")

            # ---- Dimensions ----
            f.write("** Dimensions **\n")
            dims = {
                'nhru': nhru, 'nsegment': nsegment, 'nmonths': nmonths,
                'nobs': 1, 'ngw': nhru, 'nssr': nhru, 'nsub': 1,
                'ndepl': 1, 'ndeplval': 11, 'one': 1, 'nrain': 1,
                'ntemp': 1, 'nsol': 0,
            }
            for name, size in dims.items():
                f.write("####\n")
                f.write(f"{name}\n")
                f.write(f"{size}\n")

            # ---- Parameters ----
            f.write("** Parameters **\n")

            def write_param(name, ndim, dim_name, dim_size, dtype, values):
                """Write a single PRMS parameter block.
                dtype: 1=int, 2=float, 4=string
                """
                f.write("####\n")
                f.write(f"{name}\n")
                f.write(f"{ndim}\n")
                f.write(f"{dim_name}\n")
                f.write(f"{dim_size}\n")
                f.write(f"{dtype}\n")
                if isinstance(values, list):
                    for v in values:
                        f.write(f"{v}\n")
                else:
                    f.write(f"{values}\n")

            # --- Unit system ---
            write_param("precip_units", 1, "one", 1, 1, 1)   # 1=mm (default 0=inches)
            write_param("temp_units", 1, "one", 1, 1, 1)      # 1=Celsius (default 0=Fahrenheit)
            write_param("elev_units", 1, "one", 1, 1, 1)      # 1=meters (default 0=feet)

            # --- HRU geometry & classification ---
            # PRMS requires hru_area in acres (1 km² = 247.105 acres)
            area_km2 = 2210.0  # Bow at Banff catchment area
            area_acres = area_km2 * 247.105
            write_param("hru_area", 1, "nhru", nhru, 2, f"{area_acres:.1f}")
            write_param("hru_elev", 1, "nhru", nhru, 2, elev)
            write_param("hru_slope", 1, "nhru", nhru, 2, 0.15)
            write_param("hru_lat", 1, "nhru", nhru, 2, lat)
            write_param("hru_lon", 1, "nhru", nhru, 2, lon)
            write_param("hru_aspect", 1, "nhru", nhru, 2, 180.0)
            write_param("hru_type", 1, "nhru", nhru, 1, 1)       # 1=land
            write_param("cov_type", 1, "nhru", nhru, 1, 3)       # 3=tree
            write_param("soil_type", 1, "nhru", nhru, 1, 2)      # 2=loam
            write_param("hru_percent_imperv", 1, "nhru", nhru, 2, 0.0)

            # --- Connectivity ---
            write_param("hru_segment", 1, "nhru", nhru, 1, 1)
            write_param("hru_subbasin", 1, "nhru", nhru, 1, 1)
            write_param("hru_deplcrv", 1, "nhru", nhru, 1, 1)

            # --- Canopy interception ---
            write_param("covden_sum", 1, "nhru", nhru, 2, 0.5)
            write_param("covden_win", 1, "nhru", nhru, 2, 0.3)
            write_param("snow_intcp", 1, "nhru", nhru, 2, 0.05)
            write_param("srain_intcp", 1, "nhru", nhru, 2, 0.05)
            write_param("wrain_intcp", 1, "nhru", nhru, 2, 0.05)

            # --- Snow ---
            write_param("rad_trncf", 1, "nhru", nhru, 2, 0.5)
            write_param("potet_sublim", 1, "nhru", nhru, 2, 0.5)
            write_param("emis_noppt", 1, "nhru", nhru, 2, 0.757)
            write_param("freeh2o_cap", 1, "nhru", nhru, 2, 0.05)
            write_param("cecn_coef", 1, "nmonths", nmonths, 2,
                         [5.0] * nmonths)

            # Snow depletion curve (11 points for 1 curve = ndeplval)
            write_param("snarea_curve", 1, "ndeplval", 11, 2,
                         [0.05, 0.24, 0.40, 0.53, 0.65, 0.73,
                          0.80, 0.87, 0.93, 0.97, 1.00])
            write_param("snarea_thresh", 1, "nhru", nhru, 2, 50.0)

            # --- Transpiration ---
            write_param("transp_beg", 1, "nhru", nhru, 1, 4)    # April
            write_param("transp_end", 1, "nhru", nhru, 1, 10)   # October
            write_param("transp_tmax", 1, "nhru", nhru, 2, 1.0)

            # --- PET (Jensen-Haise) ---
            write_param("jh_coef", 1, "nmonths", nmonths, 2,
                         [0.014] * nmonths)
            write_param("jh_coef_hru", 1, "nhru", nhru, 2, 13.0)

            # --- Solar radiation (ddsolrad) ---
            write_param("dday_slope", 1, "nmonths", nmonths, 2,
                         [0.30, 0.32, 0.34, 0.36, 0.38, 0.38,
                          0.36, 0.35, 0.33, 0.31, 0.30, 0.29])
            write_param("dday_intcp", 1, "nmonths", nmonths, 2,
                         [-20.0, -18.0, -14.0, -10.0, -6.0, -3.0,
                          -2.0, -3.0, -6.0, -12.0, -16.0, -20.0])

            # --- Surface runoff ---
            write_param("carea_max", 1, "nhru", nhru, 2, 0.6)
            write_param("smidx_coef", 1, "nhru", nhru, 2, 0.01)
            write_param("smidx_exp", 1, "nhru", nhru, 2, 0.3)
            write_param("imperv_stor_max", 1, "nhru", nhru, 2, 0.05)

            # --- Soil zone ---
            write_param("soil_moist_max", 1, "nhru", nhru, 2, 6.0)
            write_param("soil_rechr_max", 1, "nhru", nhru, 2, 2.0)
            write_param("soil_moist_init", 1, "nhru", nhru, 2, 3.0)
            write_param("soil_rechr_init", 1, "nhru", nhru, 2, 1.0)
            write_param("soil2gw_max", 1, "nhru", nhru, 2, 0.5)
            write_param("ssr2gw_rate", 1, "nssr", nhru, 2, 0.1)
            write_param("ssr2gw_exp", 1, "nssr", nhru, 2, 1.0)
            write_param("ssrcoef_lin", 1, "nssr", nhru, 2, 0.1)
            write_param("ssrcoef_sq", 1, "nssr", nhru, 2, 0.1)
            write_param("slowcoef_lin", 1, "nhru", nhru, 2, 0.015)
            write_param("slowcoef_sq", 1, "nhru", nhru, 2, 0.1)
            write_param("pref_flow_den", 1, "nhru", nhru, 2, 0.0)
            write_param("fastcoef_lin", 1, "nhru", nhru, 2, 0.1)
            write_param("fastcoef_sq", 1, "nhru", nhru, 2, 0.1)

            # --- Groundwater ---
            write_param("gwflow_coef", 1, "ngw", nhru, 2, 0.015)
            write_param("gwstor_init", 1, "ngw", nhru, 2, 2.0)
            write_param("gwstor_min", 1, "ngw", nhru, 2, 0.0)
            write_param("gwsink_coef", 1, "ngw", nhru, 2, 0.0)

            # --- Temperature/precipitation adjustments ---
            write_param("tmax_allrain", 1, "nmonths", nmonths, 2,
                         [3.3] * nmonths)
            write_param("tmax_allsnow", 1, "nmonths", nmonths, 2,
                         [0.0] * nmonths)
            write_param("adjmix_rain", 1, "nmonths", nmonths, 2,
                         [1.0] * nmonths)
            write_param("tmax_adj", 1, "nhru", nhru, 2, 0.0)
            write_param("tmin_adj", 1, "nhru", nhru, 2, 0.0)
            write_param("rain_adj", 1, "nmonths", nmonths, 2,
                         [1.0] * nmonths)
            write_param("snow_adj", 1, "nmonths", nmonths, 2,
                         [1.0] * nmonths)

            # --- Station parameters ---
            write_param("tsta_elev", 1, "ntemp", 1, 2, elev)
            write_param("psta_elev", 1, "nrain", 1, 2, elev)
            write_param("hru_tsta", 1, "nhru", nhru, 1, 1)
            write_param("hru_psta", 1, "nhru", nhru, 1, 1)
            write_param("basin_tsta", 1, "one", 1, 1, 1)
            # nsol=0: ddsolrad computes radiation via degree-day, no observed solar data needed

            # --- Segment routing ---
            write_param("tosegment", 1, "nsegment", nsegment, 1, 0)
            write_param("seg_length", 1, "nsegment", nsegment, 2, 10000.0)
            write_param("K_coef", 1, "nsegment", nsegment, 2, 0.1)
            write_param("x_coef", 1, "nsegment", nsegment, 2, 0.2)
            write_param("obsin_segment", 1, "nsegment", nsegment, 1, 0)

        logger.info(f"Generated PRMS parameter file: {out_path}")

    def _generate_control_file(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate the PRMS control file.

        The control file specifies simulation dates, module selection,
        file paths, and output options.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        logger.info("Generating PRMS control file...")

        control_file_name = self._get_config_value(
            lambda: self.config.model.prms.control_file,
            default='control.dat'
        )
        param_file_name = self._get_config_value(
            lambda: self.config.model.prms.parameter_file,
            default='params.dat'
        )
        data_file_name = self._get_config_value(
            lambda: self.config.model.prms.data_file,
            default='data.dat'
        )

        model_mode = self._get_config_value(
            lambda: self.config.model.prms.model_mode,
            default='DAILY'
        )

        out_path = self.settings_dir / control_file_name

        with open(out_path, 'w') as f:
            f.write("PRMS control file generated by SYMFLUENCE\n")

            def write_ctrl(name, nvals, dtype, values):
                """Write a control parameter block.
                dtype: 1=int, 2=float, 4=string
                """
                f.write("####\n")
                f.write(f"{name}\n")
                f.write(f"{nvals}\n")
                f.write(f"{dtype}\n")
                if isinstance(values, list):
                    for v in values:
                        f.write(f"{v}\n")
                else:
                    f.write(f"{values}\n")

            # Simulation dates (6 integer values: year, month, day, hour, min, sec)
            write_ctrl("start_time", 6, 1,
                        [start_date.year, start_date.month, start_date.day,
                         0, 0, 0])
            write_ctrl("end_time", 6, 1,
                        [end_date.year, end_date.month, end_date.day,
                         0, 0, 0])

            # Model mode
            write_ctrl("model_mode", 1, 4, model_mode)

            # File paths
            write_ctrl("param_file", 1, 4, str(self.settings_dir / param_file_name))
            write_ctrl("data_file", 1, 4, str(self.forcing_dir / data_file_name))

            # Module selections (required by PRMS 5)
            write_ctrl("precip_module", 1, 4, "precip_1sta")
            write_ctrl("temp_module", 1, 4, "temp_1sta")
            write_ctrl("solrad_module", 1, 4, "ddsolrad")
            write_ctrl("et_module", 1, 4, "potet_jh")
            write_ctrl("srunoff_module", 1, 4, "srunoff_smidx")
            write_ctrl("strmflow_module", 1, 4, "strmflow")
            write_ctrl("transp_module", 1, 4, "transp_tindex")
            write_ctrl("soilzone_module", 1, 4, "soilzone")

            # Disable subbasin and print_debug to avoid variable conflicts
            write_ctrl("subbasin_flag", 1, 1, 0)
            write_ctrl("print_debug", 1, 1, -1)

            # Output — use relative filenames so PRMS writes to run cwd
            write_ctrl("statsON_OFF", 1, 1, 1)
            write_ctrl("nstatVars", 1, 1, 4)
            write_ctrl("stat_var_file", 1, 4, "statvar.dat")
            write_ctrl("csv_output_file", 1, 4, "prms_output.csv")

            # Output variables
            write_ctrl("statVar_names", 4, 4,
                        ["basin_cfs", "basin_actet", "basin_soil_moist",
                         "basin_pweqv"])
            write_ctrl("statVar_element", 4, 4, [1, 1, 1, 1])
            # Note: Do NOT add a trailing #### — PRMS will try to read
            # a key after the delimiter and hit early EOF

        logger.info(f"Generated PRMS control file: {out_path}")
