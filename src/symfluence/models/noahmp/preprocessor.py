# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Noah-MP Model Preprocessor.

Prepares forcing data and configuration files for the standalone
noah-owp-modular model (NOAA-OWP).  Forcing is converted from NetCDF
(basin-averaged) to the ASCII format expected by the executable, and
parameter tables (GENPARM.TBL, SOILPARM.TBL, MPTABLE.TBL) plus
a Fortran namelist (namelist.input) are written to the settings directory.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add('NOAHMP')
class NoahMPPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Preprocessor for the standalone Noah-MP land surface model.

    Converts basin-averaged forcing (NetCDF) into the ASCII format
    expected by noah-owp-modular and generates the Fortran namelist
    plus parameter lookup tables.
    """

    MODEL_NAME = "NOAHMP"

    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.noahmp_forcing_dir = self.project_forcing_dir / 'NOAHMP_input'
        self.noahmp_settings_dir = self.project_dir / 'settings' / 'NOAHMP'

        self._forcing_start: Optional[datetime] = None
        self._forcing_end: Optional[datetime] = None

    def get_base_settings_source_dir(self) -> Path:
        """Resolve base settings against the shared ``NOAH`` resource dir.

        The standalone NOAH-MP model (MODEL_NAME=NOAHMP) shares its Noah-OWP
        base settings — the parameter lookup tables under
        resources/base_settings/NOAH — with the NGEN NOAH BMI module; there is
        no separate ``NOAHMP`` resource directory. Override the default lookup
        (which keys on MODEL_NAME) to use ``NOAH``."""
        from symfluence.core.modeling.base_settings import get_base_settings_dir
        code_dir_value = self._get_config_value(
            lambda: self.config.system.code_dir, default=None)
        if code_dir_value:
            src = (Path(code_dir_value) / "src" / "symfluence" / "resources"
                   / "base_settings" / "NOAH")
            if src.exists():
                return src
        return get_base_settings_dir("NOAH")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_preprocessing(self) -> bool:
        """Run the complete Noah-MP preprocessing workflow."""
        self.noahmp_forcing_dir = self.project_forcing_dir / 'NOAHMP_input'
        self.noahmp_settings_dir = self.project_dir / 'settings' / 'NOAHMP'
        return self.run_preprocessing_template()

    # ------------------------------------------------------------------
    # Template-method hooks
    # ------------------------------------------------------------------

    def _prepare_forcing(self) -> None:
        """Convert basin-averaged NetCDF forcing to Noah-MP ASCII format.

        noah-owp-modular expects a single ASCII file with a ``<FORCING>``
        header line followed by rows of
        ``YYYY MM DD HH MM  windspd airtemp spechum airpres SWRad LWRad pptrate RH``
        using a Fortran format of ``(I4.4, 4(1x,I2.2), 8(F17.10))``.
        """
        # Only the forcing remapped onto this run's HRUs (issue #339).
        run_files = self._select_run_forcing(sorted(self.forcing_basin_path.glob('*.nc')))
        try:
            self.noahmp_forcing_dir.mkdir(parents=True, exist_ok=True)

            # Locate forcing NetCDF
            forcing_files = run_files
            if not forcing_files:
                logger.error(f"No forcing NetCDF found in {self.forcing_basin_path}")
                return

            ds = xr.open_mfdataset(forcing_files, combine='by_coords')

            # Variable mapping (SYMFLUENCE standard -> Noah-MP names)
            # Aliases include the canonical CFIF names that the model-ready
            # store actually uses (surface_air_pressure, precipitation_flux,
            # surface_downwelling_{short,long}wave_flux) — these were missing,
            # so airpres/SW/LW/pptrate lookups failed and no forcing was written.
            var_map = {
                'windspd': ['windspd', 'wind_speed', 'WIND', 'u10'],
                'airtemp': ['airtemp', 'air_temperature', 'TAIR', 't2m', 'TMP'],
                'spechum': ['spechum', 'specific_humidity', 'SPFH', 'q2'],
                'airpres': ['airpres', 'surface_air_pressure', 'air_pressure',
                            'PRES', 'sp', 'PSFC'],
                'SWRadAtm': ['SWRadAtm', 'surface_downwelling_shortwave_flux',
                             'DSWRF', 'SWDOWN', 'ssrd'],
                'LWRadAtm': ['LWRadAtm', 'surface_downwelling_longwave_flux',
                             'DLWRF', 'LWDOWN', 'strd'],
                'pptrate': ['pptrate', 'precipitation_flux', 'APCP', 'RAINRATE',
                            'tp', 'PRATE'],
            }

            def _find_var(ds, aliases):
                for alias in aliases:
                    if alias in ds:
                        return ds[alias].values.flatten()
                return None

            windspd = _find_var(ds, var_map['windspd'])
            airtemp = _find_var(ds, var_map['airtemp'])
            spechum = _find_var(ds, var_map['spechum'])
            airpres = _find_var(ds, var_map['airpres'])
            swrad = _find_var(ds, var_map['SWRadAtm'])
            lwrad = _find_var(ds, var_map['LWRadAtm'])
            pptrate = _find_var(ds, var_map['pptrate'])

            times = pd.to_datetime(ds['time'].values)
            ds.close()

            for name, arr in [('windspd', windspd), ('airtemp', airtemp),
                              ('spechum', spechum), ('airpres', airpres),
                              ('SWRadAtm', swrad), ('LWRadAtm', lwrad),
                              ('pptrate', pptrate)]:
                if arr is None:
                    logger.error(f"Required forcing variable '{name}' not found")
                    return

            # Convert specific humidity -> relative humidity
            rh = self._spechum_to_rh(spechum, airtemp, airpres)

            # Store time bounds
            self._forcing_start = times[0].to_pydatetime()
            self._forcing_end = times[-1].to_pydatetime()

            # Write ASCII forcing file.
            # noah-owp-modular's read_forcing_text (driver/AsciiReadModule.f90)
            # reads the 8 data columns in this exact order and with these units:
            #   windspeed [m/s], winddir [deg], temperature [K], humidity [%RH],
            #   pressure [mb/hPa], shortwave [W/m2], longwave [W/m2], precip [kg/m2/s]
            # It then converts pressure*100 -> Pa, humidity*0.01 -> fraction, and
            # derives u/v from speed+direction. We have no wind direction, so we
            # write 0.0 (the model uses wind-speed magnitude regardless), supply
            # relative humidity in percent, and convert pressure Pa -> hPa.
            forcing_path = self.noahmp_forcing_dir / 'forcing.txt'
            winddir = np.zeros_like(windspd)
            airpres_hpa = airpres / 100.0
            with open(forcing_path, 'w') as f:
                f.write('<FORCING>\n')
                for i, t in enumerate(times):
                    line = (
                        f"{t.year:04d} {t.month:02d} {t.day:02d} "
                        f"{t.hour:02d} {t.minute:02d}"
                        f"{windspd[i]:17.10f}{winddir[i]:17.10f}"
                        f"{airtemp[i]:17.10f}{rh[i]:17.10f}"
                        f"{airpres_hpa[i]:17.10f}{swrad[i]:17.10f}"
                        f"{lwrad[i]:17.10f}{pptrate[i]:17.10f}"
                    )
                    f.write(line + '\n')

            logger.info(
                f"Wrote Noah-MP forcing: {len(times)} timesteps "
                f"({self._forcing_start} to {self._forcing_end})"
            )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error preparing Noah-MP forcing: {e}", exc_info=True)
            import traceback
            logger.debug(traceback.format_exc())

    def _create_model_configs(self) -> None:
        """Copy parameter tables and write namelist.input."""
        try:
            self.noahmp_settings_dir.mkdir(parents=True, exist_ok=True)

            # Copy parameter lookup tables from install directory
            install_path = self._get_config_value(
                lambda: self.config.model.noahmp.install_path,
                dict_key='NOAHMP_INSTALL_PATH',
            )
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir,
                dict_key='SYMFLUENCE_DATA_DIR',
            )
            install_dir = resolve_install_dir(install_path, data_dir, 'installs/noah-owp-modular')

            params_dir = install_dir / 'parameters'
            for tbl in ['GENPARM.TBL', 'SOILPARM.TBL', 'MPTABLE.TBL']:
                src = params_dir / tbl
                if src.exists():
                    shutil.copy2(src, self.noahmp_settings_dir / tbl)
                    logger.debug(f"Copied {tbl}")
                else:
                    logger.warning(f"Parameter table not found: {src}")

            # Write namelist.input
            self._write_namelist()

            logger.info("Noah-MP configuration files created")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error creating Noah-MP configs: {e}", exc_info=True)
            import traceback
            logger.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # Namelist generation
    # ------------------------------------------------------------------

    def _write_namelist(self) -> None:
        """Write the Fortran namelist (namelist.input) for noah-owp-modular."""
        # Determine time settings
        if self._forcing_start and self._forcing_end:
            start_str = self._forcing_start.strftime('%Y%m%d%H%M')
            end_str = self._forcing_end.strftime('%Y%m%d%H%M')
        else:
            start_str = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='200001010000',
                dict_key='EXPERIMENT_TIME_START',
            ).replace('-', '').replace(' ', '').replace(':', '')
            end_str = self._get_config_value(
                lambda: self.config.domain.time_end,
                default='200112310000',
                dict_key='EXPERIMENT_TIME_END',
            ).replace('-', '').replace(' ', '').replace(':', '')

        # Dynamic vegetation option from config (default 1)
        dynamic_veg = int(self._get_config_value(
            lambda: self.config.model.noahmp.dynamic_veg_option,
            dict_key='NOAHMP_DYNAMIC_VEG_OPTION',
        ))

        forcing_path = self.noahmp_forcing_dir / 'forcing.txt'
        # Use a RELATIVE output filename so the model writes output.nc into its
        # working directory (the settings dir the runner/worker launch from).
        # The calibration worker runs each evaluation with cwd=<process settings
        # dir> and looks for output.nc there; an absolute path would send every
        # run's output to a single shared location the worker never checks,
        # making every iteration score -9999 (counted as a crash).
        output_path = 'output.nc'

        namelist = f"""\
&timing
  dt                 = 3600
  startdate          = "{start_str}"
  enddate            = "{end_str}"
  forcing_filename   = "{forcing_path}"
  output_filename    = "{output_path}"
/

&parameters
  parameter_dir      = "{self.noahmp_settings_dir}/"
  general_table      = "GENPARM.TBL"
  soil_table         = "SOILPARM.TBL"
  noahowp_table      = "MPTABLE.TBL"
  soil_class_name    = "STAS"
  veg_class_name     = "MODIFIED_IGBP_MODIS_NOAH"
/

&location
  lat                =  {self._get_lat():.6f}
  lon                =  {self._get_lon():.6f}
  terrain_slope      =  0.0
  azimuth            =  0.0
/

&forcing
  ZREF               =  10.0
  rain_snow_thresh   =  1.0
/

&model_options
  precip_phase_option           = 1
  snow_albedo_option            = 1
  dynamic_veg_option            = {dynamic_veg}
  runoff_option                 = 8
  drainage_option               = 8
  frozen_soil_option            = 1
  dynamic_vic_option            = 1
  radiative_transfer_option     = 3
  sfc_drag_coeff_option         = 1
  canopy_stom_resist_option     = 1
  crop_model_option             = 0
  snowsoil_temp_time_option     = 3
  soil_temp_boundary_option     = 2
  supercooled_water_option      = 1
  stomatal_resistance_option    = 1
  evap_srfc_resistance_option   = 1
  subsurface_option             = 1
/

&structure
  isltyp             = 4
  nsoil              = 4
  nsnow              = 3
  nveg               = 20
  vegtyp             = 1
  croptype           = 0
  sfctyp             = 1
  soilcolor          = 4
/

&initial_values
  dzsnso             = 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 1.0
  sice               = 0.0, 0.0, 0.0, 0.0
  sh2o               = 0.3, 0.3, 0.3, 0.3
  zwt                = -2.0
/
"""
        nl_path = self.noahmp_settings_dir / 'namelist.input'
        nl_path.write_text(namelist)
        logger.info(f"Wrote namelist.input to {nl_path}")

    # ------------------------------------------------------------------
    # Helper: lat/lon from domain configuration
    # ------------------------------------------------------------------

    def _get_lat(self) -> float:
        """Get latitude from configuration or default."""
        return float(self._get_config_value(
            lambda: None,
            default=51.0,
            dict_key='DOMAIN_LATITUDE',
        ))

    def _get_lon(self) -> float:
        """Get longitude from configuration or default."""
        return float(self._get_config_value(
            lambda: None,
            default=-116.0,
            dict_key='DOMAIN_LONGITUDE',
        ))

    # ------------------------------------------------------------------
    # Static utility: specific humidity -> relative humidity
    # ------------------------------------------------------------------

    @staticmethod
    def _spechum_to_rh(
        q: np.ndarray,
        t_k: np.ndarray,
        p_pa: np.ndarray,
    ) -> np.ndarray:
        """Convert specific humidity to relative humidity.

        Uses the Tetens approximation for saturation vapour pressure.

        Args:
            q: Specific humidity (kg/kg).
            t_k: Air temperature (K).
            p_pa: Air pressure (Pa).

        Returns:
            Relative humidity (%), clipped to [0, 100].
        """
        tc = t_k - 273.15
        es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
        e = q * p_pa / (0.622 + 0.378 * q)
        rh = 100.0 * e / es
        return np.clip(rh, 0.0, 100.0)
