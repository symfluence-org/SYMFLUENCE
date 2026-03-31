# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Model Preprocessor

Handles preparation of CRHM model inputs including:
- Observation file (.obs) with meteorological forcing data
- Project file (.prj) with model configuration and parameters
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.spatial_modes import SpatialMode

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("CRHM")
class CRHMPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a CRHM model run.

    CRHM requires:
    - Project file (.prj): Model structure, modules, and parameters
    - Observation file (.obs): Meteorological forcing data in CRHM text format
      with a header block followed by space-separated data columns
    """


    MODEL_NAME = "CRHM"
    def __init__(self, config, logger):
        """
        Initialize the CRHM preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Setup CRHM-specific directories
        # Forcing (.obs) goes in data/forcing/CRHM_input
        # Settings (.prj) goes in settings/CRHM
        self.crhm_forcing_dir = self.project_forcing_dir / "CRHM_input"
        self.settings_dir = self.project_dir / "settings" / "CRHM"

        # Get CRHM-specific settings
        self.spatial_mode = self._get_config_value(
            lambda: self.config.model.crhm.spatial_mode,
            default='lumped',
            dict_key='CRHM_SPATIAL_MODE'
        )
        logger.info(f"CRHM spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        """
        Run the complete CRHM preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting CRHM preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate observation file from ERA5 forcing
            self._generate_observation_file()

            # Generate project file
            self._generate_project_file()

            logger.info("CRHM preprocessing complete.")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"CRHM preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create CRHM input directory structure."""
        self.crhm_forcing_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created CRHM input directories at {self.crhm_forcing_dir} and {self.settings_dir}")

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

                # Get centroid
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y

                # Project to UTM for accurate area
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()

                # Get elevation if available
                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0

                return {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'area_km2': area_m2 / 1e6,
                    'elev': elev
                }
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read catchment properties: {e}")

        # Defaults for cold-region catchment
        return {
            'lat': 51.0,
            'lon': -115.0,
            'area_m2': 1e8,
            'area_km2': 100.0,
            'elev': 1000.0
        }

    def _generate_observation_file(self) -> None:
        """
        Generate the CRHM observation file (.obs) from ERA5 forcing data.

        The .obs file format has:
        - Header lines starting with variable descriptions
        - Column definitions
        - Space-separated data with datetime columns

        Required variables:
        - t (air temperature, deg C)
        - rh (relative humidity, %)
        - p (precipitation, mm)
        - u (wind speed, m/s)
        - Qsi (incoming shortwave radiation, W/m2)
        """
        logger.info("Generating CRHM observation file...")

        obs_file = self._get_config_value(
            lambda: self.config.model.crhm.observation_file,
            default='forcing.obs'
        )
        obs_path = self.crhm_forcing_dir / obs_file

        start_date, end_date = self._get_simulation_dates()

        # Try to load forcing data
        try:
            forcing_ds = self._load_forcing_data()
            self._write_obs_file(forcing_ds, obs_path, start_date, end_date)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not load forcing data: {e}, using synthetic")
            self._generate_synthetic_obs(obs_path, start_date, end_date)

        logger.info(f"Observation file written: {obs_path}")

    def _load_forcing_data(self) -> xr.Dataset:
        """Load basin-averaged forcing data from NetCDF files."""
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            # Try merged_path
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

        # Subset to simulation period
        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _write_obs_file(
        self,
        forcing_ds: xr.Dataset,
        obs_path: Path,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Write forcing data in CRHM .obs format."""
        # Variable mapping from ERA5/forcing names to CRHM names
        var_map = {
            't': ['air_temperature', 'temperature', 'tas', 'temp', 't2m'],
            'p': ['precipitation_flux', 'precipitation', 'pr', 'precip', 'tp'],
            'rh': ['rh', 'relative_humidity', 'hurs'],
            'u': ['wind_speed', 'wind_speed', 'sfcWind', 'wind'],
            'Qsi': ['surface_downwelling_shortwave_flux', 'ssrd', 'rsds', 'swdown', 'shortwave'],
        }

        times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='h')
        time_index = pd.DatetimeIndex(times)

        # Determine timestep
        if len(time_index) > 1:
            dt_seconds = float((time_index[1] - time_index[0]).total_seconds())
        else:
            dt_seconds = 3600.0

        # Extract data for each variable
        data_columns = {}

        for crhm_var, candidates in var_map.items():
            found = False
            for candidate in candidates:
                if candidate in forcing_ds:
                    data = forcing_ds[candidate].values

                    # Flatten spatial dimensions if present
                    if data.ndim > 1:
                        # Average over spatial dims
                        while data.ndim > 1:
                            data = np.nanmean(data, axis=-1)

                    data = data[:len(time_index)]
                    src_units = forcing_ds[candidate].attrs.get('units', '')

                    # Unit conversions
                    if crhm_var == 't':
                        # CRHM expects deg C
                        if src_units == 'K' or np.nanmean(data) > 100:
                            data = data - 273.15
                            logger.info(f"Converted {candidate} from K to deg C")

                    elif crhm_var == 'p':
                        # CRHM expects mm per timestep
                        if 'mm/s' in src_units or candidate == 'precipitation_flux':
                            data = data * dt_seconds
                            logger.info(f"Converted {candidate} from mm/s to mm/timestep")
                        elif src_units == 'm' or candidate == 'tp':
                            data = data * 1000.0
                            logger.info(f"Converted {candidate} from m to mm")
                        elif 'kg' in src_units and 'm-2' in src_units and 's-1' in src_units:
                            data = data * dt_seconds
                            logger.info(f"Converted {candidate} from kg/m2/s to mm/timestep")

                    elif crhm_var == 'rh':
                        # CRHM expects % (0-100)
                        if np.nanmax(data) <= 1.0:
                            data = data * 100.0
                            logger.info(f"Converted {candidate} from fraction to %")

                    elif crhm_var == 'Qsi':
                        # CRHM expects W/m2
                        if candidate in ('ssrd',):
                            data = data / dt_seconds
                            logger.info(f"Converted {candidate} from J/m2 to W/m2")

                    data_columns[crhm_var] = data
                    found = True
                    break

            if not found:
                # Generate synthetic values for missing variables
                n = len(time_index)
                day_frac = np.arange(n) / 24.0

                if crhm_var == 't':
                    data_columns[crhm_var] = 5.0 + 10.0 * np.sin(2 * np.pi * day_frac / 365)
                elif crhm_var == 'p':
                    data_columns[crhm_var] = np.random.exponential(0.1, n)
                elif crhm_var == 'rh':
                    # Derive from specific humidity if available
                    derived = False
                    if 't' in data_columns:
                        for q_name in ['specific_humidity', 'specific_humidity', 'huss', 'q']:
                            if q_name in forcing_ds:
                                q_data = forcing_ds[q_name].values
                                if q_data.ndim > 1:
                                    while q_data.ndim > 1:
                                        q_data = np.nanmean(q_data, axis=-1)
                                q_data = q_data[:n]
                                temp_c = data_columns['t']
                                es = 610.8 * np.exp(17.27 * temp_c / (temp_c + 237.3))
                                # Approximate pressure
                                p_pa = 101325.0 * np.exp(-1000.0 / 8500.0)
                                e = q_data * p_pa / (0.622 + 0.378 * q_data)
                                rh = np.clip(e / es * 100.0, 0, 100)
                                data_columns[crhm_var] = rh
                                derived = True
                                logger.info(f"Derived RH from specific humidity ({q_name})")
                                break
                    if not derived:
                        data_columns[crhm_var] = 60.0 * np.ones(n)
                        logger.warning("Using default RH=60%")
                elif crhm_var == 'u':
                    data_columns[crhm_var] = np.abs(np.random.normal(3.0, 1.0, n))
                elif crhm_var == 'Qsi':
                    data_columns[crhm_var] = np.maximum(
                        0, 200 + 150 * np.sin(2 * np.pi * (day_frac - 80) / 365)
                    )
                logger.warning(f"Variable {crhm_var} not found in forcing, using synthetic")

        # Write the .obs file
        self._write_obs_text(obs_path, time_index, data_columns)

    def _write_obs_text(
        self,
        obs_path: Path,
        time_index: pd.DatetimeIndex,
        data_columns: Dict[str, np.ndarray]
    ) -> None:
        """Write CRHM .obs text file with header and data.

        The CRHM .obs format is:
            Line 1:  free-text description
            Lines 2..N:  variable declarations ``<name> <ncols> [comment]``
                         (optional ``/`` comment lines or ``$`` filter lines)
            Delimiter:  a line starting with ``#``
            Data lines: ``<datetime_fields> <val1> <val2> ...``

        Datetime can be either:
          - Decimal (Excel serial date)  --  a single float > 3000
          - Integer  --  ``YYYY MM DD HH MM`` (5 space-separated integers)
        """
        var_names = list(data_columns.keys())

        lines = []

        # Line 1 -- free-text description (read and discarded by CRHM)
        lines.append(
            f"Observation file for {self.domain_name} "
            f"- SYMFLUENCE {datetime.now().strftime('%Y-%m-%d')}"
        )

        # Variable declarations -- must come *before* the # delimiter.
        # Each line: <var_name> <number_of_columns> [optional unit comment]
        obs_units = {'t': '(C)', 'rh': '(%)', 'p': '(mm)',
                     'u': '(m/s)', 'Qsi': '(W/m2)'}
        for var_name in var_names:
            unit = obs_units.get(var_name, '')
            lines.append(f"{var_name} 1 {unit}".rstrip())

        # Section delimiter
        lines.append("########################################")

        # Data lines: YYYY MM DD HH MM followed by one value per variable
        for i, dt in enumerate(time_index):
            date_str = dt.strftime("%Y %m %d %H %M")
            values = []
            for var_name in var_names:
                val = data_columns[var_name][i] if i < len(data_columns[var_name]) else 0.0
                if np.isnan(val):
                    val = 0.0
                values.append(f"{val:.4f}")
            line = f"{date_str} " + " ".join(values)
            lines.append(line)

        obs_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def _generate_synthetic_obs(
        self,
        obs_path: Path,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Generate synthetic observation data for testing."""
        self._get_catchment_properties()
        dates = pd.date_range(start_date, end_date, freq='h')
        n = len(dates)

        day_frac = np.arange(n) / 24.0

        data_columns = {
            't': 5.0 + 10.0 * np.sin(2 * np.pi * day_frac / 365),
            'rh': 60.0 + 20.0 * np.sin(2 * np.pi * day_frac / 365 + np.pi),
            'p': np.random.exponential(0.1, n),
            'u': np.abs(np.random.normal(3.0, 1.0, n)),
            'Qsi': np.maximum(0, 200 + 150 * np.sin(2 * np.pi * (day_frac - 80) / 365)),
        }

        self._write_obs_text(obs_path, dates, data_columns)
        logger.info(f"Synthetic observation file written: {obs_path}")

    # Module version string used when writing the Modules section.
    # CRHM parses "<module> <DLLName> <version>" for each line.
    _MODULE_DLL = "CRHM"
    _MODULE_VER = "04/20/06"

    # Default module list for a lumped cold-region basin.
    # "basin" and "global" are mandatory infrastructure modules that must
    # appear before any science modules.  "obs" bridges the observation
    # file into the model.
    #
    # IMPORTANT ordering notes:
    #   - "crack" (frozen-soil infiltration) MUST appear before "evap" and
    #     "Soil".  The Soil module reads variables produced by crack; omitting
    #     it causes a null-pointer crash at the first timestep.
    #   - "Annandale", "longVt", and "Slope_Qsi" are radiation estimation
    #     modules that require additional obs variables or DEM-derived fields
    #     not available in a lumped setup.  They are omitted here; "calcsun"
    #     and "netall" provide sufficient radiation processing for most uses.
    _DEFAULT_MODULES = [
        "basin",
        "global",
        "obs",
        "calcsun",
        "intcp",
        "pbsm",
        "albedo",
        "ebsm",
        "netall",
        "crack",
        "evap",
        "Soil",
        "Netroute",
    ]

    # Mapping from parameter name to (owning_module, min, max).
    # "Shared" means the value is broadcast to every module that declares
    # the same parameter name.
    _PARAM_META = {
        # ---- Shared (broadcast to all modules) ----
        'basin_area':      ('Shared', 1e-06, 1e+09),
        'hru_area':        ('Shared', 1e-06, 1e+09),
        'hru_ASL':         ('Shared', 0, 360),
        'hru_elev':        ('Shared', 0, 1e+05),
        'hru_GSL':         ('Shared', 0, 90),
        'hru_lat':         ('Shared', -90, 90),
        'Ht':              ('Shared', 0.001, 100),
        'inhibit_evap':    ('Shared', 0, 5),
        'Sdmax':           ('Shared', 0, 1000),
        'soil_rechr_max':  ('Shared', 0, 350),
        'fetch':           ('Shared', 300, 10000),
        # ---- albedo ----
        'Albedo_bare':     ('albedo', 0, 1),
        'Albedo_snow':     ('albedo', 0, 1),
        # ---- basin ----
        'basin_name':      ('basin', None, None),      # text
        'hru_names':       ('basin', None, None),       # text
        'INIT_STATE':      ('basin', None, None),       # text
        'Loop_to':         ('basin', None, None),       # text
        'RapidAdvance_to': ('basin', None, None),       # text
        'RUN_END':         ('basin', 0, 1e+05),
        'RUN_ID':          ('basin', -1e+08, 1e+08),
        'RUN_START':       ('basin', 0, 1e+05),
        'StateVars_to_Update': ('basin', None, None),   # text
        'TraceVars':       ('basin', None, None),       # text
        # ---- crack ----
        'fallstat':        ('crack', -1, 100),
        'infDays':         ('crack', 0, 20),
        'Major':           ('crack', 1, 100),
        'PriorInfiltration': ('crack', 0, 1),
        # ---- ebsm ----
        'delay_melt':      ('ebsm', 0, 366),
        'nfactor':         ('ebsm', 0, 10),
        'Qe_subl_from_SWE': ('ebsm', 0, 1),
        'tfactor':         ('ebsm', 0, 10),
        'Use_QnD':         ('ebsm', 0, 1),
        # ---- evap ----
        'evap_type':       ('evap', 0, 2),
        'F_Qg':            ('evap', 0, 1),
        'inhibit_evap_User': ('evap', 0, 1),
        'rs':              ('evap', 0, 0.01),
        'Zwind':           ('evap', 0.01, 100),
        # ---- global ----
        'Time_Offset':     ('global', -12, 12),
        # ---- Netroute ----
        'gwKstorage':      ('Netroute', 0, 200),
        'gwLag':           ('Netroute', 0, 1e+04),
        'gwwhereto':       ('Netroute', -1000, 1000),
        'Kstorage':        ('Netroute', 0, 200),
        'Lag':             ('Netroute', 0, 1e+04),
        'order':           ('Netroute', 1, 1000),
        'preferential_flow': ('Netroute', 0, 1),
        'runKstorage':     ('Netroute', 0, 200),
        'runLag':          ('Netroute', 0, 1e+04),
        'Sd_ByPass':       ('Netroute', 0, 1),
        'soil_rechr_ByPass': ('Netroute', 0, 1),
        'ssrKstorage':     ('Netroute', 0, 200),
        'ssrLag':          ('Netroute', 0, 1e+04),
        'whereto':         ('Netroute', 0, 1000),
        # ---- obs ----
        'catchadjust':     ('obs', 0, 3),
        'ClimChng_flag':   ('obs', 0, 1),
        'ClimChng_precip': ('obs', 0, 10),
        'ClimChng_t':      ('obs', -50, 50),
        'ElevChng_flag':   ('obs', 0, 1),
        'HRU_OBS':         ('obs', 1, 100),
        'lapse_rate':      ('obs', 0, 2),
        'obs_elev':        ('obs', 0, 1e+05),
        'ppt_daily_distrib': ('obs', 0, 1),
        'precip_elev_adj': ('obs', -1, 1),
        'snow_rain_determination': ('obs', 0, 2),
        'tmax_allrain':    ('obs', -10, 10),
        'tmax_allsnow':    ('obs', -10, 10),
        # ---- pbsm ----
        'A_S':             ('pbsm', 0, 2),
        'distrib':         ('pbsm', -10, 10),
        # Note: pbsm has its own "fetch" parameter distinct from Shared fetch.
        # It is written directly in _generate_project_file to avoid name collision.
        'inhibit_bs':      ('pbsm', 0, 1),
        'inhibit_subl':    ('pbsm', 0, 1),
        'N_S':             ('pbsm', 1, 500),
        # ---- Soil ----
        'cov_type':        ('Soil', 0, 2),
        'gw_init':         ('Soil', 0, 5000),
        'gw_K':            ('Soil', 0, 100),
        'gw_max':          ('Soil', 0, 5000),
        'lower_ssr_K':     ('Soil', 0, 100),
        'rechr_ssr_K':     ('Soil', 0, 100),
        'Sdinit':          ('Soil', 0, 5000),
        'Sd_gw_K':         ('Soil', 0, 100),
        'Sd_ssr_K':        ('Soil', 0, 100),
        'soil_gw_K':       ('Soil', 0, 100),
        'soil_moist_init': ('Soil', 0, 5000),
        'soil_moist_max':  ('Soil', 0, 5000),
        'soil_rechr_init': ('Soil', 0, 250),
        'soil_ssr_runoff':  ('Soil', 0, 1),
        'soil_withdrawal': ('Soil', 1, 4),
        'transp_limited':  ('Soil', 0, 1),
        'Wetlands_scaling_factor': ('Soil', -1, 1),
    }

    def _generate_project_file(self) -> None:
        """
        Generate the CRHM project file (.prj).

        The .prj format is a section-based text file consumed by the CRHM
        binary.  Each section starts with a keyword line (e.g. ``Dimensions:``)
        followed by a ``######`` delimiter, then the section body, closed by
        another ``######`` delimiter.

        Sections (in required order):
            Header      - description + version
            Dimensions  - nhru, nlay, nobs
            Macros      - (empty)
            Observations - paths to .obs files (one per line)
            Dates       - start/end as ``YYYY MM DD``
            Modules     - ``<name> <DLL> <version>`` per line
            Parameters  - ``<module> <param> <min to max>`` header then values
            Initial_State, Final_State, Summary_period,
            Display_Variable, Display_Observation, Log_All, TChart

        IMPORTANT: The Soil module in CRHM requires the ``crack`` module
        (frozen-soil infiltration) to be present in the module list.
        Without it, CRHM will SIGSEGV at the first timestep when any
        Display_Variable entries are present.  This is not an nhru issue
        -- nhru=1 works correctly when crack is included.
        """
        logger.info("Generating CRHM project file...")

        prj_file = self._get_config_value(
            lambda: self.config.model.crhm.project_file,
            default='model.prj'
        )
        prj_path = self.settings_dir / prj_file

        obs_file = self._get_config_value(
            lambda: self.config.model.crhm.observation_file,
            default='forcing.obs'
        )

        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()

        nhru = 1  # lumped basin -> single HRU

        # Store the observation file as a bare filename.  The runner
        # passes --obs_file_directory to tell CRHM where to find it.
        obs_basename = obs_file

        # Catchment properties
        area_km2 = props.get('area_km2', 100.0)
        lat = props.get('lat', 51.0)
        elev = props.get('elev', 1000.0)

        lines = []

        # -- Header -----------------------------------------------------------
        lines.append(
            f"CRHM Project - {self.domain_name} "
            f"{'lumped' if self.spatial_mode == SpatialMode.LUMPED else self.spatial_mode}"
            f" - SYMFLUENCE"
        )
        lines.append("###### Version NON DLL 4.02")

        # -- Dimensions -------------------------------------------------------
        lines.append("Dimensions:")
        lines.append("######")
        lines.append(f"nhru {nhru}")
        lines.append("nlay 2")
        lines.append("nobs 1")
        lines.append("######")

        # -- Macros (empty but section must exist) ----------------------------
        lines.append("Macros:")
        lines.append("######")
        lines.append("######")

        # -- Observations -----------------------------------------------------
        lines.append("Observations:")
        lines.append("######")
        lines.append(obs_basename)
        lines.append("######")

        # -- Dates ------------------------------------------------------------
        lines.append("Dates:")
        lines.append("######")
        lines.append(start_date.strftime("%Y %m %d"))
        lines.append(end_date.strftime("%Y %m %d"))
        lines.append("######")

        # -- Modules ----------------------------------------------------------
        lines.append("Modules:")
        lines.append("######")
        for mod in self._DEFAULT_MODULES:
            lines.append(f"{mod} {self._MODULE_DLL} {self._MODULE_VER}")
        lines.append("######")

        # -- Parameters -------------------------------------------------------
        lines.append("Parameters:")
        lines.append("###### 'basin' parameters always first")

        # Helper to format a numeric range annotation
        def _range(pmin, pmax):
            if pmin is not None and pmax is not None:
                return f" <{pmin} to {pmax}>"
            return ""

        # Helper to write a parameter line.
        # ``values`` can be:
        #   - a string (text params like basin_name, or pre-formatted
        #     multi-line numeric blocks like HRU_OBS)
        #   - a list/tuple (one value per HRU, joined by spaces)
        #   - a scalar (single value)
        #
        # For text parameters whose _PARAM_META has (None, None) bounds,
        # no range annotation is written.  For string-encoded multi-line
        # numeric blocks, the range annotation IS written when bounds
        # are available.
        def _write_param(module, name, values, pmin=None, pmax=None):
            meta = self._PARAM_META.get(name)
            if meta:
                module = meta[0]
                pmin = meta[1] if pmin is None else pmin
                pmax = meta[2] if pmax is None else pmax

            if isinstance(values, str):
                lines.append(f"{module} {name}{_range(pmin, pmax)}")
                lines.append(values)
            elif isinstance(values, (list, tuple)):
                lines.append(f"{module} {name}{_range(pmin, pmax)}")
                lines.append(' '.join(str(v) for v in values))
            else:
                lines.append(f"{module} {name}{_range(pmin, pmax)}")
                lines.append(str(values))

        # Helper to write per-HRU value (same value for all HRUs)
        def _hru_val(v):
            return ' '.join([str(v)] * nhru)

        # ---- Shared parameters (basin-wide, broadcast to all modules) ----
        _write_param('Shared', 'basin_area', area_km2)
        _write_param('Shared', 'hru_area', _hru_val(area_km2))
        _write_param('Shared', 'hru_ASL', _hru_val(0))
        _write_param('Shared', 'hru_elev', _hru_val(int(elev)))
        _write_param('Shared', 'hru_GSL', _hru_val(5))
        _write_param('Shared', 'hru_lat', _hru_val(round(lat, 2)))
        _write_param('Shared', 'Ht', _hru_val(0.3))
        _write_param('Shared', 'inhibit_evap', _hru_val(0))
        _write_param('Shared', 'Sdmax', _hru_val(10))
        _write_param('Shared', 'soil_rechr_max', _hru_val(60))

        # ---- albedo ----
        _write_param('albedo', 'Albedo_bare', _hru_val(0.17))
        _write_param('albedo', 'Albedo_snow', _hru_val(0.85))

        # ---- basin ----
        _write_param('basin', 'basin_name', "''")
        hru_name_list = ' '.join([f"'HRU{i+1}'" for i in range(nhru)])
        _write_param('basin', 'hru_names', hru_name_list)
        _write_param('basin', 'INIT_STATE', "''")
        _write_param('basin', 'Loop_to', "'' ''")
        _write_param('basin', 'RapidAdvance_to', "''")
        _write_param('basin', 'RUN_END', 0)
        _write_param('basin', 'RUN_ID', 1)
        _write_param('basin', 'RUN_START', 0)
        _write_param('basin', 'StateVars_to_Update',
                      ' '.join(["''"] * 10))
        _write_param('basin', 'TraceVars',
                      ' '.join(["''"] * 10))

        # ---- crack (frozen-soil infiltration) ----
        # Required by the Soil module; without it CRHM will SIGSEGV.
        _write_param('crack', 'fallstat', _hru_val(46))
        _write_param('crack', 'infDays', _hru_val(6))
        _write_param('crack', 'Major', _hru_val(5))
        _write_param('crack', 'PriorInfiltration', _hru_val(0))

        # ---- ebsm (energy-balance snowmelt) ----
        _write_param('ebsm', 'delay_melt', 0)
        _write_param('ebsm', 'nfactor', 0)
        _write_param('ebsm', 'Qe_subl_from_SWE', 0)
        _write_param('ebsm', 'tfactor', 0)
        _write_param('ebsm', 'Use_QnD', 0)

        # ---- evap ----
        _write_param('evap', 'evap_type', 0)
        _write_param('evap', 'F_Qg', 0.05)
        _write_param('evap', 'inhibit_evap_User', 0)
        _write_param('evap', 'rs', 0)
        _write_param('evap', 'Zwind', 10)

        # ---- global ----
        _write_param('global', 'Time_Offset', 0)

        # ---- Netroute (routing) ----
        # For nhru=1: order=1, whereto=0 (outlet), gwwhereto=0 (outlet)
        _write_param('Netroute', 'gwKstorage', _hru_val(0))
        _write_param('Netroute', 'gwLag', _hru_val(0))
        _write_param('Netroute', 'gwwhereto', _hru_val(0))
        _write_param('Netroute', 'Kstorage', _hru_val(1))
        _write_param('Netroute', 'Lag', _hru_val(8))
        _write_param('Netroute', 'order', _hru_val(1))
        _write_param('Netroute', 'preferential_flow', _hru_val(0))
        _write_param('Netroute', 'runKstorage', _hru_val(0))
        _write_param('Netroute', 'runLag', _hru_val(0))
        _write_param('Netroute', 'Sd_ByPass', _hru_val(0))
        _write_param('Netroute', 'soil_rechr_ByPass', _hru_val(0))
        _write_param('Netroute', 'ssrKstorage', _hru_val(0))
        _write_param('Netroute', 'ssrLag', _hru_val(0))
        _write_param('Netroute', 'whereto', _hru_val(0))

        # ---- obs (observation bridge) ----
        _write_param('obs', 'catchadjust', 0)
        _write_param('obs', 'ClimChng_flag', 0)
        _write_param('obs', 'ClimChng_precip', 1)
        _write_param('obs', 'ClimChng_t', 0)
        _write_param('obs', 'ElevChng_flag', _hru_val(1))
        # HRU_OBS: maps each HRU to observation station 1.
        # This is a 2-D parameter (nobs x nhru): 5 rows of nhru values.
        hru_obs_row = _hru_val(1)
        _write_param('obs', 'HRU_OBS',
                      '\n'.join([hru_obs_row] * 5))
        _write_param('obs', 'lapse_rate', _hru_val(0.75))
        # obs_elev: 2-D (nobs x nhru), 2 rows
        _write_param('obs', 'obs_elev',
                      '\n'.join([_hru_val(int(elev))] * 2))
        _write_param('obs', 'ppt_daily_distrib', 1)
        _write_param('obs', 'precip_elev_adj', 0)
        _write_param('obs', 'snow_rain_determination', 0)
        _write_param('obs', 'tmax_allrain', 4)
        _write_param('obs', 'tmax_allsnow', 0)

        # ---- pbsm (blowing snow) ----
        _write_param('pbsm', 'A_S', 0.003)
        _write_param('pbsm', 'distrib', 1)
        # pbsm has its own "fetch" parameter distinct from the Shared one.
        # Write it directly to avoid collision with the Shared fetch key.
        lines.append("pbsm fetch <300 to 10000.0>")
        lines.append("1500")
        _write_param('pbsm', 'inhibit_bs', 0)
        _write_param('pbsm', 'inhibit_subl', 0)
        _write_param('pbsm', 'N_S', 320)

        # ---- Soil ----
        _write_param('Soil', 'cov_type', 1)
        _write_param('Soil', 'gw_init', 75)
        _write_param('Soil', 'gw_K', 0.001)
        _write_param('Soil', 'gw_max', 150)
        _write_param('Soil', 'lower_ssr_K', 0.001)
        _write_param('Soil', 'rechr_ssr_K', 0.001)
        _write_param('Soil', 'Sdinit', 0)
        _write_param('Soil', 'Sd_gw_K', 0.001)
        _write_param('Soil', 'Sd_ssr_K', 0.001)
        _write_param('Soil', 'soil_gw_K', 0.001)
        _write_param('Soil', 'soil_moist_init', 125)
        _write_param('Soil', 'soil_moist_max', 250)
        _write_param('Soil', 'soil_rechr_init', 30)
        _write_param('Soil', 'soil_ssr_runoff', 1)
        # soil_withdrawal: 2-D (nlay x nhru)
        _write_param('Soil', 'soil_withdrawal',
                      '\n'.join([_hru_val(2)] * 2))
        _write_param('Soil', 'transp_limited', 0)
        _write_param('Soil', 'Wetlands_scaling_factor', 1)

        lines.append("######")

        # -- Initial_State (empty) -------------------------------------------
        lines.append("Initial_State")
        lines.append("######")
        lines.append("######")

        # -- Final_State (empty) ---------------------------------------------
        lines.append("Final_State")
        lines.append("######")
        lines.append("######")

        # -- Summary_period --------------------------------------------------
        lines.append("Summary_period")
        lines.append("######")
        lines.append("Water_year 10")
        lines.append("######")

        # -- Log_Time_Format -------------------------------------------------
        lines.append("Log_Time_Format")
        lines.append("######")
        lines.append("MS_DateTimeIndex")
        lines.append("######")

        # -- Display_Variable ------------------------------------------------
        # CRHM requires at least one selected output variable, otherwise
        # the model runs but produces no output and crashes (SIGSEGV)
        # during cleanup.  Select key hydrological state/flux variables
        # from the modules in the module list.
        lines.append("Display_Variable:")
        lines.append("######")

        # Build HRU index string: "1 2 3 ... nhru"
        hru_indices = ' '.join(str(i + 1) for i in range(nhru))

        # Key output variables -- names verified against CRHM source.
        # IMPORTANT: 'basinflowLoss' does NOT exist in standard CRHM;
        # 'runoff' should be 'soil_runoff'.  Only use variable names
        # that the CRHM binary actually registers.
        display_vars = [
            ('pbsm',     'SWE'),              # snow water equivalent (mm)
            ('Netroute', 'basinflow'),        # basin outflow (m3/s)
            ('Netroute', 'basingw'),          # basin groundwater flow (m3/s)
            ('Soil',     'soil_moist'),       # soil moisture (mm)
            ('Soil',     'soil_rechr'),       # recharge zone storage (mm)
            ('Soil',     'gw'),              # groundwater storage (mm)
            ('Soil',     'gw_flow'),          # groundwater flow (mm)
            ('evap',     'hru_actet'),        # actual evapotranspiration (mm)
            ('obs',      'hru_t'),            # HRU temperature (C)
            ('obs',      'hru_p'),            # HRU precipitation (mm)
        ]
        for module, var in display_vars:
            if module in self._DEFAULT_MODULES or module in ('obs',):
                lines.append(f"{module} {var} {hru_indices}")

        lines.append("######")

        # -- Display_Observation (empty) -------------------------------------
        lines.append("Display_Observation:")
        lines.append("######")
        lines.append("######")

        # -- Auto_Run --------------------------------------------------------
        lines.append("Auto_Run")
        lines.append("######")

        # -- Auto_Exit -------------------------------------------------------
        lines.append("Auto_Exit")
        lines.append("######")

        # -- Log_All ---------------------------------------------------------
        lines.append("Log_All")
        lines.append("######")

        # -- TChart (empty) --------------------------------------------------
        lines.append("TChart:")
        lines.append("######")
        lines.append("")
        lines.append("")
        lines.append("()")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("######")

        content = '\n'.join(lines) + '\n'
        prj_path.write_text(content, encoding='utf-8')
        logger.info(f"Project file written: {prj_path}")

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
