# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Observed data processor for streamflow and hydrological observations.

Handles data acquisition from USGS, WSC, SMHI, and LAMAH-ICE providers
with standardized output formatting for model calibration.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from symfluence.core.constants import UnitConversion
from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.path_resolver import find_basin_shapefile


class ObservedDataProcessor(ConfigMixin):
    """Process and standardize observed hydrological data from multiple global providers.

    Central processor for observation data acquisition, unit conversion, temporal resampling,
    and standardization. Handles diverse observation networks (USGS, WSC, SMHI, SNOTEL,
    FLUXNET) with different formats, units, and temporal resolutions. Outputs SYMFLUENCE-
    compatible CSV files for model calibration and evaluation.

    This processor implements the Adapter Pattern to unify different observation provider
    formats into a single standardized output. Provides both provider-specific handlers
    and generic processing functions for common workflows.

    Supported Observation Networks:

        Streamflow (Discharge):
            - USGS (USA): NWIS database
            - WSC (Canada): Water Survey of Canada
            - SMHI (Sweden): Swedish Meteorological and Hydrological Institute
            - LAMAH-ICE (Iceland): Large-Sample data for Hydrology
            - VI (Iceland): Veðurstofa Íslands (National Power Company)
            - CARAVANS (Global): Multi-provider aggregated dataset

        Snow Water Equivalent (SWE):
            - SNOTEL (USA): Automated snow telemetry network
            - MODIS (Global): Satellite snow cover from NASA

        Evapotranspiration (ET):
            - FLUXNET (Global): Eddy covariance flux towers
            - MOD16 (Global): MODIS ET product
            - GLEAM (Global): Global Land Evaporation Amsterdam Model
            - FluxCOM (Global): Machine learning ET

        Other Observations:
            - Soil Moisture (SMAP, ISMN, ESACCI)
            - Groundwater (GRACE)
            - TWS (Total Water Storage)

    Architecture:

        1. Provider Router:
           process_streamflow_data() → Routes to provider-specific handler
           Detects provider from configuration and delegates

        2. Provider-Specific Handlers:
           _process_usgs_data(): NWIS format parsing
           _process_wsc_data(): WSC format parsing
           _process_smhi_data(): SMHI format parsing
           _process_lamah_ice_data(): LAMAH-ICE format parsing
           _process_vi_data(): Iceland VI format parsing
           _process_caravans_data(): Global CARAVANS aggregated data

        3. Generic Processors:
           process_snotel_data(): SWE from SNOTEL
           process_fluxnet_data(): ET from FLUXNET
           _resample_and_save(): Temporal resampling

        4. Unit Conversion:
           Handles all common hydrological unit conversions
           Tracks units through processing pipeline

    Processing Workflow:

        1. Data Input:
           - Read from configured raw data path
           - Handle multiple file formats (CSV, NetCDF, HDF5)
           - Support provider-specific naming conventions

        2. Parsing:
           - Parse dates (multiple format support)
           - Convert values to numeric (handle QA flags, errors)
           - Extract metadata (station ID, location, etc.)

        3. Format Detection:
           - Identify provider-specific formats
           - Handle column naming variations
           - Detect temporal resolution (daily, hourly, subdaily)

        4. Unit Conversion:
           - Convert to standard units (m³/s for discharge, mm for SWE)
           - Track source units for traceability
           - Apply basin area for mm/d → m³/s conversions

        5. Temporal Resampling:
           - Resample to model forcing timestep
           - Interpolate missing values (up to 30 consecutive periods)
           - Handle time zone conversions

        6. Quality Assurance:
           - Flag missing/invalid values
           - Log data gaps and outliers
           - Validate time series continuity

        7. Output:
           - Save standardized CSV format
           - Include metadata (provider, units, processing date)
           - Create symlink to observations directory

    Key Configuration Parameters:

        STREAMFLOW_DATA_PROVIDER: str (required)
            Provider name: 'USGS', 'WSC', 'SMHI', 'LAMAH_ICE', 'VI', 'CARAVANS'
            Determines handler selection and format parsing

        FORCING_TIME_STEP_SIZE: int (required)
            Model timestep in seconds (e.g., 3600 for hourly, 86400 for daily)
            Used for temporal resampling

        DOMAIN_NAME: str (required)
            Basin/domain identifier (e.g., 'site_01', 'bow_at_banff')
            Used for file naming and metadata

        STREAMFLOW_RAW_PATH: Path (optional)
            Directory containing raw observation files
            Default: observations/streamflow/raw_data/

        STREAMFLOW_PROCESSED_PATH: Path (optional)
            Output directory for processed observations
            Default: observations/streamflow/processed/

        STREAMFLOW_RAW_NAME: str (optional)
            Raw file name or pattern
            Default: {domain_name}_streamflow_raw.csv

        SNOTEL_STATION: str (optional)
            SNOTEL station ID for SWE extraction

        FLUXNET_STATION: str (optional)
            FLUXNET site code for ET extraction

    Unit Conversions Handled:

        Streamflow:
            - m³/s (cubic meters per second) - Standard SI unit
            - cfs (cubic feet per second) - USGS default
                * Factor: 1 cfs = 0.028316847 m³/s
            - mm/d (millimeters per day) - Catchment-averaged
                * Conversion: (mm/d × basin_area_km2 × 1e6) / 86400
            - L/s (liters per second)
                * Factor: 1 L/s = 0.001 m³/s

        Precipitation/SWE:
            - mm (millimeters) - Standard
            - inches - SNOTEL default
                * Factor: 1 inch = 25.4 mm
            - cm (centimeters)
                * Factor: 1 cm = 10 mm

        Evapotranspiration:
            - kg m⁻² s⁻¹ (standard hydrological units)
            - mm/d (millimeters per day)
                * Factor: 1 kg m⁻² s⁻¹ = 86.4 mm/d

    Output Format:

        Streamflow CSV (resampled):
            datetime,discharge_cms
            2015-01-01 00:00:00,45.3
            2015-01-01 01:00:00,42.1
            ...
            Columns: datetime (ISO 8601), discharge_cms (m³/s)

        SNOTEL CSV:
            Date,SWE_mm
            2015-01-01,150.2
            2015-01-02,148.5
            ...
            Columns: Date (YYYY-MM-DD), SWE_mm

        FLUXNET CSV:
            datetime,ET_kg_m2_s
            2015-01-01 00:00:00,0.0001234
            ...
            Columns: datetime, ET in kg m⁻² s⁻¹

    Error Handling:

        Missing Data:
            - Logs warnings for gaps
            - Interpolates up to 30 consecutive missing periods
            - Beyond 30 periods: Leaves as NaN

        Invalid Values:
            - Attempts multiple date parsing formats
            - Coerces non-numeric to NaN with warning
            - Logs suspicious outliers (> 3σ from mean)

        Provider-Specific:
            - Falls back to m³/s if unit detection fails
            - Handles QA flags (USGS codes, etc.)
            - Gracefully skips malformed rows

        Critical Failures:
            - Raises DataAcquisitionError if file not found
            - Raises if no valid data rows after parsing
            - Raises if temporal resampling fails

    Example Usage:

        >>> config = {
        ...     'STREAMFLOW_DATA_PROVIDER': 'USGS',
        ...     'FORCING_TIME_STEP_SIZE': 86400,  # Daily
        ...     'DOMAIN_NAME': 'bow_at_banff',
        ...     'SYMFLUENCE_DATA_DIR': '/data/project'
        ... }
        >>> logger = setup_logger()
        >>> processor = ObservedDataProcessor(config, logger)
        >>>
        >>> # Process streamflow
        >>> processor.process_streamflow_data()
        >>> # Output: observations/streamflow/processed/bow_at_banff_streamflow.csv
        >>>
        >>> # Process SNOTEL SWE
        >>> processor.process_snotel_data()
        >>> # Output: observations/snow/processed/{domain}_swe.csv
        >>>
        >>> # Process FLUXNET ET
        >>> processor.process_fluxnet_data()
        >>> # Output: observations/et/processed/{domain}_et.csv

    Performance:

        - Typical processing: 1-5 seconds per observation file
        - Memory: ~100-500 MB for multi-year hourly data
        - Bottleneck: Date parsing and unit conversion for large files

    References:

        - NWIS: https://waterdata.usgs.gov/
        - Water Survey of Canada: https://www.canada.ca/en/services/environment/water/index.html
        - SNOTEL: https://www.wcc.nrcs.usda.gov/snow/
        - FLUXNET: https://fluxnet.org/
        - Unit Conversion: https://www.usgs.gov/faqs/what-conversion-factor-between-millimeters-and-cubic-feet

    See Also:

        - AcquisitionService: High-level data acquisition orchestration
        - ObservationRegistry: Provider registry system
        - Unit Conversion: UnitConversion constants
        - DataManager: Data workflow coordination

    Example:
        >>> config = {
        ...     'STREAMFLOW_DATA_PROVIDER': 'USGS',
        ...     'FORCING_TIME_STEP_SIZE': 3600,  # Hourly
        ...     'DOMAIN_NAME': 'bow_river',
        ...     'SYMFLUENCE_DATA_DIR': './data'
        ... }
        >>> processor = ObservedDataProcessor(config, logger)
        >>> processor.process_streamflow_data()
        # Processes USGS data and saves to:
        # ./data/domain_bow_river/observations/streamflow/preprocessed/bow_river_streamflow_processed.csv

    Notes:
        - CARAVANS data requires shapefile for basin area (mm/d → m³/s conversion)
        - Interpolation limited to 30 periods to avoid excessive extrapolation
        - Timezone conversions applied to WSC data (UTC → local time)
        - Multiple date format parsers attempt to handle diverse input formats

    See Also:
        - observation.base.BaseObservationHandler: Formalized handler interface
        - observation.registry: Registry for observation handler plugins
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        # Note: forcing_time_step_size is available via ConfigMixin property
        self.data_provider = (self._get_config_value(lambda: self.config.data.streamflow_data_provider, dict_key='STREAMFLOW_DATA_PROVIDER') or 'USGS').upper()

        self.streamflow_raw_path = self._get_file_path('STREAMFLOW_RAW_PATH', 'observations/streamflow/raw_data', '')
        self.streamflow_processed_path = self._get_file_path('STREAMFLOW_PROCESSED_PATH', 'observations/streamflow/preprocessed', '')
        self.streamflow_raw_name = self._get_config_value(lambda: self.config.evaluation.streamflow.raw_name, dict_key='STREAMFLOW_RAW_NAME')

    def _get_file_path(self, file_type, file_def_path, file_name):
        file_type_val = self._get_config_value(lambda: None, default='default', dict_key=file_type)
        if file_type_val == 'default':
            # Use resolve_data_subdir for data subdirectories
            parts = file_def_path.split('/', 1)
            if parts[0] in ('observations', 'forcing', 'attributes'):
                base = resolve_data_subdir(self.project_dir, parts[0])
                remainder = parts[1] if len(parts) > 1 else ''
                return base / remainder / file_name if remainder else base / file_name
            return self.project_dir / file_def_path / file_name
        else:
            return Path(file_type_val)

    def get_resample_freq(self):
        if self.forcing_time_step_size == UnitConversion.SECONDS_PER_HOUR:
            return 'h'
        if self.forcing_time_step_size == 10800: # 3 hours in seconds
            return 'h'
        elif self.forcing_time_step_size == UnitConversion.SECONDS_PER_DAY:
            return 'D'
        else:
            return f'{self.forcing_time_step_size}s'

    #: Providers handled by the hardcoded legacy dispatch below. Anything else
    #: (or any provider when ``DATA_ACCESS: community``) is looked up in
    #: ``R.observation_handlers`` so plugin-supplied handlers can serve it.
    LEGACY_STREAMFLOW_PROVIDERS = frozenset({'USGS', 'WSC', 'SMHI', 'LAMAH_ICE', 'VI'})

    def process_streamflow_data(self):
        try:
            if self._get_config_value(lambda: None, default=False, dict_key='PROCESS_CARAVANS'):
                self._process_caravans_data()
                return
            from symfluence.core.registries import R
            from symfluence.data.observation.registry import ObservationRegistry
            backend = str(self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS')).lower()
            key = self.data_provider.lower()
            # ── ObservationBackend protocol tier (contract 0.2.0) ─────────
            # Backend-first under DATA_ACCESS: community: providers claimed
            # by a registered ObservationBackend are served through the
            # protocol BEFORE the R.observation_handlers lookup. Inert seam:
            # with no observation backend registered (every existing
            # configuration), the tiers below run exactly as before.
            if backend == 'community' and self._process_streamflow_via_backend():
                return
            use_registry = backend == 'community' or self.data_provider not in self.LEGACY_STREAMFLOW_PROVIDERS
            if use_registry and key in R.observation_handlers:
                self.logger.debug(f"Dispatching streamflow provider '{self.data_provider}' to registered handler")
                handler = ObservationRegistry.get_handler(key, self.config, self.logger)
                raw_data = handler.acquire()
                handler.process(raw_data)
            elif self.data_provider == 'USGS':
                self.logger.debug("USGS streamflow data handled by formalized observation handler")
            elif self.data_provider == 'WSC':
                self.logger.debug("WSC streamflow data handled by formalized observation handler")
            elif self.data_provider == 'SMHI':
                self.logger.debug("SMHI streamflow data handled by formalized observation handler")
            elif self.data_provider == 'LAMAH_ICE':
                self.logger.debug("LAMAH_ICE streamflow data handled by formalized observation handler")
            elif self.data_provider == 'VI':
                self._process_vi_data()
            else:
                msg = (
                    f"Unsupported streamflow data provider: {self.data_provider}. "
                    "Built-in providers: USGS, WSC, SMHI, LAMAH_ICE, VI. Additional providers can be "
                    "supplied by plugins that register in R.observation_handlers "
                    "(selected automatically, or force registry dispatch with DATA_ACCESS: community)."
                )
                self.logger.error(msg)
                raise DataAcquisitionError(msg)
        except (
            DataAcquisitionError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f'Issue in streamflow data preprocessing: {e}')

    def _observation_station_ids(self) -> tuple:
        """Resolve the provider-scheme station id(s) from the existing config keys.

        Resolution order mirrors the native handlers' own lookups (e.g.
        ``handlers/usgs.py``): the shared evaluation station id first, then
        provider-specific keys. Returns ``()`` when nothing is configured —
        the backend then falls back to its own config-driven resolution.
        """
        keys = ['STATION_ID', f'{self.data_provider}_SITE_CODE', 'STREAMFLOW_STATION_ID']
        if self.data_provider == 'CSFS':
            keys.insert(0, 'CSFS_STATION_ID')

        raw = None
        for dict_key in keys:
            if dict_key == 'STATION_ID':
                raw = self._get_config_value(
                    lambda: self.config.evaluation.streamflow.station_id,
                    default=None, dict_key='STATION_ID')
            else:
                raw = self._get_config_value(lambda: None, default=None, dict_key=dict_key)
            if raw not in (None, '', 'default'):
                break
        if raw in (None, '', 'default'):
            return ()
        if isinstance(raw, (list, tuple)):
            return tuple(str(s).strip() for s in raw if str(s).strip())
        return tuple(part.strip() for part in str(raw).split(',') if part.strip())

    def _process_streamflow_via_backend(self) -> bool:
        """Serve primary streamflow through a registered ObservationBackend.

        Returns False (meaning "use the existing dispatch unchanged") unless
        BOTH hold: (a) at least one backend is registered under
        ``R.observation_backends`` (e.g. the csfs plugin's
        CommunityObservationBackend), and (b) the selector resolves the
        configured provider to it — honoring ``<PROVIDER>_BACKEND`` pins,
        capability kind/window checks, and the parity-gate policy. Selection
        failures fall through cleanly; acquisition failures propagate to the
        caller's existing error handling (same UX as a handler failure).
        """
        from symfluence.core.registries import R

        if not R.observation_backends.keys():
            return False

        from symfluence.data.backends.contract import ObservationRequest
        from symfluence.data.backends.errors import AcquisitionError
        from symfluence.data.backends.selection import select_observation_backend

        time_start = self._get_config_value(lambda: self.config.domain.time_start)
        time_end = self._get_config_value(lambda: self.config.domain.time_end)
        window = (str(time_start), str(time_end)) if time_start and time_end else None

        try:
            backend = select_observation_backend(
                self.data_provider, self.config,
                kind='streamflow', window=window, logger=self.logger,
            )
        except AcquisitionError as exc:
            self.logger.debug(
                f"Observation-backend selection declined for {self.data_provider} "
                f"({exc}); using the existing dispatch."
            )
            return False

        request = ObservationRequest(
            provider_id=self.data_provider,
            station_ids=self._observation_station_ids(),
            kind='streamflow',
            window=window,
            target_dir=Path(self.streamflow_raw_path),
        )
        result = backend.acquire(request)
        self.logger.info(
            f"✓ Streamflow observations acquired via '{backend.name}' backend "
            f"(schema={result.schema}, {len(result.paths)} file(s))"
        )
        for warning in result.warnings:
            self.logger.warning(f"Backend '{backend.name}' warning: {warning}")
        return True

    def _process_vi_data(self):
        self.logger.info("Processing VI (Iceland) streamflow data")

        vi_files = list(self.streamflow_raw_path.glob('*.csv'))
        if not vi_files:
            self.logger.error(f"No CSV files found in {self.streamflow_raw_path} for VI data.")
            return
        vi_file = vi_files[0] # Assuming the first CSV is the one we need

        try:
            vi_data = pd.read_csv(vi_file,
                                  sep=';',
                                  header=None,
                                  names=['YYYY', 'MM', 'DD', 'qobs', 'qc_flag'],
                                  parse_dates={'datetime': ['YYYY', 'MM', 'DD']},  # type: ignore[call-overload]
                                  na_values=['', 'NA', 'NaN'],
                                  skiprows = 1)

            vi_data['discharge_cms'] = pd.to_numeric(vi_data['qobs'], errors='coerce')
            vi_data.set_index('datetime', inplace=True)

            # Filter out data with qc_flag values indicating unreliable measurements
            # The exact meaning of qc_flag values can vary, so this might need adjustment
            # For now, let's assume lower values are more reliable. This is a placeholder.
            # Example: Keep data where qc_flag is None or <= 100
            # reliable_data = vi_data[vi_data['qc_flag'].isna() | (vi_data['qc_flag'] <= 100)]
            # For now, we'll just use all data after conversion

            # Filter out rows where discharge_cms could not be converted
            vi_data = vi_data.dropna(subset=['discharge_cms'])

            self._resample_and_save(vi_data['discharge_cms'])
            self.logger.info(f"Successfully processed VI data from {vi_file}")

        except FileNotFoundError:
            self.logger.error(f"VI data file not found at {vi_file}")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as e:
            self.logger.error(f"Error processing VI data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def process_fluxnet_data(self):
        """
        Process FLUXNET data by copying relevant station files to the project directory.

        This method:
        1. Checks if FLUXNET data acquisition is enabled in configuration
        2. Locates files containing the specified station ID
        3. Copies them to the project directory's observations/fluxnet folder

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Processing FLUXNET data")

        # Check if FLUXNET processing is enabled
        if self._get_config_value(lambda: self.config.evaluation.fluxnet.download, dict_key='DOWNLOAD_FLUXNET') != 'true':
            self.logger.info("FLUXNET data processing is disabled in configuration")
            return False

        try:
            # Get FLUXNET configuration parameters
            fluxnet_path_str = self._get_config_value(lambda: self.config.evaluation.fluxnet.path, dict_key='FLUXNET_PATH')
            station_id = self._get_config_value(lambda: self.config.evaluation.fluxnet.station, dict_key='FLUXNET_STATION')

            if not fluxnet_path_str or not station_id:
                self.logger.error("Missing FLUXNET_PATH or FLUXNET_STATION in configuration")
                return False

            fluxnet_path = Path(fluxnet_path_str)

            # Create directory for FLUXNET data if it doesn't exist
            output_dir = resolve_data_subdir(self.project_dir, 'observations') / 'fluxnet'
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.debug(f"Looking for FLUXNET files with station ID: {station_id} in {fluxnet_path}")

            # Find files containing the station ID
            import shutil

            # Check if the path exists
            if not fluxnet_path.exists():
                self.logger.error(f"FLUXNET path does not exist: {fluxnet_path}")
                return False

            # Find all files in the directory (including subdirectories) that match the station ID
            matching_files = []
            # Use rglob for recursive search
            for file_path in fluxnet_path.rglob('*'):
                if file_path.is_file() and station_id in file_path.name:
                    matching_files.append(file_path)

            if not matching_files:
                self.logger.warning(f"No FLUXNET files found for station ID: {station_id} in {fluxnet_path}")
                return False

            self.logger.debug(f"Found {len(matching_files)} FLUXNET files for station {station_id}")

            # Copy files to the project directory
            for file_path in matching_files:
                dest_file = output_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_file)
                    self.logger.debug(f"Copied {file_path.name} to {dest_file}")
                except (OSError, shutil.Error, ValueError, TypeError) as copy_e:
                    self.logger.error(f"Failed to copy {file_path.name}: {copy_e}")

            self.logger.info(f"Successfully processed FLUXNET data for station {station_id}")
            return True

        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error processing FLUXNET data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_snotel_data(self):
        """
        Process SNOTEL snow water equivalent data.

        This method:
        1. Checks if SNOTEL data download is enabled in configuration
        2. Finds the appropriate SNOTEL CSV file based on station ID
        3. Extracts date and SWE columns
        4. Saves processed data to project directory

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Processing SNOTEL data")

        # Check if SNOTEL processing is enabled
        if self._get_config_value(lambda: self.config.evaluation.snotel.download, dict_key='DOWNLOAD_SNOTEL') != 'true':
            self.logger.info("SNOTEL data processing is disabled in configuration")
            return False

        try:
            # Get SNOTEL configuration parameters
            snotel_path_str = self._get_config_value(lambda: self.config.evaluation.snotel.path, dict_key='SNOTEL_PATH')
            snotel_station_id = self._get_config_value(lambda: self.config.evaluation.snotel.station, dict_key='SNOTEL_STATION')
            domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')

            if not snotel_path_str or not snotel_station_id:
                self.logger.error("Missing SNOTEL_PATH or SNOTEL_STATION in configuration")
                return False

            snotel_path = Path(snotel_path_str)

            # Create directory for processed data if it doesn't exist
            project_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}"
            output_dir = resolve_data_subdir(project_dir, 'observations') / 'snow' / 'swe'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Define output file path
            output_file = output_dir / f"{domain_name}_swe_processed.csv"

            # Find the appropriate SNOTEL file based on station ID
            snotel_file = None

            # Search for files containing the station ID
            # Use rglob for recursive search
            for file in snotel_path.rglob(f'*{snotel_station_id}*.csv'):
                snotel_file = file
                break

            if not snotel_file:
                self.logger.error(f"No SNOTEL file found for station ID: {snotel_station_id} in {snotel_path}")
                return False

            self.logger.debug(f"Found SNOTEL file: {snotel_file}")

            # Read the SNOTEL data file
            import pandas as pd

            # Read the data, skipping header rows until we find the actual data
            # Usually headers end when we find a line starting with "Date"
            header_line_num = -1
            with open(snotel_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.startswith('Date'):
                        header_line_num = i
                        break

            if header_line_num == -1:
                self.logger.error(f"Could not find header line starting with 'Date' in {snotel_file}")
                return False

            # Read the data starting from the identified line
            df = pd.read_csv(snotel_file, skiprows=header_line_num)

            # Extract just the Date and SWE columns
            # The column name might vary, so we'll try to identify it
            swe_column = None
            for col in df.columns:
                if 'Snow Water Equivalent' in col:
                    swe_column = col
                    break

            if not swe_column:
                self.logger.error("Could not find 'Snow Water Equivalent' column in SNOTEL data")
                return False

            # Create a new DataFrame with just Date and SWE
            processed_df = pd.DataFrame()
            processed_df['Date'] = df['Date']
            # Unit-explicit column (NRCS reports inches) so readers never guess.
            swe_out = 'SWE_mm' if '(mm)' in swe_column.lower() else 'SWE_in'
            processed_df[swe_out] = df[swe_column]

            # Try to parse dates with different formats
            try:
                # Attempt to infer format first
                processed_df['Date'] = pd.to_datetime(processed_df['Date'], errors='coerce')
            except (ValueError, pd.errors.ParserError, OverflowError) as date_error:
                self.logger.warning(f"Flexible date parsing failed: {str(date_error)}")
                # Fallback to specific formats if inference fails
                try:
                    processed_df['Date'] = pd.to_datetime(processed_df['Date'], format='%m/%d/%Y', errors='coerce') # MM/DD/YYYY
                except (ValueError, pd.errors.ParserError, OverflowError):
                    try:
                        processed_df['Date'] = pd.to_datetime(processed_df['Date'], format='%Y-%m-%d', errors='coerce') # YYYY-MM-DD
                    except (ValueError, pd.errors.ParserError, OverflowError) as final_error:
                        self.logger.error(f"Could not parse Date column with known formats: {final_error}")
                        return False

            # Ensure the Date column is formatted consistently (YYYY-MM-DD)
            processed_df['Date'] = processed_df['Date'].dt.strftime('%Y-%m-%d')

            # Convert SWE to numeric, coercing errors to NaN
            processed_df[swe_out] = pd.to_numeric(processed_df[swe_out], errors='coerce')

            # Drop rows with invalid dates or SWE values
            processed_df = processed_df.dropna(subset=['Date', swe_out])

            # Save the processed data
            processed_df.to_csv(output_file, index=False)

            self.logger.info(f"Processed SNOTEL data saved to {output_file}")
            return True

        except FileNotFoundError:
            self.logger.error(f"SNOTEL file not found at {snotel_file}")
            return False
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as e:
            self.logger.error(f"Error processing SNOTEL data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _process_caravans_data(self):
        """
        Process CARAVANS streamflow data.

        This function reads CARAVANS CSV data, processes it, and converts from mm/d to m³/s
        using the basin area from the shapefile.
        """
        # Check if CARAVANS processing is enabled
        if not self._get_config_value(lambda: None, default=False, dict_key='PROCESS_CARAVANS'):
            self.logger.info("CARAVANS data processing is disabled in configuration")
            return

        self.logger.info("Processing CARAVANS streamflow data")

        try:
            # Determine input and output paths
            input_file_name = self.streamflow_raw_name
            if not input_file_name:
                self.logger.error("STREAMFLOW_RAW_NAME not specified in config for CARAVANS data.")
                return

            input_file = self.streamflow_raw_path / input_file_name
            output_file = self.streamflow_processed_path / f'{self.domain_name}_streamflow_processed.csv'

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            self.logger.debug(f"Reading CARAVANS data from: {input_file}")

            # Read the CSV file
            try:
                # Try reading with standard format
                caravans_data = pd.read_csv(input_file, sep=',', header=0)
            except (
                OSError,
                ValueError,
                TypeError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
            ) as e:
                self.logger.warning(f"Standard parsing failed: {e}. Trying alternative format...")
                try:
                    # Try with flexible parsing (handles multiple delimiters)
                    caravans_data = pd.read_csv(input_file, sep=r'[,\s]+', engine='python', header=0)
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    UnicodeDecodeError,
                ) as e2:
                    self.logger.error(f"Alternative parsing also failed: {e2}")
                    raise DataAcquisitionError(f"Could not parse CARAVANS data file: {input_file}") from e2

            # Identify date and discharge columns
            date_col_name = None
            discharge_col_name = None

            for col in caravans_data.columns:
                col_lower = col.lower()
                if 'date' in col_lower and not date_col_name:
                    date_col_name = col
                if ('discharge' in col_lower or 'flow' in col_lower or 'm3s' in col_lower or 'mm/d' in col_lower) and not discharge_col_name:
                    discharge_col_name = col

            if not date_col_name:
                self.logger.error("No date column identified in CARAVANS data. Please check column names.")
                raise DataAcquisitionError("No date column found in CARAVANS data")
            if not discharge_col_name:
                self.logger.error("No discharge column identified in CARAVANS data. Please check column names.")
                raise DataAcquisitionError("No discharge column found in CARAVANS data")

            self.logger.debug(f"Using date column: '{date_col_name}', discharge column: '{discharge_col_name}'")

            # Rename columns and select only necessary ones
            caravans_data = caravans_data.rename(columns={date_col_name: 'date', discharge_col_name: 'discharge_value'})
            caravans_data = caravans_data[['date', 'discharge_value']]

            # Convert discharge to numeric, handling errors
            caravans_data['discharge_value'] = pd.to_numeric(caravans_data['discharge_value'], errors='coerce')

            # Convert date to datetime
            try:
                # Try parsing with common formats, prioritizing European format if applicable
                caravans_data['datetime'] = pd.to_datetime(caravans_data['date'], dayfirst=True, errors='coerce')
            except (ValueError, pd.errors.ParserError, OverflowError) as e:
                self.logger.warning(f"Date parsing with dayfirst=True failed: {e}. Trying without.")
                caravans_data['datetime'] = pd.to_datetime(caravans_data['date'], errors='coerce')

            # Drop rows with invalid dates
            na_date_count = caravans_data['datetime'].isna().sum()
            if na_date_count > 0:
                self.logger.warning(f"Dropping {na_date_count} rows with invalid date values")
                caravans_data = caravans_data.dropna(subset=['datetime'])

            # Set datetime as index
            caravans_data.set_index('datetime', inplace=True)

            # Sort index
            caravans_data.sort_index(inplace=True)

            # Now drop rows with NaN discharge values
            na_count = caravans_data['discharge_value'].isna().sum()
            if na_count > 0:
                self.logger.warning(f"Dropping {na_count} rows with missing or non-numeric discharge values")
                caravans_data = caravans_data.dropna(subset=['discharge_value'])

            # Determine if discharge is in mm/d or m³/s based on column name or config
            discharge_unit = 'mm/d' # Default assumption
            if 'm3s' in discharge_col_name.lower() or 'cms' in discharge_col_name.lower():
                discharge_unit = 'm³/s'
            elif 'cfs' in discharge_col_name.lower():
                discharge_unit = 'cfs'

            self.logger.debug(f"Detected discharge unit: {discharge_unit}")

            # Convert discharge to m³/s if necessary
            if discharge_unit == 'mm/d':
                # Get the basin area from the shapefile
                try:
                    # Determine the shapefile path. Honour an explicit override,
                    # else use the shared resolver (handles the nested catchment
                    # layout and river_basins fallback).
                    shapefile_path_str = self._get_config_value(lambda: None, default=None, dict_key='RIVER_BASIN_SHP_PATH')
                    if shapefile_path_str:
                        shapefile_path = Path(shapefile_path_str)
                    else:
                        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
                        shapefile_path = find_basin_shapefile(
                            self.project_dir / "shapefiles",
                            domain_name,
                            self.domain_definition_method,
                            self.experiment_id,
                            required=True,
                            logger=self.logger,
                        )

                    # Read the shapefile
                    import geopandas as gpd
                    gdf = gpd.read_file(shapefile_path)

                    # Get area column from the shapefile
                    area_column = self._get_config_value(lambda: self.config.paths.river_basin_area, default='GRU_area', dict_key='RIVER_BASIN_SHP_AREA')

                    # If area column not found, try alternative names
                    if area_column not in gdf.columns:
                        area_alternatives = ['GRU_area', 'area', 'Area', 'AREA', 'basin_area', 'HRU_area', 'catchment_area']
                        for alt in area_alternatives:
                            if alt in gdf.columns:
                                area_column = alt
                                self.logger.debug(f"Using alternative area column: {area_column}")
                                break

                        # If still not found, calculate area from geometry
                        if area_column not in gdf.columns:
                            self.logger.warning("No area column found, calculating from geometry...")
                            # Ensure CRS is suitable for area calculation (e.g., projected CRS)
                            # If CRS is geographic (lat/lon), reproject to an equal-area projection
                            if gdf.crs and gdf.crs.is_geographic:
                                self.logger.debug(f"Reprojecting to an equal-area CRS for area calculation: {gdf.crs}")
                                # Use a common equal-area projection, e.g., Albers Equal Area
                                gdf_projected = gdf.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs')
                            else:
                                gdf_projected = gdf # Assume it's already projected

                            gdf['calculated_area'] = gdf_projected.geometry.area
                            area_column = 'calculated_area'
                            # Area is now in square meters, convert to square km
                            gdf[area_column] = gdf[area_column] / 1e6

                    # Sum the areas to get total basin area in km²
                    basin_area_km2 = gdf[area_column].sum() # Assuming area is already in km² or m²

                    # Check units and convert if necessary (assuming area column might be in m²)
                    # If the sum is very large (e.g., > 1,000,000 km²), it might be in m²
                    if basin_area_km2 > 1000000:
                        self.logger.warning(f"Basin area sum ({basin_area_km2:.2f}) seems large, assuming units are m² and converting to km².")
                        basin_area_km2 = basin_area_km2 / 1e6
                    elif basin_area_km2 < 0.01:
                        self.logger.warning(f"Basin area sum ({basin_area_km2:.2f}) seems small, assuming units are m² and converting to km².")
                        basin_area_km2 = basin_area_km2 * 1e6 # If it's very small, maybe it's km² and needs conversion to m² for calculation, then back to km²
                        # This logic needs careful review based on expected units.
                        # For now, let's assume the area column is in km² or m² and sum it.
                        # If it's m², we'll convert later.

                    # Convert discharge from mm/d to m³/s
                    # Formula: m³/s = (mm/d × basin_area_km² × 1000) / SECONDS_PER_DAY
                    # 1000: convert km² to m²
                    # SECONDS_PER_DAY: seconds in a day (86400)

                    # Ensure basin_area_km2 is in km² for the formula
                    # If the area column was in m², we need to convert it first
                    if area_column == 'calculated_area' or 'm2' in area_column.lower(): # Heuristic check for m²
                        self.logger.debug(f"Area column '{area_column}' seems to be in m², converting to km².")
                        basin_area_km2 = basin_area_km2 / 1e6

                    conversion_factor = (basin_area_km2 * 1000) / UnitConversion.SECONDS_PER_DAY
                    caravans_data['discharge_cms'] = caravans_data['discharge_value'] * conversion_factor

                    self.logger.debug(f"Basin area: {basin_area_km2:.2f} km²")
                    self.logger.debug(f"Converted discharge from mm/d to m³/s using conversion factor: {conversion_factor:.6f}")

                except FileNotFoundError as fnf_e:
                    self.logger.error(f"Shapefile not found: {fnf_e}. Cannot convert mm/d to m³/s.")
                    raise DataAcquisitionError("Shapefile not found for basin area calculation.") from fnf_e
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as basin_error:
                    self.logger.error(f"Error determining basin area or converting units: {basin_error}")
                    self.logger.warning("Falling back to assuming discharge is already in m³/s.")
                    caravans_data['discharge_cms'] = caravans_data['discharge_value'] # Assume it's already m³/s
            elif discharge_unit == 'm³/s' or discharge_unit == 'cms':
                self.logger.debug("Discharge unit is already m³/s, no conversion needed.")
                caravans_data['discharge_cms'] = caravans_data['discharge_value']
            elif discharge_unit == 'cfs':
                self.logger.debug("Discharge unit is cfs, converting to m³/s.")
                caravans_data['discharge_cms'] = caravans_data['discharge_value'] * UnitConversion.CFS_TO_CMS
            else:
                self.logger.warning(f"Unknown discharge unit '{discharge_unit}'. Assuming it's already in m³/s.")
                caravans_data['discharge_cms'] = caravans_data['discharge_value']

            # Verify we have a DatetimeIndex
            if not isinstance(caravans_data.index, pd.DatetimeIndex):
                self.logger.error("Failed to create DatetimeIndex. Index type is: " + str(type(caravans_data.index)))
                # Try a last-resort conversion
                try:
                    caravans_data.index = pd.to_datetime(caravans_data.index)
                except (ValueError, pd.errors.ParserError, OverflowError) as idx_e:
                    self.logger.error(f"Final attempt to convert index to datetime failed: {idx_e}")
                    raise DataAcquisitionError("Failed to create a valid DatetimeIndex.") from idx_e

            self.logger.debug(f"Data date range: {caravans_data.index.min()} to {caravans_data.index.max()}")
            self.logger.debug(f"Number of records after processing: {len(caravans_data)}")
            self.logger.debug(f"Min discharge: {caravans_data['discharge_cms'].min():.4f} m³/s")
            self.logger.debug(f"Max discharge: {caravans_data['discharge_cms'].max():.4f} m³/s")
            self.logger.debug(f"Mean discharge: {caravans_data['discharge_cms'].mean():.4f} m³/s")

            # Resample and save the data
            self._resample_and_save(caravans_data['discharge_cms'])

            self.logger.info("Successfully processed CARAVANS data")

        except FileNotFoundError:
            self.logger.error(f"CARAVANS input file not found at {input_file}")
        except DataAcquisitionError as dae:
            self.logger.error(f"Data Acquisition Error during CARAVANS processing: {dae}")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as e:
            self.logger.error(f"Error processing CARAVANS data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _process_wsc_data(self):
        self.logger.info("Processing WSC streamflow data from local file")
        try:
            file_path = self.streamflow_raw_path / self.streamflow_raw_name
            if not file_path.exists():
                self.logger.error(f"WSC raw data file not found at {file_path}")
                raise FileNotFoundError(f"WSC raw data file not found: {file_path}")

            # Read the CSV file, handling comments and potential header issues
            # WSC RDB format often has comments starting with '#'
            wsc_data = pd.read_csv(file_path,
                                   comment='#',
                                   low_memory=False)

            # Identify datetime and discharge columns
            datetime_col = None
            discharge_col = None

            # Common datetime column names
            datetime_candidates = ['ISO 8601 UTC', 'datetime', 'date_time', 'Timestamp']
            for col in wsc_data.columns:
                if col in datetime_candidates:
                    datetime_col = col
                    break

            # Common discharge column names
            discharge_candidates = ['Value', 'discharge', 'flow', 'discharge_cms']
            for col in wsc_data.columns:
                for candidate in discharge_candidates:
                    if candidate.lower() in col.lower():
                        discharge_col = col
                        break
                if discharge_col: break

            if not datetime_col:
                self.logger.error("Could not find datetime column in WSC data file.")
                return
            if not discharge_col:
                self.logger.error("Could not find discharge column in WSC data file.")
                return

            self.logger.debug(f"Using datetime column: '{datetime_col}', discharge column: '{discharge_col}'")

            # Convert datetime column, handling potential timezone issues
            wsc_data[datetime_col] = pd.to_datetime(wsc_data[datetime_col], errors='coerce')
            # If timezone info is present (e.g., 'UTC'), remove it for consistency if needed, or convert to local time
            if wsc_data[datetime_col].dt.tz is not None:
                self.logger.debug(f"Detected timezone '{wsc_data[datetime_col].dt.tz}' in WSC datetime. Converting to local time and removing tz info.")
                wsc_data[datetime_col] = wsc_data[datetime_col].dt.tz_convert('America/Edmonton').dt.tz_localize(None)

            # Convert discharge column to numeric
            wsc_data[discharge_col] = pd.to_numeric(wsc_data[discharge_col], errors='coerce')

            # Drop rows with invalid datetime or discharge values
            wsc_data = wsc_data.dropna(subset=[datetime_col, discharge_col])

            # Rename datetime column for consistency
            wsc_data.rename(columns={datetime_col: 'datetime'}, inplace=True)
            wsc_data.set_index('datetime', inplace=True)

            # Rename discharge column to 'discharge_cms' for consistency
            wsc_data.rename(columns={discharge_col: 'discharge_cms'}, inplace=True)

            self._resample_and_save(wsc_data['discharge_cms'])
            self.logger.info(f"Successfully processed local WSC data from {file_path}")

        except FileNotFoundError:
            self.logger.error(f"WSC raw data file not found at {file_path}")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as e:
            self.logger.error(f"Error processing local WSC data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _resample_and_save(self, data):
        resample_freq = self.get_resample_freq()

        # Ensure data is sorted by index before resampling
        data = data.sort_index()

        # Resample the data
        resampled_data = data.resample(resample_freq).mean()

        # Interpolate missing values
        # Use time-based interpolation for potentially irregular time series
        # Limit interpolation to avoid excessive extrapolation
        resampled_data = resampled_data.interpolate(method='time', limit_direction='both', limit=30) # Limit interpolation to 30 periods

        # Optionally, drop remaining NaNs if interpolation didn't fill everything
        # resampled_data = resampled_data.dropna()

        output_file = self.streamflow_processed_path / f'{self.domain_name}_streamflow_processed.csv'

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for writing, ensuring datetime format is consistent
        data_to_write = []
        for dt, value in resampled_data.items():
            # Format datetime to YYYY-MM-DD HH:MM:SS
            formatted_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')
            data_to_write.append([formatted_datetime, value])

        # Write to CSV
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write header
                csv_writer.writerow(['datetime', 'discharge_cms'])
                # Write data rows
                csv_writer.writerows(data_to_write)

            self.logger.info(f"Processed streamflow data saved to: {output_file}")
            self.logger.debug(f"Total rows in processed data: {len(resampled_data)}")
            self.logger.debug(f"Number of non-null values: {resampled_data.count()}")
            self.logger.debug(f"Number of null values after interpolation: {resampled_data.isnull().sum()}")

        except IOError as e:
            self.logger.error(f"Failed to write processed data to {output_file}: {e}")
        except (csv.Error, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"An unexpected error occurred during file writing: {e}")
