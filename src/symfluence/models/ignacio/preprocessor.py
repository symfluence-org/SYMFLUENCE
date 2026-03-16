# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO PreProcessor for SYMFLUENCE

Prepares input data for IGNACIO fire spread simulations including:
- Terrain data (DEM, slope, aspect)
- Fuel type rasters
- Weather station data
- Ignition point configuration
- IGNACIO YAML configuration file generation
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor('IGNACIO')
class IGNACIOPreProcessor(BaseModelPreProcessor):
    """
    Preprocessor for IGNACIO fire spread model.

    Handles preparation of input data and configuration for IGNACIO simulations.
    This includes generating the YAML configuration file that IGNACIO expects.
    """


    MODEL_NAME = "IGNACIO"
    def __init__(self, config, logger_instance=None, reporting_manager=None):
        """
        Initialize the IGNACIO preprocessor.

        Args:
            config: SymfluenceConfig object with domain and model settings
            logger_instance: Optional logger for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger_instance or logger)

        # Setup IGNACIO-specific paths
        self.ignacio_input_dir = self.project_dir / "IGNACIO_input"
        self.ignacio_config_path = self.ignacio_input_dir / "ignacio_config.yaml"

    def run_preprocessing(self, **kwargs) -> bool:
        """
        Execute IGNACIO preprocessing.

        Creates the IGNACIO input directory and generates the configuration
        YAML file from SYMFLUENCE configuration.

        Returns:
            True if preprocessing completed successfully
        """
        self.logger.info("Running IGNACIO preprocessing...")

        try:
            # Create input directories
            self.ignacio_input_dir.mkdir(parents=True, exist_ok=True)

            # Get IGNACIO config from SYMFLUENCE config
            ignacio_config = self._get_ignacio_config()

            # Prepare terrain data
            terrain_paths = self._prepare_terrain_data(ignacio_config)

            # Prepare fuel data
            fuel_path = self._prepare_fuel_data(ignacio_config)

            # Prepare weather data
            weather_path = self._prepare_weather_data(ignacio_config)

            # Prepare ignition data
            ignition_path = self._prepare_ignition_data(ignacio_config)

            # Generate IGNACIO YAML config
            self._generate_ignacio_config(
                ignacio_config,
                terrain_paths,
                fuel_path,
                weather_path,
                ignition_path
            )

            self.logger.info(f"IGNACIO preprocessing complete. Config: {self.ignacio_config_path}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"IGNACIO preprocessing failed: {e}")
            return False

    def _get_ignacio_config(self) -> Dict[str, Any]:
        """Extract IGNACIO configuration from SYMFLUENCE config."""
        config_dict: Dict[str, Any] = {}

        # Try to get from typed config
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'ignacio'):
            ignacio = self.config.model.ignacio
            if ignacio is not None:
                # Convert Pydantic model to dict
                config_dict = ignacio.model_dump() if hasattr(ignacio, 'model_dump') else dict(ignacio)

        # Also check for IGNACIO_ prefixed keys in config_dict
        for key, value in self.config_dict.items():
            if key.startswith('IGNACIO_'):
                # Convert to lowercase without prefix for internal use
                internal_key = key[8:].lower()
                if internal_key not in config_dict:
                    config_dict[internal_key] = value

        # Set defaults
        config_dict.setdefault('project_name', self.config.domain.name)
        config_dict.setdefault('output_dir', str(self.project_dir / 'simulations' /
                                                   self.config.domain.experiment_id / 'IGNACIO'))

        return config_dict

    def _prepare_terrain_data(self, ignacio_config: Dict) -> Dict[str, Optional[Path]]:
        """
        Prepare terrain data (DEM, slope, aspect).

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Dictionary with paths to terrain files
        """
        terrain_paths: dict[str, Path | None] = {
            'dem_path': None,
            'slope_path': None,
            'aspect_path': None,
        }

        # Check if DEM path is specified
        dem_path = ignacio_config.get('dem_path')
        if dem_path and dem_path != 'default':
            dem_path = Path(dem_path)
            if dem_path.exists():
                terrain_paths['dem_path'] = dem_path
                self.logger.info(f"Using DEM: {dem_path}")

        # Try default DEM locations
        if terrain_paths['dem_path'] is None:
            # Try RHESSys fire DEM first (most likely for fire simulations)
            fire_dem = self.project_dir / 'settings' / 'RHESSys' / 'fire' / 'dem_grid.tif'
            if fire_dem.exists():
                terrain_paths['dem_path'] = fire_dem
                self.logger.info(f"Using RHESSys fire DEM: {fire_dem}")
            else:
                # Try standard attribute location
                default_dem = self.project_attributes_dir / 'dem' / f"{self.config.domain.name}_dem.tif"
                if default_dem.exists():
                    terrain_paths['dem_path'] = default_dem
                    self.logger.info(f"Using default DEM: {default_dem}")
                else:
                    # Try alternative location
                    alt_dem = self.project_dir / 'shapefiles' / 'dem' / 'dem.tif'
                    if alt_dem.exists():
                        terrain_paths['dem_path'] = alt_dem

        # Check for pre-computed slope/aspect
        slope_path = ignacio_config.get('slope_path')
        if slope_path and Path(slope_path).exists():
            terrain_paths['slope_path'] = Path(slope_path)

        aspect_path = ignacio_config.get('aspect_path')
        if aspect_path and Path(aspect_path).exists():
            terrain_paths['aspect_path'] = Path(aspect_path)

        return terrain_paths

    def _prepare_fuel_data(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Prepare fuel type raster data.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to fuel raster or None
        """
        fuel_path = ignacio_config.get('fuel_path')
        if fuel_path and fuel_path != 'default':
            fuel_path = Path(fuel_path)
            if fuel_path.exists():
                self.logger.info(f"Using fuel raster: {fuel_path}")
                return fuel_path

        # Try default locations
        default_paths = [
            self.project_attributes_dir / 'fuels' / 'fuels.tif',
            self.project_dir / 'shapefiles' / 'fuels' / 'fuels.tif',
            # Land class can be used as proxy for fuels (with domain_ prefix)
            self.project_attributes_dir / 'landclass' / f"domain_{self.config.domain.name}_land_classes.tif",
            # Without prefix
            self.project_attributes_dir / 'landclass' / f"{self.config.domain.name}_land_classes.tif",
        ]

        for default_path in default_paths:
            if default_path.exists():
                self.logger.info(f"Using fuel/landclass raster: {default_path}")
                return default_path

        self.logger.warning("No fuel raster found. IGNACIO will use default fuel type.")
        return None

    def _prepare_weather_data(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Prepare weather station data for FWI calculation.

        Converts ERA5 forcing data to IGNACIO weather station CSV format
        if no station data is provided.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to weather CSV or None
        """
        station_path = ignacio_config.get('station_path')
        if station_path and station_path != 'default':
            station_path = Path(station_path)
            if station_path.exists():
                self.logger.info(f"Using weather station data: {station_path}")
                return station_path

        # Try to convert ERA5 forcing data to IGNACIO weather format
        weather_csv = self._convert_era5_to_weather_csv(ignacio_config)
        if weather_csv:
            return weather_csv

        # Generate static weather as fallback (IGNACIO requires weather file)
        self.logger.info("Generating static weather data for IGNACIO")
        return self._generate_static_weather_csv(ignacio_config)

    def _convert_era5_to_weather_csv(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Convert ERA5 forcing data to IGNACIO weather station CSV format.

        IGNACIO expects columns: HOURLY, TEMP, RH, WS, WD, PRECIP
        ERA5 provides: airtemp, spechum, windspd, pptrate, airpres

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to generated weather CSV or None
        """
        try:
            import numpy as np
            import pandas as pd
            import xarray as xr

            # Find ERA5 forcing files
            forcing_dirs = [
                self.project_forcing_dir / 'basin_averaged_data',
                self.project_forcing_dir / 'merged_path',
                self.project_forcing_dir / 'raw_data',
            ]

            nc_files = []
            for forcing_dir in forcing_dirs:
                if forcing_dir.exists():
                    nc_files.extend(list(forcing_dir.glob('*.nc')))

            if not nc_files:
                self.logger.warning("No ERA5 forcing files found")
                return None

            self.logger.info(f"Found {len(nc_files)} ERA5 forcing files")

            # Load and concatenate if multiple files
            datasets = []
            for nc_file in sorted(nc_files)[:12]:  # Limit to 12 files for memory
                try:
                    ds = xr.open_dataset(nc_file)
                    datasets.append(ds)
                except Exception as exc:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Could not load {nc_file}: {exc}")

            if not datasets:
                return None

            # Concatenate datasets
            if len(datasets) > 1:
                ds = xr.concat(datasets, dim='time')
            else:
                ds = datasets[0]

            # Average across HRUs if spatial data
            if 'hru' in ds.dims:
                ds = ds.mean(dim='hru')

            # Extract variables
            times = pd.to_datetime(ds.time.values)

            # Temperature: Convert K to C
            if 'air_temperature' in ds:
                temp_c = ds['air_temperature'].values - 273.15
            else:
                temp_c = np.full(len(times), 20.0)  # Default

            # Relative Humidity: Calculate from specific humidity and pressure
            if 'specific_humidity' in ds and 'surface_air_pressure' in ds:
                spechum = ds['specific_humidity'].values
                airpres = ds['surface_air_pressure'].values
                # Convert specific humidity to relative humidity
                # e = q * P / (0.622 + 0.378 * q)
                # es = 611.2 * exp(17.67 * T / (T + 243.5))
                # RH = 100 * e / es
                e = spechum * airpres / (0.622 + 0.378 * spechum)
                es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
                rh = 100.0 * e / es
                rh = np.clip(rh, 0, 100)
            else:
                rh = np.full(len(times), 50.0)  # Default

            # Wind speed: m/s to km/h
            if 'wind_speed' in ds:
                ws_kmh = ds['wind_speed'].values * 3.6
            else:
                ws_kmh = np.full(len(times), 10.0)  # Default

            # Wind direction: Not in ERA5, use random or constant
            # Could derive from u10/v10 if available
            wd = np.random.uniform(0, 360, len(times))

            # Precipitation: mm/s to mm (hourly accumulation)
            if 'precipitation_flux' in ds:
                precip_mm = ds['precipitation_flux'].values * 3600  # mm/s to mm/hour
            else:
                precip_mm = np.zeros(len(times))

            # Create DataFrame
            weather_df = pd.DataFrame({
                'HOURLY': times.strftime('%d/%m/%Y %H:%M'),
                'TEMP': temp_c,
                'RH': rh,
                'WS': ws_kmh,
                'WD': wd,
                'PRECIP': precip_mm,
            })

            # Save to CSV
            weather_csv = self.ignacio_input_dir / 'era5_weather_stations.csv'
            weather_df.to_csv(weather_csv, index=False)

            self.logger.info(f"Converted ERA5 to weather CSV: {weather_csv}")
            self.logger.info(f"  Time range: {times[0]} to {times[-1]}")
            self.logger.info(f"  Records: {len(weather_df)}")

            return weather_csv

        except ImportError as e:
            self.logger.warning(f"Cannot convert ERA5 data (missing dependency): {e}")
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error converting ERA5 to weather CSV: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _generate_static_weather_csv(self, ignacio_config: Dict) -> Path:
        """
        Generate a static weather CSV for IGNACIO when no real data is available.

        Creates a simple hourly weather file for the simulation period with
        typical summer fire weather conditions.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to generated weather CSV
        """
        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd

        # Get simulation date from config or use default
        ignition_date = ignacio_config.get('ignition_date')
        if ignition_date:
            try:
                start_date = pd.to_datetime(ignition_date)
            except (ValueError, TypeError):
                start_date = datetime(2014, 7, 15)
        else:
            start_date = datetime(2014, 7, 15)

        # Generate 30 days of hourly data centered on ignition
        start = start_date - timedelta(days=15)
        end = start_date + timedelta(days=15)
        times = pd.date_range(start=start, end=end, freq='h')

        # Generate realistic diurnal patterns
        hours = np.array([t.hour for t in times])
        days = np.array([(t - start).days for t in times])

        # Temperature: diurnal cycle 10-30°C with some day-to-day variation
        base_temp = 20 + 5 * np.random.randn(len(set(days)))[days]
        diurnal_temp = 10 * np.sin((hours - 6) * np.pi / 12)
        temp_c = base_temp + diurnal_temp
        temp_c = np.clip(temp_c, 5, 35)

        # Relative Humidity: inverse of temperature pattern, 30-80%
        rh = 80 - 40 * (temp_c - 10) / 25
        rh = np.clip(rh, 20, 95)

        # Wind speed: higher during afternoon, 5-25 km/h
        base_ws = 10 + 5 * np.random.randn(len(set(days)))[days]
        diurnal_ws = 5 * np.sin((hours - 8) * np.pi / 12)
        ws_kmh = base_ws + diurnal_ws
        ws_kmh = np.clip(ws_kmh, 2, 40)

        # Wind direction: slowly varying, mostly westerly (240-300°)
        wd = 270 + 30 * np.sin(days * 0.5) + 20 * np.random.randn(len(times))
        wd = wd % 360

        # Precipitation: occasional, mostly zero
        precip_mm = np.zeros(len(times))
        # Add some rain events (low probability)
        rain_mask = np.random.random(len(times)) < 0.02
        precip_mm[rain_mask] = np.random.exponential(2, int(np.sum(rain_mask)))

        # Create DataFrame
        weather_df = pd.DataFrame({
            'HOURLY': times.strftime('%d/%m/%Y %H:%M'),
            'TEMP': temp_c.round(1),
            'RH': rh.round(1),
            'WS': ws_kmh.round(1),
            'WD': wd.round(0),
            'PRECIP': precip_mm.round(2),
        })

        # Save to CSV
        weather_csv = self.ignacio_input_dir / 'static_weather.csv'
        weather_df.to_csv(weather_csv, index=False)

        self.logger.info(f"Generated static weather CSV: {weather_csv}")
        self.logger.info(f"  Time range: {times[0]} to {times[-1]}")
        self.logger.info(f"  Records: {len(weather_df)}")

        return weather_csv

    def _prepare_ignition_data(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Prepare ignition point shapefile.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to ignition shapefile or None
        """
        # Check IGNACIO config
        ignition_path = ignacio_config.get('ignition_shapefile')
        if ignition_path and ignition_path != 'default':
            ignition_path = Path(ignition_path)
            if ignition_path.exists():
                self.logger.info(f"Using ignition shapefile: {ignition_path}")
                return ignition_path

        # Check WMFire config for shared ignition
        wmfire_ignition = self._get_config_value(
            lambda: self.config.model.wmfire.ignition_shapefile,
            default=None,
            dict_key='WMFIRE_IGNITION_SHAPEFILE'
        )
        if wmfire_ignition:
            wmfire_path = Path(wmfire_ignition)
            if wmfire_path.exists():
                self.logger.info(f"Using WMFire ignition shapefile: {wmfire_path}")
                return wmfire_path

        # Check default location
        ignition_dir = self.project_dir / 'shapefiles' / 'ignitions'
        if ignition_dir.exists():
            shapefiles = list(ignition_dir.glob('*.shp'))
            if shapefiles:
                self.logger.info(f"Using default ignition shapefile: {shapefiles[0]}")
                return shapefiles[0]

        self.logger.warning("No ignition shapefile found")
        return None

    def _get_season_from_date(self, date_str: Optional[str]) -> str:
        """
        Determine meteorological season from ignition date.

        Uses meteorological seasons for Northern Hemisphere:
        - Winter: December, January, February
        - Spring: March, April, May
        - Summer: June, July, August
        - Fall: September, October, November

        Args:
            date_str: Date string in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS

        Returns:
            Season string: 'Winter', 'Spring', 'Summer', or 'Fall'
        """
        if date_str is None:
            self.logger.warning("No ignition date provided, defaulting to 'Summer'")
            return 'Summer'

        try:
            from datetime import datetime

            # Parse the date string
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                self.logger.warning(f"Could not parse date '{date_str}', defaulting to 'Summer'")
                return 'Summer'

            month = dt.month

            # Meteorological seasons (Northern Hemisphere)
            if month in (12, 1, 2):
                return 'Winter'
            elif month in (3, 4, 5):
                return 'Spring'
            elif month in (6, 7, 8):
                return 'Summer'
            else:  # 9, 10, 11
                return 'Fall'

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Error determining season from date: {e}, defaulting to 'Summer'")
            return 'Summer'

    def _generate_ignacio_config(
        self,
        ignacio_config: Dict,
        terrain_paths: Dict[str, Optional[Path]],
        fuel_path: Optional[Path],
        weather_path: Optional[Path],
        ignition_path: Optional[Path]
    ) -> None:
        """
        Generate IGNACIO YAML configuration file.

        Args:
            ignacio_config: IGNACIO configuration dictionary
            terrain_paths: Dictionary with terrain file paths
            fuel_path: Path to fuel raster
            weather_path: Path to weather CSV
            ignition_path: Path to ignition shapefile
        """
        # Build IGNACIO config structure
        config = {
            'project': {
                'name': ignacio_config.get('project_name', self.config.domain.name),
                'description': f"IGNACIO simulation for {self.config.domain.name}",
                'output_dir': ignacio_config.get('output_dir', './output'),
                'random_seed': ignacio_config.get('random_seed', 42),
            },
            'crs': {
                'working_crs': ignacio_config.get('working_crs', 'EPSG:4326'),
                'output_crs': ignacio_config.get('output_crs', 'EPSG:4326'),
            },
            'terrain': {
                'dem_path': str(terrain_paths['dem_path']) if terrain_paths['dem_path'] else None,
                'slope_path': str(terrain_paths['slope_path']) if terrain_paths['slope_path'] else None,
                'aspect_path': str(terrain_paths['aspect_path']) if terrain_paths['aspect_path'] else None,
            },
            'fuel': {
                'source_type': 'raster' if fuel_path else 'constant',
                'path': str(fuel_path) if fuel_path else None,
                'non_fuel_codes': ignacio_config.get('non_fuel_codes', [0, 11, 12, 17, 255, -9999]),
                'fuel_lookup': {
                    # Map MODIS IGBP land classes to FBP fuel types
                    1: 'C-2',   # Evergreen Needleleaf Forest -> Boreal Spruce
                    2: 'C-2',   # Evergreen Broadleaf Forest
                    3: 'D-1',   # Deciduous Needleleaf Forest
                    4: 'D-1',   # Deciduous Broadleaf Forest -> Aspen
                    5: 'M-1',   # Mixed Forest -> Mixedwood
                    6: 'O-1a',  # Closed Shrublands -> Grass
                    7: 'O-1a',  # Open Shrublands -> Grass
                    8: 'O-1a',  # Woody Savannas
                    9: 'O-1a',  # Savannas
                    10: 'O-1b', # Grasslands -> Standing grass
                    12: 'O-1a', # Croplands
                    14: 'O-1a', # Cropland/Natural Vegetation
                    16: 'NF',   # Barren/Sparse Vegetation
                },
            },
            'ignition': {
                'source_type': 'shapefile' if ignition_path else 'point',
                'point_path': str(ignition_path) if ignition_path else None,
                'cause': ignacio_config.get('ignition_cause', 'Lightning'),
                'season': self._get_season_from_date(ignacio_config.get('ignition_date')),
                'n_iterations': ignacio_config.get('n_iterations', 1),
                'escaped_fire_distribution': {1: 1.0},
            },
            'weather': {
                'station_path': str(weather_path) if weather_path else None,
                'weather_path': None,
                'calculate_fwi': ignacio_config.get('calculate_fwi', True),
                'fwi_latitude': ignacio_config.get('fwi_latitude', 51.35),
                'weather_columns': {
                    'datetime': 'HOURLY',
                    'temperature': 'TEMP',
                    'relative_humidity': 'RH',
                    'wind_speed': 'WS',
                    'wind_direction': 'WD',
                    'precipitation': 'PRECIP',
                },
                'datetime_format': '%d/%m/%Y %H:%M',
                'isi_thresholds': {
                    'moderate': 0.0,
                    'high': 3.0,
                    'extreme': 6.0,
                },
                'filter_conditions': ['moderate', 'high', 'extreme'],
                'spread_event_lambda': 3.76,
            },
            'fbp': {
                'defaults': {
                    'ffmc': ignacio_config.get('default_ffmc', 90.0),
                    'dmc': ignacio_config.get('default_dmc', 40.0),
                    'dc': ignacio_config.get('default_dc', 200.0),
                    'isi': ignacio_config.get('default_isi', 8.0),
                    'bui': ignacio_config.get('default_bui', 60.0),
                    'fwi': ignacio_config.get('default_fwi', 20.0),
                },
                'fmc': ignacio_config.get('fmc', 100.0),
                'curing': ignacio_config.get('curing', 85.0),
                'slope_factor': 0.5,
                'backing_fraction': 0.2,
                'length_to_breadth': 2.0,
            },
            'simulation': {
                'dt': ignacio_config.get('dt', 1.0),
                'max_duration': ignacio_config.get('max_duration', 720),  # 12 hours
                'n_vertices': ignacio_config.get('n_vertices', 300),
                'initial_radius': ignacio_config.get('initial_radius', 10.0),
                'store_every': ignacio_config.get('store_every', 30),
                'min_ros': ignacio_config.get('min_ros', 0.01),
                'time_varying_weather': ignacio_config.get('time_varying_weather', True),
                'start_datetime': ignacio_config.get('ignition_date', '2014-08-15 12:00:00'),
                'default_start_hour': 12,
            },
            'output': {
                'save_perimeters': ignacio_config.get('save_perimeters', True),
                'save_ros_grids': ignacio_config.get('save_ros_grids', True),
                'perimeter_format': ignacio_config.get('perimeter_format', 'shapefile'),
                'generate_plots': ignacio_config.get('generate_plots', True),
                'log_level': 'INFO',
            },
        }

        # Write YAML config
        with open(self.ignacio_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"IGNACIO config written: {self.ignacio_config_path}")
