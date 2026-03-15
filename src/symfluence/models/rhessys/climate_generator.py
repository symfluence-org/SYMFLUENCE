# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Climate File Generator

Handles generation of RHESSys-compatible climate input files from forcing data.
Extracted from RHESSysPreprocessor for better organization and testability.

RHESSys uses text-based climate files with format:
    year month day hour value
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.data.utils.variable_utils import VariableHandler

logger = logging.getLogger(__name__)


class RHESSysClimateGenerator:
    """
    Generates RHESSys-compatible climate input files.

    Handles:
    - Loading forcing data from various sources
    - Converting units and aggregating to daily
    - Computing derived variables (relative humidity from specific humidity)
    - Writing base station and climate files in RHESSys format
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.project_dir = Path(project_dir)
        self.domain_name = domain_name
        self.logger = logger or logging.getLogger(__name__)

        # Setup paths — climate files are forcing data
        self.climate_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'RHESSys_input' / 'clim'
        self.forcing_basin_path = resolve_data_subdir(self.project_dir, 'forcing') / 'basin_averaged_data'
        self.forcing_raw_path = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data'

        # Get forcing dataset info
        forcing_ds = config.get('FORCING_DATASET', 'ERA5')
        self.forcing_dataset = forcing_ds.upper() if forcing_ds else 'ERA5'

    def generate_climate_files(
        self,
        start_date: datetime,
        end_date: datetime,
        catchment_path: Optional[Path] = None
    ) -> bool:
        """
        Generate all climate files for the simulation period.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            catchment_path: Path to catchment shapefile (for coordinates)

        Returns:
            True if successful
        """
        self.logger.info("Generating climate files...")
        self.climate_dir.mkdir(parents=True, exist_ok=True)

        try:
            ds = self._load_forcing_data(start_date, end_date)
        except FileNotFoundError as e:
            self.logger.warning(f"Could not load forcing data: {e}")
            self.logger.info("Creating synthetic climate data for testing")
            self._create_synthetic_climate(start_date, end_date)
            return True

        # Initialize variable handler
        VariableHandler(self.config, self.logger, self.forcing_dataset, 'RHESSys')

        # Get variable names from dataset
        precip_var = self._find_variable(ds, ['pr', 'precipitation', 'PREC', 'precip', 'precipitation_flux'])
        temp_var = self._find_variable(ds, ['tas', 't2m', 'temp', 'air_temperature', 'TEMP', 'temperature'])
        tmax_var = self._find_variable(ds, ['tasmax', 'tmax', 'TMAX'])
        tmin_var = self._find_variable(ds, ['tasmin', 'tmin', 'TMIN'])

        # Additional variables for ET calculation
        swrad_var = self._find_variable(ds, ['surface_downwelling_shortwave_flux', 'rsds', 'swdown', 'ssrd', 'shortwave_radiation', 'Kdown'])
        lwrad_var = self._find_variable(ds, ['surface_downwelling_longwave_flux', 'rlds', 'lwdown', 'strd', 'longwave_radiation', 'Ldown'])
        wind_var = self._find_variable(ds, ['wind_speed', 'wind', 'sfcWind', 'wind_speed', 'ws', 'u10', 'v10'])
        spechum_var = self._find_variable(ds, ['specific_humidity', 'huss', 'specific_humidity', 'q'])
        airpres_var = self._find_variable(ds, ['surface_air_pressure', 'ps', 'sp', 'air_pressure', 'pressure'])

        # Extract data
        time_coord = ds['time'].values
        dates = pd.to_datetime(time_coord)

        # Process each variable
        precip = self._process_precipitation(ds, precip_var, dates)
        temp, tmax, tmin = self._process_temperature(ds, temp_var, tmax_var, tmin_var, dates)

        # Process optional variables
        wind = self._process_wind(ds, wind_var, dates)
        rh = self._process_relative_humidity(ds, spechum_var, airpres_var, temp_var, dates)
        kdown = self._process_radiation(ds, swrad_var, dates, 'shortwave')
        ldown = self._process_radiation(ds, lwrad_var, dates, 'longwave')

        # Compute rain duration from hourly precip data BEFORE daily aggregation
        # daytime_rain_duration = hours per day with precipitation > 0
        # RHESSys reads this in hours and converts to seconds internally
        rain_duration_hourly = None
        if precip is not None:
            rain_duration_hourly = (precip > 0).astype(float)

        # Aggregate to daily
        df = pd.DataFrame({
            'precip': precip,
            'temp': temp,
            'tmax': tmax,
            'tmin': tmin,
            'wind': wind,
            'rh': rh,
            'kdown': kdown,
            'ldown': ldown,
        }, index=dates)
        if rain_duration_hourly is not None:
            df['rain_duration'] = rain_duration_hourly

        daily_df = self._aggregate_to_daily(df)

        # Write climate files first (before base station file, which checks for their existence)
        base_name = f"{self.domain_name}_base"

        # RHESSys expects precipitation in METERS/day in climate files
        # RHESSys internally multiplies by 1000 to convert to mm
        # Daily aggregation produces mm/day, so divide by 1000 to get m/day
        precip_m_day = daily_df['precip'].values / 1000.0
        self._write_climate_file(f"{base_name}.rain", daily_df.index, precip_m_day)
        self._write_climate_file(f"{base_name}.tmax", daily_df.index, daily_df['tmax'].values)
        self._write_climate_file(f"{base_name}.tmin", daily_df.index, daily_df['tmin'].values)
        self._write_climate_file(f"{base_name}.tavg", daily_df.index, daily_df['temp'].values)

        # Write optional climate files if data available
        if wind is not None and not np.all(np.isnan(daily_df['wind'])):
            self._write_climate_file(f"{base_name}.wind", daily_df.index, daily_df['wind'].values)
        if rh is not None and not np.all(np.isnan(daily_df['rh'])):
            self._write_climate_file(f"{base_name}.relative_humidity", daily_df.index, daily_df['rh'].values)
        if kdown is not None and not np.all(np.isnan(daily_df['kdown'])):
            # Split total shortwave radiation into direct (60%) and diffuse (40%)
            # This partitioning is needed for RHESSys canopy radiation calculations
            # Convert from W/m² (daily mean) to kJ/(m²*day) as required by RHESSys
            # 1 W/m² = 86.4 kJ/(m²*day) [1 W = 1 J/s, * 86400 s/day / 1000 J/kJ]
            kdown_total_kj = daily_df['kdown'].values * 86.4
            self._write_climate_file(f"{base_name}.Kdown_direct", daily_df.index, kdown_total_kj * 0.6)
            self._write_climate_file(f"{base_name}.Kdown_diffuse", daily_df.index, kdown_total_kj * 0.4)
        if ldown is not None and not np.all(np.isnan(daily_df['ldown'])):
            # Convert from W/m² to kJ/(m²*day) as required by RHESSys
            ldown_kj = daily_df['ldown'].values * 86.4
            self._write_climate_file(f"{base_name}.Ldown", daily_df.index, ldown_kj)

        # Write rain duration (hours/day with precipitation > 0)
        # RHESSys reads daytime_rain_duration in hours, converts to seconds internally
        # Without this, RHESSys defaults to 86400s (full day) on rainy days,
        # leaving zero time for transpiration
        if 'rain_duration' in daily_df.columns:
            self._write_climate_file(
                f"{base_name}.daytime_rain_duration",
                daily_df.index,
                daily_df['rain_duration'].values
            )

        # Write base station file AFTER climate files so it can detect which ones exist
        self._write_base_station_file(base_name, 1, daily_df.index[0], catchment_path)

        ds.close()
        self.logger.info(f"Climate files written to {self.climate_dir}")
        return True

    def _load_forcing_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> xr.Dataset:
        """Load basin-averaged forcing data from available sources."""
        # Search multiple potential locations
        search_paths = [
            self.forcing_basin_path,
            resolve_data_subdir(self.project_dir, 'forcing') / 'merged_path',
            resolve_data_subdir(self.project_dir, 'forcing') / 'SUMMA_input',
            self.forcing_raw_path,
        ]

        forcing_files = []
        for path in search_paths:
            if path.exists():
                files = list(path.glob("*.nc"))
                if files:
                    self.logger.info(f"Found {len(files)} forcing files in {path}")
                    forcing_files = files
                    break

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in any of: {search_paths}")

        self.logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')
        except ValueError as e:
            self.logger.warning(f"Failed with combine='by_coords': {e}. Retrying...")
            try:
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
            except (FileNotFoundError, OSError, ValueError, KeyError):
                self.logger.warning("Failed to concat. Attempting merge...")
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        # Subset to simulation period
        ds = ds.sel(time=slice(start_date, end_date))

        return ds

    def _find_variable(self, ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
        """Find first matching variable name in dataset."""
        for var in candidates:
            if var in ds.data_vars:
                return var
        return None

    def _basin_average(self, data_array) -> np.ndarray:
        """Average across all spatial dimensions (HRU, GRU, etc.)"""
        values = data_array.values
        if values.ndim > 1:
            spatial_axes = tuple(range(1, values.ndim))
            values = np.nanmean(values, axis=spatial_axes)
        return values.flatten()

    def _process_precipitation(
        self,
        ds: xr.Dataset,
        precip_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Process precipitation variable.

        Converts all source units to mm per timestep (hour) so that
        daily aggregation (sum) results in mm per day, which is what
        RHESSys expects in climate station files.
        """
        if precip_var:
            precip = self._basin_average(ds[precip_var])
            units = str(ds[precip_var].attrs.get('units', '')).lower()

            # If it's a rate (e.g. kg/m2/s, mm/s, m/s)
            if 's-1' in units or 's^-1' in units or '/s' in units:
                # kg/m2/s is equivalent to mm/s (density of water = 1000 kg/m3)
                if 'kg' in units or 'mm' in units:
                    self.logger.info(
                        f"Interpreting '{units}' as mm/s"
                    )
                    precip = precip * 3600.0
                elif 'm s-1' in units or 'm/s' in units:
                    # Ambiguous: could be genuine m/s or convention for mm/s.
                    # Use magnitude to distinguish: typical precip rates are
                    # ~1e-5 to 1e-4 m/s (genuine) vs ~0.01 to 0.1 mm/s.
                    raw_mean = float(np.nanmean(precip[precip > 0])) if np.any(precip > 0) else 0.0
                    if raw_mean < 0.001:
                        # Values < 0.001 are consistent with genuine m/s
                        # (e.g. 1e-5 m/s = 0.01 mm/s = 0.864 mm/day)
                        self.logger.info(
                            f"Interpreting '{units}' as genuine m/s (mean wet rate={raw_mean:.6f}). "
                            f"Converting m/s -> mm/s (* 1000) -> mm/hour (* 3600)."
                        )
                        precip = precip * 1000.0 * 3600.0
                    else:
                        # Values >= 0.001 are consistent with mm/s convention
                        self.logger.info(
                            f"Interpreting '{units}' as mm/s by convention (mean wet rate={raw_mean:.6f}). "
                            f"Converting mm/s -> mm/hour (* 3600)."
                        )
                        precip = precip * 3600.0
                else:
                    # Other rate units - assume mm/s for safety
                    self.logger.info(
                        f"Interpreting unknown rate units '{units}' as mm/s"
                    )
                    precip = precip * 3600.0

                # Sanity check: verify precipitation magnitude after conversion
                mean_precip = float(np.nanmean(precip))
                if mean_precip > 10.0:  # 10 mm/hour is extremely high as a mean
                    self.logger.warning(
                        f"Precipitation mean ({mean_precip:.4f} mm/hour) seems high. "
                        f"Original units were '{units}'. Verify unit interpretation."
                    )
                if mean_precip > 100.0:  # 100 mm/hour is physically impossible as mean
                    raise ValueError(
                        f"Precipitation rate {mean_precip:.4f} mm/hour is physically impossible. "
                        f"Check if units '{units}' should be interpreted differently."
                    )
                if mean_precip < 0.001:  # Suspiciously low - likely wrong conversion
                    self.logger.warning(
                        f"Precipitation mean ({mean_precip:.6f} mm/hour) is suspiciously low. "
                        f"Original units were '{units}'. Possible 1000x underestimate."
                    )
            # If it's already a depth per timestep (e.g. ERA5 'm' accumulated)
            elif 'm' in units:
                if 'mm' in units:
                    # Already in mm per timestep - no conversion needed
                    pass
                else:
                    # Assume m per timestep - convert to mm
                    precip = precip * 1000.0

            return precip
        else:
            self.logger.warning("No precipitation variable found, using zeros")
            return np.zeros(len(dates))

    def _process_temperature(
        self,
        ds: xr.Dataset,
        temp_var: Optional[str],
        tmax_var: Optional[str],
        tmin_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process temperature variables (mean, max, min)."""
        # Get temperature
        if temp_var:
            temp = self._basin_average(ds[temp_var])
            # Convert K to C if needed
            if np.nanmean(temp) > 100:
                temp = temp - 273.15
        else:
            self.logger.warning("No temperature variable found")
            temp = np.full(len(dates), 15.0)

        # Get tmax
        if tmax_var:
            tmax = self._basin_average(ds[tmax_var])
            if np.nanmean(tmax) > 100:
                tmax = tmax - 273.15
        else:
            # No tmax variable: use hourly temp directly.
            # Daily aggregation (max) will extract the diurnal maximum.
            # Previously this added +5C to each hourly value, which inflated
            # daily tmax by ~5C above the actual diurnal peak (double-counting
            # since hourly data already captures the diurnal cycle).
            tmax = temp.copy()

        # Get tmin
        if tmin_var:
            tmin = self._basin_average(ds[tmin_var])
            if np.nanmean(tmin) > 100:
                tmin = tmin - 273.15
        else:
            # No tmin variable: use hourly temp directly.
            # Daily aggregation (min) will extract the diurnal minimum.
            tmin = temp.copy()

        return temp, tmax, tmin

    def _process_wind(
        self,
        ds: xr.Dataset,
        wind_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> Optional[np.ndarray]:
        """Process wind speed variable."""
        if wind_var:
            return self._basin_average(ds[wind_var])
        return None

    def _process_relative_humidity(
        self,
        ds: xr.Dataset,
        spechum_var: Optional[str],
        airpres_var: Optional[str],
        temp_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> Optional[np.ndarray]:
        """Calculate relative humidity from specific humidity if available."""
        if spechum_var and airpres_var and temp_var:
            try:
                q = self._basin_average(ds[spechum_var])
                p = self._basin_average(ds[airpres_var])
                t = self._basin_average(ds[temp_var])

                # Convert to Celsius if needed
                if np.nanmean(t) > 100:
                    t = t - 273.15

                # Calculate saturation vapor pressure (Tetens formula)
                es = 6.112 * np.exp(17.67 * t / (t + 243.5)) * 100  # Pa

                # Calculate actual vapor pressure from specific humidity
                # q = 0.622 * e / (p - 0.378 * e)
                e = q * p / (0.622 + 0.378 * q)

                # Relative humidity as decimal (0-1) for RHESSys
                # RHESSys expects decimal format, not percentage
                rh = e / es
                rh = np.clip(rh, 0, 1)

                return rh
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Could not calculate relative humidity: {e}")
        return None

    def _process_radiation(
        self,
        ds: xr.Dataset,
        rad_var: Optional[str],
        dates: pd.DatetimeIndex,
        rad_type: str
    ) -> Optional[np.ndarray]:
        """Process radiation variable."""
        if rad_var:
            rad = self._basin_average(ds[rad_var])
            # Convert J/m2 to W/m2 if needed (ERA5 often in J/m2)
            if np.nanmean(rad) > 10000:
                timestep_hours = 1  # Assume hourly data
                rad = rad / (3600 * timestep_hours)
            return rad
        return None

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sub-daily data to daily values."""
        agg_dict = {
            'precip': 'sum',  # Sum precipitation
            'temp': 'mean',   # Mean temperature
            'tmax': 'max',    # Max of max temps
            'tmin': 'min',    # Min of min temps
            'wind': 'mean',   # Mean wind
            'rh': 'mean',     # Mean relative humidity
            'kdown': 'mean',  # Mean radiation
            'ldown': 'mean',
        }
        if 'rain_duration' in df.columns:
            agg_dict['rain_duration'] = 'sum'  # Sum of hourly flags = hours with rain
        daily = df.resample('D').agg(agg_dict)
        return daily

    def _write_base_station_file(
        self,
        base_name: str,
        station_id: int,
        start_date: pd.Timestamp,
        catchment_path: Optional[Path] = None
    ) -> None:
        """Write RHESSys base station file."""
        base_file = self.climate_dir / f"{base_name}"

        # Get centroid coordinates from basin shapefile
        # Set base station elevation = zone elevation (elev_mean) so no temperature
        # lapse is applied. For lumped mode, forcing data already represents
        # basin-average conditions.
        lon, lat, elev = -115.0, 51.0, None
        if catchment_path and catchment_path.exists():
            try:
                gdf = gpd.read_file(catchment_path)
                gdf_ll = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf
                minx, miny, maxx, maxy = gdf_ll.total_bounds
                lon0 = (minx + maxx) / 2
                lat0 = (miny + maxy) / 2
                utm_zone = int((lon0 + 180) / 6) + 1
                hemisphere = 'north' if lat0 >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf_ll.to_crs(utm_crs)
                centroid_proj = gdf_proj.geometry.centroid.iloc[0]
                centroid_ll = gpd.GeoSeries([centroid_proj], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
                lon, lat = float(centroid_ll.x), float(centroid_ll.y)
                elev_col = self.config.get('CATCHMENT_SHP_ELEV', 'elev_mean')
                if elev_col in gdf.columns:
                    elev = float(gdf[elev_col].iloc[0])
            except (FileNotFoundError, KeyError, IndexError, ValueError):
                pass

        # Fallback: read zone elevation from worldfile to ensure base station matches
        if elev is None:
            try:
                import re
                world_file = self.project_dir / 'settings' / 'RHESSys' / 'worldfiles' / f"{base_name.replace('_base', '')}.world"
                if world_file.exists():
                    with open(world_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.match(r'\s*([\d\.\-]+)\s+z\b', line)
                            if match:
                                elev = float(match.group(1))
                                self.logger.info(f"Base station elevation from worldfile: {elev:.1f} m")
                                break
            except Exception:  # noqa: BLE001 — model execution resilience
                pass

        if elev is None:
            elev = 1500.0
            self.logger.warning(
                "Could not determine catchment elevation for base station. "
                "Using fallback 1500m. This may cause incorrect lapse rate corrections."
            )

        # Full path to climate file prefix
        climate_prefix = self.climate_dir / base_name

        # Build list of non-critical daily sequences
        non_critical_sequences = []
        for suffix in ['wind', 'relative_humidity', 'Kdown_direct', 'Kdown_diffuse', 'Ldown', 'tavg', 'daytime_rain_duration']:
            if (self.climate_dir / f"{base_name}.{suffix}").exists():
                non_critical_sequences.append(suffix)

        num_sequences = len(non_critical_sequences)
        sequence_lines = "\n".join(non_critical_sequences) if non_critical_sequences else ""

        content = f"""{station_id}\tbase_station_id
{lon:.4f}\tx_coordinate
{lat:.4f}\ty_coordinate
{elev:.1f}\tz_coordinate
3.5\teffective_lai
2.0\tscreen_height
none\tannual_climate_prefix
0\tnumber_non_critical_annual_sequences
none\tmonthly_climate_prefix
0\tnumber_non_critical_monthly_sequences
{climate_prefix}\tdaily_climate_prefix
{num_sequences}\tnumber_non_critical_daily_sequences
{sequence_lines}
none\thourly_climate_prefix
0\tnumber_non_critical_hourly_sequences
"""
        base_file.write_text(content, encoding='utf-8')
        self.logger.info(f"Base station file written: {base_file}")

    def _write_climate_file(
        self,
        filename: str,
        dates: pd.DatetimeIndex,
        values: np.ndarray
    ) -> None:
        """
        Write a single RHESSys climate file.

        Format:
        - Line 1: start date (year month day hour)
        - Lines 2+: one value per line
        """
        filepath = self.climate_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            start_date = dates[0]
            f.write(f"{start_date.year} {start_date.month} {start_date.day} 1\n")

            for value in values:
                if np.isnan(value):
                    f.write("0.0000\n")
                else:
                    f.write(f"{value:.4f}\n")

        self.logger.debug(f"Climate file written: {filepath}")

    def _create_synthetic_climate(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Create synthetic climate data for testing."""
        self.climate_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.date_range(start_date, end_date, freq='D')

        # Simple synthetic data
        precip = np.random.exponential(2, len(dates))
        temp = 10 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates))
        tmax = temp + 5 + np.random.normal(0, 1, len(dates))
        tmin = temp - 5 + np.random.normal(0, 1, len(dates))

        base_name = f"{self.domain_name}_base"
        self._write_base_station_file(base_name, 1, dates[0])
        self._write_climate_file(f"{base_name}.rain", dates, precip)
        self._write_climate_file(f"{base_name}.tmax", dates, tmax)
        self._write_climate_file(f"{base_name}.tmin", dates, tmin)
        self._write_climate_file(f"{base_name}.tavg", dates, temp)

        self.logger.info("Synthetic climate files created")
