# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS LAI/FPAR Observation Handler

Processes MODIS MCD15A2H/MOD15A2H/MYD15A2H Leaf Area Index and Fraction of
Photosynthetically Active Radiation data for use in hydrological modeling.

Handles data from:
1. Cloud acquisition via earthaccess/CMR (consolidated NetCDF + CSV)
2. Cloud acquisition via AppEEARS (multi-file NetCDF)
3. Pre-downloaded files (legacy format)

LAI is critical for:
- Evapotranspiration partitioning
- Interception modeling
- Vegetation phenology tracking
- Carbon cycle modeling
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler

# Scale factors and valid ranges for MODIS LAI/FPAR
LAI_SCALE_FACTOR = 0.1  # To get LAI in m²/m²
FPAR_SCALE_FACTOR = 0.01  # To get FPAR as fraction (0-1)
LAI_VALID_RANGE = (0, 100)  # Valid DN range (0-10 m²/m² after scaling)
FPAR_VALID_RANGE = (0, 100)  # Valid DN range (0-1 after scaling)
FILL_VALUE = 255


@R.observation_handlers.add('modis_lai')
@R.observation_handlers.add('mcd15')
@R.observation_handlers.add('lai')
class MODISLAIHandler(BaseObservationHandler):
    """
    Handles MODIS LAI/FPAR data processing.

    Processes 8-day composite LAI and FPAR data to basin-averaged
    time series with quality filtering and optional daily interpolation.

    Supports two input formats:
    - Consolidated NetCDF/CSV from earthaccess pathway (domain-mean timeseries)
    - Multi-file spatial NetCDF from AppEEARS pathway (requires spatial averaging)

    Configuration:
        MODIS_LAI_DIR: Directory containing MODIS LAI data
        MODIS_LAI_CONVERT_TO_DAILY: Interpolate 8-day to daily (default: True)
        MODIS_LAI_MIN_QUALITY: Minimum QC quality (default: use main algorithm)
        MODIS_LAI_INTERPOLATION: Interpolation method ('linear', 'spline')
    """

    obs_type = "lai"
    source_name = "NASA_MODIS"

    def acquire(self) -> Path:
        """Acquire MODIS LAI data via cloud acquisition."""
        lai_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "vegetation" / "modis_lai", dict_key='MODIS_LAI_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        # Check for existing processed file first
        processed_file = (
            self.project_observations_dir / "vegetation" / "preprocessed"
            / f"{self.domain_name}_modis_lai_processed.csv"
        )
        if processed_file.exists() and not self._get_config_value(
            lambda: self.config.system.force_run_all_steps,
            dict_key='FORCE_RUN_ALL_STEPS',
        ):
            self.logger.info(f"Using existing processed MODIS LAI: {processed_file}")
            return processed_file.parent

        # Check for existing raw data (earthaccess consolidated NC or CSV)
        raw_nc = lai_dir / f"{self.domain_name}_MODIS_LAI.nc"
        raw_csv = lai_dir / f"{self.domain_name}_MODIS_LAI_timeseries.csv"
        has_earthaccess = raw_nc.exists() or raw_csv.exists()

        # Check for AppEEARS multi-file output
        has_appeears = lai_dir.exists() and (
            any(lai_dir.glob("*LAI*.nc")) or any(lai_dir.glob("*Lai*.nc"))
        )

        if (has_earthaccess or has_appeears) and not force_download:
            self.logger.info(f"Using existing MODIS LAI data in {lai_dir}")
            return lai_dir

        # Run cloud acquisition
        self.logger.info("Acquiring MODIS LAI data...")
        try:
            from ...acquisition.handlers.modis_lai import MODISLAIAcquirer
            acquirer = MODISLAIAcquirer(self.config, self.logger)
            result = acquirer.download(lai_dir)
            self.logger.info(f"MODIS LAI acquisition complete: {result}")
            # Result may be a file (earthaccess) or directory (appeears)
            return lai_dir
        except ImportError as e:
            self.logger.warning(f"MODIS LAI acquirer not available: {e}")
            raise
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, LookupError) as e:
            self.logger.error(f"MODIS LAI acquisition failed: {e}")
            raise

    def process(self, input_path: Path) -> Path:
        """
        Process MODIS LAI/FPAR data for the current domain.

        Handles:
        1. Consolidated NetCDF from earthaccess (timeseries, no spatial dims)
        2. CSV timeseries from earthaccess
        3. Multi-file spatial NetCDF from AppEEARS (requires spatial averaging)

        Args:
            input_path: Path to MODIS LAI data directory or file

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing MODIS LAI/FPAR for domain: {self.domain_name}")

        output_dir = self.project_observations_dir / "vegetation" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_lai_processed.csv"

        # Check for existing processed file
        if output_file.exists() and not self._get_config_value(
            lambda: self.config.system.force_run_all_steps,
            dict_key='FORCE_RUN_ALL_STEPS',
        ):
            self.logger.info(f"Using existing processed file: {output_file}")
            return output_file

        df = None

        # Determine input directory
        if input_path.is_file():
            search_dir = input_path.parent
        else:
            search_dir = input_path

        # 1. Try consolidated NetCDF from earthaccess pathway
        consolidated_nc = search_dir / f"{self.domain_name}_MODIS_LAI.nc"
        if consolidated_nc.exists():
            df = self._process_consolidated_netcdf(consolidated_nc)

        # 2. Try CSV timeseries from earthaccess pathway
        if df is None:
            csv_pattern = f"{self.domain_name}_MODIS_LAI_timeseries.csv"
            csv_file = search_dir / csv_pattern
            if csv_file.exists():
                df = self._process_timeseries_csv(csv_file)

        # 3. Try multi-file spatial NetCDF (AppEEARS pathway)
        if df is None:
            nc_files = list(search_dir.glob("*Lai*.nc")) + list(search_dir.glob("*LAI*.nc"))
            nc_files += list(search_dir.glob("*MCD15*.nc"))
            # Exclude the consolidated file we already tried
            nc_files = [f for f in nc_files if f != consolidated_nc]
            if nc_files:
                df = self._process_appeears_netcdf_files(nc_files)

        if df is None or df.empty:
            self.logger.warning("No MODIS LAI data could be processed")
            return input_path

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.index.name = 'datetime'

        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Interpolate to daily if requested
        if self._get_config_value(lambda: None, default=True, dict_key='MODIS_LAI_CONVERT_TO_DAILY'):
            df = self._interpolate_to_daily(df)

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        df.index.name = 'datetime'
        df.to_csv(output_file)
        self.logger.info(f"MODIS LAI processing complete: {output_file}")

        return output_file

    # ===== Consolidated earthaccess output processing =====

    def _process_consolidated_netcdf(self, nc_path: Path) -> Optional[pd.DataFrame]:
        """Process consolidated NetCDF from earthaccess (timeseries, no spatial dims)."""
        self.logger.info(f"Processing consolidated NetCDF: {nc_path}")

        try:
            ds = xr.open_dataset(nc_path)

            records = {}

            # Extract LAI
            lai_var = self._find_variable(ds, ['LAI', 'Lai_500m', 'lai', 'Lai'])
            if lai_var:
                da = ds[lai_var]
                # If it has spatial dims, compute spatial mean
                spatial_dims = [d for d in da.dims if d not in ['time', 'date']]
                if spatial_dims:
                    da = da.mean(dim=spatial_dims, skipna=True)
                records['lai'] = da.to_series()

            # Extract FPAR
            fpar_var = self._find_variable(ds, ['FPAR', 'Fpar_500m', 'fpar', 'Fpar'])
            if fpar_var:
                da = ds[fpar_var]
                spatial_dims = [d for d in da.dims if d not in ['time', 'date']]
                if spatial_dims:
                    da = da.mean(dim=spatial_dims, skipna=True)
                records['fpar'] = da.to_series()

            ds.close()

            if not records:
                self.logger.warning(f"No LAI/FPAR variables found in {nc_path}")
                return None

            df = pd.DataFrame(records)
            df.index = pd.to_datetime(df.index)
            df.index.name = 'datetime'
            return df

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing consolidated NetCDF: {e}")
            return None

    def _process_timeseries_csv(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Process CSV timeseries from earthaccess pathway."""
        self.logger.info(f"Processing timeseries CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            df.index.name = 'datetime'
            return df
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing timeseries CSV: {e}")
            return None

    # ===== AppEEARS multi-file spatial processing =====

    def _process_appeears_netcdf_files(
        self,
        nc_files: List[Path]
    ) -> Optional[pd.DataFrame]:
        """Process multi-file spatial NetCDF from AppEEARS."""
        self.logger.info(f"Processing {len(nc_files)} AppEEARS NetCDF files")

        basin_gdf = self._load_catchment_shapefile()

        results: dict[str, list] = {'lai': [], 'fpar': [], 'datetime': []}

        for nc_file in nc_files:
            try:
                lai_vals, fpar_vals, times = self._process_spatial_netcdf(
                    nc_file, basin_gdf
                )
                if lai_vals is not None:
                    results['lai'].extend(lai_vals)
                    results['fpar'].extend(fpar_vals)
                    results['datetime'].extend(times)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Failed to process {nc_file.name}: {e}")

        if not results['datetime']:
            return None

        df = pd.DataFrame(results)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        return df

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """Load the catchment shapefile for spatial masking.

        Delegates to the shared resolver, which handles the nested
        discretization layout and river_basins fallback. Returns None when
        no basin is found (callers fall back to the bounding box).
        """
        return self._load_catchment_gdf()

    def _process_spatial_netcdf(
        self,
        nc_file: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ):
        """Process a spatial NetCDF file containing LAI/FPAR data (AppEEARS format)."""
        ds = xr.open_dataset(nc_file)

        # Find LAI and FPAR variables
        lai_var = self._find_variable(ds, ['Lai_500m', 'LAI', 'lai', 'Lai'])
        fpar_var = self._find_variable(ds, ['Fpar_500m', 'FPAR', 'fpar', 'Fpar'])
        qc_var = self._find_variable(ds, ['FparLai_QC', 'QC', 'qc'])

        if not lai_var and not fpar_var:
            ds.close()
            return None, None, None

        # Get time dimension
        time_dim = self._find_coord(ds, ['time', 'date'])

        lai_vals = []
        fpar_vals = []
        times = []

        if time_dim:
            time_vals = ds[time_dim].values
        else:
            time_vals = [self._extract_date_from_filename(nc_file.name)]

        for i, t in enumerate(time_vals):
            # Extract LAI
            if lai_var:
                if time_dim:
                    da_lai = ds[lai_var].isel({time_dim: i})
                else:
                    da_lai = ds[lai_var]

                # Apply QC filter if available
                if qc_var:
                    qc_da = ds[qc_var].isel({time_dim: i}) if time_dim else ds[qc_var]
                    da_lai = self._apply_qc_filter(da_lai, qc_da)

                # Extract basin mean and scale
                lai_val = self._extract_basin_mean(da_lai, basin_gdf, LAI_VALID_RANGE)
                if lai_val is not None:
                    lai_val = lai_val * LAI_SCALE_FACTOR
                else:
                    lai_val = np.nan
            else:
                lai_val = np.nan

            # Extract FPAR
            if fpar_var:
                if time_dim:
                    da_fpar = ds[fpar_var].isel({time_dim: i})
                else:
                    da_fpar = ds[fpar_var]

                if qc_var:
                    qc_da = ds[qc_var].isel({time_dim: i}) if time_dim else ds[qc_var]
                    da_fpar = self._apply_qc_filter(da_fpar, qc_da)

                fpar_val = self._extract_basin_mean(da_fpar, basin_gdf, FPAR_VALID_RANGE)
                if fpar_val is not None:
                    fpar_val = fpar_val * FPAR_SCALE_FACTOR
                else:
                    fpar_val = np.nan
            else:
                fpar_val = np.nan

            lai_vals.append(lai_val)
            fpar_vals.append(fpar_val)
            times.append(pd.to_datetime(t))

        ds.close()
        return lai_vals, fpar_vals, times

    def _find_variable(self, ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
        """Find variable name from candidates."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        return None

    def _find_coord(self, ds, candidates: List[str]) -> Optional[str]:
        """Find coordinate name from candidates."""
        for name in candidates:
            if name in ds.coords or name in ds.dims:
                return name
        return None

    def _apply_qc_filter(self, da: xr.DataArray, qc_da: xr.DataArray) -> xr.DataArray:
        """Apply quality filter based on QC flags."""
        # MODIS LAI QC: bits 5-7 indicate algorithm path
        # 0 = main method, 1 = backup method, etc.
        # Generally accept main method (0) and saturation (2)
        algorithm_bits = (qc_da.values >> 5) & 0b111

        # Accept main algorithm (0) and saturation (2)
        mask = (algorithm_bits == 0) | (algorithm_bits == 2)
        return da.where(mask)

    def _extract_basin_mean(
        self,
        da: xr.DataArray,
        basin_gdf: Optional[gpd.GeoDataFrame],
        valid_range: tuple
    ) -> Optional[float]:
        """Extract basin-averaged value with valid range filtering."""
        # Apply valid range filter
        da = da.where((da >= valid_range[0]) & (da <= valid_range[1]))

        if basin_gdf is not None:
            bounds = basin_gdf.total_bounds
            lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
            lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])

            if lat_name and lon_name:
                try:
                    lat_slice = slice(bounds[1], bounds[3])
                    if da[lat_name].values[0] > da[lat_name].values[-1]:
                        lat_slice = slice(bounds[3], bounds[1])

                    da = da.sel({
                        lon_name: slice(bounds[0], bounds[2]),
                        lat_name: lat_slice
                    })
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.debug(f"Could not subset data to shapefile bounds: {e}")

        elif self.bbox:
            lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
            lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])

            if lat_name and lon_name:
                try:
                    lat_slice = slice(self.bbox['lat_min'], self.bbox['lat_max'])
                    if da[lat_name].values[0] > da[lat_name].values[-1]:
                        lat_slice = slice(self.bbox['lat_max'], self.bbox['lat_min'])

                    da = da.sel({
                        lon_name: slice(self.bbox['lon_min'], self.bbox['lon_max']),
                        lat_name: lat_slice
                    })
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.debug(f"Could not subset data to bbox: {e}")

        mean_val = float(da.mean(skipna=True).values)
        if np.isnan(mean_val):
            return None
        return mean_val

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from MODIS filename."""
        import re

        # Pattern: MCD15A2H.AYYYYDDD or similar
        match = re.search(r'\.A(\d{4})(\d{3})\.', filename)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)

        # Pattern: YYYYDDD in filename
        match = re.search(r'(\d{4})(\d{3})', filename)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)

        return None

    def _interpolate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate 8-day composites to daily values."""
        if len(df) < 2:
            return df

        # Check if already daily
        time_diff = (df.index[1] - df.index[0]).days
        if time_diff <= 1:
            return df

        # Create daily index
        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df_daily = df.reindex(daily_index)

        # Interpolation method
        method = self._get_config_value(lambda: None, default='linear', dict_key='MODIS_LAI_INTERPOLATION')

        if method == 'spline':
            # Smooth spline interpolation (better for phenology)
            df_daily = df_daily.interpolate(method='spline', order=3)
        else:
            df_daily = df_daily.interpolate(method='linear')

        # Ensure non-negative values
        df_daily = df_daily.clip(lower=0)

        df_daily.index.name = 'datetime'
        return df_daily

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed MODIS LAI data."""
        processed_path = (
            self.project_observations_dir / "vegetation" / "preprocessed"
            / f"{self.domain_name}_modis_lai_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error loading MODIS LAI data: {e}")
            return None
