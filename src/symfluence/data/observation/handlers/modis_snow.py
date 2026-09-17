# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS snow cover observation handler.

Provides acquisition and preprocessing of MODIS Snow Cover Area data
(MOD10A1/MYD10A1) for snowmelt model calibration and validation.
"""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

HAS_GEO = find_spec("geopandas") is not None
if HAS_GEO:
    import geopandas as gpd

from ..base import BaseObservationHandler
from .modis_utils import (
    CLOUD_VALUE,
    MODIS_FILL_VALUES,
    VALID_SNOW_RANGE,
)


@R.observation_handlers.add('modis_snow')
class MODISSnowHandler(BaseObservationHandler):
    """
    Handles MODIS Snow Cover Area (SCA) data.
    Supports both single-product (MOD10A1) and merged (MOD10A1+MYD10A1) data.
    """

    obs_type = "snow_cover"
    source_name = "NASA_MODIS"
    SOURCE_INFO = {
        'source': 'MODIS MOD10A1/MOD10A2',
        'source_doi': '10.5067/MODIS/MOD10A1.061',
        'url': 'https://nsidc.org/data/mod10a1',
    }

    # Use constants from modis_utils
    VALID_SNOW_RANGE = VALID_SNOW_RANGE
    CLOUD_VALUE = CLOUD_VALUE
    MISSING_VALUES = MODIS_FILL_VALUES

    def acquire(self) -> Path:
        """Locate or download MODIS snow data."""
        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='local').lower()
        snow_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "snow", dict_key='MODIS_SNOW_DIR'))

        if not snow_dir.exists():
            snow_dir.mkdir(parents=True, exist_ok=True)

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for MODIS snow")
            from ...acquisition.registry import AcquisitionRegistry

            # Check if merged SCA is requested
            use_merged = self._get_config_value(lambda: None, default=True, dict_key='MODIS_SCA_MERGE')

            if use_merged:
                # Use the new merged SCA acquirer
                acquirer = AcquisitionRegistry.get_handler('MODIS_SCA', self.config, self.logger)
            else:
                # Use legacy single-product acquirer
                acquirer = AcquisitionRegistry.get_handler('MODIS_SNOW', self.config, self.logger)

            return acquirer.download(snow_dir)

        return snow_dir

    def process(self, input_path: Path) -> Path:
        """Process MODIS SCA data (CSV or NetCDF)."""
        self.logger.info(f"Processing MODIS Snow for domain: {self.domain_name}")

        # Check for merged file first
        if input_path.is_dir():
            merged_file = input_path / f"{self.domain_name}_MODIS_SCA_merged.nc"
            if merged_file.exists():
                input_path = merged_file

        # Determine if we are processing a file or a directory
        if input_path.is_file() and input_path.suffix == '.nc':
            return self._process_netcdf(input_path)

        # Check for any NetCDF files in directory
        if input_path.is_dir():
            nc_files = list(input_path.glob("*.nc"))
            if nc_files:
                # Prefer merged file, else use first available
                merged = [f for f in nc_files if 'merged' in f.name.lower()]
                if merged:
                    return self._process_netcdf(merged[0])
                return self._process_netcdf(nc_files[0])

        # Legacy/Directory-based CSV processing
        csv_files = list(input_path.glob("SCA_Daily_Basin_*.csv"))
        if not csv_files:
            self.logger.warning(f"No MODIS SCA files found in {input_path}")
            return input_path

        # Standard filter: at least 100 valid pixels
        min_pixels = self._get_config_value(lambda: None, default=100, dict_key='MODIS_MIN_PIXELS')

        # For now, just copy and filter the first matching file
        df = pd.read_csv(csv_files[0], parse_dates=['date']).set_index('date')
        if 'valid_pixels' in df.columns:
            df = df[df['valid_pixels'] >= min_pixels]

        return self._save_processed(df)

    def _process_netcdf(self, nc_path: Path) -> Path:
        """Extract basin average snow cover from NetCDF."""
        self.logger.info(f"Extracting basin average from MODIS NetCDF: {nc_path}")

        try:
            # Try netcdf4 engine explicitly first
            ds = xr.open_dataset(nc_path, engine='netcdf4')
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to open with netcdf4, trying h5netcdf: {e}", exc_info=True)
            try:
                ds = xr.open_dataset(nc_path, engine='h5netcdf')
            except Exception as e2:  # noqa: BLE001 — must-not-raise contract
                self.logger.error(f"Failed to open NetCDF with any engine: {e2}")
                # Check if it's an error page
                if nc_path.stat().st_size < 10000:
                    with open(nc_path, 'r', encoding='utf-8', errors='ignore') as f:
                        self.logger.error(f"File snippet: {f.read(500)}")
                raise

        with ds:
            # Find snow cover variable
            var_name = self._find_snow_variable(ds)

            # Get the data array
            snow_data = ds[var_name]

            mode = self._get_config_value(lambda: None, default='mean_ndsi', dict_key='MODIS_SCA_OBSERVATION_MODE')
            if mode == 'snow_fraction':
                return self._save_processed(self._extract_snow_fraction(snow_data, ds))

            # Apply quality filtering
            snow_data = self._apply_quality_filter(snow_data)

            # Check if we should use catchment masking
            use_catchment_mask = self._get_config_value(lambda: None, default=False, dict_key='MODIS_SCA_USE_CATCHMENT_MASK')

            if use_catchment_mask and HAS_GEO:
                df = self._extract_with_catchment_mask(snow_data, ds)
            else:
                df = self._extract_spatial_average(snow_data)

            # Filter by minimum valid ratio
            min_valid_ratio = self._get_config_value(lambda: None, default=0.1, dict_key='MODIS_SCA_MIN_VALID_RATIO')
            if 'valid_ratio' in df.columns:
                df = df[df['valid_ratio'] >= min_valid_ratio]

            # Normalize SCA to fraction (0-1) if needed
            normalize = self._get_config_value(lambda: None, default=True, dict_key='MODIS_SCA_NORMALIZE')
            if normalize and df['sca'].max() > 1.0:
                df['sca'] = df['sca'] / 100.0

            return self._save_processed(df)

    def _extract_snow_fraction(self, data: xr.DataArray, ds: xr.Dataset) -> pd.DataFrame:
        """Area-weighted fraction of clear, quality-screened pixels with snow.

        NDSI is a snow index, not a subpixel snow fraction. Require retained
        Basic QA and geographic pixel centres; never silently fall back to
        an unmasked or unscreened calibration target.
        """
        if set(data.dims) != {'time', 'lat', 'lon'}:
            raise ValueError('snow_fraction requires a time/lat/lon MODIS grid')
        for coord in (data.lat, data.lon):
            if coord.ndim != 1 or (coord.size > 2 and not np.allclose(
                    np.diff(coord.values), np.diff(coord.values)[0])):
                raise ValueError('snow_fraction requires regular geographic pixel centres')
        if data.encoding.get('scale_factor', 1) != 1 or data.encoding.get('add_offset', 0) != 0:
            raise ValueError('snow_fraction requires NDSI on its encoded 0–100 scale')
        qa_name = 'NDSI_Snow_Cover_Basic_QA'
        if qa_name not in ds:
            raise ValueError('snow_fraction requires NDSI_Snow_Cover_Basic_QA')
        threshold = self._get_config_value(lambda: None, default=0.0, dict_key='MODIS_SCA_NDSI_THRESHOLD')
        max_qa = self._get_config_value(lambda: None, default=1, dict_key='MODIS_SCA_MAX_BASIC_QA')
        lon, lat = np.meshgrid(data.lon.values, data.lat.values)
        mask = np.ones(lon.shape, dtype=bool)
        if self._get_config_value(lambda: None, default=False, dict_key='MODIS_SCA_USE_CATCHMENT_MASK'):
            shp = self._resolve_catchment_shapefile()
            if not HAS_GEO or shp is None:
                raise ValueError('snow_fraction catchment masking requires a catchment shapefile and geopandas')
            from shapely import intersects_xy
            basin = gpd.read_file(shp).to_crs('EPSG:4326').geometry.union_all()
            mask = intersects_xy(basin, lon, lat)
        if not mask.any():
            raise ValueError('No MODIS pixel centres inside the requested footprint')
        weights = xr.DataArray(np.cos(np.deg2rad(lat)) * mask,
                               dims=('lat', 'lon'), coords={'lat': data.lat, 'lon': data.lon})
        qa = ds[qa_name]
        valid = (data >= 0) & (data <= 100) & qa.isin(list(range(max_qa + 1)))
        dims = ('lat', 'lon')
        valid_weight = weights.where(valid, 0).sum(dims)
        coverage = valid_weight / weights.sum()
        count = (valid & (weights > 0)).sum(dims)
        fraction = weights.where(valid & (data > threshold), 0).sum(dims) / valid_weight.where(valid_weight > 0)
        min_ratio = self._get_config_value(lambda: None, default=0.1, dict_key='MODIS_SCA_MIN_VALID_RATIO')
        min_pixels = self._get_config_value(lambda: None, default=100, dict_key='MODIS_MIN_PIXELS')
        fraction = fraction.where((coverage >= min_ratio) & (count >= min_pixels))
        return pd.DataFrame({'sca': fraction.values, 'valid_pixels': count.values,
                             'valid_ratio': coverage.values},
                            index=pd.Index(self._convert_time_to_datetime(data.time.values), name='date'))

    def _find_snow_variable(self, ds: xr.Dataset) -> str:
        """Find the snow cover variable in the dataset."""
        # Priority order for variable names
        priority_vars = [
            'NDSI_Snow_Cover',
            'snow_cover',
            'SCA',
            'sca'
        ]

        for var in priority_vars:
            if var in ds.data_vars:
                return str(var)

        # Fall back to finding any snow-related variable
        suitable_vars = [v for v in ds.data_vars if 'snow' in str(v).lower() or 'ndsi' in str(v).lower()]
        if suitable_vars:
            return str(suitable_vars[0])

        raise ValueError(f"No snow variables found in dataset. Available: {list(ds.data_vars)}")

    def _apply_quality_filter(self, data: xr.DataArray) -> xr.DataArray:
        """Apply quality filtering to MODIS snow data."""
        # Convert invalid values to NaN
        filtered = data.where(
            (data >= self.VALID_SNOW_RANGE[0]) & (data <= self.VALID_SNOW_RANGE[1])
        )
        return filtered

    def _extract_spatial_average(self, data: xr.DataArray) -> pd.DataFrame:
        """Extract spatially averaged SCA time series."""
        # Get spatial dimensions
        spatial_dims = [d for d in data.dims if d != 'time']

        # Convert time values, handling cftime objects
        time_values = self._convert_time_to_datetime(data.time.values)

        if spatial_dims:
            # Count valid pixels for each timestep
            valid_counts = data.notnull().sum(dim=spatial_dims)
            total_pixels = np.prod([data.sizes[d] for d in spatial_dims])

            # Compute mean over spatial dims
            sca_mean = data.mean(dim=spatial_dims, skipna=True)

            df = pd.DataFrame({
                'sca': sca_mean.values,
                'valid_pixels': valid_counts.values,
                'valid_ratio': valid_counts.values / total_pixels
            }, index=time_values)
        else:
            df = pd.DataFrame({
                'sca': data.values
            }, index=time_values)

        df.index.name = 'date'
        return df

    def _convert_time_to_datetime(self, time_values):
        """Convert time values to pandas DatetimeIndex, handling cftime objects."""
        try:
            # Try direct conversion first
            return pd.to_datetime(time_values)
        except (TypeError, ValueError):
            pass

        # Handle cftime objects
        try:
            import cftime
            if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                # Convert cftime to standard datetime
                converted = []
                for t in time_values:
                    try:
                        # Try to create a standard datetime
                        dt = pd.Timestamp(year=t.year, month=t.month, day=t.day,
                                         hour=getattr(t, 'hour', 0),
                                         minute=getattr(t, 'minute', 0),
                                         second=getattr(t, 'second', 0))
                        converted.append(dt)
                    except (ValueError, TypeError, AttributeError):
                        # Fall back to string parsing for invalid dates
                        converted.append(pd.to_datetime(str(t)[:10]))
                return pd.DatetimeIndex(converted)
        except ImportError:
            pass

        # Last resort: try string conversion
        return pd.to_datetime([str(t)[:10] for t in time_values])

    def _extract_with_catchment_mask(self, data: xr.DataArray, ds: xr.Dataset) -> pd.DataFrame:
        """Extract SCA using catchment shapefile as mask."""
        shp_path = self._resolve_catchment_shapefile()
        if shp_path is None:
            self.logger.warning("Catchment shapefile not found; using spatial average")
            return self._extract_spatial_average(data)

        try:
            gdf = gpd.read_file(shp_path)
            catchment_geom = gdf.geometry.union_all()

            # Get coordinate info
            lat_name = 'lat' if 'lat' in ds.coords else 'y'
            lon_name = 'lon' if 'lon' in ds.coords else 'x'

            lats = ds.coords[lat_name].values
            lons = ds.coords[lon_name].values

            # Create mask
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            from shapely.geometry import Point
            mask = np.zeros(lon_grid.shape, dtype=bool)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    pt = Point(lon_grid[i, j], lat_grid[i, j])
                    mask[i, j] = catchment_geom.contains(pt)

            # Apply mask
            masked_data = data.where(mask)

            return self._extract_spatial_average(masked_data)

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to apply catchment mask: {e}", exc_info=True)
            return self._extract_spatial_average(data)

    def _save_processed(self, df: pd.DataFrame) -> Path:
        """Save processed dataframe to standard location."""
        output_dir = self.project_observations_dir / "snow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_snow_processed.csv"

        # Ensure index name
        if df.index.name is None:
            df.index.name = 'date'

        df.to_csv(output_file)
        self.logger.info(f"MODIS snow processing complete: {output_file}")
        self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        self.logger.info(f"  SCA range: {df['sca'].min():.3f} to {df['sca'].max():.3f}")
        self.logger.info(f"  Valid observations: {df['sca'].notna().sum()}")

        return output_file


@R.observation_handlers.add('modis_sca')
@R.observation_handlers.add('modis_snow_merged')
class MODISSCAHandler(MODISSnowHandler):
    """
    Specialized handler for merged MODIS SCA (MOD10A1 + MYD10A1).
    Inherits all processing from MODISSnowHandler but defaults to merged acquisition.
    """

    def acquire(self) -> Path:
        """Acquire merged MODIS SCA data."""
        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='local').lower()
        snow_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "snow", dict_key='MODIS_SNOW_DIR'))

        if not snow_dir.exists():
            snow_dir.mkdir(parents=True, exist_ok=True)

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for merged MODIS SCA (Terra + Aqua)")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('MODIS_SCA', self.config, self.logger)
            return acquirer.download(snow_dir)

        # Check for existing merged file
        merged_file = snow_dir / f"{self.domain_name}_MODIS_SCA_merged.nc"
        if merged_file.exists():
            return merged_file

        return snow_dir
