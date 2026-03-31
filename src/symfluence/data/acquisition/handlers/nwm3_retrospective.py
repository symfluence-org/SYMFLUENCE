# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NOAA National Water Model v3.0 Retrospective data acquisition.

Provides automated download of NWM v3.0 retrospective data from the
public AWS S3 bucket ``noaa-nwm-retrospective-3-0-pds``.

Available output types (Zarr stores under ``CONUS/zarr/``):

- **forcing** — Meteorological forcing: temperature, humidity, pressure,
  radiation, precipitation, wind (8 per-variable Zarr stores)
- **chrtout** — Channel routing: streamflow, velocity, lateral/bucket runoff
- **ldasout** — Land surface: snow, soil moisture, evapotranspiration, runoff
- **gwout**   — Groundwater: bucket fluxes, drainage
- **lakeout** — Reservoir/lake: water surface elevation, inflow, outflow
- **rtout**   — Terrain routing: surface head, water table depth

The channel routing output (chrtout) is indexed by NHD ``feature_id`` (reach
comids), while forcing and all gridded outputs use Lambert Conformal Conic
x/y coordinates.

When used as a forcing dataset (``FORCING_DATASET: NWM3_RETROSPECTIVE``),
the handler downloads all 8 forcing variables, converts precipitation from
rate (mm/s) to hourly accumulation (kg/m²), and adds lat/lon coordinates.

Coverage: February 1979 -- January 2023.

Configuration keys
------------------
NWM3_OUTPUT_TYPE : str
    Which output store to download.  One of ``forcing``, ``chrtout``,
    ``ldasout``, ``gwout``, ``lakeout``, ``rtout``.
    Default: ``forcing`` when used as FORCING_DATASET, ``chrtout`` otherwise.
NWM3_VARIABLES : str
    Comma-separated list of variables to extract (e.g. ``streamflow,velocity``).
    Default: all variables in the store.
NWM3_FEATURE_IDS : str
    Comma-separated NHD feature (reach) IDs for channel routing data.
    Required when ``NWM3_OUTPUT_TYPE`` is ``chrtout``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

_S3_BUCKET = 'noaa-nwm-retrospective-3-0-pds'
_ZARR_PREFIX = 'CONUS/zarr'

# NWM Lambert Conformal Conic projection string (from Zarr metadata)
_NWM_PROJ4 = (
    '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 '
    '+lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 '
    '+x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext +no_defs'
)

# Temporal coverage
_NWM3_MIN_DATE = pd.Timestamp('1979-02-01')
_NWM3_MAX_DATE = pd.Timestamp('2023-01-31 23:00:00')

# Valid output types
_VALID_OUTPUT_TYPES = {'forcing', 'chrtout', 'ldasout', 'gwout', 'lakeout', 'rtout'}

# NWM forcing variable Zarr stores and their internal data-variable names
_NWM_FORCING_VARS = {
    't2d':    'T2D',      # 2-m air temperature (K)
    'q2d':    'Q2D',      # 2-m specific humidity (kg/kg)
    'psfc':   'PSFC',     # Surface pressure (Pa)
    'swdown': 'SWDOWN',   # Downward shortwave radiation (W/m²)
    'lwdown': 'LWDOWN',   # Downward longwave radiation (W/m²)
    'precip': 'RAINRATE', # Precipitation rate (mm/s)
    'u2d':    'U2D',      # 10-m u-wind component (m/s)
    'v2d':    'V2D',      # 10-m v-wind component (m/s)
}


@AcquisitionRegistry.register('NWM3_RETROSPECTIVE')
class NWM3RetrospectiveAcquirer(BaseAcquisitionHandler):
    """
    Download NOAA NWM v3.0 retrospective outputs from AWS S3 Zarr stores.

    Channel routing data (chrtout) is extracted by feature ID.
    Gridded outputs (ldasout, gwout, lakeout, rtout) are spatially subsetted
    using the configured bounding box projected into LCC coordinates.
    """

    def download(self, output_dir: Path) -> Path:
        """Download NWM3 retrospective data.

        Args:
            output_dir: Directory to save downloaded NetCDF file.

        Returns:
            Path to downloaded NetCDF file.
        """
        output_type = self._get_config_value(
            lambda: None, default='forcing', dict_key='NWM3_OUTPUT_TYPE'
        ).lower()

        if output_type not in _VALID_OUTPUT_TYPES:
            raise ValueError(
                f"Invalid NWM3_OUTPUT_TYPE '{output_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_OUTPUT_TYPES))}"
            )

        # Clamp dates to NWM3 coverage
        start = max(self.start_date, _NWM3_MIN_DATE)
        end = min(self.end_date, _NWM3_MAX_DATE)
        if start > end:
            raise ValueError(
                f"Requested time range ({self.start_date} to {self.end_date}) "
                f"does not overlap NWM3 retrospective coverage "
                f"({_NWM3_MIN_DATE.date()} to {_NWM3_MAX_DATE.date()})."
            )
        if start != self.start_date or end != self.end_date:
            self.logger.warning(
                f"Clamping requested dates to NWM3 coverage: "
                f"{start.date()} to {end.date()}"
            )

        # Build output path
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir
            / f"{self.domain_name}_NWM3_{output_type}_{start.year}-{end.year}.nc"
        )

        force = self._get_config_value(
            lambda: self.config.data.force_download, default=False
        )
        if self._skip_if_exists(output_file, force):
            return output_file

        self.logger.info(
            f"Downloading NWM3 retrospective {output_type} "
            f"({start.date()} to {end.date()})"
        )

        if output_type == 'forcing':
            ds = self._download_forcing(start, end)
        elif output_type == 'chrtout':
            ds = self._download_channel_routing(start, end)
        else:
            ds = self._download_gridded(output_type, start, end)

        # Optionally filter variables
        requested_vars = self._get_config_value(
            lambda: None, default=None, dict_key='NWM3_VARIABLES'
        )
        if requested_vars:
            var_list = [v.strip() for v in requested_vars.split(',')]
            available = set(ds.data_vars)
            missing = set(var_list) - available
            if missing:
                self.logger.warning(
                    f"Requested variables not found in {output_type}: "
                    f"{missing}. Available: {sorted(available)}"
                )
            keep = [v for v in var_list if v in available]
            if not keep:
                raise ValueError(
                    f"None of the requested variables {var_list} found in "
                    f"{output_type}. Available: {sorted(available)}"
                )
            ds = ds[keep]

        ds.attrs.update({
            'source': f'NOAA NWM v3.0 Retrospective ({output_type})',
            'bbox': str(self.bbox),
            's3_bucket': _S3_BUCKET,
        })

        # Clear Zarr encoding to avoid HDF5 compatibility errors
        for var in ds.data_vars:
            ds[var].encoding.clear()
        for coord in ds.coords:
            ds[coord].encoding.clear()

        ds.load().to_netcdf(output_file, engine='h5netcdf')
        self.logger.info(f"NWM3 {output_type} download complete: {output_file}")

        self.plot_diagnostics(output_file)
        return output_file

    # ------------------------------------------------------------------
    # Forcing data (8 per-variable Zarr stores)
    # ------------------------------------------------------------------

    def _download_forcing(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> xr.Dataset:
        """Download NWM3 forcing from per-variable Zarr stores.

        Downloads all 8 meteorological forcing variables, converts
        precipitation from rate (mm/s) to hourly accumulation (kg/m²),
        and adds lat/lon coordinates via inverse LCC projection.
        """
        import pyproj

        self.logger.info(
            "Downloading NWM3 retrospective forcing "
            f"({_S3_BUCKET})"
        )

        # --- Project bbox to LCC ---
        wgs84 = pyproj.Proj('EPSG:4326')
        nwm_proj = pyproj.Proj(_NWM_PROJ4)
        to_lcc = pyproj.Transformer.from_proj(wgs84, nwm_proj, always_xy=True)
        to_wgs = pyproj.Transformer.from_proj(nwm_proj, wgs84, always_xy=True)

        lon_min = min(self.bbox['lon_min'], self.bbox['lon_max'])
        lon_max = max(self.bbox['lon_min'], self.bbox['lon_max'])
        lat_min = min(self.bbox['lat_min'], self.bbox['lat_max'])
        lat_max = max(self.bbox['lat_min'], self.bbox['lat_max'])

        x_min, y_min = to_lcc.transform(lon_min, lat_min)
        x_max, y_max = to_lcc.transform(lon_max, lat_max)
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        # Buffer by ~2 grid cells (2000 m)
        x_min -= 2000
        x_max += 2000
        y_min -= 2000
        y_max += 2000

        self.logger.info(
            f"Bounding box in LCC: x=[{x_min:.0f}, {x_max:.0f}], "
            f"y=[{y_min:.0f}, {y_max:.0f}]"
        )

        fs = s3fs.S3FileSystem(anon=True)
        bucket = f'{_S3_BUCKET}/{_ZARR_PREFIX}/forcing'
        var_arrays = {}

        for zarr_name, internal_name in _NWM_FORCING_VARS.items():
            self.logger.info(f"Opening NWM3 forcing Zarr: {zarr_name}.zarr")
            store = s3fs.S3Map(f'{bucket}/{zarr_name}.zarr', s3=fs)
            ds_var = xr.open_zarr(store)

            # Spatial and temporal subset
            ds_var = ds_var.sel(
                x=slice(x_min, x_max),
                y=slice(y_min, y_max),
                time=slice(start, end),
            )

            if ds_var.sizes['time'] == 0:
                raise ValueError(
                    f"No NWM3 forcing data for {zarr_name} in "
                    f"time range {start} to {end}"
                )

            data = ds_var[internal_name]

            # Convert precipitation from rate (mm/s) to hourly
            # accumulation (kg/m²) for consistency with AORC convention
            if zarr_name == 'precip':
                data = data * 3600.0  # mm/s × 3600s = mm = kg/m²
                data.attrs['units'] = 'kg/m^2'
                data.attrs['long_name'] = 'Total precipitation (hourly accumulation)'

            var_arrays[internal_name] = data
            self.logger.info(
                f"  {zarr_name} -> {internal_name}: shape={dict(data.sizes)}"
            )

        # --- Merge all variables into a single dataset ---
        ds_merged = xr.Dataset(var_arrays)

        if 'crs' in ds_merged:
            ds_merged = ds_merged.drop_vars('crs')

        # --- Add lat/lon coordinates via inverse projection ---
        x_vals = ds_merged.x.values
        y_vals = ds_merged.y.values
        xx, yy = np.meshgrid(x_vals, y_vals)
        lons, lats = to_wgs.transform(xx, yy)

        ds_merged = ds_merged.assign_coords({
            'latitude': (('y', 'x'), lats),
            'longitude': (('y', 'x'), lons),
        })

        ds_merged.attrs.update({
            'source': 'NOAA NWM v3.0 Retrospective (forcing)',
            'projection': _NWM_PROJ4,
        })

        self.logger.info(
            f"NWM3 forcing subset: {ds_merged.sizes['time']} timesteps, "
            f"grid={ds_merged.sizes['y']}x{ds_merged.sizes['x']}, "
            f"variables: {list(ds_merged.data_vars)}"
        )
        return ds_merged

    # ------------------------------------------------------------------
    # Channel routing (indexed by feature_id / NHD reach comids)
    # ------------------------------------------------------------------

    def _download_channel_routing(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> xr.Dataset:
        """Download channel routing data for specific NHD reach feature IDs."""
        feature_ids_str = self._get_config_value(
            lambda: None, default=None, dict_key='NWM3_FEATURE_IDS'
        )
        if not feature_ids_str:
            raise ValueError(
                "NWM3_FEATURE_IDS must be set when NWM3_OUTPUT_TYPE is "
                "'chrtout'. Provide comma-separated NHD reach comids."
            )

        feature_ids = [int(fid.strip()) for fid in feature_ids_str.split(',')]
        self.logger.info(
            f"Extracting channel routing for {len(feature_ids)} feature IDs"
        )

        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(
            f'{_S3_BUCKET}/{_ZARR_PREFIX}/chrtout.zarr', s3=fs
        )
        ds = xr.open_zarr(store)

        # Select feature IDs
        ds = ds.sel(feature_id=feature_ids)

        # Temporal subset
        ds = ds.sel(time=slice(start, end))

        if ds.sizes['time'] == 0:
            raise ValueError(
                f"No chrtout data in time range {start} to {end}"
            )

        self.logger.info(
            f"Channel routing subset: {ds.sizes['time']} timesteps, "
            f"{ds.sizes['feature_id']} reaches, "
            f"variables: {list(ds.data_vars)}"
        )
        return ds

    # ------------------------------------------------------------------
    # Gridded outputs (LCC x/y spatial subsetting)
    # ------------------------------------------------------------------

    def _download_gridded(
        self,
        output_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> xr.Dataset:
        """Download gridded NWM output with spatial subsetting via LCC projection."""
        import pyproj

        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(
            f'{_S3_BUCKET}/{_ZARR_PREFIX}/{output_type}.zarr', s3=fs
        )
        ds = xr.open_zarr(store)

        # --- Project bbox to LCC ---
        wgs84 = pyproj.Proj('EPSG:4326')
        nwm_proj = pyproj.Proj(_NWM_PROJ4)
        to_lcc = pyproj.Transformer.from_proj(wgs84, nwm_proj, always_xy=True)
        to_wgs = pyproj.Transformer.from_proj(nwm_proj, wgs84, always_xy=True)

        lon_min = min(self.bbox['lon_min'], self.bbox['lon_max'])
        lon_max = max(self.bbox['lon_min'], self.bbox['lon_max'])
        lat_min = min(self.bbox['lat_min'], self.bbox['lat_max'])
        lat_max = max(self.bbox['lat_min'], self.bbox['lat_max'])

        x_min, y_min = to_lcc.transform(lon_min, lat_min)
        x_max, y_max = to_lcc.transform(lon_max, lat_max)
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        # Buffer by ~2 grid cells (2000 m)
        x_min -= 2000
        x_max += 2000
        y_min -= 2000
        y_max += 2000

        self.logger.info(
            f"Bounding box in LCC: x=[{x_min:.0f}, {x_max:.0f}], "
            f"y=[{y_min:.0f}, {y_max:.0f}]"
        )

        # Spatial subset
        ds = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

        # Temporal subset
        ds = ds.sel(time=slice(start, end))

        if ds.sizes['time'] == 0:
            raise ValueError(
                f"No {output_type} data in time range {start} to {end}"
            )

        # Drop CRS variable if present
        if 'crs' in ds:
            ds = ds.drop_vars('crs')

        # Add lat/lon coordinates via inverse projection
        x_vals = ds.x.values
        y_vals = ds.y.values
        xx, yy = np.meshgrid(x_vals, y_vals)
        lons, lats = to_wgs.transform(xx, yy)

        ds = ds.assign_coords({
            'latitude': (('y', 'x'), lats),
            'longitude': (('y', 'x'), lons),
        })
        ds.attrs['projection'] = _NWM_PROJ4

        self.logger.info(
            f"Gridded {output_type} subset: {ds.sizes['time']} timesteps, "
            f"grid={ds.sizes.get('y', '?')}x{ds.sizes.get('x', '?')}, "
            f"variables: {list(ds.data_vars)}"
        )
        return ds
