# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Daymet Climate Data Acquisition Handler

Provides acquisition for Daymet daily surface weather data for North America.
Daymet provides gridded estimates of daily weather parameters interpolated
from ground-based observations.

Daymet features:
- 1 km spatial resolution
- Daily temporal resolution
- Coverage: North America (US, Canada, Mexico)
- Period: 1980-present
- Variables: Tmin, Tmax, Precip, SWE, VP, SRAD, Dayl

Data access via ORNL DAAC:
https://daymet.ornl.gov/
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('DAYMET')
class DaymetAcquirer(BaseAcquisitionHandler):
    """
    Handles Daymet climate data acquisition from ORNL DAAC.

    Downloads daily gridded climate data for specified region
    and time period.

    Configuration:
        DAYMET_VARIABLES: Variables to download (default: all)
        DAYMET_FORMAT: Output format ('netcdf', 'csv') (default: netcdf)
    """

    # ORNL DAAC Daymet endpoints
    SINGLE_PIXEL_URL = "https://daymet.ornl.gov/single-pixel/api/data"

    # Cloud OPeNDAP endpoint (NASA Hyrax) — server-side subsetting via DAP2
    # hyperslabs. This is the only working gridded route: ORNL's legacy
    # THREDDS endpoint is retired (404s into the same DMR++ backend), and
    # this Hyrax deployment rejects plain DAP4 constraint expressions, so
    # requests must use DAP2 (pydap with an Earthdata-authenticated session).
    OPENDAP_URL = (
        "https://opendap.earthdata.nasa.gov/collections/"
        "C2532426483-ORNL_CLOUD/granules/"
        "Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc"
    )

    # Daymet Lambert Conformal Conic projection (native grid CRS)
    DAYMET_CRS = (
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 "
        "+lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )

    AVAILABLE_VARIABLES = [
        'tmax',  # Maximum temperature (°C)
        'tmin',  # Minimum temperature (°C)
        'prcp',  # Precipitation (mm)
        'swe',   # Snow water equivalent (mm)
        'vp',    # Vapor pressure (Pa)
        'srad',  # Shortwave radiation (W/m²)
        'dayl',  # Day length (seconds)
    ]

    def download(self, output_dir: Path) -> Path:
        """
        Download Daymet climate data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        self.logger.info("Starting Daymet climate data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve config values that may be missing when the typed config
        # could not be built (e.g. standalone ``data download`` CLI).
        self._resolve_flat_config()

        # Get configuration
        variables = self._get_variables()
        force_download = self._get_config_value(lambda: self.config.data.force_download, default=False)

        # Check if bbox is in North America
        if self.bbox:
            if (self.bbox['lat_min'] < 14 or self.bbox['lat_max'] > 83 or
                self.bbox['lon_min'] < -179 or self.bbox['lon_max'] > -52):
                self.logger.warning(
                    "Daymet coverage is limited to North America. "
                    "Data may not be available for this region."
                )

        # Build output filename
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        var_str = '_'.join(variables[:3])  # First 3 vars in filename
        output_file = output_dir / f"daymet_{var_str}_{start_str}_{end_str}.nc"

        if output_file.exists() and not force_download:
            self.logger.info(f"Daymet file already exists: {output_file}")
            return output_file

        # Download data
        if self._is_point_request():
            # Use single-pixel API for small areas
            self._download_single_pixel(output_dir, variables)
        else:
            # Use subset service for larger areas
            self._download_gridded(output_file, variables)

        return output_dir if not output_file.exists() else output_file

    def _get_variables(self) -> List[str]:
        """Get variables to download."""
        config_vars = self._get_config_value(lambda: None, default=None, dict_key='DAYMET_VARIABLES')

        if config_vars:
            if isinstance(config_vars, str):
                return [config_vars]
            return list(config_vars)

        # Default: core hydrological variables (temp + precip)
        return ['tmax', 'tmin', 'prcp']

    def _is_point_request(self) -> bool:
        """Check if this is a small enough area for single-pixel API."""
        if not self.bbox:
            return True

        lat_range = self.bbox['lat_max'] - self.bbox['lat_min']
        lon_range = self.bbox['lon_max'] - self.bbox['lon_min']

        # Use single-pixel for areas smaller than ~0.1 degrees
        return lat_range < 0.1 and lon_range < 0.1

    def _resolve_flat_config(self):
        """Populate bbox / dates from a flat config dict if the typed config
        coercion failed (e.g. standalone ``symfluence data download`` CLI).
        """
        cfg = self._config
        if not isinstance(cfg, dict):
            return

        if not self.bbox:
            raw = cfg.get('BOUNDING_BOX_COORDS')
            if raw:
                self.bbox = self._parse_bbox(raw)

        if self.start_date is None or pd.isna(self.start_date):
            raw = cfg.get('EXPERIMENT_TIME_START')
            if raw:
                self.start_date = pd.to_datetime(raw)

        if self.end_date is None or pd.isna(self.end_date):
            raw = cfg.get('EXPERIMENT_TIME_END')
            if raw:
                self.end_date = pd.to_datetime(raw)

    def _download_single_pixel(self, output_dir: Path, variables: List[str]):
        """Download using single-pixel API (for point locations)."""
        if not self.bbox:
            self.logger.error("Bounding box required for Daymet download")
            return

        # Use centroid for single-pixel request
        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2

        params = {
            'lat': lat,
            'lon': lon,
            'vars': ','.join(variables),
            'start': self.start_date.strftime('%Y-%m-%d'),
            'end': self.end_date.strftime('%Y-%m-%d'),
            'format': 'csv',
        }

        try:
            self.logger.info(f"Downloading Daymet single-pixel data at ({lat:.4f}, {lon:.4f})")

            response = requests.get(
                self.SINGLE_PIXEL_URL,
                params=params,
                timeout=300
            )
            response.raise_for_status()

            # Save CSV
            output_file = output_dir / f"daymet_{lat:.4f}_{lon:.4f}.csv"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            self.logger.info(f"Downloaded Daymet data: {output_file}")

        except Exception as e:  # noqa: BLE001 — data acquisition resilience
            self.logger.error(f"Daymet single-pixel download failed: {e}")

    def _get_earthdata_session(self) -> requests.Session:
        """Authenticated session for ORNL DAAC (token first, then credentials).

        Redirect-aware (see :class:`~..earthdata_auth.EarthdataSession`);
        warns and returns an unauthenticated session when nothing is found.
        """
        return self._earthdata_auth().session(
            origin_auth=False, purpose="gridded Daymet downloads (ORNL DAAC)",
        )

    def _bbox_to_lcc(self) -> Dict[str, float]:
        """Convert lat/lon bounding box to Daymet LCC x/y coordinates.

        Transforms all four corners of the geographic bounding box to
        Lambert Conformal Conic projection and returns the enclosing
        rectangle in LCC space, with a 1 km buffer (one grid cell).
        """
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326", self.DAYMET_CRS, always_xy=True
        )

        corners_lon = [
            self.bbox['lon_min'], self.bbox['lon_max'],
            self.bbox['lon_min'], self.bbox['lon_max'],
        ]
        corners_lat = [
            self.bbox['lat_min'], self.bbox['lat_min'],
            self.bbox['lat_max'], self.bbox['lat_max'],
        ]

        x_coords, y_coords = transformer.transform(corners_lon, corners_lat)

        buffer = 1000.0  # 1 km = 1 Daymet grid cell
        return {
            'x_min': min(x_coords) - buffer,
            'x_max': max(x_coords) + buffer,
            'y_min': min(y_coords) - buffer,
            'y_max': max(y_coords) + buffer,
        }

    def _opendap_dap2_url(self, var: str, year: int) -> str:
        """Build the DAP2 request URL for one Daymet variable granule.

        Appends a DAP constraint requesting only the needed fields. This
        crucially skips Daymet's ``time_bnds``, whose (time, 2) shape the
        pydap backend mis-parses, and keeps the request DAP2-shaped (this
        Hyrax deployment rejects plain constraints under DAP4).
        """
        return (
            self.OPENDAP_URL.format(var=var, year=year)
            + f"?{var},time,x,y,lat,lon"
        )

    @staticmethod
    def _lcc_subset(ds, lcc_bbox: Dict[str, float]):
        """Subset a Daymet granule to an LCC bounding box.

        Daymet's ``y`` coordinate DESCENDS (north -> south), so the y slice
        must be ordered (y_max, y_min) — an ascending ``slice(y_min, y_max)``
        silently returns an empty window. The x axis ascends normally.
        """
        y_vals = ds['y'].values
        if y_vals.size > 1 and y_vals[0] > y_vals[-1]:
            y_slice = slice(lcc_bbox['y_max'], lcc_bbox['y_min'])
        else:
            y_slice = slice(lcc_bbox['y_min'], lcc_bbox['y_max'])
        return ds.sel(
            x=slice(lcc_bbox['x_min'], lcc_bbox['x_max']),
            y=y_slice,
        )

    def _download_opendap_subset(
        self, var: str, year: int, var_file: Path, lcc_bbox: Dict[str, float],
        max_retries: int = 3,
    ) -> bool:
        """Download a Daymet variable via OPeNDAP server-side subsetting.

        Opens the Daymet granule lazily through the NASA Hyrax OPeNDAP
        endpoint using the pydap engine over DAP2 with an Earthdata-
        authenticated session (Bearer token, ~/.netrc, or env credentials),
        subsets to the LCC bounding box, and writes the result. Retries up
        to *max_retries* times with backoff to handle intermittent server
        errors.

        Returns True on success, False on failure.
        """
        import time as _time

        import xarray as xr

        try:
            import pydap  # noqa: F401
        except ImportError:
            self.logger.error(
                "pydap is required for gridded Daymet downloads (DAP2 "
                "hyperslab subsetting against NASA Hyrax). Install it with: "
                "pip install pydap"
            )
            return False

        url = self._opendap_dap2_url(var, year)
        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(
                    "OPeNDAP subsetting for Daymet %s %d (attempt %d/%d)",
                    var, year, attempt, max_retries,
                )

                ds = xr.open_dataset(
                    url, engine='pydap', session=self._get_earthdata_session()
                )

                ds_sub = self._lcc_subset(ds, lcc_bbox)

                if any(s == 0 for s in ds_sub.sizes.values()):
                    self.logger.warning(
                        "OPeNDAP subset returned empty data for %s %d",
                        var, year,
                    )
                    ds.close()
                    return False

                ds_sub.load()
                for v in ds_sub.data_vars:
                    ds_sub[v].encoding.clear()
                for c in ds_sub.coords:
                    ds_sub[c].encoding.clear()
                ds_sub.to_netcdf(var_file, engine='h5netcdf')
                ds.close()

                self.logger.info(
                    "Downloaded subsetted Daymet %s %d via OPeNDAP: %s",
                    var, year, var_file,
                )
                return True

            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_retries:
                    wait = 5 * attempt
                    self.logger.info(
                        "OPeNDAP attempt %d failed for %s %d: %s  "
                        "— retrying in %ds",
                        attempt, var, year, e, wait,
                    )
                    _time.sleep(wait)

        self.logger.error(
            "OPeNDAP subsetting failed for Daymet %s %d after %d attempts: %s",
            var, year, max_retries, last_err,
        )
        return False

    def _download_gridded(self, output_file: Path, variables: List[str]):
        """Download gridded Daymet data with server-side spatial subsetting.

        Converts the geographic bounding box to Daymet's native Lambert
        Conformal Conic projection and downloads each variable/year via
        OPeNDAP, which transfers only the subsetted region.
        """
        if not self.bbox:
            self.logger.error("Bounding box required for Daymet download")
            return

        lcc_bbox = self._bbox_to_lcc()
        self.logger.info(
            f"Daymet bbox in LCC: x=[{lcc_bbox['x_min']:.0f}, "
            f"{lcc_bbox['x_max']:.0f}], "
            f"y=[{lcc_bbox['y_min']:.0f}, {lcc_bbox['y_max']:.0f}]"
        )

        for year in range(self.start_date.year, self.end_date.year + 1):
            year_file = output_file.parent / f"daymet_{year}.nc"

            if year_file.exists():
                continue

            failed = []
            for var in variables:
                var_file = output_file.parent / f"daymet_{var}_{year}.nc"

                if var_file.exists():
                    continue

                if not self._download_opendap_subset(
                    var, year, var_file, lcc_bbox
                ):
                    failed.append(var)

            if failed:
                self.logger.error(
                    "Failed to download Daymet variables %s for %d. "
                    "Check that ~/.netrc has credentials for "
                    "urs.earthdata.nasa.gov, or set EARTHDATA_TOKEN.",
                    failed, year,
                )

            # Merge per-variable files into a single year file
            self._merge_year_files(output_file.parent, variables, year, year_file)

    def _merge_year_files(
        self, output_dir: Path, variables: List[str], year: int, year_file: Path
    ):
        """Merge per-variable Daymet files into a single per-year file.

        Collects ``daymet_{var}_{year}.nc`` files, merges them with
        ``xr.merge``, writes to ``daymet_{year}.nc``, and removes the
        per-variable files.  Skips silently if no per-variable files exist
        (e.g. all downloads failed).
        """
        import xarray as xr

        var_files = [
            output_dir / f"daymet_{var}_{year}.nc"
            for var in variables
            if (output_dir / f"daymet_{var}_{year}.nc").exists()
        ]

        if not var_files:
            self.logger.warning(
                f"No per-variable files found for Daymet {year}; "
                "cannot create merged file"
            )
            return

        try:
            datasets = [xr.open_dataset(f) for f in var_files]
            merged = xr.merge(datasets, compat='override')
            merged.to_netcdf(year_file, engine='h5netcdf')

            for ds in datasets:
                ds.close()

            # Remove per-variable files now that the merged file exists
            for f in var_files:
                f.unlink()

            self.logger.info(
                f"Merged {len(var_files)} variable files into {year_file.name}"
            )

        except Exception as e:  # noqa: BLE001 — merge is best-effort
            self.logger.warning(
                f"Failed to merge per-variable files for {year}: {e}. "
                "Per-variable files retained."
            )
