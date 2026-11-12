# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FLUXCOM Evapotranspiration Acquisition Handler

Provides acquisition for FLUXCOM machine learning upscaled ET products.

FLUXCOM Overview:
    Data Type: Machine-learning upscaled eddy covariance ET
    Resolution: 0.5° (FLUXCOM) or 0.05° (FLUXCOM-X)
    Coverage: Global
    Temporal: Daily/Monthly, 2001-2020+ depending on version
    Source: Max Planck Institute for Biogeochemistry / ICOS Carbon Portal

Products:
    FLUXCOM (original): 0.5° resolution, multiple ML methods
    FLUXCOM-X: 0.05° resolution, extended version with improved methods

Data Access Options:
    1. ICOS Carbon Portal (FLUXCOM-X): https://www.icos-cp.eu/data-products/2G60-ZHAK
       - NetCDF files, CC-BY 4.0 license (accepted programmatically)
       - Monthly ET at 0.5° resolution, 2001-2021

    2. Max Planck BGI Data Portal (original FLUXCOM):
       - Requires registration and data request
       - ftp://dataportal.bgc-jena.mpg.de/

    3. Google Earth Engine:
       - FLUXCOM available as ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
       - Contains derived ET variables

Configuration:
    FLUXCOM_DOWNLOAD_URL: Direct URL to download data file
    FLUXCOM_FILE_PATTERN: Pattern for local files (default: "*.nc")
    FLUXCOM_VARIABLE: Variable to extract (default: "ET")
    FLUXCOM_VERSION: 'original' or 'x' (default: 'x')

Unit Conversion (automatic based on source units):
    LE (W/m²) to ET (mm/day): ET = LE * 86400 / (2.45e6)
    ET (mm/hr) to ET (mm/day): ET = ET * 24

References:
    - Jung et al. (2019): https://doi.org/10.1038/s41597-019-0076-8
    - FLUXCOM-X: https://www.icos-cp.eu/data-products/2G60-ZHAK
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# Latent heat of vaporization (J/kg)
LATENT_HEAT_VAPORIZATION = 2.45e6
SECONDS_PER_DAY = 86400
HOURS_PER_DAY = 24


@R.acquisition_handlers.add('FLUXCOM_ET')
@R.acquisition_handlers.add('FLUXCOM')
class FLUXCOMETAcquirer(BaseAcquisitionHandler):
    """
    Acquires FLUXCOM Evapotranspiration data.

    FLUXCOM is a machine learning approach that upscales eddy covariance
    flux measurements using satellite and meteorological data.

    Since FLUXCOM data requires registration, this handler supports:
    1. Direct download from a user-provided URL
    2. Local file discovery (if data is manually placed)
    3. Instructions for manual data access
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download/process FLUXCOM ET data.

        Args:
            output_dir: Directory to save processed files

        Returns:
            Path to output directory or processed file
        """
        self.logger.info("Starting FLUXCOM ET acquisition")

        et_dir = output_dir / "et" / "fluxcom"
        et_dir.mkdir(parents=True, exist_ok=True)

        processed_file = et_dir / f"{self.domain_name}_FLUXCOM_ET.nc"

        if processed_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            self.logger.info(f"Using existing FLUXCOM file: {processed_file}")
            return processed_file

        # Option 1: Check for existing local files
        local_pattern = self._get_config_value(lambda: None, default="*.nc", dict_key='FLUXCOM_FILE_PATTERN')
        existing_files = list(et_dir.glob(local_pattern))

        if existing_files:
            self.logger.info(f"Found {len(existing_files)} local FLUXCOM files")
            return self._process_local_files(existing_files, processed_file)

        # Option 2: Download from URL
        download_url = self._get_config_value(lambda: None, default=None, dict_key='FLUXCOM_DOWNLOAD_URL')
        if download_url:
            self.logger.info("Downloading FLUXCOM data from URL")
            downloaded_file = self._download_from_url(download_url, et_dir)
            return self._process_local_files([downloaded_file], processed_file)

        # Option 3: Download from ICOS Carbon Portal (FLUXCOM-X 0.5° monthly)
        icos_files = self._download_from_icos(et_dir)
        if icos_files:
            return self._process_local_files(icos_files, processed_file)

        # Fail with helpful instructions
        self._provide_data_access_instructions(et_dir)
        raise ValueError(
            f"FLUXCOM data not available. Please download manually to: {et_dir}\n"
            "See log output for data access instructions."
        )

    def _download_from_url(self, url: str, output_dir: Path) -> Path:
        """Download file from URL."""
        filename = url.split('/')[-1]
        if not filename.endswith('.nc'):
            filename = "fluxcom_downloaded.nc"

        out_file = output_dir / filename

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return out_file

        self.logger.info(f"Downloading: {filename}")
        try:
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()

            with open(out_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)

            self.logger.info(f"Downloaded: {out_file}")
            return out_file

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Failed to download FLUXCOM data: {e}")
            raise

    def _download_from_icos(self, output_dir: Path) -> List[Path]:
        """
        Download FLUXCOM-X monthly ET (0.5°) from ICOS Carbon Portal.

        Queries the ICOS CP metadata API for the FLUXCOM-X-BASE ET collection,
        finds the 0.5° monthly files for each year in the requested time range,
        and downloads them. Data is CC-BY 4.0, no authentication required.

        Returns:
            List of downloaded NetCDF file paths, empty if download fails.
        """
        import json
        import urllib.request

        COLLECTION_URL = "https://meta.icos-cp.eu/collections/_l85vWiIV81AifoxCkty50YI"

        self.logger.info("Downloading FLUXCOM-X ET from ICOS Carbon Portal...")

        start_year = self.start_date.year
        end_year = self.end_date.year

        try:
            req = urllib.request.Request(COLLECTION_URL, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                collection = json.loads(resp.read())
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to query ICOS Carbon Portal: {e}", exc_info=True)
            return []

        downloaded = []
        for member in sorted(collection.get("members", []), key=lambda m: m.get("name", "")):
            # Extract year from name like "FLUXCOM-X-BASE evapotranspiration for 2016"
            name = member.get("name", "")
            year = None
            for token in name.split():
                if token.isdigit() and 2000 <= int(token) <= 2030:
                    year = int(token)
                    break
            if year is None or year < start_year or year > end_year:
                continue

            # Fetch yearly sub-collection to find the 0.5° monthly file
            year_url = member.get("res", "")
            try:
                req2 = urllib.request.Request(year_url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req2, timeout=60) as resp2:  # nosec B310
                    year_data = json.loads(resp2.read())
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to fetch {year} metadata: {e}", exc_info=True)
                continue

            for item in year_data.get("members", []):
                item_name = item.get("name", "")
                if "0.5 degree" in item_name and "monthly" in item_name.lower():
                    obj_hash = item.get("hash", "")
                    out_file = output_dir / f"ET_{year}_monthly_halfdeg.nc"

                    if out_file.exists() and not self._get_config_value(
                        lambda: self.config.data.force_download,
                        dict_key='FORCE_DOWNLOAD',
                    ):
                        self.logger.info(f"  {year}: already exists, skipping")
                        downloaded.append(out_file)
                        break

                    self.logger.info(f"  {year}: downloading from ICOS CP...")
                    try:
                        # ICOS CP requires license acceptance via cookie.
                        # Hit /licence_accept to get the cookie, which
                        # redirects to the actual file download.
                        import urllib.parse
                        ids_param = urllib.parse.quote(json.dumps([obj_hash]))
                        accept_url = f"https://data.icos-cp.eu/licence_accept?ids={ids_param}"

                        session = requests.Session()
                        resp3 = session.get(accept_url, stream=True, timeout=600)
                        resp3.raise_for_status()

                        # Verify we got binary data, not an HTML license page
                        content_type = resp3.headers.get("content-type", "")
                        if "html" in content_type:
                            self.logger.warning(f"  {year}: got HTML instead of NetCDF, skipping")
                            break

                        tmp = out_file.with_suffix(".nc.part")
                        with open(tmp, "wb") as f:
                            for chunk in resp3.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                        tmp.replace(out_file)
                        downloaded.append(out_file)
                        self.logger.info(f"  {year}: OK ({out_file.stat().st_size / 1e6:.1f} MB)")
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.warning(f"  {year}: download failed: {e}", exc_info=True)
                    break

        self.logger.info(f"Downloaded {len(downloaded)} FLUXCOM-X yearly files")
        return downloaded

    def _process_local_files(self, files: List[Path], output_file: Path) -> Path:
        """Process local FLUXCOM files."""
        self.logger.info(f"Processing {len(files)} FLUXCOM files...")

        variable = self._get_config_value(lambda: None, default='ET', dict_key='FLUXCOM_VARIABLE')

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        datasets = []
        for f in files:
            try:
                ds = xr.open_dataset(f)

                # Find the variable (ET, LE, etc.)
                var_name = None
                for v in ds.data_vars:
                    if variable.lower() == v.lower():
                        var_name = v
                        break
                if var_name is None:
                    for v in ds.data_vars:
                        if variable.lower() in v.lower():
                            var_name = v
                            break
                        if 'et' in v.lower() or 'latent' in v.lower():
                            var_name = v
                            break

                if not var_name:
                    self.logger.warning(f"Could not find {variable} in {f.name}")
                    continue

                da = ds[var_name]

                # Subset to domain
                lat_dim = 'lat' if 'lat' in da.dims else 'latitude'
                lon_dim = 'lon' if 'lon' in da.dims else 'longitude'

                if lat_dim in da.dims and lon_dim in da.dims:
                    # Handle coordinate ordering
                    lat_coords = da[lat_dim].values
                    if lat_coords[0] > lat_coords[-1]:
                        # Descending latitudes
                        da = da.sel(
                            **{lat_dim: slice(lat_max, lat_min),
                               lon_dim: slice(lon_min, lon_max)}
                        )
                    else:
                        da = da.sel(
                            **{lat_dim: slice(lat_min, lat_max),
                               lon_dim: slice(lon_min, lon_max)}
                        )

                    # If bbox is smaller than grid resolution, use nearest neighbor
                    if da.sizes.get(lat_dim, 0) == 0 or da.sizes.get(lon_dim, 0) == 0:
                        self.logger.info("Basin smaller than grid cell, using nearest neighbor")
                        lat_center = (lat_min + lat_max) / 2
                        lon_center = (lon_min + lon_max) / 2
                        da = ds[var_name].sel(
                            **{lat_dim: lat_center, lon_dim: lon_center},
                            method='nearest'
                        )

                datasets.append(da)

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Error reading {f}: {e}", exc_info=True)
                continue

        if not datasets:
            raise RuntimeError("No valid FLUXCOM data found in files")

        # Combine datasets
        if len(datasets) > 1:
            combined = xr.concat(datasets, dim='time')
        else:
            combined = datasets[0]

        combined = combined.sortby('time')

        # Convert to mm/day based on source units
        source_units = combined.attrs.get('units', '').lower()
        if 'w' in source_units and 'm' in source_units:
            # LE (W/m²) → ET (mm/day)
            self.logger.info("Converting LE (W/m²) to ET (mm/day)")
            combined = combined * SECONDS_PER_DAY / LATENT_HEAT_VAPORIZATION
            units = 'mm/day'
            long_name = 'Evapotranspiration'
        elif 'mm' in source_units and 'hr' in source_units:
            # ET (mm/hr) → ET (mm/day)
            self.logger.info("Converting ET (mm/hr) to ET (mm/day)")
            combined = combined * HOURS_PER_DAY
            units = 'mm/day'
            long_name = 'Evapotranspiration'
        elif 'mm' in source_units and 'day' in source_units:
            # Already mm/day
            units = 'mm/day'
            long_name = 'Evapotranspiration'
        else:
            self.logger.info(f"Unknown source units '{source_units}', keeping as-is")
            units = combined.attrs.get('units', 'unknown')
            long_name = combined.attrs.get('long_name', variable)

        # Compute spatial mean
        lat_dim = 'lat' if 'lat' in combined.dims else 'latitude'
        lon_dim = 'lon' if 'lon' in combined.dims else 'longitude'

        if lat_dim in combined.dims and lon_dim in combined.dims:
            # Weight by cos(lat) for proper area averaging
            weights = np.cos(np.deg2rad(combined[lat_dim]))
            weights = weights / weights.sum()
            basin_mean = combined.weighted(weights).mean(dim=[lat_dim, lon_dim])
        else:
            basin_mean = combined

        # Create output dataset
        ds_out = xr.Dataset({
            'ET': basin_mean.rename('ET'),
        })

        ds_out['ET'].attrs = {
            'long_name': long_name,
            'units': units,
            'source': 'FLUXCOM',
        }

        ds_out.attrs['title'] = 'FLUXCOM Evapotranspiration'
        ds_out.attrs['source'] = 'Max Planck Institute / ICOS Carbon Portal'
        ds_out.attrs['original_variable'] = variable
        ds_out.attrs['created'] = datetime.now().isoformat()
        ds_out.attrs['domain'] = self.domain_name
        ds_out.attrs['bbox'] = f"{lon_min},{lat_min},{lon_max},{lat_max}"

        ds_out.to_netcdf(output_file)

        self.logger.info(f"Saved FLUXCOM ET: {output_file}")

        # Create CSV
        self._create_csv_output(basin_mean, output_file.parent)

        return output_file

    def _create_csv_output(self, da: xr.DataArray, output_dir: Path):
        """Create CSV timeseries output."""
        csv_file = output_dir / f"{self.domain_name}_FLUXCOM_ET_timeseries.csv"

        df = da.to_dataframe()
        # Keep only the ET column (drop scalar lat/lon from nearest-neighbor)
        et_col = [c for c in df.columns if 'et' in c.lower()]
        if et_col:
            df = df[[et_col[0]]]
        else:
            df = df.iloc[:, :1]

        df.index.name = 'date'
        df.columns = ['et_mm_day']

        df.to_csv(csv_file)
        self.logger.info(f"Created CSV: {csv_file}")

    def _provide_data_access_instructions(self, output_dir: Path):
        """Log instructions for accessing FLUXCOM data."""
        instructions = """
========================================================================
FLUXCOM Data Access Instructions
========================================================================

FLUXCOM data requires registration. Options:

1. FLUXCOM-X (Recommended - higher resolution 0.05°)
   - Register at: https://cpauth.icos-cp.eu/
   - Download from: https://www.icos-cp.eu/data-products/2G60-ZHAK
   - Product: FLUXCOM-X-BASE daily/monthly Latent Heat Flux
   - Coverage: 2001-2020, Global

2. Original FLUXCOM (0.5° resolution)
   - Request access at: https://www.bgc-jena.mpg.de/geodb/projects/Data.php
   - FTP: ftp://dataportal.bgc-jena.mpg.de/ (requires credentials)
   - Product: FLUXCOM RS/RS+METEO daily/monthly LE

3. Google Earth Engine (indirect access)
   - FLUXCOM variables available in TerraClimate collection
   - ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')

4. Direct URL (if you have a download link)
   - Set FLUXCOM_DOWNLOAD_URL in your config file

After downloading, place NetCDF files in:
  {output_dir}

Configuration example:
  FLUXCOM_VARIABLE: 'LE'           # Latent heat flux
  FLUXCOM_CONVERT_TO_ET: true      # Convert to ET (mm/day)
  FLUXCOM_FILE_PATTERN: '*.nc'     # File pattern to search

========================================================================
"""
        self.logger.info(instructions.format(output_dir=output_dir))
