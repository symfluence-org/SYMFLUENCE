# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GLEAM evapotranspiration observation handler.

Provides acquisition and preprocessing of GLEAM ET products for
evapotranspiration model validation and multivariate calibration.
"""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('gleam_et')
class GLEAMETHandler(BaseObservationHandler):
    """
    Handles GLEAM evapotranspiration data.

    Downloads a NetCDF bundle (tar/zip or single file), subsets to the basin
    bbox, and outputs a basin-mean ET time series.
    """

    obs_type = "et"
    source_name = "GLEAM"
    SOURCE_INFO = {
        'source': 'GLEAM v3',
        'source_doi': '10.5194/gmd-10-1903-2017',
        'url': 'https://www.gleam.eu/',
    }

    def acquire(self) -> Path:
        et_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "et" / "gleam", dict_key='GLEAM_ET_PATH'))
        et_dir.mkdir(parents=True, exist_ok=True)

        download_url = self._get_config_value(lambda: None, default=None, dict_key='GLEAM_ET_DOWNLOAD_URL')
        if not download_url:
            return et_dir

        archive_name = Path(download_url).name
        target_path = et_dir / archive_name
        force_download = bool(self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ))

        if target_path.exists() and not force_download:
            return et_dir

        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        self.logger.info(f"Downloading GLEAM ET from {download_url}")
        with requests.get(download_url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(target_path)

        self._extract_if_needed(target_path, et_dir)
        return et_dir

    def process(self, input_path: Path) -> Path:
        self.logger.info(f"Processing GLEAM ET for domain: {self.domain_name}")

        nc_files = list(input_path.rglob("*.nc"))
        if not nc_files:
            self.logger.warning("No GLEAM ET NetCDF files found")
            return input_path

        series_list = []
        for nc_file in sorted(nc_files):
            with xr.open_dataset(nc_file) as ds:
                et_var = self._select_et_variable(ds)
                if et_var is None:
                    continue

                et_data = ds[et_var]
                et_data = self._subset_to_bbox(et_data)
                et_mean = et_data.mean(dim=[d for d in et_data.dims if d != 'time'])
                df_ts = et_mean.to_dataframe().reset_index()
                series_list.append(df_ts)

        if not series_list:
            self.logger.warning("No GLEAM ET data could be extracted")
            return input_path

        df = pd.concat(series_list).sort_values('time').set_index('time')

        # Optional unit conversion to mm/day
        conversion = self._get_config_value(lambda: None, default=None, dict_key='ET_UNIT_CONVERSION')
        if conversion is not None:
            try:
                df = df * float(conversion)
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid ET_UNIT_CONVERSION: {conversion}")

        output_dir = self.project_observations_dir / "et" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_gleam_et_processed.csv"
        df.to_csv(output_file)

        self.logger.info(f"GLEAM ET processing complete: {output_file}")
        return output_file

    def _extract_if_needed(self, archive_path: Path, target_dir: Path) -> None:
        if archive_path.suffix in {".tar", ".gz", ".tgz"} or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:*") as tar:
               # Security rationale: Extracting from trusted GLEAM data archive
                # nosec B202
                from symfluence.core.archive_extraction import safe_tar_extract
                safe_tar_extract(tar, target_dir)
            return
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                from symfluence.core.archive_extraction import safe_zip_extract
                safe_zip_extract(zf, target_dir)

    def _select_et_variable(self, ds: xr.Dataset) -> Optional[str]:
        preferred = self._get_config_value(lambda: None, default=None, dict_key='ET_VARIABLE_NAME')
        if preferred and preferred in ds.data_vars:
            return preferred

        candidates = []
        for name in ds.data_vars:
            lower = name.lower()
            if lower in {'et', 'e', 'evap', 'evaporation', 'evapotranspiration'}:
                candidates.append(name)
            elif 'et' in lower and 'pet' not in lower:
                candidates.append(name)

        if candidates:
            return str(candidates[0])

        if len(ds.data_vars) == 1:
            return str(next(iter(ds.data_vars.keys())))

        return None

    def _subset_to_bbox(self, data: xr.DataArray) -> xr.DataArray:
        if not self.bbox:
            return data

        lat_name = next((n for n in data.coords if n.lower() in {"lat", "latitude"}), None)
        lon_name = next((n for n in data.coords if n.lower() in {"lon", "longitude"}), None)
        if not lat_name or not lon_name:
            return data

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        lon_vals = data[lon_name].values
        if np.nanmax(lon_vals) > 180 and (lon_min < 0 or lon_max < 0):
            lon_min = lon_min % 360
            lon_max = lon_max % 360

        if lon_min <= lon_max:
            data = data.sel({lon_name: slice(lon_min, lon_max)})
        else:
            data = xr.concat(
                [
                    data.sel({lon_name: slice(lon_min, np.nanmax(lon_vals))}),
                    data.sel({lon_name: slice(np.nanmin(lon_vals), lon_max)}),
                ],
                dim=lon_name,
            )

        lat_vals = data[lat_name].values
        lat_slice = slice(lat_min, lat_max) if lat_vals[0] < lat_vals[-1] else slice(lat_max, lat_min)
        data = data.sel({lat_name: lat_slice})
        return data
