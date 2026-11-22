# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
EM-Earth climate forcing data acquisition from AWS S3.

Provides automated download of EM-Earth reanalysis data with support for
deterministic and probabilistic variants, multi-year coverage, and spatial subsetting.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import s3fs
import xarray as xr

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# EM-Earth published record (Tang et al. 2022): daily data 1950-01 .. 2019-12.
_RECORD_START = pd.Timestamp('1950-01-01')
_RECORD_END = pd.Timestamp('2019-12-31')

_FRDR_DATASET_URL = "https://www.frdr-dfdr.ca/repo/dataset/8d30ab02-f2bd-4d05-ae43-11f4a387e5ad"


@R.acquisition_handlers.add('EM-EARTH')
@R.acquisition_handlers.add('EM_EARTH')
class EMEarthAcquirer(BaseAcquisitionHandler):
    """
    Acquires EM-Earth global climate reanalysis data from AWS S3.

    EM-Earth provides daily meteorological variables at 0.1° resolution
    globally from 1950-2019, available in deterministic and probabilistic
    variants. Data includes precipitation, temperature (mean, range), and
    dewpoint temperature.

    S3 access: the ``emearth`` bucket allows anonymous LIST but currently
    denies anonymous GET. Set ``EM_EARTH_S3_ANON: false`` to use the standard
    AWS credential chain (environment variables, ``~/.aws/credentials``,
    instance profile), or stage the data from FRDR instead:
    https://www.frdr-dfdr.ca/repo/dataset/8d30ab02-f2bd-4d05-ae43-11f4a387e5ad

    Bucket layout (verified live): monthly files sit directly under
    ``emearth/nc/<product>/<var>/`` — there are no region subfolders.
    """

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading EM-Earth data from AWS S3")
        anon = bool(self._get_config_value(lambda: None, default=True, dict_key='EM_EARTH_S3_ANON'))
        fs = s3fs.S3FileSystem(anon=anon)
        emearth_type = str(self._get_config_value(
            lambda: self.config.forcing.em_earth.data_type,
            dict_key='EM_EARTH_DATA_TYPE',
        )).lower()
        base_folder = "nc/deterministic_raw_daily" if emearth_type == "deterministic" else "nc/probabilistic_daily"
        precip_var = self._get_config_value(lambda: self.config.forcing.em_earth.prcp_var, dict_key='EM_PRCP')
        variables = [precip_var, "tmean", "trange", "tdew"]

        if self.start_date > _RECORD_END or self.end_date < _RECORD_START:
            raise DataAcquisitionError(
                f"EM-Earth record covers {_RECORD_START.date()} to {_RECORD_END.date()}; the requested "
                f"window {self.start_date.date()} to {self.end_date.date()} is entirely outside it. "
                "Choose a window within the record or use a different forcing dataset."
            )
        # Only probe the months actually requested (clamped to the record).
        months = pd.period_range(
            max(self.start_date, _RECORD_START), min(self.end_date, _RECORD_END), freq='M'
        )

        all_datasets = {}
        for var in variables:
            var_datasets = []
            for period in months:
                ym = f"{period.year}{period.month:02d}"
                fname = f"EM_Earth_{emearth_type}_daily_{var}_{ym}.nc"
                # No region path component: bucket keys are
                # emearth/nc/<product>/<var>/<fname> (verified against live listing).
                key = f"emearth/{base_folder}/{var}/{fname}"
                try:
                    if fs.exists(key):
                        with fs.open(key, "rb") as f:
                            ds = xr.open_dataset(f, engine="h5netcdf")
                            ds_subset = ds.sel(lat=slice(self.bbox["lat_min"], self.bbox["lat_max"]), lon=slice(self.bbox["lon_min"], self.bbox["lon_max"]))
                            real_start = max(self.start_date, period.start_time)
                            real_end = min(self.end_date, period.end_time)
                            ds_subset = ds_subset.sel(time=slice(real_start, real_end))
                            if len(ds_subset.time) > 0:
                                var_datasets.append(ds_subset.load())
                except PermissionError as e:
                    raise DataAcquisitionError(
                        "EM-Earth S3 access denied: the 'emearth' bucket denies anonymous GET. "
                        "Provide AWS credentials with bucket access (standard AWS credential "
                        "chain) and set EM_EARTH_S3_ANON: false in your configuration, or "
                        f"stage the data from FRDR: {_FRDR_DATASET_URL}"
                    ) from e
                except (OSError, KeyError, ValueError) as e:
                    self.logger.debug(f"EM-Earth file not available for {var} {ym}: {e}")
                    continue
            if var_datasets:
                all_datasets[var] = xr.concat(var_datasets, dim="time")
        if not all_datasets:
            raise DataAcquisitionError(
                "No EM-Earth data downloaded for the requested window "
                f"({self.start_date.date()} to {self.end_date.date()})."
            )
        ds_final = xr.merge(list(all_datasets.values()))
        ds_final.attrs.update({"source": "EM-Earth", "bbox": str(self.bbox)})
        save_dir = output_dir / "raw_data_em_earth" if self._get_config_value(
            lambda: self.config.forcing.supplement,
            dict_key='SUPPLEMENT_FORCING',
        ) else output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / f"{self.domain_name}_EM-Earth_{emearth_type}_{months[0].year}-{months[-1].year}.nc"
        ds_final.to_netcdf(output_file)
        return output_file
