# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Soil moisture observation handlers.

Provides acquisition and preprocessing of satellite soil moisture data
for multivariate hydrological model calibration and validation.

Supported products:
    - SMAP: NASA L-band passive microwave (~3 km, 2015-present)
    - ISMN: International Soil Moisture Network (in-situ, multi-depth)
    - ESA CCI: ESA Climate Change Initiative merged product (~25 km, 1978-present)
    - SMOS: ESA L-band passive microwave (~25 km, 2010-present)
    - ASCAT: EUMETSAT C-band active microwave (~25 km, 2007-present)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('smap')
class SMAPHandler(BaseObservationHandler):
    """
    Handles SMAP Soil Moisture data.
    """

    obs_type = "soil_moisture"
    source_name = "NASA_SMAP"
    SOURCE_INFO = {
        'source': 'NASA SMAP L3',
        'source_doi': '10.5067/HH4SZ2PXSP6A',
        'url': 'https://nsidc.org/data/spl3smp',
    }

    def acquire(self) -> Path:
        """Locate SMAP data."""
        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS').lower()

        smap_path = self._get_config_value(lambda: self.config.evaluation.smap.path, dict_key='SMAP_PATH')
        if isinstance(smap_path, str) and smap_path.lower() == 'default':
            smap_dir = self.project_observations_dir / "soil_moisture" / "smap"
        else:
            smap_dir = Path(smap_path)

        if not smap_dir.exists():
            smap_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        use_opendap = self._get_config_value(
            lambda: self.config.evaluation.smap.use_opendap,
            dict_key='SMAP_USE_OPENDAP',
        )

        if list(smap_dir.glob("*.nc")) and not force_download:
            return smap_dir
        if not use_opendap and not force_download:
            for pattern in ("*.h5", "*.hdf5"):
                if list(smap_dir.glob(pattern)):
                    return smap_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for SMAP soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('SMAP', self.config, self.logger)
            return acquirer.download(smap_dir)
        return smap_dir

    def process(self, input_path: Path) -> Path:
        """Process SMAP NetCDF data."""
        self.logger.info(f"Processing SMAP Soil Moisture for domain: {self.domain_name}")

        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            for pattern in ("*.h5", "*.hdf5"):
                nc_files.extend(input_path.glob(pattern))
        if not nc_files:
            self.logger.warning("No SMAP NetCDF files found")
            return input_path

        # Strategy: spatial average over bounding box if multiple pixels
        # For simplicity in this implementation, we take the mean of the first file
        results = []
        for f in nc_files:
            try:
                try:
                    ds = xr.open_dataset(f, engine='netcdf4')
                except (OSError, ValueError):
                    # netcdf4 engine failed, try h5netcdf fallback
                    ds = xr.open_dataset(f, engine='h5netcdf')
            except (OSError, ValueError) as exc:
                self.logger.warning(f"Skipping unreadable SMAP file {f.name}: {exc}")
                continue
            with ds:
                # SMAP variables often named 'soil_moisture', 'sm_surface', or 'sm_rootzone'
                var_names = [
                    v for v in ds.data_vars
                    if 'soil_moisture' in v.lower() or 'sm_surface' in v.lower() or 'sm_rootzone' in v.lower()
                ]
                if not var_names:
                    continue

                file_frames = []
                for var_name in var_names:
                    output_name = var_name
                    if 'sm_surface' in var_name.lower():
                        output_name = 'surface_sm'
                    elif 'rootzone' in var_name.lower():
                        output_name = 'rootzone_sm'

                    # Spatial average
                    mean_sm = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d != 'time'])
                    df_ts = mean_sm.to_dataframe().reset_index()
                    df_ts = df_ts.rename(columns={var_name: output_name})
                    if 'time' in df_ts.columns:
                        df_ts = df_ts.set_index('time')[[output_name]]
                    file_frames.append(df_ts)

                if file_frames:
                    results.append(pd.concat(file_frames, axis=1))

        if not results:
            self.logger.warning("No SMAP data could be extracted")
            return input_path

        df = pd.concat(results).sort_index()
        if 'time' in df.columns:
            df = df.set_index('time')
        df = df.groupby(level=0).mean().sort_index()
        if self.start_date is not None and self.end_date is not None:
            df = df.loc[(df.index >= self.start_date) & (df.index <= self.end_date)]

        output_dir = self.project_observations_dir / "soil_moisture" / "smap" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_smap_processed.csv"
        df.to_csv(output_file)
        legacy_dir = self.project_observations_dir / "soil_moisture" / "preprocessed"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_file = legacy_dir / f"{self.domain_name}_smap_processed.csv"
        df.to_csv(legacy_file)

        self.logger.info(f"SMAP processing complete: {output_file}")
        return output_file


@R.observation_handlers.add('ismn')
class ISMNHandler(BaseObservationHandler):
    """
    Handles ISMN soil moisture data.
    """

    obs_type = "soil_moisture"
    source_name = "ISMN"

    def acquire(self) -> Path:
        """Locate or download ISMN data."""
        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS').lower()

        ismn_path = self._get_config_value(lambda: self.config.evaluation.ismn.path, dict_key='ISMN_PATH')
        if isinstance(ismn_path, str) and ismn_path.lower() == 'default':
            ismn_dir = self.project_observations_dir / "soil_moisture" / "ismn"
        else:
            ismn_dir = Path(ismn_path)
        ismn_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        if list(ismn_dir.glob("*.csv")) and not force_download:
            return ismn_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for ISMN soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('ISMN', self.config, self.logger)
            return acquirer.download(ismn_dir)
        return ismn_dir

    def process(self, input_path: Path) -> Path:
        """Process ISMN station data to a basin-average time series."""
        self.logger.info(f"Processing ISMN Soil Moisture for domain: {self.domain_name}")

        files: list[Any] = []
        for pattern in ("*.csv", "*.txt", "*.dat"):
            files.extend(input_path.glob(pattern))
        if not files:
            self.logger.warning("No ISMN files found")
            return input_path

        target_depth = self._get_target_depth()
        series_list = []
        depth_series = []
        for f in files:
            df = self._read_station_file(f)
            if df is None or df.empty:
                continue

            date_col = self._find_date_column(df.columns)
            if not date_col:
                continue

            sm_col = self._find_soil_moisture_column(df.columns)
            if not sm_col:
                continue

            df['DateTime'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['DateTime'])
            df = df.set_index('DateTime')

            depth_col = self._find_depth_column(df.columns)
            depth_value = None
            if depth_col:
                depth_numeric: pd.Series = pd.to_numeric(df[depth_col], errors='coerce')
                df['depth_m'] = depth_numeric.where(depth_numeric.notna(), np.nan)
                df['depth_m'] = df['depth_m'].apply(self._normalize_depth)
                df = df.dropna(subset=['depth_m'])
                if not df.empty:
                    depth_values = df['depth_m'].unique()
                    closest_depth = min(depth_values, key=lambda x: abs(x - target_depth))
                    df = df[df['depth_m'] == closest_depth]
                    depth_value = float(closest_depth)

            numeric_series = pd.to_numeric(df[sm_col], errors='coerce')  # type: ignore[call-overload,index]
            series = numeric_series.dropna()
            if series.empty:
                continue
            series_list.append(series)
            if depth_value is not None:
                depth_series.append((series, depth_value))

        if depth_series:
            min_diff = min(abs(d - target_depth) for _, d in depth_series)
            series_list = [s for s, d in depth_series if abs(d - target_depth) == min_diff]

        if not series_list:
            self.logger.warning("No ISMN soil moisture data could be extracted")
            return input_path

        combined = pd.concat(series_list, axis=1)
        combined = combined.groupby(level=0).mean().sort_index()
        if isinstance(combined, pd.DataFrame):
            combined = combined.mean(axis=1)

        if self.start_date is not None and self.end_date is not None:
            combined = combined.loc[(combined.index >= self.start_date) & (combined.index <= self.end_date)]

        aggregation = self._get_config_value(
            lambda: self.config.evaluation.ismn.temporal_aggregation,
            dict_key='ISMN_TEMPORAL_AGGREGATION',
        )
        if aggregation == 'daily_mean':
            combined = combined.resample('D').mean().dropna()

        col_name = f"sm_{target_depth:.2f}"
        output_df = combined.to_frame(name=col_name)

        output_dir = self.project_observations_dir / "soil_moisture" / "ismn" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_ismn_processed.csv"
        output_df.to_csv(output_file)

        self.logger.info(f"ISMN processing complete: {output_file}")
        return output_file

    def _read_station_file(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
            self.logger.debug(f"Standard CSV parse failed for {path.name}: {e}, trying whitespace delimiter")
            try:
                return pd.read_csv(path, sep=r'\s+')
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
                self.logger.warning(f"Skipping unreadable ISMN file {path.name}: {exc}")
                return None

    def _find_date_column(self, columns):
        candidates = [
            'timestamp', 'datetime', 'DateTime', 'date', 'Date', 'time', 'Time'
        ]
        for candidate in candidates:
            if candidate in columns:
                return candidate
        for col in columns:
            lower = col.lower()
            if any(term in lower for term in ['timestamp', 'datetime', 'date', 'time']):
                return col
        return None

    def _find_soil_moisture_column(self, columns):
        for col in columns:
            lower = col.lower()
            if any(term in lower for term in ['soil_moisture', 'soilmoisture', 'volumetric', 'vsm', 'theta']):
                return col
        for col in columns:
            lower = col.lower()
            if lower.startswith('sm') and 'flag' not in lower and 'qc' not in lower:
                return col
        return None

    def _find_depth_column(self, columns):
        for col in columns:
            if 'depth' in col.lower():
                return col
        return None

    def _normalize_depth(self, depth):
        try:
            depth_val = float(depth)
        except (ValueError, TypeError):
            return pd.NA
        if depth_val > 10:
            return depth_val / 100.0
        return depth_val

    def _get_target_depth(self) -> float:
        target_depth = self._get_config_value(
            lambda: self.config.evaluation.ismn.target_depth_m,
            dict_key='ISMN_TARGET_DEPTH_M',
        )
        try:
            return float(target_depth)
        except (ValueError, TypeError):
            self.logger.debug(f"Could not convert target_depth '{target_depth}' to float, using default 0.05")
            return 0.05

@R.observation_handlers.add('esa_cci_sm')
class ESACCISMHandler(BaseObservationHandler):
    """
    Handles ESA CCI Soil Moisture data.
    """

    obs_type = "soil_moisture"
    source_name = "ESA_CCI"

    def acquire(self) -> Path:
        """Locate ESA CCI SM data."""
        # ESA CCI SM path - fallback to default location
        esa_path = self._get_config_value(lambda: None, default=None, dict_key='ESA_CCI_SM_PATH')
        if esa_path:
            esa_dir = Path(esa_path)
        else:
            esa_dir = self.project_observations_dir / "soil_moisture" / "esa_cci"
        esa_dir.mkdir(parents=True, exist_ok=True)

        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS').lower()
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        if list(esa_dir.glob("*.nc")) and not force_download:
            return esa_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for ESA CCI soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('ESA_CCI_SM', self.config, self.logger)
            return acquirer.download(esa_dir)

        return esa_dir

    def process(self, input_path: Path) -> Path:
        """Process ESA CCI SM NetCDF data."""
        self.logger.info(f"Processing ESA CCI Soil Moisture for domain: {self.domain_name}")

        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No ESA CCI SM NetCDF files found")
            return input_path

        # Use bbox from base class (already parsed in __init__)
        north = west = south = east = None
        target_lat = target_lon = None
        if self.bbox:
            try:
                north = self.bbox.get('lat_max')
                south = self.bbox.get('lat_min')
                west = self.bbox.get('lon_min')
                east = self.bbox.get('lon_max')
                if all(v is not None for v in [north, south, west, east]):
                    target_lat = (north + south) / 2.0
                    if west <= east:
                        target_lon = (west + east) / 2.0
                    else:
                        # Dateline-crossing bbox; pick midpoint on wrapped domain
                        target_lon = ((west + 360.0 + east) / 2.0) % 360.0
                        if target_lon > 180.0:
                            target_lon -= 360.0
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning(f"Failed to parse bbox for ESA CCI SM subsetting: {e}")

        results = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                # ESA CCI SM variable is usually 'sm'
                if 'sm' not in ds.data_vars:
                    continue

                sm = ds['sm']
                if target_lat is not None and target_lon is not None and {'lat', 'lon'}.issubset(sm.dims):
                    # Use nearest pixel to reduce spatial smoothing
                    sm = sm.sel(lat=target_lat, lon=target_lon, method='nearest')
                elif None not in (north, west, south, east) and {'lat', 'lon'}.issubset(sm.dims):
                    lat_desc = sm['lat'][0] > sm['lat'][-1]
                    lat_slice = slice(north, south) if lat_desc else slice(south, north)
                    sm = sm.sel(lat=lat_slice)
                    if west <= east:
                        sm = sm.sel(lon=slice(west, east))
                    else:
                        sm = sm.where((sm['lon'] >= west) | (sm['lon'] <= east), drop=True)

                # Filter invalid values
                sm = sm.where((sm >= 0) & (sm <= 1))

                # Spatial average if still gridded, else keep point
                mean_sm = sm.mean(dim=[d for d in sm.dims if d != 'time'])
                df_ts = mean_sm.to_dataframe().reset_index()
                results.append(df_ts)

        if not results:
            self.logger.warning("No ESA CCI SM data could be extracted")
            return input_path

        df = pd.concat(results).sort_values('time')
        df = df.rename(columns={'sm': 'soil_moisture'})

        output_dir = self.project_observations_dir / "soil_moisture" / "esa_sm" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_esa_processed.csv"
        df.to_csv(output_file, index=False)

        self.logger.info(f"ESA CCI SM processing complete: {output_file}")
        return output_file


@R.observation_handlers.add('smos')
@R.observation_handlers.add('smos_sm')
class SMOSSMHandler(BaseObservationHandler):
    """
    Handles SMOS (Soil Moisture and Ocean Salinity) data.

    SMOS uses L-band passive microwave at ~25 km resolution.
    Data is acquired via CDS satellite-soil-moisture (passive-only sensor type).
    Available from 2010-present.
    """

    obs_type = "soil_moisture"
    source_name = "ESA_SMOS"

    def acquire(self) -> Path:
        """Locate or download SMOS SM data."""
        smos_path = self._get_config_value(lambda: None, default=None, dict_key='SMOS_SM_PATH')
        if smos_path:
            smos_dir = Path(smos_path)
        else:
            smos_dir = self.project_observations_dir / "soil_moisture" / "smos"
        smos_dir.mkdir(parents=True, exist_ok=True)

        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS').lower()
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        if list(smos_dir.glob("*.nc")) and not force_download:
            return smos_dir

        # Check for extracted directory
        extract_dir = smos_dir / "extracted"
        if extract_dir.exists() and list(extract_dir.glob("*.nc")) and not force_download:
            return extract_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for SMOS soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('SMOS', self.config, self.logger)
            return acquirer.download(smos_dir)

        return smos_dir

    def process(self, input_path: Path) -> Path:
        """Process SMOS SM NetCDF data to catchment-average time series."""
        self.logger.info(f"Processing SMOS Soil Moisture for domain: {self.domain_name}")

        # Search input path and extracted subdirectory
        nc_files = list(input_path.glob("*.nc"))
        extract_dir = input_path / "extracted"
        if extract_dir.exists():
            nc_files.extend(extract_dir.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No SMOS NetCDF files found")
            return input_path

        # Get bbox for spatial subsetting
        north = south = west = east = None
        if self.bbox:
            north = self.bbox.get('lat_max')
            south = self.bbox.get('lat_min')
            west = self.bbox.get('lon_min')
            east = self.bbox.get('lon_max')

        all_data = []
        for nc_file in nc_files:
            try:
                ds = self._open_dataset(nc_file)

                # Identify SM variable
                sm_var = None
                for var in ['sm', 'Soil_Moisture', 'soil_moisture',
                            'volumetric_surface_soil_moisture', 'SM']:
                    if var in ds:
                        sm_var = var
                        break
                if sm_var is None:
                    ds.close()
                    continue

                # Identify coordinates
                lat_var = 'lat' if 'lat' in ds.coords else 'latitude'
                lon_var = 'lon' if 'lon' in ds.coords else 'longitude'

                # Spatial subset with buffer for coarse resolution
                buffer = 0.25
                if all(v is not None for v in [north, south, west, east]):
                    lat_mask = ((ds[lat_var] >= south - buffer) &
                                (ds[lat_var] <= north + buffer))
                    lon_mask = ((ds[lon_var] >= west - buffer) &
                                (ds[lon_var] <= east + buffer))
                    ds_subset = ds.where(lat_mask & lon_mask, drop=True)
                else:
                    ds_subset = ds

                if ds_subset[sm_var].size == 0:
                    ds.close()
                    continue

                sm_values = ds_subset[sm_var].values
                times = pd.to_datetime(ds_subset['time'].values)

                for i, t in enumerate(times):
                    sm_slice = sm_values[i] if len(sm_values.shape) > 2 else sm_values
                    valid_mask = (sm_slice > 0) & (sm_slice < 1) & (~np.isnan(sm_slice))
                    if np.any(valid_mask):
                        all_data.append({
                            'date': t,
                            'soil_moisture': float(np.nanmean(sm_slice[valid_mask])),
                            'n_pixels': int(np.sum(valid_mask)),
                        })

                ds.close()
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Error processing SMOS file {nc_file.name}: {e}", exc_info=True)
                continue

        if not all_data:
            self.logger.warning("No SMOS SM data could be extracted")
            return input_path

        df = pd.DataFrame(all_data)
        df = df.set_index('date').sort_index()
        df = df[~df.index.duplicated(keep='first')]

        if self.start_date is not None and self.end_date is not None:
            df = df.loc[(df.index >= self.start_date) & (df.index <= self.end_date)]

        output_dir = self.project_observations_dir / "soil_moisture" / "smos" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_smos_processed.csv"
        df.to_csv(output_file)

        self.logger.info(f"SMOS processing complete: {output_file} ({len(df)} records)")
        return output_file


@R.observation_handlers.add('ascat')
@R.observation_handlers.add('ascat_sm')
class ASCATSMHandler(BaseObservationHandler):
    """
    Handles ASCAT (Advanced Scatterometer) soil moisture data.

    ASCAT uses C-band active microwave at ~25 km resolution.
    Natively provides degree of saturation; this handler converts to
    volumetric SM using a configurable porosity value.
    Data is acquired via CDS satellite-soil-moisture (active-only sensor type).
    Available from 2007-present.
    """

    obs_type = "soil_moisture"
    source_name = "EUMETSAT_ASCAT"

    # Default porosity for saturation-to-volumetric conversion
    DEFAULT_POROSITY = 0.45

    def acquire(self) -> Path:
        """Locate or download ASCAT SM data."""
        ascat_path = self._get_config_value(lambda: None, default=None, dict_key='ASCAT_SM_PATH')
        if ascat_path:
            ascat_dir = Path(ascat_path)
        else:
            ascat_dir = self.project_observations_dir / "soil_moisture" / "ascat"
        ascat_dir.mkdir(parents=True, exist_ok=True)

        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS').lower()
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        if list(ascat_dir.glob("*.nc")) and not force_download:
            return ascat_dir

        # Check for extracted directory
        extract_dir = ascat_dir / "extracted"
        if extract_dir.exists() and list(extract_dir.glob("*.nc")) and not force_download:
            return extract_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for ASCAT soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('ASCAT', self.config, self.logger)
            return acquirer.download(ascat_dir)

        return ascat_dir

    def process(self, input_path: Path) -> Path:
        """Process ASCAT SM NetCDF data to catchment-average time series.

        ASCAT provides degree of saturation (0-1). This is converted to
        volumetric SM by multiplying by porosity (configurable, default 0.45).
        """
        self.logger.info(f"Processing ASCAT Soil Moisture for domain: {self.domain_name}")

        # Search input path and extracted subdirectory
        nc_files = list(input_path.glob("*.nc"))
        extract_dir = input_path / "extracted"
        if extract_dir.exists():
            nc_files.extend(extract_dir.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No ASCAT NetCDF files found")
            return input_path

        # Get porosity for saturation-to-volumetric conversion
        porosity = float(self._get_config_value(lambda: None, default=self.DEFAULT_POROSITY, dict_key='ASCAT_POROSITY'))

        # Get bbox for spatial subsetting
        north = south = west = east = None
        if self.bbox:
            north = self.bbox.get('lat_max')
            south = self.bbox.get('lat_min')
            west = self.bbox.get('lon_min')
            east = self.bbox.get('lon_max')

        all_data = []
        for nc_file in nc_files:
            try:
                ds = self._open_dataset(nc_file)

                # Identify SM variable
                sm_var = None
                for var in ['sm', 'ssm', 'soil_moisture', 'surface_soil_moisture',
                            'surface_soil_moisture_saturation',
                            'volumetric_surface_soil_moisture']:
                    if var in ds:
                        sm_var = var
                        break
                if sm_var is None:
                    for var in ds.data_vars:
                        if 'sm' in var.lower() or 'soil' in var.lower():
                            sm_var = var
                            break
                if sm_var is None:
                    ds.close()
                    continue

                # Identify coordinates
                lat_var = 'lat' if 'lat' in ds.coords else 'latitude'
                lon_var = 'lon' if 'lon' in ds.coords else 'longitude'

                # Spatial subset with buffer
                buffer = 0.25
                if all(v is not None for v in [north, south, west, east]):
                    lat_mask = ((ds[lat_var] >= south - buffer) &
                                (ds[lat_var] <= north + buffer))
                    lon_mask = ((ds[lon_var] >= west - buffer) &
                                (ds[lon_var] <= east + buffer))
                    ds_subset = ds.where(lat_mask & lon_mask, drop=True)
                else:
                    ds_subset = ds

                if ds_subset[sm_var].size == 0:
                    ds.close()
                    continue

                sm_values = ds_subset[sm_var].values
                times = pd.to_datetime(ds_subset['time'].values)

                for i, t in enumerate(times):
                    sm_slice = sm_values[i] if len(sm_values.shape) > 2 else sm_values
                    valid_mask = ~np.isnan(sm_slice)
                    if np.any(valid_mask):
                        sm_valid = sm_slice[valid_mask]

                        # Convert percentage to fraction if needed
                        if np.nanmean(sm_valid) > 1.0:
                            sm_valid = sm_valid / 100.0

                        # Convert saturation to volumetric SM
                        # If variable is saturation-type, multiply by porosity
                        if 'saturation' in sm_var.lower() or np.nanmean(sm_valid) > 0.5:
                            sm_valid = sm_valid * porosity

                        # Filter physical range
                        phys_mask = (sm_valid > 0) & (sm_valid < 1)
                        if np.any(phys_mask):
                            all_data.append({
                                'date': t,
                                'soil_moisture': float(np.mean(sm_valid[phys_mask])),
                                'n_pixels': int(np.sum(phys_mask)),
                            })

                ds.close()
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Error processing ASCAT file {nc_file.name}: {e}", exc_info=True)
                continue

        if not all_data:
            self.logger.warning("No ASCAT SM data could be extracted")
            return input_path

        df = pd.DataFrame(all_data)
        df = df.set_index('date').sort_index()
        df = df[~df.index.duplicated(keep='first')]

        if self.start_date is not None and self.end_date is not None:
            df = df.loc[(df.index >= self.start_date) & (df.index <= self.end_date)]

        output_dir = self.project_observations_dir / "soil_moisture" / "ascat" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_ascat_processed.csv"
        df.to_csv(output_file)

        self.logger.info(
            f"ASCAT processing complete: {output_file} ({len(df)} records, "
            f"porosity={porosity})"
        )
        return output_file
