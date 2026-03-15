# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Forcing Processor

Converts basin-averaged forcing data to CLM DATM stream format.
Generates one NetCDF per year with variables mapped to CLM naming
conventions and a DATM streams file pointing to these.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# SYMFLUENCE variable → CLM DATM variable mapping
# ERA5 basin-averaged data is already in SI units (K, Pa, mm/s, W/m2)
FORCING_VAR_MAP = {
    'air_temperature': ('TBOT', 'K', None),          # Already K
    'specific_humidity': ('QBOT', 'kg/kg', None),
    'wind_speed': ('WIND', 'm/s', None),
    'surface_downwelling_shortwave_flux': ('FSDS', 'W/m2', None),
    'surface_downwelling_longwave_flux': ('FLDS', 'W/m2', None),
    'surface_air_pressure': ('PSRF', 'Pa', None),          # Already Pa
    'precipitation_flux': ('PRECTmms', 'mm/s', None),
}


class CLMForcingProcessor:
    """Processes and converts forcing data for CLM DATM.

    Reads basin-averaged forcing data from SYMFLUENCE standard format
    and writes DATM-compatible NetCDF files (one per year) with
    single-point spatial dimensions (LATIXY, LONGXY).
    """

    def __init__(self, config: Dict, logger_instance: logging.Logger):
        self.config = config
        self.logger = logger_instance

    def process_forcing(
        self,
        forcing_data_dir: Path,
        output_dir: Path,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Convert basin-averaged forcing to CLM DATM format.

        Args:
            forcing_data_dir: Directory with basin_averaged_data/
            output_dir: Output directory for DATM forcing files
            lat: Catchment centroid latitude
            lon: Catchment centroid longitude
            start_date: Simulation start date (YYYY-MM-DD)
            end_date: Simulation end date (YYYY-MM-DD)

        Returns:
            Path to output directory with forcing files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load basin-averaged forcing
        basin_avg_dir = forcing_data_dir / 'basin_averaged_data'
        if not basin_avg_dir.exists():
            basin_avg_dir = forcing_data_dir

        forcing_ds = self._load_basin_averaged_forcing(basin_avg_dir)
        if forcing_ds is None:
            raise FileNotFoundError(
                f"No basin-averaged forcing data found in {basin_avg_dir}"
            )

        # Clip to simulation period
        forcing_ds = forcing_ds.sel(
            time=slice(start_date, end_date)
        )

        # Convert variables to CLM DATM naming
        datm_ds = self._convert_to_datm(forcing_ds, lat, lon)

        # Write one file per year
        years = np.unique(datm_ds['time'].dt.year.values)
        written_files = []

        for year in years:
            year_ds = datm_ds.sel(
                time=datm_ds['time'].dt.year == year
            )
            fname = output_dir / f"clmforc.{year}.nc"
            self._write_datm_file(year_ds, fname)
            written_files.append(fname)
            self.logger.debug(f"Wrote DATM forcing: {fname}")

        # Generate streams file
        streams_file = output_dir / 'user_datm.streams.txt'
        self._write_streams_file(
            streams_file, output_dir, written_files, lat, lon
        )

        self.logger.info(
            f"CLM forcing: {len(written_files)} files written to {output_dir}"
        )
        return output_dir

    def _load_basin_averaged_forcing(
        self, data_dir: Path
    ) -> Optional[xr.Dataset]:
        """Load basin-averaged forcing from NetCDF or CSV files."""
        # Try NetCDF first
        nc_files = list(data_dir.glob('*.nc'))
        if nc_files:
            ds = xr.open_mfdataset(nc_files, combine='by_coords')
            return ds

        # Try CSV
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            dfs = []
            for f in csv_files:
                df = pd.read_csv(f, parse_dates=['time'], index_col='time')
                dfs.append(df)
            if dfs:
                combined = pd.concat(dfs, axis=1)
                return xr.Dataset.from_dataframe(combined)

        return None

    def _convert_to_datm(
        self, forcing_ds: xr.Dataset, lat: float, lon: float
    ) -> xr.Dataset:
        """Convert forcing dataset to DATM variable names and units."""
        data_vars = {}

        for sym_var, (datm_var, units, converter) in FORCING_VAR_MAP.items():
            if sym_var in forcing_ds:
                values = forcing_ds[sym_var].values
                if converter is not None:
                    values = converter(values)

                # Reshape to (time, 1, 1) for single-point DATM
                if values.ndim == 1:
                    values = values[:, np.newaxis, np.newaxis]
                elif values.ndim == 2:
                    # Already has spatial dim, take first point
                    values = values[:, 0:1, np.newaxis]

                data_vars[datm_var] = xr.DataArray(
                    data=values,
                    dims=['time', 'LATIXY', 'LONGXY'],
                    attrs={'units': units, 'long_name': datm_var},
                )
            else:
                self.logger.warning(
                    f"Forcing variable '{sym_var}' not found, "
                    f"'{datm_var}' will be missing"
                )

        # Derive missing variables if possible
        data_vars = self._derive_missing_variables(
            data_vars, forcing_ds, lat
        )

        # Create coordinate arrays
        latixy = xr.DataArray(
            data=np.array([[lat]]),
            dims=['LATIXY', 'LONGXY'],
            attrs={'units': 'degrees_north'},
        )
        longxy = xr.DataArray(
            data=np.array([[lon]]),
            dims=['LATIXY', 'LONGXY'],
            attrs={'units': 'degrees_east'},
        )

        datm_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': forcing_ds['time'],
                'LATIXY': [lat],
                'LONGXY': [lon],
            },
        )
        datm_ds['LATIXY_data'] = latixy
        datm_ds['LONGXY_data'] = longxy

        return datm_ds

    def _derive_missing_variables(
        self,
        data_vars: Dict,
        forcing_ds: xr.Dataset,
        lat: float,
    ) -> Dict:
        """Derive missing DATM variables from available data."""
        # Derive specific humidity from relative humidity + temperature + pressure
        if 'QBOT' not in data_vars:
            if all(v in forcing_ds for v in ['relative_humidity', 'air_temperature', 'surface_air_pressure']):
                T_K = forcing_ds['air_temperature'].values  # Already K
                P_Pa = forcing_ds['surface_air_pressure'].values  # Already Pa
                RH = forcing_ds['relative_humidity'].values / 100.0

                # Saturation vapor pressure (Buck equation)
                e_sat = 611.21 * np.exp(
                    (18.678 - (T_K - 273.15) / 234.5)
                    * ((T_K - 273.15) / (257.14 + (T_K - 273.15)))
                )
                e = RH * e_sat
                q = 0.622 * e / (P_Pa - 0.378 * e)

                data_vars['QBOT'] = xr.DataArray(
                    data=q[:, np.newaxis, np.newaxis],
                    dims=['time', 'LATIXY', 'LONGXY'],
                    attrs={'units': 'kg/kg', 'long_name': 'QBOT (derived)'},
                )
                self.logger.debug("Derived QBOT from relhum/airtemp/airpres")

        # Derive longwave from temperature if missing (Stefan-Boltzmann approx)
        if 'FLDS' not in data_vars and 'air_temperature' in forcing_ds:
            T_K = forcing_ds['air_temperature'].values  # Already K
            sigma = 5.67e-8
            emissivity = 0.85
            FLDS = emissivity * sigma * T_K**4

            data_vars['FLDS'] = xr.DataArray(
                data=FLDS[:, np.newaxis, np.newaxis],
                dims=['time', 'LATIXY', 'LONGXY'],
                attrs={'units': 'W/m2', 'long_name': 'FLDS (derived)'},
            )
            self.logger.debug("Derived FLDS from Stefan-Boltzmann approximation")

        return data_vars

    def _write_datm_file(self, ds: xr.Dataset, filepath: Path) -> None:
        """Write DATM-formatted NetCDF file."""
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                'dtype': 'float64',
                '_FillValue': 1.0e36,
            }

        # Time encoding — hourly data needs hours resolution
        encoding['time'] = {
            'units': 'hours since 1900-01-01 00:00:00',
            'calendar': 'standard',
            'dtype': 'float64',
        }

        ds.to_netcdf(filepath, encoding=encoding, format='NETCDF4')

    def _write_streams_file(
        self,
        streams_file: Path,
        forcing_dir: Path,
        forcing_files: list,
        lat: float,
        lon: float,
    ) -> None:
        """Write DATM streams definition file."""
        file_list = '\n'.join(
            f"      {f.name}" for f in sorted(forcing_files)
        )

        content = f"""<?xml version="1.0"?>
<file id="stream_CLM_FORCING" version="2.0">
  <dataSource>GENERIC</dataSource>
  <domainInfo>
    <variableNames>
      time          time
      LONGXY        LONGXY
      LATIXY        LATIXY
    </variableNames>
    <filePath>{forcing_dir}</filePath>
    <fileNames>
{file_list}
    </fileNames>
  </domainInfo>
  <fieldInfo>
    <variableNames>
      TBOT      tbot
      QBOT      shum
      WIND      wind
      FSDS      swdn
      FLDS      lwdn
      PSRF      pbot
      PRECTmms  prec
    </variableNames>
    <filePath>{forcing_dir}</filePath>
    <fileNames>
{file_list}
    </fileNames>
    <offset>0</offset>
  </fieldInfo>
</file>
"""
        streams_file.write_text(content)
        self.logger.debug(f"Wrote DATM streams file: {streams_file}")
