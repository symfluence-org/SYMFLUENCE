# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Forcing File Generator

Generates SWAT precipitation (.pcp) and temperature (.tmp) forcing files
from ERA5 NetCDF data, including unit conversion and daily resampling.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SWATForcingGenerator:
    """Generates SWAT forcing files (.pcp and .tmp) from ERA5 data.

    Args:
        preprocessor: Parent SWATPreProcessor instance providing access
            to config, logger, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    def generate_forcing_files(self, start_date: datetime, end_date: datetime) -> None:
        """Generate SWAT forcing files (.pcp and .tmp) from ERA5 data.

        SWAT .pcp format: Fixed-width text with daily precipitation [mm]
        SWAT .tmp format: Fixed-width text with daily max/min temperature [deg C]
        """
        logger.info("Generating SWAT forcing files...")

        try:
            forcing_ds = self.pp._load_forcing_data()
            self.write_pcp_file(forcing_ds, start_date, end_date)
            self.write_tmp_file(forcing_ds, start_date, end_date)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not load forcing data: {e}, generating synthetic")
            self.generate_synthetic_forcing(start_date, end_date)

    def write_pcp_file(self, forcing_ds, start_date, end_date) -> None:
        """Write SWAT precipitation file (.pcp)."""
        precip_data, src_var = self.pp._extract_variable(
            forcing_ds,
            ['precipitation_flux', 'precipitation', 'pr', 'precip', 'tp', 'PREC']
        )

        if precip_data is None:
            logger.warning("No precipitation variable found, using zeros")
            dates = pd.date_range(start_date, end_date, freq='D')
            precip_data = np.zeros(len(dates))
        else:
            # Unit conversion
            src_units = ''
            if src_var and src_var in forcing_ds:
                src_units = forcing_ds[src_var].attrs.get('units', '')

            # Determine timestep in seconds for rate->amount conversion
            if 'time' in forcing_ds and len(forcing_ds['time']) > 1:
                dt_ns = np.diff(forcing_ds['time'].values[:2])[0]
                dt_seconds = float(dt_ns / np.timedelta64(1, 's'))
            else:
                dt_seconds = 3600  # Default to hourly

            if 'mm/s' in src_units or src_var == 'precipitation_flux':
                precip_data = precip_data * dt_seconds  # mm/s -> mm per timestep
                logger.info(f"Converted {src_var} from mm/s to mm/timestep (dt={dt_seconds}s)")
            elif src_units == 'm' or src_var == 'tp':
                precip_data = precip_data * 1000.0  # m -> mm
                logger.info(f"Converted {src_var} from m to mm")
            elif 'kg' in src_units and 'm-2' in src_units and 's-1' in src_units:
                precip_data = precip_data * dt_seconds  # kg/m2/s -> mm per timestep
                logger.info(f"Converted {src_var} from kg/m2/s to mm/timestep (dt={dt_seconds}s)")

        # Resample to daily if sub-daily
        times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='D')
        precip_series = pd.Series(precip_data[:len(times)], index=pd.DatetimeIndex(times))
        precip_daily = precip_series.resample('D').sum()

        # Ensure non-negative
        precip_daily = precip_daily.clip(lower=0.0)

        # Write .pcp file
        # Header: 4 lines (title, column headers, lat/lon/elev, elevation integers)
        # Data: format (i4,i3,1800f5.1) -- year(4), julday(3), precip(5.1) per gage
        pcp_path = self.pp.forcing_dir / 'pcp1.pcp'
        lines = []
        lines.append("Station  1: SYMFLUENCE generated")
        lines.append("Lati    Long  Elev")
        props = self.pp._get_catchment_properties()
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i5) -- 7 spaces + 5-char int per gage
        lines.append(f"{'':7s}{int(props['elev']):5d}")

        for date, precip in precip_daily.items():
            year = date.year
            jday = date.timetuple().tm_yday
            lines.append(f"{year:4d}{jday:3d}{precip:5.1f}")

        pcp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Precipitation file written: {pcp_path}")

    def write_tmp_file(self, forcing_ds, start_date, end_date) -> None:
        """Write SWAT temperature file (.tmp) with daily max/min temperatures."""
        # Try to find temperature variable
        temp_data, src_var = self.pp._extract_variable(
            forcing_ds,
            ['air_temperature', 'temperature', 'tas', 'temp', 't2m', 'AIR_TEMP']
        )

        if temp_data is None:
            logger.warning("No temperature variable found, using synthetic")
            dates = pd.date_range(start_date, end_date, freq='D')
            doy = dates.dayofyear
            temp_data = 10 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
            tmax_daily = pd.Series(temp_data + 5, index=dates)
            tmin_daily = pd.Series(temp_data - 5, index=dates)
        else:
            # Unit conversion
            src_units = ''
            if src_var and src_var in forcing_ds:
                src_units = forcing_ds[src_var].attrs.get('units', '')

            if src_units == 'K' or np.nanmean(temp_data) > 100:
                temp_data = temp_data - 273.15
                logger.info(f"Converted {src_var} from K to deg C")

            times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='D')
            temp_series = pd.Series(temp_data[:len(times)], index=pd.DatetimeIndex(times))
            tmax_daily = temp_series.resample('D').max()
            tmin_daily = temp_series.resample('D').min()

        # Write .tmp file
        # Header: 4 lines (title, column headers, lat/lon/elev, elevation integers)
        # Data: format (i4,i3,3600f5.1) -- year(4), julday(3), tmax(5.1), tmin(5.1) per gage
        tmp_path = self.pp.forcing_dir / 'tmp1.tmp'
        lines = []
        lines.append("Station  1: SYMFLUENCE generated")
        lines.append("Lati    Long  Elev")
        props = self.pp._get_catchment_properties()
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i10) -- 7 spaces + 10-char int per gage
        lines.append(f"{'':7s}{int(props['elev']):10d}")

        for date in tmax_daily.index:
            year = date.year
            jday = date.timetuple().tm_yday
            tmax = tmax_daily[date]
            tmin = tmin_daily[date]
            lines.append(f"{year:4d}{jday:3d}{tmax:5.1f}{tmin:5.1f}")

        tmp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Temperature file written: {tmp_path}")

    def generate_synthetic_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """Generate synthetic forcing data for testing."""
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)
        props = self.pp._get_catchment_properties()

        # Synthetic precipitation
        precip = np.random.exponential(2.0, n)

        pcp_path = self.pp.forcing_dir / 'pcp1.pcp'
        lines = []
        lines.append("Station  1: SYMFLUENCE synthetic")
        lines.append("Lati    Long  Elev")
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i5)
        lines.append(f"{'':7s}{int(props['elev']):5d}")
        for i, date in enumerate(dates):
            year = date.year
            jday = date.timetuple().tm_yday
            lines.append(f"{year:4d}{jday:3d}{precip[i]:5.1f}")
        pcp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

        # Synthetic temperature
        doy = dates.dayofyear
        tmean = 10 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
        tmax = tmean + 5.0
        tmin = tmean - 5.0

        tmp_path = self.pp.forcing_dir / 'tmp1.tmp'
        lines = []
        lines.append("Station  1: SYMFLUENCE synthetic")
        lines.append("Lati    Long  Elev")
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i10)
        lines.append(f"{'':7s}{int(props['elev']):10d}")
        for i, date in enumerate(dates):
            year = date.year
            jday = date.timetuple().tm_yday
            lines.append(f"{year:4d}{jday:3d}{tmax[i]:5.1f}{tmin[i]:5.1f}")
        tmp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

        logger.info(f"Synthetic forcing files written to {self.pp.forcing_dir}")
