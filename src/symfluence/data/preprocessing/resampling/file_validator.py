# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
File Validator

Validates forcing files for proper structure and content.
"""

import logging
from pathlib import Path

import xarray as xr


class FileValidator:
    """
    Validates forcing files for proper structure and content.

    Checks for required dimensions, variables, and reasonable file sizes.
    """

    # Expected forcing variables (CFIF names + legacy SUMMA names for backward compat)
    EXPECTED_VARS = [
        # CFIF standard names
        'surface_air_pressure', 'surface_downwelling_longwave_flux',
        'surface_downwelling_shortwave_flux', 'precipitation_flux',
        'air_temperature', 'specific_humidity', 'wind_speed',
        # Legacy SUMMA-style names (for pre-existing data files)
        'airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate',
        'airtemp', 'spechum', 'windspd',
    ]

    # Minimum file size for valid data (100KB)
    MIN_FILE_SIZE = 100000

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize file validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def validate(self, file_path: Path, worker_str: str = "") -> bool:
        """
        Validate that a forcing file has proper structure.

        Args:
            file_path: Path to the NetCDF file to validate
            worker_str: Optional worker identifier for logging

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            with xr.open_dataset(file_path, engine="h5netcdf") as ds:
                # Check 1: Must have time dimension
                if 'time' not in ds.dims:
                    self.logger.warning(
                        f"{worker_str}File {file_path.name} missing time dimension"
                    )
                    return False

                # Check 2: Time dimension should have at least 1 timestep
                time_size = ds.sizes.get('time', 0)
                if time_size < 1:
                    self.logger.warning(
                        f"{worker_str}File {file_path.name} has empty time dimension"
                    )
                    return False

                # Check 3: Should have at least one forcing variable
                has_forcing_var = any(var in ds.data_vars for var in self.EXPECTED_VARS)

                if not has_forcing_var:
                    self.logger.warning(
                        f"{worker_str}File {file_path.name} missing forcing variables. "
                        f"Has: {list(ds.data_vars)}"
                    )
                    return False

                # Check 4: File should be larger than just metadata
                file_size = file_path.stat().st_size
                if file_size < self.MIN_FILE_SIZE:
                    self.logger.warning(
                        f"{worker_str}File {file_path.name} suspiciously small "
                        f"({file_size} bytes). Likely contains only metadata."
                    )
                    return False

                self.logger.debug(f"{worker_str}File {file_path.name} validated successfully")
                return True

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(
                f"{worker_str}Error validating file {file_path.name}: {str(e)}"
            )
            return False
