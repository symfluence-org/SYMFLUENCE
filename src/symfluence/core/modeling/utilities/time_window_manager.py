# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Time Window Manager

Shared utility for handling simulation time windows across model preprocessors.
Consolidates time parsing, validation, and alignment logic that was previously
duplicated in SUMMA, FUSE, NGEN, and GR preprocessors.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import xarray as xr

from symfluence.core.mixins.logging import _class_logger_name
from symfluence.core.modeling.utilities.periods import parse_period

logger = logging.getLogger(__name__)


class TimeWindowManager:
    """
    Manages simulation time windows for model preprocessors.

    Handles:
    - Parsing time strings from config (including 'default' handling)
    - Validating time formats
    - Extracting time ranges from forcing files
    - Aligning simulation times to forcing timesteps
    """

    # Supported time formats
    DATETIME_FORMATS = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the time window manager.

        Args:
            config: Configuration dictionary (typed or dict)
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(
            _class_logger_name(self.__class__))

    def get_simulation_times(
        self,
        forcing_path: Optional[Path] = None,
        default_start_offset_days: int = 0,
        default_end_offset_days: int = 0
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get simulation start and end times from config.

        Handles 'default' string by extracting times from forcing files.

        Args:
            forcing_path: Path to forcing files (required if using 'default')
            default_start_offset_days: Days to offset from forcing start (warm-up)
            default_end_offset_days: Days to offset from forcing end

        Returns:
            Tuple of (start_time, end_time) as pd.Timestamp
        """
        sim_start = self._get_config_value('SIMULATION_START_DATE', 'default')
        sim_end = self._get_config_value('SIMULATION_END_DATE', 'default')

        # Handle 'default' strings
        if sim_start == 'default' or sim_end == 'default':
            if forcing_path is None:
                raise ValueError(
                    "forcing_path required when using 'default' simulation times"
                )
            forcing_start, forcing_end = self.get_forcing_time_range(forcing_path)

            if sim_start == 'default':
                sim_start = forcing_start + pd.Timedelta(days=default_start_offset_days)
            if sim_end == 'default':
                sim_end = forcing_end - pd.Timedelta(days=default_end_offset_days)

        # Parse string times
        start_time = self.parse_time_string(sim_start)
        end_time = self.parse_time_string(sim_end)

        # Validate order
        if start_time >= end_time:
            raise ValueError(
                f"Simulation start ({start_time}) must be before end ({end_time})"
            )

        return start_time, end_time

    def get_calibration_period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get calibration period from config.

        Reads ``CALIBRATION_PERIOD_START``/``_END`` when both are set, else
        the ``CALIBRATION_PERIOD`` ``'start, end'`` string.

        Returns:
            Tuple of (start_time, end_time) as pd.Timestamp
        """
        return self._get_period('CALIBRATION', 'Calibration')

    def get_evaluation_period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get evaluation period from config.

        Reads ``EVALUATION_PERIOD_START``/``_END`` when both are set, else
        the ``EVALUATION_PERIOD`` ``'start, end'`` string.

        Returns:
            Tuple of (start_time, end_time) as pd.Timestamp
        """
        return self._get_period('EVALUATION', 'Evaluation')

    def _get_period(self, prefix: str, label: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start = self._get_config_value(f'{prefix}_PERIOD_START', None)
        end = self._get_config_value(f'{prefix}_PERIOD_END', None)
        if start is not None and end is not None:
            return self.parse_time_string(start), self.parse_time_string(end)

        period = parse_period(self._get_config_value(f'{prefix}_PERIOD', None))
        if period is None:
            raise ValueError(f"{label} period not defined in config")
        return period

    def get_forcing_time_range(
        self,
        forcing_path: Path,
        time_var: str = 'time'
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Extract time range from forcing NetCDF files.

        Args:
            forcing_path: Path to directory containing forcing files
            time_var: Name of time variable in files

        Returns:
            Tuple of (min_time, max_time) as pd.Timestamp
        """
        forcing_files = sorted(forcing_path.glob('*.nc'))
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")

        min_time = None
        max_time = None

        for f in forcing_files:
            try:
                with xr.open_dataset(f) as ds:
                    if time_var not in ds.coords and time_var not in ds.dims:
                        continue
                    times = pd.to_datetime(ds[time_var].values)
                    if min_time is None or times.min() < min_time:
                        min_time = times.min()
                    if max_time is None or times.max() > max_time:
                        max_time = times.max()
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Could not read time from {f}: {e}", exc_info=True)
                continue

        if min_time is None or max_time is None:
            raise ValueError(f"Could not extract time range from {forcing_path}")

        return pd.Timestamp(min_time), pd.Timestamp(max_time)

    def parse_time_string(
        self,
        time_value: Union[str, datetime, pd.Timestamp]
    ) -> pd.Timestamp:
        """
        Parse a time value to pd.Timestamp.

        Args:
            time_value: Time as string, datetime, or Timestamp

        Returns:
            pd.Timestamp
        """
        if isinstance(time_value, pd.Timestamp):
            return time_value
        if isinstance(time_value, datetime):
            return pd.Timestamp(time_value)
        if isinstance(time_value, str):
            for fmt in self.DATETIME_FORMATS:
                try:
                    return pd.Timestamp(datetime.strptime(time_value, fmt))
                except ValueError:
                    continue
            # Try pandas parser as fallback
            try:
                return pd.Timestamp(time_value)
            except (ValueError, TypeError):
                pass
            raise ValueError(
                f"Could not parse time string '{time_value}'. "
                f"Expected formats: {self.DATETIME_FORMATS}"
            )
        raise TypeError(f"Unsupported time type: {type(time_value)}")

    def validate_time_format(
        self,
        time_str: str,
        expected_format: str = "%Y-%m-%d %H:%M"
    ) -> bool:
        """
        Validate that a time string matches expected format.

        Args:
            time_str: Time string to validate
            expected_format: Expected datetime format

        Returns:
            True if valid, False otherwise
        """
        try:
            datetime.strptime(time_str, expected_format)
            return True
        except ValueError:
            return False

    def align_to_forcing_timestep(
        self,
        target_time: pd.Timestamp,
        forcing_times: pd.DatetimeIndex,
        direction: str = 'nearest'
    ) -> pd.Timestamp:
        """
        Align a target time to the nearest forcing timestep.

        Args:
            target_time: Time to align
            forcing_times: Available forcing timesteps
            direction: 'nearest', 'forward', or 'backward'

        Returns:
            Aligned timestamp
        """
        if direction == 'nearest':
            idx = forcing_times.get_indexer([target_time], method='nearest')[0]
        elif direction == 'forward':
            idx = forcing_times.get_indexer([target_time], method='bfill')[0]
        elif direction == 'backward':
            idx = forcing_times.get_indexer([target_time], method='ffill')[0]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        if idx < 0:
            raise ValueError(f"Could not align {target_time} to forcing times")

        return forcing_times[idx]

    def create_time_index(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str = 'D'
    ) -> pd.DatetimeIndex:
        """
        Create a time index for the simulation period.

        Args:
            start: Start time
            end: End time
            freq: Frequency string ('D' for daily, 'h' for hourly, etc.)

        Returns:
            pd.DatetimeIndex
        """
        return pd.date_range(start=start, end=end, freq=freq)

    def subset_dataset_to_time(
        self,
        ds: xr.Dataset,
        start: pd.Timestamp,
        end: pd.Timestamp,
        time_var: str = 'time',
        label: str = 'Dataset'
    ) -> xr.Dataset:
        """
        Subset a dataset to a time window.

        Args:
            ds: Dataset to subset
            start: Start time
            end: End time
            time_var: Name of time coordinate
            label: Label for logging

        Returns:
            Subsetted dataset
        """
        if time_var not in ds.coords and time_var not in ds.dims:
            self.logger.warning(f"{label} has no '{time_var}' coordinate")
            return ds

        original_len = len(ds[time_var])
        ds = ds.sel({time_var: slice(start, end)})
        new_len = len(ds[time_var])

        self.logger.debug(
            f"Subset {label} from {original_len} to {new_len} timesteps "
            f"({start} to {end})"
        )

        return ds

    def _get_config_value(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from config, handling typed config objects.

        Args:
            key: Config key
            default: Default value if not found

        Returns:
            Config value or default
        """
        # Try as typed config attribute first
        if hasattr(self.config, key):
            value = getattr(self.config, key)
            if value is not None:
                return value

        # Try as dict key
        if isinstance(self.config, dict):
            return self.config.get(key, default)

        # Try flattened typed config dict
        if hasattr(self.config, 'to_dict'):
            return self.config.to_dict(flatten=True).get(key, default)

        return default
