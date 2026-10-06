"""Reject successful exit codes with incomplete SUMMA output."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ValidationError


def validate_output_coverage(file_manager: Path, output_dir: Path, timestep_seconds=None):
    """Require every requested forcing timestamp in each timestep output shard.

    SUMMA can terminate with a Fortran STOP that returns zero. File existence
    and a zero subprocess exit status therefore do not establish completion.
    """
    text = Path(file_manager).read_text()
    def value(key):
        match = re.search(rf"^\s*{key}\s+['\"]([^'\"]+)['\"]", text, re.M)
        if not match:
            raise ValidationError(f"Missing {key} in {file_manager}")
        return match.group(1)
    start, end = pd.Timestamp(value('simStartTime')), pd.Timestamp(value('simEndTime'))
    prefix = value('outFilePrefix')
    files = sorted(Path(output_dir).glob(f'{prefix}*timestep.nc'))
    if not files:
        raise ValidationError(f'No SUMMA timestep output for {prefix}')
    for path in files:
        with xr.open_dataset(path) as ds:
            if 'time' not in ds:
                raise ValidationError(f'{path.name}: missing time coordinate')
            times = pd.DatetimeIndex(ds.time.values)
            if len(times) < 2 or times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
                raise ValidationError(f'{path.name}: invalid or insufficient timestamps')
            step = (pd.Timedelta(seconds=float(timestep_seconds)) if timestep_seconds is not None
                    else pd.Timedelta(int(np.median(np.diff(times.asi8))), unit='ns').round('ms'))
            expected = pd.date_range(start, end, freq=step)
            # NetCDF floating-point time decoding can introduce microsecond noise.
            # Compare every timestamp with an absolute tolerance; never relative
            # tolerance against epoch-sized values or just the two endpoints.
            tolerance_ns = min(1_000_000, step.value // 1000)
            if len(times) != len(expected) or not np.all(
                np.abs(times.asi8 - expected.asi8) <= tolerance_ns
            ):
                raise ValidationError(f'{path.name}: incomplete output coverage: {len(times)} records '
                                 f'{times[0]}–{times[-1]}; expected {len(expected)} '
                                 f'{start}–{end} at {step}')
    return files


def validate_routing_coverage(summa_file: Path, routing_dir: Path, prefix: str):
    """Require routed outputs to cover the same timestamps as complete SUMMA."""
    files = sorted(Path(routing_dir).glob(f'{prefix}.h.*.nc'))
    if not files:
        raise ValidationError('No mizuRoute output files')
    with xr.open_dataset(summa_file) as ds:
        expected = pd.DatetimeIndex(ds.time.values)
    parts = []
    for path in files:
        with xr.open_dataset(path) as ds:
            parts.extend(ds.time.values)
    times = pd.DatetimeIndex(parts).sort_values()
    if times.has_duplicates or not times.equals(expected):
        raise ValidationError('mizuRoute output coverage does not match complete SUMMA output')
    return files
