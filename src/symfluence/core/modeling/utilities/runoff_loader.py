# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared runoff loading utilities for routing models.

Provides model-specific runoff file resolution, variable detection,
and time precision fixing used by mizuRoute, tRoute, and dRoute.

The per-model runoff metadata itself is *not* declared here: each model
declares it as the ``runoff`` section of its registered
:class:`~symfluence.core.modeling.config_schema.ModelConfigSchema`, so an
external model package becomes routable by registering a schema and nothing
else. This module is the read side of that declaration.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.modeling.config_schema import RunoffConfig, registered_runoff_configs

logger = logging.getLogger(__name__)

#: Back-compat alias: the served value type used to be declared here.
ModelRunoffConfig = RunoffConfig


class _SchemaRunoffConfigs(Mapping):
    """Live, lowercase-keyed view of the registered ``runoff`` declarations.

    A mapping rather than a dict so late registrations (plugin discovery,
    tests registering a schema) are visible without re-import.
    """

    @staticmethod
    def _table() -> Dict[str, RunoffConfig]:
        return {name.lower(): cfg for name, cfg in registered_runoff_configs().items()}

    def __getitem__(self, key: str) -> RunoffConfig:
        return self._table()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())

    def __eq__(self, other: Any) -> bool:
        return self._table() == other

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{type(self).__name__}({self._table()!r})"


MODEL_CONFIGS: Mapping = _SchemaRunoffConfigs()


def _canonical_source_key(source_model: str) -> str:
    """Normalize a source-model spelling to a registered runoff key.

    Case- and whitespace-insensitive; model variants (GR4J/GR5J/GR6J -> GR)
    resolve through the ``aliases`` each model declares on its own runoff
    config, so the variant list is model-owned rather than hardcoded here.
    """
    key = source_model.strip().upper()
    for name, cfg in registered_runoff_configs().items():
        if key in cfg.aliases:
            return name.lower()
    return key.lower()


def get_model_config(source_model: str) -> RunoffConfig:
    """Get the runoff configuration for a source model.

    Raises:
        ConfigurationError: if *source_model* declares no runoff config.
            This used to fall back to SUMMA's layout, which turned a config
            typo — or a model that simply cannot feed a routing model — into
            a routing run against plausible-looking SUMMA paths, discovered
            only when someone noticed the routed hydrograph was wrong.
    """
    table = _SchemaRunoffConfigs._table()
    config = table.get(_canonical_source_key(source_model))
    if config is None:
        raise ConfigurationError(
            f"No runoff configuration registered for source model "
            f"'{source_model}'. Routable source models: "
            f"{sorted(name.upper() for name in table)}. Model packages declare "
            f"this as ModelConfigSchema.runoff at registration time; a model "
            f"without one cannot feed a routing model."
        )
    return config


def resolve_runoff_file(
    source_model: str,
    project_dir: Path,
    experiment_id: str,
    domain_name: str = '',
    config: Any = None,
) -> Optional[Path]:
    """
    Resolve the runoff output file path for a given source model.

    Handles FUSE's 6-char file ID truncation and model-specific
    output directory overrides from config.
    """
    model_cfg = get_model_config(source_model)
    model_key = source_model.strip().upper()

    # Resolve output directory
    output_dir: Optional[Path] = None
    if config is not None:
        try:
            config_dir_map = {
                'SUMMA': lambda: config.model.summa.experiment_output,
                'FUSE': lambda: config.model.fuse.experiment_output,
                'GR': lambda: config.model.gr.experiment_output,
                # No HYPE entry: HYPE is not a routable source (its cout is
                # already routed discharge), so get_model_config raises above
                # before this map is ever consulted for it.
                'NGEN': lambda: config.model.ngen.experiment_output,
            }
            getter = config_dir_map.get(model_key)
            if getter:
                configured = getter()
                if configured and configured != 'default':
                    output_dir = Path(configured)
        except (AttributeError, TypeError):
            pass

    if output_dir is None:
        output_dir = project_dir / f"simulations/{experiment_id}" / model_cfg.output_dir_name

    # Resolve filename
    file_id = experiment_id
    if model_key == 'FUSE':
        if config is not None:
            try:
                fuse_file_id = config.model.fuse.file_id
                if fuse_file_id:
                    file_id = fuse_file_id
            except (AttributeError, TypeError):
                pass
        if len(file_id) > 6:
            import hashlib
            file_id = hashlib.md5(file_id.encode(), usedforsecurity=False).hexdigest()[:6]

    filename = model_cfg.output_file_pattern.format(
        experiment_id=file_id if model_key == 'FUSE' else experiment_id,
        domain_name=domain_name,
    )

    runoff_path = output_dir / filename

    if not runoff_path.exists():
        # No "first *.nc in the directory" fallback: glob order is arbitrary,
        # so that handed routing a parameter trial, a restart or another
        # experiment's file — which the time-precision fix then rewrote in
        # place.
        logger.warning(f"Runoff file not found (expected {runoff_path})")
        return None

    return runoff_path


def detect_runoff_variable(ds: Any, source_model: str = 'SUMMA') -> Optional[str]:
    """
    Detect the runoff variable in a dataset based on source model.

    Tries model-specific variable first, then falls through common names.
    """
    model_cfg = get_model_config(source_model)

    # Model-specific variable first
    candidates = [model_cfg.default_var]

    # Common runoff variable names as fallback
    common_vars = [
        'averageRoutedRunoff', 'scalarTotalRunoff', 'q_routed',
        'runoff', 'q_runoff', 'cout', 'streamflow',
    ]
    for v in common_vars:
        if v not in candidates:
            candidates.append(v)

    for var in candidates:
        if var in ds.data_vars:
            return var

    # Last resort: any variable with 'runoff' in name
    for var in ds.data_vars:
        if 'runoff' in str(var).lower():
            return var

    return None


#: CF time-unit spellings -> pandas timedelta unit.
_CF_TIME_UNITS = {
    'seconds': 's', 'second': 's', 'secs': 's', 'sec': 's', 's': 's',
    'minutes': 'min', 'minute': 'min', 'mins': 'min', 'min': 'min',
    'hours': 'h', 'hour': 'h', 'hrs': 'h', 'hr': 'h', 'h': 'h',
    'days': 'D', 'day': 'D', 'd': 'D',
}
_CF_UNIT_NAMES = {'s': 'seconds', 'min': 'minutes', 'h': 'hours', 'D': 'days'}
_STANDARD_CALENDARS = {'', 'standard', 'gregorian', 'proleptic_gregorian'}


def parse_cf_time_units(units: str) -> Optional[tuple]:
    """Split CF ``"<unit> since <reference>"`` into (pandas unit, reference).

    The reference is returned as a timezone-naive UTC ``pd.Timestamp``.
    Returns None when *units* is not a numeric CF time encoding this module
    understands (no ``since``, an unknown unit such as months, an unparseable
    reference) — callers must then leave the axis alone rather than guess.
    """
    import pandas as pd

    parts = re.split(r'\s+since\s+', str(units).strip(), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    unit = _CF_TIME_UNITS.get(parts[0].strip().lower())
    if unit is None:
        return None
    try:
        reference = pd.Timestamp(parts[1].strip())
    except (ValueError, TypeError):
        return None
    if pd.isna(reference):
        return None
    if reference.tzinfo is not None:
        reference = reference.tz_convert('UTC').tz_localize(None)
    return unit, reference


def normalize_cf_time_units(units: str) -> str:
    """Same CF epoch, spelled ``YYYY-MM-DD HH:MM:SS`` in UTC.

    ISO spellings ('T' separator, timezone suffix) are valid CF but are
    rejected by Fortran readers such as mizuRoute's. The unit word is kept as
    written; unparseable strings are returned unchanged.
    """
    import pandas as pd

    parts = re.split(r'\s+since\s+', str(units), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return units
    try:
        reference = pd.Timestamp(parts[1])
    except (ValueError, TypeError):
        return units
    if pd.isna(reference):
        return units
    if reference.tzinfo is not None:
        reference = reference.tz_convert('UTC').tz_localize(None)
    return f"{parts[0]} since {reference.isoformat(sep=' ')}"


def _round_time_stamps(stamps: Any, rounding_freq: str) -> Any:
    """Round *stamps* to *rounding_freq*, or to their own step if that is finer.

    Rounding sub-hourly output to the hour would merge neighbouring stamps, so
    when the axis' own step (median spacing, to the second) is shorter than
    the requested frequency the stamps are snapped to that step instead,
    anchored on the first stamp so an off-grid phase (00:30, 01:30, ...) is
    preserved.
    """
    import numpy as np
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    anchor = pd.Timestamp('2000-01-01')
    freq_td = (anchor + to_offset(rounding_freq)) - anchor

    step = None
    if len(stamps) > 1:
        median_step = pd.Timedelta(np.median(np.diff(stamps.values))).round('s')
        if median_step > pd.Timedelta(0):
            step = median_step

    if step is None or step >= freq_td:
        return stamps.round(rounding_freq)

    origin = stamps[0].round('s')
    step_seconds = int(step / pd.Timedelta(seconds=1))
    n_steps = np.rint(np.asarray((stamps - origin) / step, dtype='float64')).astype('int64')
    return pd.DatetimeIndex(origin + pd.to_timedelta(n_steps * step_seconds, unit='s'))


def fix_time_precision(
    runoff_filepath: Path,
    rounding_freq: Optional[str] = 'h',
    normalize_epoch: bool = False,
) -> Path:
    """
    Remove sub-step noise from a runoff file's numeric CF time axis.

    The single implementation behind every routing model's "time precision
    fix". The axis is decoded from its CF ``units`` (seconds, minutes, hours
    or days since any reference), **every** stamp is rounded to
    *rounding_freq* — or to the axis' own step when that is finer — and the
    result is written back in the file's original unit and epoch.

    The file is only rewritten (temp file, then atomic replace) when a value
    or the units string actually changes; an already-clean file is untouched.

    Args:
        runoff_filepath: Path to the NetCDF runoff file.
        rounding_freq: pandas frequency to round to ('h', 'min', 's');
            ``None`` or ``'none'`` disables rounding.
        normalize_epoch: Also respell the reference epoch as
            ``YYYY-MM-DD HH:MM:SS`` UTC (needed by mizuRoute's Fortran parser).
            Applied even when rounding is disabled.

    Returns:
        *runoff_filepath*.

    Raises:
        ModelExecutionError: the file is missing, unreadable, has an empty or
            non-finite time axis, or rounding would merge distinct time stamps
            into duplicates.
    """
    import os

    import numpy as np
    import pandas as pd
    import xarray as xr

    from symfluence.core.exceptions import ModelExecutionError

    runoff_filepath = Path(runoff_filepath)
    if rounding_freq is not None and str(rounding_freq).lower() == 'none':
        rounding_freq = None
    if rounding_freq is None and not normalize_epoch:
        return runoff_filepath

    if not runoff_filepath.exists():
        raise ModelExecutionError(f"Runoff file not found: {runoff_filepath}")

    try:
        ds = xr.open_dataset(runoff_filepath, decode_times=False)
    except (OSError, RuntimeError, ValueError) as nc_err:
        raise ModelExecutionError(
            f"Failed to open runoff file {runoff_filepath}: {nc_err}. The file "
            f"appears to be corrupt or incomplete, which usually means the "
            f"upstream hydrological model failed or timed out before finishing."
        ) from nc_err

    try:
        if 'time' not in ds.variables:
            logger.warning(f"No 'time' variable in {runoff_filepath}; time axis left unchanged")
            return runoff_filepath

        time_values = np.asarray(ds['time'].values)
        if time_values.size == 0:
            raise ModelExecutionError(
                f"Empty time dimension in model output: {runoff_filepath}. The "
                f"upstream model output is incomplete or corrupted."
            )

        old_units = ds['time'].attrs.get('units')
        parsed = parse_cf_time_units(old_units) if old_units else None
        calendar = str(ds['time'].attrs.get('calendar', '')).lower()
        if (
            parsed is None
            or calendar not in _STANDARD_CALENDARS
            or not np.issubdtype(time_values.dtype, np.number)
        ):
            logger.warning(
                f"Time axis of {runoff_filepath} is not a numeric CF axis this "
                f"fix understands (units={old_units!r}, calendar={calendar or 'unset'!r}); "
                f"left unchanged"
            )
            return runoff_filepath
        time_unit, reference = parsed

        new_units = normalize_cf_time_units(old_units) if normalize_epoch else old_units
        new_values = time_values

        if rounding_freq is not None:
            if not np.all(np.isfinite(time_values.astype('float64'))):
                raise ModelExecutionError(
                    f"Non-finite values on the time axis of {runoff_filepath}"
                )
            stamps = pd.DatetimeIndex(reference + pd.to_timedelta(time_values, unit=time_unit))
            rounded = _round_time_stamps(stamps, rounding_freq)

            if rounded.has_duplicates and not stamps.has_duplicates:
                raise ModelExecutionError(
                    f"Rounding the time axis of {runoff_filepath} to "
                    f"'{rounding_freq}' would merge distinct time stamps into "
                    f"duplicates (step ~{pd.Timedelta(np.median(np.diff(stamps.values)))}). "
                    f"Refusing to rewrite the file; use a finer rounding "
                    f"frequency or 'none'."
                )

            candidate = np.asarray((rounded - reference) / pd.Timedelta(1, unit=time_unit), dtype='float64')
            if not np.array_equal(candidate, time_values.astype('float64')):
                max_shift = np.abs(rounded - stamps).max()
                logger.info(
                    f"Rounding time axis of {runoff_filepath.name} to "
                    f"'{rounding_freq}' (max shift {max_shift})"
                )
                if np.issubdtype(time_values.dtype, np.integer) and np.all(candidate == np.rint(candidate)):
                    new_values = candidate.astype(time_values.dtype)
                else:
                    new_values = candidate

        if new_values is time_values and new_units == old_units:
            logger.debug(f"Time axis of {runoff_filepath} is already clean")
            return runoff_filepath

        time_attrs = dict(ds['time'].attrs)
        time_attrs['units'] = new_units
        if new_values is not time_values and rounding_freq is not None:
            # Spell the unit and epoch canonically whenever values are rewritten.
            time_attrs['units'] = (
                f"{_CF_UNIT_NAMES[time_unit]} since {reference.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            time_attrs.setdefault('calendar', 'standard')
            time_attrs.setdefault('long_name', 'time')
        ds = ds.assign_coords(time=(ds['time'].dims, new_values, time_attrs))

        ds.load()
        temp_filepath = runoff_filepath.with_suffix('.tmp.nc')
        ds.to_netcdf(temp_filepath, format='NETCDF4')
    finally:
        ds.close()

    if os.name != 'nt':
        # Security rationale: Group-writable for HPC shared access
        os.chmod(temp_filepath, 0o664)  # nosec B103
    temp_filepath.replace(runoff_filepath)

    logger.info(f"Fixed time precision in {runoff_filepath}")
    return runoff_filepath
