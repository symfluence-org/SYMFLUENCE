# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Canonical forcing reader for the model-ready store.

Single entry point every model adapter should use to load basin-averaged
forcing. It guarantees the canonical CFIF vocabulary — CF standard names with
underscores (precipitation_flux, air_temperature, surface_downwelling_shortwave_flux,
surface_downwelling_longwave_flux, wind_speed, specific_humidity, surface_air_pressure)
— regardless of the source dataset (CARRA, ERA5, ...) and exposes the forcing
timestep via ``ds.attrs['timestep_seconds']``, so adapters never re-parse raw
variable names, units, or guess the timestep. Model-native layers (e.g. SUMMA's
Fortran binary) translate these to their shorthand via cfif.CFIF_TO_SUMMA_MAPPING.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import xarray as xr

from symfluence.core.modeling.forcing_naming import (
    discretization_key_from_name,
)
from symfluence.core.modeling.forcing_naming import (
    discretization_token as discretization_token,
)
from symfluence.core.modeling.forcing_naming import (
    forcing_name_matches_discretization as forcing_name_matches_discretization,
)
from symfluence.core.modeling.forcing_naming import (
    select_forcing_files as select_forcing_files,
)

from .cf_conventions import CANONICAL_FORCING

if TYPE_CHECKING:
    from symfluence.core.modeling.forcing_naming import HruGeometry

logger = logging.getLogger(__name__)
_discretization_key_from_name = discretization_key_from_name

# Deltas within this tolerance (seconds) of each other count as one cadence.
_CADENCE_TOLERANCE_S = 1.0


def _has_datetime_axis(ds: xr.Dataset) -> bool:
    """Report whether *ds* has a datetime64 time axis long enough to measure."""
    return (
        'time' in ds
        and ds['time'].size > 1
        and np.issubdtype(np.asarray(ds['time'].values).dtype, np.datetime64)
    )


def validated_timestep_seconds(times: np.ndarray, context: str = 'forcing') -> float:
    """Infer the cadence (seconds) from a full time axis, refusing irregular ones.

    The whole axis is checked — not just the first pair of timestamps — because a
    single leading delta cannot distinguish an hourly axis from one with a gap
    ([00:00, 01:00, 04:00] is not hourly). An axis that is non-monotonic, has
    duplicate timestamps, or mixes different deltas would silently corrupt
    accumulated precipitation and timestep counts downstream, so it is rejected.

    Args:
        times: datetime64 time coordinate values (size >= 2).
        context: label for error messages (e.g. the source file).

    Returns:
        The constant timestep in seconds.

    Raises:
        ValueError: if the axis is non-monotonic, has duplicates, or is irregular.
    """
    deltas = np.diff(times) / np.timedelta64(1, 's')
    if np.any(deltas <= 0):
        bad = int(np.argmax(deltas <= 0))
        raise ValueError(
            f"Time axis of {context} is not strictly increasing at index {bad + 1} "
            f"({times[bad]} -> {times[bad + 1]}): duplicate or out-of-order timestamps. "
            "Fix or regenerate the forcing store before using it."
        )
    if float(deltas.max() - deltas.min()) > _CADENCE_TOLERANCE_S:
        bad = int(np.argmax(np.abs(deltas - deltas[0]) > _CADENCE_TOLERANCE_S))
        raise ValueError(
            f"Time axis of {context} is irregular: steps range from {deltas.min():.0f} s "
            f"to {deltas.max():.0f} s (first differing interval at {times[bad]} -> "
            f"{times[bad + 1]}). Refusing to treat it as a fixed cadence — gaps would "
            "silently distort accumulated forcing. Regularize the axis or rebuild the store."
        )
    return float(np.median(deltas))


def _dedupe_forcing_time(ds: xr.Dataset, files: List[Path]) -> xr.Dataset:
    """Drop duplicate timestamps left by merging overlapping forcing files.

    Returns ``ds`` unchanged when the time axis is already unique (the common
    case). When duplicates are present — the signature of a store holding more
    than one file for the same period — keep the first occurrence of each
    timestamp and log a warning naming the store, since the duplicate itself
    is a data-hygiene problem worth fixing at the source.
    """
    if 'time' not in ds.dims and 'time' not in ds.coords:
        return ds
    try:
        index = ds.get_index('time')
    except (KeyError, ValueError):
        return ds
    dup_mask = index.duplicated()  # keep='first'
    n_dup = int(dup_mask.sum())
    if n_dup == 0:
        return ds
    logger.warning(
        "Forcing store has %d duplicate timestep(s) across %d files (%s) — "
        "more than one file covers the same period. Keeping the first value "
        "at each timestamp; remove the duplicate/stray forcing file to silence "
        "this. Left unhandled it doubles the series and corrupts the run.",
        n_dup, len(files), ', '.join(f.name for f in files[:4]),
    )
    return ds.isel(time=np.where(~dup_mask)[0])


# Spatial dimensions that identify the discretization of a forcing file.
_SPATIAL_DIMS = ('hru', 'gru', 'subbasin')

def assert_consistent_within_discretization(files: List[Path]) -> None:
    """Write-boundary check: each discretization namespace is internally consistent.

    Unlike :func:`assert_consistent_spatial_dims` (which refuses ANY spatial-size
    mix and guards a single ``open``), a *store* may now hold several
    discretizations of one domain side by side — each under its own
    ``_remapped_{token}_`` namespace. This groups the files by that token and
    checks size-consistency only WITHIN each group, so distinct namespaces
    (lumped ``hru=1`` + elevation ``hru=12``) coexist, while a genuine collision
    inside one namespace — including two legacy untokened files that disagree on
    size (the original #339 signature) — still fails loudly.
    """
    groups: dict = {}
    for f in files:
        groups.setdefault(_discretization_key_from_name(Path(f).name), []).append(Path(f))
    for group in groups.values():
        assert_consistent_spatial_dims(group)


def assert_consistent_spatial_dims(files: List[Path]) -> None:
    """Fail loudly when a multi-file forcing store mixes discretizations.

    A model-ready forcing store must hold ONE discretization per (domain,
    discretization): every file shares the same ``hru`` (and ``gru``) size.
    A cross-discretization filename collision (issue #339) can leave a lumped
    ``hru=1`` forcing and, say, a 12-band elevation ``hru=12`` forcing side by
    side under the shared ``{domain}_{forcing}_remapped_*`` namespace. Merging
    them with ``xr.open_mfdataset(combine='by_coords')`` then raises the cryptic
    ``cannot reindex or align along dimension 'hru' because of conflicting
    dimension sizes: {1, 12}`` — or, on the concat fallback, an even more opaque
    "could not find any dimension coordinates" — with no hint at WHICH file is
    the stray. This pre-flight opens each file's header, groups them by spatial
    size, and raises a clear, actionable error naming every file and its size so
    the offending remap can be quarantined.

    Args:
        files: the forcing NetCDF paths about to be merged (len >= 2).

    Raises:
        ValueError: if the files disagree on any spatial dimension size.
    """
    if len(files) < 2:
        return
    # dim -> {size -> [file names]}
    sizes: dict = {}
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                for dim in _SPATIAL_DIMS:
                    if dim in ds.sizes:
                        sizes.setdefault(dim, {}).setdefault(int(ds.sizes[dim]), []).append(Path(f).name)
        except (OSError, ValueError) as e:
            # A header we cannot read is not a discretization conflict; let the
            # subsequent open surface the real I/O error with full context.
            logger.debug("Could not read spatial dims of %s: %s", Path(f).name, e)
    for dim, by_size in sizes.items():
        if len(by_size) > 1:
            groups = '; '.join(
                f"{dim}={size} in [{', '.join(names)}]"
                for size, names in sorted(by_size.items())
            )
            raise ValueError(
                f"Model-ready forcing store mixes conflicting '{dim}' dimension "
                f"sizes {sorted(by_size)} across files: {groups}. A store must hold "
                "one discretization per domain; this is a cross-discretization "
                "forcing collision (see issue #339). Quarantine the stray remap "
                "(the file whose spatial size does not match this domain's "
                "discretization) from data/model_ready/forcings and its "
                "forcing/basin_averaged_data source, then rebuild the store."
            )


def open_canonical_forcing(
    forcing_files: Union[Path, List[Path]],
    discretization: object = None,
    geometry: Optional['HruGeometry'] = None,
) -> xr.Dataset:
    """Open model-ready forcing and return it under canonical CFIF names.

    - Aliased source variables (e.g. ``pptrate`` -> ``precipitation_flux``) are
      renamed to the canonical CF vocabulary.
    - ``ds.attrs['timestep_seconds']`` is set from the store's declared attribute
      or, failing that, inferred from the time axis.

    Args:
        forcing_files: a single NetCDF path or a list of them.
        discretization: the run's spatial discretization
            (``config.domain.discretization`` / ``SUB_GRID_DISCRETIZATION``).
            When given, the store is filtered to the forcing matching THIS
            discretization (see :func:`select_forcing_files`) before opening, so
            a store holding more than one discretization of the domain does not
            collide (issue #339). ``None`` keeps the legacy behaviour of reading
            every file passed, with :func:`assert_consistent_spatial_dims` as a
            loud backstop against an accidental mix.
        geometry: the run's catchment :class:`HruGeometry`; narrows the
            selection to the forcing remapped onto that HRU set.

    Returns:
        xarray.Dataset with canonical variable names and a timestep_seconds attr.
    """
    files = [forcing_files] if isinstance(forcing_files, (str, Path)) else list(forcing_files)
    files = [Path(f) for f in files]
    files = select_forcing_files(files, discretization, geometry)
    if len(files) == 1:
        ds = xr.open_dataset(files[0])
    else:
        # Pre-flight: refuse to merge a store that mixes discretizations, and
        # name the offending file, instead of letting xarray raise the cryptic
        # "conflicting dimension sizes: {1, 12}" deep inside the merge.
        assert_consistent_spatial_dims(files)
        try:
            ds = xr.open_mfdataset(files, combine='by_coords', join='outer', data_vars='minimal',
                                   coords='minimal', compat='override')
        except (ValueError, OSError):
            # Filename sort order need not be chronological; sort the axis so
            # the regularity validation below sees the true cadence.
            ds = xr.concat([xr.open_dataset(f) for f in sorted(files)], dim='time').sortby('time')
        # A legitimately-chunked store holds NON-overlapping time slices, which
        # merge cleanly. Two files describing the SAME period — a duplicate or
        # stray remap left in the store (e.g. a re-download beside the original
        # under the shared {domain}_{forcing}_remapped_* namespace) — instead
        # produce duplicate timestamps: the concat path doubles the series
        # (140254 = 2x70127 was observed), which then either raises on a
        # non-unique time index or silently feeds a garbled 2x-length forcing
        # to the model. Collapse duplicate timestamps deterministically (keep
        # the first occurrence) and warn loudly, so a contaminated store can
        # never again silently corrupt a calibration.
        ds = _dedupe_forcing_time(ds, files)

    # Resolve each canonical variable, coalescing across every spelling present.
    # A partially-regenerated store can carry more than one spelling of the same
    # variable spread across files (e.g. 'precipitation_flux' in some, 'pptrate' in
    # others); the by-coords merge then leaves each spelling NaN wherever the other
    # supplied the data. Fill the canonical name from its aliases so every timestep
    # gets a value where any source provided one. With a single spelling present
    # (the clean-store case) this reduces to a plain rename.
    for canonical, spec in CANONICAL_FORCING.items():
        # spec['aliases'] already includes the canonical name; dict.fromkeys
        # de-duplicates while preserving order so the canonical stays the primary.
        candidates = dict.fromkeys([canonical, *spec['aliases']])  # type: ignore[misc]
        present = [name for name in candidates if name in ds]
        if not present:
            continue
        primary = present[0]
        if len(present) > 1:
            filled = ds[primary]
            for alt in present[1:]:
                filled = filled.where(~filled.isnull(), ds[alt])
            ds[primary] = filled
            ds = ds.drop_vars(present[1:])
        if primary != canonical:
            ds = ds.rename({primary: canonical})

    # The measured axis is authoritative: validate the whole axis and cross-check
    # any declared attribute against it rather than trusting metadata blindly.
    # Non-datetime64 axes (cftime calendars, undecoded numerics) can't be
    # validated this way and fall back to the declared attribute.
    ts = ds.attrs.get('timestep_seconds')
    if _has_datetime_axis(ds):
        measured = validated_timestep_seconds(
            ds['time'].values, context=', '.join(f.name for f in files[:3])
        )
        if ts is not None and abs(float(ts) - measured) > _CADENCE_TOLERANCE_S:
            logger.warning(
                "Declared timestep_seconds=%s disagrees with the time axis (%.0f s); "
                "using the measured cadence", ts, measured,
            )
        ts = measured
    if ts is not None:
        ds.attrs['timestep_seconds'] = float(ts)
    return ds


def forcing_timestep_seconds(ds: xr.Dataset, default: float = 3600.0) -> float:
    """Return the canonical forcing timestep in seconds (measured or declared).

    The full time axis, when present, is validated and wins over the declared
    ``timestep_seconds`` attribute — a stale or wrong attribute must not let an
    irregular or coarser axis masquerade as the declared cadence.
    """
    ts = ds.attrs.get('timestep_seconds')
    if _has_datetime_axis(ds):
        measured = validated_timestep_seconds(ds['time'].values)
        if ts is not None and abs(float(ts) - measured) > _CADENCE_TOLERANCE_S:
            logger.warning(
                "Declared timestep_seconds=%s disagrees with the time axis (%.0f s); "
                "using the measured cadence", ts, measured,
            )
        return measured
    if ts is not None:
        return float(ts)
    return default


def resample_canonical_forcing(ds: xr.Dataset, target_seconds: float) -> xr.Dataset:
    """Resample a CF-named forcing dataset to a fixed target timestep.

    Some model pipelines assume a fixed forcing cadence (e.g. hourly) and equate
    "one timestep" with "one hour" in their downstream aggregation. Feeding such
    a pipeline a coarser source (CARRA 3-hourly, daily) silently miscounts the
    cadence. This resamples the source to ``target_seconds`` so the assumption
    holds for any input:

    - **Rate** variables (``kind == 'rate'``, e.g. ``precipitation_flux``) are
      held constant across each source interval — a step function (``ffill`` on
      upsample, interval ``mean`` on downsample) that conserves the accumulated
      total.
    - **State** variables (``kind == 'state'``, e.g. ``air_temperature``) are
      linearly interpolated on upsample, interval-averaged on downsample.

    It is a no-op when the source already matches ``target_seconds`` — so an
    hourly source (ERA5) is returned byte-for-byte unchanged.

    Args:
        ds: forcing Dataset under canonical CF names (from open_canonical_forcing).
        target_seconds: desired timestep in seconds (e.g. 3600 for hourly).

    Returns:
        Resampled Dataset with ``target_seconds`` on ``ds.attrs['timestep_seconds']``.
    """
    if 'time' not in ds or ds['time'].size < 2:
        return ds

    src_seconds = forcing_timestep_seconds(ds)
    if abs(src_seconds - target_seconds) < 1.0:
        return ds  # no-op — source already at the target cadence

    import pandas as pd

    src_times = pd.DatetimeIndex(ds['time'].values)
    freq = f'{int(target_seconds)}s'

    def _is_rate(var: str) -> bool:
        spec = CANONICAL_FORCING.get(var)
        return bool(spec) and spec.get('kind') == 'rate'

    if target_seconds < src_seconds:
        # Upsample to a finer cadence. Extend the target to cover the final
        # source interval fully so step-filled rates conserve their total.
        end = (src_times[-1]
               + pd.Timedelta(seconds=src_seconds)
               - pd.Timedelta(seconds=target_seconds))
        target_times = pd.date_range(start=src_times[0], end=end, freq=freq)
        if len(target_times) == 0:
            return ds
        rate_vars = [v for v in ds.data_vars if _is_rate(str(v))]
        state_vars = [v for v in ds.data_vars if not _is_rate(str(v))]
        parts = []
        if rate_vars:
            # Step function: each source rate spans its whole interval.
            parts.append(ds[rate_vars].reindex(time=target_times, method='ffill'))
        if state_vars:
            # Linear interpolation in the interior; hold the last value across
            # the extrapolated tail (interp yields NaN beyond the source range).
            interp = ds[state_vars].interp(time=target_times).ffill('time')
            parts.append(interp)
        out = xr.merge(parts) if parts else ds.reindex(time=target_times, method='ffill')
    else:
        # Downsample to a coarser cadence: mean over each interval works for both
        # a mean rate and an averaged state.
        out = ds.resample(time=freq).mean()

    out.attrs.update(ds.attrs)
    out.attrs['timestep_seconds'] = float(target_seconds)
    return out
