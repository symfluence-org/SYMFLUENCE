# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcing data processing utilities for HYPE model.

Handles merging of forcing data from multiple NetCDF files and conversion
to HYPE-compatible daily observation formats.
"""

# Standard library imports
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import cdo
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from symfluence.core.exceptions import ValidationError
from symfluence.core.modeling.utilities import BaseForcingProcessor


class HYPEForcingProcessor(BaseForcingProcessor):
    """
    Processor for HYPE forcing data.

    Handles:
    - Merging hourly NetCDF forcing files
    - Rolling time for time zone offsets
    - Resampling hourly data to daily HYPE format (Pobs, Tobs, TMAXobs, TMINobs)
    - Unit conversions and HYPE-specific file formatting
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        forcing_input_dir: Path,
        output_path: Path,
        cache_path: Path,
        timeshift: int = 0,
        forcing_units: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HYPE forcing processor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            forcing_input_dir: Path to input basin-averaged NetCDF files
            output_path: Path to output HYPE settings directory
            cache_path: Path for temporary processing files
            timeshift: Hour offset for time zone correction
            forcing_units: Mapping of variables to units and names
        """
        super().__init__(
            config=config,
            logger=logger,
            input_path=forcing_input_dir,
            output_path=output_path,
            cache_path=cache_path
        )
        # Keep forcing_input_dir as alias for backward compatibility
        self.forcing_input_dir = self.input_path
        self.timeshift = timeshift
        self.forcing_units = forcing_units or {}
        # Elevation-band forcing expansion (set by the preprocessor when
        # SUB_GRID_DISCRETIZATION=elevation). When present, the single
        # basin-averaged obs column is expanded into one column per band with a
        # temperature lapse correction; precipitation is replicated.
        self._elevation_bands: Optional[list] = None
        self._lapse_rate: float = 0.0065

    def set_elevation_bands(self, bands: list, lapse_rate: float = 0.0065) -> None:
        """Enable per-band forcing expansion.

        Args:
            bands: ordered band table — list of dicts with 'hru_id' and
                'elev_mean' (and optionally 'area' for the reference elevation).
            lapse_rate: temperature lapse rate in K/m (default standard ELR).
        """
        self._elevation_bands = bands or None
        self._lapse_rate = lapse_rate

    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "HYPE"

    def process_forcing(self) -> None:
        """Execute the full HYPE forcing processing workflow."""
        self.logger.info("Merging HYPE forcing files...")
        merged_forcing_path = self._merge_forcing_files()

        if not merged_forcing_path or not merged_forcing_path.exists():
            self.logger.error("Forcing merge failed, cannot proceed with daily conversion")
            return

        self.logger.info("Converting hourly forcing to HYPE daily observations...")
        self._convert_to_daily_obs(merged_forcing_path)

        # Cleanup
        if merged_forcing_path.exists():
            merged_forcing_path.unlink()

    def _select_forcing_for_banding(self, files: List[Path]) -> List[Path]:
        """Scope the glob to the series HYPE's band expansion actually needs.

        With elevation banding, HYPE consumes the SINGLE-column basin-averaged
        series and lapse-expands it into one column per band itself --
        ``_convert_to_daily_obs`` only expands when ``df.shape[1] == 1``. The
        pre-remapped banded file is therefore the wrong input: it is already
        multi-column, so the expansion silently does not run and the model gets
        bands with no lapse correction.

        Before #339 this could not arise: the lumped and banded remaps shared a
        filename, so the store only ever held one. 5279dd49 namespaced them and
        noted the store "must hold one discretization at a time until the reader
        wiring passes discretization on every model's forcing-read path
        (follow-up)"; d70c4c53 wired MESH, this path was missed. The glob then
        takes both and CDO aborts with "Grid size of the input field 'latitude'
        do not match!" (identical on CDO 2.4.4 and 2.6.3).

        Measured on experiment 02 (Bow, ERA5): selecting the single-column
        series scores KGE 0.9189 against the 0.9001 reference -- inside the
        0.02 cross-platform bound. Feeding the banded file instead scores
        0.8338, outside it.

        A no-op without banding, or when the store holds one file, so
        single-discretization and pre-namespacing layouts do not regress.
        """
        if not self._elevation_bands or len(files) < 2:
            return files

        single = []
        for path in files:
            try:
                with xr.open_dataset(path) as ds:
                    if ds.sizes.get('hru', 1) == 1:
                        single.append(path)
            except (OSError, ValueError):
                continue

        if not single:
            # Nothing single-column to expand from: leave the glob alone and let
            # the merge report whatever is actually wrong, rather than silently
            # narrowing to an arbitrary subset.
            self.logger.warning(
                "Elevation banding is configured but no single-column forcing was "
                "found in %s; the lapse expansion cannot run and bands will be "
                "used as-is.", self.forcing_input_dir
            )
            return files

        if len(single) != len(files):
            self.logger.info(
                "Elevation banding active: using the single-column basin-averaged "
                "forcing %s (of %d file(s)) so the lapse expansion applies.",
                [p.name for p in single], len(files)
            )
        return single

    def _merge_forcing_files(self) -> Optional[Path]:
        """Merge individual NetCDF files using CDO with xarray fallback."""
        easymore_nc_files = sorted(list(self.forcing_input_dir.glob('*.nc')))
        if not easymore_nc_files:
            self.logger.warning(f"No forcing files found in {self.forcing_input_dir}")
            return None
        easymore_nc_files = self._select_forcing_for_banding(easymore_nc_files)

        merged_forcing_path = self.cache_path / 'merged_forcing.nc'

        # Try CDO first (faster for large datasets)
        try:
            cdo_obj = cdo.Cdo()
            # If initialization succeeded, try merging
            self.logger.info("Merging forcing files with CDO...")

            # split the files in batches as cdo cannot mergetime long list of file names
            batch_size = 20
            if len(easymore_nc_files) < batch_size:
                batch_size = len(easymore_nc_files)

            files_split: List[Any] = np.array_split(easymore_nc_files, batch_size)
            intermediate_files = []

            for i in tqdm(range(batch_size), desc="Merging forcing batches"):
                batch_files = [str(f) for f in files_split[i].tolist()]
                batch_output = self.cache_path / f"forcing_batch_{i}.nc"
                cdo_obj.mergetime(input=batch_files, output=str(batch_output))
                intermediate_files.append(batch_output)

            # Combine intermediate results
            cdo_obj.mergetime(input=[str(f) for f in intermediate_files], output=str(merged_forcing_path))

            # Clean up intermediate files
            for f in intermediate_files:
                if f.exists():
                    f.unlink()

            self.logger.info("CDO merge successful")

        except (AttributeError, OSError, ValueError) as e:
            self.logger.warning(f"CDO merge failed or CDO not available: {e}. Falling back to xarray...")
            try:
                # Portable, CDO-free fallback (required on native Windows, where CDO
                # has no conda-forge win-64 build). See _merge_forcing_files_xarray
                # for why this must be eager rather than a lazy open_mfdataset.
                if self._merge_forcing_files_xarray(easymore_nc_files, merged_forcing_path) is None:
                    return None
                self.logger.info("Xarray merge successful")
            except Exception as xe:  # noqa: BLE001 — model execution resilience
                self.logger.error(f"Xarray merge also failed: {xe}", exc_info=True)
                return None

        # Handle time shift and calendar
        if not merged_forcing_path.exists():
            return None

        with xr.open_dataset(merged_forcing_path, engine='h5netcdf') as forcing:
            forcing = forcing.convert_calendar('standard')
            if self.timeshift != 0:
                forcing['time'] = forcing['time'] + pd.Timedelta(hours=self.timeshift)

            tmp_path = merged_forcing_path.with_suffix('.nc.tmp')
            forcing.to_netcdf(tmp_path, engine='h5netcdf')

        os.replace(tmp_path, merged_forcing_path)
        return merged_forcing_path

    def _merge_forcing_files_xarray(
        self, files: List[Path], out_path: Path
    ) -> Optional[Path]:
        """Merge forcing NetCDFs with pure xarray/numpy (no CDO), eagerly.

        This is the fallback used when the CDO binary is unavailable (notably on
        native Windows, where CDO has no conda-forge win-64 build). It must
        reproduce what ``cdo mergetime`` produces: all timesteps of all input
        files, sorted by date, with duplicate/overlapping timesteps collapsed to
        their first occurrence.

        Why not ``xr.open_mfdataset(combine='nested', concat_dim='time').sortby('time')``:
        that path is lazy (dask-backed). When two input files share the same
        timestamps — which happens for the basin-averaged (hru=1) remapped store,
        where sibling files (e.g. an ``_grus_`` variant) cover the identical
        2002-2009 range — ``sortby`` computes a *fully interleaved* permutation
        (0, N, 1, N+1, 2, N+2, ...). Applying that shuffled index to chunked dask
        arrays produces a near-quadratic task graph (every output chunk depends on
        every input chunk), which effectively never completes: on WSL this hung for
        2+ hours. Installing CDO merely routed around the fallback; it did not fix
        it.

        The fix is to stay eager: load each (small, basin-averaged) file into
        memory as plain NumPy, concatenate, then deduplicate/sort with a single
        ``np.unique`` on the time values. ``np.unique(..., return_index=True)``
        returns the unique times already in ascending order together with the
        index of each value's first occurrence, so a single ``isel`` yields a
        time-sorted, first-occurrence-wins dataset — identical to CDO mergetime —
        with a trivial, dask-free computation that finishes in well under a second.
        """
        datasets: List[xr.Dataset] = []
        try:
            for f in files:
                # .load() forces eager reads so concat/sort operate on NumPy
                # arrays — no lazy dask graph, no interleaved-index blowup.
                datasets.append(xr.open_dataset(f, engine='h5netcdf').load())

            if len(datasets) == 1:
                combined = datasets[0]
            else:
                combined = xr.concat(
                    datasets,
                    dim='time',
                    data_vars='minimal',
                    coords='minimal',
                    compat='override',
                    join='override',
                )

            # Sort by time and drop duplicate/overlapping timesteps, keeping the
            # first occurrence (matches `cdo mergetime`). np.unique yields ascending
            # unique times plus first-occurrence indices in one pass.
            time_values = combined['time'].values
            _, first_occurrence = np.unique(time_values, return_index=True)
            if len(first_occurrence) != len(time_values):
                self.logger.info(
                    "Dropped %d duplicate/overlapping timesteps during xarray merge",
                    len(time_values) - len(first_occurrence),
                )
            combined = combined.isel(time=first_occurrence)

            combined.to_netcdf(out_path, engine='h5netcdf')
        finally:
            for ds in datasets:
                ds.close()

        return out_path if out_path.exists() else None

    def _convert_to_daily_obs(self, merged_forcing_path: Path) -> None:
        """Convert hourly merged data to HYPE daily observation files."""
        def get_in_var(key):
            return self.forcing_units[key]['in_varname']

        # Get temperature units for conversion (HYPE expects Celsius)
        temp_units = self.forcing_units.get('temperature', {}).get('in_units', 'K')

        # TMAX
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMAXobs',
            stat='max',
            output_file_name_txt=self.output_path / 'TMAXobs.txt',
            unit_conversion=temp_units  # Convert K to C if needed
        )

        # TMIN
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMINobs',
            stat='min',
            output_file_name_txt=self.output_path / 'TMINobs.txt',
            unit_conversion=temp_units  # Convert K to C if needed
        )

        # Tobs (Mean)
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'Tobs',
            stat='mean',
            output_file_name_txt=self.output_path / 'Tobs.txt',
            unit_conversion=temp_units  # Convert K to C if needed
        )

        # Pobs (Sum)
        # Get precipitation units for conversion
        precip_units = self.forcing_units.get('precipitation', {}).get('in_units', 'mm/s')
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('precipitation'),
            'Pobs',
            stat='sum',
            output_file_name_txt=self.output_path / 'Pobs.txt',
            unit_conversion=precip_units  # Pass units for conversion
        )

        if self.config.get('HYPE_SNOW_MELT_MODEL', 0) == 2:
            self.write_shortwave_obs([merged_forcing_path])

    def write_shortwave_obs(self, files: List[Path]) -> Path:
        """Integrate complete, regular flux records to HYPE MJ/m2/day.

        Reads only radiation, avoiding a second full forcing merge. No radiation
        is synthesized when coverage, units or subbasin IDs are ambiguous.
        """
        chunks = []
        for path in files:
            with xr.open_dataset(path) as ds:
                variable = 'surface_downwelling_shortwave_flux'
                if variable not in ds:
                    raise ValidationError(f'Missing {variable} in {path}')
                da = ds[variable].load()
                units = str(da.attrs.get('units', '')).lower().replace(' ', '')
                if units not in ('wm-2', 'w/m2', 'wm^-2', 'w/m^2', 'wm**-2'):
                    raise ValidationError(f'Expected shortwave flux in W/m2, got {units!r}')
                frame = da.to_pandas()
                if isinstance(frame, pd.Series):
                    frame = frame.to_frame(name=1)
                if not isinstance(frame.index, pd.DatetimeIndex):
                    frame = frame.T
                if 'hruId' in ds:
                    ids = ds['hruId']
                    if 'time' in ids.dims:
                        ids = ids.isel(time=0)
                    frame.columns = ids.values.astype(int).ravel()
                frame.columns = frame.columns.astype(int)
                if 0 in frame.columns:
                    frame.columns = frame.columns + 1
                chunks.append(frame)
        if not chunks:
            raise ValidationError('No shortwave forcing files')
        frame = pd.concat(chunks).sort_index()
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValidationError('Duplicate shortwave timestamps or IDs')
        dt = np.diff(frame.index.asi8) / 1e9
        if not len(dt) or dt[0] <= 0 or not np.all(dt == dt[0]) or 86400 % dt[0] != 0:
            raise ValidationError('Shortwave forcing must have a regular timestep dividing one day')
        if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
            raise ValidationError('Missing or negative shortwave forcing')
        count = frame.resample('D').size()
        if not (count == int(86400/dt[0])).all():
            raise ValidationError('Incomplete shortwave forcing day')
        daily = (frame * dt[0] / 1e6).resample('D').sum()
        if self._elevation_bands and len(daily.columns) == 1:
            daily = self._expand_columns_to_bands(daily, 'SWobs')
        daily.index.name = 'time'
        daily.columns.name = None
        target = self.output_path / 'SWobs.txt'
        daily.to_csv(target, sep='\t', float_format='%.6f')
        return target

    def _convert_hourly_to_daily(
        self,
        input_file_name: Path,
        variable_in: str,
        variable_out: str,
        var_time: str = 'time',
        var_id: str = 'hruId',
        stat: str = 'max',
        output_file_name_txt: Optional[Path] = None,
        unit_conversion: Optional[str] = None
    ) -> xr.Dataset:
        """Helper to resample hourly NetCDF to daily text file.

        Args:
            input_file_name: Path to merged forcing NetCDF
            variable_in: Input variable name
            variable_out: Output variable name (for logging)
            var_time: Time dimension name
            var_id: HRU/subbasin ID variable name
            stat: Aggregation statistic ('max', 'min', 'mean', 'sum')
            output_file_name_txt: Output text file path
            unit_conversion: Input units for conversion. If 'mm/s' or 'kg/m²/s' or 'kg m-2 s-1',
                applies conversion factor of 3600 (seconds per hour) for hourly data.
        """
        with xr.open_dataset(input_file_name, engine='h5netcdf') as ds:
            ds = ds.copy()

            # Apply unit conversion
            if unit_conversion:
                unit_lower = unit_conversion.lower()

                # Precipitation: convert from rate (per second) to amount per timestep
                if unit_lower in ['mm/s', 'mm s-1', 'kg/m²/s', 'kg m-2 s-1', 'kg/m2/s']:
                    # Detect actual timestep from data instead of assuming hourly
                    time_diff = ds[var_time].diff(dim=var_time)
                    # Convert to seconds (time_diff is in nanoseconds for datetime64)
                    median_step_ns = float(time_diff.median())
                    if np.isnan(median_step_ns) or median_step_ns <= 0:
                        median_step_seconds = 3600.0  # Default to hourly if detection fails
                        self.logger.warning("Could not detect timestep, assuming hourly (3600s)")
                    else:
                        median_step_seconds = median_step_ns * 1e-9  # Convert ns to seconds

                    # Multiply by timestep seconds to convert rate to amount per timestep
                    self.logger.info(
                        f"Converting {variable_in} from {unit_conversion} to mm/timestep "
                        f"(multiplying by {median_step_seconds:.0f}s)"
                    )
                    ds[variable_in] = ds[variable_in] * median_step_seconds

                # Temperature: convert from Kelvin to Celsius
                elif unit_lower in ['k', 'kelvin']:
                    self.logger.info(f"Converting {variable_in} from Kelvin to Celsius (subtracting 273.15)")
                    ds[variable_in] = ds[variable_in] - 273.15

            # Get the mapping from hru dimension index to actual hruId values
            # This is needed because hruId is often a data variable, not a coordinate
            hru_id_mapping = None
            if var_id in ds.data_vars and var_id not in ds.coords:
                # hruId is a data variable - get the mapping from hru index to actual IDs
                hru_id_da = ds[var_id]
                # Find the dimension name for hruId (typically 'hru')
                hru_dim = hru_id_da.dims[0] if hru_id_da.dims else None
                if hru_dim:
                    # Handle multi-dimensional case (e.g., if hruId has time dimension)
                    # Take the first time slice if multiple dimensions exist
                    if hru_id_da.ndim > 1:
                        # Select first index along all dimensions except the hru dimension
                        sel_dict = {str(dim): 0 for dim in hru_id_da.dims if dim != hru_dim}
                        hru_id_values = hru_id_da.isel(**sel_dict).values.flatten()
                    else:
                        hru_id_values = hru_id_da.values.flatten()
                    # Convert to integer and create mapping
                    hru_id_values = hru_id_values.astype(int)
                    hru_id_mapping = {i: int(hru_id_values[i]) for i in range(len(hru_id_values))}
            elif var_id in ds.coords:
                # hruId is already a coordinate - cast to int
                ds.coords[var_id] = ds.coords[var_id].astype(int)

            # Ensure time index is sorted
            ds = ds.sortby('time')

            # Resample to daily
            if stat == 'max':
                ds_daily = ds.resample(time='D').max()
            elif stat == 'min':
                ds_daily = ds.resample(time='D').min()
            elif stat == 'mean':
                ds_daily = ds.resample(time='D').mean()
            elif stat == 'sum':
                ds_daily = ds.resample(time='D').sum()
            else:
                raise ValueError(f"Unsupported stat: {stat}")

            # Extract variable and convert to dataframe
            # Use to_series().unstack() to get time as index and IDs as columns
            series = ds_daily[variable_in].to_series()

            # Dynamically determine the ID level name
            actual_id_level = var_id
            if var_id not in series.index.names:
                for fallback in ['id', 'hru', 'subid']:
                    if fallback in series.index.names:
                        actual_id_level = fallback
                        break

            if actual_id_level in series.index.names:
                df = series.unstack(level=actual_id_level)
            else:
                # Lumped domain: the forcing has no spatial/ID dimension to unstack.
                # Usually the series is indexed by time alone, but some model-ready
                # stores carry a singleton spatial dimension whose level name isn't
                # one we unstack on (not time/id/hru/subid). That leaves a MultiIndex
                # whose entries are tuples, which pd.to_datetime later rejects with
                # "<class 'tuple'> is not convertible to datetime". Collapse any such
                # extra levels onto the time axis before framing.
                if isinstance(series.index, pd.MultiIndex):
                    for level_name in [n for n in series.index.names if n != 'time']:
                        n_unique = series.index.get_level_values(level_name).nunique()
                        if n_unique > 1:
                            self.logger.warning(
                                f"Lumped HYPE forcing has non-singleton dimension "
                                f"'{level_name}' ({n_unique} values); using its first slice."
                            )
                            first = series.index.get_level_values(level_name)[0]
                            series = series.xs(first, level=level_name)
                        else:
                            series = series.droplevel(level_name)
                # Emit a single subbasin column (id 0; the +1 shift below promotes
                # it to 1, since HYPE requires subid > 0).
                df = series.to_frame(name=0)

            # Map column indices to actual hruId values if we have the mapping
            if hru_id_mapping is not None:
                # Columns are currently hru dimension indices (0, 1, 2, ...)
                # Map them to actual hruId values
                df.columns = [hru_id_mapping.get(int(c), int(c)) for c in df.columns]
            else:
                # Ensure columns (subids) are integers
                df.columns = df.columns.astype(int)

            # Shift 0-based IDs by +1 to match GeoData (HYPE requires subid > 0)
            column_ids = [int(c) for c in df.columns]
            if 0 in column_ids or min(column_ids) == 0:
                column_ids = [c + 1 for c in column_ids]
            df.columns = column_ids

            df.columns.name = None
            df.index.name = 'time'

            # Expand the single basin-averaged column into per-elevation-band
            # columns (lapse-corrected temperature; replicated precipitation) so
            # the obs files match the banded GeoData sub-basins.
            if self._elevation_bands and df.shape[1] == 1:
                df = self._expand_columns_to_bands(df, variable_out)
            elif self._elevation_bands:
                # Banding was requested but the input is already multi-column, so
                # the lapse correction cannot be applied. Say so: silently running
                # unlapsed bands produces a plausible-looking but wrong result
                # (measured KGE 0.8338 vs 0.9189 correctly expanded), which is
                # far harder to notice than a failure.
                self.logger.warning(
                    "Elevation banding is configured (%d bands, lapse=%.4f K/m) but "
                    "%s forcing has %d columns, not 1 — the lapse expansion will NOT "
                    "be applied and the model will use these columns unchanged.",
                    len(self._elevation_bands), self._lapse_rate,
                    variable_out, df.shape[1]
                )

            # Ensure time index is formatted as YYYY-MM-DD for HYPE
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

            if output_file_name_txt:
                # HYPE observation files: header is 'time' then subids
                # Separated by tabs
                df.to_csv(output_file_name_txt, sep='\t', na_rep='-9999.0', index=True, float_format='%.3f')

            return ds_daily

    def _expand_columns_to_bands(
        self, df: pd.DataFrame, variable_out: str
    ) -> pd.DataFrame:
        """Expand a single-column daily obs frame into one column per band.

        Temperature variables (Tobs/TMAXobs/TMINobs) are lapse-corrected to each
        band's mean elevation relative to the area-weighted reference elevation
        (warmer below the reference, colder above). Precipitation (Pobs) is
        replicated unchanged. Columns are keyed by band ``hru_id`` so they match
        the banded GeoData sub-basin ids and ForcKey.
        """
        bands = self._elevation_bands or []
        if len(bands) < 2:
            return df

        elevs = np.array([b['elev_mean'] for b in bands], dtype=float)
        areas = np.array([float(b.get('area', 1.0)) for b in bands], dtype=float)
        wsum = areas.sum() or float(len(bands))
        ref_elev = float((elevs * areas).sum() / wsum)

        base = df.iloc[:, 0].to_numpy(dtype=float)
        is_temperature = variable_out.upper().startswith('T')

        expanded: dict[int, np.ndarray] = {}
        for band in bands:
            col_id = int(band['hru_id'])
            if is_temperature:
                delta = self._lapse_rate * (ref_elev - float(band['elev_mean']))
                expanded[col_id] = base + delta
            else:
                expanded[col_id] = base
        out = pd.DataFrame(expanded, index=df.index)
        if is_temperature:
            deltas = [f"{self._lapse_rate * (ref_elev - e):+.1f}" for e in elevs]
            self.logger.info(
                "HYPE %s: expanded to %d bands (ref_elev=%.0fm, deltas[C]=%s)",
                variable_out, len(bands), ref_elev, ",".join(deltas))
        return out
