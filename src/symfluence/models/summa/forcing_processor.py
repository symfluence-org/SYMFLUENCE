# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcing data processing utilities for SUMMA model.

This module contains the SummaForcingProcessor class which handles all forcing data
processing operations including lapse rate corrections, time coordinate fixes,
NaN value handling, and data validation for SUMMA model compatibility.
"""

# Standard library imports
import gc
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# Third-party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import xarray as xr

from symfluence.core.constants import PhysicalConstants

from ..utilities import BaseForcingProcessor


class SummaForcingProcessor(BaseForcingProcessor):
    """
    Processor for SUMMA forcing data with comprehensive quality control and corrections.

    This class handles:
    - Temperature lapse rate corrections
    - Time coordinate standardization for SUMMA compatibility
    - NaN value interpolation and filling
    - Data range validation and clipping
    - Batch processing for memory efficiency
    - Forcing file list generation
    - HRU ID filtering

    Attributes:
        config: Configuration dictionary containing processing parameters
        logger: Logger instance for recording operations
        forcing_basin_path: Path to basin-averaged forcing data
        forcing_summa_path: Path to output SUMMA-compatible forcing data
        intersect_path: Path to catchment intersection shapefiles
        catchment_path: Path to catchment shapefiles
        project_dir: Root project directory
        setup_dir: Path to SUMMA setup/settings directory
        domain_name: Name of the domain being processed
        forcing_dataset: Name of the forcing dataset (e.g., 'era5', 'rdrs')
        data_step: Time step size in seconds
        gruId: Name of GRU ID field in configuration
        hruId: Name of HRU ID field in configuration
        catchment_name: Name of the catchment shapefile
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        forcing_basin_path: Path,
        forcing_summa_path: Path,
        intersect_path: Path,
        catchment_path: Path,
        project_dir: Path,
        setup_dir: Path,
        domain_name: str,
        forcing_dataset: str,
        data_step: int,
        gruId: str,
        hruId: str,
        catchment_name: str
    ):
        """
        Initialize the SUMMA forcing processor.

        Args:
            config: Configuration dictionary with processing parameters
            logger: Logger instance
            forcing_basin_path: Path to input basin-averaged forcing data
            forcing_summa_path: Path to output SUMMA-compatible forcing data
            intersect_path: Path to catchment intersection shapefiles
            catchment_path: Path to catchment shapefiles
            project_dir: Root project directory
            setup_dir: SUMMA settings directory
            domain_name: Domain name for file naming
            forcing_dataset: Forcing dataset identifier
            data_step: Time step size in seconds
            gruId: GRU ID field name
            hruId: HRU ID field name
            catchment_name: Catchment shapefile filename
        """
        super().__init__(
            config=config,
            logger=logger,
            input_path=forcing_basin_path,
            output_path=forcing_summa_path,
            intersect_path=intersect_path,
            catchment_path=catchment_path,
            project_dir=project_dir,
            setup_dir=setup_dir
        )
        # Keep original attribute names for backward compatibility
        self.forcing_basin_path = self.input_path
        self.forcing_summa_path = self.output_path
        self.domain_name = domain_name
        self.forcing_dataset = forcing_dataset
        self.data_step = data_step
        self.gruId = gruId
        self.hruId = hruId
        self.catchment_name = catchment_name
        self._source_calendar = 'standard'

    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "SUMMA"

    def apply_datastep_and_lapse_rate(self):
        """Apply temperature lapse rate corrections to forcing data.

        Orchestrates file discovery, topology loading, lapse-rate
        pre-calculation, and batch processing of forcing files.

        If the model-agnostic :class:`ElevationCorrectionProcessor` has
        already corrected the basin-averaged files (indicated by the
        ``elevation_corrected`` NetCDF attribute), the expensive topology
        loading and lapse pre-calculation are skipped.  Files are still
        processed through ``_process_forcing_batches`` for SUMMA-specific
        fixes (time coordinate format, NaN interpolation, data validation).
        """
        self.logger.info("Starting memory-efficient temperature lapse rate and data step application")

        # Check if model-agnostic elevation correction already applied
        already_corrected = self._check_already_corrected()

        if already_corrected:
            self.logger.info(
                "Basin-averaged files already elevation-corrected — "
                "skipping topology loading and lapse pre-calculation"
            )
            forcing_files = self._get_forcing_files()
            self._prepare_forcing_output_dir()
            # Pass zero lapse values so SUMMA-specific processing still runs
            lapse_values = pd.DataFrame({'lapse_values': []})
            lapse_values.index.name = f'S_1_{self.hruId}'
            self._process_forcing_batches(forcing_files, lapse_values, 0.0)
        else:
            intersect_csv = self._find_intersection_file()
            topo_data = self._load_topology_data(intersect_csv)
            forcing_files = self._get_forcing_files()
            self._prepare_forcing_output_dir()
            lapse_values, lapse_rate = self._precalculate_lapse_corrections(topo_data)
            del topo_data
            gc.collect()
            self._process_forcing_batches(forcing_files, lapse_values, lapse_rate)

        del lapse_values
        gc.collect()

        if self._source_calendar in ('noleap', '365_day'):
            self._insert_noleap_leap_days()

        self.logger.info(
            f"Completed processing of {len(forcing_files)} "
            f"{self.forcing_dataset.upper()} forcing files with temperature lapsing"
        )

    # ------------------------------------------------------------------
    # Helpers for apply_datastep_and_lapse_rate
    # ------------------------------------------------------------------

    def _check_already_corrected(self) -> bool:
        """Check whether basin-averaged files carry the ``elevation_corrected`` attribute."""
        nc_files = sorted(self.forcing_basin_path.glob("*.nc"))
        if not nc_files:
            return False
        try:
            with xr.open_dataset(nc_files[0]) as ds:
                return bool(ds.attrs.get('elevation_corrected', False))
        except Exception:  # noqa: BLE001
            return False

    def _find_intersection_file(self):
        """Locate the intersection CSV (or shapefile) for topology weighting."""
        intersect_base = (
            f"{self.domain_name}_"
            f"{self._get_config_value(lambda: self.config.forcing.dataset)}"
            f"_intersected_shapefile"
        )
        intersect_csv = self.intersect_path / f"{intersect_base}.csv"
        intersect_shp = self.intersect_path / f"{intersect_base}.shp"

        # Fallback for legacy naming in data bundle
        if not intersect_csv.exists() and not intersect_shp.exists() and self.domain_name == 'bow_banff_minimal':
            legacy_base = (
                f"Bow_at_Banff_lumped_"
                f"{self._get_config_value(lambda: self.config.forcing.dataset)}"
                f"_intersected_shapefile"
            )
            if (self.intersect_path / f"{legacy_base}.csv").exists():
                intersect_csv = self.intersect_path / f"{legacy_base}.csv"
                self.logger.info(f"Using legacy intersection CSV: {intersect_csv.name}")
            elif (self.intersect_path / f"{legacy_base}.shp").exists():
                intersect_shp = self.intersect_path / f"{legacy_base}.shp"
                self.logger.info(f"Using legacy intersection SHP: {intersect_shp.name}")

        # Convert shapefile → CSV if needed
        if not intersect_csv.exists() and intersect_shp.exists():
            self.logger.info(f"Converting {intersect_shp} to CSV format")
            try:
                shp_df = gpd.read_file(intersect_shp)
                shp_df['weight'] = shp_df['AP1']
                shp_df.to_csv(intersect_csv, index=False)
                self.logger.info(f"Successfully created {intersect_csv}")
                del shp_df
                gc.collect()
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"Failed to convert shapefile to CSV: {str(e)}")
                raise
        elif not intersect_csv.exists() and not intersect_shp.exists():
            hru_id_field = self._get_config_value(lambda: self.config.domain.catchment_shp_hruid)
            case_name = f"{self.domain_name}_{self._get_config_value(lambda: self.config.forcing.dataset)}"
            remap_file = self.intersect_path / f"{case_name}_{hru_id_field}_remapping.csv"
            if remap_file.exists():
                self.logger.info(f"Intersected shapefile missing, falling back to remapping weights: {remap_file.name}")
                intersect_csv = remap_file
            else:
                self.logger.error(f"Missing both intersected shapefile and remapping weights in {self.intersect_path}")
                self.logger.error(f"Expected intersect base: {intersect_base}")
                self.logger.error(f"Expected remap file: {remap_file.name}")
                raise FileNotFoundError(f"Neither {intersect_csv} nor {intersect_shp} exist")

        return intersect_csv

    def _load_topology_data(self, intersect_csv):
        """Load intersection CSV with truncated-column handling."""
        self.logger.info("Loading topology data...")
        try:
            sample_df = pd.read_csv(intersect_csv, nrows=0)
            dtype_dict = {}

            # Handle GRU ID (may be truncated to 10 chars by shapefile)
            gru_col = f'S_1_{self.gruId}'
            if gru_col not in sample_df.columns:
                gru_col_truncated = gru_col[:10]
                if gru_col_truncated in sample_df.columns:
                    dtype_dict[gru_col_truncated] = 'int32'
                else:
                    self.logger.warning(f"Column {gru_col} not found in CSV, will try to load without dtype")
            else:
                dtype_dict[gru_col] = 'int32'

            # Handle HRU ID (may be truncated)
            hru_col = f'S_1_{self.hruId}'
            if hru_col not in sample_df.columns:
                hru_col_truncated = hru_col[:10]
                if hru_col_truncated in sample_df.columns:
                    dtype_dict[hru_col_truncated] = 'int32'
                    self.logger.info(f"Using truncated column name: {hru_col_truncated} (original: {hru_col})")
                else:
                    self.logger.warning(f"Column {hru_col} not found in CSV, will try to load without dtype")
            else:
                dtype_dict[hru_col] = 'int32'

            dtype_dict.update({
                'S_2_ID': 'Int32',
                'S_2_elev_m': 'float32',
                'weight': 'float32',
            })
            if 'S_1_elev_m' in sample_df.columns:
                dtype_dict['S_1_elev_m'] = 'float32'
            elif 'S_1_elev_mean' in sample_df.columns:
                dtype_dict['S_1_elev_mean'] = 'float32'

            topo_data = pd.read_csv(intersect_csv, dtype=dtype_dict)

            # Rename truncated columns back to full names
            rename_dict = {}
            if f'S_1_{self.hruId}' not in topo_data.columns:
                trunc = f'S_1_{self.hruId}'[:10]
                if trunc in topo_data.columns:
                    rename_dict[trunc] = f'S_1_{self.hruId}'
            if f'S_1_{self.gruId}' not in topo_data.columns:
                trunc = f'S_1_{self.gruId}'[:10]
                if trunc in topo_data.columns:
                    rename_dict[trunc] = f'S_1_{self.gruId}'
            if rename_dict:
                self.logger.info(f"Renaming truncated columns: {rename_dict}")
                topo_data.rename(columns=rename_dict, inplace=True)

            self.logger.info(f"Loaded topology data: {len(topo_data)} rows, {topo_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            self.logger.info(f"Columns after rename: {topo_data.columns.tolist()[:15]}")
            self.logger.info(f"Sample HRU IDs: {topo_data[f'S_1_{self.hruId}'].head(5).tolist()}")
            return topo_data
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error loading topology data: {str(e)}")
            raise

    def _get_forcing_files(self):
        """Return sorted list of basin forcing file names."""
        forcing_files = [
            f for f in os.listdir(self.forcing_basin_path)
            if f.startswith(f"{self.domain_name}") and f.endswith('.nc')
        ]
        forcing_files.sort()
        self.logger.info(f"Found {len(forcing_files)} forcing files to process")
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {self.forcing_basin_path}")
        return forcing_files

    def _prepare_forcing_output_dir(self):
        """Create output directory and remove stale forcing files."""
        self.forcing_summa_path.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.domain_name}_{self.forcing_dataset}".lower()
        for existing_file in self.forcing_summa_path.glob("*.nc"):
            if not existing_file.name.lower().startswith(prefix):
                continue
            try:
                existing_file.unlink()
                self.logger.info(f"Removed stale SUMMA forcing file {existing_file}")
            except OSError as exc:
                self.logger.warning(f"Failed to remove stale SUMMA forcing file {existing_file}: {exc}")

    @staticmethod
    def _extract_forcing_date(filename: str) -> Optional[datetime]:
        """Extract datetime token from common forcing filename patterns."""
        import re

        patterns = (
            (r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", "%Y-%m-%d-%H-%M-%S"),
            (r"(\d{6})\.nc$", "%Y%m"),
            (r"(\d{8})\.nc$", "%Y%m%d"),
            (r"_(\d{6})_", "%Y%m"),
        )

        for pattern, fmt in patterns:
            match = re.search(pattern, filename)
            if not match:
                continue
            try:
                return datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue
        return None

    def _precalculate_lapse_corrections(self, topo_data):
        """Pre-calculate per-HRU weighted lapse-rate corrections.

        Returns (lapse_values DataFrame indexed by hruId, lapse_rate in K/m).
        """
        gru_id = f'S_1_{self.gruId}'
        hru_id = f'S_1_{self.hruId}'
        catchment_elev = 'S_1_elev_m'
        forcing_elev = 'S_2_elev_m'
        weights = 'weight'

        raw_lapse = float(self._get_config_value(lambda: self.config.forcing.lapse_rate))
        if abs(raw_lapse) < 0.1:
            lapse_rate_km = raw_lapse * 1000.0
            lapse_rate = raw_lapse
        else:
            lapse_rate_km = raw_lapse
            lapse_rate = raw_lapse / 1000.0

        if raw_lapse < 0:
            self.logger.warning(
                f"Negative LAPSE_RATE ({raw_lapse}) detected. "
                "This will make higher elevations warmer. "
                "Standard lapse rates should be positive in SYMFLUENCE."
            )

        if catchment_elev not in topo_data.columns and 'S_1_elev_mean' in topo_data.columns:
            catchment_elev = 'S_1_elev_mean'

        self.logger.info(f"Pre-calculating lapse rate corrections (Rate: {lapse_rate_km:.2f} K/km)...")
        topo_data['lapse_values'] = (
            topo_data[weights] * lapse_rate * (topo_data[forcing_elev] - topo_data[catchment_elev])
        )

        if gru_id == hru_id:
            lapse_values = topo_data.groupby([hru_id])['lapse_values'].sum().reset_index()
        else:
            lapse_values = topo_data.groupby([gru_id, hru_id])['lapse_values'].sum().reset_index()

        lapse_values = lapse_values.sort_values(hru_id).set_index(hru_id)
        self.logger.info(f"Prepared lapse corrections for {len(lapse_values)} HRUs")
        self.logger.info(f"Lapse values HRU IDs: {lapse_values.index.tolist()}")
        return lapse_values, lapse_rate

    def _process_forcing_batches(self, forcing_files, lapse_values, lapse_rate):
        """Process forcing files in memory-efficient batches."""
        total_files = len(forcing_files)
        batch_size = self._determine_batch_size(total_files)
        self.logger.info(f"Processing files in batches of {batch_size}")

        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = forcing_files[batch_start:batch_end]

            memory_before = psutil.Process().memory_info().rss / 1024**2
            self.logger.debug(f"Memory usage before batch: {memory_before:.1f} MB")

            for i, file in enumerate(batch_files):
                try:
                    self._process_single_file(file, lapse_values, lapse_rate)
                    if (i + 1) % 10 == 0 or batch_size <= 10:
                        processed_count = batch_start + i + 1
                        self.logger.debug(f"Processed {processed_count}/{total_files} forcing files")
                except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                    self.logger.error(f"Error processing file {file}: {str(e)}")
                    raise

            gc.collect()
            memory_after = psutil.Process().memory_info().rss / 1024**2
            self.logger.debug(
                f"Memory usage after batch: {memory_after:.1f} MB "
                f"(delta: {memory_after - memory_before:+.1f} MB)"
            )

    def _process_single_file(self, file: str, lapse_values: pd.DataFrame, lapse_rate: float):
        """
        Process a single forcing file with comprehensive fixes for SUMMA compatibility.

        Fixes:
        1. Time coordinate format (convert to seconds since reference)
        2. NaN values in forcing data (interpolation)
        3. Data validation and quality checks

        Args:
            file: Filename to process
            lapse_values: Pre-calculated lapse values
            lapse_rate: Lapse rate value
        """
        input_path = self.forcing_basin_path / file
        output_path = self.forcing_summa_path / file

        self.logger.debug(f"Processing file: {file}")

        # Use context manager and process efficiently
        with xr.open_dataset(input_path) as dat:
            # Create a copy to avoid modifying the original
            dat = dat.copy()

            # 0. NORMALIZE LEGACY VARIABLE NAMES TO CFIF
            from symfluence.data.preprocessing.cfif.variables import normalize_to_cfif
            dat = normalize_to_cfif(dat)

            # 1. FIX TIME COORDINATE FIRST
            dat = self._fix_time_coordinate_comprehensive(dat, file)

            # Find which HRU IDs exist in the forcing data but not in the lapse values
            valid_hru_mask = np.isin(dat['hruId'].values, lapse_values.index)

            # Log and filter invalid HRUs
            if not np.all(valid_hru_mask):
                missing_hrus = dat['hruId'].values[~valid_hru_mask]
                if len(missing_hrus) <= 10:
                    self.logger.warning(f"File {file}: Removing {len(missing_hrus)} HRU IDs without lapse values: {missing_hrus}")
                else:
                    self.logger.warning(f"File {file}: Removing {len(missing_hrus)} HRU IDs without lapse values")

                # Filter the dataset
                dat = dat.sel(hru=valid_hru_mask)

                if len(dat.hru) == 0:
                    raise ValueError(f"File {file}: No valid HRUs found after filtering")

            # 2. FIX NaN VALUES IN FORCING DATA
            dat = self._fix_nan_values(dat, file)

            # 2a. STANDARDIZE VARIABLE NAMES
            dat = self._standardize_variable_names(dat, file)

            # 2b. ENSURE REQUIRED VARIABLES EXIST
            dat = self._ensure_required_forcing_variables(dat, file)

            # 3. VALIDATE DATA RANGES
            dat = self._validate_and_fix_data_ranges(dat, file)

            # Apply data step (memory efficient - in-place operation)
            dat['data_step'] = self.data_step
            dat.data_step.attrs.update({
                'long_name': 'data step length in seconds',
                'units': 's'
            })

            # Update precipitation units if present
            if 'precipitation_flux' in dat:
                # Handle cases where intermediate remapping (e.g. EASYMORE)
                # might have converted to m/s but SUMMA expects kg m-2 s-1 (mm/s)
                if dat.precipitation_flux.attrs.get('units') == 'm s-1' and float(dat.precipitation_flux.mean()) < 1e-6:
                    self.logger.info(f"File {file}: Converting pptrate from m s-1 to kg m-2 s-1 (x1000)")
                    dat['precipitation_flux'] = dat['precipitation_flux'] * 1000.0

                dat.precipitation_flux.attrs.update({
                    'units': 'kg m-2 s-1',
                    'long_name': 'Mean total precipitation rate'
                })

                # Apply lapse rate correction efficiently if enabled
            if (self._get_config_value(lambda: self.config.forcing.apply_lapse_rate)
                    and not dat.attrs.get('elevation_corrected', False)):
                # Get lapse values for the HRUs (vectorized operation)
                hru_lapse_values = lapse_values.loc[dat['hruId'].values, 'lapse_values'].values

                # Create correction array more efficiently, handling both (time, hru) and (hru, time)
                if dat['air_temperature'].dims == ('time', 'hru'):
                    lapse_correction = np.broadcast_to(hru_lapse_values[np.newaxis, :], dat['air_temperature'].shape)
                elif dat['air_temperature'].dims == ('hru', 'time'):
                    lapse_correction = np.broadcast_to(hru_lapse_values[:, np.newaxis], dat['air_temperature'].shape)
                else:
                    self.logger.warning(f"Unexpected airtemp dimensions {dat['air_temperature'].dims}, skipping lapse correction")
                    lapse_correction = 0

                # Store original attributes
                tmp_units = dat['air_temperature'].attrs.get('units', 'K')

                # Apply correction (in-place operation)
                if not isinstance(lapse_correction, int):
                    dat['air_temperature'].values += lapse_correction
                dat.air_temperature.attrs['units'] = tmp_units

                # Clean up temporary arrays
                del hru_lapse_values, lapse_correction

            # 4. FINAL VALIDATION BEFORE SAVING
            self._final_validation(dat, file)

            # Ensure hruId is int32 for SUMMA compatibility
            if 'hruId' in dat:
                dat['hruId'] = dat['hruId'].astype('int32')

            # Prepare encoding with time coordinate fix
            encoding: Dict[str, Any] = {
                str(var): {'zlib': True, 'complevel': 1, 'shuffle': True}
                for var in dat.data_vars
            }

            if 'hruId' in dat:
                encoding['hruId'] = {'dtype': 'int32', '_FillValue': None}

            # Ensure time coordinate is properly encoded for SUMMA
            encoding['time'] = {
                'dtype': 'float64',
                'zlib': True,
                'complevel': 1,
                '_FillValue': None
            }

            # Rename CFIF variables to SUMMA-native names for the Fortran executable
            from symfluence.data.preprocessing.cfif.variables import CFIF_TO_SUMMA_MAPPING
            summa_renames = {k: v for k, v in CFIF_TO_SUMMA_MAPPING.items() if k in dat}
            if summa_renames:
                dat = dat.rename(summa_renames)
                # Update encoding keys to match new variable names
                encoding = {
                    summa_renames.get(k, k): v for k, v in encoding.items()
                }

            dat.to_netcdf(output_path, encoding=encoding)

            # Explicit cleanup
            dat.close()
            del dat

    def _insert_noleap_leap_days(self):
        """
        Insert interpolated Feb 29 forcing files for each leap year in the
        simulation period.  Called after the main batch processing loop when
        the source calendar is ``noleap`` or ``365_day``.

        For every leap year found in the processed forcing date range the
        method:
        1. Locates the processed files that contain the last hour of Feb 28
           and the first hour of Mar 1.
        2. Extracts those two boundary time-steps.
        3. Linearly interpolates 24 hourly time-steps for Feb 29
           (00:00 – 23:00).
        4. Writes a new NetCDF file with the same encoding / compression as
           the existing files.

        The resulting files are named
        ``{domain}_{dataset}_leapday_{year}-02-29-00-00-00.nc`` and are
        automatically picked up by ``create_forcing_file_list`` because they
        share the expected prefix.
        """
        import calendar

        self.logger.info("Scanning processed forcing files for leap-day insertion (source calendar: %s)",
                         self._source_calendar)

        # ------------------------------------------------------------------
        # 1. Discover the date range from existing processed files
        # ------------------------------------------------------------------
        prefix = f"{self.domain_name}_{self.forcing_dataset}".lower()
        processed_files = sorted(
            f for f in os.listdir(self.forcing_summa_path)
            if f.lower().startswith(prefix) and f.endswith('.nc')
        )

        if not processed_files:
            self.logger.warning("No processed forcing files found – skipping leap-day insertion")
            return

        dates = [self._extract_forcing_date(f) for f in processed_files]
        dates = [d for d in dates if d is not None]

        if not dates:
            self.logger.warning("Could not extract dates from processed filenames – skipping leap-day insertion")
            return

        min_year = min(d.year for d in dates)
        max_year = max(d.year for d in dates)

        leap_years = [y for y in range(min_year, max_year + 1) if calendar.isleap(y)]
        if not leap_years:
            self.logger.info("No leap years in range %d–%d – nothing to insert", min_year, max_year)
            return

        # Reference date used for the seconds-since encoding
        reference_date = pd.Timestamp('1990-01-01 00:00:00')
        inserted = 0

        for year in leap_years:
            # Target times: Feb 28 23:00 and Mar 1 00:00
            feb28_23 = pd.Timestamp(f'{year}-02-28 23:00:00')
            mar01_00 = pd.Timestamp(f'{year}-03-01 00:00:00')

            feb28_23_sec = (feb28_23 - reference_date).total_seconds()
            mar01_00_sec = (mar01_00 - reference_date).total_seconds()

            # ------------------------------------------------------------------
            # 2. Find the file(s) containing these boundary time-steps
            # ------------------------------------------------------------------
            boundary_ds = {}  # key: 'feb28' or 'mar01', value: xr.Dataset single-time slice

            for fname in processed_files:
                fpath = self.forcing_summa_path / fname
                try:
                    with xr.open_dataset(fpath, decode_times=False) as ds:
                        time_vals = ds['time'].values
                        # Check for feb28_23
                        if 'feb28' not in boundary_ds:
                            mask = np.abs(time_vals - feb28_23_sec) < 1.0
                            if np.any(mask):
                                idx = int(np.argmax(mask))
                                boundary_ds['feb28'] = ds.isel(time=idx).load()
                        # Check for mar01_00
                        if 'mar01' not in boundary_ds:
                            mask = np.abs(time_vals - mar01_00_sec) < 1.0
                            if np.any(mask):
                                idx = int(np.argmax(mask))
                                boundary_ds['mar01'] = ds.isel(time=idx).load()
                except Exception as exc:  # noqa: BLE001 — model execution resilience
                    self.logger.debug("Could not read %s while scanning for leap-day boundaries: %s", fname, exc)
                    continue

                if len(boundary_ds) == 2:
                    break  # Found both boundaries

            if len(boundary_ds) < 2:
                self.logger.warning(
                    "Could not find Feb 28 23:00 and/or Mar 1 00:00 boundary data for %d – "
                    "skipping leap-day insertion for this year (found: %s)",
                    year, list(boundary_ds.keys())
                )
                continue

            # ------------------------------------------------------------------
            # 3. Interpolate 24 hourly time-steps for Feb 29 00:00–23:00
            # ------------------------------------------------------------------
            feb29_hours = pd.date_range(f'{year}-02-29 00:00', periods=24, freq='h')
            feb29_secs = np.array(
                [(t - reference_date).total_seconds() for t in feb29_hours],
                dtype=np.float64,
            )

            ds_feb28 = boundary_ds['feb28']
            ds_mar01 = boundary_ds['mar01']

            # Total gap = 25 hours (Feb 28 23:00 → Mar 1 00:00 next day = 25h)
            # Weights for linear interpolation: Feb 29 00:00 is 1h after feb28_23,
            # Feb 29 23:00 is 24h after feb28_23.  Mar 1 00:00 is 25h after feb28_23.
            # weight_mar = hours_since_feb28_23 / 25
            weights = np.arange(1, 25, dtype=np.float64) / 25.0  # [1/25 .. 24/25]

            # Build a new dataset
            # Start from the structure of the feb28 slice (has all vars, coords, attrs)
            data_vars = {}
            forcing_vars = [v for v in ds_feb28.data_vars if v != 'data_step']

            for var in forcing_vars:
                v28 = ds_feb28[var].values  # shape: (hru,) or scalar
                v01 = ds_mar01[var].values

                if np.ndim(v28) == 0:
                    # Scalar variable (e.g. hruId) – replicate
                    data_vars[var] = ('hru', np.full(1 if np.ndim(v28) == 0 else len(v28), v28))
                    continue

                # v28 and v01 are 1-D (hru,)
                nhru = len(v28)
                interp: np.ndarray = np.zeros((24, nhru), dtype=np.float64)
                for h_idx, w in enumerate(weights):
                    interp[h_idx, :] = (1.0 - w) * v28 + w * v01

                data_vars[var] = (('time', 'hru'), interp)

            # Handle hruId – keep from source
            if 'hruId' in ds_feb28:
                data_vars['hruId'] = ('hru', ds_feb28['hruId'].values.copy())

            # data_step scalar
            data_vars['data_step'] = self.data_step

            coords = {
                'time': ('time', feb29_secs),
                'hru': ('hru', ds_feb28['hru'].values if 'hru' in ds_feb28.coords else np.arange(len(ds_feb28.hru))),
            }

            leap_ds = xr.Dataset(data_vars, coords=coords)

            # Copy variable attributes from source
            for var in leap_ds.data_vars:
                if var in ds_feb28:
                    leap_ds[var].attrs = dict(ds_feb28[var].attrs)

            # Set time attributes
            leap_ds['time'].attrs = {
                'units': 'seconds since 1990-01-01 00:00:00',
                'calendar': 'standard',
                'long_name': 'time',
                'axis': 'T',
            }

            leap_ds['data_step'] = self.data_step
            leap_ds['data_step'].attrs = {
                'long_name': 'data step length in seconds',
                'units': 's',
            }

            # Ensure hruId is int32
            if 'hruId' in leap_ds:
                leap_ds['hruId'] = leap_ds['hruId'].astype('int32')

            # ------------------------------------------------------------------
            # 4. Write the leap-day file
            # ------------------------------------------------------------------
            out_name = f"{self.domain_name}_{self.forcing_dataset}_leapday_{year}-02-29-00-00-00.nc"
            out_path = self.forcing_summa_path / out_name

            encoding: dict[str, dict] = {
                str(v): {'zlib': True, 'complevel': 1, 'shuffle': True}
                for v in leap_ds.data_vars
            }
            if 'hruId' in leap_ds:
                encoding['hruId'] = {'dtype': 'int32', '_FillValue': None}
            encoding['time'] = {
                'dtype': 'float64',
                'zlib': True,
                'complevel': 1,
                '_FillValue': None,
            }

            leap_ds.to_netcdf(out_path, encoding=encoding)
            leap_ds.close()
            inserted += 1
            self.logger.info("Inserted leap-day forcing file: %s", out_name)

        self.logger.info("Leap-day insertion complete: inserted %d leap-day file(s)", inserted)

    def _fix_time_coordinate_comprehensive(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Fix time coordinate to ensure SUMMA compatibility using only the data's time coordinate.
        No filename parsing - just uses the actual time data which is always authoritative.

        Args:
            dataset: Input dataset
            filename: Filename for logging

        Returns:
            Dataset with corrected time coordinate
        """
        try:
            # Check if time exists in the dataset
            if 'time' not in dataset.dims and 'time' not in dataset.coords:
                raise ValueError("Dataset has no 'time' dimension or coordinate")

            # Use bracket notation to access time safely
            time_coord = dataset['time']

            self.logger.debug(f"File {filename}: Original time dtype: {time_coord.dtype}")

            # Detect source calendar before any conversion
            source_calendar = 'standard'
            if 'calendar' in time_coord.attrs:
                source_calendar = time_coord.attrs['calendar']
            elif hasattr(time_coord.values[0], 'calendar'):
                # cftime objects carry their calendar (e.g., DatetimeNoLeap)
                cal = getattr(time_coord.values[0], 'calendar', None)
                if cal:
                    source_calendar = cal
            # NEX-GDDP acquisition handler already converts all calendars
            # (noleap, 360_day, etc.) to standard Gregorian with explicit
            # calendar=standard encoding, so trust the detected calendar here.

            self._source_calendar = source_calendar
            self.logger.debug(f"File {filename}: Detected source calendar: {source_calendar}")

            # Convert any time format to pandas datetime first
            if time_coord.dtype.kind == 'M':  # datetime64
                pd_times = pd.to_datetime(time_coord.values)
            elif np.issubdtype(time_coord.dtype, np.number):
                if 'units' in time_coord.attrs and 'since' in time_coord.attrs['units']:
                    # Parse existing time units to understand the reference and time unit
                    units_str = time_coord.attrs['units']
                    if 'since' in units_str:
                        # Parse the time unit from the units string (e.g., "hours since", "days since")
                        parts = units_str.split()
                        time_unit_str = parts[0].lower() if parts else 'seconds'

                        # Map to pandas unit codes
                        unit_map = {
                            'seconds': 's', 'second': 's',
                            'hours': 'h', 'hour': 'h',
                            'days': 'D', 'day': 'D',
                            'minutes': 'm', 'minute': 'm'
                        }
                        pd_unit = unit_map.get(time_unit_str, 's')

                        reference_str = units_str.split('since ')[1]
                        self.logger.debug(f"File {filename}: Parsing time with unit='{pd_unit}' from '{time_unit_str}'")
                        pd_times = pd.to_datetime(time_coord.values, unit=pd_unit, origin=pd.Timestamp(reference_str))  # type: ignore[call-overload]
                    else:
                        # Assume seconds since unix epoch if no reference given
                        pd_times = pd.to_datetime(time_coord.values, unit='s')
                else:
                    # Try to interpret as seconds since unix epoch
                    pd_times = pd.to_datetime(time_coord.values, unit='s')
            else:
                # Try direct conversion
                pd_times = pd.to_datetime(time_coord.values)

            self.logger.debug(f"File {filename}: Time range from data: {pd_times[0]} to {pd_times[-1]}")

            # Get time step from config
            time_step_seconds = int(self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600))
            len(pd_times)

            # Convert to SUMMA's expected format: seconds since 1990-01-01 00:00:00
            reference_date = pd.Timestamp('1990-01-01 00:00:00')
            seconds_since_ref = (pd_times - reference_date).total_seconds().values

            # Ensure perfect integer seconds to avoid floating point precision issues
            seconds_since_ref = np.round(seconds_since_ref).astype(np.int64).astype(np.float64)

            # Replace the time coordinate
            dataset = dataset.assign_coords(time=seconds_since_ref)

            # Ensure time is monotonic
            dataset = dataset.sortby('time')

            # Set proper attributes for SUMMA
            dataset.time.attrs = {
                'units': 'seconds since 1990-01-01 00:00:00',
                'calendar': source_calendar,
                'long_name': 'time',
                'axis': 'T'
            }

            self.logger.debug(f"File {filename}: Final time range: {seconds_since_ref[0]:.0f} to {seconds_since_ref[-1]:.0f} seconds")

            # Validate the conversion
            if len(seconds_since_ref) == 0:
                raise ValueError("Empty time coordinate after conversion")

            if np.any(np.isnan(seconds_since_ref)):
                raise ValueError("NaN values in converted time coordinate")

            # Check time step consistency (but don't force it - preserve actual data timing)
            if len(seconds_since_ref) > 1:
                time_diffs = np.diff(seconds_since_ref)
                expected_step = time_step_seconds

                # Check if most time steps match expected (allowing for some variability)
                step_matches = np.abs(time_diffs - expected_step) < (expected_step * 0.01)  # 1% tolerance
                match_percentage = np.sum(step_matches) / len(step_matches) * 100

                if match_percentage < 90:
                    self.logger.warning(f"File {filename}: Only {match_percentage:.1f}% of time steps match expected step size")
                    actual_median_step = int(np.median(time_diffs))
                    self.logger.warning(
                        f"File {filename}: Expected step: {expected_step}s, Actual median: {actual_median_step:.0f}s"
                    )
                    if actual_median_step > 0 and abs(actual_median_step - expected_step) > expected_step * 0.01:
                        self.logger.info(
                            f"File {filename}: Updating data_step from {self.data_step}s to {actual_median_step}s "
                            f"based on actual forcing timestep"
                        )
                        self.data_step = actual_median_step
                else:
                    self.logger.debug(f"File {filename}: Time steps are consistent ({match_percentage:.1f}% match)")

            return dataset

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"File {filename}: Error fixing time coordinate: {str(e)}")
            raise ValueError(f"Cannot fix time coordinate in file {filename}: {str(e)}") from e

    def _fix_nan_values(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Fix NaN values in forcing data through interpolation and filling.
        Handles CASR data pattern where only every 3rd temperature value is valid.

        Args:
            dataset: Input dataset
            filename: Filename for logging

        Returns:
            Dataset with NaN values filled
        """
        forcing_vars = ['air_temperature', 'surface_air_pressure', 'specific_humidity', 'wind_speed', 'precipitation_flux', 'surface_downwelling_longwave_flux', 'surface_downwelling_shortwave_flux']

        for var in forcing_vars:
            if var not in dataset:
                continue

            var_data = dataset[var]

            # Count NaN values
            nan_count = np.isnan(var_data.values).sum()
            total_count = var_data.size

            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100

                # Apply interpolation strategy based on variable type
                if var == 'precipitation_flux':
                    # For precipitation, fill NaN with 0 (no precipitation)
                    filled_data = var_data.fillna(0.0)
                    self.logger.debug(f"File {filename}: Filled {var} NaN values with 0")

                elif var in ['surface_downwelling_shortwave_flux']:
                    # For solar radiation, interpolate during day, zero at night
                    filled_data = var_data.interpolate_na(dim='time', method='linear')
                    filled_data = filled_data.ffill(dim='time').bfill(dim='time')
                    filled_data = filled_data.fillna(0.0)
                    self.logger.debug(f"File {filename}: Interpolated {var} NaN values")

                elif var == 'air_temperature' and nan_percentage > 50:
                    # Special handling for CASR temperature pattern (high NaN percentage)

                    # Use scipy cubic interpolation for better results with sparse temperature data
                    try:
                        from scipy import interpolate
                        filled_data = var_data.copy()

                        # Process each HRU separately
                        for hru_idx in range(var_data.shape[-1] if len(var_data.shape) == 2 else 1):
                            if len(var_data.shape) == 2:
                                temp_values = var_data.values[:, hru_idx]
                            else:
                                temp_values = var_data.values

                            # Find valid (non-NaN) indices
                            valid_mask = ~np.isnan(temp_values)
                            valid_indices = np.where(valid_mask)[0]
                            valid_values = temp_values[valid_mask]

                            if len(valid_values) >= 2:
                                # Use cubic for smooth interpolation if enough points, otherwise linear
                                # Avoid extrapolation with cubic as it can produce wild values at boundaries
                                kind = 'cubic' if len(valid_values) >= 4 else 'linear'

                                f = interpolate.interp1d(
                                    valid_indices,
                                    valid_values,
                                    kind=kind,
                                    bounds_error=False,
                                    fill_value=(valid_values[0], valid_values[-1])
                                )

                                # Interpolate all time steps
                                all_indices = np.arange(len(temp_values))
                                interpolated_values = f(all_indices)

                                # Update the data
                                if len(var_data.shape) == 2:
                                    filled_data.values[:, hru_idx] = interpolated_values
                                else:
                                    filled_data.values[:] = interpolated_values
                            else:
                                # Not enough valid values, use default
                                if len(var_data.shape) == 2:
                                    filled_data.values[:, hru_idx] = PhysicalConstants.KELVIN_OFFSET  # 0°C
                                else:
                                    filled_data.values[:] = PhysicalConstants.KELVIN_OFFSET

                        # Clip to reasonable temperature range
                        filled_data = filled_data.clip(min=200.0, max=350.0)

                    except ImportError:
                        self.logger.warning(f"File {filename}: scipy not available, using xarray interpolation")
                        filled_data = var_data.interpolate_na(dim='time', method='linear')
                        filled_data = filled_data.ffill(dim='time').bfill(dim='time')
                        filled_data = filled_data.fillna(PhysicalConstants.KELVIN_OFFSET)
                        filled_data = filled_data.clip(min=200.0, max=350.0)

                    self.logger.debug(f"File {filename}: Applied CASR temperature interpolation")

                elif nan_percentage > 80:  # Only reject if >80% NaN for non-temperature variables
                    self.logger.error(f"File {filename}: Too many NaN values in {var} ({nan_percentage:.1f}%)")
                    raise ValueError(f"Variable {var} has too many NaN values to interpolate reliably")

                else:
                    # Standard interpolation for other variables
                    filled_data = var_data.interpolate_na(dim='time', method='linear')
                    filled_data = filled_data.ffill(dim='time').bfill(dim='time')

                    # If still NaN, use reasonable defaults
                    if np.any(np.isnan(filled_data.values)):
                        if var == 'air_temperature':
                            default_val = PhysicalConstants.KELVIN_OFFSET  # 0°C in Kelvin
                        elif var == 'surface_air_pressure':
                            default_val = 101325.0  # Standard pressure in Pa
                        elif var == 'specific_humidity':
                            default_val = 0.005  # Reasonable specific humidity
                        elif var == 'wind_speed':
                            default_val = 2.0  # Light wind in m/s
                        elif var == 'surface_downwelling_longwave_flux':
                            default_val = 300.0  # Reasonable longwave radiation
                        else:
                            default_val = 0.0

                        filled_data = filled_data.fillna(default_val)
                        self.logger.warning(f"File {filename}: Used default value {default_val} for remaining {var} NaN values")

                    self.logger.debug(f"File {filename}: Interpolated {var} NaN values")

                # Replace the variable in dataset
                dataset[var] = filled_data

                # Verify no NaN values remain
                remaining_nans = np.isnan(dataset[var].values).sum()
                if remaining_nans > 0:
                    self.logger.error(f"File {filename}: Still have {remaining_nans} NaN values in {var} after fixing")
                    raise ValueError(f"Failed to remove all NaN values from {var}")

        return dataset

    def _standardize_variable_names(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Standardize variable names to SUMMA conventions.

        Maps common alternative variable names to SUMMA expected names.
        This handles datasets that use non-standard naming conventions.

        Args:
            dataset: Input dataset with potentially non-standard variable names
            filename: Filename for logging

        Returns:
            Dataset with standardized variable names
        """
        # Common variable name aliases that need mapping
        name_mappings = {
            # Pressure variations
            'pressure': 'surface_air_pressure',
            'press': 'surface_air_pressure',
            'sp': 'surface_air_pressure',
            'surface_pressure': 'surface_air_pressure',
            # Humidity variations
            'r2': 'relative_humidity',
            'rh': 'relative_humidity',
            'relative_humidity': 'relative_humidity',
            'rh2m': 'relative_humidity',
            '2m_relative_humidity': 'relative_humidity',
            # Temperature variations
            't2m': 'air_temperature',
            'temp': 'air_temperature',
            'temperature': 'air_temperature',
            '2m_temperature': 'air_temperature',
            # Wind components
            'u10': 'eastward_wind',
            'v10': 'northward_wind',
            'u_component_of_wind': 'eastward_wind',
            'v_component_of_wind': 'northward_wind',
        }

        # Apply mappings for variables that exist
        rename_dict = {}
        for old_name, new_name in name_mappings.items():
            if old_name in dataset and new_name not in dataset:
                rename_dict[old_name] = new_name

        if rename_dict:
            self.logger.info(f"File {filename}: Renaming variables: {rename_dict}")
            dataset = dataset.rename(rename_dict)

        return dataset

    def _compute_specific_humidity(
        self, T: xr.DataArray, RH: xr.DataArray, P: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate specific humidity from temperature, relative humidity, and pressure.

        Uses the Magnus approximation formula for saturation vapor pressure.

        Args:
            T: Temperature in Kelvin
            RH: Relative humidity (0-100)
            P: Pressure in Pa

        Returns:
            Specific humidity (kg/kg)
        """
        # Convert to Celsius
        T_celsius = T - 273.15

        # Saturation vapor pressure (Magnus formula)
        es = 611.2 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))

        # Actual vapor pressure
        e = (RH / 100.0) * es

        # Specific humidity
        spechum = (0.622 * e) / (P - 0.378 * e)

        # Set attributes
        spechum.attrs = {
            'units': 'kg kg-1',
            'long_name': 'specific humidity',
            'standard_name': 'specific_humidity'
        }

        return spechum

    def _estimate_longwave_radiation(self, T: xr.DataArray) -> xr.DataArray:
        """
        Estimate incoming longwave radiation from air temperature.

        This is a rough estimate based on air temperature, assuming clear-sky
        conditions. Uses a simplified empirical relationship where LW increases
        with temperature. Should only be used when LWRadAtm is not available.

        Args:
            T: Air temperature in Kelvin

        Returns:
            Estimated longwave radiation (W/m²)
        """
        # Simple empirical relationship: LW ≈ 200 + 0.8*T (W/m²)
        # This gives reasonable values:
        #   T = 273K (0°C):  ~218 W/m² (cold, clear sky)
        #   T = 288K (15°C): ~230 W/m² (moderate)
        #   T = 303K (30°C): ~242 W/m² (warm, clear sky)
        #
        # More sophisticated would be: σ * ε * T^4
        # where ε is atmospheric emissivity (~0.7-0.9 depending on humidity)
        # but we don't have humidity data here

        # Use a more conservative approach: Stefan-Boltzmann with effective emissivity
        # Effective clear-sky emissivity typically ranges from 0.7 to 0.85
        # We'll use 0.75 as a reasonable middle ground
        sigma = 5.67e-8  # Stefan-Boltzmann constant (W m-2 K-4)
        emissivity = 0.75  # Effective atmospheric emissivity for clear sky

        LW = emissivity * sigma * T ** 4

        # Set attributes
        LW.attrs = {
            'units': 'W m-2',
            'long_name': 'estimated incoming longwave radiation',
            'standard_name': 'surface_downwelling_longwave_flux_in_air',
            'note': 'Estimated using simplified clear-sky emissivity (0.75) - use with caution'
        }

        return LW

    def _compute_wind_speed(
        self, u: xr.DataArray, v: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate wind speed magnitude from u and v components.

        Args:
            u: Eastward wind component (m/s)
            v: Northward wind component (m/s)

        Returns:
            Wind speed magnitude (m/s)
        """
        windspd = cast(xr.DataArray, np.sqrt(u ** 2 + v ** 2))

        # Set attributes
        windspd.attrs = {
            'units': 'm s-1',
            'long_name': 'wind speed',
            'standard_name': 'wind_speed'
        }

        return windspd

    def _ensure_required_forcing_variables(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Ensure all required forcing variables exist in the dataset.

        Some forcing products (e.g., CARRA) can omit variables like LWRadAtm.
        SUMMA expects a full set of forcing variables, so add missing variables
        with reasonable defaults and log a warning.
        """
        # Auto-rename legacy SUMMA-style variable names to CFIF if present
        from symfluence.data.preprocessing.cfif.variables import normalize_to_cfif
        dataset = normalize_to_cfif(dataset)

        required_vars = ['air_temperature', 'surface_air_pressure', 'specific_humidity', 'wind_speed', 'precipitation_flux', 'surface_downwelling_longwave_flux', 'surface_downwelling_shortwave_flux']
        defaults = {
            'air_temperature': PhysicalConstants.KELVIN_OFFSET,   # 0°C in Kelvin
            'surface_air_pressure': 101325.0, # Standard pressure in Pa
            'specific_humidity': 0.005,    # Reasonable specific humidity
            'wind_speed': 2.0,      # Light wind in m/s
            'precipitation_flux': 0.0,      # No precipitation
            'surface_downwelling_longwave_flux': 300.0,   # Reasonable longwave radiation W/m^2
            'surface_downwelling_shortwave_flux': 0.0      # Default shortwave radiation W/m^2
        }
        units = {
            'air_temperature': 'K',
            'surface_air_pressure': 'Pa',
            'specific_humidity': 'kg/kg',
            'wind_speed': 'm/s',
            'precipitation_flux': 'mm/s',
            'surface_downwelling_longwave_flux': 'W/m2',
            'surface_downwelling_shortwave_flux': 'W/m2'
        }

        # Check for missing required variables and try to compute them
        missing_vars = [var for var in required_vars if var not in dataset]

        if missing_vars:
            # Try to compute missing variables from available data
            computed_vars = []

            for var in missing_vars[:]:  # Use slice to allow modification during iteration
                if var == 'specific_humidity' and 'relative_humidity' in dataset and 'air_temperature' in dataset and 'surface_air_pressure' in dataset:
                    # Compute specific humidity from relative humidity
                    self.logger.info(f"File {filename}: Computing spechum from relhum, airtemp, airpres")
                    dataset['specific_humidity'] = self._compute_specific_humidity(
                        dataset['air_temperature'],
                        dataset['relative_humidity'],
                        dataset['surface_air_pressure']
                    )
                    missing_vars.remove(var)
                    computed_vars.append(var)

                elif var == 'wind_speed' and 'eastward_wind' in dataset and 'northward_wind' in dataset:
                    # Compute wind speed from components
                    self.logger.info(f"File {filename}: Computing windspd from windspd_u and windspd_v")
                    dataset['wind_speed'] = self._compute_wind_speed(
                        dataset['eastward_wind'],
                        dataset['northward_wind']
                    )
                    missing_vars.remove(var)
                    computed_vars.append(var)

                elif var == 'surface_downwelling_longwave_flux' and 'air_temperature' in dataset:
                    # Estimate longwave radiation from temperature
                    self.logger.warning(
                        f"File {filename}: LWRadAtm missing - estimating from airtemp. "
                        "This is a rough approximation and may affect model accuracy."
                    )
                    dataset['surface_downwelling_longwave_flux'] = self._estimate_longwave_radiation(dataset['air_temperature'])
                    missing_vars.remove(var)
                    computed_vars.append(var)

            # If still missing variables, raise error
            if missing_vars:
                available_vars = list(dataset.data_vars)
                error_msg = (
                    f"Missing required forcing variables in {filename}: {missing_vars}\n"
                    f"Available variables: {available_vars}\n"
                )

                if computed_vars:
                    error_msg += f"Successfully computed: {computed_vars}\n"

                error_msg += (
                    "\nSUMMA requires the following forcing variables:\n"
                )
                for var in missing_vars:
                    error_msg += f"  - {var} ({units.get(var, 'unknown units')})\n"

                error_msg += (
                    f"\nTo fix this issue:\n"
                    f"1. Ensure your forcing data source provides these variables\n"
                    f"2. Check variable naming conventions in your forcing files\n"
                    f"3. Verify forcing preprocessing pipeline includes all required variables\n"
                    f"\nNote: Previous versions filled missing variables with defaults "
                    f"({', '.join(f'{v}={defaults[v]}' for v in missing_vars[:3])}...), "
                    f"but this can produce physically unrealistic results."
                )
                raise ValueError(error_msg)

        return dataset

    def _validate_and_fix_data_ranges(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Validate and fix unrealistic data ranges that could cause SUMMA to fail.

        Args:
            dataset: Input dataset
            filename: Filename for logging

        Returns:
            Dataset with validated data ranges
        """
        # Define reasonable ranges for variables
        valid_ranges = {
            'air_temperature': (200.0, 350.0),      # -73°C to 77°C
            'surface_air_pressure': (50000.0, 110000.0), # 50-110 kPa
            'specific_humidity': (0.0, 0.1),          # 0-100 g/kg
            'wind_speed': (0.0, 100.0),        # 0-100 m/s
            'precipitation_flux': (0.0, 0.1),          # 0-360 mm/hr in mm/s
            'surface_downwelling_longwave_flux': (50.0, 600.0),      # Longwave radiation W/m²
            'surface_downwelling_shortwave_flux': (0.0, 1500.0)       # Shortwave radiation W/m²
        }

        # Tolerances for clipping trivially out-of-range values (e.g., interpolation artifacts)
        # Values within this absolute tolerance of the boundary are clipped with a warning
        # Values beyond this tolerance trigger an error
        clip_tolerances = {
            'air_temperature': 1.0,        # 1 K
            'surface_air_pressure': 500.0,      # 500 Pa
            'specific_humidity': 0.001,      # 0.001 kg/kg
            'wind_speed': 0.5,        # 0.5 m/s
            'precipitation_flux': 0.001,      # 0.001 mm/s
            'surface_downwelling_longwave_flux': 10.0,      # 10 W/m²
            'surface_downwelling_shortwave_flux': 10.0,      # 10 W/m²
        }

        out_of_range_errors = []

        for var, (min_val, max_val) in valid_ranges.items():
            if var not in dataset:
                continue

            var_data = dataset[var]
            tolerance = clip_tolerances.get(var, 0.0)

            # Clip values that are trivially out of range (within tolerance)
            below_min_trivial = (var_data < min_val) & (var_data >= min_val - tolerance)
            above_max_trivial = (var_data > max_val) & (var_data <= max_val + tolerance)
            trivial_count = int(below_min_trivial.sum()) + int(above_max_trivial.sum())

            if trivial_count > 0:
                self.logger.warning(
                    f"File {filename}: Clipping {trivial_count} trivially out-of-range "
                    f"{var} values to [{min_val}, {max_val}] (within tolerance {tolerance})"
                )
                dataset[var] = var_data.clip(min=min_val, max=max_val)
                var_data = dataset[var]

            # Check for remaining out-of-range values (beyond tolerance)
            below_min_count = int((var_data < min_val).sum())
            above_max_count = int((var_data > max_val).sum())

            if below_min_count > 0 or above_max_count > 0:
                var_min = float(var_data.min())
                var_max = float(var_data.max())
                var_mean = float(var_data.mean())

                error_details = (
                    f"  {var}: {below_min_count + above_max_count} values out of range [{min_val}, {max_val}]\n"
                    f"    Actual range: [{var_min:.3f}, {var_max:.3f}], mean: {var_mean:.3f}\n"
                )

                # Provide specific guidance based on variable
                if var == 'air_temperature':
                    if var_mean < 100:
                        error_details += "    → Temperature appears to be in °C, expected Kelvin. Add 273.15\n"
                    elif var_max > 350:
                        error_details += "    → Temperature values unrealistically high. Check source data.\n"
                elif var == 'surface_air_pressure':
                    if var_mean < 1000:
                        error_details += "    → Pressure appears to be in hPa or kPa, expected Pa. Multiply by 100 or 1000\n"
                elif var == 'precipitation_flux':
                    if var_max > 0.1:
                        error_details += "    → Precipitation rate too high. Expected mm/s, got mm/day or mm/hour?\n"
                elif var == 'specific_humidity':
                    if var_max > 0.1:
                        error_details += "    → Specific humidity too high. Expected kg/kg, got g/kg? Divide by 1000\n"

                out_of_range_errors.append(error_details)

        if out_of_range_errors:
            error_msg = (
                f"Out-of-range forcing values detected in {filename}:\n\n"
                + "".join(out_of_range_errors) +
                "\nThese values are outside physically realistic ranges and will cause SUMMA to fail or produce incorrect results.\n"
                "Please check:\n"
                "1. Unit conversions in your forcing data pipeline\n"
                "2. Source data quality and processing steps\n"
                "3. Variable naming and mapping conventions\n\n"
                "Note: Previous versions silently clipped out-of-range values, which could hide data issues and invalidate results."
            )
            raise ValueError(error_msg)

        return dataset

    def _final_validation(self, dataset: xr.Dataset, filename: str):
        """
        Final validation to ensure dataset is ready for SUMMA.

        Args:
            dataset: Dataset to validate
            filename: Filename for logging
        """
        # Check time coordinate
        time_coord = dataset.time

        if not np.issubdtype(time_coord.dtype, np.number):
            raise ValueError(f"File {filename}: Time coordinate is not numeric after fixing")

        if 'units' not in time_coord.attrs or 'since' not in time_coord.attrs['units']:
            raise ValueError(f"File {filename}: Time coordinate missing proper units")

        # Check for any remaining NaN values in critical variables
        critical_vars = ['air_temperature', 'surface_air_pressure', 'specific_humidity', 'wind_speed']
        for var in critical_vars:
            if var in dataset:
                nan_count = np.isnan(dataset[var].values).sum()
                if nan_count > 0:
                    raise ValueError(f"File {filename}: Variable {var} still has {nan_count} NaN values")

        # Check that all arrays have consistent shapes
        expected_shape = (len(dataset.time), len(dataset.hru))
        for var in dataset.data_vars:
            if var not in ['data_step', 'latitude', 'longitude', 'hruId'] and hasattr(dataset[var], 'shape'):
                if dataset[var].shape != expected_shape:
                    self.logger.warning(f"File {filename}: Variable {var} has unexpected shape {dataset[var].shape}, "
                                    f"expected {expected_shape}")

        self.logger.debug(f"File {filename}: Passed final validation for SUMMA compatibility")

    def _infer_forcing_step_from_filenames(self, forcing_files: List[str]) -> int | None:
        forcing_times = []
        for forcing_file in forcing_files:
            stem = Path(forcing_file).stem
            time_token = stem.split("_")[-1]
            try:
                forcing_times.append(datetime.strptime(time_token, "%Y-%m-%d-%H-%M-%S"))
            except ValueError:
                continue

        if len(forcing_times) < 2:
            return None

        forcing_times.sort()
        diffs = [
            (forcing_times[idx] - forcing_times[idx - 1]).total_seconds()
            for idx in range(1, len(forcing_times))
            if forcing_times[idx] > forcing_times[idx - 1]
        ]
        if not diffs:
            return None

        return int(np.median(diffs))

    def _determine_batch_size(self, total_files: int) -> int:
        """
        Determine optimal batch size based on available memory and file count.

        Args:
            total_files: Total number of files to process

        Returns:
            Optimal batch size
        """
        try:
            # Get available memory in MB
            available_memory = psutil.virtual_memory().available / 1024**2

            # Conservative estimate: assume each file uses ~50MB during processing
            # (this includes temporary arrays, xarray overhead, etc.)
            estimated_memory_per_file = 50

            # Use at most 70% of available memory for batch processing
            max_memory_for_batch = available_memory * 0.7

            # Calculate batch size based on memory constraint
            memory_based_batch_size = max(1, int(max_memory_for_batch / estimated_memory_per_file))

            # Set reasonable bounds
            min_batch_size = 1
            max_batch_size = min(100, total_files)  # Don't exceed 100 files per batch

            # Choose the most conservative estimate
            batch_size = max(min_batch_size, min(memory_based_batch_size, max_batch_size))

            self.logger.debug(f"Batch size calculation: available_memory={available_memory:.1f}MB, "
                            f"memory_based_size={memory_based_batch_size}, "
                            f"chosen_size={batch_size}")

            return batch_size

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Could not determine optimal batch size: {str(e)}. Using default.")
            return min(10, total_files)  # Conservative fallback

    def create_forcing_file_list(self):
        """
        Create a list of forcing files for SUMMA.

        This method performs the following steps:
        1. Determine the forcing dataset from the configuration
        2. Find all relevant forcing files in the SUMMA input directory
        3. Sort the files to ensure chronological order
        4. Write the sorted file list to a text file

        The resulting file list is used by SUMMA to locate and read the forcing data.

        Raises:
            FileNotFoundError: If no forcing files are found.
            IOError: If there are issues writing the file list.
        """
        self.logger.info("Creating forcing file list")

        forcing_dataset = self._get_config_value(lambda: self.config.forcing.dataset)
        domain_name = self._get_config_value(lambda: self.config.domain.name)
        forcing_path = self.project_forcing_dir / "SUMMA_input"
        file_list_path = (
            self.setup_dir / self._get_config_value(lambda: self.config.model.summa.forcing_list)
        )

        forcing_dataset_upper = forcing_dataset.upper()

        # All datasets we *know* about and expect to behave like the others
        supported_datasets = {
            "CARRA",
            "ERA5",
            "RDRS",
            "CASR",
            "AORC",
            "CONUS404",
            "NEX-GDDP-CMIP6",
            "HRRR",
        }

        if forcing_dataset_upper in supported_datasets:
            prefix = f"{domain_name}_{forcing_dataset}"
        else:
            # Fall back to a generic prefix so future datasets still work,
            # but emit a warning so we notice.
            self.logger.warning(
                "Forcing dataset %s is not in the supported list %s; "
                "using generic prefix '%s_' for SUMMA forcing files.",
                forcing_dataset,
                supported_datasets,
                domain_name,
            )
            prefix = f"{domain_name}_"

        self.logger.info(
            "Looking for SUMMA forcing files in %s with prefix '%s' and extension '.nc'",
            forcing_path,
            prefix,
        )

        if not forcing_path.exists():
            self.logger.error("Forcing SUMMA_input directory does not exist: %s", forcing_path)
            raise FileNotFoundError(f"SUMMA forcing directory not found: {forcing_path}")

        forcing_files = [
            f for f in os.listdir(forcing_path)
            if f.startswith(prefix) and f.endswith(".nc")
        ]

        if not forcing_files:
            self.logger.error(
                "No forcing files found for dataset %s in %s (prefix '%s')",
                forcing_dataset,
                forcing_path,
                prefix,
            )
            raise FileNotFoundError(
                f"No {forcing_dataset} forcing files found in {forcing_path}"
            )

        # Sort and deduplicate (prefer files with longer names which usually contain full timestamps)
        forcing_files.sort(key=lambda x: (self._extract_forcing_date(x) or datetime(1900, 1, 1), -len(x)))

        unique_files = []
        seen_dates = set()
        for f in forcing_files:
            date = self._extract_forcing_date(f) or datetime(1900, 1, 1)
            if date not in seen_dates:
                unique_files.append(f)
                seen_dates.add(date)
            else:
                self.logger.warning(f"Skipping duplicate forcing file for date {date}: {f}")

        forcing_files = unique_files

        self.logger.info(
            "Found %d unique %s forcing files for SUMMA",
            len(forcing_files),
            forcing_dataset,
        )

        with open(file_list_path, "w", encoding="utf-8") as fobj:
            for fname in forcing_files:
                fobj.write(f"{fname}\n")

        self.logger.info(
            "Forcing file list created at %s with %d files",
            file_list_path,
            len(forcing_files),
        )

    def _filter_forcing_hru_ids(self, forcing_hru_ids):
        """
        Filter forcing HRU IDs against catchment shapefile to ensure consistency.

        Args:
            forcing_hru_ids: List or array of HRU IDs from forcing data

        Returns:
            Filtered list of HRU IDs that exist in catchment shapefile
        """
        forcing_hru_ids = list(forcing_hru_ids)
        try:
            shp = gpd.read_file(self.catchment_path / self.catchment_name)
            shp = shp.set_index(self._get_config_value(lambda: self.config.domain.catchment_shp_hruid))
            shp.index = shp.index.astype(int)
            available_hru_ids = set(shp.index.astype(int))
        except Exception as exc:  # noqa: BLE001 — model execution resilience
            self.logger.warning(
                "Unable to filter forcing HRU IDs against catchment shapefile: %s",
                exc,
            )
            return forcing_hru_ids

        missing_hru_ids = [hru_id for hru_id in forcing_hru_ids if hru_id not in available_hru_ids]
        if missing_hru_ids:
            self.logger.warning(
                "Forcing HRU IDs not found in catchment shapefile; filtering missing IDs. "
                "Missing count: %s (showing first 10): %s",
                len(missing_hru_ids),
                missing_hru_ids[:10],
            )
            forcing_hru_ids = [hru_id for hru_id in forcing_hru_ids if hru_id in available_hru_ids]
        if len(forcing_hru_ids) == 0:
            raise ValueError("No forcing HRU IDs match catchment shapefile HRU IDs.")
        return forcing_hru_ids
