# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
File Processor

Handles parallel and serial processing of forcing files.
"""
from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from tqdm import tqdm

from symfluence.core.mixins import ConfigMixin

from .file_validator import FileValidator


def _init_worker_pool():
    """
    Initialize worker process for multiprocessing pool.

    This function is called once per worker process when the pool is created.
    It configures HDF5/netCDF4 thread safety to prevent segmentation faults.
    """
    from symfluence.core.hdf5_safety import apply_worker_environment
    apply_worker_environment()


class FileProcessor(ConfigMixin):
    """
    Manages parallel and serial processing of forcing files.

    Handles file filtering, output naming, and batch processing.
    """

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        logger: logging.Logger = None
    ):
        """
        Initialize file processor.

        Args:
            config: Configuration dictionary
            output_dir: Output directory for processed files
            logger: Optional logger instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.validator = FileValidator(self.logger)
        # Identity of the HRU set being remapped onto. The resampler sets it
        # before naming any output; without it names carry the method only.
        self.target_geometry = None

    def determine_output_filename(self, input_file: Path, token: Optional[str] = None) -> Path:
        """
        Determine the expected output filename for a given input file.

        Args:
            input_file: Input forcing file path
            token: Override the filename token. ``''`` gives the pre-#379
                untokened name; used to find files written under older names.

        Returns:
            Path: Expected output file path
        """
        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        forcing_dataset = self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')
        input_stem = input_file.stem

        # Namespace the remapped forcing by the HRU set it is remapped onto
        # ({method}-h{count}-{digest8}), so two discretizations or two
        # geometries of the SAME domain never share a name (issue #339).
        if token is None:
            from symfluence.core.modeling.forcing_naming import forcing_token
            disc = self._get_config_value(
                lambda: self.config.domain.discretization, dict_key='SUB_GRID_DISCRETIZATION')
            token = forcing_token(disc, self.target_geometry)
        infix = f"remapped_{token}_" if token else "remapped_"

        # Extract the date tag from the filename. Only the part AFTER the
        # domain name is searched, and the LAST digit-bounded date token wins:
        # a digit-bearing domain name (e.g. 'usgs_13201010') or dataset name
        # must never be read as the date, or every monthly file maps to the
        # same output name and the months overwrite each other.
        date_tag = self._extract_date_tag(input_stem, domain_name)

        if date_tag:
            output_filename = f"{domain_name}_{forcing_dataset}_{infix}{date_tag}.nc"
        else:
            # Fallback logic: prevent redundant prefixing
            clean_stem = input_stem
            if input_stem.startswith(f"domain_{domain_name}"):
                clean_stem = input_stem.replace(f"domain_{domain_name}_", "")
            elif input_stem.startswith(domain_name):
                clean_stem = input_stem.replace(f"{domain_name}_", "")

            clean_stem = clean_stem.replace(f"{forcing_dataset}_", "")
            clean_stem = clean_stem.replace(f"{forcing_dataset.lower()}_", "")
            clean_stem = clean_stem.replace("remapped_", "").replace("merged_", "")

            output_filename = f"{domain_name}_{forcing_dataset}_{infix}{clean_stem}.nc"

        return self.output_dir / output_filename

    _FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?!\d)")
    _COMPACT_DATE_RE = re.compile(
        r"(?<!\d)((?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])?)(?!\d)")

    @classmethod
    def _extract_date_tag(cls, input_stem: str, domain_name: Optional[str]) -> Optional[str]:
        """Return the ``YYYY-MM-DD-HH-MM-SS`` date tag encoded in *input_stem*.

        The domain name prefix is stripped before matching and the last
        digit-bounded token is used, so digits inside the domain or dataset
        name are never mistaken for the file's date.
        """
        stem = input_stem
        if domain_name:
            for prefix in (f"domain_{domain_name}_", f"{domain_name}_"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            else:
                idx = stem.rfind(str(domain_name))
                if idx >= 0:
                    stem = stem[idx + len(str(domain_name)):]

        full = cls._FULL_DATE_RE.findall(stem)
        if full:
            return full[-1]

        compact = cls._COMPACT_DATE_RE.findall(stem)
        if compact:
            date_str = compact[-1]
            try:
                if len(date_str) == 6:
                    return datetime.strptime(date_str, "%Y%m").strftime("%Y-%m-01-00-00-00")
                return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d-00-00-00")
            except ValueError:
                return None
        return None

    def filter_processed_files(self, forcing_files: List[Path]) -> List[Path]:
        """
        Filter out already processed and valid files.

        Args:
            forcing_files: List of forcing files to check

        Returns:
            List of files that need processing
        """
        remaining_files = []
        already_processed = 0
        corrupted_files = 0

        # Explicit force: FORCE_RUN_ALL_STEPS (set directly, or by the
        # `--force-rerun` workflow flag) reprocesses every file unconditionally.
        force = bool(self._get_config_value(
            lambda: self.config.system.force_run_all_steps,
            default=False, dict_key='FORCE_RUN_ALL_STEPS'))

        for file in forcing_files:
            output_file = self.determine_output_filename(file)

            if output_file.exists():
                is_valid = self.validator.validate(output_file)
                # A structurally-valid output can still be STALE: if the source
                # forcing now carries variables the cached output was built
                # without (e.g. the dataset/version changed and added wind or
                # radiation), the old remap must be regenerated. Reusing it
                # silently produces a forcing file missing variables the model
                # requires. See _output_is_current().
                is_current = (not force and is_valid
                              and self._output_is_current(file, output_file))

                if is_current:
                    self.logger.debug(f"Skipping already processed file: {file.name}")
                    already_processed += 1
                    continue
                else:
                    reason = ("forced" if force
                              else "corrupted" if not is_valid else "stale")
                    msg = f"Reprocessing {reason} output file {output_file.name}."
                    if reason == "forced":
                        self.logger.debug(msg)
                    else:
                        self.logger.warning(msg)
                    # The remapped forcing is a domain-shared, model-agnostic
                    # product: other workflows on this domain read it (directly
                    # and through data/model_ready/forcings). It must never be
                    # unlinked ahead of a rebuild — that leaves a window in
                    # which readers get FileNotFoundError or a half-written
                    # file. The applier republishes it with an atomic rename, so
                    # the old file stays complete and readable until the new one
                    # is in place.
                    corrupted_files += 1

            remaining_files.append(file)

        self.logger.debug(f"Found {already_processed} already processed files")
        if corrupted_files > 0:
            self.logger.info(f"Queued {corrupted_files} stale/corrupted files for atomic reprocessing")
        self.logger.debug(f"Found {len(remaining_files)} files that need processing")

        return remaining_files

    # Forcing variable names (CFIF + legacy SUMMA-style) that constitute the
    # remap payload. Used to detect a cached output that is stale relative to a
    # source whose variable set has since grown.
    _FORCING_VAR_NAMES = frozenset({
        'air_temperature', 'precipitation_flux', 'surface_air_pressure',
        'specific_humidity', 'wind_speed', 'relative_humidity',
        'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux',
        'eastward_wind', 'northward_wind',
        'airtemp', 'pptrate', 'airpres', 'spechum', 'windspd', 'relhum',
        'SWRadAtm', 'LWRadAtm',
    })

    def _output_is_current(self, source_file: Path, output_file: Path) -> bool:
        """Return False if the remapped output is stale w.r.t. its source.

        The output is stale when the source forcing file carries forcing
        variables that the (already-remapped) output lacks — the signature of a
        source dataset/version change (e.g. RDRS v2.1 -> CaSR v3.2 adding wind
        and radiation). Such an output must be reprocessed (republished in place
        by an atomic rename), otherwise the model receives a forcing file
        missing required variables.
        """
        import xarray as xr
        try:
            with xr.open_dataset(source_file) as src, \
                    xr.open_dataset(output_file) as out:
                src_forcing = self._FORCING_VAR_NAMES & set(src.data_vars)
                out_forcing = self._FORCING_VAR_NAMES & set(out.data_vars)
        except (OSError, ValueError, KeyError) as e:
            # If we cannot compare, err on the side of reprocessing.
            self.logger.warning(
                "Could not compare %s against its source (%s); will reprocess.",
                output_file.name, e, exc_info=True)
            return False
        missing = src_forcing - out_forcing
        if missing:
            self.logger.info(
                "Remapped output %s is stale: source has forcing variables %s "
                "absent from the cached output; reprocessing.",
                output_file.name, sorted(missing))
            return False
        return True

    def process_serial(
        self,
        files: List[Path],
        process_func: Callable[[Path], bool]
    ) -> int:
        """
        Process files in serial mode.

        Args:
            files: List of files to process
            process_func: Function to process each file (returns True on success)

        Returns:
            Number of successfully processed files
        """
        self.logger.info(f"Processing {len(files)} files in serial mode")

        success_count = 0

        # Note: tqdm monitor thread is disabled globally in configure_hdf5_safety()
        with tqdm(total=len(files), desc="Remapping forcing files", unit="file") as pbar:
            for file in files:
                try:
                    success = process_func(file)
                    if success:
                        success_count += 1
                    else:
                        self.logger.error(f"Failed to process {file.name}")
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)

                pbar.update(1)

        self.logger.info(f"Serial processing complete: {success_count}/{len(files)} successful")
        return success_count

    def process_parallel(
        self,
        files: List[Path],
        num_cpus: int,
        process_func: Callable[[Path, int], bool]
    ) -> int:
        """
        Process files in parallel mode.

        Args:
            files: List of files to process
            num_cpus: Number of CPUs to use
            process_func: Function to process each file (takes file and worker_id)

        Returns:
            Number of successfully processed files
        """
        self.logger.debug(f"Processing {len(files)} in parallel with {num_cpus} CPUs")

        batch_size = min(10, len(files))
        total_batches = (len(files) + batch_size - 1) // batch_size

        self.logger.debug(f"Processing {total_batches} batches of up to {batch_size} files each")

        success_count = 0

        # Note: tqdm monitor thread is disabled globally in configure_hdf5_safety()
        with tqdm(total=len(files), desc="Remapping forcing files", unit="file") as pbar:
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(files))
                batch_files = files[start_idx:end_idx]

                try:
                    # Use initializer to configure HDF5 safety in each worker
                    with mp.Pool(processes=num_cpus, initializer=_init_worker_pool) as pool:
                        worker_args = [
                            (file, i % num_cpus)
                            for i, file in enumerate(batch_files)
                        ]
                        results = pool.starmap(process_func, worker_args)

                    batch_success = sum(1 for r in results if r)
                    success_count += batch_success
                    pbar.update(len(batch_files))

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.error(f"Error processing batch {batch_num+1}: {str(e)}", exc_info=True)
                    pbar.update(len(batch_files))

                gc.collect()

        self.logger.debug(f"Parallel processing complete: {success_count}/{len(files)} successful")
        return success_count

    def get_forcing_files(
        self,
        forcing_path: Path,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Get list of forcing files, excluding non-temporal files.

        Filters by the configured forcing dataset when multiple datasets
        are present in the same directory (e.g., after switching from
        AORC to ERA5 without clearing old files).

        Args:
            forcing_path: Path to search for forcing files
            exclude_patterns: Patterns to exclude (default: attributes, metadata, etc.)

        Returns:
            Sorted list of forcing file paths
        """
        if exclude_patterns is None:
            exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params']

        all_nc_files = list(forcing_path.glob('*.nc'))
        forcing_files = sorted([
            f for f in all_nc_files
            if not any(pattern in f.name.lower() for pattern in exclude_patterns)
        ])

        # Filter by configured dataset to avoid processing stale files from
        # a previously configured dataset (e.g., AORC files when ERA5 is active)
        forcing_dataset = self._get_config_value(
            lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET'
        )
        if forcing_dataset:
            dataset_lower = forcing_dataset.lower()
            dataset_upper = forcing_dataset.upper()
            dataset_files = [
                f for f in forcing_files
                if dataset_lower in f.name.lower() or dataset_upper in f.name
            ]
            if dataset_files:
                other_count = len(forcing_files) - len(dataset_files)
                if other_count > 0:
                    self.logger.info(
                        f"Filtered to {len(dataset_files)} {dataset_upper} file(s), "
                        f"skipping {other_count} file(s) from other datasets"
                    )
                forcing_files = dataset_files

        excluded_count = len(all_nc_files) - len(forcing_files)
        if excluded_count > 0:
            excluded_files = [f.name for f in all_nc_files if f not in forcing_files]
            self.logger.debug(f"Excluded {excluded_count} non-forcing files: {excluded_files}")

        return forcing_files
