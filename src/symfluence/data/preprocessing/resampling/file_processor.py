# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
File Processor

Handles parallel and serial processing of forcing files.
"""

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

    def determine_output_filename(self, input_file: Path) -> Path:
        """
        Determine the expected output filename for a given input file.

        Args:
            input_file: Input forcing file path

        Returns:
            Path: Expected output file path
        """
        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        forcing_dataset = self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')
        input_stem = input_file.stem

        # Try to extract a date from the filename
        date_tag = None

        # Pattern 1: YYYY-MM-DD-HH-MM-SS
        match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", input_stem)
        if match:
            date_tag = match.group(1)
        else:
            # Pattern 2: YYYYMMDD or YYYYMM
            match = re.search(r"(19|20)(\d{4,6})", input_stem)
            if match:
                date_str = match.group(0)
                try:
                    if len(date_str) == 6:
                        dt = datetime.strptime(date_str, "%Y%m")
                        date_tag = dt.strftime("%Y-%m-01-00-00-00")
                    elif len(date_str) == 8:
                        dt = datetime.strptime(date_str, "%Y%m%d")
                        date_tag = dt.strftime("%Y-%m-%d-00-00-00")
                except ValueError:
                    pass

        if date_tag:
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{date_tag}.nc"
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

            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{clean_stem}.nc"

        return self.output_dir / output_filename

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

        for file in forcing_files:
            output_file = self.determine_output_filename(file)

            if output_file.exists():
                is_valid = self.validator.validate(output_file)

                if is_valid:
                    self.logger.debug(f"Skipping already processed file: {file.name}")
                    already_processed += 1
                    continue
                else:
                    self.logger.warning(
                        f"Found corrupted output file {output_file}. Will reprocess."
                    )
                    try:
                        output_file.unlink()
                        corrupted_files += 1
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.warning(f"Error deleting corrupted file: {e}")

            remaining_files.append(file)

        self.logger.debug(f"Found {already_processed} already processed files")
        if corrupted_files > 0:
            self.logger.info(f"Deleted {corrupted_files} corrupted files to reprocess")
        self.logger.debug(f"Found {len(remaining_files)} files that need processing")

        return remaining_files

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
                    self.logger.error(f"Error processing {file.name}: {str(e)}")

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
                    self.logger.error(f"Error processing batch {batch_num+1}: {str(e)}")
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
