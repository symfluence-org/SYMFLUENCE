# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Chunked Download Mixin for Data Acquisition Handlers.

Provides utilities for temporal chunking, parallel downloads,
and NetCDF file merging.
"""

import concurrent.futures
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeVar

import pandas as pd
import xarray as xr

T = TypeVar('T')


class ChunkedDownloadMixin:
    """
    Mixin for chunked/batched download operations.

    Provides methods for:
    - Generating temporal chunks from date ranges
    - Parallel chunk downloads with ThreadPoolExecutor
    - Merging NetCDF files along time dimension

    Expects the class to have a `logger` attribute.
    """

    def generate_temporal_chunks(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        freq: str = 'MS'
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate temporal chunks from a date range.

        Args:
            start_date: Start of the date range
            end_date: End of the date range
            freq: Pandas frequency string for chunking:
                  'MS' = month start (default)
                  'YS' = year start
                  'D' = daily
                  'W' = weekly

        Returns:
            List of (chunk_start, chunk_end) tuples

        Example:
            >>> chunks = self.generate_temporal_chunks(
            ...     pd.Timestamp('2020-01-15'),
            ...     pd.Timestamp('2020-03-20'),
            ...     freq='MS'
            ... )
            >>> # Returns: [(2020-01-15, 2020-01-31), (2020-02-01, 2020-02-29), (2020-03-01, 2020-03-20)]
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        if start_date > end_date:
            return []

        # Generate period starts
        period_starts = pd.date_range(
            start=start_date.to_period(freq[0]).start_time,
            end=end_date,
            freq=freq
        )

        if len(period_starts) == 0:
            # Single period case
            return [(start_date, end_date)]

        chunks = []
        for i, period_start in enumerate(period_starts):
            # Determine chunk boundaries
            chunk_start = max(start_date, period_start)

            if i < len(period_starts) - 1:
                # Not the last period - end at day before next period
                chunk_end = period_starts[i + 1] - pd.Timedelta(days=1)
            else:
                # Last period - use end date
                chunk_end = end_date

            # Ensure chunk_end doesn't exceed end_date
            chunk_end = min(chunk_end, end_date)

            # Only add valid chunks
            if chunk_start <= chunk_end:
                chunks.append((chunk_start, chunk_end))

        return chunks

    def generate_year_month_list(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[Tuple[int, int]]:
        """
        Generate list of (year, month) tuples covering a date range.

        Args:
            start_date: Start of the date range
            end_date: End of the date range

        Returns:
            List of (year, month) tuples

        Example:
            >>> ym_list = self.generate_year_month_list(
            ...     pd.Timestamp('2020-11-01'),
            ...     pd.Timestamp('2021-02-28')
            ... )
            >>> # Returns: [(2020, 11), (2020, 12), (2021, 1), (2021, 2)]
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        dates = pd.date_range(start_date, end_date, freq='MS')

        if dates.empty:
            # Single month case
            return [(start_date.year, start_date.month)]

        ym_list = [(d.year, d.month) for d in dates]

        # Ensure end month is included
        end_ym = (end_date.year, end_date.month)
        if end_ym not in ym_list:
            ym_list.append(end_ym)

        return ym_list

    def download_chunks_parallel(
        self,
        chunks: List[T],
        download_func: Callable[[T], Optional[Path]],
        max_workers: int = 2,
        desc: str = "Downloading",
        fail_fast: bool = True
    ) -> List[Path]:
        """
        Download chunks in parallel using ThreadPoolExecutor.

        Args:
            chunks: List of chunk specifications (e.g., date ranges, URLs)
            download_func: Function that takes a chunk and returns Path to downloaded file
            max_workers: Maximum parallel workers (default: 2, conservative for rate limits)
            desc: Description for logging
            fail_fast: If True, cancel remaining on first error (default: True)

        Returns:
            List of Paths to successfully downloaded files

        Raises:
            Exception from download_func if fail_fast is True and any download fails

        Example:
            >>> def download_month(year_month):
            ...     year, month = year_month
            ...     # Download and return path
            ...     return output_path
            >>> files = self.download_chunks_parallel(
            ...     [(2020, 1), (2020, 2), (2020, 3)],
            ...     download_month,
            ...     max_workers=2
            ... )
        """
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        chunk_files = []
        total = len(chunks)

        logger.info(f"{desc}: {total} chunks with {max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(download_func, chunk): chunk
                for chunk in chunks
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_exc = future.exception()
                except concurrent.futures.CancelledError as exc:
                    logger.error(f"Chunk {chunk} was cancelled: {exc}")
                    if fail_fast:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    continue

                if chunk_exc is not None:
                    logger.error(f"Chunk {chunk} failed: {chunk_exc}")
                    if fail_fast:
                        # Cancel remaining futures
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise chunk_exc
                    continue

                result = future.result()
                if result is not None:
                    chunk_files.append(result)
                    logger.debug(f"Completed chunk: {chunk}")

        logger.info(f"Downloaded {len(chunk_files)}/{total} chunks successfully")
        return chunk_files

    def merge_netcdf_chunks(
        self,
        chunk_files: List[Path],
        output_file: Path,
        time_slice: Tuple[pd.Timestamp, pd.Timestamp] = None,
        concat_dim: str = 'time',
        combine: str = 'by_coords',
        cleanup: bool = True,
        encoding: dict = None
    ) -> Path:
        """
        Merge multiple NetCDF files along a dimension.

        Args:
            chunk_files: List of paths to NetCDF files to merge
            output_file: Path for the merged output file
            time_slice: Optional (start, end) to subset after merging
            concat_dim: Dimension to concatenate along (default: 'time')
            combine: How to combine files - 'by_coords' or 'nested' (default: 'by_coords')
            cleanup: If True, delete chunk files after successful merge (default: True)
            encoding: Optional encoding dict for output file

        Returns:
            Path to the merged output file

        Example:
            >>> merged = self.merge_netcdf_chunks(
            ...     [Path("chunk1.nc"), Path("chunk2.nc")],
            ...     Path("merged.nc"),
            ...     time_slice=(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')),
            ...     cleanup=True
            ... )
        """
        logger = getattr(self, 'logger', logging.getLogger(__name__))

        if not chunk_files:
            raise ValueError("No chunk files provided for merging")

        # Sort files for consistent ordering
        chunk_files = sorted(chunk_files)
        logger.info(f"Merging {len(chunk_files)} NetCDF files to {output_file}")

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load each chunk fully into memory and close file handles immediately.
        # Avoids dask scheduler deadlocks AND HDF5 C-library hangs that occur
        # when open_mfdataset keeps file handles alive inside forked processes
        # (pytest-xdist workers, MPI ranks).
        datasets = []
        for f in chunk_files:
            datasets.append(xr.load_dataset(f))

        if len(datasets) == 1:
            ds_merged = datasets[0]
        else:
            ds_merged = xr.concat(
                datasets,
                dim=concat_dim,
                data_vars='minimal',
                coords='minimal',
                compat='override'
            )

        # Apply time slice if specified
        if time_slice is not None:
            start, end = time_slice
            ds_merged = ds_merged.sel(time=slice(start, end))

        # Write output
        if encoding:
            ds_merged.to_netcdf(output_file, encoding=encoding)
        else:
            ds_merged.to_netcdf(output_file)

        # Cleanup chunk files
        if cleanup:
            cleanup_count = 0
            for chunk_file in chunk_files:
                if chunk_file.exists():
                    try:
                        chunk_file.unlink()
                        cleanup_count += 1
                    except OSError as e:
                        logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")
            logger.debug(f"Cleaned up {cleanup_count} chunk files")

        logger.info(f"Merged output saved to {output_file}")
        return output_file

    def get_netcdf_encoding(
        self,
        ds: xr.Dataset,
        compression: bool = True,
        complevel: int = 1,
        chunk_time: Optional[int] = None
    ) -> dict:
        """
        Generate standard NetCDF encoding for a dataset.

        Args:
            ds: xarray Dataset to generate encoding for
            compression: Whether to enable zlib compression (default: True)
            complevel: Compression level 1-9 (default: 1 for speed)
            chunk_time: Optional time chunk size (default: min(168, time_dim_size))

        Returns:
            Encoding dictionary for ds.to_netcdf(encoding=...)
        """
        encoding = {}

        for var in ds.data_vars:
            var_encoding = {}

            if compression:
                var_encoding['zlib'] = True
                var_encoding['complevel'] = complevel

            # Determine chunksizes if variable has expected dims
            dims = ds[var].dims
            if 'time' in dims and chunk_time is None:
                time_size = ds.sizes.get('time', 1)
                chunk_time = min(168, time_size)  # 1 week of hourly data

            if chunk_time and 'time' in dims:
                chunksizes = []
                for dim in dims:
                    if dim == 'time':
                        chunksizes.append(chunk_time)
                    else:
                        chunksizes.append(ds.sizes[dim])
                var_encoding['chunksizes'] = tuple(chunksizes)

            if var_encoding:
                encoding[var] = var_encoding

        return encoding


__all__ = ['ChunkedDownloadMixin']
