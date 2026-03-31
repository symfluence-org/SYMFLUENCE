# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RemappingWeights - EASYMORE weight generation and application for forcing remapping.

This module handles:
- One-time weight generation using EASYMORE
- Efficient weight application to multiple forcing files
- Serial and parallel processing modes

Extracted from ForcingResampler to improve testability and reduce coupling.
"""

import gc
import logging
import multiprocessing as mp
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import easymore
import netCDF4 as nc4
import xarray as xr
from tqdm import tqdm

from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir

from .shapefile_manager import ShapefileManager

# Suppress verbose easymore logging
logging.getLogger('easymore').setLevel(logging.WARNING)


def _init_worker_pool():
    """
    Initialize worker process for multiprocessing pool.

    This function is called once per worker process when the pool is created.
    It configures HDF5/netCDF4 thread safety to prevent segmentation faults.
    """
    from symfluence.core.hdf5_safety import apply_worker_environment
    apply_worker_environment()


def _create_easymore_instance():
    """Create an EASYMORE instance handling different module structures."""
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")


class RemappingWeightGenerator(ConfigMixin):
    """
    Generates remapping weights for forcing data using EASYMORE.

    The weight generation is an expensive GIS operation that only needs
    to be done once per source/target shapefile combination.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        project_dir: Path,
        shapefile_manager: ShapefileManager
    ):
        """
        Initialize RemappingWeightGenerator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Project directory path
            shapefile_manager: ShapefileManager instance for CRS handling
        """
        # Use centralized config coercion (handles dict -> SymfluenceConfig with fallback)
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.project_dir = project_dir
        self.shapefile_manager = shapefile_manager

    def create_weights(
        self,
        source_shapefile: Path,
        target_shapefile: Path,
        sample_forcing_file: Path,
        output_dir: Path,
        dataset_handler,
        hru_id_field: str
    ) -> Tuple[Path, List[str]]:
        """
        Create remapping weights file using EASYMORE.

        Args:
            source_shapefile: Path to source (forcing grid) shapefile in WGS84
            target_shapefile: Path to target (catchment) shapefile in WGS84
            sample_forcing_file: Sample NetCDF file to detect variables
            output_dir: Directory for intersection/weight files
            dataset_handler: Dataset handler for coordinate names
            hru_id_field: HRU ID field name in target shapefile

        Returns:
            Tuple of (remap_file_path, detected_variables)
        """
        case_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_{self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')}"
        remap_file = output_dir / f"{case_name}_{hru_id_field}_remapping.csv"

        # Check if weights already exist
        if remap_file.exists():
            intersect_csv = output_dir / f"{case_name}_intersected_shapefile.csv"
            intersect_shp = output_dir / f"{case_name}_intersected_shapefile.shp"
            if intersect_csv.exists() or intersect_shp.exists():
                self.logger.info(f"Remapping weights file already exists: {remap_file}")
                # Detect variables from sample file
                detected_vars = self._detect_forcing_variables(sample_forcing_file)
                return remap_file, detected_vars
            self.logger.info("Remapping weights found but intersection missing. Recreating.")

        self.logger.info("Creating remapping weights (this is done only once)...")

        temp_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'temp_easymore_weights'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Handle longitude alignment
            target_for_easymore, disable_lon_correction = self.shapefile_manager.align_longitude_frame(
                target_shapefile, source_shapefile, temp_dir
            )

            # Detect variables and grid info
            var_lat, var_lon = dataset_handler.get_coordinate_names()
            detected_vars, source_resolution = self._analyze_forcing_file(
                sample_forcing_file, var_lat, var_lon
            )

            # Configure EASYMORE
            esmr = _create_easymore_instance()

            esmr.author_name = 'SUMMA public workflow scripts'
            esmr.license = 'Copernicus data use license'
            esmr.case_name = case_name
            if disable_lon_correction:
                esmr.correction_shp_lon = False

            # Source shapefile config
            esmr.source_shp = str(source_shapefile)
            esmr.source_shp_lat = self._get_config_value(lambda: self.config.forcing.shape_lat_name, dict_key='FORCING_SHAPE_LAT_NAME')
            esmr.source_shp_lon = self._get_config_value(lambda: self.config.forcing.shape_lon_name, dict_key='FORCING_SHAPE_LON_NAME')
            esmr.source_shp_ID = self._get_config_value(lambda: self.config.forcing.shape_id_name, default='ID')

            # Target shapefile config
            esmr.target_shp = str(target_for_easymore)
            esmr.target_shp_ID = hru_id_field
            esmr.target_shp_lat = self._get_config_value(lambda: self.config.paths.catchment_lat, dict_key='CATCHMENT_SHP_LAT')
            esmr.target_shp_lon = self._get_config_value(lambda: self.config.paths.catchment_lon, dict_key='CATCHMENT_SHP_LON')

            # NetCDF config
            esmr.source_nc = str(sample_forcing_file)
            esmr.var_names = detected_vars
            esmr.var_lat = var_lat
            esmr.var_lon = var_lon
            esmr.var_time = 'time'

            if source_resolution is not None:
                esmr.source_nc_resolution = source_resolution

            # Output config
            esmr.temp_dir = str(temp_dir) + '/'
            esmr.output_dir = str(temp_dir) + '/'  # Use temp_dir for both to avoid mess
            esmr.remapped_dim_id = 'hru'
            esmr.remapped_var_id = 'hruId'
            esmr.format_list = ['f4']
            esmr.fill_value_list = ['-9999']

            # Only create weights, don't apply
            esmr.only_create_remap_csv = True
            if hasattr(esmr, 'only_create_remap_nc'):
                esmr.only_create_remap_nc = True

            esmr.save_csv = True
            esmr.sort_ID = False

            # Set numcpu=1 to avoid bus errors on macOS
            esmr.numcpu = 1

            # Enable temp shp saving
            esmr.save_temp_shp = True

            self.logger.info("Running EASYMORE to create remapping weights...")
            # Use same suppressed output runner as forcing_resampler if available
            # otherwise just run it
            try:
                from .forcing_resampler import _run_easmore_with_suppressed_output
                _run_easmore_with_suppressed_output(esmr, self.logger)
            except (ImportError, AttributeError):
                esmr.nc_remapper()

            # Move output files
            case_remap_csv = temp_dir / f"{case_name}_remapping.csv"
            case_remap_nc = temp_dir / f"{case_name}_remapping.nc"
            case_attr_nc = temp_dir / f"{case_name}_attributes.nc"

            # Support EASYMORE 2.0 conversion if needed
            if not case_remap_csv.exists() and case_remap_nc.exists():
                try:
                    import xarray as xr
                    with xr.open_dataset(case_remap_nc) as ds:
                        ds.to_dataframe().to_csv(case_remap_csv)
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Could not convert NetCDF weights to CSV: {e}")

            if case_remap_csv.exists():
                shutil.move(str(case_remap_csv), str(remap_file))
                self.logger.info(f"Remapping weights created: {remap_file}")

                # Also move NC versions
                remap_nc_final = remap_file.with_suffix('.nc')
                if case_remap_nc.exists():
                    shutil.move(str(case_remap_nc), str(remap_nc_final))
                if case_attr_nc.exists():
                    attr_nc_final = remap_file.parent / f"{case_name}_attributes.nc"
                    shutil.move(str(case_attr_nc), str(attr_nc_final))
            else:
                # Fallback to searching for CSV
                mapping_patterns = ["*remapping*.csv", "*_remapping.csv", "Mapping_*.csv"]
                fallback = []
                for pattern in mapping_patterns:
                    fallback.extend(list(temp_dir.glob(pattern)))

                if fallback:
                    shutil.move(str(fallback[0]), str(remap_file))
                    self.logger.info(f"Remapping weights created (fallback): {remap_file}")
                else:
                    raise FileNotFoundError(f"Expected remapping file not created in {temp_dir}")

            for shp_file in temp_dir.glob(f"{case_name}_intersected_shapefile.*"):
                shutil.move(str(shp_file), str(output_dir / shp_file.name))

            return remap_file, detected_vars

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _detect_forcing_variables(self, forcing_file: Path) -> List[str]:
        """Detect available forcing variables in a NetCDF file (CFIF or legacy names)."""
        all_cfif_vars = [
            'surface_air_pressure', 'surface_downwelling_longwave_flux',
            'surface_downwelling_shortwave_flux', 'precipitation_flux',
            'air_temperature', 'specific_humidity', 'wind_speed', 'relative_humidity',
        ]
        all_legacy_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd', 'relhum']

        available = []
        try:
            with nc4.Dataset(forcing_file, 'r') as ncid:
                available = [v for v in all_cfif_vars if v in ncid.variables]
                if not available:
                    available = [v for v in all_legacy_vars if v in ncid.variables]
        finally:
            gc.collect()

        if not available:
            raise ValueError(f"No SUMMA forcing variables found in {forcing_file}")

        self.logger.info(f"Detected {len(available)}/{len(all_cfif_vars)} forcing variables: {available}")
        return available

    def _analyze_forcing_file(
        self,
        forcing_file: Path,
        var_lat: str,
        var_lon: str
    ) -> Tuple[List[str], Optional[float]]:
        """
        Analyze forcing file for variables and grid resolution.

        Returns:
            Tuple of (detected_variables, source_resolution_or_None)
        """
        detected_vars = self._detect_forcing_variables(forcing_file)

        source_resolution = None
        try:
            with nc4.Dataset(forcing_file, 'r') as ncid:
                if var_lat not in ncid.variables or var_lon not in ncid.variables:
                    return detected_vars, None

                lat_vals = ncid.variables[var_lat][:]
                lon_vals = ncid.variables[var_lon][:]

                # Determine grid size
                if lat_vals.ndim == 1:
                    lat_size, lon_size = len(lat_vals), len(lon_vals)
                elif lat_vals.ndim == 2:
                    lat_size, lon_size = lat_vals.shape
                else:
                    lat_size, lon_size = 1, 1

                # Calculate resolution for small grids
                if lat_size == 1 or lon_size == 1:
                    if lat_vals.ndim == 1:
                        res_lat = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.25
                        res_lon = abs(float(lon_vals[1] - lon_vals[0])) if len(lon_vals) > 1 else 0.25
                    else:
                        res_lat, res_lon = 0.25, 0.25

                    source_resolution = max(res_lat, res_lon)
                    self.logger.info(
                        f"Small grid detected ({lat_size}x{lon_size}), "
                        f"setting source_nc_resolution={source_resolution}"
                    )
        finally:
            gc.collect()

        return detected_vars, source_resolution


class RemappingWeightApplier(ConfigMixin):
    """
    Applies pre-computed remapping weights to forcing files.

    This is the fast operation that reads weights and applies them
    to each forcing file.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        project_dir: Path,
        output_dir: Path,
        dataset_handler
    ):
        """
        Initialize RemappingWeightApplier.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Project directory path
            output_dir: Directory for remapped output files
            dataset_handler: Dataset handler for coordinate names
        """
        self.config = config
        self.logger = logger
        self.project_dir = project_dir
        self.output_dir = output_dir
        self.dataset_handler = dataset_handler
        self._detected_vars: Optional[List[str]] = None

    def set_detected_variables(self, variables: List[str]) -> None:
        """Set the detected forcing variables to use."""
        self._detected_vars = variables

    def apply_weights(
        self,
        forcing_file: Path,
        remap_file: Path,
        worker_id: Optional[int] = None
    ) -> bool:
        """
        Apply pre-computed remapping weights to a single forcing file.

        Args:
            forcing_file: Path to forcing file to process
            remap_file: Path to pre-computed remapping weights CSV
            worker_id: Optional worker ID for logging

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        worker_str = f"Worker {worker_id}: " if worker_id is not None else ""

        try:
            output_file = self.determine_output_filename(forcing_file)

            # Check if already processed
            if output_file.exists() and output_file.stat().st_size > 1000:
                self.logger.debug(f"{worker_str}Output already exists: {forcing_file.name}")
                return True

            # Create unique temp directory
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = resolve_data_subdir(self.project_dir, 'forcing') / f'temp_apply_{unique_id}'
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                esmr = _create_easymore_instance()

                esmr.author_name = 'SUMMA public workflow scripts'
                esmr.case_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_{self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')}"

                var_lat, var_lon = self.dataset_handler.get_coordinate_names()

                esmr.source_nc = str(forcing_file)
                esmr.var_names = self._get_variables(forcing_file)
                esmr.var_lat = var_lat
                esmr.var_lon = var_lon
                esmr.var_time = 'time'

                esmr.temp_dir = str(temp_dir) + '/'
                esmr.output_dir = str(self.output_dir) + '/'

                esmr.remapped_dim_id = 'hru'
                esmr.remapped_var_id = 'hruId'
                esmr.format_list = ['f4']
                esmr.fill_value_list = ['-9999']

                esmr.remap_csv = str(remap_file)
                esmr.save_csv = False
                esmr.sort_ID = False

                self.logger.debug(f"{worker_str}Applying remapping weights to {forcing_file.name}")
                esmr.nc_remapper()

            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

            # Verify output
            if output_file.exists() and output_file.stat().st_size > 1000:
                elapsed = time.time() - start_time
                self.logger.debug(
                    f"{worker_str}Successfully processed {forcing_file.name} in {elapsed:.2f}s"
                )
                return True

            self.logger.error(f"{worker_str}Output file not created: {output_file}")
            return False

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"{worker_str}Error processing {forcing_file.name}: {str(e)}")
            return False

    def _get_variables(self, forcing_file: Path) -> List[str]:
        """Get variables to use, either pre-detected or from file."""
        if self._detected_vars:
            return self._detected_vars

        all_cfif_vars = [
            'surface_air_pressure', 'surface_downwelling_longwave_flux',
            'surface_downwelling_shortwave_flux', 'precipitation_flux',
            'air_temperature', 'specific_humidity', 'wind_speed', 'relative_humidity',
        ]
        all_legacy_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd', 'relhum']
        with xr.open_dataset(forcing_file) as ds:
            result = [v for v in all_cfif_vars if v in ds.data_vars]
            return result if result else [v for v in all_legacy_vars if v in ds.data_vars]

    def determine_output_filename(self, input_file: Path) -> Path:
        """
        Determine the expected output filename for a given input file.

        Args:
            input_file: Input forcing file path

        Returns:
            Expected output file path
        """
        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        forcing_dataset = self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')
        input_stem = input_file.stem

        if forcing_dataset.lower() in ('rdrs', 'casr'):
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"

        elif forcing_dataset.lower() in ('carra', 'cerra'):
            start_str = self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')
            try:
                dt_start = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
                time_tag = dt_start.strftime("%Y-%m-%d-%H-%M-%S")
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{time_tag}.nc"
            except (ValueError, TypeError):
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"

        elif forcing_dataset.lower() == 'era5':
            date_match = re.search(r"(\d{8})$", input_stem) or re.search(r"(\d{6})$", input_stem)
            if date_match:
                date_str = date_match.group(1)
                if len(date_str) == 6:
                    dt = datetime.strptime(date_str, "%Y%m")
                    time_tag = dt.strftime("%Y-%m-01-00-00-00")
                else:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    time_tag = dt.strftime("%Y-%m-%d-00-00-00")
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{time_tag}.nc"
            else:
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"

        else:
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"

        return self.output_dir / output_filename


class BatchProcessor(ConfigMixin):
    """
    Handles serial and parallel batch processing of forcing files.
    """

    def __init__(
        self,
        applier: RemappingWeightApplier,
        logger: logging.Logger
    ):
        """
        Initialize BatchProcessor.

        Args:
            applier: RemappingWeightApplier instance
            logger: Logger instance
        """
        self.applier = applier
        self.logger = logger

    def filter_unprocessed(self, files: List[Path]) -> List[Path]:
        """Filter out already processed files."""
        remaining = []
        already_processed = 0

        for file in files:
            output_file = self.applier.determine_output_filename(file)

            if output_file.exists():
                try:
                    if output_file.stat().st_size > 1000:
                        self.logger.debug(f"Skipping already processed: {file.name}")
                        already_processed += 1
                        continue
                except OSError:
                    pass  # File may have been removed or is inaccessible

            remaining.append(file)

        self.logger.info(f"Found {already_processed} already processed files")
        self.logger.info(f"Found {len(remaining)} files that need processing")

        return remaining

    def process_serial(self, files: List[Path], remap_file: Path) -> int:
        """
        Process files serially.

        Args:
            files: List of forcing files to process
            remap_file: Path to remapping weights file

        Returns:
            Number of successfully processed files
        """
        self.logger.info(f"Processing {len(files)} files in serial mode")

        success_count = 0

        # Note: tqdm monitor thread is disabled globally in configure_hdf5_safety()
        with tqdm(total=len(files), desc="Remapping forcing files", unit="file") as pbar:
            for file in files:
                try:
                    if self.applier.apply_weights(file, remap_file):
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
        remap_file: Path,
        num_cpus: int
    ) -> int:
        """
        Process files in parallel.

        Args:
            files: List of forcing files to process
            remap_file: Path to remapping weights file
            num_cpus: Number of CPUs to use

        Returns:
            Number of successfully processed files
        """
        self.logger.info(f"Processing {len(files)} files with {num_cpus} CPUs")

        batch_size = min(10, len(files))
        total_batches = (len(files) + batch_size - 1) // batch_size

        self.logger.info(f"Processing {total_batches} batches of up to {batch_size} files")

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
                            (file, remap_file, i % num_cpus)
                            for i, file in enumerate(batch_files)
                        ]
                        results = pool.starmap(self._worker_wrapper, worker_args)

                    batch_success = sum(1 for r in results if r)
                    success_count += batch_success
                    pbar.update(len(batch_files))

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.error(f"Error processing batch {batch_num+1}: {str(e)}")
                    pbar.update(len(batch_files))

                import gc
                gc.collect()

        self.logger.info(f"Parallel processing complete: {success_count}/{len(files)} successful")
        return success_count

    def _worker_wrapper(self, file: Path, remap_file: Path, worker_id: int) -> bool:
        """Worker function for parallel processing."""
        try:
            return self.applier.apply_weights(file, remap_file, worker_id)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Worker {worker_id}: Error processing {file.name}: {str(e)}")
            return False
