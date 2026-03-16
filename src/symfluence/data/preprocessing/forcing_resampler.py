# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Forcing data resampling orchestrator for catchment-based remapping.

Orchestrates efficient remapping of gridded forcing data (ERA5, AORC, etc.) from source
grids to model catchments/HRUs using EASMORE (EArth Similarity Mapping and OverRemapping).
Implements Facade Pattern delegating to specialized components: weight generation, weight
application, file processing, and elevation correction. Supports parallel processing,
memory-efficient file handling, and flexible discretization strategies (lumped, distributed,
point-scale).

Architecture:
    The ForcingResampler coordinates a multi-step workflow:

    1. Remapping Weight Generation (One-time, expensive):
       - Compute spatial intersection weights between forcing grid and model HRUs
       - Uses EASMORE for efficient weight computation
       - Weights: [n_forcing_points × n_hrus] sparse matrix
       - Saved for reuse across all forcing files

    2. Weight Application (Per-file, fast):
       - Load remapping weights
       - Apply to each forcing file (ERA5, AORC, etc.)
       - Output: Catchment/HRU-averaged forcing time series
       - Format: NetCDF with dimensions [time, HRU]

    3. File Processing (Parallel/Serial):
       - Process multiple forcing files in parallel or serial
       - Configurable batch processing
       - Error handling and progress tracking

    4. Elevation Correction:
       - Compute mean DEM elevation per catchment/HRU
       - Apply elevation lapse rates to temperature
       - Adjust precipitation with altitude

    5. CRS Handling:
       - Validate and reproject shapefiles if needed
       - Maintain consistency across all operations

Workflow:

    1. Initialization:
       resampler = ForcingResampler(config, logger)
       - Load config, paths, catchment shapefiles
       - Initialize dataset-specific handlers
       - Determine discretization method (lumped/distributed/point)

    2. Resample Forcing (Main Entry Point):
       resampler.resample_forcing()
       - Check if weights already exist (skip generation if yes)
       - Generate remapping weights if needed (expensive, one-time)
       - Apply weights to forcing files (fast, per-file)
       - Compute elevation statistics
       - Output: Catchment-averaged forcing NetCDF files

    3. Alternative Workflows:
       For point-scale domains:
           resampler.extract_point_scale_forcing()
           - Extract forcing at single point/small grid
           - No remapping needed

Component Delegation:

    RemappingWeightGenerator:
        - Input: Source grid shapefile, target HRU shapefile
        - Output: Remapping weight matrix (sparse, [n_forcing_points × n_hrus])
        - Algorithm: EASMORE similarity mapping

    RemappingWeightApplier:
        - Input: Forcing file (NetCDF), remapping weights
        - Output: Remapped forcing file ([time, HRU])
        - Operation: Matrix multiplication (weights × forcing values)

    FileProcessor:
        - Input: List of forcing files, worker function
        - Output: Results from applying worker to each file
        - Features: Parallel/serial execution, progress tracking, error handling

    PointScaleForcingExtractor:
        - Input: Domain bounding box, forcing file
        - Output: Point-scale forcing at domain center
        - Use case: FLUXNET, weather station, small-area studies

    ElevationCalculator:
        - Input: DEM raster, HRU shapefile
        - Output: Mean elevation per HRU
        - Use case: Temperature lapse rate correction

    ShapefileProcessor:
        - Input: Shapefile, optional target CRS
        - Output: Validated/reprojected shapefile
        - Features: CRS conversion, HRU ID assignment

Data Flow:

    Forcing File → RemappingWeightApplier → Remapped Forcing (NetCDF)
                  ↑
                RemappingWeightGenerator (one-time)
                  ↑
                Source Grid + Target HRUs

Configuration Parameters:

    domain.discretization: str
        Discretization method: 'lumped', 'distributed', 'point_scale'
        Determines HRU definition and remapping target

    forcing_dataset: str
        Forcing data source: 'ERA5', 'AORC', 'CONUS404', etc.
        Controls dataset-specific handling

    paths.dem_name: str (optional)
        DEM raster filename for elevation correction
        Default: domain_{domain_name}_elv.tif

    paths.catchment_shp_name: str (optional)
        Catchment shapefile name
        Default: auto-generated from domain_name and discretization

    Remapping parameters (in config):
        weight_generation_method: Method for computing weights (EASMORE)
        parallel_processing: Enable/disable parallel file processing

Input Paths:

    shapefiles/catching/: Catchment/HRU polygon shapefiles
    attributes/elevation/dem/: DEM raster for elevation correction
    forcing/raw_data/: Input forcing files (ERA5 NetCDF, etc.)

Output Paths:

    forcing/basin_averaged_data/: Remapped forcing (NetCDF files)
    shapefiles/forcing/: Processed shapefiles with weights

Supported Discretization Methods:

    1. Lumped (Single HRU):
       - One HRU per catchment
       - Remapping: Average all grid cells to single catchment value
       - Output: Time series [time] for single catchment

    2. Distributed (Multiple HRUs):
       - Multiple HRUs per catchment (elevation bands, landcover types)
       - Remapping: Average grid cells to each HRU based on weights
       - Output: Time series [time, n_hrus] with separate HRU values

    3. Point-Scale (Single Point):
       - Single point or small study area
       - Remapping: Extract nearest grid cell or interpolate
       - Output: Time series [time] at point location

Error Handling:

    - Missing weights: Regenerate if weight file corrupted/missing
    - CRS mismatch: Warn and reproject if necessary
    - Temporal alignment: Validate time dimension across files
    - Parallel failures: Fall back to serial processing

Performance Considerations:

    - Weight generation: ~1-10 hours for large domains (millions of HRUs)
    - Weight application: ~1-10 seconds per forcing file
    - Parallel processing: 4-8 cores recommended
    - Memory: ~1-5 GB for weight matrix (depends on discretization)

References:

    - EASMORE Algorithm: Vergopolan et al. (similar approach)
    - Remapping Concepts: https://confluence.ecmwf.int/
    - NetCDF I/O: https://unidata.github.io/netcdf4-python/

See Also:

    - RemappingWeightGenerator: Low-level weight computation
    - RemappingWeightApplier: Low-level weight application
    - FileProcessor: Batch file processing
    - DataManager: High-level data workflow coordination
"""

import logging
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd

from symfluence.core.path_resolver import PathResolverMixin
from symfluence.data.preprocessing.dataset_handlers import DatasetRegistry
from symfluence.data.preprocessing.dataset_handlers.base_dataset import BaseDatasetHandler

from .resampling import (
    ElevationCalculator,
    FileProcessor,
    PointScaleForcingExtractor,
    RemappingWeightApplier,
    RemappingWeightGenerator,
    ShapefileProcessor,
)


class _ParallelWorker:
    """Picklable worker class for parallel file processing.

    This class can be pickled because it only stores simple types (paths as strings,
    config dict) rather than complex objects like loggers or dataset handlers.
    """

    def __init__(
        self,
        config_dict: dict,
        project_dir: str,
        output_dir: str,
        forcing_dataset: str,
        remap_file: str,
        cached_target_shp_wgs84: str,
        cached_hru_field: str,
        domain_name: str,
    ):
        self.config_dict = config_dict
        self.project_dir = Path(project_dir)
        self.output_dir = Path(output_dir)
        self.forcing_dataset = forcing_dataset
        self.remap_file = Path(remap_file)
        self.cached_target_shp_wgs84 = Path(cached_target_shp_wgs84)
        self.cached_hru_field = cached_hru_field
        self.domain_name = domain_name

        # These will be initialized per-worker
        self._weight_applier = None
        self._file_processor = None
        self._logger = None

    def _init_worker_components(self):
        """Initialize components in the worker process."""
        if self._weight_applier is not None:
            return

        # CRITICAL: Apply HDF5/netCDF4 safety settings in worker process
        # Worker processes are fresh Python interpreters and don't inherit the
        # environment setup from the main process configure_hdf5_safety() call.
        from symfluence.core.hdf5_safety import apply_worker_environment
        apply_worker_environment()

        # Create a minimal logger for the worker
        self._logger = logging.getLogger(f"forcing_worker_{mp.current_process().pid}")
        self._logger.setLevel(logging.WARNING)  # Reduce noise in parallel workers

        # Initialize dataset handler
        from symfluence.data.preprocessing.dataset_handlers import DatasetRegistry
        dataset_handler = DatasetRegistry.get_handler(
            self.forcing_dataset,
            self.config_dict,
            self._logger,
            self.project_dir
        )

        # Initialize weight applier
        self._weight_applier = RemappingWeightApplier(
            self.config_dict,
            self.project_dir,
            self.output_dir,
            dataset_handler,
            self._logger
        )
        self._weight_applier.set_shapefile_cache(
            self.cached_target_shp_wgs84,
            self.cached_hru_field
        )

        # Initialize file processor
        self._file_processor = FileProcessor(
            self.config_dict,
            self.output_dir,
            self._logger
        )

    def __call__(self, args: Tuple[str, int]) -> bool:
        """Process a single file. Called by pool.map()."""
        file_path, worker_id = args
        file = Path(file_path)

        self._init_worker_components()

        output_file = self._file_processor.determine_output_filename(file)
        return self._weight_applier.apply_weights(file, self.remap_file, output_file, worker_id)

# Suppress verbose easmore logging
logging.getLogger('easymore').setLevel(logging.WARNING)
logging.getLogger('easymorepy').setLevel(logging.WARNING)

warnings.filterwarnings('ignore', category=DeprecationWarning, module='easymore')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='easymore')


class ForcingResampler(PathResolverMixin):
    """
    Orchestrates forcing data remapping using EASYMORE.

    Delegates specialized operations to:
    - RemappingWeightGenerator: Creates intersection weights (expensive, one-time)
    - RemappingWeightApplier: Applies weights to forcing files (fast, per-file)
    - FileProcessor: Handles parallel/serial file processing
    - PointScaleForcingExtractor: Simplified extraction for small grids
    - ElevationCalculator: DEM-based elevation statistics
    - ShapefileProcessor: CRS conversion and HRU ID handling
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'

        dem_name = self._get_config_value(lambda: self.config.paths.dem_name)
        if dem_name == "default":
            dem_name = f"domain_{self._get_config_value(lambda: self.config.domain.name)}_elv.tif"

        self.dem_path = self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.forcing_basin_path = self.project_forcing_dir / 'basin_averaged_data'
        # Use backward-compatible catchment path resolution
        self.catchment_name = self._get_config_value(lambda: self.config.paths.catchment_shp_name)
        if self.catchment_name == 'default' or self.catchment_name is None:
            self.catchment_name = f"{self._get_config_value(lambda: self.config.domain.name)}_HRUs_{str(self._get_config_value(lambda: self.config.domain.discretization)).replace(',','_')}.shp"
        # Note: catchment_path will be resolved dynamically using _get_catchment_file_path()
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/raw_data')

        # Initialize dataset-specific handler
        try:
            self.dataset_handler = DatasetRegistry.get_handler(
                self.forcing_dataset,
                self.config,
                self.logger,
                self.project_dir
            )
            self.logger.debug(f"Initialized {self.forcing_dataset.upper()} dataset handler")
        except ValueError as e:
            self.logger.error(f"Failed to initialize dataset handler: {str(e)}")
            raise

        # Merge forcings if required by dataset
        if self.dataset_handler.needs_merging():
            self.logger.debug(f"{self.forcing_dataset.upper()} requires merging of raw files")
            self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_path')
            self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
            self.merge_forcings()

        # Lazy-initialized components
        self._elevation_calculator = None
        self._file_processor = None
        self._weight_generator = None
        self._weight_applier = None
        self._point_scale_extractor = None
        self._shapefile_processor = None

    # Lazy initialization properties — each component is created on first
    # access so that unused code paths incur no import/construction cost.

    @property
    def elevation_calculator(self) -> ElevationCalculator:
        """DEM-based mean-elevation calculator, created on first access."""
        if self._elevation_calculator is None:
            self._elevation_calculator = ElevationCalculator(self.logger)
        return self._elevation_calculator

    @property
    def file_processor(self) -> FileProcessor:
        """Parallel/serial file batch processor, created on first access."""
        if self._file_processor is None:
            self._file_processor = FileProcessor(
                self.config,
                self.forcing_basin_path,
                self.logger
            )
        return self._file_processor

    @property
    def weight_generator(self) -> RemappingWeightGenerator:
        """EASMORE remapping-weight generator, created on first access."""
        if self._weight_generator is None:
            self._weight_generator = RemappingWeightGenerator(
                self.config,
                self.project_dir,
                self.dataset_handler,
                self.logger
            )
        return self._weight_generator

    @property
    def weight_applier(self) -> RemappingWeightApplier:
        """Remapping-weight applier for per-file forcing regridding, created on first access."""
        if self._weight_applier is None:
            self._weight_applier = RemappingWeightApplier(
                self.config,
                self.project_dir,
                self.forcing_basin_path,
                self.dataset_handler,
                self.logger
            )
        return self._weight_applier

    @property
    def point_scale_extractor(self) -> PointScaleForcingExtractor:
        """Simplified point-scale forcing extractor for small/single-cell grids, created on first access."""
        if self._point_scale_extractor is None:
            self._point_scale_extractor = PointScaleForcingExtractor(
                self.config,
                self.project_dir,
                self.dataset_handler,
                self.logger
            )
        return self._point_scale_extractor

    @property
    def shapefile_processor(self) -> ShapefileProcessor:
        """CRS validation and HRU-ID assignment processor, created on first access."""
        if self._shapefile_processor is None:
            self._shapefile_processor = ShapefileProcessor(self.config, self.logger)
        return self._shapefile_processor

    def _filter_forcing_files_by_period(self, forcing_files: List[Path]) -> List[Path]:
        """Filter forcing files to only those overlapping the configured time period."""
        start_year = int(self._get_config_value(lambda: self.config.domain.time_start).split('-')[0])
        end_year = int(self._get_config_value(lambda: self.config.domain.time_end).split('-')[0])

        filtered = [
            f for f in forcing_files
            if BaseDatasetHandler._file_overlaps_period(f, start_year, end_year)
        ]
        skipped = len(forcing_files) - len(filtered)
        if skipped:
            self.logger.info(
                f"Skipped {skipped} forcing file(s) outside configured period "
                f"{start_year}-{end_year}"
            )
        return filtered

    def run_resampling(self):
        """Run the complete forcing resampling process."""
        self.logger.debug("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.debug("Forcing data resampling process completed")

    def merge_forcings(self):
        """Merge forcing data files using dataset-specific handler."""
        start_year = int(self._get_config_value(lambda: self.config.domain.time_start).split('-')[0])
        end_year = int(self._get_config_value(lambda: self.config.domain.time_end).split('-')[0])

        raw_forcing_path = self.project_forcing_dir / 'raw_data'
        merged_forcing_path = self.project_forcing_dir / 'merged_path'

        self.dataset_handler.merge_forcings(
            raw_forcing_path=raw_forcing_path,
            merged_forcing_path=merged_forcing_path,
            start_year=start_year,
            end_year=end_year
        )

    def create_shapefile(self):
        """Create forcing shapefile using dataset-specific handler."""
        self.logger.debug(f"Creating {self.forcing_dataset.upper()} shapefile")

        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset)}.shp"

        if output_shapefile.exists():
            if self._validate_existing_shapefile(output_shapefile):
                return output_shapefile

        return self.dataset_handler.create_shapefile(
            shapefile_path=self.shapefile_path,
            merged_forcing_path=self.merged_forcing_path,
            dem_path=self.dem_path,
            elevation_calculator=self.elevation_calculator.calculate
        )

    def _validate_existing_shapefile(self, output_shapefile: Path) -> bool:
        """Check if existing shapefile is valid and covers current bbox."""
        try:
            gdf = gpd.read_file(output_shapefile)
            expected_columns = [
                self._get_config_value(lambda: self.config.forcing.shape_lat_name),
                self._get_config_value(lambda: self.config.forcing.shape_lon_name),
                'ID', 'elev_m'
            ]

            if not all(col in gdf.columns for col in expected_columns) or len(gdf) == 0:
                self.logger.debug("Existing forcing shapefile missing expected columns. Recreating.")
                return False

            bbox_str = self._get_config_value(lambda: self.config.domain.bounding_box_coords)
            if isinstance(bbox_str, str) and "/" in bbox_str:
                try:
                    lat_max, lon_min, lat_min, lon_max = [float(v) for v in bbox_str.split("/")]
                    lat_min, lat_max = sorted([lat_min, lat_max])
                    lon_min, lon_max = sorted([lon_min, lon_max])
                    minx, miny, maxx, maxy = gdf.total_bounds
                    tol = 1e-6
                    if (lon_min < minx - tol or lon_max > maxx + tol or
                            lat_min < miny - tol or lat_max > maxy + tol):
                        self.logger.debug("Existing forcing shapefile bounds do not cover current bbox. Recreating.")
                        return False
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Error checking bbox vs shapefile bounds: {e}. Recreating.")
                    return False

            self.logger.debug("Forcing shapefile already exists. Skipping creation.")
            return True

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Error checking existing forcing shapefile: {str(e)}. Recreating.")
            return False

    def remap_forcing(self):
        """Remap forcing data to catchment HRUs."""
        self.logger.debug("Starting forcing remapping process")

        # Check for point-scale bypass conditions
        if (self._get_config_value(lambda: self.config.domain.definition_method, default='') or '').lower() == 'point':
            self.logger.debug("Point-scale domain detected. Using simplified extraction.")
            self._process_point_scale_forcing()
        elif self.point_scale_extractor.should_use_point_scale(self.merged_forcing_path):
            self.logger.info("Tiny forcing grid detected. Using simplified extraction.")
            self._process_point_scale_forcing()
        else:
            self._create_parallelized_weighted_forcing()

        self.logger.debug("Forcing remapping process completed")

    def _process_point_scale_forcing(self):
        """Process forcing files using point-scale extraction."""
        forcing_files = self.file_processor.get_forcing_files(self.merged_forcing_path)
        forcing_files = self._filter_forcing_files_by_period(forcing_files)
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return

        # Use backward-compatible catchment path resolution
        catchment_file_path = self._get_catchment_file_path(self.catchment_name)

        self.point_scale_extractor.process(
            forcing_files=forcing_files,
            output_dir=self.forcing_basin_path,
            catchment_file_path=catchment_file_path,
            output_filename_func=self.file_processor.determine_output_filename,
            dem_path=self.dem_path
        )

    def _create_parallelized_weighted_forcing(self):
        """Create weighted forcing files with parallel/serial processing."""
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)

        # Get forcing files, filtered by configured time period
        forcing_files = self.file_processor.get_forcing_files(self.merged_forcing_path)
        forcing_files = self._filter_forcing_files_by_period(forcing_files)
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return

        self.logger.debug(f"Found {len(forcing_files)} forcing files to process")

        # STEP 1: Create remapping weights once
        source_shp_path = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset)}.shp"
        # Use backward-compatible catchment path resolution
        target_shp_path = self._get_catchment_file_path(self.catchment_name)

        remap_file = self.weight_generator.create_weights(
            sample_forcing_file=forcing_files[0],
            intersect_path=intersect_path,
            source_shp_path=source_shp_path,
            target_shp_path=target_shp_path
        )

        # Transfer cached shapefile info to weight applier
        self.weight_applier.set_shapefile_cache(
            self.weight_generator.cached_target_shp_wgs84,
            self.weight_generator.cached_hru_field
        )

        # STEP 2: Filter already processed files
        remaining_files = self.file_processor.filter_processed_files(forcing_files)
        if not remaining_files:
            self.logger.debug("All files have already been processed")
            return

        # STEP 3: Apply remapping weights
        requested_cpus = int(self._get_config_value(lambda: self.config.system.num_processes, default=1))
        max_available_cpus = mp.cpu_count()
        use_parallel = requested_cpus > 1 and max_available_cpus > 1

        if use_parallel:
            num_cpus = min(requested_cpus, max_available_cpus, 20, len(remaining_files))
            self.logger.debug(f"Using parallel processing with {num_cpus} CPUs")
            success_count = self._process_files_parallel(remaining_files, num_cpus, remap_file)
        else:
            self.logger.debug("Using serial processing")
            success_count = self._process_files_serial(remaining_files, remap_file)

        already_processed = len(forcing_files) - len(remaining_files)
        self.logger.debug(
            f"Processing complete: {success_count} files processed successfully "
            f"out of {len(remaining_files)}"
        )
        self.logger.debug(
            f"Total files processed or skipped: {success_count + already_processed} "
            f"out of {len(forcing_files)}"
        )

    def _process_files_serial(self, files, remap_file):
        """Process files in serial mode."""
        def process_func(file):
            output_file = self.file_processor.determine_output_filename(file)
            return self.weight_applier.apply_weights(file, remap_file, output_file)

        return self.file_processor.process_serial(files, process_func)

    def _process_files_parallel(self, files, num_cpus, remap_file):
        """Process files in parallel mode using picklable worker."""
        # Create a picklable worker with all necessary configuration
        worker = _ParallelWorker(
            config_dict=self.config.to_dict(flatten=True) if hasattr(self.config, 'to_dict') else dict(self.config),
            project_dir=str(self.project_dir),
            output_dir=str(self.forcing_basin_path),
            forcing_dataset=self.forcing_dataset,
            remap_file=str(remap_file),
            cached_target_shp_wgs84=str(self.weight_generator.cached_target_shp_wgs84),
            cached_hru_field=self.weight_generator.cached_hru_field,
            domain_name=self._get_config_value(lambda: self.config.domain.name),
        )

        return self._run_parallel_with_worker(files, num_cpus, worker)

    def _run_parallel_with_worker(self, files, num_cpus, worker):
        """Run parallel processing with picklable worker."""
        import gc

        from tqdm import tqdm

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
                    # Note: _ParallelWorker also calls apply_worker_environment() internally
                    # but the pool initializer ensures it's set before any imports
                    from symfluence.core.hdf5_safety import apply_worker_environment
                    with mp.Pool(processes=num_cpus, initializer=apply_worker_environment) as pool:
                        worker_args = [
                            (str(file), i % num_cpus)
                            for i, file in enumerate(batch_files)
                        ]
                        results = pool.map(worker, worker_args)

                    batch_success = sum(1 for r in results if r)
                    success_count += batch_success
                    pbar.update(len(batch_files))

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.error(f"Error processing batch {batch_num+1}: {str(e)}")
                    pbar.update(len(batch_files))

                gc.collect()

        self.logger.debug(f"Parallel processing complete: {success_count}/{len(files)} successful")
        return success_count
