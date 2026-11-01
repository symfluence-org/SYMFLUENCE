# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MAF Pipeline Runners

Wrapper classes for external HPC-based MAF (Model-Agnostic Framework) tools.
Provides programmatic interfaces to gistool (geospatial extraction) and datatool
(forcing data acquisition) for accessing large-scale datasets on supercomputers.

MAF Overview:
    Model-Agnostic Framework is a collection of tools developed for efficient
    data extraction and processing on HPC systems. SYMFLUENCE wraps these tools
    to enable seamless integration with the broader data acquisition pipeline.

    Components:
    1. gistool: Geospatial data extraction (elevation, landcover, soil class)
    2. datatool: Forcing data extraction (ERA5, RDRS, CASR)
    3. MAF Scheduler: Job queue management and caching

Data Tools:
    gistool (Geospatial Information Subsetting Tool):
    - Extracts elevation (DEM), land cover, soil class data
    - Datasets typically located on HPC shared filesystems
    - Supports bounding-box subsetting
    - Outputs GeoTIFF format
    - Can submit jobs to Slurm/PBS queues

    datatool (Dataset Extractor):
    - Extracts ERA5, RDRS, CASR forcing data
    - Downloads from cloud (CDS, AWS) or HPC cache
    - Temporal subsetting (date ranges)
    - Spatial subsetting (bounding boxes)
    - NetCDF output
    - Slurm job submission with queue monitoring

HPC Integration:
    Designed for supercomputer environments:
    - CPS (Community Petascale Data Server): CDS hosting
    - HPC login nodes or compute clusters
    - Job queue management (Slurm, PBS)
    - Shared filesystem caching
    - Environment modules (gistool, datatool modules)

Configuration:
    Required environment/config:
    - SYMPHLUENCE_DATA_DIR: Base data directory
    - SYMPHLUENCE_CODE_DIR: Code installation directory
    - DOMAIN_NAME: Domain identifier
    - CLUSTER_JSON: Cluster configuration (queue, account, etc.)
    - Data paths: GISTOOL_DATASET_ROOT, datatool datasets

    MAF scheduler command format:
    $ cd {MAIL_REPO} && python run_scheduler.py {config_file}

    Job monitoring:
    - Slurm: squeue command polling (30-second intervals)
    - Max polling: 1000 checks (~8.3 hours max wait)
    - Status tracking: SUBMITTED → RUNNING → COMPLETED

Examples:
    >>> # Geospatial extraction
    >>> from symfluence.data.acquisition.maf_pipeline import gistoolRunner
    >>> runner = gistoolRunner(config, logger)
    >>> runner.create_gistool_command(...)
    >>> runner.execute_gistool_command(cmd)

    >>> # Forcing data extraction
    >>> from symfluence.data.acquisition.maf_pipeline import datatoolRunner
    >>> runner = datatoolRunner(config, logger)
    >>> runner.create_datatool_command(...)
    >>> runner.execute_datatool_command(cmd)

References:
    - Model-Agnostic Framework: https://github.com/CH-Earth/ModelAgnosticFramework
    - MAF Documentation: Extensive README and examples in MAF repository
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from symfluence.core.mixins import ConfigMixin


class gistoolRunner(ConfigMixin):
    """
    Wrapper for gistool command-line utility for geospatial data extraction.

    gistool is a Model-Agnostic Framework (MAF) component that extracts geospatial
    attributes (elevation, land cover, soil class) from large raster datasets hosted
    on HPC systems. This class generates and executes gistool shell commands with
    appropriate parameters for bounding box subsetting and output formatting.

    Purpose:
        Provides programmatic access to gistool for automated geospatial data
        extraction as part of SYMFLUENCE workflows, particularly for domains where
        cloud-based data sources are unavailable or HPC-based datasets are preferred.

    gistool Capabilities:
        - Extract elevation data (DEM)
        - Extract land cover classification
        - Extract soil classification
        - Bounding box spatial subsetting
        - GeoTIFF output generation
        - Caching for repeated extractions
        - HPC cluster job submission (optional)

    Command Generation:
        Generates extract-gis.sh commands with parameters:
        - Dataset specification (elevation, land_class, soil_class)
        - Dataset root directory on HPC filesystem
        - Output directory for GeoTIFF files
        - Spatial bounds (lat/lon limits)
        - Variable selection
        - File prefix for output naming
        - Cache directory for intermediate files
        - Cluster configuration JSON (Slurm/PBS)

    Workflow:
        1. Initialize with configuration and logger
        2. Resolve paths (gistool installation, cache, dataset root)
        3. Call create_gistool_command() with extraction parameters
        4. Execute command via execute_gistool_command()
        5. gistool writes GeoTIFF to output directory

    Configuration Requirements:
        Required:
            - SYMFLUENCE_DATA_DIR: Base data directory
            - SYMFLUENCE_CODE_DIR: Code installation directory
            - DOMAIN_NAME: Domain identifier (for file prefixes)
            - GISTOOL_DATASET_ROOT: Root path to geospatial datasets on HPC
            - CLUSTER_JSON: Cluster configuration file path

        Optional:
            - GISTOOL_PATH: Custom gistool installation path
              Default: {SYMFLUENCE_DATA_DIR}/installs/gistool
            - TOOL_CACHE: Cache directory for intermediate files
              Default: $HOME/cache_dir/

    Output Files:
        Pattern: {output_dir}/domain_{DOMAIN_NAME}_{dataset}_{variable}.tif
        Format: GeoTIFF with CRS and geotransform metadata
        Example: ./attributes/elevation/domain_bow_river_elevation_mean.tif

    HPC Integration:
        - Designed for use on HPC systems with centralized datasets
        - Can submit jobs to Slurm/PBS queues (--submit-job flag)
        - Caching reduces redundant data transfers
        - Dataset directories typically on high-performance storage

    Example:
        >>> config = {
        ...     'SYMFLUENCE_DATA_DIR': '/project/data',
        ...     'DOMAIN_NAME': 'test_basin',
        ...     'GISTOOL_DATASET_ROOT': '/datasets/',
        ...     'CLUSTER_JSON': './cluster_config.json'
        ... }
        >>> runner = gistoolRunner(config, logger)
        >>> cmd = runner.create_gistool_command(
        ...     dataset='elevation',
        ...     output_dir='./attributes/elevation',
        ...     lat_lims='40.0/41.0',
        ...     lon_lims='-106.0/-105.0',
        ...     variables='elevation'
        ... )
        >>> runner.execute_gistool_command(cmd)
        # Executes: extract-gis.sh with parameters
        # Output: domain_test_basin_elevation_mean.tif

    Notes:
        - gistool must be installed and accessible at GISTOOL_PATH
        - Dataset directories must be readable on HPC filesystem
        - Cache directory improves performance for repeated extractions
        - Cluster JSON configures queue/partition/walltime for job submission
        - Geographic coordinates expected in decimal degrees

    See Also:
        - data.preprocessing.attribute_processor: Processes gistool outputs
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.code_dir = Path(self._get_config_value(lambda: self.config.system.code_dir, dict_key='SYMFLUENCE_CODE_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.tool_cache = self._get_config_value(lambda: self.config.paths.tool_cache, dict_key='TOOL_CACHE')

        if self.tool_cache == 'default':
            self.tool_cache = '$HOME/cache_dir/'

        #Get the path to the directory containing the gistool script
        self.gistool_path = self._get_config_value(lambda: self.config.paths.gistool_path, dict_key='GISTOOL_PATH')
        if self.gistool_path == 'default':
            self.gistool_path = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')) / 'installs/gistool'
        else:
            self.gistool_path = self._get_config_value(lambda: self.config.paths.gistool_path, dict_key='GISTOOL_PATH')

    def create_gistool_command(self, dataset, output_dir, lat_lims, lon_lims, variables, start_date=None, end_date=None):
        dataset_dir = dataset
        if dataset == 'soil_class':
            dataset_dir = 'soil_classes'

        gistool_command = [
            f"{self.gistool_path}/extract-gis.sh",
            f"--dataset={dataset}",
            f"--dataset-dir={self._get_config_value(lambda: self.config.paths.gistool_dataset_root, dict_key='GISTOOL_DATASET_ROOT')}{dataset_dir}",
            f"--output-dir={output_dir}",
            f"--lat-lims={lat_lims}",
            f"--lon-lims={lon_lims}",
            f"--variable={variables}",
            f"--prefix=domain_{self.domain_name}_",
            #f"--lib-path={self._get_config_value(lambda: self.config.paths.gistool_lib_path, dict_key='GISTOOL_LIB_PATH')}"
            #"--submit-job",
            "--print-geotiff=true",
            f"--cache={self.tool_cache}_{self.domain_name}",
            f"--cluster={self._get_config_value(lambda: self.config.paths.cluster_json, dict_key='CLUSTER_JSON')}"
        ]

        self.logger.info(f'gistool command: {gistool_command}')
        if start_date and end_date:
            gistool_command.extend([
                f"--start-date={start_date}",
                f"--end-date={end_date}"
            ])

        return gistool_command

    def execute_gistool_command(self, gistool_command):

        #Run the gistool command
        # On Windows, .sh scripts need an explicit bash interpreter
        if sys.platform == "win32":
            gistool_command = ["bash"] + list(gistool_command)
        try:
            subprocess.run(gistool_command, check=True)
            self.logger.info("gistool completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running gistool: {e}")
            raise
        self.logger.info("Geospatial data acquisition process completed")

class datatoolRunner(ConfigMixin):
    """
    Wrapper for datatool command-line utility for forcing data extraction from HPC.

    datatool is a Model-Agnostic Framework (MAF) component that extracts atmospheric
    forcing data from large NetCDF archives hosted on HPC systems.
    This class generates datatool commands, submits them as Slurm jobs, and monitors
    job completion in the queue.

    Purpose:
        Provides programmatic access to datatool for automated forcing data extraction
        on HPC systems where datasets are stored locally and cloud access is unavailable
        or inefficient. Handles job submission, queue monitoring, and completion tracking.

    datatool Capabilities:
        - Extract forcing data from 13 supported datasets
        - Spatial subsetting via bounding box
        - Temporal subsetting via date range
        - Variable selection
        - NetCDF output generation
        - Slurm job array submission
        - Caching for efficiency

    Datasets Supported:
        See DATASET_MAP for the full list of supported datasets and their
        datatool identifiers. Includes global reanalyses (ERA5), regional
        reanalyses (RDRS, CASR), dynamically downscaled products (WRF-CONUS),
        statistically downscaled products (NEX-GDDP-CMIP6, ESPO-G6-R2),
        regional climate model outputs (CanRCM4, WFDEI-GEM-CaPA, MRCC5-CMIP6),
        and gridded observation products (Daymet, Alberta Government).

    Workflow:
        1. Initialize with configuration and logger
        2. Resolve datatool installation path and dataset root
        3. Create datatool command with parameters
        4. Submit command as Slurm array job
        5. Monitor job queue until completion
        6. Return when job no longer in queue (completed or failed)

    Job Monitoring Strategy:
        - Submits job via subprocess and captures job ID
        - Polls `squeue -j {job_id}` every 30 seconds
        - Exits when job no longer appears in queue output
        - Maximum 1000 checks (~8.3 hours) before timeout warning
        - Continues workflow after job completion

    Command Generation:
        Generates extract-dataset.sh commands with parameters:
        - Dataset specification (see DATASET_MAP)
        - Dataset root directory on HPC filesystem
        - Output directory for NetCDF files
        - Temporal bounds (start/end dates)
        - Spatial bounds (lat/lon limits)
        - Variable selection (comma-separated list)
        - File prefix for output naming
        - Submit-job flag (enables Slurm submission)
        - Cache directory
        - Cluster configuration JSON

    Configuration Requirements:
        Required:
            - SYMFLUENCE_DATA_DIR: Base data directory
            - SYMFLUENCE_CODE_DIR: Code installation directory
            - DOMAIN_NAME: Domain identifier (for file prefixes)
            - DATATOOL_DATASET_ROOT: Root path to forcing datasets on HPC
            - CLUSTER_JSON: Cluster configuration file (queue/partition/walltime)

        Optional:
            - DATATOOL_PATH: Custom datatool installation path
              Default: {SYMFLUENCE_DATA_DIR}/installs/datatool
            - TOOL_CACHE: Cache directory for intermediate files
              Default: $HOME/cache_dir/

    Output Files:
        Pattern: {output_dir}/domain_{DOMAIN_NAME}_{dataset}_{YYYYMMDD}-{YYYYMMDD}.nc
        Format: NetCDF4 with CF conventions
        Variables: Depends on dataset and variable selection
        Example: ./forcing/raw/domain_basin_ERA5_20150101-20171231.nc

    Slurm Job Submission:
        - datatool submits job array to Slurm scheduler
        - Job ID extracted from "Submitted batch job {ID}" output
        - Each time step may be processed as array task
        - Parallel processing for multi-year extractions
        - Job output logged to datatool working directory

    Error Handling:
        - Raises RuntimeError if job ID extraction fails
        - Raises CalledProcessError if datatool submission fails
        - Logs stderr output on command errors
        - Warning if maximum queue checks exceeded
        - Does not validate job success (assumes completion when not in queue)

    Performance:
        - Job submission: ~1-5 seconds
        - Queue monitoring: 30-second intervals
        - Typical job duration: 5-60 minutes (depends on dataset/date range)
        - Array parallelization: Speeds up multi-year extractions
        - Caching: Reduces repeated data access

    Example:
        >>> config = {
        ...     'SYMFLUENCE_DATA_DIR': '/project/data',
        ...     'DOMAIN_NAME': 'test_basin',
        ...     'DATATOOL_DATASET_ROOT': '/datasets/',
        ...     'CLUSTER_JSON': './cluster_slurm.json'
        ... }
        >>> runner = datatoolRunner(config, logger)
        >>> cmd = runner.create_datatool_command(
        ...     dataset='ERA5',
        ...     output_dir='./forcing/raw',
        ...     start_date='2015-01-01',
        ...     end_date='2016-12-31',
        ...     lat_lims='40.0/41.0',
        ...     lon_lims='-106.0/-105.0',
        ...     variables='tas,pr,huss,ps,rsds,rlds,uas,vas'
        ... )
        >>> runner.execute_datatool_command(cmd)
        # Submits Slurm job: Submitted batch job 12345678
        # Monitors: Job 12345678 still in queue. Waiting 30 seconds...
        # Completes: Job 12345678 no longer in queue, assuming completed

    Notes:
        - datatool must be installed at DATATOOL_PATH
        - Dataset directories must be accessible on HPC filesystem
        - Slurm scheduler required (uses squeue for monitoring)
        - Job success not validated; check output files for completeness
        - Cache directory shared across runs for efficiency
        - Dataset naming conventions differ (see DATASET_MAP for mappings)

    See Also:
        - data.preprocessing.dataset_handlers: Variable processing for datasets
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.code_dir = Path(self._get_config_value(lambda: self.config.system.code_dir, dict_key='SYMFLUENCE_CODE_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.tool_cache = self._get_config_value(lambda: self.config.paths.tool_cache, dict_key='TOOL_CACHE')
        if self.tool_cache == 'default':
            self.tool_cache = '$HOME/cache_dir/'

        #Get the path to the directory containing the datatool script
        self.datatool_path = self._get_config_value(lambda: self.config.paths.datatool_path, dict_key='DATATOOL_PATH')
        if self.datatool_path == 'default':
            self.datatool_path = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')) / 'installs/datatool'
        else:
            self.datatool_path = self._get_config_value(lambda: self.config.paths.datatool_path, dict_key='DATATOOL_PATH')

    # Maps SYMFLUENCE dataset names → (datatool --dataset value, dataset directory name).
    # Directory names correspond to subdirectories under DATATOOL_DATASET_ROOT on the HPC.
    # The --dataset values must match aliases in datatool's extract-dataset.sh case statement.
    # See https://github.com/CH-Earth/datatool for the full list of supported datasets.
    DATASET_MAP = {
        # Reanalysis
        'ERA5':             ('era5',                   'era5'),
        'RDRS':             ('rdrs',                   'rdrsv2.1'),
        'CASR':             ('casr',                   'casrv3.1'),
        # Observation datasets
        'DAYMET':           ('daymet',                 'daymet-v4r1-ornl'),
        # Dynamically downscaled (WRF)
        'CONUS_I':          ('conus_i',                'conus_i'),
        'CONUS_II':         ('conus_ii',               'conus_ii'),
        # Statistically downscaled / climate projections
        'NEX-GDDP-CMIP6':  ('nex-gddp-cmip6',        'nex-gddp-cmip6'),
        'ESPO-G6-R2':      ('espo-g6-r2',             'espo-g6-r2'),
        'MRCC5-CMIP6':     ('mrcc5-cmip6',            'mrcc5-cmip6'),
        # Regional climate model outputs
        'CANRCM4-WFDEI-GEM-CAPA': ('canrcm4_wfdei_gem_capa', 'canrcm4_wfdei_gem_capa'),
        'WFDEI-GEM-CAPA':  ('wfdei_gem_capa',         'wfdei_gem_capa'),
        # Provincial datasets
        'AB-GOV':           ('ab-gov',                 'ab-gov'),
    }

    @classmethod
    def supported_datasets(cls):
        """Return the set of dataset names supported by datatool on HPC."""
        return set(cls.DATASET_MAP)

    def create_datatool_command(self, dataset, output_dir, start_date, end_date, lat_lims, lon_lims, variables):
        dataset_upper = dataset.upper()
        if dataset_upper not in self.DATASET_MAP:
            supported = ', '.join(sorted(self.DATASET_MAP))
            raise ValueError(
                f"Dataset '{dataset}' is not supported by datatool. "
                f"Supported datasets: {supported}"
            )

        datatool_name, dataset_dir = self.DATASET_MAP[dataset_upper]

        datatool_command = [
        f"{self.datatool_path}/extract-dataset.sh",
        f"--dataset={datatool_name}",
        f"--dataset-dir={self._get_config_value(lambda: self.config.paths.datatool_dataset_root, dict_key='DATATOOL_DATASET_ROOT')}{dataset_dir}",
        f"--output-dir={output_dir}",
        f"--start-date={start_date}",
        f"--end-date={end_date}",
        f"--lat-lims={lat_lims}",
        f"--lon-lims={lon_lims}",
        f"--variable={variables}",
        f"--prefix=domain_{self.domain_name}_",
        "--submit-job",
        f"--cache={self.tool_cache}",
        f"--cluster={self._get_config_value(lambda: self.config.paths.cluster_json, dict_key='CLUSTER_JSON')}",
        ]

        return datatool_command

    def execute_datatool_command(self, datatool_command):
        """
        Execute the datatool command and wait for the job to complete in the queue.

        This simplified implementation focuses on tracking the specific job ID
        until it's no longer present in the Slurm queue.
        """
        # On Windows, .sh scripts need an explicit bash interpreter
        if sys.platform == "win32":
            datatool_command = ["bash"] + list(datatool_command)
        try:
            # Submit the array job
            self.logger.info("Submitting datatool job")
            result = subprocess.run(datatool_command, check=True, capture_output=True, text=True)
            self.logger.info("datatool job submitted successfully.")

            # Extract job ID from the output
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    try:
                        job_id = line.split()[-1].strip()
                        break
                    except (IndexError, ValueError):
                        pass

            if not job_id:
                self.logger.error("Could not extract job ID from submission output")
                self.logger.debug(f"Submission output: {result.stdout}")
                raise RuntimeError("Could not extract job ID from submission output")

            self.logger.info(f"Monitoring job ID: {job_id}")

            # Wait for job to no longer be in the queue
            wait_time = 30  # seconds between checks
            max_checks = 1000  # Maximum number of checks
            check_count = 0

            while check_count < max_checks:
                # Check if job is still in the queue
                check_cmd = ['squeue', '-j', job_id, '-h']
                status_result = subprocess.run(check_cmd, capture_output=True, text=True)

                # If no output, the job is no longer in the queue
                if not status_result.stdout.strip():
                    self.logger.info(f"Job {job_id} is no longer in the queue, assuming completed.")
                    # Wait an additional minute to allow for any file system operations to complete
                    break

                self.logger.info(f"Job {job_id} still in queue. Waiting {wait_time} seconds. Check {check_count+1}/{max_checks}")
                time.sleep(wait_time)
                check_count += 1

            if check_count >= max_checks:
                self.logger.warning(f"Reached maximum checks ({max_checks}) for job {job_id}, but continuing anyway")

            self.logger.info("datatool job monitoring completed.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running datatool: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Command error output: {e.stderr}")
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Unexpected error during datatool execution: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        self.logger.info("Meteorological data acquisition process completed")
