# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Remapping Weight Applier

Applies pre-computed EASYMORE weights to forcing files.
"""

import gc
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import xarray as xr

from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir

from .file_validator import FileValidator
from .weight_generator import _create_easymore_instance, _run_easmore_with_suppressed_output


class RemappingWeightApplier(ConfigMixin):
    """
    Applies pre-computed EASYMORE remapping weights to forcing files.

    This is the fast operation that reads weights and applies them,
    unlike weight generation which is an expensive GIS operation.
    """

    def __init__(
        self,
        config: dict,
        project_dir: Path,
        output_dir: Path,
        dataset_handler,
        logger: logging.Logger = None
    ):
        """
        Initialize weight applier.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            output_dir: Output directory for remapped files
            dataset_handler: Dataset-specific handler
            logger: Optional logger instance
        """
        # Use centralized config coercion (handles dict -> SymfluenceConfig with fallback)
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.project_dir = project_dir
        self.output_dir = output_dir
        self.dataset_handler = dataset_handler
        self.logger = logger or logging.getLogger(__name__)
        self.file_validator = FileValidator(self.logger)

        # Cache for shapefile paths (set by weight generator)
        self.cached_target_shp_wgs84: Optional[Path] = None
        self.cached_hru_field: Optional[str] = None

    def set_shapefile_cache(
        self,
        target_shp_wgs84: Path,
        hru_field: str
    ) -> None:
        """
        Set cached shapefile paths from weight generation phase.

        Args:
            target_shp_wgs84: Path to WGS84 target shapefile
            hru_field: HRU ID field name
        """
        self.cached_target_shp_wgs84 = target_shp_wgs84
        self.cached_hru_field = hru_field
        self.logger.debug(f"Cached shapefile: {target_shp_wgs84}, HRU field: {hru_field}")

    def apply_weights(
        self,
        file: Path,
        remap_file: Path,
        output_file: Path,
        worker_id: Optional[int] = None
    ) -> bool:
        """
        Apply pre-computed remapping weights to a forcing file.

        Args:
            file: Path to forcing file to process
            remap_file: Path to pre-computed remapping weights CSV
            output_file: Path for output file
            worker_id: Optional worker ID for logging

        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        worker_str = f"Worker {worker_id}: " if worker_id is not None else ""

        try:
            # Check if output already exists
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:
                    self.logger.debug(f"{worker_str}Output already exists: {file.name}")
                    return True

            # Create unique temp directory
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = resolve_data_subdir(self.project_dir, 'forcing') / f'temp_apply_{unique_id}'
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Configure EASYMORE
                esmr = self._configure_easymore(file, remap_file, temp_dir, worker_str)
                if esmr is None:
                    return False

                # Apply the remapping
                self.logger.debug(f"{worker_str}Applying remapping weights to {file.name}")
                self.logger.debug(f"{worker_str}EASYMORE configured to remap variables: {esmr.var_names}")

                success, stdout, stderr = _run_easmore_with_suppressed_output(esmr, self.logger)

                # Log concerning patterns
                if 'no data' in stdout.lower() or 'no data' in stderr.lower():
                    self.logger.warning(f"{worker_str}EASYMORE reported 'no data' for {file.name}")
                if 'empty' in stdout.lower() or 'empty' in stderr.lower():
                    self.logger.warning(f"{worker_str}EASYMORE reported 'empty' for {file.name}")

                # Find and move output file
                success = self._move_output_file(temp_dir, output_file, file, worker_str)
                if not success:
                    return False

            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

            # Verify output
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:
                    elapsed_time = time.time() - start_time
                    self.logger.debug(
                        f"{worker_str}Successfully processed {file.name} in {elapsed_time:.2f} seconds"
                    )
                    return True
                else:
                    self.logger.error(f"{worker_str}Output file corrupted (size: {file_size})")
                    return False
            else:
                self.logger.error(f"{worker_str}Output file not created: {output_file}")
                return False

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"{worker_str}Error processing {file.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _configure_easymore(
        self,
        file: Path,
        remap_file: Path,
        temp_dir: Path,
        worker_str: str
    ):
        """Configure EASYMORE for weight application."""
        esmr = _create_easymore_instance()

        esmr.author_name = 'SUMMA public workflow scripts'
        esmr.case_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_{self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')}"
        esmr.correction_shp_lon = False

        # Use cached shapefile path
        if self.cached_target_shp_wgs84 is None or self.cached_hru_field is None:
            self.logger.error(f"{worker_str}Shapefile cache not available")
            return None

        esmr.target_shp = str(self.cached_target_shp_wgs84)
        esmr.target_shp_ID = self.cached_hru_field
        esmr.target_shp_lat = self._get_config_value(lambda: self.config.paths.catchment_lat, dict_key='CATCHMENT_SHP_LAT')
        esmr.target_shp_lon = self._get_config_value(lambda: self.config.paths.catchment_lon, dict_key='CATCHMENT_SHP_LON')

        # Coordinate variables
        var_lat, var_lon = self.dataset_handler.get_coordinate_names()

        # NetCDF file configuration
        esmr.source_nc = str(file)

        # Detect variables in this specific file
        available_vars = self._detect_file_variables(file, worker_str)
        if not available_vars:
            return None

        esmr.var_names = available_vars
        esmr.var_lat = var_lat
        esmr.var_lon = var_lon
        esmr.var_time = 'time'

        # Directories
        esmr.temp_dir = str(temp_dir) + '/'
        esmr.output_dir = str(temp_dir) + '/'

        # Output configuration
        esmr.remapped_dim_id = 'hru'
        esmr.remapped_var_id = 'hruId'
        esmr.format_list = ['f4']
        esmr.fill_value_list = ['-9999']

        # Point to pre-computed weights
        esmr.remap_csv = str(remap_file)

        # EASYMORE 2.0: Provide NetCDF version if available
        remap_nc = remap_file.with_suffix('.nc')
        case_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_{self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')}"
        attr_nc = remap_file.parent / f"{case_name}_attributes.nc"

        if remap_nc.exists():
            esmr.remap_nc = str(remap_nc)
            self.logger.debug(f"{worker_str}Using NetCDF remapping file: {remap_nc}")

        if attr_nc.exists():
            esmr.attr_nc = str(attr_nc)
            self.logger.debug(f"{worker_str}Using NetCDF attributes file: {attr_nc}")

        esmr.save_csv = False
        esmr.sort_ID = False
        esmr.save_temp_shp = False
        esmr.numcpu = 1

        return esmr

    def _detect_file_variables(self, file: Path, worker_str: str) -> list:
        """Detect forcing variables in the file.

        Tries, in order:
          1. CFIF standard names (e.g. 'air_temperature') — the post-merge
             expectation.
          2. Legacy SUMMA-style names (e.g. 'airtemp') — older intermediate
             format kept for back-compat.
          3. Raw dataset names from the active dataset handler's rename
             map (e.g. 'tas' for NEX-GDDP, 'pr' for RDRS-projection,
             'RDRS_v2.1_P_TT_1.5m' for native RDRS). When matches are
             found here it means the per-handler ``process_dataset`` step
             was bypassed; we materialise a renamed copy in-place so
             EASYMORE produces a CFIF-named output and SUMMA's forcing
             processor can read it. This is the defensive guard described
             in feedback item 3.1.
        """
        try:
            with xr.open_dataset(file, engine="h5netcdf") as ds:
                # CFIF standard variable names (primary)
                all_cfif_vars = [
                    'surface_air_pressure', 'surface_downwelling_longwave_flux',
                    'surface_downwelling_shortwave_flux', 'precipitation_flux',
                    'air_temperature', 'specific_humidity', 'wind_speed',
                    'relative_humidity', 'eastward_wind', 'northward_wind',
                ]
                available_vars = [v for v in all_cfif_vars if v in ds]

                if not available_vars:
                    # Fallback: legacy SUMMA-style variable names
                    all_legacy_vars = [
                        'airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate',
                        'airtemp', 'spechum', 'windspd', 'relhum',
                    ]
                    available_vars = [v for v in all_legacy_vars if v in ds]

                if not available_vars:
                    # Defensive guard: try the dataset handler's rename map.
                    # If raw dataset names are present, the per-handler
                    # process_dataset step was bypassed somewhere upstream;
                    # materialise a CFIF-renamed copy in-place so EASYMORE
                    # gets standardised names.
                    handler_vars = self._maybe_rename_with_handler(file, ds, worker_str)
                    if handler_vars:
                        return handler_vars

                if not available_vars:
                    self.logger.error(
                        f"{worker_str}No forcing variables found in {file.name} "
                        f"(checked CFIF, legacy SUMMA, and dataset-handler rename map). "
                        f"Available variables: {list(ds.variables.keys())}"
                    )
                    return []

                if 'time' not in ds.dims:
                    self.logger.error(f"{worker_str}Input file {file.name} has no time dimension!")
                    return []

                self.logger.debug(
                    f"{worker_str}Detected {len(available_vars)} variables in {file.name}: {available_vars}"
                )
                return available_vars

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"{worker_str}Error opening {file.name} for variable detection: {e}")
            return []
        finally:
            gc.collect()

    def _maybe_rename_with_handler(
        self,
        file: Path,
        ds: xr.Dataset,
        worker_str: str,
    ) -> list:
        """Apply dataset-handler standardisation in-place if raw names are present.

        Returns the list of CFIF variable names now in the file, or [] if
        nothing matched. Reopens the file because xr does not allow
        in-place rewrite of the dataset that's currently open.
        """
        if self.dataset_handler is None:
            return []
        # Discover the rename map for this handler. Older handlers expose
        # ``get_variable_mapping``; newer ones use ``process_dataset``.
        rename_map = {}
        if hasattr(self.dataset_handler, 'get_variable_mapping'):
            try:
                full_map = self.dataset_handler.get_variable_mapping() or {}
                rename_map = {k: v for k, v in full_map.items() if k in ds.variables}
            except Exception as e:  # noqa: BLE001 — defensive guard, don't kill remap
                self.logger.debug(f"{worker_str}get_variable_mapping failed: {e}")

        if not rename_map:
            return []

        self.logger.warning(
            f"{worker_str}{file.name} has raw dataset variables "
            f"({sorted(rename_map.keys())}); the per-handler standardisation "
            f"step appears to have been skipped. Materialising a CFIF-renamed "
            f"copy in-place so remapping can proceed. "
            f"This indicates an upstream wiring gap — see feedback item 3.1."
        )
        try:
            ds_load = xr.open_dataset(file, engine="h5netcdf").load()
            try:
                ds_proc = self.dataset_handler.process_dataset(ds_load)
                tmp = file.with_suffix(file.suffix + '.cfif_tmp')
                ds_proc.to_netcdf(tmp)
            finally:
                ds_load.close()
            tmp.replace(file)
        except Exception as e:  # noqa: BLE001 — defensive guard, don't kill remap
            self.logger.error(
                f"{worker_str}Failed to materialise CFIF-renamed copy of "
                f"{file.name}: {e}"
            )
            return []

        # Re-detect on the rewritten file
        with xr.open_dataset(file, engine="h5netcdf") as ds2:
            cfif_now = [
                v for v in (
                    'surface_air_pressure', 'surface_downwelling_longwave_flux',
                    'surface_downwelling_shortwave_flux', 'precipitation_flux',
                    'air_temperature', 'specific_humidity', 'wind_speed',
                    'relative_humidity', 'eastward_wind', 'northward_wind',
                )
                if v in ds2
            ]
        self.logger.info(
            f"{worker_str}{file.name} now contains CFIF variables: {cfif_now}"
        )
        return cfif_now

    def _move_output_file(
        self,
        temp_dir: Path,
        output_file: Path,
        input_file: Path,
        worker_str: str
    ) -> bool:
        """Find and move output file from temp directory."""
        exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params', 'remapping']
        all_temp_files = list(temp_dir.glob("*.nc"))
        temp_output_files = [
            f for f in all_temp_files
            if not any(pattern in f.name.lower() for pattern in exclude_patterns)
        ]

        if temp_output_files:
            temp_output = temp_output_files[0]

            # Validate before moving
            is_valid = self.file_validator.validate(temp_output, worker_str)

            if not is_valid:
                self.logger.error(
                    f"{worker_str}EASYMORE created invalid output for input {input_file.name}. "
                    f"Output file: {temp_output.name}."
                )
                all_created = list(temp_dir.glob("*"))
                self.logger.error(f"{worker_str}Files created by EASYMORE: {[f.name for f in all_created]}")
                return False

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Move to final location
            shutil.move(str(temp_output), str(output_file))
            self.logger.debug(f"{worker_str}Moved {temp_output.name} to {output_file.name}")
            return True

        elif output_file.exists():
            self.logger.debug(f"{worker_str}Output file already exists: {output_file.name}")
            return True
        else:
            self.logger.error(
                f"{worker_str}EASYMORE created NO valid output files for input {input_file.name}. "
                f"Files in temp dir: {[f.name for f in all_temp_files]}"
            )
            return False
