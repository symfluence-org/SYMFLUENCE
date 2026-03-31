# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Remapping Weight Generator

Creates EASYMORE remapping weights for forcing data.
"""

import gc
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import xarray as xr

from symfluence.core.hdf5_safety import clear_xarray_cache, prepare_for_netcdf_operation
from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir

from .shapefile_processor import ShapefileProcessor


def _create_easymore_instance():
    """Create an Easymore instance while suppressing initialization output."""
    from contextlib import redirect_stdout
    from io import StringIO

    import easymore

    captured_output = StringIO()
    with redirect_stdout(captured_output):
        if hasattr(easymore, "Easymore"):
            instance = easymore.Easymore()
        elif hasattr(easymore, "easymore"):
            instance = easymore.easymore()
        else:
            raise AttributeError("easymore module does not expose an Easymore class")
    return instance


def _run_easmore_with_suppressed_output(esmr, logger):
    """Run EASMORE's nc_remapper while suppressing verbose output."""
    import warnings
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    # Aggressive cleanup before running easymore to prevent HDF5 conflicts
    prepare_for_netcdf_operation()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            captured_output = StringIO()
            captured_error = StringIO()

            with redirect_stdout(captured_output), redirect_stderr(captured_error):
                esmr.nc_remapper()

            stdout_text = captured_output.getvalue().strip()
            stderr_text = captured_error.getvalue().strip()

            has_error = any(
                indicator in stderr_text.lower()
                for indicator in ['error', 'failed', 'exception', 'traceback']
            )

            if stdout_text:
                logger.debug(f"EASMORE stdout: {stdout_text[:200]}")

            if stderr_text:
                if has_error:
                    logger.warning(f"EASMORE stderr (errors detected):\n{stderr_text}")
                else:
                    logger.debug(f"EASMORE stderr: {stderr_text[:200]}")

            return (True, stdout_text, stderr_text)

    except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
        logger.error(f"Error running EASMORE: {str(e)}")
        raise
    finally:
        # Cleanup after easymore operation
        clear_xarray_cache()


class RemappingWeightGenerator(ConfigMixin):
    """
    Creates EASYMORE remapping weights from source to target shapefiles.

    This is an expensive GIS operation that only needs to be done once.
    The weights are then reused for all forcing files.
    """

    def __init__(
        self,
        config: dict,
        project_dir: Path,
        dataset_handler,
        logger: logging.Logger = None
    ):
        """
        Initialize weight generator.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            dataset_handler: Dataset-specific handler
            logger: Optional logger instance
        """
        # Use centralized config coercion (handles dict -> SymfluenceConfig with fallback)
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.project_dir = project_dir
        self.dataset_handler = dataset_handler
        self.logger = logger or logging.getLogger(__name__)
        self.shapefile_processor = ShapefileProcessor(config, self.logger)

        # Cache for processed shapefiles
        self.cached_target_shp_wgs84: Optional[Path] = None
        self.cached_hru_field: Optional[str] = None
        self.detected_forcing_vars: list = []

    def create_weights(
        self,
        sample_forcing_file: Path,
        intersect_path: Path,
        source_shp_path: Path,
        target_shp_path: Path
    ) -> Path:
        """
        Create remapping weights using a sample forcing file.

        Args:
            sample_forcing_file: Sample forcing file for variable detection
            intersect_path: Directory to store intersection files
            source_shp_path: Path to source (forcing) shapefile
            target_shp_path: Path to target (catchment) shapefile

        Returns:
            Path to the remapping CSV file
        """
        # Ensure shapefiles are in WGS84
        source_shp_wgs84 = self.shapefile_processor.ensure_wgs84(source_shp_path, "_wgs84")
        target_result = self.shapefile_processor.ensure_wgs84(target_shp_path, "_wgs84")

        if isinstance(target_result, tuple):
            target_shp_wgs84, actual_hru_field = target_result
        else:
            target_shp_wgs84 = target_result
            actual_hru_field = self._get_config_value(lambda: self.config.paths.catchment_hruid, dict_key='CATCHMENT_SHP_HRUID')

        # Cache for reuse during weight application
        self.cached_target_shp_wgs84 = target_shp_wgs84
        self.cached_hru_field = actual_hru_field
        self.logger.debug(f"Cached target shapefile: {target_shp_wgs84}, HRU field: {actual_hru_field}")

        # Define remap file path
        case_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_{self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET')}"
        remap_file = intersect_path / f"{case_name}_{actual_hru_field}_remapping.csv"
        remap_nc = remap_file.with_suffix('.nc')

        # Check if already exists
        if remap_file.exists() and remap_nc.exists():
            intersect_csv = intersect_path / f"{case_name}_intersected_shapefile.csv"
            intersect_shp = intersect_path / f"{case_name}_intersected_shapefile.shp"
            if intersect_csv.exists() or intersect_shp.exists():
                self.logger.debug("Remapping weights files already exist. Skipping creation.")
                return remap_file
            self.logger.debug("Remapping weights found but intersected shapefile missing. Recreating.")
        elif remap_file.exists():
            self.logger.debug("Remapping CSV found but NetCDF weights missing. Recreating.")

        self.logger.debug("Creating remapping weights...")

        temp_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'temp_easymore_weights'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Handle longitude alignment
            target_shp_for_easymore, disable_lon_correction = self._align_longitudes(
                source_shp_wgs84, target_shp_wgs84, temp_dir
            )

            # Configure EASYMORE
            esmr = self._configure_easymore(
                sample_forcing_file,
                source_shp_wgs84,
                target_shp_for_easymore,
                actual_hru_field,
                temp_dir,
                case_name,
                disable_lon_correction
            )

            # Create the weights
            self.logger.info("Running easymore to create remapping weights...")
            _run_easmore_with_suppressed_output(esmr, self.logger)

            # Move files to final location
            self._move_output_files(temp_dir, intersect_path, case_name, remap_file)

            return remap_file

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _align_longitudes(
        self,
        source_shp_wgs84: Path,
        target_shp_wgs84: Path,
        temp_dir: Path
    ) -> Tuple[Path, bool]:
        """Align target longitudes to 0-360 if source uses that frame."""
        target_shp_for_easymore = target_shp_wgs84
        disable_lon_correction = False

        try:
            source_gdf = gpd.read_file(source_shp_wgs84)
            source_lon_field = self._get_config_value(lambda: self.config.forcing.shape_lon_name, dict_key='FORCING_SHAPE_LON_NAME')

            if source_lon_field in source_gdf.columns:
                source_lon_max = float(source_gdf[source_lon_field].max())

                if source_lon_max > 180:
                    target_gdf = gpd.read_file(target_shp_wgs84)
                    minx, _, maxx, _ = target_gdf.total_bounds

                    if minx < 0 or maxx < 0:
                        from shapely.affinity import translate

                        target_gdf = target_gdf.copy()
                        target_gdf["geometry"] = target_gdf["geometry"].apply(
                            lambda geom: translate(geom, xoff=360) if geom is not None else geom
                        )

                        target_lon_field = self._get_config_value(lambda: self.config.paths.catchment_lon, dict_key='CATCHMENT_SHP_LON')
                        if target_lon_field in target_gdf.columns:
                            target_gdf[target_lon_field] = target_gdf[target_lon_field].apply(
                                lambda v: v + 360 if v < 0 else v
                            )

                        shifted_path = temp_dir / f"{target_shp_wgs84.stem}_lon360.shp"
                        target_gdf.to_file(shifted_path)
                        target_shp_for_easymore = shifted_path
                        disable_lon_correction = True
                        self.logger.debug("Shifted target shapefile longitudes to 0-360.")

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to align target longitudes: {e}")

        return target_shp_for_easymore, disable_lon_correction

    def _configure_easymore(
        self,
        sample_forcing_file: Path,
        source_shp_wgs84: Path,
        target_shp_for_easymore: Path,
        actual_hru_field: str,
        temp_dir: Path,
        case_name: str,
        disable_lon_correction: bool
    ):
        """Configure EASYMORE for weight creation."""
        esmr = _create_easymore_instance()

        esmr.author_name = 'SUMMA public workflow scripts'
        esmr.license = 'Copernicus data use license'
        esmr.case_name = case_name
        esmr.correction_shp_lon = False

        # Shapefile configuration
        esmr.source_shp = str(source_shp_wgs84)
        esmr.source_shp_lat = self._get_config_value(lambda: self.config.forcing.shape_lat_name, dict_key='FORCING_SHAPE_LAT_NAME')
        esmr.source_shp_lon = self._get_config_value(lambda: self.config.forcing.shape_lon_name, dict_key='FORCING_SHAPE_LON_NAME')
        esmr.source_shp_ID = self._get_config_value(lambda: self.config.forcing.shape_id_name, default='ID')

        esmr.target_shp = str(target_shp_for_easymore)
        esmr.target_shp_ID = actual_hru_field
        esmr.target_shp_lat = self._get_config_value(lambda: self.config.paths.catchment_lat, dict_key='CATCHMENT_SHP_LAT')
        esmr.target_shp_lon = self._get_config_value(lambda: self.config.paths.catchment_lon, dict_key='CATCHMENT_SHP_LON')

        # NetCDF configuration
        var_lat, var_lon = self.dataset_handler.get_coordinate_names()
        # Note: HDF5_USE_FILE_LOCKING is now set at package init (symfluence/__init__.py)

        # Detect variables and resolution
        available_vars, source_nc_resolution = self._detect_variables(
            sample_forcing_file, var_lat, var_lon
        )
        self.detected_forcing_vars = available_vars

        esmr.source_nc = str(sample_forcing_file)
        esmr.var_names = available_vars
        esmr.var_lat = var_lat
        esmr.var_lon = var_lon
        esmr.var_time = 'time'

        if source_nc_resolution is not None:
            esmr.source_nc_resolution = source_nc_resolution

        # Directories
        esmr.temp_dir = str(temp_dir) + '/'
        esmr.output_dir = str(temp_dir) + '/'

        # Output configuration
        esmr.remapped_dim_id = 'hru'
        esmr.remapped_var_id = 'hruId'
        esmr.format_list = ['f4']
        esmr.fill_value_list = ['-9999']

        # Weight creation only
        esmr.only_create_remap_csv = True
        if hasattr(esmr, 'only_create_remap_nc'):
            esmr.only_create_remap_nc = True

        esmr.save_csv = True
        esmr.sort_ID = False
        esmr.save_temp_shp = True
        esmr.numcpu = 1

        return esmr

    def _detect_variables(
        self,
        sample_file: Path,
        var_lat: str,
        var_lon: str
    ) -> Tuple[list, Optional[float]]:
        """Detect variables and resolution in forcing file."""
        available_vars = []
        source_nc_resolution = None

        try:
            with xr.open_dataset(sample_file, engine="h5netcdf") as ds:
                # CFIF standard variable names (primary)
                all_cfif_vars = [
                    'surface_air_pressure', 'surface_downwelling_longwave_flux',
                    'surface_downwelling_shortwave_flux', 'precipitation_flux',
                    'air_temperature', 'specific_humidity', 'wind_speed',
                    'relative_humidity',
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
                    raise ValueError(f"No forcing variables found in {sample_file}")

                self.logger.info(
                    f"Detected {len(available_vars)} "
                    f"forcing variables: {available_vars}"
                )

                # Calculate grid resolution
                if var_lat not in ds:
                    raise KeyError(f"Latitude variable '{var_lat}' not found")
                if var_lon not in ds:
                    raise KeyError(f"Longitude variable '{var_lon}' not found")

                lat_vals = ds[var_lat].values
                lon_vals = ds[var_lon].values

                if lat_vals.ndim == 1:
                    lat_size = len(lat_vals)
                    lon_size = len(lon_vals)
                elif lat_vals.ndim == 2:
                    lat_size = lat_vals.shape[0]
                    lon_size = lat_vals.shape[1]
                else:
                    lat_size = lon_size = 1

                if lat_size == 1 or lon_size == 1:
                    if lat_vals.ndim == 1:
                        res_lat = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.25
                        res_lon = abs(float(lon_vals[1] - lon_vals[0])) if len(lon_vals) > 1 else 0.25
                    else:
                        res_lat = res_lon = 0.25

                    source_nc_resolution = max(res_lat, res_lon)
                    self.logger.info(
                        f"Small grid detected ({lat_size}x{lon_size}), "
                        f"setting resolution={source_nc_resolution}"
                    )

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error detecting variables in {sample_file}: {e}")
            raise
        finally:
            gc.collect()

        return available_vars, source_nc_resolution

    def _move_output_files(
        self,
        temp_dir: Path,
        intersect_path: Path,
        case_name: str,
        remap_file: Path
    ) -> None:
        """Move output files from temp to final location."""
        case_remap_csv = temp_dir / f"{case_name}_remapping.csv"
        case_remap_nc = temp_dir / f"{case_name}_remapping.nc"
        case_attr_nc = temp_dir / f"{case_name}_attributes.nc"

        # Convert NC to CSV if needed
        if not case_remap_csv.exists() and case_remap_nc.exists():
            self.logger.debug("Converting NetCDF weights to CSV...")
            try:
                with xr.open_dataset(case_remap_nc, engine="h5netcdf") as ds:
                    ds.to_dataframe().to_csv(case_remap_csv)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to convert NetCDF weights to CSV: {e}")

        candidate_paths = [case_remap_csv]

        if not any(path.exists() for path in candidate_paths):
            mapping_patterns = ["*remapping*.csv", "*_remapping.csv", "Mapping_*.csv"]
            fallback = []
            for pattern in mapping_patterns:
                fallback.extend(list(temp_dir.glob(pattern)))
            if fallback:
                self.logger.info(f"Using fallback mapping file: {fallback[0].name}")
                candidate_paths.extend(fallback)

        remap_source = next((path for path in candidate_paths if path.exists()), None)

        if remap_source is not None:
            remap_source.replace(remap_file)
            self.logger.info(f"Remapping weights created: {remap_file}")

            remap_nc = remap_file.with_suffix('.nc')
            attr_nc = remap_file.parent / f"{case_name}_attributes.nc"

            if case_remap_nc.exists():
                shutil.move(str(case_remap_nc), str(remap_nc))
                self.logger.debug(f"Moved NetCDF remapping file to {remap_nc}")

            if case_attr_nc.exists():
                shutil.move(str(case_attr_nc), str(attr_nc))
                self.logger.debug(f"Moved NetCDF attributes file to {attr_nc}")
        else:
            self.logger.error(f"Remapping file not found. Checked: {candidate_paths}")
            self.logger.error(f"Contents of {temp_dir}: {list(temp_dir.glob('*'))}")
            raise FileNotFoundError(f"Expected remapping file not created: {case_remap_csv}")

        # Move shapefile files
        for shp_file in temp_dir.glob(f"{case_name}_intersected_shapefile.*"):
            shutil.move(str(shp_file), str(intersect_path / shp_file.name))
