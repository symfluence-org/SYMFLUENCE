# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Subcatchment Processor Module

Handles distributed FUSE execution for subcatchment-based workflows:
- Settings file management for individual subcatchments
- Forcing data extraction per subcatchment
- Elevation band creation
- Output combination from multiple subcatchments

Extracted from FUSERunner for better maintainability and testability.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.modeling.netcdf_utils import create_netcdf_encoding
from symfluence.core.path_resolver import find_basin_shapefile, find_catchment_subfile
from symfluence.core.process_exec import run as run_subprocess

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)


class SubcatchmentProcessor(ConfigMixin):
    """
    Processes FUSE subcatchments for distributed model execution.

    Handles the complete workflow for running FUSE on individual subcatchments
    and combining the results into a unified output dataset.

    Attributes:
        project_dir: Path to the project directory
        domain_name: Name of the domain being processed
        experiment_id: Identifier for the experiment
        config_dict: Configuration dictionary
        setup_dir: Path to FUSE settings directory
        output_path: Path to FUSE output directory
        fuse_exe: Path to FUSE executable
        logger: Logger instance
    """

    def __init__(
        self,
        project_dir: Path,
        domain_name: str,
        experiment_id: str,
        config_dict: Dict[str, Any],
        setup_dir: Path,
        output_path: Path,
        fuse_exe: Path,
        logger: Optional[logging.Logger] = None,
        config: Optional['SymfluenceConfig'] = None
    ):
        """
        Initialize the subcatchment processor for distributed FUSE execution.

        Args:
            project_dir: Root directory for the project.
            domain_name: Name of the hydrological domain.
            experiment_id: Unique identifier for this experiment run.
            config_dict: Configuration dictionary with model parameters.
            setup_dir: Directory containing FUSE settings files.
            output_path: Directory for FUSE simulation outputs.
            fuse_exe: Path to the FUSE executable.
            logger: Optional logger instance for status messages.
            config: Optional typed SymfluenceConfig object.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        self.experiment_id = experiment_id
        self._config = config
        self._config_dict_override = config_dict
        self.setup_dir = setup_dir
        self.output_path = output_path
        self.fuse_exe = fuse_exe
        self.logger = logger or logging.getLogger(__name__)

        # Derived paths
        self.forcing_fuse_path = resolve_data_subdir(project_dir, 'forcing') / 'FUSE_input'

    def load_subcatchment_info(self, catchment_name_col: str = 'default') -> np.ndarray:
        """
        Load subcatchment identifiers for distributed FUSE execution.

        Reads catchment shapefiles to extract unique subcatchment IDs. First checks
        for delineated subcatchments (from TauDEM or similar), then falls back to
        HRU-based discretization. Each subcatchment will be run independently.

        Args:
            catchment_name_col: Column name for catchment identification, or
                'default' to auto-detect based on domain and discretization.

        Returns:
            np.ndarray: Array of integer subcatchment IDs (GRU_ID values).
        """
        # Check if delineated catchments exist (for distributed routing).
        # Discretization writes this under the nested experiment layout.
        delineated_path = find_catchment_subfile(
            self.project_dir / 'shapefiles',
            self.domain_definition_method,
            self.experiment_id,
            f"{self.domain_name}_catchment_delineated.shp",
            logger=self.logger,
        )

        if delineated_path is not None:
            self.logger.info("Using delineated subcatchments")
            subcatchments = gpd.read_file(delineated_path)
            return subcatchments['GRU_ID'].values.astype(int)
        else:
            # Use regular HRUs. Honour an explicit catchment_name_col, else
            # resolve via the shared finder (nested layout + casing drift).
            if catchment_name_col != 'default':
                catchment_file = self._get_catchment_path() / catchment_name_col
            else:
                catchment_file = find_basin_shapefile(
                    self.project_dir / 'shapefiles',
                    self.domain_name,
                    self.domain_definition_method,
                    self.experiment_id,
                    include_river_basins=False,
                    required=True,
                    logger=self.logger,
                )

            catchment = gpd.read_file(catchment_file)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

    def run_individual_subcatchments(self, subcatchments: np.ndarray) -> bool:
        """
        Execute FUSE model for each subcatchment sequentially.

        For each subcatchment, this method:
        1. Extracts forcing data specific to that subcatchment
        2. Creates subcatchment-specific settings files
        3. Executes FUSE with those settings
        4. Collects successful outputs for combination

        After all subcatchments complete, outputs are combined into a single
        distributed results file.

        Args:
            subcatchments: Array of subcatchment IDs to process.

        Returns:
            bool: True if at least one subcatchment ran successfully and
                outputs were combined. False if all subcatchments failed.
        """
        outputs = []

        for i, subcat_id in enumerate(subcatchments):
            self.logger.info(f"Running FUSE for subcatchment {subcat_id} ({i+1}/{len(subcatchments)})")

            try:
                # Extract forcing for this subcatchment
                subcat_forcing = self.extract_subcatchment_forcing(subcat_id, i)

                # Create subcatchment-specific settings
                subcat_settings = self.create_subcatchment_settings(subcat_id, i)

                # Run FUSE for this subcatchment
                subcat_output = self.execute_fuse_subcatchment(subcat_id, subcat_forcing, subcat_settings)

                if subcat_output:
                    outputs.append((subcat_id, subcat_output))
                else:
                    self.logger.warning(f"FUSE failed for subcatchment {subcat_id}")

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.error(f"Error running subcatchment {subcat_id}: {str(e)}", exc_info=True)
                continue

        if outputs:
            # Combine outputs from all subcatchments
            self.combine_subcatchment_outputs(outputs)
            return True
        else:
            self.logger.error("No successful subcatchment runs")
            return False

    def create_subcatchment_settings(self, subcat_id: int, index: int) -> Path:
        """
        Create subcatchment-specific settings files.

        Args:
            subcat_id: Subcatchment identifier
            index: Index position in subcatchment array

        Returns:
            Path to subcatchment settings directory
        """
        try:
            # Create subcatchment-specific settings directory
            subcat_settings_dir = self.setup_dir / f"subcat_{subcat_id}"
            subcat_settings_dir.mkdir(exist_ok=True)

            # Copy base settings files
            base_settings_dir = self.setup_dir

            for file in base_settings_dir.glob("*.txt"):
                if "subcat_" not in file.name:  # Don't copy other subcatchment files
                    dest_file = subcat_settings_dir / file.name
                    shutil.copy2(file, dest_file)

            # Update file manager for this subcatchment
            fm_file = subcat_settings_dir / 'fm_catch.txt'
            if fm_file.exists():
                with open(fm_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Update paths to point to subcatchment-specific files
                content = content.replace(
                    f"{self.domain_name}_input.nc",
                    f"subcat_{subcat_id}_input.nc"
                )
                content = content.replace(
                    f"/{self.experiment_id}/FUSE/",
                    f"/{self.experiment_id}/FUSE/subcat_{subcat_id}/"
                )

                with open(fm_file, 'w', encoding='utf-8') as f:
                    f.write(content)

            return subcat_settings_dir

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error creating subcatchment settings for {subcat_id}: {str(e)}")
            raise

    def execute_fuse_subcatchment(
        self,
        subcat_id: int,
        forcing_file: Path,
        settings_dir: Path
    ) -> Optional[Path]:
        """
        Execute FUSE for a specific subcatchment.

        Args:
            subcat_id: Subcatchment identifier
            forcing_file: Path to forcing file for this subcatchment
            settings_dir: Path to settings directory for this subcatchment

        Returns:
            Path to output file if successful, None otherwise
        """
        try:
            # Create subcatchment output directory
            subcat_output_dir = self.output_path / f"subcat_{subcat_id}"
            subcat_output_dir.mkdir(parents=True, exist_ok=True)

            # Create elevation bands file for this subcatchment
            self.create_subcatchment_elevation_bands(subcat_id)

            # Run FUSE with subcatchment-specific settings
            control_file = settings_dir / 'fm_catch.txt'

            command = [
                str(self.fuse_exe),
                str(control_file),
                f"{self.domain_name}_subcat_{subcat_id}",
                "run_def"  # Run with default parameters for distributed mode
            ]

            # Create log file for this subcatchment
            log_file = subcat_output_dir / 'fuse_run.log'

            timeout = self._get_config_value(
                lambda: self.config.model.fuse.timeout,
                default=3600,
                dict_key='FUSE_TIMEOUT',
            )

            with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
                result = run_subprocess(
                    command,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=str(settings_dir),
                    timeout=timeout,
                )

            if result.returncode == 0:
                # Find and return the output file
                output_files = list(subcat_output_dir.glob("*_runs_best.nc"))
                if output_files:
                    return output_files[0]
                else:
                    self.logger.warning(f"No output file found for subcatchment {subcat_id}")
                    return None
            else:
                self.logger.error(f"FUSE failed for subcatchment {subcat_id} with return code {result.returncode}")
                return None

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error executing FUSE for subcatchment {subcat_id}: {str(e)}", exc_info=True)
            return None

    def extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """
        Extract forcing data for a specific subcatchment while preserving netCDF structure.

        Args:
            subcat_id: Subcatchment identifier
            index: Index position in subcatchment array

        Returns:
            Path to subcatchment forcing file
        """
        # Load distributed forcing data
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        ds = xr.open_dataset(forcing_file)

        # Extract data for this subcatchment based on coordinate system
        subcatchment_dim = self._get_config_value(
            lambda: self.config.model.fuse.subcatchment_dim,
            default='longitude',
            dict_key='FUSE_SUBCATCHMENT_DIM'
        )

        try:
            if subcatchment_dim == 'latitude':
                # Find the index of this subcatchment ID in the latitude coordinates
                lat_coords = ds.latitude.values
                try:
                    subcat_idx = list(lat_coords).index(float(subcat_id))
                except ValueError:
                    # If exact match not found, use the index directly
                    if index < len(lat_coords):
                        subcat_idx = index
                    else:
                        raise ModelExecutionError(f"Subcatchment index {index} out of range") from None

                # Extract data for this subcatchment but preserve the dimensional structure
                subcat_data = ds.isel(latitude=slice(subcat_idx, subcat_idx + 1))

            else:
                # Similar logic for longitude dimension
                lon_coords = ds.longitude.values
                try:
                    subcat_idx = list(lon_coords).index(float(subcat_id))
                except ValueError:
                    if index < len(lon_coords):
                        subcat_idx = index
                    else:
                        raise ModelExecutionError(f"Subcatchment index {index} out of range") from None

                subcat_data = ds.isel(longitude=slice(subcat_idx, subcat_idx + 1))

            # Verify the structure
            expected_dims = ['time', 'latitude', 'longitude']
            for var in ['pr', 'precip', 'temp', 'pet', 'q_obs']:
                if var in subcat_data:
                    actual_dims = list(subcat_data[var].dims)
                    if actual_dims != expected_dims:
                        self.logger.error(f"Dimension mismatch for {var}: got {actual_dims}, expected {expected_dims}")
                        raise ModelExecutionError(f"Dimension structure incorrect for {var}")

            # Preserve all attributes
            for var in subcat_data.data_vars:
                if var in ds:
                    subcat_data[var].attrs = ds[var].attrs.copy()

            for coord in subcat_data.coords:
                if coord in ds.coords:
                    subcat_data[coord].attrs = ds[coord].attrs.copy()

            subcat_data.attrs = ds.attrs.copy()
            subcat_data.attrs['subcatchment_id'] = subcat_id

            # Save with proper encoding
            subcat_forcing_file = self.forcing_fuse_path / f"{self.domain_name}_subcat_{subcat_id}_input.nc"

            encoding = {}
            for var in subcat_data.data_vars:
                encoding[str(var)] = {
                    '_FillValue': -9999.0,
                    'dtype': 'float32'
                }

            for coord in subcat_data.coords:
                if coord == 'time':
                    encoding[str(coord)] = {'dtype': 'float64'}
                else:
                    encoding[str(coord)] = {'dtype': 'float64'}

            subcat_data.to_netcdf(
                subcat_forcing_file,
                encoding=encoding,
                format='NETCDF4',
                unlimited_dims=['time']
            )

            ds.close()
            subcat_data.close()

            self.logger.info(f"Created forcing file for subcatchment {subcat_id}: {subcat_forcing_file}")
            return subcat_forcing_file

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting forcing for subcatchment {subcat_id}: {str(e)}")
            ds.close()
            raise

    def create_subcatchment_elevation_bands(self, subcat_id: int) -> Path:
        """
        Create elevation bands file for a specific subcatchment.

        Args:
            subcat_id: Subcatchment identifier

        Returns:
            Path to elevation bands file
        """
        try:
            # Source elevation bands file (the main one created during preprocessing)
            source_elev_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"

            # Target elevation bands file for this subcatchment
            target_elev_file = self.forcing_fuse_path / f"{self.domain_name}_subcat_{subcat_id}_elev_bands.nc"

            if source_elev_file.exists():
                # For now, copy the main elevation bands file for each subcatchment
                shutil.copy2(source_elev_file, target_elev_file)
                self.logger.debug(f"Created elevation bands file for subcatchment {subcat_id}")
            else:
                self.logger.warning(f"Source elevation bands file not found: {source_elev_file}")
                # Create a simple elevation bands file as fallback
                self._create_simple_elevation_bands(target_elev_file, subcat_id)

            return target_elev_file

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error creating elevation bands for subcatchment {subcat_id}: {str(e)}")
            raise

    def _create_simple_elevation_bands(self, target_file: Path, subcat_id: int):
        """
        Create a simple elevation bands file as fallback.

        Args:
            target_file: Path to output elevation bands file
            subcat_id: Subcatchment identifier
        """
        # Get catchment centroid for coordinates
        catchment_path = self._get_catchment_path()
        discretization = self._get_config_value(
            lambda: self.config.domain.discretization,
            default='GRUs',
            dict_key='SUB_GRID_DISCRETIZATION'
        )
        catchment_name_col = self._get_config_value(
            lambda: self.config.paths.catchment_name,
            default='default',
            dict_key='CATCHMENT_SHP_NAME'
        )

        if catchment_name_col == 'default':
            catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_name = catchment_name_col

        catchment = gpd.read_file(catchment_path / catchment_name)

        # Calculate centroid
        if catchment.crs is None:
            catchment.set_crs(epsg=4326, inplace=True)
        catchment_geo = catchment.to_crs(epsg=4326)
        bounds = catchment_geo.total_bounds
        lon = (bounds[0] + bounds[2]) / 2
        lat = (bounds[1] + bounds[3]) / 2

        # Create simple single elevation band
        ds = xr.Dataset(
            coords={
                'longitude': ('longitude', [lon]),
                'latitude': ('latitude', [lat]),
                'elevation_band': ('elevation_band', [1])
            }
        )

        # Add variables (single elevation band covering entire subcatchment)
        for var_name, data, attrs in [
            ('area_frac', [1.0], {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
            ('mean_elev', [1000.0], {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
            ('prec_frac', [1.0], {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
        ]:
            ds[var_name] = xr.DataArray(
                np.array(data).reshape(1, 1, 1),
                dims=['elevation_band', 'latitude', 'longitude'],
                coords=ds.coords,
                attrs=attrs
            )

        # Add coordinate attributes
        ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'}
        ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}

        # Save to file
        encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
        ds.to_netcdf(target_file, encoding=encoding)

        self.logger.info(f"Created simple elevation bands file for subcatchment {subcat_id}")

    def combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """
        Combine outputs from all subcatchments into a unified distributed dataset.

        Loads NetCDF output files from each subcatchment run and merges them
        along a new 'subcatchment' dimension. The combined dataset preserves
        all variables and attributes from individual runs.

        Creates two output files:
        - Full distributed results: all variables with subcatchment dimension
        - Streamflow-only file: just q_routed for easier analysis

        Args:
            outputs: List of (subcatchment_id, output_path) tuples from
                successful FUSE runs.
        """
        self.logger.info(f"Combining outputs from {len(outputs)} subcatchments")

        combined_outputs: Dict[Any, Any] = {}

        # Load and combine all subcatchment outputs
        for subcat_id, output_file in outputs:
            try:
                ds = xr.open_dataset(output_file)

                # Store with subcatchment identifier
                for var_name in ds.data_vars:
                    if var_name not in combined_outputs:
                        combined_outputs[var_name] = {}
                    combined_outputs[var_name][subcat_id] = ds[var_name]

                ds.close()

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Error loading output for subcatchment {subcat_id}: {str(e)}", exc_info=True)
                continue

        # Create combined dataset and save
        if combined_outputs:
            self._create_combined_dataset(combined_outputs)

    def _create_combined_dataset(self, combined_outputs: Dict[str, Dict[int, xr.DataArray]]):
        """
        Create a combined dataset from subcatchment outputs.

        Args:
            combined_outputs: Dictionary of {variable_name: {subcatchment_id: data_array}}
        """
        try:
            self.logger.info("Creating combined dataset from subcatchment outputs")

            if not combined_outputs:
                self.logger.warning("No outputs to combine")
                return

            # Get list of subcatchment IDs and variables
            first_var = list(combined_outputs.keys())[0]
            subcatchment_ids = list(combined_outputs[first_var].keys())
            variable_names = list(combined_outputs.keys())

            self.logger.info(f"Combining {len(subcatchment_ids)} subcatchments with {len(variable_names)} variables")

            # Create the combined dataset
            combined_ds = xr.Dataset()

            # Add subcatchment coordinate
            combined_ds.coords['subcatchment'] = ('subcatchment', subcatchment_ids)

            # Process each variable
            for var_name in variable_names:
                self.logger.debug(f"Processing variable: {var_name}")

                # Collect data arrays for this variable from all subcatchments
                var_arrays = []
                reference_da = None

                for subcat_id in subcatchment_ids:
                    if subcat_id in combined_outputs[var_name]:
                        da = combined_outputs[var_name][subcat_id]
                        var_arrays.append(da)
                        if reference_da is None:
                            reference_da = da
                    else:
                        self.logger.warning(f"Missing data for variable {var_name} in subcatchment {subcat_id}")

                if var_arrays:
                    try:
                        # Concatenate along new subcatchment dimension
                        combined_var = xr.concat(var_arrays, dim='subcatchment')

                        # Assign subcatchment coordinates
                        combined_var = combined_var.assign_coords(subcatchment=subcatchment_ids)

                        # Copy attributes from reference data array
                        if reference_da is not None:
                            combined_var.attrs = reference_da.attrs.copy()

                        # Add to combined dataset
                        combined_ds[var_name] = combined_var

                        self.logger.debug(f"Combined {var_name} with shape: {combined_var.shape}")

                    except Exception as e:  # noqa: BLE001 — model execution resilience
                        self.logger.error(f"Error combining variable {var_name}: {str(e)}", exc_info=True)
                        continue

            # Add global attributes
            combined_ds.attrs.update({
                'model': 'FUSE',
                'spatial_mode': 'distributed',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'n_subcatchments': len(subcatchment_ids),
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Combined FUSE distributed simulation results'
            })

            # Add subcatchment coordinate attributes
            combined_ds.subcatchment.attrs = {
                'long_name': 'Subcatchment identifier',
                'description': 'Unique identifier for each subcatchment in the distributed model'
            }

            # Save the combined dataset
            combined_file = self.output_path / f"{self.domain_name}_{self.experiment_id}_distributed_results.nc"

            # Use standardized encoding utility
            encoding = create_netcdf_encoding(
                combined_ds,
                compression=True,
                custom_encoding={'subcatchment': {'dtype': 'int32'}}
            )

            # Save to netCDF
            combined_ds.to_netcdf(
                combined_file,
                encoding=encoding,
                format='NETCDF4'
            )

            self.logger.info(f"Combined distributed results saved to: {combined_file}")

            # Log summary information
            self.logger.info(f"Combined dataset dimensions: {dict(combined_ds.dims)}")
            self.logger.debug(f"Combined dataset variables: {list(combined_ds.data_vars.keys())}")

            # Also create a simplified streamflow-only file for easier analysis
            if 'q_routed' in combined_ds.data_vars:
                streamflow_file = self.output_path / f"{self.domain_name}_{self.experiment_id}_streamflow_distributed.nc"
                streamflow_ds = combined_ds[['q_routed']].copy()
                streamflow_ds.to_netcdf(streamflow_file, encoding={'q_routed': encoding.get('q_routed', {})})
                self.logger.info(f"Streamflow-only file saved to: {streamflow_file}")

            combined_ds.close()

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error creating combined dataset: {str(e)}")
            raise

    def _get_catchment_path(self) -> Path:
        """
        Get path to catchment shapefiles directory.

        Returns the configured catchment path or defaults to the standard
        project subdirectory 'shapefiles/catchment'.

        Returns:
            Path: Directory containing catchment shapefiles.
        """
        catchment_path = self._get_config_value(
            lambda: self.config.paths.catchment_path,
            default='default',
            dict_key='CATCHMENT_PATH'
        )
        if catchment_path == 'default':
            return self.project_dir / 'shapefiles' / 'catchment'
        return Path(catchment_path)
