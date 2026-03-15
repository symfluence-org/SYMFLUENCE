# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
EM-Earth data integrator for remapping and basin averaging.

Handles EM-Earth reanalysis data processing including spatial remapping,
basin-averaged value calculations, and temporal alignment.
"""

import calendar
import logging
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from symfluence.core.constants import PhysicalConstants, UnitConversion
from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir


# Logic moved from DataManager
def _perform_em_earth_remapping_logic(input_file: Path, output_file: Path, basin_shapefile: Path, config: Dict[str, Any]) -> bool:
    """
    Shared logic for remapping EM-Earth data to basin-averaged values.
    """
    try:
        import geopandas as gpd
        import xarray as xr
        from exactextract import exact_extract
        from exactextract.raster import NumPyRasterSource

        # Read EM-Earth data
        em_ds = xr.open_dataset(input_file)

        # Read basin shapefile
        basins_gdf = gpd.read_file(basin_shapefile)

        # Check and reproject shapefile to WGS84 if necessary
        if basins_gdf.crs is None:
            basins_gdf = basins_gdf.set_crs('EPSG:4326')
        elif not basins_gdf.crs.equals(4326) and not basins_gdf.crs.to_string().upper() == 'EPSG:4326':
            basins_gdf = basins_gdf.to_crs('EPSG:4326')

        # Get basin ID column
        basin_id_col = config.get('RIVER_BASIN_SHP_RM_GRUID', 'GRU_ID')

        if basin_id_col not in basins_gdf.columns:
            raise ValueError(f"Basin ID column '{basin_id_col}' not found in shapefile")

        # Create output dataset structure
        basin_ids = sorted(basins_gdf[basin_id_col].unique())

        # Initialize output dataset
        output_ds = xr.Dataset()
        output_ds = output_ds.assign_coords({
            'time': em_ds.time,
            'hru': basin_ids
        })

        # Check if this is a spatially averaged small watershed
        is_single_point = (len(em_ds.lat) == 1 and len(em_ds.lon) == 1)
        small_watershed_flag = em_ds.attrs.get('small_watershed_processing', 0)
        spatial_averaging_flag = em_ds.attrs.get('spatial_averaging_applied', 0)
        is_single_point = is_single_point or (small_watershed_flag == 1) or (spatial_averaging_flag == 1)

        # Compute grid bounds once (half-pixel extension for cell-edge alignment)
        lats = em_ds.lat.values
        lons = em_ds.lon.values
        dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.1
        dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.1
        xmin = float(lons.min()) - dlon / 2
        xmax = float(lons.max()) + dlon / 2
        ymin = float(lats.min()) - dlat / 2
        ymax = float(lats.max()) + dlat / 2

        # Ensure lat is descending (north-up) for raster convention
        lat_ascending = bool(lats[-1] > lats[0]) if len(lats) > 1 else False

        # Build basin_id -> column index mapping
        basin_id_to_idx = {bid: i for i, bid in enumerate(basin_ids)}

        # Prepare GeoDataFrame with basin_id column for exact_extract
        extract_gdf = basins_gdf[[basin_id_col, 'geometry']].copy()

        # Process variables
        for var_name in em_ds.data_vars:
            if var_name in ['prcp', 'prcp_corrected', 'tmean']:
                var_data = em_ds[var_name]

                if is_single_point:
                    # Single point processing - much faster
                    if len(var_data.dims) == 3:
                        time_dim_index = var_data.dims.index('time')
                        if time_dim_index == 0:
                            time_series = var_data.values[:, 0, 0]
                        elif time_dim_index == 1:
                            time_series = var_data.values[0, :, 0]
                        else:
                            time_series = var_data.values[0, 0, :]
                    elif len(var_data.dims) == 1:
                        time_series = var_data.values
                    elif len(var_data.dims) == 2:
                        time_dim_index = var_data.dims.index('time')
                        if time_dim_index == 0:
                            time_series = var_data.values[:, 0]
                        else:
                            time_series = var_data.values[0, :]
                    else:
                        time_series = np.asarray(var_data.values).flatten()

                    time_series = np.asarray(time_series).flatten()
                    basin_values = np.tile(time_series.reshape(-1, 1), (1, len(basin_ids)))

                    output_ds[var_name] = xr.DataArray(
                        basin_values,
                        dims=['time', 'hru'],
                        coords={'time': em_ds.time, 'hru': basin_ids},
                        attrs=var_data.attrs
                    )

                else:
                    # Multi-point processing with exactextract (coverage-weighted)
                    basin_values = np.full((len(em_ds.time), len(basin_ids)), np.nan)

                    for t_idx in range(len(em_ds.time)):
                        grid = var_data.isel(time=t_idx).values.astype(np.float64)

                        # Flip to north-up if lat is ascending
                        if lat_ascending:
                            grid = grid[::-1]

                        raster_src = NumPyRasterSource(
                            grid, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                            nodata=np.nan
                        )

                        results = exact_extract(
                            raster_src, extract_gdf, 'mean',
                            include_cols=[basin_id_col], output='pandas'
                        )

                        for _, row in results.iterrows():
                            bid = row[basin_id_col]
                            if bid in basin_id_to_idx:
                                basin_values[t_idx, basin_id_to_idx[bid]] = row['mean']

                    output_ds[var_name] = xr.DataArray(
                        basin_values,
                        dims=['time', 'hru'],
                        coords={'time': em_ds.time, 'hru': basin_ids},
                        attrs=var_data.attrs
                    )

        # Add metadata
        processing_method = 'single_point_replication' if is_single_point else 'exact_extract'
        output_ds.attrs.update({
            'remapped_from': str(input_file),
            'remapping_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'remapping_method': processing_method,
            'basin_shapefile': str(basin_shapefile),
            'input_grid_size': f"{len(em_ds.lat)}x{len(em_ds.lon)}",
            'output_basins': len(basin_ids),
            'small_watershed_processing': int(is_single_point)
        })

        # Save remapped dataset
        output_ds.to_netcdf(output_file)

        # Close datasets
        em_ds.close()
        output_ds.close()

        return True

    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
        import logging
        logger = logging.getLogger('symfluence')
        logger.error(f"Error remapping {input_file.name}: {str(e)}")
        return False

def _init_worker_pool():
    """
    Initialize worker process for multiprocessing pool.

    This function is called once per worker process when the pool is created.
    It configures HDF5/netCDF4 thread safety to prevent segmentation faults.
    """
    from symfluence.core.hdf5_safety import apply_worker_environment
    apply_worker_environment()


def _remap_em_earth_worker(args):
    """Worker function for parallel EM-Earth remapping."""
    input_file_str, output_file_str, basin_shapefile_str, config = args
    return _perform_em_earth_remapping_logic(
        Path(input_file_str),
        Path(output_file_str),
        Path(basin_shapefile_str),
        config
    )

class EMEarthIntegrator(ConfigMixin):
    """
    Handles the integration of EM-Earth data into the forcing dataset.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

    def integrate_em_earth_data(self):
        """
        Integrate EM-Earth precipitation and temperature data with primary forcing dataset.
        """
        self.logger.debug("Starting EM-Earth data integration")

        try:
            # Check if EM-Earth data exists
            em_earth_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data_em_earth'
            if not em_earth_dir.exists():
                self.logger.warning("EM-Earth data directory not found, skipping integration")
                return

            # Find EM-Earth files
            em_earth_files = list(em_earth_dir.glob("watershed_subset_*.nc"))
            if not em_earth_files:
                self.logger.warning("No EM-Earth files found, skipping integration")
                return

            # Process and remap EM-Earth data
            self._remap_em_earth_to_basin_grid()

            # Replace precipitation and temperature in basin-averaged data
            self._replace_forcing_variables_with_em_earth()

            self.logger.info("EM-Earth data integration completed successfully")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error during EM-Earth data integration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _remap_em_earth_to_basin_grid(self):
        """
        Remap EM-Earth data to match the basin grid used by the primary forcing dataset.
        """
        self.logger.debug("Remapping EM-Earth data to basin grid")

        try:
            # Get basin shapefile for remapping
            subbasins_name = self._get_config_value(lambda: self.config.paths.river_basins_name, dict_key='RIVER_BASINS_NAME')
            if subbasins_name == 'default':
                subbasins_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_riverBasins_{self._get_config_value(lambda: self.config.domain.definition_method, dict_key='DOMAIN_DEFINITION_METHOD')}.shp"

            basin_shapefile = self.project_dir / "shapefiles/river_basins" / subbasins_name

            if not basin_shapefile.exists():
                raise FileNotFoundError(f"Basin shapefile not found: {basin_shapefile}")

            # Create output directory for remapped EM-Earth data
            remapped_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'em_earth_remapped'
            remapped_dir.mkdir(parents=True, exist_ok=True)

            # Find EM-Earth files
            em_earth_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'raw_data_em_earth'
            em_earth_files = sorted(em_earth_dir.glob("watershed_subset_*.nc"))

            if not em_earth_files:
                self.logger.warning("No EM-Earth files found for remapping")
                return

            # Filter out files that already exist (unless forcing rerun)
            files_to_process = []
            for em_file in em_earth_files:
                output_file = remapped_dir / f"remapped_{em_file.name}"
                if not output_file.exists() or self._get_config_value(lambda: self.config.system.force_run_all_steps, default=False, dict_key='FORCE_RUN_ALL_STEPS'):
                    files_to_process.append(em_file)

            if not files_to_process:
                self.logger.debug("All EM-Earth files already remapped, skipping")
                return

            self.logger.debug(f"Found {len(files_to_process)} EM-Earth files to remap")

            # Check if we should use parallel processing
            num_processes = self._get_config_value(lambda: self.config.system.num_processes, default=1, dict_key='NUM_PROCESSES')
            use_parallel = num_processes > 1 and len(files_to_process) > 1

            if use_parallel:
                self.logger.debug(f"Using parallel processing with {num_processes} workers")
                self._remap_em_earth_files_parallel(files_to_process, basin_shapefile, remapped_dir, num_processes)
            else:
                self.logger.debug("Using sequential processing")
                self._remap_em_earth_files_sequential(files_to_process, basin_shapefile, remapped_dir)

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error remapping EM-Earth data: {str(e)}")
            raise

    def _remap_em_earth_files_sequential(self, files_to_process, basin_shapefile, remapped_dir):
        """Remap EM-Earth files sequentially."""
        for i, em_file in enumerate(files_to_process, 1):
            output_file = remapped_dir / f"remapped_{em_file.name}"

            self.logger.info(f"Processing file {i}/{len(files_to_process)}: {em_file.name}")

            try:
                # Direct call to the module level function
                _perform_em_earth_remapping_logic(em_file, output_file, basin_shapefile, self.config)
                self.logger.info(f"✓ Successfully remapped {em_file.name}")
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"✗ Failed to remap {em_file.name}: {str(e)}")
                continue

    def _remap_em_earth_files_parallel(self, files_to_process, basin_shapefile, remapped_dir, num_processes):
        """Remap EM-Earth files in parallel."""
        num_workers = min(num_processes, len(files_to_process))

        self.logger.debug(f"Starting parallel EM-Earth remapping with {num_workers} workers")

        # Create shared arguments for workers
        worker_args = []
        for em_file in files_to_process:
            output_file = remapped_dir / f"remapped_{em_file.name}"
            worker_args.append((
                str(em_file),
                str(output_file),
                str(basin_shapefile),
                self.config.copy()
            ))

        start_time = time.time()

        # Use initializer to configure HDF5 safety in each worker
        with mp.Pool(processes=num_workers, initializer=_init_worker_pool) as pool:
            try:
                result = pool.map_async(
                    _remap_em_earth_worker,
                    worker_args,
                    chunksize=1
                )

                results = []
                while True:
                    try:
                        results = result.get(timeout=30)
                        break
                    except mp.TimeoutError:
                        if hasattr(result, '_number_left'):
                            completed = len(worker_args) - result._number_left
                            self.logger.debug(f"Parallel remapping progress: {completed}/{len(worker_args)} files completed")
                        continue

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error in parallel processing: {str(e)}")
                pool.terminate()
                pool.join()
                raise

        successful = sum(results)
        failed = len(files_to_process) - successful
        elapsed_time = time.time() - start_time

        self.logger.debug(f"Parallel EM-Earth remapping completed in {elapsed_time:.1f} seconds")
        self.logger.debug(f"Success rate: {successful}/{len(files_to_process)} files ({successful/len(files_to_process)*100:.1f}%)")

        if failed > 0:
            self.logger.warning(f"{failed} files failed to process")

        if successful == 0:
            raise ValueError("No EM-Earth files were successfully remapped")

    def _replace_forcing_variables_with_em_earth(self):
        """Replace precipitation and temperature variables in basin-averaged data with EM-Earth values."""
        self.logger.debug("Replacing forcing variables with EM-Earth data")

        try:

            basin_data_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'basin_averaged_data'
            if not basin_data_dir.exists():
                raise FileNotFoundError(f"Basin-averaged data directory not found: {basin_data_dir}")

            remapped_dir = resolve_data_subdir(self.project_dir, 'forcing') / 'em_earth_remapped'
            if not remapped_dir.exists():
                raise FileNotFoundError(f"Remapped EM-Earth directory not found: {remapped_dir}")

            forcing_files = list(basin_data_dir.glob("*.nc"))
            em_earth_files = list(remapped_dir.glob("remapped_watershed_subset_*.nc"))

            if not forcing_files:
                self.logger.warning("No basin-averaged forcing files found")
                return

            if not em_earth_files:
                self.logger.warning("No remapped EM-Earth files found")
                return

            em_earth_lookup = {}
            for em_file in em_earth_files:
                year_month = em_file.name.split('_')[-1].replace('.nc', '')
                em_earth_lookup[year_month] = em_file

            for forcing_file in forcing_files:
                try:
                    self._update_single_forcing_file(forcing_file, em_earth_lookup)
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.warning(f"Failed to update {forcing_file.name}: {str(e)}")
                    continue

            self.logger.debug("Successfully replaced forcing variables with EM-Earth data")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error replacing forcing variables: {str(e)}")
            raise

    def _update_single_forcing_file(self, forcing_file: Path, em_earth_lookup: Dict[str, Path]):
        """Update a single forcing file with EM-Earth precipitation and temperature data."""
        try:
            import xarray as xr

            forcing_ds = xr.open_dataset(forcing_file)

            # Check if time dimension/coordinate exists
            if 'time' not in forcing_ds.dims and 'time' not in forcing_ds.coords:
                self.logger.warning(f"Skipping {forcing_file.name}: no 'time' dimension or coordinate found")
                forcing_ds.close()
                return

            # Access time safely - use bracket notation to work with both coordinates and variables
            try:
                start_time = forcing_ds['time'].min().values
                end_time = forcing_ds['time'].max().values
            except (KeyError, AttributeError) as e:
                self.logger.warning(f"Skipping {forcing_file.name}: unable to access time data ({e})")
                forcing_ds.close()
                return

            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)

            em_datasets = []
            for year_month, em_file in em_earth_lookup.items():
                year = int(year_month[:4])
                month = int(year_month[4:])

                em_start = datetime(year, month, 1)
                em_end = datetime(year, month, calendar.monthrange(year, month)[1])

                if (em_start <= end_dt.to_pydatetime() and em_end >= start_dt.to_pydatetime()):
                    em_ds = xr.open_dataset(em_file)
                    em_datasets.append(em_ds)

            if not em_datasets:
                self.logger.warning(f"No matching EM-Earth data for {forcing_file.name}")
                return

            em_combined = xr.concat(em_datasets, dim='time')
            em_combined = em_combined.sortby('time')
            em_combined = em_combined.sel(time=slice(start_time, end_time))

            updated_ds = forcing_ds.copy(deep=True)
            variable_mapping = {
                'prcp': ['precipitation_flux', 'pcp', 'precipitation', 'PRCP', 'prcp'],
                'prcp_corrected': ['precipitation_flux', 'pcp', 'precipitation', 'PRCP', 'prcp'],
                'tmean': ['air_temperature', 'tmp', 'temperature', 'TEMP', 'tmean', 'tas']
            }

            for em_var, forcing_vars in variable_mapping.items():
                if em_var in em_combined.data_vars:
                    for forcing_var in forcing_vars:
                        if forcing_var in updated_ds.data_vars:
                            self.logger.debug(f"Replacing {forcing_var} with EM-Earth {em_var}")

                            if em_var in ('prcp', 'prcp_corrected'):
                                em_data_interp = em_combined[em_var].interp(time=forcing_ds['time'], method='nearest')
                            else:
                                em_data_interp = em_combined[em_var].interp(time=forcing_ds['time'])

                            if em_var in ['prcp', 'prcp_corrected']:
                                current_units = str(updated_ds[forcing_var].attrs.get('units', ''))
                                original_max = float(em_data_interp.max())

                                if 'kg m-2 s-1' in current_units or 'kg m**-2 s**-1' in current_units:
                                    em_data_interp = em_data_interp / UnitConversion.SECONDS_PER_HOUR
                                    em_data_interp.attrs['units'] = 'kg m-2 s-1'
                                elif 'm s-1' in current_units:
                                    em_data_interp = em_data_interp / (UnitConversion.SECONDS_PER_HOUR * 1000)
                                    em_data_interp.attrs['units'] = 'm s-1'
                                elif 'mm s-1' in current_units or 'mm/s' in current_units:
                                    em_data_interp = em_data_interp / UnitConversion.SECONDS_PER_HOUR
                                    em_data_interp.attrs['units'] = 'mm/s'
                                else:
                                    em_data_interp.attrs['units'] = 'mm/h'

                                converted_max = float(em_data_interp.max())
                                if converted_max == 0.0 and original_max > 0.0:
                                    raise ValueError("Precipitation conversion resulted in zero values")

                            elif em_var == 'tmean':
                                current_units = str(updated_ds[forcing_var].attrs.get('units', ''))
                                if 'K' in current_units and 'Celsius' not in current_units.lower():
                                    em_data_interp = em_data_interp + PhysicalConstants.KELVIN_OFFSET
                                    # Ensure the interpolated data knows its new units
                                    em_data_interp.attrs['units'] = 'K'
                                else:
                                    # Already Celsius or unknown, ensure attribute matches values
                                    em_data_interp.attrs['units'] = 'degC'

                            updated_ds[forcing_var] = em_data_interp

                            updated_ds[forcing_var].attrs.update({
                                'source': f'EM-Earth {em_var}',
                                'replacement_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'original_units': 'EM-Earth: mm/h (prcp) or °C (temp)',
                                'converted_units': current_units
                            })
                            break

            updated_ds.attrs.update({
                'em_earth_replacement': 1,
                'em_earth_replacement_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'em_earth_variables_replaced': 'precipitation, temperature'
            })

            temp_file = forcing_file.with_suffix('.nc.temp')
            try:
                updated_ds.to_netcdf(temp_file)
                forcing_ds.close()
                updated_ds.close()
                for em_ds in em_datasets:
                    em_ds.close()

                if forcing_file.exists():
                    forcing_file.unlink()
                temp_file.replace(forcing_file)

                self.logger.debug(f"Successfully updated {forcing_file.name} with EM-Earth data")

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as write_error:
                if temp_file.exists():
                    temp_file.unlink()
                raise write_error

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error updating {forcing_file.name}: {str(e)}")
            raise
