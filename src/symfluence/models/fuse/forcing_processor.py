# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcing data processing utilities for FUSE model.

This module contains the FuseForcingProcessor class which handles all forcing data
processing operations including spatial mode transformations, PET calculation,
observation loading, and NetCDF formatting for FUSE model compatibility.

Uses shared utilities from symfluence.core.modeling.utilities for common operations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import geopandas as gpd
import numpy as np
import xarray as xr

from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
from symfluence.core.modeling.utilities import BaseForcingProcessor, DataQualityHandler, ForcingDataProcessor
from symfluence.core.path_resolver import find_catchment_subfile


class FuseForcingProcessor(BaseForcingProcessor):
    """
    Processor for FUSE forcing data with support for lumped, semi-distributed, and distributed modes.

    This class handles:
    - Spatial mode transformations (lumped/semi-distributed/distributed)
    - PET calculation for different spatial configurations
    - Forcing data resampling and alignment
    - NetCDF encoding and formatting
    - Variable mapping and dimension handling

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        project_dir: Root project directory
        forcing_basin_path: Path to basin-averaged forcing data
        forcing_fuse_path: Path to FUSE-specific forcing output
        catchment_path: Path to catchment shapefile
        domain_name: Name of the domain
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        project_dir: Path,
        forcing_basin_path: Path,
        forcing_fuse_path: Path,
        catchment_path: Path,
        domain_name: str,
        calculate_pet_callback,
        calculate_catchment_centroid_callback,
        get_simulation_time_window_callback,
        subset_to_simulation_time_callback,
        forcing_selector: Optional[Callable[[List[Path]], List[Path]]] = None,
    ):
        """
        Initialize the FUSE forcing processor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Root project directory
            forcing_basin_path: Path to basin-averaged forcing data
            forcing_fuse_path: Path to FUSE forcing output directory
            catchment_path: Path to catchment shapefile
            domain_name: Domain name for file naming
            calculate_pet_callback: Callback to parent's _calculate_pet method
            calculate_catchment_centroid_callback: Callback to parent's calculate_catchment_centroid method
            get_simulation_time_window_callback: Callback to parent's _get_simulation_time_window method
            subset_to_simulation_time_callback: Callback to parent's _subset_to_simulation_time method
            forcing_selector: Narrows basin forcing files to this run's HRU
                set (the preprocessor's ``_select_run_forcing``, issue #339)
        """
        super().__init__(
            config=config,
            logger=logger,
            input_path=forcing_basin_path,
            output_path=forcing_fuse_path,
            project_dir=project_dir,
            catchment_path=catchment_path,
            forcing_selector=forcing_selector,
        )
        # Keep original attribute names for backward compatibility
        self.forcing_basin_path = self.input_path
        self.forcing_fuse_path = self.output_path
        self.domain_name = domain_name

        # Callbacks to parent methods
        self._calculate_pet = calculate_pet_callback
        self.calculate_catchment_centroid = calculate_catchment_centroid_callback
        self._get_simulation_time_window = get_simulation_time_window_callback
        self._subset_to_simulation_time = subset_to_simulation_time_callback

    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "FUSE"

    def _prepare_lumped_forcing(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Prepare lumped forcing data using area-weighted spatial average.

        For spatially heterogeneous catchments, area-weighted averaging provides
        more physically accurate basin-average forcing than simple mean.

        Args:
            ds: Input forcing dataset with 'hru' dimension

        Returns:
            Lumped forcing dataset (no 'hru' dimension)
        """
        if 'hru' not in ds.dims:
            return ds

        try:
            # Load catchment to get HRU areas
            catchment = gpd.read_file(self.catchment_path)
            n_hrus = len(catchment)

            # Get area for each HRU
            # First try 'area' or 'area_km2' attribute, then calculate from geometry
            if 'area' in catchment.columns:
                areas = catchment['area'].values
                self.logger.debug("Using 'area' column for area weights")
            elif 'area_km2' in catchment.columns:
                areas = catchment['area_km2'].values
                self.logger.debug("Using 'area_km2' column for area weights")
            else:
                # Calculate from geometry
                # If geographic CRS, reproject to projected CRS for accurate area
                if catchment.crs and catchment.crs.is_geographic:
                    self.logger.debug("Geographic CRS detected, reprojecting to equal-area CRS for area calculation")
                    catchment_projected = catchment.to_crs("EPSG:6933")
                    areas = catchment_projected.geometry.area.values / 1e6  # km²
                else:
                    areas = catchment.geometry.area.values
                    if areas.mean() > 1e6:
                        areas = areas / 1e6

            # Verify we have the right number of areas
            if len(areas) != ds.sizes.get('hru', 0):
                self.logger.warning(
                    f"Area count ({len(areas)}) doesn't match HRU count ({ds.sizes.get('hru', 0)}). "
                    f"Falling back to simple mean."
                )
                return ds.mean(dim='hru')

            # Normalize areas to get weights
            total_area = areas.sum()
            if total_area <= 0:
                self.logger.warning(f"Total catchment area is {total_area}. Falling back to simple mean.")
                return ds.mean(dim='hru')

            weights = areas / total_area

            # Apply area-weighted averaging
            # Create xarray DataArray for weights to ensure proper broadcasting
            weight_da = xr.DataArray(weights, dims=['hru'])

            # Weighted average: sum(data * weight) for each variable
            ds_lumped = (ds * weight_da).sum(dim='hru')

            self.logger.info(
                f"Applied area-weighted aggregation for lumped mode "
                f"({n_hrus} HRUs, total area: {total_area:.2f} km²)"
            )

            return ds_lumped

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(
                f"Failed to apply area-weighted aggregation: {e}. "
                f"Falling back to simple mean."
            , exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return ds.mean(dim='hru')

    def _prepare_semi_distributed_forcing(self, ds: xr.Dataset, subcatchment_dim: str) -> xr.Dataset:
        """Prepare semi-distributed forcing data using subcatchment IDs"""
        self.logger.info(f"Organizing subcatchments along {subcatchment_dim} dimension")

        # Load subcatchment information
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)

        # Reorganize data by subcatchments
        if 'hru' in ds.dims:
            if ds.sizes['hru'] == n_subcatchments:
                ds_subcat = ds
            else:
                ds_subcat = self._map_hrus_to_subcatchments(ds, subcatchments)
        else:
            ds_subcat = self._replicate_to_subcatchments(ds, n_subcatchments)

        return ds_subcat

    def _prepare_distributed_forcing(self, ds: xr.Dataset) -> xr.Dataset:
        """Prepare fully distributed forcing data"""
        self.logger.info("Preparing distributed forcing data")

        # Check target size from available catchment data to ensure alignment
        target_ids = self._load_subcatchment_data()
        n_target = len(target_ids)

        # Use HRU data directly if available
        if 'hru' in ds.dims:
            if ds.sizes['hru'] == n_target:
                return ds
            elif ds.sizes['hru'] == 1:
                self.logger.info(f"Broadcasting single HRU to {n_target} distributed units")
                # Replicate single HRU data to n_target HRUs
                # First squeeze out the singleton hru dimension, then expand to target size
                import numpy as np
                new_ds = xr.Dataset()
                new_ds['time'] = ds['time'].copy()
                new_ds['hru'] = xr.DataArray(range(1, n_target + 1), dims='hru')

                for var in ds.data_vars:
                    if 'hru' in ds[var].dims:
                        # Squeeze the singleton hru dimension and tile to n_target
                        data = ds[var].values
                        if data.ndim == 2 and data.shape[1] == 1:  # (time, hru=1)
                            tiled_data = np.tile(data, (1, n_target))
                            new_ds[var] = xr.DataArray(
                                tiled_data,
                                dims=('time', 'hru'),
                                attrs=ds[var].attrs
                            )
                        else:
                            new_ds[var] = ds[var].copy()
                    else:
                        new_ds[var] = ds[var].copy()

                return new_ds
            else:
                self.logger.warning(f"Mismatch in HRU count: Data has {ds.sizes['hru']}, Target has {n_target}. Proceeding with data as-is.")
                return ds
        else:
            return self._create_distributed_from_catchment(ds)

    def _calculate_distributed_pet(
        self,
        ds: xr.Dataset,
        spatial_mode: str,
        pet_method: str = 'oudin'
    ) -> xr.DataArray:
        """
        Calculate PET for distributed/semi-distributed modes.

        Uses per-subcatchment latitude and temperature for spatially varying PET
        via the Oudin formula. Falls back to single-centroid calculation if
        per-subcatchment latitudes are unavailable or for non-Oudin methods.

        Args:
            ds: xarray dataset with temperature data (must contain 'temp')
            spatial_mode: Spatial mode ('semi_distributed', 'distributed')
            pet_method: PET calculation method

        Returns:
            xr.DataArray: Calculated PET data with same shape as ds['temp']
        """
        self.logger.info(f"Calculating distributed PET for {spatial_mode} mode using {pet_method}")

        try:
            catchment = gpd.read_file(self.catchment_path)

            # Try vectorized per-subcatchment Oudin using center_lat column
            if pet_method == 'oudin' and 'hru' in ds.dims and 'center_lat' in catchment.columns:
                n_hru = ds.sizes['hru']
                lats = catchment['center_lat'].values[:n_hru].astype(np.float64)

                if len(lats) == n_hru:
                    pet = self._vectorized_oudin(ds['temp'], lats)
                    lat_std = np.std(lats)
                    pet_spatial_std = float(pet.std(dim='hru').mean())
                    self.logger.info(
                        f"Spatially varying PET: {n_hru} HRUs, "
                        f"lat range [{lats.min():.2f}, {lats.max():.2f}] (std={lat_std:.3f}), "
                        f"PET spatial std={pet_spatial_std:.4f} mm/day"
                    )
                    return pet
                else:
                    self.logger.warning(
                        f"center_lat count ({len(lats)}) != HRU count ({n_hru}), "
                        f"falling back to single-centroid PET"
                    )

            # Fallback: single-centroid calculation (non-Oudin methods or missing center_lat)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)

            if 'hru' in ds.dims:
                temp_mean = ds['temp'].mean(dim='hru')
                pet_base = self._calculate_pet(temp_mean, mean_lat, pet_method)
                pet = pet_base.broadcast_like(ds['temp'])
                self.logger.debug(f"Calculated distributed PET (single centroid) with shape: {pet.shape}")
            else:
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)

            return pet

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Error calculating distributed PET, falling back to lumped: {str(e)}", exc_info=True)
            catchment = gpd.read_file(self.catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
            return self._calculate_pet(ds['temp'], mean_lat, pet_method)

    def _vectorized_oudin(self, temp: xr.DataArray, lats: np.ndarray) -> xr.DataArray:
        """
        Vectorized Oudin PET across time and HRUs using numpy broadcasting.

        PET = Ra * (T + 5) / 100  where T > -5 C, else 0.

        Args:
            temp: Temperature DataArray with dims (time, hru), in K or C
            lats: Per-HRU latitudes in degrees, shape (n_hru,)

        Returns:
            xr.DataArray: PET in mm/day with same dims as temp
        """
        from symfluence.core.constants import PhysicalConstants

        # Convert temperature to Celsius if needed
        temp_vals = temp.values.astype(np.float64)  # (n_time, n_hru)
        if np.nanmean(temp_vals) > 100.0:
            temp_c = temp_vals - PhysicalConstants.KELVIN_OFFSET
        else:
            temp_c = temp_vals

        # (n_time, 1) day-of-year against (1, n_hru) latitudes
        doy = temp.time.dt.dayofyear.values.astype(np.float64)[:, np.newaxis]
        lat_2d = np.asarray(lats, dtype=np.float64)[np.newaxis, :]
        pet_vals = PETCalculatorMixin.oudin_pet_numpy(temp_c, doy, lat_2d)

        return xr.DataArray(
            pet_vals,
            dims=temp.dims,
            coords=temp.coords,
            attrs={'units': 'mm/day', 'long_name': 'Potential evapotranspiration (Oudin)'}
        )

    def _load_subcatchment_data(self) -> np.ndarray:
        """Load subcatchment information for semi-distributed mode"""
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
            # Use regular HRUs
            catchment = gpd.read_file(self.catchment_path)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

    def _map_hrus_to_subcatchments(self, ds: xr.Dataset, subcatchments: np.ndarray) -> xr.Dataset:
        """Map HRU data to subcatchments for semi-distributed mode"""
        self.logger.info("Mapping HRUs to subcatchments")

        n_hrus = ds.sizes['hru']
        n_subcatchments = len(subcatchments)

        if n_hrus == n_subcatchments:
            return ds.rename({'hru': 'subcatchment'})
        elif n_hrus > n_subcatchments:
            # Aggregate HRUs to subcatchments
            hrus_per_subcat = n_hrus // n_subcatchments
            subcatchment_data = []

            for i in range(n_subcatchments):
                start_idx = i * hrus_per_subcat
                end_idx = start_idx + hrus_per_subcat if i < n_subcatchments - 1 else n_hrus
                subcat_data = ds.isel(hru=slice(start_idx, end_idx)).mean(dim='hru')
                subcatchment_data.append(subcat_data)

            ds_subcat = xr.concat(subcatchment_data, dim='subcatchment')
            ds_subcat['subcatchment'] = subcatchments
            return ds_subcat
        else:
            return self._replicate_to_subcatchments(ds, n_subcatchments)

    def _replicate_to_subcatchments(self, ds: xr.Dataset, n_subcatchments: int) -> xr.Dataset:
        """Replicate lumped data to all subcatchments using broadcasting"""
        self.logger.info(f"Replicating data to {n_subcatchments} subcatchments")

        sub_ids = xr.DataArray(range(1, n_subcatchments + 1), dims='subcatchment', name='subcatchment')
        return ds.broadcast_like(sub_ids).assign_coords(subcatchment=sub_ids)

    def _create_distributed_from_catchment(self, ds: xr.Dataset) -> xr.Dataset:
        """Create HRU-level data from catchment data for distributed mode using broadcasting"""
        self.logger.info("Creating distributed data from catchment data")

        catchment = gpd.read_file(self.catchment_path)
        n_hrus = len(catchment)

        hru_ids = xr.DataArray(range(1, n_hrus + 1), dims='hru', name='hru')
        return ds.broadcast_like(hru_ids).assign_coords(hru=hru_ids)

    def get_encoding_dict(self, fuse_forcing: xr.Dataset) -> Dict[str, Dict]:
        """
        Get encoding dictionary for netCDF output.

        Uses shared DataQualityHandler for fill values and ForcingDataProcessor
        for consistent encoding patterns.

        Args:
            fuse_forcing: xarray Dataset to encode

        Returns:
            Dict: Encoding dictionary for netCDF
        """
        # Use shared utility for data variable encoding
        dqh = DataQualityHandler()
        fill_value = dqh.get_fill_value('float32')

        fdp = ForcingDataProcessor(self.config)
        encoding = fdp.create_encoding_dict(
            fuse_forcing,
            fill_value=fill_value,
            dtype='float32',
            compression=False
        )

        # Add coordinate-specific encoding (FUSE requires float64 for coords)
        for coord in fuse_forcing.coords:
            coord_str = str(coord)
            if coord_str == 'time':
                encoding[coord_str] = {'dtype': 'float64'}
            elif coord_str in ['longitude', 'latitude', 'lon', 'lat']:
                encoding[coord_str] = {'dtype': 'float64'}
            else:
                encoding[coord_str] = {'dtype': 'float32'}

        return encoding
