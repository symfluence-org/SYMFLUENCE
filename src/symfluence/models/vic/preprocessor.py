# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC Model Preprocessor

Handles preparation of VIC model inputs including:
- Domain file (grid definition)
- Parameter file (soil and vegetation parameters)
- Forcing files (meteorological data)
- Global parameter file (model control settings)
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import FileOperationError, ValidationError
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.modeling.model_ready.forcing_reader import (
    resample_canonical_forcing,
    validated_timestep_seconds,
)
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("VIC")
class VICPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a VIC model run.

    VIC requires:
    - Domain file (vic_domain.nc): Grid mask, area, coordinates
    - Parameter file (vic_params.nc): Soil and vegetation parameters
    - Forcing files: Meteorological data in VIC NetCDF format
    - Global parameter file: Model control settings
    """

    MODEL_NAME = "VIC"

    def __init__(self, config, logger):
        """
        Initialize the VIC preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # VIC directories — use the standard project layout from base class:
        #   settings  → settings/VIC/          (self.setup_dir)
        #   forcing   → data/forcing/VIC_input/ (self.forcing_dir)
        #   parameters → settings/VIC/parameters/
        self.settings_dir = self.setup_dir
        self.params_dir = self.setup_dir / "parameters"

        # Get VIC-specific settings from config
        # Resolve spatial mode: explicit config > inferred from domain method
        configured_mode = self._get_config_value(
            lambda: self.config.model.vic.spatial_mode,
            default=None,
            dict_key='VIC_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            # Infer from domain definition method (consistent with other models)
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped',
                dict_key='DOMAIN_DEFINITION_METHOD'
            )
            if domain_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        logger.info(f"VIC spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        """
        Run the complete VIC preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting VIC preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate domain file
            self._generate_domain_file()

            # Generate parameter file
            self._generate_parameter_file()

            # Generate forcing files
            self._generate_forcing_files()

            # Generate global parameter file
            self._generate_global_param_file()

            logger.info("VIC preprocessing complete.")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"VIC preprocessing failed: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create VIC input directory structure."""
        dirs = [
            self.settings_dir,
            self.forcing_dir,
            self.params_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created VIC directories: settings={self.settings_dir}, forcing={self.forcing_dir}")

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        catchment_path = self._find_basin_shapefile(required=True)
        gdf = gpd.read_file(catchment_path)
        if gdf.empty or gdf.crs is None:
            raise ValidationError("VIC requires nonempty catchment geometry with a CRS")
        geographic = gdf.to_crs(4326)
        projected = geographic.to_crs(geographic.estimate_utm_crs())
        center = gpd.GeoSeries([projected.geometry.union_all().centroid], crs=projected.crs).to_crs(4326).iloc[0]
        areas = projected.geometry.area.to_numpy()
        if not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValidationError("VIC catchment area must be finite and positive")
        if 'elev_mean' not in gdf:
            raise ValidationError("VIC catchment geometry requires elev_mean")
        elev = float(np.average(gdf.elev_mean.to_numpy(), weights=areas))
        if not np.isfinite(elev):
            raise ValidationError("VIC catchment elevation is nonfinite")
        return {'lat': center.y, 'lon': center.x, 'area_m2': float(areas.sum()), 'elev': elev}

    def _compute_elevation_bands(self, n_bands: int = 5) -> Dict:
        """
        Compute elevation bands from DEM raster for VIC snow bands.

        Divides the catchment into equal-area elevation bands based on the
        hypsometric distribution from the DEM. This is critical for mountain
        catchments where elevation-dependent snowmelt timing dominates the
        hydrograph.

        Args:
            n_bands: Number of elevation bands (default 5)

        Returns:
            Dict with 'elevations', 'area_fracs', 'pfactors' arrays
        """
        try:
            import rasterio
            dem_dir = self.project_attributes_dir / 'elevation' / 'dem'
            dem_files = list(dem_dir.glob('*.tif'))
            if not dem_files:
                logger.warning("No DEM raster found, using single elevation band")
                props = self._get_catchment_properties()
                return {
                    'elevations': np.array([props.get('elev', 1000.0)]),
                    'area_fracs': np.array([1.0]),
                    'pfactors': np.array([1.0]),
                }

            with rasterio.open(dem_files[0]) as src:
                dem = src.read(1)
                valid = dem[dem > 0].flatten()

            if len(valid) < n_bands:
                props = self._get_catchment_properties()
                return {
                    'elevations': np.array([props.get('elev', 1000.0)]),
                    'area_fracs': np.array([1.0]),
                    'pfactors': np.array([1.0]),
                }

            # Create equal-area bands using percentiles
            percentiles = np.linspace(0, 100, n_bands + 1)
            band_edges = np.percentile(valid, percentiles)

            elevations = np.zeros(n_bands)
            area_fracs = np.zeros(n_bands)
            pfactors = np.zeros(n_bands)

            mean_elev = np.mean(valid)
            total_pixels = len(valid)

            for b in range(n_bands):
                lo, hi = band_edges[b], band_edges[b + 1]
                if b == n_bands - 1:
                    in_band = (valid >= lo) & (valid <= hi)
                else:
                    in_band = (valid >= lo) & (valid < hi)

                band_pixels = valid[in_band]
                if len(band_pixels) == 0:
                    elevations[b] = (lo + hi) / 2
                    area_fracs[b] = 1.0 / n_bands
                else:
                    elevations[b] = np.mean(band_pixels)
                    area_fracs[b] = len(band_pixels) / total_pixels

                # Orographic precipitation factor: increase with elevation
                # ~5% per 100m above mean, decrease below mean
                elev_diff = elevations[b] - mean_elev
                pfactor_per_km = self._get_config_value(
                    lambda: self.config.model.vic.pfactor_per_km,
                    dict_key='VIC_PFACTOR_PER_KM',
                )
                pfactors[b] = 1.0 + pfactor_per_km * elev_diff

            # Ensure area fractions sum to 1
            area_fracs = area_fracs / area_fracs.sum()

            # Clamp pfactors to reasonable range
            pfactors = np.clip(pfactors, 0.5, 2.0)

            logger.info(f"Computed {n_bands} elevation bands from DEM "
                        f"({valid.min():.0f}-{valid.max():.0f}m)")
            return {
                'elevations': elevations,
                'area_fracs': area_fracs,
                'pfactors': pfactors,
            }

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Failed to compute elevation bands from DEM: {e}", exc_info=True)
            props = self._get_catchment_properties()
            return {
                'elevations': np.array([props.get('elev', 1000.0)]),
                'area_fracs': np.array([1.0]),
                'pfactors': np.array([1.0]),
            }

    def _generate_domain_file(self) -> None:
        """
        Generate the VIC domain file.

        The domain file defines the model grid with:
        - mask: Grid cell mask (1 = active)
        - area: Grid cell area (m²)
        - frac: Fraction of grid cell that is land
        - lat/lon: Coordinate arrays
        """
        logger.info("Generating VIC domain file...")

        domain_file = self._get_config_value(lambda: self.config.model.vic.domain_file, dict_key='VIC_DOMAIN_FILE')
        domain_path = self.params_dir / domain_file

        props = self._get_catchment_properties()

        if self.spatial_mode == SpatialMode.LUMPED:
            # Single cell domain
            self._generate_lumped_domain(domain_path, props)
        else:
            # Distributed domain
            self._generate_distributed_domain(domain_path, props)

        logger.info(f"Domain file written: {domain_path}")

    def _generate_lumped_domain(self, domain_path: Path, props: Dict) -> None:
        """Generate a single-cell domain for lumped mode."""
        # Create 1x1 grid
        lat = np.array([props['lat']])
        lon = np.array([props['lon']])

        # Create dataset
        ds = xr.Dataset(
            {
                'mask': (['lat', 'lon'], np.array([[1]], dtype=np.int32)),
                'area': (['lat', 'lon'], np.array([[props['area_m2']]])),
                'frac': (['lat', 'lon'], np.array([[1.0]])),
            },
            coords={
                'lat': lat,
                'lon': lon,
            }
        )

        # Add attributes
        ds.attrs['title'] = f'VIC domain file for {self.domain_name}'
        ds.attrs['history'] = f'Created by SYMFLUENCE on {datetime.now().isoformat()}'

        ds['mask'].attrs['long_name'] = 'domain mask'
        ds['mask'].attrs['comment'] = '1 = active grid cell'

        ds['area'].attrs['long_name'] = 'grid cell area'
        ds['area'].attrs['units'] = 'm2'

        ds['frac'].attrs['long_name'] = 'land fraction'
        ds['frac'].attrs['units'] = '1'

        ds['lat'].attrs['long_name'] = 'latitude'
        ds['lat'].attrs['units'] = 'degrees_north'

        ds['lon'].attrs['long_name'] = 'longitude'
        ds['lon'].attrs['units'] = 'degrees_east'

        ds.to_netcdf(domain_path)

    def _generate_distributed_domain(self, domain_path: Path, props: Dict) -> None:
        """Generate a distributed domain from catchment shapefile."""
        try:
            catchment_path = self.get_catchment_path()
            if not catchment_path.exists():
                logger.warning("Catchment shapefile not found, falling back to lumped domain")
                self._generate_lumped_domain(domain_path, props)
                return

            gdf = gpd.read_file(catchment_path)

            # Get bounds
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

            # Determine grid resolution (~1km = 0.01 degrees, or use config)
            resolution = 0.01  # degrees

            # Create coordinate arrays
            lon = np.arange(bounds[0], bounds[2] + resolution, resolution)
            lat = np.arange(bounds[1], bounds[3] + resolution, resolution)

            # Limit grid size
            if len(lon) > 100 or len(lat) > 100:
                # Use coarser resolution
                resolution = max((bounds[2] - bounds[0]) / 50, (bounds[3] - bounds[1]) / 50)
                lon = np.arange(bounds[0], bounds[2] + resolution, resolution)
                lat = np.arange(bounds[1], bounds[3] + resolution, resolution)

            # Create meshgrid for rasterization
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)

            # Simple rasterization: check if grid cell center is in catchment
            from shapely.geometry import Point
            catchment_union = gdf.unary_union

            mask = np.zeros((len(lat), len(lon)), dtype=np.int32)
            for i in range(len(lat)):
                for j in range(len(lon)):
                    point = Point(lon[j], lat[i])
                    if catchment_union.contains(point):
                        mask[i, j] = 1

            # If no cells were found, ensure at least the centroid cell is active
            if mask.sum() == 0:
                center_lat_idx = len(lat) // 2
                center_lon_idx = len(lon) // 2
                mask[center_lat_idx, center_lon_idx] = 1

            # Calculate cell area (approximate using center latitude)
            # Area = resolution² * cos(lat) * 111.32² km² -> m²
            lat_rad = np.radians(lat_mesh)
            cell_area = (resolution * 111320) ** 2 * np.cos(lat_rad)

            # Create dataset
            ds = xr.Dataset(
                {
                    'mask': (['lat', 'lon'], mask),
                    'area': (['lat', 'lon'], cell_area),
                    'frac': (['lat', 'lon'], mask.astype(float)),
                },
                coords={
                    'lat': lat,
                    'lon': lon,
                }
            )

            # Add attributes
            ds.attrs['title'] = f'VIC domain file for {self.domain_name}'
            ds.attrs['history'] = f'Created by SYMFLUENCE on {datetime.now().isoformat()}'
            ds.attrs['resolution'] = f'{resolution} degrees'

            ds['mask'].attrs['long_name'] = 'domain mask'
            ds['area'].attrs['long_name'] = 'grid cell area'
            ds['area'].attrs['units'] = 'm2'
            ds['frac'].attrs['long_name'] = 'land fraction'
            ds['lat'].attrs['units'] = 'degrees_north'
            ds['lon'].attrs['units'] = 'degrees_east'

            ds.to_netcdf(domain_path)
            logger.info(f"Created distributed domain: {mask.sum()} active cells")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Error creating distributed domain: {e}, falling back to lumped", exc_info=True)
            self._generate_lumped_domain(domain_path, props)

    def _generate_parameter_file(self) -> None:
        """
        Generate the VIC parameter file.

        Contains soil and vegetation parameters for each grid cell.
        Key parameters:
        - infilt: Infiltration parameter
        - Ds, Dsmax, Ws: Baseflow parameters
        - depth1, depth2, depth3: Soil layer depths
        - Ksat: Saturated hydraulic conductivity
        """
        logger.info("Generating VIC parameter file...")

        params_file = self._get_config_value(lambda: self.config.model.vic.params_file, dict_key='VIC_PARAMS_FILE')
        params_path = self.params_dir / params_file

        # Load domain to get coordinates
        domain_file = self._get_config_value(lambda: self.config.model.vic.domain_file, dict_key='VIC_DOMAIN_FILE')
        domain_path = self.params_dir / domain_file

        if not domain_path.exists():
            raise RuntimeError(f"Domain file not found: {domain_path}. Run domain generation first.")

        domain_ds = xr.open_dataset(domain_path)
        lat = domain_ds['lat'].values
        lon = domain_ds['lon'].values
        mask = domain_ds['mask'].values

        nlat, nlon = len(lat), len(lon)
        nlayers = 3  # VIC typically uses 3 soil layers

        props = self._get_catchment_properties()

        # Default parameter values (typical VIC calibration ranges)
        defaults = {
            'infilt': 0.2,      # Variable infiltration curve parameter [0.001-0.9]
            'Ds': 0.01,         # Fraction of Dsmax where nonlinear baseflow begins [0-1]
            'Dsmax': 10.0,      # Maximum velocity of baseflow [mm/day]
            'Ws': 0.9,          # Fraction of max soil moisture where nonlinear baseflow occurs [0.1-1]
            'c': 2.0,           # Exponent in baseflow curve
            'expt': np.array([8.0, 10.0, 12.0]),   # Soil layer exponent
            'Ksat': np.array([100.0, 50.0, 10.0]), # Saturated hydraulic conductivity [mm/day]
            'phi_s': np.array([-0.3, -0.5, -1.0]), # Soil bubbling pressure [m]
            'depth': np.array([0.1, 0.5, 1.5]),    # Soil layer depths [m]
            'bulk_density': np.array([1500.0, 1600.0, 1700.0]),  # Bulk density [kg/m³]
            'soil_density': np.array([2650.0, 2650.0, 2650.0]),  # Particle density [kg/m³]
            'Wcr_FRACT': np.array([0.7, 0.7, 0.7]),  # Critical moisture fraction
            # NOTE: init_moist is computed below from depth and porosity
            'Wpwp_FRACT': np.array([0.3, 0.3, 0.3]), # Wilting point fraction
            'rough': 0.001,     # Surface roughness [m]
            'snow_rough': 0.0005,  # Snow roughness [m]
            'annual_prec': 1000.0,  # Annual precipitation [mm]
            'avg_T': 5.0,       # Average temperature [°C]
            'elev': props.get('elev', 1000.0),     # Elevation [m]
            'off_gmt': -7.0,    # Offset from GMT [hours]
            'fs_active': 1,     # Frozen soil flag
            'July_Tavg': 15.0,  # July average temperature [°C]
        }

        # VIC 5 image driver dimensions
        # The image driver reserves the last library class for bare soil.
        # Keep the existing simplified grass tile plus a zero-area bare class;
        # Nveg counts active vegetated tiles, not library entries.
        nveg = 2
        nroot = 2      # Root zones

        # Compute elevation bands from DEM
        n_snow_bands = self._get_config_value(lambda: self.config.model.vic.n_snow_bands, dict_key='VIC_N_SNOW_BANDS')
        band_data = self._compute_elevation_bands(n_bands=n_snow_bands)
        nband = len(band_data['elevations'])

        # Create parameter arrays
        params = {}

        # Required VIC 5 grid variables
        params['run_cell'] = mask.copy()
        params['gridcell'] = np.arange(1, nlat * nlon + 1).reshape(nlat, nlon)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        params['lats'] = lat_mesh
        params['lons'] = lon_mesh

        # Scalar parameters (per grid cell)
        for var in ['infilt', 'Ds', 'Dsmax', 'Ws', 'c', 'rough', 'snow_rough',
                    'annual_prec', 'avg_T', 'elev', 'off_gmt', 'fs_active', 'July_Tavg']:
            params[var] = np.full((nlat, nlon), defaults[var])

        # Additional required scalar parameters
        params['dp'] = np.full((nlat, nlon), 4.0)             # Soil thermal damping depth [m]
        params['resid_moist'] = np.zeros((nlayers, nlat, nlon))  # Residual moisture [mm/mm]
        params['quartz'] = np.zeros((nlayers, nlat, nlon))
        for layer in range(nlayers):
            params['quartz'][layer, :, :] = 0.5
            params['resid_moist'][layer, :, :] = 0.02
        params['bubble'] = np.zeros((nlayers, nlat, nlon))
        for layer in range(nlayers):
            params['bubble'][layer, :, :] = 5.0  # Bubbling pressure [cm]

        # Compute init_moist as 50% of max soil capacity (depth × porosity × 1000)
        # to avoid exceeding capacity, which crashes VIC when depths shrink during calibration
        porosity = 1.0 - defaults['bulk_density'] / defaults['soil_density']
        max_moist = defaults['depth'] * porosity * 1000.0  # mm
        defaults['init_moist'] = 0.5 * max_moist
        logger.info(f"Computed init_moist from depth & porosity: {defaults['init_moist']}")

        # Layer parameters
        for var in ['expt', 'Ksat', 'phi_s', 'init_moist', 'depth',
                    'bulk_density', 'soil_density', 'Wcr_FRACT', 'Wpwp_FRACT']:
            arr = np.zeros((nlayers, nlat, nlon))
            for layer in range(nlayers):
                arr[layer, :, :] = defaults[var][layer]
            params[var] = arr

        # Vegetation tiling parameters (veg_class dimension)
        params['Nveg'] = np.ones((nlat, nlon), dtype=np.int32)
        params['Cv'] = np.zeros((nveg, nlat, nlon))
        params['Cv'][0, :, :] = 1.0  # Single veg class covers 100%

        # Vegetation library parameters (per veg_class)
        # Using grassland defaults (overstory=0)
        params['overstory'] = np.zeros((nveg, nlat, nlon), dtype=np.int32)
        params['rarc'] = np.full((nveg, nlat, nlon), 25.0)        # Architectural resistance [s/m]
        params['rmin'] = np.full((nveg, nlat, nlon), 150.0)       # Min stomatal resistance [s/m]
        params['wind_h'] = np.full((nveg, nlat, nlon), 10.0)      # Wind measurement height [m]
        params['RGL'] = np.full((nveg, nlat, nlon), 100.0)        # Min shortwave for transpiration [W/m²]
        params['rad_atten'] = np.full((nveg, nlat, nlon), 0.5)    # Radiation attenuation factor
        params['wind_atten'] = np.full((nveg, nlat, nlon), 0.5)   # Wind attenuation through canopy
        params['trunk_ratio'] = np.full((nveg, nlat, nlon), 0.0)  # Trunk ratio (0 for grass)

        # Monthly vegetation parameters (veg_class × month)
        # Seasonal LAI cycle peaking in summer
        monthly_lai = np.array([0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5])
        monthly_albedo = np.array([0.2, 0.2, 0.18, 0.16, 0.15, 0.15, 0.15, 0.15, 0.16, 0.18, 0.2, 0.2])
        monthly_veg_rough = np.array([0.01, 0.01, 0.02, 0.03, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.01, 0.01])
        monthly_displacement = monthly_veg_rough * 6.67  # ~2/3 of canopy height

        params['LAI'] = np.zeros((nveg, 12, nlat, nlon))
        params['albedo'] = np.zeros((nveg, 12, nlat, nlon))
        params['veg_rough'] = np.zeros((nveg, 12, nlat, nlon))
        params['displacement'] = np.zeros((nveg, 12, nlat, nlon))
        for m in range(12):

            params['LAI'][0, m, :, :] = monthly_lai[m]
            params['albedo'][0, m, :, :] = monthly_albedo[m]
            params['veg_rough'][0, m, :, :] = monthly_veg_rough[m]
            params['displacement'][0, m, :, :] = monthly_displacement[m]

        # Reserved bare-soil library entry: no canopy or roots. A nonzero
        # roughness avoids invalid aerodynamic inputs even at zero tile area.
        params['albedo'][-1, :, :, :] = 0.2
        params['veg_rough'][-1, :, :, :] = defaults['rough']
        params['rarc'][-1, :, :] = 0.0
        params['rmin'][-1, :, :] = 0.0

        # Root zone parameters (veg_class × root_zone)
        params['root_depth'] = np.zeros((nveg, nroot, nlat, nlon))
        params['root_depth'][0, 0, :, :] = 0.3
        params['root_depth'][0, 1, :, :] = 0.7
        params['root_fract'] = np.zeros((nveg, nroot, nlat, nlon))
        params['root_fract'][0, 0, :, :] = 0.6
        params['root_fract'][0, 1, :, :] = 0.4

        # Snow band parameters (from DEM-derived elevation bands)
        params['AreaFract'] = np.zeros((nband, nlat, nlon))
        params['elevation'] = np.zeros((nband, nlat, nlon))
        params['Pfactor'] = np.zeros((nband, nlat, nlon))
        for b in range(nband):
            params['AreaFract'][b, :, :] = band_data['area_fracs'][b]
            params['elevation'][b, :, :] = band_data['elevations'][b]
            params['Pfactor'][b, :, :] = band_data['pfactors'][b]
        logger.info(f"Created {nband} elevation bands: "
                    f"{[f'{e:.0f}m ({a:.1%})' for e, a in zip(band_data['elevations'], band_data['area_fracs'])]}")

        # Create dataset
        ds = xr.Dataset(coords={
            'lat': lat, 'lon': lon,
            'nlayer': np.arange(nlayers),
            'root_zone': np.arange(nroot),
            'veg_class': np.arange(nveg),
            'snow_band': np.arange(nband),
            'month': np.arange(1, 13),
        })

        # Required grid cell variables
        ds['run_cell'] = (['lat', 'lon'], params['run_cell'])
        ds['gridcell'] = (['lat', 'lon'], params['gridcell'])
        ds['lats'] = (['lat', 'lon'], params['lats'])
        ds['lons'] = (['lat', 'lon'], params['lons'])

        # Add max_snow_albedo to params (calibratable snow parameter)
        params['max_snow_albedo'] = np.full((nlat, nlon), 0.85)

        # Scalar parameters
        for var in ['infilt', 'Ds', 'Dsmax', 'Ws', 'c', 'rough', 'snow_rough',
                    'annual_prec', 'avg_T', 'elev', 'off_gmt', 'fs_active',
                    'July_Tavg', 'dp', 'max_snow_albedo']:
            ds[var] = (['lat', 'lon'], params[var])

        # Layer parameters
        for var in ['expt', 'Ksat', 'phi_s', 'init_moist', 'depth',
                    'bulk_density', 'soil_density', 'Wcr_FRACT', 'Wpwp_FRACT',
                    'bubble', 'quartz', 'resid_moist']:
            ds[var] = (['nlayer', 'lat', 'lon'], params[var])

        # Vegetation tiling parameters
        ds['Nveg'] = (['lat', 'lon'], params['Nveg'])
        ds['Cv'] = (['veg_class', 'lat', 'lon'], params['Cv'])

        # Vegetation library parameters
        ds['overstory'] = (['veg_class', 'lat', 'lon'], params['overstory'])
        ds['rarc'] = (['veg_class', 'lat', 'lon'], params['rarc'])
        ds['rmin'] = (['veg_class', 'lat', 'lon'], params['rmin'])
        ds['wind_h'] = (['veg_class', 'lat', 'lon'], params['wind_h'])
        ds['RGL'] = (['veg_class', 'lat', 'lon'], params['RGL'])
        ds['rad_atten'] = (['veg_class', 'lat', 'lon'], params['rad_atten'])
        ds['wind_atten'] = (['veg_class', 'lat', 'lon'], params['wind_atten'])
        ds['trunk_ratio'] = (['veg_class', 'lat', 'lon'], params['trunk_ratio'])

        # Monthly vegetation parameters
        ds['LAI'] = (['veg_class', 'month', 'lat', 'lon'], params['LAI'])
        ds['albedo'] = (['veg_class', 'month', 'lat', 'lon'], params['albedo'])
        ds['veg_rough'] = (['veg_class', 'month', 'lat', 'lon'], params['veg_rough'])
        ds['displacement'] = (['veg_class', 'month', 'lat', 'lon'], params['displacement'])

        # Root parameters (per veg class)
        ds['root_depth'] = (['veg_class', 'root_zone', 'lat', 'lon'], params['root_depth'])
        ds['root_fract'] = (['veg_class', 'root_zone', 'lat', 'lon'], params['root_fract'])

        # Snow band parameters
        ds['AreaFract'] = (['snow_band', 'lat', 'lon'], params['AreaFract'])
        ds['elevation'] = (['snow_band', 'lat', 'lon'], params['elevation'])
        ds['Pfactor'] = (['snow_band', 'lat', 'lon'], params['Pfactor'])

        # Apply mask - set inactive cells appropriately
        for var in ds.data_vars:
            if var in ('run_cell', 'gridcell'):
                continue  # Keep integer mask variables as-is
            if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
                if np.issubdtype(ds[var].dtype, np.integer):
                    ds[var] = ds[var].astype(float)
                vals = ds[var].values
                if vals.ndim == 2:
                    vals[mask == 0] = np.nan
                elif vals.ndim == 3:
                    for k in range(vals.shape[0]):
                        vals[k, mask == 0] = np.nan
                elif vals.ndim == 4:
                    for k in range(vals.shape[0]):
                        for m in range(vals.shape[1]):
                            vals[k, m, mask == 0] = np.nan

        # Add attributes
        ds.attrs['title'] = f'VIC parameter file for {self.domain_name}'
        ds.attrs['history'] = f'Created by SYMFLUENCE on {datetime.now().isoformat()}'

        ds['run_cell'].attrs = {'long_name': 'Run grid cell', 'units': 'N/A', 'comment': '1 = Run, 0 = Skip'}
        ds['gridcell'].attrs = {'long_name': 'Grid cell number', 'units': 'N/A'}
        ds['lats'].attrs = {'long_name': 'Latitude of grid cell', 'units': 'degrees_north'}
        ds['lons'].attrs = {'long_name': 'Longitude of grid cell', 'units': 'degrees_east'}
        ds['infilt'].attrs = {'long_name': 'Variable infiltration curve parameter', 'units': '-'}
        ds['Ds'].attrs = {'long_name': 'Baseflow fraction of Dsmax', 'units': '-'}
        ds['Dsmax'].attrs = {'long_name': 'Maximum baseflow velocity', 'units': 'mm/day'}
        ds['Ws'].attrs = {'long_name': 'Fraction of max soil moisture for baseflow', 'units': '-'}
        ds['depth'].attrs = {'long_name': 'Soil layer thickness', 'units': 'm'}
        ds['Ksat'].attrs = {'long_name': 'Saturated hydraulic conductivity', 'units': 'mm/day'}
        ds['elev'].attrs = {'long_name': 'Grid cell elevation', 'units': 'm'}
        ds['Nveg'].attrs = {'long_name': 'Number of vegetation tiles', 'units': 'N/A'}
        ds['Cv'].attrs = {'long_name': 'Vegetation cover fraction', 'units': '-'}
        ds['AreaFract'].attrs = {'long_name': 'Snow band area fraction', 'units': '-'}
        ds['Pfactor'].attrs = {'long_name': 'Snow band precipitation factor', 'units': '-'}

        ds.to_netcdf(params_path)
        domain_ds.close()
        logger.info(f"Parameter file written: {params_path}")

    def _generate_forcing_files(self) -> None:
        """
        Generate VIC forcing files from basin-averaged data.

        VIC requires forcing variables:
        - PREC: Precipitation [mm/timestep]
        - TMAX: Maximum temperature [°C]
        - TMIN: Minimum temperature [°C]
        - WIND: Wind speed [m/s]
        """
        logger.info("Generating VIC forcing files...")

        start_date, end_date = self._get_simulation_dates()

        # Load domain to get coordinates
        domain_file = self._get_config_value(lambda: self.config.model.vic.domain_file, dict_key='VIC_DOMAIN_FILE')
        domain_path = self.params_dir / domain_file

        if not domain_path.exists():
            raise FileOperationError(
                f"VIC domain file not found: {domain_path}. Run domain generation first."
            )

        # No synthetic fallback: missing or unreadable forcing is an error.
        with xr.open_dataset(domain_path) as domain_ds:
            forcing_ds = self._load_forcing_data()
            self._write_forcing_files(forcing_ds, domain_ds, start_date, end_date)

    def _load_forcing_data(self) -> xr.Dataset:
        """Load basin-averaged forcing data."""
        # Only the forcing remapped onto this run's HRUs (issue #339).
        forcing_files = self._select_run_forcing(sorted(self.forcing_basin_path.glob("*.nc")))

        if not forcing_files:
            # Try merged_path
            merged_path = self.project_forcing_dir / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path}")

        logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')
        except ValueError:
            try:
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
            except Exception:  # noqa: BLE001 — model execution resilience
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        # Subset to simulation period
        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    # VIC variable name → candidate forcing variable names
    VIC_VAR_MAP = {
        'AIR_TEMP': ['air_temperature', 'temperature', 'tas', 'temp', 't2m', 'AIR_TEMP'],
        'PREC': ['precipitation_flux', 'precipitation', 'pr', 'precip', 'tp', 'PREC'],
        'SWDOWN': ['surface_downwelling_shortwave_flux', 'ssrd', 'rsds', 'swdown', 'SWDOWN', 'shortwave'],
        'LWDOWN': ['surface_downwelling_longwave_flux', 'strd', 'rlds', 'lwdown', 'LWDOWN', 'longwave'],
        'PRESSURE': ['surface_air_pressure', 'sp', 'ps', 'pres', 'pressure', 'PRESSURE'],
        'VP': ['vp', 'vapor_pressure', 'VP'],
        'WIND': ['wind_speed', 'wind_speed', 'sfcWind', 'wind', 'WIND'],
    }

    def _write_forcing_files(
        self,
        forcing_ds: xr.Dataset,
        domain_ds: xr.Dataset,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Write forcing data in VIC format."""
        lat = domain_ds['lat'].values
        lon = domain_ds['lon'].values
        mask = domain_ds['mask'].values

        if 'time' not in forcing_ds.coords:
            raise ValidationError("VIC forcing: the forcing data has no 'time' coordinate")

        # VIC reads forcing at the snow-model step, which the global file sets
        # equal to MODEL_STEPS_PER_DAY (hourly by default).
        target_seconds = self._forcing_step_seconds()

        times = forcing_ds['time'].values
        if len(times) > 1:
            # Validates the whole axis: gaps, duplicates and irregular steps
            # raise rather than being filled.
            dt_seconds = validated_timestep_seconds(times, context='VIC forcing')
        else:
            dt_seconds = target_seconds

        out_ds = xr.Dataset(coords={'time': times, 'lat': lat, 'lon': lon})
        shape = (len(times), len(lat), len(lon))

        missing = []
        for vic_var, candidates in self.VIC_VAR_MAP.items():
            data = self._find_and_convert_variable(
                forcing_ds, vic_var, candidates, times, dt_seconds, mask, lat, lon
            )
            if data is not None:
                out_ds[vic_var] = (['time', 'lat', 'lon'], data[:len(times)])
            elif vic_var == 'VP' and self._derive_vapour_pressure(
                forcing_ds, out_ds, shape, times, mask, lat, lon
            ):
                continue
            else:
                missing.append(vic_var)

        if missing:
            wanted = '; '.join(f"{v} (any of {self.VIC_VAR_MAP[v]})" for v in missing)
            raise ValidationError(
                f"VIC forcing: required variable(s) not found in the forcing data: {wanted}. "
                f"Variables present: {sorted(str(v) for v in forcing_ds.data_vars)}. "
                "VP can also be derived from specific humidity and pressure."
            )

        # Derivation above uses Pa; the VIC image driver reads kPa and multiplies
        # both pressure fields by 1000 before computing density and humidity.
        for var in ('PRESSURE', 'VP'):
            out_ds[var] = out_ds[var] / 1000.0
            out_ds[var].attrs['units'] = 'kPa'

        # Bring the forcing to the VIC step. PREC is a depth per step, so it is
        # resampled as a rate (held constant over each source interval, which
        # conserves the total) and converted back with the *target* step; the
        # other variables are interpolated (finer) or averaged (coarser).
        if abs(dt_seconds - target_seconds) >= 1.0:
            logger.info(
                f"Resampling VIC forcing from {dt_seconds:.0f} s to {target_seconds:.0f} s"
            )
            attrs = {v: out_ds[v].attrs for v in out_ds.data_vars}
            rate_ds = out_ds.rename({'PREC': 'precipitation_flux'})
            rate_ds['precipitation_flux'] = rate_ds['precipitation_flux'] / dt_seconds
            rate_ds = resample_canonical_forcing(rate_ds, target_seconds)
            out_ds = rate_ds.rename({'precipitation_flux': 'PREC'})
            out_ds['PREC'] = out_ds['PREC'] * target_seconds
            out_ds = out_ds[list(self.VIC_VAR_MAP)].transpose('time', 'lat', 'lon')
            for v, a in attrs.items():
                out_ds[v].attrs = a
            times = out_ds['time'].values

        # Apply mask
        run_cell = xr.DataArray(mask == 1, dims=('lat', 'lon'))
        for var in self.VIC_VAR_MAP:
            out_ds[var] = out_ds[var].where(run_cell)

        out_ds.attrs['title'] = f'VIC forcing file for {self.domain_name}'
        out_ds.attrs['history'] = f'Created by SYMFLUENCE on {datetime.now().isoformat()}'

        # VIC reads one forcing file per calendar year. The first file may start
        # after 1 January and the last may end before 31 December: the files
        # hold only real data, and the global parameter file is limited to the
        # days they cover (see _clip_to_forcing_coverage).
        time_index = pd.DatetimeIndex(times)
        encoding = {'time': {'calendar': 'standard', 'units': 'hours since 1900-01-01'}}
        self._forcing_coverage = (
            time_index[0], time_index[-1] + pd.Timedelta(seconds=target_seconds)
        )

        for year in range(time_index[0].year, time_index[-1].year + 1):
            year_ds = out_ds.sel(time=str(year))
            if year_ds.sizes['time'] == 0:
                raise ValidationError(f"VIC forcing: no forcing data for {year}")

            forcing_path = self.forcing_dir / f'{self.domain_name}_forcing.{year}.nc'
            year_ds.to_netcdf(forcing_path, encoding=encoding)
            logger.info(f"Forcing file written: {forcing_path}")

    def _forcing_step_seconds(self) -> float:
        """Forcing step VIC expects: one snow-model step (= MODEL_STEPS_PER_DAY)."""
        steps_per_day = self._get_config_value(
            lambda: self.config.model.vic.model_steps_per_day,
            dict_key='VIC_STEPS_PER_DAY',
        )
        return 86400.0 / float(steps_per_day)

    def _clip_to_forcing_coverage(self, start_date, end_date):
        """Limit the simulation dates to whole days covered by the written forcing.

        VIC runs whole days from STARTDAY to ENDDAY, so a day only partly
        covered by forcing cannot be simulated. Rather than padding the forcing
        with invented values, the run is shortened to the covered days (with a
        warning); no overlap at all is an error. Without forcing written by
        this instance the configured dates are returned unchanged.
        """
        coverage = getattr(self, '_forcing_coverage', None)
        if coverage is None:
            return start_date, end_date
        first, end_exclusive = coverage
        first_full_day = first.ceil('D')
        last_full_day = end_exclusive.floor('D') - pd.Timedelta(days=1)
        start = max(pd.Timestamp(start_date).floor('D'), first_full_day)
        end = min(pd.Timestamp(end_date).floor('D'), last_full_day)
        if start > end:
            raise ValidationError(
                f"VIC forcing ({first} to {end_exclusive}, exclusive) does not cover a "
                f"single whole day of the simulation period {start_date} to {end_date}."
            )
        if start != pd.Timestamp(start_date).floor('D') or end != pd.Timestamp(end_date).floor('D'):
            logger.warning(
                f"VIC simulation period limited to the days covered by forcing: "
                f"{start.date()} to {end.date()} (configured {start_date} to {end_date})"
            )
        return start, end

    def _find_and_convert_variable(
        self, forcing_ds, vic_var, candidates, times, dt_seconds, mask, lat, lon
    ):
        """Search forcing dataset for a VIC variable and apply unit conversion.

        Returns converted 3-D array (time, lat, lon) or None if not found.
        """
        for candidate in candidates:
            if candidate not in forcing_ds:
                continue

            data = forcing_ds[candidate].values
            data = self._broadcast_to_grid(
                data, (len(times), len(lat), len(lon)), times, mask, lat, lon
            )
            src_units = forcing_ds[candidate].attrs.get('units', '')

            if vic_var == 'AIR_TEMP':
                if src_units == 'K' or np.nanmean(data) > 100:
                    data = data - 273.15
                    logger.info(f"Converted {candidate} from K to °C")

            elif vic_var == 'PREC':
                if 'mm/s' in src_units or candidate == 'precipitation_flux':
                    data = data * dt_seconds
                    logger.info(f"Converted {candidate} from mm/s to mm/timestep (×{dt_seconds})")
                elif src_units == 'm' or candidate == 'tp':
                    data = data * 1000.0
                    logger.info(f"Converted {candidate} from m to mm")
                elif 'kg' in src_units and 'm-2' in src_units and 's-1' in src_units:
                    data = data * dt_seconds
                    logger.info(f"Converted {candidate} from kg/m²/s to mm/timestep")

            elif vic_var in ('SWDOWN', 'LWDOWN'):
                if candidate in ('ssrd', 'strd'):
                    data = data / dt_seconds
                    logger.info(f"Converted {candidate} from J/m² to W/m² (÷{dt_seconds})")
                elif 'J' in src_units and 'm' in src_units:
                    data = data / dt_seconds
                    logger.info(f"Converted {candidate} from {src_units} to W/m²")

            elif vic_var in ('PRESSURE', 'VP'):
                units = src_units.strip().lower()
                if units in ('kpa', 'kilopascal', 'kilopascals'):
                    data = data * 1000.0
                    logger.info(f"Converted {candidate} from kPa to Pa")
                elif units in ('hpa', 'mbar', 'millibar', 'millibars'):
                    data = data * 100.0
                elif units in ('pa', 'pascal', 'pascals'):
                    pass
                elif not units:
                    # Retain the legacy atmospheric-pressure inference. Untagged
                    # vapour pressure defaults to Pa: its magnitude is ambiguous.
                    if vic_var == 'PRESSURE' and np.nanmean(data) < 200:
                        data = data * 1000.0
                    logger.warning(f"Missing units for {candidate}; inferring pressure units")
                else:
                    raise ValidationError(f"Unsupported pressure units for {candidate}: {src_units!r}")

            return data

        return None

    def _derive_vapour_pressure(
        self, forcing_ds, out_ds, shape, times, mask, lat, lon
    ) -> bool:
        """Derive VP [Pa] from specific humidity and pressure into *out_ds*.

        Returns False when either input is unavailable (no estimate is made).
        """
        spechum_var = next(
            (c for c in ('specific_humidity', 'huss', 'q') if c in forcing_ds), None
        )
        if spechum_var is None or 'PRESSURE' not in out_ds:
            return False
        logger.info(f"Deriving VP from specific humidity ({spechum_var}) and pressure")
        q_data = forcing_ds[spechum_var].values
        q_data = self._broadcast_to_grid(q_data, shape, times, mask, lat, lon)
        pressure_pa = out_ds['PRESSURE'].values
        data = q_data[:len(times)] * pressure_pa / (0.622 + 0.378 * q_data[:len(times)])
        out_ds['VP'] = (['time', 'lat', 'lon'], data)
        return True

    @staticmethod
    def _broadcast_to_grid(data, shape, times, mask, lat, lon):
        """Broadcast 1-D or 2-D data to (time, lat, lon) grid."""
        if data.ndim == 1:
            data_grid = np.zeros(shape)
            for i in range(len(lat)):
                for j in range(len(lon)):
                    if mask[i, j] == 1:
                        data_grid[:, i, j] = data[:len(times)]
            return data_grid
        elif data.ndim == 2:
            flat_data = data[:len(times), 0] if data.shape[1] >= 1 else data[:len(times)].flatten()
            data_grid = np.zeros(shape)
            for i in range(len(lat)):
                for j in range(len(lon)):
                    if mask[i, j] == 1:
                        data_grid[:, i, j] = flat_data
            return data_grid
        return data

    def _generate_global_param_file(self) -> None:
        """
        Generate the VIC global parameter file.

        This text file controls all VIC settings including:
        - File paths
        - Simulation dates
        - Output options
        - Model physics options
        """
        logger.info("Generating VIC global parameter file...")

        global_file = self._get_config_value(
            lambda: self.config.model.vic.global_param_file,
            dict_key='VIC_GLOBAL_PARAM_FILE',
        )
        global_path = self.settings_dir / global_file

        start_date, end_date = self._clip_to_forcing_coverage(*self._get_simulation_dates())

        # Get config values
        model_steps_per_day = self._get_config_value(
            lambda: self.config.model.vic.model_steps_per_day,
            dict_key='VIC_STEPS_PER_DAY',
        )
        full_energy = self._get_config_value(lambda: self.config.model.vic.full_energy, dict_key='VIC_FULL_ENERGY')
        frozen_soil = self._get_config_value(lambda: self.config.model.vic.frozen_soil, dict_key='VIC_FROZEN_SOIL')
        output_prefix = self._get_config_value(
            lambda: self.config.model.vic.output_prefix,
            dict_key='VIC_OUTPUT_PREFIX',
        )
        domain_file = self._get_config_value(lambda: self.config.model.vic.domain_file, dict_key='VIC_DOMAIN_FILE')
        params_file = self._get_config_value(lambda: self.config.model.vic.params_file, dict_key='VIC_PARAMS_FILE')

        # Build global parameter file content
        lines = [
            "#-- VIC Global Parameter File --#",
            f"# Generated by SYMFLUENCE for {self.domain_name}",
            f"# {datetime.now().isoformat()}",
            "",
            "#-- Simulation Settings --#",
            f"MODEL_STEPS_PER_DAY    {model_steps_per_day}",
            f"SNOW_STEPS_PER_DAY     {model_steps_per_day}",
            f"RUNOFF_STEPS_PER_DAY   {model_steps_per_day}",
            "",
            f"STARTYEAR              {start_date.year}",
            f"STARTMONTH             {start_date.month}",
            f"STARTDAY               {start_date.day}",
            f"ENDYEAR                {end_date.year}",
            f"ENDMONTH               {end_date.month}",
            f"ENDDAY                 {end_date.day}",
            "",
            "#-- Physics Options --#",
            f"FULL_ENERGY            {'TRUE' if full_energy else 'FALSE'}",
            f"FROZEN_SOIL            {'TRUE' if frozen_soil else 'FALSE'}",
            f"QUICK_FLUX             {'FALSE' if (full_energy or frozen_soil) else 'TRUE'}",
            f"NODES                  {10 if frozen_soil else 3}",
            "SNOW_DENSITY           DENS_SNTHRM",
            "SNOW_BAND              TRUE",
            "",
            "#-- State Options --#",
            "INIT_STATE             FALSE",
            "",
            "#-- Input Paths --#",
            f"DOMAIN                 {self.params_dir / domain_file}",
            "DOMAIN_TYPE            LAT      lat",
            "DOMAIN_TYPE            LON      lon",
            "DOMAIN_TYPE            MASK     mask",
            "DOMAIN_TYPE            AREA     area",
            "DOMAIN_TYPE            FRAC     frac",
            "DOMAIN_TYPE            YDIM     lat",
            "DOMAIN_TYPE            XDIM     lon",
            "",
            f"PARAMETERS             {self.params_dir / params_file}",
            f"FORCING1               {self.forcing_dir / f'{self.domain_name}_forcing.'}",
            "FORCE_TYPE             AIR_TEMP AIR_TEMP",
            "FORCE_TYPE             PREC     PREC",
            "FORCE_TYPE             SWDOWN   SWDOWN",
            "FORCE_TYPE             LWDOWN   LWDOWN",
            "FORCE_TYPE             PRESSURE PRESSURE",
            "FORCE_TYPE             VP       VP",
            "FORCE_TYPE             WIND     WIND",
            "",
            "#-- Output Settings --#",
            "RESULT_DIR             ./",
            "",
            "#-- Output Variables --#",
            f"OUTFILE                {output_prefix}",
            "AGGFREQ                NDAYS   1",
            "OUTVAR                 OUT_RUNOFF      *.      *       *.      AGG_TYPE_SUM",
            "OUTVAR                 OUT_BASEFLOW    *.      *       *.      AGG_TYPE_SUM",
            "OUTVAR                 OUT_EVAP        *.      *       *.      AGG_TYPE_SUM",
            "OUTVAR                 OUT_SWE",
            "OUTVAR                 OUT_SOIL_MOIST",
            "OUTVAR                 OUT_PREC        *.      *       *.      AGG_TYPE_SUM",
            "OUTVAR                 OUT_WATER_ERROR *.      *       *.      AGG_TYPE_SUM",
            "OUTVAR                 OUT_ENERGY_ERROR",
            "",
        ]

        content = '\n'.join(lines)
        global_path.write_text(content, encoding='utf-8')
        logger.info(f"Global parameter file written: {global_path}")

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
