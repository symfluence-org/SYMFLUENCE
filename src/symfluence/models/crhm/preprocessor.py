# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Model Preprocessor

Handles preparation of CRHM model inputs including:
- Observation file (.obs) with meteorological forcing data
- Project file (.prj) with model configuration and parameters
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigurationError, ValidationError
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("CRHM")
class CRHMPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a CRHM model run.

    CRHM requires:
    - Project file (.prj): Model structure, modules, and parameters
    - Observation file (.obs): Meteorological forcing data in CRHM text format
      with a header block followed by space-separated data columns
    """


    MODEL_NAME = "CRHM"
    def __init__(self, config, logger):
        """
        Initialize the CRHM preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Setup CRHM-specific directories
        # Forcing (.obs) goes in data/forcing/CRHM_input
        # Settings (.prj) goes in settings/CRHM
        self.crhm_forcing_dir = self.project_forcing_dir / "CRHM_input"
        self.settings_dir = self.project_dir / "settings" / "CRHM"

        # Get CRHM-specific settings
        self.spatial_mode = self._get_config_value(
            lambda: self.config.model.crhm.spatial_mode,
            default='lumped',
            dict_key='CRHM_SPATIAL_MODE'
        )
        logger.info(f"CRHM spatial mode: {self.spatial_mode}")

        # Elevation-band (sub-grid) discretisation. When enabled, the basin is
        # split into fixed-width elevation bands (one HRU per band). CRHM lapses
        # the basin-mean forcing to each band's mean elevation, so snow
        # accumulation and melt timing become elevation-dependent -- the single
        # biggest control on snowmelt-driven streamflow timing in mountainous
        # catchments, which a lumped (single mean-elevation) HRU cannot capture.
        self.use_elev_bands = bool(self._get_config_value(
            lambda: self.config.model.crhm.elevation_bands,
            default=False,
            dict_key='CRHM_ELEVATION_BANDS'
        ))
        self.elev_band_size = float(self._get_config_value(
            lambda: self.config.model.crhm.elevation_band_size,
            default=self._get_config_value(
                lambda: self.config.domain.elevation_band_size,
                default=200.0, dict_key='ELEVATION_BAND_SIZE'),
            dict_key='CRHM_ELEVATION_BAND_SIZE'
        ))
        if self.use_elev_bands:
            logger.info(f"CRHM elevation bands ON (band size {self.elev_band_size:.0f} m)")

        # Terrain-driven radiation: derive per-band slope (hru_GSL) and aspect
        # (hru_ASL) from the DEM instead of treating every band as flat. CRHM's
        # energy-balance snowmelt uses slope/aspect for the solar incidence
        # angle, so this sharpens elevation-band melt timing. Requires bands.
        self.use_terrain_radiation = bool(self._get_config_value(
            lambda: self.config.model.crhm.terrain_radiation,
            default=False,
            dict_key='CRHM_TERRAIN_RADIATION'
        ))
        if self.use_terrain_radiation:
            logger.info("CRHM terrain radiation ON (per-band slope/aspect from DEM)")

    def run_preprocessing(self) -> bool:
        """
        Run the complete CRHM preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting CRHM preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate observation file from ERA5 forcing
            self._generate_observation_file()

            # Generate project file
            self._generate_project_file()

            logger.info("CRHM preprocessing complete.")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"CRHM preprocessing failed: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create CRHM input directory structure."""
        self.crhm_forcing_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created CRHM input directories at {self.crhm_forcing_dir} and {self.settings_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from configuration."""
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        try:
            catchment_path = self._find_basin_shapefile(required=True)
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)

                # Centroid in GEOGRAPHIC degrees. The shapefile is often in a
                # projected CRS (e.g. EPSG:3057, metres); taking the centroid in
                # that CRS yields lon/lat in metres, which then produces an
                # absurd UTM zone (EPSG:9xxxx -> invalid) and made this whole
                # method fall back to defaults (lat=51, area=100) -> wrong PET,
                # snowmelt timing and water balance. Compute the centroid in a
                # projected CRS (to avoid the geographic-centroid warning) and
                # reproject the point to degrees.
                if gdf.crs.is_geographic:
                    cpt = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326).iloc[0]
                else:
                    cpt = gdf.geometry.centroid.to_crs(epsg=4326).iloc[0]
                lon, lat = cpt.x, cpt.y

                # Accurate area via the UTM zone derived from the degree centroid.
                utm_zone = int((lon + 180) / 6) + 1
                utm_crs = f"EPSG:{(32600 if lat >= 0 else 32700) + utm_zone}"
                area_m2 = gdf.to_crs(utm_crs).geometry.area.sum()
                if not np.isfinite(area_m2) or area_m2 <= 0:
                    raise ValidationError('Catchment area must be finite and positive')
                expected = self._get_config_value(lambda: self.config.domain.catchment_area, default=None)
                if expected is not None and not np.isclose(area_m2 / 1e6, expected, rtol=0.02):
                    raise ValidationError(f'Geometry area {area_m2 / 1e6:g} km2 disagrees with CATCHMENT_AREA={expected}')

                # Get elevation if available
                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0

                return {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'area_km2': area_m2 / 1e6,
                    'elev': elev
                }
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise ValidationError('CRHM requires readable catchment geometry; refusing default basin area/location') from e
        raise ValidationError('CRHM catchment geometry does not exist')

    def _find_dem_path(self) -> Optional[Path]:
        """Locate the catchment DEM raster (domain_<name>_elv.tif)."""
        dem_dir = self.project_attributes_dir / 'elevation' / 'dem'
        cands = sorted(dem_dir.glob('*_elv.tif')) if dem_dir.exists() else []
        if not cands:
            cands = sorted(self.project_attributes_dir.glob('**/*elv*.tif'))
        return cands[0] if cands else None

    def _get_elevation_bands(self, total_area_km2: float):
        """
        Derive fixed-width elevation bands from the catchment DEM.

        Returns a list of {'elev': mean_elev_m, 'area_km2': band_area} dicts
        (ordered low -> high) plus the area-weighted mean elevation, which is
        used as the observation (forcing) reference elevation. CRHM then lapses
        the basin-mean forcing from that reference to each band.

        Returns (bands, ref_elev) or (None, None) when bands cannot be built
        (no DEM, <2 populated bands) so the caller can fall back to lumped.
        """
        if not self.use_terrain_radiation:
            import geopandas as gpd
            for path in sorted((self.project_dir / 'shapefiles' / 'catchment').rglob('*HRUs_elevation.shp')):
                hrus = gpd.read_file(path).sort_values('elev_mean')
                if len(hrus) > 1 and {'HRU_area', 'elev_mean'}.issubset(hrus.columns):
                    areas = hrus.HRU_area.to_numpy(dtype=float)
                    elevations = hrus.elev_mean.to_numpy(dtype=float)
                    if np.all(areas > 0) and np.isfinite(elevations).all():
                        bands = [{'elev': float(e), 'area_km2': float(a / areas.sum() * total_area_km2)}
                                 for e, a in zip(elevations, areas)]
                        return bands, float(np.average(elevations, weights=areas))

        try:
            import rasterio  # local import: only needed when bands are enabled
        except ImportError:
            logger.warning("rasterio unavailable; cannot build elevation bands")
            return None, None

        dem_path = self._find_dem_path()
        if dem_path is None:
            logger.warning("No DEM found; falling back to lumped CRHM (1 HRU)")
            return None, None

        with rasterio.open(dem_path) as src:
            arr2d = src.read(1).astype('float64')
            nodata = src.nodata
            xres, yres = src.res
            bnds = src.bounds
            is_geographic = src.crs is not None and src.crs.is_geographic

        valid = np.isfinite(arr2d) & (arr2d > -100.0)
        if nodata is not None:
            valid &= arr2d != nodata
        if int(valid.sum()) == 0:
            logger.warning("DEM has no valid pixels; falling back to lumped CRHM")
            return None, None

        # Per-pixel slope (deg) and aspect (deg, 0=N) for terrain radiation.
        slope2d = aspect2d = None
        if self.use_terrain_radiation:
            if is_geographic:
                lat0 = np.radians((bnds.bottom + bnds.top) / 2.0)
                xres_m = abs(xres) * 111320.0 * max(np.cos(lat0), 0.1)
                yres_m = abs(yres) * 111320.0
            else:
                xres_m, yres_m = abs(xres), abs(yres)
            z = np.where(valid, arr2d, np.nan)
            dzdy, dzdx = np.gradient(z, yres_m, xres_m)
            slope2d = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
            aspect2d = (np.degrees(np.arctan2(-dzdy, dzdx)) + 360.0) % 360.0

        arr = arr2d[valid]
        sl = slope2d[valid] if slope2d is not None else None
        asp = aspect2d[valid] if aspect2d is not None else None

        size = self.elev_band_size
        lo_edge = np.floor(arr.min() / size) * size
        hi_edge = np.ceil(arr.max() / size) * size
        edges = np.arange(lo_edge, hi_edge + size, size)
        npix = arr.size
        bands = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = (arr >= lo) & (arr < hi) if i < len(edges) - 2 else (arr >= lo) & (arr <= hi)
            n = int(sel.sum())
            if n == 0:
                continue
            band = {
                'elev': float(arr[sel].mean()),
                'area_km2': total_area_km2 * (n / npix),
            }
            if sl is not None and asp is not None:
                band['slope'] = float(np.nanmean(sl[sel]))
                a = np.radians(asp[sel])
                band['aspect'] = float(
                    (np.degrees(np.arctan2(np.nanmean(np.sin(a)),
                                           np.nanmean(np.cos(a)))) + 360.0) % 360.0)
            bands.append(band)

        if len(bands) < 2:
            logger.info("DEM relief spans <2 bands; using lumped CRHM (1 HRU)")
            return None, None

        ref_elev = sum(b['elev'] * b['area_km2'] for b in bands) / \
            sum(b['area_km2'] for b in bands)
        logger.info(
            f"CRHM elevation bands: {len(bands)} bands "
            f"({bands[0]['elev']:.0f}-{bands[-1]['elev']:.0f} m), "
            f"ref elev {ref_elev:.0f} m"
        )
        return bands, ref_elev

    def _generate_observation_file(self) -> None:
        """
        Generate the CRHM observation file (.obs) from ERA5 forcing data.

        The .obs file format has:
        - Header lines starting with variable descriptions
        - Column definitions
        - Space-separated data with datetime columns

        Required variables:
        - t (air temperature, deg C)
        - rh (relative humidity, %)
        - p (precipitation, mm)
        - u (wind speed, m/s)
        - Qsi (incoming shortwave radiation, W/m2)
        """
        logger.info("Generating CRHM observation file...")

        obs_file = self._get_config_value(
            lambda: self.config.model.crhm.observation_file,
            default='forcing.obs'
        )
        obs_path = self.crhm_forcing_dir / obs_file

        start_date, end_date = self._get_simulation_dates()

        # Try to load forcing data
        try:
            forcing_ds = self._load_forcing_data()
            self._write_obs_file(forcing_ds, obs_path, start_date, end_date)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not load forcing data: {e}, using synthetic", exc_info=True)
            self._generate_synthetic_obs(obs_path, start_date, end_date)

        logger.info(f"Observation file written: {obs_path}")

    def _load_forcing_data(self) -> xr.Dataset:
        """Load basin-averaged forcing data from NetCDF files."""
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

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

    def _write_obs_file(
        self,
        forcing_ds: xr.Dataset,
        obs_path: Path,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Write forcing data in CRHM .obs format."""
        # Variable mapping from ERA5/forcing names to CRHM names
        var_map = {
            # NOTE: includes CARRA/SUMMA-style names (airtemp, pptrate, windspd,
            # SWRadAtm). Without them every variable fell back to SYNTHETIC data
            # (constant-ish temp ramp, ~5x-low precip) -> meaningless simulation.
            't': ['air_temperature', 'temperature', 'tas', 'temp', 't2m', 'airtemp'],
            'p': ['precipitation_flux', 'precipitation', 'pr', 'precip', 'tp', 'pptrate'],
            'rh': ['rh', 'relative_humidity', 'hurs'],
            'u': ['wind_speed', 'sfcWind', 'wind', 'windspd'],
            'Qsi': ['surface_downwelling_shortwave_flux', 'ssrd', 'rsds', 'swdown', 'shortwave', 'SWRadAtm'],
        }

        times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='h')
        time_index = pd.DatetimeIndex(times)

        # Determine timestep
        if len(time_index) > 1:
            dt_seconds = float((time_index[1] - time_index[0]).total_seconds())
        else:
            dt_seconds = 3600.0

        # Extract data for each variable
        data_columns = {}

        for crhm_var, candidates in var_map.items():
            found = False
            for candidate in candidates:
                if candidate in forcing_ds:
                    data = forcing_ds[candidate].values

                    # Flatten spatial dimensions if present
                    if data.ndim > 1:
                        # Average over spatial dims
                        while data.ndim > 1:
                            data = np.nanmean(data, axis=-1)

                    data = data[:len(time_index)]
                    src_units = forcing_ds[candidate].attrs.get('units', '')

                    # Unit conversions
                    if crhm_var == 't':
                        # CRHM expects deg C
                        if src_units == 'K' or np.nanmean(data) > 100:
                            data = data - 273.15
                            logger.info(f"Converted {candidate} from K to deg C")

                    elif crhm_var == 'p':
                        # CRHM expects mm per timestep
                        if 'mm/s' in src_units or candidate == 'precipitation_flux':
                            data = data * dt_seconds
                            logger.info(f"Converted {candidate} from mm/s to mm/timestep")
                        elif src_units == 'm' or candidate == 'tp':
                            data = data * 1000.0
                            logger.info(f"Converted {candidate} from m to mm")
                        elif 'kg' in src_units and 'm-2' in src_units and 's-1' in src_units:
                            data = data * dt_seconds
                            logger.info(f"Converted {candidate} from kg/m2/s to mm/timestep")

                    elif crhm_var == 'rh':
                        # CRHM expects % (0-100)
                        if np.nanmax(data) <= 1.0:
                            data = data * 100.0
                            logger.info(f"Converted {candidate} from fraction to %")

                    elif crhm_var == 'Qsi':
                        # CRHM expects W/m2
                        if candidate in ('ssrd',):
                            data = data / dt_seconds
                            logger.info(f"Converted {candidate} from J/m2 to W/m2")

                    data_columns[crhm_var] = data
                    found = True
                    break

            if not found:
                # Generate synthetic values for missing variables
                n = len(time_index)
                day_frac = np.arange(n) / 24.0

                if crhm_var == 't':
                    data_columns[crhm_var] = 5.0 + 10.0 * np.sin(2 * np.pi * day_frac / 365)
                elif crhm_var == 'p':
                    data_columns[crhm_var] = np.random.exponential(0.1, n)
                elif crhm_var == 'rh':
                    # Derive from specific humidity if available
                    derived = False
                    if 't' in data_columns:
                        for q_name in ['specific_humidity', 'huss', 'q', 'spechum']:
                            if q_name in forcing_ds:
                                q_data = forcing_ds[q_name].values
                                if q_data.ndim > 1:
                                    while q_data.ndim > 1:
                                        q_data = np.nanmean(q_data, axis=-1)
                                q_data = q_data[:n]
                                temp_c = data_columns['t']
                                es = 610.8 * np.exp(17.27 * temp_c / (temp_c + 237.3))
                                # Approximate pressure
                                p_pa = 101325.0 * np.exp(-1000.0 / 8500.0)
                                if 'surface_air_pressure' in forcing_ds:
                                    p_pa = forcing_ds['surface_air_pressure'].values.reshape(n, -1).mean(axis=1)
                                e = q_data * p_pa / (0.622 + 0.378 * q_data)
                                rh = np.clip(e / es * 100.0, 0, 100)
                                data_columns[crhm_var] = rh
                                derived = True
                                logger.info(f"Derived RH from specific humidity ({q_name})")
                                break
                    if not derived:
                        data_columns[crhm_var] = 60.0 * np.ones(n)
                        logger.warning("Using default RH=60%")
                elif crhm_var == 'u':
                    data_columns[crhm_var] = np.abs(np.random.normal(3.0, 1.0, n))
                elif crhm_var == 'Qsi':
                    data_columns[crhm_var] = np.maximum(
                        0, 200 + 150 * np.sin(2 * np.pi * (day_frac - 80) / 365)
                    )
                if crhm_var != 'rh' or not derived:
                    logger.warning(f"Variable {crhm_var} not found in forcing, using synthetic")

        # Write the .obs file
        self._write_obs_text(obs_path, time_index + pd.Timedelta(seconds=dt_seconds), data_columns)

    def _write_obs_text(
        self,
        obs_path: Path,
        time_index: pd.DatetimeIndex,
        data_columns: Dict[str, np.ndarray]
    ) -> None:
        """Write CRHM .obs text file with header and data.

        The CRHM .obs format is:
            Line 1:  free-text description
            Lines 2..N:  variable declarations ``<name> <ncols> [comment]``
                         (optional ``/`` comment lines or ``$`` filter lines)
            Delimiter:  a line starting with ``#``
            Data lines: ``<datetime_fields> <val1> <val2> ...``

        Datetime can be either:
          - Decimal (Excel serial date)  --  a single float > 3000
          - Integer  --  ``YYYY MM DD HH MM`` (5 space-separated integers)
        """
        var_names = list(data_columns.keys())

        lines = []

        # Line 1 -- free-text description (read and discarded by CRHM)
        lines.append(
            f"Observation file for {self.domain_name} "
            f"- SYMFLUENCE {datetime.now().strftime('%Y-%m-%d')}"
        )

        # Variable declarations -- must come *before* the # delimiter.
        # Each line: <var_name> <number_of_columns> [optional unit comment]
        obs_units = {'t': '(C)', 'rh': '(%)', 'p': '(mm)',
                     'u': '(m/s)', 'Qsi': '(W/m2)'}
        for var_name in var_names:
            unit = obs_units.get(var_name, '')
            lines.append(f"{var_name} 1 {unit}".rstrip())

        # Section delimiter
        lines.append("########################################")

        # Data lines: YYYY MM DD HH MM followed by one value per variable
        for i, dt in enumerate(time_index):
            date_str = dt.strftime("%Y %m %d %H %M")
            values = []
            for var_name in var_names:
                val = data_columns[var_name][i] if i < len(data_columns[var_name]) else 0.0
                if np.isnan(val):
                    val = 0.0
                values.append(f"{val:.4f}")
            line = f"{date_str} " + " ".join(values)
            lines.append(line)

        obs_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def _generate_synthetic_obs(
        self,
        obs_path: Path,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Generate synthetic observation data for testing."""
        self._get_catchment_properties()
        dates = pd.date_range(start_date, end_date, freq='h')
        n = len(dates)

        day_frac = np.arange(n) / 24.0

        data_columns = {
            't': 5.0 + 10.0 * np.sin(2 * np.pi * day_frac / 365),
            'rh': 60.0 + 20.0 * np.sin(2 * np.pi * day_frac / 365 + np.pi),
            'p': np.random.exponential(0.1, n),
            'u': np.abs(np.random.normal(3.0, 1.0, n)),
            'Qsi': np.maximum(0, 200 + 150 * np.sin(2 * np.pi * (day_frac - 80) / 365)),
        }

        self._write_obs_text(obs_path, dates, data_columns)
        logger.info(f"Synthetic observation file written: {obs_path}")

    # Module version string used when writing the Modules section.
    # CRHM parses "<module> <DLLName> <version>" for each line.
    _MODULE_DLL = "CRHM"
    _MODULE_VER = "04/20/06"

    # Default module list for a lumped cold-region basin.
    # "basin" and "global" are mandatory infrastructure modules that must
    # appear before any science modules.  "obs" bridges the observation
    # file into the model.
    #
    # IMPORTANT ordering notes:
    #   - "crack" (frozen-soil infiltration) MUST appear before "evap" and
    #     "Soil".  The Soil module reads variables produced by crack; omitting
    #     it causes a null-pointer crash at the first timestep.
    #   - "Annandale", "longVt", and "Slope_Qsi" are radiation estimation
    #     modules that require additional obs variables or DEM-derived fields
    #     not available in a lumped setup.  They are omitted here; "calcsun"
    #     and "netall" provide sufficient radiation processing for most uses.
    _DEFAULT_MODULES = [
        "basin",
        "global",
        "obs",
        "calcsun",
        "intcp",
        "pbsm",
        "albedo",
        "ebsm",
        "netall",
        "crack",
        "evap",
        "Soil",
        "Netroute",
    ]

    # Mapping from parameter name to (owning_module, min, max).
    # "Shared" means the value is broadcast to every module that declares
    # the same parameter name.
    _PARAM_META = {
        # ---- Shared (broadcast to all modules) ----
        'basin_area':      ('Shared', 1e-06, 1e+09),
        'hru_area':        ('Shared', 1e-06, 1e+09),
        'hru_ASL':         ('Shared', 0, 360),
        'hru_elev':        ('Shared', 0, 1e+05),
        'hru_GSL':         ('Shared', 0, 90),
        'hru_lat':         ('Shared', -90, 90),
        'Ht':              ('Shared', 0.001, 100),
        'inhibit_evap':    ('Shared', 0, 5),
        'Sdmax':           ('Shared', 0, 1000),
        'soil_rechr_max':  ('Shared', 0, 350),
        'fetch':           ('Shared', 300, 10000),
        # ---- albedo ----
        'Albedo_bare':     ('albedo', 0, 1),
        'Albedo_snow':     ('albedo', 0, 1),
        # ---- basin ----
        'basin_name':      ('basin', None, None),      # text
        'hru_names':       ('basin', None, None),       # text
        'INIT_STATE':      ('basin', None, None),       # text
        'Loop_to':         ('basin', None, None),       # text
        'RapidAdvance_to': ('basin', None, None),       # text
        'RUN_END':         ('basin', 0, 1e+05),
        'RUN_ID':          ('basin', -1e+08, 1e+08),
        'RUN_START':       ('basin', 0, 1e+05),
        'StateVars_to_Update': ('basin', None, None),   # text
        'TraceVars':       ('basin', None, None),       # text
        # ---- crack ----
        'fallstat':        ('crack', -1, 100),
        'infDays':         ('crack', 0, 20),
        'Major':           ('crack', 1, 100),
        'PriorInfiltration': ('crack', 0, 1),
        # ---- ebsm ----
        'delay_melt':      ('ebsm', 0, 366),
        'nfactor':         ('ebsm', 0, 10),
        'Qe_subl_from_SWE': ('ebsm', 0, 1),
        'tfactor':         ('ebsm', 0, 10),
        'Use_QnD':         ('ebsm', 0, 1),
        # ---- evap ----
        'evap_type':       ('evap', 0, 2),
        'F_Qg':            ('evap', 0, 1),
        'inhibit_evap_User': ('evap', 0, 1),
        'rs':              ('evap', 0, 0.01),
        'Zwind':           ('evap', 0.01, 100),
        # ---- global ----
        'Time_Offset':     ('global', -12, 12),
        # ---- Netroute ----
        'gwKstorage':      ('Netroute', 0, 200),
        'gwLag':           ('Netroute', 0, 1e+04),
        'gwwhereto':       ('Netroute', -1000, 1000),
        'Kstorage':        ('Netroute', 0, 200),
        'Lag':             ('Netroute', 0, 1e+04),
        'order':           ('Netroute', 1, 1000),
        'preferential_flow': ('Netroute', 0, 1),
        'runKstorage':     ('Netroute', 0, 200),
        'runLag':          ('Netroute', 0, 1e+04),
        'Sd_ByPass':       ('Netroute', 0, 1),
        'soil_rechr_ByPass': ('Netroute', 0, 1),
        'ssrKstorage':     ('Netroute', 0, 200),
        'ssrLag':          ('Netroute', 0, 1e+04),
        'whereto':         ('Netroute', 0, 1000),
        # ---- obs ----
        'catchadjust':     ('obs', 0, 3),
        'ClimChng_flag':   ('obs', 0, 1),
        'ClimChng_precip': ('obs', 0, 10),
        'ClimChng_t':      ('obs', -50, 50),
        'ElevChng_flag':   ('obs', 0, 1),
        'HRU_OBS':         ('obs', 1, 100),
        'lapse_rate':      ('obs', 0, 2),
        'obs_elev':        ('obs', 0, 1e+05),
        'ppt_daily_distrib': ('obs', 0, 1),
        'precip_elev_adj': ('obs', -1, 1),
        'snow_rain_determination': ('obs', 0, 2),
        'tmax_allrain':    ('obs', -10, 10),
        'tmax_allsnow':    ('obs', -10, 10),
        # ---- pbsm ----
        'A_S':             ('pbsm', 0, 2),
        'distrib':         ('pbsm', -10, 10),
        # Note: pbsm has its own "fetch" parameter distinct from Shared fetch.
        # It is written directly in _generate_project_file to avoid name collision.
        'inhibit_bs':      ('pbsm', 0, 1),
        'inhibit_subl':    ('pbsm', 0, 1),
        'N_S':             ('pbsm', 1, 500),
        # ---- Soil ----
        'cov_type':        ('Soil', 0, 2),
        'gw_init':         ('Soil', 0, 5000),
        'gw_K':            ('Soil', 0, 100),
        'gw_max':          ('Soil', 0, 5000),
        'lower_ssr_K':     ('Soil', 0, 100),
        'rechr_ssr_K':     ('Soil', 0, 100),
        'Sdinit':          ('Soil', 0, 5000),
        'Sd_gw_K':         ('Soil', 0, 100),
        'Sd_ssr_K':        ('Soil', 0, 100),
        'soil_gw_K':       ('Soil', 0, 100),
        'soil_moist_init': ('Soil', 0, 5000),
        'soil_moist_max':  ('Soil', 0, 5000),
        'soil_rechr_init': ('Soil', 0, 250),
        'soil_ssr_runoff':  ('Soil', 0, 1),
        'soil_withdrawal': ('Soil', 1, 4),
        'transp_limited':  ('Soil', 0, 1),
        'Wetlands_scaling_factor': ('Soil', -1, 1),
    }

    def _generate_project_file(self) -> None:
        """
        Generate the CRHM project file (.prj).

        The .prj format is a section-based text file consumed by the CRHM
        binary.  Each section starts with a keyword line (e.g. ``Dimensions:``)
        followed by a ``######`` delimiter, then the section body, closed by
        another ``######`` delimiter.

        Sections (in required order):
            Header      - description + version
            Dimensions  - nhru, nlay, nobs
            Macros      - (empty)
            Observations - paths to .obs files (one per line)
            Dates       - start/end as ``YYYY MM DD``
            Modules     - ``<name> <DLL> <version>`` per line
            Parameters  - ``<module> <param> <min to max>`` header then values
            Initial_State, Final_State, Summary_period,
            Display_Variable, Display_Observation, Log_All, TChart

        IMPORTANT: The Soil module in CRHM requires the ``crack`` module
        (frozen-soil infiltration) to be present in the module list.
        Without it, CRHM will SIGSEGV at the first timestep when any
        Display_Variable entries are present.  This is not an nhru issue
        -- nhru=1 works correctly when crack is included.
        """
        logger.info("Generating CRHM project file...")

        calibrated = {
            name.strip() for name in self.config.model.crhm.params_to_calibrate.split(',')
        }
        if 'nfactor' in calibrated:
            raise ConfigurationError(
                "The generated CRHM forcing has no daily net-radiation observation QnD. "
                "Remove nfactor from params_to_calibrate and leave it at zero; "
                "nonzero nfactor makes native ebsm initialization fail."
            )
        if 'Qe_subl_from_SWE' in calibrated:
            raise ValidationError('Qe_subl_from_SWE is a native integer switch; use sublimation_from_swe instead of continuous calibration.')

        prj_file = self._get_config_value(
            lambda: self.config.model.crhm.project_file,
            default='model.prj'
        )
        prj_path = self.settings_dir / prj_file

        obs_file = self._get_config_value(
            lambda: self.config.model.crhm.observation_file,
            default='forcing.obs'
        )

        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()

        # Store the observation file as a bare filename.  The runner
        # passes --obs_file_directory to tell CRHM where to find it.
        obs_basename = obs_file

        # Catchment properties
        area_km2 = props.get('area_km2', 100.0)
        lat = props.get('lat', 51.0)
        elev = props.get('elev', 1000.0)

        # Elevation-band discretisation (one HRU per band) or lumped fallback.
        # band_elevs / band_areas are per-HRU vectors; obs_ref_elev is the
        # forcing reference elevation CRHM lapses from.
        band_elevs = None
        band_areas = None
        band_slopes = None
        band_aspects = None
        obs_ref_elev = int(elev)
        if self.use_elev_bands:
            bands, ref_elev = self._get_elevation_bands(area_km2)
            if bands:
                band_elevs = [int(round(b['elev'])) for b in bands]
                band_areas = [round(b['area_km2'], 6) for b in bands]
                obs_ref_elev = int(round(ref_elev))
                if self.use_terrain_radiation and all('slope' in b for b in bands):
                    band_slopes = [round(b['slope'], 2) for b in bands]
                    band_aspects = [round(b['aspect'], 1) for b in bands]
        nhru = len(band_elevs) if band_elevs else 1  # lumped -> single HRU

        lines = []

        # -- Header -----------------------------------------------------------
        lines.append(
            f"CRHM Project - {self.domain_name} "
            f"{'lumped' if self.spatial_mode == SpatialMode.LUMPED else self.spatial_mode}"
            f" - SYMFLUENCE"
        )
        lines.append("###### Version NON DLL 4.02")

        # -- Dimensions -------------------------------------------------------
        lines.append("Dimensions:")
        lines.append("######")
        lines.append(f"nhru {nhru}")
        lines.append("nlay 2")
        lines.append("nobs 1")
        lines.append("######")

        # -- Macros (empty but section must exist) ----------------------------
        lines.append("Macros:")
        lines.append("######")
        lines.append("######")

        # -- Observations -----------------------------------------------------
        lines.append("Observations:")
        lines.append("######")
        lines.append(obs_basename)
        lines.append("######")

        # -- Dates ------------------------------------------------------------
        lines.append("Dates:")
        lines.append("######")
        lines.append(start_date.strftime("%Y %m %d"))
        lines.append((pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)).strftime("%Y %m %d"))
        lines.append("######")

        # -- Modules ----------------------------------------------------------
        lines.append("Modules:")
        lines.append("######")
        for mod in self._DEFAULT_MODULES:
            lines.append(f"{mod} {self._MODULE_DLL} {self._MODULE_VER}")
        lines.append("######")

        # -- Parameters -------------------------------------------------------
        lines.append("Parameters:")
        lines.append("###### 'basin' parameters always first")

        # Helper to format a numeric range annotation
        def _range(pmin, pmax):
            if pmin is not None and pmax is not None:
                return f" <{pmin} to {pmax}>"
            return ""

        # Helper to write a parameter line.
        # ``values`` can be:
        #   - a string (text params like basin_name, or pre-formatted
        #     multi-line numeric blocks like HRU_OBS)
        #   - a list/tuple (one value per HRU, joined by spaces)
        #   - a scalar (single value)
        #
        # For text parameters whose _PARAM_META has (None, None) bounds,
        # no range annotation is written.  For string-encoded multi-line
        # numeric blocks, the range annotation IS written when bounds
        # are available.
        def _write_param(module, name, values, pmin=None, pmax=None):
            meta = self._PARAM_META.get(name)
            if meta:
                module = meta[0]
                pmin = meta[1] if pmin is None else pmin
                pmax = meta[2] if pmax is None else pmax

            if isinstance(values, str):
                lines.append(f"{module} {name}{_range(pmin, pmax)}")
                lines.append(values)
            elif isinstance(values, (list, tuple)):
                lines.append(f"{module} {name}{_range(pmin, pmax)}")
                lines.append(' '.join(str(v) for v in values))
            else:
                lines.append(f"{module} {name}{_range(pmin, pmax)}")
                lines.append(str(values))

        # Helper to write per-HRU value (same value for all HRUs)
        def _hru_val(v):
            return ' '.join([str(v)] * nhru)

        # Helper to write an explicit per-HRU vector (one value per HRU)
        def _hru_vec(vals):
            return ' '.join(str(v) for v in vals)

        # Per-HRU area / elevation: band vectors when discretised, else lumped.
        hru_area_str = _hru_vec(band_areas) if band_areas else _hru_val(area_km2)
        hru_elev_str = _hru_vec(band_elevs) if band_elevs else _hru_val(int(elev))
        # Per-band slope/aspect from the DEM when terrain radiation is on.
        hru_asl_str = _hru_vec(band_aspects) if band_aspects else _hru_val(0)
        hru_gsl_str = _hru_vec(band_slopes) if band_slopes else _hru_val(5)

        # ---- Shared parameters (basin-wide, broadcast to all modules) ----
        _write_param('Shared', 'basin_area', area_km2)
        _write_param('Shared', 'hru_area', hru_area_str)
        _write_param('Shared', 'hru_ASL', hru_asl_str)
        _write_param('Shared', 'hru_elev', hru_elev_str)
        _write_param('Shared', 'hru_GSL', hru_gsl_str)
        _write_param('Shared', 'hru_lat', _hru_val(round(lat, 2)))
        _write_param('Shared', 'Ht', _hru_val(0.3))
        _write_param('Shared', 'inhibit_evap', _hru_val(0))
        _write_param('Shared', 'Sdmax', _hru_val(10))
        _write_param('Shared', 'soil_rechr_max', _hru_val(60))

        # ---- albedo ----
        _write_param('albedo', 'Albedo_bare', _hru_val(0.17))
        _write_param('albedo', 'Albedo_snow', _hru_val(0.85))

        # ---- basin ----
        _write_param('basin', 'basin_name', "''")
        hru_name_list = ' '.join([f"'HRU{i+1}'" for i in range(nhru)])
        _write_param('basin', 'hru_names', hru_name_list)
        _write_param('basin', 'INIT_STATE', "''")
        _write_param('basin', 'Loop_to', "'' ''")
        _write_param('basin', 'RapidAdvance_to', "''")
        _write_param('basin', 'RUN_END', 0)
        _write_param('basin', 'RUN_ID', 1)
        _write_param('basin', 'RUN_START', 0)
        _write_param('basin', 'StateVars_to_Update',
                      ' '.join(["''"] * 10))
        _write_param('basin', 'TraceVars',
                      ' '.join(["''"] * 10))

        # ---- crack (frozen-soil infiltration) ----
        # Required by the Soil module; without it CRHM will SIGSEGV.
        _write_param('crack', 'fallstat', _hru_val(46))
        _write_param('crack', 'infDays', _hru_val(6))
        _write_param('crack', 'Major', _hru_val(5))
        _write_param('crack', 'PriorInfiltration', _hru_val(0))

        # ---- ebsm (energy-balance snowmelt) ----
        _write_param('ebsm', 'delay_melt', _hru_val(0))
        _write_param('ebsm', 'nfactor', _hru_val(0))
        explicit_sublimation = self._get_config_value(
            lambda: self.config.model.crhm.sublimation_from_swe, default=False)
        _write_param('ebsm', 'Qe_subl_from_SWE', _hru_val(int(explicit_sublimation)))
        _write_param('ebsm', 'tfactor', _hru_val(0))
        _write_param('ebsm', 'Use_QnD', _hru_val(0))

        # ---- evap ----
        _write_param('evap', 'evap_type', _hru_val(0))
        _write_param('evap', 'F_Qg', _hru_val(0.05))
        _write_param('evap', 'inhibit_evap_User', _hru_val(0))
        _write_param('evap', 'rs', _hru_val(0))
        _write_param('evap', 'Zwind', _hru_val(10))

        # ---- global ----
        _write_param('global', 'Time_Offset', 0)

        # ---- Netroute (routing) ----
        # For nhru=1: order=1, whereto=0 (outlet), gwwhereto=0 (outlet)
        _write_param('Netroute', 'gwKstorage', _hru_val(0))
        _write_param('Netroute', 'gwLag', _hru_val(0))
        _write_param('Netroute', 'gwwhereto', _hru_val(nhru + 1 if self.config.model.crhm.route_groundwater_to_stream else 0))
        _write_param('Netroute', 'Kstorage', _hru_val(1))
        _write_param('Netroute', 'Lag', _hru_val(8))
        _write_param('Netroute', 'order',
                      _hru_vec(range(1, nhru + 1)))
        _write_param('Netroute', 'preferential_flow', _hru_val(0))
        _write_param('Netroute', 'runKstorage', _hru_val(0))
        _write_param('Netroute', 'runLag', _hru_val(0))
        _write_param('Netroute', 'Sd_ByPass', _hru_val(0))
        _write_param('Netroute', 'soil_rechr_ByPass', _hru_val(0))
        _write_param('Netroute', 'ssrKstorage', _hru_val(0))
        _write_param('Netroute', 'ssrLag', _hru_val(0))
        _write_param('Netroute', 'whereto', _hru_val(0))

        # ---- obs (observation bridge) ----
        _write_param('obs', 'catchadjust', 0)
        _write_param('obs', 'ClimChng_flag', 0)
        _write_param('obs', 'ClimChng_precip', 1)
        _write_param('obs', 'ClimChng_t', 0)
        _write_param('obs', 'ElevChng_flag', _hru_val(1))
        # HRU_OBS: maps each HRU to observation station 1.
        # This is a 2-D parameter (nobs x nhru): 5 rows of nhru values.
        hru_obs_row = _hru_val(1)
        _write_param('obs', 'HRU_OBS',
                      '\n'.join([hru_obs_row] * 5))
        _write_param('obs', 'lapse_rate', _hru_val(0.75))
        # obs_elev: forcing reference elevation (basin-mean) that CRHM lapses
        # FROM to each band's hru_elev. 2-D (nobs x nhru), 2 rows.
        _write_param('obs', 'obs_elev',
                      '\n'.join([_hru_val(obs_ref_elev)] * 2))
        _write_param('obs', 'ppt_daily_distrib', _hru_val(1))
        _write_param('obs', 'precip_elev_adj', _hru_val(0))
        _write_param('obs', 'snow_rain_determination', _hru_val(0))
        _write_param('obs', 'tmax_allrain', _hru_val(4))
        _write_param('obs', 'tmax_allsnow', _hru_val(0))

        # ---- pbsm (blowing snow) ----
        _write_param('pbsm', 'A_S', _hru_val(0.003))
        _write_param('pbsm', 'distrib', _hru_val(1))
        # pbsm has its own "fetch" parameter distinct from the Shared one.
        # Write it directly to avoid collision with the Shared fetch key.
        lines.append("pbsm fetch <300 to 10000.0>")
        lines.append(_hru_val(1500))
        blowing_snow = self._get_config_value(lambda: self.config.model.crhm.blowing_snow, default=True)
        _write_param('pbsm', 'inhibit_bs', _hru_val(int(not blowing_snow)))
        _write_param('pbsm', 'inhibit_subl', _hru_val(int(not blowing_snow)))
        _write_param('pbsm', 'N_S', _hru_val(320))

        # ---- Soil ----
        _write_param('Soil', 'cov_type', _hru_val(1))
        _write_param('Soil', 'gw_init', _hru_val(75))
        _write_param('Soil', 'gw_K', _hru_val(0.001))
        _write_param('Soil', 'gw_max', _hru_val(150))
        _write_param('Soil', 'lower_ssr_K', _hru_val(0.001))
        _write_param('Soil', 'rechr_ssr_K', _hru_val(0.001))
        _write_param('Soil', 'Sdinit', _hru_val(0))
        _write_param('Soil', 'Sd_gw_K', _hru_val(0.001))
        _write_param('Soil', 'Sd_ssr_K', _hru_val(0.001))
        _write_param('Soil', 'soil_gw_K', _hru_val(0.001))
        _write_param('Soil', 'soil_moist_init', _hru_val(125))
        _write_param('Soil', 'soil_moist_max', _hru_val(250))
        _write_param('Soil', 'soil_rechr_init', _hru_val(30))
        _write_param('Soil', 'soil_ssr_runoff', _hru_val(1))
        # soil_withdrawal: 2-D (nlay x nhru)
        _write_param('Soil', 'soil_withdrawal',
                      '\n'.join([_hru_val(2)] * 2))
        _write_param('Soil', 'transp_limited', _hru_val(0))
        _write_param('Soil', 'Wetlands_scaling_factor', _hru_val(1))

        lines.append("######")

        # -- Initial_State (empty) -------------------------------------------
        lines.append("Initial_State")
        lines.append("######")
        lines.append("######")

        # -- Final_State (empty) ---------------------------------------------
        lines.append("Final_State")
        lines.append("######")
        lines.append("######")

        # -- Summary_period --------------------------------------------------
        lines.append("Summary_period")
        lines.append("######")
        lines.append("Water_year 10")
        lines.append("######")

        # -- Log_Time_Format -------------------------------------------------
        lines.append("Log_Time_Format")
        lines.append("######")
        lines.append("MS_DateTimeIndex")
        lines.append("######")

        # -- Display_Variable ------------------------------------------------
        # CRHM requires at least one selected output variable, otherwise
        # the model runs but produces no output and crashes (SIGSEGV)
        # during cleanup.  Select key hydrological state/flux variables
        # from the modules in the module list.
        lines.append("Display_Variable:")
        lines.append("######")

        # Build HRU index string: "1 2 3 ... nhru"
        hru_indices = ' '.join(str(i + 1) for i in range(nhru))

        # Key output variables -- names verified against CRHM source.
        # IMPORTANT: 'basinflowLoss' does NOT exist in standard CRHM;
        # 'runoff' should be 'soil_runoff'.  Only use variable names
        # that the CRHM binary actually registers.
        display_vars = [
            ('pbsm',     'SWE'),              # snow water equivalent (mm)
            ('Netroute', 'basinflow'),        # basin outflow (m3/s)
            ('Netroute', 'basingw'),          # basin groundwater flow (m3/s)
            ('Soil',     'soil_moist'),       # soil moisture (mm)
            ('Soil',     'soil_rechr'),       # recharge zone storage (mm)
            ('Soil',     'gw'),              # groundwater storage (mm)
            ('Soil',     'gw_flow'),          # groundwater flow (mm)
            ('evap',     'hru_actet'),        # actual evapotranspiration (mm)
            ('obs',      'hru_t'),            # HRU temperature (C)
            ('obs',      'hru_p'),            # HRU precipitation (mm)
        ]
        for module, var in display_vars:
            if module in self._DEFAULT_MODULES or module in ('obs',):
                indices = "1" if var in ("basinflow", "basingw") else hru_indices
                lines.append(f"{module} {var} {indices}")

        lines.append("######")

        # -- Display_Observation (empty) -------------------------------------
        lines.append("Display_Observation:")
        lines.append("######")
        lines.append("######")

        # -- Auto_Run --------------------------------------------------------
        lines.append("Auto_Run")
        lines.append("######")

        # -- Auto_Exit -------------------------------------------------------
        lines.append("Auto_Exit")
        lines.append("######")

        # -- Log_All ---------------------------------------------------------
        lines.append("Log_All")
        lines.append("######")

        # -- TChart (empty) --------------------------------------------------
        lines.append("TChart:")
        lines.append("######")
        lines.append("")
        lines.append("")
        lines.append("()")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("######")

        content = '\n'.join(lines) + '\n'
        prj_path.write_text(content, encoding='utf-8')
        logger.info(f"Project file written: {prj_path}")

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
