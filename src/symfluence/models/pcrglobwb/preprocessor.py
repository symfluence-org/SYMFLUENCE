# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB Model Preprocessor."""
from __future__ import annotations

import configparser
import math
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.modeling.model_ready.forcing_reader import (
    forcing_timestep_seconds,
    open_canonical_forcing,
)
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

_OPENDAP_BASE = (
    "https://opendap.4tu.nl/thredds/dodsC/data2/"
    "pcrglobwb/version_2019_11_beta/pcrglobwb2_input"
)


@R.preprocessors.add("PCRGLOBWB")
class PCRGLOBWBPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Prepares inputs for a PCR-GLOBWB 2.0 model run.

    PCR-GLOBWB requires:
    - Clone map (.map format, PCRaster binary) defining the grid
    - Meteorological forcing (precipitation m/day, temperature C, ref ET m/day)
    - INI configuration file (Python configparser format)
    - Input data directory with land surface, groundwater, and routing datasets
    """

    MODEL_NAME = "PCRGLOBWB"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.settings_dir = self.setup_dir
        self.forcing_out_dir = self.setup_dir / "forcing"
        self.output_dir = self.setup_dir / "output"

        self.resolution = self._get_config_value(
            lambda: self.config.model.pcrglobwb.resolution,
            default='05min', dict_key='PCRGLOBWB_RESOLUTION',
        )

    # USDA texture classes → hydraulic properties (Saxton & Rawls 2006)
    # {class: (Ksat m/day, porosity, field_capacity, wilting_point)}
    USDA_SOIL_PROPS = {
        1: (7.128, 0.437, 0.091, 0.033),   # Sand
        2: (3.502, 0.437, 0.125, 0.055),   # Loamy Sand
        3: (1.052, 0.453, 0.207, 0.095),   # Sandy Loam
        4: (0.727, 0.463, 0.270, 0.117),   # Silt Loam
        5: (0.132, 0.501, 0.330, 0.133),   # Silt
        6: (0.625, 0.398, 0.255, 0.148),   # Loam
        7: (0.311, 0.435, 0.296, 0.186),   # Sandy Clay Loam
        8: (0.170, 0.476, 0.365, 0.210),   # Clay Loam
        9: (0.102, 0.423, 0.325, 0.172),   # Silty Clay Loam
        10: (0.119, 0.406, 0.342, 0.250),  # Sandy Clay
        11: (0.061, 0.468, 0.387, 0.272),  # Silty Clay
        12: (0.058, 0.475, 0.396, 0.290),  # Clay
    }

    def run_preprocessing(self) -> bool:
        try:
            self.logger.info("Starting PCR-GLOBWB preprocessing...")
            self._create_directory_structure()
            self._build_domain_grid()
            self._prepare_clone_map()
            self._generate_forcing()
            self._generate_parameter_files()
            self._generate_ini_config()
            self.logger.info("PCR-GLOBWB preprocessing complete.")
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"PCR-GLOBWB preprocessing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        self.params_dir = self.settings_dir / 'parameters'
        for d in [self.settings_dir, self.forcing_out_dir, self.output_dir, self.params_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_catchment_properties(self) -> Dict:
        # Raises if the catchment shapefile is missing; never fall back to a
        # default location (that silently builds the model somewhere else).
        return self.get_catchment_location()

    def _build_domain_grid(self) -> None:
        """Build the model grid from the catchment bounding box.

        Snaps the catchment extent to the configured resolution and stores
        grid coordinates on ``self`` for use by all parameter generators.
        """
        self.cellsize = 5.0 / 60.0 if self.resolution == '05min' else 30.0 / 60.0
        props = self._get_catchment_properties()

        try:
            import geopandas as gpd
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                self._catchment_gdf = gdf
            else:
                c = self.cellsize
                bounds = [props['lon'] - 1.5 * c, props['lat'] - 1.5 * c,
                          props['lon'] + 1.5 * c, props['lat'] + 1.5 * c]
                self._catchment_gdf = None
        except Exception:  # noqa: BLE001
            c = self.cellsize
            bounds = [props['lon'] - 1.5 * c, props['lat'] - 1.5 * c,
                      props['lon'] + 1.5 * c, props['lat'] + 1.5 * c]
            self._catchment_gdf = None

        cs = self.cellsize
        # Snap to grid (expand outward by 1 cell for boundary effects)
        west = math.floor(bounds[0] / cs) * cs - cs
        south = math.floor(bounds[1] / cs) * cs - cs
        east = math.ceil(bounds[2] / cs) * cs + cs
        north = math.ceil(bounds[3] / cs) * cs + cs

        # Cell centers (lat descending N→S, lon ascending W→E)
        self.grid_lats = np.arange(north - cs / 2, south, -cs)
        self.grid_lons = np.arange(west + cs / 2, east, cs)
        self.nrows = len(self.grid_lats)
        self.ncols = len(self.grid_lons)

        # Grid corners for clone map
        self.grid_north = north
        self.grid_west = west

        self.logger.info(
            f"Domain grid: {self.nrows}×{self.ncols} cells at {self.resolution} "
            f"({self.grid_lats[0]:.2f}N to {self.grid_lats[-1]:.2f}N, "
            f"{self.grid_lons[0]:.2f}E to {self.grid_lons[-1]:.2f}E)"
        )

    def _resample_raster_to_grid(self, raster_path: Path, method: str = 'average') -> np.ndarray:
        """Resample a GeoTIFF raster to the model grid.

        Args:
            raster_path: Path to input GeoTIFF
            method: 'average' for continuous (DEM, Ksat), 'mode' for categorical (soil/land class)

        Returns:
            2D array (nrows, ncols) on the model grid. NaN outside raster extent.
        """
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        resampling = Resampling.average if method == 'average' else Resampling.mode

        with rasterio.open(raster_path) as src:
            # Target transform from grid coordinates
            cs = self.cellsize
            dst_transform = from_bounds(
                self.grid_lons[0] - cs / 2, self.grid_lats[-1] - cs / 2,
                self.grid_lons[-1] + cs / 2, self.grid_lats[0] + cs / 2,
                self.ncols, self.nrows,
            )

            dst = np.full((self.nrows, self.ncols), np.nan, dtype=np.float64)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=resampling,
                dst_nodata=np.nan,
            )

        return dst

    def _mask_to_catchment(self, grid: np.ndarray) -> np.ndarray:
        """Mask a grid array to the catchment boundary. Cells outside → NaN."""
        if self._catchment_gdf is None:
            return grid

        from rasterio.features import geometry_mask
        from rasterio.transform import from_bounds

        cs = self.cellsize
        transform = from_bounds(
            self.grid_lons[0] - cs / 2, self.grid_lats[-1] - cs / 2,
            self.grid_lons[-1] + cs / 2, self.grid_lats[0] + cs / 2,
            self.ncols, self.nrows,
        )
        mask = geometry_mask(
            self._catchment_gdf.geometry,
            out_shape=(self.nrows, self.ncols),
            transform=transform,
            invert=True,
        )
        result = grid.copy()
        result[~mask] = np.nan
        return result

    def _prepare_clone_map(self) -> None:
        """Prepare the PCRaster clone map.

        If the user provides an existing .map file via PCRGLOBWB_CLONE_MAP,
        copy it into the settings directory.  Otherwise, attempt to generate
        one using the PCRaster Python API.
        """
        clone_map_cfg = self._get_config_value(
            lambda: self.config.model.pcrglobwb.clone_map,
            default='clone.map', dict_key='PCRGLOBWB_CLONE_MAP',
        )
        target = self.settings_dir / 'clone.map'

        # Check if user provided an absolute path to an existing file
        src = Path(clone_map_cfg)
        if src.is_absolute() and src.exists():
            shutil.copy2(src, target)
            self.logger.info(f"Copied user-provided clone map from {src}")
            return

        # Check relative to data dir
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )
        src_rel = Path(data_dir) / clone_map_cfg
        if src_rel.exists():
            shutil.copy2(src_rel, target)
            self.logger.info(f"Copied clone map from {src_rel}")
            return

        # Generate a clone map using PCRaster
        self._generate_clone_map(target)

        # Re-read actual clone map attributes to align NetCDF coordinates
        # PCR-GLOBWB's virtualOS.py uses mapattr with limited precision,
        # so we must match those exact values in NetCDF coords
        self._align_grid_to_clone_map(target)

    def _generate_clone_map(self, target: Path) -> None:
        """Generate clone map matching the domain grid via PCRaster."""
        self._run_pcraster_script(
            f"import pcraster as pcr; "
            f"pcr.setclone({self.nrows}, {self.ncols}, {self.cellsize}, "
            f"{self.grid_west}, {self.grid_north}); "
            f"clone = pcr.boolean(1); "
            f"pcr.report(clone, '{target}')",
            target, "clone map",
        )

    def _align_grid_to_clone_map(self, clone_path: Path) -> None:
        """Read clone map attributes via mapattr and align grid coordinates.

        PCR-GLOBWB's virtualOS uses mapattr output (limited precision) for
        exact == comparisons with NetCDF coordinates. We read those values
        and rebuild grid_lats/grid_lons to match exactly.
        """
        import subprocess
        import sys

        pyver = f"{sys.version_info.major}{sys.version_info.minor}"
        for env_name in [f"pcraster{pyver}", "pcraster"]:
            try:
                result = run_subprocess(
                    ["conda", "run", "-n", env_name, "mapattr", "-p", str(clone_path)],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    parts = result.stdout.split()
                    cs = float(parts[7])
                    xUL = float(parts[17])
                    yUL = float(parts[19])
                    nrows = int(float(parts[3]))
                    ncols = int(float(parts[5]))

                    self.cellsize = cs
                    self.grid_north = yUL
                    self.grid_west = xUL
                    self.nrows = nrows
                    self.ncols = ncols
                    # Rebuild coords from mapattr precision
                    self.grid_lats = np.array([yUL - (i + 0.5) * cs for i in range(nrows)])
                    self.grid_lons = np.array([xUL + (j + 0.5) * cs for j in range(ncols)])

                    self.logger.info(
                        f"Aligned grid to clone map: {nrows}×{ncols}, cs={cs}, "
                        f"xUL={xUL}, yUL={yUL}"
                    )
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    def _run_pcraster_script(self, script: str, target: Path, label: str,
                             timeout: int = 60) -> bool:
        """Run a PCRaster Python script via direct import or conda env."""
        import subprocess
        import sys

        # Try conda pcraster envs
        pyver = f"{sys.version_info.major}{sys.version_info.minor}"
        for env_name in [f"pcraster{pyver}", "pcraster"]:
            try:
                result = run_subprocess(
                    ["conda", "run", "-n", env_name, "python", "-c", script],
                    capture_output=True, text=True, timeout=timeout,
                )
                if result.returncode == 0 and target.exists():
                    self.logger.info(f"Generated {label} via conda env '{env_name}'")
                    return True
                if result.stderr:
                    self.logger.debug(f"PCRaster stderr: {result.stderr[-300:]}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        self.logger.warning(f"Could not generate {label}")
        return False

    def _generate_ldd_map(self, dem_grid: np.ndarray = None) -> None:
        """Generate LDD from resampled DEM using PCRaster's lddcreate.

        For distributed grids, derives flow directions from the DEM.
        Falls back to uniform pit (value 5) if DEM is unavailable.
        """
        target = self.params_dir / 'ldd.map'

        if dem_grid is not None and np.isfinite(dem_grid).any():
            # Write DEM to temporary .map, run lddcreate, clean up
            dem_safe = np.where(np.isfinite(dem_grid), dem_grid, -9999.0)
            dem_tmp = self.params_dir / '_dem_for_ldd.map'
            self._write_map('_dem_for_ldd.map', dem_safe)

            if dem_tmp.exists():
                script = (
                    f"import pcraster as pcr; "
                    f"pcr.setclone('{dem_tmp}'); "
                    f"dem = pcr.readmap('{dem_tmp}'); "
                    f"dem = pcr.cover(dem, pcr.scalar(0)); "
                    f"ldd = pcr.lddcreate(dem, 1e31, 1e31, 1e31, 1e31); "
                    f"pcr.report(ldd, '{target}')"
                )
                if self._run_pcraster_script(script, target, "LDD from DEM", timeout=120):
                    dem_tmp.unlink(missing_ok=True)
                    return
                dem_tmp.unlink(missing_ok=True)

        # Fallback: uniform pit
        script = (
            f"import pcraster as pcr; import numpy as np; "
            f"pcr.setclone({self.nrows}, {self.ncols}, {self.cellsize}, "
            f"{self.grid_west}, {self.grid_north}); "
            f"ldd_arr = np.full(({self.nrows},{self.ncols}), 5, dtype=np.uint8); "
            f"ldd_pcr = pcr.numpy2pcr(pcr.Ldd, ldd_arr, 255); "
            f"pcr.report(ldd_pcr, '{target}')"
        )
        self._run_pcraster_script(script, target, "LDD (uniform pit)")

    def _generate_forcing(self) -> None:
        """Convert SYMFLUENCE forcing to PCR-GLOBWB format.

        PCR-GLOBWB expects daily data:
        - precipitation in m/day
        - temperature in degrees Celsius
        - reference ET in m/day (or computed via Hamon)

        Sub-daily forcing is resampled to daily (sum for precip, mean for temp).
        """
        self.logger.info("Generating PCR-GLOBWB forcing data...")
        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()
        forcing_path = self._get_forcing_path()

        forcing_files = sorted(forcing_path.glob('*.nc'))
        if not forcing_files:
            forcing_files = sorted(forcing_path.glob('**/*.nc'))
        # Only the forcing remapped onto this run's HRUs (issue #339).
        forcing_files = self._select_run_forcing(forcing_files)
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")

        # Canonical reader: variables under SYMFLUENCE names with a declared
        # timestep, so we no longer guess source names, units, or the timestep.
        ds_forcing = open_canonical_forcing(forcing_files)
        ds_forcing = ds_forcing.sel(time=slice(str(start_date), str(end_date)))

        time_vals = pd.DatetimeIndex(ds_forcing.time.values)
        dt_seconds = forcing_timestep_seconds(ds_forcing)
        dt_hours = dt_seconds / 3600.0
        is_subdaily = dt_hours < 24.0
        if is_subdaily:
            self.logger.info(
                f"Sub-daily forcing detected ({dt_hours:.0f}h) — "
                f"will resample to daily for PCR-GLOBWB"
            )

        # Precipitation: canonical precipitation_flux is a rate (kg m-2 s-1);
        # convert to metres per source step (m = mm / 1000).
        precip_raw = self._extract_forcing_var(ds_forcing, ['precipitation_flux'], props['lat'], props['lon'])
        precip_raw = np.maximum(precip_raw, 0.0)
        precip_m_per_step = precip_raw * dt_seconds / 1000.0

        # Temperature: canonical air_temperature is in kelvin.
        temp_raw = self._extract_forcing_var(ds_forcing, ['air_temperature'], props['lat'], props['lon']) - 273.15

        ds_forcing.close()

        # Build daily series
        precip_series = pd.Series(precip_m_per_step, index=time_vals)
        temp_series = pd.Series(temp_raw, index=time_vals)

        if is_subdaily:
            precip_daily = precip_series.resample('D').sum()
            temp_daily = temp_series.resample('D').mean()
        else:
            precip_daily = precip_series
            temp_daily = temp_series

        daily_times = precip_daily.index
        precip = precip_daily.values
        temp = temp_daily.values

        # Estimate PET in m/day from daily temperature
        pet = self._estimate_pet(temp, daily_times, props['lat'])

        self.logger.info(
            f"Forcing summary: {len(daily_times)} days, "
            f"P mean={precip.mean()*1000:.1f} mm/day, "
            f"T mean={temp.mean():.1f} C, "
            f"PET mean={pet.mean()*1000:.2f} mm/day"
        )

        # Broadcast to domain grid (basin-averaged forcing applied uniformly)
        nt = len(daily_times)
        coords = {'time': daily_times, 'lat': self.grid_lats, 'lon': self.grid_lons}

        for varname, data, units in [
            ('precipitation', precip, 'm.day-1'),
            ('temperature', temp, 'degrees_C'),
            ('referencePotET', pet, 'm.day-1'),
        ]:
            da = xr.DataArray(
                np.broadcast_to(
                    data.reshape(nt, 1, 1), (nt, self.nrows, self.ncols)
                ).copy(),
                dims=['time', 'lat', 'lon'],
                coords=coords,
                attrs={'units': units, 'missing_value': 1e20},
                name=varname,
            )
            out_path = self.forcing_out_dir / f'{varname}.nc'
            self._write_nc(out_path, da.to_dataset())
            self.logger.info(f"Written {varname} forcing to {out_path}")

    def _extract_forcing_var(self, ds, var_names, lat, lon):
        for var in var_names:
            if var in ds.data_vars:
                data = ds[var]
                spatial_dims = [d for d in data.dims if d not in ['time']]
                if spatial_dims:
                    try:
                        data = data.sel(
                            **{d: lat if 'lat' in d else lon for d in spatial_dims},
                            method='nearest',
                        )
                    except Exception:  # noqa: BLE001
                        data = data.isel(**{d: 0 for d in spatial_dims})
                return data.values.flatten()
        raise ValueError(
            f"None of {var_names} found in forcing. Available: {list(ds.data_vars)}"
        )

    def _estimate_pet(self, temp_c, times, lat_deg):
        """Estimate PET and return values in m/day."""
        from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin

        pet_method = self._get_config_value(
            lambda: self.config.model.pcrglobwb.pet_method,
            default='hamon', dict_key='PCRGLOBWB_PET_METHOD',
        )

        doy = np.array([t.timetuple().tm_yday for t in times])

        if pet_method == 'oudin':
            pet_mm_day = PETCalculatorMixin.oudin_pet_numpy(temp_c, doy, lat_deg)
            self.logger.info(
                f"Oudin PET: annual mean = {pet_mm_day.mean() * 365.25:.0f} mm/yr"
            )
            return pet_mm_day / 1000.0  # mm/day -> m/day

        # Default: Hamon
        self.logger.info(f"Using Hamon PET (method={pet_method})")
        return self._estimate_pet_hamon(temp_c, times, lat_deg)

    def _estimate_pet_hamon(self, temp_c, times, lat_deg):
        doy = np.array([t.timetuple().tm_yday for t in times])
        lat_rad = math.radians(lat_deg)
        decl = 0.4093 * np.sin(2 * np.pi / 365 * doy - 1.405)
        cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1)
        day_length = 24 / np.pi * np.arccos(cos_omega)
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        pet_mm_day = np.maximum(
            0.1651 * (day_length / 12.0) * es * 216.7 / (temp_c + 273.3), 0.0
        )
        return pet_mm_day / 1000.0  # mm/day -> m/day

    def _get_forcing_path(self) -> Path:
        forcing_path = self._get_config_value(
            lambda: self.config.paths.forcing_path,
            default=None, dict_key='FORCING_PATH',
        )
        if forcing_path and forcing_path != 'default':
            return Path(forcing_path)
        # Store-first basin-averaged forcing (model_ready/forcings when present,
        # else legacy forcing/basin_averaged_data) — resolved by the base class.
        return self.forcing_basin_path

    def _write_nc(self, path: Path, ds: xr.Dataset) -> None:
        """Write a NetCDF without _FillValue on any variable.

        PCRaster's numpy2pcr rejects masked arrays, and PCR-GLOBWB's
        virtualOS coordinate matching breaks with masked coordinates.
        We suppress _FillValue on ALL variables including coords.
        """
        encoding = {}
        for v in ds.data_vars:
            encoding[v] = {'_FillValue': None}
        for c in ds.coords:
            encoding[c] = {'_FillValue': None}
        ds.to_netcdf(path, encoding=encoding)

    def _write_map(self, filename: str, arr: np.ndarray, pcr_type: str = 'Scalar') -> Path:
        """Write a 2D array as a PCRaster .map file via conda env.

        All grid-aligned parameter files use .map format to avoid
        floating-point coordinate mismatches in PCR-GLOBWB's NetCDF reader.
        """
        target = self.params_dir / filename
        grid = self._to_grid(arr)
        # Replace NaN with nodata
        grid_safe = np.where(np.isnan(grid), -9999.0, grid)
        self._run_pcraster_script(
            f"import pcraster as pcr; import numpy as np; "
            f"pcr.setclone({self.nrows}, {self.ncols}, {self.cellsize}, "
            f"{self.grid_west}, {self.grid_north}); "
            f"a = np.array({grid_safe.tolist()}, dtype=np.float64); "
            f"pcr.report(pcr.numpy2pcr(pcr.{pcr_type}, a, -9999.0), "
            f"'{target}')",
            target, filename,
        )
        return target

    def _to_grid(self, value) -> np.ndarray:
        """Expand a scalar or 2D array to the domain grid shape."""
        if isinstance(value, np.ndarray) and value.shape == (self.nrows, self.ncols):
            return value.astype(np.float64)
        return np.full((self.nrows, self.ncols), float(value), dtype=np.float64)

    def _make_static_nc(self, filename: str, variables: Dict) -> Path:
        """Write a static parameter NetCDF on the domain grid (no time dim).

        Values can be scalars (uniform) or 2D arrays (spatially varying).
        """
        coords = {'lat': self.grid_lats, 'lon': self.grid_lons}
        ds = xr.Dataset(coords=coords)
        for name, value in variables.items():
            ds[name] = xr.DataArray(
                self._to_grid(value), dims=['lat', 'lon'], coords=coords,
                attrs={'missing_value': 1e20},
            )
        out = self.params_dir / filename
        self._write_nc(out, ds)
        return out

    def _make_param_nc(self, filename: str, variables: Dict) -> Path:
        """Write a parameter NetCDF with a time dimension on the domain grid."""
        coords = {
            'time': pd.to_datetime(['2000-01-01']),
            'lat': self.grid_lats,
            'lon': self.grid_lons,
        }
        ds = xr.Dataset(coords=coords)
        for name, value in variables.items():
            grid = self._to_grid(value)
            ds[name] = xr.DataArray(
                grid.reshape(1, self.nrows, self.ncols),
                dims=['time', 'lat', 'lon'], coords=coords,
                attrs={'missing_value': 1e20},
            )
        out = self.params_dir / filename
        self._write_nc(out, ds)
        return out

    def _make_landcover_maps(self, prefix: str, cover_frac) -> None:
        """Write land cover properties as individual .map files."""
        for name, val in [
            ('fracVegCover', cover_frac),
            ('rootFraction1', 0.6),
            ('rootFraction2', 0.4),
            ('maxRootDepth', 1.5),
            ('minSoilDepthFrac', 0.05),
            ('maxSoilDepthFrac', 1.0),
        ]:
            self._write_map(f'{prefix}_{name}.map', val)

    def _make_timeseries_nc(self, filename: str, value, varname: str | None = None) -> Path:
        """Write a 366-day daily-cycle NetCDF on the domain grid."""
        times = pd.date_range('2000-01-01', periods=366, freq='D')
        coords = {'time': times, 'lat': self.grid_lats, 'lon': self.grid_lons}
        grid = self._to_grid(value)
        data = np.broadcast_to(grid, (366, self.nrows, self.ncols)).copy()
        if varname is None:
            varname = Path(filename).stem
        ds = xr.Dataset({
            varname: xr.DataArray(
                data, dims=['time', 'lat', 'lon'], coords=coords,
                attrs={'missing_value': 1e20},
            )
        })
        out = self.params_dir / filename
        self._write_nc(out, ds)
        return out

    def _generate_parameter_files(self) -> None:
        """Generate spatially varying PCR-GLOBWB parameter NetCDFs.

        Resamples domain DEM, soil, and land cover rasters to the model
        grid, then applies pedotransfer functions and geomorphological
        relationships to derive all required parameters.
        """
        self.logger.info("Generating PCR-GLOBWB parameter files from domain data...")

        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )
        domain_name = self._get_config_value(
            lambda: self.config.domain.name, default='', dict_key='DOMAIN_NAME',
        )
        attr_base = Path(data_dir) / f'domain_{domain_name}' / 'attributes'

        # ── Resample rasters to model grid ────────────────────────────
        dem_path = attr_base / 'elevation' / 'dem' / f'domain_{domain_name}_elv.tif'
        lc_path = attr_base / 'landclass' / f'domain_{domain_name}_land_classes.tif'
        soil_path = attr_base / 'soilclass' / f'domain_{domain_name}_soil_classes.tif'

        # DEM → elevation and slope grids
        if dem_path.exists():
            dem_grid = self._resample_raster_to_grid(dem_path, method='average')
            dem_grid = self._mask_to_catchment(dem_grid)
            dy, dx = np.gradient(dem_grid)
            cs_m = self.cellsize * 111000  # approximate meters per degree
            slope_grid = np.sqrt((dx / cs_m) ** 2 + (dy / cs_m) ** 2)
            slope_grid = np.where(np.isfinite(slope_grid), slope_grid, 0.15)
        else:
            dem_grid = np.full((self.nrows, self.ncols), 1500.0)
            slope_grid = np.full((self.nrows, self.ncols), 0.15)
            dem_grid = self._mask_to_catchment(dem_grid)

        # Soil class → hydraulic properties per cell
        if soil_path.exists():
            soil_grid = self._resample_raster_to_grid(soil_path, method='mode')
            soil_grid = self._mask_to_catchment(soil_grid)
        else:
            soil_grid = np.full((self.nrows, self.ncols), 3.0)  # sandy loam default

        ksat_grid = np.full_like(dem_grid, 0.3)
        poro_grid = np.full_like(dem_grid, 0.43)
        fc_grid = np.full_like(dem_grid, 0.18)
        wp_grid = np.full_like(dem_grid, 0.085)
        for cls, (ks, po, fc, wp) in self.USDA_SOIL_PROPS.items():
            mask = soil_grid == cls
            ksat_grid[mask] = ks
            poro_grid[mask] = po
            fc_grid[mask] = fc
            wp_grid[mask] = wp

        # Land cover → forest/grass fractions
        if lc_path.exists():
            lc_grid = self._resample_raster_to_grid(lc_path, method='mode')
            lc_grid = self._mask_to_catchment(lc_grid)
            forest_grid = np.where(np.isin(lc_grid, [1, 2, 3, 4, 5]), 1.0, 0.0)
        else:
            forest_grid = np.full((self.nrows, self.ncols), 0.3)
        grass_grid = np.where(np.isnan(forest_grid), np.nan, 1.0 - forest_grid)

        self.logger.info(
            f"Distributed grids: elev {np.nanmean(dem_grid):.0f}m, "
            f"slope {np.nanmean(slope_grid):.3f}, "
            f"forest {np.nanmean(forest_grid):.0%}, "
            f"Ksat {np.nanmean(ksat_grid):.3f} m/d"
        )

        # ── Topography — individual .map files ────────────────────────
        self._write_map('tanslope.map', slope_grid)
        self._write_map('slopeLength.map', 200.0)
        self._write_map('orographyBeta.map', 0.0)
        self._write_map('dem_average.map', dem_grid)
        self._write_map('dem_standard_deviation.map', 200.0)
        for level in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            self._write_map(f'dzRel{level:04d}.map', level / 100.0)

        # ── Soil properties — individual .map files ──────────────────
        stor_cap_upp = (poro_grid - wp_grid) * 0.3
        stor_cap_low = (poro_grid - wp_grid) * 1.2
        for name, arr in [
            ('KSat1', ksat_grid), ('KSat2', ksat_grid * 0.5),
            ('airEntryValue1', 0.1), ('airEntryValue2', 0.1),
            ('poreSizeBeta1', 5.0), ('poreSizeBeta2', 5.0),
            ('resVolWC1', wp_grid), ('resVolWC2', wp_grid),
            ('satVolWC1', poro_grid), ('satVolWC2', poro_grid),
            ('firstStorDepth', 0.3), ('secondStorDepth', 1.2),
            ('soilDepth', 1.5), ('percolationImp', 1.0e-4),
            ('soilWaterStorageCap1', stor_cap_upp), ('soilWaterStorageCap2', stor_cap_low),
        ]:
            self._write_map(f'{name}.map', arr)

        # ── Groundwater properties — individual .map files ───────────
        specific_yield = np.maximum(poro_grid - fc_grid, 0.05)
        recession_coeff = np.minimum(ksat_grid / (100.0 * specific_yield), 0.01)
        for name, arr in [
            ('kSatAquifer', ksat_grid * 0.1),
            ('specificYield', specific_yield),
            ('recessionCoeff', recession_coeff),
        ]:
            self._write_map(f'{name}.map', arr)
        self._write_map('thickness.map', 100.0)

        # ── LDD from DEM ─────────────────────────────────────────────
        self._generate_ldd_map(dem_grid)

        # ── Cell area (varies with latitude) — as .map for readPCRmapClone ──
        cell_areas = np.zeros((self.nrows, self.ncols))
        for i, lat_i in enumerate(self.grid_lats):
            dx = self.cellsize * 111320 * math.cos(math.radians(lat_i))
            dy = self.cellsize * 110540
            cell_areas[i, :] = dx * dy
        self._run_pcraster_script(
            f"import pcraster as pcr; import numpy as np; "
            f"pcr.setclone({self.nrows}, {self.ncols}, {self.cellsize}, "
            f"{self.grid_west}, {self.grid_north}); "
            f"a = np.array({cell_areas.tolist()}, dtype=np.float64); "
            f"pcr.report(pcr.numpy2pcr(pcr.Scalar, a, -9999.0), "
            f"'{self.params_dir / 'cellarea.map'}')",
            self.params_dir / 'cellarea.map', 'cell area map',
        )

        # ── Channel properties & water bodies — as .map files ────────
        chan_gradient = np.maximum(slope_grid * 0.1, 0.001)
        avg_q_est = 40.0
        chan_width_val = 7.0 * avg_q_est ** 0.5
        chan_depth_val = 0.27 * avg_q_est ** 0.39
        for fname, arr in [
            ('channel_gradient.map', chan_gradient),
            ('bankfull_depth.map', np.full((self.nrows, self.ncols), chan_depth_val)),
            ('bankfull_width.map', np.full((self.nrows, self.ncols), chan_width_val)),
        ]:
            self._run_pcraster_script(
                f"import pcraster as pcr; import numpy as np; "
                f"pcr.setclone({self.nrows}, {self.ncols}, {self.cellsize}, "
                f"{self.grid_west}, {self.grid_north}); "
                f"a = np.array({arr.tolist()}, dtype=np.float64); "
                f"pcr.report(pcr.numpy2pcr(pcr.Scalar, a, -9999.0), "
                f"'{self.params_dir / fname}')",
                self.params_dir / fname, fname,
            )

        # Water bodies — all zeros, as .map
        self._run_pcraster_script(
            f"import pcraster as pcr; import numpy as np; "
            f"pcr.setclone({self.nrows}, {self.ncols}, {self.cellsize}, "
            f"{self.grid_west}, {self.grid_north}); "
            f"a = np.zeros(({self.nrows},{self.ncols}), dtype=np.float64); "
            f"pcr.report(pcr.numpy2pcr(pcr.Nominal, a.astype(np.int32), -9999), "
            f"'{self.params_dir / 'waterBodyIds.map'}')",
            self.params_dir / 'waterBodyIds.map', 'waterBodyIds',
        )

        # ── Crop coefficients for open water ─────────────────────────
        self._make_timeseries_nc('cropCoefficientForOpenWater.nc', 1.0, varname='kc')

        # ── Forest land cover ────────────────────────────────────────
        self._make_landcover_maps('forest', forest_grid)
        self._make_timeseries_nc('cropKc_forest.nc', 1.0, varname='kc')
        self._make_timeseries_nc('interceptCap_forest.nc', 0.002, varname='interceptCapInput')
        self._make_timeseries_nc('coverFraction_forest.nc', forest_grid, varname='coverFractionInput')

        # ── Grassland land cover ─────────────────────────────────────
        self._make_landcover_maps('grassland', grass_grid)
        self._make_timeseries_nc('cropKc_grassland.nc', 0.8, varname='kc')
        self._make_timeseries_nc('interceptCap_grassland.nc', 0.001, varname='interceptCapInput')
        self._make_timeseries_nc('coverFraction_grassland.nc', grass_grid, varname='coverFractionInput')

        # ── Irrigation (zero for natural run) ────────────────────────
        self._make_landcover_maps('irrPaddy', 0.0)
        self._make_timeseries_nc('cropKc_irrPaddy.nc', 1.0, varname='kc')
        self._make_landcover_maps('irrNonPaddy', 0.0)
        self._make_timeseries_nc('cropKc_irrNonPaddy.nc', 1.0, varname='kc')

        self.logger.info(
            f"Generated {len(list(self.params_dir.glob('*.nc')))} parameter files "
            f"+ LDD on {self.nrows}×{self.ncols} grid"
        )

    def _estimate_initial_conditions(self) -> Dict:
        """Derive physically based initial conditions from forcing climatology.

        Uses mean annual precipitation, temperature, and PET to estimate
        equilibrium states for soil moisture, groundwater, snow, and
        channel discharge.
        """
        # Read back the generated daily forcing
        try:
            precip_ds = xr.open_dataset(self.forcing_out_dir / 'precipitation.nc')
            temp_ds = xr.open_dataset(self.forcing_out_dir / 'temperature.nc')
            pet_ds = xr.open_dataset(self.forcing_out_dir / 'referencePotET.nc')

            p_mean = float(precip_ds['precipitation'].mean())  # m/day
            t_mean = float(temp_ds['temperature'].mean())      # °C
            pet_mean = float(pet_ds['referencePotET'].mean())   # m/day

            precip_ds.close()
            temp_ds.close()
            pet_ds.close()
        except Exception:  # noqa: BLE001
            p_mean = 0.002   # ~730 mm/yr
            t_mean = 5.0
            pet_mean = 0.001

        # Aridity index (PET/P) — controls runoff coefficient
        aridity = pet_mean / max(p_mean, 1e-6)
        # Budyko runoff coefficient: Q/P ≈ 1 - tanh(PET/P)
        runoff_coeff = max(0.05, 1.0 - math.tanh(aridity))

        # Mean annual values
        map_mm = p_mean * 365.25 * 1000  # mean annual precip in mm
        baseflow_frac = 0.4  # baseflow index ~40% for temperate catchments

        # Soil moisture: approximate field capacity storage (m)
        # Upper soil (~0.3m): porosity ~0.4, field capacity ~60% of porosity
        stor_upp = 0.3 * 0.4 * 0.6  # ~0.072 m
        # Lower soil (~1.2m): slightly drier equilibrium
        stor_low = 1.2 * 0.4 * 0.5  # ~0.24 m

        # Scale soil moisture by wetness (wetter climates → closer to FC)
        wetness = min(1.0, p_mean / max(pet_mean, 1e-6))
        stor_upp *= wetness
        stor_low *= wetness

        # Snow: estimate SWE from temperature regime
        if t_mean < -5.0:
            # Cold climate — significant snowpack
            snow_swe = p_mean * 90  # ~3 months accumulation (m)
        elif t_mean < 0.0:
            snow_swe = p_mean * 30  # ~1 month
        elif t_mean < 5.0:
            snow_swe = p_mean * 5   # minor
        else:
            snow_swe = 0.0

        # Groundwater storage (m): recharge ≈ 10-30% of precipitation
        recharge_frac = 0.15 * wetness
        gw_storage = p_mean * 365.25 * recharge_frac  # annual recharge as proxy

        # Channel discharge (m3/s) from catchment area
        catchment_area_m2 = self._get_catchment_properties().get('area_m2', 2.0e9)

        # Q = P * runoff_coeff * area / seconds_per_day
        avg_discharge = (p_mean * runoff_coeff * catchment_area_m2) / 86400.0
        avg_baseflow = avg_discharge * baseflow_frac

        self.logger.info(
            f"Initial conditions from climatology: "
            f"MAP={map_mm:.0f} mm/yr, T_mean={t_mean:.1f}°C, "
            f"aridity={aridity:.2f}, runoff_coeff={runoff_coeff:.2f}, "
            f"Q_est={avg_discharge:.1f} m3/s, SWE={snow_swe*1000:.0f} mm"
        )

        return {
            'stor_upp': f'{stor_upp:.4f}',
            'stor_low': f'{stor_low:.4f}',
            'snow_swe': f'{snow_swe:.4f}',
            'gw_storage': f'{gw_storage:.4f}',
            'avg_discharge': f'{avg_discharge:.2f}',
            'avg_baseflow': f'{avg_baseflow:.2f}',
            'interflow': '0.0',
            'intercept_stor': '0.0',
            'snow_free_water': '0.0',
            'top_water_layer': '0.0',
            'channel_storage': '0.0',
            'water_body_storage': '0.0',
        }

    def _generate_ini_config(self) -> None:
        """Generate the PCR-GLOBWB INI configuration file.

        Produces a complete INI with all 10 sections required by
        PCR-GLOBWB 2.0, using physically based initial conditions
        estimated from the forcing climatology.
        """
        self.logger.info("Generating PCR-GLOBWB INI configuration...")
        start_date, end_date = self._get_simulation_dates()

        config_file = self._get_config_value(
            lambda: self.config.model.pcrglobwb.config_file,
            default='setup.ini',
        )
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='', dict_key='DOMAIN_NAME',
        )
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default='run_1', dict_key='EXPERIMENT_ID',
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )

        sim_output_dir = (
            Path(data_dir) / f'domain_{domain_name}'
            / 'simulations' / experiment_id / 'PCRGLOBWB'
        )
        sim_output_dir.mkdir(parents=True, exist_ok=True)

        spinup_years = self._get_config_value(
            lambda: self.config.model.pcrglobwb.spinup_years,
            default=0, dict_key='PCRGLOBWB_SPINUP_YEARS',
        )

        p = self.params_dir  # shorthand for generated parameter paths

        # Estimate initial conditions from forcing climatology
        ic = self._estimate_initial_conditions()

        ini = configparser.ConfigParser()
        ini.optionxform = str  # preserve case

        # ── globalOptions ─────────────────────────────────────────────
        ini['globalOptions'] = {
            'inputDir': str(self.settings_dir),
            'outputDir': str(sim_output_dir),
            'cloneMap': str(self.settings_dir / 'clone.map'),
            'landmask': 'None',
            'startTime': start_date.strftime('%Y-%m-%d'),
            'endTime': end_date.strftime('%Y-%m-%d'),
            'maxSpinUpsInYears': str(spinup_years),
            'minConvForSoilSto': '0.0',
            'minConvForGwatSto': '0.0',
            'minConvForChanSto': '0.0',
            'minConvForTotlSto': '0.0',
            'institution': 'SYMFLUENCE',
            'title': f'PCR-GLOBWB 2 output — {domain_name}',
            'description': f'PCR-GLOBWB run at {self.resolution} resolution, natural mode',
        }

        # ── meteoOptions ──────────────────────────────────────────────
        ini['meteoOptions'] = {
            'precipitationNC': str(self.forcing_out_dir / 'precipitation.nc'),
            'precipitationVariableName': 'precipitation',
            'precipitationCorrectionFactor': '1.0',
            'temperatureNC': str(self.forcing_out_dir / 'temperature.nc'),
            'temperatureVariableName': 'temperature',
            'referenceETPotMethod': 'Input',
            'refETPotFileNC': str(self.forcing_out_dir / 'referencePotET.nc'),
            'referenceEPotVariableName': 'referencePotET',
        }

        # ── landSurfaceOptions ────────────────────────────────────────
        ini['landSurfaceOptions'] = {
            'debugWaterBalance': 'True',
            'numberOfUpperSoilLayers': '2',
            'topographyNC': 'None',
            'soilPropertiesNC': 'None',
            # Individual .map files for topography
            'tanslope': str(p / 'tanslope.map'),
            'slopeLength': str(p / 'slopeLength.map'),
            'orographyBeta': str(p / 'orographyBeta.map'),
            'dem_average': str(p / 'dem_average.map'),
            'dem_standard_deviation': str(p / 'dem_standard_deviation.map'),
            # Individual .map files for soil
            'KSat1': str(p / 'KSat1.map'),
            'KSat2': str(p / 'KSat2.map'),
            'airEntryValue1': str(p / 'airEntryValue1.map'),
            'airEntryValue2': str(p / 'airEntryValue2.map'),
            'poreSizeBeta1': str(p / 'poreSizeBeta1.map'),
            'poreSizeBeta2': str(p / 'poreSizeBeta2.map'),
            'resVolWC1': str(p / 'resVolWC1.map'),
            'resVolWC2': str(p / 'resVolWC2.map'),
            'satVolWC1': str(p / 'satVolWC1.map'),
            'satVolWC2': str(p / 'satVolWC2.map'),
            'percolationImp': str(p / 'percolationImp.map'),
            'firstStorDepth': str(p / 'firstStorDepth.map'),
            'secondStorDepth': str(p / 'secondStorDepth.map'),
            'soilWaterStorageCap1': str(p / 'soilWaterStorageCap1.map'),
            'soilWaterStorageCap2': str(p / 'soilWaterStorageCap2.map'),
            # dzRel relative elevation levels
            **{f'dzRel{level:04d}': str(p / f'dzRel{level:04d}.map')
               for level in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
            'includeIrrigation': 'False',
            'includeDomesticWaterDemand': 'False',
            'includeIndustryWaterDemand': 'False',
            'includeLivestockWaterDemand': 'False',
        }

        # ── forestOptions (naturalTall) ───────────────────────────────
        _lc_common = {
            'debugWaterBalance': 'True',
            'snowModuleType': 'Simple',
            'freezingT': '0.0',
            'degreeDayFactor': '0.0025',
            'snowWaterHoldingCap': '0.1',
            'refreezingCoeff': '0.05',
            'minTopWaterLayer': '0.0',
            'minCropKC': '0.2',
        }

        ini['forestOptions'] = {
            'name': 'forest',
            **_lc_common,
            'cropCoefficientNC': str(p / 'cropKc_forest.nc'),
            'interceptCapNC': str(p / 'interceptCap_forest.nc'),
            'coverFractionNC': str(p / 'coverFraction_forest.nc'),
            'landCoverMapsNC': 'None',
            'fracVegCover': str(p / 'forest_fracVegCover.map'),
            'rootFraction1': str(p / 'forest_rootFraction1.map'),
            'rootFraction2': str(p / 'forest_rootFraction2.map'),
            'maxRootDepth': str(p / 'forest_maxRootDepth.map'),
            'minSoilDepthFrac': str(p / 'forest_minSoilDepthFrac.map'),
            'maxSoilDepthFrac': str(p / 'forest_maxSoilDepthFrac.map'),
            'interceptStorIni': ic['intercept_stor'],
            'snowCoverSWEIni': ic['snow_swe'],
            'snowFreeWaterIni': ic['snow_free_water'],
            'topWaterLayerIni': ic['top_water_layer'],
            'storUppIni': ic['stor_upp'],
            'storLowIni': ic['stor_low'],
            'interflowIni': ic['interflow'],
        }

        # ── grasslandOptions (naturalShort) ───────────────────────────
        ini['grasslandOptions'] = {
            'name': 'grassland',
            **_lc_common,
            'cropCoefficientNC': str(p / 'cropKc_grassland.nc'),
            'interceptCapNC': str(p / 'interceptCap_grassland.nc'),
            'coverFractionNC': str(p / 'coverFraction_grassland.nc'),
            'landCoverMapsNC': 'None',
            'fracVegCover': str(p / 'grassland_fracVegCover.map'),
            'rootFraction1': str(p / 'grassland_rootFraction1.map'),
            'rootFraction2': str(p / 'grassland_rootFraction2.map'),
            'maxRootDepth': str(p / 'grassland_maxRootDepth.map'),
            'minSoilDepthFrac': str(p / 'grassland_minSoilDepthFrac.map'),
            'maxSoilDepthFrac': str(p / 'grassland_maxSoilDepthFrac.map'),
            'interceptStorIni': ic['intercept_stor'],
            'snowCoverSWEIni': ic['snow_swe'],
            'snowFreeWaterIni': ic['snow_free_water'],
            'topWaterLayerIni': ic['top_water_layer'],
            'storUppIni': ic['stor_upp'],
            'storLowIni': ic['stor_low'],
            'interflowIni': ic['interflow'],
        }

        # ── irrPaddyOptions ───────────────────────────────────────────
        ini['irrPaddyOptions'] = {
            'name': 'irrPaddy',
            **_lc_common,
            'minTopWaterLayer': '0.05',
            'cropDeplFactor': '0.2',
            'minInterceptCap': '0.0002',
            'landCoverMapsNC': 'None',
            'fracVegCover': str(p / 'irrPaddy_fracVegCover.map'),
            'rootFraction1': str(p / 'irrPaddy_rootFraction1.map'),
            'rootFraction2': str(p / 'irrPaddy_rootFraction2.map'),
            'maxRootDepth': str(p / 'irrPaddy_maxRootDepth.map'),
            'minSoilDepthFrac': str(p / 'irrPaddy_minSoilDepthFrac.map'),
            'maxSoilDepthFrac': str(p / 'irrPaddy_maxSoilDepthFrac.map'),
            'cropCoefficientNC': str(p / 'cropKc_irrPaddy.nc'),
            'interceptStorIni': '0.0',
            'snowCoverSWEIni': ic['snow_swe'],
            'snowFreeWaterIni': '0.0',
            'topWaterLayerIni': '0.0',
            'storUppIni': ic['stor_upp'],
            'storLowIni': ic['stor_low'],
            'interflowIni': '0.0',
        }

        # ── irrNonPaddyOptions ────────────────────────────────────────
        ini['irrNonPaddyOptions'] = {
            'name': 'irrNonPaddy',
            **_lc_common,
            'cropDeplFactor': '0.5',
            'minInterceptCap': '0.0002',
            'landCoverMapsNC': 'None',
            'fracVegCover': str(p / 'irrNonPaddy_fracVegCover.map'),
            'rootFraction1': str(p / 'irrNonPaddy_rootFraction1.map'),
            'rootFraction2': str(p / 'irrNonPaddy_rootFraction2.map'),
            'maxRootDepth': str(p / 'irrNonPaddy_maxRootDepth.map'),
            'minSoilDepthFrac': str(p / 'irrNonPaddy_minSoilDepthFrac.map'),
            'maxSoilDepthFrac': str(p / 'irrNonPaddy_maxSoilDepthFrac.map'),
            'cropCoefficientNC': str(p / 'cropKc_irrNonPaddy.nc'),
            'interceptStorIni': '0.0',
            'snowCoverSWEIni': ic['snow_swe'],
            'snowFreeWaterIni': '0.0',
            'topWaterLayerIni': '0.0',
            'storUppIni': ic['stor_upp'],
            'storLowIni': ic['stor_low'],
            'interflowIni': '0.0',
        }

        # ── groundwaterOptions ────────────────────────────────────────
        ini['groundwaterOptions'] = {
            'debugWaterBalance': 'True',
            'groundwaterPropertiesNC': 'None',
            'kSatAquifer': str(p / 'kSatAquifer.map'),
            'specificYield': str(p / 'specificYield.map'),
            'recessionCoeff': str(p / 'recessionCoeff.map'),
            'minRecessionCoeff': '1.0e-4',
            'limitFossilGroundWaterAbstraction': 'True',
            'estimateOfRenewableGroundwaterCapacity': '0.0',
            'estimateOfTotalGroundwaterThickness': str(p / 'thickness.map'),
            'minimumTotalGroundwaterThickness': '100.',
            'maximumTotalGroundwaterThickness': 'None',
            'pumpingCapacityNC': 'None',
            'storGroundwaterIni': ic['gw_storage'],
            'storGroundwaterFossilIni': '0.0',
            'avgNonFossilGroundwaterAllocationLongIni': '0.0',
            'avgNonFossilGroundwaterAllocationShortIni': '0.0',
            'avgTotalGroundwaterAbstractionIni': '0.0',
            'avgTotalGroundwaterAllocationLongIni': '0.0',
            'avgTotalGroundwaterAllocationShortIni': '0.0',
            'relativeGroundwaterHeadIni': '0.0',
            'baseflowIni': ic['avg_baseflow'],
        }

        # ── routingOptions ────────────────────────────────────────────
        ini['routingOptions'] = {
            'debugWaterBalance': 'True',
            'lddMap': str(p / 'ldd.map'),
            'cellAreaMap': str(p / 'cellarea.map'),
            'routingMethod': 'accuTravelTime',
            'manningsN': '0.04',
            'dynamicFloodPlain': 'False',
            'gradient': str(p / 'channel_gradient.map'),
            'constantChannelDepth': str(p / 'bankfull_depth.map'),
            'constantChannelWidth': str(p / 'bankfull_width.map'),
            'minimumChannelWidth': str(p / 'bankfull_width.map'),
            'bankfullCapacity': 'None',
            'cropCoefficientWaterNC': str(p / 'cropCoefficientForOpenWater.nc'),
            'minCropWaterKC': '1.00',
            'waterBodyInputNC': 'None',
            'waterBodyIds': 'None',
            'waterBodyTyp': 'None',
            'fracWaterInp': 'None',
            'resSfAreaInp': 'None',
            'resMaxCapInp': 'None',
            'onlyNaturalWaterBodies': 'True',
            'waterBodyStorageIni': ic['water_body_storage'],
            'channelStorageIni': ic['channel_storage'],
            'readAvlChannelStorageIni': '0.0',
            'avgDischargeLongIni': ic['avg_discharge'],
            'avgDischargeShortIni': ic['avg_discharge'],
            'm2tDischargeLongIni': '0.0',
            'avgBaseflowLongIni': ic['avg_baseflow'],
            'riverbedExchangeIni': '0.0',
            'subDischargeIni': ic['avg_discharge'],
            'avgLakeReservoirInflowShortIni': '0.0',
            'avgLakeReservoirOutflowLongIni': '0.0',
            'timestepsToAvgDischargeIni': '0.0',
        }

        # ── reportingOptions ──────────────────────────────────────────
        ini['reportingOptions'] = {
            'outDailyTotNC': 'discharge,totalRunoff',
            'outMonthTotNC': 'totalRunoff,baseflow,directRunoff,interflowTotal,'
                             'runoff,precipitation,gwRecharge,totalEvaporation',
            'outMonthAvgNC': 'discharge,temperature,snowCoverSWE,'
                             'storUppTotal,storLowTotal,storGroundwater,'
                             'channelStorage',
            'outMonthEndNC': 'storGroundwater,channelStorage',
            'outAnnuaTotNC': 'totalEvaporation,precipitation,gwRecharge,'
                             'totalRunoff,baseflow',
            'outAnnuaAvgNC': 'discharge,storGroundwater',
        }

        ini_path = self.settings_dir / config_file
        with open(ini_path, 'w') as f:
            f.write('# PCR-GLOBWB 2.0 configuration - generated by SYMFLUENCE\n')
            f.write(f'# Domain: {domain_name}, Resolution: {self.resolution}\n')
            f.write('# Natural mode (no irrigation/water demand)\n')
            f.write('# Initial conditions estimated from forcing climatology\n\n')
            ini.write(f)

        self.logger.info(f"Written INI config to {ini_path}")
