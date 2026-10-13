# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Model Preprocessor

Handles preparation of PRMS model inputs including:
- Control file (control.dat) with simulation settings
- Parameter file (params.dat) with HRU definitions and soil parameters
- Data file (data.dat) with forcing time series
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("PRMS")
class PRMSPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a PRMS model run.

    PRMS requires:
    - control.dat: Master control file with simulation settings
    - params.dat: Parameter file with HRU definitions
    - data.dat: Data file with forcing time series (precip, temp)
    """


    MODEL_NAME = "PRMS"
    def __init__(self, config, logger):
        """
        Initialize the PRMS preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Use standard SYMFLUENCE directory layout (inherited from base):
        #   self.setup_dir   -> {project_dir}/settings/PRMS
        #   self.forcing_dir -> {project_dir}/data/forcing/PRMS_input
        self.settings_dir = self.setup_dir

        # Resolve spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.prms.spatial_mode,
            default=None,
            dict_key='PRMS_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            self.spatial_mode = 'semi_distributed'
        logger.info(f"PRMS spatial mode: {self.spatial_mode}")

        # Use measured incoming shortwave (CARRA SWRadAtm) instead of the
        # ddsolrad degree-day estimate. At high latitude with persistent cloud
        # the empirical (US-tuned) estimate is biased, and measured radiation
        # drives both PET (Jensen-Haise) and snowmelt energy. When enabled, an
        # observed solar station is added (nsol=1, swrad in data.dat, HRU mapped
        # via hru_solsta) so PRMS uses observed radiation where available.
        self.use_obs_solar = bool(self._get_config_value(
            lambda: self.config.model.prms.use_obs_solar,
            default=False,
            dict_key='PRMS_USE_OBS_SOLAR'
        ))
        if self.use_obs_solar:
            logger.info("PRMS observed solar ON (measured SWRadAtm via hru_solsta)")

    def run_preprocessing(self) -> bool:
        """
        Run the complete PRMS preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting PRMS preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate data file (forcing time series)
            self._generate_data_file(start_date, end_date)

            # Generate parameter file
            self._generate_parameter_file()

            # Generate control file
            self._generate_control_file(start_date, end_date)

            logger.info("PRMS preprocessing completed successfully")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"PRMS preprocessing failed: {str(e)}", exc_info=True)
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create PRMS input directory structure."""
        for d in [self.settings_dir, self.forcing_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created PRMS directory structure: settings={self.settings_dir}, forcing={self.forcing_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from config."""
        start = self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01'
        )
        end = self._get_config_value(
            lambda: self.config.domain.time_end,
            default='2001-12-31'
        )
        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        if isinstance(end, str):
            end = pd.Timestamp(end).to_pydatetime()
        return start, end

    def _generate_data_file(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate the PRMS data file with forcing time series.

        The data file contains daily precipitation and temperature
        (min/max) for each station or HRU.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        logger.info("Generating PRMS data file...")

        data_file_name = self._get_config_value(
            lambda: self.config.model.prms.data_file,
            default='data.dat'
        )

        # Try to load forcing data
        forcing_data = self._load_forcing_data(start_date, end_date)

        out_path = self.forcing_dir / data_file_name

        if forcing_data is not None:
            self._write_data_file(out_path, forcing_data, start_date, end_date)
        else:
            logger.warning("No forcing data found, generating synthetic data file")
            self._generate_synthetic_data_file(out_path, start_date, end_date)

        # climate_hru still needs data.dat (observed-runoff placeholder), but
        # the meteorology PRMS uses comes from the per-HRU CBH files.
        if self.config.model.prms.forcing_mode == 'climate_hru':
            self._write_cbh_files(start_date, end_date)

    CBH_VARIABLES = ('precip', 'tmax', 'tmin', 'swrad')

    def _cbh_path(self, name: str) -> Path:
        return self.forcing_dir / f'{name}.cbh'

    def _load_hru_forcing(self, start_date: datetime, end_date: datetime) -> dict:
        """Daily per-HRU forcing for climate_hru, in PRMS HRU order.

        Reads canonical forcing with an ``hru`` dimension (e.g. EASYMORE
        remapping onto the HRU polygons) and maps each PRMS HRU to its source
        hruId, so split land/glacier HRUs of one band share that band's forcing.
        Returns DataFrames (days x nhru): precip mm/day, tmax/tmin degC,
        swrad Langley/day (daily-mean shortwave x 2.0651).
        """
        from symfluence.core.modeling.model_ready.forcing_reader import (
            forcing_timestep_seconds,
            open_canonical_forcing,
        )
        prms = self.config.model.prms
        if not prms.hru_forcing_path:
            raise ValueError('PRMS climate_hru forcing requires PRMS_HRU_FORCING_PATH')
        files = sorted(Path(prms.hru_forcing_path).glob('*.nc'))
        if not files:
            raise FileNotFoundError(f'No per-HRU forcing NetCDF in {prms.hru_forcing_path}')
        source_ids = prms.hru_forcing_map or list(range(1, prms.nhru + 1))
        if len(source_ids) != prms.nhru:
            raise ValueError(f'PRMS_HRU_FORCING_MAP has {len(source_ids)} entries for {prms.nhru} HRUs')

        names = {'precipitation_flux': 'precip', 'air_temperature': 'temp',
                 'surface_downwelling_shortwave_flux': 'swrad'}
        parts = {v: [] for v in names.values()}
        step = None
        # Monthly files opened one at a time: open_mfdataset on per-timestep
        # chunks is orders of magnitude slower for these small stores.
        for f in files:
            ds = open_canonical_forcing(f)
            try:
                missing = set(names) - set(ds.data_vars)
                if missing:
                    raise ValueError(f'{f.name} lacks canonical {sorted(missing)}')
                if 'hru' not in ds.dims:
                    raise ValueError(f'{f.name} has no hru dimension; climate_hru needs per-HRU forcing')
                ids = ds['hruId']
                ids = (ids.isel(time=0) if 'time' in ids.dims else ids).values.astype(int)
                step = step or forcing_timestep_seconds(ds)
                times = pd.DatetimeIndex(ds['time'].values)
                for var, short in names.items():
                    parts[short].append(pd.DataFrame(ds[var].transpose('time', 'hru').values,
                                                     index=times, columns=ids))
            finally:
                ds.close()
        raw = {}
        for short, frames in parts.items():
            df = pd.concat(frames).sort_index()
            df = df[~df.index.duplicated()].loc[str(start_date):str(end_date)]
            absent = set(source_ids) - set(df.columns)
            if absent:
                raise ValueError(f'Per-HRU forcing lacks source hruIds {sorted(absent)}')
            raw[short] = df[source_ids]
        temp = raw['temp'] - 273.15
        daily = {
            'precip': (raw['precip'] * step).resample('D').sum(),
            'tmax': temp.resample('D').max(),
            'tmin': temp.resample('D').min(),
            'swrad': raw['swrad'].clip(lower=0.0).resample('D').mean() * 2.0651,
        }
        for name, df in daily.items():
            if df.isna().any().any() or len(df) == 0:
                raise ValueError(f'Per-HRU {name} forcing has gaps between {start_date} and {end_date}')
        return daily

    def _write_cbh_files(self, start_date: datetime, end_date: datetime) -> None:
        """Write PRMS climate-by-HRU files (precip, tmax, tmin, swrad)."""
        daily = self._load_hru_forcing(start_date, end_date)
        for name in self.CBH_VARIABLES:
            df = daily[name]
            with open(self._cbh_path(name), 'w') as f:
                f.write(f'PRMS {name} CBH generated by SYMFLUENCE from per-HRU canonical forcing\n')
                f.write(f'{name} {df.shape[1]}\n')
                f.write('#' * 40 + '\n')
                for d, row in zip(df.index, df.to_numpy()):
                    f.write(f'{d.year} {d.month} {d.day} 0 0 0 ' + ' '.join(f'{v:.4f}' for v in row) + '\n')
        p = daily['precip'].sum() / max(len(daily['precip']) / 365.25, 1e-9)
        logger.info(f"Wrote PRMS CBH files ({len(daily['precip'])} days); mean annual precip per HRU "
                    f"(mm): {np.round(p.to_numpy(), 0).tolist()}")

    def _load_forcing_data(self, start_date: datetime, end_date: datetime):
        """
        Load basin-averaged ERA5 forcing data and convert to daily PRMS format.

        Uses the same basin-averaged forcing data as all other models in the
        ensemble (from self.forcing_basin_path inherited from BaseModelPreProcessor).

        Returns:
            pd.DataFrame with columns: precip (mm/day), tmax (°C), tmin (°C)
            indexed by date, or None if no forcing data found.
        """

        # Use inherited forcing_basin_path from BaseModelPreProcessor
        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            logger.warning(f"Forcing path does not exist: {forcing_path}")
            return None

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            logger.warning(f"No NetCDF files found in {forcing_path}")
            return None

        logger.info(f"Loading forcing from {forcing_path} ({len(forcing_files)} files)")

        # Canonical model-ready forcing: variables under the canonical vocabulary
        # (pptrate kg m-2 s-1, airtemp K) with the timestep on
        # ds.attrs['timestep_seconds'] -- no per-model alias lists or hardcoded step.
        from symfluence.core.modeling.model_ready.forcing_reader import (
            forcing_timestep_seconds,
            open_canonical_forcing,
        )
        ds = open_canonical_forcing(forcing_files, discretization=self.forcing_discretization)
        ds = ds.sel(time=slice(str(start_date), str(end_date)))

        airtemp = ds['air_temperature'].values.squeeze()      # K
        pptrate = ds['precipitation_flux'].values.squeeze()   # mm s-1 (== kg m-2 s-1)

        times = pd.DatetimeIndex(ds['time'].values)
        dt_seconds = forcing_timestep_seconds(ds)
        sub = pd.DataFrame({
            'airtemp_C': airtemp - 273.15,          # K → °C
            'precip_mm': pptrate * dt_seconds,      # mm s-1 → mm per timestep
        }, index=times)

        # Resample to daily
        daily = pd.DataFrame({
            'precip': sub['precip_mm'].resample('D').sum(),    # mm/day
            'tmax': sub['airtemp_C'].resample('D').max(),      # °C
            'tmin': sub['airtemp_C'].resample('D').min(),      # °C
        })

        # Optional measured incoming shortwave -> daily mean in Langleys/day
        # (PRMS swrad units). 1 W m-2 sustained over a day = 86400 J m-2 =
        # 86400/41840 = 2.0651 Langleys.
        if self.use_obs_solar:
            if 'surface_downwelling_shortwave_flux' in ds:
                sw = pd.Series(ds['surface_downwelling_shortwave_flux'].values.squeeze(),
                               index=times).clip(lower=0.0)
                daily['swrad'] = (sw.resample('D').mean() * 2.0651)
            else:
                logger.warning("PRMS_USE_OBS_SOLAR set but no canonical shortwave "
                               "(surface_downwelling_shortwave_flux) in forcing; "
                               "falling back to ddsolrad estimate")

        # Drop any days with all NaN
        daily = daily.dropna(how='all')

        ds.close()
        logger.info(f"Loaded canonical forcing: {len(daily)} days, "
                     f"precip range [{daily['precip'].min():.1f}, {daily['precip'].max():.1f}] mm/day, "
                     f"temp range [{daily['tmin'].min():.1f}, {daily['tmax'].max():.1f}] °C")
        return daily

    def _write_data_file(self, out_path: Path, forcing_data: pd.DataFrame,
                         start_date, end_date) -> None:
        """
        Write PRMS data file from loaded ERA5 forcing data.

        Args:
            out_path: Path to output data.dat file
            forcing_data: DataFrame with columns precip, tmax, tmin indexed by date
            start_date: Simulation start date
            end_date: Simulation end date
        """
        nhru = 1  # One forcing station; HRUs map to this station.

        has_swrad = 'swrad' in forcing_data.columns

        with open(out_path, 'w') as f:
            # Header: variable names and counts. Column order here defines the
            # order PRMS reads the data values; swrad (if present) precedes the
            # runoff placeholder.
            f.write("PRMS data file generated by SYMFLUENCE from canonical forcing\n")
            f.write(f"precip {nhru}\n")
            f.write(f"tmax {nhru}\n")
            f.write(f"tmin {nhru}\n")
            if has_swrad:
                # PRMS reads observed solar from a data variable named 'solrad'
                # (Langleys/day); 'swrad' is the computed OUTPUT name.
                f.write("solrad 1\n")
            f.write("runoff 1\n")
            f.write("####\n")

            # Write daily records: date + all values on one line (PRMS MMF format)
            for date, row in forcing_data.iterrows():
                d = pd.Timestamp(date)
                swrad_str = f" {row['swrad']:.2f}" if has_swrad else ""
                f.write(f"{d.year} {d.month} {d.day} 0 0 0"
                        f" {row['precip']:.4f}"
                        f" {row['tmax']:.2f}"
                        f" {row['tmin']:.2f}"
                        f"{swrad_str}"
                        f" 0.0\n")

        logger.info(f"Generated PRMS data file with canonical forcing: {out_path}")

    def _generate_synthetic_data_file(self, out_path: Path,
                                       start_date: datetime,
                                       end_date: datetime) -> None:
        """Raise error — synthetic forcing should never be used."""
        raise FileNotFoundError(
            f"No basin-averaged ERA5 forcing data found in {self.forcing_basin_path}. "
            "PRMS requires real forcing data. Ensure the domain has been set up "
            "with forcing data before running preprocessing."
        )

    def _get_catchment_properties(self) -> dict:
        """
        Derive HRU geometry from the per-domain model-ready datastore.

        Reads the catchment shapefile for centroid lat/lon and area, and the
        DEM (attributes/elevation/dem) for mean elevation, slope and aspect.
        Mirrors the projection-safe approach used by the CRHM preprocessor:
        a projected CRS is used for the centroid (then reprojected to degrees)
        and for area, otherwise a metres-CRS centroid yields an absurd UTM zone
        and the method silently falls back to defaults (wrong PET / snowmelt).

        Returns a dict with lat, lon, area_km2, elev, slope (deg), aspect (deg).
        Falls back to physically reasonable defaults if the datastore is absent.
        """
        props = {'lat': 65.0, 'lon': -19.0, 'area_km2': 100.0,
                 'elev': 500.0, 'slope': 5.0, 'aspect': 180.0}
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path and catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                if gdf.crs.is_geographic:
                    cpt = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326).iloc[0]
                else:
                    cpt = gdf.geometry.centroid.to_crs(epsg=4326).iloc[0]
                lon, lat = cpt.x, cpt.y
                utm_zone = int((lon + 180) / 6) + 1
                utm_crs = f"EPSG:{(32600 if lat >= 0 else 32700) + utm_zone}"
                area_m2 = gdf.to_crs(utm_crs).geometry.area.sum()
                props.update({'lat': lat, 'lon': lon,
                              'area_km2': area_m2 / 1e6})
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read catchment shapefile: {e}; using defaults", exc_info=True)

        # DEM-derived elevation / slope / aspect (mean over the catchment).
        try:
            import rasterio  # local import: only needed here
            dem_dir = self.project_attributes_dir / 'elevation' / 'dem'
            dems = sorted(dem_dir.glob('*_elv.tif')) if dem_dir.exists() else []
            if dems:
                with rasterio.open(dems[0]) as src:
                    z = src.read(1).astype('float64')
                    nd = src.nodata
                    xres, yres = src.res
                    geo = src.crs is not None and src.crs.is_geographic
                    bnds = src.bounds
                m = np.isfinite(z) & (z > -100.0)
                if nd is not None:
                    m &= z != nd
                if m.any():
                    props['elev'] = float(z[m].mean())
                    if geo:
                        lat0 = np.radians((bnds.bottom + bnds.top) / 2.0)
                        xm = abs(xres) * 111320.0 * max(np.cos(lat0), 0.1)
                        ym = abs(yres) * 111320.0
                    else:
                        xm, ym = abs(xres), abs(yres)
                    zz = np.where(m, z, np.nan)
                    dy, dx = np.gradient(zz, ym, xm)
                    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
                    asp = (np.degrees(np.arctan2(-dy, dx)) + 360.0) % 360.0
                    props['slope'] = float(np.nanmean(slope[m]))
                    a = np.radians(asp[m])
                    props['aspect'] = float(
                        (np.degrees(np.arctan2(np.nanmean(np.sin(a)),
                                               np.nanmean(np.cos(a)))) + 360.0) % 360.0)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read DEM attributes: {e}; using defaults", exc_info=True)

        return props

    def _get_dominant_classes(self) -> dict:
        """
        Derive PRMS cov_type and soil_type from the land/soil-class rasters.

        cov_type: 0=bare, 1=grasses, 2=shrubs, 3=trees (mapped from the dominant
        MODIS IGBP land-cover class). soil_type: 1=sand, 2=loam, 3=clay (mapped
        from the dominant soil class; defaults to loam when unavailable).
        """
        out = {'cov_type': 1, 'soil_type': 2}  # grass / loam defaults (Iceland)
        try:
            import rasterio
            lc = self.project_attributes_dir / 'landclass'
            tifs = sorted(lc.glob('domain_*_land_classes.tif')) if lc.exists() else []
            if tifs:
                with rasterio.open(tifs[0]) as src:
                    a = src.read(1); nd = src.nodata
                v = a[a != nd] if nd is not None else a.ravel()
                v = v[v > 0]
                v = v[v != 17]  # drop open water
                if v.size:
                    dom = int(np.bincount(v).argmax())
                    # IGBP -> PRMS cov_type
                    if dom in (1, 2, 3, 4, 5):       # forests
                        out['cov_type'] = 3
                    elif dom in (6, 7, 8, 9):        # shrub/savanna
                        out['cov_type'] = 2
                    elif dom in (10, 12, 14):        # grass/crop
                        out['cov_type'] = 1
                    else:                            # 11 wetland,13 urban,15 ice,16 barren
                        out['cov_type'] = 0
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not read land class: {e}; cov_type default", exc_info=True)
        return out

    SNAREA_CURVE = [0.05, 0.24, 0.40, 0.53, 0.65, 0.73, 0.80, 0.87, 0.93, 0.97, 1.00]

    def _depletion_curve_assignment(self) -> list:
        """Depletion-curve index per HRU: 1 for all, or 1 land / 2 glacier."""
        prms = self.config.model.prms
        if not prms.separate_glacier_depletion_curve:
            return [1] * prms.nhru
        fractions = prms.parameter_overrides.get('glacier_frac_init', {}).get('values')
        if not prms.glacier_flag or not isinstance(fractions, list) or len(fractions) != prms.nhru \
                or any(v not in (0, 1) for v in fractions):
            raise ValueError('Separate glacier depletion curves need glacier_flag and pure land/glacier '
                             'HRUs (glacier_frac_init override of 0/1 per HRU)')
        if all(fractions) or not any(fractions):
            raise ValueError('Separate glacier depletion curves need both land and glacier HRUs')
        return [2 if v == 1 else 1 for v in fractions]

    def _generate_parameter_file(self) -> None:
        """
        Generate the PRMS parameter file.

        Contains all required dimensions and parameters for PRMS 5.x
        lumped-mode operation including HRU, soil, snow, runoff,
        groundwater, and PET (Jensen-Haise) parameters. HRU geometry and
        classification are derived from the per-domain model-ready datastore
        (catchment shapefile, DEM, land/soil-class rasters).
        """
        logger.info("Generating PRMS parameter file...")

        param_file_name = self._get_config_value(
            lambda: self.config.model.prms.parameter_file,
            default='params.dat'
        )

        out_path = self.settings_dir / param_file_name

        prms = self.config.model.prms
        nhru = prms.nhru
        nsegment = prms.routing_segments
        nmonths = 12
        deplcrv = self._depletion_curve_assignment()
        ndepl = max(deplcrv)

        # HRU geometry & classification from the model-ready datastore.
        props = self._get_catchment_properties()
        classes = self._get_dominant_classes()
        lat = props['lat']
        lon = props['lon']
        elev = props['elev']  # mean basin elevation (m)
        logger.info(
            f"PRMS HRU from datastore: lat={lat:.3f} lon={lon:.3f} "
            f"elev={elev:.0f} m area={props['area_km2']:.1f} km2 "
            f"slope={props['slope']:.1f} deg aspect={props['aspect']:.0f} "
            f"cov_type={classes['cov_type']} soil_type={classes['soil_type']}")

        with open(out_path, 'w') as f:
            f.write("PRMS parameter file generated by SYMFLUENCE\n")
            f.write("Version: 1.0\n")

            # ---- Dimensions ----
            f.write("** Dimensions **\n")
            # nsol = number of solar-radiation stations: 1 when feeding observed
            # shortwave, else 0 (ddsolrad estimates radiation internally).
            nsol = 1 if self.use_obs_solar else 0
            dims = {
                'nhru': nhru, 'nsegment': nsegment, 'nmonths': nmonths,
                'nobs': 1, 'ngw': nhru, 'nssr': nhru, 'nsub': 1,
                'ndepl': ndepl, 'ndeplval': 11 * ndepl, 'one': 1, 'nrain': 1,
                'ntemp': 1, 'nsol': nsol,
            }
            for name, size in dims.items():
                f.write("####\n")
                f.write(f"{name}\n")
                f.write(f"{size}\n")

            # ---- Parameters ----
            f.write("** Parameters **\n")

            def write_param(name, ndim, dim_name, dim_size, dtype, values):
                """Write a single PRMS parameter block.
                dtype: 1=int, 2=float, 4=string
                """
                f.write("####\n")
                f.write(f"{name}\n")
                f.write(f"{ndim}\n")
                f.write(f"{dim_name}\n")
                f.write(f"{dim_size}\n")
                f.write(f"{dtype}\n")
                if not isinstance(values, list):
                    values = [values] * dim_size
                for v in values:
                    f.write(f"{v}\n")

            # --- Unit system ---
            write_param("precip_units", 1, "one", 1, 1, 1)   # 1=mm (default 0=inches)
            write_param("temp_units", 1, "one", 1, 1, 1)      # 1=Celsius (default 0=Fahrenheit)
            write_param("elev_units", 1, "one", 1, 1, 1)      # 1=meters (default 0=feet)

            # --- Observed solar radiation station mapping ---
            # When feeding measured shortwave, map the HRU to solar station 1 so
            # ddsolrad uses the observed swrad (Langleys/day) from data.dat
            # instead of its degree-day estimate. rad_conv=1.0: data already in ly/day.
            if self.use_obs_solar:
                write_param("basin_solsta", 1, "one", 1, 1, 1)
                write_param("hru_solsta", 1, "nhru", nhru, 1, 1)
                write_param("rad_conv", 1, "one", 1, 2, 1.0)

            # --- HRU geometry & classification (from datastore) ---
            # PRMS requires hru_area in acres (1 km² = 247.105 acres) and
            # hru_slope as a decimal grade (rise/run = tan(slope)).
            area_acres = props['area_km2'] * 247.105
            hru_slope_grade = float(np.tan(np.radians(props['slope'])))
            write_param("hru_area", 1, "nhru", nhru, 2, f"{area_acres:.1f}")
            write_param("hru_elev", 1, "nhru", nhru, 2, f"{elev:.1f}")
            write_param("hru_slope", 1, "nhru", nhru, 2, f"{hru_slope_grade:.4f}")
            write_param("hru_lat", 1, "nhru", nhru, 2, f"{lat:.4f}")
            write_param("hru_lon", 1, "nhru", nhru, 2, f"{lon:.4f}")
            write_param("hru_aspect", 1, "nhru", nhru, 2, f"{props['aspect']:.1f}")
            write_param("hru_type", 1, "nhru", nhru, 1, 1)       # 1=land
            write_param("cov_type", 1, "nhru", nhru, 1, classes['cov_type'])
            write_param("soil_type", 1, "nhru", nhru, 1, classes['soil_type'])
            write_param("hru_percent_imperv", 1, "nhru", nhru, 2, 0.0)

            # --- Connectivity ---
            write_param("hru_segment", 1, "nhru", nhru, 1, 1)
            write_param("hru_subbasin", 1, "nhru", nhru, 1, 1)
            write_param("hru_deplcrv", 1, "nhru", nhru, 1, deplcrv)

            # --- Canopy interception ---
            write_param("covden_sum", 1, "nhru", nhru, 2, 0.5)
            write_param("covden_win", 1, "nhru", nhru, 2, 0.3)
            write_param("snow_intcp", 1, "nhru", nhru, 2, 0.05)
            write_param("srain_intcp", 1, "nhru", nhru, 2, 0.05)
            write_param("wrain_intcp", 1, "nhru", nhru, 2, 0.05)

            # --- Snow ---
            write_param("rad_trncf", 1, "nhru", nhru, 2, 0.5)
            write_param("potet_sublim", 1, "nhru", nhru, 2, 0.5)
            write_param("emis_noppt", 1, "nhru", nhru, 2, 0.757)
            write_param("freeh2o_cap", 1, "nhru", nhru, 2, 0.05)
            write_param("cecn_coef", 1, "nmonths", nmonths, 2,
                         [5.0] * nmonths)

            # Snow depletion curves (11 points each). With separate glacier
            # curves, curve 1 (land) is rescaled by land_sca_max during
            # calibration and curve 2 (glacier) stays the unscaled reference.
            write_param("snarea_curve", 1, "ndeplval", 11 * ndepl, 2,
                         self.SNAREA_CURVE * ndepl)
            write_param("snarea_thresh", 1, "nhru", nhru, 2, 50.0)

            # --- Transpiration ---
            write_param("transp_beg", 1, "nhru", nhru, 1, 4)    # April
            write_param("transp_end", 1, "nhru", nhru, 1, 10)   # October
            write_param("transp_tmax", 1, "nhru", nhru, 2, 1.0)

            # --- PET (Jensen-Haise) ---
            write_param("jh_coef", 1, "nmonths", nmonths, 2,
                         [0.014] * nmonths)
            write_param("jh_coef_hru", 1, "nhru", nhru, 2, 13.0)

            # --- Solar radiation (ddsolrad) ---
            write_param("dday_slope", 1, "nmonths", nmonths, 2,
                         [0.30, 0.32, 0.34, 0.36, 0.38, 0.38,
                          0.36, 0.35, 0.33, 0.31, 0.30, 0.29])
            write_param("dday_intcp", 1, "nmonths", nmonths, 2,
                         [-20.0, -18.0, -14.0, -10.0, -6.0, -3.0,
                          -2.0, -3.0, -6.0, -12.0, -16.0, -20.0])

            # --- Surface runoff ---
            write_param("carea_max", 1, "nhru", nhru, 2, 0.6)
            write_param("smidx_coef", 1, "nhru", nhru, 2, 0.01)
            write_param("smidx_exp", 1, "nhru", nhru, 2, 0.3)
            write_param("imperv_stor_max", 1, "nhru", nhru, 2, 0.05)

            # --- Soil zone ---
            write_param("soil_moist_max", 1, "nhru", nhru, 2, 6.0)
            if self.config.model.prms.model_mode.upper() == 'PRMS5':
                # Preserve the default depths (2/6, 3/6 and 1/2) using
                # the fractional parameters actually read by PRMS5.
                write_param("soil_rechr_max_frac", 1, "nhru", nhru, 2, 1.0 / 3.0)
                write_param("soil_moist_init_frac", 1, "nhru", nhru, 2, 0.5)
                write_param("soil_rechr_init_frac", 1, "nhru", nhru, 2, 0.5)
            else:
                write_param("soil_rechr_max", 1, "nhru", nhru, 2, 2.0)
                write_param("soil_moist_init", 1, "nhru", nhru, 2, 3.0)
                write_param("soil_rechr_init", 1, "nhru", nhru, 2, 1.0)
            write_param("soil2gw_max", 1, "nhru", nhru, 2, 0.5)
            write_param("ssr2gw_rate", 1, "nssr", nhru, 2, 0.1)
            write_param("ssr2gw_exp", 1, "nssr", nhru, 2, 1.0)
            write_param("ssrcoef_lin", 1, "nssr", nhru, 2, 0.1)
            write_param("ssrcoef_sq", 1, "nssr", nhru, 2, 0.1)
            write_param("slowcoef_lin", 1, "nhru", nhru, 2, 0.015)
            write_param("slowcoef_sq", 1, "nhru", nhru, 2, 0.1)
            write_param("pref_flow_den", 1, "nhru", nhru, 2, 0.0)
            write_param("fastcoef_lin", 1, "nhru", nhru, 2, 0.1)
            write_param("fastcoef_sq", 1, "nhru", nhru, 2, 0.1)

            # --- Groundwater ---
            write_param("gwflow_coef", 1, "ngw", nhru, 2, 0.015)
            write_param("gwstor_init", 1, "ngw", nhru, 2, 2.0)
            write_param("gwstor_min", 1, "ngw", nhru, 2, 0.0)
            write_param("gwsink_coef", 1, "ngw", nhru, 2, 0.0)

            # --- Temperature/precipitation adjustments ---
            write_param("tmax_allrain", 1, "nmonths", nmonths, 2,
                         [3.3] * nmonths)
            write_param("tmax_allsnow", 1, "nmonths", nmonths, 2,
                         [0.0] * nmonths)
            write_param("adjmix_rain", 1, "nmonths", nmonths, 2,
                         [1.0] * nmonths)
            write_param("tmax_adj", 1, "nhru", nhru, 2, 0.0)
            write_param("tmin_adj", 1, "nhru", nhru, 2, 0.0)
            write_param("rain_adj", 1, "nmonths", nmonths, 2,
                         [1.0] * nmonths)
            write_param("snow_adj", 1, "nmonths", nmonths, 2,
                         [1.0] * nmonths)

            # --- Station parameters ---
            write_param("tsta_elev", 1, "ntemp", 1, 2, elev)
            write_param("psta_elev", 1, "nrain", 1, 2, elev)
            write_param("hru_tsta", 1, "nhru", nhru, 1, 1)
            write_param("hru_psta", 1, "nhru", nhru, 1, 1)
            write_param("basin_tsta", 1, "one", 1, 1, 1)
            # nsol=0: ddsolrad computes radiation via degree-day, no observed solar data needed

            # --- Segment routing ---
            # Reaches in series: all HRUs drain to reach 1, reach i flows to
            # i+1 and the last reach is the outlet. Native K_coef <= 24 h each.
            write_param("tosegment", 1, "nsegment", nsegment, 1,
                        list(range(2, nsegment + 1)) + [0])
            write_param("seg_length", 1, "nsegment", nsegment, 2, 10000.0)
            write_param("K_coef", 1, "nsegment", nsegment, 2, 0.1 if nsegment == 1 else 12.0)
            write_param("x_coef", 1, "nsegment", nsegment, 2, 0.2)
            write_param("obsin_segment", 1, "nsegment", nsegment, 1, 0)

            if prms.forcing_mode == 'climate_hru':
                # climate_hru multipliers/offsets (nhru x nmonths, HRU fastest).
                # rain_adj/snow_adj/tmax_adj above are station-mode only.
                for name, value in (('rain_cbh_adj', 1.0), ('snow_cbh_adj', 1.0),
                                    ('tmax_cbh_adj', 0.0), ('tmin_cbh_adj', 0.0)):
                    write_param(name, 2, "nhru\nnmonths", nhru * nmonths, 2, value)

        # Explicit spatial/glacier parameters override generated defaults. PRMS
        # dimension order is Fortran order (HRU varies fastest for monthly data).
        overrides = self.config.model.prms.parameter_overrides
        if self.config.model.prms.model_mode.upper() == 'PRMS5':
            legacy = {'soil_rechr_max', 'soil_moist_init', 'soil_rechr_init'} & overrides.keys()
            if legacy:
                raise ValueError(
                    f"PRMS5 ignores depth parameters {sorted(legacy)}; use the corresponding "
                    "*_frac parameters (fractions of their respective capacities)."
                )
        if overrides:
            content = out_path.read_text()
            prefix, body = content.split("** Parameters **\n", 1)
            blocks = {b.strip().splitlines()[0]: b for b in body.split("####\n") if b.strip()}
            for name, spec in overrides.items():
                dimensions = spec['dimensions']
                count = int(np.prod([dims[d] for d in dimensions]))
                values = spec['values']
                if not isinstance(values, list):
                    values = [values] * count
                if len(values) != count:
                    raise ValueError(f"PRMS {name}: expected {count} values, got {len(values)}")
                blocks[name] = '\n'.join(map(str, [name, len(dimensions), *dimensions,
                                                       count, spec.get('dtype', 2), *values])) + '\n'
            out_path.write_text(prefix + "** Parameters **\n" + ''.join('####\n' + b for b in blocks.values()))
        logger.info(f"Generated PRMS parameter file: {out_path}")

    def _generate_control_file(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate the PRMS control file.

        The control file specifies simulation dates, module selection,
        file paths, and output options.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        logger.info("Generating PRMS control file...")

        control_file_name = self._get_config_value(
            lambda: self.config.model.prms.control_file,
            default='control.dat'
        )
        param_file_name = self._get_config_value(
            lambda: self.config.model.prms.parameter_file,
            default='params.dat'
        )
        data_file_name = self._get_config_value(
            lambda: self.config.model.prms.data_file,
            default='data.dat'
        )

        model_mode = self._get_config_value(
            lambda: self.config.model.prms.model_mode,
            default='DAILY'
        )

        out_path = self.settings_dir / control_file_name

        with open(out_path, 'w') as f:
            f.write("PRMS control file generated by SYMFLUENCE\n")

            def write_ctrl(name, nvals, dtype, values):
                """Write a control parameter block.
                dtype: 1=int, 2=float, 4=string
                """
                f.write("####\n")
                f.write(f"{name}\n")
                f.write(f"{nvals}\n")
                f.write(f"{dtype}\n")
                if isinstance(values, list):
                    for v in values:
                        f.write(f"{v}\n")
                else:
                    f.write(f"{values}\n")

            # Simulation dates (6 integer values: year, month, day, hour, min, sec)
            write_ctrl("start_time", 6, 1,
                        [start_date.year, start_date.month, start_date.day,
                         0, 0, 0])
            write_ctrl("end_time", 6, 1,
                        [end_date.year, end_date.month, end_date.day,
                         0, 0, 0])

            # Model mode
            write_ctrl("model_mode", 1, 4, model_mode)

            # File paths
            write_ctrl("param_file", 1, 4, str(self.settings_dir / param_file_name))
            write_ctrl("data_file", 1, 4, str(self.forcing_dir / data_file_name))

            # Module selections (required by PRMS 5)
            if self.config.model.prms.forcing_mode == 'climate_hru':
                for module in ("precip_module", "temp_module", "solrad_module"):
                    write_ctrl(module, 1, 4, "climate_hru")
                for name in self.CBH_VARIABLES:
                    write_ctrl(f"{name}_day", 1, 4, str(self._cbh_path(name)))
                write_ctrl("cbh_check_flag", 1, 1, 1)
            else:
                write_ctrl("precip_module", 1, 4, "precip_1sta")
                write_ctrl("temp_module", 1, 4, "temp_1sta")
                write_ctrl("solrad_module", 1, 4, "ddsolrad")
            write_ctrl("et_module", 1, 4, "potet_jh")
            write_ctrl("srunoff_module", 1, 4, "srunoff_smidx")
            routing = self.config.model.prms.streamflow_module
            if routing == 'auto':
                routing = 'muskingum' if self.config.model.prms.glacier_flag else 'strmflow'
            if self.config.model.prms.glacier_flag and routing == 'strmflow':
                raise ValueError("PRMS strmflow omits glacier discharge; use muskingum with glacier_flag.")
            if self.config.model.prms.routing_segments > 1 and routing != 'muskingum':
                raise ValueError("PRMS_ROUTING_SEGMENTS > 1 requires the muskingum streamflow module.")
            write_ctrl("strmflow_module", 1, 4, routing)
            write_ctrl("transp_module", 1, 4, "transp_tindex")
            write_ctrl("soilzone_module", 1, 4, "soilzone")

            # Disable subbasin and print_debug to avoid variable conflicts
            write_ctrl("glacier_flag", 1, 1, int(self.config.model.prms.glacier_flag))
            if self.config.model.prms.glacier_flag and not self.config.model.prms.glacier_dynamics:
                binary = Path(self.config.model.prms.install_path)
                if binary.is_dir():
                    binary = binary / self.config.model.prms.exe
                if not binary.is_file() or b'glacier_dynamics' not in binary.read_bytes():
                    raise ValueError('Fixed glacier geometry requires a native PRMS build supporting glacier_dynamics')
                fractions = self.config.model.prms.parameter_overrides.get('glacier_frac_init', {}).get('values', [])
                if not isinstance(fractions, list) or len(fractions) != self.config.model.prms.nhru or any(v not in (0, 1) for v in fractions):
                    raise ValueError('Fixed glacier geometry requires separate pure glacier and land HRUs (fractions 0 or 1)')
                write_ctrl('glacier_dynamics', 1, 1, 0)
            write_ctrl("subbasin_flag", 1, 1, 0)
            write_ctrl("print_debug", 1, 1, -1)

            # Output — use relative filenames so PRMS writes to run cwd
            write_ctrl("statsON_OFF", 1, 1, 1)
            variables = ["basin_cfs", "basin_actet", "basin_soil_moist", "basin_pweqv"]
            if self.config.model.prms.glacier_flag:
                variables += ["basin_gl_ice_melt", "basin_gl_area", "basin_gl_storage"]
            # Save every term of the native basin_sum water balance so a
            # successful exit can be checked independently of streamflow fit.
            variables += ["basin_ppt", "basin_storage", "basin_stflow_out",
                          "basin_gwsink", "basin_gwstor_minarea_wb", "basin_snowmelt"]
            if self.config.model.prms.glacier_flag:
                variables += ["basin_glacrevap", "basin_glacrb_melt",
                              "basin_gl_top_gain", "basin_gl_top_melt"]
            elements = [1] * len(variables)
            if self.config.model.prms.glacier_flag:
                # basin_storage includes glacier storage change, while
                # basin_pweqv also includes snow on ice. Retain HRU states
                # so diagnostics can identify overlapping storage terms.
                for name in ('pkwater_equiv', 'glacier_frac'):
                    variables.extend([name] * self.config.model.prms.nhru)
                    elements.extend(range(1, self.config.model.prms.nhru + 1))
            if self.config.model.prms.cryosphere_constraints:
                if not self.config.model.prms.glacier_flag or self.config.model.prms.glacier_dynamics:
                    raise ValueError('PRMS cryosphere constraints require fixed glacier geometry')
                variables.extend(['snowcov_area'] * self.config.model.prms.nhru)
                elements.extend(range(1, self.config.model.prms.nhru + 1))
            write_ctrl("nstatVars", 1, 1, len(variables))
            write_ctrl("stat_var_file", 1, 4, "statvar.dat")
            write_ctrl("csv_output_file", 1, 4, "prms_output.csv")

            # Output variables
            write_ctrl("statVar_names", len(variables), 4, variables)
            write_ctrl("statVar_element", len(variables), 4, elements)
            # Note: Do NOT add a trailing #### — PRMS will try to read
            # a key after the delimiter and hit early EOF

        logger.info(f"Generated PRMS control file: {out_path}")
