# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Model Preprocessor."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
from xml.etree import ElementTree as ET  # nosec B405

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.modeling.model_ready.forcing_reader import (
    forcing_timestep_seconds,
    open_canonical_forcing,
)
from symfluence.core.registries import R


@R.preprocessors.add("LISFLOOD")
class LisfloodPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Prepares inputs for a LISFLOOD model run."""

    MODEL_NAME = "LISFLOOD"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.settings_dir = self.setup_dir
        self.maps_dir = self.setup_dir / "maps"
        self.forcing_out_dir = self.setup_dir / "meteo"
        self.tables_dir = self.setup_dir / "tables"
        self.inits_dir = self.setup_dir / "init"

        configured_mode = self._get_config_value(
            lambda: self.config.model.lisflood.spatial_mode, default=None, dict_key="LISFLOOD_SPATIAL_MODE"
        )
        if configured_mode and configured_mode not in (None, "auto", "default"):
            self.spatial_mode = configured_mode
        else:
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method, default="lumped", dict_key="DOMAIN_DEFINITION_METHOD"
            )
            self.spatial_mode = "distributed" if domain_method == "delineate" else "lumped"
        self.logger.info(f"LISFLOOD spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        try:
            self.logger.info("Starting LISFLOOD preprocessing...")
            self._create_directory_structure()
            self._generate_static_maps()
            self._generate_forcing()
            self._generate_settings_xml()
            self.logger.info("LISFLOOD preprocessing complete.")
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"LISFLOOD preprocessing failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        for d in [self.settings_dir, self.maps_dir, self.forcing_out_dir, self.tables_dir, self.inits_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_catchment_properties(self) -> Dict:
        # Raises if the catchment shapefile is missing; never fall back to a
        # default location (that silently builds the model somewhere else).
        props = self.get_catchment_location()
        if props['elev'] is None:
            self.logger.warning("Catchment shapefile has no elev_mean column; using 1500 m for LISFLOOD elevation maps")
            props['elev'] = 1500.0
        return props

    # ------------------------------------------------------------------
    # Static map helpers
    # ------------------------------------------------------------------
    def _write_map(self, name: str, value: float, lat: np.ndarray, lon: np.ndarray, units: str = "") -> None:
        """Write a 3x3 NetCDF map with uniform values (y descending)."""
        arr = np.full((3, 3), value, dtype=np.float64)
        ds = xr.Dataset(
            {
                name: xr.DataArray(
                    arr,
                    dims=["y", "x"],
                    coords={"y": lat[::-1], "x": lon},
                    attrs={"units": units},
                ),
            }
        )
        ds.to_netcdf(self.maps_dir / f"{name}.nc")

    def _generate_static_maps(self) -> None:
        """Generate ALL LISFLOOD static maps as single-cell NetCDF files.

        For lumped mode every "map" is a 1x1 grid. The maps are written to
        ``self.maps_dir`` and cover topography, channels, soil hydraulics
        (Other + Forest variants), land-use fractions, LAI, snow/elevation
        standard deviation, gauges, lake mask, and a netCDF template.
        """
        self.logger.info("Generating LISFLOOD static maps...")
        props = self._get_catchment_properties()

        dx = 0.01
        lat = np.array([props["lat"] - dx, props["lat"], props["lat"] + dx])
        lon = np.array([props["lon"] - dx, props["lon"], props["lon"] + dx])

        wm = self._write_map  # shorthand

        # Read configurable forest fraction (default 0.3)
        forest_frac = float(
            self._get_config_value(
                lambda: self.config.model.lisflood.forest_fraction,
                default=0.3,
                dict_key="LISFLOOD_FOREST_FRACTION",
            )
        )

        # ----- Topography / base maps -----
        wm("dem", props["elev"], lat, lon, "m")
        # LDD: proper drain directions (all cells drain to centre pit)
        # PCRaster codes: 7=NW 8=N 9=NE / 4=W 5=pit 6=E / 1=SW 2=S 3=SE
        # y is descending so row 0=north, row 2=south
        ldd_arr = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0], [9.0, 8.0, 7.0]], dtype=np.float64)
        ds_ldd = xr.Dataset(
            {
                "ldd": xr.DataArray(ldd_arr, dims=["y", "x"], coords={"y": lat[::-1], "x": lon}),
            }
        )
        ds_ldd.to_netcdf(self.maps_dir / "ldd.nc")
        # Outlet/Gauges: only centre cell
        for map_name in ["outlet", "Gauges"]:
            out_arr = np.zeros((3, 3), dtype=np.float64)
            out_arr[1, 1] = 1.0
            ds_out = xr.Dataset(
                {
                    map_name: xr.DataArray(out_arr, dims=["y", "x"], coords={"y": lat[::-1], "x": lon}),
                }
            )
            ds_out.to_netcdf(self.maps_dir / f"{map_name}.nc")
        wm("catchment", 1.0, lat, lon)
        wm("cellarea", props["area_m2"], lat, lon, "m2")
        wm("gradient", 0.05, lat, lon, "m/m")
        wm("Grad", 0.05, lat, lon, "m/m")
        wm("elvstd", 100.0, lat, lon, "m")
        wm("ElevationStD", 100.0, lat, lon, "m")
        wm("mannings", 0.07, lat, lon, "s/m^(1/3)")
        wm("LakeMask", 0.0, lat, lon)

        # netCDF template (copy of elvstd)
        wm("netCDFtemplate", 100.0, lat, lon, "m")

        # Latitude map (for snow/ice modelling)
        wm("lat", props["lat"], lat, lon, "degrees_north")

        # ----- Channel maps -----
        wm("changrad", 0.01, lat, lon, "m/m")
        wm("chanman", 0.04, lat, lon, "s/m^(1/3)")
        wm("chanlength", 1000.0, lat, lon, "m")
        wm("chanbw", 10.0, lat, lon, "m")
        wm("chandepth", 2.0, lat, lon, "m")
        # Channels: only centre cell is a channel
        chan_arr = np.zeros((3, 3), dtype=np.float64)
        chan_arr[1, 1] = 1.0
        ds_chan = xr.Dataset(
            {
                "Channels": xr.DataArray(chan_arr, dims=["y", "x"], coords={"y": lat[::-1], "x": lon}),
            }
        )
        ds_chan.to_netcdf(self.maps_dir / "Channels.nc")
        wm("ChanSdXdY", 0.1, lat, lon)  # channel side slope dx/dy
        wm("chans", 0.1, lat, lon)  # alias used in some configs
        wm("chan", 1.0, lat, lon)  # boolean channel map
        wm("pixleng", 1000.0, lat, lon, "m")
        wm("pixarea", props["area_m2"], lat, lon, "m2")
        wm("lzavin", 0.5, lat, lon, "mm/day")
        wm("avgdis", 1.0, lat, lon, "m3/s")

        # ----- Soil depth maps (Other) -----
        wm("soil_depth1", 50.0, lat, lon, "mm")
        wm("soil_depth2", 250.0, lat, lon, "mm")
        wm("soil_depth3", 700.0, lat, lon, "mm")

        # Soil depth maps (Forest) - same depths as Other
        wm("SoilDepth1Forest", 50.0, lat, lon, "mm")
        wm("SoilDepth2Forest", 250.0, lat, lon, "mm")
        wm("SoilDepth3Forest", 700.0, lat, lon, "mm")

        # ----- Soil hydraulic parameters (Other land use) -----
        # Saturated conductivity [mm/day]
        wm("MapKSat1", 25.0, lat, lon, "mm/day")
        wm("MapKSat2", 15.0, lat, lon, "mm/day")
        wm("MapKSat3", 5.0, lat, lon, "mm/day")
        # Pore-size index Lambda [-]
        wm("MapLambda1", 0.25, lat, lon)
        wm("MapLambda2", 0.22, lat, lon)
        wm("MapLambda3", 0.18, lat, lon)
        # Van Genuchten Alpha [1/cm]
        wm("MapGenuAlpha1", 0.037, lat, lon, "1/cm")
        wm("MapGenuAlpha2", 0.035, lat, lon, "1/cm")
        wm("MapGenuAlpha3", 0.030, lat, lon, "1/cm")
        # Saturated soil moisture [V/V]
        wm("MapThetaSat1", 0.45, lat, lon)
        wm("MapThetaSat2", 0.43, lat, lon)
        wm("MapThetaSat3", 0.40, lat, lon)
        # Residual soil moisture [V/V]
        wm("MapThetaRes1", 0.05, lat, lon)
        wm("MapThetaRes2", 0.05, lat, lon)
        wm("MapThetaRes3", 0.05, lat, lon)

        # ----- Soil hydraulic parameters (Forest) -----
        # Use same values as "Other" for layers 1 & 2; layer 3 shared
        wm("MapKSat1Forest", 25.0, lat, lon, "mm/day")
        wm("MapKSat2Forest", 15.0, lat, lon, "mm/day")
        wm("MapLambda1Forest", 0.25, lat, lon)
        wm("MapLambda2Forest", 0.22, lat, lon)
        wm("MapGenuAlpha1Forest", 0.037, lat, lon, "1/cm")
        wm("MapGenuAlpha2Forest", 0.035, lat, lon, "1/cm")
        wm("MapThetaSat1Forest", 0.45, lat, lon)
        wm("MapThetaSat2Forest", 0.43, lat, lon)
        wm("MapThetaRes1Forest", 0.05, lat, lon)
        wm("MapThetaRes2Forest", 0.05, lat, lon)

        # ----- Crop / vegetation parameters (Other, Forest, Irrigation) -----
        wm("MapCropCoef", 1.0, lat, lon)
        wm("MapForestCropCoef", 0.8, lat, lon)
        wm("MapIrrigationCropCoef", 1.1, lat, lon)
        wm("MapCropGroupNumber", 3.0, lat, lon)
        wm("MapForestCropGroupNumber", 5.0, lat, lon)
        wm("MapIrrigationCropGroupNumber", 2.0, lat, lon)
        wm("MapN", 0.07, lat, lon, "s/m^(1/3)")
        wm("MapForestN", 0.10, lat, lon, "s/m^(1/3)")

        # Also keep backward-compatible names used by old lfbinding
        wm("cropcoef", 1.0, lat, lon)
        wm("thetas1", 0.45, lat, lon)
        wm("thetas2", 0.43, lat, lon)
        wm("thetas3", 0.40, lat, lon)
        wm("thetar1", 0.05, lat, lon)
        wm("thetar2", 0.05, lat, lon)
        wm("thetar3", 0.05, lat, lon)
        wm("ksat1", 25.0, lat, lon, "mm/day")
        wm("ksat2", 15.0, lat, lon, "mm/day")
        wm("ksat3", 5.0, lat, lon, "mm/day")
        wm("lambda1", 0.25, lat, lon)
        wm("lambda2", 0.22, lat, lon)
        wm("lambda3", 0.18, lat, lon)
        wm("genuinvalf1", 0.037, lat, lon, "1/cm")
        wm("genuinvalf2", 0.035, lat, lon, "1/cm")
        wm("genuinvalf3", 0.030, lat, lon, "1/cm")

        # ----- Land-use fractions -----
        # Fractions must sum to ~1.  For a simple lumped setup:
        #   DirectRunoff (sealed) + Forest + Other + Irrigation + Water + Rice = 1
        sealed = 0.02
        irrig = 0.0
        water = 0.01
        rice = 0.0
        other = max(0.0, 1.0 - sealed - forest_frac - irrig - water - rice)
        wm("DirectRunoffFraction", sealed, lat, lon)
        wm("ForestFraction", forest_frac, lat, lon)
        wm("OtherFraction", other, lat, lon)
        wm("IrrigationFraction", irrig, lat, lon)
        wm("WaterFraction", water, lat, lon)
        wm("RiceFraction", rice, lat, lon)

        # Keep backward-compatible name
        wm("forest_fraction", forest_frac, lat, lon)

        # ----- LAI maps (static, one per land-use type) -----
        # LISFLOOD reads LAI via a look-up table (laiofday.txt) that maps
        # day-of-year ranges to map suffixes. With static LAI we create a
        # single map per type and point the look-up table at it.
        lai_dir = self.maps_dir / "lai"
        lai_dir.mkdir(parents=True, exist_ok=True)
        n_lai_steps = 36
        for prefix, lai_val in [("laio", 3.0), ("laif", 4.0), ("laii", 3.5)]:
            arr = np.full((n_lai_steps, 3, 3), lai_val, dtype=np.float64)
            ds = xr.Dataset(
                {
                    prefix: xr.DataArray(
                        arr,
                        dims=["time", "y", "x"],
                        coords={"time": np.arange(n_lai_steps), "y": lat[::-1], "x": lon},
                        attrs={"units": "m2/m2"},
                    ),
                }
            )
            ds.to_netcdf(lai_dir / f"{prefix}.nc")

        # Also write the old-style single lai map for backward compat
        wm("lai", 3.0, lat, lon, "m2/m2")

        # ----- Snow parameters -----
        wm("snow_melt_coef", 4.5, lat, lon, "mm/degC/day")
        wm("temp_melt", 1.0, lat, lon, "degC")
        wm("temp_snow", 0.0, lat, lon, "degC")

        # ----- Groundwater -----
        wm("UpperGwLoss", 0.0, lat, lon, "mm/day")
        wm("LowerGwLoss", 0.0, lat, lon, "mm/day")
        wm("GwPercValue", 0.5, lat, lon, "mm/day")

        # ----- Generate look-up tables -----
        self._generate_lookup_tables()

        self.logger.info(f"Written static maps to {self.maps_dir}")

    # ------------------------------------------------------------------
    # Look-up tables required by LISFLOOD
    # ------------------------------------------------------------------
    def _generate_lookup_tables(self) -> None:
        """Create ``laiofday.txt`` and ``firstdayofmonth.txt`` in *tables_dir*."""
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # laiofday.txt: maps day-of-year ranges to LAI map suffixes.
        # With static (constant) LAI we use a single entry covering the
        # entire year; the suffix is empty so LISFLOOD reads e.g. "laio.nc".
        laiofday = "# Day ranges and LAI map suffixes\n# start end suffix\n1 366\n"
        (self.tables_dir / "laiofday.txt").write_text(laiofday)

        # firstdayofmonth.txt: maps day-of-year to monthly suffixes.
        # Standard 12-month table.
        lines = ["# Day ranges for monthly maps"]
        month_starts = [
            (1, 31),
            (32, 59),
            (60, 90),
            (91, 120),
            (121, 151),
            (152, 181),
            (182, 212),
            (213, 243),
            (244, 273),
            (274, 304),
            (305, 334),
            (335, 366),
        ]
        for i, (s, e) in enumerate(month_starts, 1):
            lines.append(f"{s} {e} {i:02d}")
        (self.tables_dir / "firstdayofmonth.txt").write_text("\n".join(lines) + "\n")

    def _generate_forcing(self) -> None:
        """Generate LISFLOOD meteorological forcing.

        LISFLOOD reads one NetCDF per variable (pr.nc, ta.nc, et0.nc, etc.)
        from a meteo/ directory. Source forcing is aggregated to the model
        timestep configured via ``LISFLOOD_FORCING_TIMESTEP`` (default daily).
        """
        self.logger.info("Generating LISFLOOD forcing data...")
        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()
        forcing_path = self._get_forcing_path()

        model_dt = int(
            self._get_config_value(
                lambda: self.config.model.lisflood.forcing_timestep,
                default=86400,
                dict_key="LISFLOOD_FORCING_TIMESTEP",
            )
        )

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            forcing_files = sorted(forcing_path.glob("**/*.nc"))
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")

        # Canonical reader: variables under SYMFLUENCE names with a declared
        # timestep, so we no longer guess source names, units, or the timestep.
        ds_forcing = open_canonical_forcing(forcing_files)
        ds_forcing = ds_forcing.sel(time=slice(str(start_date), str(end_date)))

        dx = 0.01
        lat = np.array([props["lat"] - dx, props["lat"], props["lat"] + dx])
        lon = np.array([props["lon"] - dx, props["lon"], props["lon"] + dx])

        # Precipitation: canonical precipitation_flux is a rate (kg m-2 s-1 = mm
        # s-1); scale by the declared timestep to mm per source step.
        precip = self._extract_forcing_var(ds_forcing, ["precipitation_flux"], props["lat"], props["lon"])
        raw_times = pd.DatetimeIndex(ds_forcing.time.values)
        source_dt = forcing_timestep_seconds(ds_forcing)
        precip = np.maximum(precip * source_dt, 0.0)

        # Temperature: canonical air_temperature is in kelvin.
        temp = self._extract_forcing_var(ds_forcing, ["air_temperature"], props["lat"], props["lon"]) - 273.15

        ds_forcing.close()

        # PET at source resolution (mm/timestep)
        doy = np.array([t.timetuple().tm_yday for t in raw_times])
        from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin

        pet_mm_day = PETCalculatorMixin.oudin_pet_numpy(temp, doy, props["lat"])
        pet = pet_mm_day * (source_dt / 86400.0)

        # Aggregate to model timestep if source is finer
        if source_dt < model_dt * 0.9:
            resample_freq = f"{model_dt}s"
            self.logger.info(f"Aggregating forcing from {source_dt:.0f}s to {model_dt}s timestep")
            idx = raw_times
            # sum mm/timestep → mm/day for precip and PET
            precip = pd.Series(precip, index=idx).resample(resample_freq).sum().values
            temp = pd.Series(temp, index=idx).resample(resample_freq).mean().values
            pet = pd.Series(pet, index=idx).resample(resample_freq).sum().values
            times = pd.Series(0, index=idx).resample(resample_freq).first().index
        else:
            times = raw_times

        nt = len(times)

        def _broadcast_1d(arr):
            return np.broadcast_to(arr.reshape(nt, 1, 1), (nt, 3, 3)).copy()

        coords = {"time": times, "y": (["y"], lat[::-1]), "x": (["x"], lon)}

        def _write_forcing(name, data, units):
            ds = xr.Dataset(
                {
                    name: xr.DataArray(_broadcast_1d(data), dims=["time", "y", "x"], attrs={"units": units}),
                },
                coords=coords,
            )
            ds.to_netcdf(self.forcing_out_dir / f"{name}.nc")

        _write_forcing("pr", precip, "mm/day")
        _write_forcing("ta", temp, "degC")
        _write_forcing("et0", pet, "mm/day")
        _write_forcing("e0", pet, "mm/day")
        _write_forcing("es", pet * 0.7, "mm/day")

        self.logger.info(f"Written forcing to {self.forcing_out_dir} ({nt} steps, dt={model_dt}s)")

    def _extract_forcing_var(self, ds, var_names, lat, lon):
        for var in var_names:
            if var in ds.data_vars:
                data = ds[var]
                spatial_dims = [d for d in data.dims if d not in ["time"]]
                if spatial_dims:
                    try:
                        data = data.sel(**{d: lat if "lat" in d else lon for d in spatial_dims}, method="nearest")
                    except Exception:  # noqa: BLE001
                        data = data.isel(**{d: 0 for d in spatial_dims})
                return data.values.flatten()
        raise ValueError(f"None of {var_names} found in forcing. Available: {list(ds.data_vars)}")

    def _estimate_pet(self, temp_c, times, lat_deg):
        from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin

        pet_method = self._get_config_value(
            lambda: self.config.model.lisflood.pet_method, default="oudin", dict_key="LISFLOOD_PET_METHOD"
        )

        doy = np.array([t.timetuple().tm_yday for t in times])
        if pet_method == "oudin":
            pet_mm_day = PETCalculatorMixin.oudin_pet_numpy(temp_c, doy, lat_deg)
            self.logger.info(f"Oudin PET: annual mean = {pet_mm_day.mean() * 365.25:.0f} mm/yr")
            return pet_mm_day

        self.logger.info(f"Using Hamon PET (method={pet_method})")
        return PETCalculatorMixin.hamon_pet_numpy(temp_c, doy, lat_deg)

    def _get_forcing_path(self) -> Path:
        forcing_path = self._get_config_value(
            lambda: self.config.paths.forcing_path, default=None, dict_key="FORCING_PATH"
        )
        if forcing_path and forcing_path != "default":
            return Path(forcing_path)
        # Store-first basin-averaged forcing (model_ready/forcings when present,
        # else legacy forcing/basin_averaged_data) — resolved by the base class.
        return self.forcing_basin_path

    def _generate_settings_xml(self) -> None:
        """Generate a complete LISFLOOD settings XML.

        The file follows the canonical three-section structure:

        ``<lfoptions>``  -- boolean switches
        ``<lfuser>``     -- user-defined scalar parameters and path variables
        ``<lfbinding>``  -- maps every LISFLOOD binding name to a value or
                            ``$(VarName)`` reference
        ``<prolog>``     -- PCRaster-style timer/mask prologue
        """
        self.logger.info("Generating LISFLOOD settings XML...")
        start_date, end_date = self._get_simulation_dates()

        domain_name = self._get_config_value(lambda: self.config.domain.name, default="", dict_key="DOMAIN_NAME")
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default="run_1",
            dict_key="EXPERIMENT_ID",
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default=".",
            dict_key="SYMFLUENCE_DATA_DIR",
        )
        output_dir = Path(data_dir) / f"domain_{domain_name}" / "simulations" / experiment_id / "LISFLOOD"
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dt = int(
            self._get_config_value(
                lambda: self.config.model.lisflood.forcing_timestep,
                default=86400,
                dict_key="LISFLOOD_FORCING_TIMESTEP",
            )
        )
        timestep_secs = str(model_dt)

        # Align calendar to midnight for daily timestep (forcing is midnight-aligned)
        if model_dt >= 86400:
            start_aligned = start_date.replace(hour=0, minute=0, second=0)
            end_aligned = end_date.replace(hour=0, minute=0, second=0)
        else:
            start_aligned = start_date
            end_aligned = end_date

        cal_start = start_aligned.strftime("%d/%m/%Y %H:%M")
        sim_start = start_aligned.strftime("%d/%m/%Y %H:%M")
        sim_end = end_aligned.strftime("%d/%m/%Y %H:%M")

        root = ET.Element("lfsettings")

        # ==============================================================
        # lfoptions -- model switches (minimal cold-start configuration)
        # ==============================================================
        lfoptions = ET.SubElement(root, "lfoptions")
        options = {
            # Temperature input
            "TemperatureInKelvin": "0",
            "gridSizeUserDefined": "1",
            # Hydrological modules
            "groundwaterSmooth": "0",
            "wateruse": "0",
            "TransientWaterDemandChange": "0",
            "wateruseRegion": "0",
            "cropsEPIC": "0",
            "drainedIrrigation": "0",
            "riceIrrigation": "0",
            "openwaterevapo": "0",
            "simulateLakes": "0",
            "simulateReservoirs": "0",
            "SplitRouting": "0",
            "inflow": "0",
            "indicator": "0",
            "TransientLandUseChange": "0",
            # Initialisation
            "InitLisflood": "0",
            "InitLisfloodwithoutsplit": "0",
            # I/O format
            "readNetcdfStack": "1",
            "writeNetcdfStack": "1",
            "writeNetcdf": "1",
            # Reporting -- time series
            "repDischargeTs": "1",
            "repStateSites": "0",
            "repRateSites": "0",
            "repStateUpsGauges": "0",
            "repRateUpsGauges": "0",
            "repMeteoUpsGauges": "0",
            "repsimulateLakes": "0",
            "repsimulateReservoirs": "0",
            # Reporting -- maps
            "repStateMaps": "1",
            "repEndMaps": "0",
            "repDischargeMaps": "1",
            "repSurfaceRunoffMaps": "0",
            "repTotalRunoffMaps": "0",
            "repPrefFlowMaps": "0",
            "repTaMaps": "0",
            "repRainMaps": "0",
            "repSnowMaps": "0",
            "repSnowCoverMaps": "0",
            "repSnowMeltMaps": "0",
            "repThetaMaps": "0",
            "repThetaForestMaps": "0",
            "repLZMaps": "0",
            "repUZMaps": "0",
            "repGwPercUZLZMaps": "0",
            "repPercolationMaps": "0",
            "repRWS": "0",
            "repPFMaps": "0",
            "repTotalWaterStorageMaps": "0",
            "repPFForestMaps": "0",
            "repTotalAbs": "0",
            "repTotalWUse": "0",
            "repWIndex": "0",
        }
        for name, choice in options.items():
            self._add_choice_option(lfoptions, name, choice)

        # ==============================================================
        # lfuser -- scalar parameters and path definitions
        # ==============================================================
        lfuser = ET.SubElement(root, "lfuser")

        # -- NetCDF / performance tuning --
        lfuser_vars = {
            "NetCDFTimeChunks": "-1",
            "MapsCaching": "False",
            "OutputMapsChunks": "1",
            "OutputMapsDataType": "float64",
            "numCPUs_parallelNumba": "1",
            # -- Projection --
            "proj4_params": "+proj=longlat +datum=WGS84 +no_defs",
            # -- Paths --
            "PathMaps": str(self.maps_dir),
            "PathMeteo": str(self.forcing_out_dir),
            "PathOut": str(output_dir),
            "PathInit": str(self.inits_dir),
            "PathTables": str(self.tables_dir),
            "PathLAI": str(self.maps_dir / "lai"),
            # -- Mask & gauges --
            "MaskMap": "$(PathMaps)/outlet.nc",
            "Gauges": "$(PathMaps)/Gauges.nc",
            "netCDFtemplate": "$(PathMaps)/netCDFtemplate.nc",
            "Latitude": "$(PathMaps)/lat.nc",
            # -- Calendar / time stepping --
            "CalendarConvention": "proleptic_gregorian",
            "CalendarDayStart": cal_start,
            "DtSec": timestep_secs,
            "DtSecChannel": timestep_secs,
            "StepStart": sim_start,
            "StepEnd": sim_end,
            "ReportSteps": "1..999999",
            # -- Monte Carlo / ensemble --
            "EnsMembers": "1",
            "FilterSteps": "0",
            # -- Meteorological prefixes --
            "PrefixPrecipitation": "pr",
            "PrefixTavg": "ta",
            "PrefixE0": "e0",
            "PrefixES0": "es",
            "PrefixET0": "et0",
            "PrefixLAIOther": "laio",
            "PrefixLAIForest": "laif",
            "PrefixLAIIrrigation": "laii",
            # -- Calibration / process parameters --
            "UpperZoneTimeConstant": "10",
            "LowerZoneTimeConstant": "100",
            "GwPercValue": "0.5",
            "GwLoss": "0.0",
            "LZThreshold": "0.0",
            "b_Xinanjiang": "0.7",
            "PowerPrefFlow": "3.5",
            "CalChanMan": "2.0",
            "SnowMeltCoef": "4.5",
            "AvWaterRateThreshold": "5.0",
            # -- ET / interception parameters --
            "PrScaling": "1",
            "CalEvaporation": "1.0",
            "LeafDrainageTimeConstant": "1",
            "kdf": "0.72",
            "SMaxSealed": "1.0",
            # -- Snow / frost parameters --
            "SnowFactor": "1",
            "SnowSeasonAdj": "1.0",
            "TempMelt": "1.0",
            "TempSnow": "1.0",
            "TemperatureLapseRate": "0.0065",
            "Afrost": "0.97",
            "Kfrost": "0.57",
            "SnowWaterEquivalent": "0.45",
            "FrostIndexThreshold": "56",
            # -- Routing parameters --
            "beta": "0.6",
            "OFDepRef": "5",
            "GradMin": "0.001",
            "ChanGradMin": "0.0001",
            # -- Numerics --
            "CourantCrit": "0.4",
            # -- Open water evaporation --
            "maxNoEva": "10",
            # -- Initial conditions (Other) --
            "timestepInit": cal_start,
            "OFDirectInitValue": "0",
            "OFOtherInitValue": "0",
            "OFForestInitValue": "0",
            "SnowCoverAInitValue": "0",
            "SnowCoverBInitValue": "0",
            "SnowCoverCInitValue": "0",
            "FrostIndexInitValue": "0",
            "CumIntInitValue": "0",
            "UZInitValue": "0",
            "DSLRInitValue": "1",
            # Auto-initialised variables (use -9999)
            "LZInitValue": "-9999",
            "TotalCrossSectionAreaInitValue": "-9999",
            "ThetaInit1Value": "-9999",
            "ThetaInit2Value": "-9999",
            "ThetaInit3Value": "-9999",
            "PrevDischarge": "0",
            # -- Initial conditions (Forest) --
            "CumIntForestInitValue": "0",
            "UZForestInitValue": "0",
            "DSLRForestInitValue": "1",
            "ThetaForestInit1Value": "-9999",
            "ThetaForestInit2Value": "-9999",
            "ThetaForestInit3Value": "-9999",
            # -- Initial conditions (Irrigation) --
            "CumIntIrrigationInitValue": "0",
            "UZIrrigationInitValue": "0",
            "DSLRIrrigationInitValue": "1",
            "ThetaIrrigationInit1Value": "-9999",
            "ThetaIrrigationInit2Value": "-9999",
            "ThetaIrrigationInit3Value": "-9999",
            # -- Initial conditions (Sealed) --
            "CumIntSealedInitValue": "0",
        }
        for name, value in lfuser_vars.items():
            self._add_text_option(lfuser, name, value)

        # ==============================================================
        # lfbinding -- map every LISFLOOD binding name to its source
        # ==============================================================
        lfbinding = ET.SubElement(root, "lfbinding")
        bindings = self._build_lfbinding_dict(
            cal_start,
            sim_start,
            sim_end,
            timestep_secs,
            str(output_dir),
        )
        for name, value in bindings.items():
            self._add_text_option(lfbinding, name, value)

        # ==============================================================
        # prolog
        # ==============================================================
        prolog = ET.SubElement(root, "prolog")
        prolog.text = "\nareamap MaskMap;\ntimer StepStart StepEnd 1;\nReportSteps=$(ReportSteps);\n"

        # ---- write ----
        settings_file = self._get_config_value(
            lambda: self.config.model.lisflood.settings_file,
            default="settings.xml",
        )
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(
            self.settings_dir / settings_file,
            encoding="unicode",
            xml_declaration=True,
        )
        self.logger.info(f"Written settings XML to {self.settings_dir / settings_file}")

    # ------------------------------------------------------------------
    # lfbinding dictionary builder
    # ------------------------------------------------------------------
    def _build_lfbinding_dict(  # noqa: PLR0913
        self,
        cal_start: str,
        sim_start: str,
        sim_end: str,
        timestep_secs: str,
        output_dir: str,
    ) -> Dict:
        """Return an *ordered* dict of all lfbinding entries."""
        b: Dict[str, str] = {}

        # -- Performance / netCDF tuning --
        b["numCPUs_parallelNumba"] = "$(numCPUs_parallelNumba)"
        b["proj4_params"] = "$(proj4_params)"
        b["OutputMapsChunks"] = "$(OutputMapsChunks)"
        b["OutputMapsDataType"] = "$(OutputMapsDataType)"
        b["NetCDFTimeChunks"] = "$(NetCDFTimeChunks)"
        b["MapsCaching"] = "$(MapsCaching)"

        # -- Mask, template, latitude --
        b["MaskMap"] = "$(MaskMap)"
        b["netCDFtemplate"] = "$(netCDFtemplate)"
        b["Latitude"] = "$(Latitude)"

        # -- Time stepping --
        b["CalendarConvention"] = "$(CalendarConvention)"
        b["CalendarDayStart"] = "$(CalendarDayStart)"
        b["DtSec"] = "$(DtSec)"
        b["DtSecChannel"] = "$(DtSecChannel)"
        b["StepStart"] = "$(StepStart)"
        b["StepEnd"] = "$(StepEnd)"

        # -- ET / interception --
        b["PrScaling"] = "$(PrScaling)"
        b["CalEvaporation"] = "$(CalEvaporation)"
        b["LeafDrainageTimeConstant"] = "$(LeafDrainageTimeConstant)"
        b["kdf"] = "$(kdf)"
        b["AvWaterRateThreshold"] = "$(AvWaterRateThreshold)"
        b["SMaxSealed"] = "$(SMaxSealed)"

        # -- Snow / frost --
        b["SnowFactor"] = "$(SnowFactor)"
        b["SnowMeltCoef"] = "$(PathMaps)/snow_melt_coef.nc"
        b["SnowSeasonAdj"] = "$(SnowSeasonAdj)"
        b["TempMelt"] = "$(PathMaps)/temp_melt.nc"
        b["TempSnow"] = "$(PathMaps)/temp_snow.nc"
        b["TemperatureLapseRate"] = "$(TemperatureLapseRate)"
        b["Afrost"] = "$(Afrost)"
        b["Kfrost"] = "$(Kfrost)"
        b["SnowWaterEquivalent"] = "$(SnowWaterEquivalent)"
        b["FrostIndexThreshold"] = "$(FrostIndexThreshold)"

        # -- Infiltration --
        b["b_Xinanjiang"] = "$(b_Xinanjiang)"
        b["PowerPrefFlow"] = "$(PowerPrefFlow)"

        # -- Groundwater --
        b["UpperZoneTimeConstant"] = "$(UpperZoneTimeConstant)"
        b["LowerZoneTimeConstant"] = "$(LowerZoneTimeConstant)"
        b["GwPercValue"] = "$(GwPercValue)"
        b["GwLoss"] = "$(GwLoss)"
        b["LZThreshold"] = "$(LZThreshold)"

        # -- Open water evaporation --
        b["LakeMask"] = "$(PathMaps)/LakeMask.nc"
        b["maxNoEva"] = "$(maxNoEva)"

        # -- Routing --
        b["CalChanMan"] = "$(CalChanMan)"
        b["beta"] = "$(beta)"
        b["OFDepRef"] = "$(OFDepRef)"
        b["GradMin"] = "$(GradMin)"
        b["ChanGradMin"] = "$(ChanGradMin)"
        b["CourantCrit"] = "$(CourantCrit)"

        # -- Initial conditions (Other) --
        b["timestepInit"] = "$(timestepInit)"
        b["OFDirectInitValue"] = "$(OFDirectInitValue)"
        b["OFOtherInitValue"] = "$(OFOtherInitValue)"
        b["OFForestInitValue"] = "$(OFForestInitValue)"
        b["SnowCoverAInitValue"] = "$(SnowCoverAInitValue)"
        b["SnowCoverBInitValue"] = "$(SnowCoverBInitValue)"
        b["SnowCoverCInitValue"] = "$(SnowCoverCInitValue)"
        b["FrostIndexInitValue"] = "$(FrostIndexInitValue)"
        b["CumIntInitValue"] = "$(CumIntInitValue)"
        b["UZInitValue"] = "$(UZInitValue)"
        b["DSLRInitValue"] = "$(DSLRInitValue)"
        b["LZInitValue"] = "$(LZInitValue)"
        b["TotalCrossSectionAreaInitValue"] = "$(TotalCrossSectionAreaInitValue)"
        b["ThetaInit1Value"] = "$(ThetaInit1Value)"
        b["ThetaInit2Value"] = "$(ThetaInit2Value)"
        b["ThetaInit3Value"] = "$(ThetaInit3Value)"
        b["PrevDischarge"] = "$(PrevDischarge)"

        # -- Initial conditions (Forest) --
        b["CumIntForestInitValue"] = "$(CumIntForestInitValue)"
        b["UZForestInitValue"] = "$(UZForestInitValue)"
        b["DSLRForestInitValue"] = "$(DSLRForestInitValue)"
        b["ThetaForestInit1Value"] = "$(ThetaForestInit1Value)"
        b["ThetaForestInit2Value"] = "$(ThetaForestInit2Value)"
        b["ThetaForestInit3Value"] = "$(ThetaForestInit3Value)"
        b["CumIntSealedInitValue"] = "$(CumIntSealedInitValue)"

        # -- Initial conditions (Irrigation) --
        b["CumIntIrrigationInitValue"] = "$(CumIntIrrigationInitValue)"
        b["UZIrrigationInitValue"] = "$(UZIrrigationInitValue)"
        b["DSLRIrrigationInitValue"] = "$(DSLRIrrigationInitValue)"
        b["ThetaIrrigationInit1Value"] = "$(ThetaIrrigationInit1Value)"
        b["ThetaIrrigationInit2Value"] = "$(ThetaIrrigationInit2Value)"
        b["ThetaIrrigationInit3Value"] = "$(ThetaIrrigationInit3Value)"

        # -- Meteorological forcing --
        b["PrecipitationMaps"] = "$(PathMeteo)/$(PrefixPrecipitation)"
        b["TavgMaps"] = "$(PathMeteo)/$(PrefixTavg)"
        b["E0Maps"] = "$(PathMeteo)/$(PrefixE0)"
        b["ES0Maps"] = "$(PathMeteo)/$(PrefixES0)"
        b["ET0Maps"] = "$(PathMeteo)/$(PrefixET0)"

        # -- LAI maps --
        b["LAIOtherMaps"] = "$(PathLAI)/$(PrefixLAIOther)"
        b["LAIForestMaps"] = "$(PathLAI)/$(PrefixLAIForest)"
        b["LAIIrrigationMaps"] = "$(PathLAI)/$(PrefixLAIIrrigation)"
        b["LAIOfDay"] = "$(PathTables)/laiofday.txt"
        b["FirstDayOfMonth"] = "$(PathTables)/firstdayofmonth.txt"

        # -- Topography / base maps --
        b["Ldd"] = "$(PathMaps)/ldd.nc"
        b["Outlets"] = "$(PathMaps)/outlet.nc"
        b["Catchments"] = "$(PathMaps)/catchment.nc"
        b["Dem"] = "$(PathMaps)/dem.nc"
        b["CellArea"] = "$(PathMaps)/cellarea.nc"
        b["Grad"] = "$(PathMaps)/Grad.nc"
        b["ElevationStD"] = "$(PathMaps)/ElevationStD.nc"
        b["Gauges"] = "$(Gauges)"

        # -- Soil depth (Other) --
        b["SoilDepth1"] = "$(PathMaps)/soil_depth1.nc"
        b["SoilDepth2"] = "$(PathMaps)/soil_depth2.nc"
        b["SoilDepth3"] = "$(PathMaps)/soil_depth3.nc"
        # -- Soil depth (Forest) --
        b["SoilDepth1Forest"] = "$(PathMaps)/SoilDepth1Forest.nc"
        b["SoilDepth2Forest"] = "$(PathMaps)/SoilDepth2Forest.nc"
        b["SoilDepth3Forest"] = "$(PathMaps)/SoilDepth3Forest.nc"

        # -- Land-use fractions --
        b["DirectRunoffFraction"] = "$(PathMaps)/DirectRunoffFraction.nc"
        b["ForestFraction"] = "$(PathMaps)/ForestFraction.nc"
        b["OtherFraction"] = "$(PathMaps)/OtherFraction.nc"
        b["IrrigationFraction"] = "$(PathMaps)/IrrigationFraction.nc"
        b["WaterFraction"] = "$(PathMaps)/WaterFraction.nc"
        b["RiceFraction"] = "$(PathMaps)/RiceFraction.nc"

        # -- Crop / vegetation --
        b["MapCropCoef"] = "$(PathMaps)/MapCropCoef.nc"
        b["MapCropGroupNumber"] = "$(PathMaps)/MapCropGroupNumber.nc"
        b["MapN"] = "$(PathMaps)/MapN.nc"
        b["MapForestCropCoef"] = "$(PathMaps)/MapForestCropCoef.nc"
        b["MapForestCropGroupNumber"] = "$(PathMaps)/MapForestCropGroupNumber.nc"
        b["MapForestN"] = "$(PathMaps)/MapForestN.nc"
        b["MapIrrigationCropCoef"] = "$(PathMaps)/MapIrrigationCropCoef.nc"
        b["MapIrrigationCropGroupNumber"] = "$(PathMaps)/MapIrrigationCropGroupNumber.nc"

        # -- Channel maps --
        b["Channels"] = "$(PathMaps)/Channels.nc"
        b["ChanGrad"] = "$(PathMaps)/changrad.nc"
        b["ChanMan"] = "$(PathMaps)/chanman.nc"
        b["ChanLength"] = "$(PathMaps)/chanlength.nc"
        b["ChanBottomWidth"] = "$(PathMaps)/chanbw.nc"
        b["ChanSdXdY"] = "$(PathMaps)/ChanSdXdY.nc"
        b["ChanDepthThreshold"] = "$(PathMaps)/chandepth.nc"

        # -- Soil hydraulic parameters (Other) --
        b["MapThetaSat1"] = "$(PathMaps)/MapThetaSat1.nc"
        b["MapThetaSat2"] = "$(PathMaps)/MapThetaSat2.nc"
        b["MapThetaSat3"] = "$(PathMaps)/MapThetaSat3.nc"
        b["MapThetaRes1"] = "$(PathMaps)/MapThetaRes1.nc"
        b["MapThetaRes2"] = "$(PathMaps)/MapThetaRes2.nc"
        b["MapThetaRes3"] = "$(PathMaps)/MapThetaRes3.nc"
        b["MapLambda1"] = "$(PathMaps)/MapLambda1.nc"
        b["MapLambda2"] = "$(PathMaps)/MapLambda2.nc"
        b["MapLambda3"] = "$(PathMaps)/MapLambda3.nc"
        b["MapGenuAlpha1"] = "$(PathMaps)/MapGenuAlpha1.nc"
        b["MapGenuAlpha2"] = "$(PathMaps)/MapGenuAlpha2.nc"
        b["MapGenuAlpha3"] = "$(PathMaps)/MapGenuAlpha3.nc"
        b["MapKSat1"] = "$(PathMaps)/MapKSat1.nc"
        b["MapKSat2"] = "$(PathMaps)/MapKSat2.nc"
        b["MapKSat3"] = "$(PathMaps)/MapKSat3.nc"

        # -- Soil hydraulic parameters (Forest, layers 1 & 2 only) --
        b["MapThetaSat1Forest"] = "$(PathMaps)/MapThetaSat1Forest.nc"
        b["MapThetaSat2Forest"] = "$(PathMaps)/MapThetaSat2Forest.nc"
        b["MapThetaRes1Forest"] = "$(PathMaps)/MapThetaRes1Forest.nc"
        b["MapThetaRes2Forest"] = "$(PathMaps)/MapThetaRes2Forest.nc"
        b["MapLambda1Forest"] = "$(PathMaps)/MapLambda1Forest.nc"
        b["MapLambda2Forest"] = "$(PathMaps)/MapLambda2Forest.nc"
        b["MapGenuAlpha1Forest"] = "$(PathMaps)/MapGenuAlpha1Forest.nc"
        b["MapGenuAlpha2Forest"] = "$(PathMaps)/MapGenuAlpha2Forest.nc"
        b["MapKSat1Forest"] = "$(PathMaps)/MapKSat1Forest.nc"
        b["MapKSat2Forest"] = "$(PathMaps)/MapKSat2Forest.nc"

        # -- Average LZ inflow and discharge --
        b["LZAvInflowMap"] = "$(PathMaps)/lzavin.nc"
        b["AvgDis"] = "$(PathMaps)/avgdis.nc"

        # -- Pixel-level geometry (for lat/lon grids) --
        b["PixelLengthUser"] = "$(PathMaps)/pixleng.nc"
        b["PixelAreaUser"] = "$(PathMaps)/pixarea.nc"

        # -- Output maps --
        b["DischargeMaps"] = "$(PathOut)/dis"
        b["DischargeTS"] = "$(PathOut)/dis"

        # -- Output time series (LISFLOOD requires a path for each) --
        tss_names = [
            "ChanqTS",
            "DisTS",
            "DSLRAvUpsTS",
            "DSLRTS",
            "ESActAvUpsTS",
            "ESActTS",
            "ETRefAvUpsTS",
            "EWIntAvUpsTS",
            "EWIntTS",
            "EWRefAvUpsTS",
            "EvaOpenWaterAvUpsTS",
            "FrostIndexAvUpsTS",
            "FrostIndexTS",
            "GwLossAvUpsTS",
            "GwLossTS",
            "GwPercUZLZAvUpsTS",
            "GwPercUZLZTS",
            "InfiltrationAvUpsTS",
            "InfiltrationTS",
            "LZAvInflowAvUps",
            "LZAvInflowTS",
            "LZAvUpsTS",
            "LZOutflowAvUpsTS",
            "LZOutflowTS",
            "LZTS",
            "PercolationAvUpsTS",
            "PercolationTS",
            "PF1AvUpsTS",
            "PF1ForestAvUpsTS",
            "PF1ForestTS",
            "PF1TS",
            "PF2AvUpsTS",
            "PF2ForestAvUpsTS",
            "PF2ForestTS",
            "PF2TS",
            "PrecipitationAvUpsTS",
            "PrefFlowAvUpsTS",
            "PrefFlowTS",
            "RainAvUpsTS",
            "RainTS",
            "SeepSubToGWAvUpsTS",
            "SeepSubToGWTS",
            "SnowAvUpsTS",
            "SnowCoverAvUpsTS",
            "SnowCoverTS",
            "SnowmeltAvUpsTS",
            "SnowmeltTS",
            "SnowTS",
            "SurfaceRunoffAvUpsTS",
            "SurfaceRunoffTS",
            "TaAvUpsTS",
            "TaTS",
            "TavgAvUpsTS",
            "Theta1AvUpsTS",
            "Theta1TS",
            "Theta2AvUpsTS",
            "Theta2TS",
            "Theta3AvUpsTS",
            "Theta3TS",
            "TotalRunoffAvUpsTS",
            "TotalRunoffTS",
            "UZAvUpsTS",
            "UZOutflowAvUpsTS",
            "UZOutflowTS",
            "WaterDepthAvUpsTS",
            "WaterDepthTS",
            "WaterLevelTS",
            "actETPUpsTS",
            "LakeInflowTS",
            "LakeLevelTS",
            "LakeOutflowTS",
            "ReservoirFillTS",
            "ReservoirInflowTS",
            "ReservoirOutflowTS",
            "PolderFluxTS",
            "PolderLevelTS",
            "WaterUseTS",
            "WaterUseSitesTS",
            "StepsWaterUseTS",
            "StepsSoilTS",
            "RunLossUpsTS",
            "MassBalanceMMTSS",
            "WaterMassBalanceTSS",
            "MBErrorStorageRatioTSS",
            "MassBalanceErrorSplitRoutingTSS",
            "OutletDischargeErrorSplitRoutingTSS",
            "AverageFractionsCatchmentTSS",
        ]
        for name in tss_names:
            b[name] = f"$(PathOut)/{name}.tss"

        # -- Output maps (state, end, discharge, etc.) --
        map_names = [
            "Theta1State",
            "Theta2State",
            "Theta3State",
            "Theta1ForestState",
            "Theta2ForestState",
            "Theta3ForestState",
            "Theta1IrrigationState",
            "Theta2IrrigationState",
            "Theta3IrrigationState",
            "DSLRState",
            "DSLRForestState",
            "DSLRIrrigationState",
            "CumIntState",
            "CumIntForestState",
            "CumIntIrrigationState",
            "CumIntSealedState",
            "UZState",
            "UZForestState",
            "UZIrrigationState",
            "LZState",
            "FrostIndexState",
            "SnowCoverAState",
            "SnowCoverBState",
            "SnowCoverCState",
            "CrossSection2State",
            "PrevSideflowState",
            "TotalCrossSectionState",
            "OFOtherState",
            "OFForestState",
            "OFDirectState",
            "DisMaps",
            "TopSoilMoistureMaps",
            "SurfaceSoilMoistureMaps",
            "MaskDischargeMaps",
            "WaterLevelMaps",
        ]
        for name in map_names:
            b[name] = f"$(PathOut)/{name}"

        return b

    @staticmethod
    def _add_text_option(parent, name, value):
        elem = ET.SubElement(parent, "textvar")
        elem.set("name", name)
        elem.set("value", value)

    @staticmethod
    def _add_choice_option(parent, name, value):
        elem = ET.SubElement(parent, "setoption")
        elem.set("name", name)
        elem.set("choice", value)
