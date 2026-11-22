# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NEX-GDDP-CMIP6 climate projection data acquisition via THREDDS.

Provides automated download of NASA NEX-GDDP-CMIP6 downscaled climate model
outputs with support for multiple models, scenarios, and ensemble members.
"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('NEX-GDDP-CMIP6')
@R.acquisition_handlers.add('NEX-GDDP')
class NEXGDDPCHandler(BaseAcquisitionHandler):
    """
    Acquires NEX-GDDP-CMIP6 downscaled climate projection data via THREDDS.

    NASA NEX-GDDP-CMIP6 provides bias-corrected, downscaled (0.25°) climate
    projections from CMIP6 models. Supports multiple models, scenarios
    (historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5), and ensemble members.
    """

    # Filename suffixes published on the THREDDS server, newest first. Most
    # files carry _v2.0, some only _v1.1, and a few are unsuffixed.
    FILENAME_SUFFIXES = ("_v2.0", "_v1.1", "")

    def download(self, output_dir: Path) -> Path:
        start_dt, end_dt = self.start_date.date(), self.end_date.date()
        bbox = self.bbox
        lat_min, lat_max = sorted([bbox["lat_min"], bbox["lat_max"]])
        lon_min, lon_max = sorted([bbox["lon_min"], bbox["lon_max"]])
        cfg_models = self._get_config_value(lambda: self.config.forcing.nex.models, dict_key='NEX_MODELS')
        cfg_scenarios = self._get_config_value(lambda: self.config.forcing.nex.scenarios, default=["historical"], dict_key='NEX_SCENARIOS')
        variables = self._get_config_value(lambda: self.config.forcing.nex.variables, default=["hurs", "huss", "pr", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"], dict_key='NEX_VARIABLES')
        cfg_members = self._get_config_value(lambda: self.config.forcing.nex.ensembles, default=["r1i1p1f1"], dict_key='NEX_ENSEMBLES')
        if not cfg_models: raise ValueError("NEX_MODELS must be set.")

        # Grid label mapping for different models (CMIP6 models use different grid labels)
        grid_labels = {
            "GFDL-ESM4": "gr1",
            "GFDL-CM4": "gr1",
            "INM-CM4-8": "gr1",
            "INM-CM5-0": "gr1",
            "IPSL-CM6A-LR": "gr",
            "CNRM-CM6-1": "gr",
            "CNRM-ESM2-1": "gr",
            # Most other models use 'gn' for native grid
        }

        # Models using a 360-day calendar (every month has 30 days, Dec 31 is invalid)
        models_360day = {
            "UKESM1-0-LL", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM",
            "CNRM-CM6-1", "CNRM-ESM2-1", "KACE-1-0-G",
        }

        ncss_base = "https://ds.nccs.nasa.gov/thredds/ncss/grid"
        cache_root = output_dir / "_nex_ncss_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        ensemble_datasets: list[Any] = []
        missing_variables: list[str] = []
        for model_name in cfg_models:
            # Get the grid label for this model, default to 'gn' (native grid)
            grid_label = grid_labels.get(model_name, "gn")
            is_360day = model_name in models_360day
            year_end_day = 30 if is_360day else 31

            for scenario_name in cfg_scenarios:
                scenario_end_dt = min(end_dt, dt.date(2014, 12, year_end_day)) if scenario_name == "historical" else end_dt
                if start_dt > scenario_end_dt: continue
                for member in cfg_members:
                    all_nc_files_for_ens = []
                    for var in variables:
                        var_cache_dir = cache_root / model_name / scenario_name / member / var
                        var_cache_dir.mkdir(parents=True, exist_ok=True)
                        var_files_found = 0
                        attempted_chunks = 0
                        for year in range(start_dt.year, scenario_end_dt.year + 1):
                            chunk_start = max(start_dt, dt.date(year, 1, 1))
                            chunk_end = min(scenario_end_dt, dt.date(year, 12, year_end_day))
                            if chunk_start > chunk_end: continue
                            attempted_chunks += 1
                            # Filenames on the server carry _v2.0, _v1.1, or no
                            # suffix depending on model/variable; try each in turn
                            # instead of hardcoding _v2.0 and silently dropping
                            # the variable on 404.
                            year_nc = self._fetch_year_chunk(
                                ncss_base, var_cache_dir, var, model_name, scenario_name,
                                member, grid_label, year, chunk_start, chunk_end,
                                lat_min, lat_max, lon_min, lon_max)
                            if year_nc is not None:
                                all_nc_files_for_ens.append(str(year_nc))
                                var_files_found += 1
                            else:
                                self.logger.warning(
                                    f"NEX-GDDP-CMIP6: no file found for {var} "
                                    f"{model_name}/{scenario_name}/{member} {year} "
                                    f"(tried suffixes {', '.join(repr(s) for s in self.FILENAME_SUFFIXES)})")
                        if attempted_chunks > 0 and var_files_found < attempted_chunks:
                            # A missing year leaves a gap in the forcing just as a
                            # missing variable does, so it fails the same way.
                            missing_variables.append(
                                f"{model_name}/{scenario_name}/{member}/{var}"
                                + ("" if var_files_found == 0
                                   else f" ({attempted_chunks - var_files_found} of {attempted_chunks} years)"))
                    if all_nc_files_for_ens:
                        ds_ens = xr.open_mfdataset(all_nc_files_for_ens, engine="netcdf4", combine="by_coords", parallel=False, data_vars='minimal', coords='minimal', compat='override').chunk({"time": -1})
                        ds_ens = ds_ens.expand_dims(ensemble=[len(ensemble_datasets)]).assign_coords(model=("ensemble", [model_name]), scenario=("ensemble", [scenario_name]), member=("ensemble", [member]))
                        ensemble_datasets.append(ds_ens)
        if missing_variables:
            # Silent partial forcing is worse than failure: a model fed a
            # subset of its forcing variables produces garbage downstream.
            raise RuntimeError(
                "NEX-GDDP-CMIP6: no files could be downloaded for requested "
                f"variable(s): {', '.join(missing_variables)}. "
                "Cached files for the other variables were kept under "
                f"{cache_root} for reuse on retry."
            )
        if not ensemble_datasets:
            if cache_root.exists(): shutil.rmtree(cache_root)
            raise RuntimeError("NEX-GDDP-CMIP6: no data written.")
        ds_all = xr.concat(ensemble_datasets, dim="ensemble")
        # Handle cftime objects (e.g., DatetimeNoLeap, Datetime360Day)
        # Convert to pandas datetime and reassign to dataset for consistent slicing
        time_raw = ds_all["time"].values
        source_cal = getattr(time_raw[0], 'calendar', None) if hasattr(time_raw[0], 'calendar') else None
        any_360day = any(m in models_360day for m in cfg_models)
        if source_cal == '360_day':
            # True cftime 360-day objects (e.g. Feb 30 exists).
            # Map each 360-day DOY proportionally to standard calendar DOY.
            import calendar as cal_mod
            def _360day_to_datetime(t):
                doy_360 = (t.month - 1) * 30 + t.day  # 1-360
                days_in_year = 366 if cal_mod.isleap(t.year) else 365
                doy_std = max(1, min(days_in_year, round(doy_360 * days_in_year / 360)))
                return dt.datetime(t.year, 1, 1) + dt.timedelta(days=doy_std - 1)
            time_vals = pd.DatetimeIndex([_360day_to_datetime(t) for t in time_raw])
            # Remove duplicates from rounding (keep first occurrence)
            mask = ~time_vals.duplicated(keep='first')
            ds_all = ds_all.isel(time=mask)
            time_vals = time_vals[mask]
        elif hasattr(time_raw[0], 'strftime'):
            time_vals = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in time_raw])
        else:
            time_vals = pd.to_datetime(time_raw)
        ds_all = ds_all.assign_coords(time=time_vals)
        # Non-standard calendars (360_day, noleap, 365_day) or known 360-day models
        # may have gaps after conversion to Gregorian (e.g. missing Dec 31, Feb 29,
        # or ~5 skipped days in 360-day mapping). Interpolate to a complete daily
        # series so every Gregorian date is present.
        needs_interp = (
            any_360day
            or source_cal == '360_day'
            or (source_cal and source_cal not in ('standard', 'gregorian', 'proleptic_gregorian'))
        )
        if needs_interp:
            full_daily = pd.date_range(time_vals.min(), time_vals.max(), freq="D")
            if len(full_daily) > len(time_vals):
                ds_all = ds_all.interp(time=full_daily, method="linear")
                self.logger.info(f"Interpolated calendar gaps: {len(time_vals)} -> {len(full_daily)} standard-calendar days")
                time_vals = full_daily
        # Clear inherited calendar encoding so output uses standard calendar
        if "time" in ds_all.encoding:
            ds_all.encoding.pop("time", None)
        for var in ds_all.data_vars:
            ds_all[var].encoding.pop("calendar", None)
        # NEX-GDDP-CMIP6 does not publish surface pressure; fabricate a constant
        # airpres = p0 * exp(-z/H) because downstream models require the variable.
        elev_cfg = self._get_config_value(lambda: None, default=None, dict_key='DOMAIN_MEAN_ELEV_M')
        if elev_cfg is None:
            self.logger.warning(
                "DOMAIN_MEAN_ELEV_M is not set: the synthetic 'airpres' variable will be a "
                "CONSTANT SEA-LEVEL pressure (101325 Pa) across the whole domain. For an "
                "elevation-adjusted estimate, set DOMAIN_MEAN_ELEV_M (domain mean elevation "
                "in metres) in your configuration."
            )
        p0, z_mean, scale_height = 101325.0, float(elev_cfg or 0.0), 8400.0
        p_surf = p0 * np.exp(-z_mean / scale_height)
        month_starts = pd.date_range(time_vals[0].replace(day=1), time_vals[-1], freq="MS")
        for ms in month_starts:
            me = (ms + pd.offsets.MonthEnd(0))
            ds_m = ds_all.sel(time=slice(ms, me))
            if "time" not in ds_m.dims or ds_m.sizes["time"] == 0: continue
            if "ensemble" in ds_m.dims: ds_m = ds_m.isel(ensemble=0, drop=True)
            if "airpres" not in ds_m:
                ds_m["airpres"] = xr.full_like(ds_m["tas"], p_surf, dtype="float32").assign_attrs(long_name="synthetic surface air pressure", units="Pa")
            month_path = output_dir / f"NEXGDDP_all_{ms.year:04d}{ms.month:02d}.nc"
            # Ensure time is written with standard calendar encoding
            time_encoding = {"units": "hours since 1990-01-01", "calendar": "standard"}
            ds_m.to_netcdf(month_path, engine="netcdf4", encoding={"time": time_encoding})
        ds_all.close()
        if cache_root.exists(): shutil.rmtree(cache_root)
        return output_dir

    def _fetch_year_chunk(
        self, ncss_base: str, var_cache_dir: Path, var: str, model_name: str,
        scenario_name: str, member: str, grid_label: str, year: int,
        chunk_start: dt.date, chunk_end: dt.date,
        lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    ) -> Path | None:
        """Download one variable-year via NCSS, trying each filename suffix.

        Returns the cached/downloaded NetCDF path, or None when every suffix
        variant failed (404 or transport error).
        """
        params = {
            "var": var, "north": lat_max, "south": lat_min,
            "west": lon_min, "east": lon_max, "horizStride": 1,
            "time_start": f"{chunk_start.isoformat()}T12:00:00Z",
            "time_end": f"{chunk_end.isoformat()}T12:00:00Z",
            "accept": "netcdf4-classic",
        }
        for suffix in self.FILENAME_SUFFIXES:
            fname = f"{var}_day_{model_name}_{scenario_name}_{member}_{grid_label}_{year}{suffix}.nc"
            dataset_path = f"AMES/NEX/GDDP-CMIP6/{model_name}/{scenario_name}/{member}/{var}/{fname}"
            out_nc = var_cache_dir / f"{fname.replace('.nc', '')}_{chunk_start:%Y%m%d}-{chunk_end:%Y%m%d}.nc"
            if out_nc.exists():
                return out_nc
            try:
                resp = requests.get(f"{ncss_base}/{dataset_path}", params=params, stream=True, timeout=600)
                if resp.status_code == 200:
                    # Write to .part and rename only when complete; a truncated
                    # file at out_nc would be reused as cached on the next run.
                    part = out_nc.with_name(out_nc.name + ".part")
                    try:
                        with open(part, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                        part.replace(out_nc)
                    finally:
                        part.unlink(missing_ok=True)
                    return out_nc
                self.logger.debug(
                    f"NCSS request failed with status {resp.status_code} for {fname}: {resp.text[:500]}")
            except Exception as e:  # noqa: BLE001 — try the next suffix variant
                self.logger.warning(f"NCSS failed for {fname}: {e}", exc_info=True)
        return None
