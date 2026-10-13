"""Validated daily forcing and geometry for elevation-band model adapters."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
from symfluence.core.modeling.model_ready.forcing_reader import forcing_timestep_seconds, open_canonical_forcing


def daily_bands(forcing_dir, hru_file, start, end):
    bands = gpd.read_file(hru_file).sort_values("HRU_ID").reset_index(drop=True)
    if not {"HRU_ID", "HRU_area", "elev_mean", "center_lat", "center_lon"} <= set(bands):
        raise ValidationError("Elevation geometry lacks required HRU attributes")
    if bands.HRU_ID.duplicated().any() or (bands.HRU_area <= 0).any():
        raise ValidationError("Invalid or duplicate elevation HRUs")
    files = sorted(Path(forcing_dir).glob("*.nc"))
    ds = open_canonical_forcing(files)
    try:
        ds = ds.sel(
            time=slice(
                str(pd.Timestamp(start).normalize()),
                str(pd.Timestamp(end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)),
            )
        )
        dt = forcing_timestep_seconds(ds)
        ids = np.asarray(ds.hruId.values).reshape(-1)
        indices = [list(ids).index(i) for i in bands.HRU_ID]
        ds = ds.isel(hru=indices).load()
        times = pd.DatetimeIndex(ds.time.values)
        expected = pd.date_range(
            pd.Timestamp(start).normalize(),
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1),
            freq=pd.Timedelta(seconds=dt),
            inclusive="left",
        )
        if not times.equals(expected):
            raise ValidationError("Forcing does not cover every requested timestep")
        p = pd.DataFrame(ds.precipitation_flux.values * dt, index=times).resample("D").sum()
        t = pd.DataFrame(ds.air_temperature.values - 273.15, index=times)
        mean = t.resample("D").mean()
        lo = t.resample("D").min()
        hi = t.resample("D").max()
        pet = np.column_stack(
            [
                PETCalculatorMixin.oudin_pet_numpy(
                    mean.iloc[:, i].values, mean.index.dayofyear.values, float(bands.center_lat.iloc[i])
                )
                for i in range(len(bands))
            ]
        )
        out = xr.Dataset(
            {
                k: (("time", "hru"), v)
                for k, v in {
                    "precip": p.values,
                    "temp": mean.values,
                    "tmin": lo.values,
                    "tmax": hi.values,
                    "pet": pet,
                }.items()
            },
            coords={"time": mean.index, "hru": bands.HRU_ID.values},
        )
        if any(not np.isfinite(out[k]).all() for k in out) or (out.precip < 0).any():
            raise ValidationError("Nonfinite forcing or negative precipitation")
        return bands, out
    finally:
        ds.close()


def upper_ice_fractions(areas, elevations, fraction):
    areas = np.asarray(areas, dtype=float)
    remaining = areas.sum() * fraction
    ice = np.zeros(len(areas))
    for i in np.argsort(elevations)[::-1]:
        a = min(areas[i], remaining)
        ice[i] = a / areas[i]
        remaining -= a
    if remaining > 1e-5 or not 0 <= fraction <= 1:
        raise ValidationError("Invalid glacier fraction")
    return ice
