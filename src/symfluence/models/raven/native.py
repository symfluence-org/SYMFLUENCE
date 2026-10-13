"""Generate native Raven HBV-EC inputs; finite ice, no external water source.

PRECIP_CORR scales rain and snow at every gauge; SNOW_CORR is an additional snow-only factor;
TEMP_CORR is an additive gauge temperature correction (K). Defaults reproduce uncorrected forcing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = dict(
    RAINSNOW_TEMP=0.0,
    MELT_FACTOR=5.0,
    REFREEZE_FACTOR=1.0,
    HBV_BETA=2.0,
    SOIL_DEPTH=0.8,
    FAST_K=0.1,
    SLOW_K=0.01,
    PERC=2.0,
    TIME_CONC=2.0,
    GLACIER_CORR=1.6,
    GLACIER_K=0.2,
    SNOW_CORR=1.0,
    PRECIP_CORR=1.0,
    TEMP_CORR=0.0,
)
BOUNDS = dict(
    RAINSNOW_TEMP=(-2, 2),
    MELT_FACTOR=(2.3, 9),
    REFREEZE_FACTOR=(0, 3),
    HBV_BETA=(0.5, 6),
    SOIL_DEPTH=(0.1, 3),
    FAST_K=(0.01, 0.5),
    SLOW_K=(0.0005, 0.08),
    PERC=(0.1, 10),
    TIME_CONC=(0.5, 15),
    GLACIER_CORR=(1, 2.5),
    GLACIER_K=(0.06, 0.8),
    SNOW_CORR=(0.7, 1.5),
    PRECIP_CORR=(0.9, 1.5),
    TEMP_CORR=(-1, 2.5),
)


def parameters(settings, params):
    d = Path(settings)
    p = dict(DEFAULTS, **params)
    (d / "model.rvp").write_text(f"""# Raven HBV-EC; bare/open land and explicit glacier HRUs
:SoilClasses
 TOP, 1, 0, 0
 FAST, 1, 0, 0
 SLOW, 1, 0, 0
:EndSoilClasses
:SoilProfiles
 LAND, 3, TOP, {p["SOIL_DEPTH"]}, FAST, 100, SLOW, 100
 GLACIER, 0
:EndSoilProfiles
:VegetationClasses
 OPEN, 0, 0, 0
:EndVegetationClasses
:VegetationParameterList
 :Parameters, MAX_CAPACITY, MAX_SNOW_CAPACITY, TFRAIN, TFSNOW
 :Units, mm, mm, frac, frac
 OPEN, 0, 0, 1, 1
:EndVegetationParameterList
:LandUseClasses
 LAND, 0, 0
 ICE, 0, 0
:EndLandUseClasses
:GlobalParameter RAINSNOW_TEMP {p["RAINSNOW_TEMP"]}
:GlobalParameter RAINSNOW_DELTA 2
:GlobalParameter SNOW_SWI 0.05
:LandUseParameterList
 :Parameters, MELT_FACTOR, MIN_MELT_FACTOR, HBV_MELT_FOR_CORR, REFREEZE_FACTOR, HBV_MELT_ASP_CORR, HBV_MELT_GLACIER_CORR, HBV_GLACIER_KMIN, GLAC_STORAGE_COEFF, HBV_GLACIER_AG
 :Units, mm/d/K, mm/d/K, none, mm/d/K, none, none, 1/d, 1/d, 1/mm
 [DEFAULT], {p["MELT_FACTOR"]}, 2.2, 1, {p["REFREEZE_FACTOR"]}, 0, {p["GLACIER_CORR"]}, 0.05, {p["GLACIER_K"]}, 0.05
:EndLandUseParameterList
:SoilParameterList
 :Parameters, POROSITY, FIELD_CAPACITY, SAT_WILT, HBV_BETA, MAX_CAP_RISE_RATE, MAX_PERC_RATE, BASEFLOW_COEFF, BASEFLOW_N
 :Units, none, none, none, none, mm/d, mm/d, 1/d, none
 [DEFAULT], 0.4, 0.7, 0.05, {p["HBV_BETA"]}, 0, 0, 0, 1
 FAST, 0.4, 0.7, 0, {p["HBV_BETA"]}, 0, {p["PERC"]}, {p["FAST_K"]}, 1
 SLOW, 0.4, 0.7, 0, {p["HBV_BETA"]}, 0, 0, {p["SLOW_K"]}, 1
:EndSoilParameterList
""")
    meta = json.loads((d / "geometry.json").read_text())
    text = ":SubBasins\n 1, outlet, -1, NONE, 0, 1\n:EndSubBasins\n"
    text += f":SubBasinProperties\n :Parameters, TIME_CONC, TIME_TO_PEAK\n :Units, d, d\n 1, {p['TIME_CONC']}, {p['TIME_CONC'] / 2}\n:EndSubBasinProperties\n:HRUs\n"
    for h in meta["hrus"]:
        text += f" {h['id']}, {h['area_km2']}, {h['elevation']}, {h['lat']}, {h['lon']}, 1, {'ICE' if h['ice'] else 'LAND'}, OPEN, {'GLACIER' if h['ice'] else 'LAND'}, [NONE], [NONE], 0, 0\n"
    text += ":EndHRUs\n"
    (d / "model.rvh").write_text(text)
    rvt = ""
    for i, b in enumerate(meta["bands"]):
        rvt += f":Gauge band{i + 1}\n :Latitude {b['lat']}\n :Longitude {b['lon']}\n :Elevation {b['elevation']}\n :RainCorrection {p['PRECIP_CORR']}\n :SnowCorrection {p['PRECIP_CORR'] * p['SNOW_CORR']}\n :TemperatureCorrection {p['TEMP_CORR']}\n :RedirectToFile {meta['forcing_dir']}/band{i + 1}.rvt\n:EndGauge\n"
    (d / "model.rvt").write_text(rvt)


def setup(settings, bands, forcing, ice, initial_ice):
    d = Path(settings)
    d.mkdir(parents=True, exist_ok=True)
    fd = d / "forcing"
    fd.mkdir(exist_ok=True)
    hrus: list[dict] = []
    bs: list[dict] = []
    for i, b in bands.iterrows():
        bs.append(dict(lat=float(b.center_lat), lon=float(b.center_lon), elevation=float(b.elev_mean)))
        for isice, f in [(False, 1 - ice[i]), (True, ice[i])]:
            if f > 0:
                hrus.append(dict(id=len(hrus) + 1, band=i, ice=isice, area_km2=float(b.HRU_area * f / 1e6), **bs[-1]))
        a = np.column_stack([forcing[k].values[:, i] for k in ["precip", "temp", "tmin", "tmax", "pet"]])
        header = f":MultiData\n {pd.Timestamp(forcing.time.values[0]).date()} 00:00:00 1 {len(a)}\n:Parameters PRECIP TEMP_DAILY_AVE TEMP_DAILY_MIN TEMP_DAILY_MAX PET\n:Units mm/d C C C mm/d"
        np.savetxt(fd / f"band{i + 1}.rvt", a, fmt="%.6f", header=header, footer=":EndMultiData", comments="")
    (d / "geometry.json").write_text(
        json.dumps(dict(hrus=hrus, bands=bs, forcing_dir=str(fd), ice_mm=initial_ice), indent=2)
    )
    (d / "model.rvi").write_text(f""":StartDate {pd.Timestamp(forcing.time.values[0]).date()} 00:00:00
:Duration {len(forcing.time)}
:TimeStep 1
:SilentMode
:FiniteGlacierStorage
:Method ORDERED_SERIES
:Routing ROUTE_NONE
:CatchmentRoute TRIANGULAR_UH
:Interpolation INTERP_NEAREST_NEIGHBOR
:Evaporation PET_DATA
:OW_Evaporation PET_DATA
:RainSnowFraction RAINSNOW_HBV
:PotentialMeltMethod POTMELT_HBV
:OroTempCorrect OROCORR_NONE
:OroPrecipCorrect OROCORR_NONE
:OroPETCorrect OROCORR_NONE
:PrecipIceptFract PRECIP_ICEPT_USER
:SoilModel SOIL_MULTILAYER 3
:HydrologicProcesses
 :SnowRefreeze FREEZE_DEGREE_DAY SNOW_LIQ SNOW
 :Precipitation PRECIP_RAVEN ATMOS_PRECIP MULTIPLE
 :CanopyEvaporation CANEVP_ALL CANOPY ATMOSPHERE
 :CanopySnowEvap CANEVP_ALL CANOPY_SNOW ATMOSPHERE
 :SnowBalance SNOBAL_SIMPLE_MELT SNOW SNOW_LIQ
 :Overflow RAVEN_DEFAULT SNOW_LIQ PONDED_WATER
 :Flush RAVEN_DEFAULT PONDED_WATER GLACIER
 :-->Conditional HRU_TYPE IS GLACIER
 :GlacierMelt GMELT_HBV GLACIER_ICE GLACIER
 :GlacierRelease GRELEASE_HBV_EC GLACIER SURFACE_WATER
 :Infiltration INF_HBV PONDED_WATER MULTIPLE
 :Flush RAVEN_DEFAULT SURFACE_WATER SOIL[1]
 :-->Conditional HRU_TYPE IS_NOT GLACIER
 :SoilEvaporation SOILEVAP_HBV SOIL[0] ATMOSPHERE
 :Percolation PERC_CONSTANT SOIL[1] SOIL[2]
 :Baseflow BASE_LINEAR SOIL[1] SURFACE_WATER
 :Baseflow BASE_LINEAR SOIL[2] SURFACE_WATER
:EndHydrologicProcesses
:CustomOutput DAILY AVERAGE SNOW BY_HRU
:CustomOutput DAILY AVERAGE GLACIER_ICE BY_HRU
:CustomOutput DAILY AVERAGE SNOW_LIQ BY_HRU
:CustomOutput DAILY AVERAGE GLACIER BY_HRU
""")
    initial = ":HRUStateVariableTable\n :Attributes, SOIL[0], SOIL[1], SOIL[2], GLACIER_ICE\n :Units, mm, mm, mm, mm\n"
    for h in hrus:
        initial += f" {h['id']}, {0 if h['ice'] else 100}, 0, 0, {initial_ice if h['ice'] else 0}\n"
    (d / "model.rvc").write_text(initial + ":EndHRUStateVariableTable\n")
    parameters(d, {})
